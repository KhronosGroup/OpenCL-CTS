//
// Copyright (c) 2017 The Khronos Group Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "testBase.h"
#include "harness/testHarness.h"

const char *context_test_kernels[] = {
    "__kernel void sample_test_1(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dst[tid] = src[tid];\n"
    "\n"
    "}\n"

    "__kernel void sample_test_2(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dst[tid] = src[tid] * 2;\n"
    "\n"
    "}\n"

    "__kernel void sample_test_3(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dst[tid] = src[tid] / 2;\n"
    "\n"
    "}\n"

    "__kernel void sample_test_4(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dst[tid] = src[tid] /3;\n"
    "\n"
    "}\n"
};

cl_uint sampleAction1(cl_uint source) { return source; }
cl_uint sampleAction2(cl_uint source) { return source * 2; }
cl_uint sampleAction3(cl_uint source) { return source / 2; }
cl_uint sampleAction4(cl_uint source) { return source / 3; }


typedef cl_uint (*sampleActionFn)(cl_uint source);

sampleActionFn    sampleActions[4] = { sampleAction1, sampleAction2, sampleAction3, sampleAction4 };

#define BUFFER_COUNT 2
#define TEST_SIZE    512

typedef struct TestItem
{
    struct TestItem     *next;
    cl_context          c;
    cl_command_queue    q;
    cl_program          p;
    cl_kernel           k[4];
    cl_mem              m[BUFFER_COUNT];
    MTdata              d;
}TestItem;

static void DestroyTestItem( TestItem *item );

// Attempt to create a context and associated objects
TestItem *CreateTestItem( cl_device_id deviceID, cl_int *err )
{
    cl_int error = 0;
    size_t i;

    // Allocate the TestItem struct
    TestItem *item = (TestItem *) malloc( sizeof(TestItem ) );
    if( NULL == item  )
    {
        if( err )
        {
            log_error( "FAILURE: Failed to allocate TestItem -- out of host memory!\n" );
            *err = CL_OUT_OF_HOST_MEMORY;
        }
        return NULL;
    }
    //zero so we know which fields we have initialized
    memset( item, 0, sizeof( *item ) );

    item->d = init_genrand( gRandomSeed );
    if( NULL == item->d )
    {
        if( err )
        {
            log_error( "FAILURE: Failed to allocate mtdata om CreateTestItem -- out of host memory!\n" );
            *err = CL_OUT_OF_HOST_MEMORY;
        }
        DestroyTestItem( item );
        return NULL;
    }


    // Create a context
    item->c = clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &error );
    if( item->c == NULL || error != CL_SUCCESS)
    {
        if (err) {
            log_error( "FAILURE: clCreateContext failed in CreateTestItem: %d\n", error);
            *err = error;
        }
        DestroyTestItem( item );
        return NULL;
    }

    // Create a queue
    item->q = clCreateCommandQueue( item->c, deviceID, 0, &error);
    if( item->q == NULL || error != CL_SUCCESS)
    {
        if (err) {
            log_error( "FAILURE: clCreateCommandQueue failed in CreateTestItem: %d\n", error );
            *err = error;
        }
        DestroyTestItem( item );
        return NULL;
    }

    // Create a program
    error = create_single_kernel_helper_create_program(item->c, &item->p, 1, context_test_kernels);
    if( NULL == item->p || CL_SUCCESS != error )
    {
        if( err )
        {
            log_error( "FAILURE: clCreateProgram failed in CreateTestItem: %d\n", error );
            *err = error;
        }
        DestroyTestItem( item );
        return NULL;
    }

    error = clBuildProgram( item->p, 1, &deviceID, "", NULL, NULL );
    if( error )
    {
        if( err )
        {
            log_error( "FAILURE: clBuildProgram failed in CreateTestItem: %d\n", error );
            *err = error;
        }
        DestroyTestItem( item );
        return NULL;
    }

    // create some kernels
    for( i = 0; i < sizeof( item->k ) / sizeof( item->k[0] ); i++ )
    {
        static const char *kernelNames[] = { "sample_test_1", "sample_test_2", "sample_test_3", "sample_test_4" };
        item->k[i] = clCreateKernel( item->p, kernelNames[i], &error );
        if( NULL == item->k[i] || CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "FAILURE: clCreateKernel( \"%s\" ) failed in CreateTestItem: %d\n", kernelNames[i], error );
                *err = error;
            }
            DestroyTestItem( item );
            return NULL;
        }
    }

    // create some mem objects
    for( i = 0; i < BUFFER_COUNT; i++ )
    {
        item->m[i] = clCreateBuffer(item->c, CL_MEM_READ_WRITE,
                                    TEST_SIZE * sizeof(cl_uint), NULL, &error);
        if( NULL == item->m[i] || CL_SUCCESS != error )
        {
            if( err )
            {
                log_error("FAILURE: clCreateBuffer( %zu bytes ) failed in "
                          "CreateTestItem: %d\n",
                          TEST_SIZE * sizeof(cl_uint), error);
                *err = error;
            }
            DestroyTestItem( item );
            return NULL;
        }
    }


    return item;
}

// Destroy a context and associate objects
static void DestroyTestItem( TestItem *item )
{
    size_t i;

    if( NULL == item )
        return;

    if( item->d )
        free_mtdata( item->d );
    if( item->c)
        clReleaseContext( item->c );
    if( item->q)
        clReleaseCommandQueue( item->q );
    if( item->p)
        clReleaseProgram( item->p );
    for( i = 0; i < sizeof( item->k ) / sizeof( item->k[0] ); i++ )
    {
        if( item->k[i])
            clReleaseKernel( item->k[i] );
    }
    for( i = 0; i < BUFFER_COUNT; i++ )
    {
        if( item->m[i])
            clReleaseMemObject( item->m[i] );
    }
    free(item );
}


cl_int UseTestItem( const TestItem *item, cl_int *err )
{
    size_t i, j;
    cl_int error = CL_SUCCESS;

    // Fill buffer 0 with random numbers
    cl_uint *mapped = (cl_uint *)clEnqueueMapBuffer(
        item->q, item->m[0], CL_TRUE, CL_MAP_WRITE, 0,
        TEST_SIZE * sizeof(cl_uint), 0, NULL, NULL, &error);
    if( NULL == mapped || CL_SUCCESS != error )
    {
        if( err )
        {
            log_error( "FAILURE: Failed to map buffer 0 for writing: %d\n", error );
            *err = error;
        }
        return error;
    }

    for( j = 0; j < TEST_SIZE; j++ )
        mapped[j] = genrand_int32(item->d);

    error = clEnqueueUnmapMemObject( item->q, item->m[0], mapped, 0, NULL, NULL );
    if( CL_SUCCESS != error )
    {
        if( err )
        {
            log_error( "FAILURE: failure to unmap buffer 0 for writing: %d\n", error );
            *err = error;
        }
        return error;
    }

    // try each kernel in turn.
    for( j = 0; j < sizeof(item->k) / sizeof( item->k[0] ); j++ )
    {
        // Fill buffer 1 with 0xdeaddead
        mapped = (cl_uint *)clEnqueueMapBuffer(
            item->q, item->m[1], CL_TRUE, CL_MAP_WRITE, 0,
            TEST_SIZE * sizeof(cl_uint), 0, NULL, NULL, &error);
        if( NULL == mapped || CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to map buffer 1 for writing: %d\n", error );
                *err = error;
            }
            return error;
        }

        for( i = 0; i < TEST_SIZE; i++ )
            mapped[i] = 0xdeaddead;

        error = clEnqueueUnmapMemObject( item->q, item->m[1], mapped, 0, NULL, NULL );
        if( CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to unmap buffer 1 for writing: %d\n", error );
                *err = error;
            }
            return error;
        }

        // Run the kernel
        error = clSetKernelArg( item->k[j], 0, sizeof( cl_mem), &item->m[0] );
        if( error )
        {
            if( err )
            {
                log_error("FAILURE to set arg 0 for kernel # %zu :  %d\n", j,
                          error);
                *err = error;
            }
            return error;
        }

        error = clSetKernelArg( item->k[j], 1, sizeof( cl_mem), &item->m[1] );
        if( error )
        {
            if( err )
            {
                log_error(
                    "FAILURE: Unable to set arg 1 for kernel # %zu :  %d\n", j,
                    error);
                *err = error;
            }
            return error;
        }

        size_t work_size = TEST_SIZE;
        size_t global_offset = 0;
        error = clEnqueueNDRangeKernel( item->q, item->k[j], 1, &global_offset, &work_size, NULL, 0, NULL, NULL );
        if( CL_SUCCESS != error )
        {
            if( err )
            {
                log_error("FAILURE: Unable to enqueue kernel %zu: %d\n", j,
                          error);
                *err = error;
            }
            return error;
        }

        // Get the results back
        mapped = (cl_uint *)clEnqueueMapBuffer(
            item->q, item->m[1], CL_TRUE, CL_MAP_READ, 0,
            TEST_SIZE * sizeof(cl_uint), 0, NULL, NULL, &error);
        if( NULL == mapped || CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to map buffer 1 for reading: %d\n", error );
                *err = error;
            }
            return error;
        }

        // Get our input data so we can check against it
        cl_uint *inputData = (cl_uint *)clEnqueueMapBuffer(
            item->q, item->m[0], CL_TRUE, CL_MAP_READ, 0,
            TEST_SIZE * sizeof(cl_uint), 0, NULL, NULL, &error);
        if( NULL == mapped || CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to map buffer 0 for reading: %d\n", error );
                *err = error;
            }
            return error;
        }


        //Verify the results
        for( i = 0; i < TEST_SIZE; i++ )
        {
            cl_uint expected = sampleActions[j](inputData[i]);
            cl_uint result = mapped[i];
            if( expected != result )
            {
                log_error("FAILURE:  Sample data at position %zu does not "
                          "match expected result: *0x%8.8x vs. 0x%8.8x\n",
                          i, expected, result);
                if( err )
                    *err = -1;
                return -1;
            }
        }

        //Clean up
        error = clEnqueueUnmapMemObject( item->q, item->m[0], inputData, 0, NULL, NULL );
        if( CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to unmap buffer 0 for reading: %d\n", error );
                *err = error;
            }
            return error;
        }

        error = clEnqueueUnmapMemObject( item->q, item->m[1], mapped, 0, NULL, NULL );
        if( CL_SUCCESS != error )
        {
            if( err )
            {
                log_error( "Failed to unmap buffer 1 for reading: %d\n", error );
                *err = error;
            }
            return error;
        }

    }

    // Make sure that the last set of unmap calls get run
    error = clFinish( item->q );
    if( CL_SUCCESS != error )
    {
        if( err )
        {
            log_error( "Failed to clFinish: %d\n", error );
            *err = error;
        }
        return error;
    }

    return CL_SUCCESS;
}



int test_context_multiple_contexts_same_device(cl_device_id deviceID, size_t maxCount, size_t minCount )
{
    size_t i, j;
    cl_int err = CL_SUCCESS;

    //Figure out how many of these we can make before the first failure
    TestItem *list = NULL;

    for( i = 0; i < maxCount; i++ )
    {
        // create a context and accompanying objects
        TestItem *current = CreateTestItem( deviceID, NULL /*no error reporting*/ );
        if( NULL == current )
            break;

        // Attempt to use it
        cl_int failed = UseTestItem( current, NULL );

        if( failed )
        {
            DestroyTestItem( current );
            break;
        }

        // Add the successful test item to the list
        current->next = list;
        list = current;
    }

    // Check to make sure we made the minimum amount
    if( i < minCount )
    {
        log_error("FAILURE: only could make %zu of %zu contexts!\n", i,
                  minCount);
        err = -1;
        goto exit;
    }

    // Report how many contexts we made
    if( i == maxCount )
        log_info("Successfully created all %zu contexts.\n", i);
    else
        log_info("Successfully created %zu contexts out of %zu\n", i, maxCount);

    // Set the count to be the number we succesfully made
    maxCount = i;

    // Make sure we can do it again a few times
    log_info( "Tring to do it 5 more times" );
    fflush( stdout);
    for( j = 0; j < 5; j++ )
    {
        //free all the contexts we already made
        while( list )
        {
            TestItem *current = list;
            list = list->next;
            current->next = NULL;
            DestroyTestItem( current );
        }

        // Attempt to make them again
        for( i = 0; i < maxCount; i++ )
        {
            // create a context and accompanying objects
            TestItem *current = CreateTestItem( deviceID, &err );
            if( err )
            {
                log_error( "\nTest Failed with error at CreateTestItem: %d\n", err );
                goto exit;
            }

            // Attempt to use it
            cl_int failed = UseTestItem( current, &err );

            if( failed || err )
            {
                DestroyTestItem( current );
                log_error( "\nTest Failed with error at UseTestItem: %d\n", err );
                goto exit;
            }

            // Add the successful test item to the list
            current->next = list;
            list = current;
        }
        log_info( "." );
        fflush( stdout );
    }

    log_info( "Done.\n" );

exit:
    //free all the contexts we already made
    while( list )
    {
        TestItem *current = list;
        list = list->next;
        current->next = NULL;

        DestroyTestItem( current );
    }

    return err;
}

//  This test tests to make sure that your implementation isn't super leaky.  We make a bunch of contexts (up to some
//  sane limit, currently 200), attempting to use each along the way. We keep track of how many we could make before
//  a failure occurred.   We then free everything and attempt to go do it again a few times.  If you are able to make
//  that many contexts 5 times over, then you pass.
int test_context_multiple_contexts_same_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_context_multiple_contexts_same_device(deviceID, 200, 1);
}

int test_context_two_contexts_same_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_context_multiple_contexts_same_device( deviceID, 2, 2 );
}

int test_context_three_contexts_same_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_context_multiple_contexts_same_device( deviceID, 3, 3 );
}

int test_context_four_contexts_same_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_context_multiple_contexts_same_device( deviceID, 4, 4 );
}

