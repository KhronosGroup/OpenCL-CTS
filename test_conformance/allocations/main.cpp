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

#include "allocation_functions.h"
#include "allocation_fill.h"
#include "allocation_execute.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"
#include <time.h>

typedef long long unsigned llu;

int g_repetition_count = 1;
int g_reduction_percentage = 100;
int g_write_allocations = 1;
int g_multiple_allocations = 0;
int g_execute_kernel = 1;

static size_t g_max_size;
static RandomSeed g_seed( gRandomSeed );

cl_long g_max_individual_allocation_size;
cl_long g_global_mem_size;

cl_uint checksum;

static void printUsage( const char *execName );

test_status init_cl( cl_device_id device ) {
    int error;

    error = clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(g_max_individual_allocation_size), &g_max_individual_allocation_size, NULL );
    if ( error ) {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_MAX_MEM_ALLOC_SIZE");
        return TEST_FAIL;
    }
    error = clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(g_global_mem_size), &g_global_mem_size, NULL );
    if ( error ) {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_GLOBAL_MEM_SIZE");
        return TEST_FAIL;
    }

    log_info("Device reports CL_DEVICE_MAX_MEM_ALLOC_SIZE=%llu bytes (%gMB), CL_DEVICE_GLOBAL_MEM_SIZE=%llu bytes (%gMB).\n",
             llu( g_max_individual_allocation_size ), toMB( g_max_individual_allocation_size ),
             llu( g_global_mem_size ), toMB( g_global_mem_size ) );

    if( g_global_mem_size > (cl_ulong)SIZE_MAX )
    {
        g_global_mem_size = (cl_ulong)SIZE_MAX;
    }

    if( g_max_individual_allocation_size > g_global_mem_size )
    {
        log_error( "FAILURE:  CL_DEVICE_MAX_MEM_ALLOC_SIZE (%llu) is greater than the CL_DEVICE_GLOBAL_MEM_SIZE (%llu)\n",
                   llu( g_max_individual_allocation_size ), llu( g_global_mem_size ) );
        return TEST_FAIL;
    }

    // We may need to back off the global_mem_size on unified memory devices to leave room for application and operating system code
    // and associated data in the working set, so we dont start pathologically paging.
    // Check to see if we are a unified memory device
    cl_bool hasUnifiedMemory = CL_FALSE;
    if( ( error = clGetDeviceInfo( device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof( hasUnifiedMemory ), &hasUnifiedMemory, NULL ) ) )
    {
        print_error( error, "clGetDeviceInfo failed for CL_DEVICE_HOST_UNIFIED_MEMORY");
        return TEST_FAIL;
    }
    // we share unified memory so back off to 1/2 the global memory size.
    if( CL_TRUE == hasUnifiedMemory )
    {
        g_global_mem_size -= g_global_mem_size /2;
        log_info( "Device shares memory with the host, so backing off the maximum combined allocation size to be %gMB to avoid rampant paging.\n",
                  toMB( g_global_mem_size ) );
    }
    else
    {
        // Lets just use 60% of total available memory as framework/driver may not allow using all of it
        // e.g. vram on GPU is used by window server and even for this test, we need some space for context,
        // queue, kernel code on GPU.
        g_global_mem_size *= 0.60;
    }

    if( gReSeed )
    {
        g_seed = RandomSeed( gRandomSeed );
    }

    return TEST_PASS;
}

int doTest( cl_device_id device, cl_context context, cl_command_queue queue, AllocType alloc_type )
{
    int error;
    int failure_counts = 0;
    size_t final_size;
    size_t current_test_size;
    cl_mem mems[MAX_NUMBER_TO_ALLOCATE];
    int number_of_mems_used;
    cl_ulong max_individual_allocation_size = g_max_individual_allocation_size;
    cl_ulong global_mem_size = g_global_mem_size ;
    const bool allocate_image =
        (alloc_type != BUFFER) && (alloc_type != BUFFER_NON_BLOCKING);

    static const char* alloc_description[] = {
        "buffer(s)",
        "read-only image(s)",
        "write-only image(s)",
        "buffer(s)",
        "read-only image(s)",
        "write-only image(s)",
    };

    // Skip image tests if we don't support images on the device
    if (allocate_image && checkForImageSupport(device))
    {
        log_info( "Can not test image allocation because device does not support images.\n" );
        return 0;
    }

    // This section was added in order to fix a bug in the test
    // If CL_DEVICE_MAX_MEM_ALLOC_SIZE is much grater than CL_DEVICE_IMAGE2D_MAX_WIDTH * CL_DEVICE_IMAGE2D_MAX_HEIGHT
    // The test will fail in image allocations as the size requested for the allocation will be much grater than the maximum size allowed for image
    if (allocate_image)
    {
        size_t max_width, max_height;

        error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( max_width ), &max_width, NULL );
        test_error_abort( error, "clGetDeviceInfo failed for CL_DEVICE_IMAGE2D_MAX_WIDTH" );

        error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( max_height ), &max_height, NULL );
        test_error_abort( error, "clGetDeviceInfo failed for CL_DEVICE_IMAGE2D_MAX_HEIGHT" );

        cl_ulong max_image2d_size = (cl_ulong)max_height * max_width * 4 * sizeof(cl_uint);

        if( max_individual_allocation_size > max_image2d_size )
        {
            max_individual_allocation_size = max_image2d_size;
        }
    }

    // Pick the baseline size based on whether we are doing a single large or multiple allocations
    g_max_size = g_multiple_allocations ? (size_t)global_mem_size : (size_t)max_individual_allocation_size;

    // Adjust based on the percentage
    if( g_reduction_percentage != 100 )
    {
        log_info( "NOTE: reducing max allocations to %d%%.\n", g_reduction_percentage );
        g_max_size = (size_t)( (double)g_max_size * (double)g_reduction_percentage / 100.0 );
    }

    // Round to nearest MB.
    g_max_size &= (size_t)(0xFFFFFFFFFF00000ULL);

    log_info( "** Target allocation size (rounded to nearest MB) is: %llu bytes (%gMB).\n", llu( g_max_size ), toMB( g_max_size ) );
    log_info( "** Allocating %s to size %gMB.\n", alloc_description[alloc_type], toMB( g_max_size ) );

    for( int count = 0; count < g_repetition_count; count++ )
    {
        current_test_size = g_max_size;
        error = FAILED_TOO_BIG;
        log_info( "  => Allocation %d\n", count + 1 );

        while( ( error == FAILED_TOO_BIG ) && ( current_test_size > g_max_size / 8 ) )
        {
            // Reset our checksum for each allocation
            checksum = 0;

            // Do the allocation
            error = allocate_size( context, &queue, device, g_multiple_allocations, current_test_size, alloc_type,
                                   mems, &number_of_mems_used, &final_size, g_write_allocations, g_seed );

            // If we succeeded and we're supposed to execute a kernel, do so.
            if( error == SUCCEEDED && g_execute_kernel )
            {
                log_info( "\tExecuting kernel with memory objects.\n" );
                error = execute_kernel( context, &queue, device, alloc_type, mems, number_of_mems_used,
                                        g_write_allocations );
            }

            // If we failed to allocate more than 1/8th of the requested amount return a failure.
            if( final_size < (size_t)g_max_size / 8 )
            {
                log_error( "===> Allocation %d failed to allocate more than 1/8th of the requested size.\n", count + 1 );
                failure_counts++;
            }

            // Clean up.
            for( int i = 0; i < number_of_mems_used; i++ )
            {
                clReleaseMemObject( mems[i] );
            }

            if( error == FAILED_ABORT )
            {
                log_error( "  => Allocation %d failed.\n", count + 1 );
                failure_counts++;
            }

            if( error == FAILED_TOO_BIG )
            {
                current_test_size -= g_max_size / 16;
                log_info( "\tFailed at this size; trying a smaller size of %gMB.\n", toMB( current_test_size ) );
            }
        }

        if( error == SUCCEEDED && current_test_size == g_max_size )
        {
            log_info("\tPASS: Allocation succeeded.\n");
        }
        else if( error == SUCCEEDED && current_test_size > g_max_size / 8 )
        {
            log_info("\tPASS: Allocation succeeded at reduced size.\n");
        }
        else
        {
            log_error("\tFAIL: Allocation failed.\n");
            failure_counts++;
        }
    }

    return failure_counts;
}

int test_buffer(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, BUFFER );
}
int test_image2d_read(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, IMAGE_READ );
}
int test_image2d_write(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, IMAGE_WRITE );
}
int test_buffer_non_blocking(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, BUFFER_NON_BLOCKING );
}
int test_image2d_read_non_blocking(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, IMAGE_READ_NON_BLOCKING );
}
int test_image2d_write_non_blocking(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, IMAGE_WRITE_NON_BLOCKING );
}

test_definition test_list[] = {
    ADD_TEST( buffer ),
    ADD_TEST( image2d_read ),
    ADD_TEST( image2d_write ),
    ADD_TEST( buffer_non_blocking ),
    ADD_TEST( image2d_read_non_blocking ),
    ADD_TEST( image2d_write_non_blocking ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    char *endPtr;
    int r;

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return 1;
    }

    const char ** argList = (const char **)calloc( argc, sizeof( char*) );

    if( NULL == argList )
    {
        log_error( "Failed to allocate memory for argList array.\n" );
        return 1;
    }

    argList[0] = argv[0];
    size_t argCount = 1;

    // Parse arguments
    for( int i = 1; i < argc; i++ )
    {
        if( strcmp( argv[i], "multiple" ) == 0 )
            g_multiple_allocations = 1;
        else if( strcmp( argv[i], "single" ) == 0 )
            g_multiple_allocations = 0;

        else if( ( r = (int)strtol( argv[i], &endPtr, 10 ) ) && ( endPtr != argv[i] ) && ( *endPtr == 0 ) )
        {
            // By spec, that means the entire string was an integer, so take it as a repetition count
            g_repetition_count = r;
        }

        else if( strchr( argv[i], '%' ) != NULL )
        {
            // Reduction percentage (let strtol ignore the percentage)
            g_reduction_percentage = (int)strtol( argv[i], NULL, 10 );
        }

        else if( strcmp( argv[i], "do_not_force_fill" ) == 0 )
        {
            g_write_allocations = 0;
        }

        else if( strcmp( argv[i], "do_not_execute" ) == 0 )
        {
            g_execute_kernel = 0;
        }

        else if ( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 )
        {
            printUsage( argv[0] );
            free(argList);
            return -1;
        }

        else
        {
            argList[argCount] = argv[i];
            argCount++;
        }
    }

    int ret = runTestHarnessWithCheck( argCount, argList, test_num, test_list, false, 0, init_cl );

    free(argList);
    return ret;
}

void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [options] [test_names]\n", execName );
    log_info( "Options:\n" );
    log_info( "\trandomize - Uses random seed\n" );
    log_info( "\tsingle - Tests using a single allocation as large as possible\n" );
    log_info( "\tmultiple - Tests using as many allocations as possible\n" );
    log_info( "\n" );
    log_info( "\tnumReps - Optional integer specifying the number of repetitions to run and average the result (defaults to 1)\n" );
    log_info( "\treduction%% - Optional integer, followed by a %% sign, that acts as a multiplier for the target amount of memory.\n" );
    log_info( "\t             Example: target amount of 512MB and a reduction of 75%% will result in a target of 384MB.\n" );
    log_info( "\n" );
    log_info( "\tdo_not_force_fill - Disable explicitly write data to all memory objects after creating them.\n" );
    log_info( "\t                    Without this, the kernel execution can not verify its checksum.\n" );
    log_info( "\tdo_not_execute - Disable executing a kernel that accesses all of the memory objects.\n" );
    log_info( "\n" );
    log_info( "Test names (Allocation Types):\n" );
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}
