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


const char *sample_single_kernel[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n" };

size_t sample_single_kernel_lengths[1];

const char *sample_two_kernels[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n",
    "__kernel void sample_test2(__global int *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)src[tid];\n"
    "\n"
    "}\n" };

size_t sample_two_kernel_lengths[2];

const char *sample_two_kernels_in_1[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n"
    "__kernel void sample_test2(__global int *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)src[tid];\n"
    "\n"
    "}\n" };

size_t sample_two_kernels_in_1_lengths[1];


const char *repeate_test_kernel =
"__kernel void test_kernel(__global int *src, __global int *dst)\n"
"{\n"
" dst[get_global_id(0)] = src[get_global_id(0)]+1;\n"
"}\n";



int test_load_single_kernel(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    cl_program testProgram;
    clKernelWrapper kernel;
    cl_context testContext;
    unsigned int numKernels;
    cl_char testName[512];
    cl_uint testArgCount;
    size_t realSize;


    error = create_single_kernel_helper(context, &program, NULL, 1, sample_single_kernel, NULL);
    test_error( error, "Unable to build test program" );

    error = clCreateKernelsInProgram(program, 1, &kernel, &numKernels);
    test_error( error, "Unable to create single kernel program" );

    /* Check program and context pointers */
    error = clGetKernelInfo( kernel, CL_KERNEL_PROGRAM, sizeof( cl_program ), &testProgram, &realSize );
    test_error( error, "Unable to get kernel's program" );
    if( (cl_program)testProgram != (cl_program)program )
    {
        log_error( "ERROR: Returned kernel's program does not match program used to create it! (Got %p, expected %p)\n", (cl_program)testProgram, (cl_program)program );
        return -1;
    }
    if( realSize != sizeof( cl_program ) )
    {
        log_error( "ERROR: Returned size of kernel's program does not match expected size (expected %d, got %d)\n", (int)sizeof( cl_program ), (int)realSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_CONTEXT, sizeof( cl_context ), &testContext, &realSize );
    test_error( error, "Unable to get kernel's context" );
    if( (cl_context)testContext != (cl_context)context )
    {
        log_error( "ERROR: Returned kernel's context does not match program used to create it! (Got %p, expected %p)\n", (cl_context)testContext, (cl_context)context );
        return -1;
    }
    if( realSize != sizeof( cl_context ) )
    {
        log_error( "ERROR: Returned size of kernel's context does not match expected size (expected %d, got %d)\n", (int)sizeof( cl_context ), (int)realSize );
        return -1;
    }

    /* Test arg count */
    error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, 0, NULL, &realSize );
    test_error( error, "Unable to get size of arg count info from kernel" );

    if( realSize != sizeof( testArgCount ) )
    {
        log_error( "ERROR: size of arg count not valid! %d\n", (int)realSize );
        return -1;
    }

    error = clGetKernelInfo( kernel, CL_KERNEL_NUM_ARGS, sizeof( testArgCount ), &testArgCount, NULL );
    test_error( error, "Unable to get arg count from kernel" );

    if( testArgCount != 2 )
    {
        log_error( "ERROR: Kernel arg count does not match!\n" );
        return -1;
    }


    /* Test function name */
    error = clGetKernelInfo( kernel, CL_KERNEL_FUNCTION_NAME, sizeof( testName ), testName, &realSize );
    test_error( error, "Unable to get name from kernel" );

    if( strcmp( (char *)testName, "sample_test" ) != 0 )
    {
        log_error( "ERROR: Kernel names do not match!\n" );
        return -1;
    }
    if( realSize != strlen( (char *)testName ) + 1 )
    {
        log_error( "ERROR: Length of kernel name returned does not validate (expected %d, got %d)\n", (int)strlen( (char *)testName ) + 1, (int)realSize );
        return -1;
    }

    /* All done */

    return 0;
}

int test_load_two_kernels(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel[2];
    unsigned int numKernels;
    cl_char testName[ 512 ];
    cl_uint testArgCount;


    error = create_single_kernel_helper(context, &program, NULL, 2, sample_two_kernels, NULL);
    test_error( error, "Unable to build test program" );

    error = clCreateKernelsInProgram(program, 2, &kernel[0], &numKernels);
    test_error( error, "Unable to create dual kernel program" );

    if( numKernels != 2 )
    {
        log_error( "ERROR: wrong # of kernels! (%d)\n", numKernels );
        return -1;
    }

    /* Check first kernel */
    error = clGetKernelInfo( kernel[0], CL_KERNEL_FUNCTION_NAME, sizeof( testName ), testName, NULL );
    test_error( error, "Unable to get function name from kernel" );

    int found_kernel1 = 0, found_kernel2 = 0;

    if( strcmp( (char *)testName, "sample_test" ) == 0 ) {
        found_kernel1 = 1;
    } else if( strcmp( (char *)testName, "sample_test2" ) == 0 ) {
        found_kernel2 = 1;
    } else {
        log_error( "ERROR: Invalid kernel name returned: \"%s\" expected \"%s\" or \"%s\".\n", testName, "sample_test", "sample_test2");
        return -1;
    }

    error = clGetKernelInfo( kernel[1], CL_KERNEL_FUNCTION_NAME, sizeof( testName ), testName, NULL );
    test_error( error, "Unable to get function name from second kernel" );

    if( strcmp( (char *)testName, "sample_test" ) == 0 ) {
        if (found_kernel1) {
            log_error("Kernel \"%s\" returned twice.\n", (char *)testName);
            return -1;
        }
        found_kernel1 = 1;
    } else if( strcmp( (char *)testName, "sample_test2" ) == 0 ) {
        if (found_kernel2) {
            log_error("Kernel \"%s\" returned twice.\n", (char *)testName);
            return -1;
        }
        found_kernel2 = 1;
    } else {
        log_error( "ERROR: Invalid kernel name returned: \"%s\" expected \"%s\" or \"%s\".\n", testName, "sample_test", "sample_test2");
        return -1;
    }

    if( !found_kernel1 || !found_kernel2 )
    {
        log_error( "ERROR: Kernel names do not match.\n" );
        if (!found_kernel1)
            log_error("Kernel \"%s\" not returned.\n", "sample_test");
        if (!found_kernel2)
            log_error("Kernel \"%s\" not returned.\n", "sample_test");
        return -1;
    }

    error = clGetKernelInfo( kernel[0], CL_KERNEL_NUM_ARGS, sizeof( testArgCount ), &testArgCount, NULL );
    test_error( error, "Unable to get arg count from kernel" );

    if( testArgCount != 2 )
    {
        log_error( "ERROR: wrong # of args for kernel\n" );
        return -1;
    }

    /* All done */
    return 0;
}

int test_load_two_kernels_in_one(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel[2];
    unsigned int numKernels;
    cl_char testName[512];
    cl_uint testArgCount;


    error = create_single_kernel_helper(context, &program, NULL, 1, sample_two_kernels_in_1, NULL);
    test_error( error, "Unable to build test program" );

    error = clCreateKernelsInProgram(program, 2, &kernel[0], &numKernels);
    test_error( error, "Unable to create dual kernel program" );

    if( numKernels != 2 )
    {
        log_error( "ERROR: wrong # of kernels! (%d)\n", numKernels );
        return -1;
    }

    /* Check first kernel */
    error = clGetKernelInfo( kernel[0], CL_KERNEL_FUNCTION_NAME, sizeof( testName ), testName, NULL );
    test_error( error, "Unable to get function name from kernel" );

    int found_kernel1 = 0, found_kernel2 = 0;

    if( strcmp( (char *)testName, "sample_test" ) == 0 ) {
        found_kernel1 = 1;
    } else if( strcmp( (char *)testName, "sample_test2" ) == 0 ) {
        found_kernel2 = 1;
    } else {
        log_error( "ERROR: Invalid kernel name returned: \"%s\" expected \"%s\" or \"%s\".\n", testName, "sample_test", "sample_test2");
        return -1;
    }

    error = clGetKernelInfo( kernel[0], CL_KERNEL_NUM_ARGS, sizeof( testArgCount ), &testArgCount, NULL );
    test_error( error, "Unable to get arg count from kernel" );

    if( testArgCount != 2 )
    {
        log_error( "ERROR: wrong # of args for kernel\n" );
        return -1;
    }

    /* Check second kernel */
    error = clGetKernelInfo( kernel[1], CL_KERNEL_FUNCTION_NAME, sizeof( testName ), testName, NULL );
    test_error( error, "Unable to get function name from kernel" );

    if( strcmp( (char *)testName, "sample_test" ) == 0 ) {
        if (found_kernel1) {
            log_error("Kernel \"%s\" returned twice.\n", (char *)testName);
            return -1;
        }
        found_kernel1 = 1;
    } else if( strcmp( (char *)testName, "sample_test2" ) == 0 ) {
        if (found_kernel2) {
            log_error("Kernel \"%s\" returned twice.\n", (char *)testName);
            return -1;
        }
        found_kernel2 = 1;
    } else {
        log_error( "ERROR: Invalid kernel name returned: \"%s\" expected \"%s\" or \"%s\".\n", testName, "sample_test", "sample_test2");
        return -1;
    }

    if( !found_kernel1 || !found_kernel2 )
    {
        log_error( "ERROR: Kernel names do not match.\n" );
        if (!found_kernel1)
            log_error("Kernel \"%s\" not returned.\n", "sample_test");
        if (!found_kernel2)
            log_error("Kernel \"%s\" not returned.\n", "sample_test");
        return -1;
    }

    /* All done */
    return 0;
}

int test_load_two_kernels_manually( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel1, kernel2;
    int error;


    /* Now create a test program */
    error = create_single_kernel_helper(context, &program, NULL, 1, sample_two_kernels_in_1, NULL);
    test_error( error, "Unable to build test program" );

    /* Try manually creating kernels (backwards just in case) */
    kernel1 = clCreateKernel( program, "sample_test2", &error );

    if( kernel1 == NULL || error != CL_SUCCESS )
    {
        print_error( error, "Could not get kernel 1" );
        return -1;
    }

    kernel2 = clCreateKernel( program, "sample_test", &error );

    if( kernel2 == NULL )
    {
        print_error( error, "Could not get kernel 2" );
        return -1;
    }

    return 0;
}

int test_get_program_info_kernel_names( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel1, kernel2;
    int error;
    size_t i;

    /* Now create a test program */
    error = create_single_kernel_helper(context, &program, NULL, 1, sample_two_kernels_in_1, NULL);
    test_error( error, "Unable to build test program" );

    /* Lookup the number of kernels in the program. */
    size_t total_kernels = 0;
    error = clGetProgramInfo(program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t),&total_kernels,NULL);
    test_error( error, "Unable to get program info num kernels");

    if (total_kernels != 2)
    {
        print_error( error, "Program did not contain two kernels" );
        return -1;
    }

    /* Lookup the kernel names. */
    const char* actual_names[] = { "sample_test;sample_test2", "sample_test2;sample_test"} ;

    size_t kernel_names_len = 0;
    error = clGetProgramInfo(program,CL_PROGRAM_KERNEL_NAMES,0,NULL,&kernel_names_len);
    test_error( error, "Unable to get length of kernel names list." );

    if (kernel_names_len != (strlen(actual_names[0])+1))
    {
        print_error( error, "Kernel names length did not match");
        return -1;
    }

    const size_t len = (kernel_names_len+1)*sizeof(char);
    char* kernel_names = (char*)malloc(len);
    error = clGetProgramInfo(program,CL_PROGRAM_KERNEL_NAMES,len,kernel_names,&kernel_names_len);
    test_error( error, "Unable to get kernel names list." );

    /* Check to see if the kernel name array is null terminated. */
    if (kernel_names[kernel_names_len-1] != '\0')
    {
        free(kernel_names);
        print_error( error, "Kernel name list was not null terminated");
        return -1;
    }

    /* Check to see if the correct kernel name string was returned. */
    for( i = 0; i < sizeof( actual_names ) / sizeof( actual_names[0] ); i++ )
        if( 0 == strcmp(actual_names[i],kernel_names) )
            break;

    if (i == sizeof( actual_names ) / sizeof( actual_names[0] ) )
    {
        free(kernel_names);
        log_error( "Kernel names \"%s\" did not match:\n", kernel_names );
        for( i = 0; i < sizeof( actual_names ) / sizeof( actual_names[0] ); i++ )
            log_error( "\t\t\"%s\"\n", actual_names[0] );
        return -1;
    }
    free(kernel_names);

    /* Try manually creating kernels (backwards just in case) */
    kernel1 = clCreateKernel( program, "sample_test", &error );
    if( kernel1 == NULL || error != CL_SUCCESS )
    {
        print_error( error, "Could not get kernel 1" );
        return -1;
    }

    kernel2 = clCreateKernel( program, "sample_test2", &error );
    if( kernel2 == NULL )
    {
        print_error( error, "Could not get kernel 2" );
        return -1;
    }

    return 0;
}

static const char *single_task_kernel[] = {
    "__kernel void sample_test(__global int *dst, int count)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    for( int i = 0; i < count; i++ )\n"
    "        dst[i] = tid + i;\n"
    "\n"
    "}\n" };

int test_enqueue_task(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper output;
    cl_int count;


    if( create_single_kernel_helper( context, &program, &kernel, 1, single_task_kernel, "sample_test" ) )
        return -1;

    // Create args
    count = 100;
    output = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof( cl_int ) * count, NULL, &error );
    test_error( error, "Unable to create output buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( cl_mem ), &output );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( cl_int ), &count );
    test_error( error, "Unable to set kernel argument" );

    // Run task
    error = clEnqueueTask( queue, kernel, 0, NULL, NULL );
    test_error( error, "Unable to run task" );

    // Read results
    cl_int *results = (cl_int*)malloc(sizeof(cl_int)*count);
    error = clEnqueueReadBuffer( queue, output, CL_TRUE, 0, sizeof( cl_int ) * count, results, 0, NULL, NULL );
    test_error( error, "Unable to read results" );

    // Validate
    for( cl_int i = 0; i < count; i++ )
    {
        if( results[ i ] != i )
        {
            log_error( "ERROR: Task result value %d did not validate! Expected %d, got %d\n", (int)i, (int)i, (int)results[ i ] );
            free(results);
            return -1;
        }
    }

    /* All done */
    free(results);
    return 0;
}



#define TEST_SIZE 1000
int test_repeated_setup_cleanup(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{

    cl_context local_context;
    cl_command_queue local_queue;
    cl_program local_program;
    cl_kernel local_kernel;
    cl_mem local_mem_in, local_mem_out;
    cl_event local_event;
    size_t global_dim[3];
    int i, j, error;
    global_dim[0] = TEST_SIZE;
    global_dim[1] = 1; global_dim[2] = 1;
    cl_int *inData, *outData;
    cl_int status;

    inData = (cl_int*)malloc(sizeof(cl_int)*TEST_SIZE);
    outData = (cl_int*)malloc(sizeof(cl_int)*TEST_SIZE);
    for (i=0; i<TEST_SIZE; i++) {
        inData[i] = i;
    }


    for (i=0; i<100; i++) {
        memset(outData, 0, sizeof(cl_int)*TEST_SIZE);

        local_context = clCreateContext(NULL, 1, &deviceID, notify_callback, NULL, &error);
        test_error( error, "clCreateContext failed");

        local_queue = clCreateCommandQueue(local_context, deviceID, 0, &error);
        test_error( error, "clCreateCommandQueue failed");

        error = create_single_kernel_helper(local_context, &local_program, NULL, 1, &repeate_test_kernel, NULL);
        test_error( error, "Unable to build test program" );

        local_kernel = clCreateKernel(local_program, "test_kernel", &error);
        test_error( error, "clCreateKernel failed");

        local_mem_in = clCreateBuffer(local_context, CL_MEM_READ_ONLY, TEST_SIZE*sizeof(cl_int), NULL, &error);
        test_error( error, "clCreateBuffer failed");

        local_mem_out = clCreateBuffer(local_context, CL_MEM_WRITE_ONLY, TEST_SIZE*sizeof(cl_int), NULL, &error);
        test_error( error, "clCreateBuffer failed");

        error = clEnqueueWriteBuffer(local_queue, local_mem_in, CL_TRUE, 0, TEST_SIZE*sizeof(cl_int), inData, 0, NULL, NULL);
        test_error( error, "clEnqueueWriteBuffer failed");

        error = clEnqueueWriteBuffer(local_queue, local_mem_out, CL_TRUE, 0, TEST_SIZE*sizeof(cl_int), outData, 0, NULL, NULL);
        test_error( error, "clEnqueueWriteBuffer failed");

        error = clSetKernelArg(local_kernel, 0, sizeof(local_mem_in), &local_mem_in);
        test_error( error, "clSetKernelArg failed");

        error = clSetKernelArg(local_kernel, 1, sizeof(local_mem_out), &local_mem_out);
        test_error( error, "clSetKernelArg failed");

        error = clEnqueueNDRangeKernel(local_queue, local_kernel, 1, NULL, global_dim, NULL, 0, NULL, &local_event);
        test_error( error, "clEnqueueNDRangeKernel failed");

        error = clWaitForEvents(1, &local_event);
        test_error( error, "clWaitForEvents failed");

        error = clGetEventInfo(local_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
        test_error( error, "clGetEventInfo failed");

        if (status != CL_COMPLETE) {
            log_error( "Kernel execution not complete: status %d.\n", status);
            free(inData);
            free(outData);
            return -1;
        }

        error = clEnqueueReadBuffer(local_queue, local_mem_out, CL_TRUE, 0, TEST_SIZE*sizeof(cl_int), outData, 0, NULL, NULL);
        test_error( error, "clEnqueueReadBuffer failed");

        clReleaseEvent(local_event);
        clReleaseMemObject(local_mem_in);
        clReleaseMemObject(local_mem_out);
        clReleaseKernel(local_kernel);
        clReleaseProgram(local_program);
        clReleaseCommandQueue(local_queue);
        clReleaseContext(local_context);

        for (j=0; j<TEST_SIZE; j++) {
            if (outData[j] != inData[j] + 1) {
                log_error("Results failed to validate at iteration %d. %d != %d.\n", i, outData[j], inData[j] + 1);
                free(inData);
                free(outData);
                return -1;
            }
        }
    }

    free(inData);
    free(outData);

    return 0;
}



