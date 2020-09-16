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

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "harness/compat.h"

int test_release_kernel_order(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_program program;
    cl_kernel kernel;
    int error;
    const char *testProgram[] = { "__kernel void sample_test(__global int *data){}" };

    /* Create a test program */
    error = create_single_kernel_helper(context, &program, NULL, 1, testProgram, NULL);
    test_error( error, "Unable to build sample program to test with" );

    /* And create a kernel from it */
    kernel = clCreateKernel( program, "sample_test", &error );
    test_error( error, "Unable to create kernel" );

    /* Now try freeing the program first, then the kernel. If refcounts are right, this should work just fine */
    clReleaseProgram( program );
    clReleaseKernel( kernel );

    /* If we got here fine, we succeeded. If not, well, we won't be able to return an error :) */
    return 0;
}

const char *sample_delay_kernel[] = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    for( int i = 0; i < 1000000; i++ ); \n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n" };

int test_release_during_execute( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_program program;
    cl_kernel kernel;
    cl_mem streams[2];
    size_t threads[1] = { 10 }, localThreadSize;


    /* We now need an event to test. So we'll execute a kernel to get one */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_delay_kernel, "sample_test" ) )
    {
        return -1;
    }

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[ 0 ]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[ 1 ]);
    test_error( error, "Unable to set indexed kernel arguments" );

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreadSize );
    test_error( error, "Unable to calc local thread size" );


    /* Execute the kernel */
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, &localThreadSize, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* The kernel should still be executing, but we should still be able to release it. It's not terribly
       useful, but we should be able to do it, if the internal refcounting is indeed correct. */

    clReleaseMemObject( streams[ 1 ] );
    clReleaseMemObject( streams[ 0 ] );
    clReleaseKernel( kernel );
    clReleaseProgram( program );

  /* Now make sure we're really finished before we go on. */
  error = clFinish(queue);
  test_error( error, "Unable to finish context.");

    return 0;
}


