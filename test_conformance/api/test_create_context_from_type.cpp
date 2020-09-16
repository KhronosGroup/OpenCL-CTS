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

#ifndef _WIN32
#include <unistd.h>
#endif

#include "harness/conversions.h"

int test_create_context_from_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper    streams[2];
    clContextWrapper context_to_test;
    clCommandQueueWrapper queue_to_test;
    size_t    threads[1], localThreads[1];
    cl_float inputData[10];
    cl_int outputData[10];
    int i;
    RandomSeed seed( gRandomSeed );

    const char *sample_single_test_kernel[] = {
    "__kernel void sample_test(__global float *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (int)src[tid];\n"
    "\n"
    "}\n" };

    cl_device_type type;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_TYPE failed\n");

    cl_platform_id platform;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed\n");

    cl_context_properties properties[3] = {
      (cl_context_properties)CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform,
      0
    };

    context_to_test = clCreateContextFromType(properties, type, notify_callback, NULL, &error);
    test_error(error, "clCreateContextFromType failed");
    if (context_to_test == NULL) {
        log_error("clCreateContextFromType returned NULL, but error was CL_SUCCESS.");
        return -1;
    }

    queue_to_test = clCreateCommandQueue(context_to_test, deviceID, 0, &error);
    test_error(error, "clCreateCommandQueue failed");
    if (queue_to_test == NULL) {
        log_error("clCreateCommandQueue returned NULL, but error was CL_SUCCESS.");
        return -1;
    }

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context_to_test, &program, &kernel, 1, sample_single_test_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer(context_to_test, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context_to_test, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * 10, NULL, &error);
    test_error( error, "Creating test array failed" );

    /* Write some test data */
    memset( outputData, 0, sizeof( outputData ) );

    for (i=0; i<10; i++)
        inputData[i] = get_random_float(-(float) 0x7fffffff, (float) 0x7fffffff, seed);

    error = clEnqueueWriteBuffer(queue_to_test, streams[0], CL_TRUE, 0, sizeof(cl_float)*10, (void *)inputData, 0, NULL, NULL);
    test_error( error, "Unable to set testing kernel data" );

    /* Test setting the arguments by index manually */
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = (size_t)10;

    error = get_max_common_work_group_size( context_to_test, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue_to_test, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Kernel execution failed" );

    error = clEnqueueReadBuffer( queue_to_test, streams[1], CL_TRUE, 0, sizeof(cl_int)*10, (void *)outputData, 0, NULL, NULL );
    test_error( error, "Unable to get result data" );

    for (i=0; i<10; i++)
    {
        if (outputData[i] != (int)inputData[i])
        {
            log_error( "ERROR: Data did not verify on first pass!\n" );
            return -1;
        }
    }

  return 0;
}


