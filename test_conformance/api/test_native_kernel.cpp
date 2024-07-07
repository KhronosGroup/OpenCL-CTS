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

#ifndef _WIN32
#include <unistd.h>
#endif

#include "harness/conversions.h"

static void CL_CALLBACK test_native_kernel_fn( void *userData )
{
    struct arg_struct {
        cl_int * source;
        cl_int * dest;
        cl_int count;
    } *args = (arg_struct *)userData;

    for( cl_int i = 0; i < args->count; i++ )
        args->dest[ i ] = args->source[ i ];
}

int test_native_kernel(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    int error;
    RandomSeed seed( gRandomSeed );
    // Check if we support native kernels
    cl_device_exec_capabilities capabilities;
    error = clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(capabilities), &capabilities, NULL);
    if (!(capabilities & CL_EXEC_NATIVE_KERNEL)) {
        log_info("Device does not support CL_EXEC_NATIVE_KERNEL.\n");
        return 0;
    }

    clMemWrapper streams[ 2 ];
    std::vector<cl_int> inBuffer(n_elems), outBuffer(n_elems);
    clEventWrapper finishEvent;

    struct arg_struct
    {
        cl_mem inputStream;
        cl_mem outputStream;
        cl_int count;
    } args;


    // Create some input values
    generate_random_data(kInt, n_elems, seed, inBuffer.data());

    // Create I/O streams
    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, n_elems * sizeof(cl_int),
                       inBuffer.data(), &error);
    test_error( error, "Unable to create I/O stream" );
    streams[ 1 ] = clCreateBuffer( context, 0, n_elems * sizeof(cl_int), NULL, &error );
    test_error( error, "Unable to create I/O stream" );


    // Set up the arrays to call with
    args.inputStream = streams[ 0 ];
    args.outputStream = streams[ 1 ];
    args.count = n_elems;

    void * memLocs[ 2 ] = { &args.inputStream, &args.outputStream };


    // Run the kernel
    error = clEnqueueNativeKernel( queue, test_native_kernel_fn,
                                      &args, sizeof( args ),
                                      2, &streams[ 0 ],
                                      (const void **)memLocs,
                                      0, NULL, &finishEvent );
    test_error( error, "Unable to queue native kernel" );

    // Finish and wait for the kernel to complete
    error = clFinish( queue );
    test_error(error, "clFinish failed");

    error = clWaitForEvents( 1, &finishEvent );
    test_error(error, "clWaitForEvents failed");

    // Now read the results and verify
    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0,
                                n_elems * sizeof(cl_int), outBuffer.data(), 0,
                                NULL, NULL);
    test_error( error, "Unable to read results" );

    for( int i = 0; i < n_elems; i++ )
    {
        if (inBuffer[i] != outBuffer[i])
        {
            log_error("ERROR: Data sample %d for native kernel did not "
                      "validate (expected %d, got %d)\n",
                      i, (int)inBuffer[i], (int)outBuffer[i]);
            return 1;
        }
    }

    return 0;
}





