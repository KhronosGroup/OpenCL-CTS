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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"

const char *constant_source_kernel_code[] = {
"__constant int outVal = 42;\n"
"__constant int outIndex = 7;\n"
"__constant int outValues[ 16 ] = { 17, 01, 11, 12, 1955, 11, 5, 1985, 113, 1, 24, 1984, 7, 23, 1979, 97 };\n"
"\n"
"__kernel void constant_kernel( __global int *out )\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    if( tid == 0 )\n"
"    {\n"
"        out[ 0 ] = outVal;\n"
"        out[ 1 ] = outValues[ outIndex ];\n"
"    }\n"
"    else\n"
"    {\n"
"        out[ tid + 1 ] = outValues[ tid ];\n"
"    }\n"
"}\n" };

int test_constant_source(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    clProgramWrapper program;
    clKernelWrapper kernel;

    clMemWrapper outStream;
    cl_int         outValues[ 17 ];
    cl_int         expectedValues[ 17 ] = { 42, 1985, 01, 11, 12, 1955, 11, 5, 1985, 113, 1, 24, 1984, 7, 23, 1979, 97 };

    cl_int        error;


    // Create a kernel to test with
    error = create_single_kernel_helper( context, &program, &kernel, 1, constant_source_kernel_code, "constant_kernel" );
    test_error( error, "Unable to create testing kernel" );

    // Create our output buffer
    outStream = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof( outValues ), NULL, &error );
    test_error( error, "Unable to create output buffer" );

    // Set the argument
    error = clSetKernelArg( kernel, 0, sizeof( outStream ), &outStream );
    test_error( error, "Unable to set kernel argument" );

    // Run test kernel
    size_t threads[ 1 ] = { 16 };
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Unable to enqueue kernel" );

    // Read results
    error = clEnqueueReadBuffer( queue, outStream, CL_TRUE, 0, sizeof( outValues ), outValues, 0, NULL, NULL );
    test_error( error, "Unable to read results" );

    // Verify results
    for( int i = 0; i < 17; i++ )
    {
        if( expectedValues[ i ] != outValues[ i ] )
        {
            if( i == 0 )
                log_error( "ERROR: Output value %d from constant source global did not validate! (Expected %d, got %d)\n", i, expectedValues[ i ], outValues[ i ] );
            else if( i == 1 )
                log_error( "ERROR: Output value %d from constant-indexed constant array did not validate! (Expected %d, got %d)\n", i, expectedValues[ i ], outValues[ i ] );
            else
                log_error( "ERROR: Output value %d from variable-indexed constant array did not validate! (Expected %d, got %d)\n", i, expectedValues[ i ], outValues[ i ] );
            return -1;
        }
    }

    return 0;
}





