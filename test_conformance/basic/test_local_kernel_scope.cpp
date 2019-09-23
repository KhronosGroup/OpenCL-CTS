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

#define MAX_LOCAL_STORAGE_SIZE  256
#define MAX_LOCAL_STORAGE_SIZE_STRING "256"

const char *kernelSource[] = {
    "__kernel void test( __global unsigned int * input, __global unsigned int *outMaxes )\n"
    "{\n"
    "   __local unsigned int localStorage[ " MAX_LOCAL_STORAGE_SIZE_STRING " ];\n"
    "   unsigned int theValue = input[ get_global_id( 0 ) ];\n"
    "\n"
    "   // If we just write linearly, there's no verification that the items in a group share local data\n"
    "   // So we write reverse-linearly, which requires items to read the local data written by at least one\n"
    "   // different item\n"
    "   localStorage[ get_local_size( 0 ) - get_local_id( 0 ) - 1 ] = theValue;\n"
    "\n"
    "   // The barrier ensures that all local items have written to the local storage\n"
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    "   // Now we loop back through the local storage and look for the max value. We only do this if\n"
    "   // we're the first item in a group\n"
    "   unsigned int max = 0;\n"
    "   if( get_local_id( 0 ) == 0 )\n"
    "   {\n"
    "       for( size_t i = 0; i < get_local_size( 0 ); i++ )\n"
    "       {\n"
    "           if( localStorage[ i ] > max )\n"
    "               max = localStorage[ i ];\n"
    "       }\n"
    "       outMaxes[ get_group_id( 0 ) ] = max;\n"
    "   }\n"
    "}\n"
};

int test_local_kernel_scope(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 2 ];
    MTdata randSeed = init_genrand( gRandomSeed );

    // Create a test kernel
    error = create_single_kernel_helper( context, &program, &kernel, 1, kernelSource, "test" );
    test_error( error, "Unable to create test kernel" );


    // Determine an appropriate test size
    size_t workGroupSize;
    error = clGetKernelWorkGroupInfo( kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof( workGroupSize ), &workGroupSize, NULL );
    test_error( error, "Unable to obtain kernel work group size" );

    // Make sure the work group size doesn't overrun our local storage size in the kernel
    while( workGroupSize > MAX_LOCAL_STORAGE_SIZE )
        workGroupSize >>= 1;

    size_t testSize = workGroupSize;
    while( testSize < 1024 )
        testSize += workGroupSize;
    size_t numGroups = testSize / workGroupSize;
    log_info( "\tTesting with %ld groups, %ld elements per group...\n", numGroups, workGroupSize );

    // Create two buffers for operation
    cl_uint *inputData = (cl_uint*)malloc( testSize * sizeof(cl_uint) );
    generate_random_data( kUInt, testSize, randSeed, inputData );
    free_mtdata( randSeed );
    streams[ 0 ] = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, testSize * sizeof(cl_uint), inputData, &error );
    test_error( error, "Unable to create input buffer" );

    cl_uint *outputData = (cl_uint*)malloc( numGroups *sizeof(cl_uint) );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_WRITE_ONLY, numGroups * sizeof(cl_uint), NULL, &error );
    test_error( error, "Unable to create output buffer" );


    // Set up the kernel args and run
    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel arg" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel arg" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &testSize, &workGroupSize, 0, NULL, NULL );
    test_error( error, "Unable to enqueue kernel" );


    // Read results and verify
    error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0, numGroups * sizeof(cl_uint), outputData, 0, NULL, NULL );
    test_error( error, "Unable to read output data" );

    // MingW compiler seems to have a bug that otimizes the code below incorrectly.
    // adding the volatile keyword to size_t decleration to avoid aggressive optimization by the compiler.
    for( volatile size_t i = 0; i < numGroups; i++ )
    {
        // Determine the max in our case
        cl_uint localMax = 0;
        for( volatile size_t j = 0; j < workGroupSize; j++ )
        {
            if( inputData[ i * workGroupSize + j ] > localMax )
                localMax = inputData[ i * workGroupSize + j ];
        }

        if( outputData[ i ] != localMax )
        {
            log_error( "ERROR: Local max validation failed! (expected %u, got %u for i=%lu)\n", localMax, outputData[ i ] , i );
            free(inputData);
            free(outputData);
            return -1;
        }
    }

    free(inputData);
    free(outputData);
    return 0;
}


