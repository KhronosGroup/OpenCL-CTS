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

// This test is designed to stress changing kernel arguments between execute calls (that are asynchronous and thus
// potentially overlapping) to make sure each kernel gets the right arguments

// Note: put a delay loop in the kernel to make sure we have time to queue the next kernel before this one finishes
const char *inspect_image_kernel_source[] = {
"__kernel void sample_test(read_only image2d_t src, __global int *outDimensions )\n"
"{\n"
"    int tid = get_global_id(0), i;\n"
"     for( i = 0; i < 100000; i++ ); \n"
"    outDimensions[tid * 2] = get_image_width(src) * tid;\n"
"    outDimensions[tid * 2 + 1] = get_image_height(src) * tid;\n"
"\n"
"}\n" };

#define NUM_TRIES    100
#define NUM_THREADS 2048

REGISTER_TEST(kernel_arg_changes)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    int error, i;
    clMemWrapper images[ NUM_TRIES ];
    size_t         sizes[ NUM_TRIES ][ 2 ];
    clMemWrapper results[ NUM_TRIES ];
    cl_image_format    imageFormat;
    size_t maxWidth, maxHeight;
    size_t threads[1], localThreads[1];
    cl_int resultArray[ NUM_THREADS * 2 ];
    char errStr[ 128 ];
    RandomSeed seed( gRandomSeed );


    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    // Just get any ol format to test with
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE, 0, &imageFormat );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    // Create our testing kernel
    error = create_single_kernel_helper( context, &program, &kernel, 1, inspect_image_kernel_source, "sample_test" );
    test_error( error, "Unable to create testing kernel" );

    // Get max dimensions for each of our images
    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    test_error( error, "Unable to get max image dimensions for device" );

    // Get the number of threads we'll be able to run
    threads[0] = NUM_THREADS;
    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size for kernel" );

    // Create a variety of images and output arrays
    for( i = 0; i < NUM_TRIES; i++ )
    {
        sizes[ i ][ 0 ] = genrand_int32(seed) % (maxWidth/32) + 1;
        sizes[ i ][ 1 ] = genrand_int32(seed) % (maxHeight/32) + 1;

        images[i] = create_image_2d(context, CL_MEM_READ_ONLY, &imageFormat,
                                    sizes[i][0], sizes[i][1], 0, NULL, &error);
        if( images[i] == NULL )
        {
            log_error("Failed to create image %d of size %d x %d (%s).\n", i, (int)sizes[i][0], (int)sizes[i][1], IGetErrorString( error ));
            return -1;
        }
        results[i] =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(cl_int) * threads[0] * 2, NULL, &error);
        if( results[i] == NULL)
        {
            log_error("Failed to create array %d of size %d.\n", i, (int)threads[0]*2);
            return -1;
        }
    }

    // Start setting arguments and executing kernels
    for( i = 0; i < NUM_TRIES; i++ )
    {
        // Set the arguments for this try
        error = clSetKernelArg( kernel, 0, sizeof( cl_mem ), &images[ i ] );
        sprintf( errStr, "Unable to set argument 0 for kernel try %d", i );
        test_error( error, errStr );

        error = clSetKernelArg( kernel, 1, sizeof( cl_mem ), &results[ i ] );
        sprintf( errStr, "Unable to set argument 1 for kernel try %d", i );
        test_error( error, errStr );

        // Queue up execution
        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
        sprintf( errStr, "Unable to execute kernel try %d", i );
        test_error( error, errStr );
    }

    // Read the results back out, one at a time, and verify
    for( i = 0; i < NUM_TRIES; i++ )
    {
        error = clEnqueueReadBuffer( queue, results[ i ], CL_TRUE, 0, sizeof( cl_int ) * threads[0] * 2, resultArray, 0, NULL, NULL );
        sprintf( errStr, "Unable to read results for kernel try %d", i );
        test_error( error, errStr );

        // Verify. Each entry should be n * the (width/height) of image i
        for( int j = 0; j < NUM_THREADS; j++ )
        {
            if( resultArray[ j * 2 + 0 ] != (int)sizes[ i ][ 0 ] * j )
            {
                log_error( "ERROR: Verficiation for kernel try %d, sample %d FAILED, expected a width of %d, got %d\n",
                          i, j, (int)sizes[ i ][ 0 ] * j, resultArray[ j * 2 + 0 ] );
                return -1;
            }
            if( resultArray[ j * 2 + 1 ] != (int)sizes[ i ][ 1 ] * j )
            {
                log_error( "ERROR: Verficiation for kernel try %d, sample %d FAILED, expected a height of %d, got %d\n",
                          i, j, (int)sizes[ i ][ 1 ] * j, resultArray[ j * 2 + 1 ] );
                return -1;
            }
        }
    }

    // If we got here, everything verified successfully
    return 0;
}
