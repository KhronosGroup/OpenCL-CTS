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
#include "harness/conversions.h"

// This test is designed to stress passing multiple vector parameters to kernels and verifying access between them all

const char *multi_arg_kernel_source_pattern =
"__kernel void sample_test(__global %s *src1, __global %s *src2, __global %s *src3, __global %s *dst1, __global %s *dst2, __global %s *dst3 )\n"
"{\n"
"    int tid = get_global_id(0);\n"
"    dst1[tid] = src1[tid];\n"
"    dst2[tid] = src2[tid];\n"
"    dst3[tid] = src3[tid];\n"
"}\n";

#define MAX_ERROR_TOLERANCE 0.0005f

int test_multi_arg_set(cl_device_id device, cl_context context, cl_command_queue queue,
                       ExplicitType vec1Type, int vec1Size,
                       ExplicitType vec2Type, int vec2Size,
                       ExplicitType vec3Type, int vec3Size, MTdata d)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    int error, i, j;
    clMemWrapper streams[ 6 ];
    size_t threads[1], localThreads[1];
    char programSrc[ 10248 ], vec1Name[ 64 ], vec2Name[ 64 ], vec3Name[ 64 ];
    char sizeNames[][ 4 ] = { "", "2", "3", "4", "", "", "", "8" };
    const char *ptr;
    void *initData[3], *resultData[3];


    // Create the program source
    sprintf( vec1Name, "%s%s", get_explicit_type_name( vec1Type ), sizeNames[ vec1Size - 1 ] );
    sprintf( vec2Name, "%s%s", get_explicit_type_name( vec2Type ), sizeNames[ vec2Size - 1 ] );
    sprintf( vec3Name, "%s%s", get_explicit_type_name( vec3Type ), sizeNames[ vec3Size - 1 ] );

    sprintf( programSrc, multi_arg_kernel_source_pattern,
            vec1Name, vec2Name, vec3Name, vec1Name, vec2Name, vec3Name,
            vec1Size, vec1Size, vec2Size, vec2Size, vec3Size, vec3Size );
    ptr = programSrc;

    // Create our testing kernel
    error = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "sample_test" );
    test_error( error, "Unable to create testing kernel" );

    // Get thread dimensions
    threads[0] = 1024;
    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size for kernel" );

    // Create input streams
    initData[ 0 ] = create_random_data( vec1Type, d, (unsigned int)threads[ 0 ] * vec1Size );
    streams[ 0 ] = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), get_explicit_type_size( vec1Type ) * threads[0] * vec1Size, initData[ 0 ], &error );
    test_error( error, "Unable to create testing stream" );

    initData[ 1 ] = create_random_data( vec2Type, d, (unsigned int)threads[ 0 ] * vec2Size );
    streams[ 1 ] = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), get_explicit_type_size( vec2Type ) * threads[0] * vec2Size, initData[ 1 ], &error );
    test_error( error, "Unable to create testing stream" );

    initData[ 2 ] = create_random_data( vec3Type, d, (unsigned int)threads[ 0 ] * vec3Size );
    streams[ 2 ] = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), get_explicit_type_size( vec3Type ) * threads[0] * vec3Size, initData[ 2 ], &error );
    test_error( error, "Unable to create testing stream" );

    streams[ 3 ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  get_explicit_type_size( vec1Type ) * threads[0] * vec1Size, NULL, &error );
    test_error( error, "Unable to create testing stream" );

    streams[ 4 ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  get_explicit_type_size( vec2Type ) * threads[0] * vec2Size, NULL, &error );
    test_error( error, "Unable to create testing stream" );

    streams[ 5 ] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  get_explicit_type_size( vec3Type ) * threads[0] * vec3Size, NULL, &error );
    test_error( error, "Unable to create testing stream" );

    // Set the arguments
    error = 0;
    for( i = 0; i < 6; i++ )
        error |= clSetKernelArg( kernel, i, sizeof( cl_mem ), &streams[ i ] );
    test_error( error, "Unable to set arguments for kernel" );

    // Execute!
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute kernel" );

    // Read results
    resultData[0] = malloc( get_explicit_type_size( vec1Type ) * vec1Size * threads[0] );
    resultData[1] = malloc( get_explicit_type_size( vec2Type ) * vec2Size * threads[0] );
    resultData[2] = malloc( get_explicit_type_size( vec3Type ) * vec3Size * threads[0] );
    error = clEnqueueReadBuffer( queue, streams[ 3 ], CL_TRUE, 0, get_explicit_type_size( vec1Type ) * vec1Size * threads[ 0 ], resultData[0], 0, NULL, NULL );
    error |= clEnqueueReadBuffer( queue, streams[ 4 ], CL_TRUE, 0, get_explicit_type_size( vec2Type ) * vec2Size * threads[ 0 ], resultData[1], 0, NULL, NULL );
    error |= clEnqueueReadBuffer( queue, streams[ 5 ], CL_TRUE, 0, get_explicit_type_size( vec3Type ) * vec3Size * threads[ 0 ], resultData[2], 0, NULL, NULL );
    test_error( error, "Unable to read result stream" );

    // Verify
    char *ptr1 = (char *)initData[ 0 ], *ptr2 = (char *)resultData[ 0 ];
    size_t span = get_explicit_type_size( vec1Type );
    for( i = 0; i < (int)threads[0]; i++ )
    {
        for( j = 0; j < vec1Size; j++ )
        {
            if( memcmp( ptr1 + span * j , ptr2 + span * j, span ) != 0 )
            {
                log_error( "ERROR: Value did not validate for component %d of item %d of stream 0!\n", j, i );
                free( initData[ 0 ] );
                free( initData[ 1 ] );
                free( initData[ 2 ] );
                free( resultData[ 0 ] );
                free( resultData[ 1 ] );
                free( resultData[ 2 ] );
                return -1;
            }
        }
        ptr1 += span * vec1Size;
        ptr2 += span * vec1Size;
    }

    ptr1 = (char *)initData[ 1 ];
    ptr2 = (char *)resultData[ 1 ];
    span = get_explicit_type_size( vec2Type );
    for( i = 0; i < (int)threads[0]; i++ )
    {
        for( j = 0; j < vec2Size; j++ )
        {
            if( memcmp( ptr1 + span * j , ptr2 + span * j, span ) != 0 )
            {
                log_error( "ERROR: Value did not validate for component %d of item %d of stream 1!\n", j, i );
                free( initData[ 0 ] );
                free( initData[ 1 ] );
                free( initData[ 2 ] );
                free( resultData[ 0 ] );
                free( resultData[ 1 ] );
                free( resultData[ 2 ] );
                return -1;
            }
        }
        ptr1 += span * vec2Size;
        ptr2 += span * vec2Size;
    }

    ptr1 = (char *)initData[ 2 ];
    ptr2 = (char *)resultData[ 2 ];
    span = get_explicit_type_size( vec3Type );
    for( i = 0; i < (int)threads[0]; i++ )
    {
        for( j = 0; j < vec3Size; j++ )
        {
            if( memcmp( ptr1 + span * j , ptr2 + span * j, span ) != 0 )
            {
                log_error( "ERROR: Value did not validate for component %d of item %d of stream 2!\n", j, i );
                free( initData[ 0 ] );
                free( initData[ 1 ] );
                free( initData[ 2 ] );
                free( resultData[ 0 ] );
                free( resultData[ 1 ] );
                free( resultData[ 2 ] );
                return -1;
            }
        }
        ptr1 += span * vec3Size;
        ptr2 += span * vec3Size;
    }

    // If we got here, everything verified successfully
    free( initData[ 0 ] );
    free( initData[ 1 ] );
    free( initData[ 2 ] );
    free( resultData[ 0 ] );
    free( resultData[ 1 ] );
    free( resultData[ 2 ] );

    return 0;
}

int test_kernel_arg_multi_setup_exhaustive(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    // Loop through every combination of input and output types
    ExplicitType types[] = { kChar, kShort, kInt, kFloat, kNumExplicitTypes };
    int type1, type2, type3;
    int size1, size2, size3;
    RandomSeed seed( gRandomSeed );

    log_info( "\n" ); // for formatting

    for( type1 = 0; types[ type1 ] != kNumExplicitTypes; type1++ )
    {
        for( type2 = 0; types[ type2 ] != kNumExplicitTypes; type2++ )
        {
            for( type3 = 0; types[ type3 ] != kNumExplicitTypes; type3++ )
            {
                log_info( "\n\ttesting %s, %s, %s...", get_explicit_type_name( types[ type1 ] ), get_explicit_type_name( types[ type2 ] ), get_explicit_type_name( types[ type3 ] ) );

                // Loop through every combination of vector size
                for( size1 = 2; size1 <= 8; size1 <<= 1 )
                {
                    for( size2 = 2; size2 <= 8; size2 <<= 1 )
                    {
                        for( size3 = 2; size3 <= 8; size3 <<= 1 )
                        {
                            log_info(".");
                            fflush( stdout);
                            if( test_multi_arg_set( device, context, queue,
                                                   types[ type1 ], size1,
                                                   types[ type2 ], size2,
                                                   types[ type3 ], size3, seed ) )
                                return -1;
                        }
                    }
                }
            }
        }
    }
    log_info( "\n" );
    return 0;
}

int test_kernel_arg_multi_setup_random(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    // Loop through a selection of combinations
    ExplicitType types[] = { kChar, kShort, kInt, kFloat, kNumExplicitTypes };
    int type1, type2, type3;
    int size1, size2, size3;
    RandomSeed seed( gRandomSeed );

    num_elements = 3*3*3*4;
    log_info( "Testing %d random configurations\n", num_elements );

    // Loop through every combination of vector size
    for( size1 = 2; size1 <= 8; size1 <<= 1 )
    {
        for( size2 = 2; size2 <= 8; size2 <<= 1 )
        {
            for( size3 = 2; size3 <= 8; size3 <<= 1 )
            {
                // Loop through 4 type combinations for each size combination
                int n;
                for (n=0; n<4; n++) {
                    type1 = (int)get_random_float(0,4, seed);
                    type2 = (int)get_random_float(0,4, seed);
                    type3 = (int)get_random_float(0,4, seed);


                    log_info( "\ttesting %s%d, %s%d, %s%d...\n",
                             get_explicit_type_name( types[ type1 ] ), size1,
                             get_explicit_type_name( types[ type2 ] ), size2,
                             get_explicit_type_name( types[ type3 ] ), size3 );

                    if( test_multi_arg_set( device, context, queue,
                                           types[ type1 ], size1,
                                           types[ type2 ], size2,
                                           types[ type3 ], size3, seed ) )
                        return -1;
                }
            }
        }
    }
    return 0;
}




