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
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"


static const char *astype_kernel_pattern =
"%s\n"
"__kernel void test_fn( __global %s%s *src, __global %s%s *dst )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"    %s%s tmp = as_%s%s( src[ tid ] );\n"
"   dst[ tid ] = tmp;\n"
"}\n";

static const char *astype_kernel_pattern_V3srcV3dst =
"%s\n"
"__kernel void test_fn( __global %s *src, __global %s *dst )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"    %s%s tmp = as_%s%s( vload3(tid,src) );\n"
"   vstore3(tmp,tid,dst);\n"
"}\n";
// in the printf, remove the third and fifth argument, each of which
// should be a "3", when copying from the printf for astype_kernel_pattern

static const char *astype_kernel_pattern_V3dst =
"%s\n"
"__kernel void test_fn( __global %s%s *src, __global %s *dst )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"    %s3 tmp = as_%s3( src[ tid ] );\n"
"   vstore3(tmp,tid,dst);\n"
"}\n";
// in the printf, remove the fifth argument, which
// should be a "3", when copying from the printf for astype_kernel_pattern


static const char *astype_kernel_pattern_V3src =
"%s\n"
"__kernel void test_fn( __global %s *src, __global %s%s *dst )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"    %s%s tmp = as_%s%s( vload3(tid,src) );\n"
"   dst[ tid ] = tmp;\n"
"}\n";
// in the printf, remove the third argument, which
// should be a "3", when copying from the printf for astype_kernel_pattern


int test_astype_set( cl_device_id device, cl_context context, cl_command_queue queue, ExplicitType inVecType, ExplicitType outVecType,
                    unsigned int vecSize, unsigned int outVecSize,
                    int numElements )
{
    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 2 ];

    char programSrc[ 10240 ];
    size_t threads[ 1 ], localThreads[ 1 ];
    size_t typeSize = get_explicit_type_size( inVecType );
    size_t outTypeSize = get_explicit_type_size(outVecType);
    char sizeNames[][ 3 ] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    MTdata d;



    // Create program
    if(outVecSize == 3 && vecSize == 3) {
        // astype_kernel_pattern_V3srcV3dst
        sprintf( programSrc, astype_kernel_pattern_V3srcV3dst,
                (outVecType == kDouble || inVecType == kDouble) ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                get_explicit_type_name( inVecType ), // sizeNames[ vecSize ],
                get_explicit_type_name( outVecType ), // sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ] );
    } else if(outVecSize == 3) {
        // astype_kernel_pattern_V3dst
        sprintf( programSrc, astype_kernel_pattern_V3dst,
                (outVecType == kDouble || inVecType == kDouble) ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                get_explicit_type_name( inVecType ), sizeNames[ vecSize ],
                get_explicit_type_name( outVecType ),
                get_explicit_type_name( outVecType ),
                get_explicit_type_name( outVecType ));

    } else if(vecSize == 3) {
        // astype_kernel_pattern_V3src
        sprintf( programSrc, astype_kernel_pattern_V3src,
                (outVecType == kDouble || inVecType == kDouble) ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                get_explicit_type_name( inVecType ),// sizeNames[ vecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ]);
    } else {
        sprintf( programSrc, astype_kernel_pattern,
                (outVecType == kDouble || inVecType == kDouble) ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" : "",
                get_explicit_type_name( inVecType ), sizeNames[ vecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ],
                get_explicit_type_name( outVecType ), sizeNames[ outVecSize ]);
    }

    const char *ptr = programSrc;
    error = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "test_fn" );
    test_error( error, "Unable to create testing kernel" );


    // Create some input values
    size_t inBufferSize = sizeof(char)* numElements * get_explicit_type_size( inVecType ) * vecSize;
    char *inBuffer = (char*)malloc( inBufferSize );
    size_t outBufferSize = sizeof(char)* numElements * get_explicit_type_size( outVecType ) *outVecSize;
    char *outBuffer = (char*)malloc( outBufferSize );

    d = init_genrand( gRandomSeed );
    generate_random_data( inVecType, numElements * vecSize,
                         d, inBuffer );
    free_mtdata(d); d = NULL;

    // Create I/O streams and set arguments
    streams[ 0 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, inBufferSize, inBuffer, &error );
    test_error( error, "Unable to create I/O stream" );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE, outBufferSize, NULL, &error );
    test_error( error, "Unable to create I/O stream" );

    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel argument" );


    // Run the kernel
    threads[ 0 ] = numElements;
    error = get_max_common_work_group_size( context, kernel, threads[ 0 ], &localThreads[ 0 ] );
    test_error( error, "Unable to get group size to run with" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to run kernel" );


    // Get the results and compare
    // The beauty is that astype is supposed to return the bit pattern as a different type, which means
    // the output should have the exact same bit pattern as the input. No interpretation necessary!
    error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0, outBufferSize, outBuffer, 0, NULL, NULL );
    test_error( error, "Unable to read results" );

    char *expected = inBuffer;
    char *actual = outBuffer;
    size_t compSize = typeSize*vecSize;
    if(outTypeSize*outVecSize < compSize) {
        compSize = outTypeSize*outVecSize;
    }

    if(outVecSize == 4 && vecSize == 3)
    {
        // as_type4(vec3) should compile but produce undefined results??
        free(inBuffer);
        free(outBuffer);
        return 0;
    }

    if(outVecSize != 3 && vecSize != 3 && outVecSize != vecSize)
    {
        // as_typen(vecm) should compile and run but produce
        // implementation-defined results for m != n
        // and n*sizeof(type) = sizeof(vecm)
        free(inBuffer);
        free(outBuffer);
        return 0;
    }

    for( int i = 0; i < numElements; i++ )
    {
        if( memcmp( expected, actual, compSize ) != 0 )
        {
            char expectedString[ 1024 ], actualString[ 1024 ];
            log_error( "ERROR: Data sample %d of %d for as_%s%d( %s%d ) did not validate (expected {%s}, got {%s})\n",
                      (int)i, (int)numElements, get_explicit_type_name( outVecType ), vecSize, get_explicit_type_name( inVecType ), vecSize,
                      GetDataVectorString( expected, typeSize, vecSize, expectedString ),
                      GetDataVectorString( actual, typeSize, vecSize, actualString ) );
            log_error("Src is :\n%s\n----\n%d threads %d localthreads\n",
                      programSrc, (int)threads[0],(int) localThreads[0]);
            free(inBuffer);
            free(outBuffer);
            return 1;
        }
        expected += typeSize * vecSize;
        actual += outTypeSize * outVecSize;
    }

    free(inBuffer);
    free(outBuffer);
    return 0;
}

int test_astype(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Note: although casting to different vector element sizes that match the same size (i.e. short2 -> char4) is
    // legal in OpenCL 1.0, the result is dependent on the device it runs on, which means there's no actual way
    // for us to verify what is "valid". So the only thing we can test are types that match in size independent
    // of the element count (char -> uchar, etc)
    ExplicitType vecTypes[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int inTypeIdx, outTypeIdx, sizeIdx, outSizeIdx;
    size_t inTypeSize, outTypeSize;
    int error = 0;

    for( inTypeIdx = 0; vecTypes[ inTypeIdx ] != kNumExplicitTypes; inTypeIdx++ )
    {
        inTypeSize = get_explicit_type_size(vecTypes[inTypeIdx]);

        if( vecTypes[ inTypeIdx ] == kDouble && !is_extension_available( device, "cl_khr_fp64" ) )
            continue;

        if (( vecTypes[ inTypeIdx ] == kLong || vecTypes[ inTypeIdx ] == kULong ) && !gHasLong )
            continue;

        for( outTypeIdx = 0; vecTypes[ outTypeIdx ] != kNumExplicitTypes; outTypeIdx++ )
        {
            outTypeSize = get_explicit_type_size(vecTypes[outTypeIdx]);
            if( vecTypes[ outTypeIdx ] == kDouble && !is_extension_available( device, "cl_khr_fp64" ) ) {
                continue;
            }

            if (( vecTypes[ outTypeIdx ] == kLong || vecTypes[ outTypeIdx ] == kULong ) && !gHasLong )
                continue;

            // change this check
            if( inTypeIdx == outTypeIdx ) {
                continue;
            }

            log_info( " (%s->%s)\n", get_explicit_type_name( vecTypes[ inTypeIdx ] ), get_explicit_type_name( vecTypes[ outTypeIdx ] ) );
            fflush( stdout );

            for( sizeIdx = 0; vecSizes[ sizeIdx ] != 0; sizeIdx++ )
            {

                for(outSizeIdx = 0; vecSizes[outSizeIdx] != 0; outSizeIdx++)
                {
                    if(vecSizes[sizeIdx]*inTypeSize !=
                       vecSizes[outSizeIdx]*outTypeSize )
                    {
                        continue;
                    }
                    error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], vecSizes[ sizeIdx ], vecSizes[outSizeIdx], n_elems );


                }

            }
            if(get_explicit_type_size(vecTypes[inTypeIdx]) ==
               get_explicit_type_size(vecTypes[outTypeIdx])) {
                // as_type3(vec4) allowed, as_type4(vec3) not allowed
                error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], 3, 4, n_elems );
                error += test_astype_set( device, context, queue, vecTypes[ inTypeIdx ], vecTypes[ outTypeIdx ], 4, 3, n_elems );
            }

        }
    }
    return error;
}


