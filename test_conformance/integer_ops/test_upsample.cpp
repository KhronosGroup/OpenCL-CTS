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

static const int vector_sizes[] = {1, 2, 3, 4, 8, 16};
#define NUM_VECTOR_SIZES 6

const char *permute_2_param_kernel_pattern =
"__kernel void test_upsample(__global %s *sourceA, __global %s *sourceB, __global %s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid] );\n"
"\n"
"}\n";


const char *permute_2_param_kernel_pattern_v3srcdst =
"__kernel void test_upsample(__global %s *sourceA, __global %s *sourceB, __global %s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    vstore3( %s( vload3(tid,sourceA), vload3(tid, sourceB) ), tid, destValues);\n"
"\n"
"}\n";

int test_upsample_2_param_fn(cl_command_queue queue, cl_context context, const char *fnName, ExplicitType sourceAType, ExplicitType sourceBType, ExplicitType outType,
                            size_t sourceAVecSize, size_t sourceBVecSize, size_t outVecSize, size_t count,
                            void *sourceA, void *sourceB, void *expectedResults )
{
    cl_program program;
    cl_kernel kernel;
    int error, retCode = 0;
    cl_mem streams[3];
    void *outData;
    size_t threadSize, groupSize, i;
    unsigned char *expectedPtr, *outPtr;
    size_t sourceATypeSize, sourceBTypeSize, outTypeSize, outStride;
    char programSource[ 10240 ], aType[ 64 ], bType[ 64 ], tType[ 64 ];
    const char *progPtr;


    sourceATypeSize = get_explicit_type_size( sourceAType );
    sourceBTypeSize = get_explicit_type_size( sourceBType );
    outTypeSize = get_explicit_type_size( outType );

    outStride = outTypeSize * outVecSize;
    outData = malloc( outStride * count );

    /* Construct the program */
    strcpy( aType, get_explicit_type_name( sourceAType ) );
    strcpy( bType, get_explicit_type_name( sourceBType ) );
    strcpy( tType, get_explicit_type_name( outType ) );
    if( sourceAVecSize > 1 && sourceAVecSize != 3)
        sprintf( aType + strlen( aType ), "%d", (int)sourceAVecSize );
    if( sourceBVecSize > 1  && sourceBVecSize != 3)
        sprintf( bType + strlen( bType ), "%d", (int)sourceBVecSize );
    if( outVecSize > 1  && outVecSize != 3)
        sprintf( tType + strlen( tType ), "%d", (int)outVecSize );

    if(sourceAVecSize == 3 && sourceBVecSize == 3 && outVecSize == 3)
    {
        // permute_2_param_kernel_pattern_v3srcdst
        sprintf( programSource, permute_2_param_kernel_pattern_v3srcdst, aType, bType, tType, fnName );
    }
    else if(sourceAVecSize != 3 && sourceBVecSize != 3 && outVecSize != 3)
    {
    sprintf( programSource, permute_2_param_kernel_pattern, aType, bType, tType, fnName );
    } else {
        vlog_error("Not implemented for %d,%d -> %d\n",
                   (int)sourceAVecSize, (int)sourceBVecSize, (int)outVecSize);
        return -1;
    }

    progPtr = (const char *)programSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, &progPtr, "test_upsample" ) )
    {
        free( outData );
        return -1;
    }

    /* Set up parameters */
    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sourceATypeSize * sourceAVecSize * count, sourceA, NULL);
    if (!streams[0])
    {
        log_error("ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sourceBTypeSize * sourceBVecSize * count, sourceB, NULL);
    if (!streams[1])
    {
        log_error("ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, outStride * count,
                                NULL, NULL);
    if (!streams[2])
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg(kernel, 2, sizeof( streams[2] ), &streams[2] );
    test_error( error, "Unable to set kernel arguments" );

    /* Run the kernel */
    threadSize = count;

    error = get_max_common_work_group_size( context, kernel, threadSize, &groupSize );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, &threadSize, &groupSize, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now verify the results. Each value should have been duplicated four times, and we should be able to just
     do a memcpy instead of relying on the actual type of data */
    error = clEnqueueReadBuffer( queue, streams[2], CL_TRUE, 0, outStride * count, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output values!" );

    expectedPtr = (unsigned char *)expectedResults;
    outPtr = (unsigned char *)outData;

    for( i = 0; i < count; i++ )
    {
        if( memcmp( outPtr, expectedPtr, outTypeSize * outVecSize ) != 0 )
        {
            log_error( "ERROR: Output value %d does not validate!\n", (int)i );
            retCode = -1;
            break;
        }
        expectedPtr += outTypeSize * outVecSize;
        outPtr += outStride;
    }

    clReleaseMemObject( streams[0] );
    clReleaseMemObject( streams[1] );
    clReleaseMemObject( streams[2] );
    clReleaseKernel( kernel );
    clReleaseProgram( program );
    free( outData );

    return retCode;
}

void * create_upsample_data( ExplicitType type, void *sourceA, void *sourceB, size_t count )
{
    void *outData;
    size_t i, tSize;

    tSize = get_explicit_type_size( type );
    outData = malloc( tSize * count * 2 );

    switch( tSize )
    {
        case 1:
            {
                const cl_uchar *aPtr = (const cl_uchar *) sourceA;
                const cl_uchar *bPtr = (const cl_uchar *) sourceB;
                cl_ushort       *dPtr = (cl_ushort*) outData;
                for( i = 0; i < count; i++ )
                {
                    cl_ushort u = *bPtr++;
                    u |= ((cl_ushort) *aPtr++) << 8;
                    *dPtr++ = u;
                }
            }
            break;
        case 2:
            {
                const cl_ushort *aPtr = (const cl_ushort *) sourceA;
                const cl_ushort *bPtr = (const cl_ushort *) sourceB;
                cl_uint       *dPtr = (cl_uint*) outData;
                for( i = 0; i < count; i++ )
                {
                    cl_uint u = *bPtr++;
                    u |= ((cl_uint) *aPtr++) << 16;
                    *dPtr++ = u;
                }
            }
            break;
        case 4:
            {
                const cl_uint *aPtr = (const cl_uint *) sourceA;
                const cl_uint *bPtr = (const cl_uint *) sourceB;
                cl_ulong       *dPtr = (cl_ulong*) outData;
                for( i = 0; i < count; i++ )
                {
                    cl_ulong u = *bPtr++;
                    u |= ((cl_ulong) *aPtr++) << 32;
                    *dPtr++ = u;
                }
            }
            break;
        default:
            log_error("ERROR: unknown type size: %zu\n", tSize);
            return NULL;
    }

    return outData;
}

int test_integer_upsample(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    ExplicitType typesToTest[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kNumExplicitTypes };
    ExplicitType baseTypes[] = { kUChar, kUChar, kUShort, kUShort, kUInt, kUInt, kNumExplicitTypes };
    ExplicitType outTypes[] = { kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    int i, err = 0;
    int sizeIndex;
    size_t size;
    void *sourceA, *sourceB, *expected;
    RandomSeed seed(gRandomSeed );

    for( i = 0; typesToTest[ i ] != kNumExplicitTypes; i++ )
    {
        if ((outTypes[i] == kLong || outTypes[i] == kULong) && !gHasLong)
        {
            log_info( "Longs unsupported on this device. Skipping...\n");
            continue;
        }

        for( sizeIndex = 0; sizeIndex < NUM_VECTOR_SIZES; sizeIndex++)
        {
            size = (size_t)vector_sizes[sizeIndex];
            log_info("running upsample test for %s %s vector size %d\n", get_explicit_type_name(typesToTest[i]), get_explicit_type_name(baseTypes[i]), (int)size);
            sourceA = create_random_data( typesToTest[ i ], seed, 256 );
            sourceB = create_random_data( baseTypes[ i ], seed, 256 );
            expected = create_upsample_data( typesToTest[ i ], sourceA, sourceB, 256 );

            if( test_upsample_2_param_fn( queue, context, "upsample",
                                          typesToTest[ i ], baseTypes[ i ],
                                          outTypes[ i ],
                                          size, size, size,
                                          256 / size,
                                          sourceA, sourceB, expected ) != 0 )
            {
                log_error( "TEST FAILED: %s for %s%d\n", "upsample", get_explicit_type_name( typesToTest[ i ] ), (int)size );
                err = -1;
            }
            free( sourceA );
            free( sourceB );
            free( expected );
        }
    }
    return err;
}


