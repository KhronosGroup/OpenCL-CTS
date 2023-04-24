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
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"

// clang-format off

const char *anyAllTestKernelPattern =
"%s\n" // optional pragma
"%s\n" // optional pragma
"__kernel void sample_test(__global %s%s *sourceA, __global int *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid] );\n"
"\n"
"}\n";

const char *anyAllTestKernelPatternVload =
"%s\n" // optional pragma
"%s\n" // optional pragma
"__kernel void sample_test(__global %s%s *sourceA, __global int *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s(vload3(tid, (__global %s *)sourceA));\n" // ugh, almost
"\n"
"}\n";

// clang-format on

#define TEST_SIZE 512

typedef int (*anyAllVerifyFn)( ExplicitType vecType, unsigned int vecSize, void *inData );

int test_any_all_kernel(cl_context context, cl_command_queue queue,
                        const char *fnName, ExplicitType vecType,
                        unsigned int vecSize, anyAllVerifyFn verifyFn,
                        MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    cl_long inDataA[TEST_SIZE * 16], clearData[TEST_SIZE * 16];
    int outData[TEST_SIZE];
    int error, i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4];


    /* Create the source */
    if( g_vector_aligns[vecSize] == 1 ) {
        sizeName[ 0 ] = 0;
    } else {
        sprintf( sizeName, "%d", vecSize );
    }
    log_info("Testing any/all on %s%s\n",
             get_explicit_type_name( vecType ), sizeName);
    if(DENSE_PACK_VECS && vecSize == 3) {
        // anyAllTestKernelPatternVload
        sprintf(
            kernelSource, anyAllTestKernelPatternVload,
            vecType == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                               : "",
            vecType == kHalf ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
                             : "",
            get_explicit_type_name(vecType), sizeName, fnName,
            get_explicit_type_name(vecType));
    } else {
        sprintf(
            kernelSource, anyAllTestKernelPattern,
            vecType == kDouble ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                               : "",
            vecType == kHalf ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
                             : "",
            get_explicit_type_name(vecType), sizeName, fnName);
    }
    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1,
                                    (const char **)&programPtr,
                                    "sample_test" ) )
    {
        return -1;
    }

    /* Generate some streams */
    generate_random_data( vecType, TEST_SIZE * g_vector_aligns[vecSize], d, inDataA );
    memset( clearData, 0, sizeof( clearData ) );

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                get_explicit_type_size(vecType)
                                    * g_vector_aligns[vecSize] * TEST_SIZE,
                                &inDataA, &error);
    if( streams[0] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_int) * g_vector_aligns[vecSize] * TEST_SIZE,
                       clearData, &error);
    if( streams[1] == NULL )
    {
        print_error( error, "Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof( int ) * TEST_SIZE, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < TEST_SIZE; i++ )
    {
        int expected = verifyFn( vecType, vecSize, (char *)inDataA + i * get_explicit_type_size( vecType ) * g_vector_aligns[vecSize] );
        if( expected != outData[ i ] )
        {
            unsigned int *ptr = (unsigned int *)( (char *)inDataA + i * get_explicit_type_size( vecType ) * g_vector_aligns[vecSize] );
            log_error( "ERROR: Data sample %d does not validate! Expected (%d), got (%d), source 0x%08x\n",
                      i, expected, outData[i], *ptr );
            return -1;
        }
    }

    return 0;
}

int anyVerifyFn( ExplicitType vecType, unsigned int vecSize, void *inData )
{
    unsigned int i;
    switch( vecType )
    {
        case kChar:
        {
            char sum = 0;
            char *tData = (char *)inData;
            for( i = 0; i < vecSize; i++ )
                sum |= tData[ i ] & 0x80;
            return (sum != 0) ? 1 : 0;
        }
        case kShort:
        {
            short sum = 0;
            short *tData = (short *)inData;
            for( i = 0; i < vecSize; i++ )
                sum |= tData[ i ] & 0x8000;
            return (sum != 0);
        }
        case kInt:
        {
            cl_int sum = 0;
            cl_int *tData = (cl_int *)inData;
            for( i = 0; i < vecSize; i++ )
                sum |= tData[ i ] & (cl_int)0x80000000L;
            return (sum != 0);
        }
        case kLong:
        {
            cl_long sum = 0;
            cl_long *tData = (cl_long *)inData;
            for( i = 0; i < vecSize; i++ )
                sum |= tData[ i ] & 0x8000000000000000LL;
            return (sum != 0);
        }
        default:
            return 0;
    }
}

int test_relational_any(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    ExplicitType vecType[] = { kChar, kShort, kInt, kLong };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );

    for( typeIndex = 0; typeIndex < 4; typeIndex++ )
    {
        if (vecType[typeIndex] == kLong && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            // Test!
            if( test_any_all_kernel(context, queue, "any", vecType[ typeIndex ], vecSizes[ index ], anyVerifyFn, seed ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( vecType[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

int allVerifyFn( ExplicitType vecType, unsigned int vecSize, void *inData )
{
    unsigned int i;
    switch( vecType )
    {
        case kChar:
        {
            char sum = 0x80;
            char *tData = (char *)inData;
            for( i = 0; i < vecSize; i++ )
                sum &= tData[ i ] & 0x80;
            return (sum != 0) ? 1 : 0;
        }
        case kShort:
        {
            short sum = 0x8000;
            short *tData = (short *)inData;
            for( i = 0; i < vecSize; i++ )
                sum &= tData[ i ] & 0x8000;
            return (sum != 0);
        }
        case kInt:
        {
            cl_int sum = 0x80000000L;
            cl_int *tData = (cl_int *)inData;
            for( i = 0; i < vecSize; i++ )
                sum &= tData[ i ] & (cl_int)0x80000000L;
            return (sum != 0);
        }
        case kLong:
        {
            cl_long sum = 0x8000000000000000LL;
            cl_long *tData = (cl_long *)inData;
            for( i = 0; i < vecSize; i++ )
                sum &= tData[ i ] & 0x8000000000000000LL;
            return (sum != 0);
        }
        default:
            return 0;
    }
}

int test_relational_all(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    ExplicitType vecType[] = { kChar, kShort, kInt, kLong };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );


    for( typeIndex = 0; typeIndex < 4; typeIndex++ )
    {
        if (vecType[typeIndex] == kLong && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            // Test!
            if( test_any_all_kernel(context, queue, "all", vecType[ typeIndex ], vecSizes[ index ], allVerifyFn, seed ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( vecType[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

// clang-format off

const char *selectTestKernelPattern =
"%s\n" // optional pragma
"%s\n" // optional pragma
"__kernel void sample_test(__global %s%s *sourceA, __global %s%s *sourceB, __global %s%s *sourceC, __global %s%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    destValues[tid] = %s( sourceA[tid], sourceB[tid], sourceC[tid] );\n"
"\n"
"}\n";


const char *selectTestKernelPatternVload =
"%s\n" // optional pragma
"%s\n" // optional pragma
"__kernel void sample_test(__global %s%s *sourceA, __global %s%s *sourceB, __global %s%s *sourceC, __global %s%s *destValues)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    %s%s tmp = %s( vload3(tid, (__global %s *)sourceA), vload3(tid, (__global %s *)sourceB), vload3(tid, (__global %s *)sourceC) );\n"
"    vstore3(tmp, tid, (__global %s *)destValues);\n"
"\n"
"}\n";

// clang-format on

typedef void (*selectVerifyFn)( ExplicitType vecType, ExplicitType testVecType, unsigned int vecSize, void *inDataA, void *inDataB, void *inDataTest, void *outData );

int test_select_kernel(cl_context context, cl_command_queue queue, const char *fnName,
                       ExplicitType vecType, unsigned int vecSize, ExplicitType testVecType, selectVerifyFn verifyFn, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[4];
    cl_long inDataA[TEST_SIZE * 16], inDataB[ TEST_SIZE * 16 ], inDataC[ TEST_SIZE * 16 ];
    cl_long outData[TEST_SIZE * 16], expected[16];
    int error, i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;
    char sizeName[4], outSizeName[4];
    unsigned int outVecSize;


    /* Create the source */
    if( vecSize == 1 )
        sizeName[ 0 ] = 0;
    else
        sprintf( sizeName, "%d", vecSize );

    outVecSize = vecSize;

    if( outVecSize == 1 )
        outSizeName[ 0 ] = 0;
    else
        sprintf( outSizeName, "%d", outVecSize );

    if(DENSE_PACK_VECS && vecSize == 3) {
        // anyAllTestKernelPatternVload
        sprintf(kernelSource, selectTestKernelPatternVload,
                (vecType == kDouble || testVecType == kDouble)
                    ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                    : "",
                (vecType == kHalf || testVecType == kHalf)
                    ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
                    : "",
                get_explicit_type_name(vecType), sizeName,
                get_explicit_type_name(vecType), sizeName,
                get_explicit_type_name(testVecType), sizeName,
                get_explicit_type_name(vecType), outSizeName,
                get_explicit_type_name(vecType), sizeName, fnName,
                get_explicit_type_name(vecType),
                get_explicit_type_name(vecType),
                get_explicit_type_name(vecType),
                get_explicit_type_name(testVecType));
    } else {
        sprintf(kernelSource, selectTestKernelPattern,
                (vecType == kDouble || testVecType == kDouble)
                    ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                    : "",
                (vecType == kHalf || testVecType == kHalf)
                    ? "#pragma OPENCL EXTENSION cl_khr_fp16 : enable"
                    : "",
                get_explicit_type_name(vecType), sizeName,
                get_explicit_type_name(vecType), sizeName,
                get_explicit_type_name(testVecType), sizeName,
                get_explicit_type_name(vecType), outSizeName, fnName);
    }

    /* Create kernels */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }

    /* Generate some streams */
    generate_random_data( vecType, TEST_SIZE * g_vector_aligns[vecSize], d, inDataA );
    generate_random_data( vecType, TEST_SIZE * g_vector_aligns[vecSize], d, inDataB );
    generate_random_data( testVecType, TEST_SIZE * g_vector_aligns[vecSize], d, inDataC );

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                get_explicit_type_size(vecType)
                                    * g_vector_aligns[vecSize] * TEST_SIZE,
                                &inDataA, &error);
    if( streams[0] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                get_explicit_type_size(vecType)
                                    * g_vector_aligns[vecSize] * TEST_SIZE,
                                &inDataB, &error);
    if( streams[1] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                get_explicit_type_size(testVecType)
                                    * g_vector_aligns[vecSize] * TEST_SIZE,
                                &inDataC, &error);
    if( streams[2] == NULL )
    {
        print_error( error, "Creating input array A failed!\n");
        return -1;
    }
    streams[3] = clCreateBuffer( context, CL_MEM_READ_WRITE, get_explicit_type_size( vecType ) * g_vector_aligns[outVecSize] * TEST_SIZE, NULL, &error);
    if( streams[3] == NULL )
    {
        print_error( error, "Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 2, sizeof( streams[2] ), &streams[2] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 3, sizeof( streams[3] ), &streams[3] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now get the results */
    error = clEnqueueReadBuffer( queue, streams[3], true, 0, get_explicit_type_size( vecType ) * TEST_SIZE * g_vector_aligns[outVecSize], outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    /* And verify! */
    for( i = 0; i < (int)(TEST_SIZE * g_vector_aligns[vecSize]); i++ )
    {
        if(i%g_vector_aligns[vecSize] >= (int) vecSize) {
            continue;
        }
        verifyFn( vecType, testVecType, vecSize, (char *)inDataA + i * get_explicit_type_size( vecType ),
                 (char *)inDataB + i * get_explicit_type_size( vecType ),
                 (char *)inDataC + i * get_explicit_type_size( testVecType ),
                 expected);

        char *outPtr = (char *)outData;
        outPtr += ( i / g_vector_aligns[vecSize] ) * get_explicit_type_size( vecType ) * g_vector_aligns[outVecSize];
        outPtr += ( i % g_vector_aligns[vecSize] ) * get_explicit_type_size( vecType );
        if( memcmp( expected, outPtr, get_explicit_type_size( vecType ) ) != 0 )
        {
            log_error( "ERROR: Data sample %d:%d does not validate! Expected (0x%08x), got (0x%08x) from (0x%08x) and (0x%08x) with test (0x%08x)\n",
                      i / g_vector_aligns[vecSize],
                      i % g_vector_aligns[vecSize],
                      *( (int *)expected ),
                      *( (int *)( (char *)outData +
                                 i * get_explicit_type_size( vecType
                                                            ) ) ),
                      *( (int *)( (char *)inDataA +
                                 i * get_explicit_type_size( vecType
                                                            ) ) ),
                      *( (int *)( (char *)inDataB +
                                 i * get_explicit_type_size( vecType
                                                            ) ) ),
                      *( (int *)( (char *)inDataC +
                                 i*get_explicit_type_size( testVecType
                                                          ) ) ) );
            int j;
            log_error( "inA: " );
            unsigned char *a = (unsigned char *)( (char *)inDataA + i * get_explicit_type_size( vecType ) );
            unsigned char *b = (unsigned char *)( (char *)inDataB + i * get_explicit_type_size( vecType ) );
            unsigned char *c = (unsigned char *)( (char *)inDataC + i * get_explicit_type_size( testVecType ) );
            unsigned char *e = (unsigned char *)( expected );
            unsigned char *g = (unsigned char *)( (char *)outData + i * get_explicit_type_size( vecType ) );
            for( j = 0; j < 16; j++ )
                log_error( "0x%02x ", a[ j ] );
            log_error( "\ninB: " );
            for( j = 0; j < 16; j++ )
                log_error( "0x%02x ", b[ j ] );
            log_error( "\ninC: " );
            for( j = 0; j < 16; j++ )
                log_error( "0x%02x ", c[ j ] );
            log_error( "\nexp: " );
            for( j = 0; j < 16; j++ )
                log_error( "0x%02x ", e[ j ] );
            log_error( "\ngot: " );
            for( j = 0; j < 16; j++ )
                log_error( "0x%02x ", g[ j ] );
            return -1;
        }
    }

    return 0;
}

void bitselect_verify_fn( ExplicitType vecType, ExplicitType testVecType, unsigned int vecSize, void *inDataA, void *inDataB, void *inDataTest, void *outData )
{
    char *inA = (char *)inDataA, *inB = (char *)inDataB, *inT = (char *)inDataTest, *out = (char *)outData;
    size_t i, numBytes = get_explicit_type_size( vecType );

    // Type is meaningless, this is all bitwise!
    for( i = 0; i < numBytes; i++ )
    {
        out[ i ] = ( inA[ i ] & ~inT[ i ] ) | ( inB[ i ] & inT[ i ] );
    }
}

int test_relational_bitselect(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    constexpr ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort,
                                         kInt,  kUInt,  kLong,  kULong,
                                         kHalf, kFloat, kDouble };
    constexpr auto vecTypeSize = sizeof(vecType) / sizeof(ExplicitType);
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed( gRandomSeed );


    for (typeIndex = 0; typeIndex < vecTypeSize; typeIndex++)
    {
        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong) && !gHasLong)
            continue;

        if (vecType[typeIndex] == kDouble)
        {
            if(!is_extension_available(device, "cl_khr_fp64"))
            {
                log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
                continue;
            }
            else
                log_info("Testing doubles.\n");
        }

        if (vecType[typeIndex] == kHalf)
        {
            if (!is_extension_available(device, "cl_khr_fp16"))
            {
                log_info("Extension cl_khr_fp16 not supported; skipping half "
                         "tests.\n");
                continue;
            }
            else
                log_info("Testing halfs.\n");
        }

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            // Test!
            if( test_select_kernel(context, queue, "bitselect", vecType[ typeIndex ], vecSizes[ index ], vecType[typeIndex], bitselect_verify_fn, seed ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( vecType[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

void select_signed_verify_fn( ExplicitType vecType, ExplicitType testVecType, unsigned int vecSize, void *inDataA, void *inDataB, void *inDataTest, void *outData )
{
    bool yep = false;
    if (vecSize == 1)  {
        switch( testVecType )
        {
            case kChar:
                yep = *( (char *)inDataTest ) ? true : false;
                break;
            case kShort:
                yep = *( (short *)inDataTest ) ? true : false;
                break;
            case kInt:
                yep = *( (int *)inDataTest ) ? true : false;
                break;
            case kLong:
                yep = *( (cl_long *)inDataTest ) ? true : false;
                break;
            default:
                // Should never get here
                return;
        }
    }
    else {
        switch( testVecType )
        {
            case kChar:
                yep = *( (char *)inDataTest ) & 0x80 ? true : false;
                break;
            case kShort:
                yep = *( (short *)inDataTest ) & 0x8000 ? true : false;
                break;
            case kInt:
                yep = *( (int *)inDataTest ) & 0x80000000L ? true : false;
                break;
            case kLong:
                yep = *( (cl_long *)inDataTest ) & 0x8000000000000000LL ? true : false;
                break;
            default:
                // Should never get here
                return;
        }
    }
    memcpy( outData, ( yep ) ? inDataB : inDataA, get_explicit_type_size( vecType ) );
}

int test_relational_select_signed(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    constexpr ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort,
                                         kInt,  kUInt,  kLong,  kULong,
                                         kHalf, kFloat, kDouble };
    constexpr auto vecTypeSize = sizeof(vecType) / sizeof(ExplicitType);

    ExplicitType testVecType[] = { kChar, kShort, kInt, kLong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 4, 8, 16, 0 };
    unsigned int index, typeIndex, testTypeIndex;
    int retVal = 0;
    RandomSeed seed( gRandomSeed );

    for (typeIndex = 0; typeIndex < vecTypeSize; typeIndex++)
    {
        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong) && !gHasLong)
            continue;

        if (vecType[typeIndex] == kDouble) {
            if(!is_extension_available(device, "cl_khr_fp64")) {
                log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
                continue;
            } else {
                log_info("Testing doubles.\n");
            }
        }
        if (vecType[typeIndex] == kHalf)
        {
            if (!is_extension_available(device, "cl_khr_fp16"))
            {
                log_info("Extension cl_khr_fp16 not supported; skipping half "
                         "tests.\n");
                continue;
            }
            else
            {
                log_info("Testing halfs.\n");
            }
        }
        for( testTypeIndex = 0; testVecType[ testTypeIndex ] != kNumExplicitTypes; testTypeIndex++ )
        {
            if( testVecType[ testTypeIndex ] != vecType[ typeIndex ] )
                continue;

            for( index = 0; vecSizes[ index ] != 0; index++ )
            {
                // Test!
                if( test_select_kernel(context, queue, "select", vecType[ typeIndex ], vecSizes[ index ], testVecType[ testTypeIndex ], select_signed_verify_fn, seed ) != 0 )
                {
                    log_error( "   Vector %s%d, test vector %s%d FAILED\n", get_explicit_type_name( vecType[ typeIndex ] ), vecSizes[ index ],
                              get_explicit_type_name( testVecType[ testTypeIndex ] ), vecSizes[ index ] );
                    retVal = -1;
                }
            }
        }
    }

    return retVal;
}

void select_unsigned_verify_fn( ExplicitType vecType, ExplicitType testVecType, unsigned int vecSize, void *inDataA, void *inDataB, void *inDataTest, void *outData )
{
    bool yep = false;
    if (vecSize == 1)  {
        switch( testVecType )
        {
            case kUChar:
                yep = *( (unsigned char *)inDataTest ) ? true : false;
                break;
            case kUShort:
                yep = *( (unsigned short *)inDataTest ) ? true : false;
                break;
            case kUInt:
                yep = *( (unsigned int *)inDataTest ) ? true : false;
                break;
            case kULong:
                yep = *( (cl_ulong *)inDataTest ) ? true : false;
                break;
            default:
                // Should never get here
                return;
        }
    }
    else {
        switch( testVecType )
        {
            case kUChar:
                yep = *( (unsigned char *)inDataTest ) & 0x80 ? true : false;
                break;
            case kUShort:
                yep = *( (unsigned short *)inDataTest ) & 0x8000 ? true : false;
                break;
            case kUInt:
                yep = *( (unsigned int *)inDataTest ) & 0x80000000L ? true : false;
                break;
            case kULong:
                yep = *( (cl_ulong *)inDataTest ) & 0x8000000000000000LL ? true : false;
                break;
            default:
                // Should never get here
                return;
        }
    }
    memcpy( outData, ( yep ) ? inDataB : inDataA, get_explicit_type_size( vecType ) );
}

int test_relational_select_unsigned(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    constexpr ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort,
                                         kInt,  kUInt,  kLong,  kULong,
                                         kHalf, kFloat, kDouble };
    constexpr auto vecTypeSize = sizeof(vecType) / sizeof(ExplicitType);

    ExplicitType testVecType[] = { kUChar, kUShort, kUInt, kULong, kNumExplicitTypes };
    unsigned int vecSizes[] = { 1, 2, 4, 8, 16, 0 };
    unsigned int index, typeIndex, testTypeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed);


    for (typeIndex = 0; typeIndex < vecTypeSize; typeIndex++)
    {
        if ((vecType[typeIndex] == kLong || vecType[typeIndex] == kULong) && !gHasLong)
            continue;

        if (vecType[typeIndex] == kDouble) {
            if(!is_extension_available(device, "cl_khr_fp64")) {
                log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
                continue;
            } else {
                log_info("Testing doubles.\n");
            }
        }
        if (vecType[typeIndex] == kHalf)
        {
            if (!is_extension_available(device, "cl_khr_fp16"))
            {
                log_info("Extension cl_khr_fp16 not supported; skipping half "
                         "tests.\n");
                continue;
            }
            else
            {
                log_info("Testing halfs.\n");
            }
        }
        for( testTypeIndex = 0; testVecType[ testTypeIndex ] != kNumExplicitTypes; testTypeIndex++ )
        {
            if( testVecType[ testTypeIndex ] != vecType[ typeIndex ] )
                continue;

            for( index = 0; vecSizes[ index ] != 0; index++ )
            {
                // Test!
                if( test_select_kernel(context, queue, "select", vecType[ typeIndex ], vecSizes[ index ], testVecType[ testTypeIndex ], select_unsigned_verify_fn, seed ) != 0 )
                {
                    log_error( "   Vector %s%d, test vector %s%d FAILED\n", get_explicit_type_name( vecType[ typeIndex ] ), vecSizes[ index ],
                              get_explicit_type_name( testVecType[ testTypeIndex ] ), vecSizes[ index ] );
                    retVal = -1;
                }
            }
        }
    }

    return retVal;
}
