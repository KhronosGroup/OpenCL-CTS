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

#include <cinttypes>

#define TEST_SIZE 512

enum OpKonstants
{
    kIncrement = 0,
    kDecrement,
    kBoth
};

const char *testKernel =
"__kernel void test( __global %s *inOut, __global char * control )\n"
"{\n"
"    size_t tid = get_global_id(0);\n"
"\n"
"   %s%s inOutVal = %s;\n"
"\n"
"   if( control[tid] == 0 )\n"
"        inOutVal++;\n"
"   else if( control[tid] == 1 )\n"
"        ++inOutVal;\n"
"   else if( control[tid] == 2 )\n"
"        inOutVal--;\n"
"   else // if( control[tid] == 3 )\n"
"        --inOutVal;\n"
"\n"
"   %s;\n"
"}\n";

typedef int (*OpVerifyFn)( void * actualPtr, void * inputPtr, size_t vecSize, size_t numVecs, cl_char * controls );

int test_unary_op( cl_command_queue queue, cl_context context, OpKonstants whichOp,
                                     ExplicitType vecType, size_t vecSize,
                                     MTdata d, OpVerifyFn verifyFn )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];
    cl_long inData[TEST_SIZE * 16], outData[TEST_SIZE * 16];
    cl_char controlData[TEST_SIZE];
    int error;
    size_t i;
    size_t threads[1], localThreads[1];
    char kernelSource[10240];
    char *programPtr;


    // Create the source
    char loadLine[ 1024 ], storeLine[ 1024 ];
    if( vecSize == 1 )
    {
        sprintf( loadLine, "inOut[tid]" );
        sprintf( storeLine, "inOut[tid] = inOutVal" );
    }
    else
    {
        sprintf(loadLine, "vload%zu( tid, inOut )", vecSize);
        sprintf(storeLine, "vstore%zu( inOutVal, tid, inOut )", vecSize);
    }

    char sizeNames[][4] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    sprintf( kernelSource, testKernel, get_explicit_type_name( vecType ), /*sizeNames[ vecSize ],*/
                                        get_explicit_type_name( vecType ), sizeNames[ vecSize ],
                                        loadLine, storeLine );

    // Create the kernel
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "test" ) )
    {
        log_error( "ERROR: Unable to create test program!\n" );
        return -1;
    }

    // Generate two streams. The first is our random data to test against, the second is our control stream
    generate_random_data( vecType, vecSize * TEST_SIZE, d, inData );
    streams[0] = clCreateBuffer(
        context, CL_MEM_COPY_HOST_PTR,
        get_explicit_type_size(vecType) * vecSize * TEST_SIZE, inData, &error);
    test_error( error, "Creating input data array failed" );

    cl_uint bits;
    for( i = 0; i < TEST_SIZE; i++ )
    {
        size_t which = i & 7;
        if( which == 0 )
            bits = genrand_int32(d);

        controlData[ i ] = ( bits >> ( which << 1 ) ) & 0x03;
        if( whichOp == kDecrement )
            // For sub ops, the min control value is 2. Otherwise, it's 0
            controlData[ i ] |= 0x02;
        else if( whichOp == kIncrement )
            // For addition ops, the max control value is 1. Otherwise, it's 3
            controlData[ i ] &= ~0x02;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                sizeof(controlData), controlData, &error);
    test_error( error, "Unable to create control stream" );

    // Assign streams and execute
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );


    // Run the kernel
    threads[0] = TEST_SIZE;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );


    // Read the results
    error = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0,
                                get_explicit_type_size( vecType ) * TEST_SIZE * vecSize,
                                outData, 0, NULL, NULL );
    test_error( error, "Unable to read output array!" );

    // Now verify the results
    return verifyFn( outData, inData, vecSize, TEST_SIZE, controlData );
}

template<typename T> int VerifyFn( void * actualPtr, void * inputPtr, size_t vecSize, size_t numVecs, cl_char * controls )
{
    T * actualData = (T *)actualPtr;
    T * inputData = (T *)inputPtr;

    size_t index = 0;
    for( size_t i = 0; i < numVecs; i++ )
    {
        for( size_t j = 0; j < vecSize; j++, index++ )
        {
            T nextVal = inputData[ index ];
            if( controls[ i ] & 0x02 )
                nextVal--;
            else
                nextVal++;

            if( actualData[ index ] != nextVal )
            {
                log_error("ERROR: Validation failed on vector %zu:%zu "
                          "(expected %" PRId64 ", got %" PRId64 ")",
                          i, j, (cl_long)nextVal, (cl_long)actualData[index]);
                return -1;
            }
        }
    }
    return 0;
}

int test_unary_op_set( cl_command_queue queue, cl_context context, OpKonstants whichOp )
{
    ExplicitType types[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kNumExplicitTypes };
    OpVerifyFn verifys[] = { VerifyFn<cl_char>, VerifyFn<cl_uchar>, VerifyFn<cl_short>, VerifyFn<cl_ushort>, VerifyFn<cl_int>, VerifyFn<cl_uint>, VerifyFn<cl_long>, VerifyFn<cl_ulong>, NULL };
    unsigned int vecSizes[] = { 1, 2, 3, 4, 8, 16, 0 };
    unsigned int index, typeIndex;
    int retVal = 0;
    RandomSeed seed(gRandomSeed );

    for( typeIndex = 0; types[ typeIndex ] != kNumExplicitTypes; typeIndex++ )
    {
        if ((types[ typeIndex ] == kLong || types[ typeIndex ] == kULong) && !gHasLong)
            continue;

        for( index = 0; vecSizes[ index ] != 0; index++ )
        {
            if( test_unary_op( queue, context, whichOp, types[ typeIndex ], vecSizes[ index ], seed, verifys[ typeIndex ] ) != 0 )
            {
                log_error( "   Vector %s%d FAILED\n", get_explicit_type_name( types[ typeIndex ] ), vecSizes[ index ] );
                retVal = -1;
            }
        }
    }

    return retVal;
}

int test_unary_ops_full(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_unary_op_set( queue, context, kBoth );
}

int test_unary_ops_increment(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_unary_op_set( queue, context, kIncrement );
}

int test_unary_ops_decrement(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_unary_op_set( queue, context, kDecrement );
}
