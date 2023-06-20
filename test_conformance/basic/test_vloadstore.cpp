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
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include <CL/cl_half.h>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include "harness/stringHelpers.h"
#include "harness/typeWrappers.h"

// Outputs debug information for stores
#define DEBUG 0
// Forces stores/loads to be done with offsets = tid
#define LINEAR_OFFSETS 0
#define NUM_LOADS    512
#define HFF(num) cl_half_from_float(num, halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

char pragma_str[128] = { 0 };
char mem_type[64] = { 0 };
char store_str[128] = { 0 };
char load_str[128] = { 0 };

extern cl_half_rounding_mode halfRoundingMode;

// clang-format off
static const char *store_pattern= "results[ tid ] = tmp;\n";
static const char *store_patternV3 = "results[3*tid] = tmp.s0; results[3*tid+1] = tmp.s1; results[3*tid+2] = tmp.s2;\n";
static const char *load_pattern = "sSharedStorage[ i ] = src[ i ];\n";
static const char *load_patternV3 = "sSharedStorage[3*i] = src[ 3*i]; sSharedStorage[3*i+1] = src[3*i+1]; sSharedStorage[3*i+2] = src[3*i+2];\n";
static const char *kernel_pattern[] = {
pragma_str,
"#define STYPE %s\n"
"__kernel void test_fn( ", mem_type, " STYPE *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"    %s%d tmp = vload%d( offsets[ tid ], ( (", mem_type, " STYPE *) src ) + alignmentOffsets[ tid ] );\n"
"    ", store_str,
"}\n"
};

const char *pattern_local [] = {
pragma_str,
"__kernel void test_fn(__local %s *sSharedStorage, __global %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
"{\n"
"    int tid = get_global_id( 0 );\n"
"   int lid = get_local_id( 0 );\n"
"\n"
"    if( lid == 0 )\n"
"    {\n"
"        for( int i = 0; i < %d; i++ ) {\n"
"           ", load_str,
"        }\n"
"    }\n"
//  Note: the above loop will only run on the first thread of each local group, but this barrier should ensure that all
//  threads are caught up (including the first one with the copy) before any proceed, i.e. the shared storage should be
//  updated on all threads at that point
"   barrier( CLK_LOCAL_MEM_FENCE );\n"
"\n"
"    %s%d tmp = vload%d( offsets[ tid ], ( (__local %s *) sSharedStorage ) + alignmentOffsets[ tid ] );\n"
"    ", store_str,
"}\n" };

const char *pattern_priv [] = {
pragma_str,
// Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
// for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
"#define PRIV_TYPE %s\n"
"#define PRIV_SIZE %d\n"
"__kernel void test_fn( __global %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
"{\n"
"    __private PRIV_TYPE sPrivateStorage[ PRIV_SIZE ];\n"
"    int tid = get_global_id( 0 );\n"
"\n"
"    for( int i = 0; i < PRIV_SIZE; i++ )\n"
"      sPrivateStorage[ i ] = src[ i ];\n"
//    Note: unlike the local test, each thread runs the above copy loop independently, so nobody needs to wait for
//  anybody else to sync up
"\n"
"    %s%d tmp = vload%d( offsets[ tid ], ( (__private %s *) sPrivateStorage ) + alignmentOffsets[ tid ] );\n"
"    ", store_str,
"}\n"};
// clang-format on

#pragma mark -------------------- vload harness --------------------------

typedef void (*create_program_fn)(std::string &, size_t, ExplicitType, size_t,
                                  size_t);
typedef int (*test_fn)(cl_device_id, cl_context, cl_command_queue, ExplicitType,
                       unsigned int, create_program_fn, size_t);

int test_vload(cl_device_id device, cl_context context, cl_command_queue queue,
               ExplicitType type, unsigned int vecSize,
               create_program_fn createFn, size_t bufferSize)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 4 ];
    MTdataHolder d(gRandomSeed);
    const size_t numLoads = (DEBUG) ? 16 : NUM_LOADS;

    if (DEBUG) bufferSize = (bufferSize < 128) ? bufferSize : 128;

    size_t threads[ 1 ], localThreads[ 1 ];
    clProtectedArray inBuffer( bufferSize );
    cl_uint offsets[ numLoads ], alignmentOffsets[ numLoads ];
    size_t numElements, typeSize, i;
    unsigned int outVectorSize;

    pragma_str[0] = '\0';
    if (type == kDouble)
        std::snprintf(pragma_str, sizeof(pragma_str),
                      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    else if (type == kHalf)
        std::snprintf(pragma_str, sizeof(pragma_str),
                      "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");

    typeSize = get_explicit_type_size( type );
    numElements = bufferSize / ( typeSize * vecSize );
    bufferSize = numElements * typeSize * vecSize;    // To account for rounding

    if (DEBUG) log_info("Testing: numLoads: %d, typeSize: %d, vecSize: %d, numElements: %d, bufferSize: %d\n", (int)numLoads, (int)typeSize, vecSize, (int)numElements, (int)bufferSize);

    // Create some random input data and random offsets to load from
    generate_random_data( type, numElements * vecSize, d, (void *)inBuffer );
    for( i = 0; i < numLoads; i++ )
    {
        offsets[ i ] = (cl_uint)random_in_range( 0, (int)numElements - 1, d );
        if( offsets[ i ] < numElements - 2 )
            alignmentOffsets[ i ] = (cl_uint)random_in_range( 0, (int)vecSize - 1, d );
        else
            alignmentOffsets[ i ] = 0;
        if (LINEAR_OFFSETS) offsets[i] = (cl_uint)i;
    }
    if (LINEAR_OFFSETS) log_info("Offsets set to thread IDs to simplify output.\n");

    // 32-bit fixup
    outVectorSize = vecSize;

    // Declare output buffers now
    std::vector<char> outBuffer(numLoads * typeSize * outVectorSize);
    std::vector<char> referenceBuffer(numLoads * typeSize * vecSize);

    // Create the program
    std::string programSrc;
    createFn( programSrc, numElements, type, vecSize, outVectorSize);

    // Create our kernel
    const char *ptr = programSrc.c_str();
    cl_int error = create_single_kernel_helper(context, &program, &kernel, 1,
                                               &ptr, "test_fn");
    test_error( error, "Unable to create testing kernel" );
    if (DEBUG) log_info("Kernel: \n%s\n", programSrc.c_str());

    // Get the number of args to differentiate the kernels with local storage. (They have 5)
    cl_uint numArgs;
    error = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, NULL);
    test_error( error, "clGetKernelInfo failed");

    // Set up parameters
    streams[ 0 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, bufferSize, (void *)inBuffer, &error );
    test_error( error, "Unable to create kernel stream" );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numLoads*sizeof(offsets[0]), offsets, &error );
    test_error( error, "Unable to create kernel stream" );
    streams[ 2 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numLoads*sizeof(alignmentOffsets[0]), alignmentOffsets, &error );
    test_error( error, "Unable to create kernel stream" );
    streams[3] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                numLoads * typeSize * outVectorSize,
                                (void *)outBuffer.data(), &error);
    test_error( error, "Unable to create kernel stream" );

    // Set parameters and run
    if (numArgs == 5) {
        // We need to set the size of the local storage
        error = clSetKernelArg(kernel, 0, bufferSize, NULL);
        test_error( error, "clSetKernelArg for buffer failed");
        for( i = 0; i < 4; i++ )
        {
            error = clSetKernelArg( kernel, (int)i+1, sizeof( streams[ i ] ), &streams[ i ] );
            test_error( error, "Unable to set kernel argument" );
        }
    } else {
        // No local storage
        for( i = 0; i < 4; i++ )
        {
            error = clSetKernelArg( kernel, (int)i, sizeof( streams[ i ] ), &streams[ i ] );
            test_error( error, "Unable to set kernel argument" );
        }
    }

    threads[ 0 ] = numLoads;
    error = get_max_common_work_group_size( context, kernel, threads[ 0 ], &localThreads[ 0 ] );
    test_error( error, "Unable to get local thread size" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to exec kernel" );

    // Get the results
    error = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0,
                                numLoads * typeSize * outVectorSize
                                    * sizeof(cl_char),
                                (void *)outBuffer.data(), 0, NULL, NULL);
    test_error( error, "Unable to read results" );

    // Create the reference results
    referenceBuffer.assign(numLoads * typeSize * vecSize, 0);
    for( i = 0; i < numLoads; i++ )
    {
        memcpy(&referenceBuffer[i * typeSize * vecSize],
               ((char *)(void *)inBuffer)
                   + ((offsets[i] * vecSize) + alignmentOffsets[i]) * typeSize,
               typeSize * vecSize);
    }

    // Validate the results now
    char *expected = referenceBuffer.data();
    char *actual = outBuffer.data();
    char *in = (char *)(void *)inBuffer;

    if (DEBUG) {
        log_info("Memory contents:\n");
        char inString[1024];
        char expectedString[1024], actualString[1024];
        for (i=0; i<numElements; i++) {
            if (i < numLoads) {
                log_info("buffer %3d: input: %s expected: %s got: %s (load offset %3d, alignment offset %3d)", (int)i, GetDataVectorString( &(in[i*typeSize*vecSize]), typeSize, vecSize, inString ),
                         GetDataVectorString( &(expected[i*typeSize*vecSize]), typeSize, vecSize, expectedString ),
                         GetDataVectorString( &(actual[i*typeSize*outVectorSize]), typeSize, vecSize, actualString ),
                         offsets[i], alignmentOffsets[i]);
                if (memcmp(&(expected[i*typeSize*vecSize]), &(actual[i*typeSize*outVectorSize]), typeSize * vecSize) != 0)
                    log_error(" << ERROR\n");
                else
                    log_info("\n");
            } else {
                log_info("buffer %3d: input: %s expected: %s got: %s\n", (int)i, GetDataVectorString( &(in[i*typeSize*vecSize]), typeSize, vecSize, inString ),
                         GetDataVectorString( &(expected[i*typeSize*vecSize]), typeSize, vecSize, expectedString ),
                         GetDataVectorString( &(actual[i*typeSize*outVectorSize]), typeSize, vecSize, actualString ));
            }
        }
    }

    for( i = 0; i < numLoads; i++ )
    {
        if( memcmp( expected, actual, typeSize * vecSize ) != 0 )
        {
            char expectedString[ 1024 ], actualString[ 1024 ];
            log_error( "ERROR: Data sample %d for vload of %s%d did not validate (expected {%s}, got {%s}, loaded from offset %d)\n",
                      (int)i, get_explicit_type_name( type ), vecSize, GetDataVectorString( expected, typeSize, vecSize, expectedString ),
                      GetDataVectorString( actual, typeSize, vecSize, actualString ), (int)offsets[ i ] );
            return 1;
        }
        expected += typeSize * vecSize;
        actual += typeSize * outVectorSize;
    }
    return 0;
}

template <test_fn test_func_ptr>
int test_vset(cl_device_id device, cl_context context, cl_command_queue queue,
              create_program_fn createFn, size_t bufferSize)
{
    std::vector<ExplicitType> vecType = { kChar,  kUChar, kShort, kUShort,
                                          kInt,   kUInt,  kLong,  kULong,
                                          kFloat, kHalf,  kDouble };
    unsigned int vecSizes[] = { 2, 3, 4, 8, 16, 0 };
    const char *size_names[] = { "2", "3", "4", "8", "16"};
    int error = 0;

    log_info("Testing with buffer size of %d.\n", (int)bufferSize);

    bool hasDouble = is_extension_available(device, "cl_khr_fp64");
    bool hasHalf = is_extension_available(device, "cl_khr_fp16");

    for (unsigned typeIdx = 0; typeIdx < vecType.size(); typeIdx++)
    {
        if (vecType[typeIdx] == kDouble && !hasDouble)
            continue;
        else if (vecType[typeIdx] == kHalf && !hasHalf)
            continue;
        else if ((vecType[typeIdx] == kLong || vecType[typeIdx] == kULong)
                 && !gHasLong)
            continue;

        for (unsigned sizeIdx = 0; vecSizes[sizeIdx] != 0; sizeIdx++)
        {
            log_info("Testing %s%s...\n", get_explicit_type_name(vecType[typeIdx]), size_names[sizeIdx]);

            int error_this_type =
                test_func_ptr(device, context, queue, vecType[typeIdx],
                              vecSizes[sizeIdx], createFn, bufferSize);
            if (error_this_type) {
                error += error_this_type;
                log_error("Failure; skipping further sizes for this type.");
                break;
            }
        }
    }
    return error;
}

#pragma mark -------------------- vload test cases --------------------------

void create_global_load_code(std::string &destBuffer, size_t inBufferSize,
                             ExplicitType type, size_t inVectorSize,
                             size_t outVectorSize)
{
    std::snprintf(mem_type, sizeof(mem_type), "__global");
    std::snprintf(store_str, sizeof(store_str), store_patternV3);
    const char *typeName = get_explicit_type_name(type);
    std::string outTypeName = typeName;
    if (inVectorSize != 3)
    {
        outTypeName = str_sprintf("%s%d", typeName, (int)outVectorSize);
        std::snprintf(store_str, sizeof(store_str), store_pattern);
    }

    std::string kernel_src = concat_kernel(
        kernel_pattern, sizeof(kernel_pattern) / sizeof(kernel_pattern[0]));
    destBuffer = str_sprintf(kernel_src, typeName, outTypeName.c_str(),
                             typeName, (int)inVectorSize, (int)inVectorSize);
}

int test_vload_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_vset<test_vload>(device, context, queue,
                                 create_global_load_code, 10240);
}

void create_local_load_code(std::string &destBuffer, size_t inBufferSize,
                            ExplicitType type, size_t inVectorSize,
                            size_t outVectorSize)
{
    std::snprintf(store_str, sizeof(store_str), store_patternV3);
    std::snprintf(load_str, sizeof(load_str), load_patternV3);
    const char *typeName = get_explicit_type_name(type);
    std::string outTypeName = typeName;
    std::string inTypeName = typeName;
    if (inVectorSize != 3)
    {
        outTypeName = str_sprintf("%s%d", typeName, (int)outVectorSize);
        inTypeName = str_sprintf("%s%d", typeName, (int)inVectorSize);
        std::snprintf(store_str, sizeof(store_str), store_pattern);
        std::snprintf(load_str, sizeof(load_str), load_pattern);
    }

    std::string kernel_src = concat_kernel(
        pattern_local, sizeof(pattern_local) / sizeof(pattern_local[0]));
    destBuffer = str_sprintf(kernel_src, inTypeName.c_str(), inTypeName.c_str(),
                             outTypeName.c_str(), (int)inBufferSize, typeName,
                             (int)inVectorSize, (int)inVectorSize, typeName);
}

int test_vload_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong localSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( localSize ), &localSize, NULL );
    test_error( error, "Unable to get max size of local memory buffer" );
    if (localSize > 10240) localSize = 10240;
    if (localSize > 4096)
        localSize -= 2048;
    else
        localSize /= 2;

    return test_vset<test_vload>(device, context, queue, create_local_load_code,
                                 (size_t)localSize);
}

void create_constant_load_code(std::string &destBuffer, size_t inBufferSize,
                               ExplicitType type, size_t inVectorSize,
                               size_t outVectorSize)
{
    std::snprintf(mem_type, sizeof(mem_type), "__constant");
    std::snprintf(store_str, sizeof(store_str), store_patternV3);
    const char *typeName = get_explicit_type_name(type);
    std::string outTypeName = typeName;
    if (inVectorSize != 3)
    {
        outTypeName = str_sprintf("%s%d", typeName, (int)outVectorSize);
        std::snprintf(store_str, sizeof(store_str), store_pattern);
    }

    std::string kernel_src = concat_kernel(
        kernel_pattern, sizeof(kernel_pattern) / sizeof(kernel_pattern[0]));
    destBuffer = str_sprintf(kernel_src, typeName, outTypeName.c_str(),
                             typeName, (int)inVectorSize, (int)inVectorSize);
}

int test_vload_constant(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong maxSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof( maxSize ), &maxSize, NULL );
    test_error( error, "Unable to get max size of constant memory buffer" );
    if (maxSize > 10240) maxSize = 10240;
    if (maxSize > 4096)
        maxSize -= 2048;
    else
        maxSize /= 2;

    return test_vset<test_vload>(device, context, queue,
                                 create_constant_load_code, (size_t)maxSize);
}

void create_private_load_code(std::string &destBuffer, size_t inBufferSize,
                              ExplicitType type, size_t inVectorSize,
                              size_t outVectorSize)
{
    std::snprintf(store_str, sizeof(store_str), store_patternV3);
    const char *typeName = get_explicit_type_name(type);
    std::string outTypeName = typeName;
    std::string inTypeName = typeName;
    int bufSize = (int)inBufferSize * 3;
    if (inVectorSize != 3)
    {
        outTypeName = str_sprintf("%s%d", typeName, (int)outVectorSize);
        inTypeName = str_sprintf("%s%d", typeName, (int)inVectorSize);
        bufSize = (int)inBufferSize;
        std::snprintf(store_str, sizeof(store_str), store_pattern);
    }

    std::string kernel_src = concat_kernel(
        pattern_priv, sizeof(pattern_priv) / sizeof(pattern_priv[0]));
    destBuffer = str_sprintf(kernel_src, inTypeName.c_str(), bufSize,
                             inTypeName.c_str(), outTypeName.c_str(), typeName,
                             (int)inVectorSize, (int)inVectorSize, typeName);
}

int test_vload_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // We have no idea how much actual private storage is available, so just pick a reasonable value,
    // which is that we can fit at least two 16-element long, which is 2*8 bytes * 16 = 256 bytes
    return test_vset<test_vload>(device, context, queue,
                                 create_private_load_code, 256);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark -------------------- vstore harness --------------------------

int test_vstore(cl_device_id device, cl_context context, cl_command_queue queue,
                ExplicitType type, unsigned int vecSize,
                create_program_fn createFn, size_t bufferSize)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 3 ];
    MTdataHolder d(gRandomSeed);

    size_t threads[ 1 ], localThreads[ 1 ];
    size_t numElements, typeSize, numStores = (DEBUG) ? 16 : NUM_LOADS;

    pragma_str[0] = '\0';
    if (type == kDouble)
        std::snprintf(pragma_str, sizeof(pragma_str),
                      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
    else if (type == kHalf)
        std::snprintf(pragma_str, sizeof(pragma_str),
                      "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");

    if (DEBUG)
        bufferSize = (bufferSize < 128) ? bufferSize : 128;

    typeSize = get_explicit_type_size( type );
    numElements = bufferSize / ( typeSize * vecSize );
    bufferSize = numElements * typeSize * vecSize;    // To account for rounding
    if( numStores > numElements * 2 / 3 )
    {
        // Note: unlike load, we have to restrict the # of stores here, since all offsets must be unique for our test
        // (Plus, we leave some room for extra values to make sure didn't get written)
        numStores = numElements * 2 / 3;
        if( numStores < 1 )
            numStores = 1;
    }
    if (DEBUG)
        log_info("Testing: numStores: %d, typeSize: %d, vecSize: %d, numElements: %d, bufferSize: %d\n", (int)numStores, (int)typeSize, vecSize, (int)numElements, (int)bufferSize);

    std::vector<cl_uint> offsets(numStores);
    std::vector<char> inBuffer(numStores * typeSize * vecSize);

    clProtectedArray outBuffer( numElements * typeSize * vecSize );
    std::vector<char> referenceBuffer(numElements * typeSize * vecSize);

    // Create some random input data and random offsets to load from
    generate_random_data(type, numStores * vecSize, d, (void *)inBuffer.data());

    // Note: make sure no two offsets are the same, otherwise the output would depend on
    // the order that threads ran in, and that would be next to impossible to verify
    std::vector<char> flags(numElements);
    flags.assign(flags.size(), 0);

    for (size_t i = 0; i < numStores; i++)
    {
        do
        {
            offsets[ i ] = (cl_uint)random_in_range( 0, (int)numElements - 2, d );    // Note: keep it one vec below the end for offset testing
        } while( flags[ offsets[ i ] ] != 0 );
        flags[ offsets[ i ] ] = -1;
        if (LINEAR_OFFSETS)
            offsets[i] = (int)i;
    }
    if (LINEAR_OFFSETS)
        log_info("Offsets set to thread IDs to simplify output.\n");

    std::string programSrc;
    createFn(programSrc, numElements, type, vecSize, vecSize);

    // Create our kernel
    const char *ptr = programSrc.c_str();
    cl_int error = create_single_kernel_helper(context, &program, &kernel, 1,
                                               &ptr, "test_fn");
    test_error( error, "Unable to create testing kernel" );
    if (DEBUG) log_info("Kernel: \n%s\n", programSrc.c_str());

    // Get the number of args to differentiate the kernels with local storage. (They have 5)
    cl_uint numArgs;
    error = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, NULL);
    test_error( error, "clGetKernelInfo failed");

    // Set up parameters
    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       numStores * typeSize * vecSize * sizeof(cl_char),
                       (void *)inBuffer.data(), &error);
    test_error( error, "Unable to create kernel stream" );
    streams[1] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       numStores * sizeof(cl_uint), offsets.data(), &error);
    test_error( error, "Unable to create kernel stream" );
    streams[ 2 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numElements * typeSize * vecSize, (void *)outBuffer, &error );
    test_error( error, "Unable to create kernel stream" );

    // Set parameters and run
    if (numArgs == 5)
    {
        // We need to set the size of the local storage
        error = clSetKernelArg(kernel, 0, bufferSize, NULL);
        test_error( error, "clSetKernelArg for buffer failed");
        for (size_t i = 0; i < 3; i++)
        {
            error = clSetKernelArg( kernel, (int)i+1, sizeof( streams[ i ] ), &streams[ i ] );
            test_error( error, "Unable to set kernel argument" );
        }
    }
    else
    {
        // No local storage
        for (size_t i = 0; i < 3; i++)
        {
            error = clSetKernelArg( kernel, (int)i, sizeof( streams[ i ] ), &streams[ i ] );
            if (error) log_info("%s\n", programSrc.c_str());
            test_error( error, "Unable to set kernel argument" );
        }
    }

    threads[ 0 ] = numStores;
    error = get_max_common_work_group_size( context, kernel, threads[ 0 ], &localThreads[ 0 ] );
    test_error( error, "Unable to get local thread size" );

    // Run in a loop, changing the address offset from 0 to ( vecSize - 1 ) each time, since
    // otherwise stores might overlap each other, and it'd be a nightmare to test!
    for( cl_uint addressOffset = 0; addressOffset < vecSize; addressOffset++ )
    {
        if (DEBUG)
            log_info("\tstore addressOffset is %d, executing with threads %d\n", addressOffset, (int)threads[0]);

        // Clear the results first
        memset( outBuffer, 0, numElements * typeSize * vecSize );
        error = clEnqueueWriteBuffer( queue, streams[ 2 ], CL_TRUE, 0, numElements * typeSize * vecSize, (void *)outBuffer, 0, NULL, NULL );
        test_error( error, "Unable to erase result stream" );

        // Set up the new offset and run
        if (numArgs == 5)
            error = clSetKernelArg( kernel, 3+1, sizeof( cl_uint ), &addressOffset );
        else
            error = clSetKernelArg( kernel, 3, sizeof( cl_uint ), &addressOffset );
        test_error( error, "Unable to set address offset argument" );

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
        test_error( error, "Unable to exec kernel" );

        // Get the results
        error = clEnqueueReadBuffer( queue, streams[ 2 ], CL_TRUE, 0, numElements * typeSize * vecSize, (void *)outBuffer, 0, NULL, NULL );
        test_error( error, "Unable to read results" );

        // Create the reference results
        referenceBuffer.assign(referenceBuffer.size(), 0);
        for (size_t i = 0; i < numStores; i++)
        {
            memcpy(&referenceBuffer[((offsets[i] * vecSize) + addressOffset)
                                    * typeSize],
                   &inBuffer[i * typeSize * vecSize], typeSize * vecSize);
        }

        // Validate the results now
        char *expected = referenceBuffer.data();
        char *actual = (char *)(void *)outBuffer;

        if (DEBUG)
        {
            log_info("Memory contents:\n");
            char inString[1024];
            char expectedString[1024], actualString[1024];
            for (size_t i = 0; i < numElements; i++)
            {
                if (i < numStores)
                {
                    log_info("buffer %3d: input: %s expected: %s got: %s (store offset %3d)", (int)i, GetDataVectorString( &(inBuffer[i*typeSize*vecSize]), typeSize, vecSize, inString ),
                             GetDataVectorString( &(expected[i*typeSize*vecSize]), typeSize, vecSize, expectedString ),
                             GetDataVectorString( &(actual[i*typeSize*vecSize]), typeSize, vecSize, actualString ),
                             offsets[i]);
                    if (memcmp(&(expected[i*typeSize*vecSize]), &(actual[i*typeSize*vecSize]), typeSize * vecSize) != 0)
                        log_error(" << ERROR\n");
                    else
                        log_info("\n");
                }
                else
                {
                    log_info("buffer %3d: input: %s expected: %s got: %s\n", (int)i, GetDataVectorString( &(inBuffer[i*typeSize*vecSize]), typeSize, vecSize, inString ),
                             GetDataVectorString( &(expected[i*typeSize*vecSize]), typeSize, vecSize, expectedString ),
                             GetDataVectorString( &(actual[i*typeSize*vecSize]), typeSize, vecSize, actualString ));
                }
            }
        }

        for (size_t i = 0; i < numElements; i++)
        {
            if( memcmp( expected, actual, typeSize * vecSize ) != 0 )
            {
                char expectedString[ 1024 ], actualString[ 1024 ];
                log_error( "ERROR: Data sample %d for vstore of %s%d did not validate (expected {%s}, got {%s}",
                          (int)i, get_explicit_type_name( type ), vecSize, GetDataVectorString( expected, typeSize, vecSize, expectedString ),
                          GetDataVectorString( actual, typeSize, vecSize, actualString ) );
                size_t j;
                for( j = 0; j < numStores; j++ )
                {
                    if( offsets[ j ] == (cl_uint)i )
                    {
                        log_error( ", stored from store #%d (of %d, offset = %d) with address offset of %d", (int)j, (int)numStores, offsets[j], (int)addressOffset );
                        break;
                    }
                }
                if( j == numStores )
                    log_error( ", supposed to be canary value" );
                log_error( ")\n" );
                return 1;
            }
            expected += typeSize * vecSize;
            actual += typeSize * vecSize;
        }
    }
    return 0;
}

#pragma mark -------------------- vstore test cases --------------------------

void create_global_store_code(std::string &destBuffer, size_t inBufferSize,
                              ExplicitType type, size_t inVectorSize,
                              size_t /*unused*/)
{
    // clang-format off
    const char *pattern [] = {
    pragma_str,
    "__kernel void test_fn( __global %s%d *srcValues, __global uint *offsets, __global %s *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    vstore%d( srcValues[ tid ], offsets[ tid ], destBuffer + alignmentOffset );\n"
    "}\n" };

    const char *patternV3 [] = {
    pragma_str,
    "__kernel void test_fn( __global %s3 *srcValues, __global uint *offsets, __global %s *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    if((tid&3) == 0) { // if \"tid\" is a multiple of 4 \n"
    "      vstore3( srcValues[ 3*(tid>>2) ], offsets[ tid ], destBuffer + alignmentOffset );\n"
    "    } else {\n"
    "      vstore3( vload3(tid, (__global %s *)srcValues), offsets[ tid ], destBuffer + alignmentOffset );\n"
    "    }\n"
    "}\n" };
    // clang-format on

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        std::string kernel_src =
            concat_kernel(patternV3, sizeof(patternV3) / sizeof(patternV3[0]));
        destBuffer = str_sprintf(kernel_src, typeName, typeName, typeName);
    }
    else
    {
        std::string kernel_src =
            concat_kernel(pattern, sizeof(pattern) / sizeof(pattern[0]));
        destBuffer = str_sprintf(kernel_src, typeName, (int)inVectorSize,
                                 typeName, (int)inVectorSize);
    }
}

int test_vstore_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_vset<test_vstore>(device, context, queue,
                                  create_global_store_code, 10240);
}

void create_local_store_code(std::string &destBuffer, size_t inBufferSize,
                             ExplicitType type, size_t inVectorSize,
                             size_t /*unused*/)
{
    // clang-format off
    const char *pattern[] = {
    pragma_str,
    "#define LOC_TYPE %s\n"
    "#define LOC_VTYPE %s%d\n"
    "__kernel void test_fn(__local LOC_VTYPE *sSharedStorage, __global LOC_VTYPE *srcValues, __global uint *offsets, __global LOC_VTYPE *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sSharedStorage[ offsets[tid] ] = (LOC_VTYPE)(LOC_TYPE)0;\n"
    " sSharedStorage[ offsets[tid] +1 ] =  sSharedStorage[ offsets[tid] ];\n"
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    "    vstore%d( srcValues[ tid ], offsets[ tid ], ( (__local LOC_TYPE *)sSharedStorage ) + alignmentOffset );\n"
    "\n"
    // Note: Once all threads are done vstore'ing into our shared storage, we then copy into the global output
    // buffer, but we have to make sure ALL threads are done vstore'ing before we do the copy
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  int i;\n"
    "  __local LOC_TYPE *sp = (__local LOC_TYPE*) (sSharedStorage + offsets[tid]) + alignmentOffset;\n"
    "  __global LOC_TYPE *dp = (__global LOC_TYPE*) (destBuffer + offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; (size_t)i < sizeof( sSharedStorage[0]) / sizeof( *sp ); i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n" };

    const char *patternV3 [] = {
    pragma_str,
    "#define LOC_TYPE %s\n"
    "__kernel void test_fn(__local LOC_TYPE *sSharedStorage, __global LOC_TYPE *srcValues, __global uint *offsets, __global LOC_TYPE *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    "    sSharedStorage[ 3*offsets[tid]   ] = (LOC_TYPE)0;\n"
    "    sSharedStorage[ 3*offsets[tid] +1 ] =  \n"
    "        sSharedStorage[ 3*offsets[tid] ];\n"
    "    sSharedStorage[ 3*offsets[tid] +2 ] =  \n"
    "        sSharedStorage[ 3*offsets[tid]];\n"
    "    sSharedStorage[ 3*offsets[tid] +3 ] =  \n"
    "        sSharedStorage[ 3*offsets[tid]];\n"
    "    sSharedStorage[ 3*offsets[tid] +4 ] =  \n"
    "        sSharedStorage[ 3*offsets[tid] ];\n"
    "    sSharedStorage[ 3*offsets[tid] +5 ] =  \n"
    "        sSharedStorage[ 3*offsets[tid]];\n"
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    " vstore3( vload3(tid,srcValues), offsets[ tid ], sSharedStorage  + alignmentOffset );\n"
    "\n"
    // Note: Once all threads are done vstore'ing into our shared storage, we then copy into the global output
    // buffer, but we have to make sure ALL threads are done vstore'ing before we do the copy
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  int i;\n"
    "  __local LOC_TYPE *sp =  (sSharedStorage + 3*offsets[tid]) + alignmentOffset;\n"
    "  __global LOC_TYPE *dp = (destBuffer + 3*offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; i < 3; i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n" };
    // clang-format on

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        std::string kernel_src =
            concat_kernel(patternV3, sizeof(patternV3) / sizeof(patternV3[0]));
        destBuffer = str_sprintf(kernel_src, typeName);
    }
    else
    {
        std::string kernel_src =
            concat_kernel(pattern, sizeof(pattern) / sizeof(pattern[0]));
        destBuffer = str_sprintf(kernel_src, typeName, typeName,
                                 (int)inVectorSize, (int)inVectorSize);
    }
}

int test_vstore_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong localSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( localSize ), &localSize, NULL );
    test_error( error, "Unable to get max size of local memory buffer" );
    if (localSize > 10240) localSize = 10240;
    if (localSize > 4096)
        localSize -= 2048;
    else
        localSize /= 2;
    return test_vset<test_vstore>(device, context, queue,
                                  create_local_store_code, (size_t)localSize);
}

void create_private_store_code(std::string &destBuffer, size_t inBufferSize,
                               ExplicitType type, size_t inVectorSize,
                               size_t /*unused*/)
{
    // clang-format off
    const char *pattern [] = {
    pragma_str,
    "#define PRIV_TYPE %s\n"
    "#define PRIV_VTYPE %s%d\n"
    // Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
    // for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
    "\n"
    "__kernel void test_fn( __global PRIV_VTYPE *srcValues, __global uint *offsets, __global PRIV_VTYPE *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "  __private PRIV_VTYPE sPrivateStorage[ %d ];\n"
    "  int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sPrivateStorage[tid] = (PRIV_VTYPE)(PRIV_TYPE)0;\n"
    "\n"
    "  vstore%d( srcValues[ tid ], offsets[ tid ], ( (__private PRIV_TYPE *)sPrivateStorage ) + alignmentOffset );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  uint i;\n"
    "  __private PRIV_TYPE *sp = (__private PRIV_TYPE*) (sPrivateStorage + offsets[tid]) + alignmentOffset;\n"
    "  __global PRIV_TYPE *dp = (__global PRIV_TYPE*) (destBuffer + offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; i < sizeof( sPrivateStorage[0]) / sizeof( *sp ); i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n"};

    const char *patternV3  [] = {
    pragma_str,
    "#define PRIV_TYPE %s\n"
    "#define PRIV_VTYPE %s3\n"
    // Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
    // for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
    "\n"
    "__kernel void test_fn( __global PRIV_TYPE *srcValues, __global uint *offsets, __global PRIV_VTYPE *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "  __private PRIV_VTYPE sPrivateStorage[ %d ];\n" // keep this %d
    "  int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sPrivateStorage[tid] = (PRIV_VTYPE)(PRIV_TYPE)0;\n"
    "\n"
    "  vstore3( vload3(tid,srcValues), offsets[ tid ], ( (__private PRIV_TYPE *)sPrivateStorage ) + alignmentOffset );\n"
    "  uint i;\n"
    "  __private PRIV_TYPE *sp = ((__private PRIV_TYPE*) sPrivateStorage) + 3*offsets[tid] + alignmentOffset;\n"
    "  __global PRIV_TYPE *dp = ((__global PRIV_TYPE*) destBuffer) + 3*offsets[tid] + alignmentOffset;\n"
    "  for( i = 0; i < 3; i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n"};
    // clang-format on

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        std::string kernel_src =
            concat_kernel(patternV3, sizeof(patternV3) / sizeof(patternV3[0]));
        destBuffer =
            str_sprintf(kernel_src, typeName, typeName, (int)inBufferSize);
    }
    else
    {
        std::string kernel_src =
            concat_kernel(pattern, sizeof(pattern) / sizeof(pattern[0]));
        destBuffer =
            str_sprintf(kernel_src, typeName, typeName, (int)inVectorSize,
                        (int)inBufferSize, (int)inVectorSize);
    }
}

int test_vstore_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // We have no idea how much actual private storage is available, so just pick a reasonable value,
    // which is that we can fit at least two 16-element long, which is 2*8 bytes * 16 = 256 bytes
    return test_vset<test_vstore>(device, context, queue,
                                  create_private_store_code, 256);
}



