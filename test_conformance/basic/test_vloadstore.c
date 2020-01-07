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
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"

// Outputs debug information for stores
#define DEBUG 0
// Forces stores/loads to be done with offsets = tid
#define LINEAR_OFFSETS 0
#define NUM_LOADS    512

static const char *doubleExtensionPragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

#pragma mark -------------------- vload harness --------------------------

typedef void (*create_vload_program_fn)( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize, size_t outVectorSize );

int test_vload( cl_device_id device, cl_context context, cl_command_queue queue, ExplicitType type, unsigned int vecSize,
               create_vload_program_fn createFn, size_t bufferSize, MTdata d )
{
    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 4 ];
    const size_t numLoads = (DEBUG) ? 16 : NUM_LOADS;

    if (DEBUG) bufferSize = (bufferSize < 128) ? bufferSize : 128;

    size_t threads[ 1 ], localThreads[ 1 ];
    clProtectedArray inBuffer( bufferSize );
    char programSrc[ 10240 ];
    cl_uint offsets[ numLoads ], alignmentOffsets[ numLoads ];
    size_t numElements, typeSize, i;
    unsigned int outVectorSize;


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
#if !(defined(_WIN32) && defined(_MSC_VER))
    char outBuffer[ numLoads * typeSize * outVectorSize ];
    char referenceBuffer[ numLoads * typeSize * vecSize ];
#else
    char* outBuffer = (char*)_malloca(numLoads * typeSize * outVectorSize * sizeof(cl_char));
    char* referenceBuffer = (char*)_malloca(numLoads * typeSize * vecSize * sizeof(cl_char));
#endif

    // Create the program


    createFn( programSrc, numElements, type, vecSize, outVectorSize);

    // Create our kernel
    const char *ptr = programSrc;

    error = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "test_fn" );
    test_error( error, "Unable to create testing kernel" );
    if (DEBUG) log_info("Kernel: \n%s\n", programSrc);

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
    streams[ 3 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numLoads*typeSize*outVectorSize, (void *)outBuffer, &error );
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
    error = clEnqueueReadBuffer( queue, streams[ 3 ], CL_TRUE, 0, numLoads * typeSize * outVectorSize * sizeof(cl_char), (void *)outBuffer, 0, NULL, NULL );
    test_error( error, "Unable to read results" );


    // Create the reference results
    memset( referenceBuffer, 0, numLoads * typeSize * vecSize * sizeof(cl_char));
    for( i = 0; i < numLoads; i++ )
    {
        memcpy( referenceBuffer + i * typeSize * vecSize, ( (char *)(void *)inBuffer ) + ( ( offsets[ i ] * vecSize ) + alignmentOffsets[ i ] ) * typeSize,
               typeSize * vecSize );
    }

    // Validate the results now
    char *expected = referenceBuffer;
    char *actual = outBuffer;
    char *in = (char *)(void *)inBuffer;

    if (DEBUG) {
        log_info("Memory contents:\n");
        for (i=0; i<numElements; i++) {
            char  inString[1024];
            char expectedString[ 1024 ], actualString[ 1024 ];
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

int test_vloadset(cl_device_id device, cl_context context, cl_command_queue queue, create_vload_program_fn createFn, size_t bufferSize )
{
    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble, kNumExplicitTypes };
    unsigned int vecSizes[] = { 2, 3, 4, 8, 16, 0 };
    const char *size_names[] = { "2", "3", "4", "8", "16"};
    unsigned int typeIdx, sizeIdx;
    int error = 0;
    MTdata mtData = init_genrand( gRandomSeed );

    log_info("Testing with buffer size of %d.\n", (int)bufferSize);

    for( typeIdx = 0; vecType[ typeIdx ] != kNumExplicitTypes; typeIdx++ )
    {

        if( vecType[ typeIdx ] == kDouble && !is_extension_available( device, "cl_khr_fp64" ) )
            continue;

        if(( vecType[ typeIdx ] == kLong || vecType[ typeIdx ] == kULong ) && !gHasLong )
            continue;

        for( sizeIdx = 0; vecSizes[ sizeIdx ] != 0; sizeIdx++ )
        {
            log_info("Testing %s%s...\n", get_explicit_type_name(vecType[typeIdx]), size_names[sizeIdx]);

            int error_this_type = test_vload( device, context, queue, vecType[ typeIdx ], vecSizes[ sizeIdx ], createFn, bufferSize, mtData );
            if (error_this_type) {
                error += error_this_type;
                log_error("Failure; skipping further sizes for this type.");
                break;
            }
        }
    }

    free_mtdata(mtData);

    return error;
}

#pragma mark -------------------- vload test cases --------------------------

void create_global_load_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize, size_t outVectorSize )
{
    const char *pattern =
    "%s%s"
    "__kernel void test_fn( __global %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s%d *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    %s%d tmp = vload%d( offsets[ tid ], ( (__global %s *) src ) + alignmentOffsets[ tid ] );\n"
    "   results[ tid ] = tmp;\n"
    "}\n";

    const char *patternV3 =
    "%s%s"
    "__kernel void test_fn( __global %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    %s3 tmp = vload3( offsets[ tid ], ( (__global %s *) src ) + alignmentOffsets[ tid ] );\n"
    "   results[ 3*tid ] = tmp.s0;\n"
    "   results[ 3*tid+1 ] = tmp.s1;\n"
    "   results[ 3*tid+2 ] = tmp.s2;\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, typeName, typeName, typeName );
    } else {
        sprintf( destBuffer, pattern, type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, typeName, (int)outVectorSize, typeName, (int)inVectorSize,
                (int)inVectorSize, typeName );
    }
}

int test_vload_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_vloadset( device, context, queue, create_global_load_code, 10240 );
}


void create_local_load_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize, size_t outVectorSize )
{
    const char *pattern =
    "%s%s"
    //"   __local %s%d sSharedStorage[ %d ];\n"
    "__kernel void test_fn(__local %s%d *sSharedStorage, __global %s%d *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s%d *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "   int lid = get_local_id( 0 );\n"
    "\n"
    "    if( lid == 0 )\n"
    "    {\n"
    "        for( int i = 0; i < %d; i++ )\n"
    "           sSharedStorage[ i ] = src[ i ];\n"
    "    }\n"
    //  Note: the above loop will only run on the first thread of each local group, but this barrier should ensure that all
    //  threads are caught up (including the first one with the copy) before any proceed, i.e. the shared storage should be
    //  updated on all threads at that point
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    "    %s%d tmp = vload%d( offsets[ tid ], ( (__local %s *) sSharedStorage ) + alignmentOffsets[ tid ] );\n"
    "   results[ tid ] = tmp;\n"
    "}\n";

    const char *patternV3 =
    "%s%s"
    //"   __local %s%d sSharedStorage[ %d ];\n"
    "__kernel void test_fn(__local %s *sSharedStorage, __global %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "   int lid = get_local_id( 0 );\n"
    "\n"
    "    if( lid == 0 )\n"
    "    {\n"
    "        for( int i = 0; i < %d; i++ ) {\n"
    "           sSharedStorage[ 3*i   ] = src[ 3*i   ];\n"
    "           sSharedStorage[ 3*i +1] = src[ 3*i +1];\n"
    "           sSharedStorage[ 3*i +2] = src[ 3*i +2];\n"
    "        }\n"
    "    }\n"
    //  Note: the above loop will only run on the first thread of each local group, but this barrier should ensure that all
    //  threads are caught up (including the first one with the copy) before any proceed, i.e. the shared storage should be
    //  updated on all threads at that point
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    "    %s3 tmp = vload3( offsets[ tid ], ( (__local %s *) sSharedStorage ) + alignmentOffsets[ tid ] );\n"
    "   results[ 3*tid   ] = tmp.s0;\n"
    "   results[ 3*tid +1] = tmp.s1;\n"
    "   results[ 3*tid +2] = tmp.s2;\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble  ? doubleExtensionPragma : "",
                "",
                typeName, /*(int)inBufferSize,*/
                typeName, typeName,
                (int)inBufferSize,
                typeName, typeName );
    } else {
        sprintf( destBuffer, pattern,
                type == kDouble  ? doubleExtensionPragma : "",
                "",
                typeName, (int)inVectorSize, /*(int)inBufferSize,*/
                typeName, (int)inVectorSize, typeName, (int)outVectorSize,
                (int)inBufferSize,
                typeName, (int)inVectorSize, (int)inVectorSize, typeName );
    }
}

int test_vload_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong localSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( localSize ), &localSize, NULL );
    test_error( error, "Unable to get max size of local memory buffer" );
    if( localSize > 10240 )
        localSize = 10240;
    if (localSize > 4096)
        localSize -= 2048;
    else
        localSize /= 2;

    return test_vloadset( device, context, queue, create_local_load_code, (size_t)localSize );
}


void create_constant_load_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize, size_t outVectorSize )
{
    const char *pattern =
    "%s%s"
    "__kernel void test_fn( __constant %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s%d *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    %s%d tmp = vload%d( offsets[ tid ], ( (__constant %s *) src ) + alignmentOffsets[ tid ] );\n"
    "   results[ tid ] = tmp;\n"
    "}\n";

    const char *patternV3 =
    "%s%s"
    "__kernel void test_fn( __constant %s *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s *results )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    %s3 tmp = vload3( offsets[ tid ], ( (__constant %s *) src ) + alignmentOffsets[ tid ] );\n"
    "   results[ 3*tid   ] = tmp.s0;\n"
    "   results[ 3*tid+1 ] = tmp.s1;\n"
    "   results[ 3*tid+2 ] = tmp.s2;\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, typeName,  typeName,
                typeName );
    } else {
        sprintf( destBuffer, pattern,
                type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, typeName, (int)outVectorSize, typeName, (int)inVectorSize,
                (int)inVectorSize, typeName );
    }
}

int test_vload_constant(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong maxSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof( maxSize ), &maxSize, NULL );
    test_error( error, "Unable to get max size of constant memory buffer" );
    if( maxSize > 10240 )
        maxSize = 10240;
    if (maxSize > 4096)
        maxSize -= 2048;
    else
        maxSize /= 2;

    return test_vloadset( device, context, queue, create_constant_load_code, (size_t)maxSize );
}


void create_private_load_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize, size_t outVectorSize )
{
    const char *pattern =
    "%s%s"
    // Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
    // for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
    "#define PRIV_TYPE %s%d\n"
    "#define PRIV_SIZE %d\n"
    "__kernel void test_fn( __global %s%d *src, __global uint *offsets, __global uint *alignmentOffsets, __global %s%d *results )\n"
    "{\n"
    "    __private PRIV_TYPE sPrivateStorage[ PRIV_SIZE ];\n"
    "    int tid = get_global_id( 0 );\n"
    "\n"
    "    for( int i = 0; i < %d; i++ )\n"
    "      sPrivateStorage[ i ] = src[ i ];\n"
    //    Note: unlike the local test, each thread runs the above copy loop independently, so nobody needs to wait for
    //  anybody else to sync up
    "\n"
    "    %s%d tmp = vload%d( offsets[ tid ], ( (__private %s *) sPrivateStorage ) + alignmentOffsets[ tid ] );\n"
    "   results[ tid ] = tmp;\n"
    "}\n";

    const char *patternV3 =
    "%s%s"
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
    "    {\n"
    "        sPrivateStorage[ i ] = src[ i ];\n"
    "    }\n"
    //    Note: unlike the local test, each thread runs the above copy loop independently, so nobody needs to wait for
    //  anybody else to sync up
    "\n"
    "    %s3 tmp = vload3( offsets[ tid ], ( sPrivateStorage ) + alignmentOffsets[ tid ] );\n"
    "   results[ 3*tid   ] = tmp.s0;\n"
    "   results[ 3*tid+1 ] = tmp.s1;\n"
    "   results[ 3*tid+2 ] = tmp.s2;\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize ==3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, 3*((int)inBufferSize),
                typeName, typeName,
                typeName );
        // log_info("Src is \"\n%s\n\"\n", destBuffer);
    } else {
        sprintf( destBuffer, pattern,
                type == kDouble ? doubleExtensionPragma : "",
                "",
                typeName, (int)inVectorSize, (int)inBufferSize,
                typeName, (int)inVectorSize, typeName, (int)outVectorSize,
                (int)inBufferSize,
                typeName, (int)inVectorSize, (int)inVectorSize, typeName );
    }
}

int test_vload_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // We have no idea how much actual private storage is available, so just pick a reasonable value,
    // which is that we can fit at least two 16-element long, which is 2*8 bytes * 16 = 256 bytes
    return test_vloadset( device, context, queue, create_private_load_code, 256 );
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma mark -------------------- vstore harness --------------------------

typedef void (*create_vstore_program_fn)( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize );

int test_vstore( cl_device_id device, cl_context context, cl_command_queue queue, ExplicitType type, unsigned int vecSize,
                create_vstore_program_fn createFn, size_t bufferSize, MTdata d )
{
    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 3 ];

    size_t threads[ 1 ], localThreads[ 1 ];

    size_t numElements, typeSize, numStores = (DEBUG) ? 16 : NUM_LOADS;

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
#if !(defined(_WIN32) && defined(_MSC_VER))
    cl_uint offsets[ numStores ];
#else
    cl_uint* offsets = (cl_uint*)_malloca(numStores * sizeof(cl_uint));
#endif
    char programSrc[ 10240 ];
    size_t i;

#if !(defined(_WIN32) && defined(_MSC_VER))
    char inBuffer[ numStores * typeSize * vecSize ];
#else
    char* inBuffer = (char*)_malloca( numStores * typeSize * vecSize * sizeof(cl_char));
#endif
    clProtectedArray outBuffer( numElements * typeSize * vecSize );
#if !(defined(_WIN32) && defined(_MSC_VER))
    char referenceBuffer[ numElements * typeSize * vecSize ];
#else
    char* referenceBuffer = (char*)_malloca(numElements * typeSize * vecSize * sizeof(cl_char));
#endif

    // Create some random input data and random offsets to load from
    generate_random_data( type, numStores * vecSize, d, (void *)inBuffer );

    // Note: make sure no two offsets are the same, otherwise the output would depend on
    // the order that threads ran in, and that would be next to impossible to verify
#if !(defined(_WIN32) && defined(_MSC_VER))
    char flags[ numElements ];
#else
    char* flags = (char*)_malloca( numElements * sizeof(char));
#endif

    memset( flags, 0, numElements * sizeof(char) );
    for( i = 0; i < numStores; i++ )
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

    createFn( programSrc, numElements, type, vecSize );

    // Create our kernel
    const char *ptr = programSrc;
    error = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "test_fn" );
    test_error( error, "Unable to create testing kernel" );
    if (DEBUG) log_info("Kernel: \n%s\n", programSrc);

    // Get the number of args to differentiate the kernels with local storage. (They have 5)
    cl_uint numArgs;
    error = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(numArgs), &numArgs, NULL);
    test_error( error, "clGetKernelInfo failed");

    // Set up parameters
    streams[ 0 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numStores * typeSize * vecSize * sizeof(cl_char), (void *)inBuffer, &error );
    test_error( error, "Unable to create kernel stream" );
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numStores * sizeof(cl_uint), offsets, &error );
    test_error( error, "Unable to create kernel stream" );
    streams[ 2 ] = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, numElements * typeSize * vecSize, (void *)outBuffer, &error );
    test_error( error, "Unable to create kernel stream" );

    // Set parameters and run
    if (numArgs == 5)
    {
        // We need to set the size of the local storage
        error = clSetKernelArg(kernel, 0, bufferSize, NULL);
        test_error( error, "clSetKernelArg for buffer failed");
        for( i = 0; i < 3; i++ )
        {
            error = clSetKernelArg( kernel, (int)i+1, sizeof( streams[ i ] ), &streams[ i ] );
            test_error( error, "Unable to set kernel argument" );
        }
    }
    else
    {
        // No local storage
        for( i = 0; i < 3; i++ )
        {
            error = clSetKernelArg( kernel, (int)i, sizeof( streams[ i ] ), &streams[ i ] );
            if (error)
                log_info("%s\n", programSrc);
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
        memset( referenceBuffer, 0, numElements * typeSize * vecSize * sizeof(cl_char) );
        for( i = 0; i < numStores; i++ )
        {
            memcpy( referenceBuffer + ( ( offsets[ i ] * vecSize ) + addressOffset ) * typeSize, inBuffer + i * typeSize * vecSize, typeSize * vecSize );
        }

        // Validate the results now
        char *expected = referenceBuffer;
        char *actual = (char *)(void *)outBuffer;

        if (DEBUG)
        {
            log_info("Memory contents:\n");
            for (i=0; i<numElements; i++)
            {
                char  inString[1024];
                char expectedString[ 1024 ], actualString[ 1024 ];
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

        for( i = 0; i < numElements; i++ )
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

int test_vstoreset(cl_device_id device, cl_context context, cl_command_queue queue, create_vstore_program_fn createFn, size_t bufferSize )
{
    ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt, kUInt, kLong, kULong, kFloat, kDouble, kNumExplicitTypes };
    unsigned int vecSizes[] = { 2, 3, 4, 8, 16, 0 };
    const char *size_names[] = { "2", "3", "4", "8", "16"};
    unsigned int typeIdx, sizeIdx;
    int error = 0;
    MTdata d = init_genrand( gRandomSeed );

    log_info("Testing with buffer size of %d.\n", (int)bufferSize);

    for( typeIdx = 0; vecType[ typeIdx ] != kNumExplicitTypes; typeIdx++ )
    {
        if( vecType[ typeIdx ] == kDouble && !is_extension_available( device, "cl_khr_fp64" ) )
            continue;

        if(( vecType[ typeIdx ] == kLong || vecType[ typeIdx ] == kULong ) && !gHasLong )
            continue;

        for( sizeIdx = 0; vecSizes[ sizeIdx ] != 0; sizeIdx++ )
        {
            log_info("Testing %s%s...\n", get_explicit_type_name(vecType[typeIdx]), size_names[sizeIdx]);

            int error_this_type = test_vstore( device, context, queue, vecType[ typeIdx ], vecSizes[ sizeIdx ], createFn, bufferSize, d );
            if (error_this_type)
            {
                log_error("Failure; skipping further sizes for this type.\n");
                error += error_this_type;
                break;
            }
        }
    }

    free_mtdata(d);
    return error;
}


#pragma mark -------------------- vstore test cases --------------------------

void create_global_store_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize )
{
    const char *pattern =
    "%s"
    "__kernel void test_fn( __global %s%d *srcValues, __global uint *offsets, __global %s *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    vstore%d( srcValues[ tid ], offsets[ tid ], destBuffer + alignmentOffset );\n"
    "}\n";

    const char *patternV3 =
    "%s"
    "__kernel void test_fn( __global %s3 *srcValues, __global uint *offsets, __global %s *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    "    if((tid&3) == 0) { // if \"tid\" is a multiple of 4 \n"
    "      vstore3( srcValues[ 3*(tid>>2) ], offsets[ tid ], destBuffer + alignmentOffset );\n"
    "    } else {\n"
    "      vstore3( vload3(tid, (__global %s *)srcValues), offsets[ tid ], destBuffer + alignmentOffset );\n"
    "    }\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);

    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                typeName, typeName, typeName);

    } else {
        sprintf( destBuffer, pattern,
                type == kDouble ? doubleExtensionPragma : "",
                typeName, (int)inVectorSize, typeName, (int)inVectorSize );
    }
    // if(inVectorSize == 3 || inVectorSize == 4) {
    //     log_info("\n----\n%s\n----\n", destBuffer);
    // }
}

int test_vstore_global(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    return test_vstoreset( device, context, queue, create_global_store_code, 10240 );
}


void create_local_store_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize )
{
    const char *pattern =
    "%s"
    "\n"
    "__kernel void test_fn(__local %s%d *sSharedStorage, __global %s%d *srcValues, __global uint *offsets, __global %s%d *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sSharedStorage[ offsets[tid] ] = (%s%d)(%s)0;\n"
    " sSharedStorage[ offsets[tid] +1 ] =  sSharedStorage[ offsets[tid] ];\n"
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    "    vstore%d( srcValues[ tid ], offsets[ tid ], ( (__local %s *)sSharedStorage ) + alignmentOffset );\n"
    "\n"
    // Note: Once all threads are done vstore'ing into our shared storage, we then copy into the global output
    // buffer, but we have to make sure ALL threads are done vstore'ing before we do the copy
    "   barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  int i;\n"
    "  __local %s *sp = (__local %s*) (sSharedStorage + offsets[tid]) + alignmentOffset;\n"
    "  __global %s *dp = (__global %s*) (destBuffer + offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; (size_t)i < sizeof( sSharedStorage[0]) / sizeof( *sp ); i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n";

    const char *patternV3 =
    "%s"
    "\n"
    "__kernel void test_fn(__local %s *sSharedStorage, __global %s *srcValues, __global uint *offsets, __global %s *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    "    sSharedStorage[ 3*offsets[tid]   ] = (%s)0;\n"
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
    "  __local %s *sp =  (sSharedStorage + 3*offsets[tid]) + alignmentOffset;\n"
    "  __global %s *dp = (destBuffer + 3*offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; i < 3; i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                typeName,
                typeName,
                typeName,  typeName,
                typeName, typeName, typeName  );
    } else {
        sprintf( destBuffer, pattern,
                type == kDouble ? doubleExtensionPragma : "",
                typeName, (int)inVectorSize,
                typeName, (int)inVectorSize, typeName, (int)inVectorSize,
                typeName, (int)inVectorSize, typeName,
                (int)inVectorSize, typeName, typeName,
                typeName, typeName, typeName  );
    }
    // log_info(destBuffer);
}

int test_vstore_local(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // Determine the max size of a local buffer that we can test against
    cl_ulong localSize;
    int error = clGetDeviceInfo( device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( localSize ), &localSize, NULL );
    test_error( error, "Unable to get max size of local memory buffer" );
    if( localSize > 10240 )
        localSize = 10240;
    if (localSize > 4096)
        localSize -= 2048;
    else
        localSize /= 2;
    return test_vstoreset( device, context, queue, create_local_store_code, (size_t)localSize );
}


void create_private_store_code( char *destBuffer, size_t inBufferSize, ExplicitType type, size_t inVectorSize )
{
    const char *pattern =
    "%s"
    // Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
    // for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
    "\n"
    "__kernel void test_fn( __global %s%d *srcValues, __global uint *offsets, __global %s%d *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    __private %s%d sPrivateStorage[ %d ];\n"
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sPrivateStorage[tid] = (%s%d)(%s)0;\n"
    "\n"
    "   vstore%d( srcValues[ tid ], offsets[ tid ], ( (__private %s *)sPrivateStorage ) + alignmentOffset );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  uint i;\n"
    "  __private %s *sp = (__private %s*) (sPrivateStorage + offsets[tid]) + alignmentOffset;\n"
    "  __global %s *dp = (__global %s*) (destBuffer + offsets[tid]) + alignmentOffset;\n"
    "  for( i = 0; i < sizeof( sPrivateStorage[0]) / sizeof( *sp ); i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n";


    const char *patternV3 =
    "%s"
    // Private memory is unique per thread, unlike local storage which is unique per local work group. Which means
    // for this test, we have to copy the entire test buffer into private storage ON EACH THREAD to be an effective test
    "\n"
    "__kernel void test_fn( __global %s *srcValues, __global uint *offsets, __global %s3 *destBuffer, uint alignmentOffset )\n"
    "{\n"
    "    __private %s3 sPrivateStorage[ %d ];\n" // keep this %d
    "    int tid = get_global_id( 0 );\n"
    // We need to zero the shared storage since any locations we don't write to will have garbage otherwise.
    " sPrivateStorage[tid] = (%s3)(%s)0;\n"
    "\n"

    "   vstore3( vload3(tid,srcValues), offsets[ tid ], ( (__private %s *)sPrivateStorage ) + alignmentOffset );\n"
    "\n"
    // Note: we only copy the relevant portion of our local storage over to the dest buffer, because
    // otherwise, local threads would be overwriting results from other local threads
    "  uint i;\n"
    "  __private %s *sp = ((__private %s*) sPrivateStorage) + 3*offsets[tid] + alignmentOffset;\n"
    "  __global %s *dp = ((__global %s*) destBuffer) + 3*offsets[tid] + alignmentOffset;\n"
    "  for( i = 0; i < 3; i++ ) \n"
    "       dp[i] = sp[i];\n"
    "}\n";

    const char *typeName = get_explicit_type_name(type);
    if(inVectorSize == 3) {
        sprintf( destBuffer, patternV3,
                type == kDouble ? doubleExtensionPragma : "",
                typeName,  typeName,
                typeName, (int)inBufferSize,
                typeName, typeName,
                typeName, typeName, typeName, typeName, typeName );
    } else {
        sprintf( destBuffer, pattern,
                type == kDouble ? doubleExtensionPragma : "",
                typeName, (int)inVectorSize, typeName, (int)inVectorSize,
                typeName, (int)inVectorSize, (int)inBufferSize,
                typeName, (int)inVectorSize, typeName,
                (int)inVectorSize, typeName, typeName, typeName, typeName, typeName );
    }
}

int test_vstore_private(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
    // We have no idea how much actual private storage is available, so just pick a reasonable value,
    // which is that we can fit at least two 16-element long, which is 2*8 bytes * 16 = 256 bytes
    return test_vstoreset( device, context, queue, create_private_store_code, 256 );
}



