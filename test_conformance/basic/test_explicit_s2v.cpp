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
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

#define DECLARE_S2V_IDENT_KERNEL(srctype,dsttype,size) \
"__kernel void test_conversion(__global " srctype " *sourceValues, __global " dsttype #size " *destValues )\n"        \
"{\n"                                                                            \
"    int  tid = get_global_id(0);\n"                                        \
"    " srctype "  src = sourceValues[tid];\n"                                        \
"\n"                                                                            \
"    destValues[tid] = (" dsttype #size ")src;\n"                        \
"\n"                                                                            \
"}\n"

#define DECLARE_S2V_IDENT_KERNELS(srctype,dsttype) \
{        \
DECLARE_S2V_IDENT_KERNEL(srctype,#dsttype,2), \
DECLARE_S2V_IDENT_KERNEL(srctype,#dsttype,4), \
DECLARE_S2V_IDENT_KERNEL(srctype,#dsttype,8), \
DECLARE_S2V_IDENT_KERNEL(srctype,#dsttype,16) \
}

#define DECLARE_EMPTY { NULL, NULL, NULL, NULL, NULL }

/* Note: the next four arrays all must match in order and size to the ExplicitTypes enum in conversions.h!!! */

#define DECLARE_S2V_IDENT_KERNELS_SET(srctype)    \
{                                                    \
DECLARE_S2V_IDENT_KERNELS(#srctype,bool),            \
            DECLARE_S2V_IDENT_KERNELS(#srctype,char),            \
            DECLARE_S2V_IDENT_KERNELS(#srctype,uchar),            \
            DECLARE_S2V_IDENT_KERNELS(#srctype,unsigned char),    \
DECLARE_S2V_IDENT_KERNELS(#srctype,short),            \
DECLARE_S2V_IDENT_KERNELS(#srctype,ushort),            \
DECLARE_S2V_IDENT_KERNELS(#srctype,unsigned short),    \
DECLARE_S2V_IDENT_KERNELS(#srctype,int),                \
DECLARE_S2V_IDENT_KERNELS(#srctype,uint),            \
DECLARE_S2V_IDENT_KERNELS(#srctype,unsigned int),    \
DECLARE_S2V_IDENT_KERNELS(#srctype,long),            \
DECLARE_S2V_IDENT_KERNELS(#srctype,ulong),            \
DECLARE_S2V_IDENT_KERNELS(#srctype,unsigned long),    \
DECLARE_S2V_IDENT_KERNELS(#srctype,float),            \
DECLARE_EMPTY                                        \
}

#define DECLARE_EMPTY_SET                \
{                                                    \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY, \
DECLARE_EMPTY    \
}


/* The overall array */
const char * kernel_explicit_s2v_set[kNumExplicitTypes][kNumExplicitTypes][5] = {
    DECLARE_S2V_IDENT_KERNELS_SET(bool),
    DECLARE_S2V_IDENT_KERNELS_SET(char),
    DECLARE_S2V_IDENT_KERNELS_SET(uchar),
    DECLARE_S2V_IDENT_KERNELS_SET(unsigned char),
    DECLARE_S2V_IDENT_KERNELS_SET(short),
    DECLARE_S2V_IDENT_KERNELS_SET(ushort),
    DECLARE_S2V_IDENT_KERNELS_SET(unsigned short),
    DECLARE_S2V_IDENT_KERNELS_SET(int),
    DECLARE_S2V_IDENT_KERNELS_SET(uint),
    DECLARE_S2V_IDENT_KERNELS_SET(unsigned int),
    DECLARE_S2V_IDENT_KERNELS_SET(long),
    DECLARE_S2V_IDENT_KERNELS_SET(ulong),
    DECLARE_S2V_IDENT_KERNELS_SET(unsigned long),
    DECLARE_S2V_IDENT_KERNELS_SET(float),
    DECLARE_EMPTY_SET
};

int test_explicit_s2v_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, const char *programSrc,
                               ExplicitType srcType, unsigned int count, ExplicitType destType, unsigned int vecSize, void *inputData )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    int error;
    clMemWrapper streams[2];
    void *outData;
    unsigned char convertedData[ 8 ];    /* Max type size is 8 bytes */
    size_t threadSize[3], groupSize[3];
    unsigned int i, s;
    unsigned char *inPtr, *outPtr;
    size_t paramSize, destTypeSize;

    const char* finalProgramSrc[2] = {
        "", // optional pragma
        programSrc
    };

    if (srcType == kDouble || destType == kDouble) {
        finalProgramSrc[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }


    if( programSrc == NULL )
        return 0;

    paramSize = get_explicit_type_size( srcType );
    destTypeSize = get_explicit_type_size( destType );

    size_t destStride = destTypeSize * vecSize;

    outData = malloc( destStride * count );

    if( create_single_kernel_helper( context, &program, &kernel, 2, finalProgramSrc, "test_conversion" ) )
    {
        log_info( "****** %s%s *******\n", finalProgramSrc[0], finalProgramSrc[1] );
        return -1;
    }

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), paramSize * count, inputData, &error);
    test_error( error, "clCreateBuffer failed");
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  destStride * count, NULL, &error);
    test_error( error, "clCreateBuffer failed");

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1] );
    test_error( error, "Unable to set indexed kernel arguments" );

    /* Run the kernel */
    threadSize[0] = count;

    error = get_max_common_work_group_size( context, kernel, threadSize[0], &groupSize[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threadSize, groupSize, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    /* Now verify the results. Each value should have been duplicated four times, and we should be able to just
     do a memcpy instead of relying on the actual type of data */
    error = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, destStride * count, outData, 0, NULL, NULL );
    test_error( error, "Unable to read output values!" );

    inPtr = (unsigned char *)inputData;
    outPtr = (unsigned char *)outData;

    for( i = 0; i < count; i++ )
    {
        /* Convert the input data element to our output data type to compare against */
        convert_explicit_value( (void *)inPtr, (void *)convertedData, srcType, false, kDefaultRoundingType, destType );

        /* Now compare every element of the vector */
        for( s = 0; s < vecSize; s++ )
        {
            if( memcmp( convertedData, outPtr + destTypeSize * s, destTypeSize ) != 0 )
            {
                unsigned int *p = (unsigned int *)outPtr;
                log_error( "ERROR: Output value %d:%d does not validate for size %d:%d!\n", i, s, vecSize, (int)destTypeSize );
                log_error( "       Input:   0x%0*x\n", (int)( paramSize * 2 ), *(unsigned int *)inPtr & ( 0xffffffff >> ( 32 - paramSize * 8 ) ) );
                log_error( "       Actual:  0x%08x 0x%08x 0x%08x 0x%08x\n", p[ 0 ], p[ 1 ], p[ 2 ], p[ 3 ] );
                return -1;
            }
        }
        inPtr += paramSize;
        outPtr += destStride;
    }

    free( outData );

    return 0;
}

int test_explicit_s2v_function_set(cl_device_id deviceID, cl_context context, cl_command_queue queue, ExplicitType srcType,
                                   unsigned int count, void *inputData )
{
    unsigned int sizes[] = { 2, 4, 8, 16, 0 };
    int i, dstType, failed = 0;


    for( dstType = kBool; dstType < kNumExplicitTypes; dstType++ )
    {
        if( dstType == kDouble && !is_extension_available( deviceID, "cl_khr_fp64" ) )
            continue;

        if (( dstType == kLong || dstType == kULong ) && !gHasLong )
            continue;

        for( i = 0; sizes[i] != 0; i++ )
        {
            if( dstType != srcType )
                continue;
            if( strchr( get_explicit_type_name( (ExplicitType)srcType ), ' ' ) != NULL ||
               strchr( get_explicit_type_name( (ExplicitType)dstType ), ' ' ) != NULL )
                continue;

            if( test_explicit_s2v_function( deviceID, context, queue, kernel_explicit_s2v_set[ srcType ][ dstType ][ i ],
                                           srcType, count, (ExplicitType)dstType, sizes[ i ], inputData ) != 0 )
            {
                log_error( "ERROR: Explicit cast of scalar %s to vector %s%d FAILED; skipping other %s vector tests\n",
                          get_explicit_type_name(srcType), get_explicit_type_name((ExplicitType)dstType), sizes[i], get_explicit_type_name((ExplicitType)dstType) );
                failed = -1;
                break;
            }
        }
    }

    return failed;
}

int test_explicit_s2v_char(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kChar, 128, seed, data );

    return test_explicit_s2v_function_set( deviceID, context, queue, kChar, 128, data );
}

int test_explicit_s2v_uchar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    unsigned char    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kUChar, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kUChar, 128, data ) != 0 )
        return -1;
    if( test_explicit_s2v_function_set( deviceID, context, queue, kUnsignedChar, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_short(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    short            data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kShort, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kShort, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_ushort(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    unsigned short    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kUShort, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kUShort, 128, data ) != 0 )
        return -1;
    if( test_explicit_s2v_function_set( deviceID, context, queue, kUnsignedShort, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int                data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kInt, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kInt, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    unsigned int    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kUInt, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kUInt, 128, data ) != 0 )
        return -1;
    if( test_explicit_s2v_function_set( deviceID, context, queue, kUnsignedInt, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_long    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kLong, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kLong,  128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_ulong    data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kULong, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kULong,  128, data ) != 0 )
        return -1;
    if( test_explicit_s2v_function_set( deviceID, context, queue, kUnsignedLong, 128, data ) != 0 )
        return -1;
    return 0;
}

int test_explicit_s2v_float(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    float            data[128];
    RandomSeed seed(gRandomSeed);

    generate_random_data( kFloat, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kFloat, 128, data ) != 0 )
        return -1;
    return 0;
}


int test_explicit_s2v_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    double            data[128];
    RandomSeed seed(gRandomSeed);

    if( !is_extension_available( deviceID, "cl_khr_fp64" ) ) {
        log_info("Extension cl_khr_fp64 not supported. Skipping test.\n");
        return 0;
    }

    generate_random_data( kDouble, 128, seed, data );

    if( test_explicit_s2v_function_set( deviceID, context, queue, kDouble, 128, data ) != 0 )
        return -1;
    return 0;
}


