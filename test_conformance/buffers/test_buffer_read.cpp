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
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl_half.h>

#include "procs.h"

//#define HK_DO_NOT_RUN_SHORT_ASYNC    1
//#define HK_DO_NOT_RUN_USHORT_ASYNC    1
//#define HK_DO_NOT_RUN_CHAR_ASYNC    1
//#define HK_DO_NOT_RUN_UCHAR_ASYNC    1

#define TEST_PRIME_INT        ((1<<16)+1)
#define TEST_PRIME_UINT        ((1U<<16)+1U)
#define TEST_PRIME_LONG        ((1LL<<32)+1LL)
#define TEST_PRIME_ULONG    ((1ULL<<32)+1ULL)
#define TEST_PRIME_SHORT    ((1S<<8)+1S)
#define TEST_PRIME_FLOAT    (float)3.40282346638528860e+38
#define TEST_PRIME_HALF        119.f
#define TEST_BOOL            true
#define TEST_PRIME_CHAR        0x77

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef TestStruct
typedef struct{
    int     a;
    float   b;
} TestStruct;
#endif

//--- the code for the kernel executables
static const char *buffer_read_int_kernel_code[] = {
    "__kernel void test_buffer_read_int(__global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1<<16)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_int2(__global int2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1<<16)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_int4(__global int4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1<<16)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_int8(__global int8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1<<16)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_int16(__global int16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1<<16)+1);\n"
    "}\n" };

static const char *int_kernel_name[] = { "test_buffer_read_int", "test_buffer_read_int2", "test_buffer_read_int4", "test_buffer_read_int8", "test_buffer_read_int16" };

static const char *buffer_read_uint_kernel_code[] = {
    "__kernel void test_buffer_read_uint(__global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1U<<16)+1U);\n"
    "}\n",

    "__kernel void test_buffer_read_uint2(__global uint2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1U<<16)+1U);\n"
    "}\n",

    "__kernel void test_buffer_read_uint4(__global uint4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1U<<16)+1U);\n"
    "}\n",

    "__kernel void test_buffer_read_uint8(__global uint8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1U<<16)+1U);\n"
    "}\n",

    "__kernel void test_buffer_read_uint16(__global uint16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1U<<16)+1U);\n"
    "}\n" };

static const char *uint_kernel_name[] = { "test_buffer_read_uint", "test_buffer_read_uint2", "test_buffer_read_uint4", "test_buffer_read_uint8", "test_buffer_read_uint16" };

static const char *buffer_read_long_kernel_code[] = {
    "__kernel void test_buffer_read_long(__global long *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1L<<32)+1L);\n"
    "}\n",

    "__kernel void test_buffer_read_long2(__global long2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1L<<32)+1L);\n"
    "}\n",

    "__kernel void test_buffer_read_long4(__global long4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1L<<32)+1L);\n"
    "}\n",

    "__kernel void test_buffer_read_long8(__global long8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1L<<32)+1L);\n"
    "}\n",

    "__kernel void test_buffer_read_long16(__global long16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1L<<32)+1L);\n"
    "}\n" };

static const char *long_kernel_name[] = { "test_buffer_read_long", "test_buffer_read_long2", "test_buffer_read_long4", "test_buffer_read_long8", "test_buffer_read_long16" };

static const char *buffer_read_ulong_kernel_code[] = {
    "__kernel void test_buffer_read_ulong(__global ulong *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1UL<<32)+1UL);\n"
    "}\n",

    "__kernel void test_buffer_read_ulong2(__global ulong2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1UL<<32)+1UL);\n"
    "}\n",

    "__kernel void test_buffer_read_ulong4(__global ulong4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1UL<<32)+1UL);\n"
    "}\n",

    "__kernel void test_buffer_read_ulong8(__global ulong8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1UL<<32)+1UL);\n"
    "}\n",

    "__kernel void test_buffer_read_ulong16(__global ulong16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = ((1UL<<32)+1UL);\n"
    "}\n" };

static const char *ulong_kernel_name[] = { "test_buffer_read_ulong", "test_buffer_read_ulong2", "test_buffer_read_ulong4", "test_buffer_read_ulong8", "test_buffer_read_ulong16" };

static const char *buffer_read_short_kernel_code[] = {
    "__kernel void test_buffer_read_short(__global short *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (short)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_short2(__global short2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (short)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_short4(__global short4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (short)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_short8(__global short8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (short)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_short16(__global short16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (short)((1<<8)+1);\n"
    "}\n" };

static const char *short_kernel_name[] = { "test_buffer_read_short", "test_buffer_read_short2", "test_buffer_read_short4", "test_buffer_read_short8", "test_buffer_read_short16" };


static const char *buffer_read_ushort_kernel_code[] = {
    "__kernel void test_buffer_read_ushort(__global ushort *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (ushort)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_ushort2(__global ushort2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (ushort)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_ushort4(__global ushort4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (ushort)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_ushort8(__global ushort8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (ushort)((1<<8)+1);\n"
    "}\n",

    "__kernel void test_buffer_read_ushort16(__global ushort16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (ushort)((1<<8)+1);\n"
    "}\n" };

static const char *ushort_kernel_name[] = { "test_buffer_read_ushort", "test_buffer_read_ushort2", "test_buffer_read_ushort4", "test_buffer_read_ushort8", "test_buffer_read_ushort16" };


static const char *buffer_read_float_kernel_code[] = {
    "__kernel void test_buffer_read_float(__global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)3.40282346638528860e+38;\n"
    "}\n",

    "__kernel void test_buffer_read_float2(__global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)3.40282346638528860e+38;\n"
    "}\n",

    "__kernel void test_buffer_read_float4(__global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)3.40282346638528860e+38;\n"
    "}\n",

    "__kernel void test_buffer_read_float8(__global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)3.40282346638528860e+38;\n"
    "}\n",

    "__kernel void test_buffer_read_float16(__global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (float)3.40282346638528860e+38;\n"
    "}\n" };

static const char *float_kernel_name[] = { "test_buffer_read_float", "test_buffer_read_float2", "test_buffer_read_float4", "test_buffer_read_float8", "test_buffer_read_float16" };


static const char *buffer_read_half_kernel_code[] = {
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_buffer_read_half(__global half *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (half)119;\n"
    "}\n",

    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_buffer_read_half2(__global half2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (half)119;\n"
    "}\n",

    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_buffer_read_half4(__global half4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (half)119;\n"
    "}\n",

    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_buffer_read_half8(__global half8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (half)119;\n"
    "}\n",

    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "__kernel void test_buffer_read_half16(__global half16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (half)119;\n"
    "}\n"
};

static const char *half_kernel_name[] = { "test_buffer_read_half", "test_buffer_read_half2", "test_buffer_read_half4", "test_buffer_read_half8", "test_buffer_read_half16" };


static const char *buffer_read_char_kernel_code[] = {
    "__kernel void test_buffer_read_char(__global char *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (char)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_char2(__global char2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (char)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_char4(__global char4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (char)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_char8(__global char8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (char)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_char16(__global char16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (char)'w';\n"
    "}\n" };

static const char *char_kernel_name[] = { "test_buffer_read_char", "test_buffer_read_char2", "test_buffer_read_char4", "test_buffer_read_char8", "test_buffer_read_char16" };


static const char *buffer_read_uchar_kernel_code[] = {
    "__kernel void test_buffer_read_uchar(__global uchar *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = 'w';\n"
    "}\n",

    "__kernel void test_buffer_read_uchar2(__global uchar2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (uchar)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_uchar4(__global uchar4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (uchar)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_uchar8(__global uchar8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (uchar)'w';\n"
    "}\n",

    "__kernel void test_buffer_read_uchar16(__global uchar16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = (uchar)'w';\n"
    "}\n" };

static const char *uchar_kernel_name[] = { "test_buffer_read_uchar", "test_buffer_read_uchar2", "test_buffer_read_uchar4", "test_buffer_read_uchar8", "test_buffer_read_uchar16" };


static const char *buffer_read_struct_kernel_code =
"typedef struct{\n"
"int    a;\n"
"float    b;\n"
"} TestStruct;\n"
"__kernel void test_buffer_read_struct(__global TestStruct *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid].a = ((1<<16)+1);\n"
"     dst[tid].b = (float)3.40282346638528860e+38;\n"
"}\n";


//--- the verify functions
static int verify_read_int(void *ptr, int n)
{
    int     i;
    cl_int  *outptr = (cl_int *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_INT )
            return -1;
    }

    return 0;
}


static int verify_read_uint(void *ptr, int n)
{
    int     i;
    cl_uint *outptr = (cl_uint *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_UINT )
            return -1;
    }

    return 0;
}


static int verify_read_long(void *ptr, int n)
{
    int     i;
    cl_long *outptr = (cl_long *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_LONG )
            return -1;
    }

    return 0;
}


static int verify_read_ulong(void *ptr, int n)
{
    int      i;
    cl_ulong *outptr = (cl_ulong *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_ULONG )
            return -1;
    }

    return 0;
}


static int verify_read_short(void *ptr, int n)
{
    int      i;
    cl_short *outptr = (cl_short *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != (cl_short)((1<<8)+1) )
            return -1;
    }

    return 0;
}


static int verify_read_ushort(void *ptr, int n)
{
    int       i;
    cl_ushort *outptr = (cl_ushort *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != (cl_ushort)((1<<8)+1) )
            return -1;
    }

    return 0;
}


static int verify_read_float( void *ptr, int n )
{
    int      i;
    cl_float *outptr = (cl_float *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_FLOAT )
            return -1;
    }

    return 0;
}


static int verify_read_half( void *ptr, int n )
{
    int     i;
    cl_half *outptr = (cl_half *)ptr;

    for (i = 0; i < n; i++)
    {
        if (cl_half_to_float(outptr[i]) != TEST_PRIME_HALF) return -1;
    }

    return 0;
}


static int verify_read_char(void *ptr, int n)
{
    int     i;
    cl_char *outptr = (cl_char *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_uchar(void *ptr, int n)
{
    int      i;
    cl_uchar *outptr = (cl_uchar *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_struct(TestStruct *outptr, int n)
{
    int     i;

    for (i=0; i<n; i++)
    {
        if ( ( outptr[i].a != TEST_PRIME_INT ) ||
             ( outptr[i].b != TEST_PRIME_FLOAT ) )
            return -1;
    }

    return 0;
}

//----- the test functions
int test_buffer_read( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                      const char *kernelCode[], const char *kernelName[], int (*fn)(void *,int) )
{
    void        *outptr[5];
    void        *inptr[5];
    clProgramWrapper program[5];
    clKernelWrapper kernel[5];
    size_t      global_work_size[3];
    cl_int      err;
    int         i;
    size_t      ptrSizes[5];
    int         src_flag_id;
    int         total_errors = 0;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong && strstr(type,"long") )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for (i = 0; i < loops; i++)
    {

        err = create_single_kernel_helper(context, &program[i], &kernel[i], 1,
                                          &kernelCode[i], kernelName[i]);
        if (err)
        {
            log_error("Creating program for %s\n", type);
            print_error(err, " Error creating program ");
            return -1;
        }

        for (src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
        {
            clMemWrapper buffer;
            outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
            if ( ! outptr[i] ){
                log_error( " unable to allocate %d bytes for outptr\n", (int)( ptrSizes[i] * num_elements ) );
                return -1;
            }
            inptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
            if ( ! inptr[i] ){
                log_error( " unable to allocate %d bytes for inptr\n", (int)( ptrSizes[i] * num_elements ) );
                return -1;
            }


            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffer =
                    clCreateBuffer(context, flag_set[src_flag_id],
                                   ptrSizes[i] * num_elements, inptr[i], &err);
            else
                buffer = clCreateBuffer(context, flag_set[src_flag_id],
                                        ptrSizes[i] * num_elements, NULL, &err);
            if (err != CL_SUCCESS)
            {
                print_error(err, " clCreateBuffer failed\n" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&buffer);
            if ( err != CL_SUCCESS ){
                print_error( err, "clSetKernelArg failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL,
                                         global_work_size, NULL, 0, NULL, NULL);
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueNDRangeKernel failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                                      ptrSizes[i] * num_elements, outptr[i], 0,
                                      NULL, NULL);
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueReadBuffer failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            if (fn(outptr[i], num_elements*(1<<i))){
                log_error(" %s%d test failed. cl_mem_flags src: %s\n", type,
                          1 << i, flag_set_names[src_flag_id]);
                total_errors++;
            }
            else{
                log_info(" %s%d test passed. cl_mem_flags src: %s\n", type,
                         1 << i, flag_set_names[src_flag_id]);
            }

            err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                                      ptrSizes[i] * num_elements, inptr[i], 0,
                                      NULL, NULL);
            if (err != CL_SUCCESS)
            {
                print_error( err, "clEnqueueReadBuffer failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            if (fn(inptr[i], num_elements*(1<<i))){
                log_error( " %s%d test failed in-place readback\n", type, 1<<i );
                total_errors++;
            }
            else{
                log_info( " %s%d test passed in-place readback\n", type, 1<<i );
            }


            // cleanup
            align_free( outptr[i] );
            align_free( inptr[i] );
        }
    } // mem flag

    return total_errors;

}   // end test_buffer_read()

int test_buffer_read_async( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                            const char *kernelCode[], const char *kernelName[], int (*fn)(void *,int) )
{
    clProgramWrapper program[5];
    clKernelWrapper kernel[5];
    void        *outptr[5];
    void        *inptr[5];
    size_t      global_work_size[3];
    cl_int      err;
    int         i;
    size_t      lastIndex;
    size_t      ptrSizes[5];
    int         src_flag_id;
    int         total_errors = 0;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong && strstr(type,"long") )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for (i = 0; i < loops; i++)
    {

        err = create_single_kernel_helper(context, &program[i], &kernel[i], 1,
                                          &kernelCode[i], kernelName[i]);
        if (err)
        {
            log_error(" Error creating program for %s\n", type);
            return -1;
        }

        for (src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
        {
            clMemWrapper buffer;
            clEventWrapper event;
            outptr[i] = align_malloc(ptrSizes[i] * num_elements, min_alignment);
            if ( ! outptr[i] ){
                log_error( " unable to allocate %d bytes for outptr\n", (int)(ptrSizes[i] * num_elements) );
                return -1;
            }
            memset( outptr[i], 0, ptrSizes[i] * num_elements ); // initialize to zero to tell difference
            inptr[i] = align_malloc(ptrSizes[i] * num_elements, min_alignment);
            if ( ! inptr[i] ){
                log_error( " unable to allocate %d bytes for inptr\n", (int)(ptrSizes[i] * num_elements) );
                return -1;
            }
            memset( inptr[i], 0, ptrSizes[i] * num_elements );  // initialize to zero to tell difference


            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffer =
                    clCreateBuffer(context, flag_set[src_flag_id],
                                   ptrSizes[i] * num_elements, inptr[i], &err);
            else
                buffer = clCreateBuffer(context, flag_set[src_flag_id],
                                        ptrSizes[i] * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&buffer);
            if ( err != CL_SUCCESS ){
                print_error( err, "clSetKernelArg failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueNDRangeKernel failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            lastIndex = ( num_elements * ( 1 << i ) - 1 ) * ptrSizes[0];
            err = clEnqueueReadBuffer(queue, buffer, false, 0,
                                      ptrSizes[i] * num_elements, outptr[i], 0,
                                      NULL, &event);
#ifdef CHECK_FOR_NON_WAIT
            if ( ((uchar *)outptr[i])[lastIndex] ){
                log_error( "    clEnqueueReadBuffer() possibly returned only after inappropriately waiting for execution to be finished\n" );
                log_error( "    Function was run asynchornously, but last value in array was set in code line following clEnqueueReadBuffer()\n" );
            }
#endif
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueReadBuffer failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }
            err = clWaitForEvents(1, &event );
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            if ( fn(outptr[i], num_elements*(1<<i)) ){
                log_error(" %s%d test failed. cl_mem_flags src: %s\n", type,
                          1 << i, flag_set_names[src_flag_id]);
                total_errors++;
            }
            else{
                log_info(" %s%d test passed. cl_mem_flags src: %s\n", type,
                         1 << i, flag_set_names[src_flag_id]);
            }

            // cleanup
            align_free( outptr[i] );
            align_free( inptr[i] );
        }
    } // mem flags


    return total_errors;

}   // end test_buffer_read_array_async()


int test_buffer_read_array_barrier( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                                    const char *kernelCode[], const char *kernelName[], int (*fn)(void *,int) )
{
    clProgramWrapper program[5];
    clKernelWrapper kernel[5];
    void        *outptr[5], *inptr[5];
    size_t      global_work_size[3];
    cl_int      err;
    int         i;
    size_t      lastIndex;
    size_t      ptrSizes[5];
    int         src_flag_id;
    int         total_errors = 0;

    size_t min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong && strstr(type,"long") )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for (i = 0; i < loops; i++)
    {

        err = create_single_kernel_helper(context, &program[i], &kernel[i], 1,
                                          &kernelCode[i], kernelName[i]);
        if (err)
        {
            log_error(" Error creating program for %s\n", type);
            return -1;
        }

        for (src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
        {
            clMemWrapper buffer;
            clEventWrapper event;
            outptr[i] = align_malloc(ptrSizes[i] * num_elements, min_alignment);
            if ( ! outptr[i] ){
                log_error( " unable to allocate %d bytes for outptr\n", (int)(ptrSizes[i] * num_elements) );
                return -1;
            }
            memset( outptr[i], 0, ptrSizes[i] * num_elements ); // initialize to zero to tell difference
            inptr[i] = align_malloc(ptrSizes[i] * num_elements, min_alignment);
            if ( ! inptr[i] ){
                log_error( " unable to allocate %d bytes for inptr\n", (int)(ptrSizes[i] * num_elements) );
                return -1;
            }
            memset( inptr[i], 0, ptrSizes[i] * num_elements );  // initialize to zero to tell difference

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffer =
                    clCreateBuffer(context, flag_set[src_flag_id],
                                   ptrSizes[i] * num_elements, inptr[i], &err);
            else
                buffer = clCreateBuffer(context, flag_set[src_flag_id],
                                        ptrSizes[i] * num_elements, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error(err, " clCreateBuffer failed\n" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&buffer);
            if ( err != CL_SUCCESS ){
                print_error( err, "clSetKernelArgs failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueNDRangeKernel failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            lastIndex = ( num_elements * ( 1 << i ) - 1 ) * ptrSizes[0];
            err = clEnqueueReadBuffer(queue, buffer, false, 0,
                                      ptrSizes[i] * num_elements,
                                      (void *)(outptr[i]), 0, NULL, &event);
#ifdef CHECK_FOR_NON_WAIT
            if ( ((uchar *)outptr[i])[lastIndex] ){
                log_error( "    clEnqueueReadBuffer() possibly returned only after inappropriately waiting for execution to be finished\n" );
                log_error( "    Function was run asynchornously, but last value in array was set in code line following clEnqueueReadBuffer()\n" );
            }
#endif
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueReadBuffer failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }
            err = clEnqueueBarrierWithWaitList(queue, 0, NULL, NULL);
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueBarrierWithWaitList() failed" );
                align_free( outptr[i] );
                return -1;
            }

            err = clWaitForEvents(1, &event);
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
                align_free( outptr[i] );
                align_free( inptr[i] );
                return -1;
            }

            if ( fn(outptr[i], num_elements*(1<<i)) ){
                log_error(" %s%d test failed. cl_mem_flags src: %s\n", type,
                          1 << i, flag_set_names[src_flag_id]);
                total_errors++;
            }
            else{
                log_info(" %s%d test passed. cl_mem_flags src: %s\n", type,
                         1 << i, flag_set_names[src_flag_id]);
            }

            // cleanup
            align_free( outptr[i] );
            align_free( inptr[i] );
        }
    } // cl_mem flags
    return total_errors;

}   // end test_buffer_read_array_barrier()


#define DECLARE_READ_TEST(type, realType) \
int test_buffer_read_##type( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )    \
{ \
return test_buffer_read( deviceID, context, queue, num_elements, sizeof( realType ), (char*)#type, 5, \
buffer_read_##type##_kernel_code, type##_kernel_name, verify_read_##type ); \
}

DECLARE_READ_TEST(int, cl_int)
DECLARE_READ_TEST(uint, cl_uint)
DECLARE_READ_TEST(long, cl_long)
DECLARE_READ_TEST(ulong, cl_ulong)
DECLARE_READ_TEST(short, cl_short)
DECLARE_READ_TEST(ushort, cl_ushort)
DECLARE_READ_TEST(float, cl_float)
DECLARE_READ_TEST(char, cl_char)
DECLARE_READ_TEST(uchar, cl_uchar)

int test_buffer_read_half(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    PASSIVE_REQUIRE_FP16_SUPPORT(deviceID)
    return test_buffer_read( deviceID, context, queue, num_elements, sizeof( cl_float ) / 2, (char*)"half", 5,
                             buffer_read_half_kernel_code, half_kernel_name, verify_read_half );
}


#define DECLARE_ASYNC_TEST(type, realType) \
int test_buffer_read_async_##type( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )    \
{ \
return test_buffer_read_async( deviceID, context, queue, num_elements, sizeof( realType ), (char*)#type, 5, \
buffer_read_##type##_kernel_code, type##_kernel_name, verify_read_##type ); \
}

DECLARE_ASYNC_TEST(char, cl_char)
DECLARE_ASYNC_TEST(uchar, cl_uchar)
DECLARE_ASYNC_TEST(short, cl_short)
DECLARE_ASYNC_TEST(ushort, cl_ushort)
DECLARE_ASYNC_TEST(int, cl_int)
DECLARE_ASYNC_TEST(uint, cl_uint)
DECLARE_ASYNC_TEST(long, cl_long)
DECLARE_ASYNC_TEST(ulong, cl_ulong)
DECLARE_ASYNC_TEST(float, cl_float)


#define DECLARE_BARRIER_TEST(type, realType) \
int test_buffer_read_array_barrier_##type( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )    \
{ \
return test_buffer_read_array_barrier( deviceID, context, queue, num_elements, sizeof( realType ), (char*)#type, 5, \
buffer_read_##type##_kernel_code, type##_kernel_name, verify_read_##type ); \
}

DECLARE_BARRIER_TEST(int, cl_int)
DECLARE_BARRIER_TEST(uint, cl_uint)
DECLARE_BARRIER_TEST(long, cl_long)
DECLARE_BARRIER_TEST(ulong, cl_ulong)
DECLARE_BARRIER_TEST(short, cl_short)
DECLARE_BARRIER_TEST(ushort, cl_ushort)
DECLARE_BARRIER_TEST(char, cl_char)
DECLARE_BARRIER_TEST(uchar, cl_uchar)
DECLARE_BARRIER_TEST(float, cl_float)

int test_buffer_read_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem      buffers[1];
    TestStruct  *output_ptr;
    cl_program  program[1];
    cl_kernel   kernel[1];
    size_t      global_work_size[3];
    cl_int      err;
    size_t      objSize = sizeof(TestStruct);

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    output_ptr = (TestStruct*)align_malloc(objSize * num_elements, min_alignment);
    if ( ! output_ptr ){
        log_error( " unable to allocate %d bytes for output_ptr\n", (int)(objSize * num_elements) );
        return -1;
    }
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                objSize * num_elements, NULL, &err);
    if ( err != CL_SUCCESS ){
        print_error( err, " clCreateBuffer failed\n" );
        align_free( output_ptr );
        return -1;
    }

    err = create_single_kernel_helper(  context, &program[0], &kernel[0], 1, &buffer_read_struct_kernel_code, "test_buffer_read_struct" );
    if ( err ){
        clReleaseProgram( program[0] );
        align_free( output_ptr );
        return -1;
    }

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&buffers[0] );
    if ( err != CL_SUCCESS){
        print_error( err, "clSetKernelArg failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( output_ptr );
        return -1;
    }

    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueNDRangeKernel failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( output_ptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, buffers[0], true, 0, objSize*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if ( err != CL_SUCCESS){
        print_error( err, "clEnqueueReadBuffer failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( output_ptr );
        return -1;
    }

    if (verify_read_struct(output_ptr, num_elements)){
        log_error(" struct test failed\n");
        err = -1;
    }
    else{
        log_info(" struct test passed\n");
        err = 0;
    }

    // cleanup
    clReleaseMemObject( buffers[0] );
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    align_free( output_ptr );

    return err;
}


static int testRandomReadSize( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, cl_uint startOfRead, size_t sizeOfRead )
{
    cl_mem      buffers[3];
    int         *outptr[3];
    cl_program  program[3];
    cl_kernel   kernel[3];
    size_t      global_work_size[3];
    cl_int      err;
    int         i, j;
    size_t      ptrSizes[3];    // sizeof(int), sizeof(int2), sizeof(int4)
    int         total_errors = 0;
    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    for ( i = 0; i < 3; i++ ){
        outptr[i] = (int *)align_malloc( ptrSizes[i] * num_elements, min_alignment);
        if ( ! outptr[i] ){
            log_error( " Unable to allocate %d bytes for outptr[%d]\n", (int)(ptrSizes[i] * num_elements), i );
            for ( j = 0; j < i; j++ ){
                clReleaseMemObject( buffers[j] );
                align_free( outptr[j] );
            }
            return -1;
        }
        buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    ptrSizes[i] * num_elements, NULL, &err);
        if ( err != CL_SUCCESS ){
            print_error(err, " clCreateBuffer failed\n" );
            for ( j = 0; j < i; j++ ){
                clReleaseMemObject( buffers[j] );
                align_free( outptr[j] );
            }
            align_free( outptr[i] );
            return -1;
        }
    }

    err = create_single_kernel_helper(  context, &program[0], &kernel[0], 1, &buffer_read_int_kernel_code[0], "test_buffer_read_int" );
    if ( err ){
        log_error( " Error creating program for int\n" );
        for ( i = 0; i < 3; i++ ){
            clReleaseMemObject( buffers[i] );
            align_free( outptr[i] );
        }
        return -1;
    }

    err = create_single_kernel_helper(  context, &program[1], &kernel[1], 1, &buffer_read_int_kernel_code[1], "test_buffer_read_int2" );
    if ( err ){
        log_error( " Error creating program for int2\n" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        for ( i = 0; i < 3; i++ ){
            clReleaseMemObject( buffers[i] );
            align_free( outptr[i] );
        }
        return -1;
    }

    err = create_single_kernel_helper(  context, &program[2], &kernel[2], 1, &buffer_read_int_kernel_code[2], "test_buffer_read_int4" );
    if ( err ){
        log_error( " Error creating program for int4\n" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseKernel( kernel[1] );
        clReleaseProgram( program[1] );
        for ( i = 0; i < 3; i++ ){
            clReleaseMemObject( buffers[i] );
            align_free( outptr[i] );
        }
        return -1;
    }

    for (i=0; i<3; i++){
        err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&buffers[i] );
        if ( err != CL_SUCCESS ){
            print_error( err, "clSetKernelArgs failed" );
            clReleaseMemObject( buffers[i] );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            align_free( outptr[i] );
            return -1;
        }

        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
        if ( err != CL_SUCCESS ){
            print_error( err, "clEnqueueNDRangeKernel failed" );
            clReleaseMemObject( buffers[i] );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            align_free( outptr[i] );
            return -1;
        }

        err = clEnqueueReadBuffer( queue, buffers[i], true, startOfRead*ptrSizes[i], ptrSizes[i]*sizeOfRead, (void *)(outptr[i]), 0, NULL, NULL );
        if ( err != CL_SUCCESS ){
            print_error( err, "clEnqueueReadBuffer failed" );
            clReleaseMemObject( buffers[i] );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            align_free( outptr[i] );
            return -1;
        }

        if ( verify_read_int( outptr[i], (int)sizeOfRead*(1<<i) ) ){
            log_error(" random size from %d, size: %d test failed on i%d\n", (int)startOfRead, (int)sizeOfRead, 1<<i);
            total_errors++;
        }
        else{
            log_info(" random size from %d, size: %d test passed on i%d\n", (int)startOfRead, (int)sizeOfRead, 1<<i);
        }

        // cleanup
        clReleaseMemObject( buffers[i] );
        clReleaseKernel( kernel[i] );
        clReleaseProgram( program[i] );
        align_free( outptr[i] );
    }

    return total_errors;

}   // end testRandomReadSize()


int test_buffer_read_random_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int     err = 0;
    int     i;
    cl_uint start;
    size_t  size;
    MTdata  d = init_genrand( gRandomSeed );

    // now test for random sizes of array being read
    for ( i = 0; i < 8; i++ ){
        start = (cl_uint)get_random_float( 0.f, (float)(num_elements - 8), d );
        size = (size_t)get_random_float( 8.f, (float)(num_elements - start), d );
        if (testRandomReadSize( deviceID, context, queue, num_elements, start, size ))
            err++;
    }

    free_mtdata(d);

    return err;
}

