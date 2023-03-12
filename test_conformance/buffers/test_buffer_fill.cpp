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
#include "harness/errorHelpers.h"

#define TEST_PRIME_CHAR        0x77
#define TEST_PRIME_INT        ((1<<16)+1)
#define TEST_PRIME_UINT        ((1U<<16)+1U)
#define TEST_PRIME_LONG        ((1LL<<32)+1LL)
#define TEST_PRIME_ULONG    ((1ULL<<32)+1ULL)
#define TEST_PRIME_SHORT    (cl_short)((1<<8)+1)
#define TEST_PRIME_USHORT   (cl_ushort)((1<<8)+1)
#define TEST_PRIME_FLOAT    (cl_float)3.40282346638528860e+38
#define TEST_PRIME_HALF        119.f

#ifndef TestStruct
typedef struct{
    cl_int     a;
    cl_float   b;
} TestStruct;
#endif

const char *buffer_fill_int_kernel_code[] = {
    "__kernel void test_buffer_fill_int(__global int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_int2(__global int2 *src, __global int2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_int4(__global int4 *src, __global int4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_int8(__global int8 *src, __global int8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_int16(__global int16 *src, __global int16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *int_kernel_name[] = { "test_buffer_fill_int", "test_buffer_fill_int2", "test_buffer_fill_int4", "test_buffer_fill_int8", "test_buffer_fill_int16" };


const char *buffer_fill_uint_kernel_code[] = {
    "__kernel void test_buffer_fill_uint(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uint2(__global uint2 *src, __global uint2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uint4(__global uint4 *src, __global uint4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uint8(__global uint8 *src, __global uint8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uint16(__global uint16 *src, __global uint16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uint_kernel_name[] = { "test_buffer_fill_uint", "test_buffer_fill_uint2", "test_buffer_fill_uint4", "test_buffer_fill_uint8", "test_buffer_fill_uint16" };


const char *buffer_fill_short_kernel_code[] = {
    "__kernel void test_buffer_fill_short(__global short *src, __global short *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_short2(__global short2 *src, __global short2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_short4(__global short4 *src, __global short4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_short8(__global short8 *src, __global short8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_short16(__global short16 *src, __global short16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *short_kernel_name[] = { "test_buffer_fill_short", "test_buffer_fill_short2", "test_buffer_fill_short4", "test_buffer_fill_short8", "test_buffer_fill_short16" };


const char *buffer_fill_ushort_kernel_code[] = {
    "__kernel void test_buffer_fill_ushort(__global ushort *src, __global ushort *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ushort2(__global ushort2 *src, __global ushort2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ushort4(__global ushort4 *src, __global ushort4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ushort8(__global ushort8 *src, __global ushort8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ushort16(__global ushort16 *src, __global ushort16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ushort_kernel_name[] = { "test_buffer_fill_ushort", "test_buffer_fill_ushort2", "test_buffer_fill_ushort4", "test_buffer_fill_ushort8", "test_buffer_fill_ushort16" };


const char *buffer_fill_char_kernel_code[] = {
    "__kernel void test_buffer_fill_char(__global char *src, __global char *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_char2(__global char2 *src, __global char2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_char4(__global char4 *src, __global char4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_char8(__global char8 *src, __global char8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_char16(__global char16 *src, __global char16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *char_kernel_name[] = { "test_buffer_fill_char", "test_buffer_fill_char2", "test_buffer_fill_char4", "test_buffer_fill_char8", "test_buffer_fill_char16" };


const char *buffer_fill_uchar_kernel_code[] = {
    "__kernel void test_buffer_fill_uchar(__global uchar *src, __global uchar *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uchar2(__global uchar2 *src, __global uchar2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uchar4(__global uchar4 *src, __global uchar4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uchar8(__global uchar8 *src, __global uchar8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_uchar16(__global uchar16 *src, __global uchar16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uchar_kernel_name[] = { "test_buffer_fill_uchar", "test_buffer_fill_uchar2", "test_buffer_fill_uchar4", "test_buffer_fill_uchar8", "test_buffer_fill_uchar16" };


const char *buffer_fill_long_kernel_code[] = {
    "__kernel void test_buffer_fill_long(__global long *src, __global long *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_long2(__global long2 *src, __global long2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_long4(__global long4 *src, __global long4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_long8(__global long8 *src, __global long8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_long16(__global long16 *src, __global long16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *long_kernel_name[] = { "test_buffer_fill_long", "test_buffer_fill_long2", "test_buffer_fill_long4", "test_buffer_fill_long8", "test_buffer_fill_long16" };


const char *buffer_fill_ulong_kernel_code[] = {
    "__kernel void test_buffer_fill_ulong(__global ulong *src, __global ulong *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ulong2(__global ulong2 *src, __global ulong2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ulong4(__global ulong4 *src, __global ulong4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ulong8(__global ulong8 *src, __global ulong8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_ulong16(__global ulong16 *src, __global ulong16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ulong_kernel_name[] = { "test_buffer_fill_ulong", "test_buffer_fill_ulong2", "test_buffer_fill_ulong4", "test_buffer_fill_ulong8", "test_buffer_fill_ulong16" };


const char *buffer_fill_float_kernel_code[] = {
    "__kernel void test_buffer_fill_float(__global float *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_float2(__global float2 *src, __global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_float4(__global float4 *src, __global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_float8(__global float8 *src, __global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_fill_float16(__global float16 *src, __global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *float_kernel_name[] = { "test_buffer_fill_float", "test_buffer_fill_float2", "test_buffer_fill_float4", "test_buffer_fill_float8", "test_buffer_fill_float16" };


static const char *struct_kernel_code =
"typedef struct{\n"
"int    a;\n"
"float    b;\n"
"} TestStruct;\n"
"__kernel void read_fill_struct(__global TestStruct *src, __global TestStruct *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid].a = src[tid].a;\n"
"     dst[tid].b = src[tid].b;\n"
"}\n";



static int verify_fill_int( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_int  *inptr = (cl_int *)ptr1;
    cl_int  *outptr = (cl_int *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_uint( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_uint *inptr = (cl_uint *)ptr1;
    cl_uint *outptr = (cl_uint *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_short( void *ptr1, void *ptr2, int n )
{
    int      i;
    cl_short *inptr = (cl_short *)ptr1;
    cl_short *outptr = (cl_short *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_ushort( void *ptr1, void *ptr2, int n )
{
    int       i;
    cl_ushort *inptr = (cl_ushort *)ptr1;
    cl_ushort *outptr = (cl_ushort *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_char( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_char *inptr = (cl_char *)ptr1;
    cl_char *outptr = (cl_char *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_uchar( void *ptr1, void *ptr2, int n )
{
    int      i;
    cl_uchar *inptr = (cl_uchar *)ptr1;
    cl_uchar *outptr = (cl_uchar *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_long( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_long *inptr = (cl_long *)ptr1;
    cl_long *outptr = (cl_long *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_ulong( void *ptr1, void *ptr2, int n )
{
    int      i;
    cl_ulong *inptr = (cl_ulong *)ptr1;
    cl_ulong *outptr = (cl_ulong *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_float( void *ptr1, void *ptr2, int n )
{
    int      i;
    cl_float *inptr = (cl_float *)ptr1;
    cl_float *outptr = (cl_float *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_fill_struct( void *ptr1, void *ptr2, int n )
{
    int         i;
    TestStruct  *inptr = (TestStruct *)ptr1;
    TestStruct  *outptr = (TestStruct *)ptr2;

    for (i=0; i<n; i++){
        if ( ( outptr[i].a != inptr[i].a ) || ( outptr[i].b != outptr[i].b ) )
            return -1;
    }

    return 0;
}



int test_buffer_fill( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type,
                     int loops, void *inptr[5], void *hostptr[5], void *pattern[5], size_t offset_elements, size_t fill_elements,
                     const char *kernelCode[], const char *kernelName[], int (*fn)(void *,void *,int) )
{
    void        *outptr[5];
    clProgramWrapper program[5];
    clKernelWrapper kernel[5];
    size_t      ptrSizes[5];
    size_t      global_work_size[3];
    int         err;
    int i;
    int         src_flag_id;
    int         total_errors = 0;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    loops = (loops < 5 ? loops : 5);
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
            clEventWrapper event[2];
            clMemWrapper buffers[2];
            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],
                                            ptrSizes[i] * num_elements,
                                            hostptr[i], &err);
            else
                buffers[0] =
                    clCreateBuffer(context, flag_set[src_flag_id],
                                   ptrSizes[i] * num_elements, NULL, &err);
            if (!buffers[0] || err)
            {
                print_error(err, "clCreateBuffer failed\n" );
                return -1;
            }
            // Initialize source buffer with 0, since the validation code expects 0(s) outside of the fill region.
            if (!((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))) {
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_FALSE, 0,
                                           ptrSizes[i] * num_elements,
                                           hostptr[i], 0, NULL, NULL);
                if ( err != CL_SUCCESS ){
                    print_error(err, "clEnqueueWriteBuffer failed\n" );
                    return -1;
                }
            }

            outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
            memset(outptr[i], 0, ptrSizes[i] * num_elements);
            buffers[1] =
                clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                               ptrSizes[i] * num_elements, outptr[i], &err);
            if (!buffers[1] || err)
            {
                print_error(err, "clCreateBuffer failed\n" );
                align_free( outptr[i] );
                return -1;
            }

            err = clEnqueueFillBuffer(
                queue, buffers[0], pattern[i], ptrSizes[i],
                ptrSizes[i] * offset_elements, ptrSizes[i] * fill_elements, 0,
                NULL, &(event[0]));

            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueFillBuffer failed" );
                align_free( outptr[i] );
                return -1;
            }

            err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem),
                                 (void *)&buffers[0]);
            err |= clSetKernelArg(kernel[i], 1, sizeof(cl_mem),
                                  (void *)&buffers[1]);
            if ( err != CL_SUCCESS ){
                print_error( err, "clSetKernelArg failed" );
                align_free( outptr[i] );
                return -1;
            }

            err = clWaitForEvents(  1, &(event[0]) );
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
                align_free( outptr[i] );
                return -1;
            }

            err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS){
                print_error( err, "clEnqueueNDRangeKernel failed" );
                return -1;
            }

            err = clEnqueueReadBuffer(queue, buffers[1], false, 0,
                                      ptrSizes[i] * num_elements, outptr[i], 0,
                                      NULL, &(event[1]));
            if (err != CL_SUCCESS){
                print_error( err, "clEnqueueReadBuffer failed" );
                return -1;
            }

            err = clWaitForEvents( 1, &(event[1]) );
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
            }

            if ( fn( inptr[i], outptr[i], (int)(ptrSizes[i] * (size_t)num_elements / ptrSizes[0]) ) ){
                log_error(" %s%d test failed. (cl_mem_flags: %s)\n", type,
                          1 << i, flag_set_names[src_flag_id]);
                total_errors++;
            }
            else{
                log_info(" %s%d test passed (cl_mem_flags: %s)\n", type, 1 << i,
                         flag_set_names[src_flag_id]);
            }

            // cleanup
            align_free( outptr[i] );
        }
    } // src cl_mem_flag

    return total_errors;

}   // end test_buffer_fill()


int test_buffer_fill_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    TestStruct pattern;
    size_t      ptrSize = sizeof( TestStruct );
    size_t      global_work_size[3];
    int         n, err;
    size_t      j, offset_elements, fill_elements;
    int         src_flag_id;
    int         total_errors = 0;
    MTdata      d = init_genrand( gRandomSeed );

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;


    for (src_flag_id = 0; src_flag_id < NUM_FLAGS; src_flag_id++)
    {
        clProgramWrapper program;
        clKernelWrapper kernel;
        log_info("Testing with cl_mem_flags: %s\n",
                 flag_set_names[src_flag_id]);

        err = create_single_kernel_helper(context, &program, &kernel, 1,
                                          &struct_kernel_code,
                                          "read_fill_struct");
        if (err)
        {
            log_error(" Error creating program for struct\n");
            free_mtdata(d);
            return -1;
        }

        // Test with random offsets and fill sizes
        for (n = 0; n < 8; n++)
        {
            clEventWrapper event[2];
            clMemWrapper buffers[2];
            void *outptr;
            TestStruct *inptr;
            TestStruct *hostptr;

            offset_elements =
                (size_t)get_random_float(0.f, (float)(num_elements - 8), d);
            fill_elements = (size_t)get_random_float(
                8.f, (float)(num_elements - offset_elements), d);
            log_info("Testing random fill from offset %d for %d elements: \n",
                     (int)offset_elements, (int)fill_elements);

            pattern.a = (cl_int)genrand_int32(d);
            pattern.b = (cl_float)get_random_float(-FLT_MAX, FLT_MAX, d);

            inptr = (TestStruct *)align_malloc(ptrSize * num_elements,
                                               min_alignment);
            for (j = 0; j < offset_elements; j++)
            {
                inptr[j].a = 0;
                inptr[j].b = 0;
            }
            for (j = offset_elements; j < offset_elements + fill_elements; j++)
            {
                inptr[j].a = pattern.a;
                inptr[j].b = pattern.b;
            }
            for (j = offset_elements + fill_elements; j < (size_t)num_elements;
                 j++)
            {
                inptr[j].a = 0;
                inptr[j].b = 0;
            }

            hostptr = (TestStruct *)align_malloc(ptrSize * num_elements,
                                                 min_alignment);
            memset(hostptr, 0, ptrSize * num_elements);

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSize * num_elements, hostptr, &err);
            else
                buffers[0] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSize * num_elements, NULL, &err);
            if ( err ){
                print_error(err, " clCreateBuffer failed\n" );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }
            if (!((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))) {
                err = clEnqueueWriteBuffer(queue, buffers[0], CL_FALSE, 0, ptrSize * num_elements, hostptr, 0, NULL, NULL);
                if ( err != CL_SUCCESS ){
                    print_error(err, " clEnqueueWriteBuffer failed\n" );
                    align_free( (void *)inptr );
                    align_free( (void *)hostptr );
                    free_mtdata(d);
                    return -1;
                }
            }
            outptr = align_malloc( ptrSize * num_elements, min_alignment);
            memset(outptr, 0, ptrSize * num_elements);
            buffers[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  ptrSize * num_elements, outptr, &err);
            if ( ! buffers[1] || err){
                print_error(err, " clCreateBuffer failed\n" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clEnqueueFillBuffer(
                queue, buffers[0], &pattern, ptrSize, ptrSize * offset_elements,
                ptrSize * fill_elements, 0, NULL, &(event[0]));
            /* uncomment for test debugging
             err = clEnqueueWriteBuffer(queue, buffers[0], CL_FALSE, 0, ptrSize * num_elements, inptr, 0, NULL, &(event[0]));
             */
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueFillBuffer failed" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clSetKernelArg( kernel, 0, sizeof( cl_mem ), (void *)&buffers[0] );
            err |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), (void *)&buffers[1] );
            if ( err != CL_SUCCESS ){
                print_error( err, " clSetKernelArg failed" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clWaitForEvents(  1, &(event[0]) );
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueNDRangeKernel failed" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clEnqueueReadBuffer( queue, buffers[1], CL_FALSE, 0, ptrSize * num_elements, outptr, 0, NULL, &(event[1]) );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueReadBuffer failed" );
                align_free( outptr );
                align_free( (void *)inptr );
                align_free( (void *)hostptr );
                free_mtdata(d);
                return -1;
            }

            err = clWaitForEvents( 1, &(event[1]) );
            if ( err != CL_SUCCESS ){
                print_error( err, "clWaitForEvents() failed" );
            }

            if ( verify_fill_struct( inptr, outptr, num_elements) ) {
                log_error( " buffer_FILL async struct test failed\n" );
                total_errors++;
            }
            else{
                log_info( " buffer_FILL async struct test passed\n" );
            }
            // cleanup
            align_free( outptr );
            align_free((void *)inptr);
            align_free((void *)hostptr);
        } // src cl_mem_flag
    }

    free_mtdata(d);

    return total_errors;

}   // end test_buffer_fill_struct()


int test_buffer_fill_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_int  *inptr[5];
    cl_int  *hostptr[5];
    cl_int  *pattern[5];
    size_t  ptrSizes[5];
    int     n, i, err=0;
    size_t  j, offset_elements, fill_elements;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_int;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_int *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_INT;

            inptr[i] = (cl_int *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_INT;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_int *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_int_kernel_code, int_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_int_fill()


int test_buffer_fill_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uint *inptr[5];
    cl_uint *hostptr[5];
    cl_uint *pattern[5];
    size_t  ptrSizes[5];
    int     n, i, err=0;
    size_t  j, offset_elements, fill_elements;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_uint;

    ptrSizes[0] = sizeof(cl_uint);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_uint *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_UINT;

            inptr[i] = (cl_uint *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_UINT;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_uint *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_uint_kernel_code, uint_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_uint_fill()


int test_buffer_fill_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_short *inptr[5];
    cl_short *hostptr[5];
    cl_short *pattern[5];
    size_t   ptrSizes[5];
    int      n, i, err=0;
    size_t   j, offset_elements, fill_elements;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_short;

    ptrSizes[0] = sizeof(cl_short);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_short *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_SHORT;

            inptr[i] = (cl_short *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_SHORT;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_short *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_short_kernel_code, short_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_short_fill()


int test_buffer_fill_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ushort *inptr[5];
    cl_ushort *hostptr[5];
    cl_ushort *pattern[5];
    size_t    ptrSizes[5];
    int       n, i, err=0;
    size_t    j, offset_elements, fill_elements;
    MTdata    d = init_genrand( gRandomSeed );
    int       (*foo)(void *,void *,int);

    size_t    min_alignment = get_min_alignment(context);

    foo = verify_fill_ushort;

    ptrSizes[0] = sizeof(cl_ushort);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_ushort *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_USHORT;

            inptr[i] = (cl_ushort *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_USHORT;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_ushort *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_ushort_kernel_code, ushort_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_ushort_fill()


int test_buffer_fill_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_char *inptr[5];
    cl_char *hostptr[5];
    cl_char *pattern[5];
    size_t  ptrSizes[5];
    int     n, i, err=0;
    size_t  j, offset_elements, fill_elements;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_char;

    ptrSizes[0] = sizeof(cl_char);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_char *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_CHAR;

            inptr[i] = (cl_char *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_CHAR;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_char *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_char_kernel_code, char_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_char_fill()


int test_buffer_fill_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uchar *inptr[5];
    cl_uchar *hostptr[5];
    cl_uchar *pattern[5];
    size_t   ptrSizes[5];
    int      n, i, err=0;
    size_t   j, offset_elements, fill_elements;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_uchar;

    ptrSizes[0] = sizeof(cl_uchar);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_uchar *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_CHAR;

            inptr[i] = (cl_uchar *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_CHAR;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_uchar *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_uchar_kernel_code, uchar_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_uchar_fill()


int test_buffer_fill_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_long *inptr[5];
    cl_long *hostptr[5];
    cl_long *pattern[5];
    size_t  ptrSizes[5];
    int     n, i, err=0;
    size_t  j, offset_elements, fill_elements;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_long;

    ptrSizes[0] = sizeof(cl_long);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_long *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_LONG;

            inptr[i] = (cl_long *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_LONG;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_long *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"long",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_long_kernel_code, long_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_long_fill()


int test_buffer_fill_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ulong *inptr[5];
    cl_ulong *hostptr[5];
    cl_ulong *pattern[5];
    size_t   ptrSizes[5];
    int      n, i, err=0;
    size_t   j, offset_elements, fill_elements;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t   min_alignment = get_min_alignment(context);

    foo = verify_fill_ulong;

    ptrSizes[0] = sizeof(cl_ulong);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    if (! gHasLong )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_ulong *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_ULONG;

            inptr[i] = (cl_ulong *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_ULONG;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_ulong *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_ulong_kernel_code, ulong_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_ulong_fill()


int test_buffer_fill_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_float *inptr[5];
    cl_float *hostptr[5];
    cl_float *pattern[5];
    size_t   ptrSizes[5];
    int      n, i, err=0;
    size_t   j, offset_elements, fill_elements;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_fill_float;

    ptrSizes[0] = sizeof(cl_float);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    // Test with random offsets and fill sizes
    for ( n = 0; n < 8; n++ ){
        offset_elements = (size_t)get_random_float( 0.f, (float)(num_elements - 8), d );
        fill_elements = (size_t)get_random_float( 8.f, (float)(num_elements - offset_elements), d );
        log_info( "Testing random fill from offset %d for %d elements: \n", (int)offset_elements, (int)fill_elements );

        for ( i = 0; i < 5; i++ ){
            pattern[i] = (cl_float *)malloc(ptrSizes[i]);
            for ( j = 0; j < ptrSizes[i] / ptrSizes[0]; j++ )
                pattern[i][j] = TEST_PRIME_FLOAT;

            inptr[i] = (cl_float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            for ( j = 0; j < ptrSizes[i] * offset_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;
            for ( j = ptrSizes[i] * offset_elements / ptrSizes[0]; j < ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j++ )
                inptr[i][j] = TEST_PRIME_FLOAT;
            for ( j = ptrSizes[i] * (offset_elements + fill_elements) / ptrSizes[0]; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
                inptr[i][j] = 0;

            hostptr[i] = (cl_float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);
            memset(hostptr[i], 0, ptrSizes[i] * num_elements);
        }

        if (test_buffer_fill( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float",
                             5, (void**)inptr, (void**)hostptr, (void**)pattern,
                             offset_elements, fill_elements,
                             buffer_fill_float_kernel_code, float_kernel_name, foo ))
            err++;

        for ( i = 0; i < 5; i++ ){
            free( (void *)pattern[i] );
            align_free( (void *)inptr[i] );
            align_free( (void *)hostptr[i] );
        }

    }

    free_mtdata(d);

    return err;

}   // end test_buffer_float_fill()
