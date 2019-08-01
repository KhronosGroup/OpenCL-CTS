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

#define USE_LOCAL_WORK_GROUP    1

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef TestStruct
typedef struct{
    int     a;
    float   b;
} TestStruct;
#endif

// If this is set to 1 the writes are done via map/unmap
static int gTestMap = 0;

const char *buffer_write_int_kernel_code[] = {
    "__kernel void test_buffer_write_int(__global int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_int2(__global int2 *src, __global int2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_int4(__global int4 *src, __global int4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_int8(__global int8 *src, __global int8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_int16(__global int16 *src, __global int16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *int_kernel_name[] = { "test_buffer_write_int", "test_buffer_write_int2", "test_buffer_write_int4", "test_buffer_write_int8", "test_buffer_write_int16" };


const char *buffer_write_uint_kernel_code[] = {
    "__kernel void test_buffer_write_uint(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uint2(__global uint2 *src, __global uint2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uint4(__global uint4 *src, __global uint4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uint8(__global uint8 *src, __global uint8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uint16(__global uint16 *src, __global uint16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uint_kernel_name[] = { "test_buffer_write_uint", "test_buffer_write_uint2", "test_buffer_write_uint4", "test_buffer_write_uint8", "test_buffer_write_uint16" };


const char *buffer_write_ushort_kernel_code[] = {
    "__kernel void test_buffer_write_ushort(__global ushort *src, __global ushort *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ushort2(__global ushort2 *src, __global ushort2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ushort4(__global ushort4 *src, __global ushort4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ushort8(__global ushort8 *src, __global ushort8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ushort16(__global ushort16 *src, __global ushort16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ushort_kernel_name[] = { "test_buffer_write_ushort", "test_buffer_write_ushort2", "test_buffer_write_ushort4", "test_buffer_write_ushort8", "test_buffer_write_ushort16" };



const char *buffer_write_short_kernel_code[] = {
    "__kernel void test_buffer_write_short(__global short *src, __global short *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_short2(__global short2 *src, __global short2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_short4(__global short4 *src, __global short4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_short8(__global short8 *src, __global short8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_short16(__global short16 *src, __global short16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *short_kernel_name[] = { "test_buffer_write_short", "test_buffer_write_short2", "test_buffer_write_short4", "test_buffer_write_short8", "test_buffer_write_short16" };


const char *buffer_write_char_kernel_code[] = {
    "__kernel void test_buffer_write_char(__global char *src, __global char *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_char2(__global char2 *src, __global char2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_char4(__global char4 *src, __global char4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_char8(__global char8 *src, __global char8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_char16(__global char16 *src, __global char16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *char_kernel_name[] = { "test_buffer_write_char", "test_buffer_write_char2", "test_buffer_write_char4", "test_buffer_write_char8", "test_buffer_write_char16" };


const char *buffer_write_uchar_kernel_code[] = {
    "__kernel void test_buffer_write_uchar(__global uchar *src, __global uchar *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uchar2(__global uchar2 *src, __global uchar2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uchar4(__global uchar4 *src, __global uchar4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uchar8(__global uchar8 *src, __global uchar8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_uchar16(__global uchar16 *src, __global uchar16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uchar_kernel_name[] = { "test_buffer_write_uchar", "test_buffer_write_uchar2", "test_buffer_write_uchar4", "test_buffer_write_uchar8", "test_buffer_write_uchar16" };


const char *buffer_write_float_kernel_code[] = {
    "__kernel void test_buffer_write_float(__global float *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_float2(__global float2 *src, __global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_float4(__global float4 *src, __global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_float8(__global float8 *src, __global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_float16(__global float16 *src, __global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *float_kernel_name[] = { "test_buffer_write_float", "test_buffer_write_float2", "test_buffer_write_float4", "test_buffer_write_float8", "test_buffer_write_float16" };


const char *buffer_write_half_kernel_code[] = {
    "__kernel void test_buffer_write_half(__global half *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = vload_half( tid * 2, src );\n"
    "}\n",

    "__kernel void test_buffer_write_half2(__global half2 *src, __global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = vload_half2( tid * 2, src );\n"
    "}\n",

    "__kernel void test_buffer_write_half4(__global half4 *src, __global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = vload_half4( tid * 2, src );\n"
    "}\n",

    "__kernel void test_buffer_write_half8(__global half8 *src, __global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = vload_half8( tid * 2, src );\n"
    "}\n",

    "__kernel void test_buffer_write_half16(__global half16 *src, __global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = vload_half16( tid * 2, src );\n"
    "}\n" };

static const char *half_kernel_name[] = { "test_buffer_write_half", "test_buffer_write_half2", "test_buffer_write_half4", "test_buffer_write_half8", "test_buffer_write_half16" };


const char *buffer_write_long_kernel_code[] = {
    "__kernel void test_buffer_write_long(__global long *src, __global long *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_long2(__global long2 *src, __global long2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_long4(__global long4 *src, __global long4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_long8(__global long8 *src, __global long8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_long16(__global long16 *src, __global long16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *long_kernel_name[] = { "test_buffer_write_long", "test_buffer_write_long2", "test_buffer_write_long4", "test_buffer_write_long8", "test_buffer_write_long16" };


const char *buffer_write_ulong_kernel_code[] = {
    "__kernel void test_buffer_write_ulong(__global ulong *src, __global ulong *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ulong2(__global ulong2 *src, __global ulong2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ulong4(__global ulong4 *src, __global ulong4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ulong8(__global ulong8 *src, __global ulong8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_buffer_write_ulong16(__global ulong16 *src, __global ulong16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ulong_kernel_name[] = { "test_buffer_write_ulong", "test_buffer_write_ulong2", "test_buffer_write_ulong4", "test_buffer_write_ulong8", "test_buffer_write_ulong16" };


static const char *struct_kernel_code =
"typedef struct{\n"
"int    a;\n"
"float    b;\n"
"} TestStruct;\n"
"__kernel void read_write_struct(__global TestStruct *src, __global TestStruct *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid].a = src[tid].a;\n"
"     dst[tid].b = src[tid].b;\n"
"}\n";



static int verify_write_int( void *ptr1, void *ptr2, int n )
{
    int     i;
    int     *inptr = (int *)ptr1;
    int     *outptr = (int *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_uint( void *ptr1, void *ptr2, int n )
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


static int verify_write_short( void *ptr1, void *ptr2, int n )
{
    int     i;
    short   *inptr = (short *)ptr1;
    short   *outptr = (short *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_ushort( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_ushort   *inptr = (cl_ushort *)ptr1;
    cl_ushort   *outptr = (cl_ushort *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_char( void *ptr1, void *ptr2, int n )
{
    int     i;
    char    *inptr = (char *)ptr1;
    char    *outptr = (char *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_uchar( void *ptr1, void *ptr2, int n )
{
    int     i;
    uchar   *inptr = (uchar *)ptr1;
    uchar   *outptr = (uchar *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_float( void *ptr1, void *ptr2, int n )
{
    int     i;
    float   *inptr = (float *)ptr1;
    float   *outptr = (float *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_half( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_ushort   *inptr = (cl_ushort *)ptr1;
    cl_ushort   *outptr = (cl_ushort *)ptr2;

    for ( i = 0; i < n; i++ ){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_long( void *ptr1, void *ptr2, int n )
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


static int verify_write_ulong( void *ptr1, void *ptr2, int n )
{
    int     i;
    cl_ulong    *inptr = (cl_ulong *)ptr1;
    cl_ulong    *outptr = (cl_ulong *)ptr2;

    for (i=0; i<n; i++){
        if ( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_struct( void *ptr1, void *ptr2, int n )
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


int test_buffer_write( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                       void *inptr[5], const char *kernelCode[], const char *kernelName[], int (*fn)(void *,void *,int), MTdata d )
{
    cl_mem      buffers[10];
    void        *outptr[5];
    cl_program  program[5];
    cl_kernel   kernel[5];
    size_t      ptrSizes[5];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i, ii;
    int         src_flag_id, dst_flag_id;
    int         total_errors = 0;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        for (dst_flag_id=0; dst_flag_id < NUM_FLAGS; dst_flag_id++) {
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            loops = ( loops < 5 ? loops : 5 );
            for ( i = 0; i < loops; i++ ){
                ii = i << 1;
                if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, inptr[i], &err);
                else
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, NULL, &err);

                if ( ! buffers[ii] || err){
                    align_free( outptr[i] );
                    print_error(err, " clCreateBuffer failed\n" );
                    return -1;
                }
                if ( ! strcmp( type, "half" ) ){
                    outptr[i] = align_malloc( ptrSizes[i] * (num_elements * 2 ), min_alignment);
                    if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                        buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * 2 * num_elements, outptr[i], &err);
                    else
                        buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * 2 * num_elements, NULL, &err);
                }
                else{
                    outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
                    if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                        buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, outptr[i], &err);
                    else
                        buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, NULL, &err);
                }
                if ( err ){
                    clReleaseMemObject(buffers[ii]);
                    align_free( outptr[i] );
                    print_error(err, " clCreateBuffer failed\n" );
                    return -1;
                }

                if (gTestMap) {
                    void *dataPtr;
                    dataPtr = clEnqueueMapBuffer(queue, buffers[ii], CL_TRUE, CL_MAP_WRITE, 0, ptrSizes[i]*num_elements, 0, NULL, NULL, &err);
                    if (err) {
                        print_error(err, "clEnqueueMapBuffer failed");
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        return -1;
                    }

                    memcpy(dataPtr, inptr[i], ptrSizes[i]*num_elements);

                    err = clEnqueueUnmapMemObject(queue, buffers[ii], dataPtr, 0, NULL, NULL);
                    if (err) {
                        print_error(err, "clEnqueueUnmapMemObject failed");
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        return -1;
                    }
                }
                else if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR)) {
                    err = clEnqueueWriteBuffer(queue, buffers[ii], CL_TRUE, 0, ptrSizes[i]*num_elements, inptr[i], 0, NULL, NULL);
                    if ( err != CL_SUCCESS ){
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        print_error( err, " clWriteBuffer failed" );
                        return -1;
                    }
                }

                err = create_single_kernel_helper( context, &program[i], &kernel[i], 1, &kernelCode[i], kernelName[i] );
                if ( err ){
                    clReleaseMemObject(buffers[ii]);
                    clReleaseMemObject(buffers[ii+1]);
                    align_free( outptr[i] );
                    log_error( " Error creating program for %s\n", type );
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = get_max_common_work_group_size( context, kernel[i], global_work_size[0], &local_work_size[0] );
                test_error( err, "Unable to get work group size to use" );
#endif

                err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&buffers[ii] );
                err |= clSetKernelArg( kernel[i], 1, sizeof( cl_mem ), (void *)&buffers[ii+1] );
                if ( err != CL_SUCCESS ){
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    print_error( err, " clSetKernelArg failed" );
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
                if ( err != CL_SUCCESS ){
                    print_error( err, " clEnqueueNDRangeKernel failed" );
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    return -1;
                }

                if ( ! strcmp( type, "half" ) ){
                    err = clEnqueueReadBuffer( queue, buffers[ii+1], true, 0, ptrSizes[i]*num_elements, outptr[i], 0, NULL, NULL );
                }
                else{
                    err = clEnqueueReadBuffer( queue, buffers[ii+1], true, 0, ptrSizes[i]*num_elements, outptr[i], 0, NULL, NULL );
                }
                if ( err != CL_SUCCESS ){
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    print_error( err, " clEnqueueReadBuffer failed" );
                    return -1;
                }

                if ( fn( inptr[i], outptr[i], (int)(ptrSizes[i] * (size_t)num_elements / ptrSizes[0]) ) ){
                    log_error( " %s%d test failed\n", type, 1<<i );
                    total_errors++;
                }
                else{
                    log_info( " %s%d test passed\n", type, 1<<i );
                }
                // cleanup
                clReleaseMemObject( buffers[ii] );
                clReleaseMemObject( buffers[ii+1] );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                align_free( outptr[i] );
            }
        } // dst cl_mem_flag
    } // src cl_mem_flag

    return total_errors;

}   // end test_buffer_write()




int test_buffer_write_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem      buffers[10];
    void        *outptr[5];
    TestStruct  *inptr[5];
    cl_program  program[5];
    cl_kernel   kernel[5];
    size_t      ptrSizes[5];
    size_t      size = sizeof( TestStruct );
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i, ii;
    cl_uint     j;
    int         loops = 1;      // no vector for structs
    int         src_flag_id, dst_flag_id;
    int         total_errors = 0;
    MTdata      d = init_genrand( gRandomSeed );

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        for (dst_flag_id=0; dst_flag_id < NUM_FLAGS; dst_flag_id++) {
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            loops = ( loops < 5 ? loops : 5 );
            for ( i = 0; i < loops; i++ ){

                inptr[i] = (TestStruct *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

                for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ ){
                    inptr[i][j].a = (int)genrand_int32(d);
                    inptr[i][j].b = get_random_float( -FLT_MAX, FLT_MAX, d );
                }

                ii = i << 1;
                if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, inptr[i], &err);
                else
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, NULL, &err);
                if ( err ){
                    align_free( outptr[i] );
                    print_error(err, " clCreateBuffer failed\n" );
                    free_mtdata(d);
                    return -1;
                }
                outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
                if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                    buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, outptr[i], &err);
                else
                    buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, NULL, &err);
                if ( ! buffers[ii+1] || err){
                    clReleaseMemObject(buffers[ii]);
                    align_free( outptr[i] );
                    print_error(err, " clCreateBuffer failed\n" );
                    free_mtdata(d);
                    return -1;
                }

                if (gTestMap) {
                    void *dataPtr;
                    dataPtr = clEnqueueMapBuffer(queue, buffers[ii], CL_TRUE, CL_MAP_WRITE, 0, ptrSizes[i]*num_elements, 0, NULL, NULL, &err);
                    if (err) {
                        print_error(err, "clEnqueueMapBuffer failed");
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        free_mtdata(d);
                        return -1;
                    }

                    memcpy(dataPtr, inptr[i], ptrSizes[i]*num_elements);

                    err = clEnqueueUnmapMemObject(queue, buffers[ii], dataPtr, 0, NULL, NULL);
                    if (err) {
                        print_error(err, "clEnqueueUnmapMemObject failed");
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        free_mtdata(d);
                        return -1;
                    }
                }
                else if (!(flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) && !(flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR)) {
                    err = clEnqueueWriteBuffer(queue, buffers[ii], CL_TRUE, 0, ptrSizes[i]*num_elements, inptr[i], 0, NULL, NULL);
                    if ( err != CL_SUCCESS ){
                        clReleaseMemObject(buffers[ii]);
                        clReleaseMemObject(buffers[ii+1]);
                        align_free( outptr[i] );
                        print_error( err, " clWriteBuffer failed" );
                        free_mtdata(d);
                        return -1;
                    }
                }

                err = create_single_kernel_helper( context, &program[i], &kernel[i], 1, &struct_kernel_code, "read_write_struct" );
                if ( err ){
                    clReleaseMemObject(buffers[ii]);
                    clReleaseMemObject(buffers[ii+1]);
                    align_free( outptr[i] );
                    log_error( " Error creating program for struct\n" );
                    free_mtdata(d);
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = get_max_common_work_group_size( context, kernel[i], global_work_size[0], &local_work_size[0] );
                test_error( err, "Unable to get work group size to use" );
#endif

                err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&buffers[ii] );
                err |= clSetKernelArg( kernel[i], 1, sizeof( cl_mem ), (void *)&buffers[ii+1] );
                if ( err != CL_SUCCESS ){
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    print_error( err, " clSetKernelArg failed" );
                    free_mtdata(d);
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
                if ( err != CL_SUCCESS ){
                    print_error( err, " clEnqueueNDRangeKernel failed" );
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    free_mtdata(d);
                    return -1;
                }

                err = clEnqueueReadBuffer( queue, buffers[ii+1], true, 0, ptrSizes[i]*num_elements, outptr[i], 0, NULL, NULL );
                if ( err != CL_SUCCESS ){
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    align_free( outptr[i] );
                    print_error( err, " clEnqueueReadBuffer failed" );
                    free_mtdata(d);
                    return -1;
                }

                if ( verify_write_struct( inptr[i], outptr[i], (int)(ptrSizes[i] * (size_t)num_elements / ptrSizes[0]) ) ){
                    log_error( " buffer_WRITE struct%d test failed\n", 1<<i );
                    total_errors++;
                }
                else{
                    log_info( " buffer_WRITE struct%d test passed\n", 1<<i );
                }
                // cleanup
                clReleaseMemObject( buffers[ii] );
                clReleaseMemObject( buffers[ii+1] );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                align_free( outptr[i] );
                align_free( (void *)inptr[i] );
            }
        } // dst cl_mem_flag
    } // src cl_mem_flag

    free_mtdata(d);

    return total_errors;

}   // end test_buffer_struct_write()


int test_buffer_write_array_async( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                                   void *inptr[5], const char *kernelCode[], const char *kernelName[], int (*fn)(void *,void *,int) )
{
    cl_mem      buffers[10];
    void        *outptr[5];
    cl_program  program[5];
    cl_kernel   kernel[5];
    cl_event    event[2];
    size_t      ptrSizes[5];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i, ii;
    int         src_flag_id, dst_flag_id;
    int         total_errors = 0;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        for (dst_flag_id=0; dst_flag_id < NUM_FLAGS; dst_flag_id++) {
            log_info("Testing with cl_mem_flags src: %s dst: %s\n", flag_set_names[src_flag_id], flag_set_names[dst_flag_id]);

            loops = ( loops < 5 ? loops : 5 );
            for ( i = 0; i < loops; i++ ){
                ii = i << 1;
                if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, inptr[i], &err);
                else
                    buffers[ii] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, NULL, &err);
                if ( !buffers[ii] || err){
                    print_error(err, "clCreateBuffer failed\n" );
                    return -1;
                }

                outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
                if ((flag_set[dst_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[dst_flag_id] & CL_MEM_COPY_HOST_PTR))
                    buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, outptr[i], &err);
                else
                    buffers[ii+1] = clCreateBuffer(context, flag_set[dst_flag_id],  ptrSizes[i] * num_elements, NULL, &err);
                if ( !buffers[ii+1] || err){
                    print_error(err, "clCreateBuffer failed\n" );
                    return -1;
                }

                err = clEnqueueWriteBuffer(queue, buffers[ii], CL_FALSE, 0, ptrSizes[i]*num_elements, inptr[i], 0, NULL, &(event[0]));
                if ( err != CL_SUCCESS ){
                    print_error( err, "clEnqueueWriteBuffer failed" );
                    return -1;
                }

                err = create_single_kernel_helper( context, &program[i], &kernel[i], 1, &kernelCode[i], kernelName[i] );
                if ( err ){
                    log_error( " Error creating program for %s\n", type );
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    align_free( outptr[i] );
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = get_max_common_work_group_size( context, kernel[i], global_work_size[0], &local_work_size[0] );
                test_error( err, "Unable to get work group size to use" );
#endif

                err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&buffers[ii] );
                err |= clSetKernelArg( kernel[i], 1, sizeof( cl_mem ), (void *)&buffers[ii+1] );
                if ( err != CL_SUCCESS ){
                    print_error( err, "clSetKernelArg failed" );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    align_free( outptr[i] );
                    return -1;
                }

                err = clWaitForEvents(  1, &(event[0]) );
                if ( err != CL_SUCCESS ){
                    print_error( err, "clWaitForEvents() failed" );
                    clReleaseKernel( kernel[i] );
                    clReleaseProgram( program[i] );
                    clReleaseMemObject( buffers[ii] );
                    clReleaseMemObject( buffers[ii+1] );
                    align_free( outptr[i] );
                    return -1;
                }

#ifdef USE_LOCAL_WORK_GROUP
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
                err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
                if (err != CL_SUCCESS){
                    print_error( err, "clEnqueueNDRangeKernel failed" );
                    return -1;
                }

                err = clEnqueueReadBuffer( queue, buffers[ii+1], false, 0, ptrSizes[i]*num_elements, outptr[i], 0, NULL, &(event[1]) );
                if (err != CL_SUCCESS){
                    print_error( err, "clEnqueueReadBuffer failed" );
                    return -1;
                }

                err = clWaitForEvents( 1, &(event[1]) );
                if ( err != CL_SUCCESS ){
                    print_error( err, "clWaitForEvents() failed" );
                }

                if ( fn( inptr[i], outptr[i], (int)(ptrSizes[i] * (size_t)num_elements / ptrSizes[0]) ) ){
                    log_error( " %s%d test failed\n", type, 1<<i );
                    total_errors++;
                }
                else{
                    log_info( " %s%d test passed\n", type, 1<<i );
                }

                // cleanup
                clReleaseEvent( event[0] );
                clReleaseEvent( event[1] );
                clReleaseMemObject( buffers[ii] );
                clReleaseMemObject( buffers[ii+1] );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                align_free( outptr[i] );
            }
        } // dst cl_mem_flag
    } // src cl_mem_flag

    return total_errors;

}   // end test_buffer_write_array_async()


int test_buffer_write_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_int;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (int *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (int)genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                             buffer_write_int_kernel_code, int_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}   // end test_buffer_int_write()


int test_buffer_write_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uint *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_uint;

    ptrSizes[0] = sizeof(cl_uint);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_uint *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                             buffer_write_uint_kernel_code, uint_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_uint_write()


int test_buffer_write_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    short   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_short;

    ptrSizes[0] = sizeof(cl_short);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_short *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_short)genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                             buffer_write_short_kernel_code, short_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_short_write()


int test_buffer_write_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ushort *inptr[5];
    size_t    ptrSizes[5];
    int       i, err;
    cl_uint   j;
    MTdata    d = init_genrand( gRandomSeed );
    int       (*foo)(void *,void *,int);

    size_t    min_alignment = get_min_alignment(context);

    foo = verify_write_ushort;

    ptrSizes[0] = sizeof(cl_ushort);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ushort *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ushort)genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                             buffer_write_ushort_kernel_code, ushort_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_ushort_write()


int test_buffer_write_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    char    *inptr[5];
    size_t  ptrSizes[5];
    int     i,  err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_char;

    ptrSizes[0] = sizeof(cl_char);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (char *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (char)genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                             buffer_write_char_kernel_code, char_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_char_write()


int test_buffer_write_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    uchar   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_uchar;

    ptrSizes[0] = sizeof(cl_uchar);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (uchar *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (uchar)genrand_int32(d);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                             buffer_write_uchar_kernel_code, uchar_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_uchar_write()


int test_buffer_write_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    float   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_float;

    ptrSizes[0] = sizeof(cl_float);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = get_random_float( -FLT_MAX, FLT_MAX, d );
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                             buffer_write_float_kernel_code, float_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_float_write()


int test_buffer_write_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    float   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_half;

    ptrSizes[0] = sizeof( cl_float ) / 2;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ( ptrSizes[0] * 2 ); j++ )
            inptr[i][j] = get_random_float( -FLT_MAX, FLT_MAX, d );
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_float ) / 2, (char*)"half", 5, (void**)inptr,
                             buffer_write_half_kernel_code, half_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_half_write()


int test_buffer_write_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_long *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_long;

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

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_long *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_long) genrand_int32(d) ^ ((cl_long) genrand_int32(d) << 32);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"cl_long", 5, (void**)inptr,
                             buffer_write_long_kernel_code, long_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_long_write()


int test_buffer_write_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ulong *inptr[5];
    size_t   ptrSizes[5];
    int      i, err;
    cl_uint  j;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t   min_alignment = get_min_alignment(context);

    foo = verify_write_ulong;

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

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ulong *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
    }

    err = test_buffer_write( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong long", 5, (void**)inptr,
                             buffer_write_ulong_kernel_code, ulong_kernel_name, foo, d );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);

    return err;

}   // end test_buffer_ulong_write()


int test_buffer_map_write_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_int(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_uint(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_long(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_ulong(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_short(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_ushort(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_char(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_uchar(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_float(deviceID, context, queue, num_elements);
}

int test_buffer_map_write_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    gTestMap = 1;
    return test_buffer_write_struct(deviceID, context, queue, num_elements);
}


int test_buffer_write_async_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_int;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (int *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (int)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                                         buffer_write_int_kernel_code, int_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_int_write_array_async()


int test_buffer_write_async_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uint *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_uint;

    ptrSizes[0] = sizeof(cl_uint);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_uint *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_uint)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                                         buffer_write_uint_kernel_code, uint_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_uint_write_array_async()


int test_buffer_write_async_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    short   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_short;

    ptrSizes[0] = sizeof(cl_short);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (short *)align_malloc(ptrSizes[i] * num_elements + min_alignment, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (short)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                                         buffer_write_short_kernel_code, short_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_short_write_array_async()


int test_buffer_write_async_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ushort *inptr[5];
    size_t    ptrSizes[5];
    int       i, err;
    cl_uint   j;
    MTdata    d = init_genrand( gRandomSeed );
    int       (*foo)(void *,void *,int);

    size_t    min_alignment = get_min_alignment(context);

    foo = verify_write_ushort;

    ptrSizes[0] = sizeof(cl_ushort);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ushort *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ushort)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                                         buffer_write_ushort_kernel_code, ushort_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_ushort_write_array_async()


int test_buffer_write_async_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    char    *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_char;

    ptrSizes[0] = sizeof(cl_char);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (char *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (char)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                                         buffer_write_char_kernel_code, char_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_char_write_array_async()


int test_buffer_write_async_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    uchar   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_uchar;

    ptrSizes[0] = sizeof(cl_uchar);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (uchar *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (uchar)genrand_int32(d);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                                         buffer_write_uchar_kernel_code, uchar_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );

    }

    free_mtdata(d);
    return err;

}   // end test_buffer_uchar_write_array_async()


int test_buffer_write_async_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    float   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_float;

    ptrSizes[0] = sizeof(cl_float);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = get_random_float( -FLT_MAX, FLT_MAX, d );
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                                         buffer_write_float_kernel_code, float_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_float_write_array_async()


int test_buffer_write_async_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_long *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    MTdata  d = init_genrand( gRandomSeed );
    int     (*foo)(void *,void *,int);

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_write_long;

    ptrSizes[0] = sizeof(cl_long);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    if (! gHasLong )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_long *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = ((cl_long) genrand_int32(d)) ^ ((cl_long) genrand_int32(d) << 32);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"cl_long", 5, (void**)inptr,
                                         buffer_write_long_kernel_code, long_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_long_write_array_async()


int test_buffer_write_async_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ulong *inptr[5];
    size_t   ptrSizes[5];
    int      i, err;
    cl_uint  j;
    MTdata   d = init_genrand( gRandomSeed );
    int      (*foo)(void *,void *,int);

    size_t   min_alignment = get_min_alignment(context);

    foo = verify_write_ulong;

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

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ulong *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
    }

    err = test_buffer_write_array_async( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong long", 5, (void**)inptr,
                                         buffer_write_ulong_kernel_code, ulong_kernel_name, foo );

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}   // end test_buffer_ulong_write_array_async()

