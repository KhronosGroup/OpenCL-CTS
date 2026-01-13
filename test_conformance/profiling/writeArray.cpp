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
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/conversions.h"

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef TestStruct
typedef struct{
    int        a;
    float    b;
} TestStruct;
#endif

const char *stream_write_int_kernel_code[] = {
    "__kernel void test_stream_write_int(__global int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_int2(__global int2 *src, __global int2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_int4(__global int4 *src, __global int4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_int8(__global int8 *src, __global int8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_int16(__global int16 *src, __global int16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *int_kernel_name[] = { "test_stream_write_int", "test_stream_write_int2", "test_stream_write_int4", "test_stream_write_int8", "test_stream_write_int16" };


const char *stream_write_uint_kernel_code[] = {
    "__kernel void test_stream_write_uint(__global uint *src, __global uint *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uint2(__global uint2 *src, __global uint2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uint4(__global uint4 *src, __global uint4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uint8(__global uint8 *src, __global uint8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uint16(__global uint16 *src, __global uint16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uint_kernel_name[] = { "test_stream_write_uint", "test_stream_write_uint2", "test_stream_write_uint4", "test_stream_write_uint8", "test_stream_write_uint16" };


const char *stream_write_ushort_kernel_code[] = {
    "__kernel void test_stream_write_ushort(__global ushort *src, __global ushort *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ushort2(__global ushort2 *src, __global ushort2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ushort4(__global ushort4 *src, __global ushort4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ushort8(__global ushort8 *src, __global ushort8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ushort16(__global ushort16 *src, __global ushort16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ushort_kernel_name[] = { "test_stream_write_ushort", "test_stream_write_ushort2", "test_stream_write_ushort4", "test_stream_write_ushort8", "test_stream_write_ushort16" };



const char *stream_write_short_kernel_code[] = {
    "__kernel void test_stream_write_short(__global short *src, __global short *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_short2(__global short2 *src, __global short2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_short4(__global short4 *src, __global short4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_short8(__global short8 *src, __global short8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_short16(__global short16 *src, __global short16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *short_kernel_name[] = { "test_stream_write_short", "test_stream_write_short2", "test_stream_write_short4", "test_stream_write_short8", "test_stream_write_short16" };


const char *stream_write_char_kernel_code[] = {
    "__kernel void test_stream_write_char(__global char *src, __global char *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_char2(__global char2 *src, __global char2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_char4(__global char4 *src, __global char4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_char8(__global char8 *src, __global char8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_char16(__global char16 *src, __global char16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *char_kernel_name[] = { "test_stream_write_char", "test_stream_write_char2", "test_stream_write_char4", "test_stream_write_char8", "test_stream_write_char16" };


const char *stream_write_uchar_kernel_code[] = {
    "__kernel void test_stream_write_uchar(__global uchar *src, __global uchar *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uchar2(__global uchar2 *src, __global uchar2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uchar4(__global uchar4 *src, __global uchar4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uchar8(__global uchar8 *src, __global uchar8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_uchar16(__global uchar16 *src, __global uchar16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *uchar_kernel_name[] = { "test_stream_write_uchar", "test_stream_write_uchar2", "test_stream_write_uchar4", "test_stream_write_uchar8", "test_stream_write_uchar16" };


const char *stream_write_float_kernel_code[] = {
    "__kernel void test_stream_write_float(__global float *src, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_float2(__global float2 *src, __global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_float4(__global float4 *src, __global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_float8(__global float8 *src, __global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_float16(__global float16 *src, __global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *float_kernel_name[] = { "test_stream_write_float", "test_stream_write_float2", "test_stream_write_float4", "test_stream_write_float8", "test_stream_write_float16" };


const char *stream_write_long_kernel_code[] = {
    "__kernel void test_stream_write_long(__global long *src, __global long *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_long2(__global long2 *src, __global long2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_long4(__global long4 *src, __global long4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_long8(__global long8 *src, __global long8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_long16(__global long16 *src, __global long16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *long_kernel_name[] = { "test_stream_write_long", "test_stream_write_long2", "test_stream_write_long4", "test_stream_write_long8", "test_stream_write_long16" };


const char *stream_write_ulong_kernel_code[] = {
    "__kernel void test_stream_write_ulong(__global ulong *src, __global ulong *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ulong2(__global ulong2 *src, __global ulong2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ulong4(__global ulong4 *src, __global ulong4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ulong8(__global ulong8 *src, __global ulong8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n",

    "__kernel void test_stream_write_ulong16(__global ulong16 *src, __global ulong16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src[tid];\n"
    "}\n" };

static const char *ulong_kernel_name[] = { "test_stream_write_ulong", "test_stream_write_ulong2", "test_stream_write_ulong4", "test_stream_write_ulong8", "test_stream_write_ulong16" };


static const char *stream_write_struct_kernel_code[] = {
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
    "}\n" };

static const char *struct_kernel_name[] = { "read_write_struct" };


static int verify_write_int( void *ptr1, void *ptr2, int n )
{
    int        i;
    int        *inptr = (int *)ptr1;
    int        *outptr = (int *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_uint( void *ptr1, void *ptr2, int n )
{
    int        i;
    cl_uint    *inptr = (cl_uint *)ptr1;
    cl_uint    *outptr = (cl_uint *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_short( void *ptr1, void *ptr2, int n )
{
    int        i;
    short    *inptr = (short *)ptr1;
    short    *outptr = (short *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_ushort( void *ptr1, void *ptr2, int n )
{
    int        i;
    cl_ushort    *inptr = (cl_ushort *)ptr1;
    cl_ushort    *outptr = (cl_ushort *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_char( void *ptr1, void *ptr2, int n )
{
    int        i;
    char    *inptr = (char *)ptr1;
    char    *outptr = (char *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_uchar( void *ptr1, void *ptr2, int n )
{
    int        i;
    uchar    *inptr = (uchar *)ptr1;
    uchar    *outptr = (uchar *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_float( void *ptr1, void *ptr2, int n )
{
    int        i;
    float    *inptr = (float *)ptr1;
    float    *outptr = (float *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_long( void *ptr1, void *ptr2, int n )
{
    int        i;
    cl_long    *inptr = (cl_long *)ptr1;
    cl_long    *outptr = (cl_long *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_ulong( void *ptr1, void *ptr2, int n )
{
    int        i;
    cl_ulong    *inptr = (cl_ulong *)ptr1;
    cl_ulong    *outptr = (cl_ulong *)ptr2;

    for (i=0; i<n; i++){
        if( outptr[i] != inptr[i] )
            return -1;
    }

    return 0;
}


static int verify_write_struct( void *ptr1, void *ptr2, int n )
{
    int            i;
    TestStruct    *inptr = (TestStruct *)ptr1;
    TestStruct    *outptr = (TestStruct *)ptr2;

    for (i=0; i<n; i++){
        if( ( outptr[i].a != inptr[i].a ) || ( outptr[i].b != outptr[i].b ) )
            return -1;
    }

    return 0;
}


int test_stream_write( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements, size_t size, const char *type, int loops,
                      void *inptr[5], const char *kernelCode[], const char *kernelName[], int (*fn)(void *,void *,int), MTdata d )
{
    cl_mem            streams[10];
    void            *outptr[5];
    cl_program        program[5];
    cl_kernel        kernel[5];
    cl_event        writeEvent;
    cl_ulong    queueStart, submitStart, writeStart, writeEnd;
    size_t            ptrSizes[5], outPtrSizes[5];
    size_t            threads[1];
    int                err, err_count = 0;
    int                i, ii;

    threads[0] = (size_t)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    loops = ( loops < 5 ? loops : 5 );

    for( i = 0; i < loops; i++ )
    {
        outPtrSizes[i] = ptrSizes[i];
    }

    for( i = 0; i < loops; i++ ){
        ii = i << 1;
        streams[ii] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     ptrSizes[i] * num_elements, NULL, &err);
        if( ! streams[ii] ){
            free( outptr[i] );
            log_error( " clCreateBuffer failed\n" );
            return -1;
        }
        if( ! strcmp( type, "half" ) ){
            outptr[i] = malloc( outPtrSizes[i] * num_elements * 2 );
            streams[ii + 1] =
                clCreateBuffer(context, CL_MEM_READ_WRITE,
                               outPtrSizes[i] * 2 * num_elements, NULL, &err);
        }
        else{
            outptr[i] = malloc( outPtrSizes[i] * num_elements );
            streams[ii + 1] =
                clCreateBuffer(context, CL_MEM_READ_WRITE,
                               outPtrSizes[i] * num_elements, NULL, &err);
        }
        if( ! streams[ii+1] ){
            clReleaseMemObject(streams[ii]);
            free( outptr[i] );
            log_error( " clCreateBuffer failed\n" );
            return -1;
        }

        err = clEnqueueWriteBuffer( queue, streams[ii], false, 0, ptrSizes[i]*num_elements, inptr[i], 0, NULL, &writeEvent );
        if( err != CL_SUCCESS ){
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            print_error( err, " clWriteArray failed" );
            return -1;
        }

        // This synchronization point is needed in order to assume the data is valid.
        // Getting profiling information is not a synchronization point.
        err = clWaitForEvents( 1, &writeEvent );
        if( err != CL_SUCCESS )
        {
            print_error( err, "Unable to wait for event completion" );
            clReleaseEvent(writeEvent);
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }

        // test profiling
        while( ( err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
              CL_PROFILING_INFO_NOT_AVAILABLE );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseEvent(writeEvent);
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }

        while( ( err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
              CL_PROFILING_INFO_NOT_AVAILABLE );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseEvent(writeEvent);
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }

        err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseEvent(writeEvent);
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }

        err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseEvent(writeEvent);
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }


        err = create_single_kernel_helper( context, &program[i], &kernel[i], 1, &kernelCode[i], kernelName[i] );
        if( err ){
            clReleaseEvent(writeEvent);
            clReleaseMemObject(streams[ii]);
            clReleaseMemObject(streams[ii+1]);
            free( outptr[i] );
            log_error( " Error creating program for %s\n", type );
            return -1;
        }

        err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&streams[ii] );
        err |= clSetKernelArg( kernel[i], 1, sizeof( cl_mem ), (void *)&streams[ii+1] );
        if (err != CL_SUCCESS){
            clReleaseEvent(writeEvent);
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            print_error( err, " clSetKernelArg failed" );
            return -1;
        }

        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );

        if( err != CL_SUCCESS ){
            print_error( err, " clEnqueueNDRangeKernel failed" );
            clReleaseEvent(writeEvent);
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            return -1;
        }

        if( ! strcmp( type, "half" ) ){
            err = clEnqueueReadBuffer( queue, streams[ii+1], true, 0, outPtrSizes[i]*num_elements, outptr[i], 0, NULL, NULL );
        }
        else{
            err = clEnqueueReadBuffer( queue, streams[ii+1], true, 0, outPtrSizes[i]*num_elements, outptr[i], 0, NULL, NULL );
        }
        if( err != CL_SUCCESS ){
            clReleaseEvent(writeEvent);
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[ii] );
            clReleaseMemObject( streams[ii+1] );
            free( outptr[i] );
            print_error( err, " clEnqueueReadBuffer failed" );
            return -1;
        }

        char *inP = (char *)inptr[i];
        char *outP = (char *)outptr[i];
        int err2 = 0;
        for( size_t p = 0; p < (size_t)num_elements; p++ )
        {
            if( fn( inP, outP, (int)(ptrSizes[i] / ptrSizes[0]) ) )
            {
                log_error( " %s%d data failed to verify\n", type, 1<<i );
                err2 = -1;
                err_count++;
            }
            inP += ptrSizes[i];
            outP += outPtrSizes[i];
        }
        if( !err2 )
        {
            log_info(" %s%d data verified\n", type, 1 << i);
        }
        err = err2;

        if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
            err_count++;

        // cleanup
        clReleaseEvent(writeEvent);
        clReleaseKernel( kernel[i] );
        clReleaseProgram( program[i] );
        clReleaseMemObject( streams[ii] );
        clReleaseMemObject( streams[ii+1] );
        free( outptr[i] );
    }

    return err_count;

}    // end test_stream_write()


REGISTER_TEST(write_array_int)
{
    int    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_int;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (int *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_int ), "int", 5, (void**)inptr,
                            stream_write_int_kernel_code, int_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);

    return err;

}    // end write_int_array()


REGISTER_TEST(write_array_uint)
{
    cl_uint    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_uint;

    ptrSizes[0] = sizeof(cl_uint);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (cl_uint *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_uint ), "uint", 5, (void **)inptr,
                            stream_write_uint_kernel_code, uint_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_uint_array()


REGISTER_TEST(write_array_short)
{
    short    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_short;

    ptrSizes[0] = sizeof(cl_short);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (short *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (short)genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_short ), "short", 5, (void **)inptr,
                            stream_write_short_kernel_code, short_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_short_array()


REGISTER_TEST(write_array_ushort)
{
    cl_ushort    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_ushort;

    ptrSizes[0] = sizeof(cl_ushort);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ushort *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ushort)genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_ushort ), "ushort", 5, (void **)inptr,
                            stream_write_ushort_kernel_code, ushort_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_ushort_array()


REGISTER_TEST(write_array_char)
{
    char    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_char;

    ptrSizes[0] = sizeof(cl_char);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (char *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (char)genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_char ), "char", 5, (void **)inptr,
                            stream_write_char_kernel_code, char_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_char_array()


REGISTER_TEST(write_array_uchar)
{
    uchar    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_uchar;

    ptrSizes[0] = sizeof(cl_uchar);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (uchar *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (uchar)genrand_int32(d);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_uchar ), "uchar", 5, (void **)inptr,
                            stream_write_uchar_kernel_code, uchar_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_uchar_array()


REGISTER_TEST(write_array_float)
{
    float    *inptr[5];
    size_t    ptrSizes[5];
    int        i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_float;

    ptrSizes[0] = sizeof(cl_float);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (float *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = get_random_float( -FLT_MAX, FLT_MAX, d );
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_float ), "float", 5, (void **)inptr,
                            stream_write_float_kernel_code, float_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_float_array()


REGISTER_TEST(write_array_long)
{
    cl_long    *inptr[5];
    size_t        ptrSizes[5];
    int            i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_long;

    if (!gHasLong)
    {
        log_info("write_long_array: Long types unsupported, skipping.");
        return CL_SUCCESS;
    }

    ptrSizes[0] = sizeof(cl_long);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (cl_long *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_long) genrand_int32(d) ^ ((cl_long) genrand_int32(d) << 32);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_long ), "cl_long", 5, (void **)inptr,
                            stream_write_long_kernel_code, long_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_long_array()


REGISTER_TEST(write_array_ulong)
{
    cl_ulong    *inptr[5];
    size_t                ptrSizes[5];
    int                    i, j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_ulong;

    if (!gHasLong)
    {
        log_info("write_long_array: Long types unsupported, skipping.");
        return CL_SUCCESS;
    }

    ptrSizes[0] = sizeof(cl_ulong);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ulong *)malloc(ptrSizes[i] * num_elements);

        for( j = 0; (unsigned int)j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( cl_ulong ), "ulong long", 5, (void **)inptr,
                            stream_write_ulong_kernel_code, ulong_kernel_name, foo, d );

    for( i = 0; i < 5; i++ ){
        free( (void *)inptr[i] );
    }

    free_mtdata(d);
    return err;

}    // end write_ulong_array()


REGISTER_TEST(write_array_struct)
{
    TestStruct            *inptr[1];
    size_t                ptrSizes[1];
    int                    j, err;
    int    (*foo)(void *,void *,int);
    MTdata d = init_genrand( gRandomSeed );
    foo = verify_write_struct;

    ptrSizes[0] = sizeof( TestStruct );

    inptr[0] = (TestStruct *)malloc( ptrSizes[0] * num_elements );

    for( j = 0; (unsigned int)j < ptrSizes[0] * num_elements / ptrSizes[0]; j++ ){
        inptr[0][j].a = (int)genrand_int32(d);
        inptr[0][j].b = get_random_float( 0.f, 1.844674407370954e+19f, d );
    }

    err = test_stream_write( device, context, queue, num_elements, sizeof( TestStruct ), "struct", 1, (void **)inptr,
                            stream_write_struct_kernel_code, struct_kernel_name, foo, d );

    free( (void *)inptr[0] );

    free_mtdata(d);
    return err;

}    // end write_struct_array()





