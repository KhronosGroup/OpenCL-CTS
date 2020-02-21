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
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"

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
    int        a;
    float    b;
} TestStruct;
#endif



//--- the code for the kernel executables
static const char *stream_read_int_kernel_code[] = {
"__kernel void test_stream_read_int(__global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1<<16)+1);\n"
"}\n",

"__kernel void test_stream_read_int2(__global int2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1<<16)+1);\n"
"}\n",

"__kernel void test_stream_read_int4(__global int4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1<<16)+1);\n"
"}\n",

"__kernel void test_stream_read_int8(__global int8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1<<16)+1);\n"
"}\n",

"__kernel void test_stream_read_int16(__global int16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1<<16)+1);\n"
"}\n" };

static const char *int_kernel_name[] = { "test_stream_read_int", "test_stream_read_int2", "test_stream_read_int4", "test_stream_read_int8", "test_stream_read_int16" };

const char *stream_read_uint_kernel_code[] = {
"__kernel void test_stream_read_uint(__global uint *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1U<<16)+1U);\n"
"}\n",

"__kernel void test_stream_read_uint2(__global uint2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1U<<16)+1U);\n"
"}\n",

"__kernel void test_stream_read_uint4(__global uint4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1U<<16)+1U);\n"
"}\n",

"__kernel void test_stream_read_uint8(__global uint8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1U<<16)+1U);\n"
"}\n",

"__kernel void test_stream_read_uint16(__global uint16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1U<<16)+1U);\n"
"}\n" };

const char *uint_kernel_name[] = { "test_stream_read_uint", "test_stream_read_uint2", "test_stream_read_uint4", "test_stream_read_uint8", "test_stream_read_uint16" };

const char *stream_read_long_kernel_code[] = {
"__kernel void test_stream_read_long(__global long *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1L<<32)+1L);\n"
"}\n",

"__kernel void test_stream_read_long2(__global long2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1L<<32)+1L);\n"
"}\n",

"__kernel void test_stream_read_long4(__global long4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1L<<32)+1L);\n"
"}\n",

"__kernel void test_stream_read_long8(__global long8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1L<<32)+1L);\n"
"}\n",

"__kernel void test_stream_read_long16(__global long16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1L<<32)+1L);\n"
"}\n" };

const char *long_kernel_name[] = { "test_stream_read_long", "test_stream_read_long2", "test_stream_read_long4", "test_stream_read_long8", "test_stream_read_long16" };

const char *stream_read_ulong_kernel_code[] = {
"__kernel void test_stream_read_ulong(__global ulong *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1UL<<32)+1UL);\n"
"}\n",

"__kernel void test_stream_read_ulong2(__global ulong2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1UL<<32)+1UL);\n"
"}\n",

"__kernel void test_stream_read_ulong4(__global ulong4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1UL<<32)+1UL);\n"
"}\n",

"__kernel void test_stream_read_ulong8(__global ulong8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1UL<<32)+1UL);\n"
"}\n",

"__kernel void test_stream_read_ulong16(__global ulong16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = ((1UL<<32)+1UL);\n"
"}\n" };

const char *ulong_kernel_name[] = { "test_stream_read_ulong", "test_stream_read_ulong2", "test_stream_read_ulong4", "test_stream_read_ulong8", "test_stream_read_ulong16" };

const char *stream_read_short_kernel_code[] = {
"__kernel void test_stream_read_short(__global short *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (short)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_short2(__global short2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (short)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_short4(__global short4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (short)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_short8(__global short8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (short)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_short16(__global short16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (short)((1<<8)+1);\n"
"}\n" };

const char *short_kernel_name[] = { "test_stream_read_short", "test_stream_read_short2", "test_stream_read_short4", "test_stream_read_short8", "test_stream_read_short16" };


const char *stream_read_ushort_kernel_code[] = {
"__kernel void test_stream_read_ushort(__global ushort *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (ushort)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_ushort2(__global ushort2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (ushort)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_ushort4(__global ushort4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (ushort)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_ushort8(__global ushort8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (ushort)((1<<8)+1);\n"
"}\n",

"__kernel void test_stream_read_ushort16(__global ushort16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (ushort)((1<<8)+1);\n"
"}\n" };

static const char *ushort_kernel_name[] = { "test_stream_read_ushort", "test_stream_read_ushort2", "test_stream_read_ushort4", "test_stream_read_ushort8", "test_stream_read_ushort16" };


const char *stream_read_float_kernel_code[] = {
"__kernel void test_stream_read_float(__global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)3.40282346638528860e+38;\n"
"}\n",

"__kernel void test_stream_read_float2(__global float2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)3.40282346638528860e+38;\n"
"}\n",

"__kernel void test_stream_read_float4(__global float4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)3.40282346638528860e+38;\n"
"}\n",

"__kernel void test_stream_read_float8(__global float8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)3.40282346638528860e+38;\n"
"}\n",

"__kernel void test_stream_read_float16(__global float16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (float)3.40282346638528860e+38;\n"
"}\n" };

const char *float_kernel_name[] = { "test_stream_read_float", "test_stream_read_float2", "test_stream_read_float4", "test_stream_read_float8", "test_stream_read_float16" };


const char *stream_read_half_kernel_code[] = {
"__kernel void test_stream_read_half(__global half *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (half)119;\n"
"}\n",

"__kernel void test_stream_read_half2(__global half2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (half)119;\n"
"}\n",

"__kernel void test_stream_read_half4(__global half4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (half)119;\n"
"}\n",

"__kernel void test_stream_read_half8(__global half8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (half)119;\n"
"}\n",

"__kernel void test_stream_read_half16(__global half16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (half)119;\n"
"}\n" };

const char *half_kernel_name[] = { "test_stream_read_half", "test_stream_read_half2", "test_stream_read_half4", "test_stream_read_half8", "test_stream_read_half16" };


const char *stream_read_char_kernel_code[] = {
"__kernel void test_stream_read_char(__global char *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (char)'w';\n"
"}\n",

"__kernel void test_stream_read_char2(__global char2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (char)'w';\n"
"}\n",

"__kernel void test_stream_read_char4(__global char4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (char)'w';\n"
"}\n",

"__kernel void test_stream_read_char8(__global char8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (char)'w';\n"
"}\n",

"__kernel void test_stream_read_char16(__global char16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (char)'w';\n"
"}\n" };

const char *char_kernel_name[] = { "test_stream_read_char", "test_stream_read_char2", "test_stream_read_char4", "test_stream_read_char8", "test_stream_read_char16" };


const char *stream_read_uchar_kernel_code[] = {
"__kernel void test_stream_read_uchar(__global uchar *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = 'w';\n"
"}\n",

"__kernel void test_stream_read_uchar2(__global uchar2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (uchar)'w';\n"
"}\n",

"__kernel void test_stream_read_uchar4(__global uchar4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (uchar)'w';\n"
"}\n",

"__kernel void test_stream_read_uchar8(__global uchar8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (uchar)'w';\n"
"}\n",

"__kernel void test_stream_read_uchar16(__global uchar16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (uchar)'w';\n"
"}\n" };

const char *uchar_kernel_name[] = { "test_stream_read_uchar", "test_stream_read_uchar2", "test_stream_read_uchar4", "test_stream_read_uchar8", "test_stream_read_uchar16" };


const char *stream_read_struct_kernel_code[] = {
"typedef struct{\n"
"int    a;\n"
"float    b;\n"
"} TestStruct;\n"
"__kernel void test_stream_read_struct(__global TestStruct *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid].a = ((1<<16)+1);\n"
"     dst[tid].b = (float)3.40282346638528860e+38;\n"
"}\n" };

const char *struct_kernel_name[] = { "test_stream_read_struct" };



//--- the verify functions
static int verify_read_int(void *ptr, int n)
{
    int        i;
    int        *outptr = (int *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_INT )
            return -1;
    }

    return 0;
}


static int verify_read_uint(void *ptr, int n)
{
    int        i;
    cl_uint    *outptr = (cl_uint *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_UINT )
            return -1;
    }

    return 0;
}


static int verify_read_long(void *ptr, int n)
{
    int        i;
    cl_long    *outptr = (cl_long *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_LONG )
            return -1;
    }

    return 0;
}


static int verify_read_ulong(void *ptr, int n)
{
    int        i;
    cl_ulong    *outptr = (cl_ulong *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_ULONG )
            return -1;
    }

    return 0;
}


static int verify_read_short(void *ptr, int n)
{
    int        i;
    short    *outptr = (short *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != (short)((1<<8)+1) )
            return -1;
    }

    return 0;
}


static int verify_read_ushort(void *ptr, int n)
{
    int        i;
    cl_ushort    *outptr = (cl_ushort *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != (cl_ushort)((1<<8)+1) )
            return -1;
    }

    return 0;
}


static int verify_read_float( void *ptr, int n )
{
    int        i;
    float    *outptr = (float *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_FLOAT )
            return -1;
    }

    return 0;
}


static int verify_read_half( void *ptr, int n )
{
    int        i;
    float    *outptr = (float *)ptr;

    for( i = 0; i < n / 2; i++ ){
        if( outptr[i] != TEST_PRIME_HALF )
            return -1;
    }

    return 0;
}


static int verify_read_char(void *ptr, int n)
{
    int        i;
    char    *outptr = (char *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_uchar( void *ptr, int n )
{
    int        i;
    uchar    *outptr = (uchar *)ptr;

    for (i=0; i<n; i++){
        if( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_struct( void *ptr, int n )
{
    int         i;
    TestStruct    *outptr = (TestStruct *)ptr;

    for ( i = 0; i < n; i++ ){
        if( ( outptr[i].a != TEST_PRIME_INT ) ||
           ( outptr[i].b != TEST_PRIME_FLOAT ) )
            return -1;
    }

    return 0;
}

//----- the test functions
int test_stream_read( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements, size_t size, const char *type, int loops,
                     const char *kernelCode[], const char *kernelName[], int (*fn)(void *,int) )
{
    cl_mem            streams[5];
    void            *outptr[5];
    cl_program        program[5];
    cl_kernel        kernel[5];
    cl_event        readEvent;
    cl_ulong    queueStart, submitStart, readStart, readEnd;
    size_t            threads[1];
#ifdef USE_LOCAL_THREADS
    size_t            localThreads[1];
#endif
    int                err, err_count = 0;
    int                i;
    size_t            ptrSizes[5];

    threads[0] = (size_t)num_elements;

#ifdef USE_LOCAL_THREADS
    err = clGetDeviceConfigInfo( id, CL_DEVICE_MAX_THREAD_GROUP_SIZE, localThreads, sizeof( cl_uint ), NULL );
    if( err != CL_SUCCESS ){
        log_error( "Unable to get thread group max size: %d", err );
        return -1;
    }
    if( localThreads[0] > threads[0] )
        localThreads[0] = threads[0];
#endif

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;
    for( i = 0; i < loops; i++ ){
        outptr[i] = malloc( ptrSizes[i] * num_elements );
        if( ! outptr[i] ){
            log_error( " unable to allocate %d bytes for outptr\n", (int)( ptrSizes[i] * num_elements ) );
            return -1;
        }
        streams[i] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  ptrSizes[i] * num_elements, NULL, &err );
        if( !streams[i] ){
            log_error( " clCreateBuffer failed\n" );
            free( outptr[i] );
            return -1;
        }
        err = create_single_kernel_helper( context, &program[i], &kernel[i], 1, &kernelCode[i], kernelName[i] );
        if( err ){
            log_error( " Error creating program for %s\n", type );
            clReleaseMemObject(streams[i]);
            free( outptr[i] );
            return -1;
        }

        err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&streams[i] );
        if( err != CL_SUCCESS ){
            print_error( err, "clSetKernelArg failed" );
            clReleaseProgram( program[i] );
            clReleaseKernel( kernel[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

#ifdef USE_LOCAL_THREADS
        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, localThreads, 0, NULL, NULL );
#else
        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
#endif
        if( err != CL_SUCCESS ){
            print_error( err, "clEnqueueNDRangeKernel failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

        err = clEnqueueReadBuffer( queue, streams[i], false, 0, ptrSizes[i]*num_elements, outptr[i], 0, NULL, &readEvent );
        if( err != CL_SUCCESS ){
            print_error( err, "clEnqueueReadBuffer failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }
        err = clWaitForEvents( 1, &readEvent );
        if( err != CL_SUCCESS )
        {
            print_error( err, "Unable to wait for event completion" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }
        err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

        err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

        err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &readStart, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

        err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &readEnd, NULL );
        if( err != CL_SUCCESS ){
            print_error( err, "clGetEventProfilingInfo failed" );
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( streams[i] );
            free( outptr[i] );
            return -1;
        }

        if (fn(outptr[i], num_elements*(1<<i))){
            log_error( " %s%d data failed to verify\n", type, 1<<i );
            err_count++;
        }
        else{
            log_info( " %s%d data verified\n", type, 1<<i );
        }

    if (check_times(queueStart, submitStart, readStart, readEnd, device))
      err_count++;

        // cleanup
        clReleaseEvent(readEvent);
        clReleaseKernel( kernel[i] );
        clReleaseProgram( program[i] );
        clReleaseMemObject( streams[i] );
        free( outptr[i] );
    }

    return err_count;

}    // end test_stream_read()


int test_read_array_int( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_int;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_int ), "int", 5,
                             stream_read_int_kernel_code, int_kernel_name, foo );
}


int test_read_array_uint( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_uint;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_uint ), "uint", 5,
                             stream_read_uint_kernel_code, uint_kernel_name, foo );
}


int test_read_array_long( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_long;

    if (!gHasLong)
    {
        log_info("read_long_array: Long types unsupported, skipping.");
        return CL_SUCCESS;
    }

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_long ), "long", 5,
                             stream_read_long_kernel_code, long_kernel_name, foo );
}


int test_read_array_ulong( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_ulong;

    if (!gHasLong)
    {
        log_info("read_long_array: Long types unsupported, skipping.");
        return CL_SUCCESS;
    }

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_ulong ), "ulong", 5,
                             stream_read_ulong_kernel_code, ulong_kernel_name, foo );
}


int test_read_array_short( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_short;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_short ), "short", 5,
                             stream_read_short_kernel_code, short_kernel_name, foo );
}


int test_read_array_ushort( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_ushort;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_ushort ), "ushort", 5,
                             stream_read_ushort_kernel_code, ushort_kernel_name, foo );
}


int test_read_array_float( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_float;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_float ), "float", 5,
                             stream_read_float_kernel_code, float_kernel_name, foo );
}


int test_read_array_half( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_half;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_half ), "half", 5,
                             stream_read_half_kernel_code, half_kernel_name, foo );
}


int test_read_array_char( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_char;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_char ), "char", 5,
                             stream_read_char_kernel_code, char_kernel_name, foo );
}


int test_read_array_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_uchar;

    return test_stream_read( device, context, queue, num_elements, sizeof( cl_uchar ), "uchar", 5,
                             stream_read_uchar_kernel_code, uchar_kernel_name, foo );
}


int test_read_array_struct( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int    (*foo)(void *,int);
    foo = verify_read_struct;

    return test_stream_read( device, context, queue, num_elements, sizeof( TestStruct ), "struct", 1,
                             stream_read_struct_kernel_code, struct_kernel_name, foo );
}

/*
int read_struct_array(cl_device_group device, cl_device id, cl_context context, int num_elements)
{
    cl_mem            streams[1];
    TestStruct        *output_ptr;
    cl_program        program[1];
    cl_kernel        kernel[1];
    void            *values[1];
    size_t            sizes[1] = { sizeof(cl_stream) };
    size_t            threads[1];
#ifdef USE_LOCAL_THREADS
    size_t            localThreads[1];
#endif
    int                err;
    size_t            objSize = sizeof(TestStruct);

    threads[0] = (size_t)num_elements;

#ifdef USE_LOCAL_THREADS
    err = clGetDeviceConfigInfo( id, CL_DEVICE_MAX_THREAD_GROUP_SIZE, localThreads, sizeof( cl_uint ), NULL );
    if( err != CL_SUCCESS ){
        log_error( "Unable to get thread group max size: %d", err );
        return -1;
    }
    if( localThreads[0] > threads[0] )
        localThreads[0] = threads[0];
#endif

    output_ptr = malloc(objSize * num_elements);
    if( ! output_ptr ){
        log_error( " unable to allocate %d bytes for output_ptr\n", (int)(objSize * num_elements) );
        return -1;
    }
    streams[0] = clCreateBuffer( device, (cl_mem_flags)(CL_MEM_READ_WRITE),  objSize * num_elements, NULL );
    if( !streams[0] ){
        log_error( " clCreateBuffer failed\n" );
        free( output_ptr );
        return -1;
    }

    err = create_program_and_kernel( device, stream_read_struct_kernel_code, "test_stream_read_struct", &program[0], &kernel[0]);
    if( err ){
        clReleaseProgram( program[0] );
        free( output_ptr );
        return -1;
    }

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&streams[0] );
    if( err != CL_SUCCESS){
        print_error( err, "clSetKernelArg failed" );
        clReleaseProgram( program[0] );
        clReleaseKernel( kernel[0] );
        clReleaseMemObject( streams[0] );
        free( output_ptr );
        return -1;
    }

#ifdef USE_LOCAL_THREADS
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, threads, localThreads, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, threads, NULL, 0, NULL, NULL );
#endif
    if( err != CL_SUCCESS ){
        print_error( err, "clEnqueueNDRangeKernel failed" );
        clReleaseProgram( program[0] );
        clReleaseKernel( kernel[0] );
        clReleaseMemObject( streams[0] );
        free( output_ptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[0], true, 0, objSize*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if( err != CL_SUCCESS){
        print_error( err, "clEnqueueReadBuffer failed" );
        clReleaseProgram( program[0] );
        clReleaseKernel( kernel[0] );
        clReleaseMemObject( streams[0] );
        free( output_ptr );
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
    clReleaseProgram( program[0] );
    clReleaseKernel( kernel[0] );
    clReleaseMemObject( streams[0] );
    free( output_ptr );

    return err;
}
*/


