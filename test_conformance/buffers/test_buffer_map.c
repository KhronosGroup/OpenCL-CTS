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


#define TEST_PRIME_INT        ((1<<16)+1)
#define TEST_PRIME_UINT        ((1U<<16)+1U)
#define TEST_PRIME_LONG        ((1LL<<32)+1LL)
#define TEST_PRIME_ULONG    ((1ULL<<32)+1ULL)
#define TEST_PRIME_SHORT    ((1S<<8)+1S)
#define TEST_PRIME_FLOAT    (float)3.40282346638528860e+38
#define TEST_PRIME_HALF        119.f
#define TEST_BOOL            true
#define TEST_PRIME_CHAR        0x77


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


static const char *buffer_read_struct_kernel_code[] = {
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
    "}\n" };

static const char *struct_kernel_name[] = { "test_buffer_read_struct" };


//--- the verify functions
static int verify_read_int(void *ptr, int n)
{
    int     i;
    int     *outptr = (int *)ptr;

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
    int     i;
    short   *outptr = (short *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != (short)((1<<8)+1) )
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
    int     i;
    float   *outptr = (float *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_FLOAT )
            return -1;
    }

    return 0;
}


static int verify_read_char(void *ptr, int n)
{
    int     i;
    char    *outptr = (char *)ptr;

    for (i=0; i<n; i++){
        if ( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_uchar( void *ptr, int n )
{
    int      i;
    cl_uchar *outptr = (cl_uchar *)ptr;

    for ( i = 0; i < n; i++ ){
        if ( outptr[i] != TEST_PRIME_CHAR )
            return -1;
    }

    return 0;
}


static int verify_read_struct( void *ptr, int n )
{
    int         i;
    TestStruct  *outptr = (TestStruct *)ptr;

    for ( i = 0; i < n; i++ ){
        if ( ( outptr[i].a != TEST_PRIME_INT ) ||
             ( outptr[i].b != TEST_PRIME_FLOAT ) )
            return -1;
    }

    return 0;
}


//----- the test functions
static int test_buffer_map_read( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                                 const char *kernelCode[], const char *kernelName[], int (*fn)(void *,int) )
{
    cl_mem      buffers[5];
    void        *outptr[5];
    cl_program  program[5];
    cl_kernel   kernel[5];
    size_t      threads[3], localThreads[3];
    cl_int      err;
    int         i;
    size_t      ptrSizes[5];
    int         src_flag_id;
    int         total_errors = 0;
    void        *mappedPtr;

    size_t      min_alignment = get_min_alignment(context);

    threads[0] = (cl_uint)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //embedded devices don't support long/ulong so skip over
    if (! gHasLong && strstr(type,"long"))
        return 0;

    for (src_flag_id=0; src_flag_id < NUM_FLAGS; src_flag_id++) {
        log_info("Testing with cl_mem_flags src: %s\n", flag_set_names[src_flag_id]);

        for ( i = 0; i < loops; i++ ){
            outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
            if ( ! outptr[i] ){
                log_error( " unable to allocate %d bytes of memory\n", (int)ptrSizes[i] * num_elements );
                return -1;
            }

            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR) || (flag_set[src_flag_id] & CL_MEM_COPY_HOST_PTR))
                buffers[i] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, outptr[i], &err);
            else
                buffers[i] = clCreateBuffer(context, flag_set[src_flag_id],  ptrSizes[i] * num_elements, NULL, &err);

            if ( ! buffers[i] | err){
                print_error(err, "clCreateBuffer failed\n" );
                align_free( outptr[i] );
                return -1;
            }

            err = create_single_kernel_helper(context, &program[i], &kernel[i], 1, &kernelCode[i], kernelName[i] );
            if ( err ){
                log_error( " Error creating program for %s\n", type );
                clReleaseMemObject( buffers[i] );
                align_free( outptr[i] );
                return -1;
            }

            err = clSetKernelArg( kernel[i], 0, sizeof( cl_mem ), (void *)&buffers[i] );
            if ( err != CL_SUCCESS ){
                print_error( err, "clSetKernelArg failed\n" );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                clReleaseMemObject( buffers[i] );
                align_free( outptr[i] );
                return -1;
            }

            threads[0] = (cl_uint)num_elements;

            err = get_max_common_work_group_size( context, kernel[i], threads[0], &localThreads[0] );
            test_error( err, "Unable to get work group size to use" );

            err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, localThreads, 0, NULL, NULL );
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueNDRangeKernel failed\n" );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                clReleaseMemObject( buffers[i] );
                align_free( outptr[i] );
                return -1;
            }

            mappedPtr = clEnqueueMapBuffer(queue, buffers[i], CL_TRUE, CL_MAP_READ, 0, ptrSizes[i]*num_elements, 0, NULL, NULL, &err);
            if ( err != CL_SUCCESS ){
                print_error( err, "clEnqueueMapBuffer failed" );
                clReleaseKernel( kernel[i] );
                clReleaseProgram( program[i] );
                clReleaseMemObject( buffers[i] );
                align_free( outptr[i] );
                return -1;
            }

            if (fn(mappedPtr, num_elements*(1<<i))){
                log_error(" %s%d test failed\n", type, 1<<i);
                total_errors++;
            }
            else{
                log_info(" %s%d test passed\n", type, 1<<i);
            }

            err = clEnqueueUnmapMemObject(queue, buffers[i], mappedPtr, 0, NULL, NULL);
            test_error(err, "clEnqueueUnmapMemObject failed");

            // cleanup
            clReleaseKernel( kernel[i] );
            clReleaseProgram( program[i] );
            clReleaseMemObject( buffers[i] );

            // If we are using the outptr[i] as backing via USE_HOST_PTR we need to make sure we are done before freeing.
            if ((flag_set[src_flag_id] & CL_MEM_USE_HOST_PTR)) {
                err = clFinish(queue);
                test_error(err, "clFinish failed");
            }
            align_free( outptr[i] );
        }
    } // cl_mem_flags

    return total_errors;

}   // end test_buffer_map_read()


#define DECLARE_LOCK_TEST(type, realType) \
int test_buffer_map_read_##type( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )    \
{ \
return test_buffer_map_read( deviceID, context, queue,  num_elements, sizeof( realType ), (char*)#type, 5, \
buffer_read_##type##_kernel_code, type##_kernel_name, verify_read_##type ); \
}

DECLARE_LOCK_TEST(int, cl_int)
DECLARE_LOCK_TEST(uint, cl_uint)
DECLARE_LOCK_TEST(long, cl_long)
DECLARE_LOCK_TEST(ulong, cl_ulong)
DECLARE_LOCK_TEST(short, cl_short)
DECLARE_LOCK_TEST(ushort, cl_ushort)
DECLARE_LOCK_TEST(char, cl_char)
DECLARE_LOCK_TEST(uchar, cl_uchar)
DECLARE_LOCK_TEST(float, cl_float)

int test_buffer_map_read_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    int (*foo)(void *,int);
    foo = verify_read_struct;

    return test_buffer_map_read( deviceID, context, queue, num_elements, sizeof( TestStruct ), (char*)"struct", 1,
                                 buffer_read_struct_kernel_code, struct_kernel_name, foo );

}   // end test_buffer_map_struct_read()

