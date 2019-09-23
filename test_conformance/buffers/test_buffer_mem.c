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

#ifndef uchar
typedef unsigned char uchar;
#endif

#define USE_LOCAL_WORK_GROUP 1


const char *mem_read_write_kernel_code =
"__kernel void test_mem_read_write(__global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = dst[tid]+1;\n"
"}\n";

const char *mem_read_kernel_code =
"__kernel void test_mem_read(__global int *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src[tid]+1;\n"
"}\n";

const char *mem_write_kernel_code =
"__kernel void test_mem_write(__global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = dst[tid]+1;\n"
"}\n";


static int verify_mem( int *outptr, int n )
{
    int i;

    for ( i = 0; i < n; i++ ){
        if ( outptr[i] != ( i + 1 ) )
            return -1;
    }

    return 0;
}



int test_mem_read_write_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem      buffers[1];
    cl_int      *inptr, *outptr;
    cl_program  program[1];
    cl_kernel   kernel[1];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    inptr = (cl_int*)align_malloc(sizeof(cl_int)  * num_elements, min_alignment);
    outptr = (cl_int*)align_malloc(sizeof(cl_int) * num_elements, min_alignment);
    buffers[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * num_elements, NULL, &err);
    if (err != CL_SUCCESS) {
        print_error( err, "clCreateBuffer failed");
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    for (i=0; i<num_elements; i++)
        inptr[i] = i;

    err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)inptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        print_error( err, "clEnqueueWriteBuffer failed");
        clReleaseMemObject( buffers[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &mem_read_write_kernel_code, "test_mem_read_write" );
    if (err){
        clReleaseMemObject( buffers[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = get_max_common_work_group_size( context, kernel[0], global_work_size[0], &local_work_size[0] );
    test_error( err, "Unable to get work group size to use" );
#endif

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&buffers[0] );
    if ( err != CL_SUCCESS ){
        print_error( err, "clSetKernelArg failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
    if (err != CL_SUCCESS){
        log_error("clEnqueueNDRangeKernel failed\n");
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, buffers[0], true, 0, sizeof(cl_int)*num_elements, (void *)outptr, 0, NULL, NULL );
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueReadBuffer failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    if (verify_mem(outptr, num_elements)){
        log_error("buffer_MEM_READ_WRITE test failed\n");
        err = -1;
    }
    else{
        log_info("buffer_MEM_READ_WRITE test passed\n");
        err = 0;
    }

    // cleanup
    clReleaseMemObject( buffers[0] );
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    align_free( (void *)outptr );
    align_free( (void *)inptr );

    return err;
}   // end test_mem_read_write()


int test_mem_write_only_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem      buffers[1];
    int         *inptr, *outptr;
    cl_program  program[1];
    cl_kernel   kernel[1];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    inptr = (int *)align_malloc( sizeof(cl_int) * num_elements, min_alignment);
    if ( ! inptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(cl_int) * num_elements );
        return -1;
    }
    outptr = (int *)align_malloc( sizeof(cl_int) * num_elements, min_alignment);
    if ( ! outptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(cl_int) * num_elements );
        align_free( (void *)inptr );
        return -1;
    }
    buffers[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * num_elements, NULL, &err);
    if (err != CL_SUCCESS)
    {
        print_error(err, "clCreateBuffer failed\n");
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    for (i=0; i<num_elements; i++)
        inptr[i] = i;

    err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)inptr, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        print_error( err, "clEnqueueWriteBuffer failed" );
        clReleaseMemObject( buffers[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &mem_write_kernel_code, "test_mem_write" );
    if (err){
        clReleaseMemObject( buffers[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = get_max_common_work_group_size( context, kernel[0], global_work_size[0], &local_work_size[0] );
    test_error( err, "Unable to get work group size to use" );
#endif

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&buffers[0] );
    if ( err != CL_SUCCESS ){
        print_error( err, "clSetKernelArg failed");
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueNDRangeKernel failed" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, buffers[0], true, 0, sizeof(cl_int)*num_elements, (void *)outptr, 0, NULL, NULL );
    if ( err != CL_SUCCESS ){
        print_error( err, "Error reading array" );
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    // cleanup
    clReleaseMemObject( buffers[0] );
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    align_free( (void *)outptr );
    align_free( (void *)inptr );

    return err;
}   // end test_mem_write()


int test_mem_read_only_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem      buffers[2];
    int         *inptr, *outptr;
    cl_program  program[1];
    cl_kernel   kernel[1];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    inptr = (int *)align_malloc( sizeof(cl_int) * num_elements, min_alignment);
    if ( ! inptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(cl_int) * num_elements );
        return -1;
    }
    outptr = (int *)align_malloc( sizeof(cl_int) * num_elements, min_alignment);
    if ( ! outptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(cl_int) * num_elements );
        align_free( (void *)inptr );
        return -1;
    }

    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * num_elements, NULL, &err);
    if ( err != CL_SUCCESS ){
        print_error(err, " clCreateBuffer failed to create READ_ONLY array\n" );
        align_free( (void *)outptr );
        align_free( (void *)inptr );
        return -1;
    }

    for (i=0; i<num_elements; i++)
        inptr[i] = i;

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, &err);
    if ( err != CL_SUCCESS ){
        print_error(err, " clCreateBuffer failed to create MEM_ALLOC_GLOBAL_POOL array\n" );
        clReleaseMemObject( buffers[0]) ;
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

    err = clEnqueueWriteBuffer(queue, buffers[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)inptr, 0, NULL, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueWriteBuffer() failed");
        clReleaseMemObject( buffers[1]) ;
        clReleaseMemObject( buffers[0]) ;
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &mem_read_kernel_code, "test_mem_read" );
    if ( err ){
        clReleaseMemObject( buffers[1]) ;
        clReleaseMemObject( buffers[0]) ;
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = get_max_common_work_group_size( context, kernel[0], global_work_size[0], &local_work_size[0] );
    test_error( err, "Unable to get work group size to use" );
#endif

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&buffers[0] );
    err |= clSetKernelArg( kernel[0], 1, sizeof( cl_mem ), (void *)&buffers[1] );
    if ( err != CL_SUCCESS ){
        print_error( err, "clSetKernelArgs failed" );
        clReleaseMemObject( buffers[1]) ;
        clReleaseMemObject( buffers[0]) ;
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
    if (err != CL_SUCCESS){
        print_error( err, "clEnqueueNDRangeKernel failed" );
        clReleaseMemObject( buffers[1]) ;
        clReleaseMemObject( buffers[0]) ;
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, buffers[1], true, 0, sizeof(cl_int)*num_elements, (void *)outptr, 0, NULL, NULL );
    if ( err != CL_SUCCESS ){
        print_error( err, "clEnqueueReadBuffer failed" );
        clReleaseMemObject( buffers[1]) ;
        clReleaseMemObject( buffers[0]) ;
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)inptr );
        align_free( (void *)outptr );
        return -1;
    }

    if (verify_mem(outptr, num_elements)){
        log_error( " CL_MEM_READ_ONLY test failed\n" );
        err = -1;
    }
    else{
        log_info( " CL_MEM_READ_ONLY test passed\n" );
        err = 0;
    }

    // cleanup
    clReleaseMemObject( buffers[1]) ;
    clReleaseMemObject( buffers[0]) ;
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    align_free( (void *)inptr );
    align_free( (void *)outptr );

    return err;

}   // end test_mem_read()


int test_mem_copy_host_flags( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_mem      buffers[1];
    int         *ptr;
    cl_program  program[1];
    cl_kernel   kernel[1];
    size_t      global_work_size[3];
#ifdef USE_LOCAL_WORK_GROUP
    size_t      local_work_size[3];
#endif
    cl_int      err;
    int         i;

    size_t min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptr = (int *)align_malloc( sizeof(cl_int) * num_elements, min_alignment);
    if ( ! ptr ){
        log_error( " unable to allocate %d bytes of memory\n", (int)sizeof(cl_int) * num_elements );
        return -1;
    }

    for (i=0; i<num_elements; i++)
        ptr[i] = i;

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sizeof(cl_int) * num_elements, (void *)ptr, &err);
    if (err != CL_SUCCESS){
        print_error(err, "clCreateBuffer failed for CL_MEM_COPY_HOST_PTR\n");
        align_free( (void *)ptr );
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &mem_read_write_kernel_code, "test_mem_read_write" );
    if (err){
        clReleaseMemObject( buffers[0] );
        align_free( (void *)ptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = get_max_common_work_group_size( context, kernel[0], global_work_size[0], &local_work_size[0] );
    test_error( err, "Unable to get work group size to use" );
#endif

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&buffers[0] );
    if (err != CL_SUCCESS){
        log_error("clSetKernelArgs failed\n");
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)ptr );
        return -1;
    }

#ifdef USE_LOCAL_WORK_GROUP
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, local_work_size, 0, NULL, NULL );
#else
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, NULL );
#endif
    if (err != CL_SUCCESS){
        log_error("clEnqueueNDRangeKernel failed\n");
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)ptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, buffers[0], true, 0, sizeof(cl_int)*num_elements, (void *)ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS){
        log_error("CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_CONSTANT_POOL failed.\n");
        clReleaseMemObject( buffers[0] );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        align_free( (void *)ptr );
        return -1;
    }

    if ( verify_mem( ptr, num_elements ) ){
        log_error("CL_MEM_COPY_HOST_PTR test failed\n");
        err = -1;
    }
    else{
        log_info("CL_MEM_COPY_HOST_PTR test passed\n");
        err = 0;
    }

    // cleanup
    clReleaseMemObject( buffers[0] );
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    align_free( (void *)ptr );

    return err;

}   // end test_mem_copy_host_flags()

