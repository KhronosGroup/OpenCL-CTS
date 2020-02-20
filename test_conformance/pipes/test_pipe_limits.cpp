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
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/errorHelpers.h"

#define STRING_LENGTH  1024

void createKernelSourceCode(char *source, int num_pipes)
{
    int i;
    char str[256];
    int str_length;

    strcpy(source, "__kernel void test_multiple_pipe_write(__global int *src, ");

    for(i = 0; i < num_pipes; i++) {
        sprintf(str, "__write_only pipe int pipe%d, ", i);
        strcat(source, str);
    }
    sprintf(str, "int num_pipes ) \n{\n  int gid = get_global_id(0);\n  reserve_id_t res_id;\n\n");
    strcat(source, str);
    sprintf(str, "  if(gid < (get_global_size(0))/num_pipes)\n  {\n    res_id = reserve_write_pipe(pipe0, 1);\n    if(is_valid_reserve_id(res_id))\n    {\n");
    strcat(source, str);
    sprintf(str, "      write_pipe(pipe0, res_id, 0, &src[gid]);\n      commit_write_pipe(pipe0, res_id);\n    }\n  }\n");
    strcat(source, str);
    for(i = 1; i < num_pipes; i++){
        sprintf(str, "  else if(gid < (%d*get_global_size(0))/num_pipes)\n  {\n    res_id = reserve_write_pipe(pipe%d, 1);\n    if(is_valid_reserve_id(res_id))\n    {\n", i+1, i);
        strcat(source, str);
        sprintf(str, "      write_pipe(pipe%d, res_id, 0, &src[gid]);\n      commit_write_pipe(pipe%d, res_id);\n    }\n  }\n", i, i);
        strcat(source, str);
    }
    strcat(source, "}\n\n__kernel void test_multiple_pipe_read(__global int *dst, ");

    for(i = 0; i < num_pipes; i++) {
        sprintf(str, "__read_only pipe int pipe%d, ", i);
        strcat(source, str);
    }
    sprintf(str, "int num_pipes ) \n{\n  int gid = get_global_id(0);\n  reserve_id_t res_id;\n\n");
    strcat(source, str);
    sprintf(str, "  if(gid < (get_global_size(0))/num_pipes)\n  {\n    res_id = reserve_read_pipe(pipe0, 1);\n    if(is_valid_reserve_id(res_id))\n    {\n");
    strcat(source, str);
    sprintf(str, "      read_pipe(pipe0, res_id, 0, &dst[gid]);\n      commit_read_pipe(pipe0, res_id);\n    }\n  }\n");
    strcat(source, str);
    for(i = 1; i < num_pipes; i++){
        sprintf(str, "  else if(gid < (%d*get_global_size(0))/num_pipes)\n  {\n    res_id = reserve_read_pipe(pipe%d, 1);\n    if(is_valid_reserve_id(res_id))\n    {\n", i+1, i);
        strcat(source, str);
        sprintf(str, "      read_pipe(pipe%d, res_id, 0, &dst[gid]);\n      commit_read_pipe(pipe%d, res_id);\n    }\n  }\n", i, i);
        strcat(source, str);
    }
    strcat(source, "}");

    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH*num_pipes);
}

static int verify_result(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_char    *inptr = (cl_char *)ptr1;
    cl_char    *outptr = (cl_char *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_result_int(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_int    *inptr = (cl_int *)ptr1;
    cl_int    *outptr = (cl_int *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

int test_pipe_max_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{

    cl_mem        pipes[1024];
    cl_mem      buffers[2];
    void        *outptr;
    cl_int        *inptr;
    cl_program  program;
    cl_kernel   kernel[2];
    size_t      global_work_size[3];
    cl_int      err;
    cl_int        size;
    int            num_pipe_elements = 1024;
    int         i, j;
    int            max_pipe_args;
    char        *source;
    cl_event    producer_sync_event = NULL;
    cl_event    consumer_sync_event = NULL;
    MTdata      d = init_genrand( gRandomSeed );
    const char*    kernelName[] = {"test_multiple_pipe_write", "test_multiple_pipe_read"};

    size_t      min_alignment = get_min_alignment(context);

    err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PIPE_ARGS, sizeof(max_pipe_args), (void*)&max_pipe_args, NULL);
    if(err){
        print_error(err, " clGetDeviceInfo failed\n");
        return -1;
    }
    if(max_pipe_args < 16){
        log_error("The device should support minimum 16 pipe objects that could be passed as arguments to the kernel");
        return -1;
    }

    global_work_size[0] = (cl_uint)num_pipe_elements * max_pipe_args;
    size = sizeof(int) * num_pipe_elements * max_pipe_args;
    source = (char *)malloc(STRING_LENGTH * sizeof(char) * max_pipe_args);

    inptr = (cl_int *)align_malloc(size, min_alignment);

    for(i = 0; i < num_pipe_elements * max_pipe_args; i++){
        inptr[i] = (int)genrand_int32(d);
    }

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        free(source);
        print_error(err, " clCreateBuffer failed\n");
        return -1;
    }
    outptr = align_malloc(size, min_alignment);
    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size, outptr, &err);
    if ( err ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        free(source);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    for(i = 0; i < max_pipe_args; i++){
        pipes[i] = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), num_pipe_elements, NULL, &err);
        if(err){
            clReleaseMemObject(buffers[0]);
            clReleaseMemObject(buffers[1]);
            align_free( outptr );
            free(source);
            for(j = 0; j < i; j++) {
                clReleaseMemObject(pipes[j]);
            }
            print_error(err, " clCreatePipe failed\n");
            return -1;
        }
    }

    createKernelSourceCode(source, max_pipe_args);

    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, (const char**)&source, kernelName[0], "-cl-std=CL2.0");
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        align_free(outptr);
        free(source);
        print_error(err, "Error creating program\n");
        return -1;
    }
    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    if( kernel[1] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        align_free(outptr);
        free(source);
        print_error(err, " Error creating kernel\n");
        return -1;
    }

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    for( i = 0; i < max_pipe_args; i++){
        err |= clSetKernelArg(kernel[0], i+1, sizeof(cl_mem), (void*)&pipes[i]);
    }
    err |= clSetKernelArg(kernel[0], max_pipe_args + 1, sizeof(int), (void*)&max_pipe_args);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&buffers[1]);
    for( i = 0; i < max_pipe_args; i++){
        err |= clSetKernelArg(kernel[1], i+1, sizeof(cl_mem), (void*)&pipes[i]);
    }
    err |= clSetKernelArg(kernel[1], max_pipe_args + 1, sizeof(int), (void*)&max_pipe_args);
    if ( err != CL_SUCCESS ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &consumer_sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    err = clWaitForEvents(1, &consumer_sync_event);
    if ( err != CL_SUCCESS ){
        print_error( err, " clWaitForEvents failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    if( verify_result( inptr, outptr, num_pipe_elements*sizeof(cl_int))){
        log_error("test_pipe_max_args failed\n");
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        for(j = 0; j < max_pipe_args; j++) {
            clReleaseMemObject(pipes[j]);
        }
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }
    else {
        log_info("test_pipe_max_args passed\n");
    }
    //cleanup
    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    for(j = 0; j < max_pipe_args; j++) {
        clReleaseMemObject(pipes[j]);
    }
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseEvent(producer_sync_event);
    clReleaseEvent(consumer_sync_event);
    clReleaseProgram(program);
    align_free(outptr);
    free(source);

    return 0;
}


int test_pipe_max_packet_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem        pipe;
    cl_mem      buffers[2];
    void        *outptr;
    cl_char        *inptr;
    cl_program  program;
    cl_kernel   kernel[2];
    size_t      global_work_size[3];
    cl_int      err;
    size_t        size;
    int            num_pipe_elements = 1024;
    int         i;
    cl_uint        max_pipe_packet_size;
    char        *source;
    char        str[256];
    int            str_length;
    cl_event    producer_sync_event = NULL;
    cl_event    consumer_sync_event = NULL;
    MTdata      d = init_genrand( gRandomSeed );
    const char*    kernelName[] = {"test_pipe_max_packet_size_write", "test_pipe_max_packet_size_read"};

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_pipe_elements;

    source = (char*)malloc(STRING_LENGTH*sizeof(char));

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_PACKET_SIZE, sizeof(max_pipe_packet_size), (void*)&max_pipe_packet_size, NULL);
    if(err){
        print_error(err, " clGetDeviceInfo failed\n");
        return -1;
    }
    if(max_pipe_packet_size < 1024){
        log_error("The device should support minimum packet size of 1024 bytes");
        return -1;
    }

    if(max_pipe_packet_size > (32*1024*1024/num_pipe_elements))
    {
        max_pipe_packet_size = 32*1024*1024/num_pipe_elements;
    }

    size = max_pipe_packet_size * num_pipe_elements;

    inptr = (cl_char *)align_malloc(size, min_alignment);

    for(i = 0; i < size; i++){
        inptr[i] = (char)genrand_int32(d);
    }

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        free(source);
        print_error(err, " clCreateBuffer failed\n");
        return -1;
    }
    outptr = align_malloc(size, min_alignment);
    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size, outptr, &err);
    if ( err ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        free(source);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, max_pipe_packet_size, num_pipe_elements, NULL, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        free(source);
        clReleaseMemObject(pipe);
        print_error(err, " clCreatePipe failed\n");
        return -1;
    }

    sprintf(str, "typedef struct{\n  char a[%d];\n}TestStruct;\n\n__kernel void test_pipe_max_packet_size_write(__global TestStruct *src, __write_only pipe TestStruct out_pipe)\n{\n", max_pipe_packet_size);
    strcpy(source,str);
    strcat(source, "  int gid = get_global_id(0);\n  reserve_id_t res_id;\n\n");
    sprintf(str, "  res_id = reserve_write_pipe(out_pipe, 1);\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    write_pipe(out_pipe, res_id, 0, &src[gid]);\n    commit_write_pipe(out_pipe, res_id);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_max_packet_size_read(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)\n{\n");
    strcat(source, str);
    strcat(source, "  int gid = get_global_id(0);\n  reserve_id_t res_id;\n\n");
    sprintf(str, "  res_id = reserve_read_pipe(in_pipe, 1);\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    read_pipe(in_pipe, res_id, 0, &dst[gid]);\n    commit_read_pipe(in_pipe, res_id);\n  }\n}\n\n");
    strcat(source, str);

    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH);

    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, (const char**)&source, kernelName[0], "-cl-std=CL2.0");
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        free(source);
        print_error(err, "Error creating program\n");
        return -1;
    }
    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    if( kernel[1] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        free(source);
        print_error(err, "Error creating kernel\n");
        return -1;
    }

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    if ( err != CL_SUCCESS ){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &consumer_sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }

    if( verify_result( inptr, outptr, size)){
        log_error("test_pipe_max_packet_size failed\n");
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        free(source);
        return -1;
    }
    else {
        log_info("test_pipe_max_packet_size passed\n");
    }
    //cleanup
    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(pipe);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseEvent(producer_sync_event);
    clReleaseEvent(consumer_sync_event);
    clReleaseProgram(program);
    align_free(outptr);
    free(source);

    return 0;
}

int test_pipe_max_active_reservations(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem        pipe;
    cl_mem      buffers[2];
    cl_mem        buf_reservations;
    cl_mem        buf_status;
    cl_mem        buf_reserve_id_t_size;
    cl_mem        buf_reserve_id_t_size_aligned;
    cl_int      *inptr;
    void        *outptr;
    int            size, i;
    cl_program  program;
    cl_kernel   kernel[3];
    size_t      global_work_size[3];
    cl_int      err;
    int            status = 0;
    cl_uint        max_active_reservations = 0;
    cl_ulong    max_global_size = 0;
    int            reserve_id_t_size;
    int            temp;
    char        *source;
    char        str[256];
    int            str_length;
    cl_event    sync_event = NULL;
    cl_event    read_event = NULL;
    MTdata      d = init_genrand( gRandomSeed );
    const char*    kernelName[3] = {"test_pipe_max_active_reservations_write", "test_pipe_max_active_reservations_read", "pipe_get_reserve_id_t_size"};

    size_t      min_alignment = get_min_alignment(context);

    source = (char*)malloc(2*STRING_LENGTH*sizeof(char));

    global_work_size[0] = 1;

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS, sizeof(max_active_reservations), (void*)&max_active_reservations, NULL);
    if(err){
        print_error(err, " clGetDeviceInfo failed\n");
        return -1;
    }

    err = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(max_global_size), (void*)&max_global_size, NULL);
    if(err){
        print_error(err, " clGetDeviceInfo failed\n");
        return -1;
    }

    max_active_reservations = (max_active_reservations > max_global_size) ? 1<<16 : max_active_reservations;

    if(max_active_reservations < 1){
        log_error("The device should support minimum active reservations of 1");
        return -1;
    }

    // To get reserve_id_t size
    buf_reserve_id_t_size = clCreateBuffer(context, CL_MEM_HOST_READ_ONLY, sizeof(reserve_id_t_size), NULL, &err);
    if ( err ){
        clReleaseMemObject(buf_reserve_id_t_size);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    sprintf(str, "__kernel void test_pipe_max_active_reservations_write(__global int *src, __write_only pipe int out_pipe, __global char *reserve_id, __global int *reserve_id_t_size_aligned, __global int *status)\n{\n");
    strcpy(source,str);
    sprintf(str, "  __global reserve_id_t *res_id_ptr;\n  int reserve_idx;\n  int commit_idx;\n");
    strcat(source, str);
    sprintf(str, "  for(reserve_idx = 0; reserve_idx < %d; reserve_idx++)\n  {\n", max_active_reservations);
    strcat(source, str);
    sprintf(str, "    res_id_ptr = (__global reserve_id_t*)(reserve_id + reserve_idx*reserve_id_t_size_aligned[0]);\n");
    strcat(source, str);
    sprintf(str, "    *res_id_ptr = reserve_write_pipe(out_pipe, 1);\n");
    strcat(source, str);
    sprintf(str, "    if(is_valid_reserve_id(res_id_ptr[0]))\n    {\n      write_pipe(out_pipe, res_id_ptr[0], 0, &src[reserve_idx]);\n    }\n");
    strcat(source, str);
    sprintf(str, "    else\n    {\n      *status = -1;\n      return;\n    }\n  }\n");
    strcat(source, str);
    sprintf(str, "  for(commit_idx = 0; commit_idx < %d; commit_idx++)\n  {\n", max_active_reservations);
    strcat(source, str);
    sprintf(str, "    res_id_ptr = (__global reserve_id_t*)(reserve_id + commit_idx*reserve_id_t_size_aligned[0]);\n");
    strcat(source, str);
    sprintf(str, "    commit_write_pipe(out_pipe, res_id_ptr[0]);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_max_active_reservations_read(__read_only pipe int in_pipe, __global int *dst, __global char *reserve_id, __global int *reserve_id_t_size_aligned, __global int *status)\n{\n");
    strcat(source, str);
    sprintf(str, "  __global reserve_id_t *res_id_ptr;\n  int reserve_idx;\n  int commit_idx;\n");
    strcat(source, str);
    sprintf(str, "  for(reserve_idx = 0; reserve_idx < %d; reserve_idx++)\n  {\n", max_active_reservations);
    strcat(source, str);
    sprintf(str, "    res_id_ptr = (__global reserve_id_t*)(reserve_id + reserve_idx*reserve_id_t_size_aligned[0]);\n");
    strcat(source, str);
    sprintf(str, "    *res_id_ptr = reserve_read_pipe(in_pipe, 1);\n");
    strcat(source, str);
    sprintf(str, "    if(is_valid_reserve_id(res_id_ptr[0]))\n    {\n      read_pipe(in_pipe, res_id_ptr[0], 0, &dst[reserve_idx]);\n    }\n");
    strcat(source, str);
    sprintf(str, "    else\n    {\n      *status = -1;\n      return;\n    }\n  }\n");
    strcat(source, str);
    sprintf(str, "  for(commit_idx = 0; commit_idx < %d; commit_idx++)\n  {\n", max_active_reservations);
    strcat(source, str);
    sprintf(str, "    res_id_ptr = (__global reserve_id_t*)(reserve_id + commit_idx*reserve_id_t_size_aligned[0]);\n");
    strcat(source, str);
    sprintf(str, "    commit_read_pipe(in_pipe, res_id_ptr[0]);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void pipe_get_reserve_id_t_size(__global int *reserve_id_t_size) \n");
    strcat(source, str);
    sprintf(str, "{\n  *reserve_id_t_size = sizeof(reserve_id_t);\n}\n");
    strcat(source, str);

    str_length = strlen(source);
    assert(str_length <= 2*STRING_LENGTH);

    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, (const char**)&source, kernelName[0], "-cl-std=CL2.0");
    if(err){
        clReleaseMemObject(buf_reserve_id_t_size);
        print_error(err, "Error creating program\n");
        return -1;
    }

    // Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    if( kernel[1] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buf_reserve_id_t_size);
        print_error(err, "Error creating kernel\n");
        return -1;
    }

    // Create size query kernel for reserve_id_t
    kernel[2] = clCreateKernel(program, kernelName[2], &err);
    if( kernel[2] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buf_reserve_id_t_size);
        print_error(err, "Error creating kernel\n");
        return -1;
    }
    err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void*)&buf_reserve_id_t_size);
    if(err){
        clReleaseMemObject(buf_reserve_id_t_size);
        print_error(err, "Error creating program\n");
        return -1;
    }
    //Launch size query kernel for reserve_id_t
    err = clEnqueueNDRangeKernel( queue, kernel[2], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event );
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buf_reserve_id_t_size, true, 0, sizeof(reserve_id_t_size), &reserve_id_t_size, 1, &sync_event, &read_event);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clWaitForEvents(1, &read_event);
    if ( err != CL_SUCCESS ){
        print_error( err, " clWaitForEvents failed" );
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        return -1;
    }

    // Round reserve_id_t_size to the nearest power of 2
    temp = 1;
    while(temp < reserve_id_t_size)
        temp *= 2;
    reserve_id_t_size = temp;

    size = sizeof(cl_int) * max_active_reservations;
    inptr = (cl_int *)align_malloc(size, min_alignment);

    for(i = 0; i < max_active_reservations; i++){
        inptr[i] = (int)genrand_int32(d);
    }

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    if ( err ){
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buffers[0]);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    outptr = align_malloc(size, min_alignment);
    buffers[1] = clCreateBuffer(context, CL_MEM_HOST_READ_ONLY, size, NULL, &err);
    if ( err ){
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        align_free(outptr);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    buf_reserve_id_t_size_aligned = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(reserve_id_t_size), &reserve_id_t_size, &err);
    if ( err ){
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    //For error status
    buf_status = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  sizeof(int), &status, &err);
    if ( err ){
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), max_active_reservations, NULL, &err);
    if(err){
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        print_error(err, " clCreatePipe failed\n");
        return -1;
    }

    // Global buffer to hold all active reservation ids
    buf_reservations = clCreateBuffer(context, CL_MEM_HOST_NO_ACCESS, reserve_id_t_size*max_active_reservations, NULL, &err);
    if ( err != CL_SUCCESS ){
        print_error( err, " clCreateBuffer failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseEvent(read_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void*)&buf_reservations);
    err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), (void*)&buf_reserve_id_t_size_aligned);
    err |= clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void*)&buf_status);
    if ( err != CL_SUCCESS ){
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseProgram(program);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void*)&buf_reservations);
    err |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void*)&buf_reserve_id_t_size_aligned);
    err |= clSetKernelArg(kernel[1], 4, sizeof(cl_mem), (void*)&buf_status);
    if ( err != CL_SUCCESS ){
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseProgram(program);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buf_status, true, 0, sizeof(int), &status, 1, &sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    if(status != 0)
    {
        log_error("test_pipe_max_active_reservations failed\n");
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buf_status, true, 0, sizeof(int), &status, 1, &sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    if(status != 0)
    {
        log_error("test_pipe_max_active_reservations failed\n");
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &sync_event, NULL);
    if ( err != CL_SUCCESS ){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }

    if( verify_result_int( inptr, outptr, max_active_reservations)){
        log_error("test_pipe_max_active_reservations failed\n");
        clReleaseMemObject(buf_status);
        clReleaseMemObject(buf_reserve_id_t_size);
        clReleaseMemObject(buf_reserve_id_t_size_aligned);
        clReleaseMemObject(buf_reservations);
        clReleaseMemObject(pipe);
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free(outptr);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseKernel(kernel[2]);
        clReleaseEvent(sync_event);
        clReleaseProgram(program);
        return -1;
    }
    else {
        log_info("test_pipe_max_active_reservations passed\n");
    }

    //cleanup
    clReleaseMemObject(buf_status);
    clReleaseMemObject(buf_reserve_id_t_size);
    clReleaseMemObject(buf_reserve_id_t_size_aligned);
    clReleaseMemObject(buf_reservations);
    clReleaseMemObject(pipe);
    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    align_free(outptr);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseEvent(sync_event);
    clReleaseProgram(program);
    return 0;
}