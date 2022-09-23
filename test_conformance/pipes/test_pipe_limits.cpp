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

#include <assert.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "procs.h"
#include "harness/errorHelpers.h"

#define STRING_LENGTH  1024

void createKernelSourceCode(std::stringstream &stream, int num_pipes)
{
    int i;

    stream << "__kernel void test_multiple_pipe_write(__global int *src, ";
    for (i = 0; i < num_pipes; i++)
    {
        stream << "__write_only pipe int pipe" << i << ", ";
    }
    stream << R"(int num_pipes )
    {
          int gid = get_global_id(0);
          reserve_id_t res_id;


          if(gid < (get_global_size(0))/num_pipes)
          {
                 res_id = reserve_write_pipe(pipe0, 1);
                 if(is_valid_reserve_id(res_id))
                 {
                     write_pipe(pipe0, res_id, 0, &src[gid]);
                     commit_write_pipe(pipe0, res_id);
                 }
          })";

    for (i = 1; i < num_pipes; i++)
    {
        // clang-format off
        stream << R"(
          else if(gid < ()" << (i + 1) << R"(*get_global_size(0))/num_pipes)
          {
                 res_id = reserve_write_pipe(pipe)" << i << R"(, 1);
                 if(is_valid_reserve_id(res_id))
                 {
                     write_pipe(pipe)" << i << R"(, res_id, 0, &src[gid]);
                     commit_write_pipe(pipe)" << i << R"(, res_id);
                  }
          }
          )";
        // clang-format on
    }
    stream << R"(
    }

    __kernel void test_multiple_pipe_read(__global int *dst, )";

    for (i = 0; i < num_pipes; i++)
    {
        stream << "__read_only pipe int pipe" << i << ", ";
    }
    stream << R"(int num_pipes )
    {
            int gid = get_global_id(0);
            reserve_id_t res_id;


            if(gid < (get_global_size(0))/num_pipes)
            {
                res_id = reserve_read_pipe(pipe0, 1);
                if(is_valid_reserve_id(res_id))
                {
                    read_pipe(pipe0, res_id, 0, &dst[gid]);
                    commit_read_pipe(pipe0, res_id);
                }
            })";

    for (i = 1; i < num_pipes; i++)
    {
        // clang-format off
        stream << R"(
            else if(gid < ()"    << (i + 1) << R"(*get_global_size(0))/num_pipes)
            {
                res_id = reserve_read_pipe(pipe)" << i << R"(, 1);
                if(is_valid_reserve_id(res_id))
                {
                    read_pipe(pipe)" << i << R"(, res_id, 0, &dst[gid]);
                    commit_read_pipe(pipe)" << i << R"(, res_id);
                }
            })";
        // clang-format on
    }
    stream << "}";
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

    clMemWrapper pipes[1024];
    clMemWrapper buffers[2];
    void *outptr;
    cl_int *inptr;
    clProgramWrapper program;
    clKernelWrapper kernel[2];
    size_t global_work_size[3];
    cl_int err;
    cl_int size;
    int num_pipe_elements = 1024;
    int i;
    int max_pipe_args;
    std::stringstream source;
    clEventWrapper producer_sync_event = NULL;
    clEventWrapper consumer_sync_event = NULL;
    BufferOwningPtr<cl_int> BufferInPtr;
    BufferOwningPtr<cl_int> BufferOutPtr;

    MTdataHolder d(gRandomSeed);
    const char *kernelName[] = { "test_multiple_pipe_write",
                                 "test_multiple_pipe_read" };

    size_t min_alignment = get_min_alignment(context);

    err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_PIPE_ARGS,
                          sizeof(max_pipe_args), (void *)&max_pipe_args, NULL);
    if (err)
    {
        print_error(err, " clGetDeviceInfo failed\n");
        return -1;
    }
    if(max_pipe_args < 16){
        log_error("The device should support minimum 16 pipe objects that could be passed as arguments to the kernel");
        return -1;
    }

    global_work_size[0] = (cl_uint)num_pipe_elements * max_pipe_args;
    size = sizeof(int) * num_pipe_elements * max_pipe_args;

    inptr = (cl_int *)align_malloc(size, min_alignment);

    for(i = 0; i < num_pipe_elements * max_pipe_args; i++){
        inptr[i] = (int)genrand_int32(d);
    }
    BufferInPtr.reset(inptr, nullptr, 0, size, true);

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    outptr = align_malloc(size, min_alignment);
    BufferOutPtr.reset(outptr, nullptr, 0, size, true);
    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size, outptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    for(i = 0; i < max_pipe_args; i++){
        pipes[i] = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), num_pipe_elements, NULL, &err);
        test_error_ret(err, " clCreatePipe failed", -1);
    }

    createKernelSourceCode(source, max_pipe_args);

    std::string kernel_source = source.str();
    const char *sources[] = { kernel_source.c_str() };

    // Create producer kernel
    err = create_single_kernel_helper(context, &program, &kernel[0], 1, sources,
                                      kernelName[0]);
    test_error_ret(err, " Error creating program", -1);

    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    test_error_ret(err, " Error creating kernel", -1);

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
    test_error_ret(err, " clSetKernelArg failed", -1);

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &consumer_sync_event, NULL);
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clWaitForEvents(1, &consumer_sync_event);
    test_error_ret(err, " clWaitForEvents failed", -1);

    if( verify_result( inptr, outptr, num_pipe_elements*sizeof(cl_int))){
        log_error("test_pipe_max_args failed\n");
    }
    else {
        log_info("test_pipe_max_args passed\n");
    }

    return 0;
}


int test_pipe_max_packet_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clMemWrapper pipe;
    clMemWrapper buffers[2];
    void *outptr;
    cl_char *inptr;
    clProgramWrapper program;
    clKernelWrapper kernel[2];
    size_t global_work_size[3];
    cl_int err;
    size_t size;
    int num_pipe_elements = 1024;
    int i;
    cl_uint max_pipe_packet_size;
    clEventWrapper producer_sync_event = NULL;
    clEventWrapper consumer_sync_event = NULL;
    BufferOwningPtr<cl_int> BufferInPtr;
    BufferOwningPtr<cl_int> BufferOutPtr;
    MTdataHolder d(gRandomSeed);
    const char *kernelName[] = { "test_pipe_max_packet_size_write",
                                 "test_pipe_max_packet_size_read" };

    size_t min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_pipe_elements;

    std::stringstream source;

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_PACKET_SIZE,
                          sizeof(max_pipe_packet_size),
                          (void *)&max_pipe_packet_size, NULL);
    test_error_ret(err, " clCreatePipe failed", -1);

    if (max_pipe_packet_size < 1024)
    {
        log_error(
            "The device should support minimum packet size of 1024 bytes");
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
    BufferInPtr.reset(inptr, nullptr, 0, size, true);

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    outptr = align_malloc(size, min_alignment);
    BufferOutPtr.reset(outptr, nullptr, 0, size, true);

    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size, outptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, max_pipe_packet_size, num_pipe_elements, NULL, &err);
    test_error_ret(err, " clCreatePipe failed", -1);

    // clang-format off
    source << R"(
        typedef struct{
            char a[)" << max_pipe_packet_size << R"(];
        }TestStruct;

        __kernel void test_pipe_max_packet_size_write(__global TestStruct *src, __write_only pipe TestStruct out_pipe)
        {
            int gid = get_global_id(0);
            reserve_id_t res_id;

            res_id = reserve_write_pipe(out_pipe, 1);
            if(is_valid_reserve_id(res_id))
            {
                write_pipe(out_pipe, res_id, 0, &src[gid]);
                commit_write_pipe(out_pipe, res_id);
            }
        }

        __kernel void test_pipe_max_packet_size_read(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)
        {
            int gid = get_global_id(0);
            reserve_id_t res_id;

            res_id = reserve_read_pipe(in_pipe, 1);
            if(is_valid_reserve_id(res_id))
            {
                read_pipe(in_pipe, res_id, 0, &dst[gid]);
                commit_read_pipe(in_pipe, res_id);
            }
        }
        )";
    // clang-format on

    std::string kernel_source = source.str();
    const char *sources[] = { kernel_source.c_str() };

    // Create producer kernel
    err = create_single_kernel_helper(context, &program, &kernel[0], 1, sources,
                                      kernelName[0]);
    test_error_ret(err, " Error creating program", -1);

    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    test_error_ret(err, " Error creating kernel", -1);

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    test_error_ret(err, " clSetKernelArg failed", -1);

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &consumer_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if( verify_result( inptr, outptr, size)){
        log_error("test_pipe_max_packet_size failed\n");
    }
    else {
        log_info("test_pipe_max_packet_size passed\n");
    }

    return 0;
}

int test_pipe_max_active_reservations(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clMemWrapper pipe;
    clMemWrapper buffers[2];
    clMemWrapper buf_reservations;
    clMemWrapper buf_status;
    clMemWrapper buf_reserve_id_t_size;
    clMemWrapper buf_reserve_id_t_size_aligned;
    cl_int *inptr;
    void *outptr;
    int size, i;
    clProgramWrapper program;
    clKernelWrapper kernel[3];
    size_t global_work_size[3];
    cl_int err;
    int status = 0;
    cl_uint max_active_reservations = 0;
    cl_ulong max_global_size = 0;
    int reserve_id_t_size;
    int temp;
    clEventWrapper sync_event = NULL;
    clEventWrapper read_event = NULL;
    BufferOwningPtr<cl_int> BufferInPtr;
    BufferOwningPtr<cl_int> BufferOutPtr;
    MTdataHolder d(gRandomSeed);
    const char *kernelName[3] = { "test_pipe_max_active_reservations_write",
                                  "test_pipe_max_active_reservations_read",
                                  "pipe_get_reserve_id_t_size" };

    size_t min_alignment = get_min_alignment(context);

    std::stringstream source;

    global_work_size[0] = 1;

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS,
                          sizeof(max_active_reservations),
                          (void *)&max_active_reservations, NULL);
    test_error_ret(err, " clGetDeviceInfo failed", -1);

    err = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(max_global_size), (void *)&max_global_size,
                          NULL);
    test_error_ret(err, " clGetDeviceInfo failed", -1);

    max_active_reservations = (max_active_reservations > max_global_size)
        ? 1 << 16
        : max_active_reservations;

    if (max_active_reservations < 1)
    {
        log_error("The device should support minimum active reservations of 1");
        return -1;
    }

    // To get reserve_id_t size
    buf_reserve_id_t_size = clCreateBuffer(context, CL_MEM_HOST_READ_ONLY, sizeof(reserve_id_t_size), NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    // clang-format off
    source << R"(
        __kernel void test_pipe_max_active_reservations_write(__global int *src, __write_only pipe int out_pipe, __global char *reserve_id, __global int *reserve_id_t_size_aligned, __global int *status)
        {
            __global reserve_id_t *res_id_ptr;
            int reserve_idx;
            int commit_idx;

            for(reserve_idx = 0; reserve_idx < )" << max_active_reservations << R"(; reserve_idx++)
            {
                res_id_ptr = (__global reserve_id_t*)(reserve_id + reserve_idx*reserve_id_t_size_aligned[0]);
                *res_id_ptr = reserve_write_pipe(out_pipe, 1);
                if(is_valid_reserve_id(res_id_ptr[0]))
                {
                    write_pipe(out_pipe, res_id_ptr[0], 0, &src[reserve_idx]);
                }
                else
                {
                    *status = -1;
                    return;
                }
            }

            for(commit_idx = 0; commit_idx < )" << max_active_reservations << R"(; commit_idx++)
            {
                res_id_ptr = (__global reserve_id_t*)(reserve_id + commit_idx*reserve_id_t_size_aligned[0]);
                commit_write_pipe(out_pipe, res_id_ptr[0]);
            }
        }

        __kernel void test_pipe_max_active_reservations_read(__read_only pipe int in_pipe, __global int *dst, __global char *reserve_id, __global int *reserve_id_t_size_aligned, __global int *status)
        {
            __global reserve_id_t *res_id_ptr;
            int reserve_idx;
            int commit_idx;

            for(reserve_idx = 0; reserve_idx < )" << max_active_reservations << R"(; reserve_idx++)
            {
                res_id_ptr = (__global reserve_id_t*)(reserve_id + reserve_idx*reserve_id_t_size_aligned[0]);
                *res_id_ptr = reserve_read_pipe(in_pipe, 1);

                if(is_valid_reserve_id(res_id_ptr[0]))
                {
                    read_pipe(in_pipe, res_id_ptr[0], 0, &dst[reserve_idx]);
                }
                else
                {
                    *status = -1;
                    return;
                }
            }

            for(commit_idx = 0; commit_idx < )" << max_active_reservations << R"(; commit_idx++)
            {
                res_id_ptr = (__global reserve_id_t*)(reserve_id + commit_idx*reserve_id_t_size_aligned[0]);
                commit_read_pipe(in_pipe, res_id_ptr[0]);
            }
        }

        __kernel void pipe_get_reserve_id_t_size(__global int *reserve_id_t_size)
        {
            *reserve_id_t_size = sizeof(reserve_id_t);
        }
        )";
    // clang-format on

    std::string kernel_source = source.str();
    const char *sources[] = { kernel_source.c_str() };

    // Create producer kernel
    err = create_single_kernel_helper(context, &program, &kernel[0], 1, sources,
                                      kernelName[0]);
    test_error_ret(err, " Error creating program", -1);

    // Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    test_error_ret(err, " Error creating kernel", -1);

    // Create size query kernel for reserve_id_t
    kernel[2] = clCreateKernel(program, kernelName[2], &err);
    test_error_ret(err, " Error creating kernel", -1);

    err = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void*)&buf_reserve_id_t_size);
    test_error_ret(err, " clSetKernelArg failed", -1);

    //Launch size query kernel for reserve_id_t
    err = clEnqueueNDRangeKernel( queue, kernel[2], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buf_reserve_id_t_size, true, 0, sizeof(reserve_id_t_size), &reserve_id_t_size, 1, &sync_event, &read_event);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    err = clWaitForEvents(1, &read_event);
    test_error_ret(err, " clWaitForEvents failed", -1);

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
    BufferInPtr.reset(inptr, nullptr, 0, size, true);

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    outptr = align_malloc(size, min_alignment);
    BufferOutPtr.reset(outptr, nullptr, 0, size, true);

    buffers[1] = clCreateBuffer(context, CL_MEM_HOST_READ_ONLY, size, NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    buf_reserve_id_t_size_aligned = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeof(reserve_id_t_size), &reserve_id_t_size, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    //For error status
    buf_status = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  sizeof(int), &status, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), max_active_reservations, NULL, &err);
    test_error_ret(err, " clCreatePipe failed", -1);

    // Global buffer to hold all active reservation ids
    buf_reservations = clCreateBuffer(context, CL_MEM_HOST_NO_ACCESS, reserve_id_t_size*max_active_reservations, NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void*)&buf_reservations);
    err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), (void*)&buf_reserve_id_t_size_aligned);
    err |= clSetKernelArg(kernel[0], 4, sizeof(cl_mem), (void*)&buf_status);
    test_error_ret(err, " clSetKernelArg failed", -1);

    err = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void*)&buf_reservations);
    err |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void*)&buf_reserve_id_t_size_aligned);
    err |= clSetKernelArg(kernel[1], 4, sizeof(cl_mem), (void*)&buf_status);
    test_error_ret(err, " clSetKernelArg failed", -1);

    clReleaseEvent(sync_event);

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event);
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buf_status, true, 0, sizeof(int), &status, 1, &sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if(status != 0)
    {
        log_error("test_pipe_max_active_reservations failed\n");
        return -1;
    }

    clReleaseEvent(sync_event);
    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, global_work_size, NULL, 0, NULL, &sync_event);
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buf_status, true, 0, sizeof(int), &status, 1, &sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if(status != 0)
    {
        log_error("test_pipe_max_active_reservations failed\n");
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if( verify_result_int( inptr, outptr, max_active_reservations)){
        log_error("test_pipe_max_active_reservations failed\n");
        return -1;
    }
    else {
        log_info("test_pipe_max_active_reservations passed\n");
    }

    return 0;
}
