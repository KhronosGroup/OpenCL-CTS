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
#include "harness/errorHelpers.h"

#define TEST_PRIME_INT        ((1<<16)+1)

const char* pipe_query_functions_kernel_code = {
    "__kernel void test_pipe_write(__global int *src, __write_only pipe int out_pipe)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id;\n"
    "    res_id = reserve_write_pipe(out_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        write_pipe(out_pipe, res_id, 0, &src[gid]);\n"
    "        commit_write_pipe(out_pipe, res_id);\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_query_functions(__write_only pipe int out_pipe, __global int *num_packets, __global int *max_packets)\n"
    "{\n"
    "    *max_packets = get_pipe_max_packets(out_pipe);\n"
    "    *num_packets = get_pipe_num_packets(out_pipe);\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_read(__read_only pipe int in_pipe, __global int *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id;\n"
    "    res_id = reserve_read_pipe(in_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        read_pipe(in_pipe, res_id, 0, &dst[gid]);\n"
    "        commit_read_pipe(in_pipe, res_id);\n"
    "    }\n"
    "}\n" };

static int verify_result(void *ptr1, void *ptr2, int n)
{
    int     i, sum_output = 0;
    cl_int    *outptr1 = (int *)ptr1;
    cl_int    *outptr2 = (int *)ptr2;
    int        cmp_val = ((n*3)/2) * TEST_PRIME_INT;

    for(i = 0; i < n/2; i++)
    {
        sum_output += outptr1[i];
    }
    for(i = 0; i < n; i++)
    {
        sum_output += outptr2[i];
    }
    if(sum_output != cmp_val){
        return -1;
    }
    return 0;
}

int test_pipe_query_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clMemWrapper pipe;
    clMemWrapper buffers[4];
    void *outptr1;
    void *outptr2;
    cl_int *inptr;
    clProgramWrapper program;
    clKernelWrapper kernel[3];
    size_t global_work_size[3];
    size_t half_global_work_size[3];
    size_t global_work_size_pipe_query[3];
    cl_int pipe_max_packets, pipe_num_packets;
    cl_int err;
    cl_int size;
    cl_int i;
    clEventWrapper producer_sync_event = NULL;
    clEventWrapper consumer_sync_event = NULL;
    clEventWrapper pipe_query_sync_event = NULL;
    clEventWrapper pipe_read_sync_event = NULL;
    BufferOwningPtr<cl_int> BufferInPtr;
    BufferOwningPtr<cl_int> BufferOutPtr1;
    BufferOwningPtr<cl_int> BufferOutPtr2;
    MTdataHolder d(gRandomSeed);
    const char *kernelName[] = { "test_pipe_write", "test_pipe_read",
                                 "test_pipe_query_functions" };

    size_t min_alignment = get_min_alignment(context);

    size = sizeof(int) * num_elements;
    global_work_size[0] = (cl_uint)num_elements;
    half_global_work_size[0] = (cl_uint)(num_elements / 2);
    global_work_size_pipe_query[0] = 1;

    inptr = (int *)align_malloc(size, min_alignment);

    for (i = 0; i < num_elements; i++)
    {
        inptr[i] = TEST_PRIME_INT;
    }
    BufferInPtr.reset(inptr, nullptr, 0, size, true);

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    outptr1 = align_malloc(size/2, min_alignment);
    outptr2 = align_malloc(size, min_alignment);
    BufferOutPtr1.reset(outptr1, nullptr, 0, size, true);
    BufferOutPtr2.reset(outptr2, nullptr, 0, size, true);

    buffers[1] = clCreateBuffer(context, CL_MEM_HOST_READ_ONLY,  size, NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    buffers[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    buffers[3] = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), num_elements, NULL, &err);
    test_error_ret(err, " clCreatePipe failed", -1);

    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, (const char**)&pipe_query_functions_kernel_code, kernelName[0], "-cl-std=CL2.0");
    test_error_ret(err, " Error creating program", -1);

    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    test_error_ret(err, " Error creating kernel", -1);

    //Create pipe query functions kernel
    kernel[2] = clCreateKernel(program, kernelName[2], &err);
    test_error_ret(err, " Error creating kernel", -1);

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    err |= clSetKernelArg(kernel[2], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), (void*)&buffers[2]);
    err |= clSetKernelArg(kernel[2], 2, sizeof(cl_mem), (void*)&buffers[3]);
    test_error_ret(err, " clSetKernelArg failed", -1);

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    // Launch Pipe query kernel
    err = clEnqueueNDRangeKernel( queue, kernel[2], 1, NULL, global_work_size_pipe_query, NULL, 1, &producer_sync_event, &pipe_query_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(cl_int), &pipe_num_packets, 1, &pipe_query_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[3], true, 0, sizeof(cl_int), &pipe_max_packets, 1, &pipe_query_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if(pipe_num_packets != num_elements || pipe_max_packets != num_elements)
    {
        log_error("test_pipe_query_functions failed\n");
        return -1;
    }

    // Launch Consumer kernel with half the previous global size
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, half_global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size / 2, outptr1, 1, &consumer_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    // We will reuse this variable so release the previous referred event.
    clReleaseEvent(pipe_query_sync_event);

    // Launch Pipe query kernel
    err = clEnqueueNDRangeKernel( queue, kernel[2], 1, NULL, global_work_size_pipe_query, NULL, 1, &consumer_sync_event, &pipe_query_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(cl_int), &pipe_num_packets, 1, &pipe_query_sync_event, &pipe_read_sync_event);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    // After consumer kernel consumes num_elements/2 from the pipe,
    // there are (num_elements - num_elements/2) remaining package in the pipe.
    if(pipe_num_packets != (num_elements - num_elements/2))
    {
        log_error("test_pipe_query_functions failed\n");
        return -1;
    }

    // We will reuse this variable so release the previous referred event.
    clReleaseEvent(producer_sync_event);

    // Launch Producer kernel to fill the pipe again
    global_work_size[0] = pipe_num_packets;
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 1, &pipe_read_sync_event, &producer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    // We will reuse this variable so release the previous referred event.
    clReleaseEvent(pipe_query_sync_event);
    // Launch Pipe query kernel
    err = clEnqueueNDRangeKernel( queue, kernel[2], 1, NULL, global_work_size_pipe_query, NULL, 1, &producer_sync_event, &pipe_query_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    // We will reuse this variable so release the previous referred event.
    clReleaseEvent(pipe_read_sync_event);
    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, sizeof(cl_int), &pipe_num_packets, 1, &pipe_query_sync_event, &pipe_read_sync_event);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if(pipe_num_packets != num_elements)
    {
        log_error("test_pipe_query_functions failed\n");
        return -1;
    }

    // We will reuse this variable so release the previous referred event.
    clReleaseEvent(consumer_sync_event);

    // Launch Consumer kernel to consume all packets from pipe
    global_work_size[0] = pipe_num_packets;
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &pipe_read_sync_event, &consumer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr2, 1, &consumer_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if( verify_result(outptr1, outptr2, num_elements )){
        log_error("test_pipe_query_functions failed\n");
        return -1;
    }
    else {
        log_info("test_pipe_query_functions passed\n");
    }
    return 0;
}

