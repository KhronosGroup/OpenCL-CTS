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
#define TEST_INT_VALUE 100

const char* pipe_subgroups_kernel_code = {
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "__kernel void test_pipe_subgroups_divergence_write(__global int *src, __write_only pipe int out_pipe, __global int *active_work_item_buffer)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    if(get_sub_group_id() % 2 == 0)\n"
    "    {\n"
    "        active_work_item_buffer[gid] = 1;\n"
    "        res_id = sub_group_reserve_write_pipe(out_pipe, get_sub_group_size());\n"
    "        if(is_valid_reserve_id(res_id))\n"
    "        {\n"
    "            write_pipe(out_pipe, res_id, get_sub_group_local_id(), &src[gid]);\n"
    "            sub_group_commit_write_pipe(out_pipe, res_id);\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_subgroups_divergence_read(__read_only pipe int in_pipe, __global int *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    if(get_sub_group_id() % 2 == 0)\n"
    "    {\n"
    "        res_id = sub_group_reserve_read_pipe(in_pipe, get_sub_group_size());\n"
    "        if(is_valid_reserve_id(res_id))\n"
    "        {\n"
    "            read_pipe(in_pipe, res_id, get_sub_group_local_id(), &dst[gid]);\n"
    "            sub_group_commit_read_pipe(in_pipe, res_id);\n"
    "        }\n"
    "    }\n"
    "}\n"
};

static int verify_result(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_int    *inptr = (cl_int *)ptr1;
    cl_int    *outptr = (cl_int *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
    }
    sum_input *= TEST_INT_VALUE;
    for(i = 0; i < n; i++)
    {
        if(outptr[i] == TEST_INT_VALUE){
            sum_output += outptr[i];
        }
    }

    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

int test_pipe_subgroups_divergence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    clMemWrapper pipe;
    clMemWrapper buffers[3];
    cl_int *outptr;
    cl_int *inptr;
    cl_int *active_work_item_buffer;
    clProgramWrapper program;
    clKernelWrapper kernel[2];
    size_t global_work_size[3];
    size_t local_work_size[3];
    cl_int err;
    cl_int size;
    int i;
    size_t subgroup_count;
    clEventWrapper producer_sync_event = NULL;
    clEventWrapper consumer_sync_event = NULL;
    BufferOwningPtr<cl_int> BufferInPtr;
    BufferOwningPtr<cl_int> BufferOutPtr;
    const char *kernelName[] = { "test_pipe_subgroups_divergence_write",
                                 "test_pipe_subgroups_divergence_read" };

    size_t min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    if (!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platform. Skipping "
                 "test.\n");
        return CL_SUCCESS;
    }

    size = sizeof(int) * num_elements;
    inptr = (cl_int *)align_malloc(size, min_alignment);
    outptr = (cl_int *)align_malloc(size, min_alignment);
    active_work_item_buffer = (cl_int *)align_malloc(size, min_alignment);

    for(i = 0; i < num_elements; i++){
        inptr[i] = TEST_INT_VALUE;
        outptr[i] = 0;
        active_work_item_buffer[i] = 0;
    }
    BufferInPtr.reset(inptr, nullptr, 0, size, true);
    BufferOutPtr.reset(outptr, nullptr, 0, size, true);

    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size, inptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    buffers[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  size, outptr, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    buffers[2] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,  size, active_work_item_buffer, &err);
    test_error_ret(err, " clCreateBuffer failed", -1);

    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, sizeof(int), num_elements, NULL, &err);
    test_error_ret(err, " clCreatePipe failed", -1);

    // Create producer kernel
    err = create_single_kernel_helper(
        context, &program, &kernel[0], 1,
        (const char **)&pipe_subgroups_kernel_code, kernelName[0]);
    test_error_ret(err, " Error creating program", -1);

    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    test_error_ret(err, " Error creating kernel", -1);

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), (void*)&buffers[2]);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    test_error_ret(err, " clSetKernelArg failed", -1);

    err = get_max_common_work_group_size( context, kernel[0], global_work_size[0], &local_work_size[0] );
    test_error_ret(err, " Unable to get work group size to use", -1);

    cl_platform_id platform;
    err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform),
                          &platform, NULL);
    test_error_ret(err, " clGetDeviceInfo failed", -1);

    clGetKernelSubGroupInfoKHR_fn clGetKernelSubGroupInfoKHR =
        (clGetKernelSubGroupInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(
            platform, "clGetKernelSubGroupInfoKHR");

    err = clGetKernelSubGroupInfoKHR(kernel[0], deviceID, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR, sizeof(local_work_size[0]), &local_work_size[0], sizeof(subgroup_count), &subgroup_count, NULL);
    test_error_ret(err, " clGetKernelSubGroupInfoKHR failed", -1);
    if(subgroup_count <= 1)
    {
        log_info("Only 1 subgroup per workgroup for the kernel. Hence no divergence among subgroups possible. Skipping test.\n");
        return CL_SUCCESS;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, local_work_size, 0, NULL, &producer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[2], true, 0, size, active_work_item_buffer, 1, &producer_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, local_work_size, 1, &producer_sync_event, &consumer_sync_event );
    test_error_ret(err, " clEnqueueNDRangeKernel failed", -1);

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size, outptr, 1, &consumer_sync_event, NULL);
    test_error_ret(err, " clEnqueueReadBuffer failed", -1);

    if( verify_result( active_work_item_buffer, outptr, num_elements)){
        log_error("test_pipe_subgroups_divergence failed\n");
        return -1;
    }
    else {
        log_info("test_pipe_subgroups_divergence passed\n");
    }

    return 0;
}
