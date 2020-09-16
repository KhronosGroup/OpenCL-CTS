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
#include <stdio.h>
#include <string.h>
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#include <vector>

#include "procs.h"
#include "utils.h"
#include <time.h>


#ifdef CL_VERSION_2_0
extern int gWimpyMode;
static const char* multi_queue_simple_block1[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void multi_queue_simple_block1(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "  res[tid] = -1;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* multi_queue_simple_block2[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void multi_queue_simple_block2(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "  res[tid] = -1;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* multi_queue_simple_block3[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void multi_queue_simple_block3(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "  res[tid] = -1;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* multi_queue_simple_block4[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void multi_queue_simple_block4(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "  res[tid] = -1;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const kernel_src sources_multi_queue_block[] =
{
    KERNEL(multi_queue_simple_block1),
    KERNEL(multi_queue_simple_block2),
    KERNEL(multi_queue_simple_block3),
    KERNEL(multi_queue_simple_block4),
};
static const size_t num_kernels_multi_queue_block = arr_size(sources_multi_queue_block);


int test_host_multi_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint i;
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    cl_int kernel_results[MAX_GWS] = {0};

    size_t ret_len;
    cl_uint max_queues = 1;
    cl_uint maxQueueSize = 0;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, sizeof(maxQueueSize), &maxQueueSize, 0);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");

    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof(max_queues), &max_queues, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_ON_DEVICE_QUEUES) failed");

    size_t max_local_size = 1;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed");

    cl_queue_properties queue_prop_def[] =
    {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE|CL_QUEUE_ON_DEVICE_DEFAULT,
        CL_QUEUE_SIZE, maxQueueSize,
        0
    };

    dev_queue = clCreateCommandQueueWithProperties(context, device, queue_prop_def, &err_ret);
    test_error(err_ret, "clCreateCommandQueueWithProperties(CL_QUEUE_DEVICE|CL_QUEUE_DEFAULT) failed");

    cl_uint n = num_kernels_multi_queue_block; // Number of host queues
    std::vector<clCommandQueueWrapper> queues(n);
    std::vector<cl_command_queue> q(n);
    std::vector<clProgramWrapper> program(n);
    std::vector<clKernelWrapper> kernel(n);
    std::vector<clMemWrapper> mem(n);
    std::vector<clEventWrapper> event(n);

    for(i = 0; i < n; ++i)
    {
        queues[i] = clCreateCommandQueueWithProperties(context, device, NULL, &err_ret);
        if(check_error(err_ret, "clCreateCommandQueueWithProperties() failed")) { res = -1; break; }
        q[i] = queues[i];
    }

    if(err_ret == CL_SUCCESS)
    {
        for(i = 0; i < n; ++i)
        {
            size_t global = MAX_GWS;
            if(gWimpyMode)
            {
                global = 16;
            }

            err_ret |= create_single_kernel_helper_with_build_options(context, &program[i], &kernel[i], sources_multi_queue_block[i].num_lines, sources_multi_queue_block[i].lines, sources_multi_queue_block[i].kernel_name, "-cl-std=CL2.0");
            if(check_error(err_ret, "Create single kernel failed")) { res = -1; break; }

            mem[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(kernel_results), kernel_results, &err_ret);
            if(check_error(err_ret, "clCreateBuffer() failed")) { res = -1; break; }

            err_ret |= clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &mem[i]);
            if(check_error(err_ret, "clSetKernelArg(0) failed")) { res = -1; break; }

            err_ret |= clEnqueueNDRangeKernel(q[i], kernel[i], 1, NULL, &global, 0, 0, NULL, &event[i]);
            if(check_error(err_ret, "clEnqueueNDRangeKernel() failed")) { res = -1; break; }
        }
    }

    if(err_ret == CL_SUCCESS)
    {
        for(i = 0; i < n; ++i)
        {
            cl_int status;
            err_ret = clEnqueueReadBuffer(q[i], mem[i], CL_TRUE, 0, sizeof(kernel_results), kernel_results, 0, NULL, NULL);
            if(check_error(err_ret, "clEnqueueReadBuffer() failed")) { res = -1; break; }

            err_ret = clGetEventInfo(event[i], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &ret_len);
            if(check_error(err_ret, "clGetEventInfo() failed")) { res = -1; break; }

#if CL_COMPLETE != CL_SUCCESS
#error Fix me!
#endif
            // This hack is possible because both CL_COMPLETE and CL_SUCCESS defined as 0x00
            if(check_error(status, "Kernel execution status %d", status)) { err_ret = status; res = -1; break; }
            else if(kernel_results[0] != 0 && check_error(-1, "'%s' kernel results validation failed = %d", sources_multi_queue_block[i].kernel_name, kernel_results[0])) { res = -1; break; }
        }
    }

    return res;
}




#endif

