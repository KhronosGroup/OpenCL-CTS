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

extern int gWimpyMode;

#ifdef CL_VERSION_2_0

static const char* enqueue_block_first_kernel[] =
{
    NL, "void block_fn(uint num, __global int* res)"
    NL, "{"
    NL, "    size_t tid = get_global_id(0);"
    NL, ""
    NL, "    for(int i = 1 ; i < tid ; i++)"
    NL, "    {"
    NL, "      for(int j = 0 ; j < num ; j++)"
    NL, "        atomic_add(res+tid, 1);"
    NL, "    }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_first_kernel(uint num, __global int* res)"
    NL, "{"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(num, res); };"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(num, 1);"
    NL, ""
    NL, "  int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[0] = -1; return; }"
    NL, ""
    NL, "}"
    NL
};

static const char* enqueue_block_second_kernel[] =
{
    NL, "void block_fn(uint num, __global int* res)"
    NL, "{"
    NL, "    for(int i = 2 ; i < num ; i++)"
    NL, "    {"
    NL, "      res[i] = res[i]/num - (i-1);"
    NL, "    }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_second_kernel(uint num, __global int* res)"
    NL, "{"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(num, res); };"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, ""
    NL, "  int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[0] = -1; return; }"
    NL, ""
    NL, "}"
    NL
};

static int check_kernel_results(cl_int* results, cl_int len)
{
    for(cl_int i = 0; i < len; ++i)
    {
        if(results[i] != 0) return i;
    }
    return -1;
}

/*
    Test checks kernel block execution order in case of two different kernels with enqueue block submitted to one ordered host queue.
*/
int test_host_queue_order(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_int k, err_ret, res = 0;
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

    cl_int status;
    size_t size = 1;
    cl_int result[MAX_GWS] = { 0 };
    cl_uint num = arr_size(result);
    if( gWimpyMode )
    {
        num = MAX(num / 16, 4);
    }

    clMemWrapper res_mem;
    clProgramWrapper program1, program2;
    clKernelWrapper kernel1, kernel2;

    cl_event kernel_event;

    err_ret = create_single_kernel_helper_with_build_options(context, &program1, &kernel1,  arr_size(enqueue_block_first_kernel), enqueue_block_first_kernel, "enqueue_block_first_kernel", "-cl-std=CL2.0");
    if(check_error(err_ret, "Create single kernel failed")) return -1;

    err_ret = create_single_kernel_helper_with_build_options(context, &program2, &kernel2, arr_size(enqueue_block_second_kernel), enqueue_block_second_kernel, "enqueue_block_second_kernel", "-cl-std=CL2.0");
    if(check_error(err_ret, "Create single kernel failed")) return -1;

    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(kernel_results), kernel_results, &err_ret);
    test_error(err_ret, "clCreateBuffer() failed");

    // Enqueue first kernel
    err_ret = clSetKernelArg(kernel1, 0, sizeof(num), &num);
    test_error(err_ret, "clSetKernelArg(0) failed");
    err_ret = clSetKernelArg(kernel1, 1, sizeof(cl_mem), &res_mem);
    test_error(err_ret, "clSetKernelArg(1) failed");

    cl_event event1 = clCreateUserEvent(context, &err_ret);
    if(check_error(err_ret, "Create user event failed")) return -1;

    err_ret = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL, &size, &size, 1, &event1, NULL);
    test_error(err_ret, "clEnqueueNDRangeKernel('enqueue_block_first_kernel') failed");

    // Enqueue second kernel
    err_ret = clSetKernelArg(kernel2, 0, sizeof(num), &num);
    test_error(err_ret, "clSetKernelArg(0) failed");
    err_ret = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &res_mem);
    test_error(err_ret, "clSetKernelArg(1) failed");

    err_ret = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &size, &size, 0, NULL, &kernel_event);
    test_error(err_ret, "clEnqueueNDRangeKernel('enqueue_block_second_kernel') failed");

    //Triger execution of first kernel
    err_ret = clSetUserEventStatus(event1, CL_COMPLETE);
    test_error(err_ret, "clSetUserEventStatus() failed");

    // Collect resulsts
    err_ret = clEnqueueReadBuffer(queue, res_mem, CL_TRUE, 0, sizeof(result), result, 0, NULL, NULL);
    test_error(err_ret, "clEnqueueReadBuffer() failed");

    err_ret = clGetEventInfo(kernel_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, &ret_len);
    test_error(err_ret, "clGetEventInfo() failed");

    if(check_error(status, "Kernel execution status %d", status)) return status;

    if((k = check_kernel_results(result, num)) >= 0 && check_error(-1, "'%s' results validation failed: [%d] returned %d expected 0", "test_host_queue_order", k, result[k])) res = -1;

    clReleaseEvent(kernel_event);
    clReleaseEvent(event1);

    return res;
}

#endif

