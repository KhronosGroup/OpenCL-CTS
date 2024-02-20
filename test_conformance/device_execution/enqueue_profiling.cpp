//
// Copyright (c) 2020 The Khronos Group Inc.
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

static int max_nestingLevel = 10;

static const char* enqueue_multi_level = R"(
    void block_fn(__global int* res, int level)
    {
      queue_t def_q = get_default_queue();
      if(--level < 0) return;
      void (^kernelBlock)(void) = ^{ block_fn(res, level); };
      ndrange_t ndrange = ndrange_1D(1);
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
      if(enq_res != CLK_SUCCESS) { (*res) = -1; return; }
      else if(*res != -1) { (*res)++; }
    }
    kernel void enqueue_multi_level(__global int* res, int level)
    {
      *res = 0;
      block_fn(res, level);
    })";

int test_enqueue_profiling(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    clCommandQueueWrapper host_queue;

    cl_uint maxQueueSize = 0;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                              sizeof(maxQueueSize), &maxQueueSize, 0);
    test_error(err_ret,
               "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");

    cl_queue_properties dev_queue_prop_def[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
            | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE,
        CL_QUEUE_SIZE, maxQueueSize, 0
    };

    dev_queue = clCreateCommandQueueWithProperties(
        context, device, dev_queue_prop_def, &err_ret);
    test_error(err_ret,
               "clCreateCommandQueueWithProperties(CL_QUEUE_ON_DEVICE | "
               "CL_QUEUE_ON_DEVICE_DEFAULT) failed");

    cl_queue_properties host_queue_prop_def[] = { CL_QUEUE_PROPERTIES,
                                                  CL_QUEUE_PROFILING_ENABLE,
                                                  0 };

    host_queue = clCreateCommandQueueWithProperties(
        context, device, host_queue_prop_def, &err_ret);
    test_error(
        err_ret,
        "clCreateCommandQueueWithProperties(CL_QUEUE_PROFILING_ENABLE) failed");

    cl_int status;
    size_t size = 1;
    cl_int result = 0;

    clMemWrapper res_mem;
    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_event kernel_event;

    err_ret = create_single_kernel_helper(context, &program, &kernel, 1,
                                          &enqueue_multi_level,
                                          "enqueue_multi_level");
    if (check_error(err_ret, "Create single kernel failed")) return -1;

    res_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             sizeof(result), &result, &err_ret);
    test_error(err_ret, "clCreateBuffer() failed");

    err_ret = clSetKernelArg(kernel, 0, sizeof(res_mem), &res_mem);
    test_error(err_ret, "clSetKernelArg(0) failed");

    for (int level = 0; level < max_nestingLevel; level++)
    {
        err_ret = clSetKernelArg(kernel, 1, sizeof(level), &level);
        test_error(err_ret, "clSetKernelArg(1) failed");

        err_ret = clEnqueueNDRangeKernel(host_queue, kernel, 1, NULL, &size,
                                         &size, 0, NULL, &kernel_event);
        test_error(err_ret,
                   "clEnqueueNDRangeKernel('enqueue_multi_level') failed");

        err_ret = clEnqueueReadBuffer(host_queue, res_mem, CL_TRUE, 0,
                                      sizeof(result), &result, 0, NULL, NULL);
        test_error(err_ret, "clEnqueueReadBuffer() failed");

        if (result != level)
        {
            log_error("Kernel execution should return the maximum nesting "
                      " level (got %d instead of %d)",
                      result, level);
            return -1;
        }

        err_ret =
            clGetEventInfo(kernel_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(status), &status, NULL);
        test_error(err_ret, "clGetEventInfo() failed");

        if (check_error(status, "Kernel execution status %d", status))
            return status;

        cl_ulong end;
        err_ret = clGetEventProfilingInfo(
            kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        test_error(err_ret, "clGetEventProfilingInfo() failed");

        cl_ulong complete;
        err_ret =
            clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_COMPLETE,
                                    sizeof(complete), &complete, NULL);
        test_error(err_ret, "clGetEventProfilingInfo() failed");

        if (end > complete)
        {
            log_error(
                "Profiling END should be smaller than or equal to COMPLETE for "
                "kernels that use the on-device queue");
            return -1;
        }

        log_info("Profiling info for '%s' kernel is OK for level %d.\n",
                 "enqueue_multi_level", level);

        clReleaseEvent(kernel_event);
    }

    return res;
}
