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
static const char enqueue_block_multi_queue[] =
    NL "#define BLOCK_COMPLETED 0"
    NL "#define BLOCK_SUBMITTED 1"
    NL ""
    NL "kernel void enqueue_block_multi_queue(__global int* res, __global int* buff %s)"
    NL "{"
    NL "  uint i, n = %d;"
    NL "  clk_event_t block_evt[%d];"
    NL "  queue_t q[] = { %s };"
    NL "  queue_t *queue = q;"
    NL ""
    NL "  clk_event_t user_evt = create_user_event();"
    NL "  queue_t def_q = get_default_queue();"
    NL "  size_t tid = get_global_id(0);"
    NL "  res[tid] = -1;"
    NL "  __global int* b = buff + tid*n;"
    NL "  for(i=0; i<n; ++i) b[i] = -1;"
    NL ""
    NL "  ndrange_t ndrange = ndrange_1D(1);"
    NL "  for(i = 0; i < n; ++i)"
    NL "  {"
    NL "    b[i] = BLOCK_SUBMITTED;"
    NL "    int enq_res = enqueue_kernel(queue[i], CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt[i], "
    NL "    ^{"
    NL "       b[i] = BLOCK_COMPLETED;"
    NL "     });"
    NL "    if(enq_res != CLK_SUCCESS) { res[tid] = -2; return; }"
    NL "  }"
    NL ""
    NL "  // check blocks are not started"
    NL "  for(i = 0; i < n; ++i)"
    NL "  {"
    NL "    if(b[i] != BLOCK_SUBMITTED) { res[tid] = -5; return; }"
    NL "  }"
    NL ""
    NL "  res[tid] = BLOCK_SUBMITTED;"
    NL "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, n, block_evt, NULL, "
    NL "  ^{"
    NL "     uint k;"
    NL "     // check blocks are finished"
    NL "     for(k = 0; k < n; ++k)"
    NL "     {"
    NL "       if(b[k] != BLOCK_COMPLETED) { res[tid] = -3; return; }"
    NL "     }"
    NL "     res[tid] = BLOCK_COMPLETED;"
    NL "   });"
    NL "  for(i = 0; i < n; ++i)"
    NL "  {"
    NL "    release_event(block_evt[i]);"
    NL "  }"
    NL "  if(enq_res != CLK_SUCCESS) { res[tid] = -4; return; }"
    NL ""
    NL "  set_user_event_status(user_evt, CL_COMPLETE);"
    NL "  release_event(user_evt);"
    NL "}";


static int check_kernel_results(cl_int* results, cl_int len)
{
    for(cl_int i = 0; i < len; ++i)
    {
        if(results[i] != 0) return i;
    }
    return -1;
}

int test_enqueue_multi_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint i;
    cl_int k, err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    cl_int kernel_results[MAX_GWS] = {0};

    size_t ret_len;
    cl_uint n, max_queues = 1;
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

    if(max_queues > 1)
    {
        n = (max_queues > MAX_QUEUES) ? MAX_QUEUES : max_queues-1;
        clMemWrapper mem, buff, evt;
        std::vector<clCommandQueueWrapper> queues(n);
        std::vector<cl_command_queue> q(n);
        cl_queue_properties queue_prop[] =
        {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE,
            CL_QUEUE_SIZE, maxQueueSize,
            0
        };

        for(i = 0; i < n; ++i)
        {
            queues[i] = clCreateCommandQueueWithProperties(context, device, queue_prop, &err_ret);
            test_error(err_ret, "clCreateCommandQueueWithProperties(CL_QUEUE_DEVICE) failed");
            q[i] = queues[i];
        }

        size_t global_size = MAX_GWS;
        size_t local_size = (max_local_size > global_size/16) ? global_size/16 : max_local_size;
        if(gWimpyMode)
        {
            global_size = 4;
            local_size = 2;
        }

        evt = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(cl_event), NULL, &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        mem = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, n * sizeof(cl_command_queue), &q[0], &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        buff = clCreateBuffer(context, CL_MEM_READ_WRITE, global_size * n * sizeof(cl_int), NULL, &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        // Prepare CL source
        char cl[65536] = { 0 };
        char q_args[16384] = { 0 };
        char q_list[8192] = { 0 };

        kernel_arg arg_res = { sizeof(cl_mem), &buff };

        std::vector<kernel_arg> args(n+1);
        args[0] = arg_res;

        for(i = 0; i < n; ++i)
        {
            snprintf(q_args+strlen(q_args), sizeof(q_args)-strlen(q_args)-1, ", queue_t q%d", i);
            snprintf(q_list+strlen(q_list), sizeof(q_list)-strlen(q_list)-1, "q%d, ", i);
            kernel_arg arg_q = { sizeof(cl_command_queue), &q[i] };
            args[i+1] = arg_q;
        }

        snprintf(cl, sizeof(cl)-1, enqueue_block_multi_queue, q_args, n, n, q_list);
        const char *source = cl;

        err_ret = run_n_kernel_args(context, queue, &source, 1, "enqueue_block_multi_queue", local_size, global_size, kernel_results, sizeof(kernel_results), args.size(), &args[0]);
        if(check_error(err_ret, "'%s' kernel execution failed", "enqueue_block_multi_queue")) res = -1;
        else if((k = check_kernel_results(kernel_results, arr_size(kernel_results))) >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", "enqueue_block_multi_queue", k, kernel_results[k])) res = -1;
        else log_info("'%s' kernel is OK.\n", "enqueue_block_multi_queue");
    }
    return res;
}



#endif


