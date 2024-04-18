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

// clang-format off
static const char* enqueue_simple_block[] = { R"(
    void block_fn(size_t tid, int mul, __global int* res)
    {
      res[tid] = mul * 7 - 21;
    }

    kernel void enqueue_simple_block(__global int* res)
    {
      int multiplier = 3;
      size_t tid = get_global_id(0);

      void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };

      res[tid] = -1;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_block_with_local_arg1[] = { R"(
    #define LOCAL_MEM_SIZE 10

    void block_fn_local_arg1(size_t tid, int mul, __global int* res, __local int* tmp)
    {
      for (int i = 0; i < LOCAL_MEM_SIZE; i++)
      {
        tmp[i] = mul * 7 - 21;
        res[tid] += tmp[i];
      }
      res[tid] += 2;
    }

    kernel void enqueue_block_with_local_arg1(__global int* res)
    {
      int multiplier = 3;
      size_t tid = get_global_id(0);

      void (^kernelBlock)(__local void*) = ^(__local void* buf){ block_fn_local_arg1(tid, multiplier, res, (local int*)buf); };

      res[tid] = -2;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock, (uint)(LOCAL_MEM_SIZE*sizeof(int)));
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_block_with_local_arg2[] = { R"(
    #define LOCAL_MEM_SIZE 10

    void block_fn_local_arg1(size_t tid, int mul, __global int* res, __local int* tmp1, __local float4* tmp2)
    {
      for (int i = 0; i < LOCAL_MEM_SIZE; i++)
      {
        tmp1[i]   = mul * 7 - 21;
        tmp2[i].x = (float)(mul * 7 - 21);
        tmp2[i].y = (float)(mul * 7 - 21);
        tmp2[i].z = (float)(mul * 7 - 21);
        tmp2[i].w = (float)(mul * 7 - 21);

        res[tid] += tmp1[i];
        res[tid] += (int)(tmp2[i].x+tmp2[i].y+tmp2[i].z+tmp2[i].w);
      }
      res[tid] += 2;
    }

    kernel void enqueue_block_with_local_arg2(__global int* res)
    {
      int multiplier = 3;
      size_t tid = get_global_id(0);

      void (^kernelBlock)(__local void*, __local void*) = ^(__local void* buf1, __local void* buf2)
        { block_fn_local_arg1(tid, multiplier, res, (local int*)buf1, (local float4*)buf2); };

      res[tid] = -2;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock, (uint)(LOCAL_MEM_SIZE*sizeof(int)), (uint)(LOCAL_MEM_SIZE*sizeof(float4)));
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_block_with_wait_list[] = { R"(
    #define BLOCK_SUBMITTED 1
    #define BLOCK_COMPLETED 2
    #define CHECK_SUCCESS   0

    kernel void enqueue_block_with_wait_list(__global int* res)
    {
      size_t tid = get_global_id(0);

      clk_event_t user_evt = create_user_event();

      res[tid] = BLOCK_SUBMITTED;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      clk_event_t block_evt;
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt,
      ^{
          res[tid] = BLOCK_COMPLETED;
       });
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

      retain_event(block_evt);
      release_event(block_evt);

      //check block is not started
      if (res[tid] == BLOCK_SUBMITTED)
      {
        clk_event_t my_evt;
        enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt, &my_evt,
        ^{
           //check block is completed
           if (res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;
         });
        release_event(my_evt);
      }

      set_user_event_status(user_evt, CL_COMPLETE);

      release_event(user_evt);
      release_event(block_evt);
    }
)" };

static const char* enqueue_block_with_wait_list_and_local_arg[] = { R"(
    #define LOCAL_MEM_SIZE 10
    #define BLOCK_COMPLETED 1
    #define BLOCK_SUBMITTED 2
    #define BLOCK_STARTED   3
    #define CHECK_SUCCESS   0

    void block_fn_local_arg(size_t tid, int mul, __global int* res, __local int* tmp)
    {
      res[tid] = BLOCK_STARTED;
      for (int i = 0; i < LOCAL_MEM_SIZE; i++)
      {
        tmp[i] = mul * 7 - 21;
        res[tid] += tmp[i];
      }
      if (res[tid] == BLOCK_STARTED) res[tid] = BLOCK_COMPLETED;
    }

    kernel void enqueue_block_with_wait_list_and_local_arg(__global int* res)
    {
      int multiplier = 3;
      size_t tid = get_global_id(0);
      clk_event_t user_evt = create_user_event();

      res[tid] = BLOCK_SUBMITTED;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      clk_event_t block_evt;
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt,
        ^(__local void* buf) {
           block_fn_local_arg(tid, multiplier, res, (__local int*)buf);
         }, LOCAL_MEM_SIZE*sizeof(int));
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

      retain_event(block_evt);
      release_event(block_evt);

      //check block is not started
      if (res[tid] == BLOCK_SUBMITTED)
      {
        clk_event_t my_evt;
        enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt, &my_evt,
        ^{
           //check block is completed
           if (res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;
         });
        release_event(my_evt);
      }

      set_user_event_status(user_evt, CL_COMPLETE);

      release_event(user_evt);
      release_event(block_evt);
    }
)" };

static const char* enqueue_block_get_kernel_work_group_size[] = { R"(
    void block_fn(size_t tid, int mul, __global int* res)
    {
      res[tid] = mul * 7 - 21;
    }

    kernel void enqueue_block_get_kernel_work_group_size(__global int* res)
    {
        int multiplier = 3;
        size_t tid = get_global_id(0);

        void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };

        size_t local_work_size = get_kernel_work_group_size(kernelBlock);
        if (local_work_size <= 0){ res[tid] = -1; return; }
        size_t global_work_size = local_work_size * 4;

        res[tid] = -1;
        queue_t q1 = get_default_queue();
        ndrange_t ndrange = ndrange_1D(global_work_size, local_work_size);

        int enq_res = enqueue_kernel(q1, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_block_get_kernel_preferred_work_group_size_multiple[] = { R"(
    void block_fn(size_t tid, int mul, __global int* res)
    {
      res[tid] = mul * 7 - 21;
    }

    kernel void enqueue_block_get_kernel_preferred_work_group_size_multiple(__global int* res)
    {
        int multiplier = 3;
        size_t tid = get_global_id(0);

        void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };

        size_t local_work_size = get_kernel_preferred_work_group_size_multiple(kernelBlock);
        if (local_work_size <= 0){ res[tid] = -1; return; }
        size_t global_work_size = local_work_size * 4;

        res[tid] = -1;
        queue_t q1 = get_default_queue();
        ndrange_t ndrange = ndrange_1D(global_work_size, local_work_size);

        int enq_res = enqueue_kernel(q1, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_block_capture_event_profiling_info_after_execution[] = {
    "#define MAX_GWS " STRINGIFY_VALUE(MAX_GWS) "\n"
    , R"(
    __global ulong value[MAX_GWS*2] = {0};

    void block_fn(size_t tid, __global int* res)
    {
        res[tid] = -2;
    }

    void check_res(size_t tid, const clk_event_t evt, __global int* res)
    {
        capture_event_profiling_info (evt, CLK_PROFILING_COMMAND_EXEC_TIME, &value[tid*2]);

        if (value[tid*2] > 0 && value[tid*2+1] > 0) res[tid] =  0;
        else                                        res[tid] = -4;
        release_event(evt);
    }

    kernel void enqueue_block_capture_event_profiling_info_after_execution(__global int* res)
    {
        size_t tid = get_global_id(0);

        res[tid] = -1;
        queue_t def_q = get_default_queue();
        ndrange_t ndrange = ndrange_1D(1);
        clk_event_t block_evt1;

        void (^kernelBlock)(void)  = ^{ block_fn (tid, res);                   };

        int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 0, NULL, &block_evt1, kernelBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

        void (^checkBlock) (void)  = ^{ check_res(tid, block_evt1, res);      };

        enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt1, NULL, checkBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -3; return; }
    }
)" };

static const char* enqueue_block_capture_event_profiling_info_before_execution[] = {
    "#define MAX_GWS " STRINGIFY_VALUE(MAX_GWS) "\n"
    , R"(
    __global ulong value[MAX_GWS*2] = {0};

    void block_fn(size_t tid, __global int* res)
    {
        res[tid] = -2;
    }

    void check_res(size_t tid, const ulong *value, __global int* res)
    {
        if (value[tid*2] > 0 && value[tid*2+1] > 0) res[tid] =  0;
        else                                        res[tid] = -4;
    }

    kernel void enqueue_block_capture_event_profiling_info_before_execution(__global int* res)
    {
        int multiplier = 3;
        size_t tid = get_global_id(0);
        clk_event_t user_evt = create_user_event();

        res[tid] = -1;
        queue_t def_q = get_default_queue();
        ndrange_t ndrange = ndrange_1D(1);
        clk_event_t block_evt1;
        clk_event_t block_evt2;

        void (^kernelBlock)(void)  = ^{ block_fn (tid, res);                   };

        int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt1, kernelBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

        capture_event_profiling_info (block_evt1, CLK_PROFILING_COMMAND_EXEC_TIME, &value[tid*2]);

        set_user_event_status(user_evt, CL_COMPLETE);

        void (^checkBlock) (void)  = ^{ check_res(tid, &value, res);      };

        enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt1, &block_evt2, checkBlock);
        if (enq_res != CLK_SUCCESS) { res[tid] = -3; return; }

        release_event(user_evt);
        release_event(block_evt1);
        release_event(block_evt2);
    }
)" };

static const char* enqueue_block_with_barrier[] = { R"(
    void block_fn(size_t tid, int mul, __global int* res)
    {
      if (mul > 0) barrier(CLK_GLOBAL_MEM_FENCE);
      res[tid] = mul * 7 -21;
    }

    void loop_fn(size_t tid, int n, __global int* res)
    {
      while (n > 0)
      {
        barrier(CLK_GLOBAL_MEM_FENCE);
        res[tid] = 0;
        --n;
      }
    }

    kernel void enqueue_block_with_barrier(__global int* res)
    {
      int multiplier = 3;
      size_t tid = get_global_id(0);
      queue_t def_q = get_default_queue();
      res[tid] = -1;
      size_t n = 256;

      void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };

      ndrange_t ndrange = ndrange_1D(n);
      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

      void (^loopBlock)(void) = ^{ loop_fn(tid, n, res); };

      enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, loopBlock);
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }
    }
)" };

static const char* enqueue_marker_with_block_event[] = { R"(
    #define BLOCK_COMPLETED 1
    #define BLOCK_SUBMITTED 2
    #define CHECK_SUCCESS   0

    kernel void enqueue_marker_with_block_event(__global int* res)
    {
      size_t tid = get_global_id(0);

      clk_event_t user_evt = create_user_event();

      res[tid] = BLOCK_SUBMITTED;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);

      clk_event_t block_evt1;
      clk_event_t marker_evt;

      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt1,
      ^{
         res[tid] = BLOCK_COMPLETED;
       });
      if (enq_res != CLK_SUCCESS) { res[tid] = -2; return; }

      enq_res = enqueue_marker(def_q, 1, &block_evt1, &marker_evt);
      if (enq_res != CLK_SUCCESS) { res[tid] = -3; return; }

      retain_event(marker_evt);
      release_event(marker_evt);

      //check block is not started
      if (res[tid] == BLOCK_SUBMITTED)
      {
        clk_event_t my_evt;
        enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &my_evt,
        ^{
           //check block is completed
           if (res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;
         });
        release_event(my_evt);
      }

      set_user_event_status(user_evt, CL_COMPLETE);

      release_event(block_evt1);
      release_event(marker_evt);
      release_event(user_evt);
    }
)" };

static const char* enqueue_marker_with_user_event[] = { R"(
    #define BLOCK_COMPLETED 1
    #define BLOCK_SUBMITTED 2
    #define CHECK_SUCCESS   0

    kernel void enqueue_marker_with_user_event(__global int* res)
    {
      size_t tid = get_global_id(0);
      uint multiplier = 7;

      clk_event_t user_evt = create_user_event();

      res[tid] = BLOCK_SUBMITTED;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);

      clk_event_t marker_evt;
      clk_event_t block_evt;

      int enq_res = enqueue_marker(def_q, 1, &user_evt, &marker_evt);
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

      retain_event(marker_evt);
      release_event(marker_evt);

      enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &block_evt,
      ^{
         if (res[tid] == BLOCK_SUBMITTED) res[tid] = CHECK_SUCCESS;
       });

      //check block is not started
      if (res[tid] != BLOCK_SUBMITTED)  { res[tid] = -2; return; }

      set_user_event_status(user_evt, CL_COMPLETE);

      release_event(block_evt);
      release_event(marker_evt);
      release_event(user_evt);
    }
)" };

static const char* enqueue_marker_with_mixed_events[] = { R"(
    #define BLOCK_COMPLETED 1
    #define BLOCK_SUBMITTED 2
    #define CHECK_SUCCESS   0

    kernel void enqueue_marker_with_mixed_events(__global int* res)
    {
      size_t tid = get_global_id(0);

      clk_event_t mix_ev[2];
      mix_ev[0] = create_user_event();

      res[tid] = BLOCK_SUBMITTED;
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);

      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &mix_ev[0], &mix_ev[1],
      ^{
         res[tid] = BLOCK_COMPLETED;
       });
      if (enq_res != CLK_SUCCESS) { res[tid] = -2; return; }

      clk_event_t marker_evt;

      enq_res = enqueue_marker(def_q, 2, mix_ev, &marker_evt);
      if (enq_res != CLK_SUCCESS) { res[tid] = -3; return; }

      retain_event(marker_evt);
      release_event(marker_evt);

      //check block is not started
      if (res[tid] == BLOCK_SUBMITTED)
      {
        clk_event_t my_evt;
        enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &my_evt,
        ^{
           //check block is completed
           if (res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;
         });
        release_event(my_evt);
      }

      set_user_event_status(mix_ev[0], CL_COMPLETE);

      release_event(mix_ev[1]);
      release_event(marker_evt);
      release_event(mix_ev[0]);
    }
)" };

static const char* enqueue_block_with_mixed_events[] = { R"(
    kernel void enqueue_block_with_mixed_events(__global int* res)
    {
      int enq_res;
      size_t tid = get_global_id(0);
      clk_event_t mix_ev[3];
      mix_ev[0] = create_user_event();
      queue_t def_q = get_default_queue();
      ndrange_t ndrange = ndrange_1D(1);
      res[tid] = -2;

      enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &mix_ev[0], &mix_ev[1], ^{ res[tid]++; });
      if (enq_res != CLK_SUCCESS) { res[tid] = -1; return; }

      enq_res = enqueue_marker(def_q, 1, &mix_ev[1], &mix_ev[2]);
      if (enq_res != CLK_SUCCESS) { res[tid] = -3; return; }

      enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, sizeof(mix_ev)/sizeof(mix_ev[0]), mix_ev, NULL, ^{ res[tid]++; });
      if (enq_res != CLK_SUCCESS) { res[tid] = -4; return; }

      set_user_event_status(mix_ev[0], CL_COMPLETE);

      release_event(mix_ev[0]);
      release_event(mix_ev[1]);
      release_event(mix_ev[2]);
    }
)" };
// clang-format on

static const kernel_src sources_enqueue_block[] =
{
    KERNEL(enqueue_simple_block),
    // Block with local mem
    KERNEL(enqueue_block_with_local_arg1),
    KERNEL(enqueue_block_with_local_arg2),
    KERNEL(enqueue_block_with_wait_list),
    KERNEL(enqueue_block_with_wait_list_and_local_arg),
    // WG size built-ins
    KERNEL(enqueue_block_get_kernel_work_group_size),
    KERNEL(enqueue_block_get_kernel_preferred_work_group_size_multiple),
    // Event profiling info
    KERNEL(enqueue_block_capture_event_profiling_info_after_execution),
    KERNEL(enqueue_block_capture_event_profiling_info_before_execution),
    // Marker
    KERNEL(enqueue_marker_with_block_event),
    KERNEL(enqueue_marker_with_user_event),
    // Mixed events
    KERNEL(enqueue_marker_with_mixed_events),
    KERNEL(enqueue_block_with_mixed_events),
    // Barrier
    KERNEL(enqueue_block_with_barrier),

};
static const size_t num_kernels_enqueue_block = arr_size(sources_enqueue_block);

static int check_kernel_results(cl_int* results, cl_int len)
{
    for(cl_int i = 0; i < len; ++i)
    {
        if(results[i] != 0) return i;
    }
    return -1;
}

int test_enqueue_block(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint i;
    cl_int n, err_ret, res = 0;
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
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE|CL_QUEUE_ON_DEVICE_DEFAULT|CL_QUEUE_PROFILING_ENABLE,
        CL_QUEUE_SIZE, maxQueueSize,
        0
    };

    dev_queue = clCreateCommandQueueWithProperties(context, device, queue_prop_def, &err_ret);
    test_error(err_ret,
               "clCreateCommandQueueWithProperties(CL_QUEUE_ON_DEVICE | "
               "CL_QUEUE_ON_DEVICE_DEFAULT) failed");

    size_t global_size = MAX_GWS;
    size_t local_size = (max_local_size > global_size/16) ? global_size/16 : max_local_size;
    if(gWimpyMode)
    {
        global_size = 4;
        local_size = 2;
    }

    size_t failCnt = 0;
    for(i = 0; i < num_kernels_enqueue_block; ++i)
    {
        if (!gKernelName.empty() && gKernelName != sources_enqueue_block[i].kernel_name)
            continue;

        log_info("Running '%s' kernel (%d of %zu) ...\n",
                 sources_enqueue_block[i].kernel_name, i + 1,
                 num_kernels_enqueue_block);
        err_ret = run_n_kernel_args(context, queue, sources_enqueue_block[i].lines, sources_enqueue_block[i].num_lines, sources_enqueue_block[i].kernel_name, local_size, global_size, kernel_results, sizeof(kernel_results), 0, NULL);
        if(check_error(err_ret, "'%s' kernel execution failed", sources_enqueue_block[i].kernel_name)) { ++failCnt; res = -1; }
        else if((n = check_kernel_results(kernel_results, arr_size(kernel_results))) >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", sources_enqueue_block[i].kernel_name, n, kernel_results[n])) res = -1;
        else log_info("'%s' kernel is OK.\n", sources_enqueue_block[i].kernel_name);
    }

    if (failCnt > 0)
    {
        log_error("ERROR: %zu of %zu kernels failed.\n", failCnt,
                  num_kernels_enqueue_block);
    }

    return res;
}



#endif


