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
static const char* enqueue_simple_block[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_simple_block(__global int* res)"
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

static const char* enqueue_block_with_local_arg1[] =
{
    NL, "#define LOCAL_MEM_SIZE 10"
    NL, ""
    NL, "void block_fn_local_arg1(size_t tid, int mul, __global int* res, __local int* tmp)"
    NL, "{"
    NL, "  for(int i = 0; i < LOCAL_MEM_SIZE; i++)"
    NL, "  {"
    NL, "    tmp[i] = mul * 7 - 21;"
    NL, "    res[tid] += tmp[i];"
    NL, "  }"
    NL, "  res[tid] += 2;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_with_local_arg1(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(__local void*) = ^(__local void* buf){ block_fn_local_arg1(tid, multiplier, res, (local int*)buf); };"
    NL, ""
    NL, "  res[tid] = -2;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock, (uint)(LOCAL_MEM_SIZE*sizeof(int)));"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* enqueue_block_with_local_arg2[] =
{
    NL, "#define LOCAL_MEM_SIZE 10"
    NL, ""
    NL, "void block_fn_local_arg1(size_t tid, int mul, __global int* res, __local int* tmp1, __local float4* tmp2)"
    NL, "{"
    NL, "  for(int i = 0; i < LOCAL_MEM_SIZE; i++)"
    NL, "  {"
    NL, "    tmp1[i]   = mul * 7 - 21;"
    NL, "    tmp2[i].x = (float)(mul * 7 - 21);"
    NL, "    tmp2[i].y = (float)(mul * 7 - 21);"
    NL, "    tmp2[i].z = (float)(mul * 7 - 21);"
    NL, "    tmp2[i].w = (float)(mul * 7 - 21);"
    NL, ""
    NL, "    res[tid] += tmp1[i];"
    NL, "    res[tid] += (int)(tmp2[i].x+tmp2[i].y+tmp2[i].z+tmp2[i].w);"
    NL, "  }"
    NL, "  res[tid] += 2;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_with_local_arg2(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  void (^kernelBlock)(__local void*, __local void*) = ^(__local void* buf1, __local void* buf2)"
    NL, "    { block_fn_local_arg1(tid, multiplier, res, (local int*)buf1, (local float4*)buf2); };"
    NL, ""
    NL, "  res[tid] = -2;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock, (uint)(LOCAL_MEM_SIZE*sizeof(int)), (uint)(LOCAL_MEM_SIZE*sizeof(float4)));"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* enqueue_block_with_wait_list[] =
{
    NL, "#define BLOCK_SUBMITTED 1"
    NL, "#define BLOCK_COMPLETED 2"
    NL, "#define CHECK_SUCCESS   0"
    NL, ""
    NL, "kernel void enqueue_block_with_wait_list(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  clk_event_t user_evt = create_user_event();"
    NL, ""
    NL, "  res[tid] = BLOCK_SUBMITTED;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  clk_event_t block_evt;"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt,"
    NL, "  ^{"
    NL, "      res[tid] = BLOCK_COMPLETED;"
    NL, "   });"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "  retain_event(block_evt);"
    NL, "  release_event(block_evt);"
    NL, ""
    NL, "  //check block is not started"
    NL, "  if(res[tid] == BLOCK_SUBMITTED)"
    NL, "  {"
    NL, "    clk_event_t my_evt;"
    NL, "    enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt, &my_evt, "
    NL, "    ^{"
    NL, "       //check block is completed"
    NL, "       if(res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;"
    NL, "     });"
    NL, "    release_event(my_evt);"
    NL, "  }"
    NL, ""
    NL, "  set_user_event_status(user_evt, CL_COMPLETE);"
    NL, ""
    NL, "  release_event(user_evt);"
    NL, "  release_event(block_evt);"
    NL, "}"
    NL
};

static const char* enqueue_block_with_wait_list_and_local_arg[] =
{
    NL, "#define LOCAL_MEM_SIZE 10"
    NL, "#define BLOCK_COMPLETED 1"
    NL, "#define BLOCK_SUBMITTED 2"
    NL, "#define BLOCK_STARTED   3"
    NL, "#define CHECK_SUCCESS   0"
    NL, ""
    NL, "void block_fn_local_arg(size_t tid, int mul, __global int* res, __local int* tmp)"
    NL, "{"
    NL, "  res[tid] = BLOCK_STARTED;"
    NL, "  for(int i = 0; i < LOCAL_MEM_SIZE; i++)"
    NL, "  {"
    NL, "    tmp[i] = mul * 7 - 21;"
    NL, "    res[tid] += tmp[i];"
    NL, "  }"
    NL, "  if(res[tid] == BLOCK_STARTED) res[tid] = BLOCK_COMPLETED;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_with_wait_list_and_local_arg(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  clk_event_t user_evt = create_user_event();"
    NL, ""
    NL, "  res[tid] = BLOCK_SUBMITTED;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  clk_event_t block_evt;"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt, "
    NL, "    ^(__local void* buf) {"
    NL, "       block_fn_local_arg(tid, multiplier, res, (__local int*)buf);"
    NL, "     }, LOCAL_MEM_SIZE*sizeof(int));"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "  retain_event(block_evt);"
    NL, "  release_event(block_evt);"
    NL, ""
    NL, "  //check block is not started"
    NL, "  if(res[tid] == BLOCK_SUBMITTED)"
    NL, "  {"
    NL, "    clk_event_t my_evt;"
    NL, "    enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt, &my_evt, "
    NL, "    ^{"
    NL, "       //check block is completed"
    NL, "       if(res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;"
    NL, "     });"
    NL, "    release_event(my_evt);"
    NL, "  }"
    NL, ""
    NL, "  set_user_event_status(user_evt, CL_COMPLETE);"
    NL, ""
    NL, "  release_event(user_evt);"
    NL, "  release_event(block_evt);"
    NL, "}"
    NL
};

static const char* enqueue_block_get_kernel_work_group_size[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_get_kernel_work_group_size(__global int* res)"
    NL, "{"
    NL, "    int multiplier = 3;"
    NL, "    size_t tid = get_global_id(0);"
    NL, ""
    NL, "    void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "    size_t local_work_size = get_kernel_work_group_size(kernelBlock);"
    NL, "    if (local_work_size <= 0){ res[tid] = -1; return; }"
    NL, "    size_t global_work_size = local_work_size * 4;"
    NL, ""
    NL, "    res[tid] = -1;"
    NL, "    queue_t q1 = get_default_queue();"
    NL, "    ndrange_t ndrange = ndrange_1D(global_work_size, local_work_size);"
    NL, ""
    NL, "    int enq_res = enqueue_kernel(q1, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
};

static const char* enqueue_block_get_kernel_preferred_work_group_size_multiple[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  res[tid] = mul * 7 - 21;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_get_kernel_preferred_work_group_size_multiple(__global int* res)"
    NL, "{"
    NL, "    int multiplier = 3;"
    NL, "    size_t tid = get_global_id(0);"
    NL, ""
    NL, "    void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "    size_t local_work_size = get_kernel_preferred_work_group_size_multiple(kernelBlock);"
    NL, "    if (local_work_size <= 0){ res[tid] = -1; return; }"
    NL, "    size_t global_work_size = local_work_size * 4;"
    NL, ""
    NL, "    res[tid] = -1;"
    NL, "    queue_t q1 = get_default_queue();"
    NL, "    ndrange_t ndrange = ndrange_1D(global_work_size, local_work_size);"
    NL, ""
    NL, "    int enq_res = enqueue_kernel(q1, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
};

static const char* enqueue_block_capture_event_profiling_info_after_execution[] =
{
    NL, "#define MAX_GWS " STRINGIFY_VALUE(MAX_GWS)
    NL, ""
    NL, "__global ulong value[MAX_GWS*2] = {0};"
    NL, ""
    NL, "void block_fn(size_t tid, __global int* res)"
    NL, "{"
    NL, "    res[tid] = -2;"
    NL, "}"
    NL, ""
    NL, "void check_res(size_t tid, const clk_event_t evt, __global int* res)"
    NL, "{"
    NL, "    capture_event_profiling_info (evt, CLK_PROFILING_COMMAND_EXEC_TIME, &value[tid*2]);"
    NL, ""
    NL, "    if (value[tid*2] > 0 && value[tid*2+1] > 0) res[tid] =  0;"
    NL, "    else                                        res[tid] = -4;"
    NL, "    release_event(evt);"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_capture_event_profiling_info_after_execution(__global int* res)"
    NL, "{"
    NL, "    size_t tid = get_global_id(0);"
    NL, ""
    NL, "    res[tid] = -1;"
    NL, "    queue_t def_q = get_default_queue();"
    NL, "    ndrange_t ndrange = ndrange_1D(1);"
    NL, "    clk_event_t block_evt1;"
    NL, ""
    NL, "    void (^kernelBlock)(void)  = ^{ block_fn (tid, res);                   };"
    NL, ""
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 0, NULL, &block_evt1, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "    void (^checkBlock) (void)  = ^{ check_res(tid, block_evt1, res);      };"
    NL, ""
    NL, "    enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt1, NULL, checkBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -3; return; }"
    NL, "}"
    NL
};

static const char* enqueue_block_capture_event_profiling_info_before_execution[] =
{
    NL, "#define MAX_GWS " STRINGIFY_VALUE(MAX_GWS)
    NL, ""
    NL, "__global ulong value[MAX_GWS*2] = {0};"
    NL, ""
    NL, "void block_fn(size_t tid, __global int* res)"
    NL, "{"
    NL, "    res[tid] = -2;"
    NL, "}"
    NL, ""
    NL, "void check_res(size_t tid, const ulong *value, __global int* res)"
    NL, "{"
    NL, "    if (value[tid*2] > 0 && value[tid*2+1] > 0) res[tid] =  0;"
    NL, "    else                                        res[tid] = -4;"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_capture_event_profiling_info_before_execution(__global int* res)"
    NL, "{"
    NL, "    int multiplier = 3;"
    NL, "    size_t tid = get_global_id(0);"
    NL, "    clk_event_t user_evt = create_user_event();"
    NL, ""
    NL, "    res[tid] = -1;"
    NL, "    queue_t def_q = get_default_queue();"
    NL, "    ndrange_t ndrange = ndrange_1D(1);"
    NL, "    clk_event_t block_evt1;"
    NL, "    clk_event_t block_evt2;"
    NL, ""
    NL, "    void (^kernelBlock)(void)  = ^{ block_fn (tid, res);                   };"
    NL, ""
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt1, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "    capture_event_profiling_info (block_evt1, CLK_PROFILING_COMMAND_EXEC_TIME, &value[tid*2]);"
    NL, ""
    NL, "    set_user_event_status(user_evt, CL_COMPLETE);"
    NL, ""
    NL, "    void (^checkBlock) (void)  = ^{ check_res(tid, &value, res);      };"
    NL, ""
    NL, "    enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &block_evt1, &block_evt2, checkBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -3; return; }"
    NL, ""
    NL, "    release_event(user_evt);"
    NL, "    release_event(block_evt1);"
    NL, "    release_event(block_evt2);"
    NL, "}"
    NL
};

static const char* enqueue_block_with_barrier[] =
{
    NL, "void block_fn(size_t tid, int mul, __global int* res)"
    NL, "{"
    NL, "  if(mul > 0) barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, "  res[tid] = mul * 7 -21;"
    NL, "}"
    NL, ""
    NL, "void loop_fn(size_t tid, int n, __global int* res)"
    NL, "{"
    NL, "  while(n > 0)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, "    res[tid] = 0;"
    NL, "    --n;"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_block_with_barrier(__global int* res)"
    NL, "{"
    NL, "  int multiplier = 3;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  res[tid] = -1;"
    NL, "  size_t n = 256;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(tid, multiplier, res); };"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(n);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "  void (^loopBlock)(void) = ^{ loop_fn(tid, n, res); };"
    NL, ""
    NL, "  enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, loopBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL
};

static const char* enqueue_marker_with_block_event[] =
{
    NL, "#define BLOCK_COMPLETED 1"
    NL, "#define BLOCK_SUBMITTED 2"
    NL, "#define CHECK_SUCCESS   0"
    NL, ""
    NL, "kernel void enqueue_marker_with_block_event(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  clk_event_t user_evt = create_user_event();"
    NL, ""
    NL, "  res[tid] = BLOCK_SUBMITTED;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, ""
    NL, "  clk_event_t block_evt1;"
    NL, "  clk_event_t marker_evt;"
    NL, ""
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &user_evt, &block_evt1,"
    NL, "  ^{"
    NL, "     res[tid] = BLOCK_COMPLETED;"
    NL, "   });"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -2; return; }"
    NL, ""
    NL, "  enq_res = enqueue_marker(def_q, 1, &block_evt1, &marker_evt);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -3; return; }"
    NL, ""
    NL, "  retain_event(marker_evt);"
    NL, "  release_event(marker_evt);"
    NL, ""
    NL, "  //check block is not started"
    NL, "  if(res[tid] == BLOCK_SUBMITTED)"
    NL, "  {"
    NL, "    clk_event_t my_evt;"
    NL, "    enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &my_evt, "
    NL, "    ^{"
    NL, "       //check block is completed"
    NL, "       if(res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;"
    NL, "     });"
    NL, "    release_event(my_evt);"
    NL, "  }"
    NL, ""
    NL, "  set_user_event_status(user_evt, CL_COMPLETE);"
    NL, ""
    NL, "  release_event(block_evt1);"
    NL, "  release_event(marker_evt);"
    NL, "  release_event(user_evt);"
    NL, "}"
    NL
};

static const char* enqueue_marker_with_user_event[] =
{
    NL, "#define BLOCK_COMPLETED 1"
    NL, "#define BLOCK_SUBMITTED 2"
    NL, "#define CHECK_SUCCESS   0"
    NL, ""
    NL, "kernel void enqueue_marker_with_user_event(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  uint multiplier = 7;"
    NL, ""
    NL, "  clk_event_t user_evt = create_user_event();"
    NL, ""
    NL, "  res[tid] = BLOCK_SUBMITTED;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, ""
    NL, "  clk_event_t marker_evt;"
    NL, "  clk_event_t block_evt;"
    NL, ""
    NL, "  int enq_res = enqueue_marker(def_q, 1, &user_evt, &marker_evt);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "  retain_event(marker_evt);"
    NL, "  release_event(marker_evt);"
    NL, ""
    NL, "  enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &block_evt, "
    NL, "  ^{"
    NL, "     if(res[tid] == BLOCK_SUBMITTED) res[tid] = CHECK_SUCCESS;"
    NL, "   });"
    NL, ""
    NL, "  //check block is not started"
    NL, "  if(res[tid] != BLOCK_SUBMITTED)  { res[tid] = -2; return; }"
    NL, ""
    NL, "  set_user_event_status(user_evt, CL_COMPLETE);"
    NL, ""
    NL, "  release_event(block_evt);"
    NL, "  release_event(marker_evt);"
    NL, "  release_event(user_evt);"
    NL, "}"
    NL
};

static const char* enqueue_marker_with_mixed_events[] =
{
    NL, "#define BLOCK_COMPLETED 1"
    NL, "#define BLOCK_SUBMITTED 2"
    NL, "#define CHECK_SUCCESS   0"
    NL, ""
    NL, "kernel void enqueue_marker_with_mixed_events(__global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  clk_event_t mix_ev[2];"
    NL, "  mix_ev[0] = create_user_event();"
    NL, ""
    NL, "  res[tid] = BLOCK_SUBMITTED;"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, ""
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &mix_ev[0], &mix_ev[1],"
    NL, "  ^{"
    NL, "     res[tid] = BLOCK_COMPLETED;"
    NL, "   });"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -2; return; }"
    NL, ""
    NL, "  clk_event_t marker_evt;"
    NL, ""
    NL, "  enq_res = enqueue_marker(def_q, 2, mix_ev, &marker_evt);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -3; return; }"
    NL, ""
    NL, "  retain_event(marker_evt);"
    NL, "  release_event(marker_evt);"
    NL, ""
    NL, "  //check block is not started"
    NL, "  if(res[tid] == BLOCK_SUBMITTED)"
    NL, "  {"
    NL, "    clk_event_t my_evt;"
    NL, "    enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &marker_evt, &my_evt, "
    NL, "    ^{"
    NL, "       //check block is completed"
    NL, "       if(res[tid] == BLOCK_COMPLETED) res[tid] = CHECK_SUCCESS;"
    NL, "     });"
    NL, "    release_event(my_evt);"
    NL, "  }"
    NL, ""
    NL, "  set_user_event_status(mix_ev[0], CL_COMPLETE);"
    NL, ""
    NL, "  release_event(mix_ev[1]);"
    NL, "  release_event(marker_evt);"
    NL, "  release_event(mix_ev[0]);"
    NL, "}"
    NL
};

static const char* enqueue_block_with_mixed_events[] =
{
    NL, "kernel void enqueue_block_with_mixed_events(__global int* res)"
    NL, "{"
    NL, "  int enq_res;"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  clk_event_t mix_ev[3];"
    NL, "  mix_ev[0] = create_user_event();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(1);"
    NL, "  res[tid] = -2;"
    NL, ""
    NL, "  enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, 1, &mix_ev[0], &mix_ev[1], ^{ res[tid]++; });"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, ""
    NL, "  enq_res = enqueue_marker(def_q, 1, &mix_ev[1], &mix_ev[2]);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -3; return; }"
    NL, ""
    NL, "  enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, sizeof(mix_ev)/sizeof(mix_ev[0]), mix_ev, NULL, ^{ res[tid]++; });"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -4; return; }"
    NL, ""
    NL, "  set_user_event_status(mix_ev[0], CL_COMPLETE);"
    NL, ""
    NL, "  release_event(mix_ev[0]);"
    NL, "  release_event(mix_ev[1]);"
    NL, "  release_event(mix_ev[2]);"
    NL, "}"
    NL
};

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
    test_error(err_ret, "clCreateCommandQueueWithProperties(CL_QUEUE_DEVICE|CL_QUEUE_DEFAULT) failed");

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

        log_info("Running '%s' kernel (%d of %d) ...\n", sources_enqueue_block[i].kernel_name, i + 1, num_kernels_enqueue_block);
        err_ret = run_n_kernel_args(context, queue, sources_enqueue_block[i].lines, sources_enqueue_block[i].num_lines, sources_enqueue_block[i].kernel_name, local_size, global_size, kernel_results, sizeof(kernel_results), 0, NULL);
        if(check_error(err_ret, "'%s' kernel execution failed", sources_enqueue_block[i].kernel_name)) { ++failCnt; res = -1; }
        else if((n = check_kernel_results(kernel_results, arr_size(kernel_results))) >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", sources_enqueue_block[i].kernel_name, n, kernel_results[n])) res = -1;
        else log_info("'%s' kernel is OK.\n", sources_enqueue_block[i].kernel_name);
    }

    if (failCnt > 0)
    {
      log_error("ERROR: %d of %d kernels failed.\n", failCnt, num_kernels_enqueue_block);
    }

    return res;
}



#endif


