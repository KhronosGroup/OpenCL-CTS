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

static int gNestingLevel = 4;
extern int gWimpyMode;

static const char* enqueue_nested_blocks_single[] =
{
    NL, "void block_fn(__global int* res, int level)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(3);"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(res, level); };"
    NL, ""
    NL, "  // Only 1 work-item enqueues block"
    NL, "  if(tid == 1)"
    NL, "  {"
    NL, "    res[tid]++;"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_nested_blocks_single(__global int* res, int level)"
    NL, "{"
    NL, "  block_fn(res, level);"
    NL, "}"
    NL
};

static const char* enqueue_nested_blocks_some_eq[] =
{
    NL, "void block_fn(int level, __global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(10);"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, res); };"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with the same level"
    NL, "  if(tid < (get_global_size(0) >> 1))"
    NL, "  {"
    NL, "    atomic_inc(&res[tid]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_nested_blocks_some_eq(__global int* res, int level)"
    NL, "{"
    NL, "  block_fn(level, res);"
    NL, "}"
    NL
};

static const char* enqueue_nested_blocks_some_diff[] =
{
    NL, "void block_fn(int level, __global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(10);"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, res); };"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with different levels"
    NL, "  if(tid % 2)"
    NL, "  {"
    NL, "    atomic_inc(&res[tid]);"
    NL, "    if(level >= tid)"
    NL, "    {"
    NL, "      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_nested_blocks_some_diff(__global int* res, int level)"
    NL, "{"
    NL, "  block_fn(level, res);"
    NL, "}"
    NL
};

static const char* enqueue_nested_blocks_all_eq[] =
{
    NL, "void block_fn(int level, __global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(4);"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, res); };"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with the same level"
    NL, "  atomic_inc(&res[tid]);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_nested_blocks_all_eq(__global int* res, int level)"
    NL, "{"
    NL, "  block_fn(level, res);"
    NL, "}"
    NL
};

static const char* enqueue_nested_blocks_all_diff[] =
{
    NL, "void block_fn(int level, __global int* res)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  ndrange_t ndrange = ndrange_1D(10);"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, res); };"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with different levels"
    NL, "  atomic_inc(&res[tid]);"
    NL, "  if(level >= tid)"
    NL, "  {"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_nested_blocks_all_diff(__global int* res, int level)"
    NL, "{"
    NL, "  block_fn(level, res);"
    NL, "}"
    NL
};

static int check_single(cl_int* results, cl_int len, cl_int nesting_level)
{
    int i, fail = -1;
    const cl_uint tid = 1;

    for(i = 0; i < len; ++i)
    {
        if(i != tid && results[i] != 0) { fail = i; break; }
        if(i == tid && results[i] != nesting_level) { fail = i; break; }
    }
    return fail;
}

void generate_reference_some_eq(std::vector<cl_int> &referenceResults, cl_int len, cl_int nesting_level)
{
    size_t globalWorkSize = (nesting_level == gNestingLevel)? len: 10;
    if(--nesting_level < 0) return;

    for (size_t tid = 0; tid < globalWorkSize; ++tid)
    {
        if (tid < (globalWorkSize >> 1))
        {
            ++referenceResults[tid];
            generate_reference_some_eq(referenceResults, len, nesting_level);
        }
    }
}

static int check_some_eq(cl_int* results, cl_int len, cl_int nesting_level)
{
    int i, fail = -1;
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_some_eq(referenceResults, len, nesting_level);

    for(i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i]) { fail = i; break; }
    }

    return fail;
}

void generate_reference_some_diff(std::vector<cl_int> &referenceResults, cl_int len, cl_int nesting_level)
{
    size_t globalWorkSize = (nesting_level == gNestingLevel)? len: 10;
    if(--nesting_level < 0) return;

    for (size_t tid = 0; tid < globalWorkSize; ++tid)
    {
        if (tid % 2)
        {
            ++referenceResults[tid];
            if (nesting_level >= tid)
            {
                generate_reference_some_diff(referenceResults, len, nesting_level);
            }
        }
    }
}

static int check_some_diff(cl_int* results, cl_int len, cl_int nesting_level)
{
    int i, fail = -1;
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_some_diff(referenceResults, len, nesting_level);

    for(i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i]) { fail = i; break; }
    }

    return fail;
}

void generate_reference_all_eq(std::vector<cl_int> &referenceResults, cl_int len, cl_int nesting_level)
{
    size_t globalWorkSize = (nesting_level == gNestingLevel)? len: 4;
    if(--nesting_level < 0) return;

    for (size_t tid = 0; tid < globalWorkSize; ++tid)
    {
        ++referenceResults[tid];
        generate_reference_all_eq(referenceResults, len, nesting_level);
    }
}

static int check_all_eq(cl_int* results, cl_int len, cl_int nesting_level)
{
    int i, fail = -1;
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_all_eq(referenceResults, len, nesting_level);

    for(i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i]) { fail = i; break; }
    }

    return fail;
}

void generate_reference_all_diff(std::vector<cl_int> &referenceResults, cl_int len, cl_int nesting_level)
{
    size_t globalWorkSize = (nesting_level == gNestingLevel)? len: 10;
    if(--nesting_level < 0) return;

    for (size_t tid = 0; tid < globalWorkSize; ++tid)
    {
        ++referenceResults[tid];
        if (nesting_level >= tid)
        {
            generate_reference_all_diff(referenceResults, len, nesting_level);
        }
    }
}

static int check_all_diff(cl_int* results, cl_int len, cl_int nesting_level)
{
    int i, fail = -1;
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_all_diff(referenceResults, len, nesting_level);

    for(i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i]) { fail = i; break; }
    }

    return fail;
}

static const kernel_src_check sources_nested_blocks[] =
{
    { KERNEL(enqueue_nested_blocks_single), check_single },
    { KERNEL(enqueue_nested_blocks_some_eq), check_some_eq },
    { KERNEL(enqueue_nested_blocks_some_diff), check_some_diff },
    { KERNEL(enqueue_nested_blocks_all_eq), check_all_eq },
    { KERNEL(enqueue_nested_blocks_all_diff), check_all_diff }
};

int test_enqueue_nested_blocks(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint i, k;
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    const size_t MAX_GLOBAL_WORK_SIZE = MAX_GWS / 4;
    cl_int kernel_results[MAX_GLOBAL_WORK_SIZE] = {0};

    if(gWimpyMode)
    {
        gNestingLevel = 2;
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
    }

    size_t ret_len;
    cl_uint max_queues = 1;
    cl_uint maxQueueSize = 0;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, sizeof(maxQueueSize), &maxQueueSize, 0);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");

    err_ret = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof(max_queues), &max_queues, &ret_len);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_MAX_ON_DEVICE_QUEUES) failed");

    cl_queue_properties queue_prop_def[] =
    {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_ON_DEVICE|CL_QUEUE_ON_DEVICE_DEFAULT,
        CL_QUEUE_SIZE, maxQueueSize,
        0
    };

    dev_queue = clCreateCommandQueueWithProperties(context, device, queue_prop_def, &err_ret);
    test_error(err_ret,
               "clCreateCommandQueueWithProperties(CL_QUEUE_ON_DEVICE | "
               "CL_QUEUE_ON_DEVICE_DEFAULT) failed");

    kernel_arg args[] =
    {
        { sizeof(cl_int), &gNestingLevel }
    };

    size_t failCnt = 0;
    for(k = 0; k < arr_size(sources_nested_blocks); ++k)
    {
        if (!gKernelName.empty() && gKernelName != sources_nested_blocks[k].src.kernel_name)
            continue;

        log_info("Running '%s' kernel (%d of %zu) ...\n",
                 sources_nested_blocks[k].src.kernel_name, k + 1,
                 arr_size(sources_nested_blocks));
        for(i = 0; i < MAX_GLOBAL_WORK_SIZE; ++i) kernel_results[i] = 0;

        err_ret = run_n_kernel_args(context, queue, sources_nested_blocks[k].src.lines, sources_nested_blocks[k].src.num_lines, sources_nested_blocks[k].src.kernel_name, 0, MAX_GLOBAL_WORK_SIZE, kernel_results, sizeof(kernel_results), arr_size(args), args);
        if(check_error(err_ret, "'%s' kernel execution failed", sources_nested_blocks[k].src.kernel_name)) { res = -1; continue ; }

        //check results
        int fail = sources_nested_blocks[k].check(kernel_results, MAX_GLOBAL_WORK_SIZE, gNestingLevel);

        if(check_error(err_ret, "'%s' kernel execution failed", sources_nested_blocks[k].src.kernel_name)) { ++failCnt; res = -1; continue; }
        else if(fail >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", sources_nested_blocks[k].src.kernel_name, fail, kernel_results[fail])) { ++failCnt; res = -1; continue; }
        else log_info("'%s' kernel is OK.\n", sources_nested_blocks[k].src.kernel_name);
    }

    if (failCnt > 0)
    {
        log_error("ERROR: %zu of %zu kernels failed.\n", failCnt,
                  arr_size(sources_nested_blocks));
    }

    return res;
}

#endif

