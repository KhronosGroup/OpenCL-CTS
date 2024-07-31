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
static int nestingLevel = 3;

static const char* enqueue_1D_wg_size_single[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs = 64 * 64 * 64;"
    NL, "  size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "  ls = ls? ls: 1;"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(gs, ls);"
    NL, ""
    NL, "  // Only 1 work-item enqueues block"
    NL, "  if(tidX == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[tidX % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tidX % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_1D_wg_size_single(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

static int check_single(cl_int* results, cl_int len, cl_int nesting_level)
{
    for(size_t i = 0; i < len; ++i)
    {
        if(i == 0 && results[i] != nestingLevel)
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], nestingLevel, i);
            return (int)i;
        }

        if(i > 0 && results[i] != 0)
        {
            log_error("ERROR: Kernel returned %d vs. expected 0, index: %zu\n",
                      results[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_1D_wg_size_some_eq[] =
{
    NL, "void block_fn(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(res, level, maxGlobalWorkSize, rnd); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs = 8 * 8 * 2;"
    NL, "  size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "  ls = ls? ls: 1;"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with the same level"
    NL, "  if((tidX % (maxGlobalWorkSize / 8)) == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[tidX % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tidX % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_1D_wg_size_some_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(res, level, maxGlobalWorkSize, rnd);"
    NL, "}"
    NL
};

void generate_reference_results_some_eq_1D(std::vector<cl_int> &referenceResults, cl_int maxGlobalWorkSize, cl_int level)
{
    size_t globalSize = (level == nestingLevel) ? maxGlobalWorkSize: (8 * 8 * 2);
    if(--level < 0)
    {
        return;
    }

    for (size_t tidX = 0; tidX < globalSize; ++tidX)
    {
        if ((tidX % (maxGlobalWorkSize / 8)) == 0)
        {
            ++referenceResults[tidX % maxGlobalWorkSize];
            generate_reference_results_some_eq_1D(referenceResults, maxGlobalWorkSize, level);
        }
    }
}

static int check_some_eq_1D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_eq_1D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_1D_wg_size_some_diff[] =
{
    NL, "void block_fn(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(res, level, maxGlobalWorkSize, rnd); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs = 8 * 8 * 8;"
    NL, "  size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "  ls = ls? ls: 1;"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with different levels"
    NL, "  if((tidX % 2) == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[tidX % maxGlobalWorkSize]);"
    NL, "    if(level >= tidX)"
    NL, "    {"
    NL, "      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[tidX % maxGlobalWorkSize] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_1D_wg_size_some_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(res, level, maxGlobalWorkSize, rnd);"
    NL, "}"
    NL
};

void generate_reference_results_some_diff_1D(std::vector<cl_int> &referenceResults, cl_int maxGlobalWorkSize, cl_int level)
{
    size_t globalSize = (level == nestingLevel) ? maxGlobalWorkSize: (8 * 8 * 8);
    if(--level < 0)
    {
        return;
    }

    for (size_t tidX = 0; tidX < globalSize; ++tidX)
    {
        if ((tidX % 2) == 0)
        {
            ++referenceResults[tidX % maxGlobalWorkSize];
            if (level >= tidX)
            {
                generate_reference_results_some_diff_1D(referenceResults, maxGlobalWorkSize, level);
            }
        }
    }
}

static int check_some_diff_1D(cl_int* results, cl_int maxGlobalWorkSize, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(maxGlobalWorkSize, 0);
    generate_reference_results_some_diff_1D(referenceResults, maxGlobalWorkSize, nesting_level);

    for(size_t i = 0; i < maxGlobalWorkSize; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_1D_wg_size_all_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs = 8;"
    NL, "  size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "  ls = ls? ls: 1;"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with the same level"
    NL, "  atomic_inc(&res[tidX % maxGlobalWorkSize]);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[tidX % maxGlobalWorkSize] = -1; return; }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_1D_wg_size_all_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_eq_1D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSize = (level == nestingLevel) ? len: 8;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidX = 0; tidX < globalSize; ++tidX)
    {
        ++referenceResults[tidX % len];
        generate_reference_results_all_eq_1D(referenceResults, len, level);
    }
}

static int check_all_eq_1D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_eq_1D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_1D_wg_size_all_diff[] =
{
    NL, "void block_fn(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if((--level) < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(res, level, maxGlobalWorkSize, rnd); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs = 8 * 8 * 8;"
    NL, "  size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "  ls = ls? ls: 1;"
    NL, ""
    NL, "  ndrange_t ndrange = ndrange_1D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with different levels"
    NL, "  atomic_inc(&res[tidX % maxGlobalWorkSize]);"
    NL, "  if(level >= tidX)"
    NL, "  {"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tidX % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_1D_wg_size_all_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(res, level, maxGlobalWorkSize, rnd);"
    NL, "}"
    NL
};

void generate_reference_results_all_diff_1D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSize = (level == nestingLevel) ? len: (8 * 8 * 8);
    if((--level) < 0)
    {
        return;
    }

    for (size_t threadIdx = 0; threadIdx < globalSize; ++threadIdx)
    {
        ++referenceResults[threadIdx % len];
        if (level >= threadIdx)
        {
            generate_reference_results_all_diff_1D(referenceResults, len, level);
        }
    }
}

static int check_all_diff_1D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_diff_1D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_2D_wg_size_single[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 64, 64 * 64 };"
    NL, "  size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "  ls[1] = ls[1]? ls[1]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_2D(gs, ls);"
    NL, ""
    NL, "  // Only 1 work-item enqueues block"
    NL, "  if(tidX == 0 && tidY == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_2D_wg_size_single(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

static const char* enqueue_2D_wg_size_some_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 4, 4 };"
    NL, "  size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "  ls[1] = ls[1]? ls[1]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_2D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with the same level"
    NL, "  if((tidX < (get_global_size(0) >> 1)) && ((tidY < (get_global_size(1) >> 1)) || get_global_size(1) == 1))"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_2D_wg_size_some_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_eq_2D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 4;
    size_t globalSizeY = (level == nestingLevel) ? 1: 4;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
    {
        for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
        {
            if ((tidX < (globalSizeX >> 1)) && ((tidY < (globalSizeY >> 1)) || globalSizeY == 1))
            {
                ++referenceResults[(globalSizeX * tidY + tidX) % len];
                generate_reference_results_some_eq_2D(referenceResults, len, level);
            }
        }
    }
}

static int check_some_eq_2D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_eq_2D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_2D_wg_size_some_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 8, 8 };"
    NL, "  size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "  ls[1] = ls[1]? ls[1]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_2D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with different levels"
    NL, "  if((tidX % 2) == 0 && (tidY % 2) == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    if(level >= tidX && level >= tidY)"
    NL, "    {"
    NL, "      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_2D_wg_size_some_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_diff_2D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 8;
    size_t globalSizeY = (level == nestingLevel) ? 1: 8;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
    {
        for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
        {
            if ((tidX % 2) == 0 && (tidY % 2) == 0)
            {
                ++referenceResults[(globalSizeX * tidY + tidX) % len];
                if (level >= tidX && level >= tidY)
                {
                    generate_reference_results_some_diff_2D(referenceResults, len, level);
                }
            }
        }
    }
}

static int check_some_diff_2D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_diff_2D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_2D_wg_size_all_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 2, 2 };"
    NL, "  size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "  ls[1] = ls[1]? ls[1]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_2D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with the same level"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_2D_wg_size_all_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_eq_2D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 2;
    size_t globalSizeY = (level == nestingLevel) ? 1: 2;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
    {
        for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
        {
            ++referenceResults[(globalSizeX * tidY + tidX) % len];
            generate_reference_results_all_eq_2D(referenceResults, len, level);
        }
    }
}

static int check_all_eq_2D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_eq_2D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_2D_wg_size_all_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  size_t gs[] = { 8, 8 * 8 };"
    NL, "  size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "  ls[1] = ls[1]? ls[1]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_2D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with different levels"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  if(level >= tidX && level >= tidY)"
    NL, "  {"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_2D_wg_size_all_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_diff_2D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 8;
    size_t globalSizeY = (level == nestingLevel) ? 1: (8 * 8);
    if(--level < 0)
    {
        return;
    }

    for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
    {
        for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
        {
            ++referenceResults[(globalSizeX * tidY + tidX) % len];
            if (level >= tidX && level >= tidY)
            {
                generate_reference_results_all_diff_2D(referenceResults, len, level);
            }
        }
    }
}

static int check_all_diff_2D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_diff_2D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_3D_wg_size_single[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 64, 64, 64 };"
    NL, "  size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "  ls[2] = ls[2]? ls[2]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_3D(gs, ls);"
    NL, ""
    NL, "  // Only 1 work-item enqueues block"
    NL, "  if(tidX == 0 && tidY == 0 && tidZ == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_3D_wg_size_single(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

static const char* enqueue_3D_wg_size_some_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 4, 4, 4 };"
    NL, "  size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "  ls[2] = ls[2]? ls[2]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_3D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with the same level"
    NL, "  if((tidX < (get_global_size(0) >> 1)) && "
    NL, "    ((tidY < (get_global_size(1) >> 1)) || get_global_size(1) == 1) &&"
    NL, "    ((tidZ < (get_global_size(2) >> 1)) || get_global_size(2) == 1))"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_3D_wg_size_some_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_eq_3D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 4;
    size_t globalSizeY = (level == nestingLevel) ? 1: 4;
    size_t globalSizeZ = (level == nestingLevel) ? 1: 4;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                if ((tidX < (globalSizeX >> 1)) && ((tidY < (globalSizeY >> 1)) || globalSizeY == 1) && ((tidZ < (globalSizeZ >> 1)) || globalSizeZ == 1))
                {
                    ++referenceResults[(globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX) % len];
                    generate_reference_results_some_eq_3D(referenceResults, len, level);
                }
            }
        }
    }
}

static int check_some_eq_3D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_eq_3D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_3D_wg_size_some_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 8, 8, 8 };"
    NL, "  size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "  ls[2] = ls[2]? ls[2]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_3D(gs, ls);"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with different levels"
    NL, "  if((tidX % 2) == 0 && (tidY % 2) == 0 && (tidZ % 2) == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    if(level >= tidX && level >= tidY && level >= tidZ)"
    NL, "    {"
    NL, "      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_3D_wg_size_some_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_diff_3D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 8;
    size_t globalSizeY = (level == nestingLevel) ? 1: 8;
    size_t globalSizeZ = (level == nestingLevel) ? 1: 8;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                if ((tidX % 2) == 0 && (tidY % 2) == 0 && (tidZ % 2) == 0)
                {
                    ++referenceResults[(globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX) % len];
                    if (level >= tidX && level >= tidY && level >= tidZ)
                    {
                        generate_reference_results_some_diff_3D(referenceResults, len, level);
                    }
                }
            }
        }
    }
}

static int check_some_diff_3D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_diff_3D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_3D_wg_size_all_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 2, 2, 2 };"
    NL, "  size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "  ls[2] = ls[2]? ls[2]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_3D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with the same level"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_3D_wg_size_all_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_eq_3D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 2;
    size_t globalSizeY = (level == nestingLevel) ? 1: 2;
    size_t globalSizeZ = (level == nestingLevel) ? 1: 2;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                ++referenceResults[(globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX) % len];
                generate_reference_results_all_eq_3D(referenceResults, len, level);
            }
        }
    }
}

static int check_all_eq_3D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_eq_3D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_3D_wg_size_all_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  const size_t gs[] = { 8, 8, 8 };"
    NL, "  size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "  ls[2] = ls[2]? ls[2]: 1;"
    NL, "  "
    NL, "  ndrange_t ndrange = ndrange_3D(gs, ls);"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with different levels"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  if(level >= tidX && level >= tidY && level >= tidZ)"
    NL, "  {"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_3D_wg_size_all_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_diff_3D(std::vector<cl_int> &referenceResults, cl_int len, cl_int level)
{
    size_t globalSizeX = (level == nestingLevel) ? len: 8;
    size_t globalSizeY = (level == nestingLevel) ? 1: 8;
    size_t globalSizeZ = (level == nestingLevel) ? 1: 8;
    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                ++referenceResults[(globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX) % len];
                if (level >= tidX && level >= tidY && level >= tidZ)
                {
                    generate_reference_results_all_diff_3D(referenceResults, len, level);
                }
            }
        }
    }
}

static int check_all_diff_3D(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_diff_3D(referenceResults, len, nesting_level);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_mix_wg_size_single[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  ndrange_t ndrange;"
    NL, "  switch((linearId + level) % 3)"
    NL, "  {"
    NL, "    case 0:"
    NL, "      {"
    NL, "        const size_t gs = 64 * 64 * 64;"
    NL, "        size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "        ls = ls? ls: 1;"
    NL, "        ndrange = ndrange_1D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 1:"
    NL, "      {"
    NL, "        const size_t gs[] = { 64, 64 * 64 };"
    NL, "        size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "        ls[1] = ls[1]? ls[1]: 1;"
    NL, "        ndrange = ndrange_2D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 2:"
    NL, "      {"
    NL, "        const size_t gs[] = { 64, 64, 64 };"
    NL, "        size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "        ls[2] = ls[2]? ls[2]: 1;"
    NL, "        ndrange = ndrange_3D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    default:"
    NL, "      break;"
    NL, "  }"
    NL, ""
    NL, "  // Only 1 work-item enqueues block"
    NL, "  if(tidX == 0 && (tidY == 0 || get_global_size(1) == 1) && (tidZ == 0 || get_global_size(2) == 1))"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_mix_wg_size_single(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

static const char* enqueue_mix_wg_size_some_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  ndrange_t ndrange;"
    NL, "  switch((linearId + level) % 3)"
    NL, "  {"
    NL, "    case 0:"
    NL, "      {"
    NL, "        const size_t gs = 2 * 4 * 4;"
    NL, "        size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "        ls = ls? ls: 1;"
    NL, "        ndrange = ndrange_1D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 1:"
    NL, "      {"
    NL, "        const size_t gs[] = { 2, 4 * 4 };"
    NL, "        size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "        ls[1] = ls[1]? ls[1]: 1;"
    NL, "        ndrange = ndrange_2D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 2:"
    NL, "      {"
    NL, "        const size_t gs[] = { 2, 4, 4 };"
    NL, "        size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "        ls[2] = ls[2]? ls[2]: 1;"
    NL, "        ndrange = ndrange_3D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    default:"
    NL, "      break;"
    NL, "  }"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with the same level"
    NL, "  size_t globalSizeX = get_global_size(0);"
    NL, "  size_t globalSizeY = get_global_size(1);"
    NL, "  size_t globalSizeZ = get_global_size(2);"
    NL, "  if((tidX < (globalSizeX >> 1)) && ((tidY < (globalSizeY >> 1)) || globalSizeY == 1) && ((tidZ < (globalSizeZ >> 1)) || globalSizeZ == 1))"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_mix_wg_size_some_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_eq_mix(std::vector<cl_int> &referenceResults, cl_int len, cl_int level, cl_int dim)
{
    size_t globalSizeX = 1, globalSizeY = 1, globalSizeZ = 1;
    switch (dim)
    {
    case 0:
      globalSizeX = (level == nestingLevel) ? len: (2 * 4 * 4);
      break;
    case 1:
      globalSizeX = 2;
      globalSizeY = 4 * 4;
      break;
    case 2:
      globalSizeX = 2;
      globalSizeY = 4;
      globalSizeZ = 4;
      break;
    default:
      break;
    }

    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                size_t linearID = globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX;
                cl_int nextDim = (linearID + level) % 3;
                if ((tidX < (globalSizeX >> 1)) && ((tidY < (globalSizeY >> 1)) || globalSizeY == 1) && ((tidZ < (globalSizeZ >> 1)) || globalSizeZ == 1))
                {
                    ++referenceResults[linearID % len];
                    generate_reference_results_some_eq_mix(referenceResults, len, level, nextDim);
                }
            }
        }
    }
}

static int check_some_eq_mix(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_eq_mix(referenceResults, len, nesting_level, 0);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_mix_wg_size_some_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  ndrange_t ndrange;"
    NL, "  switch((linearId + level) % 3)"
    NL, "  {"
    NL, "    case 0:"
    NL, "      {"
    NL, "        const size_t gs = 8 * 8 * 8;"
    NL, "        size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "        ls = ls? ls: 1;"
    NL, "        ndrange = ndrange_1D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 1:"
    NL, "      {"
    NL, "        const size_t gs[] = { 8, 8 * 8 };"
    NL, "        size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "        ls[1] = ls[1]? ls[1]: 1;"
    NL, "        ndrange = ndrange_2D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 2:"
    NL, "      {"
    NL, "        const size_t gs[] = { 8, 8, 8 };"
    NL, "        size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "        ls[2] = ls[2]? ls[2]: 1;"
    NL, "        ndrange = ndrange_3D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    default:"
    NL, "      break;"
    NL, "  }"
    NL, ""
    NL, "  // Some work-items enqueues nested blocks with different levels"
    NL, "  if((tidX % 2) == 0 && (tidY % 2) == 0 && (tidZ % 2) == 0)"
    NL, "  {"
    NL, "    atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "    if(level >= tidX && level >= tidY && level >= tidZ)"
    NL, "    {"
    NL, "      int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_mix_wg_size_some_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_some_diff_mix(std::vector<cl_int> &referenceResults, cl_int len, cl_int level, cl_int dim)
{
    size_t globalSizeX = 1, globalSizeY = 1, globalSizeZ = 1;
    switch (dim)
    {
    case 0:
      globalSizeX = (level == nestingLevel) ? len: (8 * 8 * 8);
      break;
    case 1:
      globalSizeX = 8;
      globalSizeY = 8 * 8;
      break;
    case 2:
      globalSizeX = 8;
      globalSizeY = 8;
      globalSizeZ = 8;
      break;
    default:
      return;
      break;
    }

    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                size_t linearID = globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX;
                cl_int nextDim = (linearID + level) % 3;
                if ((tidX % 2) == 0 && (tidY % 2) == 0 && (tidZ % 2) == 0)
                {
                    ++referenceResults[linearID % len];
                    if (level >= tidX && level >= tidY && level >= tidZ)
                    {
                        generate_reference_results_some_diff_mix(referenceResults, len, level, nextDim);
                    }
                }
            }
        }
    }
}

static int check_some_diff_mix(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_some_diff_mix(referenceResults, len, nesting_level, 0);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_mix_wg_size_all_eq[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  ndrange_t ndrange;"
    NL, "  switch((linearId + level) % 3)"
    NL, "  {"
    NL, "    case 0:"
    NL, "      {"
    NL, "        const size_t gs = 2 * 2 * 2;"
    NL, "        size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "        ls = ls? ls: 1;"
    NL, "        ndrange = ndrange_1D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 1:"
    NL, "      {"
    NL, "        const size_t gs[] = { 2, 2 * 2 };"
    NL, "        size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "        ls[1] = ls[1]? ls[1]: 1;"
    NL, "        ndrange = ndrange_2D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 2:"
    NL, "      {"
    NL, "        const size_t gs[] = { 2, 2, 2 };"
    NL, "        size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "        ls[2] = ls[2]? ls[2]: 1;"
    NL, "        ndrange = ndrange_3D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    default:"
    NL, "      break;"
    NL, "  }"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with the same level"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "  if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_mix_wg_size_all_eq(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_eq_mix(std::vector<cl_int> &referenceResults, cl_int len, cl_int level, cl_int dim)
{
    size_t globalSizeX = 1, globalSizeY = 1, globalSizeZ = 1;
    switch (dim)
    {
    case 0:
      globalSizeX = (level == nestingLevel) ? len: (2 * 2 * 2);
      break;
    case 1:
      globalSizeX = 2;
      globalSizeY = 2 * 2;
      break;
    case 2:
      globalSizeX = 2;
      globalSizeY = 2;
      globalSizeZ = 2;
      break;
    default:
      break;
    }

    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                size_t linearID = globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX;
                cl_int nextDim = (linearID + level) % 3;
                ++referenceResults[linearID % len];
                generate_reference_results_all_eq_mix(referenceResults, len, level, nextDim);
            }
        }
    }
}

static int check_all_eq_mix(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_eq_mix(referenceResults, len, nesting_level, 0);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const char* enqueue_mix_wg_size_all_diff[] =
{
    NL, "void block_fn(int level, int maxGlobalWorkSize, __global int* rnd, __global int* res)"
    NL, "{"
    NL, "  queue_t def_q = get_default_queue();"
    NL, "  size_t tidX = get_global_id(0);"
    NL, "  size_t tidY = get_global_id(1);"
    NL, "  size_t tidZ = get_global_id(2);"
    NL, "  size_t linearId = get_global_linear_id();"
    NL, "  if(--level < 0) return;"
    NL, ""
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(level, maxGlobalWorkSize, rnd, res); };"
    NL, "  uint wg = get_kernel_work_group_size(kernelBlock);"
    NL, ""
    NL, "  ndrange_t ndrange;"
    NL, "  switch((linearId + level) % 3)"
    NL, "  {"
    NL, "    case 0:"
    NL, "      {"
    NL, "        const size_t gs = 8 * 8 * 8;"
    NL, "        size_t ls = rnd[tidX % maxGlobalWorkSize] % wg % gs;"
    NL, "        ls = ls? ls: 1;"
    NL, "        ndrange = ndrange_1D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 1:"
    NL, "      {"
    NL, "        const size_t gs[] = { 8, 8 * 8 };"
    NL, "        size_t ls[] = { 1, rnd[tidY % maxGlobalWorkSize] % wg % gs[1] };"
    NL, "        ls[1] = ls[1]? ls[1]: 1;"
    NL, "        ndrange = ndrange_2D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    case 2:"
    NL, "      {"
    NL, "        const size_t gs[] = { 8, 8, 8 };"
    NL, "        size_t ls[] = { 1, 1, rnd[tidZ % maxGlobalWorkSize] % wg % gs[2] };"
    NL, "        ls[2] = ls[2]? ls[2]: 1;"
    NL, "        ndrange = ndrange_3D(gs, ls);"
    NL, "      }"
    NL, "      break;"
    NL, "    default:"
    NL, "      break;"
    NL, "  }"
    NL, ""
    NL, "  // All work-items enqueues nested blocks with different levels"
    NL, "  atomic_inc(&res[linearId % maxGlobalWorkSize]);"
    NL, "  if(level >= tidX && level >= tidY && level >= tidZ)"
    NL, "  {"
    NL, "    int enq_res = enqueue_kernel(def_q, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[linearId % maxGlobalWorkSize] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_mix_wg_size_all_diff(__global int* res, int level, int maxGlobalWorkSize, __global int* rnd)"
    NL, "{"
    NL, "  block_fn(level, maxGlobalWorkSize, rnd, res);"
    NL, "}"
    NL
};

void generate_reference_results_all_diff_mix(std::vector<cl_int> &referenceResults, cl_int len, cl_int level, cl_int dim)
{
    size_t globalSizeX = 1, globalSizeY = 1, globalSizeZ = 1;
    switch (dim)
    {
    case 0:
      globalSizeX = (level == nestingLevel) ? len: (8 * 8 * 8);
      break;
    case 1:
      globalSizeX = 8;
      globalSizeY = 8 * 8;
      break;
    case 2:
      globalSizeX = 8;
      globalSizeY = 8;
      globalSizeZ = 8;
      break;
    default:
      break;
    }

    if(--level < 0)
    {
        return;
    }

    for (size_t tidZ = 0; tidZ < globalSizeZ; ++tidZ)
    {
        for (size_t tidY = 0; tidY < globalSizeY; ++tidY)
        {
            for (size_t tidX = 0; tidX < globalSizeX; ++tidX)
            {
                size_t linearID = globalSizeX * globalSizeY * tidZ + globalSizeX * tidY + tidX;
                cl_int nextDim = (linearID + level) % 3;
                ++referenceResults[linearID % len];
                if (level >= tidX && level >= tidY && level >= tidZ)
                {
                    generate_reference_results_all_diff_mix(referenceResults, len, level, nextDim);
                }
            }
        }
    }
}

static int check_all_diff_mix(cl_int* results, cl_int len, cl_int nesting_level)
{
    std::vector<cl_int> referenceResults(len, 0);
    generate_reference_results_all_diff_mix(referenceResults, len, nesting_level, 0);

    for(size_t i = 0; i < len; ++i)
    {
        if (results[i] != referenceResults[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d, index: %zu\n",
                      results[i], referenceResults[i], i);
            return (int)i;
        }
    }

    return -1;
}

static const kernel_src_check sources_enqueue_wg_size[] =
{
    { KERNEL(enqueue_1D_wg_size_single), check_single },
    { KERNEL(enqueue_1D_wg_size_some_eq), check_some_eq_1D },
    { KERNEL(enqueue_1D_wg_size_some_diff), check_some_diff_1D },
    { KERNEL(enqueue_1D_wg_size_all_eq), check_all_eq_1D },
    { KERNEL(enqueue_1D_wg_size_all_diff), check_all_diff_1D },

    { KERNEL(enqueue_2D_wg_size_single), check_single },
    { KERNEL(enqueue_2D_wg_size_some_eq), check_some_eq_2D },
    { KERNEL(enqueue_2D_wg_size_some_diff), check_some_diff_2D },
    { KERNEL(enqueue_2D_wg_size_all_eq), check_all_eq_2D },
    { KERNEL(enqueue_2D_wg_size_all_diff), check_all_diff_2D },

    { KERNEL(enqueue_3D_wg_size_single), check_single },
    { KERNEL(enqueue_3D_wg_size_some_eq), check_some_eq_3D },
    { KERNEL(enqueue_3D_wg_size_some_diff), check_some_diff_3D },
    { KERNEL(enqueue_3D_wg_size_all_eq), check_all_eq_3D },
    { KERNEL(enqueue_3D_wg_size_all_diff), check_all_diff_3D },

    { KERNEL(enqueue_mix_wg_size_single), check_single },
    { KERNEL(enqueue_mix_wg_size_some_eq), check_some_eq_mix },
    { KERNEL(enqueue_mix_wg_size_some_diff), check_some_diff_mix },
    { KERNEL(enqueue_mix_wg_size_all_eq), check_all_eq_mix },
    { KERNEL(enqueue_mix_wg_size_all_diff), check_all_diff_mix }
};

int test_enqueue_wg_size(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    MTdata d;
    cl_uint i, k;
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    const cl_int MAX_GLOBAL_WORK_SIZE = MAX_GWS / 4;
    cl_int kernel_results[MAX_GLOBAL_WORK_SIZE] = { 0 };
    cl_uint vrnd[MAX_GLOBAL_WORK_SIZE] = { 0 };

    size_t ret_len;
    cl_uint max_queues = 1;
    cl_uint maxQueueSize = 0;
    d = init_genrand(gRandomSeed);

    if(gWimpyMode)
    {
        nestingLevel = 2;
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
    }

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
    test_error(err_ret,
               "clCreateCommandQueueWithProperties(CL_QUEUE_ON_DEVICE | "
               "CL_QUEUE_ON_DEVICE_DEFAULT) failed");


    size_t failCnt = 0;
    for(k = 0; k < arr_size(sources_enqueue_wg_size); ++k)
    {
        if (!gKernelName.empty() && gKernelName != sources_enqueue_wg_size[k].src.kernel_name)
            continue;

        log_info("Running '%s' kernel (%d of %zu) ...\n",
                 sources_enqueue_wg_size[k].src.kernel_name, k + 1,
                 arr_size(sources_enqueue_wg_size));
        for(i = 0; i < MAX_GLOBAL_WORK_SIZE; ++i)
        {
            kernel_results[i] = 0;
            vrnd[i] = genrand_int32(d);
        }

        // Fill some elements with prime numbers
        cl_uint prime[] = { 3,   5,   7,  11,  13,  17,  19,  23,
            29,  31,  37,  41,  43,  47,  53,  59,
            61,  67,  71,  73,  79,  83,  89,  97,
            101, 103, 107, 109, 113, 127 };

        for(i = 0; i < arr_size(prime); ++i)
        {
            vrnd[genrand_int32(d) % MAX_GLOBAL_WORK_SIZE] = prime[i];
        }

        clMemWrapper mem;
        mem = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(vrnd), vrnd, &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        kernel_arg args[] =
        {
            { sizeof(cl_uint), &nestingLevel },
            { sizeof(cl_uint), &MAX_GLOBAL_WORK_SIZE },
            { sizeof(cl_mem),  &mem }
        };

        size_t global_size = MAX_GLOBAL_WORK_SIZE;
        size_t local_size = (max_local_size > global_size) ? global_size : max_local_size;

        err_ret = run_n_kernel_args(context, queue, sources_enqueue_wg_size[k].src.lines, sources_enqueue_wg_size[k].src.num_lines, sources_enqueue_wg_size[k].src.kernel_name, local_size, global_size, kernel_results, sizeof(kernel_results), arr_size(args), args);

        //check results
        int fail = sources_enqueue_wg_size[k].check(kernel_results, global_size, nestingLevel);

        if(check_error(err_ret, "'%s' kernel execution failed", sources_enqueue_wg_size[k].src.kernel_name)) { ++failCnt; res = -1; continue; }
        else if(fail >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d]", sources_enqueue_wg_size[k].src.kernel_name, fail)) { ++failCnt; res = -1; continue; }
        else log_info("'%s' kernel is OK.\n", sources_enqueue_wg_size[k].src.kernel_name);
    }

    if (failCnt > 0)
    {
        log_error("ERROR: %zu of %zu kernels failed.\n", failCnt,
                  arr_size(sources_enqueue_wg_size));
    }

    free_mtdata(d);

    return res;
}

#endif

