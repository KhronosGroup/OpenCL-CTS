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
static const char* helper_ndrange_1d_glo[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_1d_glo(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global atomic_uint* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int i = 0; i < n; i++)"
    NL, "  {"
    NL, "    ndrange_t ndrange = ndrange_1D(glob_size_arr[i]);"
    NL, "    int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL
};

static const char* helper_ndrange_1d_loc[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_1d_loc(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global atomic_uint* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int k = 0; k < n; k++)"
    NL, "  {"
    NL, "    for(int i = 0; i < n; i++)"
    NL, "    {"
    NL, "      if (glob_size_arr[i] >= loc_size_arr[k])"
    NL, "      {"
    NL, "        ndrange_t ndrange = ndrange_1D(glob_size_arr[i], loc_size_arr[k]);"
    NL, "        int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "        if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};

static const char* helper_ndrange_1d_ofs[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[(get_global_offset(0) + get_global_linear_id()) % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_1d_ofs(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global atomic_uint* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int l = 0; l < n; l++)"
    NL, "  {"
    NL, "    for(int k = 0; k < n; k++)"
    NL, "    {"
    NL, "      for(int i = 0; i < n; i++)"
    NL, "      {"
    NL, "        if (glob_size_arr[i] >= loc_size_arr[k])"
    NL, "        {"
    NL, "          ndrange_t ndrange = ndrange_1D(ofs_arr[l], glob_size_arr[i], loc_size_arr[k]);"
    NL, "          int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "          if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};

static const char* helper_ndrange_2d_glo[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_2d_glo(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int i = 0; i < n; i++)"
    NL, "  {"
    NL, "    size_t glob_size[2] = { glob_size_arr[i], glob_size_arr[(i + 1) % n] };"
    NL, "    ndrange_t ndrange = ndrange_2D(glob_size);"
    NL, "    int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "    if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "  }"
    NL, "}"
    NL
};

static const char* helper_ndrange_2d_loc[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_2d_loc(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int k = 0; k < n; k++)"
    NL, "  {"
    NL, "    for(int i = 0; i < n; i++)"
    NL, "    {"
    NL, "      if (glob_size_arr[(i + 1) % n] >= loc_size_arr[k])"
    NL, "      {"
    NL, "        size_t glob_size[] = { glob_size_arr[i], glob_size_arr[(i + 1) % n] };"
    NL, "        size_t loc_size[] = { 1, loc_size_arr[k] };"
    NL, ""
    NL, "        ndrange_t ndrange = ndrange_2D(glob_size, loc_size);"
    NL, "        int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "        if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};


static const char* helper_ndrange_2d_ofs[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[(get_global_offset(1) * get_global_size(0) + get_global_offset(0) + get_global_linear_id()) % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_2d_ofs(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int l = 0; l < n; l++)"
    NL, "  {"
    NL, "    for(int k = 0; k < n; k++)"
    NL, "    {"
    NL, "      for(int i = 0; i < n; i++)"
    NL, "      {"
    NL, "        if (glob_size_arr[(i + 1) % n] >= loc_size_arr[k])"
    NL, "        {"
    NL, "          size_t glob_size[] = { glob_size_arr[i], glob_size_arr[(i + 1) % n]};"
    NL, "          size_t loc_size[] = { 1, loc_size_arr[k] };"
    NL, "          size_t ofs[] = { ofs_arr[l], ofs_arr[(l + 1) % n] };"
    NL, ""
    NL, "          ndrange_t ndrange = ndrange_2D(ofs,glob_size,loc_size);"
    NL, "          int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "          if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};


static const char* helper_ndrange_3d_glo[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_3d_glo(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int i = 0; i < n; i++)"
    NL, "  {"
    NL, "    uint global_work_size = glob_size_arr[i] *  glob_size_arr[(i + 1) % n] * glob_size_arr[(i + 2) % n];"
    NL, "    if (global_work_size <= (len * len))"
    NL, "    {"
    NL, "      size_t glob_size[3] = { glob_size_arr[i], glob_size_arr[(i + 1) % n], glob_size_arr[(i + 2) % n] };"
    NL, "      ndrange_t ndrange = ndrange_3D(glob_size);"
    NL, "      int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};


static const char* helper_ndrange_3d_loc[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[get_global_linear_id() % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_3d_loc(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int k = 0; k < n; k++)"
    NL, "  {"
    NL, "    for(int i = 0; i < n; i++)"
    NL, "    {"
    NL, "      uint global_work_size = glob_size_arr[i] *  glob_size_arr[(i + 1) % n] * glob_size_arr[(i + 2) % n];"
    NL, "      if (glob_size_arr[(i + 2) % n] >= loc_size_arr[k] && global_work_size <= (len * len))"
    NL, "      {"
    NL, "        size_t glob_size[] = { glob_size_arr[i], glob_size_arr[(i + 1) % n], glob_size_arr[(i + 2) % n] };"
    NL, "        size_t loc_size[] = { 1, 1, loc_size_arr[k] };"
    NL, "        ndrange_t ndrange = ndrange_3D(glob_size,loc_size);"
    NL, "        int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "      "
    NL, "        if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};

static const char* helper_ndrange_3d_ofs[] =
{
    NL, "void block_fn(int len, __global atomic_uint* val)"
    NL, "{"
    NL, "  atomic_fetch_add_explicit(&val[(get_global_offset(2) * get_global_size(0) * get_global_size(1) + get_global_offset(1) * get_global_size(0) + get_global_offset(0) + get_global_linear_id()) % len], 1, memory_order_relaxed, memory_scope_device);"
    NL, "}"
    NL, ""
    NL, "kernel void helper_ndrange_3d_ofs(__global int* res, uint n, uint len, __global uint* glob_size_arr, __global uint* loc_size_arr, __global int* val,  __global uint* ofs_arr)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  void (^kernelBlock)(void) = ^{ block_fn(len, val); };"
    NL, ""
    NL, "  for(int l = 0; l < n; l++)"
    NL, "  {"
    NL, "    for(int k = 0; k < n; k++)"
    NL, "    {"
    NL, "      for(int i = 0; i < n; i++)"
    NL, "      {"
    NL, "        uint global_work_size = glob_size_arr[i] *  glob_size_arr[(i + 1) % n] * glob_size_arr[(i + 2) % n];"
    NL, "        if (glob_size_arr[(i + 2) % n] >= loc_size_arr[k] && global_work_size <= (len * len))"
    NL, "        {"
    NL, "          size_t glob_size[3] = { glob_size_arr[i], glob_size_arr[(i + 1) % n], glob_size_arr[(i + 2) % n]};"
    NL, "          size_t loc_size[3] = { 1, 1, loc_size_arr[k] };"
    NL, "          size_t ofs[3] = { ofs_arr[l], ofs_arr[(l + 1) % n], ofs_arr[(l + 2) % n] };"
    NL, "          ndrange_t ndrange = ndrange_3D(ofs,glob_size,loc_size);"
    NL, "          int enq_res = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, kernelBlock);"
    NL, "          if(enq_res != CLK_SUCCESS) { res[tid] = -1; return; }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL
};

static const kernel_src_dim_check sources_ndrange_Xd[] =
{
    { KERNEL(helper_ndrange_1d_glo), 1, CL_FALSE, CL_FALSE},
    { KERNEL(helper_ndrange_1d_loc), 1, CL_TRUE, CL_FALSE},
    { KERNEL(helper_ndrange_1d_ofs), 1, CL_TRUE, CL_TRUE},
    { KERNEL(helper_ndrange_2d_glo), 2, CL_FALSE, CL_FALSE},
    { KERNEL(helper_ndrange_2d_loc), 2, CL_TRUE, CL_FALSE},
    { KERNEL(helper_ndrange_2d_ofs), 2, CL_TRUE, CL_TRUE},
    { KERNEL(helper_ndrange_3d_glo), 3, CL_FALSE, CL_FALSE},
    { KERNEL(helper_ndrange_3d_loc), 3, CL_TRUE, CL_FALSE},
    { KERNEL(helper_ndrange_3d_ofs), 3, CL_TRUE, CL_TRUE},
};
static const size_t num_kernels_ndrange_Xd = arr_size(sources_ndrange_Xd);

static int check_kernel_results(cl_int* results, cl_int len)
{
    for(cl_int i = 0; i < len; ++i)
    {
        if(results[i] != 0) return i;
    }
    return -1;
}

void generate_reference_1D(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr)
{
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t w = 0; w < glob_size_arr[g]; ++w)
        {
            ++reference_results[w];
        }
    }
}

void generate_reference_1D_local(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr)
{
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            if (glob_size_arr[g] >= loc_size_arr[l])
            {
                for (size_t w = 0; w < glob_size_arr[g]; ++w)
                {
                    ++reference_results[w];
                }
            }
        }
    }
}

void generate_reference_1D_offset(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, std::vector<cl_uint> &offset, cl_uint len)
{
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            if (glob_size_arr[g] >= loc_size_arr[l])
            {
                for (size_t o = 0; o < offset.size(); ++o)
                {
                    for (size_t w = 0; w < glob_size_arr[g]; ++w)
                    {
                        ++reference_results[(offset[o] + w) % len];
                    }
                }
            }
        }
    }
}

void generate_reference_2D(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, cl_uint len)
{
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t h = 0; h < glob_size_arr[(g + 1) % glob_size_arr.size()]; ++h)
        {
            for (size_t w = 0; w < glob_size_arr[g]; ++w)
            {
                ++reference_results[(h * glob_size_arr[g] + w) % len];
            }
        }
    }
}

void generate_reference_2D_local(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, cl_uint len)
{
    size_t n = glob_size_arr.size();
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            if (glob_size_arr[(g + 1) % n] >= loc_size_arr[l])
            {
                for (size_t h = 0; h < glob_size_arr[(g + 1) % n]; ++h)
                {
                    for (size_t w = 0; w < glob_size_arr[g]; ++w)
                    {
                        ++reference_results[(h * glob_size_arr[g] + w) % len];
                    }
                }
            }
        }
    }
}

void generate_reference_2D_offset(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, std::vector<cl_uint> &offset, cl_uint len)
{
    size_t n = glob_size_arr.size();
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            if (glob_size_arr[(g + 1) % n] >= loc_size_arr[l])
            {
                for (size_t o = 0; o < offset.size(); ++o)
                {
                    for (size_t h = 0; h < glob_size_arr[(g + 1) % n]; ++h)
                    {
                        for (size_t w = 0; w < glob_size_arr[g]; ++w)
                        {
                            ++reference_results[(glob_size_arr[g] * offset[(o + 1) % n] + offset[o] + h * glob_size_arr[g] + w) % len];
                        }
                    }
                }
            }
        }
    }
}

void generate_reference_3D(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, cl_uint len)
{
    size_t n = glob_size_arr.size();
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        size_t global_work_size = glob_size_arr[(g + 2) % n] * glob_size_arr[(g + 1) % n] * glob_size_arr[g];
        if(global_work_size <= (len * len))
        {
            for (size_t d = 0; d < glob_size_arr[(g + 2) % n]; ++d)
            {
                for (size_t h = 0; h < glob_size_arr[(g + 1) % n]; ++h)
                {
                    for (size_t w = 0; w < glob_size_arr[g]; ++w)
                    {
                        ++reference_results[(d * glob_size_arr[(g + 1) % n] * glob_size_arr[g] + h * glob_size_arr[g] + w) % len];
                    }
                }
            }
        }
    }
}

void generate_reference_3D_local(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, cl_uint len)
{
    size_t n = glob_size_arr.size();
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            size_t global_work_size = glob_size_arr[(g + 2) % n] * glob_size_arr[(g + 1) % n] * glob_size_arr[g];
            if (glob_size_arr[(g + 2) % n] >= loc_size_arr[l] && global_work_size <= (len * len))
            {
                for (size_t d = 0; d < glob_size_arr[(g + 2) % n]; ++d)
                {
                    for (size_t h = 0; h < glob_size_arr[(g + 1) % n]; ++h)
                    {
                        for (size_t w = 0; w < glob_size_arr[g]; ++w)
                        {
                            ++reference_results[(d * glob_size_arr[(g + 1) % n] * glob_size_arr[g] + h * glob_size_arr[g] + w) % len];
                        }
                    }
                }
            }
        }
    }
}

void generate_reference_3D_offset(std::vector<cl_int> &reference_results, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, std::vector<cl_uint> &offset, cl_uint len)
{
    size_t n = glob_size_arr.size();
    for (size_t g = 0; g < glob_size_arr.size(); ++g)
    {
        for (size_t l = 0; l < loc_size_arr.size(); ++l)
        {
            size_t global_work_size = glob_size_arr[(g + 2) % n] * glob_size_arr[(g + 1) % n] * glob_size_arr[g];
            if (glob_size_arr[(g + 2) % n] >= loc_size_arr[l] && global_work_size <= (len * len))
            {
                for (size_t o = 0; o < offset.size(); ++o)
                {
                    for (size_t d = 0; d < glob_size_arr[(g + 2) % n]; ++d)
                    {
                        for (size_t h = 0; h < glob_size_arr[(g + 1) % n]; ++h)
                        {
                            for (size_t w = 0; w < glob_size_arr[g]; ++w)
                            {
                                ++reference_results[(glob_size_arr[g] * glob_size_arr[(g + 1) % n] * offset[(o + 2) % n] + glob_size_arr[g] * offset[(o + 1) % n] + offset[o] + d * glob_size_arr[(g + 1) % n] * glob_size_arr[g] + h * glob_size_arr[g] + w) % len];
                            }
                        }
                    }
                }
            }
        }
    }
}

static int check_kernel_results(cl_int* results, cl_int len, std::vector<cl_uint> &glob_size_arr, std::vector<cl_uint> &loc_size_arr, std::vector<cl_uint> &offset, cl_int dim, cl_bool use_local, cl_bool use_offset)
{
    std::vector<cl_int> reference_results(len, 0);
    switch (dim)
    {
    case 1:
        if (use_local == CL_FALSE)
        {
            generate_reference_1D(reference_results, glob_size_arr);
        }
        else if(use_local == CL_TRUE && use_offset == CL_FALSE)
        {
            generate_reference_1D_local(reference_results, glob_size_arr, loc_size_arr);
        }
        else
        {
            generate_reference_1D_offset(reference_results, glob_size_arr, loc_size_arr, offset, len);
        }
        break;
    case 2:
        if (use_local == CL_FALSE)
        {
            generate_reference_2D(reference_results, glob_size_arr, len);
        }
        else if (use_local == CL_TRUE && use_offset == CL_FALSE)
        {
            generate_reference_2D_local(reference_results, glob_size_arr, loc_size_arr, len);
        }
        else
        {
            generate_reference_2D_offset(reference_results, glob_size_arr, loc_size_arr, offset, len);
        }
        break;
    case 3:
        if (use_local == CL_FALSE)
        {
            generate_reference_3D(reference_results, glob_size_arr, len);
        }
        else if (use_local == CL_TRUE && use_offset == CL_FALSE)
        {
            generate_reference_3D_local(reference_results, glob_size_arr, loc_size_arr, len);
        }
        else
        {
            generate_reference_3D_offset(reference_results, glob_size_arr, loc_size_arr, offset, len);
        }
        break;
    default:
        return 0;
        break;
    }

    for (cl_int i = 0; i < len; ++i)
    {
        if (results[i] != reference_results[i])
        {
            log_error("ERROR: Kernel returned %d vs. expected %d\n", results[i], reference_results[i]);
            return i;
        }
    }

    return -1;
}

int test_enqueue_ndrange(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    MTdata d;
    cl_uint i;
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    cl_int k, kernel_results[MAX_GWS] = { 0 };

    size_t ret_len;
    cl_uint max_queues = 1;
    cl_uint maxQueueSize = 0;

    d = init_genrand(gRandomSeed);

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

    max_local_size = (max_local_size > MAX_GWS)? MAX_GWS: max_local_size;
    if(gWimpyMode)
    {
        max_local_size = MIN(8, max_local_size);
    }

    cl_uint num = 10;
    cl_uint global_work_size = max_local_size * 2;
    std::vector<cl_uint> glob_size_arr(num);
    std::vector<cl_uint> loc_size_arr(num);
    std::vector<cl_uint> ofs_arr(num);
    std::vector<cl_int> glob_results(global_work_size, 0);

    glob_size_arr[0] = 1;
    glob_size_arr[1] = global_work_size;
    loc_size_arr[0] = 1;
    loc_size_arr[1] = max_local_size;
    ofs_arr[0] = 0;
    ofs_arr[1] = 1;

    for(i = 2; i < num; ++i)
    {
        glob_size_arr[i] = genrand_int32(d) % global_work_size;
        glob_size_arr[i] = glob_size_arr[i] ? glob_size_arr[i]: 1;
        loc_size_arr[i] = genrand_int32(d) % max_local_size;
        loc_size_arr[i] = loc_size_arr[i] ? loc_size_arr[i]: 1;
        ofs_arr[i] = genrand_int32(d) % global_work_size;
    }

    // check ndrange_dX functions
    size_t failCnt = 0;
    for(i = 0; i < num_kernels_ndrange_Xd; ++i)
    {
        if (!gKernelName.empty() && gKernelName != sources_ndrange_Xd[i].src.kernel_name)
            continue;

        clMemWrapper mem1 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, glob_size_arr.size() * sizeof(cl_uint), &glob_size_arr[0], &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");
        clMemWrapper mem2 = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, loc_size_arr.size() * sizeof(cl_uint), &loc_size_arr[0], &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");
        clMemWrapper mem3 = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, glob_results.size() * sizeof(cl_int), &glob_results[0], &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");
        clMemWrapper mem4 = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, ofs_arr.size() * sizeof(cl_uint), &ofs_arr[0], &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        kernel_arg args[] =
        {
            { sizeof(cl_uint), &num },
            { sizeof(cl_uint), &global_work_size },
            { sizeof(cl_mem), &mem1 },
            { sizeof(cl_mem), &mem2 },
            { sizeof(cl_mem), &mem3 },
            { sizeof(cl_mem), &mem4 },
        };

        log_info("Running '%s' kernel (%d of %d) ...\n",  sources_ndrange_Xd[i].src.kernel_name, i + 1, num_kernels_ndrange_Xd);
        err_ret = run_single_kernel_args(context, queue, sources_ndrange_Xd[i].src.lines, sources_ndrange_Xd[i].src.num_lines, sources_ndrange_Xd[i].src.kernel_name, kernel_results, sizeof(kernel_results), arr_size(args), args);

        cl_int *ptr = (cl_int *)clEnqueueMapBuffer(queue, mem3, CL_TRUE, CL_MAP_READ, 0, glob_results.size() * sizeof(cl_int), 0, 0, 0, &err_ret);
        test_error(err_ret, "clEnqueueMapBuffer() failed");

        if(check_error(err_ret, "'%s' kernel execution failed", sources_ndrange_Xd[i].src.kernel_name)) { ++failCnt; res = -1; }
        else if((k = check_kernel_results(kernel_results, arr_size(kernel_results))) >= 0 && check_error(-1, "'%s' kernel results validation failed: [%d] returned %d expected 0", sources_ndrange_Xd[i].src.kernel_name, k, kernel_results[k])) res = -1;
        else if((k = check_kernel_results(ptr, global_work_size, glob_size_arr, loc_size_arr, ofs_arr, sources_ndrange_Xd[i].dim, sources_ndrange_Xd[i].localSize, sources_ndrange_Xd[i].offset)) >= 0 && check_error(-1, "'%s' global kernel results validation failed: [%d] returned %d expected 0", sources_ndrange_Xd[i].src.kernel_name, k, glob_results[k])) res = -1;
        else log_info("'%s' kernel is OK.\n", sources_ndrange_Xd[i].src.kernel_name);

        err_ret = clEnqueueUnmapMemObject(queue, mem3, ptr, 0, 0, 0);
        test_error(err_ret, "clEnqueueUnmapMemObject() failed");

    }

    if (failCnt > 0)
    {
        log_error("ERROR: %d of %d kernels failed.\n", failCnt, num_kernels_ndrange_Xd);
    }

    return res;
}


#endif

