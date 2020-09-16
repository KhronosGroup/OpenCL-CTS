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
#define BITS_DEPTH 28

static const char* enqueue_flags_wait_kernel_simple[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, size_t gs, __global int* res)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH)"
    NL, "    {"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), "
    NL, "      ^{"
    NL, "         block_fn(array, index + 1, ls, gs, res);"
    NL, "       });"
    NL, "    }"
    NL, "  }"
    NL, ""
    NL, "  array[index * gs + tid] = array[(index - 1) * gs + tid] + 1;"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0)"
    NL, "    {"
    NL, "      size_t gid = get_group_id(0);"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_kernel_simple(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls  = get_local_size(0);"
    NL, "  size_t gs  = get_global_size(0);"
    NL, "  size_t tid  = get_global_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), "
    NL, "    ^{"
    NL, "       block_fn(array, 1, ls, gs, res);"
    NL, "     });"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_kernel_event[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, size_t gs, __global int* res)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH)"
    NL, "    {"
    NL, "      clk_event_t block_evt;"
    NL, "      clk_event_t user_evt = create_user_event();"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "      ^{"
    NL, "         block_fn(array, index + 1, ls, gs, res);"
    NL, "       });"
    NL, "      set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "      release_event(user_evt);"
    NL, "      release_event(block_evt);"
    NL, "    }"
    NL, "  }"
    NL, ""
    NL, "  array[index * gs + tid] = array[(index - 1) * gs + tid] + 1;"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0)"
    NL, "    {"
    NL, "      size_t gid = get_group_id(0);"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_kernel_event(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t tid  = get_global_id(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t ls = get_local_size(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    clk_event_t block_evt;"
    NL, "    clk_event_t user_evt = create_user_event();"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "    ^{"
    NL, "       block_fn(array, 1, ls, gs, res);"
    NL, "     });"
    NL, "    set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "    release_event(user_evt);"
    NL, "    release_event(block_evt);"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_kernel_local[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, size_t gs, __global int* res, __local int* sub_array)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  sub_array[lid] = array[(index - 1) * gs + tid];"
    NL, "  barrier(CLK_LOCAL_MEM_FENCE);"
    NL, ""
    NL, "  for(int i = 0; i < ls; i++)"
    NL, "  {"
    NL, "    int id = gid * ls + i;"
    NL, "    val += sub_array[i];"
    NL, "    val -= (tid == id)? 0: (id + index - 1);"
    NL, "  }"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH)"
    NL, "    {"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), "
    NL, "      ^(__local void* sub_array){"
    NL, "        block_fn(array, index + 1, ls, gs, res, sub_array);"
    NL, "      }, ls * sizeof(int));"
    NL, "    }"
    NL, "  }"
    NL, ""
    NL, "  array[index * gs + tid] = val + 1;"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_kernel_local(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls  = get_local_size(0);"
    NL, "  size_t gs  = get_global_size(0);"
    NL, "  size_t tid  = get_global_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), "
    NL, "    ^(__local void* sub_array){"
    NL, "      block_fn(array, 1, ls, gs, res, sub_array);"
    NL, "    }, ls * sizeof(int));"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_kernel_event_local[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, size_t gs, __global int* res, __local int* sub_array)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, ""
    NL, "  sub_array[lid] = array[(index - 1) * gs + tid];"
    NL, "  barrier(CLK_LOCAL_MEM_FENCE);"
    NL, ""
    NL, "  for(int i = 0; i < ls; i++)"
    NL, "  {"
    NL, "    int id = gid * ls + i;"
    NL, "    val += sub_array[i];"
    NL, "    val -= (tid == id)? 0: (id + index - 1);"
    NL, "  }"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH)"
    NL, "    {"
    NL, "      clk_event_t block_evt;"
    NL, "      clk_event_t user_evt = create_user_event();"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "      ^(__local void* sub_array){"
    NL, "        block_fn(array, index + 1, ls, gs, res, sub_array);"
    NL, "      }, ls * sizeof(int));"
    NL, "      set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "      release_event(user_evt);"
    NL, "      release_event(block_evt);"
    NL, "    }"
    NL, "  }"
    NL, ""
    NL, "  array[index * gs + tid] = val + 1;"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_kernel_event_local(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls  = get_local_size(0);"
    NL, "  size_t gs  = get_global_size(0);"
    NL, "  size_t tid  = get_global_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(tid == 0)"
    NL, "  {"
    NL, "    clk_event_t block_evt;"
    NL, "    clk_event_t user_evt = create_user_event();"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "    ^(__local void* sub_array){"
    NL, "      block_fn(array, 1, ls, gs, res, sub_array);"
    NL, "    }, ls * sizeof(int));"
    NL, "    set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "    release_event(user_evt);"
    NL, "    release_event(block_evt);"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_work_group_simple[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, __global int* res, int group_id)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  "
    NL, "  if(gid == group_id)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH && lid == 0)"
    NL, "    {"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), "
    NL, "      ^{"
    NL, "         block_fn(array, index + 1, ls, res, gid);"
    NL, "       });"
    NL, "    }"
    NL, "   "
    NL, "    array[index * gs + tid] = array[(index - 1) * gs + tid] + 1;"
    NL, "  }"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0 && gid == group_id)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_work_group_simple(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls = get_local_size(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(lid == 0)"
    NL, "  {"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), "
    NL, "    ^{"
    NL, "       block_fn(array, 1, ls, res, gid);"
    NL, "     });"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_work_group_event[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, __global int* res, int group_id)"
    NL, "{"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  "
    NL, "  if(gid == group_id)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH && lid == 0)"
    NL, "    {"
    NL, "      clk_event_t block_evt;"
    NL, "      clk_event_t user_evt = create_user_event();"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "      ^{"
    NL, "         block_fn(array, index + 1, ls, res, gid);"
    NL, "       });"
    NL, "      set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "      release_event(user_evt);"
    NL, "      release_event(block_evt);"
    NL, "    }"
    NL, "   "
    NL, "    array[index * gs + tid] = array[(index - 1) * gs + tid] + 1;"
    NL, "  }"
    NL, ""
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0 && gid == group_id)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_work_group_event(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls = get_local_size(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(lid == 0)"
    NL, "  {"
    NL, "    clk_event_t block_evt;"
    NL, "    clk_event_t user_evt = create_user_event();"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "    ^{"
    NL, "       block_fn(array, 1, ls, res, gid);"
    NL, "     });"
    NL, "    set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "    release_event(user_evt);"
    NL, "    release_event(block_evt);"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_work_group_local[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, __global int* res, __local int* sub_array, int group_id)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, ""
    NL, "  sub_array[lid] = array[(index - 1) * gs + tid];"
    NL, "  barrier(CLK_LOCAL_MEM_FENCE);"
    NL, ""
    NL, "  for(int i = 0; i < ls; i++)"
    NL, "  {"
    NL, "    int id = gid * ls + i;"
    NL, "    val += sub_array[i];"
    NL, "    val -= (tid == id)? 0: (id + index - 1);"
    NL, "  }"
    NL, " "
    NL, "  if(gid == group_id)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH && lid == 0)"
    NL, "    {"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), "
    NL, "      ^(__local void* sub_array){"
    NL, "        block_fn(array, index + 1, ls, res, sub_array, gid);"
    NL, "      }, ls * sizeof(int));"
    NL, "    }"
    NL, " "
    NL, "    array[index * gs + tid] = val + 1;"
    NL, "  }"
    NL, ""
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0 && gid == group_id)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_work_group_local(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls  = get_local_size(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, "  size_t tid  = get_global_id(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(lid == 0)"
    NL, "  {"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), "
    NL, "    ^(__local void* sub_array){"
    NL, "      block_fn(array, 1, ls, res, sub_array, gid);"
    NL, "    }, ls * sizeof(int));"
    NL, "  }"
    NL, "}"
    NL
};

static const char* enqueue_flags_wait_work_group_event_local[] =
{
    NL, "#define BITS_DEPTH " STRINGIFY_VALUE(BITS_DEPTH)
    NL, ""
    NL, "void block_fn(__global int* array, int index, size_t ls, __global int* res, __local int* sub_array, int group_id)"
    NL, "{"
    NL, "  int val = 0;"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid = get_local_id(0);"
    NL, "  size_t tid = get_global_id(0);"
    NL, "  size_t gs = get_global_size(0);"
    NL, ""
    NL, "  sub_array[lid] = array[(index - 1) * gs + tid];"
    NL, "  barrier(CLK_LOCAL_MEM_FENCE);"
    NL, ""
    NL, "  for(int i = 0; i < ls; i++)"
    NL, "  {"
    NL, "    int id = gid * ls + i;"
    NL, "    val += sub_array[i];"
    NL, "    val -= (tid == id)? 0: (id + index - 1);"
    NL, "  }"
    NL, ""
    NL, "  if(gid == group_id)"
    NL, "  {"
    NL, "    if((index + 1) < BITS_DEPTH && lid == 0)"
    NL, "    {"
    NL, "      clk_event_t block_evt;"
    NL, "      clk_event_t user_evt = create_user_event();"
    NL, "      enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "      ^(__local void* sub_array){"
    NL, "        block_fn(array, index + 1, ls, res, sub_array, gid);"
    NL, "      }, ls * sizeof(int));"
    NL, "      set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "      release_event(user_evt);"
    NL, "      release_event(block_evt);"
    NL, "    }"
    NL, " "
    NL, "    array[index * gs + tid] = val + 1;"
    NL, "  }"
    NL, ""
    NL, "  if((index + 1) == BITS_DEPTH)"
    NL, "  {"
    NL, "    barrier(CLK_GLOBAL_MEM_FENCE);"
    NL, ""
    NL, "    if(lid == 0 && gid == group_id)"
    NL, "    {"
    NL, "      res[gid] = 1;"
    NL, ""
    NL, "      for(int j = 0; j < BITS_DEPTH; j++)"
    NL, "      {"
    NL, "        for(int i = 0; i < ls; i++)"
    NL, "        {"
    NL, "          if(array[j * gs + ls * gid + i] != ((ls * gid + i) + j))"
    NL, "          {"
    NL, "            res[gid] = 2;"
    NL, "            break;"
    NL, "          }"
    NL, "        }"
    NL, "      }"
    NL, "    }"
    NL, "  }"
    NL, "}"
    NL, ""
    NL, "kernel void enqueue_flags_wait_work_group_event_local(__global int* res, __global int* array)"
    NL, "{"
    NL, "  size_t ls  = get_local_size(0);"
    NL, "  size_t gs  = get_global_size(0);"
    NL, "  size_t tid  = get_global_id(0);"
    NL, "  size_t gid = get_group_id(0);"
    NL, "  size_t lid  = get_local_id(0);"
    NL, ""
    NL, "  res[tid] = 0;"
    NL, "  array[tid] = tid;"
    NL, ""
    NL, "  if(lid == 0)"
    NL, "  {"
    NL, "    clk_event_t block_evt;"
    NL, "    clk_event_t user_evt = create_user_event();"
    NL, "    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange_1D(gs, ls), 1, &user_evt, &block_evt, "
    NL, "    ^(__local void* sub_array){"
    NL, "      block_fn(array, 1, ls, res, sub_array, gid);"
    NL, "    }, ls * sizeof(int));"
    NL, "    set_user_event_status(user_evt, CL_COMPLETE);"
    NL, "    release_event(user_evt);"
    NL, "    release_event(block_evt);"
    NL, "  }"
    NL, "}"
    NL
};

static const kernel_src sources_enqueue_block_flags[] =
{
    KERNEL(enqueue_flags_wait_kernel_simple),
    KERNEL(enqueue_flags_wait_kernel_event),
    KERNEL(enqueue_flags_wait_kernel_local),
    KERNEL(enqueue_flags_wait_kernel_event_local),
    KERNEL(enqueue_flags_wait_work_group_simple),
    KERNEL(enqueue_flags_wait_work_group_event),
    KERNEL(enqueue_flags_wait_work_group_local),
    KERNEL(enqueue_flags_wait_work_group_event_local)
};
static const size_t num_enqueue_block_flags = arr_size(sources_enqueue_block_flags);


int test_enqueue_flags(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_uint i;
    cl_int err_ret, res = 0;
    clCommandQueueWrapper dev_queue;
    cl_int kernel_results[MAX_GWS] = { -1 };
    int buff[MAX_GWS * BITS_DEPTH] = { 0 };

    size_t ret_len;
    size_t max_local_size = 1;
    cl_uint maxQueueSize = 0;
    err_ret = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, sizeof(maxQueueSize), &maxQueueSize, 0);
    test_error(err_ret, "clGetDeviceInfo(CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE) failed");

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

    size_t global_size = MAX_GWS;
    size_t local_size = (max_local_size > global_size/16) ? global_size/16 : max_local_size;
    if(gWimpyMode)
    {
        global_size = 4;
        local_size = 2;
    }

    size_t failCnt = 0;
    for(i = 0; i < num_enqueue_block_flags; ++i)
    {
        if (!gKernelName.empty() && gKernelName != sources_enqueue_block_flags[i].kernel_name)
            continue;

        log_info("Running '%s' kernel (%d of %d) ...\n", sources_enqueue_block_flags[i].kernel_name, i + 1, num_enqueue_block_flags);

        clMemWrapper mem = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, global_size * BITS_DEPTH * sizeof(cl_int), buff, &err_ret);
        test_error(err_ret, "clCreateBuffer() failed");

        kernel_arg args[] =
        {
            { sizeof(cl_mem),  &mem }
        };

        err_ret = run_n_kernel_args(context, queue, sources_enqueue_block_flags[i].lines, sources_enqueue_block_flags[i].num_lines, sources_enqueue_block_flags[i].kernel_name, local_size, global_size, kernel_results, sizeof(kernel_results), arr_size(args), args);
        if(check_error(err_ret, "'%s' kernel execution failed", sources_enqueue_block_flags[i].kernel_name)) { ++failCnt; res = -1; }
        else
        {
            int r = 0;
            for (int j=0; j<global_size; j++)
            {
                if (kernel_results[j] != 1 && j < (global_size / local_size) && check_error(-1, "'%s' kernel result[idx: %d] validation failed (test) %d != (expected) 1", sources_enqueue_block_flags[i].kernel_name, j, kernel_results[j]))
                {
                    r = -1;
                    break;
                }
                else if (kernel_results[j] != 0 && j >= (global_size / local_size) && check_error(-1, "'%s' kernel result[idx: %d] validation failed (test) %d != (expected) 0", sources_enqueue_block_flags[i].kernel_name, j, kernel_results[j]))
                {
                    r = -1;
                    break;
                }
            }
            if(r == 0) log_info("'%s' kernel is OK.\n", sources_enqueue_block_flags[i].kernel_name);
            else res = -1;
        }
    }

    if (failCnt > 0)
    {
        log_error("ERROR: %d of %d kernels failed.\n", failCnt, num_enqueue_block_flags);
    }

    return res;
}



#endif
