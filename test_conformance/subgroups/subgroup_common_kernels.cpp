//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "subgroup_common_kernels.h"

const char* bcast_source =
    "__kernel void test_bcast(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    uint which_sub_group_local_id = xy[gid].z;\n"
    "    out[gid] = sub_group_broadcast(x, which_sub_group_local_id);\n"

    "}\n";

const char* redadd_source = "__kernel void test_redadd(const __global Type "
                            "*in, __global int4 *xy, __global Type *out)\n"
                            "{\n"
                            "    int gid = get_global_id(0);\n"
                            "    XY(xy,gid);\n"
                            "    out[gid] = sub_group_reduce_add(in[gid]);\n"
                            "}\n";

const char* redmax_source = "__kernel void test_redmax(const __global Type "
                            "*in, __global int4 *xy, __global Type *out)\n"
                            "{\n"
                            "    int gid = get_global_id(0);\n"
                            "    XY(xy,gid);\n"
                            "    out[gid] = sub_group_reduce_max(in[gid]);\n"
                            "}\n";

const char* redmin_source = "__kernel void test_redmin(const __global Type "
                            "*in, __global int4 *xy, __global Type *out)\n"
                            "{\n"
                            "    int gid = get_global_id(0);\n"
                            "    XY(xy,gid);\n"
                            "    out[gid] = sub_group_reduce_min(in[gid]);\n"
                            "}\n";

const char* scinadd_source =
    "__kernel void test_scinadd(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_add(in[gid]);\n"
    "}\n";

const char* scinmax_source =
    "__kernel void test_scinmax(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_max(in[gid]);\n"
    "}\n";

const char* scinmin_source =
    "__kernel void test_scinmin(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_inclusive_min(in[gid]);\n"
    "}\n";

const char* scexadd_source =
    "__kernel void test_scexadd(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_add(in[gid]);\n"
    "}\n";

const char* scexmax_source =
    "__kernel void test_scexmax(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_max(in[gid]);\n"
    "}\n";

const char* scexmin_source =
    "__kernel void test_scexmin(const __global Type *in, __global int4 *xy, "
    "__global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    out[gid] = sub_group_scan_exclusive_min(in[gid]);\n"
    "}\n";
