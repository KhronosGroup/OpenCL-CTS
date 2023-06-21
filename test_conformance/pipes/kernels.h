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
#ifndef KERNELS_H_
#define KERNELS_H_

static const char* pipe_readwrite_struct_kernel_code = {
    "typedef struct{\n"
    "char    a;\n"
    "int    b;\n"
    "}TestStruct;\n"
    "__kernel void test_pipe_write_struct(__global TestStruct *src, __write_only pipe TestStruct out_pipe)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    res_id = reserve_write_pipe(out_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        write_pipe(out_pipe, res_id, 0, &src[gid]);\n"
    "        commit_write_pipe(out_pipe, res_id);\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_read_struct(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    res_id = reserve_read_pipe(in_pipe, 1);\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        read_pipe(in_pipe, res_id, 0, &dst[gid]);\n"
    "        commit_read_pipe(in_pipe, res_id);\n"
    "    }\n"
    "}\n" };

static const char* pipe_workgroup_readwrite_struct_kernel_code = {
    "typedef struct{\n"
    "char    a;\n"
    "int    b;\n"
    "}TestStruct;\n"
    "__kernel void test_pipe_workgroup_write_struct(__global TestStruct *src, __write_only pipe TestStruct out_pipe)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    __local reserve_id_t res_id; \n"
    "\n"
    "    res_id = work_group_reserve_write_pipe(out_pipe, get_local_size(0));\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        write_pipe(out_pipe, res_id, get_local_id(0), &src[gid]);\n"
    "        work_group_commit_write_pipe(out_pipe, res_id);\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_workgroup_read_struct(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    __local reserve_id_t res_id; \n"
    "\n"
    "    res_id = work_group_reserve_read_pipe(in_pipe, get_local_size(0));\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        read_pipe(in_pipe, res_id, get_local_id(0), &dst[gid]);\n"
    "        work_group_commit_read_pipe(in_pipe, res_id);\n"
    "    }\n"
    "}\n" };

static const char* pipe_subgroup_readwrite_struct_kernel_code = {
    "typedef struct{\n"
    "char    a;\n"
    "int    b;\n"
    "}TestStruct;\n"
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "__kernel void test_pipe_subgroup_write_struct(__global TestStruct *src, __write_only pipe TestStruct out_pipe)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    res_id = sub_group_reserve_write_pipe(out_pipe, get_sub_group_size());\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        write_pipe(out_pipe, res_id, get_sub_group_local_id(), &src[gid]);\n"
    "        sub_group_commit_write_pipe(out_pipe, res_id);\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_subgroup_read_struct(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    reserve_id_t res_id; \n"
    "\n"
    "    res_id = sub_group_reserve_read_pipe(in_pipe, get_sub_group_size());\n"
    "    if(is_valid_reserve_id(res_id))\n"
    "    {\n"
    "        read_pipe(in_pipe, res_id, get_sub_group_local_id(), &dst[gid]);\n"
    "        sub_group_commit_read_pipe(in_pipe, res_id);\n"
    "    }\n"
    "}\n" };

static const char* pipe_convenience_readwrite_struct_kernel_code = {
    "typedef struct{\n"
    "char    a;\n"
    "int    b;\n"
    "}TestStruct;\n"
    "__kernel void test_pipe_convenience_write_struct(__global TestStruct *src, __write_only pipe TestStruct out_pipe)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    write_pipe(out_pipe, &src[gid]);\n"
    "}\n"
    "\n"
    "__kernel void test_pipe_convenience_read_struct(__read_only pipe TestStruct in_pipe, __global TestStruct *dst)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    read_pipe(in_pipe, &dst[gid]);\n"
    "}\n" };

#endif // KERNELS_H_
