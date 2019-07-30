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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>

#include "procs.h"
#include "kernels.h"
#include "../../test_common/harness/errorHelpers.h"

#ifndef uchar
typedef unsigned char uchar;
#endif

typedef struct{
    char    a;
    int        b;
} TestStruct;

#define STRING_LENGTH  1024

static int useWorkgroupReserve = 0;
static int useSubgroupReserve = 0;
static int useConvenienceBuiltIn = 0;

static const char* int_kernel_name[] = { "test_pipe_write_int", "test_pipe_read_int", "test_pipe_write_int2", "test_pipe_read_int2", "test_pipe_write_int4", "test_pipe_read_int4", "test_pipe_write_int8", "test_pipe_read_int8", "test_pipe_write_int16", "test_pipe_read_int16" };
static const char* uint_kernel_name[] = { "test_pipe_write_uint", "test_pipe_read_uint", "test_pipe_write_uint2", "test_pipe_read_uint2", "test_pipe_write_uint4", "test_pipe_read_uint4", "test_pipe_write_uint8", "test_pipe_read_uint8", "test_pipe_write_uint16", "test_pipe_read_uint16" };
static const char* long_kernel_name[] = { "test_pipe_write_long", "test_pipe_read_long", "test_pipe_write_long2", "test_pipe_read_long2", "test_pipe_write_long4", "test_pipe_read_long4", "test_pipe_write_long8", "test_pipe_read_long8", "test_pipe_write_long16", "test_pipe_read_long16" };
static const char* ulong_kernel_name[] = { "test_pipe_write_ulong", "test_pipe_read_ulong", "test_pipe_write_ulong2", "test_pipe_read_ulong2", "test_pipe_write_ulong4", "test_pipe_read_ulong4", "test_pipe_write_ulong8", "test_pipe_read_ulong8", "test_pipe_write_ulong16", "test_pipe_read_ulong16" };
static const char* char_kernel_name[] = { "test_pipe_write_char", "test_pipe_read_char", "test_pipe_write_char2", "test_pipe_read_char2", "test_pipe_write_char4", "test_pipe_read_char4", "test_pipe_write_char8", "test_pipe_read_char8", "test_pipe_write_char16", "test_pipe_read_char16" };
static const char* uchar_kernel_name[] = { "test_pipe_write_uchar", "test_pipe_read_uchar", "test_pipe_write_uchar2", "test_pipe_read_uchar2", "test_pipe_write_uchar4", "test_pipe_read_uchar4", "test_pipe_write_uchar8", "test_pipe_read_uchar8", "test_pipe_write_uchar16", "test_pipe_read_uchar16" };
static const char* short_kernel_name[] = { "test_pipe_write_short", "test_pipe_read_short", "test_pipe_write_short2", "test_pipe_read_short2", "test_pipe_write_short4", "test_pipe_read_short4", "test_pipe_write_short8", "test_pipe_read_short8", "test_pipe_write_short16", "test_pipe_read_short16" };
static const char* ushort_kernel_name[] = { "test_pipe_write_ushort", "test_pipe_read_ushort", "test_pipe_write_ushort2", "test_pipe_read_ushort2", "test_pipe_write_ushort4", "test_pipe_read_ushort4", "test_pipe_write_ushort8", "test_pipe_read_ushort8", "test_pipe_write_ushort16", "test_pipe_read_ushort16" };
static const char* float_kernel_name[] = { "test_pipe_write_float", "test_pipe_read_float", "test_pipe_write_float2", "test_pipe_read_float2", "test_pipe_write_float4", "test_pipe_read_float4", "test_pipe_write_float8", "test_pipe_read_float8", "test_pipe_write_float16", "test_pipe_read_float16" };
static const char* half_kernel_name[] = { "test_pipe_write_half", "test_pipe_read_half", "test_pipe_write_half2", "test_pipe_read_half2", "test_pipe_write_half4", "test_pipe_read_half4", "test_pipe_write_half8", "test_pipe_read_half8", "test_pipe_write_half16", "test_pipe_read_half16" };
static const char* double_kernel_name[] = { "test_pipe_write_double", "test_pipe_read_double", "test_pipe_write_double2", "test_pipe_read_double2", "test_pipe_write_double4", "test_pipe_read_double4", "test_pipe_write_double8", "test_pipe_read_double8", "test_pipe_write_double16", "test_pipe_read_double16" };

static const char* workgroup_int_kernel_name[] = { "test_pipe_workgroup_write_int", "test_pipe_workgroup_read_int", "test_pipe_workgroup_write_int2", "test_pipe_workgroup_read_int2", "test_pipe_workgroup_write_int4", "test_pipe_workgroup_read_int4", "test_pipe_workgroup_write_int8", "test_pipe_workgroup_read_int8", "test_pipe_workgroup_write_int16", "test_pipe_workgroup_read_int16" };
static const char* workgroup_uint_kernel_name[] = { "test_pipe_workgroup_write_uint", "test_pipe_workgroup_read_uint", "test_pipe_workgroup_write_uint2", "test_pipe_workgroup_read_uint2", "test_pipe_workgroup_write_uint4", "test_pipe_workgroup_read_uint4", "test_pipe_workgroup_write_uint8", "test_pipe_workgroup_read_uint8", "test_pipe_workgroup_write_uint16", "test_pipe_workgroup_read_uint16" };
static const char* workgroup_long_kernel_name[] = { "test_pipe_workgroup_write_long", "test_pipe_workgroup_read_long", "test_pipe_workgroup_write_long2", "test_pipe_workgroup_read_long2", "test_pipe_workgroup_write_long4", "test_pipe_workgroup_read_long4", "test_pipe_workgroup_write_long8", "test_pipe_workgroup_read_long8", "test_pipe_workgroup_write_long16", "test_pipe_workgroup_read_long16" };
static const char* workgroup_ulong_kernel_name[] = { "test_pipe_workgroup_write_ulong", "test_pipe_workgroup_read_ulong", "test_pipe_workgroup_write_ulong2", "test_pipe_workgroup_read_ulong2", "test_pipe_workgroup_write_ulong4", "test_pipe_workgroup_read_ulong4", "test_pipe_workgroup_write_ulong8", "test_pipe_workgroup_read_ulong8", "test_pipe_workgroup_write_ulong16", "test_pipe_workgroup_read_ulong16" };
static const char* workgroup_char_kernel_name[] = { "test_pipe_workgroup_write_char", "test_pipe_workgroup_read_char", "test_pipe_workgroup_write_char2", "test_pipe_workgroup_read_char2", "test_pipe_workgroup_write_char4", "test_pipe_workgroup_read_char4", "test_pipe_workgroup_write_char8", "test_pipe_workgroup_read_char8", "test_pipe_workgroup_write_char16", "test_pipe_workgroup_read_char16" };
static const char* workgroup_uchar_kernel_name[] = { "test_pipe_workgroup_write_uchar", "test_pipe_workgroup_read_uchar", "test_pipe_workgroup_write_uchar2", "test_pipe_workgroup_read_uchar2", "test_pipe_workgroup_write_uchar4", "test_pipe_workgroup_read_uchar4", "test_pipe_workgroup_write_uchar8", "test_pipe_workgroup_read_uchar8", "test_pipe_workgroup_write_uchar16", "test_pipe_workgroup_read_uchar16" };
static const char* workgroup_short_kernel_name[] = { "test_pipe_workgroup_write_short", "test_pipe_workgroup_read_short", "test_pipe_workgroup_write_short2", "test_pipe_workgroup_read_short2", "test_pipe_workgroup_write_short4", "test_pipe_workgroup_read_short4", "test_pipe_workgroup_write_short8", "test_pipe_workgroup_read_short8", "test_pipe_workgroup_write_short16", "test_pipe_workgroup_read_short16" };
static const char* workgroup_ushort_kernel_name[] = { "test_pipe_workgroup_write_ushort", "test_pipe_workgroup_read_ushort", "test_pipe_workgroup_write_ushort2", "test_pipe_workgroup_read_ushort2", "test_pipe_workgroup_write_ushort4", "test_pipe_workgroup_read_ushort4", "test_pipe_workgroup_write_ushort8", "test_pipe_workgroup_read_ushort8", "test_pipe_workgroup_write_ushort16", "test_pipe_workgroup_read_ushort16" };
static const char* workgroup_float_kernel_name[] = { "test_pipe_workgroup_write_float", "test_pipe_workgroup_read_float", "test_pipe_workgroup_write_float2", "test_pipe_workgroup_read_float2", "test_pipe_workgroup_write_float4", "test_pipe_workgroup_read_float4", "test_pipe_workgroup_write_float8", "test_pipe_workgroup_read_float8", "test_pipe_workgroup_write_float16", "test_pipe_workgroup_read_float16" };
static const char* workgroup_half_kernel_name[] = { "test_pipe_workgroup_write_half", "test_pipe_workgroup_read_half", "test_pipe_workgroup_write_half2", "test_pipe_workgroup_read_half2", "test_pipe_workgroup_write_half4", "test_pipe_workgroup_read_half4", "test_pipe_workgroup_write_half8", "test_pipe_workgroup_read_half8", "test_pipe_workgroup_write_half16", "test_pipe_workgroup_read_half16" };
static const char* workgroup_double_kernel_name[] = { "test_pipe_workgroup_write_double", "test_pipe_workgroup_read_double", "test_pipe_workgroup_write_double2", "test_pipe_workgroup_read_double2", "test_pipe_workgroup_write_double4", "test_pipe_workgroup_read_double4", "test_pipe_workgroup_write_double8", "test_pipe_workgroup_read_double8", "test_pipe_workgroup_write_double16", "test_pipe_workgroup_read_double16" };

static const char* subgroup_int_kernel_name[] = { "test_pipe_subgroup_write_int", "test_pipe_subgroup_read_int", "test_pipe_subgroup_write_int2", "test_pipe_subgroup_read_int2", "test_pipe_subgroup_write_int4", "test_pipe_subgroup_read_int4", "test_pipe_subgroup_write_int8", "test_pipe_subgroup_read_int8", "test_pipe_subgroup_write_int16", "test_pipe_subgroup_read_int16" };
static const char* subgroup_uint_kernel_name[] = { "test_pipe_subgroup_write_uint", "test_pipe_subgroup_read_uint", "test_pipe_subgroup_write_uint2", "test_pipe_subgroup_read_uint2", "test_pipe_subgroup_write_uint4", "test_pipe_subgroup_read_uint4", "test_pipe_subgroup_write_uint8", "test_pipe_subgroup_read_uint8", "test_pipe_subgroup_write_uint16", "test_pipe_subgroup_read_uint16" };
static const char* subgroup_long_kernel_name[] = { "test_pipe_subgroup_write_long", "test_pipe_subgroup_read_long", "test_pipe_subgroup_write_long2", "test_pipe_subgroup_read_long2", "test_pipe_subgroup_write_long4", "test_pipe_subgroup_read_long4", "test_pipe_subgroup_write_long8", "test_pipe_subgroup_read_long8", "test_pipe_subgroup_write_long16", "test_pipe_subgroup_read_long16" };
static const char* subgroup_ulong_kernel_name[] = { "test_pipe_subgroup_write_ulong", "test_pipe_subgroup_read_ulong", "test_pipe_subgroup_write_ulong2", "test_pipe_subgroup_read_ulong2", "test_pipe_subgroup_write_ulong4", "test_pipe_subgroup_read_ulong4", "test_pipe_subgroup_write_ulong8", "test_pipe_subgroup_read_ulong8", "test_pipe_subgroup_write_ulong16", "test_pipe_subgroup_read_ulong16" };
static const char* subgroup_char_kernel_name[] = { "test_pipe_subgroup_write_char", "test_pipe_subgroup_read_char", "test_pipe_subgroup_write_char2", "test_pipe_subgroup_read_char2", "test_pipe_subgroup_write_char4", "test_pipe_subgroup_read_char4", "test_pipe_subgroup_write_char8", "test_pipe_subgroup_read_char8", "test_pipe_subgroup_write_char16", "test_pipe_subgroup_read_char16" };
static const char* subgroup_uchar_kernel_name[] = { "test_pipe_subgroup_write_uchar", "test_pipe_subgroup_read_uchar", "test_pipe_subgroup_write_uchar2", "test_pipe_subgroup_read_uchar2", "test_pipe_subgroup_write_uchar4", "test_pipe_subgroup_read_uchar4", "test_pipe_subgroup_write_uchar8", "test_pipe_subgroup_read_uchar8", "test_pipe_subgroup_write_uchar16", "test_pipe_subgroup_read_uchar16" };
static const char* subgroup_short_kernel_name[] = { "test_pipe_subgroup_write_short", "test_pipe_subgroup_read_short", "test_pipe_subgroup_write_short2", "test_pipe_subgroup_read_short2", "test_pipe_subgroup_write_short4", "test_pipe_subgroup_read_short4", "test_pipe_subgroup_write_short8", "test_pipe_subgroup_read_short8", "test_pipe_subgroup_write_short16", "test_pipe_subgroup_read_short16" };
static const char* subgroup_ushort_kernel_name[] = { "test_pipe_subgroup_write_ushort", "test_pipe_subgroup_read_ushort", "test_pipe_subgroup_write_ushort2", "test_pipe_subgroup_read_ushort2", "test_pipe_subgroup_write_ushort4", "test_pipe_subgroup_read_ushort4", "test_pipe_subgroup_write_ushort8", "test_pipe_subgroup_read_ushort8", "test_pipe_subgroup_write_ushort16", "test_pipe_subgroup_read_ushort16" };
static const char* subgroup_float_kernel_name[] = { "test_pipe_subgroup_write_float", "test_pipe_subgroup_read_float", "test_pipe_subgroup_write_float2", "test_pipe_subgroup_read_float2", "test_pipe_subgroup_write_float4", "test_pipe_subgroup_read_float4", "test_pipe_subgroup_write_float8", "test_pipe_subgroup_read_float8", "test_pipe_subgroup_write_float16", "test_pipe_subgroup_read_float16" };
static const char* subgroup_half_kernel_name[] = { "test_pipe_subgroup_write_half", "test_pipe_subgroup_read_half", "test_pipe_subgroup_write_half2", "test_pipe_subgroup_read_half2", "test_pipe_subgroup_write_half4", "test_pipe_subgroup_read_half4", "test_pipe_subgroup_write_half8", "test_pipe_subgroup_read_half8", "test_pipe_subgroup_write_half16", "test_pipe_subgroup_read_half16" };
static const char* subgroup_double_kernel_name[] = { "test_pipe_subgroup_write_double", "test_pipe_subgroup_read_double", "test_pipe_subgroup_write_double2", "test_pipe_subgroup_read_double2", "test_pipe_subgroup_write_double4", "test_pipe_subgroup_read_double4", "test_pipe_subgroup_write_double8", "test_pipe_subgroup_read_double8", "test_pipe_subgroup_write_double16", "test_pipe_subgroup_read_double16" };


static const char* convenience_int_kernel_name[] = { "test_pipe_convenience_write_int", "test_pipe_convenience_read_int", "test_pipe_convenience_write_int2", "test_pipe_convenience_read_int2", "test_pipe_convenience_write_int4", "test_pipe_convenience_read_int4", "test_pipe_convenience_write_int8", "test_pipe_convenience_read_int8", "test_pipe_convenience_write_int16", "test_pipe_convenience_read_int16" };
static const char* convenience_uint_kernel_name[] = { "test_pipe_convenience_write_uint", "test_pipe_convenience_read_uint", "test_pipe_convenience_write_uint2", "test_pipe_convenience_read_uint2", "test_pipe_convenience_write_uint4", "test_pipe_convenience_read_uint4", "test_pipe_convenience_write_uint8", "test_pipe_convenience_read_uint8", "test_pipe_convenience_write_uint16", "test_pipe_convenience_read_uint16" };
static const char* convenience_long_kernel_name[] = { "test_pipe_convenience_write_long", "test_pipe_convenience_read_long", "test_pipe_convenience_write_long2", "test_pipe_convenience_read_long2", "test_pipe_convenience_write_long4", "test_pipe_convenience_read_long4", "test_pipe_convenience_write_long8", "test_pipe_convenience_read_long8", "test_pipe_convenience_write_long16", "test_pipe_convenience_read_long16" };
static const char* convenience_ulong_kernel_name[] = { "test_pipe_convenience_write_ulong", "test_pipe_convenience_read_ulong", "test_pipe_convenience_write_ulong2", "test_pipe_convenience_read_ulong2", "test_pipe_convenience_write_ulong4", "test_pipe_convenience_read_ulong4", "test_pipe_convenience_write_ulong8", "test_pipe_convenience_read_ulong8", "test_pipe_convenience_write_ulong16", "test_pipe_convenience_read_ulong16" };
static const char* convenience_char_kernel_name[] = { "test_pipe_convenience_write_char", "test_pipe_convenience_read_char", "test_pipe_convenience_write_char2", "test_pipe_convenience_read_char2", "test_pipe_convenience_write_char4", "test_pipe_convenience_read_char4", "test_pipe_convenience_write_char8", "test_pipe_convenience_read_char8", "test_pipe_convenience_write_char16", "test_pipe_convenience_read_char16" };
static const char* convenience_uchar_kernel_name[] = { "test_pipe_convenience_write_uchar", "test_pipe_convenience_read_uchar", "test_pipe_convenience_write_uchar2", "test_pipe_convenience_read_uchar2", "test_pipe_convenience_write_uchar4", "test_pipe_convenience_read_uchar4", "test_pipe_convenience_write_uchar8", "test_pipe_convenience_read_uchar8", "test_pipe_convenience_write_uchar16", "test_pipe_convenience_read_uchar16" };
static const char* convenience_short_kernel_name[] = { "test_pipe_convenience_write_short", "test_pipe_convenience_read_short", "test_pipe_convenience_write_short2", "test_pipe_convenience_read_short2", "test_pipe_convenience_write_short4", "test_pipe_convenience_read_short4", "test_pipe_convenience_write_short8", "test_pipe_convenience_read_short8", "test_pipe_convenience_write_short16", "test_pipe_convenience_read_short16" };
static const char* convenience_ushort_kernel_name[] = { "test_pipe_convenience_write_ushort", "test_pipe_convenience_read_ushort", "test_pipe_convenience_write_ushort2", "test_pipe_convenience_read_ushort2", "test_pipe_convenience_write_ushort4", "test_pipe_convenience_read_ushort4", "test_pipe_convenience_write_ushort8", "test_pipe_convenience_read_ushort8", "test_pipe_convenience_write_ushort16", "test_pipe_convenience_read_ushort16" };
static const char* convenience_float_kernel_name[] = { "test_pipe_convenience_write_float", "test_pipe_convenience_read_float", "test_pipe_convenience_write_float2", "test_pipe_convenience_read_float2", "test_pipe_convenience_write_float4", "test_pipe_convenience_read_float4", "test_pipe_convenience_write_float8", "test_pipe_convenience_read_float8", "test_pipe_convenience_write_float16", "test_pipe_convenience_read_float16" };
static const char* convenience_half_kernel_name[] = { "test_pipe_convenience_write_half", "test_pipe_convenience_read_half", "test_pipe_convenience_write_half2", "test_pipe_convenience_read_half2", "test_pipe_convenience_write_half4", "test_pipe_convenience_read_half4", "test_pipe_convenience_write_half8", "test_pipe_convenience_read_half8", "test_pipe_convenience_write_half16", "test_pipe_convenience_read_half16" };
static const char* convenience_double_kernel_name[] = { "test_pipe_convenience_write_double", "test_pipe_convenience_read_double", "test_pipe_convenience_write_double2", "test_pipe_convenience_read_double2", "test_pipe_convenience_write_double4", "test_pipe_convenience_read_double4", "test_pipe_convenience_write_double8", "test_pipe_convenience_read_double8", "test_pipe_convenience_write_double16", "test_pipe_convenience_read_double16" };

static void insertPragmaForHalfType(char *source, char *type)
{
    source[0] = 0;
    if(strncmp(type, "half",4) == 0)
    {
        strcat(source, "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
    }
}

void createKernelSource(char *source, char *type)
{
    char str[512];
    int     str_length;

    insertPragmaForHalfType(source, type);

    sprintf(str, "__kernel void test_pipe_write_%s(__global %s *src, __write_only pipe %s out_pipe)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = reserve_write_pipe(out_pipe, 1);\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    write_pipe(out_pipe, res_id, 0, &src[gid]);\n    commit_write_pipe(out_pipe, res_id);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_read_%s(__read_only pipe %s in_pipe, __global %s *dst)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = reserve_read_pipe(in_pipe, 1);\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    read_pipe(in_pipe, res_id, 0, &dst[gid]);\n    commit_read_pipe(in_pipe, res_id);\n  }\n}\n");
    strcat(source, str);
    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH);
}

void createKernelSourceWorkGroup(char *source, char *type)
{
    char str[512];
    int     str_length;

    insertPragmaForHalfType(source, type);

    sprintf(str, "__kernel void test_pipe_workgroup_write_%s(__global %s *src, __write_only pipe %s out_pipe)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  __local reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = work_group_reserve_write_pipe(out_pipe, get_local_size(0));\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    write_pipe(out_pipe, res_id, get_local_id(0), &src[gid]);\n    work_group_commit_write_pipe(out_pipe, res_id);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_workgroup_read_%s(__read_only pipe %s in_pipe, __global %s *dst)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  __local reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = work_group_reserve_read_pipe(in_pipe, get_local_size(0));\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    read_pipe(in_pipe, res_id, get_local_id(0), &dst[gid]);\n    work_group_commit_read_pipe(in_pipe, res_id);\n  }\n}\n");
    strcat(source, str);
    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH);
}

void createKernelSourceSubGroup(char *source, char *type)
{
    char str[512];
    int  str_length;

    insertPragmaForHalfType(source, type);

    sprintf(str, "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n__kernel void test_pipe_subgroup_write_%s(__global %s *src, __write_only pipe %s out_pipe)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = sub_group_reserve_write_pipe(out_pipe, get_sub_group_size());\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    write_pipe(out_pipe, res_id, get_sub_group_local_id(), &src[gid]);\n    sub_group_commit_write_pipe(out_pipe, res_id);\n  }\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_subgroup_read_%s(__read_only pipe %s in_pipe, __global %s *dst)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  reserve_id_t res_id; \n\n");
    strcat(source, str);
    sprintf(str, "  res_id = sub_group_reserve_read_pipe(in_pipe, get_sub_group_size());\n  if(is_valid_reserve_id(res_id))\n  {\n");
    strcat(source, str);
    sprintf(str, "    read_pipe(in_pipe, res_id, get_sub_group_local_id(), &dst[gid]);\n    sub_group_commit_read_pipe(in_pipe, res_id);\n  }\n}\n");
    strcat(source, str);
    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH);
}

void createKernelSourceConvenience(char *source, char *type)
{
    char str[512];
    int     str_length;

    insertPragmaForHalfType(source, type);

    sprintf(str, "__kernel void test_pipe_convenience_write_%s(__global %s *src, __write_only pipe %s out_pipe)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  write_pipe(out_pipe, &src[gid]);\n}\n\n");
    strcat(source, str);
    sprintf(str, "__kernel void test_pipe_convenience_read_%s(__read_only pipe %s in_pipe, __global %s *dst)\n", type, type, type);
    strcat(source, str);
    sprintf(str, "{\n  int gid = get_global_id(0);\n  read_pipe(in_pipe, &dst[gid]);\n}\n");
    strcat(source, str);
    str_length = strlen(source);
    assert(str_length <= STRING_LENGTH);
}

// verify functions
static int verify_readwrite_int(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_int  *inptr = (cl_int *)ptr1;
    cl_int  *outptr = (cl_int *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }

    return 0;
}

static int verify_readwrite_uint(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_uint *inptr = (cl_uint *)ptr1;
    cl_uint *outptr = (cl_uint *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }

    return 0;
}

static int verify_readwrite_short(void *ptr1, void *ptr2, int n)
{
    int            i;
    int            sum_input = 0, sum_output = 0;
    cl_short    *inptr = (cl_short *)ptr1;
    cl_short    *outptr = (cl_short *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_ushort(void *ptr1, void *ptr2, int n)
{
    int            i;
    int            sum_input = 0, sum_output = 0;
    cl_ushort    *inptr = (cl_ushort *)ptr1;
    cl_ushort    *outptr = (cl_ushort *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_char(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    cl_char    *inptr = (cl_char *)ptr1;
    cl_char    *outptr = (cl_char *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_uchar(void *ptr1, void *ptr2, int n)
{
    int            i;
    int            sum_input = 0, sum_output = 0;
    cl_uchar    *inptr = (cl_uchar *)ptr1;
    cl_uchar    *outptr = (cl_uchar *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_float(void *ptr1, void *ptr2, int n)
{
    int     i;
    int        sum_input = 0, sum_output = 0;
    int        *inptr = (int *)ptr1;
    int        *outptr = (int *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_half(void *ptr1, void *ptr2, int n)
{
    int            i;
    int            sum_input = 0, sum_output = 0;
    cl_ushort    *inptr = (cl_ushort *)ptr1;
    cl_ushort    *outptr = (cl_ushort *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_long(void *ptr1, void *ptr2, int n)
{
    int            i;
    cl_long        sum_input = 0, sum_output = 0;
    cl_long        *inptr = (cl_long *)ptr1;
    cl_long        *outptr = (cl_long *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_ulong(void *ptr1, void *ptr2, int n)
{
    int            i;
    cl_ulong    sum_input = 0, sum_output = 0;
    cl_ulong    *inptr = (cl_ulong *)ptr1;
    cl_ulong    *outptr = (cl_ulong *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_double(void *ptr1, void *ptr2, int n)
{
    int                i;
    long long int    sum_input = 0, sum_output = 0;
    long long int    *inptr = (long long int *)ptr1;
    long long int    *outptr = (long long int *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input += inptr[i];
        sum_output += outptr[i];
    }
    if(sum_input != sum_output){
        return -1;
    }
    return 0;
}

static int verify_readwrite_struct(void *ptr1, void *ptr2, int n)
{
    int            i;
    int            sum_input_char = 0, sum_output_char = 0;
    int            sum_input_int = 0, sum_output_int = 0;
    TestStruct    *inptr = (TestStruct *)ptr1;
    TestStruct    *outptr = (TestStruct *)ptr2;

    for(i = 0; i < n; i++)
    {
        sum_input_char += inptr[i].a;
        sum_input_int += inptr[i].b;
        sum_output_char += outptr[i].a;
        sum_output_int += outptr[i].b;
    }
    if( (sum_input_char != sum_output_char) && (sum_input_int != sum_output_int) ){
        return -1;
    }

    return 0;
}

int test_pipe_readwrite( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, size_t size, char *type, int loops,
                         void *inptr[5], const char *kernelName[], int (*fn)(void *, void *, int) )
{
    cl_mem        pipes[5];
    cl_mem      buffers[10];
    void        *outptr[5];
    cl_program  program[5];
    cl_kernel   kernel[10];
    size_t      global_work_size[3];
    size_t        local_work_size[3];
    cl_int      err;
    int         i, ii;
    size_t      ptrSizes[5];
    int         total_errors = 0, compilation_count = 0;
    cl_event    producer_sync_event[5];
    cl_event    consumer_sync_event[5];
    char        *sourceCode[5];
    char        vector_type[10];

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (cl_uint)num_elements;

    ptrSizes[0] = size;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for( i = 0; i < loops; i++)
    {
        ii = i << 1;
        sourceCode[i] = (char*) malloc(STRING_LENGTH * sizeof(char));
        buffers[ii] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, ptrSizes[i] * num_elements, inptr[i], &err);
        if(err){
            clReleaseMemObject(buffers[ii]);
            align_free( outptr[i] );
            print_error(err, " clCreateBuffer failed\n");
            return -1;
        }
        outptr[i] = align_malloc( ptrSizes[i] * num_elements, min_alignment);
        buffers[ii+1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  ptrSizes[i] * num_elements, outptr[i], &err);

        if ( err ){
            clReleaseMemObject(buffers[ii]);
            align_free( outptr[i] );
            print_error(err, " clCreateBuffer failed\n" );
            return -1;
        }
        // Creating pipe with non-power of 2 size
        pipes[i] = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, ptrSizes[i], num_elements+3, NULL, &err);
        if(err){
            clReleaseMemObject(pipes[i]);
            print_error(err, " clCreatePipe failed\n");
            return -1;
        }

        switch(i)
        {
        case 0:
            sprintf(vector_type, "%s", type);
            break;
        case 1:
            sprintf(vector_type, "%s%d", type, 2);
            break;
        case 2:
            sprintf(vector_type, "%s%d", type, 4);
            break;
        case 3:
            sprintf(vector_type, "%s%d", type, 8);
            break;
        case 4:
            sprintf(vector_type, "%s%d", type, 16);
            break;
        }

        if(useWorkgroupReserve == 1){
            createKernelSourceWorkGroup(sourceCode[i], vector_type);
        }
        else if(useSubgroupReserve == 1){
            createKernelSourceSubGroup(sourceCode[i], vector_type);
        }
        else if(useConvenienceBuiltIn == 1){
            createKernelSourceConvenience(sourceCode[i], vector_type);
        }
        else{
            createKernelSource(sourceCode[i], vector_type);
        }

        // Create producer kernel
        err = create_single_kernel_helper_with_build_options(context, &program[i], &kernel[ii], 1, (const char**)&sourceCode[i], kernelName[ii], "-cl-std=CL2.0");
        if(err){
            compilation_count++;
            clReleaseMemObject(buffers[ii]);
            clReleaseMemObject(buffers[ii+1]);
            clReleaseMemObject(pipes[i]);
            align_free( outptr[i] );
            print_error(err, "Error creating program\n");
            continue;
        }
        //Create consumer kernel
        kernel[ii + 1] = clCreateKernel(program[i], kernelName[ii + 1], &err);
        if( kernel[ii + 1] == NULL || err != CL_SUCCESS)
        {
            clReleaseMemObject(buffers[ii]);
            clReleaseMemObject(buffers[ii+1]);
            clReleaseMemObject(pipes[i]);
            align_free( outptr[i] );
            log_error("Creating program for %s\n", type);
            print_error( err, "Unable to create kernel" );
            return -1;
        }

        err = clSetKernelArg(kernel[ii], 0, sizeof(cl_mem), (void*)&buffers[ii]);
        err |= clSetKernelArg(kernel[ii], 1, sizeof(cl_mem), (void*)&pipes[i]);
        err |= clSetKernelArg(kernel[ii + 1], 0, sizeof(cl_mem), (void*)&pipes[i]);
        err |= clSetKernelArg(kernel[ii + 1], 1, sizeof(cl_mem), (void*)&buffers[ii + 1]);
        if ( err != CL_SUCCESS ){
            clReleaseMemObject(buffers[ii]);
            clReleaseMemObject(buffers[ii+1]);
            clReleaseMemObject(pipes[i]);
            clReleaseKernel(kernel[ii]);
            clReleaseKernel(kernel[ii+1]);
            clReleaseProgram(program[i]);
            align_free(outptr[i]);
            print_error(err, " clSetKernelArg failed");
            return -1;
        }

        if(useWorkgroupReserve == 1 || useSubgroupReserve == 1)
        {
            err = get_max_common_work_group_size( context, kernel[ii], global_work_size[0], &local_work_size[0] );
            test_error( err, "Unable to get work group size to use" );
            // Launch Producer kernel
            err = clEnqueueNDRangeKernel( queue, kernel[ii], 1, NULL, global_work_size, local_work_size, 0, NULL, &producer_sync_event[i] );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueNDRangeKernel failed" );
                clReleaseMemObject(buffers[ii]);
                clReleaseMemObject(buffers[ii+1]);
                clReleaseMemObject(pipes[i]);
                clReleaseKernel(kernel[ii]);
                clReleaseKernel(kernel[ii+1]);
                clReleaseEvent(producer_sync_event[i]);
                clReleaseEvent(consumer_sync_event[i]);
                clReleaseProgram(program[i]);
                align_free(outptr[i]);
                return -1;
            }
        }
        else
        {
            // Launch Producer kernel
            err = clEnqueueNDRangeKernel( queue, kernel[ii], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event[i] );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueNDRangeKernel failed" );
                clReleaseMemObject(buffers[ii]);
                clReleaseMemObject(buffers[ii+1]);
                clReleaseMemObject(pipes[i]);
                clReleaseKernel(kernel[ii]);
                clReleaseKernel(kernel[ii+1]);
                clReleaseEvent(producer_sync_event[i]);
                clReleaseEvent(consumer_sync_event[i]);
                clReleaseProgram(program[i]);
                align_free(outptr[i]);
                return -1;
            }
        }

        if(useWorkgroupReserve == 1 || useSubgroupReserve == 1)
        {
            err = get_max_common_work_group_size( context, kernel[ii + 1], global_work_size[0], &local_work_size[0] );
            test_error( err, "Unable to get work group size to use" );

            // Launch Consumer kernel
            err = clEnqueueNDRangeKernel( queue, kernel[ii + 1], 1, NULL, global_work_size, local_work_size, 1, &producer_sync_event[i], &consumer_sync_event[i] );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueNDRangeKernel failed" );
                clReleaseMemObject(buffers[ii]);
                clReleaseMemObject(buffers[ii+1]);
                clReleaseMemObject(pipes[i]);
                clReleaseKernel(kernel[ii]);
                clReleaseKernel(kernel[ii+1]);
                clReleaseEvent(producer_sync_event[i]);
                clReleaseEvent(consumer_sync_event[i]);
                clReleaseProgram(program[i]);
                align_free(outptr[i]);
                return -1;
            }
        }
        else
        {
            // Launch Consumer kernel
            err = clEnqueueNDRangeKernel( queue, kernel[ii + 1], 1, NULL, global_work_size, NULL, 1,  &producer_sync_event[i], &consumer_sync_event[i] );
            if ( err != CL_SUCCESS ){
                print_error( err, " clEnqueueNDRangeKernel failed" );
                clReleaseMemObject(buffers[ii]);
                clReleaseMemObject(buffers[ii+1]);
                clReleaseMemObject(pipes[i]);
                clReleaseKernel(kernel[ii]);
                clReleaseKernel(kernel[ii+1]);
                clReleaseEvent(producer_sync_event[i]);
                clReleaseEvent(consumer_sync_event[i]);
                clReleaseProgram(program[i]);
                align_free(outptr[i]);
                return -1;
            }
        }

        err = clEnqueueReadBuffer(queue, buffers[ii+1], true, 0, ptrSizes[i]*num_elements, outptr[i], 1, &consumer_sync_event[i], NULL);
        if ( err != CL_SUCCESS ){
            print_error( err, " clEnqueueReadBuffer failed" );
            clReleaseMemObject(buffers[ii]);
            clReleaseMemObject(buffers[ii+1]);
            clReleaseMemObject(pipes[i]);
            clReleaseKernel(kernel[ii]);
            clReleaseKernel(kernel[ii+1]);
            clReleaseEvent(producer_sync_event[i]);
            clReleaseEvent(consumer_sync_event[i]);
            clReleaseProgram(program[i]);
            align_free(outptr[i]);
            return -1;
        }

        if( fn( inptr[i], outptr[i], (int)(ptrSizes[i] * (size_t)num_elements / ptrSizes[0]))){
            log_error("%s%d test failed\n", type, 1<<i);
            total_errors++;
        }
        else {
            log_info("%s%d test passed\n", type, 1<<i);
        }
        //cleanup
        clReleaseMemObject(buffers[ii]);
        clReleaseMemObject(buffers[ii+1]);
        clReleaseMemObject(pipes[i]);
        clReleaseKernel(kernel[ii]);
        clReleaseKernel(kernel[ii+1]);
        clReleaseEvent(producer_sync_event[i]);
        clReleaseEvent(consumer_sync_event[i]);
        clReleaseProgram(program[i]);
        align_free(outptr[i]);

    }
    if(compilation_count) {
        return -1;
    }
    return total_errors;
}

int test_pipe_readwrite_struct_generic( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements,
                                const char *kernelCode, const char *kernelName[])
{
    cl_mem        buffers[2];
    cl_mem        pipe;
    void        *outptr;
    TestStruct    *inptr;
    cl_program    program;
    cl_kernel    kernel[2];
    size_t        size = sizeof(TestStruct);
    size_t      global_work_size[3];
    cl_int      err;
    int         total_errors = 0;
    int            i;
    MTdata      d = init_genrand( gRandomSeed );
    cl_event    producer_sync_event = NULL;
    cl_event    consumer_sync_event = NULL;

    size_t      min_alignment = get_min_alignment(context);

    global_work_size[0] = (size_t)num_elements;

    inptr = (TestStruct *)align_malloc(size * num_elements, min_alignment);

    for ( i = 0; i < num_elements; i++ ){
        inptr[i].a = (char)genrand_int32(d);
        inptr[i].b = genrand_int32(d);
    }
    buffers[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size * num_elements, inptr, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        print_error(err, " clCreateBuffer failed\n");
        return -1;
    }
    outptr = align_malloc( size * num_elements, min_alignment);
    buffers[1] = clCreateBuffer(context, CL_MEM_USE_HOST_PTR,  size * num_elements, outptr, &err);
    if (err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        print_error(err, " clCreateBuffer failed\n" );
        return -1;
    }
    pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, size, num_elements, NULL, &err);
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        align_free( outptr );
        clReleaseMemObject(pipe);
        print_error(err, " clCreatePipe failed\n");
        return -1;
    }
    // Create producer kernel
    err = create_single_kernel_helper_with_build_options(context, &program, &kernel[0], 1, &kernelCode, kernelName[0], "-cl-std=CL2.0");
    if(err){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        log_error(" Error creating program for struct\n");
        print_error(err, "Error creating program\n");
        return -1;
    }
    //Create consumer kernel
    kernel[1] = clCreateKernel(program, kernelName[1], &err);
    if( kernel[1] == NULL || err != CL_SUCCESS)
    {
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        align_free(outptr);
        print_error(err, "Error creating kernel\n");
        return -1;
    }

    err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), (void*)&buffers[0]);
    err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void*)&pipe);
    err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void*)&buffers[1]);
    if (err != CL_SUCCESS){
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseProgram(program);
        align_free(outptr);
        print_error(err, " clSetKernelArg failed");
        return -1;
    }

    // Launch Producer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[0], 1, NULL, global_work_size, NULL, 0, NULL, &producer_sync_event );
    if (err != CL_SUCCESS){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    // Launch Consumer kernel
    err = clEnqueueNDRangeKernel( queue, kernel[1], 1, NULL, global_work_size, NULL, 1, &producer_sync_event, &consumer_sync_event );
    if (err != CL_SUCCESS){
        print_error( err, " clEnqueueNDRangeKernel failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, buffers[1], true, 0, size*num_elements, outptr, 1, &consumer_sync_event, NULL);
    if (err != CL_SUCCESS){
        print_error( err, " clEnqueueReadBuffer failed" );
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }

    if( verify_readwrite_struct( inptr, outptr, num_elements)){
        log_error("struct_readwrite test failed\n");
        clReleaseMemObject(buffers[0]);
        clReleaseMemObject(buffers[1]);
        clReleaseMemObject(pipe);
        clReleaseKernel(kernel[0]);
        clReleaseKernel(kernel[1]);
        clReleaseEvent(producer_sync_event);
        clReleaseEvent(consumer_sync_event);
        clReleaseProgram(program);
        align_free(outptr);
        return -1;
    }
    else {
        log_info("struct_readwrite test passed\n");
    }
    //cleanup
    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(pipe);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseEvent(producer_sync_event);
    clReleaseEvent(consumer_sync_event);
    clReleaseProgram(program);
    align_free(outptr);

    return 0;
}


int test_pipe_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_int  *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_int;

    ptrSizes[0] = sizeof(cl_int);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_int *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (int)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                                   workgroup_int_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                                   subgroup_int_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1) {
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                                   convenience_int_kernel_name, foo);
    }
    else {
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_int ), (char*)"int", 5, (void**)inptr,
                                   int_kernel_name, foo);
    }


    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uint     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_uint;

    ptrSizes[0] = sizeof(cl_uint);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_uint *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_uint)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                                   workgroup_uint_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                                   subgroup_uint_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1) {
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                                   convenience_uint_kernel_name, foo);
    }
    else {
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uint ), (char*)"uint", 5, (void**)inptr,
                                   uint_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_short     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_short;

    ptrSizes[0] = sizeof(cl_short);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_short *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_short)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                                   workgroup_short_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                                   subgroup_short_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                                   convenience_short_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_short ), (char*)"short", 5, (void**)inptr,
                                   short_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ushort     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_ushort;

    ptrSizes[0] = sizeof(cl_ushort);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ushort *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ushort)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                                   workgroup_ushort_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                                   subgroup_ushort_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                                   convenience_ushort_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ushort ), (char*)"ushort", 5, (void**)inptr,
                                   ushort_kernel_name, foo);
    }


    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_char *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_char;

    ptrSizes[0] = sizeof(cl_char);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_char *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (char)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                                   workgroup_char_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                                   subgroup_char_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                                   convenience_char_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_char ), (char*)"char", 5, (void**)inptr,
                                   char_kernel_name, foo);
    }


    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_uchar    *inptr[5];
    size_t        ptrSizes[5];
    int            i, err;
    cl_uint        j;
    int            (*foo)(void *,void *,int);
    MTdata        d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_uchar;

    ptrSizes[0] = sizeof(cl_uchar);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_uchar *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (uchar)genrand_int32(d);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                                   workgroup_uchar_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                                   subgroup_uchar_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                                   convenience_uchar_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_uchar ), (char*)"uchar", 5, (void**)inptr,
                                   uchar_kernel_name, foo);
    }
    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    float     *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_float;

    ptrSizes[0] = sizeof(cl_float);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = get_random_float( -32, 32, d );
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                                   workgroup_float_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                                   subgroup_float_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                                   convenience_float_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_float ), (char*)"float", 5, (void**)inptr,
                                   float_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    float   *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_half;

    if(!is_extension_available(deviceID, "cl_khr_fp16"))
    {
        log_info("cl_khr_fp16 is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    ptrSizes[0] = sizeof(cl_float) / 2;
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (float *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / (ptrSizes[0] * 2); j++ )
            inptr[i][j] = get_random_float( -32, 32, d );
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_half ), (char*)"half", 5, (void**)inptr,
                                    workgroup_half_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_half ), (char*)"half", 5, (void**)inptr,
                                    subgroup_half_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_half ), (char*)"half", 5, (void**)inptr,
                                    convenience_half_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_half ), (char*)"half", 5, (void**)inptr,
                                    half_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;
}

int test_pipe_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_long *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_long;

    ptrSizes[0] = sizeof(cl_long);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_long *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_long) genrand_int32(d) ^ ((cl_long) genrand_int32(d) << 32);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"long", 5, (void**)inptr,
                                   workgroup_long_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"long", 5, (void**)inptr,
                                   subgroup_long_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"long", 5, (void**)inptr,
                                   convenience_long_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_long ), (char*)"long", 5, (void**)inptr,
                                   long_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_ulong *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_ulong;

    ptrSizes[0] = sizeof(cl_ulong);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support long
    if (! gHasLong )
    {
        log_info( "Device does not support 64-bit integers. Skipping test.\n" );
        return CL_SUCCESS;
    }

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_ulong *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong", 5, (void**)inptr,
                                   workgroup_ulong_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong", 5, (void**)inptr,
                                   subgroup_ulong_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong", 5, (void**)inptr,
                                   convenience_ulong_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_ulong ), (char*)"ulong", 5, (void**)inptr,
                                   ulong_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    cl_double *inptr[5];
    size_t  ptrSizes[5];
    int     i, err;
    cl_uint j;
    int     (*foo)(void *,void *,int);
    MTdata  d = init_genrand( gRandomSeed );

    size_t  min_alignment = get_min_alignment(context);

    foo = verify_readwrite_long;

    ptrSizes[0] = sizeof(cl_double);
    ptrSizes[1] = ptrSizes[0] << 1;
    ptrSizes[2] = ptrSizes[1] << 1;
    ptrSizes[3] = ptrSizes[2] << 1;
    ptrSizes[4] = ptrSizes[3] << 1;

    //skip devices that don't support double
    if(!is_extension_available(deviceID, "cl_khr_fp64"))
    {
        log_info("cl_khr_fp64 is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }

    for ( i = 0; i < 5; i++ ){
        inptr[i] = (cl_double *)align_malloc(ptrSizes[i] * num_elements, min_alignment);

        for ( j = 0; j < ptrSizes[i] * num_elements / ptrSizes[0]; j++ )
            inptr[i][j] = get_random_double( -32, 32, d );
    }

    if(useWorkgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_double ), (char*)"double", 5, (void**)inptr,
                                   workgroup_double_kernel_name, foo);
    }
    else if(useSubgroupReserve == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_double ), (char*)"double", 5, (void**)inptr,
                                   subgroup_double_kernel_name, foo);
    }
    else if(useConvenienceBuiltIn == 1){
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_double ), (char*)"double", 5, (void**)inptr,
                                   convenience_double_kernel_name, foo);
    }
    else{
        err = test_pipe_readwrite( deviceID, context, queue, num_elements, sizeof( cl_double ), (char*)"double", 5, (void**)inptr,
                                   double_kernel_name, foo);
    }

    for ( i = 0; i < 5; i++ ){
        align_free( (void *)inptr[i] );
    }
    free_mtdata(d);

    return err;

}

int test_pipe_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    const char *kernelNames[] = {"test_pipe_write_struct","test_pipe_read_struct"};
    return test_pipe_readwrite_struct_generic(deviceID, context, queue, num_elements, pipe_readwrite_struct_kernel_code, kernelNames);
}

// Work-group functions for pipe reserve/commits
int test_pipe_workgroup_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_int(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_uint(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_short(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_ushort(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_char(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_uchar(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_float(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_half(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_long(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_ulong(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useWorkgroupReserve = 1;
    useSubgroupReserve = 0;
    useConvenienceBuiltIn = 0;
    return test_pipe_readwrite_double(deviceID, context, queue, num_elements);
}

int test_pipe_workgroup_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    const char *kernelNames[] = {"test_pipe_workgroup_write_struct","test_pipe_workgroup_read_struct"};
    return test_pipe_readwrite_struct_generic(deviceID, context, queue, num_elements, pipe_workgroup_readwrite_struct_kernel_code, kernelNames);
}

// Sub-group functions for pipe reserve/commits
int test_pipe_subgroup_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_int(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_uint(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_short(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_ushort(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_char(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_uchar(deviceID, context, queue, num_elements);

}

int test_pipe_subgroup_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_float(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_half(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_long(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_ulong(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useSubgroupReserve = 1;
    useWorkgroupReserve = 0;
    useConvenienceBuiltIn = 0;

    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    return test_pipe_readwrite_double(deviceID, context, queue, num_elements);
}

int test_pipe_subgroup_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    if(!is_extension_available(deviceID, "cl_khr_subgroups"))
    {
        log_info("cl_khr_subgroups is not supported on this platoform. Skipping test.\n");
        return CL_SUCCESS;
    }
    const char *kernelNames[] = {"test_pipe_subgroup_write_struct","test_pipe_subgroup_read_struct"};
    return test_pipe_readwrite_struct_generic(deviceID, context, queue, num_elements, pipe_subgroup_readwrite_struct_kernel_code, kernelNames);
}

// Convenience functions for pipe reserve/commits
int test_pipe_convenience_readwrite_int( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_int(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_uint( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_uint(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_short( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_short(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_ushort( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_ushort(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_char( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_char(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_uchar( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_uchar(deviceID, context, queue, num_elements);
}


int test_pipe_convenience_readwrite_float( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_float(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_half( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_half(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_long( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_long(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_ulong( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_ulong(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_double( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    useConvenienceBuiltIn = 1;
    useSubgroupReserve = 0;
    useWorkgroupReserve = 0;

    return test_pipe_readwrite_double(deviceID, context, queue, num_elements);
}

int test_pipe_convenience_readwrite_struct( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements )
{
    const char *kernelNames[] = {"test_pipe_convenience_write_struct","test_pipe_convenience_read_struct"};
    return test_pipe_readwrite_struct_generic(deviceID, context, queue, num_elements, pipe_convenience_readwrite_struct_kernel_code, kernelNames);
}
