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
#include "procs.h"
#include "subhelpers.h"
#include "subgroup_common_templates.h"
#include "harness/typeWrappers.h"
#include <bitset>

namespace {

static const char* shuffle_xor_source =
    "__kernel void test_sub_group_shuffle_xor(const __global Type *in, "
    "__global int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    out[gid] = sub_group_shuffle_xor(x, xy[gid].z);"
    "}\n";

static const char* shuffle_source =
    "__kernel void test_sub_group_shuffle(const __global Type *in, __global "
    "int4 *xy, __global Type *out)\n"
    "{\n"
    "    int gid = get_global_id(0);\n"
    "    XY(xy,gid);\n"
    "    Type x = in[gid];\n"
    "    out[gid] = sub_group_shuffle(x, xy[gid].z);"
    "}\n";

template <typename T> int run_shuffle_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SHF<T, ShuffleOp::shuffle>>(
        "test_sub_group_shuffle", shuffle_source);
    error |= rft.run_impl<T, SHF<T, ShuffleOp::shuffle_xor>>(
        "test_sub_group_shuffle_xor", shuffle_xor_source);
    return error;
}

}

int test_subgroup_functions_shuffle(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    std::vector<std::string> required_extensions{ "cl_khr_subgroup_shuffle" };
    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size,
                                required_extensions);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_shuffle_for_type<cl_int>(rft);
    error |= run_shuffle_for_type<cl_uint>(rft);
    error |= run_shuffle_for_type<cl_long>(rft);
    error |= run_shuffle_for_type<cl_ulong>(rft);
    error |= run_shuffle_for_type<cl_short>(rft);
    error |= run_shuffle_for_type<cl_ushort>(rft);
    error |= run_shuffle_for_type<cl_char>(rft);
    error |= run_shuffle_for_type<cl_uchar>(rft);
    error |= run_shuffle_for_type<cl_float>(rft);
    error |= run_shuffle_for_type<cl_double>(rft);
    error |= run_shuffle_for_type<subgroups::cl_half>(rft);

    return error;
}
