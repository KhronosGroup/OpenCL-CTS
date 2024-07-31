//
// Copyright (c) 2022 The Khronos Group Inc.
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
#include "subgroup_common_kernels.h"
#include "subgroup_common_templates.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

namespace {

template <typename T> int run_rotate_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SHF<T, ShuffleOp::rotate>>("sub_group_rotate");
    return error;
}

std::string sub_group_clustered_rotate_source = R"(
    __kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out,
                          uint cluster_size) {
        Type r;
        int gid = get_global_id(0);
        XY(xy,gid);
        Type x = in[gid];
        int delta = xy[gid].z;
        switch (cluster_size) {
            case 1: r = %s(x, delta, 1); break;
            case 2: r = %s(x, delta, 2); break;
            case 4: r = %s(x, delta, 4); break;
            case 8: r = %s(x, delta, 8); break;
            case 16: r = %s(x, delta, 16); break;
            case 32: r = %s(x, delta, 32); break;
            case 64: r = %s(x, delta, 64); break;
            case 128: r = %s(x, delta, 128); break;
        }
        out[gid] = r;
    }
)";

template <typename T> int run_clustered_rotate_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SHF<T, ShuffleOp::clustered_rotate>>(
        "sub_group_clustered_rotate");
    return error;
}

}

int test_subgroup_functions_rotate(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    if (!is_extension_available(device, "cl_khr_subgroup_rotate"))
    {
        log_info("cl_khr_subgroup_rotate is not supported on this device, "
                 "skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.save_kernel_source(sub_group_generic_source);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_rotate_for_type<cl_int>(rft);
    error |= run_rotate_for_type<cl_uint>(rft);
    error |= run_rotate_for_type<cl_long>(rft);
    error |= run_rotate_for_type<cl_ulong>(rft);
    error |= run_rotate_for_type<cl_short>(rft);
    error |= run_rotate_for_type<cl_ushort>(rft);
    error |= run_rotate_for_type<cl_char>(rft);
    error |= run_rotate_for_type<cl_uchar>(rft);
    error |= run_rotate_for_type<cl_float>(rft);
    error |= run_rotate_for_type<cl_double>(rft);
    error |= run_rotate_for_type<subgroups::cl_half>(rft);

    WorkGroupParams test_params_clustered(global_work_size, local_work_size, -1,
                                          3);
    test_params_clustered.save_kernel_source(sub_group_clustered_rotate_source);
    RunTestForType rft_clustered(device, context, queue, num_elements,
                                 test_params_clustered);

    error |= run_clustered_rotate_for_type<cl_int>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_uint>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_long>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_ulong>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_short>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_ushort>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_char>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_uchar>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_float>(rft_clustered);
    error |= run_clustered_rotate_for_type<cl_double>(rft_clustered);
    error |= run_clustered_rotate_for_type<subgroups::cl_half>(rft_clustered);

    return error;
}
