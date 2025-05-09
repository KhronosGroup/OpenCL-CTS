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
#include "subhelpers.h"
#include "subgroup_common_kernels.h"
#include "subgroup_common_templates.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

namespace {

template <typename T> int run_shuffle_relative_for_type(RunTestForType rft)
{
    int error =
        rft.run_impl<T, SHF<T, ShuffleOp::shuffle_up>>("sub_group_shuffle_up");
    error |= rft.run_impl<T, SHF<T, ShuffleOp::shuffle_down>>(
        "sub_group_shuffle_down");
    return error;
}

}

REGISTER_TEST(subgroup_functions_shuffle_relative)
{
    if (!is_extension_available(device, "cl_khr_subgroup_shuffle_relative"))
    {
        log_info("cl_khr_subgroup_shuffle_relative is not supported on this "
                 "device, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.save_kernel_source(sub_group_generic_source);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_shuffle_relative_for_type<cl_int>(rft);
    error |= run_shuffle_relative_for_type<cl_uint>(rft);
    error |= run_shuffle_relative_for_type<cl_long>(rft);
    error |= run_shuffle_relative_for_type<cl_ulong>(rft);
    error |= run_shuffle_relative_for_type<cl_short>(rft);
    error |= run_shuffle_relative_for_type<cl_ushort>(rft);
    error |= run_shuffle_relative_for_type<cl_char>(rft);
    error |= run_shuffle_relative_for_type<cl_uchar>(rft);
    error |= run_shuffle_relative_for_type<cl_float>(rft);
    error |= run_shuffle_relative_for_type<cl_double>(rft);
    error |= run_shuffle_relative_for_type<subgroups::cl_half>(rft);

    return error;
}
