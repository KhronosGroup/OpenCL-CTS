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
#include "harness/typeWrappers.h"
#include "subgroup_common_templates.h"

namespace {

std::string sub_group_non_uniform_arithmetic_source = R"(
    __kernel void test_%s(const __global Type *in, __global int4 *xy, __global Type *out) {
        int gid = get_global_id(0);
        XY(xy,gid);
        uint subgroup_local_id = get_sub_group_local_id();
        uint elect_work_item = 1 << (subgroup_local_id % 32);
        uint work_item_mask;
        if(subgroup_local_id < 32) {
            work_item_mask = work_item_mask_vector.x;
        } else if(subgroup_local_id < 64) {
            work_item_mask = work_item_mask_vector.y;
        } else if(subgroup_local_id < 96) {
            work_item_mask = work_item_mask_vector.w;
        } else if(subgroup_local_id < 128) {
            work_item_mask = work_item_mask_vector.z;
        }
        if (elect_work_item & work_item_mask){
            out[gid] = %s(in[gid]);
        }
    }
)";

template <typename T>
int run_functions_add_mul_max_min_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::add_>>(
        "sub_group_non_uniform_scan_inclusive_add");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::mul_>>(
        "sub_group_non_uniform_scan_inclusive_mul");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::max_>>(
        "sub_group_non_uniform_scan_inclusive_max");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::min_>>(
        "sub_group_non_uniform_scan_inclusive_min");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::add_>>(
        "sub_group_non_uniform_scan_exclusive_add");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::mul_>>(
        "sub_group_non_uniform_scan_exclusive_mul");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::max_>>(
        "sub_group_non_uniform_scan_exclusive_max");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::min_>>(
        "sub_group_non_uniform_scan_exclusive_min");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::add_>>(
        "sub_group_non_uniform_reduce_add");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::mul_>>(
        "sub_group_non_uniform_reduce_mul");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::max_>>(
        "sub_group_non_uniform_reduce_max");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::min_>>(
        "sub_group_non_uniform_reduce_min");
    return error;
}

template <typename T> int run_functions_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::and_>>(
        "sub_group_non_uniform_scan_inclusive_and");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::or_>>(
        "sub_group_non_uniform_scan_inclusive_or");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::xor_>>(
        "sub_group_non_uniform_scan_inclusive_xor");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::and_>>(
        "sub_group_non_uniform_scan_exclusive_and");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::or_>>(
        "sub_group_non_uniform_scan_exclusive_or");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::xor_>>(
        "sub_group_non_uniform_scan_exclusive_xor");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::and_>>(
        "sub_group_non_uniform_reduce_and");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::or_>>(
        "sub_group_non_uniform_reduce_or");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::xor_>>(
        "sub_group_non_uniform_reduce_xor");
    return error;
}

template <typename T>
int run_functions_logical_and_or_xor_for_type(RunTestForType rft)
{
    int error = rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_and>>(
        "sub_group_non_uniform_scan_inclusive_logical_and");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_or>>(
        "sub_group_non_uniform_scan_inclusive_logical_or");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::logical_xor>>(
        "sub_group_non_uniform_scan_inclusive_logical_xor");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_and>>(
        "sub_group_non_uniform_scan_exclusive_logical_and");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_or>>(
        "sub_group_non_uniform_scan_exclusive_logical_or");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::logical_xor>>(
        "sub_group_non_uniform_scan_exclusive_logical_xor");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_and>>(
        "sub_group_non_uniform_reduce_logical_and");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_or>>(
        "sub_group_non_uniform_reduce_logical_or");
    error |= rft.run_impl<T, RED_NU<T, ArithmeticOp::logical_xor>>(
        "sub_group_non_uniform_reduce_logical_xor");
    return error;
}

}

int test_subgroup_functions_non_uniform_arithmetic(cl_device_id device,
                                                   cl_context context,
                                                   cl_command_queue queue,
                                                   int num_elements)
{
    if (!is_extension_available(device,
                                "cl_khr_subgroup_non_uniform_arithmetic"))
    {
        log_info("cl_khr_subgroup_non_uniform_arithmetic is not supported on "
                 "this device, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size, true);
    test_params.save_kernel_source(sub_group_non_uniform_arithmetic_source);
    RunTestForType rft(device, context, queue, num_elements, test_params);

    int error = run_functions_add_mul_max_min_for_type<cl_int>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_uint>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_long>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_ulong>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_short>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_ushort>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_char>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_uchar>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_float>(rft);
    error |= run_functions_add_mul_max_min_for_type<cl_double>(rft);
    error |= run_functions_add_mul_max_min_for_type<subgroups::cl_half>(rft);

    error |= run_functions_and_or_xor_for_type<cl_int>(rft);
    error |= run_functions_and_or_xor_for_type<cl_uint>(rft);
    error |= run_functions_and_or_xor_for_type<cl_long>(rft);
    error |= run_functions_and_or_xor_for_type<cl_ulong>(rft);
    error |= run_functions_and_or_xor_for_type<cl_short>(rft);
    error |= run_functions_and_or_xor_for_type<cl_ushort>(rft);
    error |= run_functions_and_or_xor_for_type<cl_char>(rft);
    error |= run_functions_and_or_xor_for_type<cl_uchar>(rft);

    error |= run_functions_logical_and_or_xor_for_type<cl_int>(rft);
    return error;
}
