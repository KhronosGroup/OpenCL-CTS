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
#include "subgroup_common_kernels.h"
#include "subgroup_common_templates.h"
#include "harness/typeWrappers.h"

namespace {

template <typename T> int run_broadcast_for_extended_type(RunTestForType rft)
{
    int error = rft.run_impl<T, BC<T, SubgroupsBroadcastOp::broadcast>>(
        "sub_group_broadcast");
    return error;
}

template <typename T> int run_scan_reduction_for_type(RunTestForType rft)
{
    int error =
        rft.run_impl<T, RED_NU<T, ArithmeticOp::add_>>("sub_group_reduce_add");
    error |=
        rft.run_impl<T, RED_NU<T, ArithmeticOp::max_>>("sub_group_reduce_max");
    error |=
        rft.run_impl<T, RED_NU<T, ArithmeticOp::min_>>("sub_group_reduce_min");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::add_>>(
        "sub_group_scan_inclusive_add");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::max_>>(
        "sub_group_scan_inclusive_max");
    error |= rft.run_impl<T, SCIN_NU<T, ArithmeticOp::min_>>(
        "sub_group_scan_inclusive_min");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::add_>>(
        "sub_group_scan_exclusive_add");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::max_>>(
        "sub_group_scan_exclusive_max");
    error |= rft.run_impl<T, SCEX_NU<T, ArithmeticOp::min_>>(
        "sub_group_scan_exclusive_min");
    return error;
}


}

int test_subgroup_functions_extended_types(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    if (!is_extension_available(device, "cl_khr_subgroup_extended_types"))
    {
        log_info("cl_khr_subgroup_extended_types is not supported on this "
                 "device, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    constexpr size_t global_work_size = 2000;
    constexpr size_t local_work_size = 200;
    WorkGroupParams test_params(global_work_size, local_work_size);
    test_params.save_kernel_source(sub_group_reduction_scan_source);
    test_params.save_kernel_source(sub_group_generic_source,
                                   "sub_group_broadcast");

    RunTestForType rft(device, context, queue, num_elements, test_params);
    int error = run_broadcast_for_extended_type<cl_uint2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_uint3>(rft);
    error |= run_broadcast_for_extended_type<cl_uint4>(rft);
    error |= run_broadcast_for_extended_type<cl_uint8>(rft);
    error |= run_broadcast_for_extended_type<cl_uint16>(rft);
    error |= run_broadcast_for_extended_type<cl_int2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_int3>(rft);
    error |= run_broadcast_for_extended_type<cl_int4>(rft);
    error |= run_broadcast_for_extended_type<cl_int8>(rft);
    error |= run_broadcast_for_extended_type<cl_int16>(rft);

    error |= run_broadcast_for_extended_type<cl_ulong2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_ulong3>(rft);
    error |= run_broadcast_for_extended_type<cl_ulong4>(rft);
    error |= run_broadcast_for_extended_type<cl_ulong8>(rft);
    error |= run_broadcast_for_extended_type<cl_ulong16>(rft);
    error |= run_broadcast_for_extended_type<cl_long2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_long3>(rft);
    error |= run_broadcast_for_extended_type<cl_long4>(rft);
    error |= run_broadcast_for_extended_type<cl_long8>(rft);
    error |= run_broadcast_for_extended_type<cl_long16>(rft);

    error |= run_broadcast_for_extended_type<cl_float2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_float3>(rft);
    error |= run_broadcast_for_extended_type<cl_float4>(rft);
    error |= run_broadcast_for_extended_type<cl_float8>(rft);
    error |= run_broadcast_for_extended_type<cl_float16>(rft);

    error |= run_broadcast_for_extended_type<cl_double2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_double3>(rft);
    error |= run_broadcast_for_extended_type<cl_double4>(rft);
    error |= run_broadcast_for_extended_type<cl_double8>(rft);
    error |= run_broadcast_for_extended_type<cl_double16>(rft);

    error |= run_broadcast_for_extended_type<cl_ushort>(rft);
    error |= run_broadcast_for_extended_type<cl_ushort2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_ushort3>(rft);
    error |= run_broadcast_for_extended_type<cl_ushort4>(rft);
    error |= run_broadcast_for_extended_type<cl_ushort8>(rft);
    error |= run_broadcast_for_extended_type<cl_ushort16>(rft);
    error |= run_broadcast_for_extended_type<cl_short>(rft);
    error |= run_broadcast_for_extended_type<cl_short2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_short3>(rft);
    error |= run_broadcast_for_extended_type<cl_short4>(rft);
    error |= run_broadcast_for_extended_type<cl_short8>(rft);
    error |= run_broadcast_for_extended_type<cl_short16>(rft);

    error |= run_broadcast_for_extended_type<cl_uchar>(rft);
    error |= run_broadcast_for_extended_type<cl_uchar2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_uchar3>(rft);
    error |= run_broadcast_for_extended_type<cl_uchar4>(rft);
    error |= run_broadcast_for_extended_type<cl_uchar8>(rft);
    error |= run_broadcast_for_extended_type<cl_uchar16>(rft);
    error |= run_broadcast_for_extended_type<cl_char>(rft);
    error |= run_broadcast_for_extended_type<cl_char2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_char3>(rft);
    error |= run_broadcast_for_extended_type<cl_char4>(rft);
    error |= run_broadcast_for_extended_type<cl_char8>(rft);
    error |= run_broadcast_for_extended_type<cl_char16>(rft);

    error |= run_broadcast_for_extended_type<subgroups::cl_half2>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_half3>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_half4>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_half8>(rft);
    error |= run_broadcast_for_extended_type<subgroups::cl_half16>(rft);

    error |= run_scan_reduction_for_type<cl_uchar>(rft);
    error |= run_scan_reduction_for_type<cl_char>(rft);
    error |= run_scan_reduction_for_type<cl_ushort>(rft);
    error |= run_scan_reduction_for_type<cl_short>(rft);
    return error;
}
