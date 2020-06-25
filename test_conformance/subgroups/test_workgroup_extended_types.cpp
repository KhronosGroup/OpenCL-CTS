//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "workgroup_common_kernels.h"
#include "workgroup_common_templates.h"
#include "harness/typeWrappers.h"

struct run_for_type
{
    run_for_type(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements,
                 bool useCoreSubgroups,
                 std::vector<std::string> required_extensions = {})
    {
        device_ = device;
        context_ = context;
        queue_ = queue;
        num_elements_ = num_elements;
        useCoreSubgroups_ = useCoreSubgroups;
        required_extensions_ = required_extensions;
    }

    template <typename T> cl_int run_bc()
    {
        cl_int error;
        error = test<T, BC<T>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_bcast",
            bcast_source, 0, useCoreSubgroups_, required_extensions_);
        return error;
    }
    template <typename T> cl_int run_red_cin_scex()
    {
        cl_int error;
        error = test<T, RED<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redadd",
            redadd_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, RED<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmax",
            redmax_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, RED<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_redmin",
            redmin_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCIN<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinadd",
            scinadd_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCIN<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinmax",
            scinmax_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCIN<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scinmin",
            scinmin_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCEX<T, 0>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexadd",
            scexadd_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCEX<T, 1>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexmax",
            scexmax_source, 0, useCoreSubgroups_, required_extensions_);
        error |= test<T, SCEX<T, 2>, G, L>::run(
            device_, context_, queue_, num_elements_, "test_scexmin",
            scexmin_source, 0, useCoreSubgroups_, required_extensions_);

        return error;
    }

private:
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    int num_elements_;
    bool useCoreSubgroups_;
    std::vector<std::string> required_extensions_;
};

int test_work_group_functions_extended_types(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    int error;
    std::vector<std::string> required_extensions;
    required_extensions = { "cl_khr_subgroup_extended_types" };
    run_for_type rft(device, context, queue, num_elements, true,
                     required_extensions);

    error = rft.run_bc<cl_uint2>();
    error |= rft.run_bc<subgroups::cl_uint3>();
    error |= rft.run_bc<cl_uint4>();
    error |= rft.run_bc<cl_uint8>();
    error |= rft.run_bc<cl_uint16>();
    error |= rft.run_bc<cl_int2>();
    error |= rft.run_bc<subgroups::cl_int3>();
    error |= rft.run_bc<cl_int4>();
    error |= rft.run_bc<cl_int8>();
    error |= rft.run_bc<cl_int16>();

    error |= rft.run_bc<cl_ulong2>();
    error |= rft.run_bc<subgroups::cl_ulong3>();
    error |= rft.run_bc<cl_ulong4>();
    error |= rft.run_bc<cl_ulong8>();
    error |= rft.run_bc<cl_ulong16>();
    error |= rft.run_bc<cl_long2>();
    error |= rft.run_bc<subgroups::cl_long3>();
    error |= rft.run_bc<cl_long4>();
    error |= rft.run_bc<cl_long8>();
    error |= rft.run_bc<cl_long16>();

    error |= rft.run_bc<cl_float2>();
    error |= rft.run_bc<subgroups::cl_float3>();
    error |= rft.run_bc<cl_float4>();
    error |= rft.run_bc<cl_float8>();
    error |= rft.run_bc<cl_float16>();

    error |= rft.run_bc<cl_double2>();
    error |= rft.run_bc<subgroups::cl_double3>();
    error |= rft.run_bc<cl_double4>();
    error |= rft.run_bc<cl_double8>();
    error |= rft.run_bc<cl_double16>();

    error |= rft.run_bc<cl_ushort2>();
    error |= rft.run_bc<subgroups::cl_ushort3>();
    error |= rft.run_bc<cl_ushort4>();
    error |= rft.run_bc<cl_ushort8>();
    error |= rft.run_bc<cl_ushort16>();
    error |= rft.run_bc<cl_short2>();
    error |= rft.run_bc<subgroups::cl_short3>();
    error |= rft.run_bc<cl_short4>();
    error |= rft.run_bc<cl_short8>();
    error |= rft.run_bc<cl_short16>();

    error |= rft.run_bc<cl_uchar2>();
    error |= rft.run_bc<subgroups::cl_uchar3>();
    error |= rft.run_bc<cl_uchar4>();
    error |= rft.run_bc<cl_uchar8>();
    error |= rft.run_bc<cl_uchar16>();
    error |= rft.run_bc<cl_char2>();
    error |= rft.run_bc<subgroups::cl_char3>();
    error |= rft.run_bc<cl_char4>();
    error |= rft.run_bc<cl_char8>();
    error |= rft.run_bc<cl_char16>();

    error |= rft.run_bc<subgroups::cl_half2>();
    error |= rft.run_bc<subgroups::cl_half3>();
    error |= rft.run_bc<subgroups::cl_half4>();
    error |= rft.run_bc<subgroups::cl_half8>();
    error |= rft.run_bc<subgroups::cl_half16>();

    error |= rft.run_red_cin_scex<cl_uchar>();
    error |= rft.run_red_cin_scex<cl_char>();
    error |= rft.run_red_cin_scex<cl_ushort>();
    error |= rft.run_red_cin_scex<cl_short>();
    return error;
}