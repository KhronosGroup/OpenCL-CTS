//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include "testBase.h"
#include "spirvInfo.hpp"
#include "types.hpp"

#include <algorithm>
#include <cinttypes>
#include <vector>

REGISTER_TEST(spirv15_ptr_bitcast)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.5"))
    {
        log_info("SPIR-V 1.5 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    cl_uint address_bits;
    error = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint),
                            &address_bits, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to get address bits");

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, "spv1.5/ptr_bitcast");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "ptr_bitcast_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    cl_ulong result_ulong =
        address_bits == 32 ? 0xAAAAAAAAUL : 0xAAAAAAAAAAAAAAAAUL;
    cl_ulong result_uint2 =
        address_bits == 32 ? 0x55555555UL : 0x5555555555555555UL;

    clMemWrapper dst_ulong =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(result_ulong), &result_ulong, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst_ulong buffer");

    clMemWrapper dst_uint2 =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(result_uint2), &result_uint2, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst_uint2 buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst_ulong), &dst_ulong);
    error |= clSetKernelArg(kernel, 1, sizeof(dst_uint2), &dst_uint2);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error =
        clEnqueueReadBuffer(queue, dst_ulong, CL_TRUE, 0, sizeof(result_ulong),
                            &result_ulong, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read dst_ulong buffer");

    error =
        clEnqueueReadBuffer(queue, dst_uint2, CL_TRUE, 0, sizeof(result_uint2),
                            &result_uint2, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read dst_uint2 buffer");

    if (result_ulong != result_uint2)
    {
        log_error("Results mismatch!  ulong = 0x%016" PRIx64
                  " vs. uint2 = 0x%016" PRIx64 "\n",
                  result_ulong, result_uint2);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv15_non_uniform_broadcast)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.5"))
    {
        log_info("SPIR-V 1.5 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_khr_subgroup_ballot"))
    {
        log_info("cl_khr_subgroup_ballot is not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context,
                                "spv1.5/non_uniform_broadcast_dynamic_index");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(
        prog, "non_uniform_broadcast_dynamic_index_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    // Get the local work-group size for one sub-group per work-group.
    size_t lws = 0;
    size_t one = 1;
    error = clGetKernelSubGroupInfo(
        kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
        sizeof(size_t), &one, sizeof(size_t), &lws, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to get local work size for one sub-group");

    // Use four work-groups, unless the local-group size is less than four.
    size_t wgcount = std::min<size_t>(lws, 4);
    size_t gws = wgcount * lws;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(cl_int) * gws, NULL, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL,
                                   NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    std::vector<cl_int> results(gws);
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(cl_int) * gws,
                                results.data(), 0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    // Remember: the test kernel did:
    //  sub_group_non_uniform_broadcast(get_global_id(0), get_group_id(0))
    for (size_t g = 0; g < wgcount; g++)
    {
        for (size_t l = 0; l < lws; l++)
        {
            size_t index = g * lws + l;
            size_t check = g * lws + g;
            if (results[index] != static_cast<cl_int>(check))
            {
                log_error("Result mismatch at index %zu!  Got %d, Wanted %zu\n",
                          index, results[index], check);
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}
