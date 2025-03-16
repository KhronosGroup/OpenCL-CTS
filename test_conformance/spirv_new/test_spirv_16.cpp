//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include <string>
#include <vector>

REGISTER_TEST(spirv16_image_operand_nontemporal)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.6"))
    {
        log_info("SPIR-V 1.6 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context,
                                "spv1.6/image_operand_nontemporal");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel =
        clCreateKernel(prog, "read_write_image_nontemporal", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    cl_image_format image_format = {
        CL_RGBA,
        CL_FLOAT,
    };

    std::vector<cl_float> imgData({ 1.0f, 2.0f, -1.0f, 128.0f });
    clMemWrapper src =
        clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format, 1, 1, 0, imgData.data(), &error);
    SPIRV_CHECK_ERROR(error, "Failed to create src image");

    std::vector<cl_float> h_dst({ 0, 0, 0, 0 });
    clMemWrapper dst =
        clCreateImage2D(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        &image_format, 1, 1, 0, h_dst.data(), &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst image");

    error |= clSetKernelArg(kernel, 0, sizeof(src), &src);
    error |= clSetKernelArg(kernel, 1, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    const size_t origin[] = { 0, 0, 0 };
    const size_t region[] = { 1, 1, 1 };
    error = clEnqueueReadImage(queue, dst, CL_TRUE, origin, region, 0, 0,
                               h_dst.data(), 0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination image");

    if (h_dst != imgData)
    {
        log_error("Mismatch! Got: %f, %f, %f, %f, Wanted: %f, %f, %f, %f\n",
                  h_dst[0], h_dst[1], h_dst[2], h_dst[3], imgData[0],
                  imgData[1], imgData[2], imgData[3]);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

static int test_uniformdecoration_helper(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         bool test_uniformid)
{
    constexpr size_t global_size = 16;
    const cl_uint value = 42;
    const cl_uint check = value + 1;

    const char* filename = test_uniformid ? "spv1.6/uniformdecoration_uniformid"
                                          : "spv1.6/uniformdecoration_uniform";

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, filename);
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel =
        clCreateKernel(prog, "test_uniformdecoration", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    std::vector<cl_uint> h_dst(global_size);
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       h_dst.size() * sizeof(cl_uint), nullptr, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(value), &value);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                h_dst.size() * sizeof(cl_uint), h_dst.data(), 0,
                                nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Unable to read dst buffer");

    for (size_t i = 0; i < global_size; i++)
    {
        if (h_dst[i] != check)
        {
            log_error("Result mismatch at index %zu!  Got %u, wanted %u.\n", i,
                      h_dst[i], check);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv16_uniformdecoration_uniform)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.6"))
    {
        log_info("SPIR-V 1.6 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_uniformdecoration_helper(device, context, queue, false);
}

REGISTER_TEST(spirv16_uniformdecoration_uniformid)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.6"))
    {
        log_info("SPIR-V 1.6 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_uniformdecoration_helper(device, context, queue, true);
}
