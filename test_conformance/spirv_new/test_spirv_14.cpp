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
#include "types.hpp"

#include <string>

extern bool gVersionSkip;

static int check_spirv_14_support(cl_device_id deviceID)
{
    const char* cVersionString = "SPIR-V 1.4";

    std::string ilVersions = get_device_il_version_string(deviceID);

    if (gVersionSkip)
    {
        log_info("    Skipping version check for %s.\n", cVersionString);
    }
    else if (ilVersions.find(cVersionString) == std::string::npos)
    {
        log_info("    Version %s is not supported; skipping test.\n",
                 cVersionString);
        return TEST_SKIPPED_ITSELF;
    }

    return TEST_PASS;
}

static int test_image_operand_helper(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, bool signExtend)
{
    const char* filename = signExtend ? "spv1.4/image_operand_signextend"
                                      : "spv1.4/image_operand_zeroextend";
    cl_image_format image_format = {
        CL_RGBA,
        signExtend ? CL_SIGNED_INT8 : CL_UNSIGNED_INT8,
    };

    cl_int error = CL_SUCCESS;

    std::vector<cl_uchar> imgData({ 0x1, 0x80, 0xFF, 0x0 });
    std::vector<cl_uint> expected;
    for (auto v : imgData)
    {
        if (signExtend)
        {
            expected.push_back((cl_int)(cl_char)v);
        }
        else
        {
            expected.push_back(v);
        }
    }

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, filename);
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "read_image_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    std::vector<cl_uint> h_dst({ 0, 0, 0, 0 });
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       h_dst.size() * sizeof(cl_uint), h_dst.data(), &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    std::vector<cl_uchar> h_imgdata({ 0x1, 0x80, 0xFF, 0x0 });
    clMemWrapper src =
        clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format, 1, 1, 0, h_imgdata.data(), &error);
    SPIRV_CHECK_ERROR(error, "Failed to create src image");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(src), &src);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                h_dst.size() * sizeof(cl_uint), h_dst.data(), 0,
                                NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (h_dst != expected)
    {
        log_error("Mismatch! Got: %u, %u, %u, %u, Wanted: %u, %u, %u, %u\n",
                  h_dst[0], h_dst[1], h_dst[2], h_dst[3], expected[0],
                  expected[1], expected[2], expected[3]);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

TEST_SPIRV_FUNC(spirv14_image_operand_signextend)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_image_operand_helper(deviceID, context, queue, true);
}

TEST_SPIRV_FUNC(spirv14_image_operand_zeroextend)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_image_operand_helper(deviceID, context, queue, false);
}

static int test_loop_control_helper(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue,
                                    const char* filename)
{
    const int count = 10;
    const int value = 5;

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    std::string full_filename = "spv1.4/" + std::string(filename);
    error = get_program_with_il(prog, deviceID, context, full_filename.c_str());
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "loop_control_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    int h_dst = 0;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                      sizeof(h_dst), &h_dst, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(count), &count);
    error |= clSetKernelArg(kernel, 2, sizeof(value), &value);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(h_dst), &h_dst,
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (h_dst != count * value)
    {
        log_error("Mismatch! Got: %i, Wanted: %i\n", h_dst, count * value);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

TEST_SPIRV_FUNC(spirv14_loop_control_miniterations)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_loop_control_helper(deviceID, context, queue,
                                    "loop_control_miniterations");
}

TEST_SPIRV_FUNC(spirv14_loop_control_maxiterations)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_loop_control_helper(deviceID, context, queue,
                                    "loop_control_maxiterations");
}

TEST_SPIRV_FUNC(spirv14_loop_control_iterationmultiple)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_loop_control_helper(deviceID, context, queue,
                                    "loop_control_iterationmultiple");
}

TEST_SPIRV_FUNC(spirv14_loop_control_peelcount)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_loop_control_helper(deviceID, context, queue,
                                    "loop_control_peelcount");
}

TEST_SPIRV_FUNC(spirv14_loop_control_partialcount)
{
    int check = check_spirv_14_support(deviceID);
    if (check != TEST_PASS)
    {
        return check;
    }

    return test_loop_control_helper(deviceID, context, queue,
                                    "loop_control_partialcount");
}
