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

#include <string>
#include <vector>

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

    clMemWrapper src =
        clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format, 1, 1, 0, imgData.data(), &error);
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

REGISTER_TEST(spirv14_image_operand_signextend)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_image_operand_helper(device, context, queue, true);
}

REGISTER_TEST(spirv14_image_operand_zeroextend)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_image_operand_helper(device, context, queue, false);
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

REGISTER_TEST(spirv14_loop_control_miniterations)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_loop_control_helper(device, context, queue,
                                    "loop_control_miniterations");
}

REGISTER_TEST(spirv14_loop_control_maxiterations)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_loop_control_helper(device, context, queue,
                                    "loop_control_maxiterations");
}

REGISTER_TEST(spirv14_loop_control_iterationmultiple)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_loop_control_helper(device, context, queue,
                                    "loop_control_iterationmultiple");
}

REGISTER_TEST(spirv14_loop_control_peelcount)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_loop_control_helper(device, context, queue,
                                    "loop_control_peelcount");
}

REGISTER_TEST(spirv14_loop_control_partialcount)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }
    return test_loop_control_helper(device, context, queue,
                                    "loop_control_partialcount");
}

REGISTER_TEST(spirv14_ptrops)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, "spv1.4/ptrops");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "ptrops_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    std::vector<cl_int> results(3);

    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       results.size() * sizeof(cl_int), NULL, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    clMemWrapper tst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(cl_int), NULL, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create tst buffer");

    // Test with different pointers:
    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(tst), &tst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                results.size() * sizeof(cl_int), results.data(),
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (results[0] != (dst == tst) || results[1] != (dst != tst)
        || results[2] == 0 /* dst - tst */)
    {
        log_error(
            "Results mismatch with different pointers!  Got: %i, %i, %i\n",
            results[0], results[1], results[2]);
        return TEST_FAIL;
    }

    // Test with equal pointers:
    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                results.size() * sizeof(cl_int), results.data(),
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (results[0] != (dst == dst) || results[1] != (dst != dst)
        || results[2] != 0 /* dst - dst */)
    {
        log_error("Results mismatch with equal pointers!  Got: %i, %i, %i\n",
                  results[0], results[1], results[2]);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

static int test_usersemantic_decoration(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        bool test_memberdecoratestring)
{
    cl_int error = CL_SUCCESS;

    const char* filename = test_memberdecoratestring
        ? "spv1.4/usersemantic_memberdecoratestring"
        : "spv1.4/usersemantic_decoratestring";

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, filename);
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "usersemantic_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    int h_dst = -1;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                      sizeof(h_dst), &h_dst, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(h_dst), &h_dst,
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (h_dst != 0)
    {
        log_error("Mismatch! Got: %i, Wanted: %i\n", h_dst, 0);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv14_usersemantic_decoratestring)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_usersemantic_decoration(device, context, queue, false);
}

REGISTER_TEST(spirv14_usersemantic_memberdecoratestring)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_usersemantic_decoration(device, context, queue, true);
}

REGISTER_TEST(spirv14_nonwriteable_decoration)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(
        prog, device, context,
        "spv1.4/nonwriteable_decoration_function_storage_class");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "nonwriteable_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    int result = 0;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(result), nullptr, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr,
                                   0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(result), &result,
                                0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    int expected = 42;
    if (result != expected)
    {
        log_error("Result mismatch!  Got %d, Wanted %d\n", result, expected);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv14_copymemory_memory_operands)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context,
                                "spv1.4/copymemory_memory_operands");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "copymemory_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    std::vector<int> results(6);
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       results.size() * sizeof(results[0]), nullptr, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr,
                                   0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                results.size() * sizeof(results[0]),
                                results.data(), 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    const int expected = 42;
    for (auto result : results)
    {
        if (result != expected)
        {
            log_error("Result mismatch!  Got %d, Wanted %d\n", result,
                      expected);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv14_select_composite)
{
    constexpr size_t global_size = 16;

    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, "spv1.4/select_struct");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "select_struct_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    struct TestStruct
    {
        cl_int i;
        cl_float f;
    };

    std::vector<TestStruct> results(global_size);
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       results.size() * sizeof(results[0]), nullptr, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                results.size() * sizeof(results[0]),
                                results.data(), 0, nullptr, nullptr);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    const TestStruct struct_a{ 1024, 3.1415f };
    const TestStruct struct_b{ 2048, 2.7128f };

    for (size_t i = 0; i < global_size; i++)
    {
        const TestStruct& expected = (i & 1) ? struct_a : struct_b;
        if (results[i].i != expected.i || results[i].f != expected.f)
        {
            log_error("Result mismatch at index %zu!  Got {%d, %f}, Wanted "
                      "{%d, %f}\n",
                      i, results[i].i, results[i].f, expected.i, expected.f);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(spirv14_copylogical)
{
    if (!is_spirv_version_supported(device, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;
    clProgramWrapper prog;
    error =
        get_program_with_il(prog, device, context, "spv1.4/copylogical_struct");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "copylogical_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    struct TestStruct
    {
        cl_int i;
        cl_float f;
    };
    TestStruct results{ 0, 0.0f };
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(results), NULL, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(results),
                                &results, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (results.i != 1024 || results.f != 3.1415f)
    {
        log_error("Results mismatch!  Got: { %d, %f }\n", results.i, results.f);
        return TEST_FAIL;
    }

    return TEST_PASS;
}
