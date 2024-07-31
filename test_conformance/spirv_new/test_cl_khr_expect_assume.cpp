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

#include <vector>

template <typename T> struct TestInfo
{
};
template <> struct TestInfo<cl_char>
{
    using argType = cl_char;
    static constexpr const char* typeName = "char";
    static constexpr const char* testName = "expect_char";
};
template <> struct TestInfo<cl_short>
{
    using argType = cl_short;
    static constexpr const char* typeName = "short";
    static constexpr const char* testName = "expect_short";
};
template <> struct TestInfo<cl_int>
{
    using argType = cl_int;
    static constexpr const char* typeName = "int";
    static constexpr const char* testName = "expect_int";
};
template <> struct TestInfo<cl_long>
{
    using argType = cl_long;
    static constexpr const char* typeName = "long";
    static constexpr const char* testName = "expect_long";
};
template <> struct TestInfo<cl_bool>
{
    using argType = cl_int;
    static constexpr const char* typeName = "bool";
    static constexpr const char* testName = "expect_bool";
};

template <typename T>
static int test_expect_type(cl_device_id device, cl_context context,
                            cl_command_queue queue)
{
    using ArgType = typename TestInfo<T>::argType;

    log_info("    testing type %s\n", TestInfo<T>::typeName);

    const ArgType value = 42;
    cl_int error = CL_SUCCESS;

    std::vector<size_t> vecSizes({ 1, 2, 3, 4, 8, 16 });
    std::vector<ArgType> testData;
    testData.reserve(16 * vecSizes.size());

    for (auto v : vecSizes)
    {
        size_t i;
        for (i = 0; i < v; i++)
        {
            testData.push_back(value);
        }
        for (; i < 16; i++)
        {
            testData.push_back(0);
        }
    }

    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       testData.size() * sizeof(ArgType), nullptr, &error);
    test_error(error, "Unable to create destination buffer");

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, TestInfo<T>::testName);
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel =
        clCreateKernel(prog, TestInfo<T>::testName, &error);
    test_error(error, "Unable to create SPIR-V kernel");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(value), &value);
    test_error(error, "Unable to set kernel arguments");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    test_error(error, "Unable to enqueue kernel");

    std::vector<ArgType> resData(testData.size());
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                resData.size() * sizeof(ArgType),
                                resData.data(), 0, NULL, NULL);
    test_error(error, "Unable to read destination buffer");

    if (resData != testData)
    {
        log_error("Values do not match!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

TEST_SPIRV_FUNC(op_expect)
{
    if (!is_extension_available(deviceID, "cl_khr_expect_assume"))
    {
        log_info("cl_khr_expect_assume is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int result = TEST_PASS;

    result |= test_expect_type<cl_char>(deviceID, context, queue);
    result |= test_expect_type<cl_short>(deviceID, context, queue);
    result |= test_expect_type<cl_int>(deviceID, context, queue);
    if (gHasLong)
    {
        result |= test_expect_type<cl_long>(deviceID, context, queue);
    }
    result |= test_expect_type<cl_bool>(deviceID, context, queue);

    return result;
}

TEST_SPIRV_FUNC(op_assume)
{
    if (!is_extension_available(deviceID, "cl_khr_expect_assume"))
    {
        log_info("cl_khr_expect_assume is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clMemWrapper dst =
        clCreateBuffer(context, 0, num_elements * sizeof(cl_int), NULL, &error);
    test_error(error, "Unable to create destination buffer");

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, "assume");
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel = clCreateKernel(prog, "test_assume", &error);
    test_error(error, "Unable to create SPIR-V kernel");

    const cl_int value = 42;
    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(value), &value);
    test_error(error, "Unable to set kernel arguments");

    size_t global = num_elements;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    test_error(error, "Unable to enqueue kernel");

    std::vector<cl_int> h_dst(num_elements);
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                h_dst.size() * sizeof(cl_int), h_dst.data(), 0,
                                NULL, NULL);
    test_error(error, "Unable to read destination buffer");

    for (int i = 0; i < num_elements; i++)
    {
        if (h_dst[i] != value)
        {
            log_error("Values do not match at location %d\n", i);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}
