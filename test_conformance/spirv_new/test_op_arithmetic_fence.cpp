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

template <typename T> struct TestInfo
{
};
template <> struct TestInfo<cl_half>
{
    // Note: for the half test, we will still pass floats as the kernel
    // arguments, and we will convert from float <-> half in the kernel.
    using argType = cl_float;
    static constexpr const char* typeName = "half";
    static constexpr const char* filename = "ext_arithmetic_fence_half";
    static constexpr const char* testName = "test_ext_arithmetic_fence_half";
    static constexpr argType big = 65000.0f;
    static constexpr argType small = 1e-3f;
};
template <> struct TestInfo<cl_float>
{
    using argType = cl_float;
    static constexpr const char* typeName = "float";
    static constexpr const char* filename = "ext_arithmetic_fence_float";
    static constexpr const char* testName = "test_ext_arithmetic_fence_float";
    static constexpr argType big = 1e20f;
    static constexpr argType small = 1e-3f;
};
template <> struct TestInfo<cl_double>
{
    using argType = cl_double;
    static constexpr const char* typeName = "double";
    static constexpr const char* filename = "ext_arithmetic_fence_double";
    static constexpr const char* testName = "test_ext_arithmetic_fence_double";
    static constexpr argType big = 1e200;
    static constexpr argType small = 1e-30;
};


template <typename T>
int arithmetic_fence_helper(cl_context context, cl_device_id device,
                            cl_command_queue queue)
{
    using ArgType = typename TestInfo<T>::argType;

    log_info("    testing type %s\n", TestInfo<T>::typeName);

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, TestInfo<T>::filename);
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel =
        clCreateKernel(prog, TestInfo<T>::testName, &error);
    test_error(error, "Unable to create SPIR-V kernel");

    clMemWrapper dst =
        clCreateBuffer(context, 0, sizeof(ArgType), NULL, &error);
    test_error(error, "Unable to create destination buffer");

    const auto big = TestInfo<T>::big;
    const auto small = TestInfo<T>::small;

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(big), &big);
    error |= clSetKernelArg(kernel, 2, sizeof(small), &small);
    test_error(error, "Unable to set kernel arguments");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    test_error(error, "Unable to enqueue kernel");

    ArgType value = TestInfo<T>::big;
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(value), &value,
                                0, NULL, NULL);
    test_error(error, "Unable to read destination buffer");

    // The read back value should be zero.  If it is not, then the arithmetic
    // fence is probably ignored.
    log_info("      read value: %.10e\n", (double)value);
    test_assert_error(value == 0, "read value is incorrect");

    return TEST_PASS;
}

REGISTER_TEST(op_arithmetic_fence)
{
    // REQUIRE_SPIRV_EXTENSION("SPV_EXT_arithmetic_fence");

    int result = TEST_PASS;

    // if (is_extension_available(device, "cl_khr_fp16"))
    // {
    //     result |= arithmetic_fence_helper<cl_half>(context, device, queue);
    // }
    result |= arithmetic_fence_helper<cl_float>(context, device, queue);
    // if (is_extension_available(device, "cl_khr_fp64"))
    // {
    //     result |= arithmetic_fence_helper<cl_double>(context, device, queue);
    // }

    return result;
}
