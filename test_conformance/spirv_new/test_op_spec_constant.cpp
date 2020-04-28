/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
in whole or in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include "testBase.h"
#include "types.hpp"



template<typename T>
int test_spec_constant(cl_device_id deviceID, cl_context context,
    cl_command_queue queue, const char *name,
    std::vector<T> &inputs,
    std::vector<T> &references,
    T spec_constant_value,
    bool(*notEqual)(const T&, const T&) = isNotEqual<T>)
{
    if (name == "double") {
        if (!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return TEST_SKIP;
        }
    }
    if (name == "half") {
        if (!is_extension_available(deviceID, "cl_khr_fp16")) {
            log_info("Extension cl_khr_fp16 not supported; skipping half tests.\n");
            return TEST_SKIP;
        }
    }

    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, "spec_const_kernel", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");
    int num = (int)references.size();
    size_t bytes = num * sizeof(T);
    clMemWrapper numbers_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, bytes, inputs.data(), &err);
    SPIRV_CHECK_ERROR(err, "Failed to create numbers_buffer");

    clMemWrapper spec_buffer = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(spec_constant_value), &spec_constant_value, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spec_buffer");

    err = clSetKernelArg(kernel, 0, sizeof(clMemWrapper), &numbers_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument inputs_buffer");
    err = clSetKernelArg(kernel, 1, sizeof(clMemWrapper), &spec_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument spec_constant_value");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");
    clFinish(queue);

    err = clSetProgramSpecializationConstant(prog, 101, sizeof(T), &spec_constant_value);
    SPIRV_CHECK_ERROR(err, "Failed to run clSetProgramSpecializationConstant");   

    std::vector<T> device_results(num, 0);
    err = clEnqueueReadBuffer(queue, numbers_buffer, CL_TRUE, 0, bytes, device_results.data(), 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from numbers_buffer");

    for (int i = 0; i < num; i++) {
        if (device_results[i] == references[i]) {
            log_error("Values match but should not at location %d expected %d obtained %d\n", i, references[i], device_results[i]);
            return TEST_FAIL;
        }
    }

    err = clBuildProgram(prog, 1, &deviceID, NULL, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    kernel = clCreateKernel(prog, "spec_const_kernel", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    err = clSetKernelArg(kernel, 0, sizeof(clMemWrapper), &numbers_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument inputs_buffer");
    err = clSetKernelArg(kernel, 1, sizeof(clMemWrapper), &spec_buffer);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument spec_constant_value");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");
    clFinish(queue);

    std::fill(device_results.begin(), device_results.end(), 0);
    err = clEnqueueReadBuffer(queue, numbers_buffer, CL_TRUE, 0, bytes, device_results.data(), 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from numbers_buffer");

    for (int i = 0; i < num; i++) {
        if (device_results[i] != references[i]) {
            log_error("Values do not match in index %d expected %d obtained %d\n", i, references[i], device_results[i]);
            return TEST_FAIL;
        }
    }
    return TEST_PASS;
}

#define TEST_SPEC_CONSTANT(NAME, type, value, spec_constant_value)                   \
    TEST_SPIRV_FUNC_VERSION(op_spec_constant_##NAME##_simple, Version(2,1))          \
    {                                                                                \
        std::vector<type> inputs(1024, (type)value);                                 \
        std::vector<type> references(1024, (type)value + (type)spec_constant_value); \
        return test_spec_constant(deviceID, context, queue,                          \
                             "op_spec_constant_" #NAME "_simple",                    \
                             inputs,                                                 \
                             references,                                             \
                             (type)spec_constant_value);                             \
    }                                                                                \

//type name, type, value init, spec constant value
TEST_SPEC_CONSTANT(int, cl_int, 20, -11)
TEST_SPEC_CONSTANT(uint, cl_uint, 25, 43)
TEST_SPEC_CONSTANT(char, cl_char, -20, -3)
TEST_SPEC_CONSTANT(uchar, cl_uchar, 19, 4)
TEST_SPEC_CONSTANT(short, cl_short, -6000, -1000)
TEST_SPEC_CONSTANT(ushort, cl_ushort, 6000, 3000)
TEST_SPEC_CONSTANT(long, cl_long, 34359738360L, 2)
TEST_SPEC_CONSTANT(ulong, cl_ulong, 9223372036854775000UL, 200)
TEST_SPEC_CONSTANT(float, cl_float, 1.5, -3.7)
TEST_SPEC_CONSTANT(half, cl_half, 1, 2)
TEST_SPEC_CONSTANT(double, cl_double, 14534.53453, 1.53453)

// Boolean tests
// documenation: 'If a specialization constant is a boolean
// constant, spec_value should be a pointer to a cl_uchar value'

TEST_SPIRV_FUNC(op_spec_constant_true_simple)
{
    //1-st ndrange use default spec const (true) value = value + 1 (first check verifies that values are different)
    //2-nd ndrange sets spec const false so value = value - 1
    cl_uchar value = (cl_uchar)7;
    std::vector<cl_uchar> inputs(1024, value);
    std::vector<cl_uchar> references(1024, value);
    return test_spec_constant<cl_uchar>(deviceID, context, queue,
                                "op_spec_constant_true_simple",
                                inputs,
                                references,
                                0);
}

TEST_SPIRV_FUNC(op_spec_constant_false_simple)
{
    //1-st ndrange use default spec const (false) value = value - 1 (first check verifies that values are different)
    //2-nd ndrange sets spec const true so value = value + 1
    cl_uchar value = (cl_uchar)8;
    std::vector<cl_uchar> inputs(1024, value);
    std::vector<cl_uchar> references(1024, value);
    return test_spec_constant<cl_uchar>(deviceID, context, queue,
        "op_spec_constant_false_simple",
        inputs,
        references,
        1);
}