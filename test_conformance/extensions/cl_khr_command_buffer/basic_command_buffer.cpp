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
#include "command_buffer_test_base.h"
#include "procs.h"
#include "harness/typeWrappers.h"

#include <algorithm>
#include <cstring>
#include <vector>

#define CHECK_VERIFICATION_ERROR(reference, result, index)                     \
    {                                                                          \
        if (reference != result)                                               \
        {                                                                      \
            log_error("Expected %d was %d at index %u\n", reference, result,   \
                      index);                                                  \
            return TEST_FAIL;                                                  \
        }                                                                      \
    }

namespace {

// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct BasicCommandBufferTest : CommandBufferTestBase
{

    BasicCommandBufferTest(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : CommandBufferTestBase(device), context(context), queue(queue),
          command_buffer(nullptr), simultaneous_use(false), num_elements(0)
    {}

    cl_int TearDown()
    {
        if (nullptr != command_buffer)
        {
            cl_int error = clReleaseCommandBufferKHR(command_buffer);
            test_error(error, "clReleaseCommandBufferKHR failed");
        }
        return CL_SUCCESS;
    }

    bool Skip()
    {
        cl_command_queue_properties queue_properties;
        cl_int error = clGetDeviceInfo(
            device, CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR,
            sizeof(queue_properties), &queue_properties, NULL);
        test_error(error,
                   "Unable to query "
                   "CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR");

        // Skip if queue properties required, tests don't account for this
        return (error || queue_properties);
    }

    cl_int SetUp(int elements)
    {
        cl_int error = init_extension_functions();
        if (error != CL_SUCCESS)
        {
            return error;
        }

        // Query if device supports simultaneous use
        cl_device_command_buffer_capabilities_khr capabilities;
        error =
            clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, NULL);
        test_error(error, "Unable to query CL_COMMAND_BUFFER_PROPERTIES_KHR");
        simultaneous_use =
            capabilities & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR;

        if (elements <= 0)
        {
            return CL_INVALID_VALUE;
        }
        num_elements = static_cast<size_t>(elements);


        // Kernel performs a parallel copy from an input buffer to output buffer
        // is created.
        const char *kernel_str =
            R"(
        __kernel void copy(__global int* in, __global int* out) {
            size_t id = get_global_id(0);
            out[id] = in[id];
        })";
        const size_t lengths[1] = { std::strlen(kernel_str) };

        program =
            clCreateProgramWithSource(context, 1, &kernel_str, lengths, &error);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        kernel = clCreateKernel(program, "copy", &error);
        test_error(error, "Failed to create copy kernel");

        error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        if (simultaneous_use)
        {
            cl_command_buffer_properties_khr properties[3] = {
                CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
                CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR, 0
            };
            command_buffer =
                clCreateCommandBufferKHR(1, &queue, properties, &error);
        }
        else
        {
            command_buffer =
                clCreateCommandBufferKHR(1, &queue, nullptr, &error);
        }
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    /*
     * Subtests returning an OpenCL error code
     */

    // Test enqueuing a command-buffer containing a single NDRange command
    // once
    cl_int BasicEnqueue();

    // Test enqueuing a command-buffer containing multiple command, including
    // operations other than NDRange kernel execution.
    cl_int MixedCommandsTest();

    // Test enqueueing a command-buffer blocked on a user-event
    cl_int UserEventTest();

    // Test flushing the command-queue between command-buffer enqueues
    cl_int ExplicitFlushTest();

    // Test enqueueing a command-buffer twice separated by another enqueue
    // operation
    cl_int InterleavedEnqueueTest();

    // Device supports the simultaneous-use capability
    bool simultaneous_use;

protected:
    size_t data_size() const { return num_elements * sizeof(cl_int); }

    cl_context context;
    cl_command_queue queue;
    cl_command_buffer_khr command_buffer;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper in_mem, out_mem;
    size_t num_elements;
};

cl_int BasicCommandBufferTest::BasicEnqueue()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    const cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
    }

    return CL_SUCCESS;
}

cl_int BasicCommandBufferTest::MixedCommandsTest()
{
    cl_int error;
    const size_t iterations = 4;
    clMemWrapper result_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * iterations,
                       nullptr, &error);
    test_error(error, "clCreateBuffer failed");

    const cl_int pattern_base = 42;
    for (size_t i = 0; i < iterations; i++)
    {
        const cl_int pattern = pattern_base + i;
        cl_int error = clCommandFillBufferKHR(
            command_buffer, queue, in_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        const size_t result_offset = i * sizeof(cl_int);
        error = clCommandCopyBufferKHR(
            command_buffer, queue, out_mem, result_mem, 0, result_offset,
            sizeof(cl_int), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandCopyBufferKHR failed");
    }

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> result_data(num_elements);
    error = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0,
                                iterations * sizeof(cl_int), result_data.data(),
                                0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");


    for (size_t i = 0; i < iterations; i++)
    {
        const cl_int ref = pattern_base + i;
        CHECK_VERIFICATION_ERROR(ref, result_data[i], i);
    }

    return CL_SUCCESS;
}

cl_int BasicCommandBufferTest::UserEventTest()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    clEventWrapper user_event = clCreateUserEvent(context, &error);
    test_error(error, "clCreateUserEvent failed");

    const cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                      &user_event, nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clSetUserEventStatus(user_event, CL_COMPLETE);
    test_error(error, "clSetUserEventStatus failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
    }

    return CL_SUCCESS;
}

cl_int BasicCommandBufferTest::ExplicitFlushTest()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    const cl_int pattern_A = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern_A, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data_A(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data_A.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    const cl_int pattern_B = 0xA;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern_B, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    std::vector<cl_int> output_data_B(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data_B.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern_A, output_data_A[i], i);

        CHECK_VERIFICATION_ERROR(pattern_B, output_data_B[i], i);
    }
    return CL_SUCCESS;
}


cl_int BasicCommandBufferTest::InterleavedEnqueueTest()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    pattern = 0xABCD;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    error = clEnqueueCopyBuffer(queue, in_mem, out_mem, 0, 0, data_size(), 0,
                                nullptr, nullptr);
    test_error(error, "clEnqueueCopyBuffer failed");

    std::vector<cl_int> output_data(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
    }

    return CL_SUCCESS;
}
#undef CHECK_VERIFICATION_ERROR
} // anonymous namespace

int test_single_ndrange(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = BasicCommandBufferTest(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.BasicEnqueue();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    error = test_fixture.TearDown();
    test_error_ret(error, "Error in test destruction", TEST_FAIL);

    return TEST_PASS;
}

int test_interleaved_enqueue(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = BasicCommandBufferTest(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (!test_fixture.simultaneous_use || test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.InterleavedEnqueueTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    error = test_fixture.TearDown();
    test_error_ret(error, "Error in test destruction", TEST_FAIL);

    return TEST_PASS;
}

int test_mixed_commands(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = BasicCommandBufferTest(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.MixedCommandsTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    error = test_fixture.TearDown();
    test_error_ret(error, "Error in test destruction", TEST_FAIL);

    return TEST_PASS;
}

int test_explicit_flush(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = BasicCommandBufferTest(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (!test_fixture.simultaneous_use || test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.ExplicitFlushTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    error = test_fixture.TearDown();
    test_error_ret(error, "Error in test destruction", TEST_FAIL);

    return TEST_PASS;
}

int test_user_events(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = BasicCommandBufferTest(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.UserEventTest();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    error = test_fixture.TearDown();
    test_error_ret(error, "Error in test destruction", TEST_FAIL);

    return TEST_PASS;
}
