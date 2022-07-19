//
// Copyright (c) 2022 The Khronos Group Inc.
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
          command_buffer(this), simultaneous_use(false),
          out_of_order_support(false), num_elements(0)
    {}

    virtual bool Skip()
    {
        cl_command_queue_properties required_properties;
        cl_int error = clGetDeviceInfo(
            device, CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR,
            sizeof(required_properties), &required_properties, NULL);
        test_error(error,
                   "Unable to query "
                   "CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR");

        cl_command_queue_properties queue_properties;

        error = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                      sizeof(queue_properties),
                                      &queue_properties, NULL);
        test_error(error, "Unable to query CL_QUEUE_PROPERTIES");

        // Skip if queue properties don't contain those required
        return required_properties != (required_properties & queue_properties);
    }

    virtual cl_int SetUp(int elements)
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
        test_error(error,
                   "Unable to query CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR");
        simultaneous_use =
            capabilities & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR;
        out_of_order_support =
            capabilities & CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR;

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

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
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
                CL_COMMAND_BUFFER_FLAGS_KHR,
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

    // Test body returning an OpenCL error code
    virtual cl_int Run() = 0;


protected:
    size_t data_size() const { return num_elements * sizeof(cl_int); }

    cl_context context;
    cl_command_queue queue;
    clCommandBufferWrapper command_buffer;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper in_mem, out_mem;
    size_t num_elements;

    // Device support query results
    bool simultaneous_use;
    bool out_of_order_support;
};

// Test enqueuing a command-buffer containing a single NDRange command once
struct BasicEnqueueTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        const cl_int pattern = 42;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
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
};

// Test enqueuing a command-buffer containing multiple command, including
// operations other than NDRange kernel execution.
struct MixedCommandsTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error;
        const size_t iterations = 4;
        clMemWrapper result_mem =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(cl_int) * iterations, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        const cl_int pattern_base = 42;
        for (size_t i = 0; i < iterations; i++)
        {
            const cl_int pattern = pattern_base + i;
            cl_int error = clCommandFillBufferKHR(
                command_buffer, nullptr, in_mem, &pattern, sizeof(cl_int), 0,
                data_size(), 0, nullptr, nullptr, nullptr);
            test_error(error, "clCommandFillBufferKHR failed");

            error = clCommandNDRangeKernelKHR(
                command_buffer, nullptr, nullptr, kernel, 1, nullptr,
                &num_elements, nullptr, 0, nullptr, nullptr, nullptr);
            test_error(error, "clCommandNDRangeKernelKHR failed");

            const size_t result_offset = i * sizeof(cl_int);
            error = clCommandCopyBufferKHR(
                command_buffer, nullptr, out_mem, result_mem, 0, result_offset,
                sizeof(cl_int), 0, nullptr, nullptr, nullptr);
            test_error(error, "clCommandCopyBufferKHR failed");
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> result_data(num_elements);
        error = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0,
                                    iterations * sizeof(cl_int),
                                    result_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < iterations; i++)
        {
            const cl_int ref = pattern_base + i;
            CHECK_VERIFICATION_ERROR(ref, result_data[i], i);
        }

        return CL_SUCCESS;
    }
};

// Test enqueueing a command-buffer blocked on a user-event
struct UserEventTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
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
};

// Test flushing the command-queue between command-buffer enqueues
struct ExplicitFlushTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        const cl_int pattern_A = 42;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern_A, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clFlush(queue);
        test_error(error, "clFlush failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data_A(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data_A.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        const cl_int pattern_B = 0xA;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern_B, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clFlush(queue);
        test_error(error, "clFlush failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
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

    bool Skip() override
    {
        return !simultaneous_use || BasicCommandBufferTest::Skip();
    }
};

// Test enqueueing a command-buffer twice separated by another enqueue operation
struct InterleavedEnqueueTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        cl_int pattern = 42;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        pattern = 0xABCD;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueCopyBuffer(queue, in_mem, out_mem, 0, 0, data_size(),
                                    0, nullptr, nullptr);
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

    bool Skip() override
    {
        return !simultaneous_use || BasicCommandBufferTest::Skip();
    }
};

// Test sync-points with an out-of-order command-buffer
struct OutOfOrderTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;
    OutOfOrderTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          out_of_order_command_buffer(this), out_of_order_queue(nullptr),
          event(nullptr)
    {}

    cl_int Run() override
    {
        cl_sync_point_khr sync_points[2];

        const cl_int pattern = 42;
        cl_int error =
            clCommandFillBufferKHR(out_of_order_command_buffer, nullptr, in_mem,
                                   &pattern, sizeof(cl_int), 0, data_size(), 0,
                                   nullptr, &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        const cl_int overwritten_pattern = 0xACDC;
        error = clCommandFillBufferKHR(out_of_order_command_buffer, nullptr,
                                       out_mem, &overwritten_pattern,
                                       sizeof(cl_int), 0, data_size(), 0,
                                       nullptr, &sync_points[1], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            out_of_order_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(out_of_order_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 1, &event,
                                    nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        if (!out_of_order_support)
        {
            // Test will skip as device doesn't support out-of-order
            // command-buffers
            return CL_SUCCESS;
        }

        out_of_order_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error, "Unable to create command queue to test with");

        out_of_order_command_buffer =
            clCreateCommandBufferKHR(1, &out_of_order_queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return !out_of_order_support || BasicCommandBufferTest::Skip();
    }

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;
    clEventWrapper event;
};

#undef CHECK_VERIFICATION_ERROR

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device);

    auto test_fixture = T(device, context, queue);
    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    if (test_fixture.Skip())
    {
        return TEST_SKIPPED_ITSELF;
    }

    error = test_fixture.Run();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}
} // anonymous namespace

int test_single_ndrange(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<BasicEnqueueTest>(device, context, queue,
                                            num_elements);
}

int test_interleaved_enqueue(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<InterleavedEnqueueTest>(device, context, queue,
                                                  num_elements);
}

int test_mixed_commands(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MixedCommandsTest>(device, context, queue,
                                             num_elements);
}

int test_explicit_flush(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ExplicitFlushTest>(device, context, queue,
                                             num_elements);
}

int test_user_events(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<UserEventTest>(device, context, queue, num_elements);
}

int test_out_of_order(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<OutOfOrderTest>(device, context, queue, num_elements);
}
