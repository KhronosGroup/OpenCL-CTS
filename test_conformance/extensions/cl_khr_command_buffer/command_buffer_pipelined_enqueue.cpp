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

#include "basic_command_buffer.h"

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Tests for multiple sequential submissions of a command-buffer without a
// blocking wait between them, but using the following mechanisms to serialize
// execution of the submissions.
// * In-order queue dependencies
// * Event dependencies in command-buffer submissions to an out-of-order queue
// * Barrier submissions between command-buffer submissions to an out-of-order
//   queue

// Base class that individual test fixtures are derived from
struct CommandBufferPipelined : public BasicCommandBufferTest
{
    CommandBufferPipelined(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    cl_int SetUpKernel() override
    {
        const char* mul_kernel_str =
            R"(
            __kernel void mul_by_val(int in, __global int* data)
            {
                size_t id = get_global_id(0);
                data[id] *= in;
            }

            __kernel void increment(__global int* data)
            {
                size_t id = get_global_id(0);
                data[id]++;
            })";

        cl_int error = create_single_kernel_helper_create_program(
            context, &program, 1, &mul_kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        mul_kernel = clCreateKernel(program, "mul_by_val", &error);
        test_error(error, "Failed to create mul_by_val kernel");

        inc_kernel = clCreateKernel(program, "increment", &error);
        test_error(error, "Failed to create increment kernel");

        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 num_elements * buffer_size_multiplier
                                     * sizeof(cl_int),
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int val_arg = pattern;
        error = clSetKernelArg(mul_kernel, 0, sizeof(cl_int), &val_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(mul_kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(inc_kernel, 0, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    cl_int RecordCommandBuffer(clCommandBufferWrapper& cmd_buf)
    {
        cl_int error = clCommandNDRangeKernelKHR(
            cmd_buf, nullptr, nullptr, inc_kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(cmd_buf);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Zero initialize buffer before starting test
        cl_int zero_pattern = 0;
        error =
            clEnqueueFillBuffer(queue, out_mem, &zero_pattern, sizeof(cl_int),
                                0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        return CL_SUCCESS;
    }

    const cl_int pattern = 42;

    clKernelWrapper inc_kernel = nullptr;
    clKernelWrapper mul_kernel = nullptr;
};

struct InOrderPipelined : public CommandBufferPipelined
{
    InOrderPipelined(cl_device_id device, cl_context context,
                     cl_command_queue queue)
        : CommandBufferPipelined(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error =
            clEnqueueNDRangeKernel(queue, mul_kernel, 1, nullptr, &num_elements,
                                   nullptr, 0, nullptr, nullptr);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // Verify
        const cl_int ref = pattern + 1;
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(ref, output_data[i], i);
        }
        return CL_SUCCESS;
    }
};

struct EventPipelined : public CommandBufferPipelined
{
    EventPipelined(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : CommandBufferPipelined(device, context, queue),
          out_of_order_queue(nullptr), out_of_order_command_buffer(this)
    {}

    bool Skip() override
    {
        return CommandBufferPipelined::Skip() || !out_of_order_support;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CommandBufferPipelined::SetUp(elements);
        test_error(error, "EventPipelined::SetUp failed");

        out_of_order_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error, "Unable to create command queue to test with");

        out_of_order_command_buffer =
            clCreateCommandBufferKHR(1, &out_of_order_queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = RecordCommandBuffer(out_of_order_command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &events[0]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueNDRangeKernel(out_of_order_queue, mul_kernel, 1,
                                       nullptr, &num_elements, nullptr, 1,
                                       &events[0], &events[1]);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 1, &events[1], &events[2]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 1,
                                    &events[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // Verify
        const cl_int ref = pattern + 1;
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(ref, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;
    clEventWrapper events[3] = { nullptr, nullptr, nullptr };
};

struct BarrierPipelined : public CommandBufferPipelined
{
    BarrierPipelined(cl_device_id device, cl_context context,
                     cl_command_queue queue)
        : CommandBufferPipelined(device, context, queue),
          out_of_order_queue(nullptr), out_of_order_command_buffer(this)
    {}

    bool Skip() override
    {
        return CommandBufferPipelined::Skip() || !out_of_order_support;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CommandBufferPipelined::SetUp(elements);
        test_error(error, "EventPipelined::SetUp failed");

        out_of_order_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error, "Unable to create command queue to test with");

        out_of_order_command_buffer =
            clCreateCommandBufferKHR(1, &out_of_order_queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = RecordCommandBuffer(out_of_order_command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueBarrier(out_of_order_queue);
        test_error(error, "clEnqueueBarrier failed");

        error =
            clEnqueueNDRangeKernel(out_of_order_queue, mul_kernel, 1, nullptr,
                                   &num_elements, nullptr, 0, nullptr, nullptr);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clEnqueueBarrier(out_of_order_queue);
        test_error(error, "clEnqueueBarrier failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // Verify
        const cl_int ref = pattern + 1;
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(ref, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;
};
} // anonymous namespace

REGISTER_TEST(pipeline_in_order_deps)
{
    return MakeAndRunTest<InOrderPipelined>(device, context, queue,
                                            num_elements);
}

REGISTER_TEST(pipeline_event_deps)
{
    return MakeAndRunTest<EventPipelined>(device, context, queue, num_elements);
}

REGISTER_TEST(pipeline_barrier_deps)
{
    return MakeAndRunTest<BarrierPipelined>(device, context, queue,
                                            num_elements);
}
