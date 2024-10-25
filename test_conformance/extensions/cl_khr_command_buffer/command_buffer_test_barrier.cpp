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
#include "basic_command_buffer.h"
#include "procs.h"

#include <vector>


namespace {

////////////////////////////////////////////////////////////////////////////////
// Command-bufer barrier tests which handles below cases:
//
// - barrier wait list

struct BarrierWithWaitListKHR : public BasicCommandBufferTest
{

    using BasicCommandBufferTest::BasicCommandBufferTest;

    BarrierWithWaitListKHR(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          out_of_order_queue(nullptr), out_of_order_command_buffer(this),
          event(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            out_of_order_command_buffer, nullptr, nullptr, in_mem, &pattern,
            sizeof(cl_int), 0, data_size(), 0, nullptr, &sync_points[0],
            nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        const cl_int overwritten_pattern = 0xACDC;
        error = clCommandFillBufferKHR(out_of_order_command_buffer, nullptr,
                                       nullptr, out_mem, &overwritten_pattern,
                                       sizeof(cl_int), 0, data_size(), 0,
                                       nullptr, &sync_points[1], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandBarrierWithWaitListKHR(out_of_order_command_buffer,
                                                nullptr, nullptr, 2,
                                                sync_points, nullptr, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");

        error = clCommandNDRangeKernelKHR(
            out_of_order_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(out_of_order_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data_1(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data_1.data(), 1,
                                    &event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */

        error =
            clEnqueueFillBuffer(queue, in_mem, &zero_pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBufferKHR failed");

        error =
            clEnqueueFillBuffer(queue, out_mem, &zero_pattern, sizeof(cl_int),
                                0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data_2(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data_2.data(), 1,
                                    &event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data_2[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

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
        return BasicCommandBufferTest::Skip() || !out_of_order_support;
    }

    const cl_int pattern = 0x16;
    const cl_int zero_pattern = 0x0;
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;
    clEventWrapper event;
    cl_sync_point_khr sync_points[2];
    clEventWrapper user_event;
};
};


int test_barrier_wait_list(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<BarrierWithWaitListKHR>(device, context, queue,
                                                  num_elements);
}
