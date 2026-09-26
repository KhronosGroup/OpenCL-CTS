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

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Tests for cl_khr_command_buffer which handles submitting a command-buffer to
// an out-of-order queue.

struct OutOfOrderTest : public BasicCommandBufferTest
{
    OutOfOrderTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          out_of_order_queue(nullptr), out_of_order_command_buffer(this),
          user_event(nullptr), wait_pass_event(nullptr), kernel_fill(nullptr),
          program_fill(nullptr)
    {
        buffer_size_multiplier = 2; // two enqueues of command-buffer
    }

    cl_int SetUpKernel() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // create additional kernel to properly prepare output buffer for test
        const char* kernel_str =
            R"(
          __kernel void fill(int pattern, __global int* out, __global int* offset)
          {
              size_t id = get_global_id(0);
              size_t ind = offset[0] + id;
              out[ind] = pattern;
          })";

        error = create_single_kernel_helper_create_program(
            context, &program_fill, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error =
            clBuildProgram(program_fill, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel_fill = clCreateKernel(program_fill, "fill", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernelArgs failed");

        error = clSetKernelArg(kernel_fill, 0, sizeof(cl_int),
                               &overwritten_pattern);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 2, sizeof(off_mem), &off_mem);
        test_error(error, "clSetKernelArg failed");

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
        if (BasicCommandBufferTest::Skip()) return true;
        return !out_of_order_support;
    }

    cl_int RecordCommandBuffer() const
    {
        cl_sync_point_khr sync_points[2];
        // fill entire in_mem buffer
        cl_int error = clCommandFillBufferKHR(
            out_of_order_command_buffer, nullptr, nullptr, in_mem, &pattern_pri,
            sizeof(cl_int), 0, data_size() * buffer_size_multiplier, 0, nullptr,
            &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        // to avoid overwriting the entire result buffer instead of filling only
        // relevant part this additional kernel was introduced
        error = clCommandNDRangeKernelKHR(out_of_order_command_buffer, nullptr,
                                          nullptr, kernel_fill, 1, nullptr,
                                          &num_elements, nullptr, 0, nullptr,
                                          &sync_points[1], nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clCommandNDRangeKernelKHR(
            out_of_order_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(out_of_order_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    struct EnqueuePassData
    {
        cl_int offset;
        std::vector<cl_int> output_buffer;
        // 0: offset-buffer fill event, 2:kernel done event
        clEventWrapper wait_events[2];
    };

    cl_int EnqueuePass(EnqueuePassData& pd)
    {
        // filling offset buffer must wait for previous pass completeness
        cl_int error = clEnqueueFillBuffer(
            out_of_order_queue, off_mem, &pd.offset, sizeof(cl_int), 0,
            sizeof(cl_int), (wait_pass_event != nullptr ? 1 : 0),
            (wait_pass_event != nullptr ? &wait_pass_event : nullptr),
            &pd.wait_events[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        // command buffer execution must wait for two wait-events
        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 1, &pd.wait_events[0],
            &pd.wait_events[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_FALSE,
                                    pd.offset * sizeof(cl_int), data_size(),
                                    pd.output_buffer.data(), 1,
                                    &pd.wait_events[1], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        cl_int offset = static_cast<cl_int>(num_elements);
        std::vector<EnqueuePassData> enqueue_passes = {
            { 0, std::vector<cl_int>(num_elements) },
            { offset, std::vector<cl_int>(num_elements) }
        };

        for (auto&& pass : enqueue_passes)
        {
            error = EnqueuePass(pass);
            test_error(error, "EnqueuePass failed");

            wait_pass_event = pass.wait_events[1];
        }

        error = clFinish(out_of_order_queue);
        test_error(error, "clFinish failed");

        // verify the result buffers
        for (auto&& pass : enqueue_passes)
        {
            auto& res_data = pass.output_buffer;
            for (size_t i = 0; i < num_elements; i++)
            {
                CHECK_VERIFICATION_ERROR(pattern_pri, res_data[i], i);
            }
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;

    clEventWrapper user_event;
    clEventWrapper wait_pass_event;

    clKernelWrapper kernel_fill;
    clProgramWrapper program_fill;

    const cl_int overwritten_pattern = 0xACDC;
    const cl_int pattern_pri = 42;
};

} // anonymous namespace

REGISTER_TEST(out_of_order)
{
    return MakeAndRunTest<OutOfOrderTest>(device, context, queue, num_elements);
}
