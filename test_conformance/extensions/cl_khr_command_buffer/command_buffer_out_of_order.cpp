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
#include <fstream>
#include <stdio.h>

namespace {

////////////////////////////////////////////////////////////////////////////////

template <bool simul_use> struct OutOfOrderTest : public BasicCommandBufferTest
{
    OutOfOrderTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          trigger_event(nullptr), out_of_order_command_buffer(this),
          out_of_order_queue(nullptr), event(nullptr)
    {
        simultaneous_use_requested = simul_use;
        if (simul_use) buffer_size_multiplier = 2;
    }

    //--------------------------------------------------------------------------
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

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (!out_of_order_support
            || (simultaneous_use_requested && !simultaneous_use_support))
            return true;

        return BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        if (simultaneous_use_requested)
        {
            // enqueue simultaneous command-buffers with clSetKernelArg calls
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with  clSetKernelArg calls
            error = RunSingle();
            test_error(error, "RunSingle failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        const cl_int pattern = pattern_pri;
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

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle()
    {
        cl_int error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 1, &event,
                                    nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    struct SimulPassData
    {
        cl_int pattern;
        cl_int offset;
        std::vector<cl_int> output_buffer;
    };

    //--------------------------------------------------------------------------
    cl_int RecordSimultaneousCommandBuffer() const
    {
        cl_int error = CL_SUCCESS;

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = clEnqueueFillBuffer(
            queue, out_mem, &pd.pattern, sizeof(cl_int),
            pd.offset * sizeof(cl_int), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillBuffer(queue, off_mem, &pd.offset, sizeof(cl_int),
                                    0, sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        if (!trigger_event)
        {
            trigger_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &trigger_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            queue, out_mem, CL_FALSE, pd.offset * sizeof(cl_int), data_size(),
            pd.output_buffer.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer with primary queue
        error = RecordSimultaneousCommandBuffer();
        test_error(error, "RecordSimultaneousCommandBuffer failed");

        std::vector<SimulPassData> simul_passes = {
            { 0, 0, std::vector<cl_int>(num_elements) }
        };

        error = EnqueueSimultaneousPass(simul_passes.front());
        test_error(error, "EnqueueSimultaneousPass 1 failed");

        if (simultaneous_use_support)
        {
            cl_int offset = static_cast<cl_int>(num_elements);
            simul_passes.push_back(
                { 1, offset, std::vector<cl_int>(num_elements) });

            error = EnqueueSimultaneousPass(simul_passes.back());
            test_error(error, "EnqueueSimultaneousPass 2 failed");
        }

        error = clSetUserEventStatus(trigger_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result buffer
        for (auto&& pass : simul_passes)
        {
            auto& res_data = pass.output_buffer;
            for (size_t i = 0; i < num_elements; i++)
            {
                CHECK_VERIFICATION_ERROR(pattern_pri, res_data[i], i);
            }
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    clEventWrapper trigger_event = nullptr;

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;
    clEventWrapper event;

    const cl_int pattern_pri = 2;
    const cl_int pattern_sec = 3;
};

} // anonymous namespace

int test_out_of_order(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<OutOfOrderTest<false>>(device, context, queue,
                                                 num_elements);
}

int test_simultaneous_out_of_order(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<OutOfOrderTest<true>>(device, context, queue,
                                                num_elements);
}
