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
// Command-queue substitution tests which handles below cases:
// -substitution on queue without properties
// -substitution on queue with properties
// -simultaneous use queue substitution

template <bool prop_use, bool simul_use>
struct SubstituteQueueTest : public BasicCommandBufferTest
{
    SubstituteQueueTest(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          properties_use_requested(prop_use), user_event(nullptr)
    {
        simultaneous_use_requested = simul_use;
        if (simul_use) buffer_size_multiplier = 2;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (properties_use_requested)
        {
            Version version = get_device_cl_version(device);
            const cl_device_info host_queue_query = version >= Version(2, 0)
                ? CL_DEVICE_QUEUE_ON_HOST_PROPERTIES
                : CL_DEVICE_QUEUE_PROPERTIES;

            cl_queue_properties host_queue_props = 0;
            int error = clGetDeviceInfo(device, host_queue_query,
                                        sizeof(host_queue_props),
                                        &host_queue_props, NULL);
            test_error(error, "clGetDeviceInfo failed");

            if ((host_queue_props & CL_QUEUE_PROFILING_ENABLE) == 0)
                return true;
        }

        return BasicCommandBufferTest::Skip()
            || (simultaneous_use_requested && !simultaneous_use_support);
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        // By default command queue is created without properties,
        // if test requires queue with properties default queue must be
        // replaced.
        if (properties_use_requested)
        {
            // due to the skip condition
            cl_int error = CL_SUCCESS;
            queue = clCreateCommandQueue(context, device,
                                         CL_QUEUE_PROFILING_ENABLE, &error);
            test_error(
                error,
                "clCreateCommandQueue with CL_QUEUE_PROFILING_ENABLE failed");
        }

        return BasicCommandBufferTest::SetUp(elements);
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        // record command buffer with primary queue
        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        // create substitute queue
        clCommandQueueWrapper new_queue;
        if (properties_use_requested)
        {
            new_queue = clCreateCommandQueue(context, device,
                                             CL_QUEUE_PROFILING_ENABLE, &error);
            test_error(
                error,
                "clCreateCommandQueue with CL_QUEUE_PROFILING_ENABLE failed");
        }
        else
        {
            const cl_command_queue_properties queue_properties = 0;
            new_queue =
                clCreateCommandQueue(context, device, queue_properties, &error);
            test_error(error, "clCreateCommandQueue failed");
        }

        if (simultaneous_use_support)
        {
            // enque simultaneous command-buffers with substitute queue
            error = RunSimultaneous(new_queue);
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enque single command-buffer with substitute queue
            error = RunSingle(new_queue);
            test_error(error, "RunSingle failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle(const cl_command_queue& q)
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_int> output_data(num_elements);

        error = clEnqueueFillBuffer(q, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_command_queue queues[] = { q };
        error = clEnqueueCommandBufferKHR(1, queues, command_buffer, 0, nullptr,
                                          nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(q, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(q);
        test_error(error, "clFinish failed");

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
        cl_command_queue queue;
        std::vector<cl_int> output_buffer;
    };

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = clEnqueueFillBuffer(
            pd.queue, in_mem, &pd.pattern, sizeof(cl_int),
            pd.offset * sizeof(cl_int), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error =
            clEnqueueFillBuffer(pd.queue, off_mem, &pd.offset, sizeof(cl_int),
                                0, sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        if (!user_event)
        {
            user_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        cl_command_queue queues[] = { pd.queue };
        error = clEnqueueCommandBufferKHR(1, queues, command_buffer, 1,
                                          &user_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            pd.queue, out_mem, CL_FALSE, pd.offset * sizeof(cl_int),
            data_size(), pd.output_buffer.data(), 0, nullptr, nullptr);

        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous(const cl_command_queue& q)
    {
        cl_int error = CL_SUCCESS;
        cl_int offset = static_cast<cl_int>(num_elements);

        std::vector<SimulPassData> simul_passes = {
            { pattern_pri, 0, q, std::vector<cl_int>(num_elements) },
            { pattern_sec, offset, q, std::vector<cl_int>(num_elements) }
        };

        for (auto&& pass : simul_passes)
        {
            error = EnqueueSimultaneousPass(pass);
            test_error(error, "EnqueuePass failed");
        }

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        for (auto&& pass : simul_passes)
        {
            error = clFinish(pass.queue);
            test_error(error, "clFinish failed");

            auto& res_data = pass.output_buffer;

            for (size_t i = 0; i < num_elements; i++)
            {
                CHECK_VERIFICATION_ERROR(pass.pattern, res_data[i], i);
            }
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    const cl_int pattern_pri = 0xB;
    const cl_int pattern_sec = 0xC;

    bool properties_use_requested;
    clEventWrapper user_event;
};

// Command-queue substitution tests which handles below cases:
// * Template param is true - Create a command-buffer with an in-order queue,
//   and enqueue command-buffer to an out-of-order queue.
// * Template param is false - Create a command-buffer with an out-of-order
//   queue, and enqueue command-buffer to an in-order queue.
template <bool is_ooo_test>
struct QueueOrderTest : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    QueueOrderTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), ooo_queue(nullptr),
          ooo_command_buffer(this)
    {}

    cl_int RecordOutOfOrderCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        const cl_int pattern = pattern_pri;
        cl_int error =
            clCommandFillBufferKHR(ooo_command_buffer, nullptr, nullptr, in_mem,
                                   &pattern, sizeof(cl_int), 0, data_size(), 0,
                                   nullptr, &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandFillBufferKHR(ooo_command_buffer, nullptr, nullptr,
                                       out_mem, &overwritten_pattern,
                                       sizeof(cl_int), 0, data_size(), 0,
                                       nullptr, &sync_points[1], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            ooo_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        return CL_SUCCESS;
    }

    cl_int RecordInOrderCommandBuffer()
    {
        const cl_int pattern = pattern_pri;
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, nullptr, in_mem, &pattern, sizeof(cl_int),
            0, data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandFillBufferKHR(
            command_buffer, nullptr, nullptr, out_mem, &overwritten_pattern,
            sizeof(cl_int), 0, data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        if (is_ooo_test)
        {
            // command-buffer created in-order, but executed on ooo queue
            error = RecordInOrderCommandBuffer();
            test_error(error, "RecordInOrderCommandBuffer failed");
        }
        else
        {
            // command-buffer created ooo with sync point deps, but
            // executed on in-order queue
            error = RecordOutOfOrderCommandBuffer();
            test_error(error, "RecordOutOfOrderCommandBuffer failed");
        }

        clCommandBufferWrapper& test_command_buffer =
            is_ooo_test ? command_buffer : ooo_command_buffer;
        error = clFinalizeCommandBufferKHR(test_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        clCommandQueueWrapper& test_queue = is_ooo_test ? ooo_queue : queue;
        error = clEnqueueCommandBufferKHR(1, &test_queue, test_command_buffer,
                                          0, nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(test_queue);
        test_error(error, "clFinish failed");

        // Verify output
        std::vector<cl_int> output_buffer(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_buffer.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_buffer[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        ooo_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error,
                   "clCreateCommandQueue with "
                   "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE failed");

        ooo_command_buffer =
            clCreateCommandBufferKHR(1, &ooo_queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;

        // Skip if we want to enqueue to an out-of-order command-queue,
        // and this isn't supported.
        bool skip = is_ooo_test ? !out_of_order_support : false;

        // Skip if device doesn't support out-of-order queues, we need
        // to create one for both instantiations of the test.
        return skip || !queue_out_of_order_support;
    }

    clCommandQueueWrapper ooo_queue;
    clCommandBufferWrapper ooo_command_buffer;

    const cl_int overwritten_pattern = 0xACDC;
    const cl_int pattern_pri = 42;
};
} // anonymous namespace

REGISTER_TEST(queue_substitution)
{
    return MakeAndRunTest<SubstituteQueueTest<false, false>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(properties_queue_substitution)
{
    return MakeAndRunTest<SubstituteQueueTest<true, false>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(simultaneous_queue_substitution)
{
    return MakeAndRunTest<SubstituteQueueTest<false, true>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(queue_substitute_in_order)
{
    return MakeAndRunTest<QueueOrderTest<false>>(device, context, queue,
                                                 num_elements);
}

REGISTER_TEST(queue_substitute_out_of_order)
{
    return MakeAndRunTest<QueueOrderTest<true>>(device, context, queue,
                                                num_elements);
}
