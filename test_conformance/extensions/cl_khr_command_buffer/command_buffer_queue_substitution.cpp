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

} // anonymous namespace

int test_queue_substitution(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<SubstituteQueueTest<false, false>>(
        device, context, queue, num_elements);
}

int test_properties_queue_substitution(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<SubstituteQueueTest<true, false>>(
        device, context, queue, num_elements);
}

int test_simultaneous_queue_substitution(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<SubstituteQueueTest<false, true>>(
        device, context, queue, num_elements);
}
