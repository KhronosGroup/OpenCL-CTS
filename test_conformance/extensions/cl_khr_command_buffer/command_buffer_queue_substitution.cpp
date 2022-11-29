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
        if (properties_use_requested && queue == nullptr) return true;

        return (simultaneous_use_requested && !simultaneous_use_support)
            || BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_command_queue CreateCommandQueueWithProperties(cl_int& error)
    {
        cl_command_queue ret_queue = nullptr;
        cl_queue_properties_khr device_props = 0;

        error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                                sizeof(device_props), &device_props, nullptr);
        test_error_ret(error,
                       "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed",
                       nullptr);

        using PropPair = std::pair<cl_queue_properties_khr, std::string>;

        auto check_property = [&](const PropPair& prop) {
            if (device_props & prop.first)
            {
                log_info("Queue property %s supported. Testing ... \n",
                         prop.second.c_str());
                ret_queue =
                    clCreateCommandQueue(context, device, prop.first, &error);
            }
            else
                log_info("Queue property %s not supported \n",
                         prop.second.c_str());
        };

        // in case of extending property list in future
        std::vector<PropPair> props = {
            ADD_PROP(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
            ADD_PROP(CL_QUEUE_PROFILING_ENABLE)
        };

        for (auto&& prop : props)
        {
            check_property(prop);
            test_error_ret(error, "clCreateCommandQueue failed", ret_queue);
            if (ret_queue != nullptr) return ret_queue;
        }

        return ret_queue;
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
            queue = CreateCommandQueueWithProperties(error);
            test_error(error, "CreateCommandQueueWithProperties failed");

            cl_command_queue_properties cqp;
            error = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                          sizeof(cqp), &cqp, NULL);
            test_error(error, "clGetCommandQueueInfo failed");

            if (simultaneous_use_support
                && (cqp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE))
            {
                log_info(
                    "Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE "
                    "not supported with simultaneous use in this test\n");
                return CL_INVALID_QUEUE_PROPERTIES;
            }
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
            new_queue = CreateCommandQueueWithProperties(error);
            test_error(error, "CreateCommandQueueWithProperties failed");
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

        if (properties_use_requested)
        {
            clReleaseCommandQueue(queue);
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

        // this could be out-of-order queue, cover such possibility with events
        clEventWrapper events[2] = { nullptr, nullptr };

        error = clEnqueueFillBuffer(q, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, &events[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_command_queue queues[] = { q };
        error = clEnqueueCommandBufferKHR(1, queues, command_buffer, 1,
                                          &events[0], &events[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(q, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 1, &events[1], nullptr);
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

    bool properties_use_requested = false;
    clEventWrapper user_event = nullptr;
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
