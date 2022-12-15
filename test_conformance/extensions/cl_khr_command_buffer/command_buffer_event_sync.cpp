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

enum class ReturnEventMode
{
    REM_REGULAR_WAIT_FOR_COMBUF = 0,
    REM_COMBUF_WAIT_FOR_COMBUF,
    REM_COMBUF_WAIT_FOR_OTHER_COMBUF,
    REM_EVENT_CALLBACK,
    REM_CLWAITFOREVENTS_WAIT_FOR_EVENT,
    REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS
};

enum class UserEventMode
{
    UEM_USER_EVENT_WAIT = 0,
    UEM_USER_EVENTS_WAIT,
    UEM_COMBUF_WAIT_FOR_RETULAR,
    UEM_WAIT_FOR_OTHER_QUEUE_EVENT
};

namespace {

////////////////////////////////////////////////////////////////////////////////

template <ReturnEventMode return_event_mode>
struct CommandBufferReturnEvent : public BasicCommandBufferTest
{
    CommandBufferReturnEvent(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), return_event(nullptr)
    {
        simultaneous_use_requested = false;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        switch (return_event_mode)
        {
            case ReturnEventMode::REM_REGULAR_WAIT_FOR_COMBUF:
                error = RunRegularWaitForCombuf();
                test_error(error, "RunSingle failed");
                break;
            case ReturnEventMode::REM_COMBUF_WAIT_FOR_COMBUF: break;
            case ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF: break;
            case ReturnEventMode::REM_EVENT_CALLBACK: break;
            case ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENT: break;
            case ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS: break;
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
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
    cl_int RunRegularWaitForCombuf()
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_int> output_data(num_elements);

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        const cl_int pattern = 0xA;
        error = clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    clEventWrapper return_event;
};

////////////////////////////////////////////////////////////////////////////////

// Test enqueueing a command-buffer blocked on a user-event
template <UserEventMode user_event_mode>
struct CommandBufferUserEvent : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    CommandBufferUserEvent(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), user_event(nullptr)
    {
        simultaneous_use_requested = false;
    }


    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        switch (user_event_mode)
        {
            case UserEventMode::UEM_USER_EVENT_WAIT:
                error = RunUserEventWait();
                test_error(error, "RunSingle failed");
                break;
            case UserEventMode::UEM_USER_EVENTS_WAIT: break;
            case UserEventMode::UEM_COMBUF_WAIT_FOR_RETULAR: break;
            case UserEventMode::UEM_WAIT_FOR_OTHER_QUEUE_EVENT: break;
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunUserEventWait()
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

    clEventWrapper user_event;
};

} // anonymous namespace

int test_regular_wait_for_command_buffer(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<
        CommandBufferReturnEvent<ReturnEventMode::REM_REGULAR_WAIT_FOR_COMBUF>>(
        device, context, queue, num_elements);
}

int test_command_buffer_wait_for_command_buffer(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<
        CommandBufferReturnEvent<ReturnEventMode::REM_COMBUF_WAIT_FOR_COMBUF>>(
        device, context, queue, num_elements);
}

int test_command_buffer_wait_for_other_command_buffer(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF>>(
        device, context, queue, num_elements);
}

int test_event_callback(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferReturnEvent<ReturnEventMode::REM_EVENT_CALLBACK>>(
        device, context, queue, num_elements);
}

int test_clwaitforevents_wait_for_event(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENT>>(
        device, context, queue, num_elements);
}

int test_clwaitforevents_wait_for_events(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS>>(
        device, context, queue, num_elements);
}

int test_user_event_wait(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferUserEvent<UserEventMode::UEM_USER_EVENT_WAIT>>(
        device, context, queue, num_elements);
}

int test_user_events_wait(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferUserEvent<UserEventMode::UEM_USER_EVENTS_WAIT>>(
        device, context, queue, num_elements);
}

int test_command_buffer_wait_for_regular(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<
        CommandBufferUserEvent<UserEventMode::UEM_COMBUF_WAIT_FOR_RETULAR>>(
        device, context, queue, num_elements);
}

int test_wait_for_other_queue_event(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferUserEvent<UserEventMode::UEM_WAIT_FOR_OTHER_QUEUE_EVENT>>(
        device, context, queue, num_elements);
}
