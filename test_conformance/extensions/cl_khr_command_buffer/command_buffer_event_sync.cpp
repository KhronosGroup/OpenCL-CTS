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

//--------------------------------------------------------------------------
enum class ReturnEventMode
{
    REM_REGULAR_WAIT_FOR_COMBUF = 0,
    REM_COMBUF_WAIT_FOR_COMBUF,
    REM_COMBUF_WAIT_FOR_OTHER_COMBUF,
    REM_EVENT_CALLBACK,
    REM_CLWAITFOREVENTS_WAIT_FOR_EVENT,
    REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS
};

//--------------------------------------------------------------------------
enum class UserEventMode
{
    UEM_USER_EVENT_WAIT = 0,
    UEM_USER_EVENTS_WAIT,
    UEM_COMBUF_WAIT_FOR_RETULAR,
    UEM_WAIT_FOR_OTHER_QUEUE_EVENT
};

//--------------------------------------------------------------------------
void CL_CALLBACK combuf_event_callback_function(cl_event event,
                                                cl_int commandStatus,
                                                void *userData)
{
    bool *pdata = static_cast<bool *>(userData);
    log_info("\tEvent callback of clEnqueueCommandBufferKHR triggered\n");
    *pdata = true;
}

namespace {

////////////////////////////////////////////////////////////////////////////////
//
template <ReturnEventMode return_event_mode, bool out_of_order_requested>
struct CommandBufferReturnEvent : public BasicCommandBufferTest
{
    CommandBufferReturnEvent(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          out_of_order_queue(nullptr), command_buffer_sec(nullptr),
          kernel_sec(nullptr), in_mem_sec(nullptr), out_mem_sec(nullptr),
          off_mem_sec(nullptr), test_event(nullptr)
    {
        simultaneous_use_requested = false;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // copy of the copy kernel for out-of-order case scenarios
        if (out_of_order_requested && out_of_order_support
            && (return_event_mode
                    == ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF
                || return_event_mode
                    == ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS))
        {
            kernel_sec = clCreateKernel(program, "copy", &error);
            test_error(error, "Failed to create copy kernel");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        if (out_of_order_requested && out_of_order_support
            && (return_event_mode
                    == ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF
                || return_event_mode
                    == ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS))
        {
            // setup arguments for secondary kernel
            std::swap(kernel, kernel_sec);

            cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
            test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

            // restore arguments for base class setup
            std::swap(in_mem_sec, in_mem);
            std::swap(out_mem_sec, out_mem);
            std::swap(off_mem_sec, off_mem);
            std::swap(kernel, kernel_sec);
        }

        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        cl_int error = CL_SUCCESS;

        if (out_of_order_requested && out_of_order_support
            && (return_event_mode
                    == ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF
                || return_event_mode
                    == ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS))
        {
            out_of_order_queue = clCreateCommandQueue(
                context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                &error);
            test_error(error, "Unable to create command queue to test with");

            command_buffer_sec = clCreateCommandBufferKHR(
                1, &out_of_order_queue, nullptr, &error);
            test_error(error, "clCreateCommandBufferKHR failed");
        }
        else
        {
            command_buffer_sec =
                clCreateCommandBufferKHR(1, &queue, nullptr, &error);
            test_error(error, "clCreateCommandBufferKHR failed");
        }

        return BasicCommandBufferTest::SetUp(elements);
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        switch (return_event_mode)
        {
            case ReturnEventMode::REM_REGULAR_WAIT_FOR_COMBUF:
                error = RunRegularWaitForCombuf();
                test_error(error, "RunRegularWaitForCombuf failed");
                break;
            case ReturnEventMode::REM_COMBUF_WAIT_FOR_COMBUF:
                error = RunCombufWaitForCombuf();
                test_error(error, "RunCombufWaitForCombuf failed");
                break;
            case ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF:
                error = RunCombufWaitForOtherCombuf();
                test_error(error, "RunCombufWaitForCombuf failed");
                break;
            case ReturnEventMode::REM_EVENT_CALLBACK:
                error = RunCombufEventCallback();
                test_error(error, "RunCombufEventCallback failed");
                break;
            case ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENT:
                error = RunWaitForCombufEvent();
                test_error(error, "RunWaitForCombufEvent failed");
                break;
            case ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS:
                error = RunWaitForCombufEvents();
                test_error(error, "RunWaitForCombufEvents failed");
                break;
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer(clCommandBufferWrapper &combuf)
    {
        cl_int error = clCommandNDRangeKernelKHR(
            combuf, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(combuf);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    void InitInOrderEvents(std::vector<cl_event *> &event_ptrs)
    {
        if (out_of_order_requested)
        {
            for (auto &&eptr : event_ptrs)
            {
                in_order_events.emplace_back(clEventWrapper());
                eptr = &in_order_events.back();
            }
        }
    }

    //--------------------------------------------------------------------------
    cl_int RunRegularWaitForCombuf()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of commands
        std::vector<cl_event *> event_ptrs = { nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error =
            clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, command_buffer, wait_count, event_ptrs[0], &test_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error =
            clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data.data(), 1, &test_event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunCombufWaitForCombuf()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error =
            clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, command_buffer, wait_count, event_ptrs[0], &test_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &test_event, event_ptrs[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[1], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_sec, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunCombufWaitForOtherCombuf()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        // record other command buffer
        cl_int error = RecordCommandBuffer(command_buffer_sec);
        test_error(error, "RecordCommandBuffer failed");

        error =
            clEnqueueFillBuffer(queue, in_mem_sec, &pattern_pri, sizeof(cl_int),
                                0, data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer_sec,
                                      wait_count, event_ptrs[0], &test_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_sec, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, event_ptrs[1]);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_event wait_list[] = { test_event,
                                 event_ptrs[1] != nullptr ? *event_ptrs[1]
                                                          : nullptr };
        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer,
                                      1 + wait_count, wait_list, event_ptrs[2]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_sec, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunCombufEventCallback()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error =
            clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, command_buffer, wait_count, event_ptrs[0], &test_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        bool confirmation = false;
        error =
            clSetEventCallback(test_event, CL_COMPLETE,
                               combuf_event_callback_function, &confirmation);
        test_error(error, "clSetEventCallback failed");

        error = clWaitForEvents(1, &test_event);
        test_error(error, "clWaitForEvents failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        if (!confirmation)
        {
            log_error("combuf_event_callback_function invocation failure\n");
            return TEST_FAIL;
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunWaitForCombufEvent()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error =
            clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, command_buffer, wait_count, event_ptrs[0], &test_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &test_event);
        test_error(error, "clWaitForEvents failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunWaitForCombufEvents()
    {
        std::vector<cl_int> output_data(num_elements);
        clEventWrapper test_events[2];

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        // record other command buffer
        cl_int error = RecordCommandBuffer(command_buffer_sec);
        test_error(error, "RecordCommandBuffer failed");

        error =
            clEnqueueFillBuffer(queue, in_mem_sec, &pattern_pri, sizeof(cl_int),
                                0, data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer_sec,
                                          wait_count, event_ptrs[0],
                                          &test_events[0]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_sec, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, event_ptrs[1]);
        test_error(error, "clEnqueueFillBuffer failed");

        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer, wait_count,
                                      event_ptrs[1], &test_events[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_event wait_list[] = { test_events[0], test_events[1] };
        error = clWaitForEvents(2, wait_list);
        test_error(error, "clWaitForEvents failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_sec, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------

    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper command_buffer_sec;
    clKernelWrapper kernel_sec;
    clMemWrapper in_mem_sec, out_mem_sec, off_mem_sec;
    clEventWrapper test_event;

    std::vector<clEventWrapper> in_order_events;

    const cl_int pattern_pri = 0xA;
    const cl_int pattern_sec = 0xB;
    const cl_int wait_count = out_of_order_requested ? 1 : 0;
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
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_REGULAR_WAIT_FOR_COMBUF, false>>(
        device, context, queue, num_elements);
}

int test_command_buffer_wait_for_command_buffer(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_COMBUF_WAIT_FOR_COMBUF, false>>(
        device, context, queue, num_elements);
}

int test_command_buffer_wait_for_other_command_buffer(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_COMBUF_WAIT_FOR_OTHER_COMBUF, false>>(
        device, context, queue, num_elements);
}

int test_event_callback(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferReturnEvent<ReturnEventMode::REM_EVENT_CALLBACK, false>>(
        device, context, queue, num_elements);
}

int test_clwaitforevents_wait_for_event(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENT, false>>(
        device, context, queue, num_elements);
}

int test_clwaitforevents_wait_for_events(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<CommandBufferReturnEvent<
        ReturnEventMode::REM_CLWAITFOREVENTS_WAIT_FOR_EVENTS, false>>(
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
