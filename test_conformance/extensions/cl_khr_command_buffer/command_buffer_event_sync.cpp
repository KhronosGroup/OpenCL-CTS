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
enum class EventMode
{
    RET_REGULAR_WAIT_FOR_COMBUF = 0,
    RET_COMBUF_WAIT_FOR_COMBUF,
    RET_COMBUF_WAIT_FOR_SEC_COMBUF,
    RET_EVENT_CALLBACK,
    RET_CLWAITFOREVENTS_SINGLE,
    RET_CLWAITFOREVENTS,
    RET_COMBUF_WAIT_FOR_REGULAR,
    RET_WAIT_FOR_SEC_QUEUE_EVENT,
    USER_EVENT_WAIT,
    USER_EVENTS_WAIT,
    USER_EVENT_CALLBACK
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
// event sync test cases for cl_khr_command_buffer which handles:
// -test that an event returned by a command-buffer enqueue can be waited on by
//  regular commands
// -test that an event returned by a command-buffer enqueue can
//  be waited on an enqueue of the same command-buffer
// -tests that a command buffer enqueue can wait on the enqueue of a different
//  command buffer
// -test clSetEventCallback works correctly on an event returned by
//  clEnqueueCommandBufferKHR
// -test clWaitForEvents on a single event returned from a
//  clEnqueueCommandBufferKHR
// -test clWaitForEvents on multiple events returned from different
//  clEnqueueCommandBufferKHR calls


//
//
// -test clSetEventCallback works correctly on an user defined event waited by
// clEnqueueCommandBufferKHR
//
//

template <EventMode event_mode, bool out_of_order_requested>
struct CommandBufferEventSync : public BasicCommandBufferTest
{
    CommandBufferEventSync(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          command_buffer_sec(this), kernel_sec(nullptr), in_mem_sec(nullptr),
          out_mem_sec(nullptr), off_mem_sec(nullptr), test_event(nullptr)
    {
        simultaneous_use_requested =
            (event_mode == EventMode::RET_COMBUF_WAIT_FOR_COMBUF) ? true
                                                                  : false;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // due to possible out-of-order command queue copy the kernel for below
        // case scenarios
        if (event_mode == EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF
            || event_mode == EventMode::RET_CLWAITFOREVENTS)
        {
            kernel_sec = clCreateKernel(program, "copy", &error);
            test_error(error, "Failed to create copy kernel");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        // due to possible out-of-order command queue it is necessary to create
        // separate set of kernel args for below cases
        if (event_mode == EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF
            || event_mode == EventMode::RET_CLWAITFOREVENTS)
        {
            // setup arguments for secondary kernel
            std::swap(kernel, kernel_sec);

            cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
            test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

            // swap arguments for base class setup
            in_mem_sec = in_mem;
            out_mem_sec = out_mem;
            off_mem_sec = off_mem;
            std::swap(kernel, kernel_sec);
        }

        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        if (out_of_order_requested && out_of_order_support)
        {
            queue = clCreateCommandQueue(context, device,
                                         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                         &error);
            test_error(error, "Unable to create command queue to test with");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        if (event_mode == EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF
            || event_mode == EventMode::RET_CLWAITFOREVENTS)
        {
            command_buffer_sec =
                clCreateCommandBufferKHR(1, &queue, nullptr, &error);
            test_error(error, "clCreateCommandBufferKHR failed");
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;

        if (simultaneous_use_requested && !simultaneous_use_support)
            return true;

        if (out_of_order_requested && !out_of_order_support) return true;

        return false;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer(command_buffer, kernel);
        test_error(error, "RecordCommandBuffer failed");

        switch (event_mode)
        {
            case EventMode::RET_REGULAR_WAIT_FOR_COMBUF:
                error = RunRegularWaitForCombuf();
                test_error(error, "RunRegularWaitForCombuf failed");
                break;
            case EventMode::RET_COMBUF_WAIT_FOR_COMBUF:
                error = RunCombufWaitForCombuf();
                test_error(error, "RunCombufWaitForCombuf failed");
                break;
            case EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF:
                error = RunCombufWaitForSecCombuf();
                test_error(error, "RunCombufWaitForSecCombuf failed");
                break;
            case EventMode::RET_EVENT_CALLBACK:
                error = RunReturnEventCallback();
                test_error(error, "RunReturnEventCallback failed");
                break;
            case EventMode::RET_CLWAITFOREVENTS_SINGLE:
                error = RunWaitForEvent();
                test_error(error, "RunWaitForEvent failed");
                break;
            case EventMode::RET_CLWAITFOREVENTS:
                error = RunWaitForEvents();
                test_error(error, "RunWaitForEvents failed");
                break;
            case EventMode::RET_COMBUF_WAIT_FOR_REGULAR:
                error = RunCombufWaitForRegular();
                test_error(error, "RunCombufWaitForRegular failed");
                break;
            case EventMode::RET_WAIT_FOR_SEC_QUEUE_EVENT:
                error = RunCombufWaitForSecQueueCombuf();
                test_error(error, "RunCombufWaitForSecQueueCombuf failed");
                break;
            case EventMode::USER_EVENT_WAIT:
                error = RunUserEventWait();
                test_error(error, "RunUserEventWait failed");
                break;
            case EventMode::USER_EVENTS_WAIT:
                error = RunUserEventsWait();
                test_error(error, "RunUserEventsWait failed");
                break;
            case EventMode::USER_EVENT_CALLBACK:
                error = RunUserEventCallback();
                test_error(error, "RunUserEventCallback failed");
                break;
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer(clCommandBufferWrapper &combuf,
                               clKernelWrapper &kern)
    {
        cl_int error = clCommandNDRangeKernelKHR(
            combuf, nullptr, nullptr, kern, 1, nullptr, &num_elements, nullptr,
            0, nullptr, nullptr, nullptr);
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
            in_order_events.resize(event_ptrs.size());
            for (size_t i = 0; i < in_order_events.size(); i++)
            {
                event_ptrs[i] = &in_order_events[i];
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
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunCombufWaitForSecCombuf()
    {
        std::vector<cl_int> output_data(num_elements);

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        // record other command buffer
        cl_int error = RecordCommandBuffer(command_buffer_sec, kernel_sec);
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
    cl_int RunReturnEventCallback()
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

        // verify the result
        if (!confirmation)
        {
            log_error("combuf_event_callback_function invocation failure\n");
            return TEST_FAIL;
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunWaitForEvent()
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
    cl_int RunWaitForEvents()
    {
        std::vector<cl_int> output_data(num_elements);
        clEventWrapper test_events[2];

        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        // record other command buffer
        cl_int error = RecordCommandBuffer(command_buffer_sec, kernel_sec);
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
                                    output_data.data(), 0, nullptr, nullptr);
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
    cl_int RunCombufWaitForRegular()
    {
        // if out-of-order queue requested it is necessary to secure proper
        // order of commands
        std::vector<cl_event *> event_ptrs = { nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error =
            clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, &test_event);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &test_event, event_ptrs[0]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[0], nullptr);
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
    cl_int RunCombufWaitForSecQueueCombuf()
    {
        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error = CL_SUCCESS;

        // create secondary command queue and command buffer
        clCommandQueueWrapper queue_sec =
            clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "Unable to create command queue to test with");

        command_buffer_sec =
            clCreateCommandBufferKHR(1, &queue_sec, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        // record secondary command buffer
        error = RecordCommandBuffer(command_buffer_sec, kernel);
        test_error(error, "RecordCommandBuffer failed");

        // process secondary queue
        error =
            clEnqueueFillBuffer(queue_sec, in_mem, &pattern_pri, sizeof(cl_int),
                                0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer_sec, 0,
                                          nullptr, &test_event);
        test_error(error,
                   "clEnqueueCommandBufferKHR in secondary queue failed");

        // process primary queue
        error = clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_event wait_list[] = { test_event,
                                 event_ptrs[0] != nullptr ? *event_ptrs[0]
                                                          : nullptr };
        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer,
                                      1 + wait_count, wait_list, event_ptrs[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[1], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFlush(queue);
        test_error(error, "clFlush failed");

        error = clFinish(queue_sec);
        test_error(error, "clFinish failed");

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
    cl_int RunUserEventWait()
    {
        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error = CL_SUCCESS;
        clEventWrapper user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        const cl_int pattern = 42;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_event wait_list[] = { user_event,
                                 event_ptrs[0] != nullptr ? *event_ptrs[0]
                                                          : nullptr };
        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer,
                                      wait_count + 1, wait_list, event_ptrs[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[1], nullptr);
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

    //--------------------------------------------------------------------------
    cl_int RunUserEventsWait()
    {
        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error = CL_SUCCESS;
        std::vector<clEventWrapper> user_events(user_event_num);

        for (size_t i = 0; i < user_event_num; i++)
        {
            user_events[i] = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        std::vector<cl_event> wait_list(user_event_num + wait_count);
        for (size_t i = 0; i < user_event_num; i++)
        {
            wait_list[i] = user_events[i];
        }
        if (out_of_order_requested) wait_list[user_event_num] = *event_ptrs[0];

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer,
                                          user_event_num + wait_count,
                                          &wait_list.front(), event_ptrs[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[1], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < user_event_num; i++)
        {
            error = clSetUserEventStatus(user_events[i], CL_COMPLETE);
            test_error(error, "clSetUserEventStatus failed");
        }

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunUserEventCallback()
    {
        // if out-of-order queue requested it is necessary to secure proper
        // order of all commands
        std::vector<cl_event *> event_ptrs = { nullptr, nullptr };
        InitInOrderEvents(event_ptrs);

        cl_int error = CL_SUCCESS;
        clEventWrapper user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int),
                                    0, data_size(), 0, nullptr, event_ptrs[0]);
        test_error(error, "clEnqueueFillBuffer failed");

        cl_event wait_list[] = { user_event,
                                 event_ptrs[0] != nullptr ? *event_ptrs[0]
                                                          : nullptr };
        error =
            clEnqueueCommandBufferKHR(0, nullptr, command_buffer,
                                      wait_count + 1, wait_list, event_ptrs[1]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        bool confirmation = false;
        error =
            clSetEventCallback(user_event, CL_COMPLETE,
                               combuf_event_callback_function, &confirmation);
        test_error(error, "clSetEventCallback failed");

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                    output_data.data(), wait_count,
                                    event_ptrs[1], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result
        if (!confirmation)
        {
            log_error("combuf_event_callback_function invocation failure\n");
            return TEST_FAIL;
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------

    clCommandBufferWrapper command_buffer_sec;
    clKernelWrapper kernel_sec;
    clMemWrapper in_mem_sec, out_mem_sec, off_mem_sec;
    clEventWrapper test_event;

    std::vector<clEventWrapper> in_order_events;

    const cl_int pattern_pri = 0xA;
    const cl_int pattern_sec = 0xB;
    const cl_int wait_count = out_of_order_requested ? 1 : 0;

    const cl_int user_event_num = 3;
};

} // anonymous namespace

// helper macros
#define IN_ORDER_MSG(name) #name " test with in-order command queue"
#define OUT_OF_ORDER_MSG(name) #name " test with out-of-order command queue"
#define test_status_val(code, msg)                                             \
    {                                                                          \
        if (code == TEST_FAIL)                                                 \
        {                                                                      \
            print_failure_error(code, TEST_PASS, msg " failed\n");             \
            return TEST_FAIL;                                                  \
        }                                                                      \
        else if (code == TEST_SKIP)                                            \
        {                                                                      \
            log_info(msg " skipped\n");                                        \
        }                                                                      \
    }

//--------------------------------------------------------------------------
// return-events test cases for regular queue
int test_regular_wait_for_command_buffer(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    int status = TEST_PASS;
    // The approach here is that test scenario which involves out-of-order
    // command queue may be skipped without breaking in-order queue test.
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_REGULAR_WAIT_FOR_COMBUF, true>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    OUT_OF_ORDER_MSG(EventMode::RET_REGULAR_WAIT_FOR_COMBUF));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_REGULAR_WAIT_FOR_COMBUF, false>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_REGULAR_WAIT_FOR_COMBUF));

    return status;
}

int test_command_buffer_wait_for_command_buffer(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_COMBUF_WAIT_FOR_COMBUF, true>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    OUT_OF_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_COMBUF));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_COMBUF_WAIT_FOR_COMBUF, false>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_COMBUF));

    return status;
}

int test_command_buffer_wait_for_sec_command_buffer(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<CommandBufferEventSync<
        EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF, true>>(device, context,
                                                          queue, num_elements);
    test_status_val(
        status, OUT_OF_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF));

    // in-order command queue test
    status = MakeAndRunTest<CommandBufferEventSync<
        EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF, false>>(device, context,
                                                           queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_SEC_COMBUF));

    return status;
}

int test_return_event_callback(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_EVENT_CALLBACK, true>>(
        device, context, queue, num_elements);
    test_status_val(status, OUT_OF_ORDER_MSG(EventMode::RET_EVENT_CALLBACK));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_EVENT_CALLBACK, false>>(
        device, context, queue, num_elements);
    test_status_val(status, IN_ORDER_MSG(EventMode::RET_EVENT_CALLBACK));

    return status;
}

int test_clwaitforevents_single(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_CLWAITFOREVENTS_SINGLE, true>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    OUT_OF_ORDER_MSG(EventMode::RET_CLWAITFOREVENTS_SINGLE));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_CLWAITFOREVENTS_SINGLE, false>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_CLWAITFOREVENTS_SINGLE));

    return status;
}

int test_clwaitforevents(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_CLWAITFOREVENTS, true>>(
        device, context, queue, num_elements);
    test_status_val(status, OUT_OF_ORDER_MSG(EventMode::RET_CLWAITFOREVENTS));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_CLWAITFOREVENTS, false>>(
        device, context, queue, num_elements);
    test_status_val(status, IN_ORDER_MSG(EventMode::RET_CLWAITFOREVENTS));

    return status;
}

int test_command_buffer_wait_for_regular(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_COMBUF_WAIT_FOR_REGULAR, true>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    OUT_OF_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_REGULAR));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_COMBUF_WAIT_FOR_REGULAR, false>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_COMBUF_WAIT_FOR_REGULAR));

    return status;
}

int test_wait_for_sec_queue_event(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_WAIT_FOR_SEC_QUEUE_EVENT, true>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    OUT_OF_ORDER_MSG(EventMode::RET_WAIT_FOR_SEC_QUEUE_EVENT));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::RET_WAIT_FOR_SEC_QUEUE_EVENT, false>>(
        device, context, queue, num_elements);
    test_status_val(status,
                    IN_ORDER_MSG(EventMode::RET_WAIT_FOR_SEC_QUEUE_EVENT));

    return status;
}

//--------------------------------------------------------------------------
// user-events test cases

int test_user_event_wait(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENT_WAIT, true>>(
        device, context, queue, num_elements);
    test_status_val(status, OUT_OF_ORDER_MSG(EventMode::USER_EVENT_WAIT));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENT_WAIT, false>>(
        device, context, queue, num_elements);
    test_status_val(status, IN_ORDER_MSG(EventMode::USER_EVENT_WAIT));

    return status;
}

int test_user_events_wait(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENTS_WAIT, true>>(
        device, context, queue, num_elements);
    test_status_val(status, OUT_OF_ORDER_MSG(EventMode::USER_EVENTS_WAIT));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENTS_WAIT, false>>(
        device, context, queue, num_elements);
    test_status_val(status, IN_ORDER_MSG(EventMode::USER_EVENTS_WAIT));

    return status;
}

int test_user_event_callback(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    int status = TEST_PASS;
    // out-of-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENT_CALLBACK, true>>(
        device, context, queue, num_elements);
    test_status_val(status, OUT_OF_ORDER_MSG(EventMode::USER_EVENT_CALLBACK));

    // in-order command queue test
    status = MakeAndRunTest<
        CommandBufferEventSync<EventMode::USER_EVENT_CALLBACK, false>>(
        device, context, queue, num_elements);
    test_status_val(status, IN_ORDER_MSG(EventMode::USER_EVENT_CALLBACK));

    return status;
}
