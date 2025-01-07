//
// Copyright (c) 2024 The Khronos Group Inc.
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


//--------------------------------------------------------------------------
namespace {

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct EnqueueCommandBufferInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clEnqueueCommandBufferKHR(0, nullptr, nullptr, 0, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has not been finalized.
struct EnqueueCommandBufferNotFinalized : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_command_buffer_state_khr state = ~cl_command_buffer_state_khr(0);

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state), &state,
            nullptr);
        test_error_ret(error, "clGetCommandBufferInfoKHR failed", TEST_FAIL);
        test_assert_error(state == CL_COMMAND_BUFFER_STATE_RECORDING_KHR,
                          "Command buffer not in recording state!");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer was not created with the
// CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR flag and is in the Pending state.
struct EnqueueCommandBufferWithoutSimultaneousUseNotInPendingState
    : public BasicCommandBufferTest
{
    EnqueueCommandBufferWithoutSimultaneousUseNotInPendingState(
        cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), user_event(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                                 nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");
        clFinish(queue);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        auto verify_state = [&](const cl_command_buffer_state_khr &expected) {
            cl_command_buffer_state_khr state = ~cl_command_buffer_state_khr(0);

            cl_int error = clGetCommandBufferInfoKHR(
                command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state),
                &state, nullptr);
            test_error_ret(error, "clGetCommandBufferInfoKHR failed",
                           TEST_FAIL);

            test_assert_error(
                state == expected,
                "Unexpected result of CL_COMMAND_BUFFER_STATE_KHR query!");

            return TEST_PASS;
        };

        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        command_buffer = clCreateCommandBufferKHR(1, &queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");
        error = verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        test_error(error, "State is not Executable");

        error = EnqueueCommandBuffer();
        test_error(error, "EnqueueCommandBuffer failed");
        error = verify_state(CL_COMMAND_BUFFER_STATE_PENDING_KHR);
        test_error(error, "State is not Pending");

        return CL_SUCCESS;
    }

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

    cl_int EnqueueCommandBuffer()
    {
        cl_int pattern = 0xE;

        cl_int error =
            clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &user_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        return CL_SUCCESS;
    }
    clEventWrapper user_event;
};

// CL_INVALID_VALUE if queues is NULL and num_queues is > 0, or queues is not
// NULL and num_queues is 0.
struct EnqueueCommandBufferNullQueuesNumQueues : public BasicCommandBufferTest
{
    EnqueueCommandBufferNullQueuesNumQueues(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), queue1(nullptr),
          queue2(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(1, nullptr, command_buffer, 0,
                                          nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        cl_command_queue queues[1] = { queue1 };
        error = clEnqueueCommandBufferKHR(0, queues, command_buffer, 0, nullptr,
                                          nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        queue1 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");
        queue2 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue1;
    clCommandQueueWrapper queue2;
};

// CL_INVALID_VALUE if num_queues is > 0 and not the same value as num_queues
// set on command_buffer creation.
struct EnqueueCommandBufferNumQueuesNotZeroAndDifferentThanWhileBufferCreation
    : public BasicCommandBufferTest
{
    EnqueueCommandBufferNumQueuesNotZeroAndDifferentThanWhileBufferCreation(
        cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), queue1(nullptr),
          queue2(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        command_buffer = clCreateCommandBufferKHR(1, &queue, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");
        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        const auto num_queues = 2;
        cl_command_queue queues[num_queues] = { queue1, queue2 };
        error = clEnqueueCommandBufferKHR(num_queues, queues, command_buffer, 0,
                                          nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        queue1 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");
        queue2 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue1;
    clCommandQueueWrapper queue2;
};

// CL_INVALID_COMMAND_QUEUE if any element of queues is not a valid
// command-queue.
struct EnqueueCommandBufferNotValidQueueInQueues : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_command_queue queues[1] = { nullptr };
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(1, queues, command_buffer, 0, nullptr,
                                          nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INCOMPATIBLE_COMMAND_QUEUE_KHR if any element of queues is not compatible
// with the command-queue set on command_buffer creation at the same list index.
struct EnqueueCommandBufferQueueNotCompatible : public BasicCommandBufferTest
{
    EnqueueCommandBufferQueueNotCompatible(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          queue_not_compatible(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(1, &queue_not_compatible,
                                          command_buffer, 0, nullptr, nullptr);

        test_failure_error_ret(error, CL_INCOMPATIBLE_COMMAND_QUEUE_KHR,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INCOMPATIBLE_COMMAND_QUEUE_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        queue_not_compatible = clCreateCommandQueue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &error);
        test_error(error, "clCreateCommandQueue failed");

        cl_command_queue_properties queue_properties;
        error = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                      sizeof(queue_properties),
                                      &queue_properties, NULL);
        test_error(error, "Unable to query CL_QUEUE_PROPERTIES");

        cl_command_queue_properties queue_not_compatible_properties;
        error = clGetCommandQueueInfo(queue_not_compatible, CL_QUEUE_PROPERTIES,
                                      sizeof(queue_not_compatible_properties),
                                      &queue_not_compatible_properties, NULL);
        test_error(error, "Unable to query CL_QUEUE_PROPERTIES");

        test_assert_error(queue_properties != queue_not_compatible_properties,
                          "Queues properties must be different");

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue_not_compatible;
};

// CL_INVALID_CONTEXT if any element of queues does not have the same context as
// the command-queue set on command_buffer creation at the same list index.
struct EnqueueCommandBufferQueueWithDifferentContext
    : public BasicCommandBufferTest
{
    EnqueueCommandBufferQueueWithDifferentContext(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          queue_different_context(nullptr)
    {}

    cl_int Run() override
    {
        cl_command_queue queues[1] = { queue };
        cl_command_queue queues1[1] = { queue_different_context };
        test_assert_error(queues[0] != queues1[0], "Queues must be different");

        cl_int error = CL_SUCCESS;
        command_buffer = clCreateCommandBufferKHR(1, queues, nullptr, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(1, queues1, command_buffer, 0,
                                          nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        clContextWrapper context1 =
            clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        queue_different_context =
            clCreateCommandQueue(context1, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue_different_context;
};

// CL_INVALID_CONTEXT if context associated with command_buffer and events in
// event_wait_list are not the same.
struct EnqueueCommandBufferWithDiferentContextThanEvent
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        clContextWrapper context1 =
            clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        clEventWrapper event_different_context =
            clCreateUserEvent(context1, &error);
        test_error(error, "Event creation failed");

        clEventWrapper event_list[1] = { event_different_context };

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &event_list[0], nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        cl_int signal_error =
            clSetUserEventStatus(event_different_context, CL_COMPLETE);
        test_error(signal_error, "clSetUserEventStatus failed");
        clFinish(queue);

        return CL_SUCCESS;
    }
};

// CL_INVALID_EVENT_WAIT_LIST if event_wait_list is NULL and
// num_events_in_wait_list > 0, or event_wait_list is not NULL and
// num_events_in_wait_list is 0, or if event objects in event_wait_list are not
// valid events.
struct EnqueueCommandBufferEventWaitListNullOrEventsNull
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        clEventWrapper invalid_event_list[2] = { nullptr, nullptr };
        clEventWrapper event = clCreateUserEvent(context, &error);
        test_error(error, "Event creation failed");

        clEventWrapper event_list[1] = { event };

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_EVENT_WAIT_LIST,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_EVENT_WAIT_LIST",
                               TEST_FAIL);

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          &event_list[0], nullptr);

        test_failure_error_ret(error, CL_INVALID_EVENT_WAIT_LIST,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_EVENT_WAIT_LIST",
                               TEST_FAIL);

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 2,
                                          &invalid_event_list[0], nullptr);

        test_failure_error_ret(error, CL_INVALID_EVENT_WAIT_LIST,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_EVENT_WAIT_LIST",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};
};

int test_negative_enqueue_command_buffer_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_not_finalized(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferNotFinalized>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_without_simultaneous_no_pending_state(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<
        EnqueueCommandBufferWithoutSimultaneousUseNotInPendingState>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_null_queues_num_queues(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferNullQueuesNumQueues>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_num_queues_not_zero_different_while_buffer_creation(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<
        EnqueueCommandBufferNumQueuesNotZeroAndDifferentThanWhileBufferCreation>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_not_valid_queue_in_queues(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferNotValidQueueInQueues>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_queue_not_compatible(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferQueueNotCompatible>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_queue_with_different_context(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferQueueWithDifferentContext>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_command_buffer_different_context_than_event(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferWithDiferentContextThanEvent>(
        device, context, queue, num_elements);
}

int test_negative_enqueue_event_wait_list_null_or_events_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<EnqueueCommandBufferEventWaitListNullOrEventsNull>(
        device, context, queue, num_elements);
}
