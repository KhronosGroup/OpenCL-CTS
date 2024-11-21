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

#include <thread>

#include "semaphore_base.h"

#include "semaphore_base.h"

#define FLUSH_DELAY_S 5

namespace {

const char* source = "__kernel void empty() {}";

struct SimpleSemaphore1 : public SemaphoreTestBase
{
    SimpleSemaphore1(cl_device_id device, cl_context context,
                     cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_event);

        return CL_SUCCESS;
    }
};

struct SimpleSemaphore2 : public SemaphoreTestBase
{
    SimpleSemaphore2(cl_device_id device, cl_context context,
                     cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context, &err);
        test_error(err, "Could not create user event");

        // Create Kernel
        clProgramWrapper program;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &program, &kernel, 1,
                                          &source, "empty");
        test_error(err, "Could not create kernel");

        // Enqueue task_1 (dependency on user_event)
        clEventWrapper task_1_event;
        err = clEnqueueTask(queue, kernel, 1, &user_event, &task_1_event);
        test_error(err, "Could not enqueue task 1");

        // Signal semaphore
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Flush and delay
        err = clFlush(queue);
        test_error(err, "Could not flush queue");
        std::this_thread::sleep_for(std::chrono::seconds(FLUSH_DELAY_S));

        // Ensure all events are completed except for task_1
        test_assert_event_inprogress(task_1_event);
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_event);

        // Complete user_event
        err = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(err, "Could not set user event to CL_COMPLETE");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(task_1_event);
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_event);

        return CL_SUCCESS;
    }
};

struct SemaphoreReuse : public SemaphoreTestBase
{
    SemaphoreReuse(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Create Kernel
        clProgramWrapper program;
        clKernelWrapper kernel;
        err = create_single_kernel_helper(context, &program, &kernel, 1,
                                          &source, "empty");
        test_error(err, "Could not create kernel");

        constexpr size_t loop_count = 10;
        clEventWrapper signal_events[loop_count];
        clEventWrapper wait_events[loop_count];
        clEventWrapper task_events[loop_count];

        // Enqueue task_1
        err = clEnqueueTask(queue, kernel, 0, nullptr, &task_events[0]);
        test_error(err, "Unable to enqueue task_1");

        // Signal semaphore (dependency on task_1)
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                           &task_events[0], &signal_events[0]);
        test_error(err, "Could not signal semaphore");

        // In a loop
        size_t loop;
        for (loop = 1; loop < loop_count; ++loop)
        {
            // Wait semaphore
            err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                             nullptr, &wait_events[loop - 1]);
            test_error(err, "Could not wait semaphore");

            // Enqueue task_loop (dependency on wait)
            err = clEnqueueTask(queue, kernel, 1, &wait_events[loop - 1],
                                &task_events[loop]);
            test_error(err, "Unable to enqueue task_loop");

            // Wait for the "wait semaphore" to complete
            err = clWaitForEvents(1, &wait_events[loop - 1]);
            test_error(err, "Unable to wait for wait semaphore to complete");

            // Signal semaphore (dependency on task_loop)
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                               &task_events[loop],
                                               &signal_events[loop]);
            test_error(err, "Could not signal semaphore");
        }

        // Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_events[loop - 1]);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        for (loop = 0; loop < loop_count; ++loop)
        {
            test_assert_event_complete(wait_events[loop]);
            test_assert_event_complete(signal_events[loop]);
            test_assert_event_complete(task_events[loop]);
        }

        return CL_SUCCESS;
    }
};

struct SemaphoreMultiSignal : public SemaphoreTestBase
{
    SemaphoreMultiSignal(cl_device_id device, cl_context context,
                         cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems),
          semaphore_second(this)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        semaphore_second =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore 1 and 2
        clEventWrapper signal_event;
        cl_semaphore_khr sema_list[] = { semaphore, semaphore_second };
        err = clEnqueueSignalSemaphoresKHR(queue, 2, sema_list, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore 1
        clEventWrapper wait_1_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_1_event);
        test_error(err, "Could not wait semaphore");

        // Wait semaphore 2
        clEventWrapper wait_2_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore_second, nullptr, 0,
                                         nullptr, &wait_2_event);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_1_event);
        test_assert_event_complete(wait_2_event);

        return CL_SUCCESS;
    }
    clSemaphoreWrapper semaphore_second = nullptr;
};

struct SemaphoreMultiWait : public SemaphoreTestBase
{
    SemaphoreMultiWait(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems),
          semaphore_second(this)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphores
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        semaphore_second =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore 1
        clEventWrapper signal_1_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_1_event);
        test_error(err, "Could not signal semaphore");

        // Signal semaphore 2
        clEventWrapper signal_2_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore_second, nullptr,
                                           0, nullptr, &signal_2_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore 1 and 2
        clEventWrapper wait_event;
        cl_semaphore_khr sema_list[] = { semaphore, semaphore_second };
        err = clEnqueueWaitSemaphoresKHR(queue, 2, sema_list, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(signal_1_event);
        test_assert_event_complete(signal_2_event);
        test_assert_event_complete(wait_event);

        return CL_SUCCESS;
    }
    clSemaphoreWrapper semaphore_second = nullptr;
};

} // anonymous namespace

// Confirm that a signal followed by a wait will complete successfully
int test_semaphores_simple_1(cl_device_id deviceID, cl_context context,
                             cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SimpleSemaphore1>(deviceID, context, defaultQueue,
                                            num_elements);
}

// Confirm that signal a semaphore with no event dependencies will not result
// in an implicit dependency on everything previously submitted
int test_semaphores_simple_2(cl_device_id deviceID, cl_context context,
                             cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SimpleSemaphore2>(deviceID, context, defaultQueue,
                                            num_elements);
}

// Confirm that a semaphore can be reused multiple times
int test_semaphores_reuse(cl_device_id deviceID, cl_context context,
                          cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SemaphoreReuse>(deviceID, context, defaultQueue,
                                          num_elements);
}

// Confirm that we can signal multiple semaphores with one command
int test_semaphores_multi_signal(cl_device_id deviceID, cl_context context,
                                 cl_command_queue defaultQueue,
                                 int num_elements)
{
    return MakeAndRunTest<SemaphoreMultiSignal>(deviceID, context, defaultQueue,
                                                num_elements);
}

// Confirm that we can wait for multiple semaphores with one command
int test_semaphores_multi_wait(cl_device_id deviceID, cl_context context,
                               cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SemaphoreMultiWait>(deviceID, context, defaultQueue,
                                              num_elements);
}
