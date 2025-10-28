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

const char* source_write_int = "__kernel void write_int(__global int* out, int val) { out[0] = val; }";

#define CREATE_KERNEL \
    clProgramWrapper program_write_int; \
    clKernelWrapper kernel_write_int; \
    err = create_single_kernel_helper(context, &program_write_int, &kernel_write_int, 1, \
                                      &source_write_int, "write_int"); \
    size_t threads = 1; \
    test_error(err, "Could not create kernel")

#define CREATE_BUFFER \
    int int_val = 45; \
    clMemWrapper buffer_write_int = clCreateBuffer(context, CL_MEM_READ_WRITE, \
                                                   sizeof(cl_int), nullptr, &err); \
    test_error(err, "clCreateBuffer failed")

#define ENQUEUE_KERNEL(QUEUE, NUM_LIST, WAITLIST) \
    err = clSetKernelArg(kernel_write_int, 0, sizeof(buffer_write_int), &buffer_write_int); \
    test_error(err, "clSetKernelArg failed"); \
    err = clSetKernelArg(kernel_write_int, 1, sizeof(int), &int_val); \
    test_error(err, "clSetKernelArg failed"); \
    err = clEnqueueNDRangeKernel(QUEUE, kernel_write_int, 1, nullptr, \
                                 &threads, nullptr, NUM_LIST, WAITLIST, nullptr); \
    test_error(err, "clEnqueueNDRangeKernel failed")

#define ENQUEUE_KERNEL_WITH_EVENT(QUEUE, NUM_LIST, WAITLIST, EVENT) \
    err = clSetKernelArg(kernel_write_int, 0, sizeof(buffer_write_int), &buffer_write_int); \
    test_error(err, "clSetKernelArg failed"); \
    err = clSetKernelArg(kernel_write_int, 1, sizeof(int), &int_val); \
    test_error(err, "clSetKernelArg failed"); \
    clEventWrapper EVENT; \
    err = clEnqueueNDRangeKernel(QUEUE, kernel_write_int, 1, nullptr, \
                                 &threads, nullptr, NUM_LIST, WAITLIST, &EVENT); \
    test_error(err, "clEnqueueNDRangeKernel failed")

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

        CREATE_KERNEL;
        CREATE_BUFFER;

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

        ENQUEUE_KERNEL_WITH_EVENT(queue, 0, nullptr, write_int_event);

        // Signal semaphore
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                           &write_int_event, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");
        
        ENQUEUE_KERNEL(queue, 1, &wait_event);

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
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

        CREATE_KERNEL;
        CREATE_BUFFER;
        
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

        ENQUEUE_KERNEL(queue, 1, &wait_events[loop - 1]);
        
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

        CREATE_KERNEL;
        CREATE_BUFFER;
        
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

        ENQUEUE_KERNEL_WITH_EVENT(queue, 0, nullptr, write_int_event);
        
        // Signal semaphore 1 and 2
        clEventWrapper signal_event;
        cl_semaphore_khr sema_list[] = { semaphore, semaphore_second };
        err = clEnqueueSignalSemaphoresKHR(queue, 2, sema_list, nullptr, 1,
                                           &write_int_event, &signal_event);
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

        cl_event waitlist[2] = { wait_1_event, wait_2_event };
        ENQUEUE_KERNEL(queue, 2, waitlist);

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

        CREATE_KERNEL;
        CREATE_BUFFER;
        
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

        ENQUEUE_KERNEL_WITH_EVENT(queue, 0, nullptr, write_int_event1);
        ENQUEUE_KERNEL_WITH_EVENT(queue, 0, nullptr, write_int_event2);

        // Signal semaphore 1
        clEventWrapper signal_1_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                           &write_int_event1, &signal_1_event);
        test_error(err, "Could not signal semaphore");

        // Signal semaphore 2
        clEventWrapper signal_2_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore_second, nullptr,
                                           1, &write_int_event2, &signal_2_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore 1 and 2
        clEventWrapper wait_event;
        cl_semaphore_khr sema_list[] = { semaphore, semaphore_second };
        err = clEnqueueWaitSemaphoresKHR(queue, 2, sema_list, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");
            
        ENQUEUE_KERNEL(queue, 1, &wait_event);

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
REGISTER_TEST_VERSION(semaphores_simple_1, Version(1, 2))
{
    return MakeAndRunTest<SimpleSemaphore1>(device, context, queue,
                                            num_elements);
}

// Confirm that a semaphore can be reused multiple times
REGISTER_TEST_VERSION(semaphores_reuse, Version(1, 2))
{
    return MakeAndRunTest<SemaphoreReuse>(device, context, queue, num_elements);
}

// Confirm that we can signal multiple semaphores with one command
REGISTER_TEST_VERSION(semaphores_multi_signal, Version(1, 2))
{
    return MakeAndRunTest<SemaphoreMultiSignal>(device, context, queue,
                                                num_elements);
}

// Confirm that we can wait for multiple semaphores with one command
REGISTER_TEST_VERSION(semaphores_multi_wait, Version(1, 2))
{
    return MakeAndRunTest<SemaphoreMultiWait>(device, context, queue,
                                              num_elements);
}
