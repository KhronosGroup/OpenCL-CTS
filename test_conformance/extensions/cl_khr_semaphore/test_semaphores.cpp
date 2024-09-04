//
// Copyright (c) 2023 The Khronos Group Inc.
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


#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"
#include <system_error>
#include <thread>
#include <chrono>
#include <vector>

#include "semaphore_base.h"

#define FLUSH_DELAY_S 5

#define SEMAPHORE_PARAM_TEST(param_name, param_type, expected)                 \
    do                                                                         \
    {                                                                          \
        param_type value;                                                      \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(semaphore, param_name,            \
                                             sizeof(value), &value, &size);    \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (value != expected)                                                 \
        {                                                                      \
            test_fail("ERROR: Parameter %s did not validate! (expected %d, "   \
                      "got %d)\n",                                             \
                      #param_name, expected, value);                           \
        }                                                                      \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
    } while (false)

#define SEMAPHORE_PARAM_TEST_ARRAY(param_name, param_type, num_params,         \
                                   expected)                                   \
    do                                                                         \
    {                                                                          \
        param_type value[num_params];                                          \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(semaphore, param_name,            \
                                             sizeof(value), &value, &size);    \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
        if (memcmp(value, expected, size) != 0)                                \
        {                                                                      \
            test_fail("ERROR: Parameter %s did not validate!\n", #param_name); \
        }                                                                      \
    } while (false)

namespace {

const char* source = "__kernel void empty() {}";

struct SimpleSemaphore1 : public SemaphoreTestBase
{
    SimpleSemaphore1(cl_device_id device, cl_context context,
                     cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
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
                     cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
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
                   cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
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

template <bool in_order> struct SemaphoreCrossQueue : public SemaphoreTestBase
{
    SemaphoreCrossQueue(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    // Helper function that signals and waits on semaphore across two different
    // queues.
    int semaphore_cross_queue_helper(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue_1,
                                     cl_command_queue queue_2)
    {
        cl_int err = CL_SUCCESS;
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

        // Signal semaphore on queue_1
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue_1, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore on queue_2
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue_2, 1, semaphore, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Finish queue_1 and queue_2
        err = clFinish(queue_1);
        test_error(err, "Could not finish queue");

        err = clFinish(queue_2);
        test_error(err, "Could not finish queue");

        // Ensure all events are completed
        test_assert_event_complete(signal_event);
        test_assert_event_complete(wait_event);

        return TEST_PASS;
    }

    cl_int run_in_order()
    {
        cl_int err = CL_SUCCESS;
        // Create in-order queues
        clCommandQueueWrapper queue_1 =
            clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "Could not create command queue");

        clCommandQueueWrapper queue_2 =
            clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "Could not create command queue");

        return semaphore_cross_queue_helper(device, context, queue_1, queue_2);
    }

    cl_int run_out_of_order()
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queues
        clCommandQueueWrapper queue_1 = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        clCommandQueueWrapper queue_2 = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        return semaphore_cross_queue_helper(device, context, queue_1, queue_2);
    }

    cl_int Run() override
    {
        if (in_order)
            return run_in_order();
        else
            return run_out_of_order();
    }
};

struct SemaphoreMultiSignal : public SemaphoreTestBase
{
    SemaphoreMultiSignal(cl_device_id device, cl_context context,
                         cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue), semaphore_second(this)
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
                       cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue), semaphore_second(this)
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

struct SemaphoreQueries : public SemaphoreTestBase
{
    SemaphoreQueries(cl_device_id device, cl_context context,
                     cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create binary semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR),
            (cl_semaphore_properties_khr)device,
            CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Confirm that querying CL_SEMAPHORE_TYPE_KHR returns
        // CL_SEMAPHORE_TYPE_BINARY_KHR
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_TYPE_KHR, cl_semaphore_type_khr,
                             CL_SEMAPHORE_TYPE_BINARY_KHR);

        // Confirm that querying CL_SEMAPHORE_CONTEXT_KHR returns the right
        // context
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_CONTEXT_KHR, cl_context, context);

        // Confirm that querying CL_SEMAPHORE_REFERENCE_COUNT_KHR returns the
        // right value
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 1);

        err = clRetainSemaphoreKHR(semaphore);
        test_error(err, "Could not retain semaphore");
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 2);

        err = clReleaseSemaphoreKHR(semaphore);
        test_error(err, "Could not release semaphore");
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, 1);

        // Confirm that querying CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR returns the
        // same device id the semaphore was created with
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR, cl_device_id,
                             device);

        // Confirm that querying CL_SEMAPHORE_PROPERTIES_KHR returns the same
        // properties the semaphore was created with
        SEMAPHORE_PARAM_TEST_ARRAY(CL_SEMAPHORE_PROPERTIES_KHR,
                                   cl_semaphore_properties_khr, 6, sema_props);

        // Confirm that querying CL_SEMAPHORE_PAYLOAD_KHR returns the unsignaled
        // state
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_PAYLOAD_KHR, cl_semaphore_payload_khr,
                             0);

        return CL_SUCCESS;
    }
};

struct SemaphoreImportExportFD : public SemaphoreTestBase
{
    SemaphoreImportExportFD(cl_device_id device, cl_context context,
                            cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue), semaphore_second(this)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        if (!is_extension_available(device,
                                    "cl_khr_external_semaphore_sync_fd"))
        {
            log_info(
                "cl_khr_external_semaphore_sync_fd is not supported on this "
                "platform. Skipping test.\n");
            return TEST_SKIPPED_ITSELF;
        }

        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_1_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_SYNC_FD_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_1_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore
        clEventWrapper signal_event;
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, &signal_event);
        test_error(err, "Could not signal semaphore");

        // Extract sync fd
        int handle = -1;
        size_t handle_size;
        err = clGetSemaphoreHandleForTypeKHR(
            semaphore, device, CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, sizeof(handle),
            &handle, &handle_size);
        test_error(err, "Could not extract semaphore handle");
        test_assert_error(sizeof(handle) == handle_size, "Invalid handle size");
        test_assert_error(handle >= 0, "Invalid handle");

        // Create semaphore from sync fd
        cl_semaphore_properties_khr sema_2_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
            static_cast<cl_semaphore_properties_khr>(handle), 0
        };

        semaphore_second =
            clCreateSemaphoreWithPropertiesKHR(context, sema_2_props, &err);
        test_error(err, "Could not create semaphore");

        // Wait semaphore
        clEventWrapper wait_event;
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore_second, nullptr, 0,
                                         nullptr, &wait_event);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        // Check all events are completed
        test_assert_event_complete(signal_event);
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
    return MakeAndRunTest<SimpleSemaphore1>(deviceID, context, defaultQueue);
}

// Confirm that signal a semaphore with no event dependencies will not result
// in an implicit dependency on everything previously submitted
int test_semaphores_simple_2(cl_device_id deviceID, cl_context context,
                             cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SimpleSemaphore2>(deviceID, context, defaultQueue);
}

// Confirm that a semaphore can be reused multiple times
int test_semaphores_reuse(cl_device_id deviceID, cl_context context,
                          cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SemaphoreReuse>(deviceID, context, defaultQueue);
}

// Confirm that a semaphore works across different ooo queues
int test_semaphores_cross_queues_ooo(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    return MakeAndRunTest<SemaphoreCrossQueue<false>>(deviceID, context,
                                                      defaultQueue);
}

// Confirm that a semaphore works across different in-order queues
int test_semaphores_cross_queues_io(cl_device_id deviceID, cl_context context,
                                    cl_command_queue defaultQueue,
                                    int num_elements)
{
    return MakeAndRunTest<SemaphoreCrossQueue<true>>(deviceID, context,
                                                     defaultQueue);
}

// Confirm that we can signal multiple semaphores with one command
int test_semaphores_multi_signal(cl_device_id deviceID, cl_context context,
                                 cl_command_queue defaultQueue,
                                 int num_elements)
{
    return MakeAndRunTest<SemaphoreMultiSignal>(deviceID, context,
                                                defaultQueue);
}

// Confirm that we can wait for multiple semaphores with one command
int test_semaphores_multi_wait(cl_device_id deviceID, cl_context context,
                               cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SemaphoreMultiWait>(deviceID, context, defaultQueue);
}

// Confirm the semaphores can be successfully queried
int test_semaphores_queries(cl_device_id deviceID, cl_context context,
                            cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<SemaphoreQueries>(deviceID, context, defaultQueue);
}

// Test it is possible to export a semaphore to a sync fd and import the same
// sync fd to a new semaphore
int test_semaphores_import_export_fd(cl_device_id deviceID, cl_context context,
                                     cl_command_queue defaultQueue,
                                     int num_elements)
{
    return MakeAndRunTest<SemaphoreImportExportFD>(deviceID, context,
                                                   defaultQueue);
}
