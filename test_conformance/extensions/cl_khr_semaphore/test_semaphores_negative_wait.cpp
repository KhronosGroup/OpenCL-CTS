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

#include "semaphore_base.h"

#include "harness/errorHelpers.h"
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>

namespace {

// the device associated with command_queue is not same as one of the devices
// specified by CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR at the time of creating one
// or more of sema_objects.

struct WaitInvalidCommandQueue : public SemaphoreTestBase
{
    WaitInvalidCommandQueue(cl_device_id device, cl_context context,
                            cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Create semaphore
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

        cl_int err = CL_SUCCESS;
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // find other device
        cl_platform_id platform_id = 0;
        // find out what platform the harness is using.
        err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                              sizeof(cl_platform_id), &platform_id, nullptr);
        test_error(err, "clGetDeviceInfo failed");

        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(16, nullptr, &num_platforms);
        test_error(err, "clGetPlatformIDs failed");

        std::vector<cl_platform_id> platforms(num_platforms);

        err = clGetPlatformIDs(num_platforms, platforms.data(), &num_platforms);
        test_error(err, "clGetPlatformIDs failed");

        cl_device_id device_sec = nullptr;
        cl_uint num_devices = 0;
        for (int p = 0; p < (int)num_platforms; p++)
        {
            if (platform_id == platforms[p]) continue;

            err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr,
                                 &num_devices);
            test_error(err, "clGetDeviceIDs failed");

            std::vector<cl_device_id> devices(num_devices);
            err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices,
                                 devices.data(), nullptr);
            test_error(err, "clGetDeviceIDs failed");

            device_sec = devices.front();
            break;
        }

        if (device_sec == nullptr)
        {
            log_info("Can't find needed resources. Skipping the test.\n");
            return TEST_SKIPPED_ITSELF;
        }

        // Create secondary context
        clContextWrapper context_sec =
            clCreateContext(0, 1, &device_sec, nullptr, nullptr, &err);
        test_error(err, "Failed to create context");

        // Create secondary queue
        clCommandQueueWrapper queue_sec =
            clCreateCommandQueue(context_sec, device_sec, 0, &err);
        test_error(err, "Could not create command queue");

        // Signal semaphore
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, nullptr);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue_sec, 1, semaphore, nullptr, 0,
                                         nullptr, nullptr);
        test_failure_error(err, CL_INVALID_COMMAND_QUEUE,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return TEST_PASS;
    }
};


// num_sema_objects is 0.

struct WaitInvalidValue : public SemaphoreTestBase
{
    WaitInvalidValue(cl_device_id device, cl_context context,
                     cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Wait semaphore
        cl_int err = CL_SUCCESS;
        err = clEnqueueWaitSemaphoresKHR(queue, 0, semaphore, nullptr, 0,
                                         nullptr, nullptr);
        test_failure_error(err, CL_INVALID_VALUE,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return CL_SUCCESS;
    }
};

// any of the semaphore objects specified by sema_objects is not valid.

struct WaitInvalidSemaphore : public SemaphoreTestBase
{
    WaitInvalidSemaphore(cl_device_id device, cl_context context,
                         cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Wait semaphore
        cl_semaphore_khr sema_objects[] = { nullptr, nullptr, nullptr };
        cl_int err = CL_SUCCESS;
        err = clEnqueueWaitSemaphoresKHR(
            queue, sizeof(sema_objects) / sizeof(sema_objects[0]), sema_objects,
            nullptr, 0, nullptr, nullptr);
        test_failure_error(err, CL_INVALID_SEMAPHORE_KHR,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return CL_SUCCESS;
    }
};

// 1) the context associated with command_queue and any of the semaphore objects
// in sema_objects are not the same, or
// 2) the context associated with command_queue and that associated with events
// in event_wait_list are not the same.

struct WaitInvalidContext : public SemaphoreTestBase
{
    WaitInvalidContext(cl_device_id device, cl_context context,
                       cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };

        cl_int err = CL_SUCCESS;
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Create secondary context
        clContextWrapper context_sec =
            clCreateContext(0, 1, &device, nullptr, nullptr, &err);
        test_error(err, "Failed to create context");

        // Create secondary queue
        clCommandQueueWrapper queue_sec =
            clCreateCommandQueue(context_sec, device, 0, &err);
        test_error(err, "Could not create command queue");

        // Signal semaphore
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, nullptr);
        test_error(err, "Could not signal semaphore");

        // (1) Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue_sec, 1, semaphore, nullptr, 0,
                                         nullptr, nullptr);
        test_failure_error(err, CL_INVALID_CONTEXT,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context_sec, &err);
        test_error(err, "Could not create user event");

        // (2) Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                         &user_event, nullptr);

        cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(signal_error, "clSetUserEventStatus failed");

        test_failure_error(err, CL_INVALID_CONTEXT,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return TEST_PASS;
    }
};

// (1) event_wait_list is NULL and num_events_in_wait_list is not 0, or
// (2) event_wait_list is not NULL and num_events_in_wait_list is 0, or
// (3) event objects in event_wait_list are not valid events.

struct WaitInvalidEventWaitList : public SemaphoreTestBase
{
    WaitInvalidEventWaitList(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };

        cl_int err = CL_SUCCESS;
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");


        // Signal semaphore
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, nullptr);
        test_error(err, "Could not signal semaphore");

        // (1) Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                         nullptr, nullptr);
        test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context, &err);
        test_error(err, "Could not create user event");

        // (2) Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                         &user_event, nullptr);

        cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(signal_error, "clSetUserEventStatus failed");

        test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        // (3) Wait semaphore
        cl_event wait_list[] = { nullptr, nullptr, nullptr };
        err = clEnqueueWaitSemaphoresKHR(
            queue, 1, semaphore, nullptr,
            sizeof(wait_list) / sizeof(wait_list[0]), wait_list, nullptr);
        test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return CL_SUCCESS;
    }
};

// the execution status of any of the events in event_wait_list is a negative
// integer value.

struct WaitInvalidEventStatus : public SemaphoreTestBase
{
    WaitInvalidEventStatus(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };

        cl_int err = CL_SUCCESS;
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Signal semaphore
        err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                           nullptr, nullptr);
        test_error(err, "Could not signal semaphore");

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context, &err);
        test_error(err, "Could not create user event");

        // Now release the user event, which will allow our actual action to run
        err = clSetUserEventStatus(user_event, -1);
        test_error(err, "Unable to set event status");

        // Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                         &user_event, nullptr);
        test_failure_error(err, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
                           "Unexpected clEnqueueWaitSemaphoresKHR return");

        return CL_SUCCESS;
    }
};

}

int test_semaphores_negative_wait_invalid_command_queue(cl_device_id device,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements)
{
    return MakeAndRunTest<WaitInvalidCommandQueue>(device, context, queue);
}

int test_semaphores_negative_wait_invalid_value(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<WaitInvalidValue>(device, context, queue);
}

int test_semaphores_negative_wait_invalid_semaphore(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<WaitInvalidSemaphore>(device, context, queue);
}

int test_semaphores_negative_wait_invalid_context(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements)
{
    return MakeAndRunTest<WaitInvalidContext>(device, context, queue);
}

int test_semaphores_negative_wait_invalid_event_wait_list(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<WaitInvalidEventWaitList>(device, context, queue);
}

int test_semaphores_negative_wait_invalid_event_status(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    return MakeAndRunTest<WaitInvalidEventStatus>(device, context, queue);
}
