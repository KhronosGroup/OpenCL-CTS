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

enum class RunMode
{
    RM_WAIT = 0,
    RM_SIGNAL
};

// scope guard helper to ensure proper releasing of sub devices
struct SubDevicesScopeGuarded
{
    SubDevicesScopeGuarded(const cl_int dev_count)
    {
        sub_devices.resize(dev_count);
    }
    ~SubDevicesScopeGuarded()
    {
        for (auto& device : sub_devices)
        {
            cl_int err = clReleaseDevice(device);
            if (err != CL_SUCCESS)
                log_error("\n Releasing sub-device failed \n");
        }
    }

    std::vector<cl_device_id> sub_devices;
};

// the device associated with command_queue is not same as one of the devices
// specified by CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR at the time of creating one
// or more of sema_objects.

template <RunMode mode> struct InvalidCommandQueue : public SemaphoreTestBase
{
    InvalidCommandQueue(cl_device_id device, cl_context context,
                        cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;

        // Below test makes sense only if semaphore and command queue share the
        // same context, otherwise CL_INVALID_CONTEXT could be the result. Thus,
        // multi device context must be created, then semaphore and command
        // queue with the same associated context but different devices.

        // partition device and create new context if possible
        cl_uint maxComputeUnits = 0;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                              sizeof(maxComputeUnits), &maxComputeUnits, NULL);
        test_error(err, "Unable to get maximal number of compute units");

        cl_uint maxSubDevices = 0;
        err = clGetDeviceInfo(device, CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                              sizeof(maxSubDevices), &maxSubDevices, NULL);
        test_error(err, "Unable to get maximal number of sub-devices");

        if (maxSubDevices < 2)
        {
            log_info("Can't partition device, test not supported\n");
            return TEST_SKIPPED_ITSELF;
        }

        cl_device_partition_property partitionProp[] = {
            CL_DEVICE_PARTITION_EQUALLY,
            static_cast<cl_device_partition_property>(maxComputeUnits / 2), 0
        };

        cl_uint deviceCount = 0;
        // how many sub-devices can we create?
        err =
            clCreateSubDevices(device, partitionProp, 0, nullptr, &deviceCount);
        if (err != CL_SUCCESS)
        {
            log_info("Can't partition device, test not supported\n");
            return TEST_SKIPPED_ITSELF;
        }

        if (deviceCount < 2)
            test_error_ret(
                CL_INVALID_VALUE,
                "Multi context test for CL_INVALID_COMMAND_QUEUE not supported",
                TEST_SKIPPED_ITSELF);

        // get the list of subDevices
        SubDevicesScopeGuarded scope_guard(deviceCount);
        err = clCreateSubDevices(device, partitionProp, deviceCount,
                                 scope_guard.sub_devices.data(), &deviceCount);
        if (err != CL_SUCCESS)
        {
            log_info("Can't partition device, test not supported\n");
            return TEST_SKIPPED_ITSELF;
        }

        /* Create a multi device context */
        clContextWrapper multi_device_context = clCreateContext(
            NULL, (cl_uint)deviceCount, scope_guard.sub_devices.data(), nullptr,
            nullptr, &err);
        test_error_ret(err, "Unable to create testing context", CL_SUCCESS);

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR),
            (cl_semaphore_properties_khr)scope_guard.sub_devices.front(),
            CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
            0
        };

        semaphore = clCreateSemaphoreWithPropertiesKHR(multi_device_context,
                                                       sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Create secondary queue associated with device not the same as one
        // associated with semaphore
        clCommandQueueWrapper queue_sec = clCreateCommandQueue(
            multi_device_context, scope_guard.sub_devices.back(), 0, &err);
        test_error(err, "Could not create command queue");

        if (mode == RunMode::RM_SIGNAL)
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue_sec, 1, semaphore, nullptr,
                                               0, nullptr, nullptr);
            test_failure_error(
                err, CL_INVALID_COMMAND_QUEUE,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                               nullptr, nullptr);
            test_error(err, "Could not signal semaphore");

            // Wait semaphore
            err = clEnqueueWaitSemaphoresKHR(queue_sec, 1, semaphore, nullptr,
                                             0, nullptr, nullptr);
            test_failure_error(err, CL_INVALID_COMMAND_QUEUE,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return TEST_PASS;
    }
};


// num_sema_objects is 0.

template <RunMode mode> struct InvalidValue : public SemaphoreTestBase
{
    InvalidValue(cl_device_id device, cl_context context,
                 cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        if (mode == RunMode::RM_SIGNAL)
        {
            // Signal semaphore
            cl_int err = CL_SUCCESS;
            err = clEnqueueSignalSemaphoresKHR(queue, 0, semaphore, nullptr, 0,
                                               nullptr, nullptr);
            test_failure_error(
                err, CL_INVALID_VALUE,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Wait semaphore
            cl_int err = CL_SUCCESS;
            err = clEnqueueWaitSemaphoresKHR(queue, 0, semaphore, nullptr, 0,
                                             nullptr, nullptr);
            test_failure_error(err, CL_INVALID_VALUE,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return CL_SUCCESS;
    }
};

// any of the semaphore objects specified by sema_objects is not valid.

template <RunMode mode> struct InvalidSemaphore : public SemaphoreTestBase
{
    InvalidSemaphore(cl_device_id device, cl_context context,
                     cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_semaphore_khr sema_objects[] = { nullptr, nullptr, nullptr };
        cl_int err = CL_SUCCESS;

        if (mode == RunMode::RM_SIGNAL)
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(
                queue, sizeof(sema_objects) / sizeof(sema_objects[0]),
                sema_objects, nullptr, 0, nullptr, nullptr);
            test_failure_error(
                err, CL_INVALID_SEMAPHORE_KHR,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Wait semaphore
            err = clEnqueueWaitSemaphoresKHR(
                queue, sizeof(sema_objects) / sizeof(sema_objects[0]),
                sema_objects, nullptr, 0, nullptr, nullptr);
            test_failure_error(err, CL_INVALID_SEMAPHORE_KHR,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return CL_SUCCESS;
    }
};

// (1) the context associated with command_queue and any of the semaphore
// objects in sema_objects are not the same, or (2) the context associated with
// command_queue and that associated with events in event_wait_list are not the
// same.

template <RunMode mode> struct InvalidContext : public SemaphoreTestBase
{
    InvalidContext(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
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

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context_sec, &err);
        test_error(err, "Could not create user event");

        if (mode == RunMode::RM_SIGNAL)
        {
            // (1)
            err = clEnqueueSignalSemaphoresKHR(queue_sec, 1, semaphore, nullptr,
                                               0, nullptr, nullptr);
            test_failure_error(
                err, CL_INVALID_CONTEXT,
                "Unexpected clEnqueueSignalSemaphoresKHR return");

            // (2)
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                               &user_event, nullptr);

            cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
            test_error(signal_error, "clSetUserEventStatus failed");

            test_failure_error(
                err, CL_INVALID_CONTEXT,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                               nullptr, nullptr);
            test_error(err, "Could not signal semaphore");

            // (1)
            err = clEnqueueWaitSemaphoresKHR(queue_sec, 1, semaphore, nullptr,
                                             0, nullptr, nullptr);
            test_failure_error(err, CL_INVALID_CONTEXT,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");

            // (2)
            err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                             &user_event, nullptr);

            cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
            test_error(signal_error, "clSetUserEventStatus failed");

            test_failure_error(err, CL_INVALID_CONTEXT,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return TEST_PASS;
    }
};

// (1) event_wait_list is NULL and num_events_in_wait_list is not 0, or
// (2) event_wait_list is not NULL and num_events_in_wait_list is 0, or
// (3) event objects in event_wait_list are not valid events.

template <RunMode mode> struct InvalidEventWaitList : public SemaphoreTestBase
{
    InvalidEventWaitList(cl_device_id device, cl_context context,
                         cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
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

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context, &err);
        test_error(err, "Could not create user event");

        cl_event wait_list[] = { nullptr, nullptr, nullptr };

        if (mode == RunMode::RM_SIGNAL)
        {
            // (1)
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                               nullptr, nullptr);
            test_failure_error(
                err, CL_INVALID_EVENT_WAIT_LIST,
                "Unexpected clEnqueueSignalSemaphoresKHR return");

            // (2)
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                               &user_event, nullptr);

            cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
            test_error(signal_error, "clSetUserEventStatus failed");

            test_failure_error(
                err, CL_INVALID_EVENT_WAIT_LIST,
                "Unexpected clEnqueueSignalSemaphoresKHR return");

            // (3)
            err = clEnqueueSignalSemaphoresKHR(
                queue, 1, semaphore, nullptr,
                sizeof(wait_list) / sizeof(wait_list[0]), wait_list, nullptr);
            test_failure_error(
                err, CL_INVALID_EVENT_WAIT_LIST,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                               nullptr, nullptr);
            test_error(err, "Could not signal semaphore");

            // (1)
            err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                             nullptr, nullptr);
            test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");

            // (2)
            err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                             &user_event, nullptr);

            cl_int signal_error = clSetUserEventStatus(user_event, CL_COMPLETE);
            test_error(signal_error, "clSetUserEventStatus failed");

            test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");

            // (3)
            err = clEnqueueWaitSemaphoresKHR(
                queue, 1, semaphore, nullptr,
                sizeof(wait_list) / sizeof(wait_list[0]), wait_list, nullptr);
            test_failure_error(err, CL_INVALID_EVENT_WAIT_LIST,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return CL_SUCCESS;
    }
};

// the execution status of any of the events in event_wait_list is a negative
// integer value.

template <RunMode mode> struct InvalidEventStatus : public SemaphoreTestBase
{
    InvalidEventStatus(cl_device_id device, cl_context context,
                       cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
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

        // Create user event
        clEventWrapper user_event = clCreateUserEvent(context, &err);
        test_error(err, "Could not create user event");

        // set the negative integer value status of the event in event_wait_list
        err = clSetUserEventStatus(user_event, -1);
        test_error(err, "Unable to set event status");

        if (mode == RunMode::RM_SIGNAL)
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                               &user_event, nullptr);
            test_failure_error(
                err, CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
                "Unexpected clEnqueueSignalSemaphoresKHR return");
        }
        else
        {
            // Signal semaphore
            err = clEnqueueSignalSemaphoresKHR(queue, 1, semaphore, nullptr, 0,
                                               nullptr, nullptr);
            test_error(err, "Could not signal semaphore");

            // Wait semaphore
            err = clEnqueueWaitSemaphoresKHR(queue, 1, semaphore, nullptr, 1,
                                             &user_event, nullptr);
            test_failure_error(err,
                               CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
                               "Unexpected clEnqueueWaitSemaphoresKHR return");
        }

        return CL_SUCCESS;
    }
};

}

int test_semaphores_negative_wait_invalid_command_queue(cl_device_id device,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements)
{
    return MakeAndRunTest<InvalidCommandQueue<RunMode::RM_WAIT>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_wait_invalid_value(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<InvalidValue<RunMode::RM_WAIT>>(device, context,
                                                          queue, num_elements);
}

int test_semaphores_negative_wait_invalid_semaphore(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<InvalidSemaphore<RunMode::RM_WAIT>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_wait_invalid_context(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements)
{
    return MakeAndRunTest<InvalidContext<RunMode::RM_WAIT>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_wait_invalid_event_wait_list(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<InvalidEventWaitList<RunMode::RM_WAIT>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_wait_invalid_event_status(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    return MakeAndRunTest<InvalidEventStatus<RunMode::RM_WAIT>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_command_queue(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<InvalidCommandQueue<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_value(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements)
{
    return MakeAndRunTest<InvalidValue<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_semaphore(cl_device_id device,
                                                      cl_context context,
                                                      cl_command_queue queue,
                                                      int num_elements)
{
    return MakeAndRunTest<InvalidSemaphore<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_context(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<InvalidContext<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_event_wait_list(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<InvalidEventWaitList<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}

int test_semaphores_negative_signal_invalid_event_status(cl_device_id device,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements)
{
    return MakeAndRunTest<InvalidEventStatus<RunMode::RM_SIGNAL>>(
        device, context, queue, num_elements);
}
