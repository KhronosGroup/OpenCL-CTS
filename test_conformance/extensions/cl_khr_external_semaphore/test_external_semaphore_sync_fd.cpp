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

#include "harness/typeWrappers.h"
#include "harness/extensionHelpers.h"
#include "harness/errorHelpers.h"

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

// Test it is possible to export a semaphore to a sync fd and import the same
// sync fd to a new semaphore
REGISTER_TEST_VERSION(external_semaphores_import_export_fd, Version(1, 2))
{
    cl_int err = CL_SUCCESS;

    if (!is_extension_available(device, "cl_khr_external_semaphore"))
    {
        log_info(
            "cl_khr_external_semaphore is not supported on this platoform. "
            "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    if (!is_extension_available(device, "cl_khr_external_semaphore_sync_fd"))
    {
        log_info("cl_khr_external_semaphore_sync_fd is not supported on this "
                 "platoform. Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_command_queue_properties device_props = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                          sizeof(device_props), &device_props, NULL);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

    if ((device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0)
    {
        log_info("Queue property CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE not "
                 "supported. Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);
    GET_PFN(device, clGetSemaphoreHandleForTypeKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);

    // Create ooo queue
    clCommandQueueWrapper test_queue = clCreateCommandQueue(
        context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    test_error(err, "Could not create command queue");

    CREATE_KERNEL;
    CREATE_BUFFER;

    // Create semaphore
    cl_semaphore_properties_khr sema_1_props[] = {
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_BINARY_KHR),
        static_cast<cl_semaphore_properties_khr>(
            CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
        static_cast<cl_semaphore_properties_khr>(
            CL_SEMAPHORE_HANDLE_SYNC_FD_KHR),
        static_cast<cl_semaphore_properties_khr>(
            CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR),
        0
    };
    cl_semaphore_khr sema_1 =
        clCreateSemaphoreWithPropertiesKHR(context, sema_1_props, &err);
    test_error(err, "Could not create semaphore");

    ENQUEUE_KERNEL_WITH_EVENT(test_queue, 0, nullptr, write_int_event);
    
    // Signal semaphore
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(test_queue, 1, &sema_1, nullptr, 1,
                                       &write_int_event, &signal_event);
    test_error(err, "Could not signal semaphore");

    // Extract sync fd
    int handle = -1;
    size_t handle_size;
    err = clGetSemaphoreHandleForTypeKHR(sema_1, device,
                                         CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                                         sizeof(handle), &handle, &handle_size);
    test_error(err, "Could not extract semaphore handle");
    test_assert_error(sizeof(handle) == handle_size, "Invalid handle size");
    test_assert_error(handle >= 0, "Invalid handle");

    // Create semaphore from sync fd
    cl_semaphore_properties_khr sema_2_props[] = {
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_BINARY_KHR),
        CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
        static_cast<cl_semaphore_properties_khr>(handle), 0
    };

    cl_semaphore_khr sema_2 =
        clCreateSemaphoreWithPropertiesKHR(context, sema_2_props, &err);
    test_error(err, "Could not create semaphore");

    // Wait semaphore
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(test_queue, 1, &sema_2, nullptr, 0,
                                     nullptr, &wait_event);
    test_error(err, "Could not wait semaphore");

    ENQUEUE_KERNEL(test_queue, 1, &wait_event);

    // Finish
    err = clFinish(test_queue);
    test_error(err, "Could not finish queue");

    // Check all events are completed
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    // Release semaphore
    err = clReleaseSemaphoreKHR(sema_1);
    test_error(err, "Could not release semaphore");

    err = clReleaseSemaphoreKHR(sema_2);
    test_error(err, "Could not release semaphore");
    return TEST_PASS;
}
