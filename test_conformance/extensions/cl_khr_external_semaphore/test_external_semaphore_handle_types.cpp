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

cl_int doTest(cl_device_id device, cl_context context, cl_command_queue queue,
              const cl_semaphore_properties_khr& handle_type)
{
    cl_int err = CL_SUCCESS;

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);
    GET_PFN(device, clGetSemaphoreHandleForTypeKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);

    // Create semaphore
    cl_semaphore_properties_khr sema_1_props[] = {
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_BINARY_KHR),
        static_cast<cl_semaphore_properties_khr>(
            CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
        static_cast<cl_semaphore_properties_khr>(handle_type),
        static_cast<cl_semaphore_properties_khr>(
            CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR),
        0
    };
    cl_semaphore_khr sema_1 =
        clCreateSemaphoreWithPropertiesKHR(context, sema_1_props, &err);
    test_error(err, "Could not create semaphore");

    // Signal semaphore
    clEventWrapper signal_event;
    err = clEnqueueSignalSemaphoresKHR(queue, 1, &sema_1, nullptr, 0, nullptr,
                                       &signal_event);
    test_error(err, "Could not signal semaphore");

#ifdef _WIN32
    HANDLE handle;
#else
    // Extract sync fd
    int handle = -1;
#endif
    size_t handle_size;
    err = clGetSemaphoreHandleForTypeKHR(
        sema_1, device, handle_type, sizeof(handle),
        reinterpret_cast<cl_semaphore_properties_khr*>(&handle), &handle_size);
    test_error(err, "Could not extract semaphore handle");
    test_assert_error(sizeof(handle) == handle_size, "Invalid handle size");
    test_assert_error(handle >= 0, "Invalid handle");

    // Create semaphore from sync fd
    cl_semaphore_properties_khr sema_2_props[] = {
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
        static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_BINARY_KHR),
        handle_type, static_cast<cl_semaphore_properties_khr>(handle), 0
    };

    cl_semaphore_khr sema_2 =
        clCreateSemaphoreWithPropertiesKHR(context, sema_2_props, &err);
    test_error(err, "Could not create semaphore");

    // Wait semaphore
    clEventWrapper wait_event;
    err = clEnqueueWaitSemaphoresKHR(queue, 1, &sema_2, nullptr, 0, nullptr,
                                     &wait_event);
    test_error(err, "Could not wait semaphore");

    // Finish
    err = clFinish(queue);
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

// Test it is possible to export a semaphore and import the same to a new
// semaphore
REGISTER_TEST_VERSION(external_semaphores_import_export_fd, Version(1, 2))
{
    REQUIRE_EXTENSION("cl_khr_external_semaphore");

    std::vector<cl_semaphore_properties_khr> handle_types = {
        CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR,
#ifdef _WIN32
        CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR,
        CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR
#endif
    };

    cl_int err = CL_SUCCESS;
    cl_int total_status = TEST_PASS;

    for (const auto& handle_type : handle_types)
    {
        if (handle_type == CL_SEMAPHORE_HANDLE_SYNC_FD_KHR)
            REQUIRE_EXTENSION("cl_khr_external_semaphore_sync_fd");
        else if (handle_type == CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR)
            REQUIRE_EXTENSION("cl_khr_external_semaphore_opaque_fd");
#ifdef _WIN32
        else if (handle_type == CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR)
            REQUIRE_EXTENSION("cl_khr_external_semaphore_win32");
        else if (handle_type == CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR)
            REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");
#endif
        else
            continue;

        // test external semaphore with out-of-order queue
        {
            cl_command_queue_properties device_props = 0;
            err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                                  sizeof(device_props), &device_props, NULL);
            test_error(err,
                       "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

            if ((device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0)
            {
                // Create ooo queue
                clCommandQueueWrapper test_queue = clCreateCommandQueue(
                    context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                    &err);
                test_error(err, "Could not create command queue");

                cl_int status =
                    doTest(device, context, test_queue, handle_type);
                if (status != TEST_PASS && status != TEST_SKIPPED_ITSELF)
                {
                    total_status = TEST_FAIL;
                }
            }
        }

        // test external semaphore with in-order harness queue
        {
            cl_int status = doTest(device, context, queue, handle_type);
            if (status != TEST_PASS && status != TEST_SKIPPED_ITSELF)
            {
                total_status = TEST_FAIL;
            }
        }
    }

    return total_status;
}
