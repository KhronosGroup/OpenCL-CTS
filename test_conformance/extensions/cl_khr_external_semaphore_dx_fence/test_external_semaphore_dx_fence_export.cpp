//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "semaphore_dx_fence_base.h"

struct ExportDXSignal final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    int Run() override
    {
        size_t export_types_size = 0;
        errcode =
            clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                            0, nullptr, &export_types_size);
        test_error(errcode, "Could not query export semaphore handle types");
        std::vector<cl_external_semaphore_handle_type_khr> export_types(
            export_types_size / sizeof(cl_external_semaphore_handle_type_khr));
        errcode =
            clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                            export_types_size, export_types.data(), nullptr);
        test_error(errcode, "Could not query export semaphore handle types");

        if (std::find(export_types.begin(), export_types.end(),
                      CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR)
            == export_types.end())
        {
            log_info(
                "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between "
                "the supported export types\n");
            return TEST_FAIL;
        }

        constexpr cl_semaphore_properties_khr sem_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR),
            0
        };
        cl_semaphore_khr exportable_semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sem_props, &errcode);
        test_error(errcode, "Could not create semaphore");

        cl_bool is_exportable = CL_FALSE;
        errcode = clGetSemaphoreInfoKHR(
            exportable_semaphore, CL_SEMAPHORE_EXPORTABLE_KHR,
            sizeof(is_exportable), &is_exportable, nullptr);
        test_error(errcode, "Could not get semaphore info");
        test_error(!is_exportable, "Semaphore is not exportable");

        log_info("Calling clEnqueueWaitSemaphoresKHR\n");
        clEventWrapper wait_event;
        errcode = clEnqueueWaitSemaphoresKHR(queue, 1, &exportable_semaphore,
                                             &semaphore_payload, 0, nullptr,
                                             &wait_event);
        test_error(errcode, "Failed to wait semaphore");

        HANDLE semaphore_handle = nullptr;
        errcode = clGetSemaphoreHandleForTypeKHR(
            exportable_semaphore, device, CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR,
            sizeof(semaphore_handle), &semaphore_handle, nullptr);
        test_error(errcode, "Could not get semaphore handle");

        ID3D12Fence *exported_fence = nullptr;
        errcode = dx_wrapper.getDXDevice()->OpenSharedHandle(
            semaphore_handle, IID_PPV_ARGS(&exported_fence));
        test_error(errcode, "Could not open semaphore handle");

        log_info("Calling fence->Signal()\n");
        const HRESULT hr = exported_fence->Signal(semaphore_payload);
        test_error(FAILED(hr), "Failed to signal D3D12 fence");

        errcode = clFinish(queue);
        test_error(errcode, "Could not finish queue");

        test_assert_event_complete(wait_event);

        // Release resources
        CloseHandle(semaphore_handle);
        test_error(clReleaseSemaphoreKHR(exportable_semaphore),
                   "Could not release semaphore");
        exported_fence->Release();

        return TEST_PASS;
    }
};

// Confirm that a wait followed by a signal in DirectX 12 using an exported
// semaphore will complete successfully
REGISTER_TEST(test_external_semaphores_export_dx_signal)
{
    return MakeAndRunTest<ExportDXSignal>(device, context, queue, num_elements);
}

struct ExportDXWait final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    int Run() override
    {
        size_t export_types_size = 0;
        errcode =
            clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                            0, nullptr, &export_types_size);
        test_error(errcode, "Could not query export semaphore handle types");
        std::vector<cl_external_semaphore_handle_type_khr> export_types(
            export_types_size / sizeof(cl_external_semaphore_handle_type_khr));
        errcode =
            clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                            export_types_size, export_types.data(), nullptr);
        test_error(errcode, "Could not query export semaphore handle types");

        if (std::find(export_types.begin(), export_types.end(),
                      CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR)
            == export_types.end())
        {
            log_info(
                "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between "
                "the supported export types\n");
            return TEST_FAIL;
        }

        constexpr cl_semaphore_properties_khr sem_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR),
            0
        };
        cl_semaphore_khr exportable_semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sem_props, &errcode);
        test_error(errcode, "Could not create semaphore");

        cl_bool is_exportable = CL_FALSE;
        errcode = clGetSemaphoreInfoKHR(
            exportable_semaphore, CL_SEMAPHORE_EXPORTABLE_KHR,
            sizeof(is_exportable), &is_exportable, nullptr);
        test_error(errcode, "Could not get semaphore info");
        test_error(!is_exportable, "Semaphore is not exportable");

        log_info("Calling clEnqueueSignalSemaphoresKHR\n");
        constexpr cl_semaphore_payload_khr semaphore_payload = 1;
        clEventWrapper signal_event;
        errcode = clEnqueueSignalSemaphoresKHR(queue, 1, &exportable_semaphore,
                                               &semaphore_payload, 0, nullptr,
                                               &signal_event);
        test_error(errcode, "Failed to signal semaphore");

        HANDLE semaphore_handle = nullptr;
        errcode = clGetSemaphoreHandleForTypeKHR(
            exportable_semaphore, device, CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR,
            sizeof(semaphore_handle), &semaphore_handle, nullptr);
        test_error(errcode, "Could not get semaphore handle");

        ID3D12Fence *exported_fence = nullptr;
        errcode = dx_wrapper.getDXDevice()->OpenSharedHandle(
            semaphore_handle, IID_PPV_ARGS(&exported_fence));
        test_error(errcode, "Could not open semaphore handle");

        log_info("Calling dx_wrapper.get_d3d12_command_queue()->Wait()\n");
        HRESULT hr = dx_wrapper.getDXCommandQueue()->Wait(exported_fence,
                                                          semaphore_payload);
        test_error(FAILED(hr), "Failed to wait on D3D12 fence");

        log_info("Calling WaitForSingleObject\n");
        if (exported_fence->GetCompletedValue() < semaphore_payload)
        {
            const HANDLE event =
                CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            hr = exported_fence->SetEventOnCompletion(semaphore_payload, event);
            test_error(FAILED(hr), "Failed to set event on completion");
            WaitForSingleObject(event, INFINITE);
            CloseHandle(event);
        }

        errcode = clFinish(queue);
        test_error(errcode, "Could not finish queue");

        test_assert_event_complete(signal_event);

        // Release resources
        CloseHandle(semaphore_handle);
        test_error(clReleaseSemaphoreKHR(exportable_semaphore),
                   "Could not release semaphore");
        exported_fence->Release();

        return TEST_PASS;
    }
};

// Confirm that a signal in OpenCL followed by a wait in DirectX 12 using an
// exported semaphore will complete successfully
REGISTER_TEST(test_external_semaphores_export_dx_wait)
{
    return MakeAndRunTest<ExportDXWait>(device, context, queue, num_elements);
}