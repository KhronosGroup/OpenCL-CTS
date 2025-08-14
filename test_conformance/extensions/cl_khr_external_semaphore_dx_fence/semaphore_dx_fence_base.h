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

#pragma once

#include "harness/typeWrappers.h"
#include "harness/extensionHelpers.h"
#include "harness/errorHelpers.h"
#include "directx_wrapper.hpp"

class CLDXSemaphoreWrapper {
public:
    CLDXSemaphoreWrapper(cl_device_id device, cl_context context,
                         ID3D12Device* dx_device)
        : device(device), context(context), dx_device(dx_device){};

    int createSemaphoreFromFence(ID3D12Fence* fence)
    {
        cl_int errcode = CL_SUCCESS;

        GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);

        const HRESULT hr = dx_device->CreateSharedHandle(
            fence, nullptr, GENERIC_ALL, nullptr, &fence_handle);
        test_error(FAILED(hr), "Failed to get shared handle from D3D12 fence");

        cl_semaphore_properties_khr sem_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
            reinterpret_cast<cl_semaphore_properties_khr>(fence_handle), 0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sem_props, &errcode);
        test_error(errcode, "Could not create semaphore");

        return CL_SUCCESS;
    }

    ~CLDXSemaphoreWrapper()
    {
        releaseSemaphore();
        if (fence_handle)
        {
            CloseHandle(fence_handle);
        }
    };

    const cl_semaphore_khr* operator&() const { return &semaphore; };
    cl_semaphore_khr operator*() const { return semaphore; };

    HANDLE getHandle() const { return fence_handle; };

private:
    cl_semaphore_khr semaphore;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_handle;
    cl_device_id device;
    cl_context context;
    ComPtr<ID3D12Device> dx_device;

    int releaseSemaphore() const
    {
        GET_PFN(device, clReleaseSemaphoreKHR);

        if (semaphore)
        {
            clReleaseSemaphoreKHR(semaphore);
        }

        return CL_SUCCESS;
    }
};

static bool
is_import_handle_available(cl_device_id device,
                           const cl_external_memory_handle_type_khr handle_type)
{
    int errcode = CL_SUCCESS;
    size_t import_types_size = 0;
    errcode =
        clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR, 0,
                        nullptr, &import_types_size);
    if (errcode != CL_SUCCESS)
    {
        log_error("Could not query import semaphore handle types");
        return false;
    }
    std::vector<cl_external_semaphore_handle_type_khr> import_types(
        import_types_size / sizeof(cl_external_semaphore_handle_type_khr));
    errcode =
        clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,
                        import_types_size, import_types.data(), nullptr);
    if (errcode != CL_SUCCESS)
    {
        log_error("Could not query import semaphore handle types");
        return false;
    }

    return std::find(import_types.begin(), import_types.end(), handle_type)
        != import_types.end();
}