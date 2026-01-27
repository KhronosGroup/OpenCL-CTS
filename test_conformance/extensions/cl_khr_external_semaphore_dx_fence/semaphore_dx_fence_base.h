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

struct DXFenceTestBase
{
    DXFenceTestBase(cl_device_id device, cl_context context,
                    cl_command_queue queue, cl_int num_elems)
        : device(device), context(context), queue(queue), num_elems(num_elems)
    {}
    virtual ~DXFenceTestBase()
    {
        if (fence_handle)
        {
            CloseHandle(fence_handle);
            fence_handle = nullptr;
        }
        if (fence_wrapper)
        {
            delete fence_wrapper;
            fence_wrapper = nullptr;
        }
        if (semaphore)
        {
            clReleaseSemaphoreKHR(semaphore);
            semaphore = nullptr;
        }
    };

    virtual int SetUp()
    {
        REQUIRE_EXTENSION("cl_khr_external_semaphore");
        REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

        // Obtain pointers to semaphore's API
        GET_FUNCTION_EXTENSION_ADDRESS(device,
                                       clCreateSemaphoreWithPropertiesKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clReleaseSemaphoreKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clEnqueueSignalSemaphoresKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clEnqueueWaitSemaphoresKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clGetSemaphoreHandleForTypeKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clRetainSemaphoreKHR);
        GET_FUNCTION_EXTENSION_ADDRESS(device, clGetSemaphoreInfoKHR);

        test_error(
            !is_import_handle_available(CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
            "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
            "supported import types");

        // Import D3D12 fence into OpenCL
        fence_wrapper = new DirectXFenceWrapper(dx_wrapper.getDXDevice());
        semaphore = createSemaphoreFromFence(fence_wrapper->get());
        test_assert_error(!!semaphore, "Could not create semaphore");

        return TEST_PASS;
    }

    virtual cl_int Run() = 0;

protected:
    int errcode = CL_SUCCESS;

    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_int num_elems = 0;
    DirectXWrapper dx_wrapper;

    cl_semaphore_payload_khr semaphore_payload = 1;
    cl_semaphore_khr semaphore = nullptr;
    HANDLE fence_handle = nullptr;
    DirectXFenceWrapper *fence_wrapper = nullptr;

    clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR =
        nullptr;
    clEnqueueSignalSemaphoresKHR_fn clEnqueueSignalSemaphoresKHR = nullptr;
    clEnqueueWaitSemaphoresKHR_fn clEnqueueWaitSemaphoresKHR = nullptr;
    clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR = nullptr;
    clGetSemaphoreInfoKHR_fn clGetSemaphoreInfoKHR = nullptr;
    clRetainSemaphoreKHR_fn clRetainSemaphoreKHR = nullptr;
    clGetSemaphoreHandleForTypeKHR_fn clGetSemaphoreHandleForTypeKHR = nullptr;

    [[nodiscard]] bool is_import_handle_available(
        const cl_external_memory_handle_type_khr handle_type)
    {
        size_t import_types_size = 0;
        errcode =
            clGetDeviceInfo(device, CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,
                            0, nullptr, &import_types_size);
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

    cl_semaphore_khr createSemaphoreFromFence(ID3D12Fence *src_fence)
    {
        const HRESULT hr = dx_wrapper.getDXDevice()->CreateSharedHandle(
            src_fence, nullptr, GENERIC_ALL, nullptr, &fence_handle);
        if (FAILED(hr)) return nullptr;

        const cl_semaphore_properties_khr sem_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
            reinterpret_cast<cl_semaphore_properties_khr>(fence_handle), 0
        };
        cl_semaphore_khr tmp_semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sem_props, &errcode);
        if (errcode != CL_SUCCESS) return nullptr;

        return tmp_semaphore;
    }
};

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
{
    cl_int status = TEST_PASS;
    try
    {
        auto test_fixture = T(device, context, queue, nelems);
        status = test_fixture.SetUp();
        if (status != TEST_PASS) return status;
        status = test_fixture.Run();
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return status;
}