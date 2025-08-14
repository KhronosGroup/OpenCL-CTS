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

// Confirm that a signal followed by a wait in OpenCL will complete successfully
REGISTER_TEST(test_external_semaphores_signal_wait)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);

    test_error(!is_import_handle_available(device,
                                           CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
               "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
               "supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore.createSemaphoreFromFence(*fence),
               "Could not create semaphore");

    log_info("Calling clEnqueueSignalSemaphoresKHR\n");
    constexpr cl_semaphore_payload_khr semaphore_payload = 1;
    clEventWrapper signal_event;
    errcode = clEnqueueSignalSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &signal_event);
    test_error(errcode, "Failed to signal semaphore");

    log_info("Calling clEnqueueWaitSemaphoresKHR\n");
    clEventWrapper wait_event;
    errcode = clEnqueueWaitSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
    test_error(errcode, "Failed to wait semaphore");

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    // Verify that the events completed.
    test_assert_event_complete(signal_event);
    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that a wait in OpenCL followed by a CPU signal in DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_signal_dx_cpu)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);

    test_error(!is_import_handle_available(device,
                                           CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
               "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
               "supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore.createSemaphoreFromFence(*fence),
               "Could not create semaphore");

    log_info("Calling clEnqueueWaitSemaphoresKHR\n");
    constexpr cl_semaphore_payload_khr semaphore_payload = 1;
    clEventWrapper wait_event;
    errcode = clEnqueueWaitSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
    test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

    log_info("Calling d3d12_fence->Signal()\n");
    const HRESULT hr = (*fence)->Signal(semaphore_payload);
    test_error(FAILED(hr), "Failed to signal D3D12 fence");

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that a wait in OpenCL followed by a GPU signal in DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_signal_dx_gpu)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);

    test_error(!is_import_handle_available(device,
                                           CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
               "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
               "supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore.createSemaphoreFromFence(*fence),
               "Could not create semaphore");

    log_info("Calling clEnqueueWaitSemaphoresKHR\n");
    constexpr cl_semaphore_payload_khr semaphore_payload = 1;
    clEventWrapper wait_event;
    errcode = clEnqueueWaitSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
    test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

    log_info("Calling d3d12_command_queue->Signal()\n");
    const HRESULT hr =
        dx_wrapper.getDXCommandQueue()->Signal(*fence, semaphore_payload);
    test_error(FAILED(hr), "Failed to signal D3D12 fence");

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    test_assert_event_complete(wait_event);

    return TEST_PASS;
}

// Confirm that interlocking waits between OpenCL and DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_cl_dx_interlock)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);

    test_error(!is_import_handle_available(device,
                                           CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
               "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
               "supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore.createSemaphoreFromFence(*fence),
               "Could not create semaphore");

    log_info("Calling d3d12_command_queue->Wait(1)\n");
    cl_semaphore_payload_khr semaphore_payload = 1;
    HRESULT hr =
        dx_wrapper.getDXCommandQueue()->Wait(*fence, semaphore_payload);
    test_error(FAILED(hr), "Failed to wait on D3D12 fence");

    log_info("Calling d3d12_command_queue->Signal(2)\n");
    hr = dx_wrapper.getDXCommandQueue()->Signal(*fence, semaphore_payload + 1);
    test_error(FAILED(hr), "Failed to signal D3D12 fence");

    log_info("Calling clEnqueueSignalSemaphoresKHR(1)\n");
    clEventWrapper signal_event;
    errcode = clEnqueueSignalSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &signal_event);
    test_error(errcode, "Failed to call clEnqueueSignalSemaphoresKHR");

    log_info("Calling clEnqueueWaitSemaphoresKHR(2)\n");
    semaphore_payload += 1;
    clEventWrapper wait_event;
    errcode = clEnqueueWaitSemaphoresKHR(
        queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
    test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    test_assert_event_complete(wait_event);
    test_assert_event_complete(signal_event);

    return TEST_PASS;
}

// Confirm that multiple waits in OpenCL followed by signals in DX12 and waits
// in DX12 followed by signals in OpenCL complete successfully
REGISTER_TEST(test_external_semaphores_multiple_wait_signal)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);
    GET_PFN(device, clEnqueueWaitSemaphoresKHR);

    test_error(!is_import_handle_available(device,
                                           CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
               "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the "
               "supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence_1(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore_1(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore_1.createSemaphoreFromFence(*fence_1),
               "Could not create semaphore");

    const DirectXFenceWrapper fence_2(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore_2(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore_2.createSemaphoreFromFence(*fence_2),
               "Could not create semaphore");

    const cl_semaphore_khr semaphore_list[] = { *semaphore_1, *semaphore_2 };
    constexpr cl_semaphore_payload_khr semaphore_payload = 1;
    cl_semaphore_payload_khr semaphore_payload_list[] = {
        semaphore_payload, semaphore_payload + 1
    };

    log_info("Calling clEnqueueWaitSemaphoresKHR\n");
    clEventWrapper wait_event;
    errcode = clEnqueueWaitSemaphoresKHR(queue, 2, semaphore_list,
                                         semaphore_payload_list, 0, nullptr,
                                         &wait_event);
    test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

    log_info("Calling d3d12_command_queue->Signal()\n");
    HRESULT hr =
        dx_wrapper.getDXCommandQueue()->Signal(*fence_2, semaphore_payload + 1);
    test_error(FAILED(hr), "Failed to signal D3D12 fence 2");
    hr = dx_wrapper.getDXCommandQueue()->Signal(*fence_1, semaphore_payload);
    test_error(FAILED(hr), "Failed to signal D3D12 fence 1");

    log_info("Calling d3d12_command_queue->Wait() with different payloads\n");
    hr = dx_wrapper.getDXCommandQueue()->Wait(*fence_1, semaphore_payload + 3);
    test_error(FAILED(hr), "Failed to wait on D3D12 fence 1");
    hr = dx_wrapper.getDXCommandQueue()->Wait(*fence_2, semaphore_payload + 2);
    test_error(FAILED(hr), "Failed to wait on D3D12 fence 2");

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    test_assert_event_complete(wait_event);

    semaphore_payload_list[0] = semaphore_payload + 3;
    semaphore_payload_list[1] = semaphore_payload + 2;

    log_info("Calling clEnqueueSignalSemaphoresKHR\n");
    clEventWrapper signal_event;
    errcode = clEnqueueSignalSemaphoresKHR(queue, 2, semaphore_list,
                                           semaphore_payload_list, 0, nullptr,
                                           &signal_event);
    test_error(errcode, "Could not call clEnqueueSignalSemaphoresKHR");

    // Wait until the GPU has completed commands up to this fence point.
    log_info("Waiting for D3D12 command queue completion\n");
    if ((*fence_1)->GetCompletedValue() < semaphore_payload_list[0])
    {
        const HANDLE event_handle =
            CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
        hr = (*fence_1)->SetEventOnCompletion(semaphore_payload_list[0],
                                              event_handle);
        test_error(FAILED(hr),
                   "Failed to set D3D12 fence 1 event on completion");
        WaitForSingleObject(event_handle, INFINITE);
        CloseHandle(event_handle);
    }
    if ((*fence_2)->GetCompletedValue() < semaphore_payload_list[1])
    {
        const HANDLE event_handle =
            CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
        hr = (*fence_2)->SetEventOnCompletion(semaphore_payload_list[1],
                                              event_handle);
        test_error(FAILED(hr),
                   "Failed to set D3D12 fence 2 event on completion");
        WaitForSingleObject(event_handle, INFINITE);
        CloseHandle(event_handle);
    }

    errcode = clFinish(queue);
    test_error(errcode, "Could not finish queue");

    test_assert_event_complete(signal_event);

    return TEST_PASS;
}