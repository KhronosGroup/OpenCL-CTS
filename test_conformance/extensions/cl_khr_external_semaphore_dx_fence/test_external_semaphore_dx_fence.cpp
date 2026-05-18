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

struct SignalWait final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    cl_int Run() override
    {
        log_info("Calling clEnqueueSignalSemaphoresKHR\n");
        clEventWrapper signal_event;
        errcode = clEnqueueSignalSemaphoresKHR(queue, 1, &semaphore,
                                               &semaphore_payload, 0, nullptr,
                                               &signal_event);
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
};

// Confirm that a signal followed by a wait in OpenCL will complete successfully
REGISTER_TEST(test_external_semaphores_signal_wait)
{
    return MakeAndRunTest<SignalWait>(device, context, queue, num_elements);
}

struct SignalDXCPU final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    cl_int Run() override
    {
        log_info("Calling clEnqueueWaitSemaphoresKHR\n");
        clEventWrapper wait_event;
        errcode = clEnqueueWaitSemaphoresKHR(
            queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
        test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

        log_info("Calling d3d12_fence->Signal()\n");
        const HRESULT hr = fence_wrapper->get()->Signal(semaphore_payload);
        test_error(FAILED(hr), "Failed to signal D3D12 fence");

        errcode = clFinish(queue);
        test_error(errcode, "Could not finish queue");

        test_assert_event_complete(wait_event);

        return TEST_PASS;
    }
};

// Confirm that a wait in OpenCL followed by a CPU signal in DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_signal_dx_cpu)
{
    return MakeAndRunTest<SignalDXCPU>(device, context, queue, num_elements);
}

struct SignalDXGPU final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    cl_int Run() override
    {
        log_info("Calling clEnqueueWaitSemaphoresKHR\n");
        clEventWrapper wait_event;
        errcode = clEnqueueWaitSemaphoresKHR(
            queue, 1, &semaphore, &semaphore_payload, 0, nullptr, &wait_event);
        test_error(errcode, "Failed to call clEnqueueWaitSemaphoresKHR");

        log_info("Calling d3d12_command_queue->Signal()\n");
        const HRESULT hr = dx_wrapper.getDXCommandQueue()->Signal(
            fence_wrapper->get(), semaphore_payload);
        test_error(FAILED(hr), "Failed to signal D3D12 fence");

        errcode = clFinish(queue);
        test_error(errcode, "Could not finish queue");

        test_assert_event_complete(wait_event);

        return TEST_PASS;
    }
};

// Confirm that a wait in OpenCL followed by a GPU signal in DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_signal_dx_gpu)
{
    return MakeAndRunTest<SignalDXGPU>(device, context, queue, num_elements);
}

struct CLDXInterlock final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    cl_int Run() override
    {
        log_info("Calling d3d12_command_queue->Wait(1)\n");
        HRESULT hr = dx_wrapper.getDXCommandQueue()->Wait(fence_wrapper->get(),
                                                          semaphore_payload);
        test_error(FAILED(hr), "Failed to wait on D3D12 fence");

        log_info("Calling d3d12_command_queue->Signal(2)\n");
        hr = dx_wrapper.getDXCommandQueue()->Signal(fence_wrapper->get(),
                                                    semaphore_payload + 1);
        test_error(FAILED(hr), "Failed to signal D3D12 fence");

        log_info("Calling clEnqueueSignalSemaphoresKHR(1)\n");
        clEventWrapper signal_event;
        errcode = clEnqueueSignalSemaphoresKHR(queue, 1, &semaphore,
                                               &semaphore_payload, 0, nullptr,
                                               &signal_event);
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
};

// Confirm that interlocking waits between OpenCL and DX12 will complete
// successfully
REGISTER_TEST(test_external_semaphores_cl_dx_interlock)
{
    return MakeAndRunTest<CLDXInterlock>(device, context, queue, num_elements);
}

struct MultipleWaitSignal final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    ~MultipleWaitSignal() override
    {
        if (fence_handle_2)
        {
            CloseHandle(fence_handle_2);
            fence_handle_2 = nullptr;
        }
        if (fence_wrapper_2)
        {
            delete fence_wrapper_2;
            fence_wrapper_2 = nullptr;
        }
        if (semaphore_2)
        {
            clReleaseSemaphoreKHR(semaphore_2);
            semaphore_2 = nullptr;
        }
        DXFenceTestBase::~DXFenceTestBase();
    };

    int SetUp() override
    {
        DXFenceTestBase::SetUp();
        fence_wrapper_2 = new DirectXFenceWrapper(dx_wrapper.getDXDevice());
        semaphore_2 = createSemaphoreFromFence(fence_wrapper_2->get());
        test_assert_error(!!semaphore_2, "Could not create semaphore");

        return TEST_PASS;
    }

    cl_int Run() override
    {
        const cl_semaphore_khr semaphore_list[] = { semaphore, semaphore_2 };
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
        HRESULT hr = dx_wrapper.getDXCommandQueue()->Signal(
            fence_wrapper_2->get(), semaphore_payload + 1);
        test_error(FAILED(hr), "Failed to signal D3D12 fence 2");
        hr = dx_wrapper.getDXCommandQueue()->Signal(fence_wrapper->get(),
                                                    semaphore_payload);
        test_error(FAILED(hr), "Failed to signal D3D12 fence 1");

        log_info(
            "Calling d3d12_command_queue->Wait() with different payloads\n");
        hr = dx_wrapper.getDXCommandQueue()->Wait(fence_wrapper->get(),
                                                  semaphore_payload + 3);
        test_error(FAILED(hr), "Failed to wait on D3D12 fence 1");
        hr = dx_wrapper.getDXCommandQueue()->Wait(fence_wrapper_2->get(),
                                                  semaphore_payload + 2);
        test_error(FAILED(hr), "Failed to wait on D3D12 fence 2");

        errcode = clFinish(queue);
        test_error(errcode, "Could not finish queue");

        test_assert_event_complete(wait_event);

        semaphore_payload_list[0] = semaphore_payload + 3;
        semaphore_payload_list[1] = semaphore_payload + 2;

        log_info("Calling clEnqueueSignalSemaphoresKHR\n");
        clEventWrapper signal_event;
        errcode = clEnqueueSignalSemaphoresKHR(queue, 2, semaphore_list,
                                               semaphore_payload_list, 0,
                                               nullptr, &signal_event);
        test_error(errcode, "Could not call clEnqueueSignalSemaphoresKHR");

        // Wait until the GPU has completed commands up to this fence point.
        log_info("Waiting for D3D12 command queue completion\n");
        if (fence_wrapper->get()->GetCompletedValue()
            < semaphore_payload_list[0])
        {
            const HANDLE event_handle =
                CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            hr = fence_wrapper->get()->SetEventOnCompletion(
                semaphore_payload_list[0], event_handle);
            test_error(FAILED(hr),
                       "Failed to set D3D12 fence 1 event on completion");
            WaitForSingleObject(event_handle, INFINITE);
            CloseHandle(event_handle);
        }
        if (fence_wrapper_2->get()->GetCompletedValue()
            < semaphore_payload_list[1])
        {
            const HANDLE event_handle =
                CreateEventEx(nullptr, nullptr, false, EVENT_ALL_ACCESS);
            hr = fence_wrapper_2->get()->SetEventOnCompletion(
                semaphore_payload_list[1], event_handle);
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

protected:
    cl_semaphore_khr semaphore_2 = nullptr;
    HANDLE fence_handle_2 = nullptr;
    DirectXFenceWrapper *fence_wrapper_2 = nullptr;
};

// Confirm that multiple waits in OpenCL followed by signals in DX12 and waits
// in DX12 followed by signals in OpenCL complete successfully
REGISTER_TEST(test_external_semaphores_multiple_wait_signal)
{
    return MakeAndRunTest<MultipleWaitSignal>(device, context, queue,
                                              num_elements);
}