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

// Confirm that a wait without a semaphore payload list will return
// CL_INVALID_VALUE
REGISTER_TEST(test_external_semaphores_dx_fence_negative_wait)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
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
    errcode = clEnqueueWaitSemaphoresKHR(queue, 1, &semaphore, nullptr, 0,
                                         nullptr, nullptr);
    test_assert_error(
        errcode == CL_INVALID_VALUE,
        "Unexpected error code returned from clEnqueueWaitSemaphores");

    return TEST_PASS;
}

// Confirm that a signal without a semaphore payload list will return
// CL_INVALID_VALUE
REGISTER_TEST(test_external_semaphores_dx_fence_negative_signal)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clEnqueueSignalSemaphoresKHR);

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
    errcode = clEnqueueSignalSemaphoresKHR(queue, 1, &semaphore, nullptr, 0,
                                           nullptr, nullptr);
    test_assert_error(
        errcode == CL_INVALID_VALUE,
        "Unexpected error code returned from clEnqueueSignalSemaphores");

    return TEST_PASS;
}