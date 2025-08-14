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

// Confirm that the CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR property is in the properties returned
// by clGetSemaphoreInfo
REGISTER_TEST(test_external_semaphores_dx_fence_query_properties)
{
    int errcode = CL_SUCCESS;
    const DirectXWrapper dx_wrapper;

    REQUIRE_EXTENSION("cl_khr_external_semaphore");
    REQUIRE_EXTENSION("cl_khr_external_semaphore_dx_fence");

    // Obtain pointers to semaphore's API
    GET_PFN(device, clCreateSemaphoreWithPropertiesKHR);
    GET_PFN(device, clReleaseSemaphoreKHR);
    GET_PFN(device, clGetSemaphoreInfoKHR);

    test_error(!is_import_handle_available(device, CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR),
    "Could not find CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR between the supported import types");

    // Import D3D12 fence into OpenCL
    const DirectXFenceWrapper fence(dx_wrapper.getDXDevice());
    CLDXSemaphoreWrapper semaphore(device, context, dx_wrapper.getDXDevice());
    test_error(semaphore.createSemaphoreFromFence(*fence),
        "Could not create semaphore");

    size_t properties_size_bytes = 0;
    errcode = clGetSemaphoreInfoKHR(*semaphore, CL_SEMAPHORE_PROPERTIES_KHR, 0, nullptr, &properties_size_bytes);
    test_error(errcode, "Could not get semaphore info");
    std::vector<cl_semaphore_properties_khr> semaphore_properties(properties_size_bytes / sizeof(cl_semaphore_properties_khr));
    errcode = clGetSemaphoreInfoKHR(*semaphore, CL_SEMAPHORE_PROPERTIES_KHR, properties_size_bytes, semaphore_properties.data(), nullptr);
    test_error(errcode, "Could not get semaphore info");

    for (unsigned i = 0; i < semaphore_properties.size()-1; i++)
    {
        if (semaphore_properties[i] == CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR &&
            semaphore_properties[i+1] == reinterpret_cast<cl_semaphore_properties_khr>(semaphore.getHandle()))
        {
            return TEST_PASS;
        }
    }
    log_error("Failed to find the dx fence handle type in the semaphore properties");
    return TEST_FAIL;
}