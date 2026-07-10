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

struct DXFenceNegativeWait final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    int Run() override
    {
        log_info("Calling clEnqueueWaitSemaphoresKHR\n");
        errcode = clEnqueueWaitSemaphoresKHR(queue, 1, &semaphore, nullptr, 0,
                                             nullptr, nullptr);
        test_assert_error(
            errcode == CL_INVALID_VALUE,
            "Unexpected error code returned from clEnqueueWaitSemaphores");

        return TEST_PASS;
    }
};

// Confirm that a wait without a semaphore payload list will return
// CL_INVALID_VALUE
REGISTER_TEST(test_external_semaphores_dx_fence_negative_wait)
{
    return MakeAndRunTest<DXFenceNegativeWait>(device, context, queue,
                                               num_elements);
}

struct DXFenceNegativeSignal final : DXFenceTestBase
{
    using DXFenceTestBase::DXFenceTestBase;

    int Run() override
    {
        log_info("Calling clEnqueueWaitSemaphoresKHR\n");
        errcode = clEnqueueSignalSemaphoresKHR(queue, 1, &semaphore, nullptr, 0,
                                               nullptr, nullptr);
        test_assert_error(
            errcode == CL_INVALID_VALUE,
            "Unexpected error code returned from clEnqueueSignalSemaphores");

        return TEST_PASS;
    }
};

// Confirm that a signal without a semaphore payload list will return
// CL_INVALID_VALUE
REGISTER_TEST(test_external_semaphores_dx_fence_negative_signal)
{
    return MakeAndRunTest<DXFenceNegativeSignal>(device, context, queue,
                                                 num_elements);
}