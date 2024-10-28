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

#include "semaphore_base.h"

#include "harness/errorHelpers.h"
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>

namespace {

// sema_object is not a valid semaphore object

struct ReleaseInvalidSemaphore : public SemaphoreTestBase
{
    ReleaseInvalidSemaphore(cl_device_id device, cl_context context,
                            cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        // Release invalid semaphore
        cl_int err = CL_SUCCESS;
        err = clReleaseSemaphoreKHR(nullptr);
        if (err != CL_INVALID_SEMAPHORE_KHR)
        {
            log_error("Unexpected clReleaseSemaphoreKHR result, expected "
                      "CL_INVALID_SEMAPHORE_KHR, get %s\n",
                      IGetErrorString(err));
            return TEST_FAIL;
        }

        return TEST_PASS;
    }
};

struct RetainInvalidSemaphore : public SemaphoreTestBase
{
    RetainInvalidSemaphore(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        // Release invalid semaphore
        cl_int err = CL_SUCCESS;
        err = clRetainSemaphoreKHR(nullptr);
        if (err != CL_INVALID_SEMAPHORE_KHR)
        {
            log_error("Unexpected clRetainSemaphoreKHR result, expected "
                      "CL_INVALID_SEMAPHORE_KHR, get %s\n",
                      IGetErrorString(err));
            return TEST_FAIL;
        }

        return TEST_PASS;
    }
};

}

int test_semaphores_negative_release(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ReleaseInvalidSemaphore>(device, context, queue,
                                                   num_elements);
}

int test_semaphores_negative_retain(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<RetainInvalidSemaphore>(device, context, queue,
                                                  num_elements);
}
