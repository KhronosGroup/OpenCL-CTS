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

namespace {

// sema_object is not a valid semaphore.

struct GetInfoInvalidSemaphore : public SemaphoreTestBase
{
    GetInfoInvalidSemaphore(cl_device_id device, cl_context context,
                            cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Wait semaphore
        cl_semaphore_type_khr type = 0;
        size_t ret_size = 0;
        cl_int err = clGetSemaphoreInfoKHR(nullptr, CL_SEMAPHORE_TYPE_KHR,
                                           sizeof(cl_semaphore_type_khr), &type,
                                           &ret_size);
        test_failure_error(err, CL_INVALID_SEMAPHORE_KHR,
                           "Unexpected clGetSemaphoreInfoKHR return");

        return CL_SUCCESS;
    }
};

// 1) param_name is not one of the attribute defined in the Semaphore Queries
// table

// 2) param_value_size is less than the size of Return Type of the corresponding
// param_name attribute as defined in the Semaphore Queries table.

struct GetInfoInvalidValue : public SemaphoreTestBase
{
    GetInfoInvalidValue(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : SemaphoreTestBase(device, context, queue)
    {}

    cl_int Run() override
    {
        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR),
            (cl_semaphore_properties_khr)device,
            CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
            0
        };

        cl_int err = CL_SUCCESS;
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // (1)
        cl_semaphore_info_khr param_name = ~0;
        err = clGetSemaphoreInfoKHR(semaphore, param_name, 0, nullptr, nullptr);
        test_failure_error(err, CL_INVALID_VALUE,
                           "Unexpected clGetSemaphoreInfoKHR return");

        // (2)
        size_t size = 0;
        err = clGetSemaphoreInfoKHR(semaphore, CL_SEMAPHORE_PROPERTIES_KHR, 0,
                                    nullptr, &size);
        test_error(err, "Could not query semaphore");

        // make sure that first test provides too small param size
        if (size != sizeof(sema_props))
            test_fail("Error: expected size %d, returned %d",
                      sizeof(sema_props), size);

        // first test with non-zero property size but not enough
        cl_semaphore_properties_khr ret_props = 0;
        err = clGetSemaphoreInfoKHR(semaphore, CL_SEMAPHORE_PROPERTIES_KHR,
                                    sizeof(ret_props), &ret_props, nullptr);
        test_failure_error(err, CL_INVALID_VALUE,
                           "Unexpected clGetSemaphoreInfoKHR return");

        // second test with zero property size
        cl_semaphore_type_khr type = 0;
        err = clGetSemaphoreInfoKHR(semaphore, CL_SEMAPHORE_TYPE_KHR, 0, &type,
                                    nullptr);
        test_failure_error(err, CL_INVALID_VALUE,
                           "Unexpected clGetSemaphoreInfoKHR return");

        return CL_SUCCESS;
    }
};

}

int test_semaphores_negative_get_info_invalid_semaphore(cl_device_id device,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements)
{
    return MakeAndRunTest<GetInfoInvalidSemaphore>(device, context, queue);
}

int test_semaphores_negative_get_info_invalid_value(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<GetInfoInvalidValue>(device, context, queue);
}
