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

namespace {

struct PayloadSemaphore : public SemaphoreTestBase
{
    PayloadSemaphore(cl_device_id device, cl_context context,
                     cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;
        // Create ooo queue
        clCommandQueueWrapper queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        test_error(err, "Could not create command queue");

        // Create semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");


        clSemaphoreWrapper sema_sec = nullptr;
        sema_sec =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        cl_semaphore_payload_khr payload_list[] = { 1, 2 };
        cl_semaphore_khr semaphores[2] = { semaphore, sema_sec };

        // Signal semaphore
        err = clEnqueueSignalSemaphoresKHR(queue, 2, semaphores, payload_list,
                                           0, nullptr, nullptr);
        test_error(err, "Could not signal semaphore");

        // Wait semaphore
        err = clEnqueueWaitSemaphoresKHR(queue, 2, semaphores, payload_list, 0,
                                         nullptr, nullptr);
        test_error(err, "Could not wait semaphore");

        // Finish
        err = clFinish(queue);
        test_error(err, "Could not finish queue");

        return CL_SUCCESS;
    }
};


} // anonymous namespace

// Confirm that a valid semaphore payload values list will be ignored if no
// semaphores in the list of sema_objects require a payload
REGISTER_TEST_VERSION(semaphores_payload, Version(1, 2))
{
    return MakeAndRunTest<PayloadSemaphore>(device, context, queue,
                                            num_elements);
}
