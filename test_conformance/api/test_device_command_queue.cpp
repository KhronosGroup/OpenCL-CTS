//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "testBase.h"
#include <array>

using namespace std;


int test_command_queue_helper(cl_context context, cl_device_id deviceID,
                              cl_command_queue queue)
{
    cl_int error;
    cl_command_queue check_queue;

    error = clSetDefaultDeviceCommandQueue(context, deviceID, queue);
    test_error(error, "clSetDefaultDeviceCommandQueue failed ");

    error = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE_DEFAULT,
                                  sizeof(check_queue), &check_queue, nullptr);
    test_error(error,
               "clGetCommandQueueInfo failed for CL_QUEUE_DEVICE_DEFAULT");
    test_assert_error(
        (check_queue == queue),
        "Expected the queue to be returned as default device queue failed");

    return TEST_PASS;
}

REGISTER_TEST_VERSION(set_default_device_command_queue, Version(2, 1))
{
    cl_int error;
    constexpr cl_command_queue_properties PROPERTIES = CL_QUEUE_ON_DEVICE
        | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
    std::array<cl_queue_properties, 3> properties = {
        CL_QUEUE_PROPERTIES, (PROPERTIES | CL_QUEUE_ON_DEVICE_DEFAULT), 0
    };

    if (get_device_cl_version(device) >= Version(3, 0))
    {
        cl_device_device_enqueue_capabilities dseCaps = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                                sizeof(dseCaps), &dseCaps, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES");

        if (0 == (dseCaps & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT))
            return TEST_SKIPPED_ITSELF;
    }

    clCommandQueueWrapper cmd_queue_1 = clCreateCommandQueueWithProperties(
        context, device, properties.data(), &error);
    test_error(error, "clCreateCommandQueueWithProperties failed");

    properties[1] = PROPERTIES;
    clCommandQueueWrapper cmd_queue_2 = clCreateCommandQueueWithProperties(
        context, device, properties.data(), &error);
    test_error(error, "clCreateCommandQueueWithProperties failed");

    // cmd_queue_1
    if (test_command_queue_helper(context, device, cmd_queue_1) != 0)
    {
        test_fail("test_command_queue_helper failed for cmd_queue_1.\n");
    }

    // cmd_queue_2 - without CL_QUEUE_ON_DEVICE_DEFAULT
    if (test_command_queue_helper(context, device, cmd_queue_2) != 0)
    {
        test_fail("test_command_queue_helper failed for cmd_queue_2.\n");
    }

    return TEST_PASS;
}
