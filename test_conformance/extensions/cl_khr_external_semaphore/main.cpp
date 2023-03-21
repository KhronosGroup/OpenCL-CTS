// Copyright (c) 2022 The Khronos Group Inc.
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
#include "procs.h"
#include "harness/testHarness.h"

test_definition test_list[] = {
    ADD_TEST(external_semaphores_queries),
    ADD_TEST(external_semaphores_multi_context),
    ADD_TEST(external_semaphores_simple_1),
    // ADD_TEST(external_semaphores_simple_2),
    ADD_TEST(external_semaphores_reuse),
    ADD_TEST(external_semaphores_cross_queues_ooo),
    ADD_TEST(external_semaphores_cross_queues_io),
    ADD_TEST(external_semaphores_cross_queues_io2),
    ADD_TEST(external_semaphores_multi_signal),
    ADD_TEST(external_semaphores_multi_wait),
    // ADD_TEST(external_semaphores_order_1),
    // ADD_TEST(external_semaphores_order_2),
    // ADD_TEST(external_semaphores_order_3),
    // ADD_TEST(external_semaphores_invalid_command)
};


int main(int argc, const char *argv[])
{
    // A device may report the required properties of a queue that
    // is compatible with command-buffers via the query
    // CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR. We account
    // for this in the tests themselves, rather than here, where we have a
    // device to query.
    const cl_command_queue_properties queue_properties = 0;
    return runTestHarnessWithCheck(argc, argv, ARRAY_SIZE(test_list), test_list,
                                   false, queue_properties, nullptr);
}
