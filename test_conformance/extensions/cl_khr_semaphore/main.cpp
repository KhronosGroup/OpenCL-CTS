//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "harness/compat.h"
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST_VERSION(semaphores_simple_1, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_simple_2, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_reuse, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_cross_queues_ooo, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_cross_queues_io, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_multi_signal, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_multi_wait, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_device_list_queries, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_no_device_list_queries, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_multi_device_context_queries, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_ooo_ops_single_queue, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_ooo_ops_cross_queue, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_invalid_context, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_invalid_property,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_multi_device_property,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_invalid_device, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_import_invalid_device,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_invalid_value, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_create_invalid_operation,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_get_info_invalid_semaphore,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_get_info_invalid_value, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_command_queue,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_value, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_semaphore, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_context, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_event_wait_list,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_wait_invalid_event_status,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_command_queue,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_value, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_semaphore,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_context, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_event_wait_list,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_signal_invalid_event_status,
                     Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_release, Version(1, 2)),
    ADD_TEST_VERSION(semaphores_negative_retain, Version(1, 2)),
};

const int test_num = ARRAY_SIZE(test_list);

int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}
