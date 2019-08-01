//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include <stdio.h>
#include <stdlib.h>
#if !defined(_WIN32)
#include <stdbool.h>
#endif
#include <math.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
            test_event_get_execute_status,
            test_event_get_write_array_status,
            test_event_get_read_array_status,
            test_event_get_info,
            test_event_wait_for_execute,
            test_event_wait_for_array,
            test_event_flush,
            test_event_finish_execute,
            test_event_finish_array,
            test_event_release_before_done,
            test_event_enqueue_marker,
    #ifdef CL_VERSION_1_2
            test_event_enqueue_marker_with_list,
            test_event_enqueue_barrier_with_list,
    #endif


      test_event_waitlist_single_queue,
      test_event_waitlist_multi_queue,
      test_event_waitlist_multi_queue_multi_device,
      test_event_enqueue_wait_for_events_single_queue,
            test_event_enqueue_wait_for_events_multi_queue,
            test_event_enqueue_wait_for_events_multi_queue_multi_device,
      test_event_enqueue_marker_single_queue,
      test_event_enqueue_marker_multi_queue,
      test_event_enqueue_marker_multi_queue_multi_device,
          test_event_enqueue_barrier_single_queue,

            test_waitlists,
            test_userevents,
            test_callbacks,
            test_callbacks_simultaneous,
            test_userevents_multithreaded,
};

const char    *basefn_names[] = {
            "event_get_execute_status",
            "event_get_write_array_status",
            "event_get_read_array_status",
            "event_get_info",
            "event_wait_for_execute",
            "event_wait_for_array",
            "event_flush",
            "event_finish_execute",
            "event_finish_array",
            "event_release_before_done",
            "event_enqueue_marker",
#ifdef CL_VERSION_1_2
    "event_enqueue_marker_with_event_list",
    "event_enqueue_barrier_with_event_list",
#endif

      "out_of_order_event_waitlist_single_queue",
      "out_of_order_event_waitlist_multi_queue",
      "out_of_order_event_waitlist_multi_queue_multi_device",
      "out_of_order_event_enqueue_wait_for_events_single_queue",
      "out_of_order_event_enqueue_wait_for_events_multi_queue",
      "out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device",
      "out_of_order_event_enqueue_marker_single_queue",
      "out_of_order_event_enqueue_marker_multi_queue",
      "out_of_order_event_enqueue_marker_multi_queue_multi_device",
      "out_of_order_event_enqueue_barrier_single_queue",

            "waitlists",
            "test_userevents",

            "callbacks",
            "callbacks_simultaneous",

            "userevents_multithreaded",

            "all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


