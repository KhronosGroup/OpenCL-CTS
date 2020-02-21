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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST( event_get_execute_status ),
    ADD_TEST( event_get_write_array_status ),
    ADD_TEST( event_get_read_array_status ),
    ADD_TEST( event_get_info ),
    ADD_TEST( event_wait_for_execute ),
    ADD_TEST( event_wait_for_array ),
    ADD_TEST( event_flush ),
    ADD_TEST( event_finish_execute ),
    ADD_TEST( event_finish_array ),
    ADD_TEST( event_release_before_done ),
    ADD_TEST( event_enqueue_marker ),
#ifdef CL_VERSION_1_2
    ADD_TEST( event_enqueue_marker_with_event_list ),
    ADD_TEST( event_enqueue_barrier_with_event_list ),
#endif

    ADD_TEST( out_of_order_event_waitlist_single_queue ),
    ADD_TEST( out_of_order_event_waitlist_multi_queue ),
    ADD_TEST( out_of_order_event_waitlist_multi_queue_multi_device ),
    ADD_TEST( out_of_order_event_enqueue_wait_for_events_single_queue ),
    ADD_TEST( out_of_order_event_enqueue_wait_for_events_multi_queue ),
    ADD_TEST( out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device ),
    ADD_TEST( out_of_order_event_enqueue_marker_single_queue ),
    ADD_TEST( out_of_order_event_enqueue_marker_multi_queue ),
    ADD_TEST( out_of_order_event_enqueue_marker_multi_queue_multi_device ),
    ADD_TEST( out_of_order_event_enqueue_barrier_single_queue ),

    ADD_TEST( waitlists ),
    ADD_TEST( userevents ),
    ADD_TEST( callbacks ),
    ADD_TEST( callbacks_simultaneous ),
    ADD_TEST( userevents_multithreaded ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

