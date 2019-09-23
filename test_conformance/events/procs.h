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
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/clImageHelper.h"

extern float    random_float(float low, float high);
extern float    calculate_ulperror(float a, float b);


extern int        test_event_get_execute_status(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_get_write_array_status(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_get_read_array_status(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_get_info( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_wait_for_execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_wait_for_array(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_flush(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_finish_execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_finish_array(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_release_before_done(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_enqueue_marker(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
#ifdef CL_VERSION_1_2
extern int        test_event_enqueue_marker_with_event_list(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_event_enqueue_barrier_with_event_list(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
#endif

extern int        test_out_of_order_event_waitlist_single_queue(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_waitlist_multi_queue( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_waitlist_multi_queue_multi_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_out_of_order_event_enqueue_wait_for_events_single_queue(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_enqueue_wait_for_events_multi_queue( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_enqueue_wait_for_events_multi_queue_multi_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_out_of_order_event_enqueue_barrier_single_queue(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_out_of_order_event_enqueue_marker_single_queue(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_enqueue_marker_multi_queue( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_out_of_order_event_enqueue_marker_multi_queue_multi_device(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_waitlists( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_userevents( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_callbacks( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_callbacks_simultaneous( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );
extern int        test_userevents_multithreaded( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements );


