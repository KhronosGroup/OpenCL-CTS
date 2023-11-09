//
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
#ifndef CL_KHR_COMMAND_BUFFER_PROCS_H
#define CL_KHR_COMMAND_BUFFER_PROCS_H

#include <CL/cl.h>

// Basic command-buffer tests
extern int test_single_ndrange(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_interleaved_enqueue(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_mixed_commands(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_explicit_flush(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_out_of_order(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements);
extern int test_basic_printf(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements);
extern int test_simultaneous_printf(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_info_queues(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements);
extern int test_info_ref_count(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_info_state(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements);
extern int test_info_prop_array(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_info_context(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elements);
extern int test_basic_set_kernel_arg(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);
extern int test_pending_set_kernel_arg(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_regular_wait_for_command_buffer(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements);
extern int test_command_buffer_wait_for_command_buffer(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements);
extern int test_command_buffer_wait_for_sec_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_return_event_callback(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_clwaitforevents_single(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_clwaitforevents(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_command_buffer_wait_for_regular(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements);
extern int test_wait_for_sec_queue_event(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements);
extern int test_user_event_wait(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_user_events_wait(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements);
extern int test_user_event_callback(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_simultaneous_out_of_order(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_basic_profiling(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_simultaneous_profiling(cl_device_id device, cl_context context,
                                       cl_command_queue queue,
                                       int num_elements);
extern int test_queue_substitution(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements);
extern int test_properties_queue_substitution(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_simultaneous_queue_substitution(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements);
extern int test_fill_image(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements);
extern int test_fill_buffer(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements);
extern int test_fill_svm_buffer(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_copy_image(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements);
extern int test_copy_buffer(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements);
extern int test_copy_svm_buffer(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements);
extern int test_copy_buffer_to_image(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);
extern int test_copy_image_to_buffer(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements);
extern int test_copy_buffer_rect(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements);
extern int test_barrier_wait_list(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements);
extern int test_event_info_command_type(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_event_info_command_queue(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements);
extern int test_event_info_context(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements);
extern int test_event_info_execution_status(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_event_info_reference_count(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_finalize_invalid(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements);
extern int test_finalize_empty(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements);

#endif // CL_KHR_COMMAND_BUFFER_PROCS_H
