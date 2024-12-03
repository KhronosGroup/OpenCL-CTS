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
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/clImageHelper.h"
#include "harness/imageHelpers.h"

extern int test_semaphores_simple_1(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_semaphores_simple_2(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements);
extern int test_semaphores_reuse(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements);
extern int test_semaphores_cross_queues_ooo(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_semaphores_cross_queues_io(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_semaphores_multi_signal(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_semaphores_multi_wait(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_semaphores_device_list_queries(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
extern int test_semaphores_no_device_list_queries(cl_device_id deviceID,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements);
extern int test_semaphores_multi_device_context_queries(cl_device_id deviceID,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements);
extern int test_semaphores_import_export_fd(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_semaphores_negative_create_invalid_context(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_create_invalid_property(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_create_multi_device_property(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_create_invalid_device(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_create_import_invalid_device(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_create_invalid_value(cl_device_id device,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements);
extern int test_semaphores_negative_create_invalid_operation(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_get_info_invalid_semaphore(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_get_info_invalid_value(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_wait_invalid_command_queue(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_wait_invalid_value(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements);
extern int test_semaphores_negative_wait_invalid_semaphore(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_wait_invalid_context(cl_device_id device,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements);
extern int test_semaphores_negative_wait_invalid_event_wait_list(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_wait_invalid_event_status(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_signal_invalid_command_queue(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_signal_invalid_value(cl_device_id device,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements);
extern int test_semaphores_negative_signal_invalid_semaphore(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_signal_invalid_context(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_signal_invalid_event_wait_list(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_signal_invalid_event_status(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements);
extern int test_semaphores_negative_release(cl_device_id device,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements);
extern int test_semaphores_negative_retain(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements);
extern int test_semaphores_ooo_ops_single_queue(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue defaultQueue,
                                                int num_elements);
extern int test_semaphores_ooo_ops_cross_queue(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue defaultQueue,
                                               int num_elements);
