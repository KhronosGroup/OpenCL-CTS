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
#ifndef CL_KHR_EXTERNAL_SEMAPHORE_PROCS_H
#define CL_KHR_EXTERNAL_SEMAPHORE_PROCS_H

#include <CL/cl.h>

// Basic command-buffer tests

extern int test_external_semaphores_queries(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue defaultQueue,
                                            int num_elements);
extern int test_external_semaphores_cross_context(cl_device_id deviceID,
                                                  cl_context context,
                                                  cl_command_queue defaultQueue,
                                                  int num_elements);
extern int test_external_semaphores_simple_1(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_external_semaphores_simple_2(cl_device_id deviceID,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements);
extern int test_external_semaphores_reuse(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements);
extern int test_external_semaphores_cross_queues_ooo(cl_device_id deviceID,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements);
extern int test_external_semaphores_cross_queues_io(cl_device_id deviceID,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements);
extern int test_external_semaphores_cross_queues_io2(
    cl_device_id deviceID, cl_context context, cl_command_queue defaultQueue,
    int num_elements);
extern int test_external_semaphores_multi_signal(cl_device_id deviceID,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements);
extern int test_external_semaphores_multi_wait(cl_device_id deviceID,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements);
extern int test_external_semaphores_import_export_fd(cl_device_id deviceID,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements);
#endif // CL_KHR_EXTERNAL_SEMAPHORE_PROCS_H
