//
// Copyright (c) 2017, 2021 The Khronos Group Inc.
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
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

extern int create_program_and_kernel(const char *source,
                                     const char *kernel_name,
                                     cl_program *program_ret,
                                     cl_kernel *kernel_ret);

extern int test_work_group_all(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_work_group_any(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements);
extern int test_work_group_broadcast_1D(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_work_group_broadcast_2D(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_work_group_broadcast_3D(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements);
extern int test_work_group_reduce_add(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_work_group_reduce_min(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements);
extern int test_work_group_reduce_max(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements);

extern int test_work_group_scan_exclusive_add(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_work_group_scan_exclusive_min(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_work_group_scan_exclusive_max(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_work_group_scan_inclusive_add(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_work_group_scan_inclusive_min(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
extern int test_work_group_scan_inclusive_max(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements);
