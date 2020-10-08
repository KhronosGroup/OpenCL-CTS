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
#include "harness/testHarness.h"

extern int test_device_info(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_device_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_execute_block(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_block(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_nested_blocks(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_wg_size(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_flags(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_multi_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_host_multi_queue(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_ndrange(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_host_queue_order(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);
extern int test_enqueue_profiling(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements);

extern int test_execution_stress(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements);


