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
#include "harness/typeWrappers.h"

extern int test_non_uniform_1d_basic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_1d_atomics(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_1d_barriers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_non_uniform_2d_basic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_2d_atomics(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_2d_barriers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_non_uniform_3d_basic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_3d_atomics(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_3d_barriers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_non_uniform_other_basic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_other_atomics(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_non_uniform_other_barriers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
