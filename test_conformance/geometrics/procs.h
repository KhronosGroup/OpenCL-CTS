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
#include "harness/threadTesting.h"
#include "harness/typeWrappers.h"

extern int      create_program_and_kernel(const char *source, const char *kernel_name, cl_program *program_ret, cl_kernel *kernel_ret);

extern int test_geom_cross(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_dot(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_distance(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_fast_distance(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_length(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_fast_length(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_normalize(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_geom_fast_normalize(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_geom_cross_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d);
extern int test_geom_dot_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d);
extern int test_geom_distance_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d);
extern int test_geom_length_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d);
extern int test_geom_normalize_double(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, MTdata d);
