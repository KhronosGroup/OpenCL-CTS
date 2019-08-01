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

extern int        test_atomic_add(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_sub(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_xchg(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_min(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_max(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_inc(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_dec(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_cmpxchg(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_and(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_or(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_xor(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int        test_atomic_add_index(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int        test_atomic_add_index_bin(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);



