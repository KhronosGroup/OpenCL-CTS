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
#include "harness/conversions.h"
#include "harness/mt19937.h"

// The number of errors to print out for each test in the shuffle tests
#define MAX_ERRORS_TO_PRINT 1

extern const int g_vector_aligns[];
extern const int g_vector_allocs[];

#define DENSE_PACK_VECS 1

extern int      create_program_and_kernel(const char *source, const char *kernel_name, cl_program *program_ret, cl_kernel *kernel_ret);

extern int test_relational_any(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_all(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_bitselect(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_select_signed(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_select_unsigned(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_relational_isequal(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_isnotequal(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_isgreater(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_isgreaterequal(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_isless(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_islessequal(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_relational_islessgreater(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_shuffles(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffles_16(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffles_dual(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffle_copy(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffle_function_call(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffle_array_cast(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffle_built_in(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_shuffle_built_in_dual_input(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);


