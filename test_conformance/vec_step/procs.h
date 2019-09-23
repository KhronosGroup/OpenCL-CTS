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
#include "harness/conversions.h"
#include "harness/mt19937.h"

// The number of errors to print out for each test in the shuffle tests
#define MAX_ERRORS_TO_PRINT 1


extern int      create_program_and_kernel(const char *source, const char *kernel_name, cl_program *program_ret, cl_kernel *kernel_ret);


/*
    test_step_type,
    test_step_var,
    test_step_typedef_type,
    test_step_typedef_var,
*/

extern int test_step_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_step_var(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_step_typedef_type(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

extern int test_step_typedef_var(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
