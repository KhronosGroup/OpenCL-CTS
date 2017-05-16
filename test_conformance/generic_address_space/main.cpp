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
#include "../../test_common/harness/testHarness.h"

#include <iostream>

// basic tests
extern int test_function_params_get_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_function_params_to_address_space(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_variable_get_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_variable_to_address_space(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_conditional_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_chain_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_ternary_operator_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_language_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_language_union(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_multiple_calls_same_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_compare_pointers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
// advanced tests
extern int test_library_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_generic_variable_volatile(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_generic_variable_const(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_generic_variable_gentype(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_builtin_functions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_generic_advanced_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_generic_ptr_to_host_mem(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_max_number_of_params(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);

basefn basefn_list[] = {
    // basic tests
    test_function_params_get_fence,
    test_function_params_to_address_space,
    test_variable_get_fence,
    test_variable_to_address_space,
    test_casting,
    test_conditional_casting,
    test_chain_casting,
    test_ternary_operator_casting,
    test_language_struct,
    test_language_union,
    test_multiple_calls_same_function,
    test_compare_pointers,
    // advanced tests
    test_library_function,
    test_generic_variable_volatile,
    test_generic_variable_const,
    test_generic_variable_gentype,
    test_builtin_functions,
    test_generic_advanced_casting,
    test_generic_ptr_to_host_mem,
    test_max_number_of_params,
};

const char *basefn_names[] = {
    //basic tests
    "function_get_fence",
    "function_to_address_space",
    "variable_get_fence",
    "variable_to_address_space",
    "casting",
    "conditional_casting",
    "chain_casting",
    "ternary_operator_casting",
    "language_struct",
    "language_union",
    "multiple_calls_same_function",
    "compare_pointers",
    // advanced tests
    "library_function",
    "generic_variable_volatile",
    "generic_variable_const",
    "generic_variable_gentype",
    "builtin_functions",
    "generic_advanced_casting",
    "generic_ptr_to_host_mem",
    "max_number_of_params",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int num_fns = sizeof(basefn_names) / sizeof(char *);

/*
    Generic Address Space
    Tests for unnamed generic address space. This feature allows developers to create single generic functions
    that are able to operate on pointers from various address spaces instead of writing separate instances for every combination.
*/

int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, num_fns, basefn_list, basefn_names, false, false, NULL);
}
