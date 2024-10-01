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

#include <iostream>

// basic tests
extern int test_function_get_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_function_to_address_space(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
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
extern int test_generic_ptr_to_host_mem_svm(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
extern int test_max_number_of_params(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements);
// atomic tests
int test_generic_atomics_invariant(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements);
int test_generic_atomics_variant(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements);

test_definition test_list[] = {
    // basic tests
    ADD_TEST(function_get_fence),
    ADD_TEST(function_to_address_space),
    ADD_TEST(variable_get_fence),
    ADD_TEST(variable_to_address_space),
    ADD_TEST(casting),
    ADD_TEST(conditional_casting),
    ADD_TEST(chain_casting),
    ADD_TEST(ternary_operator_casting),
    ADD_TEST(language_struct),
    ADD_TEST(language_union),
    ADD_TEST(multiple_calls_same_function),
    ADD_TEST(compare_pointers),
    // advanced tests
    ADD_TEST(library_function),
    ADD_TEST(generic_variable_volatile),
    ADD_TEST(generic_variable_const),
    ADD_TEST(generic_variable_gentype),
    ADD_TEST(builtin_functions),
    ADD_TEST(generic_advanced_casting),
    ADD_TEST(generic_ptr_to_host_mem),
    ADD_TEST(generic_ptr_to_host_mem_svm),
    ADD_TEST(max_number_of_params),
    // atomic tests
    ADD_TEST(generic_atomics_invariant),
    ADD_TEST(generic_atomics_variant),
};

const int test_num = ARRAY_SIZE( test_list );

test_status InitCL(cl_device_id device) {
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);

    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    if (version >= Version(3, 0))
    {
        cl_int error;
        cl_bool support_generic = CL_FALSE;
        size_t max_gvar_size = 0;

        error = clGetDeviceInfo(device, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                                sizeof(support_generic), &support_generic, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get generic address space support");
            return TEST_FAIL;
        }

        if (!support_generic)
        {
            return TEST_SKIP;
        }

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                sizeof(max_gvar_size), &max_gvar_size, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error,
                        "Unable to query CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE.");
            return TEST_FAIL;
        }

        if (!max_gvar_size)
        {
            return TEST_SKIP;
        }
    }

    return TEST_PASS;
}

/*
    Generic Address Space
    Tests for unnamed generic address space. This feature allows developers to create single generic functions
    that are able to operate on pointers from various address spaces instead of writing separate instances for every combination.
*/

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, false, InitCL);
}
