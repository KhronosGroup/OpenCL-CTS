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
#ifndef TEST_CONFORMANCE_CLCPP_FUNCS_TEST_UTILS_HPP
#define TEST_CONFORMANCE_CLCPP_FUNCS_TEST_UTILS_HPP

// This file contains helper classes and functions for testing various unary, binary
// and ternary OpenCL functions (for example cl::abs(x) or cl::abs_diff(x, y)), 
// as well as other helper functions/classes.

#include "common.hpp"

#define TEST_UNARY_FUNC_MACRO(TEST_CLASS) \
    last_error = test_unary_func(  \
        device, context, queue, n_elems, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

#define TEST_BINARY_FUNC_MACRO(TEST_CLASS) \
    last_error = test_binary_func(  \
        device, context, queue, n_elems, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

#define TEST_TERNARY_FUNC_MACRO(TEST_CLASS) \
    last_error = test_ternary_func(  \
        device, context, queue, n_elems, TEST_CLASS \
    );  \
    CHECK_ERROR(last_error) \
    error |= last_error;

#include "utils_test/compare.hpp"
#include "utils_test/generate_inputs.hpp"

// HOWTO:
//
// unary_func, binary_func, ternary_func - base classes wrapping OpenCL functions that
// you want to test.
// 
// To create a wrapper class for given function, you need to create a class derived from correct
// base class (unary_func, binary_func, ternary_func), and define:
//
// * std::string str() method which should return class name in OpenCL ("abs", "abs_diff"),
// * operator(x), operator(x, y) or operator(x,y,z) depending on arity of the function you wish
// to test, method should work exactly as the tested function works in OpenCL
// * if it's needed you can overload min1, max1, min2, max2, min3, max3 methods with returns min 
// and max values that can be generated for given input (function argument) [required for vec 
// arguments],
// * if you want to use vector arguments (for example: cl_int2, cl_ulong16), you should look at
// how int_func_clamp<> is implemented in integer_funcs/numeric_funcs.hpp.
//
// To see how you should use class you've just created see AUTO_TEST_CASE(test_int_numeric_funcs)
// in integer_funcs/numeric_funcs.hpp.
#include "utils_test/unary.hpp"
#include "utils_test/binary.hpp"
#include "utils_test/ternary.hpp"

#endif // TEST_CONFORMANCE_CLCPP_FUNCS_TEST_UTILS_HPP
