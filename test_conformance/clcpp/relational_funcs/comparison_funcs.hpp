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
#ifndef TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMPARISON_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMPARISON_FUNCS_HPP

#include "common.hpp"

// This marco creates a class wrapper for comparision function we want to test.
#define DEF_COMPARISION_FUNC(CLASS_NAME, FUNC_NAME, HOST_FUNC_EXPRESSION) \
template <cl_int N /* Vector size */> \
struct CLASS_NAME : public binary_func< \
                                    typename make_vector_type<cl_float, N>::type, /* create cl_floatN type */ \
                                    typename make_vector_type<cl_float, N>::type, /* create cl_floatN type */ \
                                    typename make_vector_type<cl_int, N>::type /* create cl_intN type */ \
                                 > \
{ \
    typedef typename make_vector_type<cl_float, N>::type input_type; \
    typedef typename make_vector_type<cl_int, N>::type result_type; \
    \
    std::string str() \
    { \
        return #FUNC_NAME; \
    } \
    \
    std::string headers() \
    { \
        return "#include <opencl_relational>\n"; \
    } \
    \
    result_type operator()(const input_type& x, const input_type& y) \
    {    \
        typedef typename scalar_type<input_type>::type SCALAR; \
        return perform_function<input_type, input_type, result_type>( \
            x, y, \
            [](const SCALAR& a, const SCALAR& b) \
            { \
                if(HOST_FUNC_EXPRESSION) \
                { \
                    return cl_int(1); \
                } \
                return cl_int(0); \
            } \
        ); \
    } \
    \
    bool is_out_bool() \
    { \
        return true; \
    } \
    \
    input_type min1() \
    { \
        return detail::def_limit<input_type>(-10000.0f); \
    } \
    \
    input_type max1() \
    { \
        return detail::def_limit<input_type>(10000.0f); \
    } \
    \
    input_type min2() \
    { \
        return detail::def_limit<input_type>(-10000.0f); \
    } \
    \
    input_type max2() \
    { \
        return detail::def_limit<input_type>(10000.0f); \
    } \
    \
    std::vector<input_type> in1_special_cases() \
    { \
        typedef typename scalar_type<input_type>::type SCALAR; \
        return {  \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(-std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::quiet_NaN()), \
            detail::make_value<input_type>(0.0f), \
            detail::make_value<input_type>(-0.0f) \
        }; \
    } \
    \
    std::vector<input_type> in2_special_cases() \
    { \
        typedef typename scalar_type<input_type>::type SCALAR; \
        return {  \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(-std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::quiet_NaN()), \
            detail::make_value<input_type>(0.0f), \
            detail::make_value<input_type>(-0.0f) \
        }; \
    } \
};

DEF_COMPARISION_FUNC(comparison_func_isequal, isequal, (a == b))
DEF_COMPARISION_FUNC(comparison_func_isnotequal, isnotequal, !(a == b))
DEF_COMPARISION_FUNC(comparison_func_isgreater, isgreater, (std::isgreater)(a, b))
DEF_COMPARISION_FUNC(comparison_func_isgreaterequal, isgreaterequal, ((std::isgreater)(a, b) || a == b))
DEF_COMPARISION_FUNC(comparison_func_isless, isless, (std::isless)(a, b))
DEF_COMPARISION_FUNC(comparison_func_islessequal, islessequal, ((std::isless)(a, b) || a == b))
DEF_COMPARISION_FUNC(comparison_func_islessgreater, islessgreater, ((a < b) || (a > b)))

#undef DEF_COMPARISION_FUNC

AUTO_TEST_CASE(test_relational_comparison_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

// Helper macro, so we don't have to repreat the same code.  
#define TEST_BINARY_REL_FUNC_MACRO(CLASS_NAME) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<1>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<2>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<4>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<8>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<16>())

    TEST_BINARY_REL_FUNC_MACRO(comparison_func_isequal)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_isnotequal)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_isgreater)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_isgreaterequal)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_isless)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_islessequal)
    TEST_BINARY_REL_FUNC_MACRO(comparison_func_islessgreater)

#undef TEST_BINARY_REL_FUNC_MACRO

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_COMPARISON_FUNCS_HPP
