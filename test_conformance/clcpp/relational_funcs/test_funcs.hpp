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
#ifndef TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_TEST_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_TEST_FUNCS_HPP

#include "common.hpp"

// This marco creates a class wrapper for unary test function we want to test.
#define DEF_UNARY_TEST_FUNC(CLASS_NAME, FUNC_NAME, HOST_FUNC_EXPRESSION) \
template <cl_int N /* Vector size */> \
struct CLASS_NAME : public unary_func< \
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
    result_type operator()(const input_type& x) \
    {    \
        typedef typename scalar_type<input_type>::type SCALAR; \
        return perform_function<input_type, result_type>( \
            x, \
            [](const SCALAR& a) \
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
    std::vector<input_type> in1_special_cases() \
    { \
        typedef typename scalar_type<input_type>::type SCALAR; \
        return {  \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(-std::numeric_limits<SCALAR>::infinity()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::quiet_NaN()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::signaling_NaN()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::denorm_min()), \
            detail::make_value<input_type>(0.0f), \
            detail::make_value<input_type>(-0.0f) \
        }; \
    } \
};

// This marco creates a class wrapper for binary test function we want to test.
#define DEF_BINARY_TEST_FUNC(CLASS_NAME, FUNC_NAME, HOST_FUNC_EXPRESSION) \
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
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::signaling_NaN()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::denorm_min()), \
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
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::signaling_NaN()), \
            detail::make_value<input_type>(std::numeric_limits<SCALAR>::denorm_min()), \
            detail::make_value<input_type>(0.0f), \
            detail::make_value<input_type>(-0.0f) \
        }; \
    } \
};

DEF_UNARY_TEST_FUNC(test_func_isfinite, isfinite, (std::isfinite)(a))
DEF_UNARY_TEST_FUNC(test_func_isinf, isinf, (std::isinf)(a))
DEF_UNARY_TEST_FUNC(test_func_isnan, isnan, (std::isnan)(a))
DEF_UNARY_TEST_FUNC(test_func_isnormal, isnormal, (std::isnormal)(a))
DEF_UNARY_TEST_FUNC(test_func_signbit, signbit , (std::signbit)(a))

DEF_BINARY_TEST_FUNC(test_func_isordered, isordered, !(std::isunordered)(a, b))
DEF_BINARY_TEST_FUNC(test_func_isunordered, isunordered, (std::isunordered)(a, b))

#undef DEF_UNARY_TEST_FUNC
#undef DEF_BINARY_TEST_FUNC

template <cl_int N /* Vector size */>
struct test_func_all : public unary_func<
                                    typename make_vector_type<cl_int, N>::type, /* create cl_intN type */
                                    cl_int /* create cl_intN type */
                                 >
{
    typedef typename make_vector_type<cl_int, N>::type input_type;
    typedef cl_int result_type;

    std::string str()
    {
        return "all";
    }

    std::string headers()
    {
        return "#include <opencl_relational>\n";
    }

    result_type operator()(const input_type& x)
    {
        return perform_all_function(x);
    }

    bool is_out_bool()
    {
        return true;
    }

    bool is_in1_bool()
    {
        return true;
    }

    std::vector<input_type> in1_special_cases()
    {
        return {
            detail::make_value<input_type>(0),
            detail::make_value<input_type>(1),
            detail::make_value<input_type>(12),
            detail::make_value<input_type>(-12)
        };
    }
};

template <cl_int N /* Vector size */>
struct test_func_any : public unary_func<
                                    typename make_vector_type<cl_int, N>::type, /* create cl_intN type */
                                    cl_int /* create cl_intN type */
                                 >
{
    typedef typename make_vector_type<cl_int, N>::type input_type;
    typedef cl_int result_type;

    std::string str()
    {
        return "any";
    }

    std::string headers()
    {
        return "#include <opencl_relational>\n";
    }

    result_type operator()(const input_type& x)
    {
        return perform_any_function(x);
    }

    bool is_out_bool()
    {
        return true;
    }

    bool is_in1_bool()
    {
        return true;
    }

    std::vector<input_type> in1_special_cases()
    {
        return {
            detail::make_value<input_type>(0),
            detail::make_value<input_type>(1),
            detail::make_value<input_type>(12),
            detail::make_value<input_type>(-12)
        };
    }
};

AUTO_TEST_CASE(test_relational_test_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

// Helper macro, so we don't have to repreat the same code.
#define TEST_UNARY_REL_FUNC_MACRO(CLASS_NAME) \
    TEST_UNARY_FUNC_MACRO(CLASS_NAME<1>()) \
    TEST_UNARY_FUNC_MACRO(CLASS_NAME<2>()) \
    TEST_UNARY_FUNC_MACRO(CLASS_NAME<4>()) \
    TEST_UNARY_FUNC_MACRO(CLASS_NAME<8>()) \
    TEST_UNARY_FUNC_MACRO(CLASS_NAME<16>())

    TEST_UNARY_REL_FUNC_MACRO(test_func_isfinite)
    TEST_UNARY_REL_FUNC_MACRO(test_func_isinf)
    TEST_UNARY_REL_FUNC_MACRO(test_func_isnan)
    TEST_UNARY_REL_FUNC_MACRO(test_func_isnormal)
    TEST_UNARY_REL_FUNC_MACRO(test_func_signbit)

// Tests for all(booln x) and any(booln x) are not run in USE_OPENCLC_KERNELS mode,
// because those functions in OpenCL C require different reference functions on host
// compared to their equivalents from OpenCL C++.
// (In OpenCL C those functions returns true/false based on the most significant bits
// in any/all component/s of x)
#ifndef USE_OPENCLC_KERNELS
    TEST_UNARY_REL_FUNC_MACRO(test_func_all)
    TEST_UNARY_REL_FUNC_MACRO(test_func_any)
#else
    log_info("WARNING:\n\tTests for bool all(booln x) are not run in USE_OPENCLC_KERNELS mode\n");
    log_info("WARNING:\n\tTests for bool any(booln x) are not run in USE_OPENCLC_KERNELS mode\n");
#endif

#undef TEST_UNARY_REL_FUNC_MACRO

#define TEST_BINARY_REL_FUNC_MACRO(CLASS_NAME) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<1>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<2>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<4>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<8>()) \
    TEST_BINARY_FUNC_MACRO(CLASS_NAME<16>())

    TEST_BINARY_REL_FUNC_MACRO(test_func_isordered)
    TEST_BINARY_REL_FUNC_MACRO(test_func_isunordered)

#undef TEST_BINARY_REL_FUNC_MACRO

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_RELATIONAL_FUNCS_TEST_FUNCS_HPP
