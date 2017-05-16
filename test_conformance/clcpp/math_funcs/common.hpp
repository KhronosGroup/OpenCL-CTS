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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMMON_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMMON_FUNCS_HPP

#include <cmath>
#include <limits>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include "reference.hpp"

#ifndef MATH_FUNCS_CLASS_NAME
    #define MATH_FUNCS_CLASS_NAME(x, y) x ## _func_ ## y        
#endif 

#define MATH_FUNCS_DEFINE_UNARY_FUNC1(GROUP_NAME, NAME, OCL_FUNC, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1) \
struct MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME) : public unary_func<cl_float, cl_float> \
{ \
    MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME)(bool is_embedded) : m_is_embedded(is_embedded)  \
    { \
    \
    } \
    \
    std::string str() \
    { \
        return #OCL_FUNC; \
    } \
    \
    std::string headers()  \
    { \
        return "#include <opencl_math>\n"; \
    } \
    /* Reference value type is cl_double */ \
    cl_double operator()(const cl_float& x)  \
    { \
        return (HOST_FUNC)(static_cast<cl_double>(x)); \
    } \
    \
    cl_float min1() \
    { \
        return MIN1; \
    } \
    \
    cl_float max1() \
    { \
        return MAX1; \
    } \
    \
    std::vector<cl_float> in1_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    bool use_ulp() \
    { \
        return USE_ULP; \
    } \
    \
    template<class T> \
    typename make_vector_type<cl_double, vector_size<T>::value>::type \
    delta(const cl_float& in1, const T& expected) \
    { \
        typedef  \
            typename make_vector_type<cl_double, vector_size<T>::value>::type \
            delta_vector_type; \
        (void) in1; \
        auto e = detail::make_value<delta_vector_type>(DELTA); \
        return detail::multiply<delta_vector_type>(e, expected); \
    } \
    \
    float ulp() \
    { \
        if(m_is_embedded) \
        { \
            return ULP_EMBEDDED; \
        } \
        return ULP; \
    } \
private: \
    bool m_is_embedded; \
};

#define MATH_FUNCS_DEFINE_BINARY_FUNC1(GROUP_NAME, NAME, OCL_NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2) \
struct MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME) : public binary_func<cl_float, cl_float, cl_float> \
{ \
    MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME)(bool is_embedded) : m_is_embedded(is_embedded)  \
    { \
    \
    } \
    \
    std::string str() \
    { \
        return #OCL_NAME; \
    } \
    \
    std::string headers()  \
    { \
        return "#include <opencl_math>\n"; \
    } \
    \
    cl_float operator()(const cl_float& x, const cl_float& y)  \
    { \
        return (HOST_FUNC)(x, y); \
    } \
    \
    cl_float min1() \
    { \
        return MIN1; \
    } \
    \
    cl_float max1() \
    { \
        return MAX1; \
    } \
    \
    cl_float min2() \
    { \
        return MIN2; \
    } \
    \
    cl_float max2() \
    { \
        return MAX2; \
    } \
    \
    std::vector<cl_float> in1_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    std::vector<cl_float> in2_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    template<class T> \
    typename make_vector_type<cl_double, vector_size<T>::value>::type \
    delta(const cl_float& in1, const cl_float& in2, const T& expected) \
    { \
        typedef \
            typename make_vector_type<cl_double, vector_size<T>::value>::type \
            delta_vector_type; \
        (void) in1; \
        (void) in2; \
        auto e = detail::make_value<delta_vector_type>(DELTA); \
        return detail::multiply<delta_vector_type>(e, expected); \
    } \
    \
    bool use_ulp() \
    { \
        return USE_ULP; \
    } \
    \
    float ulp() \
    { \
        if(m_is_embedded) \
        { \
            return ULP_EMBEDDED; \
        } \
        return ULP; \
    } \
private: \
    bool m_is_embedded; \
};

#define MATH_FUNCS_DEFINE_TERNARY_FUNC1(GROUP_NAME, NAME, OCL_NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2, MIN3, MAX3) \
struct MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME) : public ternary_func<cl_float, cl_float, cl_float, cl_float> \
{ \
    MATH_FUNCS_CLASS_NAME(GROUP_NAME, NAME)(bool is_embedded) : m_is_embedded(is_embedded)  \
    { \
    \
    } \
    \
    std::string str() \
    { \
        return #OCL_NAME; \
    } \
    \
    std::string headers() \
    { \
        return "#include <opencl_math>\n"; \
    } \
    \
    cl_double operator()(const cl_float& x, const cl_float& y, const cl_float& z)  \
    { \
        return (HOST_FUNC)(static_cast<cl_double>(x), static_cast<cl_double>(y), static_cast<cl_double>(z)); \
    } \
    \
    cl_float min1() \
    { \
        return MIN1; \
    } \
    \
    cl_float max1() \
    { \
        return MAX1; \
    } \
    \
    cl_float min2() \
    { \
        return MIN2; \
    } \
    \
    cl_float max2() \
    { \
        return MAX2; \
    } \
    \
    cl_float min3() \
    { \
        return MIN3; \
    } \
    \
    cl_float max3() \
    { \
        return MAX3; \
    } \
    \
    std::vector<cl_float> in1_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    std::vector<cl_float> in2_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    std::vector<cl_float> in3_special_cases() \
    { \
        return {  \
            cl_float(0.0f), \
            cl_float(-0.0f), \
            cl_float(1.0f), \
            cl_float(-1.0f), \
            cl_float(2.0f), \
            cl_float(-2.0f), \
            std::numeric_limits<cl_float>::infinity(), \
            -std::numeric_limits<cl_float>::infinity(), \
            std::numeric_limits<cl_float>::quiet_NaN() \
        }; \
    } \
    \
    template<class T> \
    typename make_vector_type<cl_double, vector_size<T>::value>::type \
    delta(const cl_float& in1, const cl_float& in2, const cl_float& in3, const T& expected) \
    { \
        typedef \
            typename make_vector_type<cl_double, vector_size<T>::value>::type \
            delta_vector_type; \
        (void) in1; \
        (void) in2; \
        (void) in3; \
        auto e = detail::make_value<delta_vector_type>(DELTA); \
        return detail::multiply<delta_vector_type>(e, expected); \
    } \
    \
    bool use_ulp() \
    { \
        return USE_ULP; \
    } \
    \
    float ulp() \
    { \
        if(m_is_embedded) \
        { \
            return ULP_EMBEDDED; \
        } \
        return ULP; \
    } \
private: \
    bool m_is_embedded; \
};

#define MATH_FUNCS_DEFINE_UNARY_FUNC(GROUP_NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1) \
    MATH_FUNCS_DEFINE_UNARY_FUNC1(GROUP_NAME, NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1)
#define MATH_FUNCS_DEFINE_BINARY_FUNC(GROUP_NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2) \
    MATH_FUNCS_DEFINE_BINARY_FUNC1(GROUP_NAME, NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2)
#define MATH_FUNCS_DEFINE_TERNARY_FUNC(GROUP_NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2, MIN3, MAX3) \
    MATH_FUNCS_DEFINE_TERNARY_FUNC1(GROUP_NAME, NAME, NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, DELTA, MIN1, MAX1, MIN2, MAX2, MIN3, MAX3)

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMMON_FUNCS_HPP
