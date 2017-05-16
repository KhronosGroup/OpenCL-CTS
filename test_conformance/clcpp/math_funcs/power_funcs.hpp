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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_POWER_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_POWER_FUNCS_HPP

#include <limits>
#include <type_traits>
#include <cmath>

#include "common.hpp"

#define DEFINE_BINARY_POWER_FUNC_INT(NAME, HOST_FUNC, USE_ULP, ULP, ULP_EMBEDDED, MIN1, MAX1, MIN2, MAX2) \
struct power_func_ ## NAME : public binary_func<cl_float, cl_int, cl_float> \
{ \
    power_func_ ## NAME(bool is_embedded) : m_is_embedded(is_embedded)  \
    { \
    \
    } \
    \
    std::string str() \
    { \
        return #NAME; \
    } \
    \
    std::string headers()  \
    { \
        return "#include <opencl_math>\n"; \
    } \
    /* Reference value type is cl_double */ \
    cl_double operator()(const cl_float& x, const cl_int& y)  \
    { \
        return (HOST_FUNC)(static_cast<cl_double>(x), y); \
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
    cl_int min2() \
    { \
        return MIN2; \
    } \
    \
    cl_int max2() \
    { \
        return MAX2; \
    } \
    \
    std::vector<cl_float> in1_special_cases() \
    { \
        return {  \
            cl_float(-1.0f), \
            cl_float(0.0f), \
            cl_float(-0.0f), \
        }; \
    } \
    \
    std::vector<cl_int> in2_special_cases() \
    { \
        return {  \
            2, 3, -1, 1, -2, 2 \
        }; \
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

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(power, cbrt, std::cbrt, true, 2.0f, 4.0f, 0.001f, -1000.0f, -9.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(power, rsqrt, reference::rsqrt, true, 2.0f, 4.0f, 0.001f, 1.0f, 100.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(power, sqrt, std::sqrt, true, 3.0f, 4.0f, 0.001f, 1.0f, 100.0f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC(power, pow, std::pow, true, 16.0f, 16.0f, 0.001f, 1.0f, 100.0f, 1.0f, 10.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(power, powr, reference::powr, true, 16.0f, 16.0f, 0.001f, 1.0f, 100.0f, 1.0f, 10.0f)

// func_name, reference_func, use_ulp, ulp, ulp_for_embedded, min1, max1, min2, max2
DEFINE_BINARY_POWER_FUNC_INT(pown, std::pow, true, 16.0f, 16.0f, 1.0f, 100.0f, 1, 10)
DEFINE_BINARY_POWER_FUNC_INT(rootn, reference::rootn, true, 16.0f, 16.0f, -100.0f, 100.0f, -10, 10)

// power functions
AUTO_TEST_CASE(test_power_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{    
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // Check for EMBEDDED_PROFILE
    bool is_embedded_profile = false;
    char profile[128];
    last_error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), (void *)&profile, NULL);
    RETURN_ON_CL_ERROR(last_error, "clGetDeviceInfo")
    if (std::strcmp(profile, "EMBEDDED_PROFILE") == 0)
        is_embedded_profile = true;

    // gentype cbrt(gentype x);
    // gentype rsqrt(gentype x);
    // gentype sqrt(gentype x);
    TEST_UNARY_FUNC_MACRO((power_func_cbrt(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((power_func_sqrt(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((power_func_rsqrt(is_embedded_profile)))

    // gentype pow(gentype x, gentype y);
    // gentype powr(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((power_func_pow(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((power_func_powr(is_embedded_profile)))

    // gentype pown(gentype x, intn y);
    // gentype rootn(gentype x, intn y);
    TEST_BINARY_FUNC_MACRO((power_func_pown(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((power_func_rootn(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_POWER_FUNCS_HPP
