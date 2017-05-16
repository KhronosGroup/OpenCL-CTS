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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_HALF_MATH_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_HALF_MATH_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)  
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, cos, half_cos, std::cos, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, sin, half_sin, std::sin, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, tan, half_tan, std::tan, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp, half_exp, std::exp, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp2, half_exp2, std::exp2, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp10, half_exp10, reference::exp10, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log, half_log, std::log, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log2, half_log2, std::log2, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log10, half_log10, std::log10, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, rsqrt, half_rsqrt, reference::rsqrt, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, sqrt, half_sqrt, std::sqrt, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, recip, half_recip, reference::recip, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC1(half_math, divide, half_divide, reference::divide, true, 8192.0f, 8192.0f, 0.1f, -1024.0f, 1024.0f, -1024.0f, 1024.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC1(half_math, powr, half_powr, reference::powr, true, 8192.0f, 8192.0f, 0.1f, -1024.0f, 1024.0f, -1024.0f, 1024.0f)
#else
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, cos, half_math::cos, std::cos, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, sin, half_math::sin, std::sin, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, tan, half_math::tan, std::tan, true, 8192.0f, 8192.0f, 0.1f, -CL_M_PI_F, CL_M_PI_F)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp, half_math::exp, std::exp, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp2, half_math::exp2, std::exp2, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, exp10, half_math::exp10, reference::exp10, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log, half_math::log, std::log, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log2, half_math::log2, std::log2, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, log10, half_math::log10, std::log10, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, rsqrt, half_math::rsqrt, reference::rsqrt, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, sqrt, half_math::sqrt, std::sqrt, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

MATH_FUNCS_DEFINE_UNARY_FUNC1(half_math, recip, half_math::recip, reference::recip, true, 8192.0f, 8192.0f, 0.1f, -1000.0f, 1000.0f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC1(half_math, divide, half_math::divide, reference::divide, true, 8192.0f, 8192.0f, 0.1f, -1024.0f, 1024.0f, -1024.0f, 1024.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC1(half_math, powr, half_math::powr, reference::powr, true, 8192.0f, 8192.0f, 0.1f, -1024.0f, 1024.0f, -1024.0f, 1024.0f)
#endif

// comparison functions
AUTO_TEST_CASE(test_half_math_funcs)
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

    TEST_UNARY_FUNC_MACRO((half_math_func_cos(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_sin(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_tan(is_embedded_profile)))

    TEST_UNARY_FUNC_MACRO((half_math_func_exp(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_exp2(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_exp10(is_embedded_profile)))

    TEST_UNARY_FUNC_MACRO((half_math_func_log(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_log2(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((half_math_func_log10(is_embedded_profile)))

    TEST_BINARY_FUNC_MACRO((half_math_func_divide(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_HALF_MATH_FUNCS_HPP
