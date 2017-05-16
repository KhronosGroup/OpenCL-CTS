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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_OTHER_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_OTHER_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(other, erfc, std::erfc, true, 16.0f, 16.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(other, erf, std::erf, true, 16.0f, 16.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(other, fabs, std::fabs, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(other, tgamma, std::tgamma, true, 16.0f, 16.0f, 0.001f, -1000.0f, 1000.0f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC(other, hypot, std::hypot, true, 4.0f, 4.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2, min3, max3
MATH_FUNCS_DEFINE_TERNARY_FUNC(other, mad, reference::mad, false, 0.0f, 0.0f, 0.1f, -10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f)

// other functions
AUTO_TEST_CASE(test_other_funcs)
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

    // gentype erf(gentype x);
    // gentype erfc(gentype x);
    TEST_UNARY_FUNC_MACRO((other_func_erfc(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((other_func_erf(is_embedded_profile)))

    // gentype fabs(gentype x);
    TEST_UNARY_FUNC_MACRO((other_func_fabs(is_embedded_profile)))

    // gentype tgamma(gentype x);
    TEST_UNARY_FUNC_MACRO((other_func_tgamma(is_embedded_profile)))

    // gentype hypot(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((other_func_hypot(is_embedded_profile)))

    // gentype mad(gentype a, gentype b, gentype c);
    TEST_TERNARY_FUNC_MACRO((other_func_mad(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_OTHER_FUNCS_HPP
