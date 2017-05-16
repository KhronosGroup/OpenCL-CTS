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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMPARISON_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMPARISON_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_BINARY_FUNC(comparison, fdim, std::fdim, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(comparison, fmax, std::fmax, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(comparison, fmin, std::fmin, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(comparison, maxmag, reference::maxmag, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(comparison, minmag, reference::minmag, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)

// comparison functions
AUTO_TEST_CASE(test_comparison_funcs)
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

    TEST_BINARY_FUNC_MACRO((comparison_func_fdim(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((comparison_func_fmax(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((comparison_func_fmin(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((comparison_func_maxmag(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((comparison_func_minmag(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_COMPARISON_FUNCS_HPP
