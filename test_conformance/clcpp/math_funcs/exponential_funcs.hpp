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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_EXP_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_EXP_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(exponential, exp, std::exp, true, 3.0f, 4.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(exponential, expm1, std::expm1, true, 3.0f, 4.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(exponential, exp2, std::exp2, true, 3.0f, 4.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(exponential, exp10, reference::exp10, true, 3.0f, 4.0f, 0.001f, -1000.0f, 1000.0f)

struct exponential_func_ldexp : public binary_func<cl_float, cl_int, cl_float>
{
    exponential_func_ldexp(bool is_embedded) : m_is_embedded(is_embedded) 
    {
   
    }
   
    std::string str()
    {
        return "ldexp";
    }
   
    std::string headers() 
    {
        return "#include <opencl_math>\n";
    }

    /* Reference value type is cl_double */
    cl_double operator()(const cl_float& x, const cl_int& y) 
    {
        return (std::ldexp)(static_cast<cl_double>(x), y);
    }
   
    cl_float min1()
    {
        return -1000.0f;
    }
   
    cl_float max1()
    {
        return 1000.0f;
    }

    cl_int min2()
    {
        return -8;
    }
   
    cl_int max2()
    {
        return 8;
    }
   
    std::vector<cl_float> in1_special_cases()
    {
        return { 
            cl_float(0.0f),
            cl_float(-0.0f),
            cl_float(1.0f),
            cl_float(-1.0f),
            cl_float(2.0f),
            cl_float(-2.0f),
            std::numeric_limits<cl_float>::infinity(),
            -std::numeric_limits<cl_float>::infinity(),
            std::numeric_limits<cl_float>::quiet_NaN()
        };
    }
   
    bool use_ulp()
    {
        return true;
    }
   
    float ulp()
    {
        if(m_is_embedded)
        {
            return 0.0f;
        }
        return 0.0f;
    }
private:
    bool m_is_embedded;
};

// exponential functions
AUTO_TEST_CASE(test_exponential_funcs)
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

    // auto exp(gentype x);
    // auto expm1(gentype x);
    // auto exp2(gentype x);
    // auto exp10(gentype x);
    TEST_UNARY_FUNC_MACRO((exponential_func_exp(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((exponential_func_expm1(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((exponential_func_exp2(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((exponential_func_exp10(is_embedded_profile)))

    // auto ldexp(gentype x, intn k);
    TEST_BINARY_FUNC_MACRO((exponential_func_ldexp(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_EXP_FUNCS_HPP
