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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_TRI_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_TRI_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, acos, std::acos, true, 4.0f, 4.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, acosh, std::acosh, true, 4.0f, 4.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, acospi, reference::acospi, true, 5.0f, 5.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, asin, std::asin, true, 4.0f, 4.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, asinh, std::asinh, true, 4.0f, 4.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, asinpi, reference::asinpi, true, 5.0f, 5.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, atan, std::atan, true, 5.0f, 5.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, atanh, std::atanh, true, 5.0f, 5.0f, 0.001f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, atanpi, reference::atanpi, true, 5.0f, 5.0f, 0.001f, -1.0f, 1.0f)

// For (sin/cos/tan)pi functions min input value is -0.24 and max input value is 0.24,
// so (CL_M_PI * x) is never greater than CL_M_PI_F.
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, cos, std::cos, true, 4.0f, 4.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, cosh, std::cosh, true, 4.0f, 4.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, cospi, reference::cospi, true, 4.0f, 4.0f, 0.001f, -0.24, -0.24f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, sin, std::sin, true, 4.0f, 4.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, sinh, std::sinh, true, 4.0f, 4.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, sinpi, reference::sinpi, true, 4.0f, 4.0f, 0.001f, -0.24, -0.24f)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, tan, std::tan, true, 5.0f, 5.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, tanh, std::tanh, true, 5.0f, 5.0f, 0.001f, -CL_M_PI_F, CL_M_PI_F)
MATH_FUNCS_DEFINE_UNARY_FUNC(trigonometric, tanpi, reference::tanpi, true, 6.0f, 6.0f, 0.001f, -0.24, -0.24f)

// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC(trigonometric, atan2, reference::atan2, true, 6.0f, 6.0f, 0.001f, -1.0f, 1.0f, -1.0f, 1.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(trigonometric, atan2pi, reference::atan2pi, true, 6.0f, 6.0f, 0.001f, -1.0f, 1.0f, -1.0f, 1.0f)

// gentype sincos(gentype x, gentype * cosval);
//
// Fact that second argument is a pointer is inconvenient.
//
// We don't want to modify all helper functions defined in funcs_test_utils.hpp
// that run test kernels generated based on this class and check if results are
// correct, so instead of having two output cl_float buffers, one for sines and
// one for cosines values, we use one cl_float2 output buffer (first component is
// sine, second is cosine).
//
// Below we also define specialization of generate_kernel_unary function template
// for trigonometric_func_sincos.
struct trigonometric_func_sincos : public unary_func<cl_float, cl_float2>
{
    trigonometric_func_sincos(bool is_embedded) : m_is_embedded(is_embedded) 
    {

    }

    std::string str()
    {
        return "sincos";
    }

    std::string headers() 
    {
        return "#include <opencl_math>\n";
    }

    /* Reference value type is cl_double */
    cl_double2 operator()(const cl_float& x) 
    {
        return (reference::sincos)(static_cast<cl_double>(x));
    }

    cl_float min1()
    {
        return -CL_M_PI_F;
    }

    cl_float max1()
    {
        return CL_M_PI_F;
    }

    bool use_ulp()
    {
        return true;
    }

    float ulp()
    {
        if(m_is_embedded)
        {
            return 4.0f;
        }
        return 4.0f;
    }
private:
    bool m_is_embedded;
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)    
template <>
std::string generate_kernel_unary<trigonometric_func_sincos, cl_float, cl_float2>(trigonometric_func_sincos func)
{    
    return 
        "__kernel void test_sincos(global float *input, global float2 *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 sine_cosine_of_x;\n"
        "    float cosine_of_x = 0;\n"
        "    sine_cosine_of_x.x = sincos(input[gid], &(cosine_of_x));\n"
        "    sine_cosine_of_x.y = cosine_of_x;\n"
        "    output[gid] = sine_cosine_of_x;\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_unary<trigonometric_func_sincos, cl_float, cl_float2>(trigonometric_func_sincos func)
{
    return         
        "" + func.defs() + 
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_sincos(global_ptr<float[]> input, global_ptr<float2[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 sine_cosine_of_x;\n"
        "    float cosine_of_x = 0;\n"
        "    sine_cosine_of_x.x = sincos(input[gid], &(cosine_of_x));\n"
        "    sine_cosine_of_x.y = cosine_of_x;\n"
        "    output[gid] = sine_cosine_of_x;\n"
        "}\n";
}
#endif

// trigonometric functions
AUTO_TEST_CASE(test_trigonometric_funcs)
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

    // gentype acos(gentype x);
    // gentype acosh(gentype x);
    // gentype acospi(gentype x);
    // gentype asin(gentype x);
    // gentype asinh(gentype x);
    // gentype asinpi(gentype x);
    // gentype atan(gentype x);
    // gentype atanh(gentype x);
    // gentype atanpi(gentype x);
    TEST_UNARY_FUNC_MACRO((trigonometric_func_acos(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_acosh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_acospi(is_embedded_profile))) 
    TEST_UNARY_FUNC_MACRO((trigonometric_func_asin(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_asinh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_asinpi(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_atan(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_atanh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_atanpi(is_embedded_profile)))

    // gentype cos(gentype x);
    // gentype cosh(gentype x);
    // gentype cospi(gentype x);
    // gentype sin(gentype x);
    // gentype sinh(gentype x);
    // gentype sinpi(gentype x);
    // gentype tan(gentype x);
    // gentype tanh(gentype x);
    // gentype tanpi(gentype x);
    TEST_UNARY_FUNC_MACRO((trigonometric_func_cos(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_cosh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_cospi(is_embedded_profile))) 
    TEST_UNARY_FUNC_MACRO((trigonometric_func_sin(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_sinh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_sinpi(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_tan(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_tanh(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((trigonometric_func_tanpi(is_embedded_profile)))

    // gentype atan2(gentype y, gentype x);
    // gentype atan2pi(gentype y, gentype x);
    TEST_BINARY_FUNC_MACRO((trigonometric_func_atan2(is_embedded_profile)))
    TEST_BINARY_FUNC_MACRO((trigonometric_func_atan2pi(is_embedded_profile)))

    // gentype sincos(gentype x, gentype * cosval);
    TEST_UNARY_FUNC_MACRO((trigonometric_func_sincos(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }    
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_TRI_FUNCS_HPP
