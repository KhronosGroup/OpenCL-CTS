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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_FP_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_FP_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

// -------------- UNARY FUNCTIONS

// gentype ceil(gentype x);
// gentype floor(gentype x);
// gentype rint(gentype x);
// gentype round(gentype x);
// gentype trunc(gentype x);
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(fp, ceil, std::ceil, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(fp, floor, std::floor, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(fp, rint, std::rint, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(fp, round, std::round, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(fp, trunc, std::trunc, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f)

// floatn nan(uintn nancode);
struct fp_func_nan : public unary_func<cl_uint, cl_float>
{
    std::string str()
    {
        return "nan";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_float operator()(const cl_uint& x)
    {
        cl_uint r = x | 0x7fc00000U;
        // cl_float and cl_int have the same size so that's correct
        cl_float rf = *reinterpret_cast<cl_float*>(&r);
        return rf;
    }

    cl_uint min1()
    {
        return 0;
    }

    cl_uint max1()
    {
        return 100;
    }

    std::vector<cl_uint> in1_special_cases()
    {
        return {
            0, 1
        };
    }
};

// -------------- UNARY FUNCTIONS, 2ND ARG IS POINTER

// gentype fract(gentype x, gentype* iptr);
//
// Fuction fract() returns additional value via pointer (2nd argument). In order to test
// if it's correct output buffer type is cl_float2. In first compontent we store what
// fract() function returns, and in the 2nd component we store what is returned via its
// 2nd argument (gentype* iptr).
struct fp_func_fract : public unary_func<cl_float, cl_float2>
{
    fp_func_fract(bool is_embedded) : m_is_embedded(is_embedded)
    {

    }

    std::string str()
    {
        return "fract";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_double2 operator()(const cl_float& x)
    {
        return reference::fract(static_cast<cl_double>(x));
    }

    cl_float min1()
    {
        return -1000.0f;
    }

    cl_float max1()
    {
        return 1000.0f;
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

// We need to specialize generate_kernel_unary<>() function template for fp_func_fract.
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <>
std::string generate_kernel_unary<fp_func_fract, cl_float, cl_float2>(fp_func_fract func)
{
    return
        "__kernel void test_fract(global float *input, global float2 *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    float itpr = 0;\n"
        "    result.x = fract(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_unary<fp_func_fract, cl_float, cl_float2>(fp_func_fract func)
{
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_fract(global_ptr<float[]> input, global_ptr<float2[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    float itpr = 0;\n"
        "    result.x = fract(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#endif

// gentype modf(gentype x, gentype* iptr);
//
// Fuction modf() returns additional value via pointer (2nd argument). In order to test
// if it's correct output buffer type is cl_float2. In first compontent we store what
// modf() function returns, and in the 2nd component we store what is returned via its
// 2nd argument (gentype* iptr).
struct fp_func_modf : public unary_func<cl_float, cl_float2>
{
    fp_func_modf(bool is_embedded) : m_is_embedded(is_embedded)
    {

    }

    std::string str()
    {
        return "modf";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_double2 operator()(const cl_float& x)
    {
        cl_double2 r;
        r.s[0] = (std::modf)(static_cast<cl_double>(x), &(r.s[1]));
        return r;
    }

    cl_float min1()
    {
        return -1000.0f;
    }

    cl_float max1()
    {
        return 1000.0f;
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

// We need to specialize generate_kernel_unary<>() function template for fp_func_modf.
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <>
std::string generate_kernel_unary<fp_func_modf, cl_float, cl_float2>(fp_func_modf func)
{
    return
        "__kernel void test_modf(global float *input, global float2 *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    float itpr = 0;\n"
        "    result.x = modf(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_unary<fp_func_modf, cl_float, cl_float2>(fp_func_modf func)
{
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_modf(global_ptr<float[]> input, global_ptr<float2[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    float itpr = 0;\n"
        "    result.x = modf(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#endif

// gentype frexp(gentype x, intn* exp);
//
// Fuction frexp() returns additional value via pointer (2nd argument). In order to test
// if it's correct output buffer type is cl_float2. In first compontent we store what
// modf() function returns, and in the 2nd component we store what is returned via its
// 2nd argument (intn* exp).
struct fp_func_frexp : public unary_func<cl_float, cl_float2>
{
    fp_func_frexp(bool is_embedded) : m_is_embedded(is_embedded)
    {

    }

    std::string str()
    {
        return "frexp";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_double2 operator()(const cl_float& x)
    {
        cl_double2 r;
        cl_int e;
        r.s[0] = (std::frexp)(static_cast<cl_double>(x), &e);
        r.s[1] = static_cast<cl_float>(e);
        return r;
    }

    cl_float min1()
    {
        return -1000.0f;
    }

    cl_float max1()
    {
        return 1000.0f;
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

// We need to specialize generate_kernel_unary<>() function template for fp_func_frexp.
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <>
std::string generate_kernel_unary<fp_func_frexp, cl_float, cl_float2>(fp_func_frexp func)
{
    return
        "__kernel void test_frexp(global float *input, global float2 *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    int itpr = 0;\n"
        "    result.x = frexp(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_unary<fp_func_frexp, cl_float, cl_float2>(fp_func_frexp func)
{
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_frexp(global_ptr<float[]> input, global_ptr<float2[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    int itpr = 0;\n"
        "    result.x = frexp(input[gid], &itpr);\n"
        "    result.y = itpr;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#endif

// -------------- BINARY FUNCTIONS

// gentype copysign(gentype x, gentype y);
// gentype fmod(gentype x, gentype y);
// gentype remainder(gentype x, gentype y);
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2
MATH_FUNCS_DEFINE_BINARY_FUNC(fp, copysign, std::copysign, true, 0.0f, 0.0f, 0.001f, -100.0f, 100.0f, -10.0f, 10.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(fp, fmod, std::fmod, true, 0.0f, 0.0f, 0.001f, -100.0f, 100.0f, -10.0f, 10.0f)
MATH_FUNCS_DEFINE_BINARY_FUNC(fp, remainder, std::remainder, true, 0.0f, 0.001f, 0.0f, -100.0f, 100.0f, -10.0f, 10.0f)

// In case of function float nextafter(float, float) reference function must
// operate on floats and return float.
struct fp_func_nextafter : public binary_func<cl_float, cl_float, cl_float>
{
    fp_func_nextafter(bool is_embedded) : m_is_embedded(is_embedded)
    {

    }

    std::string str()
    {
        return "nextafter";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    /* In this case reference value type MUST BE cl_float */
    cl_float operator()(const cl_float& x, const cl_float& y)
    {
        return (std::nextafter)(x, y);
    }

    cl_float min1()
    {
        return -1000.0f;
    }

    cl_float max1()
    {
        return 500.0f;
    }

    cl_float min2()
    {
        return 501.0f;
    }

    cl_float max2()
    {
        return 1000.0f;
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

    std::vector<cl_float> in2_special_cases()
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

// gentype remquo(gentype x, gentype y, intn* quo);
struct fp_func_remquo : public binary_func<cl_float, cl_float, cl_float2>
{
    fp_func_remquo(bool is_embedded) : m_is_embedded(is_embedded)
    {

    }

    std::string str()
    {
        return "remquo";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_double2 operator()(const cl_float& x, const cl_float& y)
    {
        return reference::remquo(static_cast<cl_double>(x), static_cast<cl_double>(y));
    }

    cl_float min1()
    {
        return -1000.0f;
    }

    cl_float max1()
    {
        return 1000.0f;
    }

    cl_float min2()
    {
        return -1000.0f;
    }

    cl_float max2()
    {
        return 1000.0f;
    }

    std::vector<cl_float> in1_special_cases()
    {
        return {
            cl_float(0.0f),
            cl_float(-0.0f),
            cl_float(1.0f),
            cl_float(-1.0f),
            std::numeric_limits<cl_float>::infinity(),
            -std::numeric_limits<cl_float>::infinity(),
            std::numeric_limits<cl_float>::quiet_NaN()
        };
    }

    std::vector<cl_float> in2_special_cases()
    {
        return {
            cl_float(0.0f),
            cl_float(-0.0f),
            cl_float(1.0f),
            cl_float(-1.0f),
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


// We need to specialize generate_kernel_binary<>() function template for fp_func_remquo.
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <>
std::string generate_kernel_binary<fp_func_remquo, cl_float, cl_float, cl_float2>(fp_func_remquo func)
{
    return
        "__kernel void test_remquo(global float *input1, global float *input2, global float2 *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    int quo = 0;\n"
        "    int sign = 0;\n"
        "    result.x = remquo(input1[gid], input2[gid], &quo);\n"
        // Specification say:
        // "remquo also calculates the lower seven bits of the integral quotient x/y,
        // and gives that value the same sign as x/y. It stores this signed value in
        // the object pointed to by quo."
        // Implemenation may save into quo more than seven bits. We need to take
        // care of that here.
        "    sign = (quo < 0) ? -1 : 1;\n"
        "    quo = (quo < 0) ? -quo : quo;\n"
        "    quo &= 0x0000007f;\n"
        "    result.y = (sign < 0) ? -quo : quo;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_binary<fp_func_remquo, cl_float, cl_float, cl_float2>(fp_func_remquo func)
{
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_remquo(global_ptr<float[]> input1, global_ptr<float[]> input2, global_ptr<float2[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    float2 result;\n"
        "    int quo = 0;\n"
        "    int sign = 0;\n"
        "    result.x = remquo(input1[gid], input2[gid], &quo);\n"
        // Specification say:
        // "remquo also calculates the lower seven bits of the integral quotient x/y,
        // and gives that value the same sign as x/y. It stores this signed value in
        // the object pointed to by quo."
        // Implemenation may save into quo more than seven bits. We need to take
        // care of that here.
        "    sign = (quo < 0) ? -1 : 1;\n"
        "    quo = (quo < 0) ? -quo : quo;\n"
        "    quo &= 0x0000007f;\n"
        "    result.y = (sign < 0) ? -quo : quo;\n"
        "    output[gid] = result;\n"
        "}\n";
}
#endif

// -------------- TERNARY FUNCTIONS

// gentype fma(gentype a, gentype b, gentype c);
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1, min2, max2, min3, max3
MATH_FUNCS_DEFINE_TERNARY_FUNC(fp, fma, std::fma, true, 0.0f, 0.0f, 0.001f, -1000.0f, 1000.0f, -1000.0f, 1000.0f, -1000.0f, 1000.0f)

// floating point functions
AUTO_TEST_CASE(test_fp_funcs)
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

    // gentype ceil(gentype x);
    TEST_UNARY_FUNC_MACRO((fp_func_ceil(is_embedded_profile)))
    // gentype floor(gentype x);
    TEST_UNARY_FUNC_MACRO((fp_func_floor(is_embedded_profile)))
    // gentype rint(gentype x);
    TEST_UNARY_FUNC_MACRO((fp_func_rint(is_embedded_profile)))
    // gentype round(gentype x);
    TEST_UNARY_FUNC_MACRO((fp_func_round(is_embedded_profile)))
    // gentype trunc(gentype x);
    TEST_UNARY_FUNC_MACRO((fp_func_trunc(is_embedded_profile)))

    // floatn nan(uintn nancode);
    TEST_UNARY_FUNC_MACRO((fp_func_nan()))

    // gentype fract(gentype x, gentype* iptr);
    TEST_UNARY_FUNC_MACRO((fp_func_fract(is_embedded_profile)))
    // gentype modf(gentype x, gentype* iptr);
    TEST_UNARY_FUNC_MACRO((fp_func_modf(is_embedded_profile)))
    // gentype frexp(gentype x, intn* exp);
    TEST_UNARY_FUNC_MACRO((fp_func_frexp(is_embedded_profile)))

    // gentype remainder(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((fp_func_remainder(is_embedded_profile)))
    // gentype copysign(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((fp_func_copysign(is_embedded_profile)))
    // gentype fmod(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((fp_func_fmod(is_embedded_profile)))

    // gentype nextafter(gentype x, gentype y);
    TEST_BINARY_FUNC_MACRO((fp_func_nextafter(is_embedded_profile)))

    // gentype remquo(gentype x, gentype y, intn* quo);
    TEST_BINARY_FUNC_MACRO((fp_func_remquo(is_embedded_profile)))

    // gentype fma(gentype a, gentype b, gentype c);
    TEST_TERNARY_FUNC_MACRO((fp_func_fma(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_FP_FUNCS_HPP
