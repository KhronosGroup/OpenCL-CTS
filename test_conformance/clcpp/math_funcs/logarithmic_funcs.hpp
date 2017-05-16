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
#ifndef TEST_CONFORMANCE_CLCPP_MATH_FUNCS_LOG_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_MATH_FUNCS_LOG_FUNCS_HPP

#include <type_traits>
#include <cmath>

#include "common.hpp"

namespace detail
{

// This function reads values of FP_ILOGB0 and FP_ILOGBNAN macros defined on the device.
// OpenCL C++ Spec:
// The value of FP_ILOGB0 shall be either {INT_MIN} or {INT_MAX}. The value of FP_ILOGBNAN
// shall be either {INT_MAX} or {INT_MIN}.
int get_ilogb_nan_zero(cl_device_id device, cl_context context, cl_command_queue queue, cl_int& ilogb_nan, cl_int& ilogb_zero)
{
    cl_mem buffers[1];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int err;

    std::string code_str =
        "__kernel void get_ilogb_nan_zero(__global int *out)\n"
        "{\n"
        "   out[0] = FP_ILOGB0;\n"
        "   out[1] = FP_ILOGBNAN;\n"
        "}\n";
    std::string kernel_name("get_ilogb_nan_zero");

    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name, "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)

    std::vector<cl_int> output = generate_output<cl_int>(2);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    work_size[0] = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(cl_int) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    // Save
    ilogb_zero = output[0];
    ilogb_nan = output[1];

    clReleaseMemObject(buffers[0]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

} // namespace detail

struct logarithmic_func_ilogb : public unary_func<cl_float, cl_int>
{
    logarithmic_func_ilogb(cl_int ilogb_nan, cl_int ilogb_zero)
        : m_ilogb_nan(ilogb_nan), m_ilogb_zero(ilogb_zero)
    {

    }

    std::string str()
    {
        return "ilogb";
    }

    std::string headers()
    {
        return "#include <opencl_math>\n";
    }

    cl_int operator()(const cl_float& x)
    {
        if((std::isnan)(x))
        {
            return m_ilogb_nan;
        }
        else if(x == 0.0 || x == -0.0)
        {
            return m_ilogb_zero;
        }
        static_assert(
            sizeof(cl_int) == sizeof(int),
            "Tests assumes that sizeof(cl_int) == sizeof(int)"
        );
        return (std::ilogb)(x);
    }

    cl_float min1()
    {
        return -100.0f;
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
private:
    cl_int m_ilogb_nan;
    cl_int m_ilogb_zero;
};

// gentype log(gentype x);
// gentype logb(gentype x);
// gentype log2(gentype x);
// gentype log10(gentype x);
// gentype log1p(gentype x);
// group_name, func_name, reference_func, use_ulp, ulp, ulp_for_embedded, max_delta, min1, max1
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, log, std::log, true, 3.0f, 4.0f, 0.001f, -10.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, logb, std::logb, true, 0.0f, 0.0f, 0.001f, -10.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, log2, std::log2, true, 3.0f, 4.0f, 0.001f, -10.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, log10, std::log10, true, 3.0f, 4.0f, 0.001f, -10.0f, 1000.0f)
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, log1p, std::log1p, true, 2.0f, 4.0f, 0.001f, -10.0f, 1000.0f)

// gentype lgamma(gentype x);
// OpenCL C++ Spec.:
// The ULP values for built-in math functions lgamma and lgamma_r is currently undefined.
// Because of that we don't check ULP and set acceptable delta to 0.2f (20%).
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, lgamma, std::lgamma, false, 0.0f, 0.0f, 0.2f, -10.0f, 1000.0f)

// gentype lgamma_r(gentype x, intn* signp);
// OpenCL C++ Spec.:
// The ULP values for built-in math functions lgamma and lgamma_r is currently undefined.
// Because of that we don't check ULP and set acceptable delta to 0.2f (20%).
//
// Note:
// We DO NOT test if sign of the gamma function return by lgamma_r is correct.
MATH_FUNCS_DEFINE_UNARY_FUNC(logarithmic, lgamma_r, std::lgamma, false, 0.0f, 0.0f, 0.2f, -10.0f, 1000.0f)

// We need to specialize generate_kernel_unary<>() function template for logarithmic_func_lgamma_r
// because it takes two arguments, but only one of it is input, the 2nd one is used to return
// the sign of the gamma function.
// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <>
std::string generate_kernel_unary<logarithmic_func_lgamma_r, cl_float, cl_float>(logarithmic_func_lgamma_r func)
{
    return
        "__kernel void test_lgamma_r(global float *input, global float *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    int sign;\n"
        "    output[gid] = lgamma_r(input[gid], &sign);\n"
        "}\n";
}
#else
template <>
std::string generate_kernel_unary<logarithmic_func_lgamma_r, cl_float, cl_float>(logarithmic_func_lgamma_r func)
{
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_lgamma_r(global_ptr<float[]> input, global_ptr<float[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    int sign;\n"
        "    output[gid] = lgamma_r(input[gid], &sign);\n"
        "}\n";
}
#endif

// logarithmic functions
AUTO_TEST_CASE(test_logarithmic_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

    // Check for EMBEDDED_PROFILE
    bool is_embedded_profile = false;
    char profile[128];
    error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), (void *)&profile, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")
    if (std::strcmp(profile, "EMBEDDED_PROFILE") == 0)
        is_embedded_profile = true;

    // Write values of FP_ILOGB0 and FP_ILOGBNAN, which are macros defined on the device, to
    // ilogb_zero and ilogb_nan.
    cl_int ilogb_nan = 0;
    cl_int ilogb_zero = 0;
    error = detail::get_ilogb_nan_zero(device, context, queue, ilogb_nan, ilogb_zero);
    RETURN_ON_ERROR_MSG(error, "detail::get_ilogb_nan_zero function failed");

    // intn ilogb(gentype x);
    TEST_UNARY_FUNC_MACRO((logarithmic_func_ilogb(ilogb_nan, ilogb_zero)))

    // gentype log(gentype x);
    // gentype logb(gentype x);
    // gentype log2(gentype x);
    // gentype log10(gentype x);
    // gentype log1p(gentype x);
    TEST_UNARY_FUNC_MACRO((logarithmic_func_log(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((logarithmic_func_logb(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((logarithmic_func_log2(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((logarithmic_func_log10(is_embedded_profile)))
    TEST_UNARY_FUNC_MACRO((logarithmic_func_log1p(is_embedded_profile)))

    // gentype lgamma(gentype x);
    TEST_UNARY_FUNC_MACRO((logarithmic_func_lgamma(is_embedded_profile)))

    // gentype lgamma(gentype x);
    //
    // Note:
    // We DO NOT test if sign of the gamma function return by lgamma_r is correct
    TEST_UNARY_FUNC_MACRO((logarithmic_func_lgamma_r(is_embedded_profile)))

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_MATH_FUNCS_LOG_FUNCS_HPP
