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
#ifndef TEST_CONFORMANCE_CLCPP_CONVERT_CONVERT_CAST_HPP
#define TEST_CONFORMANCE_CLCPP_CONVERT_CONVERT_CAST_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <functional>


enum class rounding_mode
{
    def,
    /*rte, not implemented here */
    rtz,
    rtp,
    rtn
};

enum class saturate { def, off, on };

std::string rounding_mode_name(rounding_mode rmode)
{
    switch (rmode)
    {
        case rounding_mode::rtz: return "rtz";
        case rounding_mode::rtp: return "rtp";
        case rounding_mode::rtn: return "rtn";
        default: return "";
    }
}

std::string saturate_name(saturate smode)
{
    switch (smode)
    {
        case saturate::off: return "off";
        case saturate::on:  return "on";
        default: return "";
    }
}

template<class T>
T clamp(T x, T a, T b)
{
    return (std::min)(b, (std::max)(a, x));
}

template<class IN1, class OUT1>
struct convert_cast : public unary_func<IN1, OUT1>
{
    static_assert(vector_size<IN1>::value == vector_size<OUT1>::value, "The operand and result type must have the same number of elements");

    typedef typename scalar_type<IN1>::type in_scalar_type;
    typedef typename scalar_type<OUT1>::type out_scalar_type;

    in_scalar_type in_min;
    in_scalar_type in_max;
    rounding_mode rmode;
    saturate smode;

    convert_cast(in_scalar_type min, in_scalar_type max, rounding_mode rmode, saturate smode)
        : in_min(min), in_max(max), rmode(rmode), smode(smode)
    {
    }

    std::string str()
    {
        return "convert_cast";
    }

    std::string headers()
    {
        return "#include <opencl_convert>\n";
    }

    IN1 min1()
    {
        return detail::def_limit<IN1>(in_min);
    }

    IN1 max1()
    {
        return detail::def_limit<IN1>(in_max);
    }

    OUT1 operator()(const IN1& x)
    {
        OUT1 y;
        for (size_t i = 0; i < vector_size<IN1>::value; i++)
        {
            in_scalar_type v;
            if (smode == saturate::on)
                v = clamp(x.s[i],
                    static_cast<in_scalar_type>((std::numeric_limits<out_scalar_type>::min)()),
                    static_cast<in_scalar_type>((std::numeric_limits<out_scalar_type>::max)())
                );
            else
                v = x.s[i];

            if (std::is_integral<out_scalar_type>::value)
            {
                switch (rmode)
                {
                    case rounding_mode::rtp:
                        y.s[i] = static_cast<out_scalar_type>(std::ceil(v));
                        break;
                    case rounding_mode::rtn:
                        y.s[i] = static_cast<out_scalar_type>(std::floor(v));
                        break;
                    default:
                        y.s[i] = static_cast<out_scalar_type>(v);
                }
            }
            else
            {
                y.s[i] = static_cast<out_scalar_type>(v);
            }
        }
        return y;
    }
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in_type, class out_type>
std::string generate_kernel_convert_cast(func_type func)
{
    std::string in1_value = "input[gid]";
    std::string function_call = "convert_" + type_name<out_type>();
    if (func.smode == saturate::on)
        function_call += "_sat";
    if (func.rmode != rounding_mode::def)
        function_call += "_" + rounding_mode_name(func.rmode);
    function_call += "(" + in1_value + ")";
    return
        "__kernel void test_" + func.str() + "(global " + type_name<in_type>() + " *input, global " + type_name<out_type>() + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#else
template <class func_type, class in_type, class out_type>
std::string generate_kernel_convert_cast(func_type func)
{
    std::string headers = func.headers();
    std::string in1_value = "input[gid]";
    std::string function_call = "convert_cast<" + type_name<out_type>();
    if (func.rmode != rounding_mode::def)
        function_call += ", rounding_mode::" + rounding_mode_name(func.rmode);
    if (func.smode != saturate::def)
        function_call += ", saturate::" + saturate_name(func.smode);
    function_call += ">(" + in1_value + ")";
    return
        "" + func.defs() +
        "" + headers +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_" + func.str() + "(global_ptr<" + type_name<in_type>() +  "[]> input,"
                                              "global_ptr<" + type_name<out_type>() + "[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#endif

template <class convert_cast_op>
int test_convert_cast_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, convert_cast_op op)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int error;

    typedef typename convert_cast_op::in_type INPUT;
    typedef typename convert_cast_op::out_type OUTPUT;

    // Don't run test for unsupported types
    if (!(type_supported<INPUT>(device) && type_supported<OUTPUT>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_convert_cast<convert_cast_op, INPUT, OUTPUT>(op);
    std::string kernel_name("test_"); kernel_name += op.str();

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name, "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(error)
#else
    error = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    RETURN_ON_ERROR(error)
#endif

    std::vector<INPUT> input = generate_input<INPUT>(count, op.min1(), op.max1(), op.in_special_cases());
    std::vector<OUTPUT> output = generate_output<OUTPUT>(count);

    buffers[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(INPUT) * input.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(OUTPUT) * output.size(), NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(INPUT) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueWriteBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    work_size[0] = count;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    error = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(OUTPUT) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    if (!verify_unary(input, output, op))
    {
        RETURN_ON_ERROR_MSG(-1, "test_%s %s(%s) failed", op.str().c_str(), type_name<OUTPUT>().c_str(), type_name<INPUT>().c_str());
    }
    log_info("test_%s %s(%s) passed\n", op.str().c_str(), type_name<OUTPUT>().c_str(), type_name<INPUT>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}


AUTO_TEST_CASE(test_convert_cast)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#define TEST_CONVERT_CAST_MACRO(OP) \
    last_error = test_convert_cast_func( \
        device, context, queue, n_elems, OP \
    ); \
    CHECK_ERROR(last_error) \
    error |= last_error;

    // No-op
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_float2, cl_float2>(-100.0f, +100.0f, rounding_mode::rtn, saturate::def)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_uchar2, cl_uchar2>(0, 255, rounding_mode::def, saturate::def)))

    // int to int
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_int4, cl_short4>(40000, 40000, rounding_mode::def, saturate::on)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_uchar8, cl_char8>(0, 127, rounding_mode::def, saturate::off)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_char8, cl_int8>(-100, 100, rounding_mode::def, saturate::off)))

    // float to int
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_float2, cl_uchar2>(-100.0f, +400.0f, rounding_mode::def, saturate::on)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_double4, cl_char4>(-127.0, +127.0, rounding_mode::rtp, saturate::off)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_float8, cl_uint8>(-1000.0f, +10000.0f, rounding_mode::rtp, saturate::on)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_float16, cl_ushort16>(-10000.0f, +70000.0f, rounding_mode::rtn, saturate::on)))

    // int to float
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_short8, cl_float8>(0, 12345, rounding_mode::def, saturate::def)))
    TEST_CONVERT_CAST_MACRO((convert_cast<cl_long2, cl_float2>(-1000000, +1000000, rounding_mode::rtz, saturate::def)))

#undef TEST_CONVERT_CAST_MACRO

    if (error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_CONVERT_CONVERT_CAST_HPP
