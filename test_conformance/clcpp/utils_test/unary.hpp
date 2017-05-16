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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_UNARY_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_UNARY_HPP

#include <type_traits>
#include <algorithm>
#include <string>
#include <cmath>

#include "../common.hpp"

#include "detail/base_func_type.hpp"
#include "generate_inputs.hpp"
#include "compare.hpp"

template<class IN1, class OUT1>
struct unary_func : public detail::base_func_type<OUT1>
{
    typedef IN1 in_type;
    typedef OUT1 out_type;

    virtual ~unary_func() {};
    virtual std::string str() = 0;

    // Return string with function type, for example: int(float).
    std::string decl_str()
    {
        return type_name<OUT1>() + "(" + type_name<IN1>() + ")";
    }

    // Return true if IN1 type in OpenCL kernel should be treated
    // as bool type; false otherwise.
    bool is_in1_bool()
    {
        return false;
    }

    // Return min value that can be used as a first argument.
    IN1 min1()
    {
        return detail::get_min<IN1>();
    }

    // Return max value that can be used as a first argument.
    IN1 max1()
    {
        return detail::get_max<IN1>();
    }

    // This returns a list of special cases input values we want to
    // test.
    std::vector<IN1> in_special_cases()
    {
        return { };
    }

    // Max error. Error should be raised if
    // abs(result - expected) > delta(.., expected)
    //
    // Default value: 0.001 * expected
    //
    // (This effects how are_equal() function works,
    // it may not have effect if verify() method in derived
    // class does not use are_equal() function.)
    //
    // Only for FP numbers/vectors
    template<class T>
    typename make_vector_type<cl_double, vector_size<T>::value>::type
    delta(const IN1& in1, const T& expected)
    {
        typedef
            typename make_vector_type<cl_double, vector_size<T>::value>::type
            delta_vector_type;
        // Take care of unused variable warning
        (void) in1;
        auto e = detail::make_value<delta_vector_type>(1e-3);
        return detail::multiply<delta_vector_type>(e, expected);
    }
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in_type, class out_type>
std::string generate_kernel_unary(func_type func)
{
    std::string in1_value = "input[gid]";
    // Convert uintN to boolN values
    if(func.is_in1_bool())
    {
        std::string i = vector_size<in_type>::value == 1 ? "" : std::to_string(vector_size<in_type>::value);
        in1_value = "(input[gid] != (int" + i + ")(0))";
    }
    std::string function_call = func.str() + "(" + in1_value + ");";
    // Convert boolN result of funtion func_type to uintN
    if(func.is_out_bool())
    {
        std::string i = vector_size<out_type>::value == 1 ? "" : std::to_string(vector_size<out_type>::value);
        function_call = "convert_int" + i + "(" + func.str() + "(" + in1_value + "))";
    }
    return
        "__kernel void " + func.get_kernel_name() + "(global " + type_name<in_type>() + " *input, global " + type_name<out_type>() + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#else
template <class func_type, class in_type, class out_type>
std::string generate_kernel_unary(func_type func)
{
    std::string headers = func.headers();
    std::string in1_value = "input[gid]";
    if(func.is_in1_bool())
    {
        std::string i = vector_size<in_type>::value == 1 ? "" : std::to_string(vector_size<in_type>::value);
        in1_value = "(input[gid] != (int" + i + ")(0))";
    }
    std::string function_call = func.str() + "(" + in1_value + ")";
    if(func.is_out_bool())
    {
        std::string i = vector_size<out_type>::value == 1 ? "" : std::to_string(vector_size<out_type>::value);
        function_call = "convert_cast<int" + i + ">(" + func.str() + "(" + in1_value + "))";
    }
    if(func.is_out_bool() || func.is_in1_bool())
    {
        if(headers.find("#include <opencl_convert>") == std::string::npos)
        {
            headers += "#include <opencl_convert>\n";
        }
    }
    return
        "" + func.defs() +
        "" + headers +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void " + func.get_kernel_name() + "(global_ptr<" + type_name<in_type>() +  "[]> input,"
                                              "global_ptr<" + type_name<out_type>() + "[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#endif

template<class INPUT, class OUTPUT, class unary_op>
bool verify_unary(const std::vector<INPUT> &in, const std::vector<OUTPUT> &out, unary_op op)
{
    for(size_t i = 0; i < in.size(); i++)
    {
        auto expected = op(in[i]);
        if(!are_equal(expected, out[i], op.delta(in[i], expected), op))
        {
            print_error_msg(expected, out[i], i, op);
            return false;
        }
    }
    return true;
}

template <class unary_op>
int test_unary_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, unary_op op)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int err;

    typedef typename unary_op::in_type INPUT;
    typedef typename unary_op::out_type OUTPUT;

    // Don't run test for unsupported types
    if(!(type_supported<INPUT>(device) && type_supported<OUTPUT>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_unary<unary_op, INPUT, OUTPUT>(op);
    std::string kernel_name = op.get_kernel_name();

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    RETURN_ON_ERROR(err)
    return err;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name, "-cl-std=CL2.0", false);
    RETURN_ON_ERROR(err)
#else
    err = create_opencl_kernel(context, &program, &kernel, code_str, kernel_name);
    RETURN_ON_ERROR(err)
#endif

    std::vector<INPUT> input = generate_input<INPUT>(count, op.min1(), op.max1(), op.in_special_cases());
    std::vector<OUTPUT> output = generate_output<OUTPUT>(count);

    buffers[0] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(INPUT) * input.size(), NULL,  &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    buffers[1] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(OUTPUT) * output.size(), NULL,  &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(INPUT) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer");

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    work_size[0] = count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(OUTPUT) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    if (!verify_unary(input, output, op))
    {
        RETURN_ON_ERROR_MSG(-1, "test_%s %s(%s) failed", op.str().c_str(), type_name<OUTPUT>().c_str(), type_name<INPUT>().c_str());
    }
    log_info("test_%s %s(%s) passed\n", op.str().c_str(), type_name<OUTPUT>().c_str(), type_name<INPUT>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_UNARY_HPP
