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
#ifndef TEST_CONFORMANCE_CLCPP_UTILS_TEST_BINARY_HPP
#define TEST_CONFORMANCE_CLCPP_UTILS_TEST_BINARY_HPP

#include <type_traits>
#include <algorithm>
#include <string>
#include <cmath>

#include "../common.hpp"

#include "detail/base_func_type.hpp"
#include "generate_inputs.hpp"
#include "compare.hpp"

template<class IN1, class IN2, class OUT1>
struct binary_func : public detail::base_func_type<OUT1>
{
    typedef IN1 in1_type;
    typedef IN2 in2_type;
    typedef OUT1 out_type;

    virtual ~binary_func() {};
    virtual std::string str() = 0;

    std::string decl_str()
    {
        return type_name<OUT1>() + "(" + type_name<IN1>() + ", " + type_name<IN2>() + ")";
    }

    bool is_in1_bool()
    {
        return false;
    }

    bool is_in2_bool()
    {
        return false;
    }

    IN1 min1()
    {
        return detail::get_min<IN1>();
    }

    IN1 max1()
    {
        return detail::get_max<IN1>();
    }

    IN2 min2()
    {
        return detail::get_min<IN2>();
    }

    IN2 max2()
    {
        return detail::get_max<IN2>();
    }

    std::vector<IN1> in1_special_cases()
    {
        return { };
    }

    std::vector<IN2> in2_special_cases()
    {
        return { };
    }

    template<class T>
    typename make_vector_type<cl_double, vector_size<T>::value>::type
    delta(const IN1& in1, const IN2& in2, const T& expected)
    {
        typedef
            typename make_vector_type<cl_double, vector_size<T>::value>::type
            delta_vector_type;
        // Take care of unused variable warning
        (void) in1;
        (void) in2;
        auto e = detail::make_value<delta_vector_type>(1e-3);
        return detail::multiply<delta_vector_type>(e, expected);
    }
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in1_type, class in2_type, class out_type>
std::string generate_kernel_binary(func_type func)
{
    std::string in1_value = "input1[gid]";
    if(func.is_in1_bool())
    {
        std::string i = vector_size<in1_type>::value == 1 ? "" : std::to_string(vector_size<in1_type>::value);
        in1_value = "(input1[gid] != (int" + i + ")(0))";
    }
    std::string in2_value = "input2[gid]";
    if(func.is_in2_bool())
    {
        std::string i = vector_size<in2_type>::value == 1 ? "" : std::to_string(vector_size<in2_type>::value);
        in2_value = "(input2[gid] != (int" + i + ")(0))";
    }
    std::string function_call = func.str() + "(" + in1_value + ", " + in2_value + ")";
    if(func.is_out_bool())
    {
        std::string i = vector_size<out_type>::value == 1 ? "" : std::to_string(vector_size<out_type>::value);
        function_call = "convert_int" + i + "(" + func.str() + "(" + in1_value + ", " + in2_value + "))";
    }
    return
        "__kernel void " + func.get_kernel_name() + "(global " + type_name<in1_type>() + " *input1,\n"
        "                                      global " + type_name<in2_type>() + " *input2,\n"
        "                                      global " + type_name<out_type>() + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#else
template <class func_type, class in1_type, class in2_type, class out_type>
std::string generate_kernel_binary(func_type func)
{
    std::string headers = func.headers();
    std::string in1_value = "input1[gid]";
    if(func.is_in1_bool())
    {
        std::string i = vector_size<in1_type>::value == 1 ? "" : std::to_string(vector_size<in1_type>::value);
        in1_value = "(input1[gid] != (int" + i + ")(0))";
    }
    std::string in2_value = "input2[gid]";
    if(func.is_in2_bool())
    {
        std::string i = vector_size<in2_type>::value == 1 ? "" : std::to_string(vector_size<in2_type>::value);
        in2_value = "(input2[gid] != (int" + i + ")(0))";
    }
    std::string function_call = func.str() + "(" + in1_value + ", " + in2_value + ")";
    if(func.is_out_bool())
    {
        std::string i = vector_size<out_type>::value == 1 ? "" : std::to_string(vector_size<out_type>::value);
        function_call = "convert_cast<int" + i + ">(" + func.str() + "(" + in1_value + ", " + in2_value + "))";
    }
    if(func.is_out_bool() || func.is_in1_bool() || func.is_in2_bool())
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
        "__kernel void " + func.get_kernel_name() + "(global_ptr<" + type_name<in1_type>() + "[]> input1,\n"
        "                                      global_ptr<" + type_name<in2_type>() + "[]> input2,\n"
        "                                      global_ptr<" + type_name<out_type>() + "[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#endif

template<class INPUT1, class INPUT2, class OUTPUT, class binary_op>
bool verify_binary(const std::vector<INPUT1> &in1,
                   const std::vector<INPUT2> &in2,
                   const std::vector<OUTPUT> &out,
                   binary_op op)
{
    for(size_t i = 0; i < in1.size(); i++)
    {
        auto expected = op(in1[i], in2[i]);
        if(!are_equal(expected, out[i], op.delta(in1[i], in2[i], expected), op))
        {
            print_error_msg(expected, out[i], i, op);
            return false;
        }
    }
    return true;
}

template <class binary_op>
int test_binary_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, binary_op op)
{
    cl_mem buffers[3];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int err;

    typedef typename binary_op::in1_type INPUT1;
    typedef typename binary_op::in2_type INPUT2;
    typedef typename binary_op::out_type OUTPUT;

    // Don't run test for unsupported types
    if(!(type_supported<INPUT1>(device)
         && type_supported<INPUT2>(device)
         && type_supported<OUTPUT>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_binary<binary_op, INPUT1, INPUT2, OUTPUT>(op);
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

    std::vector<INPUT1> in1_spec_cases = op.in1_special_cases();
    std::vector<INPUT2> in2_spec_cases = op.in2_special_cases();
    prepare_special_cases(in1_spec_cases, in2_spec_cases);
    std::vector<INPUT1> input1 = generate_input<INPUT1>(count, op.min1(), op.max1(), in1_spec_cases);
    std::vector<INPUT2> input2 = generate_input<INPUT2>(count, op.min2(), op.max2(), in2_spec_cases);
    std::vector<OUTPUT> output = generate_output<OUTPUT>(count);

    buffers[0] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(INPUT1) * input1.size(), NULL, &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    buffers[1] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(INPUT2) * input2.size(), NULL, &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    buffers[2] = clCreateBuffer(
        context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(OUTPUT) * output.size(), NULL, &err
    );
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(INPUT1) * input1.size(),
        static_cast<void *>(input1.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer")

    err = clEnqueueWriteBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(INPUT2) * input2.size(),
        static_cast<void *>(input2.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer")

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    err |= clSetKernelArg(kernel, 2, sizeof(buffers[2]), &buffers[2]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg");

    work_size[0] = count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue, buffers[2], CL_TRUE, 0, sizeof(OUTPUT) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer");

    if (!verify_binary(input1, input2, output, op))
    {
        RETURN_ON_ERROR_MSG(-1,
            "test_%s %s(%s, %s) failed", op.str().c_str(),
            type_name<OUTPUT>().c_str(), type_name<INPUT1>().c_str(), type_name<INPUT2>().c_str()
        );
    }
    log_info(
        "test_%s %s(%s, %s) passed\n", op.str().c_str(),
        type_name<OUTPUT>().c_str(), type_name<INPUT1>().c_str(), type_name<INPUT2>().c_str()
    );

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseMemObject(buffers[2]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

#endif // TEST_CONFORMANCE_CLCPP_UTILS_TEST_BINARY_HPP
