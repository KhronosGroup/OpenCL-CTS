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
#ifndef TEST_CONFORMANCE_CLCPP_REINTERPRET_AS_TYPE_HPP
#define TEST_CONFORMANCE_CLCPP_REINTERPRET_AS_TYPE_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <cstring>


template<class IN1, class OUT1>
struct as_type : public unary_func<IN1, OUT1>
{
    static_assert(sizeof(IN1) == sizeof(OUT1), "It is an error to use the as_type<T> operator to reinterpret data to a type of a different number of bytes");

    std::string str()
    {
        return "as_type";
    }

    std::string headers()
    {
        return "#include <opencl_reinterpret>\n";
    }

    OUT1 operator()(const IN1& x)
    {
        return *reinterpret_cast<const OUT1*>(&x);
    }
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in_type, class out_type>
std::string generate_kernel_as_type(func_type func)
{
    std::string in1_value = "input[gid]";
    std::string function_call = "as_" + type_name<out_type>() + "(" + in1_value + ");";
    return
        "__kernel void test_" + func.str() + "(global " + type_name<in_type>() + " *input, global " + type_name<out_type>() + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + function_call + ";\n"
        "}\n";
}
#else
template <class func_type, class in_type, class out_type>
std::string generate_kernel_as_type(func_type func)
{
    std::string headers = func.headers();
    std::string in1_value = "input[gid]";
    std::string function_call = "as_type<" + type_name<out_type>() + ">(" + in1_value + ")";
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

template<class INPUT, class OUTPUT, class as_type_op>
bool verify_as_type(const std::vector<INPUT> &in, const std::vector<OUTPUT> &out, as_type_op op)
{
    // When the operand and result type contain a different number of elements, the result is implementation-defined,
    // i.e. any result is correct
    if (vector_size<INPUT>::value == vector_size<OUTPUT>::value)
    {
        for (size_t i = 0; i < in.size(); i++)
        {
            auto expected = op(in[i]);
            if (std::memcmp(&expected, &out[i], sizeof(expected)) != 0)
            {
                print_error_msg(expected, out[i], i, op);
                return false;
            }
        }
    }
    return true;
}

template <class as_type_op>
int test_as_type_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, as_type_op op)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int error;

    typedef typename as_type_op::in_type INPUT;
    typedef typename as_type_op::out_type OUTPUT;

    // Don't run test for unsupported types
    if (!(type_supported<INPUT>(device) && type_supported<OUTPUT>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_as_type<as_type_op, INPUT, OUTPUT>(op);
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

    if (!verify_as_type(input, output, op))
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

AUTO_TEST_CASE(test_as_type)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#define TEST_AS_TYPE_MACRO(TYPE1, TYPE2) \
    last_error = test_as_type_func( \
        device, context, queue, n_elems, as_type<TYPE1, TYPE2>() \
    ); \
    CHECK_ERROR(last_error) \
    error |= last_error;

    TEST_AS_TYPE_MACRO(cl_int, cl_int)
    TEST_AS_TYPE_MACRO(cl_uint, cl_int)
    TEST_AS_TYPE_MACRO(cl_int, cl_ushort2)
    TEST_AS_TYPE_MACRO(cl_uchar, cl_uchar)
    TEST_AS_TYPE_MACRO(cl_char4, cl_ushort2)
    TEST_AS_TYPE_MACRO(cl_uchar16, cl_char16)
    TEST_AS_TYPE_MACRO(cl_short8, cl_uchar16)
    TEST_AS_TYPE_MACRO(cl_float4, cl_uint4)
    TEST_AS_TYPE_MACRO(cl_float16, cl_int16)
    TEST_AS_TYPE_MACRO(cl_long2, cl_float4)
    TEST_AS_TYPE_MACRO(cl_ulong, cl_long)
    TEST_AS_TYPE_MACRO(cl_ulong16, cl_double16)
    TEST_AS_TYPE_MACRO(cl_uchar16, cl_double2)
    TEST_AS_TYPE_MACRO(cl_ulong4, cl_short16)

#undef TEST_AS_TYPE_MACRO

    if (error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}


#endif // TEST_CONFORMANCE_CLCPP_REINTERPRET_AS_TYPE_HPP
