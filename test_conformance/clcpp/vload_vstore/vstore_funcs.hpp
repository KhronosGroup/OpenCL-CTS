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
#ifndef TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VSTORE_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VSTORE_FUNCS_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include <iterator>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include "common.hpp"

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in_type, class out_type, size_t N>
std::string generate_kernel_vstore(func_type func)
{
    std::string input1_type_str = type_name<in_type>();
    if(N == 3)
    {
        input1_type_str[input1_type_str.size() - 1] = '3';
    }
    std::string output1_type_str = type_name<out_type>();
    if(func.is_out_half())
    {
        output1_type_str = "half";
    }
    return
        "__kernel void test_" + func.str() + "(global " + input1_type_str + " *input, global " + output1_type_str + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    " + func.str() + std::to_string(N) + "(input[gid], gid, output);\n"
        "}\n";
}
#else
template <class func_type, class in_type, class out_type, size_t N>
std::string generate_kernel_vstore(func_type func)
{
    std::string input1_type_str = type_name<in_type>();
    if(N == 3)
    {
        input1_type_str[input1_type_str.size() - 1] = '3';
    }
    std::string output1_type_str = type_name<out_type>();
    if(func.is_out_half())
    {
        output1_type_str = "half";
    }
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_" + func.str() + "(global_ptr<" + input1_type_str +  "[]> input,"
                                              "global_ptr<" + output1_type_str + "[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    " + func.str() + "(input[gid], gid, output.get());\n"
        "}\n";
}
#endif

template<class INPUT, class OUTPUT, class vload_op>
bool verify_vstore(const std::vector<INPUT> &in, const std::vector<OUTPUT> &out, vload_op op)
{
    for(size_t i = 0; i < in.size(); i++)
    {
        auto expected = op(in[i]);
        for(size_t j = 0; j < vload_op::vector_size; j++)
        {
            size_t idx = (i * vload_op::vec_alignment) + j;
            if(!are_equal(expected.s[j], out[idx], op.delta(in[i], expected).s[j], op))
            {
                print_error_msg(expected.s[j], out[idx], idx, op);
                return false;
            }
        }
    }
    return true;
}

template <class vload_op>
int test_vstore_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, vload_op op)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int err;

    typedef typename vload_op::in_type INPUT;
    typedef typename vload_op::out_type OUTPUT;

    // Don't run test for unsupported types
    if(!(type_supported<INPUT>(device) && type_supported<OUTPUT>(device)))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_vstore<vload_op, INPUT, OUTPUT, vload_op::vector_size>(op);
    std::string kernel_name("test_"); kernel_name += op.str();

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
    std::vector<OUTPUT> output = generate_output<OUTPUT>(count * vector_size<INPUT>::value);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(INPUT) * input.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(OUTPUT) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer");

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

    if (!verify_vstore(input, output, op))
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

template <class T, cl_int N /* Vector size */>
struct vstore_func : public unary_func<
                        typename make_vector_type<T, N>::type,
                        T
                     >
{
    typedef typename make_vector_type<T, N>::type input1_type;
    typedef typename make_vector_type<T, N>::type result_type;
    const static size_t vector_size = N;
    const static size_t vec_alignment = N;

    std::string str()
    {
        return "vstore";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    result_type operator()(const input1_type& in)
    {
        static_assert(
            !is_vector_type<T>::value,
            "T must be scalar type"
        );
        return in;
    }

    bool is_out_half()
    {
        return false;
    }
};

template <cl_int N /* Vector size */>
struct vstore_half_func : public unary_func<
                            typename make_vector_type<cl_float, N>::type,
                            cl_half
                          >
{
    typedef typename make_vector_type<cl_float, N>::type input1_type;
    typedef typename make_vector_type<cl_half, N>::type result_type;
    const static size_t vector_size = N;
    const static size_t vec_alignment = N;

    std::string str()
    {
        return "vstore_half";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    result_type operator()(const input1_type& in)
    {
        result_type r;
        for(size_t i = 0; i < N; i++)
        {
            r.s[i] = float2half_rte(in.s[i]);
        }
        return r;
    }

    input1_type min1()
    {
        return detail::make_value<input1_type>(-512.f);
    }

    input1_type max1()
    {
        return detail::make_value<input1_type>(512.f);
    }

    bool is_out_half()
    {
        return true;
    }
};

template <cl_int N /* Vector size */>
struct vstorea_half_func : public unary_func<
                            typename make_vector_type<cl_float, N>::type,
                            cl_half
                          >
{
    typedef typename make_vector_type<cl_float, N>::type input1_type;
    typedef typename make_vector_type<cl_half, N>::type result_type;
    const static size_t vector_size = N;
    const static size_t vec_alignment = N == 3 ? 4 : N;

    std::string str()
    {
        return "vstorea_half";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    result_type operator()(const input1_type& in)
    {
        result_type r;
        for(size_t i = 0; i < N; i++)
        {
            r.s[i] = float2half_rte(in.s[i]);
        }
        return r;
    }

    input1_type min1()
    {
        return detail::make_value<input1_type>(-512.f);
    }

    input1_type max1()
    {
        return detail::make_value<input1_type>(512.f);
    }

    bool is_out_half()
    {
        return true;
    }
};

AUTO_TEST_CASE(test_vstore_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#define TEST_VSTORE_FUNC_MACRO(CLASS) \
    last_error = test_vstore_func( \
        device, context, queue, n_elems, CLASS \
    ); \
    CHECK_ERROR(last_error) \
    error |= last_error;

    TEST_VSTORE_FUNC_MACRO((vstore_func<cl_uint, 2>()))
    TEST_VSTORE_FUNC_MACRO((vstore_func<cl_uint, 3>()))
    TEST_VSTORE_FUNC_MACRO((vstore_func<cl_int, 4>()))
    TEST_VSTORE_FUNC_MACRO((vstore_func<cl_float, 8>()))
    TEST_VSTORE_FUNC_MACRO((vstore_func<cl_uchar, 16>()))

    TEST_VSTORE_FUNC_MACRO((vstore_half_func<2>()))
    TEST_VSTORE_FUNC_MACRO((vstore_half_func<3>()))
    TEST_VSTORE_FUNC_MACRO((vstore_half_func<4>()))
    TEST_VSTORE_FUNC_MACRO((vstore_half_func<8>()))
    TEST_VSTORE_FUNC_MACRO((vstore_half_func<16>()))

    TEST_VSTORE_FUNC_MACRO((vstorea_half_func<2>()))
    TEST_VSTORE_FUNC_MACRO((vstorea_half_func<3>()))

#undef TEST_VSTORE_FUNC_MACRO

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VSTORE_FUNCS_HPP
