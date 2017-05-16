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
#ifndef TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VLOAD_FUNCS_HPP
#define TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VLOAD_FUNCS_HPP

#include <iterator>

#include "../common.hpp"
#include "../funcs_test_utils.hpp"

#include "common.hpp"

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class in_type, class out_type, size_t N>
std::string generate_kernel_vload(func_type func)
{
    std::string input1_type_str = type_name<in_type>();
    if(func.is_in1_half())
    {
        input1_type_str = "half";
    }
    std::string output1_type_str = type_name<out_type>();
    if(N == 3)
    {
        output1_type_str[output1_type_str.size() - 1] = '3';
    }
    return
        "__kernel void test_" + func.str() + "(global " + input1_type_str + " *input, global " + output1_type_str + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    output[gid] = " + func.str() + std::to_string(N) + "(gid, input);\n"
        "}\n";
}
#else
template <class func_type, class in_type, class out_type, size_t N>
std::string generate_kernel_vload(func_type func)
{
    std::string input1_type_str = type_name<in_type>();
    if(func.is_in1_half())
    {
        input1_type_str = "half";
    }
    std::string output1_type_str = type_name<out_type>();
    if(N == 3)
    {
        output1_type_str[output1_type_str.size() - 1] = '3';
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
        "    output[gid] = " + func.str() + "<" + std::to_string(N) + ">(gid, input.get());\n"
        "}\n";
}
#endif

template<class INPUT, class OUTPUT, class vload_op>
bool verify_vload(const std::vector<INPUT> &in, const std::vector<OUTPUT> &out, vload_op op)
{
    for(size_t i = 0; i < out.size(); i++)
    {
        auto expected = op(i, in.begin());
        for(size_t j = 0; j < vload_op::vector_size; j++)
        {
            size_t idx = (i * vector_size<OUTPUT>::value) + j;
            if(!are_equal(expected.s[j], out[i].s[j], op.delta(in[idx], expected.s[j]), op))
            {
                print_error_msg(expected, out[i], i, op);
                return false;
            }
        }
    }
    return true;
}

template <class vload_op>
int test_vload_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, vload_op op)
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

    std::string code_str = generate_kernel_vload<vload_op, INPUT, OUTPUT, vload_op::vector_size>(op);
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

    std::vector<INPUT> input = vload_vstore_generate_input<INPUT>(
        count * vector_size<OUTPUT>::value, op.min1(), op.max1(), op.in_special_cases(), op.is_in1_half()
    );
    std::vector<OUTPUT> output = generate_output<OUTPUT>(count);

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

    if (!verify_vload(input, output, op))
    {
        RETURN_ON_ERROR_MSG(-1, "test_%s %s(%s) failed",
            op.str().c_str(),
            type_name<OUTPUT>().c_str(),
            type_name<INPUT>().c_str()
        );
    }
    log_info("test_%s %s(%s) passed\n", op.str().c_str(), type_name<OUTPUT>().c_str(), type_name<INPUT>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}

template <class IN1, cl_int N /* Vector size */>
struct vload_func : public unary_func<
                        IN1,
                        typename make_vector_type<IN1, N>::type /* create IN1N type */
                    >
{
    typedef typename make_vector_type<IN1, N>::type result_type;
    const static size_t vector_size = N;

    std::string str()
    {
        return "vload";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    template<class Iterator>
    result_type operator()(const size_t offset, Iterator x)
    {
        static_assert(
            !is_vector_type<IN1>::value,
            "IN1 must be scalar type"
        );
        static_assert(
            std::is_same<typename std::iterator_traits<Iterator>::value_type, IN1>::value,
            "std::iterator_traits<Iterator>::value_type must be IN1"
        );

        typedef typename std::iterator_traits<Iterator>::difference_type diff_type;

        result_type r;
        Iterator temp = x + static_cast<diff_type>(offset * N);
        for(size_t i = 0; i < N; i++)
        {
            r.s[i] = *temp;
            temp++;
        }
        return r;
    }

    bool is_in1_half()
    {
        return false;
    }
};

template <cl_int N /* Vector size */>
struct vload_half_func : public unary_func<
                            cl_half,
                            typename make_vector_type<cl_float, N>::type /* create IN1N type */
                         >
{
    typedef typename make_vector_type<cl_float, N>::type result_type;
    const static size_t vector_size = N;

    std::string str()
    {
        return "vload_half";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    template<class Iterator>
    result_type operator()(const size_t offset, Iterator x)
    {
        static_assert(
            std::is_same<typename std::iterator_traits<Iterator>::value_type, cl_half>::value,
            "std::iterator_traits<Iterator>::value_type must be cl_half"
        );

        typedef typename std::iterator_traits<Iterator>::difference_type diff_type;

        result_type r;
        Iterator temp = x + static_cast<diff_type>(offset * N);
        for(size_t i = 0; i < N; i++)
        {
            r.s[i] = half2float(*temp);
            temp++;
        }
        return r;
    }

    bool is_in1_half()
    {
        return true;
    }
};

template <cl_int N /* Vector size */>
struct vloada_half_func : public unary_func<
                            cl_half,
                            typename make_vector_type<cl_float, N>::type /* create IN1N type */
                         >
{
    typedef typename make_vector_type<cl_float, N>::type result_type;
    const static size_t vector_size = N;

    std::string str()
    {
        return "vloada_half";
    }

    std::string headers()
    {
        return "#include <opencl_vector_load_store>\n";
    }

    template<class Iterator>
    result_type operator()(const size_t offset, Iterator x)
    {
        static_assert(
            std::is_same<typename std::iterator_traits<Iterator>::value_type, cl_half>::value,
            "std::iterator_traits<Iterator>::value_type must be cl_half"
        );

        typedef typename std::iterator_traits<Iterator>::difference_type diff_type;

        result_type r;
        size_t alignment = N == 3 ? 4 : N;
        Iterator temp = x + static_cast<diff_type>(offset * alignment);
        for(size_t i = 0; i < N; i++)
        {
            r.s[i] = half2float(*temp);
            temp++;
        }
        return r;
    }

    bool is_in1_half()
    {
        return true;
    }
};

AUTO_TEST_CASE(test_vload_funcs)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#define TEST_VLOAD_FUNC_MACRO(CLASS) \
    last_error = test_vload_func( \
        device, context, queue, n_elems, CLASS \
    ); \
    CHECK_ERROR(last_error) \
    error |= last_error;

    TEST_VLOAD_FUNC_MACRO((vload_func<cl_uint,  2>()))
    TEST_VLOAD_FUNC_MACRO((vload_func<cl_float, 4>()))
    TEST_VLOAD_FUNC_MACRO((vload_func<cl_short, 8>()))
    TEST_VLOAD_FUNC_MACRO((vload_func<cl_int, 16>()))

    TEST_VLOAD_FUNC_MACRO((vload_half_func<2>()))
    TEST_VLOAD_FUNC_MACRO((vload_half_func<3>()))
    TEST_VLOAD_FUNC_MACRO((vload_half_func<4>()))
    TEST_VLOAD_FUNC_MACRO((vload_half_func<8>()))
    TEST_VLOAD_FUNC_MACRO((vload_half_func<16>()))

    TEST_VLOAD_FUNC_MACRO((vloada_half_func<2>()))
    TEST_VLOAD_FUNC_MACRO((vloada_half_func<3>()))
    TEST_VLOAD_FUNC_MACRO((vloada_half_func<4>()))
    TEST_VLOAD_FUNC_MACRO((vloada_half_func<8>()))
    TEST_VLOAD_FUNC_MACRO((vloada_half_func<16>()))

#undef TEST_VLOAD_FUNC_MACRO

    if(error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_VLOAD_VSTORE_FUNCS_VLOAD_FUNCS_HPP
