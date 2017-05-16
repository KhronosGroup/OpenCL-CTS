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
#ifndef TEST_CONFORMANCE_CLCPP_ATOMICS_ATOMIC_FETCH_HPP
#define TEST_CONFORMANCE_CLCPP_ATOMICS_ATOMIC_FETCH_HPP

#include "../common.hpp"
#include "../funcs_test_utils.hpp"


const size_t atomic_bucket_size = 100;

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
template <class func_type, class type>
std::string generate_kernel_atomic_fetch(func_type func)
{
    std::string in1_value = "input[gid]";
    std::string out1_value = "output[gid / " + std::to_string(atomic_bucket_size) + "]";
    std::string function_call = "atomic_" + func.str() + "(&" + out1_value + ", " + in1_value + ")";
    return
        "" + func.defs() +
        "__kernel void test_" + func.str() + "(global " + type_name<type>() + " *input, global atomic_" + type_name<type>() + " *output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    " + function_call + ";\n"
        "}\n";
}
#else
template <class func_type, class type>
std::string generate_kernel_atomic_fetch(func_type func)
{
    std::string in1_value = "input[gid]";
    std::string out1_value = "output[gid / " + std::to_string(atomic_bucket_size) + "]";
    std::string function_call = func.str() + "(" + in1_value + ")";
    return
        "" + func.defs() +
        "" + func.headers() +
        "#include <opencl_memory>\n"
        "#include <opencl_work_item>\n"
        "using namespace cl;\n"
        "__kernel void test_" + func.str() + "(global_ptr<" + type_name<type>() +  "[]> input,"
                                              "global_ptr<atomic<" + type_name<type>() + ">[]> output)\n"
        "{\n"
        "    size_t gid = get_global_id(0);\n"
        "    " + out1_value + "." + function_call + ";\n"
        "}\n";
}
#endif

template<class TYPE, class atomic_fetch>
bool verify_atomic_fetch(const std::vector<TYPE> &in, const std::vector<TYPE> &out, atomic_fetch op)
{
    for (size_t i = 0; i < out.size(); i++)
    {
        TYPE expected = op.init_out();
        for (size_t k = 0; k < atomic_bucket_size; k++)
        {
            const size_t in_i = i * atomic_bucket_size + k;
            if (in_i >= in.size())
                break;
            expected = op(expected, in[in_i]);
        }
        if (expected != out[i])
        {
            print_error_msg(expected, out[i], i, op);
            return false;
        }
    }
    return true;
}

template <class atomic_fetch>
int test_atomic_fetch_func(cl_device_id device, cl_context context, cl_command_queue queue, size_t count, atomic_fetch op)
{
    cl_mem buffers[2];
    cl_program program;
    cl_kernel kernel;
    size_t work_size[1];
    int err;

    typedef typename atomic_fetch::in_type TYPE;

    // Don't run test for unsupported types
    if (!(type_supported<TYPE>(device)))
    {
        return CL_SUCCESS;
    }
    if (sizeof(TYPE) == 8 &&
        (!is_extension_available(device, "cl_khr_int64_base_atomics") ||
         !is_extension_available(device, "cl_khr_int64_extended_atomics")))
    {
        return CL_SUCCESS;
    }

    std::string code_str = generate_kernel_atomic_fetch<atomic_fetch, TYPE>(op);
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

    std::vector<TYPE> input = generate_input<TYPE>(count, op.min1(), op.max1(), std::vector<TYPE>());
    std::vector<TYPE> output = generate_output<TYPE>((count - 1) / atomic_bucket_size + 1);

    buffers[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(TYPE) * input.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    buffers[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(TYPE) * output.size(), NULL, &err);
    RETURN_ON_CL_ERROR(err, "clCreateBuffer")

    err = clEnqueueWriteBuffer(
        queue, buffers[0], CL_TRUE, 0, sizeof(TYPE) * input.size(),
        static_cast<void *>(input.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueWriteBuffer")

    const TYPE pattern = op.init_out();
    err = clEnqueueFillBuffer(queue, buffers[1], &pattern, sizeof(pattern), 0, sizeof(TYPE) * output.size(), 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueFillBuffer")

    err = clSetKernelArg(kernel, 0, sizeof(buffers[0]), &buffers[0]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg")
    err = clSetKernelArg(kernel, 1, sizeof(buffers[1]), &buffers[1]);
    RETURN_ON_CL_ERROR(err, "clSetKernelArg")

    work_size[0] = count;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, work_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(err, "clEnqueueNDRangeKernel")

    err = clEnqueueReadBuffer(
        queue, buffers[1], CL_TRUE, 0, sizeof(TYPE) * output.size(),
        static_cast<void *>(output.data()), 0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(err, "clEnqueueReadBuffer")

    if (!verify_atomic_fetch(input, output, op))
    {
        RETURN_ON_ERROR_MSG(-1, "test_%s %s failed", op.str().c_str(), type_name<TYPE>().c_str());
    }
    log_info("test_%s %s passed\n", op.str().c_str(), type_name<TYPE>().c_str());

    clReleaseMemObject(buffers[0]);
    clReleaseMemObject(buffers[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return err;
}


template<class TYPE>
struct atomic_fetch
{
    typedef TYPE in_type;

    std::string decl_str()
    {
        return type_name<TYPE>();
    }

    std::string defs()
    {
        std::string defs;
        if (sizeof(TYPE) == 8)
        {
            defs += "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n";
            defs += "#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable\n";
        }
        return defs;
    }

    std::string headers()
    {
        return "#include <opencl_atomic>\n";
    }

    TYPE min1()
    {
        return 0;
    }

    TYPE max1()
    {
        return 1000;
    }
};


#define DEF_ATOMIC_FETCH_FUNC(CLASS_NAME, FUNC_NAME, HOST_FUNC_EXPRESSION, INIT_OUT) \
template<class TYPE> \
struct CLASS_NAME : public atomic_fetch<TYPE> \
{ \
    std::string str() \
    { \
        return #FUNC_NAME; \
    } \
    \
    TYPE init_out() \
    { \
        return INIT_OUT; \
    } \
    \
    TYPE operator()(const TYPE& x, const TYPE& y) \
    { \
        return HOST_FUNC_EXPRESSION; \
    } \
};

DEF_ATOMIC_FETCH_FUNC(atomic_fetch_add, fetch_add, x + y, 0)
DEF_ATOMIC_FETCH_FUNC(atomic_fetch_sub, fetch_sub, x - y, (std::numeric_limits<TYPE>::max)())

DEF_ATOMIC_FETCH_FUNC(atomic_fetch_and, fetch_and, x & y, (std::numeric_limits<TYPE>::max)())
DEF_ATOMIC_FETCH_FUNC(atomic_fetch_or,  fetch_or,  x | y, 0)
DEF_ATOMIC_FETCH_FUNC(atomic_fetch_xor, fetch_xor, x ^ y, 0)

DEF_ATOMIC_FETCH_FUNC(atomic_fetch_max, fetch_max, (std::max)(x, y), 0)
DEF_ATOMIC_FETCH_FUNC(atomic_fetch_min, fetch_min, (std::min)(x, y), (std::numeric_limits<TYPE>::max)())

#undef DEF_ATOMIC_FETCH_FUNC


AUTO_TEST_CASE(test_atomic_fetch)
(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    int error = CL_SUCCESS;
    int last_error = CL_SUCCESS;

#define TEST_ATOMIC_MACRO(TEST_CLASS) \
    last_error = test_atomic_fetch_func( \
        device, context, queue, n_elems, TEST_CLASS \
    ); \
    CHECK_ERROR(last_error) \
    error |= last_error;

    TEST_ATOMIC_MACRO((atomic_fetch_add<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_add<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_add<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_add<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_sub<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_sub<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_sub<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_sub<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_and<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_and<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_and<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_and<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_or<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_or<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_or<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_or<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_xor<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_xor<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_xor<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_xor<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_max<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_max<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_max<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_max<cl_ulong>()))

    TEST_ATOMIC_MACRO((atomic_fetch_min<cl_int>()))
    TEST_ATOMIC_MACRO((atomic_fetch_min<cl_uint>()))
    TEST_ATOMIC_MACRO((atomic_fetch_min<cl_long>()))
    TEST_ATOMIC_MACRO((atomic_fetch_min<cl_ulong>()))

#undef TEST_ATOMIC_MACRO

    if (error != CL_SUCCESS)
    {
        return -1;
    }
    return error;
}

#endif // TEST_CONFORMANCE_CLCPP_ATOMICS_ATOMIC_FETCH_HPP
