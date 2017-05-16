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
#ifndef TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_IVDEP_HPP
#define TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_IVDEP_HPP

#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_ivdep {

enum class loop_kind
{
    for_loop,
    while_loop,
    do_loop
};

struct test_options
{
    loop_kind loop;
    int ivdep_length;
    int offset1;
    int offset2;
    int iter_count;
    bool offset1_param;
    bool offset2_param;
    bool iter_count_param;
    bool cond_in_header;
    bool init_in_header;
    bool incr_in_header;
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_source(test_options options)
{
    std::string offset1s = options.offset1_param ? "offset1" : std::to_string(options.offset1);
    std::string offset2s = options.offset2_param ? "offset2" : std::to_string(options.offset2);

    std::string init = "i = 0";
    std::string cond = std::string("i < ") + (options.iter_count_param ? "iter_count" : std::to_string(options.iter_count));
    std::string incr = "i += 2";

    std::stringstream s;
    s << R"(
    kernel void test(global int *a, global int *b, global int *c, int offset1, int offset2, int iter_count)
    {
        int i;
    )";

    // Loop #1
    if (!options.init_in_header) s << init << ";" << std::endl;
    if (options.loop == loop_kind::for_loop)
        s << "for (" <<
            (options.init_in_header ? init : "") << ";" <<
            (options.cond_in_header ? cond : "") << ";" <<
            (options.incr_in_header ? incr : "") << ")";
    else if (options.loop == loop_kind::while_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ")";
    else if (options.loop == loop_kind::do_loop)
        s << "do";
    s << "{" << std::endl;
    if (!options.cond_in_header) s << "if (!(" << cond << ")) break;" << std::endl;
    s << "a[i + " << offset1s << "] = b[i + " << offset1s << "] * c[i + " << offset1s << "];" << std::endl;
    if (!options.incr_in_header) s << incr << ";" << std::endl;
    s << "}" << std::endl;
    if (options.loop == loop_kind::do_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ");" << std::endl;

    // Loop #2
    if (!options.init_in_header) s << init << ";" << std::endl;
    if (options.loop == loop_kind::for_loop)
        s << "for (" <<
            (options.init_in_header ? init : "") << ";" <<
            (options.cond_in_header ? cond : "") << ";" <<
            (options.incr_in_header ? incr : "") << ")";
    else if (options.loop == loop_kind::while_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ")";
    else if (options.loop == loop_kind::do_loop)
        s << "do";
    s << "{" << std::endl;
    if (!options.cond_in_header) s << "if (!(" << cond << ")) break;" << std::endl;
    s << "a[i + " << offset2s << "] = a[i] + b[i];" << std::endl;
    if (!options.incr_in_header) s << incr << ";" << std::endl;
    s << "}" << std::endl;
    if (options.loop == loop_kind::do_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ");" << std::endl;

    s << "}" << std::endl;

    return s.str();
}
#else
std::string generate_source(test_options options)
{
    std::string offset1s = options.offset1_param ? "offset1" : std::to_string(options.offset1);
    std::string offset2s = options.offset2_param ? "offset2" : std::to_string(options.offset2);

    std::string init = "i = 0";
    std::string cond = std::string("i < ") + (options.iter_count_param ? "iter_count" : std::to_string(options.iter_count));
    std::string incr = "i += 2";

    std::stringstream s;
    s << R"(
    #include <opencl_memory>
    #include <opencl_work_item>

    using namespace cl;
    )";
    s << R"(
    kernel void test(global_ptr<int[]> a, global_ptr<int[]> b, global_ptr<int[]> c, int offset1, int offset2, int iter_count)
    {
        int i;
    )";

    // Loop #1
    if (!options.init_in_header) s << init << ";" << std::endl;
    if (options.ivdep_length > 0) s << "[[cl::ivdep]]" << std::endl;
    if (options.loop == loop_kind::for_loop)
        s << "for (" <<
            (options.init_in_header ? init : "") << ";" <<
            (options.cond_in_header ? cond : "") << ";" <<
            (options.incr_in_header ? incr : "") << ")";
    else if (options.loop == loop_kind::while_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ")";
    else if (options.loop == loop_kind::do_loop)
        s << "do";
    s << "{" << std::endl;
    if (!options.cond_in_header) s << "if (!(" << cond << ")) break;" << std::endl;
    s << "a[i + " << offset1s << "] = b[i + " << offset1s << "] * c[i + " << offset1s << "];" << std::endl;
    if (!options.incr_in_header) s << incr << ";" << std::endl;
    s << "}" << std::endl;
    if (options.loop == loop_kind::do_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ");" << std::endl;

    // Loop #2
    if (!options.init_in_header) s << init << ";" << std::endl;
    if (options.ivdep_length > 0) s << "[[cl::ivdep(" << options.ivdep_length << ")]]" << std::endl;
    if (options.loop == loop_kind::for_loop)
        s << "for (" <<
            (options.init_in_header ? init : "") << ";" <<
            (options.cond_in_header ? cond : "") << ";" <<
            (options.incr_in_header ? incr : "") << ")";
    else if (options.loop == loop_kind::while_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ")";
    else if (options.loop == loop_kind::do_loop)
        s << "do";
    s << "{" << std::endl;
    if (!options.cond_in_header) s << "if (!(" << cond << ")) break;" << std::endl;
    s << "a[i + " << offset2s << "] = a[i] + b[i];" << std::endl;
    if (!options.incr_in_header) s << incr << ";" << std::endl;
    s << "}" << std::endl;
    if (options.loop == loop_kind::do_loop)
        s << "while (" << (options.cond_in_header ? cond : "true") << ");" << std::endl;

    s << "}" << std::endl;

    return s.str();
}
#endif

int test(cl_device_id device, cl_context context, cl_command_queue queue, test_options options)
{
    int error = CL_SUCCESS;

    cl_program program;
    cl_kernel kernel;

    std::string kernel_name = "test";
    std::string source = generate_source(options);

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
// Only OpenCL C++ to SPIR-V compilation
#if defined(DEVELOPMENT) && defined(ONLY_SPIRV_COMPILATION)
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name
    );
    RETURN_ON_ERROR(error)
    return error;
// Use OpenCL C kernels instead of OpenCL C++ kernels (test C++ host code)
#elif defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name, "", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name
    );
    RETURN_ON_ERROR(error)
#endif

    const size_t count = 100;
    const size_t global_size = 1;

    std::vector<int> a(count);
    std::vector<int> b(count);
    std::vector<int> c(count);
    for (size_t i = 0; i < count; i++)
    {
        a[i] = 0;
        b[i] = i;
        c[i] = 1;
    }

    cl_mem a_buffer;
    a_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * count, static_cast<void *>(a.data()), &error
    );
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    cl_mem b_buffer;
    b_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * count, static_cast<void *>(b.data()), &error
    );
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    cl_mem c_buffer;
    c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * count, static_cast<void *>(c.data()),&error
    );
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 3, sizeof(cl_int), &options.offset1);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 4, sizeof(cl_int), &options.offset2);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")
    error = clSetKernelArg(kernel, 5, sizeof(cl_int), &options.iter_count);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

    std::vector<int> a_output(count);
    error = clEnqueueReadBuffer(
        queue, a_buffer, CL_TRUE,
        0, sizeof(int) * count,
        static_cast<void *>(a_output.data()),
        0, NULL, NULL
    );
    RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

    for (int i = 0; i < options.iter_count; i += 2)
    {
        a[i + options.offset1] = b[i + options.offset1] * c[i + options.offset1];
    }

    for (int i = 0; i < options.iter_count; i += 2)
    {
        a[i + options.offset2] = a[i] + b[i];
    }

    for (size_t i = 0; i < count; i++)
    {
        const int value = a_output[i];
        const int expected = a[i];
        if (value != expected)
        {
            RETURN_ON_ERROR_MSG(-1,
                "Test failed. Element %lu: %d should be: %d",
                i, value, expected
            );
        }
    }

    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

const std::vector<std::tuple<int, int, int>> params{
    std::make_tuple<int, int, int>( -1, 0, 0 ),
    std::make_tuple<int, int, int>( -1, 3, 4 ),
    std::make_tuple<int, int, int>( 1, 1, 1 ),
    std::make_tuple<int, int, int>( 3, 4, 2 ),
    std::make_tuple<int, int, int>( 3, 4, 3 ),
    std::make_tuple<int, int, int>( 8, 10, 7 ),
    std::make_tuple<int, int, int>( 16, 16, 16 )
};
const std::vector<int> iter_counts{ { 1, 4, 12, 40 } };

AUTO_TEST_CASE(test_ivdep_for)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    for (auto param : params)
    for (auto iter_count : iter_counts)
    for (bool offset1_param : { false, true })
    for (bool offset2_param : { false, true })
    for (bool iter_count_param : { false, true })
    for (bool cond_in_header : { false, true })
    for (bool init_in_header : { false, true })
    for (bool incr_in_header : { false, true })
    {
        test_options options;
        options.loop = loop_kind::for_loop;
        options.ivdep_length = std::get<0>(param);
        options.offset1 = std::get<1>(param);
        options.offset2 = std::get<2>(param);
        options.iter_count = iter_count;
        options.offset1_param = offset1_param;
        options.offset2_param = offset2_param;
        options.iter_count_param = iter_count_param;
        options.cond_in_header = cond_in_header;
        options.init_in_header = init_in_header;
        options.incr_in_header = incr_in_header;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

AUTO_TEST_CASE(test_ivdep_while)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    for (auto param : params)
    for (auto iter_count : iter_counts)
    for (bool offset1_param : { false, true })
    for (bool offset2_param : { false, true })
    for (bool iter_count_param : { false, true })
    for (bool cond_in_header : { false, true })
    {
        test_options options;
        options.loop = loop_kind::while_loop;
        options.ivdep_length = std::get<0>(param);
        options.offset1 = std::get<1>(param);
        options.offset2 = std::get<2>(param);
        options.iter_count = iter_count;
        options.offset1_param = offset1_param;
        options.offset2_param = offset2_param;
        options.iter_count_param = iter_count_param;
        options.cond_in_header = cond_in_header;
        options.init_in_header = false;
        options.incr_in_header = false;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

AUTO_TEST_CASE(test_ivdep_do)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    for (auto param : params)
    for (auto iter_count : iter_counts)
    for (bool offset1_param : { false, true })
    for (bool offset2_param : { false, true })
    for (bool iter_count_param : { false, true })
    for (bool cond_in_header : { false, true })
    {
        test_options options;
        options.loop = loop_kind::do_loop;
        options.ivdep_length = std::get<0>(param);
        options.offset1 = std::get<1>(param);
        options.offset2 = std::get<2>(param);
        options.iter_count = iter_count;
        options.offset1_param = offset1_param;
        options.offset2_param = offset2_param;
        options.iter_count_param = iter_count_param;
        options.cond_in_header = cond_in_header;
        options.init_in_header = false;
        options.incr_in_header = false;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_IVDEP_HPP
