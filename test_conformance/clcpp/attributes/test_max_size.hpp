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
#ifndef TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_MAX_SIZE_HPP
#define TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_MAX_SIZE_HPP

#include <sstream>
#include <string>
#include <vector>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_max_size {

enum class address_space
{
    constant,
    local
};

enum class param_kind
{
    ptr_type, // constant_ptr<T>
    ptr,      // constant<T>*
    ref       // constant<T>&
};

const param_kind param_kinds[] =
{
    param_kind::ptr_type,
    param_kind::ptr,
    param_kind::ref
};

struct test_options
{
    address_space space;
    int max_size;
    bool spec_const;
    param_kind kind;
    bool array;
};

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << "kernel void test(";
    s << (options.space == address_space::constant ? "constant" : "local");
    s << " int2 *input) { }" << std::endl;

    return s.str();
}
#else
std::string generate_source(test_options options)
{
    std::string type_str = "int2";
    if (options.array)
        type_str += "[]";

    std::stringstream s;
    s << "#include <opencl_memory>" << std::endl;

    if (options.spec_const)
    {
        s << "#include <opencl_spec_constant>" << std::endl;
        s << "cl::spec_constant<int, 1> max_size_spec{ 1234567890 };" << std::endl;
    }

    s << "kernel void test(";
    s << "[[cl::max_size(" << (options.spec_const ? "max_size_spec" : std::to_string(options.max_size)) << ")]] ";
    s << (options.space == address_space::constant ? "cl::constant" : "cl::local");
    if (options.kind == param_kind::ptr_type)
        s << "_ptr<" << type_str << ">";
    else if (options.kind == param_kind::ptr)
        s << "<" << type_str << ">*";
    else if (options.kind == param_kind::ref)
        s << "<" << type_str << ">&";
    s << " input) { }" << std::endl;

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
    const char *source_c_str = source.c_str();
    error = create_openclcpp_program(context, &program, 1, &source_c_str, "");
    RETURN_ON_ERROR(error)

    if (options.spec_const)
    {
        error = clSetProgramSpecializationConstant(program, 1, sizeof(cl_int), &options.max_size);
        RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    }

    error = build_program_create_kernel_helper(
        context, &program, &kernel, 1, &source_c_str, kernel_name.c_str()
    );
    RETURN_ON_ERROR(error)
#endif

    const int max_size = options.max_size;
    const int sizes[] = {
        1,
        max_size / 2,
        max_size,
        max_size + 1,
        max_size * 2
    };

    for (int size : sizes)
    {
        cl_mem const_buffer;
        if (options.space == address_space::constant)
        {
            const_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &error);
            RETURN_ON_CL_ERROR(error, "clCreateBuffer")

            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &const_buffer);
            // Check the status later (depending on size and max_size values)
        }
        else if (options.space == address_space::local)
        {
            error = clSetKernelArg(kernel, 0, size, NULL);
            // Check the status later (depending on size and max_size values)
        }

        if (size <= max_size)
        {
            // Correct value, must not fail
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")

            const size_t global_size = 123;
            error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
            RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

            error = clFinish(queue);
            RETURN_ON_CL_ERROR(error, "clFinish")
        }
        else
        {
            // Incorrect value, must fail
            if (error != CL_MAX_SIZE_RESTRICTION_EXCEEDED)
            {
                RETURN_ON_ERROR_MSG(-1,
                    "clSetKernelArg must fail with CL_MAX_SIZE_RESTRICTION_EXCEEDED,"
                    " but returned %s (%d)", get_cl_error_string(error).c_str(), error
                );
            }
        }

        if (options.space == address_space::constant)
        {
            error = clReleaseMemObject(const_buffer);
            RETURN_ON_CL_ERROR(error, "clReleaseMemObject")
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_max_size_constant)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    cl_ulong max_size;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(max_size), &max_size, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    for (bool spec_const : { false, true })
    for (auto kind : param_kinds)
    for (bool array : { false, true })
    {
        test_options options;
        options.space = address_space::constant;
        options.max_size = max_size / 2;
        options.spec_const = spec_const;
        options.kind = kind;
        options.array = array;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

AUTO_TEST_CASE(test_max_size_local)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    cl_ulong max_size;
    error = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(max_size), &max_size, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    for (bool spec_const : { false, true })
    for (auto kind : param_kinds)
    for (bool array : { false, true })
    {
        test_options options;
        options.space = address_space::local;
        options.max_size = max_size / 2;
        options.spec_const = spec_const;
        options.kind = kind;
        options.array = array;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_MAX_SIZE_HPP
