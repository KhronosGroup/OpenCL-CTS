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
#ifndef TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_REQUIRED_NUM_SUB_GROUPS_HPP
#define TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_REQUIRED_NUM_SUB_GROUPS_HPP

#include <sstream>
#include <string>
#include <vector>
#include <random>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_required_num_sub_groups {

struct test_options
{
    size_t num_sub_groups;
    bool spec_const;
    size_t max_count;
    size_t num_tests;
};

struct output_type
{
    cl_ulong num_sub_groups;
    cl_ulong enqueued_num_sub_groups;
};

const std::string source_common = R"(
struct output_type
{
    ulong num_sub_groups;
    ulong enqueued_num_sub_groups;
};
)";

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << source_common;
    s << R"(
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable

    kernel void test(global struct output_type *output)
    {
        const ulong gid = get_global_linear_id();
        output[gid].num_sub_groups = get_num_sub_groups();
        output[gid].enqueued_num_sub_groups = get_enqueued_num_sub_groups();
    }
    )";

    return s.str();
}
#else
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << R"(
    #include <opencl_memory>
    #include <opencl_work_item>
    using namespace cl;
    )";

    if (options.spec_const)
    {
        s << "#include <opencl_spec_constant>" << std::endl;
        s << "cl::spec_constant<uint, 1> num_sub_groups_spec{ 1234567890 };" << std::endl;
    }

    s << source_common << std::endl;
    s << "[[cl::required_num_sub_groups(" << (options.spec_const ? "num_sub_groups_spec" : std::to_string(options.num_sub_groups)) << ")]]";
    s << R"(
    kernel void test(global_ptr<output_type[]> output)
    {
        const ulong gid = get_global_linear_id();
        output[gid].num_sub_groups = get_num_sub_groups();
        output[gid].enqueued_num_sub_groups = get_enqueued_num_sub_groups();
    }
    )";

    return s.str();
}
#endif

int test(cl_device_id device, cl_context context, cl_command_queue queue, test_options options)
{
    int error = CL_SUCCESS;

#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    if (!is_extension_available(device, "cl_khr_subgroups"))
    {
        log_info("SKIPPED: Extension `cl_khr_subgroups` is not supported. Skipping tests.\n");
        return CL_SUCCESS;
    }
#endif

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
        source, kernel_name, "-cl-std=CL2.0", false
    );
    RETURN_ON_ERROR(error)
// Normal run
#else
    const char *source_c_str = source.c_str();
    error = create_openclcpp_program(context, &program, 1, &source_c_str, "");
    RETURN_ON_ERROR(error)

    if (options.spec_const)
    {
        cl_uint spec_num_sub_groups = static_cast<cl_uint>(options.num_sub_groups);
        error = clSetProgramSpecializationConstant(program, 1, sizeof(cl_uint), &spec_num_sub_groups);
        RETURN_ON_CL_ERROR(error, "clSetProgramSpecializationConstant")
    }

    error = build_program_create_kernel_helper(
        context, &program, &kernel, 1, &source_c_str, kernel_name.c_str()
    );
    RETURN_ON_ERROR(error)
#endif

    size_t compile_num_sub_groups;
    error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_COMPILE_NUM_SUB_GROUPS,
        0, NULL,
        sizeof(size_t), &compile_num_sub_groups, NULL);
    RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
    if (compile_num_sub_groups != options.num_sub_groups)
    {
        RETURN_ON_ERROR_MSG(-1,
            "CL_KERNEL_COMPILE_NUM_SUB_GROUPS did not return correct value (expected %lu, got %lu)",
            options.num_sub_groups, compile_num_sub_groups
        )
    }

    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output_type) * options.max_count, NULL, &error);
    RETURN_ON_CL_ERROR(error, "clCreateBuffer")

    error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
    RETURN_ON_CL_ERROR(error, "clSetKernelArg")

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> count_dis(1, options.max_count);

    for (size_t test = 0; test < options.num_tests; test++)
    {
        for (size_t dim = 1; dim <= 3; dim++)
        {
            size_t global_size[3] = { 1, 1, 1 };
            size_t count = count_dis(gen);
            std::uniform_int_distribution<size_t> global_size_dis(1, static_cast<size_t>(pow(count, 1.0 / dim)));
            for (size_t d = 0; d < dim; d++)
            {
                global_size[d] = global_size_dis(gen);
            }
            count = global_size[0] * global_size[1] * global_size[2];

            size_t local_size[3] = { 1, 1, 1 };
            error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
                sizeof(size_t), &options.num_sub_groups,
                sizeof(size_t) * dim, local_size, NULL);
            RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
            if (local_size[0] == 0 || local_size[1] != 1 || local_size[2] != 1)
            {
                RETURN_ON_ERROR_MSG(-1,
                    "CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT did not return correct value"
                )
            }

            size_t sub_group_count_for_ndrange;
            error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
                sizeof(size_t) * dim, local_size,
                sizeof(size_t), &sub_group_count_for_ndrange, NULL);
            RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
            if (sub_group_count_for_ndrange != options.num_sub_groups)
            {
                RETURN_ON_ERROR_MSG(-1,
                    "CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE did not return correct value (expected %lu, got %lu)",
                    options.num_sub_groups, sub_group_count_for_ndrange
                )
            }

            const char pattern = 0;
            error = clEnqueueFillBuffer(queue, output_buffer, &pattern, sizeof(pattern), 0, sizeof(output_type) * count, 0, NULL, NULL);
            RETURN_ON_CL_ERROR(error, "clEnqueueFillBuffer")

            error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, global_size, local_size, 0, NULL, NULL);
            RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

            std::vector<output_type> output(count);
            error = clEnqueueReadBuffer(
                queue, output_buffer, CL_TRUE,
                0, sizeof(output_type) * count,
                static_cast<void *>(output.data()),
                0, NULL, NULL
            );
            RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

            for (size_t gid = 0; gid < count; gid++)
            {
                const output_type &o = output[gid];

                if (o.enqueued_num_sub_groups != options.num_sub_groups)
                {
                    RETURN_ON_ERROR_MSG(-1, "get_enqueued_num_sub_groups does not equal to required_num_sub_groups")
                }
                if (o.num_sub_groups > options.num_sub_groups)
                {
                    RETURN_ON_ERROR_MSG(-1, "get_num_sub_groups did not return correct value")
                }
            }
        }
    }

    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_required_num_sub_groups)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;

    cl_uint max_num_sub_groups;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(max_num_sub_groups), &max_num_sub_groups, NULL);
    RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

    for (bool spec_const : { false, true })
    for (size_t num_sub_groups = 1; num_sub_groups <= max_num_sub_groups; num_sub_groups++)
    {
        test_options options;
        options.spec_const = spec_const;
        options.num_sub_groups = num_sub_groups;
        options.num_tests = 100;
        options.max_count = num_elements;

        error = test(device, context, queue, options);
        RETURN_ON_ERROR(error)
    }

    return error;
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_ATTRIBUTES_TEST_REQUIRED_NUM_SUB_GROUPS_HPP
