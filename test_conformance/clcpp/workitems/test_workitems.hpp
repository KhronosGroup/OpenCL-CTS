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
#ifndef TEST_CONFORMANCE_CLCPP_WI_TEST_WORKITEMS_HPP
#define TEST_CONFORMANCE_CLCPP_WI_TEST_WORKITEMS_HPP

#include <vector>
#include <algorithm>
#include <random>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_workitems {

struct test_options
{
    bool uniform_work_group_size;
    size_t max_count;
    size_t num_tests;
};

struct output_type
{
    cl_uint  work_dim;
    cl_ulong global_size[3];
    cl_ulong global_id[3];
    cl_ulong local_size[3];
    cl_ulong enqueued_local_size[3];
    cl_ulong local_id[3];
    cl_ulong num_groups[3];
    cl_ulong group_id[3];
    cl_ulong global_offset[3];
    cl_ulong global_linear_id;
    cl_ulong local_linear_id;
    cl_ulong sub_group_size;
    cl_ulong max_sub_group_size;
    cl_ulong num_sub_groups;
    cl_ulong enqueued_num_sub_groups;
    cl_ulong sub_group_id;
    cl_ulong sub_group_local_id;
};

const std::string source_common = R"(
struct output_type
{
    uint  work_dim;
    ulong global_size[3];
    ulong global_id[3];
    ulong local_size[3];
    ulong enqueued_local_size[3];
    ulong local_id[3];
    ulong num_groups[3];
    ulong group_id[3];
    ulong global_offset[3];
    ulong global_linear_id;
    ulong local_linear_id;
    ulong sub_group_size;
    ulong max_sub_group_size;
    ulong num_sub_groups;
    ulong enqueued_num_sub_groups;
    ulong sub_group_id;
    ulong sub_group_local_id;
};
)";

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
const std::string source =
    source_common +
    R"(
        #ifdef cl_khr_subgroups
        #pragma OPENCL EXTENSION cl_khr_subgroups : enable
        #endif

        kernel void test(global struct output_type *output)
        {
           const ulong gid = get_global_linear_id();
           output[gid].work_dim = get_work_dim();
           for (uint dimindx = 0; dimindx < 3; dimindx++)
           {
               output[gid].global_size[dimindx] = get_global_size(dimindx);
               output[gid].global_id[dimindx] = get_global_id(dimindx);
               output[gid].local_size[dimindx] = get_local_size(dimindx);
               output[gid].enqueued_local_size[dimindx] = get_enqueued_local_size(dimindx);
               output[gid].local_id[dimindx] = get_local_id(dimindx);
               output[gid].num_groups[dimindx] = get_num_groups(dimindx);
               output[gid].group_id[dimindx] = get_group_id(dimindx);
               output[gid].global_offset[dimindx] = get_global_offset(dimindx);
           }
           output[gid].global_linear_id = get_global_linear_id();
           output[gid].local_linear_id = get_local_linear_id();
        #ifdef cl_khr_subgroups
           output[gid].sub_group_size = get_sub_group_size();
           output[gid].max_sub_group_size = get_max_sub_group_size();
           output[gid].num_sub_groups = get_num_sub_groups();
           output[gid].enqueued_num_sub_groups = get_enqueued_num_sub_groups();
           output[gid].sub_group_id = get_sub_group_id();
           output[gid].sub_group_local_id = get_sub_group_local_id();
        #endif
        }
    )";
#else
const std::string source =
    R"(
        #include <opencl_memory>
        #include <opencl_work_item>
        using namespace cl;
    )" +
    source_common +
    R"(

        kernel void test(global_ptr<output_type[]> output)
        {
           const size_t gid = get_global_linear_id();
           output[gid].work_dim = get_work_dim();
           for (uint dimindx = 0; dimindx < 3; dimindx++)
           {
               output[gid].global_size[dimindx] = get_global_size(dimindx);
               output[gid].global_id[dimindx] = get_global_id(dimindx);
               output[gid].local_size[dimindx] = get_local_size(dimindx);
               output[gid].enqueued_local_size[dimindx] = get_enqueued_local_size(dimindx);
               output[gid].local_id[dimindx] = get_local_id(dimindx);
               output[gid].num_groups[dimindx] = get_num_groups(dimindx);
               output[gid].group_id[dimindx] = get_group_id(dimindx);
               output[gid].global_offset[dimindx] = get_global_offset(dimindx);
           }
           output[gid].global_linear_id = get_global_linear_id();
           output[gid].local_linear_id = get_local_linear_id();
           output[gid].sub_group_size = get_sub_group_size();
           output[gid].max_sub_group_size = get_max_sub_group_size();
           output[gid].num_sub_groups = get_num_sub_groups();
           output[gid].enqueued_num_sub_groups = get_enqueued_num_sub_groups();
           output[gid].sub_group_id = get_sub_group_id();
           output[gid].sub_group_local_id = get_sub_group_local_id();
        }

    )";
#endif

#define CHECK_EQUAL(result, expected, func_name) \
    if (result != expected) \
    { \
        RETURN_ON_ERROR_MSG(-1, \
            "Function %s failed. Expected: %s, got: %s", func_name, \
            format_value(expected).c_str(), format_value(result).c_str() \
        ); \
    }

#define CHECK(expression, func_name) \
    if (expression) \
    { \
        RETURN_ON_ERROR_MSG(-1, \
            "Function %s returned incorrect result", func_name \
        ); \
    }

int test_workitems(cl_device_id device, cl_context context, cl_command_queue queue, test_options options)
{
    int error = CL_SUCCESS;

    cl_program program;
    cl_kernel kernel;

    std::string kernel_name = "test";

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
    error = create_opencl_kernel(
        context, &program, &kernel,
        source, kernel_name
    );
    RETURN_ON_ERROR(error)
#endif

    size_t max_work_group_size;
    size_t max_local_sizes[3];
    error = get_max_allowed_work_group_size(context, kernel, &max_work_group_size, max_local_sizes);
    RETURN_ON_ERROR(error)

    bool check_sub_groups = true;
    bool check_sub_groups_limits = true;
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
    check_sub_groups = false;
    check_sub_groups_limits = false;
    if (is_extension_available(device, "cl_khr_subgroups"))
    {
        Version version = get_device_cl_version(device);
        RETURN_ON_ERROR(error)
        check_sub_groups_limits = (version >= Version(2,1)); // clGetKernelSubGroupInfo is from 2.1
        check_sub_groups = true;
    }
#endif

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> count_dis(1, options.max_count);

    for (int test = 0; test < options.num_tests; test++)
    {
        for (size_t dim = 1; dim <= 3; dim++)
        {
            size_t global_size[3] = { 1, 1, 1 };
            size_t global_offset[3] = { 0, 0, 0 };
            size_t enqueued_local_size[3] = { 1, 1, 1 };
            size_t count = count_dis(gen);
            std::uniform_int_distribution<size_t> global_size_dis(1, static_cast<size_t>(pow(count, 1.0 / dim)));
            for (int d = 0; d < dim; d++)
            {
                std::uniform_int_distribution<size_t> enqueued_local_size_dis(1, max_local_sizes[d]);
                global_size[d] = global_size_dis(gen);
                global_offset[d] = global_size_dis(gen);
                enqueued_local_size[d] = enqueued_local_size_dis(gen);
            }
            // Local work size must not exceed CL_KERNEL_WORK_GROUP_SIZE for this kernel
            while (enqueued_local_size[0] * enqueued_local_size[1] * enqueued_local_size[2] > max_work_group_size)
            {
                // otherwise decrease it until it fits
                for (int d = 0; d < dim; d++)
                {
                    enqueued_local_size[d] = (std::max)((size_t)1, enqueued_local_size[d] / 2);
                }
            }
            if (options.uniform_work_group_size)
            {
                for (int d = 0; d < dim; d++)
                {
                    global_size[d] = get_uniform_global_size(global_size[d], enqueued_local_size[d]);
                }
            }
            count = global_size[0] * global_size[1] * global_size[2];

            cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output_type) * count, NULL, &error);
            RETURN_ON_CL_ERROR(error, "clCreateBuffer")

            const char pattern = 0;
            error = clEnqueueFillBuffer(queue, output_buffer, &pattern, sizeof(pattern), 0, sizeof(output_type) * count, 0, NULL, NULL);
            RETURN_ON_CL_ERROR(error, "clEnqueueFillBuffer")

            error = clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer);
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")

            error = clEnqueueNDRangeKernel(queue, kernel, dim, global_offset, global_size, enqueued_local_size, 0, NULL, NULL);
            RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

            std::vector<output_type> output(count);
            error = clEnqueueReadBuffer(
                queue, output_buffer, CL_TRUE,
                0, sizeof(output_type) * count,
                static_cast<void *>(output.data()),
                0, NULL, NULL
            );
            RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

            error = clReleaseMemObject(output_buffer);
            RETURN_ON_CL_ERROR(error, "clReleaseMemObject")

            size_t sub_group_count_for_ndrange = 0;
            size_t max_sub_group_size_for_ndrange = 0;
            if (check_sub_groups_limits)
            {
                error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE,
                    sizeof(size_t) * dim, enqueued_local_size,
                    sizeof(size_t), &sub_group_count_for_ndrange, NULL);
                RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")

                error = clGetKernelSubGroupInfo(kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                    sizeof(size_t) * dim, enqueued_local_size,
                    sizeof(size_t), &max_sub_group_size_for_ndrange, NULL);
                RETURN_ON_CL_ERROR(error, "clGetKernelSubGroupInfo")
            }

            size_t num_groups[3];
            for (int d = 0; d < 3; d++)
                num_groups[d] = static_cast<size_t>(std::ceil(static_cast<double>(global_size[d]) / enqueued_local_size[d]));

            size_t group_id[3];
            for (group_id[0] = 0; group_id[0] < num_groups[0]; group_id[0]++)
            for (group_id[1] = 0; group_id[1] < num_groups[1]; group_id[1]++)
            for (group_id[2] = 0; group_id[2] < num_groups[2]; group_id[2]++)
            {
                size_t local_size[3];
                for (int d = 0; d < 3; d++)
                {
                    if (group_id[d] == num_groups[d] - 1)
                        local_size[d] = global_size[d] - group_id[d] * enqueued_local_size[d];
                    else
                        local_size[d] = enqueued_local_size[d];
                }

                size_t local_id[3];
                for (local_id[0] = 0; local_id[0] < local_size[0]; local_id[0]++)
                for (local_id[1] = 0; local_id[1] < local_size[1]; local_id[1]++)
                for (local_id[2] = 0; local_id[2] < local_size[2]; local_id[2]++)
                {
                    size_t global_id_wo_offset[3];
                    size_t global_id[3];
                    for (int d = 0; d < 3; d++)
                    {
                        global_id_wo_offset[d] = group_id[d] * enqueued_local_size[d] + local_id[d];
                        global_id[d] = global_id_wo_offset[d] + global_offset[d];
                    }

                    // Ignore if the current work-item is outside of global work size (i.e. the work-group is non-uniform)
                    if (global_id_wo_offset[0] >= global_size[0] ||
                        global_id_wo_offset[1] >= global_size[1] ||
                        global_id_wo_offset[2] >= global_size[2]) break;

                    const size_t global_linear_id =
                        global_id_wo_offset[2] * global_size[1] * global_size[0] +
                        global_id_wo_offset[1] * global_size[0] +
                        global_id_wo_offset[0];
                    const size_t local_linear_id =
                        local_id[2] * local_size[1] * local_size[0] +
                        local_id[1] * local_size[0] +
                        local_id[0];

                    const output_type &o = output[global_linear_id];

                    CHECK_EQUAL(o.work_dim, dim, "get_work_dim")
                    for (int d = 0; d < 3; d++)
                    {
                        CHECK_EQUAL(o.global_size[d], global_size[d], "get_global_size")
                        CHECK_EQUAL(o.global_id[d], global_id[d], "get_global_id")
                        CHECK_EQUAL(o.local_size[d], local_size[d], "get_local_size")
                        CHECK_EQUAL(o.enqueued_local_size[d], enqueued_local_size[d], "get_enqueued_local_size")
                        CHECK_EQUAL(o.local_id[d], local_id[d], "get_local_id")
                        CHECK_EQUAL(o.num_groups[d], num_groups[d], "get_num_groups")
                        CHECK_EQUAL(o.group_id[d], group_id[d], "get_group_id")
                        CHECK_EQUAL(o.global_offset[d], global_offset[d], "get_global_offset")
                    }

                    CHECK_EQUAL(o.global_linear_id, global_linear_id, "get_global_linear_id")
                    CHECK_EQUAL(o.local_linear_id, local_linear_id, "get_local_linear_id")

                    // A few (but not all possible) sub-groups related checks
                    if (check_sub_groups)
                    {
                        if (check_sub_groups_limits)
                        {
                            CHECK_EQUAL(o.max_sub_group_size, max_sub_group_size_for_ndrange, "get_max_sub_group_size")
                            CHECK_EQUAL(o.enqueued_num_sub_groups, sub_group_count_for_ndrange, "get_enqueued_num_sub_groups")
                        }
                        CHECK(o.sub_group_size == 0 || o.sub_group_size > o.max_sub_group_size, "get_sub_group_size or get_max_sub_group_size")
                        CHECK(o.num_sub_groups == 0 || o.num_sub_groups > o.enqueued_num_sub_groups, "get_enqueued_num_sub_groups")
                        CHECK(o.sub_group_id >= o.num_sub_groups, "get_sub_group_id or get_num_sub_groups")
                        CHECK(o.sub_group_local_id >= o.sub_group_size, "get_sub_group_local_id or get_sub_group_size")
                    }
                }
            }
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

#undef CHECK_EQUAL
#undef CHECK

AUTO_TEST_CASE(test_workitems_uniform)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.uniform_work_group_size = true;
    options.max_count = num_elements;
    options.num_tests = 1000;
    return test_workitems(device, context, queue, options);
}

AUTO_TEST_CASE(test_workitems_non_uniform)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.uniform_work_group_size = false;
    options.max_count = num_elements;
    options.num_tests = 1000;
    return test_workitems(device, context, queue, options);
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_WI_TEST_WORKITEMS_HPP
