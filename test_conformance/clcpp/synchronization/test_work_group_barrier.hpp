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
#ifndef TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_TEST_WORK_GROUP_BARRIER_HPP
#define TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_TEST_WORK_GROUP_BARRIER_HPP

#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

// Common for all OpenCL C++ tests
#include "../common.hpp"


namespace test_work_group_barrier {

enum class barrier_type
{
    local,
    global
};

struct test_options
{
    barrier_type barrier;
    size_t max_count;
    size_t num_tests;
};

const std::string source_common = R"(
    // Circular shift of local ids
    size_t get_shifted_local_id(int local_id_delta)
    {
        const int local_size = (int)get_local_size(0);
        return (((int)get_local_id(0) + local_id_delta) % local_size + local_size) % local_size;
    }

    // Get global ids from shifted local ids
    size_t get_shifted_global_id(int local_id_delta)
    {
        return get_group_id(0) * get_enqueued_local_size(0) + get_shifted_local_id(local_id_delta);
    }
)";

// -----------------------------------------------------------------------------------
// ------------- ONLY FOR OPENCL 22 CONFORMANCE TEST 22 DEVELOPMENT ------------------
// -----------------------------------------------------------------------------------
#if defined(DEVELOPMENT) && defined(USE_OPENCLC_KERNELS)
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << source_common;
    if (options.barrier == barrier_type::global)
    {
        s << R"(
    kernel void test(const int iter_lo, const int iter_hi, global long *output)
    {
        const size_t gid = get_shifted_global_id(0);

        output[gid] = gid;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);

        for (int i = iter_lo; i < iter_hi; i++)
        {
            const size_t other_gid = get_shifted_global_id(i);

            output[other_gid] += other_gid;
            work_group_barrier(CLK_GLOBAL_MEM_FENCE);

            output[gid] += gid;
            work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
    )";
    }
    else if (options.barrier == barrier_type::local)
    {
        s << R"(
    kernel void test(const int iter_lo, const int iter_hi, global long *output, local long *values)
    {
        const size_t gid = get_shifted_global_id(0);
        const size_t lid = get_shifted_local_id(0);

        values[lid] = gid;
        work_group_barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = iter_lo; i < iter_hi; i++)
        {
            const size_t other_lid = get_shifted_local_id(i);
            const size_t other_gid = get_shifted_global_id(i);

            values[other_lid] += other_gid;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);

            values[lid] += gid;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
        }

        output[gid] = values[lid];
    }
    )";
    }

    return s.str();
}
#else
std::string generate_source(test_options options)
{
    std::stringstream s;
    s << R"(
    #include <opencl_memory>
    #include <opencl_work_item>
    #include <opencl_synchronization>

    using namespace cl;

    )";
    s << source_common;

    if (options.barrier == barrier_type::global)
    {
        s << R"(
    kernel void test(const int iter_lo, const int iter_hi, global_ptr<long[]> output)
    {
        const size_t gid = get_shifted_global_id(0);

        output[gid] = gid;
        work_group_barrier(mem_fence::global);

        for (int i = iter_lo; i < iter_hi; i++)
        {
            const size_t other_gid = get_shifted_global_id(i);

            output[other_gid] += other_gid;
            work_group_barrier(mem_fence::global);

            output[gid] += gid;
            work_group_barrier(mem_fence::global);
        }
    }
    )";
    }
    else if (options.barrier == barrier_type::local)
    {
        s << R"(
    kernel void test(const int iter_lo, const int iter_hi, global_ptr<long[]> output, local_ptr<long[]> values)
    {
        const size_t gid = get_shifted_global_id(0);
        const size_t lid = get_shifted_local_id(0);

        values[lid] = gid;
        work_group_barrier(mem_fence::local);

        for (int i = iter_lo; i < iter_hi; i++)
        {
            const size_t other_lid = get_shifted_local_id(i);
            const size_t other_gid = get_shifted_global_id(i);

            values[other_lid] += other_gid;
            work_group_barrier(mem_fence::local);

            values[lid] += gid;
            work_group_barrier(mem_fence::local);
        }

        output[gid] = values[lid];
    }
    )";
    }

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
    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")

    if (options.barrier == barrier_type::local)
    {
        cl_ulong kernel_local_mem_size;
        error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL);
        RETURN_ON_CL_ERROR(error, "clGetKernelWorkGroupInfo")

        cl_ulong device_local_mem_size;
        error = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_local_mem_size), &device_local_mem_size, NULL);
        RETURN_ON_CL_ERROR(error, "clGetDeviceInfo")

        max_work_group_size = (std::min<cl_ulong>)(max_work_group_size, (device_local_mem_size - kernel_local_mem_size) / sizeof(cl_long));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> global_size_dis(1, options.max_count);
    std::uniform_int_distribution<size_t> local_size_dis(1, max_work_group_size);
    std::uniform_int_distribution<int> iter_dis(0, 20);

    for (size_t test = 0; test < options.num_tests; test++)
    {
        const size_t global_size = global_size_dis(gen);
        const size_t local_size = local_size_dis(gen);
        const size_t count = global_size;

        const int iter_lo = -iter_dis(gen);
        const int iter_hi = +iter_dis(gen);

        cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_long) * count, NULL, &error);
        RETURN_ON_CL_ERROR(error, "clCreateBuffer")

        error = clSetKernelArg(kernel, 0, sizeof(iter_lo), &iter_lo);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 1, sizeof(iter_hi), &iter_hi);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        error = clSetKernelArg(kernel, 2, sizeof(output_buffer), &output_buffer);
        RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        if (options.barrier == barrier_type::local)
        {
            error = clSetKernelArg(kernel, 3, sizeof(cl_long) * local_size, NULL);
            RETURN_ON_CL_ERROR(error, "clSetKernelArg")
        }

        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        RETURN_ON_CL_ERROR(error, "clEnqueueNDRangeKernel")

        std::vector<cl_long> output(count);
        error = clEnqueueReadBuffer(
            queue, output_buffer, CL_TRUE,
            0, sizeof(cl_long) * count,
            static_cast<void *>(output.data()),
            0, NULL, NULL
        );
        RETURN_ON_CL_ERROR(error, "clEnqueueReadBuffer")

        error = clReleaseMemObject(output_buffer);
        RETURN_ON_CL_ERROR(error, "clReleaseMemObject")

        for (size_t gid = 0; gid < count; gid++)
        {
            const long value = output[gid];
            const long expected = gid + 2 * gid * (iter_hi - iter_lo);

            if (value != expected)
            {
                RETURN_ON_ERROR_MSG(-1,
                    "Element %lu has incorrect value. Expected: %ld, got: %ld",
                    gid, expected, value
                );
            }
        }
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    return error;
}

AUTO_TEST_CASE(test_work_group_barrier_global)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.barrier = barrier_type::global;
    options.num_tests = 1000;
    options.max_count = num_elements;
    return test(device, context, queue, options);
}

AUTO_TEST_CASE(test_work_group_barrier_local)
(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    test_options options;
    options.barrier = barrier_type::local;
    options.num_tests = 1000;
    options.max_count = num_elements;
    return test(device, context, queue, options);
}

} // namespace

#endif // TEST_CONFORMANCE_CLCPP_SYNCHRONIZATION_TEST_WORK_GROUP_BARRIER_HPP
