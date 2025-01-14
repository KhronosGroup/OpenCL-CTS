//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include <cinttypes>

#include "testBase.h"
#include "harness/testHarness.h"
#include <memory>

static const char* local_memory_kernel = R"CLC(
__kernel void local_memory_kernel(global int* data) {
    __local int array[10];

    size_t id = get_global_id(0);
    array[id] = 2 * id;
    data[id] = array[id];

    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0)
    {
        for(size_t i = 0; i < 10; i++)
            data[id] += array[i];
    }
}
)CLC";

static const char* local_param_kernel = R"CLC(
__kernel void local_param_kernel(__local int* local_ptr, __global int* src,
    __global int* dst) {

    size_t id = get_global_id(0);

    local_ptr[id] = src[id];
    barrier(CLK_GLOBAL_MEM_FENCE);
    dst[id] = local_ptr[id];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 9)
    {
        for(size_t i = 0; i < 10; i++)
            dst[id] += local_ptr[i];
    }
}
)CLC";

static const char* local_param_local_memory_kernel = R"CLC(
__kernel void local_param_local_memory_kernel(__local int* local_ptr,
    __global int* src, __global int* dst) {

    size_t id = get_global_id(0);

    __local int local_data[10];
    local_ptr[id] = src[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    local_data[id] = local_ptr[id] * 2;
    barrier(CLK_LOCAL_MEM_FENCE);

    dst[id] = local_data[id];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 9)
    {
        for(size_t i = 0; i < 10; i++)
            dst[id] += local_data[i];
        dst[id] += 666;
    }
}
)CLC";

REGISTER_TEST(kernel_local_memory_size)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_ulong kernel_local_usage = 0;
    size_t param_value_size_ret = 0;

    // Check memory needed to execute empty kernel with __local variable
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    &local_memory_kernel, "local_memory_kernel")
        != 0)
    {
        return TEST_FAIL;
    }

    error = clGetKernelWorkGroupInfo(
        kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernel_local_usage),
        &kernel_local_usage, &param_value_size_ret);
    test_error(error,
               "clGetKernelWorkGroupInfo for CL_KERNEL_LOCAL_MEM_SIZE failed");

    test_assert_error(param_value_size_ret == sizeof(cl_ulong),
                      "param_value_size_ret failed");

    constexpr size_t size = 10;
    constexpr size_t memory = size * sizeof(cl_int);

    const size_t global_work_size[] = { size };
    const size_t local_work_size[] = { size };

    int data[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = 0;
    }
    clMemWrapper streams[2];

    streams[0] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, memory, NULL, &error);
    test_error(error, "Creating test array failed");

    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size,
                                   local_work_size, 0, NULL, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed.");

    error = clEnqueueReadBuffer(queue, streams[0], CL_TRUE, 0, memory, data, 0,
                                NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    auto local_memory_kernel_verify = [&]() {
        constexpr size_t size = 10;
        int testData[size];
        for (size_t i = 0; i < size; i++)
        {
            testData[i] = i * 2;
            testData[0] += testData[i];
        }
        for (size_t i = 0; i < size; i++)
        {
            if (data[i] != testData[i]) return false;
        }
        return true;
    };
    test_assert_error(local_memory_kernel_verify(),
                      "local_memory_kernel data verification failed");

    test_assert_error(kernel_local_usage >= memory,
                      "kernel local mem size failed");


    // Check memory needed to execute empty kernel with __local parameter with
    // setKernelArg
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    &local_param_kernel, "local_param_kernel")
        != 0)
    {
        return TEST_FAIL;
    }

    kernel_local_usage = 0;
    param_value_size_ret = 0;

    for (size_t i = 0; i < size; i++)
    {
        data[i] = i;
    }

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, memory, data, &error);
    test_error(error, "Creating test array failed");
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, memory, nullptr, &error);
    test_error(error, "Creating test array failed");

    error = clSetKernelArg(kernel, 0, memory, NULL);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 2, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size,
                                   local_work_size, 0, NULL, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed.");

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, memory, data, 0,
                                NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clGetKernelWorkGroupInfo(
        kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernel_local_usage),
        &kernel_local_usage, &param_value_size_ret);
    test_error(error,
               "clGetKernelWorkGroupInfo for CL_KERNEL_LOCAL_MEM_SIZE failed");

    test_assert_error(param_value_size_ret == sizeof(cl_ulong),
                      "param_value_size_ret failed");

    auto local_param_kernel_verify = [&]() {
        constexpr size_t size = 10;
        int testData[size];
        int sum = 0;
        for (size_t i = 0; i < size; i++)
        {
            testData[i] = i;
            sum += testData[i];
        }
        testData[9] += sum;
        for (size_t i = 0; i < size; i++)
        {
            if (data[i] != testData[i]) return false;
        }

        return true;
    };
    test_assert_error(local_param_kernel_verify(),
                      "local_param_kernel data verificaion failed");

    test_assert_error(kernel_local_usage >= memory,
                      "kernel local mem size failed");


    // Check memory needed to execute kernel with __local variable and __local
    // parameter with setKernelArg
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    &local_param_local_memory_kernel,
                                    "local_param_local_memory_kernel")
        != 0)
    {
        return TEST_FAIL;
    }

    kernel_local_usage = 0;
    param_value_size_ret = 0;

    for (size_t i = 0; i < size; i++)
    {
        data[i] = i;
    }

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, memory, data, &error);
    test_error(error, "Creating test array failed");
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, memory, nullptr, &error);
    test_error(error, "Creating test array failed");

    error = clSetKernelArg(kernel, 0, memory, NULL);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 2, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size,
                                   local_work_size, 0, NULL, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed.");

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, memory, data, 0,
                                NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");


    error = clGetKernelWorkGroupInfo(
        kernel, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernel_local_usage),
        &kernel_local_usage, &param_value_size_ret);
    test_error(error,
               "clGetKernelWorkGroupInfo for CL_KERNEL_LOCAL_MEM_SIZE failed");

    test_assert_error(param_value_size_ret == sizeof(cl_ulong),
                      "param_value_size_ret failed");

    auto local_param_local_memory_kernel_verify = [&]() {
        constexpr size_t size = 10;
        int testData[size];
        for (size_t i = 0; i < size; i++)
        {
            testData[i] = i * 2;
        }

        int temp = testData[9];
        for (size_t i = 0; i < size; i++)
        {
            if (i == 9)
                testData[9] += temp;
            else
                testData[9] += testData[i];
        }
        testData[9] += 666;

        for (size_t i = 0; i < size; i++)
        {
            if (data[i] != testData[i]) return false;
        }

        return true;
    };
    test_assert_error(
        local_param_local_memory_kernel_verify(),
        "local_param_local_memory_kernel data verificaion failed");

    test_assert_error(kernel_local_usage >= 2 * memory,
                      "kernel local mem size failed");

    return CL_SUCCESS;
}
