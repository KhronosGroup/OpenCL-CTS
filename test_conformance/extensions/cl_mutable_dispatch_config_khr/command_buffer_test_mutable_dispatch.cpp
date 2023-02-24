//
// Copyright (c) 2022 The Khronos Group Inc.
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
#include <extensionHelpers.h>
#include "procs.h"
#include "testHarness.h"
#include <vector>
#include <iostream>
#include <random>
#include <cstring>
#include <algorithm>
#include <memory>

#include <CL/cl.h>
#include <CL/cl_ext.h>
////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle below cases:
//
// CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR
// CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR
// CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR
// CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR
// CL_MUTABLE_DISPATCH_KERNEL_KHR
// CL_MUTABLE_DISPATCH_DIMENSIONS_KHR
// CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR
// CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR
// CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR
// CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR


#define CL_CHECK(ERROR)                                                        \
    if (ERROR)                                                                 \
    {                                                                          \
        std::cerr << "OpenCL error: " << ERROR << "\n";                        \
        return ERROR;                                                          \
    }


static const char* kernelString = R"CLC(
        kernel void CopyBuffer( global uint* dst, global uint* src )
        {
            uint id = get_global_id(0);
            dst[id] = src[id];
        }
        )CLC";


int check_capabilities(cl_device_id device, cl_context context)
{
    GET_PFN(device, clGetMutableCommandInfoKHR);

    cl_mutable_dispatch_fields_khr mutable_capabilities;
    CL_CHECK(clGetDeviceInfo(
        device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
        sizeof(mutable_capabilities), &mutable_capabilities, nullptr));
    if (!(mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR))
    {
        std::cerr << "Device does not support update arguments to a "
                     "mutable-dispatch, "
                     "skipping example.\n";
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int query_test_common(cl_device_id& device, cl_context& context,
                      cl_command_queue& queue,
                      cl_command_buffer_khr& command_buffer, cl_kernel& kernel)
{
    cl_int error = CL_SUCCESS;

    GET_PFN(device, clCreateCommandBufferKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);

    const cl_command_buffer_properties_khr props[] = {
        CL_COMMAND_BUFFER_FLAGS_KHR,
        CL_COMMAND_BUFFER_MUTABLE_KHR,
        0,
    };

    command_buffer = clCreateCommandBufferKHR(1, &queue, props, nullptr);

    cl_program program =
        clCreateProgramWithSource(context, 1, &kernelString, nullptr, &error);
    test_error(error, "Unable to create program");

    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    test_error(error, "Unable to build program");

    kernel = clCreateKernel(program, "CopyBuffer", &error);
    test_error(error, "Unable to create kernel");

    return error;
}

int test_mutable_dispatch_device_query(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    GET_PFN(device, clUpdateMutableCommandsKHR);
    GET_PFN(device, clCreateCommandBufferKHR);
    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_mutable_dispatch_fields_khr mutable_capabilities;
    CL_CHECK(clGetDeviceInfo(
        device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
        sizeof(mutable_capabilities), &mutable_capabilities, nullptr));
    if (!(mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR))
    {
        std::cerr << "Device does not support update arguments to a "
                     "mutable-dispatch, "
                     "skipping example.\n";
        return 0;
    }

    return TEST_PASS;
}

int test_mutable_dispatch_command_buffer(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_int error = check_capabilities(device, context);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t global_work_size = 1;
    cl_sync_point_khr sync_point;

    error = clCommandNDRangeKernelKHR(command_buffer, nullptr, nullptr, kernel,
                                      1, nullptr, &global_work_size, nullptr, 0,
                                      nullptr, &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    cl_command_buffer_khr test_command_buffer = nullptr;
    error = clGetMutableCommandInfoKHR(
        command, CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR,
        sizeof(test_command_buffer), &test_command_buffer, NULL);
    test_error(error, "clGetMutableCommandInfoKHR failed");

    if (test_command_buffer != command_buffer)
    {
        log_error("ERROR: Incorrect command buffer returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}

int test_mutable_dispatch_command_type(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    cl_int error = check_capabilities(device, context);

    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t global_work_size = 1;
    cl_sync_point_khr sync_point;

    error = clCommandNDRangeKernelKHR(command_buffer, nullptr, nullptr, kernel,
                                      1, nullptr, &global_work_size, nullptr, 0,
                                      nullptr, &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    cl_command_type type = 0;
    error =
        clGetMutableCommandInfoKHR(command, CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR,
                                   sizeof(type), &type, NULL);
    test_error(error, "clGetMutableCommandInfoKHR failed");

    if (type == 0)
    {
        log_error("ERROR: Wrong type returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}

int test_mutable_dispatch_command_queue(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    cl_int error = check_capabilities(device, context);

    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t global_work_size = 1;
    cl_sync_point_khr sync_point;

    error = clCommandNDRangeKernelKHR(command_buffer, nullptr, nullptr, kernel,
                                      1, nullptr, &global_work_size, nullptr, 0,
                                      nullptr, &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    cl_command_queue testQueue = nullptr;
    error = clGetMutableCommandInfoKHR(command,
                                       CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR,
                                       sizeof(testQueue), &testQueue, nullptr);
    test_error(error, "clGetMutableCommandInfoKHR failed");

    if (!testQueue)
    {
        log_error("ERROR: Incorrect queue returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}

int test_mutable_dispatch_global_work_offset(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    cl_int error = check_capabilities(device, context);

    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t global_work_size = 1;
    cl_sync_point_khr sync_point;
    const size_t work_offset[] = { 8, 8, 8 };

    error = clCommandNDRangeKernelKHR(
        command_buffer, nullptr, nullptr, kernel, 1, nullptr, &global_work_size,
        work_offset, 0, nullptr, &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    std::shared_ptr<size_t> global_work_offset = nullptr;

    error = clGetMutableCommandInfoKHR(
        command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR,
        sizeof(global_work_offset), global_work_offset.get(), NULL);

    if (!global_work_offset.get())
    {
        log_error("ERROR: Wrong size returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}

int test_mutable_dispatch_local_work_size(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    cl_int error = check_capabilities(device, context);

    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t global_work_size = 1;
    cl_sync_point_khr sync_point;

    error = clCommandNDRangeKernelKHR(command_buffer, nullptr, nullptr, kernel,
                                      1, nullptr, &global_work_size, nullptr, 0,
                                      nullptr, &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    std::shared_ptr<size_t> local_work_size = nullptr;

    error = clGetMutableCommandInfoKHR(
        command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
        sizeof(local_work_size), &local_work_size, NULL);

    if (local_work_size != 0u)
    {
        log_error("ERROR: Wrong size returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}

int test_mutable_dispatch_global_work_size(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    GET_PFN(device, clFinalizeCommandBufferKHR);
    GET_PFN(device, clGetMutableCommandInfoKHR);
    GET_PFN(device, clCommandNDRangeKernelKHR);

    cl_int error = check_capabilities(device, context);

    cl_command_buffer_khr command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    cl_kernel kernel = nullptr;

    error = query_test_common(device, context, queue, command_buffer, kernel);

    const size_t size = 1;
    cl_sync_point_khr sync_point;

    error = clCommandNDRangeKernelKHR(command_buffer, nullptr, nullptr, kernel,
                                      1, nullptr, &size, nullptr, 0, nullptr,
                                      &sync_point, &command);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    std::shared_ptr<size_t> global_work_size = nullptr;
    error = clGetMutableCommandInfoKHR(
        command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
        sizeof(global_work_size), &global_work_size, NULL);

    if (global_work_size != 0u)
    {
        log_error("ERROR: Wrong size returned from "
                  "clGetMutableCommandInfoKHR.");
        return TEST_FAIL;
    }

    clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    return TEST_PASS;
}
