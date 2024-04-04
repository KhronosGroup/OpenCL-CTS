//
// Copyright (c) 2024 The Khronos Group Inc.
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
#include "basic_command_buffer.h"
#include "procs.h"


//--------------------------------------------------------------------------
namespace {

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandNDRangeKernelQueueNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, queue, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, and kernel are not the same.
struct CommandNDRangeKernelKernelWithDifferentContext
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = CreateKernelWithDifferentContext();
        test_error(error, "Failed to create kernel");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int CreateKernelWithDifferentContext()
    {
        cl_int error = CL_SUCCESS;

        new_context = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        const char* kernel_str =
            R"(
      __kernel void copy(__global int* in, __global int* out, __global int* offset) {
          size_t id = get_global_id(0);
          int ind = offset[0] + id;
          out[ind] = in[ind];
      })";

        error = create_single_kernel_helper_create_program(
            new_context, &program, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "copy", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }
    clContextWrapper new_context;
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandNDRangeKerneSyncPointsNullOrNumZero
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        cl_sync_point_khr point = 1;
        cl_sync_point_khr* sync_points[] = { &point };
        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 0, nullptr, &num_elements,
            nullptr, 0, sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        cl_sync_point_khr* invalid_point = nullptr;
        cl_sync_point_khr* invalid_sync_points[] = { invalid_point };
        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 0, nullptr, &num_elements,
            nullptr, 0, invalid_sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandNDRangeKernelInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            nullptr, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if values specified in properties are not valid.
struct CommandNDRangeKernelInvalidProperties : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_ndrange_kernel_command_properties_khr empty_properties;

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, &empty_properties, kernel, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        cl_ndrange_kernel_command_properties_khr props_invalid[3] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MEM_USE_CACHED_CPU_MEMORY_IMG, 1
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props_invalid, kernel, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandNDRangeKernelCommandBufferFinalized
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandNDRangeKernelMutableHandleNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if the device associated with command_queue does not
// support CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR and kernel contains a
// printf call.
struct CommandNDRangeKernelWithPrintDeviceDoesNotSupportPrint
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandNDRangeKernelKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        cl_device_command_buffer_capabilities_khr capabilities;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR");

        bool printf_support =
            (capabilities & CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR)
            != 0;

        return !printf_support;
    }

    cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;

        const char* kernel_str =
            R"(
      __kernel void print(__global char* in, __global char* out, __global int* offset)
      {
          size_t id = get_global_id(0);
          int ind = offset[0] + offset[1] * id;
          for(int i=0; i<offset[1]; i++) {
              out[ind+i] = in[i];
              printf("%c", in[i]);
          }
      })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "print", &error);
        test_error(error, "Failed to create print kernel");

        return CL_SUCCESS;
    }
};
};

int test_negative_command_ndrange_queue_not_null(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelQueueNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_with_different_context(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelKernelWithDifferentContext>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKerneSyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_invalid_properties(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelInvalidProperties>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_command_buffer_finalized(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelCommandBufferFinalized>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandNDRangeKernelMutableHandleNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_ndrange_kernel_with_print_device_does_not_support_print(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<
        CommandNDRangeKernelWithPrintDeviceDoesNotSupportPrint>(
        device, context, queue, num_elements);
}
