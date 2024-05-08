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
#include "svm_command_basic.h"
#include "procs.h"


//--------------------------------------------------------------------------
namespace {

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandBufferCommandSVMQueueNotNull : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandSVMMemcpyKHR(
            command_buffer, queue, svm_out_mem(), svm_in_mem(), data_size(), 0,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_COMMAND_QUEUE,
            "clCommandSVMMemcpyKHR should return CL_INVALID_COMMAND_QUEUE",
            TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, queue, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_COMMAND_QUEUE,
            "clCommandSVMMemFillKHR should return CL_INVALID_COMMAND_QUEUE",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    const cl_char pattern_1 = 0x14;
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, and kernel are not the same.
struct CommandBufferCommandSVMDifferentContext
    : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandSVMMemcpyKHR(
            command_buffer, nullptr, svm_out_mem(), svm_in_mem(), data_size(),
            0, nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_CONTEXT,
            "clCommandSVMMemcpyKHR should return CL_INVALID_CONTEXT",
            TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_CONTEXT,
            "clCommandSVMMemFillKHR should return CL_INVALID_CONTEXT",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        error = CreateKernelWithDifferentContext();
        test_error(error, "Failed to create kernel");

        size_t size = sizeof(cl_int) * num_elements * buffer_size_multiplier;
        svm_in_mem = clSVMWrapper(new_context, size);
        if (svm_in_mem() == nullptr)
        {
            test_error(CL_OUT_OF_RESOURCES, "Unable to allocate SVM memory");
        }
        svm_out_mem = clSVMWrapper(new_context, size);
        if (svm_out_mem() == nullptr)
        {
            test_error(CL_OUT_OF_RESOURCES, "Unable to allocate SVM memory");
        }


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

    const cl_char pattern_1 = 0x14;
    clContextWrapper new_context = nullptr;
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferCommandSVMSyncPointsNullOrNumZero
    : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_sync_point_khr invalid_point = 0;
        cl_sync_point_khr* invalid_sync_points[] = { &invalid_point };

        cl_int error = clCommandSVMMemcpyKHR(
            command_buffer, nullptr, svm_out_mem(), svm_in_mem(), data_size(),
            1, invalid_sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemcpyKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandSVMMemFillKHR(
            command_buffer, nullptr, svm_in_mem(), &pattern_1, sizeof(cl_char),
            data_size(), 1, invalid_sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemFillKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandSVMMemcpyKHR(command_buffer, nullptr, svm_out_mem(),
                                      svm_in_mem(), data_size(), 1, nullptr,
                                      nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemcpyKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemFillKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        cl_sync_point_khr point;
        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                nullptr, &point, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");
        cl_sync_point_khr* sync_points[] = { &point };

        error = clCommandSVMMemcpyKHR(command_buffer, nullptr, svm_out_mem(),
                                      svm_in_mem(), data_size(), 0,
                                      sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemcpyKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, sync_points[0], nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandSVMMemFillKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    const cl_char pattern_1 = 0x14;
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCommandSVMInvalidCommandBuffer
    : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandSVMMemcpyKHR(nullptr, nullptr, svm_out_mem(), svm_in_mem(),
                                  data_size(), 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_COMMAND_BUFFER_KHR,
            "clCommandSVMMemcpyKHR should return CL_INVALID_COMMAND_BUFFER_KHR",
            TEST_FAIL);

        error = clCommandSVMMemFillKHR(nullptr, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandSVMMemFillKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    const cl_char pattern_1 = 0x14;
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCommandSVMFinalizedCommandBuffer
    : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandSVMMemcpyKHR(command_buffer, nullptr, svm_out_mem(),
                                      svm_in_mem(), data_size(), 0, nullptr,
                                      nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_OPERATION,
            "clCommandSVMMemcpyKHR should return CL_INVALID_OPERATION",
            TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, nullptr, nullptr, nullptr);

        test_failure_error_ret(
            error, CL_INVALID_OPERATION,
            "clCommandSVMMemFillKHR should return CL_INVALID_OPERATION",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    const cl_char pattern_1 = 0x14;
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCommandSVMMutableHandleNotNull
    : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandSVMMemcpyKHR(
            command_buffer, nullptr, svm_out_mem(), svm_in_mem(), data_size(),
            0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCommandSVMMemcpyKHR should return CL_INVALID_VALUE", TEST_FAIL);

        error = clCommandSVMMemFillKHR(command_buffer, nullptr, svm_in_mem(),
                                       &pattern_1, sizeof(cl_char), data_size(),
                                       0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCommandSVMMemFillKHR should return CL_INVALID_VALUE", TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicSVMCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    const cl_char pattern_1 = 0x14;
};
};

int test_negative_command_buffer_command_svm_queue_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMQueueNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_svm_different_context(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMDifferentContext>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_svm_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMSyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_svm_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_svm_finalized_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMFinalizedCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_svm_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandSVMMutableHandleNotNull>(
        device, context, queue, num_elements);
}
