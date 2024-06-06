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
#include <vector>

//--------------------------------------------------------------------------
namespace {

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandBufferBarrierNotNullQueue : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandBarrierWithWaitListKHR(
            command_buffer, queue, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;
        return is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferBarrierInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandBarrierWithWaitListKHR(
            nullptr, queue, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferBarrierBufferFinalized : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferBarrierMutableHandleNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandBarrierWithWaitListKHR(
            command_buffer, nullptr, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferBarrierSyncPointsNullOrNumZero
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_sync_point_khr invalid_point = 0;

        cl_int error = clCommandBarrierWithWaitListKHR(
            command_buffer, nullptr, 1, &invalid_point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 1,
                                                nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        cl_sync_point_khr point;
        error =
            clCommandCopyBufferKHR(command_buffer, nullptr, in_mem, out_mem, 0,
                                   0, data_size(), 0, nullptr, &point, nullptr);
        test_error(error, "clCommandCopyBufferKHR failed");

        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                &point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandBarrierWithWaitListKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};
};

int test_negative_command_buffer_barrier_not_null_queue(cl_device_id device,
                                                        cl_context context,
                                                        cl_command_queue queue,
                                                        int num_elements)
{
    return MakeAndRunTest<CommandBufferBarrierNotNullQueue>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_barrier_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferBarrierInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_barrier_buffer_finalized(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferBarrierBufferFinalized>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_barrier_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferBarrierMutableHandleNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_barrier_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferBarrierSyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}
