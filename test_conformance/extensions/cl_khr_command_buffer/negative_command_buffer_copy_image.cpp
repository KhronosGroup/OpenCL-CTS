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

struct CommandCopyBaseTest : BasicCommandBufferTest
{
    CommandCopyBaseTest(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    cl_int SetUp(int elements) override
    {
        num_elements = elements;
        origin[0] = origin[1] = origin[2] = 0;
        region[0] = elements / 64;
        region[1] = 64;
        region[2] = 1;
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats,
                                    elements / 64, 64, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats,
                                    elements / 64, 64, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        cl_bool image_support;

        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(image_support), &image_support, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

        return (!image_support || BasicCommandBufferTest::Skip());
    }

protected:
    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    size_t origin[3];
    size_t region[3];
};

namespace {

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandBufferCopyImageQueueNotNull : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(
            command_buffer, queue, nullptr, src_image, dst_image, origin,
            origin, region, 0, nullptr, nullptr, nullptr);


        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, queue, nullptr, src_image, out_mem, origin, region,
            0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return CommandCopyBaseTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, src_image, and dst_image are not the same.
struct CommandBufferCopyImageContextNotSame : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(
            command_buffer, nullptr, nullptr, src_image_ctx, dst_image, origin,
            origin, region, 0, 0, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image_ctx, out_mem, origin,
            region, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image_ctx, origin, origin,
                                      region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem_ctx, origin,
            region, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);


        command_buffer = clCreateCommandBufferKHR(1, &queue1, 0, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image, origin, origin,
                                      region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CommandCopyBaseTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        context1 = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        src_image_ctx = create_image_2d(context1, CL_MEM_READ_ONLY, &formats,
                                        elements / 64, 64, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image_ctx = create_image_2d(context1, CL_MEM_WRITE_ONLY, &formats,
                                        elements / 64, 64, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        out_mem_ctx = clCreateBuffer(context1, CL_MEM_WRITE_ONLY,
                                     sizeof(cl_int) * num_elements
                                         * buffer_size_multiplier,
                                     nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        queue1 = clCreateCommandQueue(context1, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        return CL_SUCCESS;
    }

    clContextWrapper context1;
    clCommandQueueWrapper queue1;
    clMemWrapper src_image_ctx;
    clMemWrapper dst_image_ctx;
    clMemWrapper out_mem_ctx;
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferCopySyncPointsNullOrNumZero : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_sync_point_khr invalid_point = 0;

        cl_int error = clCommandCopyImageKHR(
            command_buffer, nullptr, nullptr, src_image, dst_image, origin,
            origin, region, 1, &invalid_point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 1, &invalid_point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image, origin, origin,
                                      region, 1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        cl_sync_point_khr point;
        error = clCommandBarrierWithWaitListKHR(
            command_buffer, nullptr, nullptr, 0, nullptr, &point, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image, origin, origin,
                                      region, 0, &point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 0, &point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCopyImageInvalidCommandBuffer : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(
            nullptr, nullptr, nullptr, src_image, dst_image, origin, origin,
            region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            nullptr, nullptr, nullptr, src_image, out_mem, origin, region, 0, 0,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCopyImageFinalizedCommandBuffer : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image, origin, origin,
                                      region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCopyImageMutableHandleNotNull : public CommandCopyBaseTest
{
    using CommandCopyBaseTest::CommandCopyBaseTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;
        cl_int error = clCommandCopyImageKHR(
            command_buffer, nullptr, nullptr, src_image, dst_image, origin,
            origin, region, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, nullptr, src_image, out_mem, origin,
            region, 0, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};
}

int test_negative_command_buffer_copy_image_queue_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageQueueNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_image_context_not_same(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageContextNotSame>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_image_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopySyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_image_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_image_finalized_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageFinalizedCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_image_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageMutableHandleNotNull>(
        device, context, queue, num_elements);
}
