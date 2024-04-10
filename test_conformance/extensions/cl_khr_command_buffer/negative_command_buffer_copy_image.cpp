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
struct CommandBufferCopyImageQueueNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(command_buffer, queue, src_image,
                                             dst_image, origin, origin, region,
                                             0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(command_buffer, queue, src_image,
                                              dst_image, origin, origin, 0, 0,
                                              nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, src_image, and dst_image are not the same.
struct CommandBufferCopyImageContextNotSame : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(
            command_buffer, nullptr, src_image_ctx, dst_image, origin, origin,
            region, 0, 0, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image_ctx, dst_image, origin, origin,
            0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                      dst_image_ctx, origin, origin, region, 0,
                                      nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image_ctx, origin, origin,
            0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);


        command_buffer = clCreateCommandBufferKHR(1, &queue1, 0, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                      dst_image, origin, origin, region, 0,
                                      nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 0,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        context1 = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        src_image_ctx = create_image_2d(context1, CL_MEM_READ_ONLY, &formats,
                                        512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image_ctx = create_image_2d(context1, CL_MEM_WRITE_ONLY, &formats,
                                        512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        queue1 = clCreateCommandQueue(context1, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clContextWrapper context1;
    clCommandQueueWrapper queue1;
    clMemWrapper src_image;
    clMemWrapper dst_image;
    clMemWrapper src_image_ctx;
    clMemWrapper dst_image_ctx;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferCopySyncPointsNullOrNumZero : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                             dst_image, origin, origin, region,
                                             1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 1,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        cl_sync_point_khr point = 1;
        std::vector<cl_sync_point_khr> sync_points;
        sync_points.push_back(point);

        error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                      dst_image, origin, origin, region, 0,
                                      sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 0,
            sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        cl_sync_point_khr* invalid_point = nullptr;
        std::vector<cl_sync_point_khr*> invalid_sync_points;
        invalid_sync_points.push_back(invalid_point);

        error = clCommandCopyImageKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin,
            region, 1, *invalid_sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 1,
            *invalid_sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCopyImageInvalidCommandBuffer
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandCopyImageKHR(nullptr, nullptr, src_image,
                                             dst_image, origin, origin, region,
                                             0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(nullptr, nullptr, src_image,
                                              dst_image, origin, origin, 0, 0,
                                              nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCopyImageFinalizedCommandBuffer
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                      dst_image, origin, origin, region, 0,
                                      nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 0,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCopyImageMutableHandleNotNull
    : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;
        cl_int error = clCommandCopyImageKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin,
            region, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, src_image, dst_image, origin, origin, 0, 0,
            nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyImageKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);


        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats, 512,
                                    512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
};
};

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
