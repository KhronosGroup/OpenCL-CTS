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
struct CommandBufferCopyQueueNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandCopyBufferKHR(command_buffer, queue, in_mem, out_mem, 0, 0,
                                   data_size(), 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };

        error = clCommandCopyBufferRectKHR(
            command_buffer, queue, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, 512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             data_size(), nullptr, &error);
        test_error(error, "Unable to create buffer");

        error = clCommandCopyImageToBufferKHR(command_buffer, queue, image,
                                              buffer, origin, region, 0, 0,
                                              nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, src_buffer, and dst_buffer are not the same.
struct CommandBufferCopyDifferentContexts : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        context1 = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        in_mem_ctx =
            clCreateBuffer(context1, CL_MEM_READ_ONLY,
                           sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem_ctx =
            clCreateBuffer(context1, CL_MEM_WRITE_ONLY,
                           sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        image_ctx = create_image_2d(context1, CL_MEM_READ_WRITE, &formats, 512,
                                    512, 0, NULL, &error);

        test_error(error, "create_image_2d failed");
        buffer_ctx = clCreateBuffer(context1, CL_MEM_READ_WRITE, data_size(),
                                    nullptr, &error);
        test_error(error, "Unable to create buffer");

        image = create_image_2d(context, CL_MEM_READ_WRITE, &formats, 512, 512,
                                0, NULL, &error);

        test_error(error, "create_image_2d failed");
        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size(),
                                nullptr, &error);
        test_error(error, "Unable to create buffer");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = clCommandCopyBufferKHR(
            command_buffer, nullptr, in_mem_ctx, out_mem, 0, 0, data_size(), 0,
            nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem_ctx, out_mem, origin, origin,
            region, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(command_buffer, nullptr,
                                              image_ctx, buffer, origin, region,
                                              0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyBufferKHR(command_buffer, nullptr, in_mem,
                                       out_mem_ctx, 0, 0, data_size(), 0,
                                       nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem_ctx, origin, origin,
            region, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(command_buffer, nullptr, image,
                                              buffer_ctx, origin, region, 0, 0,
                                              nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
    clMemWrapper in_mem_ctx = nullptr;
    clMemWrapper out_mem_ctx = nullptr;
    clMemWrapper image_ctx = nullptr;
    clMemWrapper buffer_ctx = nullptr;
    clMemWrapper image = nullptr;
    clMemWrapper buffer = nullptr;
    clContextWrapper context1 = nullptr;
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0,
// or sync_point_wait_list is not NULL and num_sync_points_in_wait_list is 0,
// or if synchronization-point objects in sync_point_wait_list are not valid
// synchronization-points.
struct CommandBufferCopySyncPointsNullOrNumZero : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };
        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, 512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             data_size(), nullptr, &error);
        test_error(error, "Unable to create buffer");


        cl_sync_point_khr invalid_point = 0;
        std::vector<cl_sync_point_khr*> invalid_sync_points;
        invalid_sync_points.push_back(&invalid_point);

        error = clCommandCopyBufferKHR(
            command_buffer, nullptr, in_mem, out_mem, 0, 0, data_size(), 1,
            *invalid_sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 1, *invalid_sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, image, buffer, origin, region, 0, 1,
            *invalid_sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandCopyBufferKHR(command_buffer, nullptr, in_mem, out_mem,
                                       0, 0, data_size(), 1, nullptr, nullptr,
                                       nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandCopyImageToBufferKHR(command_buffer, nullptr, image,
                                              buffer, origin, region, 0, 1,
                                              nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        cl_sync_point_khr point;
        std::vector<cl_sync_point_khr> sync_points;
        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                nullptr, &point, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");
        sync_points.push_back(point);

        error = clCommandCopyBufferKHR(command_buffer, nullptr, in_mem, out_mem,
                                       0, 0, data_size(), 0, sync_points.data(),
                                       nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 0, sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, image, buffer, origin, region, 0, 0,
            sync_points.data(), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCopyInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandCopyBufferKHR(nullptr, nullptr, in_mem, out_mem, 0, 0,
                                   data_size(), 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };

        error = clCommandCopyBufferRectKHR(nullptr, nullptr, in_mem, out_mem,
                                           origin, origin, region, 0, 0, 0, 0,
                                           0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, 512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             data_size(), nullptr, &error);
        test_error(error, "Unable to create buffer");

        error = clCommandCopyImageToBufferKHR(nullptr, nullptr, image, buffer,
                                              origin, region, 0, 0, nullptr,
                                              nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCopyFinalizedCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandCopyBufferKHR(command_buffer, nullptr, in_mem, out_mem,
                                       0, 0, data_size(), 0, nullptr, nullptr,
                                       nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, 512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             data_size(), nullptr, &error);
        test_error(error, "Unable to create buffer");

        error = clCommandCopyImageToBufferKHR(nullptr, nullptr, image, buffer,
                                              origin, region, 0, 0, nullptr,
                                              nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCopyMutableHandleNotNull : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandCopyBufferKHR(
            command_buffer, nullptr, in_mem, out_mem, 0, 0, data_size(), 0,
            nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        const size_t origin[3] = { 0, 0, 0 }, region[3] = { 512, 512, 1 };

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyBufferRectKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, 512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             data_size(), nullptr, &error);
        test_error(error, "Unable to create buffer");

        error = clCommandCopyImageToBufferKHR(
            command_buffer, nullptr, image, buffer, origin, region, 0, 0,
            nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandCopyImageToBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};
};

int test_negative_command_buffer_copy_queue_not_null(cl_device_id device,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyQueueNotNull>(device, context, queue,
                                                         num_elements);
}

int test_negative_command_buffer_copy_different_contexts(cl_device_id device,
                                                         cl_context context,
                                                         cl_command_queue queue,
                                                         int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyDifferentContexts>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopySyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_finalized_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyFinalizedCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_copy_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyMutableHandleNotNull>(
        device, context, queue, num_elements);
}
