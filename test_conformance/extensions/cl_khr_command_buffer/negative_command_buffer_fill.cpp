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
template <bool check_image_support>
struct CommandFillBaseTest : BasicCommandBufferTest
{
    CommandFillBaseTest(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");
        if (check_image_support)
        {
            src_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats,
                                        512, 512, 0, NULL, &error);
            test_error(error, "create_image_2d failed");

            dst_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats,
                                        512, 512, 0, NULL, &error);
            test_error(error, "create_image_2d failed");
        }

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (check_image_support)
        {
            cl_bool image_support;

            cl_int error =
                clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                                sizeof(image_support), &image_support, nullptr);
            test_error(error,
                       "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

            return (!image_support || BasicCommandBufferTest::Skip());
        }
        return BasicCommandBufferTest::Skip();
    }

protected:
    clMemWrapper src_image;
    clMemWrapper dst_image;
    const cl_uint fill_color_1[4] = { 0x05, 0x05, 0x05, 0x05 };
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { 512, 512, 1 };
    const cl_int pattern = 0xFF;
};

namespace {

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandBufferCommandFillBufferQueueNotNull
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, queue, out_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return CommandFillBaseTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }
};

// CL_INVALID_COMMAND_QUEUE if command_queue is not NULL.
struct CommandBufferCommandFillImageQueueNotNull
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillImageKHR(command_buffer, queue, src_image,
                                             fill_color_1, origin, region, 0,
                                             nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_QUEUE,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_COMMAND_QUEUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return CommandFillBaseTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, and buffer are not the same.
struct CommandBufferCommandFillBufferContextNotSame
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, out_mem_ctx, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CommandFillBaseTest::SetUp(elements);
        test_error(error, "CommandFillBaseTest::SetUp failed");

        context1 = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        out_mem_ctx =
            clCreateBuffer(context1, CL_MEM_WRITE_ONLY,
                           sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");


        return CL_SUCCESS;
    }

    clContextWrapper context1;
    clMemWrapper out_mem_ctx;
};

// CL_INVALID_CONTEXT if the context associated with command_queue,
// command_buffer, and buffer are not the same.
struct CommandBufferCommandFillImageContextNotSame
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillImageKHR(
            command_buffer, nullptr, dst_image_ctx, fill_color_1, origin,
            region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_CONTEXT,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_CONTEXT",
                               TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CommandFillBaseTest::SetUp(elements);
        test_error(error, "CommandFillBaseTest::SetUp failed");

        context1 = clCreateContext(0, 1, &device, nullptr, nullptr, &error);
        test_error(error, "Failed to create context");

        dst_image_ctx = create_image_2d(context1, CL_MEM_WRITE_ONLY, &formats,
                                        512, 512, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    clContextWrapper context1;
    clMemWrapper dst_image_ctx;
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferCommandFillBufferSyncPointsNullOrNumZero
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_sync_point_khr invalid_point = 0;

        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, out_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 1, &invalid_point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandFillBufferKHR(command_buffer, nullptr, out_mem,
                                       &pattern, sizeof(cl_int), 0, data_size(),
                                       1, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        cl_sync_point_khr point;
        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                nullptr, &point, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");

        error = clCommandFillBufferKHR(command_buffer, nullptr, out_mem,
                                       &pattern, sizeof(cl_int), 0, data_size(),
                                       0, &point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_SYNC_POINT_WAIT_LIST_KHR if sync_point_wait_list is NULL and
// num_sync_points_in_wait_list is > 0, or sync_point_wait_list is not NULL and
// num_sync_points_in_wait_list is 0, or if synchronization-point objects in
// sync_point_wait_list are not valid synchronization-points.
struct CommandBufferCommandFillImageSyncPointsNullOrNumZero
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_sync_point_khr invalid_point = 0;

        cl_int error = clCommandFillImageKHR(command_buffer, nullptr, dst_image,
                                             fill_color_1, origin, region, 1,
                                             &invalid_point, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        error = clCommandFillImageKHR(command_buffer, nullptr, dst_image,
                                      fill_color_1, origin, region, 1, nullptr,
                                      nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);

        cl_sync_point_khr point;
        error = clCommandBarrierWithWaitListKHR(command_buffer, nullptr, 0,
                                                nullptr, &point, nullptr);
        test_error(error, "clCommandBarrierWithWaitListKHR failed");


        error = clCommandFillImageKHR(command_buffer, nullptr, dst_image,
                                      fill_color_1, origin, region, 0, &point,
                                      nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_SYNC_POINT_WAIT_LIST_KHR,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_SYNC_POINT_WAIT_LIST_KHR",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};


// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCommandFillBufferInvalidCommandBuffer
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            nullptr, nullptr, out_mem, &pattern, sizeof(cl_int), 0, data_size(),
            0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct CommandBufferCommandFillImageInvalidCommandBuffer
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandFillImageKHR(nullptr, nullptr, dst_image, fill_color_1,
                                  origin, region, 0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCommandFillBufferFinalizedCommandBuffer
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clCommandFillBufferKHR(command_buffer, nullptr, out_mem,
                                       &pattern, sizeof(cl_int), 0, data_size(),
                                       0, nullptr, nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer has been finalized.
struct CommandBufferCommandFillImageFinalizedCommandBuffer
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");


        error = clCommandFillImageKHR(command_buffer, nullptr, dst_image,
                                      fill_color_1, origin, region, 0, nullptr,
                                      nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCommandFillBufferMutableHandleNotNull
    : public CommandFillBaseTest<false>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, out_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandFillBufferKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);


        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if mutable_handle is not NULL.
struct CommandBufferCommandFillImageMutableHandleNotNull
    : public CommandFillBaseTest<true>
{
    using CommandFillBaseTest::CommandFillBaseTest;

    cl_int Run() override
    {
        cl_mutable_command_khr mutable_handle;

        cl_int error = clCommandFillImageKHR(command_buffer, nullptr, dst_image,
                                             fill_color_1, origin, region, 0,
                                             nullptr, nullptr, &mutable_handle);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clCommandFillImageKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

}

int test_negative_command_buffer_command_fill_buffer_queue_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillBufferQueueNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_queue_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageQueueNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_buffer_context_not_same(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillBufferContextNotSame>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_context_not_same(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageContextNotSame>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_buffer_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<
        CommandBufferCommandFillBufferSyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_sync_points_null_or_num_zero(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageSyncPointsNullOrNumZero>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_buffer_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillBufferInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_buffer_finalized_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillBufferFinalizedCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_finalized_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageFinalizedCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_buffer_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillBufferMutableHandleNotNull>(
        device, context, queue, num_elements);
}

int test_negative_command_buffer_command_fill_image_mutable_handle_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCommandFillImageMutableHandleNotNull>(
        device, context, queue, num_elements);
}
