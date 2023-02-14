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
#include "basic_command_buffer.h"
#include "harness/typeWrappers.h"
#include "procs.h"

#include <vector>


namespace {

////////////////////////////////////////////////////////////////////////////////
// Command-buffer copy tests which handles below cases:
//
// -copy image
// -copy buffer
// -copy buffer to image
// -copy image to buffer
// -copy buffer rect

struct CopyImageKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillImageKHR(command_buffer, nullptr, src_image,
                                             fill_color, origin, region, 0,
                                             nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillImageKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, src_image,
                                      dst_image, origin, origin, region, 0, 0,
                                      nullptr, nullptr);

        test_error(error, "clCommandCopyImageKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size);
        error = clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, 0,
                                   0, output_data.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        src_image = create_image_2d(context, CL_MEM_READ_ONLY, &formats,
                                    img_width, img_height, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        dst_image = create_image_2d(context, CL_MEM_WRITE_ONLY, &formats,
                                    img_width, img_height, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        bool imageSupport =
            checkForImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED;

        return imageSupport || BasicCommandBufferTest::Skip();
    }

    const size_t img_width = 512;
    const size_t img_height = 512;
    const size_t data_size = img_width * img_height * 4 * sizeof(cl_char);
    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, 1 };
    const cl_uint pattern = 0x05;
    const cl_uint fill_color[4] = { pattern, pattern, pattern, pattern };
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };
    clMemWrapper src_image;
    clMemWrapper dst_image;
};

struct CopyBufferKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, in_mem, &pattern, sizeof(cl_char), 0,
            data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferKHR(command_buffer, nullptr, in_mem, out_mem,
                                       0, 0, data_size(), 0, nullptr, nullptr,
                                       nullptr);
        test_error(error, "clCommandCopyBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size());
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    const cl_char pattern = 0x14;
};

struct CopyBufferToImageKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, buffer, &pattern, sizeof(cl_char), 0,
            data_size, 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferToImageKHR(command_buffer, nullptr, buffer,
                                              image, 0, origin, region, 0, 0,
                                              nullptr, nullptr);

        test_error(error, "clCommandCopyBufferToImageKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size);

        error = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                   output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        image = create_image_2d(context, CL_MEM_READ_WRITE, &formats, img_width,
                                img_height, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr,
                                &error);
        test_error(error, "Unable to create buffer");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        bool imageSupport =
            checkForImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED;

        return imageSupport || BasicCommandBufferTest::Skip();
    }

    const size_t img_width = 512;
    const size_t img_height = 512;
    const size_t data_size = img_width * img_height * 4 * sizeof(cl_char);
    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, 1 };
    const cl_char pattern = 0x11;
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

    clMemWrapper buffer;
    clMemWrapper image;
};

struct CopyImageToBufferKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandFillImageKHR(command_buffer, nullptr, image, fill_color,
                                  origin, region, 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillImageKHR failed");

        error = clCommandCopyImageToBufferKHR(command_buffer, nullptr, image,
                                              buffer, origin, region, 0, 0,
                                              nullptr, nullptr, nullptr);

        test_error(error, "clCommandCopyImageToBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size,
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(static_cast<cl_char>(pattern),
                                     output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        image = create_image_2d(context, CL_MEM_READ_WRITE, &formats, img_width,
                                img_height, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr,
                                &error);
        test_error(error, "Unable to create buffer");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        bool imageSupport =
            checkForImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED;

        return imageSupport || BasicCommandBufferTest::Skip();
    }

    const size_t img_width = 512;
    const size_t img_height = 512;
    const size_t data_size = img_width * img_height * 4 * sizeof(cl_char);
    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, 1 };
    const cl_uint pattern = 0x12;
    const cl_uint fill_color[4] = { pattern, pattern, pattern, pattern };
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

    clMemWrapper image;
    clMemWrapper buffer;
};

struct CopyBufferRectKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, in_mem, &pattern, sizeof(cl_char), 0,
            data_size, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, in_mem, out_mem, origin, origin, region, 0,
            0, 0, 0, 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandCopyBufferRectKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size,
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr,
                                &error);
        test_error(error, "clCreateBuffer failed");

        out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, data_size, nullptr,
                                 &error);
        test_error(error, "Unable to create buffer");

        return CL_SUCCESS;
    }

    const size_t img_width = 512;
    const size_t img_height = 512;
    const size_t data_size = img_width * img_height * sizeof(cl_char);
    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, 1 };
    const cl_char pattern = 0x13;

    clMemWrapper in_mem;
    clMemWrapper out_mem;
};
};

int test_copy_image(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopyImageKHR>(device, context, queue, num_elements);
}

int test_copy_buffer(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopyBufferKHR>(device, context, queue, num_elements);
}

int test_copy_buffer_to_image(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopyBufferToImageKHR>(device, context, queue,
                                                num_elements);
}

int test_copy_image_to_buffer(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopyImageToBufferKHR>(device, context, queue,
                                                num_elements);
}

int test_copy_buffer_rect(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopyBufferRectKHR>(device, context, queue,
                                             num_elements);
}
