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
#include "svm_command_basic.h"
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
        cl_int error = clCommandFillImageKHR(
            command_buffer, nullptr, nullptr, src_image, fill_color_1, origin,
            region, 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillImageKHR failed");

        error = clCommandCopyImageKHR(command_buffer, nullptr, nullptr,
                                      src_image, dst_image, origin, origin,
                                      region, 0, 0, nullptr, nullptr);

        test_error(error, "clCommandCopyImageKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size);
        error =
            clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, 0, 0,
                               output_data_1.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */

        error = clEnqueueFillImage(queue, src_image, fill_color_2, origin,
                                   region, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillImageKHR failed");

        error = clEnqueueFillImage(queue, dst_image, fill_color_2, origin,
                                   region, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillImageKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size);
        error =
            clEnqueueReadImage(queue, dst_image, CL_TRUE, origin, region, 0, 0,
                               output_data_2.data(), 0, nullptr, nullptr);

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_2[i], i);
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
    const cl_uint pattern_1 = 0x05;
    const cl_uint fill_color_1[4] = { pattern_1, pattern_1, pattern_1,
                                      pattern_1 };
    const cl_uint pattern_2 = 0x1;
    const cl_uint fill_color_2[4] = { pattern_2, pattern_2, pattern_2,
                                      pattern_2 };
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
            command_buffer, nullptr, nullptr, in_mem, &pattern_1,
            sizeof(cl_char), 0, data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferKHR(command_buffer, nullptr, nullptr, in_mem,
                                       out_mem, 0, 0, data_size(), 0, nullptr,
                                       nullptr, nullptr);
        test_error(error, "clCommandCopyBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size());
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data_1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_2, sizeof(cl_char),
                                    0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBufferKHR failed");

        error = clEnqueueFillBuffer(queue, out_mem, &pattern_2, sizeof(cl_char),
                                    0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size());
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data_2.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_2[i], i);
        }

        return CL_SUCCESS;
    }

    const cl_char pattern_1 = 0x14;
    const cl_char pattern_2 = 0x28;
};

struct CopySVMBufferKHR : public BasicSVMCommandBufferTest
{
    using BasicSVMCommandBufferTest::BasicSVMCommandBufferTest;

    cl_int Run() override
    {

        cl_int error = clCommandSVMMemFillKHR(
            command_buffer, nullptr, nullptr, svm_in_mem(), &pattern_1,
            sizeof(cl_char), data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandSVMMemFillKHR failed");

        error = clCommandSVMMemcpyKHR(command_buffer, nullptr, nullptr,
                                      svm_out_mem(), svm_in_mem(), data_size(),
                                      0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandSVMMemcpyKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size());
        error =
            clEnqueueSVMMemcpy(queue, CL_TRUE, output_data_1.data(),
                               svm_out_mem(), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */
        error = clEnqueueSVMMemFill(queue, svm_in_mem(), &pattern_2,
                                    sizeof(cl_char), data_size(), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed");

        error = clEnqueueSVMMemFill(queue, svm_out_mem(), &pattern_2,
                                    sizeof(cl_char), data_size(), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size());

        error =
            clEnqueueSVMMemcpy(queue, CL_TRUE, output_data_2.data(),
                               svm_out_mem(), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_2[i], i);
        }

        return CL_SUCCESS;
    }

    const cl_char pattern_1 = 0x14;
    const cl_char pattern_2 = 0x28;
};

struct CopyBufferToImageKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, nullptr, buffer, &pattern_1,
            sizeof(cl_char), 0, data_size, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferToImageKHR(command_buffer, nullptr, nullptr,
                                              buffer, image, 0, origin, region,
                                              0, 0, nullptr, nullptr);
        test_error(error, "clCommandCopyBufferToImageKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size);

        error = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                   output_data_1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */

        error = clEnqueueFillBuffer(queue, buffer, &pattern_2, sizeof(cl_char),
                                    0, data_size, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillImage(queue, image, &fill_color_2, origin, region,
                                   0, nullptr, nullptr);
        test_error(error, "clEnqueueFillImage failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size);

        error = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                   output_data_2.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_2[i], i);
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
    const cl_char pattern_1 = 0x11;
    const cl_char pattern_2 = 0x22;

    const cl_uint fill_color_2[4] = { static_cast<cl_uint>(pattern_2),
                                      static_cast<cl_uint>(pattern_2),
                                      static_cast<cl_uint>(pattern_2),
                                      static_cast<cl_uint>(pattern_2) };

    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

    clMemWrapper buffer;
    clMemWrapper image;
};

struct CopyImageToBufferKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillImageKHR(
            command_buffer, nullptr, nullptr, image, fill_color_1, origin,
            region, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillImageKHR failed");

        error = clCommandCopyImageToBufferKHR(command_buffer, nullptr, nullptr,
                                              image, buffer, origin, region, 0,
                                              0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandCopyImageToBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size,
                                    output_data_1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(static_cast<cl_char>(pattern_1),
                                     output_data_1[i], i);
        }

        error = clEnqueueFillImage(queue, image, fill_color_2, origin, region,
                                   0, nullptr, nullptr);
        test_error(error, "clEnqueueFillImage failed");

        error = clEnqueueFillBuffer(queue, buffer, &pattern_2, sizeof(cl_char),
                                    0, data_size, 0, nullptr, nullptr);

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size,
                                    output_data_2.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(static_cast<cl_char>(pattern_1),
                                     output_data_2[i], i);
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
    const cl_uint pattern_1 = 0x12;
    const cl_uint fill_color_1[4] = { pattern_1, pattern_1, pattern_1,
                                      pattern_1 };
    const cl_uint pattern_2 = 0x24;
    const cl_uint fill_color_2[4] = { pattern_2, pattern_2, pattern_2,
                                      pattern_2 };
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
            command_buffer, nullptr, nullptr, in_mem, &pattern_1,
            sizeof(cl_char), 0, data_size, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandCopyBufferRectKHR(
            command_buffer, nullptr, nullptr, in_mem, out_mem, origin, origin,
            region, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandCopyBufferRectKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_1(data_size);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size,
                                    output_data_1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_1[i], i);
        }

        /* Check second enqueue of command buffer */

        error = clEnqueueFillBuffer(queue, in_mem, &pattern_2, sizeof(cl_char),
                                    0, data_size, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillBuffer(queue, out_mem, &pattern_2, sizeof(cl_char),
                                    0, data_size, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data_2(data_size);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size,
                                    output_data_2.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_1, output_data_2[i], i);
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
    const cl_char pattern_1 = 0x13;
    const cl_char pattern_2 = 0x26;

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

int test_copy_svm_buffer(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CopySVMBufferKHR>(device, context, queue,
                                            num_elements);
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
