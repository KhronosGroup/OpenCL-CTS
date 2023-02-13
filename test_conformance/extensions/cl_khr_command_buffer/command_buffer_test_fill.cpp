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
// Command-buffer fill tests which handles below cases:
//
// -fill image
// -fill buffer

struct FillImageKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clCommandFillImageKHR(command_buffer, nullptr, image, fill_color,
                                  origin, region, 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillImageKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size);
        error = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                   output_data.data(), 0, nullptr, nullptr);

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
    const cl_uint pattern = 0x10;
    const cl_uint fill_color[4] = { pattern, pattern, pattern, pattern };
    const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

    clMemWrapper image;
};

struct FillBufferKHR : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, in_mem, &pattern, sizeof(cl_char), 0,
            data_size(), 0, nullptr, nullptr, nullptr);

        test_error(error, "clCommandFillBufferKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_char> output_data(data_size());
        error = clEnqueueReadBuffer(queue, in_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < data_size(); i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    const char pattern = 0x15;
};

};

int test_fill_buffer(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<FillBufferKHR>(device, context, queue, num_elements);
}

int test_fill_image(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<FillImageKHR>(device, context, queue, num_elements);
}
