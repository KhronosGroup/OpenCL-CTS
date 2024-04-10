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
                                             0, 0, nullptr, nullptr);

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
};

int test_negative_command_buffer_copy_image_queue_not_null(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CommandBufferCopyImageQueueNotNull>(
        device, context, queue, num_elements);
}
