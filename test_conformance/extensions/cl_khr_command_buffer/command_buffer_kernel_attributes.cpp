//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Tests for cl_khr_command_buffer while enqueueing a kernel with a
// reqd_work_group_size with a NULL local_work_size.

struct KernelAttributesReqGroupSizeTest : public BasicCommandBufferTest
{
    KernelAttributesReqGroupSizeTest(cl_device_id device, cl_context context,
                                     cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), dst(nullptr),
          clGetKernelSuggestedLocalWorkSizeKHR(nullptr)
    {}

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        if (is_extension_available(device, "cl_khr_suggested_local_work_size"))
        {
            cl_platform_id platform = nullptr;
            error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                    sizeof(platform), &platform, NULL);
            test_error(error, "clGetDeviceInfo for platform failed");

            clGetKernelSuggestedLocalWorkSizeKHR =
                (clGetKernelSuggestedLocalWorkSizeKHR_fn)
                    clGetExtensionFunctionAddressForPlatform(
                        platform, "clGetKernelSuggestedLocalWorkSizeKHR");
            test_assert_error(clGetKernelSuggestedLocalWorkSizeKHR != nullptr,
                              "Couldn't get function pointer for "
                              "clGetKernelSuggestedLocalWorkSizeKHR");
        }

        dst = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * sizeof(cl_int),
                             nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_uint device_max_dim = 0;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof(device_max_dim), &device_max_dim, nullptr);
        test_error(
            error,
            "clGetDeviceInfo for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS failed");
        test_assert_error(
            device_max_dim >= 3,
            "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS must be at least 3!");

        std::vector<size_t> device_max_work_item_sizes(device_max_dim);
        error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                sizeof(size_t) * device_max_dim,
                                device_max_work_item_sizes.data(), nullptr);

        size_t device_max_work_group_size = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                sizeof(device_max_work_group_size),
                                &device_max_work_group_size, nullptr);
        test_error(error,
                   "clGetDeviceInfo for CL_DEVICE_MAX_WORK_GROUP_SIZE failed");


        std::vector<std::pair<std::string, std::uint16_t>> attribs = {
            { "__attribute__((reqd_work_group_size(2,1,1)))", 1 },
            { "__attribute__((reqd_work_group_size(2,3,1)))", 2 },
            { "__attribute__((reqd_work_group_size(2,3,4)))", 3 }
        };

        const std::string body_str = R"(
                    __kernel void wg_size(__global int* dst)
                    {
                        if (get_global_id(0) == 0 &&
                            get_global_id(1) == 0 &&
                            get_global_id(2) == 0) {
                            dst[0] = get_local_size(0);
                            dst[1] = get_local_size(1);
                            dst[2] = get_local_size(2);
                        }
                    }
                )";


        for (auto& attrib : attribs)
        {
            const std::string source_str = attrib.first + body_str;
            const char* source = source_str.c_str();

            clProgramWrapper program;
            clKernelWrapper kernel;
            error = create_single_kernel_helper(context, &program, &kernel, 1,
                                                &source, "wg_size");
            test_error(error, "Unable to create test kernel");

            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dst);
            test_error(error, "clSetKernelArg failed");

            for (cl_uint work_dim = 1; work_dim <= attrib.second; work_dim++)
            {
                const cl_int expected[3] = { 2, work_dim >= 2 ? 3 : 1,
                                             work_dim >= 3 ? 4 : 1 };
                const size_t test_work_group_size =
                    expected[0] * expected[1] * expected[2];
                if ((size_t)expected[0] > device_max_work_item_sizes[0]
                    || (size_t)expected[1] > device_max_work_item_sizes[1]
                    || (size_t)expected[2] > device_max_work_item_sizes[2]
                    || test_work_group_size > device_max_work_group_size)
                {
                    log_info(
                        "Skipping test for work_dim = %u: required work group "
                        "size (%i, %i, %i) (total %zu) exceeds device max "
                        "work group size (%zu, %zu, %zu) (total %zu)\n",
                        work_dim, expected[0], expected[1], expected[2],
                        test_work_group_size, device_max_work_item_sizes[0],
                        device_max_work_item_sizes[1],
                        device_max_work_item_sizes[2],
                        device_max_work_group_size);
                    continue;
                }

                const cl_int zero = 0;
                error = clCommandFillBufferKHR(
                    command_buffer, nullptr, nullptr, dst, &zero, sizeof(zero),
                    0, sizeof(expected), 0, nullptr, nullptr, nullptr);
                test_error(error, "clCommandFillBufferKHR failed");

                const size_t global_work_size[3] = { 2 * 32, 3 * 32, 4 * 32 };
                error = clCommandNDRangeKernelKHR(
                    command_buffer, nullptr, nullptr, kernel, work_dim, nullptr,
                    global_work_size, nullptr, 0, nullptr, nullptr, nullptr);
                test_error(error, "clCommandNDRangeKernelKHR failed");

                error = clFinalizeCommandBufferKHR(command_buffer);
                test_error(error, "clFinalizeCommandBufferKHR failed");

                error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                                  nullptr, nullptr);
                test_error(error, "clEnqueueCommandBufferKHR failed");

                cl_int results[3] = { -1, -1, -1 };
                error =
                    clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(results),
                                        results, 0, nullptr, nullptr);
                test_error(error, "clEnqueueReadBuffer failed");

                // Verify the result
                if (results[0] != expected[0] || results[1] != expected[1]
                    || results[2] != expected[2])
                {
                    log_error(
                        "Executed local size mismatch with work_dim = %u: "
                        "Expected (%d,%d,%d) got (%d,%d,%d)\n",
                        work_dim, expected[0], expected[1], expected[2],
                        results[0], results[1], results[2]);
                    return TEST_FAIL;
                }

                if (clGetKernelSuggestedLocalWorkSizeKHR != nullptr)
                {
                    size_t suggested[3] = { 1, 1, 1 };
                    error = clGetKernelSuggestedLocalWorkSizeKHR(
                        queue, kernel, work_dim, nullptr, global_work_size,
                        suggested);
                    test_error(error,
                               "clGetKernelSuggestedLocalWorkSizeKHR failed");

                    if ((cl_int)suggested[0] != expected[0]
                        || (cl_int)suggested[1] != expected[1]
                        || (cl_int)suggested[2] != expected[2])
                    {
                        log_error(
                            "Suggested local size mismatch with work_dim = "
                            "%u: Expected (%d,%d,%d) got (%d,%d,%d)\n",
                            work_dim, expected[0], expected[1], expected[2],
                            (cl_int)suggested[0], (cl_int)suggested[1],
                            (cl_int)suggested[2]);
                        return TEST_FAIL;
                    }
                }

                // create new command buffer
                command_buffer =
                    clCreateCommandBufferKHR(1, &queue, nullptr, &error);
                test_error(error, "clCreateCommandBufferKHR failed");
            }
        }

        return CL_SUCCESS;
    }

    clMemWrapper dst;
    clGetKernelSuggestedLocalWorkSizeKHR_fn
        clGetKernelSuggestedLocalWorkSizeKHR;
};

} // anonymous namespace

REGISTER_TEST(command_null_required_work_group_size)
{
    return MakeAndRunTest<KernelAttributesReqGroupSizeTest>(
        device, context, queue, num_elements);
}
