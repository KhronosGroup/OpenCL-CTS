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

#include <extensionHelpers.h>
#include "mutable_command_basic.h"

#include <array>
#include <cstring>

#include <CL/cl_ext.h>

// mutable dispatch tests setting `work_dim` to the original 3D value
// behaves as expected.

struct MutableDispatchWorkDim : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    MutableDispatchWorkDim(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int SetUp(int elements) override
    {
        result_data.resize(update_elements);
        return InfoMutableCommandBufferTest::SetUp(elements);
    }

    bool Skip() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && (mutable_capabilities & CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR);

        return !mutable_support || InfoMutableCommandBufferTest::Skip();
    }

    bool Verify(cl_mem buffer, cl_uint gid_elements)
    {
        std::memset(result_data.data(), 0, alloc_size);
        cl_int error =
            clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, alloc_size,
                                result_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < gid_elements; i++)
        {
            if (result_data[i] != gid_elements)
            {
                log_error("Data failed to verify at index %zu. "
                          "Expected %u, result was %u\n",
                          i, gid_elements, result_data[i]);
                return false;
            }
        }
        return true;
    }

    cl_int Run() override
    {
        const char *global_size_kernel =
            R"(
                __kernel void three_dim(__global uint *dst0,
                                        __global uint *dst1,
                                        __global uint *dst2)
            {
                size_t gid0 = get_global_id(0);
                dst0[gid0] = get_global_size(0);

                size_t gid1 = get_global_id(1);
                dst1[gid1] = get_global_size(1);

                size_t gid2 = get_global_id(2);
                dst2[gid2] = get_global_size(2);
            })";

        cl_int error = create_single_kernel_helper(
            context, &program, &kernel, 1, &global_size_kernel, "three_dim");
        test_error(error, "Creating kernel failed");

        // Create a buffer for each of the three dimensions to write the
        // global size into.
        clMemWrapper stream1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              alloc_size, nullptr, &error);
        test_error(error, "Creating test array failed");

        clMemWrapper stream2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              alloc_size, nullptr, &error);
        test_error(error, "Creating test array failed");

        clMemWrapper stream3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              alloc_size, nullptr, &error);
        test_error(error, "Creating test array failed");

        // Set the arguments
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream1);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &stream2);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &stream3);
        test_error(error, "Unable to set indexed kernel arguments");

        // Command-buffer contains a single kernel
        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, work_dim, nullptr,
            global_size_3D.data(), nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Enqueue command-buffer and wait on completion
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        // Verify results before any update
        if (!Verify(stream1, global_size_3D[0]))
        {
            return TEST_FAIL;
        }
        if (!Verify(stream2, global_size_3D[1]))
        {
            return TEST_FAIL;
        }
        if (!Verify(stream3, global_size_3D[2]))
        {
            return TEST_FAIL;
        }

        // Update command with a mutable config where we use a different 3D
        // global size, but hardcode `work_dim` to 3 (the original value).
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            work_dim /* work_dim */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            update_global_size_3D.data() /* global_work_size */,
            nullptr /* local_work_size */
        };

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        // Enqueue updated command-buffer
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Verify update is reflected in buffer output.
        if (!Verify(stream1, update_global_size_3D[0]))
        {
            return TEST_FAIL;
        }
        if (!Verify(stream2, update_global_size_3D[1]))
        {
            return TEST_FAIL;
        }
        if (!Verify(stream3, update_global_size_3D[2]))
        {
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    static const cl_uint work_dim = 3;
    // 3D global size of kernel command when created
    static const size_t original_elements = 2;
    static constexpr std::array<size_t, work_dim> global_size_3D = {
        original_elements, original_elements, original_elements
    };
    // 3D global size to update kernel command to.
    static const size_t update_elements = 4;
    static constexpr std::array<size_t, work_dim> update_global_size_3D = {
        update_elements, update_elements, update_elements
    };
    // Size in bytes of each of the 3 cl_mem buffers
    static const size_t alloc_size = update_elements * sizeof(cl_uint);

    cl_mutable_command_khr command = nullptr;
    std::vector<cl_uint> result_data;
};

REGISTER_TEST(mutable_dispatch_work_dim)
{
    return MakeAndRunTest<MutableDispatchWorkDim>(device, context, queue,
                                                  num_elements);
}
