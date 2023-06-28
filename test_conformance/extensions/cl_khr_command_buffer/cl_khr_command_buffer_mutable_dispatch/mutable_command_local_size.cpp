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

#include <extensionHelpers.h>
#include "typeWrappers.h"
#include "procs.h"
#include "testHarness.h"
#include "mutable_command_basic.h"
#include <vector>

#include <CL/cl.h>
#include <CL/cl_ext.h>

////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases:
//
// CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR

struct MutableDispatchLocalSize : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    MutableDispatchLocalSize(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    bool Skip() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR;

        return !mutable_support || InfoMutableCommandBufferTest::Skip();
    }

    cl_int Run() override
    {
        const char *local_size_kernel =
            R"(
                __kernel void sample_test(__global int *dst)
            {
                size_t tid = get_global_id(0);
                dst[tid] = get_local_size(0);
            })";

        cl_int error = create_single_kernel_helper(
            context, &program, &kernel, 1, &local_size_kernel, "sample_test");
        test_error(error, "Creating kernel failed");

        clMemWrapper stream;
        stream = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                nullptr, &error);
        test_error(error, "Creating test array failed");

        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, &local_work_size, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            &update_global_size /* global_work_size */,
            &update_local_size /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
            sizeof(info_local_size), &info_local_size, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (info_local_size != update_local_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        std::vector<cl_int> resultData;
        resultData.resize(num_elements);

        error = clEnqueueReadBuffer(queue, stream, CL_TRUE, 0, sizeToAllocate,
                                    resultData.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
            if (i < update_global_size && update_local_size != resultData[i])
            {
                log_error("Data failed to verify: update_local_size != "
                          "resultData[%d]=%d\n",
                          i, resultData[i]);
                return TEST_FAIL;
            }
            else if (i >= update_global_size
                     && local_work_size != resultData[i])
            {
                log_error("Data failed to verify: update_local_size != "
                          "resultData[%d]=%d\n",
                          i, resultData[i]);
                return TEST_FAIL;
            }

        return CL_SUCCESS;
    }

    size_t info_local_size = 0;
    const size_t global_work_size = 16;
    const size_t local_work_size = 8;
    const size_t update_global_size = 8;
    const size_t update_local_size = 4;
    const size_t sizeToAllocate = 64;
    const size_t num_elements = sizeToAllocate / sizeof(cl_int);

    cl_mutable_command_khr command = nullptr;
};

int test_mutable_dispatch_local_size(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableDispatchLocalSize>(device, context, queue,
                                                    num_elements);
}
