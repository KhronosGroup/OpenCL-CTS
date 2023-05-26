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
#include "imageHelpers.h"
#include <vector>
#include <iostream>
#include <random>
#include <cstring>
#include <algorithm>
#include <memory>
#include "mutable_command_basic.h"

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

    virtual cl_int SetUp(int elements) override
    {
        InfoMutableCommandBufferTest::SetUp(elements);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities != CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR;

        return !mutable_support || InfoMutableCommandBufferTest::Skip();
    }

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

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
            nullptr /* global_work_size */,
            &local_size /* local_work_size */
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
            sizeof(test_local_work_size), &test_local_work_size, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_local_work_size != local_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    size_t test_local_work_size = 0;
    const size_t local_size = 3;
    size_t size;
    cl_mutable_command_khr command = nullptr;
};

int test_mutable_dispatch_local_size(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableDispatchLocalSize>(device, context, queue,
                                                    num_elements);
}
