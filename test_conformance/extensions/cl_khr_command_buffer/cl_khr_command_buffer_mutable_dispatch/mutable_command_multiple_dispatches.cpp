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

#include <extensionHelpers.h>
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// command buffer with multiple command handles dispatch test

struct MultipleCommandsDispatch : BasicMutableCommandBufferTest
{
    MultipleCommandsDispatch(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          command_pri(nullptr), command_sec(nullptr)
    {
        simultaneous_use_requested = false;
    }

    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;
        cl_mutable_dispatch_fields_khr mutable_capabilities;
        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        // require mutable arguments capabillity
        return !mutable_support;
    }

    // setup default and fill kernels program
    cl_int SetUpKernel() override
    {
        // default command buffer kernel
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // fill command buffer kernel
        const char *kernel_fill_str =
            R"(
            __kernel void fill(int pattern, __global int *dst)
            {
                size_t gid = get_global_id(0);
                dst[gid] = pattern;
            })";

        error = create_single_kernel_helper_create_program(
            context, &program_fill, 1, &kernel_fill_str);
        test_error(error, "Failed to create program with source");

        error =
            clBuildProgram(program_fill, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel_fill = clCreateKernel(program_fill, "fill", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }

    // setup kernel arguments for both default and fill kernels
    cl_int SetUpKernelArgs() override
    {
        // arguments for default kernel
        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernelArgs failed");

        // fill kernel applies pattern for input data of default kernel
        error = clSetKernelArg(kernel_fill, 0, sizeof(cl_int), &pattern_pri);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 1, sizeof(in_mem), &in_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    // Check the results of command buffer execution
    bool verify_result(const cl_mem &buffer, const cl_int pattern)
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_int> data(num_elements);
        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size(),
                                    data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (data[i] != pattern)
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted %d\n",
                          i, data[i], pattern);
                return false;
            }
        }

        return true;
    }

    // run command buffer with multiple command dispatches test
    cl_int Run() override
    {
        // record fill kernel and collect first mutable command handle
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel_fill, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, nullptr, &command_pri);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        // record default kernel and collect second mutable command handle
        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command_sec);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // check the results of the initial execution
        if (!verify_result(out_mem, pattern_pri)) return TEST_FAIL;

        // new output buffer for default kernel
        clMemWrapper new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                  data_size(), nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        // apply dispatch for mutable arguments of both fill and default kernels
        cl_mutable_dispatch_arg_khr arg_pri{ 0, sizeof(cl_int), &pattern_sec };
        cl_mutable_dispatch_arg_khr args_pri[] = { arg_pri };

        cl_mutable_dispatch_arg_khr arg_sec{ 1, sizeof(new_out_mem),
                                             &new_out_mem };
        cl_mutable_dispatch_arg_khr args_sec[] = { arg_sec };

        // modify two mutable parameters, each one with separate handle
        cl_mutable_dispatch_config_khr dispatch_config[] = {
            { CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR, nullptr,
              command_pri, 1, 0, 0, 0, args_pri, nullptr, nullptr, nullptr,
              nullptr, nullptr },
            { CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR, nullptr,
              command_sec, 1, 0, 0, 0, args_sec, nullptr, nullptr, nullptr,
              nullptr, nullptr },
        };

        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 2,
            dispatch_config
        };

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        // repeat execution of modified command buffer
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // check the results of the modified execution
        if (!verify_result(new_out_mem, pattern_sec)) return TEST_FAIL;

        return TEST_PASS;
    }

    // mutable dispatch test attributes
    cl_mutable_command_khr command_pri;
    cl_mutable_command_khr command_sec;

    clKernelWrapper kernel_fill;
    clProgramWrapper program_fill;

    const cl_int pattern_pri = 0xACDC;
    const cl_int pattern_sec = 0xDEAD;
};

}

int test_mutable_command_multiple_dispatches(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    return MakeAndRunTest<MultipleCommandsDispatch>(device, context, queue,
                                                    num_elements);
}
