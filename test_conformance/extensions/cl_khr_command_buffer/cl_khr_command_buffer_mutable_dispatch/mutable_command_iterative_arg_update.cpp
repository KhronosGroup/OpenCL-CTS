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
// Test clUpdateMutableCommandsKHR() being called twice on the same command
// before an enqueue, but with different arguments. Verifies that the combined
// updates are made correctly.

struct IterativeArgUpdateDispatch : BasicMutableCommandBufferTest
{
    IterativeArgUpdateDispatch(cl_device_id device, cl_context context,
                               cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          command(nullptr)
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

    // setup kernel program
    cl_int SetUpKernel() override
    {
        const char *kernel_fill_str =
            R"(
            __kernel void fill(int pattern, __global int *dst)
            {
                size_t gid = get_global_id(0);
                dst[gid] = pattern;
            })";

        cl_int error = create_single_kernel_helper_create_program(
            context, &program, 1, &kernel_fill_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "fill", &error);
        test_error(error, "Failed to create fill kernel");

        return CL_SUCCESS;
    }

    // setup kernel arguments
    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size(),
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &pattern_pri);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
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

    // run command buffer with iterative update of mutable dispatch
    cl_int Run() override
    {
        // record command buffer with fill pattern kernel
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        const cl_int pattern = 0;
        error = clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // check the results of the initial execution
        if (!verify_result(out_mem, pattern_pri)) return TEST_FAIL;

        // new output buffer for command buffer kernel
        clMemWrapper new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                  data_size(), nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        // apply dispatch for mutable arguments
        cl_mutable_dispatch_arg_khr args = { 0, sizeof(cl_int), &pattern_sec };

        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            &args /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
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

        // update parameter of previous mutable dispatch by using the same
        // arguments list variable but this time modify other kernel argument
        args.arg_index = 1;
        args.arg_size = sizeof(new_out_mem);
        args.arg_value = &new_out_mem;

        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueFillBuffer(queue, new_out_mem, &pattern_pri,
                                    sizeof(cl_int), 0, data_size(), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        // repeat execution of modified command buffer
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Check the results of the modified execution
        if (!verify_result(new_out_mem, pattern_sec)) return TEST_FAIL;

        return TEST_PASS;
    }

    // mutable dispatch test attributes
    cl_mutable_command_khr command;

    const cl_int pattern_pri = 0xD0DA;
    const cl_int pattern_sec = 0xCAFE;
};

}

int test_mutable_command_iterative_arg_update(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements)
{
    return MakeAndRunTest<IterativeArgUpdateDispatch>(device, context, queue,
                                                      num_elements);
}
