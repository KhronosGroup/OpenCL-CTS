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

#include "testHarness.h"
#include "mutable_command_basic.h"

#include <CL/cl_ext.h>

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Tests related to ensuring the state of the updated command-buffer is expected
// and the effects of operations on it don't have side effects on other objects.
//
// - Tests the updates applied to a command-buffer persist over all subsequent
// enqueues.
// - Tests interaction of `clSetKernelArg` with mutable-dispatch extension.

struct MutableDispatchUpdateStateTest : public BasicMutableCommandBufferTest
{
    MutableDispatchUpdateStateTest(cl_device_id device, cl_context context,
                                   cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          buffer(nullptr), command(nullptr)
    {}

    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;

        cl_mutable_dispatch_fields_khr mutable_capabilities;
        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;
        return !mutable_support;
    }

    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                num_elements * sizeof(cl_int), nullptr, &error);
        test_error(error, "clCreateBuffer error");

        // Zero initialize buffer
        const cl_int zero_pattern = 0;
        error = clEnqueueFillBuffer(
            queue, buffer, &zero_pattern, sizeof(cl_int), 0,
            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        error = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        test_error(error, "Unable to set kernel argument 0");

        return CL_SUCCESS;
    }

    cl_int SetUpKernel() override
    {
        const char *add_kernel =
            R"(
            __kernel void add_kernel(__global int *data, int value)
            {
                size_t tid = get_global_id(0);
                data[tid] += value;
            })";

        cl_int error = create_single_kernel_helper(
            context, &program, &kernel, 1, &add_kernel, "add_kernel");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    bool verify_result(cl_int ref)
    {
        std::vector<cl_int> data(num_elements);
        cl_int error =
            clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size(),
                                data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (data[i] != ref)
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted %d\n",
                          i, data[i], ref);
                return false;
            }
        }
        return true;
    }

    clMemWrapper buffer;
    cl_mutable_command_khr command;
};

struct MutableDispatchUpdatesPersistTest : public MutableDispatchUpdateStateTest
{
    MutableDispatchUpdatesPersistTest(cl_device_id device, cl_context context,
                                      cl_command_queue queue)
        : MutableDispatchUpdateStateTest(device, context, queue)
    {}

    cl_int Run() override
    {
        const cl_int original_val = 42;
        cl_int error =
            clSetKernelArg(kernel, 1, sizeof(original_val), &original_val);
        test_error(error, "Unable to set kernel argument 1");

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Modify the command buffer before executing
        const cl_int new_command_val = 5;
        cl_mutable_dispatch_arg_khr arg{ 1, sizeof(new_command_val),
                                         &new_command_val };
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            &arg /* arg_list */,
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

        const unsigned iterations = 5;
        for (unsigned i = 0; i < iterations; i++)
        {
            error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                              nullptr, nullptr);
            test_error(error, "clEnqueueCommandBufferKHR failed");

            error = clFinish(queue);
            test_error(error, "clFinish failed");
        }

        // Check the results execution sequence is the clEnqueueNDRangeKernel
        // value + the updated command-buffer value, not using the original
        // command value in the operation.
        constexpr cl_int ref = iterations * new_command_val;
        return verify_result(ref) ? TEST_PASS : TEST_FAIL;
    }
};

struct MutableDispatchSetKernelArgTest : public MutableDispatchUpdateStateTest
{
    MutableDispatchSetKernelArgTest(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
        : MutableDispatchUpdateStateTest(device, context, queue)
    {}

    cl_int Run() override
    {
        const cl_int original_val = 42;
        cl_int error =
            clSetKernelArg(kernel, 1, sizeof(original_val), &original_val);
        test_error(error, "Unable to set kernel argument 1");

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Set new kernel argument for later clEnqueueNDRangeKernel
        const cl_int new_eager_val = 10;
        error =
            clSetKernelArg(kernel, 1, sizeof(new_eager_val), &new_eager_val);
        test_error(error, "Unable to set kernel argument 1");

        // Modify the command buffer before executing
        const cl_int new_command_val = 5;
        cl_mutable_dispatch_arg_khr arg{ 1, sizeof(new_command_val),
                                         &new_command_val };
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            &arg /* arg_list */,
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

        // Eager kernel enqueue, followed by command-buffer enqueue
        error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &num_elements,
                                       nullptr, 0, nullptr, nullptr);
        test_error(error, "clEnqueueNDRangeKernel failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results execution sequence is the clEnqueueNDRangeKernel
        // value + the updated command-buffer value, not using the original
        // command value in the operation.
        constexpr cl_int ref = new_eager_val + new_command_val;
        return verify_result(ref) ? TEST_PASS : TEST_FAIL;
    }
};
}

REGISTER_TEST(mutable_dispatch_updates_persist)
{
    return MakeAndRunTest<MutableDispatchUpdatesPersistTest>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_set_kernel_arg)
{
    return MakeAndRunTest<MutableDispatchSetKernelArgTest>(device, context,
                                                           queue, num_elements);
}
