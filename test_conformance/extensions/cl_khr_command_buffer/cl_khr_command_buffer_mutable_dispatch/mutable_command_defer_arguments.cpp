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

namespace {

////////////////////////////////////////////////////////////////////////////////
// Mutable dispatch test which handles the case where all the arguments of a
// kernel aren't set when a kernel is initially added to a mutable
// command-buffer, but deferred until an update is made to the command to set
// them before command-buffer enqueue.
struct MutableDispatchDeferArguments : public BasicMutableCommandBufferTest
{
    MutableDispatchDeferArguments(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
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

        // require mutable arguments capability
        return !mutable_support;
    }

    cl_int SetUpKernel() override
    {
        // Create kernel
        const char *defer_args_kernel =
            R"(
            __kernel void defer_args_test(__constant int *src, __global int *dst)
            {
                size_t tid = get_global_id(0);
                dst[tid] = src[tid];
            })";

        cl_int error =
            create_single_kernel_helper(context, &program, &kernel, 1,
                                        &defer_args_kernel, "defer_args_test");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        // Create and initialize buffers
        MTdataHolder d(gRandomSeed);

        src_data.resize(num_elements);
        for (size_t i = 0; i < num_elements; i++)
            src_data[i] = (cl_int)genrand_int32(d);

        cl_int error = CL_SUCCESS;
        in_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                num_elements * sizeof(cl_int), src_data.data(),
                                &error);
        test_error(error, "Creating src buffer");

        out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating initial dst buffer failed");

        // Only set a single kernel argument, leaving argument at index 1 unset
        error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
        test_error(error, "Unable to set src kernel arguments");

        return CL_SUCCESS;
    }

    bool verify_state(cl_command_buffer_state_khr expected)
    {
        cl_command_buffer_state_khr state = ~cl_command_buffer_state_khr(0);
        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state), &state,
            nullptr);
        if (error != CL_SUCCESS)
        {
            log_error("clGetCommandBufferInfoKHR failed: %d", error);
            return false;
        }

        if (state != expected)
        {
            log_error("Unexpected result of CL_COMMAND_BUFFER_STATE_KHR query. "
                      "Expected %u, but was %u\n",
                      expected, state);
            return false;
        }
        return true;
    }

    bool verify_result(const cl_mem &buffer)
    {
        std::vector<cl_int> data(num_elements);
        cl_int error =
            clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size(),
                                data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (data[i] != src_data[i])
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted %d\n",
                          i, data[i], src_data[i]);
                return false;
            }
        }
        return true;
    }

    cl_int Run() override
    {
        // Create command while the kernel still has the second argument unset.
        // Passing 'CL_MUTABLE_DISPATCH_ARGUMENTS_KHR' as a property means this
        // shouldn't be an error.
        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        // Finalizing the command buffer shouldn't be an error, but result in
        // the command-buffer entering the CL_COMMAND_BUFFER_STATE_FINALIZED
        // state.
        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        if (!verify_state(CL_COMMAND_BUFFER_STATE_FINALIZED_KHR))
        {
            return TEST_FAIL;
        }

        // Check that trying to enqueue the command-buffer in this state is an
        // error, as it needs to be in the executable state.
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clEnqueueCommandBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        // Update the kernel command to set the missing argument.
        cl_mutable_dispatch_arg_khr arg{ 1, sizeof(out_mem), &out_mem };
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

        // Now that all the arguments have been set, verify the
        // command-buffer has entered the executable state.
        if (!verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR))
        {
            return TEST_FAIL;
        }

        // Execute command-buffer and verify results are expected
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");
        if (!verify_result(out_mem)) return TEST_FAIL;

        return TEST_PASS;
    }

    cl_mutable_command_khr command;
    std::vector<cl_int> src_data;
};

} // anonymous namespace

REGISTER_TEST(mutable_dispatch_defer_arguments)
{
    return MakeAndRunTest<MutableDispatchDeferArguments>(device, context, queue,
                                                         num_elements);
}
