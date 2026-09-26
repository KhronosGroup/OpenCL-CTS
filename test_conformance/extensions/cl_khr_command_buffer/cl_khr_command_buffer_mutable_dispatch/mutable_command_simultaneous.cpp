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
#include <vector>
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handles
// - out-of-order queue with dependencies between command-buffer enqueues
// - out-of-order queue with simultaneous use
// - in-order queue with dependencies between command-buffer enqueues
// - in-order queue with simultaneous use
// - cross queue with dependencies between command-buffer enqueues
// - cross-queue with simultaneous use

namespace {

template <bool simultaneous_request, bool out_of_order_request>
struct SimultaneousMutableDispatchTest : public BasicMutableCommandBufferTest
{
    SimultaneousMutableDispatchTest(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          work_queue(nullptr), work_command_buffer(this), new_in_mem(nullptr),
          command(nullptr)
    {
        simultaneous_use_requested = simultaneous_request;
    }

    cl_int SetUpKernel() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // create additional kernel to properly prepare output buffer for test
        const char *kernel_str =
            R"(
          __kernel void mul(__global int* out, __global int* in, int mul_val)
          {
              size_t id = get_global_id(0);
              out[id] = in[id] * mul_val;
          })";

        error = create_single_kernel_helper_create_program(
            context, &program_mul, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error =
            clBuildProgram(program_mul, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel_mul = clCreateKernel(program_mul, "mul", &error);
        test_error(error, "Failed to create multiply kernel");

        new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(cl_int) * num_elements
                                         * buffer_size_multiplier,
                                     nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        new_in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(cl_int) * num_elements
                                        * buffer_size_multiplier,
                                    nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernelArgs failed");

        error = clSetKernelArg(kernel_mul, 0, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_mul, 1, sizeof(off_mem), &in_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_mul, 2, sizeof(cl_int), &pattern_pri);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicMutableCommandBufferTest::SetUp(elements);
        test_error(error, "BasicMutableCommandBufferTest::SetUp failed");

        if (out_of_order_request)
        {
            work_queue = clCreateCommandQueue(
                context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                &error);
            test_error(error, "Unable to create command queue to test with");
        }
        else
        {
            work_queue = queue;
        }

        cl_command_buffer_properties_khr prop = CL_COMMAND_BUFFER_MUTABLE_KHR;

        if (simultaneous_use_requested)
        {
            prop |= CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;
        }

        const cl_command_buffer_properties_khr props[] = {
            CL_COMMAND_BUFFER_FLAGS_KHR,
            prop,
            0,
        };

        work_command_buffer =
            clCreateCommandBufferKHR(1, &work_queue, props, &error);
        test_error(error, "clCreateCommandBufferKHR failed");
        return CL_SUCCESS;
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

        return (out_of_order_request && !out_of_order_support)
            || (simultaneous_use_requested && !simultaneous_use_support)
            || !mutable_support;
    }

    cl_int RecordCommandBuffer()
    {
        cl_int error = clCommandNDRangeKernelKHR(
            work_command_buffer, nullptr, nullptr, kernel_mul, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(work_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int RunSerializedPass(std::vector<cl_int> &first_enqueue_output,
                             std::vector<cl_int> &second_enqueue_output)
    {
        /* Serialize command-buffer enqueue, is a linear sequence of
         * commands, with dependencies enforced using an in-order queue
         * or cl_event dependencies.
         *
         * 1. Fill input buffer
         * 2. Enqueue command-buffer doing: `output = a * input;
         * 3. Read output buffer to host data so it can be verified later
         * -  Update command to new input buffer, new `a` val and use output
         *    buffer from previous invocation as new input buffer.
         * 4. Enqueue command-buffer again.
         * 5. Read new output buffer back to host data so it can be verified
         *    later
         *
         */
        clEventWrapper E[4];
        cl_int error = clEnqueueFillBuffer(
            work_queue, in_mem, &pattern_fill, sizeof(cl_int), 0, data_size(),
            0, nullptr, (out_of_order_request ? &E[0] : nullptr));
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, work_command_buffer, (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[0] : nullptr),
            (out_of_order_request ? &E[1] : nullptr));
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(work_queue, out_mem, CL_FALSE, 0,
                                    data_size(), first_enqueue_output.data(),
                                    (out_of_order_request ? 1 : 0),
                                    (out_of_order_request ? &E[1] : nullptr),
                                    (out_of_order_request ? &E[2] : nullptr));
        test_error(error, "clEnqueueReadBuffer failed");

        cl_mutable_dispatch_arg_khr arg_1{ 0, sizeof(new_out_mem),
                                           &new_out_mem };

        cl_mutable_dispatch_arg_khr arg_2{ 1, sizeof(cl_mem), &out_mem };
        cl_mutable_dispatch_arg_khr arg_3{ 2, sizeof(cl_int), &pattern_sec };

        cl_mutable_dispatch_arg_khr args[] = { arg_1, arg_2, arg_3 };
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            3 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            args /* arg_list */,
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
        const void* configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(work_command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, work_command_buffer, (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[2] : nullptr),
            (out_of_order_request ? &E[3] : nullptr));
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            work_queue, new_out_mem, CL_FALSE, 0, data_size(),
            second_enqueue_output.data(), (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[3] : nullptr), nullptr);
        test_error(error, "clEnqueueReadBuffer failed");
        return CL_SUCCESS;
    }

    cl_int RunSimultaneousPass(std::vector<cl_int> &first_enqueue_output,
                               std::vector<cl_int> &second_enqueue_output)
    {
        /* Simultaneous command-buffer pass enqueues a command-buffer twice
         * without dependencies between the enqueues, but an update so that
         * all the parameters are different to avoid race conditions in the
         * kernel execution. The asynchronous task graph looks like:
         *
         *  (Fill input A buffer)         (Fill input B buffer)
         *          |                               |
         *  (Enqueue command_buffer)      (Enqueue updated command_buffer)
         *          |                               |
         *  (Read output A buffer)        (Read output B buffer)
         */
        clEventWrapper E[4];
        cl_int error = clEnqueueFillBuffer(
            work_queue, in_mem, &pattern_fill, sizeof(cl_int), 0, data_size(),
            0, nullptr, (out_of_order_request ? &E[0] : nullptr));
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillBuffer(work_queue, new_in_mem, &pattern_fill_2,
                                    sizeof(cl_int), 0, data_size(), 0, nullptr,
                                    (out_of_order_request ? &E[1] : nullptr));
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, work_command_buffer, (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[0] : nullptr),
            (out_of_order_request ? &E[2] : nullptr));
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_arg_khr arg_1{ 0, sizeof(new_out_mem),
                                           &new_out_mem };
        cl_mutable_dispatch_arg_khr arg_2{ 1, sizeof(cl_mem), &new_in_mem };
        cl_mutable_dispatch_arg_khr arg_3{ 2, sizeof(cl_int), &pattern_sec };

        cl_mutable_dispatch_arg_khr args[] = { arg_1, arg_2, arg_3 };
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            3 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            args /* arg_list */,
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
        const void* configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(work_command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, work_command_buffer, (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[1] : nullptr),
            (out_of_order_request ? &E[3] : nullptr));
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            work_queue, out_mem, CL_FALSE, 0, data_size(),
            first_enqueue_output.data(), (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[2] : nullptr), nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clEnqueueReadBuffer(
            work_queue, new_out_mem, CL_FALSE, 0, data_size(),
            second_enqueue_output.data(), (out_of_order_request ? 1 : 0),
            (out_of_order_request ? &E[3] : nullptr), nullptr);
        test_error(error, "clEnqueueReadBuffer failed");
        return CL_SUCCESS;
    }

    cl_int VerifySerializedPass(std::vector<cl_int> &first_enqueue_output,
                                std::vector<cl_int> &second_enqueue_output)
    {
        const cl_int first_enqueue_ref = pattern_pri * pattern_fill;
        const cl_int second_enqueue_ref = pattern_sec * first_enqueue_ref;
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(first_enqueue_ref, first_enqueue_output[i],
                                     i);
            CHECK_VERIFICATION_ERROR(second_enqueue_ref,
                                     second_enqueue_output[i], i);
        }
        return CL_SUCCESS;
    }

    cl_int VerifySimultaneousPass(std::vector<cl_int> &first_enqueue_output,
                                  std::vector<cl_int> &second_enqueue_output)
    {
        const cl_int first_enqueue_ref = pattern_pri * pattern_fill;
        const cl_int second_enqueue_ref = pattern_sec * pattern_fill_2;
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(first_enqueue_ref, first_enqueue_output[i],
                                     i);
            CHECK_VERIFICATION_ERROR(second_enqueue_ref,
                                     second_enqueue_output[i], i);
        }
        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        std::vector<cl_int> first_enqueue_output(num_elements);
        std::vector<cl_int> second_enqueue_output(num_elements);

        if (simultaneous_use_requested)
        {
            error = RunSimultaneousPass(first_enqueue_output,
                                        second_enqueue_output);
            test_error(error, "RunSimultaneousPass failed");
        }
        else
        {
            error =
                RunSerializedPass(first_enqueue_output, second_enqueue_output);
            test_error(error, "RunSerializedPass failed");
        }

        error = clFinish(work_queue);
        test_error(error, "clFinish failed");

        // verify the result buffers
        if (simultaneous_use_requested)
        {
            error = VerifySimultaneousPass(first_enqueue_output,
                                           second_enqueue_output);
            test_error(error, "VerifySimultaneousPass failed");
        }
        else
        {
            error = VerifySerializedPass(first_enqueue_output,
                                         second_enqueue_output);
            test_error(error, "VerifySerializedPass failed");
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper work_queue;
    clCommandBufferWrapper work_command_buffer;

    clKernelWrapper kernel_mul;
    clProgramWrapper program_mul;

    clMemWrapper new_out_mem, new_in_mem;

    const cl_int pattern_pri = 42;
    const cl_int pattern_sec = 0xACDC;
    const cl_int pattern_fill = 0xA;
    const cl_int pattern_fill_2 = -3;

    cl_mutable_command_khr command;
};

template <bool simultaneous_use_request>
struct CrossQueueSimultaneousMutableDispatchTest
    : public BasicMutableCommandBufferTest
{
    CrossQueueSimultaneousMutableDispatchTest(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          queue_sec(nullptr), new_out_mem(nullptr), command(nullptr)
    {
        simultaneous_use_requested = simultaneous_use_request;
    }

    cl_int SetUpKernel() override
    {
        const char* kernel_str =
            R"(
          __kernel void fill(int pattern, __global int* out)
          {
              size_t id = get_global_id(0);
              out[id] = pattern;
          })";

        cl_int error = create_single_kernel_helper_create_program(
            context, &program, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "fill", &error);
        test_error(error, "Failed to create copy kernel");

        new_out_mem =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           sizeof(cl_int) * num_elements, nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        return CL_SUCCESS;
    }

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

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicMutableCommandBufferTest::SetUp(elements);
        test_error(error, "BasicMutableCommandBufferTest::SetUp failed");

        queue_sec = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "Unable to create command queue to test with");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;

        cl_mutable_dispatch_fields_khr mutable_capabilities = { 0 };

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        return (simultaneous_use_requested && !simultaneous_use_support)
            || !mutable_support;
    }

    cl_int Run() override
    {
        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // If we are testing not using simultaneous-use then we need to use
        // an event to serialize the execution order to the command-buffer
        // submission to each queue.
        clEventWrapper E;
        error = clEnqueueCommandBufferKHR(
            0, nullptr, command_buffer, 0, nullptr,
            (simultaneous_use_requested ? nullptr : &E));
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_arg_khr arg_0{ 0, sizeof(cl_int), &pattern_sec };
        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(new_out_mem),
                                           &new_out_mem };
        cl_mutable_dispatch_arg_khr args[] = { arg_0, arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            2 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            args /* arg_list */,
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
        const void* configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        // enqueue command buffer to non-default queue
        error = clEnqueueCommandBufferKHR(
            1, &queue_sec, command_buffer, (simultaneous_use_requested ? 0 : 1),
            (simultaneous_use_requested ? nullptr : &E), nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // read result of command buffer execution
        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        std::vector<cl_int> sec_output_data(num_elements);
        error =
            clEnqueueReadBuffer(queue_sec, new_out_mem, CL_TRUE, 0, data_size(),
                                sec_output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // verify the result
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
            CHECK_VERIFICATION_ERROR(pattern_sec, sec_output_data[i], i);
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue_sec;
    clMemWrapper new_out_mem;
    const cl_int pattern_pri = 42;
    const cl_int pattern_sec = 0xACDC;
    cl_mutable_command_khr command;
};

} // anonymous namespace

REGISTER_TEST(mutable_dispatch_out_of_order)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<false, true>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_simultaneous_out_of_order)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<true, true>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_in_order)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<false, false>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_simultaneous_in_order)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<true, false>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_cross_queue)
{
    return MakeAndRunTest<CrossQueueSimultaneousMutableDispatchTest<false>>(
        device, context, queue, num_elements);
}

REGISTER_TEST(mutable_dispatch_simultaneous_cross_queue)
{
    return MakeAndRunTest<CrossQueueSimultaneousMutableDispatchTest<true>>(
        device, context, queue, num_elements);
}
