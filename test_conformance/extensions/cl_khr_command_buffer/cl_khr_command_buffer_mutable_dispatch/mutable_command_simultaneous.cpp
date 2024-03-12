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
// mutable dispatch tests which handle following cases:
// - out-of-order queue use
// - out-of-order queue with simultaneous use
// - in-order queue with simultaneous use
// - cross-queue with simultaneous use

namespace {

template <bool simultaneous_request, bool out_of_order_request>
struct SimultaneousMutableDispatchTest : public BasicMutableCommandBufferTest
{
    SimultaneousMutableDispatchTest(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          work_queue(nullptr), work_command_buffer(this), user_event(nullptr),
          wait_pass_event(nullptr), command(nullptr)
    {
        simultaneous_use_requested = simultaneous_request;
        if (simultaneous_request) buffer_size_multiplier = 2;
    }

    cl_int SetUpKernel() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernel();
        test_error(error, "BasicCommandBufferTest::SetUpKernel failed");

        // create additional kernel to properly prepare output buffer for test
        const char* kernel_str =
            R"(
          __kernel void fill(int pattern, __global int* out, __global int*
        offset)
          {
              size_t id = get_global_id(0);
              size_t ind = offset[0] + id ;
              out[ind] = pattern;
          })";

        error = create_single_kernel_helper_create_program(
            context, &program_fill, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error =
            clBuildProgram(program_fill, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel_fill = clCreateKernel(program_fill, "fill", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        cl_int error = BasicCommandBufferTest::SetUpKernelArgs();
        test_error(error, "BasicCommandBufferTest::SetUpKernelArgs failed");

        error = clSetKernelArg(kernel_fill, 0, sizeof(cl_int),
                               &overwritten_pattern);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 2, sizeof(off_mem), &off_mem);
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

            cl_command_buffer_properties_khr prop =
                CL_COMMAND_BUFFER_MUTABLE_KHR;
            if (simultaneous_use_support)
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
        }
        else
        {
            work_queue = queue;
            work_command_buffer = command_buffer;
        }

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

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        if (simultaneous_use_support)
        {
            // enqueue simultaneous command-buffers with out-of-order calls
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with out-of-order calls
            error = RunSingle();
            test_error(error, "RunSingle failed");
        }

        return CL_SUCCESS;
    }

    cl_int RecordCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        const cl_int pattern = pattern_pri;
        cl_int error = clCommandFillBufferKHR(
            work_command_buffer, nullptr, in_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandFillBufferKHR(work_command_buffer, nullptr, out_mem,
                                       &overwritten_pattern, sizeof(cl_int), 0,
                                       data_size(), 0, nullptr, &sync_points[1],
                                       nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            work_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(work_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int RunSingle()
    {
        cl_int error;

        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, work_command_buffer, 0,
                                          nullptr, &single_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error =
            clEnqueueReadBuffer(work_queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 1, &single_event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        clMemWrapper new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                  sizeof(cl_int) * num_elements
                                                      * buffer_size_multiplier,
                                                  nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(new_out_mem),
                                           &new_out_mem };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            1 /* num_args */,
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
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(work_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, work_command_buffer, 0,
                                          nullptr, &single_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(work_queue, new_out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 1,
                                    &single_event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    cl_int RecordSimultaneousCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        // for both simultaneous passes this call will fill entire in_mem buffer
        cl_int error = clCommandFillBufferKHR(
            work_command_buffer, nullptr, in_mem, &pattern_pri, sizeof(cl_int),
            0, data_size() * buffer_size_multiplier, 0, nullptr,
            &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        // to avoid overwriting the entire result buffer instead of filling
        // only relevant part this additional kernel was introduced

        error = clCommandNDRangeKernelKHR(
            work_command_buffer, nullptr, nullptr, kernel_fill, 1, nullptr,
            &num_elements, nullptr, 0, nullptr, &sync_points[1], &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clCommandNDRangeKernelKHR(
            work_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(work_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    struct SimulPassData
    {
        cl_int offset;
        std::vector<cl_int> output_buffer;
        // 0:user event, 1:offset-buffer fill event, 2:kernel done event
        clEventWrapper wait_events[3];
    };

    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = CL_SUCCESS;
        if (!user_event)
        {
            user_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        pd.wait_events[0] = user_event;

        // filling offset buffer must wait for previous pass completeness
        error = clEnqueueFillBuffer(
            work_queue, off_mem, &pd.offset, sizeof(cl_int), 0, sizeof(cl_int),
            (wait_pass_event != nullptr ? 1 : 0),
            (wait_pass_event != nullptr ? &wait_pass_event : nullptr),
            &pd.wait_events[1]);
        test_error(error, "clEnqueueFillBuffer failed");

        // command buffer execution must wait for two wait-events
        error =
            clEnqueueCommandBufferKHR(0, nullptr, work_command_buffer, 2,
                                      &pd.wait_events[0], &pd.wait_events[2]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(work_queue, out_mem, CL_FALSE,
                                    pd.offset * sizeof(cl_int), data_size(),
                                    pd.output_buffer.data(), 1,
                                    &pd.wait_events[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        clMemWrapper new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                  sizeof(cl_int) * num_elements
                                                      * buffer_size_multiplier,
                                                  nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(new_out_mem),
                                           &new_out_mem };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            1 /* num_args */,
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
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(work_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        // command buffer execution must wait for two wait-events
        error =
            clEnqueueCommandBufferKHR(0, nullptr, work_command_buffer, 2,
                                      &pd.wait_events[0], &pd.wait_events[2]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(work_queue, new_out_mem, CL_FALSE,
                                    pd.offset * sizeof(cl_int), data_size(),
                                    pd.output_buffer.data(), 1,
                                    &pd.wait_events[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    cl_int RunSimultaneous()
    {
        cl_int error = RecordSimultaneousCommandBuffer();
        test_error(error, "RecordSimultaneousCommandBuffer failed");

        cl_int offset = static_cast<cl_int>(num_elements);

        std::vector<SimulPassData> simul_passes = {
            { 0, std::vector<cl_int>(num_elements) },
            { offset, std::vector<cl_int>(num_elements) }
        };

        for (auto&& pass : simul_passes)
        {
            error = EnqueueSimultaneousPass(pass);
            test_error(error, "EnqueueSimultaneousPass failed");

            wait_pass_event = pass.wait_events[2];
        }

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(work_queue);
        test_error(error, "clFinish failed");

        // verify the result buffers
        for (auto&& pass : simul_passes)
        {
            auto& res_data = pass.output_buffer;
            for (size_t i = 0; i < num_elements; i++)
            {
                CHECK_VERIFICATION_ERROR(pattern_pri, res_data[i], i);
            }
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper work_queue;
    clCommandBufferWrapper work_command_buffer;

    clEventWrapper user_event;
    clEventWrapper single_event;
    clEventWrapper wait_pass_event;

    clKernelWrapper kernel_fill;
    clProgramWrapper program_fill;

    const size_t test_global_work_size = 3 * sizeof(cl_int);
    const cl_int pattern_pri = 42;

    const cl_int overwritten_pattern = 0xACDC;
    cl_mutable_command_khr command;
};

struct CrossQueueSimultaneousMutableDispatchTest
    : public BasicMutableCommandBufferTest
{
    CrossQueueSimultaneousMutableDispatchTest(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          queue_sec(nullptr), command(nullptr)
    {
        simultaneous_use_requested = true;
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

        return !simultaneous_use_support || !mutable_support;
    }

    cl_int Run() override
    {
        // record command buffer
        cl_int pattern = 0;
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, out_mem, &pattern, sizeof(cl_int), 0,
            data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // enqueue command buffer to default queue
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // update mutable parameters
        clMemWrapper new_out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                  data_size(), nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_mutable_dispatch_arg_khr arg_0{ 0, sizeof(cl_int), &pattern_sec };
        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(new_out_mem),
                                           &new_out_mem };
        cl_mutable_dispatch_arg_khr args[] = { arg_0, arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
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
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        // enqueue command buffer to non-default queue
        error = clEnqueueCommandBufferKHR(1, &queue_sec, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue_sec);
        test_error(error, "clFinish failed");

        // read result of command buffer execution
        std::vector<cl_int> output_data(num_elements);
        error =
            clEnqueueReadBuffer(queue_sec, new_out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // verify the result
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_sec, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper queue_sec;
    const cl_int pattern_pri = 42;
    const cl_int pattern_sec = 0xACDC;
    cl_mutable_command_khr command;
};

} // anonymous namespace

int test_mutable_dispatch_out_of_order(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<false, true>>(
        device, context, queue, num_elements);
}

int test_mutable_dispatch_simultaneous_out_of_order(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<true, true>>(
        device, context, queue, num_elements);
}

int test_mutable_dispatch_simultaneous_in_order(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<SimultaneousMutableDispatchTest<true, false>>(
        device, context, queue, num_elements);
}

int test_mutable_dispatch_simultaneous_cross_queue(cl_device_id device,
                                                   cl_context context,
                                                   cl_command_queue queue,
                                                   int num_elements)
{
    return MakeAndRunTest<CrossQueueSimultaneousMutableDispatchTest>(
        device, context, queue, num_elements);
}
