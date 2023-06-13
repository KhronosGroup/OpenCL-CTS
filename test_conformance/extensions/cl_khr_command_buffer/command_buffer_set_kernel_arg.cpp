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

#include "basic_command_buffer.h"
#include "procs.h"

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// clSetKernelArg tests for cl_khr_command_buffer which handles below cases:
// -test interactions of clSetKernelArg with command-buffers
// -test interactions of clSetKernelArg on a command-buffer pending execution

template <bool simul_use>
struct CommandBufferSetKernelArg : public BasicCommandBufferTest
{
    CommandBufferSetKernelArg(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), trigger_event(nullptr)
    {
        simultaneous_use_requested = simul_use;
        if (simul_use) buffer_size_multiplier = 2;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;

        const char* kernel_str =
            R"(
            __kernel void copy(int in, __global int* out, __global int* offset)
            {
                size_t id = get_global_id(0);
                size_t ind = offset[0] + id;
                out[ind] = in;
            })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "copy", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 num_elements * buffer_size_multiplier
                                     * sizeof(cl_int),
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        // create secondary output buffer to test kernel args substitution
        out_mem_k2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    num_elements * buffer_size_multiplier
                                        * sizeof(cl_int),
                                    nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int offset = 0;
        off_mem =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int), &offset, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int in_arg = pattern_pri;
        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &in_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 2, sizeof(off_mem), &off_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        if (simultaneous_use_requested)
        {
            // enqueue simultaneous command-buffers with clSetKernelArg calls
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with  clSetKernelArg calls
            error = RunSingle();
            test_error(error, "RunSingle failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
    {
        cl_int error = CL_SUCCESS;

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        // changing kernel args at this point should have no effect,
        // test will verify if clSetKernelArg didn't affect the first command
        cl_int in_arg = pattern_sec;
        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &in_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem_k2), &out_mem_k2);
        test_error(error, "clSetKernelArg failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle()
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_int> output_data(num_elements);

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        const cl_int pattern_base = 0;
        error =
            clEnqueueFillBuffer(queue, out_mem, &pattern_base, sizeof(cl_int),
                                0, data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        const cl_int pattern_base_k2 = 1;
        error = clEnqueueFillBuffer(queue, out_mem_k2, &pattern_base_k2,
                                    sizeof(cl_int), 0, data_size(), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        // verify the result - result buffer must contain initial pattern
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    struct SimulPassData
    {
        cl_int pattern;
        cl_int offset;
        std::vector<cl_int> output_buffer;
    };

    //--------------------------------------------------------------------------
    cl_int RecordSimultaneousCommandBuffer() const
    {
        cl_int error = CL_SUCCESS;

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = clEnqueueFillBuffer(
            queue, out_mem, &pd.pattern, sizeof(cl_int),
            pd.offset * sizeof(cl_int), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillBuffer(queue, off_mem, &pd.offset, sizeof(cl_int),
                                    0, sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        if (!trigger_event)
        {
            trigger_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &trigger_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            queue, out_mem, CL_FALSE, pd.offset * sizeof(cl_int), data_size(),
            pd.output_buffer.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer with primary queue
        error = RecordSimultaneousCommandBuffer();
        test_error(error, "RecordSimultaneousCommandBuffer failed");

        std::vector<SimulPassData> simul_passes = {
            { 0, 0, std::vector<cl_int>(num_elements) }
        };

        error = EnqueueSimultaneousPass(simul_passes.front());
        test_error(error, "EnqueueSimultaneousPass 1 failed");

        // changing kernel args at this point should have no effect,
        // test will verify if clSetKernelArg didn't affect command-buffer
        cl_int in_arg = pattern_sec;
        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &in_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem_k2), &out_mem_k2);
        test_error(error, "clSetKernelArg failed");

        if (simultaneous_use_support)
        {
            cl_int offset = static_cast<cl_int>(num_elements);
            simul_passes.push_back(
                { 1, offset, std::vector<cl_int>(num_elements) });

            error = EnqueueSimultaneousPass(simul_passes.back());
            test_error(error, "EnqueueSimultaneousPass 2 failed");
        }

        error = clSetUserEventStatus(trigger_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result buffer
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

    //--------------------------------------------------------------------------
    clEventWrapper trigger_event = nullptr;

    const cl_int pattern_pri = 2;
    const cl_int pattern_sec = 3;

    clMemWrapper out_mem_k2 = nullptr;
};

} // anonymous namespace

int test_basic_set_kernel_arg(cl_device_id device, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandBufferSetKernelArg<false>>(
        device, context, queue, num_elements);
}

int test_pending_set_kernel_arg(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandBufferSetKernelArg<true>>(device, context,
                                                           queue, num_elements);
}
