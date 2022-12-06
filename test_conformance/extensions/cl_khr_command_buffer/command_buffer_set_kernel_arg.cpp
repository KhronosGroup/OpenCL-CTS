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
#include <fstream>
#include <stdio.h>

namespace {

////////////////////////////////////////////////////////////////////////////////

template <bool simul_use>
struct CommandBufferSetKernelArg : public BasicCommandBufferTest
{
    CommandBufferSetKernelArg(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          trigger_event(nullptr), wait_event(nullptr)
    {
        simultaneous_use_requested = simul_use;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        return (simultaneous_use_requested && !simultaneous_use_support)
            || BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;

        const char* kernel_str =
            R"(
            __kernel void fill(int in, __global int* out) {
                size_t id = get_global_id(0);
                out[id] = in;
            })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "fill", &error);
        test_error(error, "Failed to create print kernel");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size(),
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem_k2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size(),
                                    nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int in_arg = pattern_pri;
        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &in_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        // record command buffer with primary queue
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        if (simultaneous_use_support)
        {
            // enqueue simultaneous command-buffers with printf calls
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with printf calls
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

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // verify the result - we are interested in first kernel launch
        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }


        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous() { return CL_SUCCESS; }

    //--------------------------------------------------------------------------
    clEventWrapper trigger_event = nullptr;
    clEventWrapper wait_event = nullptr;

    const cl_int pattern_pri = 2;
    const cl_int pattern_sec = 3;

    clMemWrapper out_mem_k2;
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
