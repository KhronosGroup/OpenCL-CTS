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
// Command-buffer profiling test cases:
// -all commands are recorded to a single command-queue
// -profiling a command-buffer with simultaneous use

template <bool simultaneous_request>
struct CommandBufferProfiling : public BasicCommandBufferTest
{
    CommandBufferProfiling(cl_device_id device, cl_context context,
                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), wait_event(nullptr)
    {
        simultaneous_use_requested = simultaneous_request;
        if (simultaneous_request) buffer_size_multiplier = 2;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (queue == nullptr) return true;

        return (simultaneous_use_requested && !simultaneous_use_support)
            || BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        cl_int error = CL_SUCCESS;
        cl_queue_properties_khr device_props = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                                sizeof(device_props), &device_props, nullptr);
        test_error(error,
                   "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

        queue = nullptr;
        if (device_props & CL_QUEUE_PROFILING_ENABLE)
        {
            queue = clCreateCommandQueue(context, device,
                                         CL_QUEUE_PROFILING_ENABLE, &error);
        }
        else
        {
            log_info(
                "Queue property CL_QUEUE_PROFILING_ENABLE not supported \n");
            return CL_SUCCESS;
        }

        return BasicCommandBufferTest::SetUp(elements);
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        if (simultaneous_use_requested)
        {
            // enqueue simultaneous command-buffers with profiling command queue
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with profiling command queue
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

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
#define ADD_PROF_PARAM(prop)                                                   \
    {                                                                          \
        prop, #prop, 0                                                         \
    }
    struct ProfilingParam
    {
        cl_profiling_info param;
        std::string name;
        cl_ulong value;
    };

    //--------------------------------------------------------------------------
    cl_int VerifyResult(const clEventWrapper & event)
    {
        cl_int error = CL_SUCCESS;
        cl_int status;
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(status), &status, NULL);
        test_error(error, "clGetEventInfo() failed");

        if (status != CL_SUCCESS)
            test_fail("Kernel execution status %d! (%s:%d)\n", status, __FILE__,
                      __LINE__);

        std::vector<ProfilingParam> prof_params = {
            ADD_PROF_PARAM(CL_PROFILING_COMMAND_QUEUED),
            ADD_PROF_PARAM(CL_PROFILING_COMMAND_SUBMIT),
            ADD_PROF_PARAM(CL_PROFILING_COMMAND_START),
            ADD_PROF_PARAM(CL_PROFILING_COMMAND_END),
            ADD_PROF_PARAM(CL_PROFILING_COMMAND_COMPLETE)
        };

        // gather profiling timestamps
        for (auto&& p : prof_params)
        {
            error = clGetEventProfilingInfo(event, p.param,
                                            sizeof(p.value), &p.value, NULL);
            test_error(error, "clGetEventProfilingInfo() failed");
        }

        // verify the results by comparing timestamps
        for (int i = 1; i < prof_params.size(); i++)
        {
            if (prof_params[i - 1].value > prof_params[i].value)
            {
                log_error(
                    "Profiling %s should be smaller than or equal to %s for "
                    "kernels that use the on-device queue",
                    prof_params[i - 1].name, prof_params[i].name);
                return TEST_FAIL;
            }
        }

        log_info("Profiling info for command-buffer kernel succeeded.\n");
        return TEST_PASS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle()
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_int> output_data(num_elements);

        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        clEventWrapper query_event;
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &query_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        error = VerifyResult(query_event);
        test_error(error, "VerifyResult failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    struct SimulPassData
    {
        cl_int offset;
        std::vector<cl_int> output_buffer;
        clEventWrapper query_event;
    };

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = clEnqueueFillBuffer(
            queue, out_mem, &pattern, sizeof(cl_int),
            pd.offset * sizeof(cl_int), data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueFillBuffer(queue, off_mem, &pd.offset, sizeof(cl_int),
                                    0, sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        if (!wait_event)
        {
            wait_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &wait_event, &pd.query_event);
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
        cl_int offset = static_cast<cl_int>(num_elements);

        std::vector<SimulPassData> simul_passes = {
            { 0, std::vector<cl_int>(num_elements) },
            { offset, std::vector<cl_int>(num_elements) }
        };

        for (auto&& pass : simul_passes)
        {
            error = EnqueueSimultaneousPass(pass);
            test_error(error, "EnqueueSimultaneousPass failed");
        }

        error = clSetUserEventStatus(wait_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        for (auto&& pass : simul_passes)
        {
          error = VerifyResult(pass.query_event);
          test_error(error, "VerifyResult failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    clEventWrapper wait_event;

    const cl_int pattern = 0xA;
};

} // anonymous namespace

int test_basic_profiling(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandBufferProfiling<false>>(device, context, queue,
                                                         num_elements);
}

int test_simultaneous_profiling(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandBufferProfiling<true>>(device, context, queue,
                                                        num_elements);
}
