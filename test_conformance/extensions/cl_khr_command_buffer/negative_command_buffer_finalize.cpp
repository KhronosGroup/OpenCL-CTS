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
#include "basic_command_buffer.h"
#include "procs.h"


//--------------------------------------------------------------------------
namespace {

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct FinalizeCommandBufferInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clFinalizeCommandBufferKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_OPERATION if command_buffer is not in the Recording state.
struct FinalizeCommandBufferNotRecordingState : public BasicCommandBufferTest
{
    FinalizeCommandBufferNotRecordingState(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), user_event(nullptr)
    {}

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        auto verify_state = [&](const cl_command_buffer_state_khr &expected) {
            cl_command_buffer_state_khr state = ~cl_command_buffer_state_khr(0);

            cl_int error = clGetCommandBufferInfoKHR(
                command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state),
                &state, nullptr);
            test_error_ret(error, "clGetCommandBufferInfoKHR failed",
                           TEST_FAIL);

            test_assert_error(
                state == expected,
                "Unexpected result of CL_COMMAND_BUFFER_STATE_KHR query!");

            return TEST_PASS;
        };

        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");
        error = verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        test_error(error, "State is not Executable");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clFinalizeCommandBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        error = EnqueueCommandBuffer();
        test_error(error, "EnqueueCommandBuffer failed");
        error = verify_state(CL_COMMAND_BUFFER_STATE_PENDING_KHR);
        test_error(error, "State is not Pending");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_failure_error_ret(error, CL_INVALID_OPERATION,
                               "clFinalizeCommandBufferKHR should return "
                               "CL_INVALID_OPERATION",
                               TEST_FAIL);

        clSetUserEventStatus(user_event, CL_COMPLETE);
        clFinish(queue);

        return CL_SUCCESS;
    }

    cl_int RecordCommandBuffer()
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int EnqueueCommandBuffer()
    {
        cl_int pattern = 0xE;
        cl_int error =
            clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &user_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        return CL_SUCCESS;
    }
    clEventWrapper user_event;
};
};

int test_negative_finalize_command_buffer_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<FinalizeCommandBufferInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_finalize_command_buffer_not_recording_state(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<FinalizeCommandBufferNotRecordingState>(
        device, context, queue, num_elements);
}
