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

//--------------------------------------------------------------------------
enum class CombufInfoTestMode
{
    CITM_QUEUES = 0,
    CITM_REF_COUNT,
    CITM_STATE,
    CITM_PROP_ARRAY,
    CITM_CONTEXT,
};

namespace {

////////////////////////////////////////////////////////////////////////////////
// clGetCommandBufferInfoKHR tests for cl_khr_command_buffer which handles below
// cases:
// -test case for CL_COMMAND_BUFFER_NUM_QUEUES_KHR &
//  CL_COMMAND_BUFFER_QUEUES_KHR queries
// -test case for CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR query
// -test case for CL_COMMAND_BUFFER_STATE_KHR query
// -test case for CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR query
// -test case for CL_COMMAND_BUFFER_CONTEXT_KHR query

template <CombufInfoTestMode test_mode>
struct CommandBufferGetCommandBufferInfo : public BasicCommandBufferTest
{
    CommandBufferGetCommandBufferInfo(cl_device_id device, cl_context context,
                                      cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        switch (test_mode)
        {
            case CombufInfoTestMode::CITM_QUEUES:
                error = RunQueuesInfoTest();
                test_error(error, "RunQueuesInfoTest failed");
                break;
            case CombufInfoTestMode::CITM_REF_COUNT:
                error = RunRefCountInfoTest();
                test_error(error, "RunRefCountInfoTest failed");
                break;
            case CombufInfoTestMode::CITM_STATE:
                error = RunStateInfoTest();
                test_error(error, "RunStateInfoTest failed");
                break;
            case CombufInfoTestMode::CITM_PROP_ARRAY:
                error = RunPropArrayInfoTest();
                test_error(error, "RunPropArrayInfoTest failed");
                break;
            case CombufInfoTestMode::CITM_CONTEXT:
                error = RunContextInfoTest();
                test_error(error, "RunContextInfoTest failed");
                break;
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
    cl_int RunQueuesInfoTest()
    {
        cl_int error = TEST_PASS;

        // record command buffers
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        // vector containter added due to potential future growth, at the moment
        // spec of cl_khr_command_buffer says command-buffer accepts only 1
        // queue
        std::vector<cl_command_queue> expect_queue_list = { queue };
        cl_uint num_queues = 0;
        size_t ret_value_size = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_NUM_QUEUES_KHR, sizeof(cl_uint),
            &num_queues, &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        test_assert_error(
            ret_value_size == sizeof(cl_int),
            "Unexpected result of CL_COMMAND_BUFFER_NUM_QUEUES_KHR query!");

        test_assert_error(num_queues == expect_queue_list.size(),
                          "Unexpected queue list size!");

        std::vector<cl_command_queue> queue_list(num_queues);
        size_t expect_size = queue_list.size() * sizeof(cl_command_queue);
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_QUEUES_KHR, expect_size,
            &queue_list.front(), &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        test_assert_error(
            ret_value_size == expect_size,
            "Unexpected result of CL_COMMAND_BUFFER_NUM_QUEUES_KHR query!");

        // We can not check if this is the right queue because this is an opaque
        // object, test against NULL.
        for (size_t i = 0; i < queue_list.size(); i++)
        {
            test_assert_error(
                queue_list[i] == queue,
                "clGetCommandBufferInfoKHR return values not as expected\n");
        }
        return TEST_PASS;
    }

    //--------------------------------------------------------------------------
    cl_int RunRefCountInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        // collect initial reference count
        cl_uint init_ref_count = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &init_ref_count, nullptr);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        // increase reference count through clRetainCommandBufferKHR calls
        const cl_int min_retain_count = 2;
        const cl_int max_retain_count = 6;
        cl_int retain_count = std::max(
            min_retain_count, min_retain_count + rand() % max_retain_count);

        for (int i = 0; i < retain_count; i++)
        {
            error = clRetainCommandBufferKHR(command_buffer);
            test_error(error, "clRetainCommandBufferKHR failed");
        }

        // verify new reference count value
        cl_uint new_ref_count = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &new_ref_count, nullptr);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        test_assert_error(new_ref_count == (retain_count + init_ref_count),
                          "Unexpected result of "
                          "CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR query!");

        // decrease reference count through clReleaseCommandBufferKHR calls
        for (int i = 0; i < retain_count; i++)
        {
            error = clReleaseCommandBufferKHR(command_buffer);
            test_error(error, "clReleaseCommandBufferKHR failed");
        }

        // verify new reference count value
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &new_ref_count, nullptr);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        test_assert_error(new_ref_count == init_ref_count,
                          "Unexpected result of "
                          "CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR query!");

        return TEST_PASS;
    }

    //--------------------------------------------------------------------------
    cl_int RunStateInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // lambda to verify given state
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

        // verify recording state
        error = verify_state(CL_COMMAND_BUFFER_STATE_RECORDING_KHR);
        test_error(error, "verify_state failed");

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        // verify executable state
        error = verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        test_error(error, "verify_state failed");

        error = clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        clEventWrapper trigger_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        clEventWrapper execute_event;
        // enqueued command buffer blocked on user event
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &trigger_event, &execute_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // verify pending state
        error = verify_state(CL_COMMAND_BUFFER_STATE_PENDING_KHR);

        // execute command buffer
        cl_int signal_error = clSetUserEventStatus(trigger_event, CL_COMPLETE);

        test_error(error, "verify_state failed");

        test_error(signal_error, "clSetUserEventStatus failed");

        error = clWaitForEvents(1, &execute_event);
        test_error(error, "Unable to wait for execute event");

        // verify executable state
        error = verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        test_error(error, "verify_state failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunPropArrayInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        size_t ret_value_size = 0;
        std::vector<cl_command_buffer_properties_khr> combuf_props;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR, 0, nullptr,
            &ret_value_size);
        test_error_ret(error, "clGetCommandBufferInfoKHR failed", TEST_FAIL);

        // command buffer created without sumultaneous use ? 0 size possible
        if (!simultaneous_use_support && ret_value_size == 0) return TEST_PASS;

        // ... otherwise 0 size prop array is not an acceptable value
        test_assert_error(ret_value_size != 0,
                          "Unexpected result of "
                          "CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR query!");

        cl_uint num_ret_props =
            ret_value_size / sizeof(cl_command_buffer_properties_khr);
        test_assert_error(num_ret_props != 0,
                          "Unexpected result of "
                          "CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR query!");

        combuf_props.resize(num_ret_props);
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
            num_ret_props * sizeof(cl_command_buffer_properties_khr),
            combuf_props.data(), nullptr);
        test_error_ret(error, "clGetCommandBufferInfoKHR failed", TEST_FAIL);

        if (simultaneous_use_support)
        {
            // in simultaneous use case at least 3 elements in array expected
            test_assert_error(num_ret_props >= 3,
                              "Unexpected result of "
                              "CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR query!");

            if (combuf_props[0] == CL_COMMAND_BUFFER_FLAGS_KHR
                && combuf_props[1] == CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR
                && combuf_props.back() == 0)
                return TEST_PASS;
        }
        else
        {
            if (combuf_props.back() == 0) return TEST_PASS;
        }

        return TEST_FAIL;
    }

    cl_int RunContextInfoTest()
    {
        cl_int error = TEST_PASS;

        // record command buffers
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        size_t ret_value_size = 0;
        error = clGetCommandBufferInfoKHR(command_buffer,
                                          CL_COMMAND_BUFFER_CONTEXT_KHR, 0,
                                          nullptr, &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        test_assert_error(
            ret_value_size == sizeof(cl_context),
            "Unexpected result of CL_COMMAND_BUFFER_CONTEXT_KHR query!");

        cl_context ret_context = nullptr;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_CONTEXT_KHR, sizeof(cl_context),
            &ret_context, nullptr);
        test_error(error, "clGetCommandBufferInfoKHR failed");
        test_assert_error(
            ret_context != nullptr,
            "Unexpected result of CL_COMMAND_BUFFER_CONTEXT_KHR query!");

        cl_context expected_context = nullptr;
        error =
            clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context),
                                  &expected_context, nullptr);
        test_error(error, "clGetCommandQueueInfo failed");

        test_assert_error(
            ret_context == expected_context,
            "Unexpected result of CL_COMMAND_BUFFER_CONTEXT_KHR query!");

        return TEST_PASS;
    }

    const cl_int pattern = 0xE;
};

} // anonymous namespace


int test_info_queues(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferGetCommandBufferInfo<CombufInfoTestMode::CITM_QUEUES>>(
        device, context, queue, num_elements);
}

int test_info_ref_count(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferGetCommandBufferInfo<CombufInfoTestMode::CITM_REF_COUNT>>(
        device, context, queue, num_elements);
}

int test_info_state(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferGetCommandBufferInfo<CombufInfoTestMode::CITM_STATE>>(
        device, context, queue, num_elements);
}

int test_info_prop_array(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferGetCommandBufferInfo<CombufInfoTestMode::CITM_PROP_ARRAY>>(
        device, context, queue, num_elements);
}

int test_info_context(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<
        CommandBufferGetCommandBufferInfo<CombufInfoTestMode::CITM_CONTEXT>>(
        device, context, queue, num_elements);
}
