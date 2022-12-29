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
#include <array>

//--------------------------------------------------------------------------
enum class CombufInfoTestMode
{
    CITM_QUEUES = 0,
    CITM_REF_COUNT,
    CITM_STATE,
    CITM_PROP_ARRAY,
};

namespace {

////////////////////////////////////////////////////////////////////////////////

template <CombufInfoTestMode test_mode>
struct CommandBufferGetCommandBufferInfo : public BasicCommandBufferTest
{
    CommandBufferGetCommandBufferInfo(cl_device_id device, cl_context context,
                                      cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), queue_sec(nullptr),
          command_buffer_sec(nullptr)
    {}

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        if (test_mode == CombufInfoTestMode::CITM_QUEUES)
        {
            // create secondary command queue and command buffer
            queue_sec = clCreateCommandQueue(context, device, 0, &error);
            test_error(error, "clCreateCommandQueue failed");

            cl_command_queue queue_list[] = { queue, queue_sec };
            int testv = sizeof(queue_list) / sizeof(cl_command_queue);
            command_buffer_sec =
                clCreateCommandBufferKHR(testv, queue_list, nullptr, &error);
            test_error(error, "clCreateCommandBufferKHR failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (test_mode == CombufInfoTestMode::CITM_PROP_ARRAY
            && !simultaneous_use_support)
            return true;

        return BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        switch (test_mode)
        {
            case CombufInfoTestMode::CITM_QUEUES:
                error = RunQueuesInfoTest();
                test_error(error, "RunCombufWaitForCombuf failed");
                break;
            case CombufInfoTestMode::CITM_REF_COUNT:
                error = RunRefCountInfoTest();
                test_error(error, "RunCombufWaitForSecCombuf failed");
                break;
            case CombufInfoTestMode::CITM_STATE:
                error = RunStateInfoTest();
                test_error(error, "RunReturnEventCallback failed");
                break;
            case CombufInfoTestMode::CITM_PROP_ARRAY:
                error = RunPropArrayInfoTest();
                test_error(error, "RunWaitForEvent failed");
                break;
        }


        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer(clCommandBufferWrapper &combuf)
    {
        cl_int error = CL_SUCCESS;

        error = clCommandNDRangeKernelKHR(combuf, nullptr, nullptr, kernel, 1,
                                          nullptr, &num_elements, nullptr, 0,
                                          nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(combuf);
        test_error(error, "clFinalizeCommandBufferKHR failed");
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunQueuesInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // record command buffers
        error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        error = RecordCommandBuffer(command_buffer_sec);
        test_error(error, "RecordCommandBuffer failed");

        auto test_combuf_queues = [&](std::vector<cl_command_queue> &queue_list,
                                      clCommandBufferWrapper &combuf) {
            cl_uint num_queues = 0;
            size_t ret_value_size = 0;
            cl_int error = clGetCommandBufferInfoKHR(
                combuf, CL_COMMAND_BUFFER_NUM_QUEUES_KHR, sizeof(cl_uint),
                &num_queues, &ret_value_size);
            test_error_ret(error, "clGetCommandBufferInfoKHR failed",
                           TEST_FAIL);

            if (ret_value_size > sizeof(cl_int)
                || num_queues != queue_list.size())
            {
                log_error("clGetCommandBufferInfoKHR return values not as "
                          "expected\n");
                return TEST_FAIL;
            }

            std::vector<cl_command_queue> ql(num_queues);
            size_t expect_size = ql.size() * sizeof(cl_command_queue);
            error = clGetCommandBufferInfoKHR(
                combuf, CL_COMMAND_BUFFER_QUEUES_KHR, expect_size, &ql.front(),
                &ret_value_size);
            test_error_ret(error, "clGetCommandBufferInfoKHR failed",
                           TEST_FAIL);

            if (ret_value_size > expect_size)
            {
                log_error("clGetCommandBufferInfoKHR return values not as "
                          "expected\n");
                return TEST_FAIL;
            }

            for (int i = 0; i < ql.size(); i++)
            {
                if (ql[i] != queue_list[i])
                {
                    log_error("clGetCommandBufferInfoKHR return values not as "
                              "expected\n");
                    return TEST_FAIL;
                }
            }
            return TEST_PASS;
        };

        {
            std::vector<cl_command_queue> queue_list = { queue };
            error = test_combuf_queues(queue_list, command_buffer);
            test_error(error, "test_combuf_queues failed");
        }

        {
            std::vector<cl_command_queue> queue_list = { queue, queue_sec };
            error = test_combuf_queues(queue_list, command_buffer_sec);
            test_error(error, "test_combuf_queues failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunRefCountInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        cl_uint init_ref_count = 0;
        size_t ret_value_size = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &init_ref_count, &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        const cl_int min_retain_count = 2;
        const cl_int max_retain_count = 6;
        cl_int retain_count = std::max(
            min_retain_count, min_retain_count + rand() % max_retain_count);

        for (int i = 0; i < retain_count; i++)
        {
            error = clRetainCommandBufferKHR(command_buffer);
            test_error(error, "clRetainCommandBufferKHR failed");
        }

        cl_uint new_ref_count = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &new_ref_count, &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        if (new_ref_count != (retain_count + init_ref_count))
        {
            log_error("clGetCommandBufferInfoKHR return values not as "
                      "expected\n");
            return TEST_FAIL;
        }

        for (int i = 0; i < retain_count; i++)
        {
            error = clReleaseCommandBufferKHR(command_buffer);
            test_error(error, "clReleaseCommandBufferKHR failed");
        }

        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(cl_uint), &new_ref_count, &ret_value_size);
        test_error(error, "clGetCommandBufferInfoKHR failed");

        if (new_ref_count != init_ref_count)
        {
            log_error("clGetCommandBufferInfoKHR return values not as "
                      "expected\n");
            return TEST_FAIL;
        }

        return TEST_PASS;
    }

    //--------------------------------------------------------------------------
    cl_int RunStateInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // lambda to verify given state
        auto verify_state = [&](const cl_command_buffer_state_khr &expected) {
            cl_command_buffer_state_khr state =
                CL_COMMAND_BUFFER_STATE_INVALID_KHR;
            size_t ret_value_size = 0;

            cl_int error = clGetCommandBufferInfoKHR(
                command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state),
                &state, &ret_value_size);
            test_error_ret(error, "clGetCommandBufferInfoKHR failed",
                           TEST_FAIL);

            if (state != expected)
            {
                log_error("clGetCommandBufferInfoKHR return values not as "
                          "expected\n");
                return TEST_FAIL;
            }
            return TEST_PASS;
        };

        // verify recording state
        error = verify_state(CL_COMMAND_BUFFER_STATE_RECORDING_KHR);
        test_error(error, "verify_state failed");

        // record command buffer
        error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        // verify executable state
        error = verify_state(CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
        test_error(error, "verify_state failed");

        error = clEnqueueFillBuffer(queue, out_mem, &pattern, sizeof(cl_int), 0,
                                    data_size(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        clEventWrapper trigger_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");

        // enqueued command buffer blocked on user event
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &trigger_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // verify pending state
        error = verify_state(CL_COMMAND_BUFFER_STATE_PENDING_KHR);
        test_error(error, "verify_state failed");

        // execute command buffer
        error = clSetUserEventStatus(trigger_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunPropArrayInfoTest()
    {
        cl_int error = CL_SUCCESS;

        // record command buffer
        error = RecordCommandBuffer(command_buffer);
        test_error(error, "RecordCommandBuffer failed");

        size_t ret_value_size = 0;
        cl_command_buffer_properties_khr combuf_props[16];
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
            sizeof(combuf_props), combuf_props, &ret_value_size);
        test_error_ret(error, "clGetCommandBufferInfoKHR failed", TEST_FAIL);

        if (ret_value_size > sizeof(combuf_props) || ret_value_size == 0)
        {
            log_error("clGetCommandBufferInfoKHR return values not as "
                      "expected\n");
            return TEST_FAIL;
        }

        int num_ret_props =
            ret_value_size / sizeof(cl_command_buffer_properties_khr);
        for (int i = 0; i < num_ret_props; i++)
            if (combuf_props[i] == CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR)
                return TEST_PASS;

        return TEST_FAIL;
    }

    //--------------------------------------------------------------------------

    clCommandBufferWrapper command_buffer_sec;
    clCommandQueueWrapper queue_sec;

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
