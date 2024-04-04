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
#include <vector>

//--------------------------------------------------------------------------

namespace {

enum class CombufInfoTestMode
{
    CITM_PARAM_NAME = 0,
    CITM_QUEUES,
    CITM_REF_COUNT,
    CITM_STATE,
    CITM_PROP_ARRAY,
    CITM_CONTEXT
};

// CL_INVALID_COMMAND_BUFFER_KHR if command_buffer is not a valid
// command-buffer.
struct GetCommandBufferInfoInvalidCommandBuffer : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error =
            clGetCommandBufferInfoKHR(nullptr, CL_COMMAND_BUFFER_NUM_QUEUES_KHR,
                                      sizeof(cl_uint), nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_COMMAND_BUFFER_KHR,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_COMMAND_BUFFER_KHR",
                               TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if param_name is not one of the supported values or if size
// in bytes specified by param_value_size is less than size of return type and
// param_value is not a NULL value.
template <CombufInfoTestMode test_mode>
struct GetCommandBufferInfo : public BasicCommandBufferTest
{
    GetCommandBufferInfo(cl_device_id device, cl_context context,
                         cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), number_of_queues(0),
          number_of_properties(0)
    {}

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        switch (test_mode)
        {
            case CombufInfoTestMode::CITM_PARAM_NAME:
                error = RunParamNameTest();
                test_error(error, "RunParamNameTest failed");
                break;
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

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_NUM_QUEUES_KHR, sizeof(cl_uint),
            &number_of_queues, nullptr);
        test_error(error, "Unable to query CL_COMMAND_BUFFER_NUM_QUEUES_KHR");


        size_t ret_value_size = 0;
        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR, 0, nullptr,
            &ret_value_size);
        test_error(error,
                   "Unable to query CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR");

        number_of_properties =
            ret_value_size / sizeof(cl_command_buffer_properties_khr);

        return CL_SUCCESS;
    }

    cl_int RunParamNameTest()
    {
        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_FLAGS_KHR, sizeof(cl_uint),
            nullptr, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        cl_uint ret_val = 0;

        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_FLAGS_KHR, sizeof(ret_val) - 1,
            &ret_val, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_NUM_QUEUES_KHR,
            sizeof(ret_val) - 1, &ret_val, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return TEST_PASS;
    }

    cl_int RunQueuesInfoTest()
    {
        cl_uint num_queues = 0;

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_NUM_QUEUES_KHR,
            sizeof(num_queues) - 1, &num_queues, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        // cl_command_queue queues[number_of_queues];
        std::vector<cl_command_queue> queues;
        queues.resize(number_of_queues);

        error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_QUEUES_KHR,
            number_of_queues * sizeof(cl_command_queue) - 1, queues.data(),
            nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return TEST_PASS;
    }

    cl_int RunRefCountInfoTest()
    {
        cl_uint ref_count = 0;

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR,
            sizeof(ref_count) - 1, &ref_count, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);
        return TEST_PASS;
    }

    cl_int RunStateInfoTest()
    {
        cl_uint state = 0;

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_STATE_KHR, sizeof(state) - 1,
            &state, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);
        return TEST_PASS;
    }

    cl_int RunPropArrayInfoTest()
    {
        std::vector<cl_command_buffer_properties_khr> properties;
        properties.resize(number_of_properties);

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR,
            number_of_properties * sizeof(cl_command_buffer_properties_khr) - 1,
            properties.data(), nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return TEST_PASS;
    }

    cl_int RunContextInfoTest()
    {
        cl_context context = nullptr;

        cl_int error = clGetCommandBufferInfoKHR(
            command_buffer, CL_COMMAND_BUFFER_CONTEXT_KHR, sizeof(context) - 1,
            &context, nullptr);

        test_failure_error_ret(error, CL_INVALID_VALUE,
                               "clGetCommandBufferInfoKHR should return "
                               "CL_INVALID_VALUE",
                               TEST_FAIL);

        return TEST_PASS;
    }

    cl_uint number_of_queues;
    cl_uint number_of_properties;
};
};

int test_negative_get_command_buffer_info_invalid_command_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<GetCommandBufferInfoInvalidCommandBuffer>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_not_supported_param_name(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<
        GetCommandBufferInfo<CombufInfoTestMode::CITM_PARAM_NAME>>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_queues(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements)
{
    return MakeAndRunTest<
        GetCommandBufferInfo<CombufInfoTestMode::CITM_QUEUES>>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_ref_count(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<
        GetCommandBufferInfo<CombufInfoTestMode::CITM_REF_COUNT>>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_state(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{
    return MakeAndRunTest<GetCommandBufferInfo<CombufInfoTestMode::CITM_STATE>>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_prop_array(cl_device_id device,
                                                     cl_context context,
                                                     cl_command_queue queue,
                                                     int num_elements)
{
    return MakeAndRunTest<
        GetCommandBufferInfo<CombufInfoTestMode::CITM_PROP_ARRAY>>(
        device, context, queue, num_elements);
}

int test_negative_get_command_buffer_info_context(cl_device_id device,
                                                  cl_context context,
                                                  cl_command_queue queue,
                                                  int num_elements)
{
    return MakeAndRunTest<
        GetCommandBufferInfo<CombufInfoTestMode::CITM_CONTEXT>>(
        device, context, queue, num_elements);
}
