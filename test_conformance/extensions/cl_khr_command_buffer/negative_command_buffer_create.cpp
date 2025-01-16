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

// CL_INVALID_VALUE if num_queues is not one.
struct CreateCommandBufferNumQueues : public BasicCommandBufferTest
{
    CreateCommandBufferNumQueues(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue), queue1(nullptr),
          queue2(nullptr)
    {}

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        cl_command_queue queues[2] = { queue1, queue2 };

        command_buffer = clCreateCommandBufferKHR(2, queues, nullptr, &error);

        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCreateCommandBufferKHR should return CL_INVALID_VALUE",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = CL_SUCCESS;

        error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        queue1 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");
        queue2 = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        return BasicCommandBufferTest::Skip()
            || is_extension_available(device,
                                      "cl_khr_command_buffer_multi_device");
    }

    clCommandQueueWrapper queue1;
    clCommandQueueWrapper queue2;
};

// CL_INVALID_VALUE if queues is NULL.
struct CreateCommandBufferNullQueues : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        command_buffer = clCreateCommandBufferKHR(1, nullptr, nullptr, &error);

        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCreateCommandBufferKHR should return CL_INVALID_VALUE",
            TEST_FAIL);

        return CL_SUCCESS;
    }
};

// CL_INVALID_VALUE if values specified in properties are not valid,
// or if the same property name is specified more than once.
struct CreateCommandBufferRepeatedProperties : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        cl_command_buffer_properties_khr repeated_properties[5] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, rep_prop, CL_COMMAND_BUFFER_FLAGS_KHR,
            rep_prop, 0
        };

        command_buffer =
            clCreateCommandBufferKHR(1, &queue, repeated_properties, &error);
        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCreateCommandBufferKHR should return CL_INVALID_VALUE",
            TEST_FAIL);

        cl_command_buffer_properties_khr invalid_properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, (cl_command_buffer_properties_khr)-1, 0
        };

        command_buffer =
            clCreateCommandBufferKHR(1, &queue, invalid_properties, &error);
        test_failure_error_ret(
            error, CL_INVALID_VALUE,
            "clCreateCommandBufferKHR should return CL_INVALID_VALUE",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;

        bool skip = true;
        if (simultaneous_use_support)
        {
            rep_prop = CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;
            skip = false;
        }
        else if (device_side_enqueue_support)
        {
            rep_prop = CL_COMMAND_BUFFER_DEVICE_SIDE_SYNC_KHR;
            skip = false;
        }
        else if (is_extension_available(
                     device,
                     CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME))
        {
            rep_prop = CL_COMMAND_BUFFER_MUTABLE_KHR;
            skip = false;
        }

        return skip;
    }

    cl_command_buffer_properties_khr rep_prop = 0;
};

// CL_INVALID_PROPERTY if values specified in properties are valid but are not
// supported by all the devices associated with command-queues in queues.
struct CreateCommandBufferNotSupportedProperties : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, unsupported_prop, 0
        };

        command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_failure_error_ret(
            error, CL_INVALID_PROPERTY,
            "clCreateCommandBufferKHR should return CL_INVALID_PROPERTY",
            TEST_FAIL);

        return CL_SUCCESS;
    }

    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;

        bool skip = true;
        if (!simultaneous_use_support)
        {
            unsupported_prop = CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;
            skip = false;
        }
        else if (!device_side_enqueue_support)
        {
            unsupported_prop = CL_COMMAND_BUFFER_DEVICE_SIDE_SYNC_KHR;
            skip = false;
        }

        return skip;
    }

    cl_command_buffer_properties_khr unsupported_prop = 0;
};
};

int test_negative_create_command_buffer_num_queues(cl_device_id device,
                                                   cl_context context,
                                                   cl_command_queue queue,
                                                   int num_elements)
{
    return MakeAndRunTest<CreateCommandBufferNumQueues>(device, context, queue,
                                                        num_elements);
}

int test_negative_create_command_buffer_null_queues(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<CreateCommandBufferNullQueues>(device, context, queue,
                                                         num_elements);
}

int test_negative_create_command_buffer_repeated_properties(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CreateCommandBufferRepeatedProperties>(
        device, context, queue, num_elements);
}

int test_negative_create_command_buffer_not_supported_properties(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{
    return MakeAndRunTest<CreateCommandBufferNotSupportedProperties>(
        device, context, queue, num_elements);
}
