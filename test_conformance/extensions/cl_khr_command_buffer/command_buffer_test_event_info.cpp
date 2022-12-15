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
#include "harness/typeWrappers.h"
#include "procs.h"

#include <vector>


namespace {

////////////////////////////////////////////////////////////////////////////////
// get event info tests which handles below cases:
//
// -command type
// -command queue
// -context
// -execution status
// -reference count

struct CommandType : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "Unable to wait for event");

        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(status),
                               &status, NULL);
        test_error(error, "clGetEventInfo failed");

        if (status != CL_COMMAND_COMMAND_BUFFER_KHR)
        {
            log_error(
                "ERROR: Incorrect status returned from clGetEventInfo (%d)\n",
                status);

            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    clEventWrapper event;
    cl_int status;
};

struct CommandQueue : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "Unable to wait for event");

        error = clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE, sizeof(ret_queue),
                               &ret_queue, &size);
        test_error(error, "clGetEventInfo failed");

        if (ret_queue != queue)
        {
            log_error("ERROR: Wrong command queue returned by clGetEventInfo");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    clEventWrapper event;
    cl_command_queue ret_queue = nullptr;
    size_t size;
};

struct Context : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed");

        error = clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(ret_context),
                               &ret_context, &size);
        test_error(error, "clGetEventInfo failed");

        if (ret_context != context)
        {
            log_error("ERROR: Wrong context returned by clGetEventInfo");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    clEventWrapper event;
    cl_context ret_context = nullptr;
    size_t size;
};

struct ExecutionStatus : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed");

        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(status), &status, NULL);
        test_error(error, "clGetEventInfo failed");

        if (status != CL_COMPLETE)
        {
            log_error(
                "ERROR: Incorrect status returned from clGetEventInfo (%d)\n",
                status);
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    clEventWrapper event;
    cl_int status;
};

struct ReferenceCount : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed");

        error = clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(count),
                               &count, &size);
        test_error(error, "clGetEventInfo failed");

        if (count != expected_count)
        {
            log_error(
                "ERROR: Wrong command reference count (expected %d, got %d)\n",
                (int)expected_count, (int)count);
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    clEventWrapper event;
    size_t size;
    cl_uint count;
    const cl_uint expected_count = 1;
};
};

int test_command_type(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandType>(device, context, queue, num_elements);
}

int test_command_queue(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandQueue>(device, context, queue, num_elements);
}

int test_context(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<Context>(device, context, queue, num_elements);
}

int test_execution_status(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ExecutionStatus>(device, context, queue,
                                           num_elements);
}

int test_reference_count(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ReferenceCount>(device, context, queue, num_elements);
}
