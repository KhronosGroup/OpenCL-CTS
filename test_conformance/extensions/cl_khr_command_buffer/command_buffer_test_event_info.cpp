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
        clEventWrapper event;
        cl_int status;

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
};

struct CommandQueue : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        clEventWrapper event;
        size_t size;

        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_command_queue otherQueue;
        error = clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE,
                               sizeof(otherQueue), &otherQueue, &size);
        test_error(error, "Unable to get event info!");

        // We can not check if this is the right queue because this is an opaque
        // object.
        if (size != sizeof(queue) || otherQueue == NULL)
        {
            log_error("ERROR: Returned command queue size does not validate "
                      "(expected %zu, got %zu)\n",
                      sizeof(queue), size);
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }
};

struct Context : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        clEventWrapper event;
        size_t size;

        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_context testCtx;
        error = clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(testCtx),
                               &testCtx, &size);
        test_error(error, "Unable to get event context info!");
        if (size != sizeof(context))
        {
            log_error(
                "ERROR: Returned context size does not validate (expected "
                "%zu, got %zu)\n",
                sizeof(context), size);
            return TEST_FAIL;
        }
        if (testCtx != context)
        {
            log_error("ERROR: Returned context does not match (expected %p, "
                      "got %p)\n",
                      (void *)context, (void *)testCtx);
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }
};

struct ExecutionStatus : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        clEventWrapper event;
        cl_int status;

        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(status), &status, NULL);
        test_error(error, "clGetEventInfo failed");

        if (!(status == CL_QUEUED || status == CL_SUBMITTED
              || status == CL_RUNNING || status == CL_COMPLETE))
        {
            log_error(
                "ERROR: Incorrect status returned from clGetEventInfo (%d)\n",
                status);
            return TEST_FAIL;
        }

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
};

struct ReferenceCount : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        clEventWrapper event;
        size_t size;
        cl_uint count;

        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(count),
                               &count, &size);
        test_error(error, "clGetEventInfo failed");

        if (size != sizeof(count) || count == 0)
        {
            log_error(
                "ERROR: Wrong command reference count (expected return value 1 "
                "of size %zu, returned size %zu, returned value %u)\n",
                sizeof(count), size, count);
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }
};
};

int test_event_info_command_type(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandType>(device, context, queue, num_elements);
}

int test_event_info_command_queue(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CommandQueue>(device, context, queue, num_elements);
}

int test_event_info_context(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<Context>(device, context, queue, num_elements);
}

int test_event_info_execution_status(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ExecutionStatus>(device, context, queue,
                                           num_elements);
}

int test_event_info_reference_count(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<ReferenceCount>(device, context, queue, num_elements);
}