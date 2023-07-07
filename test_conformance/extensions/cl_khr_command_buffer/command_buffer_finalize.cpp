//
// Copyright (c) 2023 The Khronos Group Inc.
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

#include "basic_command_buffer.h"
#include "procs.h"

namespace {

// Test that finalizing a command-buffer that has already been finalized returns
// the correct error code.
struct FinalizeInvalid : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Finalizing an already finalized command-buffer must return
        // CL_INVALID_OPERATION
        error = clFinalizeCommandBufferKHR(command_buffer);
        test_failure_error_ret(
            error, CL_INVALID_OPERATION,
            "clFinalizeCommandBufferKHR should return CL_INVALID_OPERATION",
            TEST_FAIL);

        return CL_SUCCESS;
    }
};

// Check that an empty command-buffer can be finalized and then executed.
struct FinalizeEmpty : public BasicCommandBufferTest
{
    using BasicCommandBufferTest::BasicCommandBufferTest;

    cl_int Run() override
    {
        // Finalize an empty command-buffer
        cl_int error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        // Execute empty command-buffer and then wait to complete
        clEventWrapper event;
        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, &event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &event);
        test_error(error, "clWaitForEvents failed");

        return CL_SUCCESS;
    }
};
} // anonymous namespace

int test_finalize_invalid(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<FinalizeInvalid>(device, context, queue,
                                           num_elements);
}

int test_finalize_empty(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<FinalizeEmpty>(device, context, queue, num_elements);
}
