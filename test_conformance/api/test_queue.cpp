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


#include "testBase.h"
#include "harness/typeWrappers.h"

REGISTER_TEST(queue_flush_on_release)
{
    cl_int err;

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device, 0, &err);
    test_error(err, "Could not create command queue");

    // Create a kernel
    clProgramWrapper program;
    clKernelWrapper kernel;
    const char *source = "void kernel test(){}";
    err = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                      "test");
    test_error(err, "Could not create kernel");

    // Enqueue the kernel
    size_t gws = 1;
    clEventWrapper event;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, &gws, nullptr,
                                 0, nullptr, &event);
    test_error(err, "Could not enqueue kernel");

    // Release the queue
    err = clReleaseCommandQueue(cmd_queue);

    // Wait for kernel to execute since the queue must flush on release
    bool success = poll_until(2000, 50, [&event]() {
        cl_int status;
        cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int), &status, nullptr);
        if ((err != CL_SUCCESS) || (status != CL_COMPLETE))
        {
            return false;
        }
        return true;
    });

    return success ? TEST_PASS : TEST_FAIL;
}

REGISTER_TEST(multi_queue_flush_on_release)
{
    cl_int err;

    // Create A command queue
    cl_command_queue queue_A = clCreateCommandQueue(context, device, 0, &err);
    test_error(err, "Could not create command queue A");

    // Create B command queue
    cl_command_queue queue_B = clCreateCommandQueue(context, device, 0, &err);
    test_error(err, "Could not create command queue B");

    // Create a kernel
    clProgramWrapper program;
    clKernelWrapper kernel;
    const char *source = "void kernel test(){}";
    err = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                      "test");
    test_error(err, "Could not create kernel");

    // Enqueue the kernel on queue_A and obtain event to synchronize with
    // queue_B
    size_t gws = num_elements;
    clEventWrapper event_A;
    err = clEnqueueNDRangeKernel(queue_A, kernel, 1, nullptr, &gws, nullptr, 0,
                                 nullptr, &event_A);
    test_error(err, "Could not enqueue kernel");

    // Enqueue the kernel on queue_B using event_A for synchronization and
    // create event_B to track completion
    clEventWrapper event_B;
    err = clEnqueueNDRangeKernel(queue_B, kernel, 1, nullptr, &gws, nullptr, 1,
                                 &event_A, &event_B);
    test_error(err, "Could not enqueue kernel");

    // Release queue_A, which performs an implicit flush to issue any previously
    // queued OpenCL commands
    err = clReleaseCommandQueue(queue_A);
    test_error(err, "clReleaseCommandQueue failed");

    err = clFlush(queue_B);
    test_error(err, "clFlush failed");

    // Wait for kernel to execute since the queue must flush on release
    bool success = poll_until(2000, 50, [&event_B]() {
        cl_int status;
        cl_int err = clGetEventInfo(event_B, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int), &status, nullptr);
        if ((err != CL_SUCCESS) || (status != CL_COMPLETE))
        {
            return false;
        }
        return true;
    });

    return success ? TEST_PASS : TEST_FAIL;
}
