//
// Copyright (c) 2026 The Khronos Group Inc.
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

#include <harness/os_helpers.h>

#include "basic_command_buffer.h"

#include <vector>
#include <stdio.h>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Test for cl_khr_command_buffer which handles a command-buffer containing a
// device-side execution kernel being enqueued.

struct CommandBufferDeviceExecutionTest : public BasicCommandBufferTest
{
    CommandBufferDeviceExecutionTest(cl_device_id device, cl_context context,
                                     cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {
        buffer_size_multiplier = device_execution_size_mult;
    }

    bool Skip() override
    {
        if (BasicCommandBufferTest::Skip()) return true;

        // Query if device supports device-side execution use
        cl_device_command_buffer_capabilities_khr capabilities;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR");

        if (!(capabilities
              & CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR))
            return true;

        // check if device-side queue is supported
        cl_device_device_enqueue_capabilities enqueue_caps = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                                sizeof(enqueue_caps), &enqueue_caps, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES");

        if (!(enqueue_caps & CL_DEVICE_QUEUE_SUPPORTED)) return true;

        // check device supports creating queues with on-device properties
        cl_command_queue_properties supported_props = 0;
        error =
            clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                            sizeof(supported_props), &supported_props, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES");

        if (supported_props == 0) return true;

        return false;
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicCommandBufferTest::SetUp(elements);
        test_error(error, "BasicCommandBufferTest::SetUp failed");

        cl_uint preferred_size = 0;
        error =
            clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
                            sizeof(preferred_size), &preferred_size, NULL);
        test_error(error,
                   "Unable to query CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE");

        // create default device-side queue
        cl_queue_properties dev_queue_props[] = {
            CL_QUEUE_PROPERTIES,
            (cl_queue_properties)(CL_QUEUE_ON_DEVICE
                                  | CL_QUEUE_ON_DEVICE_DEFAULT
                                  | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
            CL_QUEUE_SIZE, preferred_size, 0
        };
        device_queue = clCreateCommandQueueWithProperties(
            context, device, dev_queue_props, &error);
        test_error(error, "Failed to create default device queue");

        return CL_SUCCESS;
    }

    cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;

        const char* kernel_str =
            R"(
            __kernel void device_enqueue(int mult, __global int* in, __global int* out, __global int* offset)
            {
                size_t id = get_global_id(0);
                int ind = offset[0] + id * mult;

                void (^command_buffer_device_block)(void) =
                ^{
                    size_t child_id = get_global_id(0);
                    out[ind + child_id] = in[ind + child_id];
                };

                ndrange_t ndrange = ndrange_1D(mult);
                enqueue_kernel(get_default_queue(),
                               CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                               ndrange,
                               command_buffer_device_block);
              })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "device_enqueue", &error);
        test_error(error, "Failed to create device execution kernel");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;
        in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_int) * num_elements
                                    * buffer_size_multiplier,
                                nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(cl_int) * num_elements
                                     * buffer_size_multiplier,
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int offset = 0;
        off_mem =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int), &offset, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int in_arg = device_execution_size_mult;
        error = clSetKernelArg(kernel, 0, sizeof(cl_int), &in_arg);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 1, sizeof(in_mem), &in_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 2, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel, 3, sizeof(off_mem), &off_mem);
        test_error(error, "clSetKernelArg failed");

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

    cl_int Run() override
    {
        // record command buffer
        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        const cl_int pattern = 42;
        error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                    data_size() * buffer_size_multiplier, 0,
                                    nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements * buffer_size_multiplier);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0,
                                    data_size() * buffer_size_multiplier,
                                    output_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements * buffer_size_multiplier; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    clCommandQueueWrapper device_queue;
    const cl_int device_execution_size_mult = 2;
};

} // anonymous namespace

REGISTER_TEST(device_execution)
{
    return MakeAndRunTest<CommandBufferDeviceExecutionTest>(
        device, context, queue, num_elements);
}
