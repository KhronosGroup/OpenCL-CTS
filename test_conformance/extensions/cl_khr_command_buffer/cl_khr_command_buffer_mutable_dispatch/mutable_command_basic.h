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

#ifndef _CL_KHR_MUTABLE_COMMAND_BASIC_H
#define _CL_KHR_MUTABLE_COMMAND_BASIC_H

#include "../basic_command_buffer.h"
#include "../command_buffer_test_base.h"

struct BasicMutableCommandBufferTest : BasicCommandBufferTest
{
    BasicMutableCommandBufferTest(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicCommandBufferTest::SetUp(elements);

        cl_int error = init_extension_functions();
        test_error(error, "Unable to initialise extension functions");

        const cl_command_buffer_properties_khr props[] = {
            CL_COMMAND_BUFFER_FLAGS_KHR,
            CL_COMMAND_BUFFER_MUTABLE_KHR,
            0,
        };

        command_buffer = clCreateCommandBufferKHR(1, &queue, props, &error);
        test_error(error, "Unable to create command buffer");

        clProgramWrapper program = clCreateProgramWithSource(
            context, 1, &kernelString, nullptr, &error);
        test_error(error, "Unable to create program");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Unable to build program");

        kernel = clCreateKernel(program, "empty", &error);
        test_error(error, "Unable to create kernel");

        return error;
    }

    bool Skip() override
    {
        bool extension_avaliable =
            is_extension_available(device,
                                   "cl_khr_command_buffer_mutable_dispatch")
            == true;

        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities != 0;

        return !mutable_support || !extension_avaliable
            || BasicCommandBufferTest::Skip();
    }

    cl_int init_extension_functions()
    {
        BasicCommandBufferTest::init_extension_functions();

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        // If it is supported get the addresses of all the APIs here.
#define GET_EXTENSION_ADDRESS(FUNC)                                            \
    FUNC = reinterpret_cast<FUNC##_fn>(                                        \
        clGetExtensionFunctionAddressForPlatform(platform, #FUNC));            \
    if (FUNC == nullptr)                                                       \
    {                                                                          \
        log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed"     \
                  " with " #FUNC "\n");                                        \
        return TEST_FAIL;                                                      \
    }
        GET_EXTENSION_ADDRESS(clGetMutableCommandInfoKHR);

        return CL_SUCCESS;
    }

    clGetMutableCommandInfoKHR_fn clGetMutableCommandInfoKHR = nullptr;
    const char* kernelString = "__kernel void empty() {}";
    const size_t global_work_size = 4 * sizeof(cl_int);
};

#endif //_CL_KHR_MUTABLE_COMMAND_BASIC_H