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

#ifndef CL_KHR_MUTABLE_COMMAND_BASIC_H
#define CL_KHR_MUTABLE_COMMAND_BASIC_H

#include "../basic_command_buffer.h"
#include "../command_buffer_test_base.h"

struct BasicMutableCommandBufferTest : BasicCommandBufferTest
{
    BasicMutableCommandBufferTest(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;
        clProgramWrapper program = clCreateProgramWithSource(
            context, 1, &kernelString, nullptr, &error);
        test_error(error, "Unable to create program");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Unable to build program");

        kernel = clCreateKernel(program, "empty", &error);
        test_error(error, "Unable to create kernel");

        return CL_SUCCESS;
    }

    virtual cl_int SetUpKernelArgs() override
    {
        /* Left blank intentionally */
        return CL_SUCCESS;
    }

    virtual cl_int SetUp(int elements) override
    {
        BasicCommandBufferTest::SetUp(elements);

        cl_int error = init_extension_functions();
        test_error(error, "Unable to initialise extension functions");

        cl_command_buffer_properties_khr prop = CL_COMMAND_BUFFER_MUTABLE_KHR;
        if (simultaneous_use_support)
        {
            prop |= CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;
        }

        const cl_command_buffer_properties_khr props[] = {
            CL_COMMAND_BUFFER_FLAGS_KHR,
            prop,
            0,
        };

        command_buffer = clCreateCommandBufferKHR(1, &queue, props, &error);
        test_error(error, "Unable to create command buffer");

        return error;
    }

    bool Skip() override
    {
        bool extension_avaliable =
            is_extension_available(device,
                                   "cl_khr_command_buffer_mutable_dispatch")
            == true;

        if (extension_avaliable)
        {
            Version device_version = get_device_cl_version(device);
            if ((device_version >= Version(3, 0))
                || is_extension_available(device, "cl_khr_extended_versioning"))
            {

                cl_version extension_version = get_extension_version(
                    device, "cl_khr_command_buffer_mutable_dispatch");

                if (extension_version != CL_MAKE_VERSION(0, 9, 3))
                {
                    log_info("cl_khr_command_buffer_mutable_dispatch version "
                             "0.9.3 is "
                             "required to run the test, skipping.\n ");
                    extension_avaliable = false;
                }
            }
        }

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

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        return CL_SUCCESS;
    }

    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;

    const char* kernelString = "__kernel void empty() {}";
    const size_t global_work_size = 4 * 16;
};

struct InfoMutableCommandBufferTest : BasicMutableCommandBufferTest
{
    InfoMutableCommandBufferTest(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_int error = init_extension_functions();
        test_error(error, "Unable to initialise extension functions");

        return CL_SUCCESS;
    }

    cl_int init_extension_functions()
    {
        BasicCommandBufferTest::init_extension_functions();

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clGetMutableCommandInfoKHR);

        return CL_SUCCESS;
    }

    clGetMutableCommandInfoKHR_fn clGetMutableCommandInfoKHR = nullptr;
};

#undef GET_EXTENSION_ADDRESS

#endif //_CL_KHR_MUTABLE_COMMAND_BASIC_H
