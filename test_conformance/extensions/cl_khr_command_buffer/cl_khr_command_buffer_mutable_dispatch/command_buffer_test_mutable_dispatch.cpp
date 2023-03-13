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
#include "../basic_command_buffer.h"
#include "../command_buffer_test_base.h"
#include <extensionHelpers.h>
#include "typeWrappers.h"
#include "procs.h"
#include "testHarness.h"
#include <vector>
#include <iostream>
#include <random>
#include <cstring>
#include <algorithm>
#include <memory>

#include <CL/cl.h>
#include <CL/cl_ext.h>
////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases:
//
// CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR
// CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR
// CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR
// CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR
// CL_MUTABLE_DISPATCH_KERNEL_KHR
// CL_MUTABLE_DISPATCH_DIMENSIONS_KHR
// CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR
// CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR
// CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR
// CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR

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
            && (mutable_capabilities
                & CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR)
                != 0;

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
};

struct InfoDeviceQuery : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoDeviceQuery(cl_device_id device, cl_context context,
                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        cl_int error = clGetDeviceInfo(
            device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
            sizeof(mutable_capabilities), &mutable_capabilities, nullptr);
        test_error(error, "clGetDeviceInfo failed");

        if (!(mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR))
        {
            log_error("Device does not support update arguments to a "
                      "mutable-dispatch.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }
};

struct InfoBuffer : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoBuffer(cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          test_command_buffer(this)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR,
            sizeof(test_command_buffer), &test_command_buffer, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_command_buffer != command_buffer)
        {
            log_error("ERROR: Incorrect command buffer returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    clCommandBufferWrapper test_command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
};

struct PropertiesArray : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    PropertiesArray(cl_device_id device, cl_context context,
                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          test_command_buffer(this)
    {}

    cl_int Run() override
    {
        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_ndrange_kernel_command_properties_khr test_props[] = { 0, 0, 0 };
        size_t size;

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR,
            sizeof(test_props), test_props, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (size != sizeof(props) || test_props[0] != props[0]
            || test_props[1] != props[1])
        {
            log_error("ERROR: Incorrect command buffer returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    clCommandBufferWrapper test_command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
};

struct Kernel : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    Kernel(cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          test_command_buffer(this)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        clKernelWrapper test_kernel;
        size_t size;

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_KERNEL_KHR, sizeof(test_kernel),
            &test_kernel, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        // We can not check if this is the right kernel because this is an
        // opaque object.
        if (size != sizeof(kernel) || test_kernel == nullptr)
        {
            log_error("ERROR: Incorrect command buffer returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    clCommandBufferWrapper test_command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
};

struct Dimensions : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    Dimensions(cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          test_command_buffer(this)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, dimensions, nullptr,
            nullptr, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        size_t test_dimensions;

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_DIMENSIONS_KHR,
            sizeof(test_dimensions), &test_dimensions, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_dimensions != dimensions)
        {
            log_error("ERROR: Incorrect command buffer returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    clCommandBufferWrapper test_command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
    const size_t dimensions = 3;
};

struct InfoType : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoType(cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_command_type type = 0;
        error = clGetMutableCommandInfoKHR(command,
                                           CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR,
                                           sizeof(type), &type, NULL);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (type == 0)
        {
            log_error("ERROR: Wrong type returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct InfoQueue : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoQueue(cl_device_id device, cl_context context, cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, nullptr,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_command_queue testQueue = nullptr;
        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR, sizeof(testQueue),
            &testQueue, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (!testQueue)
        {
            log_error("ERROR: Incorrect queue returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct InfoGlobalWorkOffset : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoGlobalWorkOffset(cl_device_id device, cl_context context,
                         cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, &global_work_offset,
            nullptr, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR,
            sizeof(test_global_work_offset), &test_global_work_offset, nullptr);

        if (test_global_work_offset != global_work_offset)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
    const size_t global_work_offset = 4 * sizeof(cl_int);
    size_t test_global_work_offset = 0;
};

struct InfoGlobalWorkSize : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoGlobalWorkSize(cl_device_id device, cl_context context,
                       cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR,
            sizeof(test_global_work_size), &test_global_work_size, nullptr);

        if (test_global_work_size != global_work_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const size_t global_work_size = 4 * sizeof(cl_int);
    size_t test_global_work_size = 0;
};

struct InfoLocalWorkSize : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    InfoLocalWorkSize(cl_device_id device, cl_context context,
                      cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, nullptr,
            &local_work_size, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
            sizeof(test_local_work_size), &test_local_work_size, nullptr);

        if (test_local_work_size != local_work_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
    const size_t local_work_size = 4 * sizeof(cl_int);
    size_t test_local_work_size = 0;
};

int test_mutable_command_info_device_query(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    return MakeAndRunTest<InfoDeviceQuery>(device, context, queue,
                                           num_elements);
}

int test_mutable_command_info_buffer(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<InfoBuffer>(device, context, queue, num_elements);
}

int test_mutable_command_properties_array(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    return MakeAndRunTest<PropertiesArray>(device, context, queue,
                                           num_elements);
}

int test_mutable_command_kernel(cl_device_id device, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<Kernel>(device, context, queue, num_elements);
}

int test_mutable_command_dimensions(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<Dimensions>(device, context, queue, num_elements);
}

int test_mutable_command_info_type(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<InfoType>(device, context, queue, num_elements);
}

int test_mutable_command_info_queue(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<InfoQueue>(device, context, queue, num_elements);
}

int test_mutable_command_info_global_work_offset(cl_device_id device,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 int num_elements)
{
    return MakeAndRunTest<InfoGlobalWorkOffset>(device, context, queue,
                                                num_elements);
}

int test_mutable_command_info_global_work_size(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue,
                                               int num_elements)
{
    return MakeAndRunTest<InfoGlobalWorkSize>(device, context, queue,
                                              num_elements);
}

int test_mutable_command_info_local_work_size(cl_device_id device,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements)
{
    return MakeAndRunTest<InfoLocalWorkSize>(device, context, queue,
                                             num_elements);
}
