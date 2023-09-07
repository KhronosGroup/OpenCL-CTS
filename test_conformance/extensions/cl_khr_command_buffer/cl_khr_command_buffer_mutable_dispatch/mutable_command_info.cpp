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
#include "mutable_command_basic.h"

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

struct InfoDeviceQuery : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoDeviceQuery(cl_device_id device, cl_context context,
                    cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        cl_int error = clGetDeviceInfo(
            device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
            sizeof(mutable_capabilities), &mutable_capabilities, nullptr);
        test_error(error, "clGetDeviceInfo failed");

        if (!mutable_capabilities)
        {
            log_error("Device does not support update arguments to a "
                      "mutable-dispatch.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }
};

struct InfoBuffer : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoBuffer(cl_device_id device, cl_context context, cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
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

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_command_buffer_khr test_command_buffer = nullptr;
    cl_mutable_command_khr command = nullptr;
};

struct PropertiesArray : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    PropertiesArray(cl_device_id device, cl_context context,
                    cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
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
            log_error("ERROR: Incorrect properties returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct Kernel : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    Kernel(cl_device_id device, cl_context context, cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_kernel test_kernel;
        size_t size;

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_KERNEL_KHR, sizeof(test_kernel),
            &test_kernel, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        // We can not check if this is the right kernel because this is an
        // opaque object.
        if (test_kernel != kernel)
        {
            log_error("ERROR: Incorrect kernel returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct Dimensions : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    Dimensions(cl_device_id device, cl_context context, cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, dimensions, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_uint test_dimensions = 0;
        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_DIMENSIONS_KHR,
            sizeof(test_dimensions), &test_dimensions, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_dimensions != dimensions)
        {
            log_error("ERROR: Incorrect dimensions returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
    const size_t dimensions = 3;
};

struct InfoType : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoType(cl_device_id device, cl_context context, cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_command_type type = 0;
        error = clGetMutableCommandInfoKHR(command,
                                           CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR,
                                           sizeof(type), &type, NULL);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (type != CL_COMMAND_NDRANGE_KERNEL)
        {
            log_error("ERROR: Wrong type returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct InfoQueue : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoQueue(cl_device_id device, cl_context context, cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        cl_command_queue test_queue = nullptr;
        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR, sizeof(test_queue),
            &test_queue, nullptr);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_queue != queue)
        {
            log_error("ERROR: Incorrect queue returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct InfoGlobalWorkOffset : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoGlobalWorkOffset(cl_device_id device, cl_context context,
                         cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, &global_work_offset,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR,
            sizeof(test_global_work_offset), &test_global_work_offset, nullptr);

        if (test_global_work_offset != global_work_offset)
        {
            log_error("ERROR: Wrong global work offset returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
    const size_t global_work_offset = 4 * sizeof(cl_int);
    size_t test_global_work_offset = 0;
};

struct InfoGlobalWorkSize : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoGlobalWorkSize(cl_device_id device, cl_context context,
                       cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
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
            log_error("ERROR: Wrong global work size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    size_t test_global_work_size = 0;
};

struct InfoLocalWorkSize : public InfoMutableCommandBufferTest
{
    using InfoMutableCommandBufferTest::InfoMutableCommandBufferTest;

    InfoLocalWorkSize(cl_device_id device, cl_context context,
                      cl_command_queue queue)
        : InfoMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, &local_work_size, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
            sizeof(test_local_work_size), &test_local_work_size, nullptr);

        if (test_local_work_size != local_work_size)
        {
            log_error("ERROR: Wrong local work size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        error = clFinalizeCommandBufferKHR(command_buffer);
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
