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
#include "imageHelpers.h"
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

struct MutableDispatchGlobalOffset : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchGlobalOffset(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          mutable_command_buffer(this), out_of_order_queue(nullptr)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, 0, 0
        };
        properties[1] = CL_COMMAND_BUFFER_MUTABLE_KHR;

        mutable_command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return 0;
    }

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            mutable_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(mutable_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            &global_offset /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(mutable_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR,
            sizeof(test_global_work_offset), &test_global_work_offset, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_global_work_offset != global_offset)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    size_t test_global_work_offset = 0;
    const size_t global_offset = 3;
    size_t size;

    cl_mutable_command_khr command = nullptr;

    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper mutable_command_buffer;
};

struct MutableDispatchGlobalSize : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchGlobalSize(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          mutable_command_buffer(this), out_of_order_queue(nullptr)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, 0, 0
        };
        properties[1] = CL_COMMAND_BUFFER_MUTABLE_KHR;

        mutable_command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return 0;
    }

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            mutable_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(mutable_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            &global_size /* global_work_size */,
            nullptr /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(mutable_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR,
            sizeof(test_global_work_size), &test_global_work_size, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_global_work_size != global_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    size_t test_global_work_size = 0;
    const size_t global_size = 3;
    size_t size;

    cl_mutable_command_khr command = nullptr;

    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper mutable_command_buffer;
};

struct MutableDispatchLocalSize : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchLocalSize(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          mutable_command_buffer(this), out_of_order_queue(nullptr)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, 0, 0
        };
        properties[1] = CL_COMMAND_BUFFER_MUTABLE_KHR;

        mutable_command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return 0;
    }

    cl_int Run() override
    {
        cl_int error = clCommandNDRangeKernelKHR(
            mutable_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(mutable_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            &local_size /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(mutable_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR,
            sizeof(test_local_work_size), &test_local_work_size, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        if (test_local_work_size != local_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    size_t test_local_work_size = 0;
    const size_t local_size = 3;
    size_t size;

    cl_mutable_command_khr command = nullptr;

    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper mutable_command_buffer;
};

struct MutableDispatchArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchArguments(cl_device_id device, cl_context context,
                             cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          mutable_command_buffer(this), out_of_order_queue(nullptr)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, 0, 0
        };
        properties[1] = CL_COMMAND_BUFFER_MUTABLE_KHR;

        mutable_command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return 0;
    }

    cl_int Run() override
    {
        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            mutable_command_buffer, nullptr, props, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(mutable_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(mutable_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, mutable_command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_ndrange_kernel_command_properties_khr test_props[] = { 0, 0, 0 };
        size_t size;

        error = clGetMutableCommandInfoKHR(
            command, CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, sizeof(test_props),
            &test_props, &size);
        test_error(error, "clGetMutableCommandInfoKHR failed");

        /*
        if (test_local_work_size != local_size)
        {
            log_error("ERROR: Wrong size returned from "
                      "clGetMutableCommandInfoKHR.");
            return TEST_FAIL;
        }
        */

        return CL_SUCCESS;
    }

    size_t test_local_work_size = 0;
    const size_t local_size = 3;
    size_t size;

    cl_mutable_command_khr command = nullptr;

    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper mutable_command_buffer;
};

int test_mutable_dispatch_global_offset(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{

    return MakeAndRunTest<MutableDispatchGlobalOffset>(device, context, queue,
                                                       num_elements);
}

int test_mutable_dispatch_global_size(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableDispatchGlobalSize>(device, context, queue,
                                                     num_elements);
}

int test_mutable_dispatch_local_size(cl_device_id device, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableDispatchLocalSize>(device, context, queue,
                                                    num_elements);
}

int test_mutable_dispatch_arguments(cl_device_id device, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<MutableDispatchArguments>(device, context, queue,
                                                    num_elements);
}
