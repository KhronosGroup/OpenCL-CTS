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

#include <extensionHelpers.h>
#include "imageHelpers.h"
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases:
//
// 1. The command buffer is created with
// CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR in its properties.
// 2. The ND-range command is recorded with
// CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR in its properties.
// 3. Both the command buffer and ND-range command have
// CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR in their properties.

struct Configuration
{
    const cl_command_buffer_properties_khr *command_buffer_properties;
    const cl_ndrange_kernel_command_properties_khr *ndrange_properties;
};

// Define the command buffer properties for each configuration
const cl_command_buffer_properties_khr command_buffer_properties[] = {
    CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR,
    CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR, 0
};

// Define the ndrange properties
const cl_ndrange_kernel_command_properties_khr ndrange_properties[] = {
    CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
    CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR, CL_MUTABLE_DISPATCH_ASSERTS_KHR,
    CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR, 0
};
// Initialize the array of configurations
Configuration configurations[] = { { command_buffer_properties, nullptr },
                                   { nullptr, ndrange_properties },
                                   { command_buffer_properties,
                                     ndrange_properties } };

template <int test_case>
struct MutableDispatchWorkGroups : public BasicMutableCommandBufferTest
{

    MutableDispatchWorkGroups(cl_device_id device, cl_context context,
                              cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          single_command_buffer(this)
    {
        config = configurations[test_case];
    }

    bool Skip() override
    {
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool no_additional_wgs_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities
                & CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR;

        return !no_additional_wgs_support
            || BasicMutableCommandBufferTest::Skip();
    }

    cl_int SetUp(int elements) override
    {
        cl_int error = BasicMutableCommandBufferTest::SetUp(elements);
        test_error(error, "BasicMutableCommandBufferTest::SetUp failed");

        error = SetUpKernel();
        test_error(error, "SetUpKernel failed");

        cl_command_buffer_properties_khr properties[5];
        int index = 0;
        properties[index++] = CL_COMMAND_BUFFER_FLAGS_KHR;
        properties[index++] = CL_COMMAND_BUFFER_MUTABLE_KHR;
        if (config.command_buffer_properties != nullptr)
        {
            properties[index++] = *config.command_buffer_properties;
            properties[index++] = *(config.command_buffer_properties + 1);
        }
        properties[index] = 0;

        single_command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        const char *num_groups_kernel =
            R"(
                __kernel void sample_test(__global int *dst)
            {
                size_t tid = get_global_id(0);
                dst[tid] = get_num_groups(0);
            })";
        cl_int error = create_single_kernel_helper(
            context, &program, &kernel, 1, &num_groups_kernel, "sample_test");
        test_error(error, "Creating kernel failed");

        clMemWrapper stream = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeToAllocate, nullptr, &error);

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        // Record an ND-range kernel of the kernel above in the command buffer
        // with a non-null local work size so that the resulting number of
        // workgroups will be greater than 1.
        error = clCommandNDRangeKernelKHR(
            single_command_buffer, nullptr, config.ndrange_properties, kernel,
            1, nullptr, &global_work_size, &local_work_size, 0, nullptr,
            nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(single_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        clEventWrapper events[2];
        error = clEnqueueCommandBufferKHR(0, nullptr, single_command_buffer, 0,
                                          nullptr, &events[0]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &events[0]);
        test_error(error, "clWaitForEvents failed");

        std::vector<cl_int> resultData;
        resultData.resize(global_work_size);
        error = clEnqueueReadBuffer(queue, stream, CL_FALSE, 0, sizeToAllocate,
                                    resultData.data(), 0, nullptr, &events[1]);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clWaitForEvents(1, &events[1]);
        test_error(error, "clWaitForEvents failed");

        for (size_t i = 0; i < global_work_size; i++)
            if (global_work_size / local_work_size != resultData[i])
            {
                log_error("Data failed to verify: global_work_size != "
                          "resultData[%zu]=%d\n",
                          i, resultData[i]);
                return TEST_FAIL;
            }
        // Test Case 1: WGu = WG0 if the user explicitly sets the number of
        // workgroups (by specifying a non-null local work size) to be WG0 when
        // the ND-range kernel command is recorded, the new number of workgroups
        // - if updated - WGu, will be equal to WG0.
        error = TestUpdateWorkGroups(global_work_size, stream, resultData);
        if (error != CL_SUCCESS) return error;

        // Test Case 2: WGu < WG0 if the user explicitly sets the number of
        // workgroups to be WG0 when the ND-range kernel command is recorded,
        // the new number of workgroups - if updated - WGu, will be less that
        // WG0.
        static_assert(update_global_size != 0,
                      "update_global_size should not be zero");
        error = TestUpdateWorkGroups(update_global_size, stream, resultData,
                                     global_work_size);
        if (error != CL_SUCCESS) return error;

        // Test Case 3: WG0 ≥ WGu > WG1 if the user explicitly sets the number
        // of workgroups to be WG0 when the ND-range kernel command is recorded,
        // the new number of workgroups - if updated - WG1, will less that WG0.
        // Then, call the API function again to update the number of workgroups
        // to be WGu so that WG0 ≥ WGu > WG1.
        error = TestUpdateWorkGroups(update_global_size * 2, stream, resultData,
                                     global_work_size);
        if (error != CL_SUCCESS) return error;

        return CL_SUCCESS;
    }

    cl_int TestUpdateWorkGroups(size_t new_global_size, clMemWrapper &stream,
                                std::vector<cl_int> &resultData,
                                size_t old_global_size = 0)
    {
        cl_int error;
        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0, // num_args
            0, // num_svm_arg
            0, // num_exec_infos
            0, // work_dim (0 means no change to dimensions)
            nullptr, // arg_list
            nullptr, // arg_svm_list (nullptr means no change)
            nullptr, // exec_info_list
            nullptr, // global_work_offset
            &new_global_size, // global_work_size
            nullptr // local_work_size
        };

        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error =
            clUpdateMutableCommandsKHR(single_command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        clEventWrapper events[2];
        error = clEnqueueCommandBufferKHR(0, nullptr, single_command_buffer, 0,
                                          nullptr, &events[0]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clWaitForEvents(1, &events[0]);
        test_error(error, "clWaitForEvents failed");

        error = clEnqueueReadBuffer(queue, stream, CL_FALSE, 0, sizeToAllocate,
                                    resultData.data(), 0, nullptr, &events[1]);
        test_error(error, "clEnqueueReadBuffer failed");

        error = clWaitForEvents(1, &events[1]);
        test_error(error, "clWaitForEvents failed");

        size_t expected_groups = new_global_size / local_work_size;
        size_t old_num_of_groups = old_global_size / local_work_size;
        for (size_t i = 0; i < global_work_size; ++i)
        {
            if (i >= new_global_size && old_num_of_groups != resultData[i])
            {
                log_error("Data failed to verify: old_num_of_groups != "
                          "resultData[%zu]=%d\n",
                          i, resultData[i]);
                return TEST_FAIL;
            }
            else if (i < new_global_size && expected_groups != resultData[i])
            {
                log_error("Data failed to verify: expected_groups != "
                          "resultData[%zu]=%d\n",
                          i, resultData[i]);
                return TEST_FAIL;
            }
        }

        return CL_SUCCESS;
    }

    clCommandBufferWrapper single_command_buffer;
    cl_mutable_command_khr command = nullptr;
    Configuration config;

    size_t info_global_size = 0;
    static constexpr size_t test_global_work_size = 64;
    static constexpr size_t update_global_size = 16;
    const size_t local_work_size = 8;
    const size_t sizeToAllocate = 64 * sizeof(cl_int);
};

int test_command_buffer_with_no_additional_work_groups(cl_device_id device,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{

    return MakeAndRunTest<MutableDispatchWorkGroups<0>>(device, context, queue,
                                                        num_elements);
}

int test_ndrange_with_no_additional_work_groups(cl_device_id device,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements)
{

    return MakeAndRunTest<MutableDispatchWorkGroups<1>>(device, context, queue,
                                                        num_elements);
}

int test_ndrange_command_buffer_with_no_additional_work_groups(
    cl_device_id device, cl_context context, cl_command_queue queue,
    int num_elements)
{

    return MakeAndRunTest<MutableDispatchWorkGroups<2>>(device, context, queue,
                                                        num_elements);
}
