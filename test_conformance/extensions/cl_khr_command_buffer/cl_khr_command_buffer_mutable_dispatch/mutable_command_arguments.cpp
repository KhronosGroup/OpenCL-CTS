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

#include "testHarness.h"
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases for
// CL_MUTABLE_DISPATCH_ARGUMENTS_KHR:
// - __global arguments
// - __local arguments
// - plain-old-data arguments
// - NULL arguments
// - SVM arguments

struct MutableDispatchArgumentsTest : public BasicMutableCommandBufferTest
{
    MutableDispatchArgumentsTest(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          command(nullptr)
    {}

    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;
        cl_mutable_dispatch_fields_khr mutable_capabilities;
        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        // require mutable arguments capabillity
        return !mutable_support;
    }

    cl_mutable_command_khr command;
};

struct MutableDispatchGlobalArguments : public MutableDispatchArgumentsTest
{
    MutableDispatchGlobalArguments(cl_device_id device, cl_context context,
                                   cl_command_queue queue)
        : MutableDispatchArgumentsTest(device, context, queue)
    {}

    cl_int SetUpKernel() override
    {
        // Create kernel
        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                dst[tid] = src[tid];
            })";

        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &sample_const_arg_kernel,
                                                   "sample_test");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        // Create and initialize buffers
        MTdataHolder d(gRandomSeed);

        src_data.resize(num_elements);
        for (size_t i = 0; i < num_elements; i++)
            src_data[i] = (cl_int)genrand_int32(d);

        cl_int error = CL_SUCCESS;
        in_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                num_elements * sizeof(cl_int), src_data.data(),
                                &error);
        test_error(error, "Creating src buffer");

        dst_buf_0 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating initial dst buffer failed");

        dst_buf_1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating updated dst buffer failed");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
        test_error(error, "Unable to set src kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(dst_buf_0), &dst_buf_0);
        test_error(error, "Unable to set initial dst kernel argument");
        return CL_SUCCESS;
    }

    // verify the result
    bool verify_result(const cl_mem &buffer)
    {
        std::vector<cl_int> data(num_elements);
        cl_int error =
            clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, data_size(),
                                data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (data[i] != src_data[i])
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted %d\n",
                          i, data[i], src_data[i]);
                return false;
            }
        }
        return true;
    }

    cl_int Run() override
    {
        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // check the results of the initial execution
        if (!verify_result(dst_buf_0)) return TEST_FAIL;

        // Modify and execute the command buffer

        cl_mutable_dispatch_arg_khr arg{ 1, sizeof(dst_buf_1), &dst_buf_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            &arg /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution
        if (!verify_result(dst_buf_1)) return TEST_FAIL;

        return TEST_PASS;
    }

    std::vector<cl_int> src_data;

    clMemWrapper dst_buf_0;
    clMemWrapper dst_buf_1;
};

struct MutableDispatchLocalArguments : public MutableDispatchArgumentsTest
{
    MutableDispatchLocalArguments(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
        : MutableDispatchArgumentsTest(device, context, queue),
          number_of_ints(0), size_to_allocate(0)
    {}

    cl_int SetUpKernel() override
    {
        // Create kernel
        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src1, __local int
            *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                src[tid] = src1[tid];
                dst[tid] = src[tid];
            })";

        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &sample_const_arg_kernel,
                                                   "sample_test");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        MTdataHolder d(gRandomSeed);
        size_to_allocate = ((size_t)max_size / sizeof(cl_int)) * sizeof(cl_int);
        number_of_ints = size_to_allocate / sizeof(cl_int);
        constant_data.resize(size_to_allocate / sizeof(cl_int));
        result_data.resize(size_to_allocate / sizeof(cl_int));

        for (size_t i = 0; i < number_of_ints; i++)
            constant_data[i] = (cl_int)genrand_int32(d);

        cl_int error = CL_SUCCESS;
        streams[0] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size_to_allocate,
                           constant_data.data(), &error);
        test_error(error, "Creating test array failed");
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    size_to_allocate, nullptr, &error);
        test_error(error, "Creating test array failed");

        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &streams[0]);
        test_error(error, "Unable to set indexed kernel arguments");
        error =
            clSetKernelArg(kernel, 1, number_of_ints * sizeof(cl_int), nullptr);
        test_error(error, "Unable to set indexed kernel arguments");
        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &streams[1]);
        test_error(error, "Unable to set indexed kernel arguments");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        size_t threads[1], local_threads[1];

        threads[0] = number_of_ints;
        local_threads[0] = 1;

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, threads,
            local_threads, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(cl_mem), nullptr };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            args /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error =
            clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, size_to_allocate,
                                result_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < number_of_ints; i++)
            if (constant_data[i] != result_data[i])
            {
                log_error("Data failed to verify: constant_data[%zu]=%d != "
                          "result_data[%zu]=%d\n",
                          i, constant_data[i], i, result_data[i]);
                return TEST_FAIL;
            }

        return TEST_PASS;
    }

    const cl_ulong max_size = 16;

    std::vector<cl_int> constant_data;
    std::vector<cl_int> result_data;

    size_t number_of_ints;
    size_t size_to_allocate;

    clMemWrapper streams[2];
};

struct MutableDispatchPODArguments : public MutableDispatchArgumentsTest
{
    MutableDispatchPODArguments(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : MutableDispatchArgumentsTest(device, context, queue),
          number_of_ints(0), size_to_allocate(0), int_arg(10)
    {}

    cl_int SetUpKernel() override
    {
        // Create kernel
        const char *sample_const_arg_kernel =
            R"(
                __kernel void sample_test(__constant int *src, int dst)
            {
                size_t  tid = get_global_id(0);
                dst = src[tid];
            })";

        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &sample_const_arg_kernel,
                                                   "sample_test");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        MTdataHolder d(gRandomSeed);
        size_to_allocate = ((size_t)max_size / sizeof(cl_int)) * sizeof(cl_int);
        number_of_ints = size_to_allocate / sizeof(cl_int);
        constant_data.resize(size_to_allocate / sizeof(cl_int));
        result_data.resize(size_to_allocate / sizeof(cl_int));

        for (size_t i = 0; i < number_of_ints; i++)
            constant_data[i] = (cl_int)genrand_int32(d);

        cl_int error = CL_SUCCESS;
        stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size_to_allocate,
                                constant_data.data(), &error);
        test_error(error, "Creating test array failed");

        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_int), &int_arg);
        test_error(error, "Unable to set indexed kernel arguments");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        size_t threads[1], local_threads[1];

        threads[0] = number_of_ints;
        local_threads[0] = 1;

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, threads,
            local_threads, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        int_arg = 20;
        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(cl_int), &int_arg };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            args /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueReadBuffer(queue, stream, CL_TRUE, 0, size_to_allocate,
                                    result_data.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < number_of_ints; i++)
            if (constant_data[i] != result_data[i])
            {
                log_error("Data failed to verify: constant_data[%zu]=%d != "
                          "result_data[%zu]=%d\n",
                          i, constant_data[i], i, result_data[i]);
                return TEST_FAIL;
            }

        return TEST_PASS;
    }

    const cl_ulong max_size = 16;

    size_t number_of_ints;
    size_t size_to_allocate;
    cl_int int_arg;

    std::vector<cl_int> constant_data;
    std::vector<cl_int> result_data;

    clMemWrapper stream;
};

struct MutableDispatchNullArguments : public MutableDispatchArgumentsTest
{
    MutableDispatchNullArguments(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : MutableDispatchArgumentsTest(device, context, queue)
    {}

    cl_int SetUpKernel() override
    {
        // Create kernel
        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                dst[tid] = src ? src[tid] : 12345;
            })";

        cl_int error = create_single_kernel_helper(context, &program, &kernel,
                                                   1, &sample_const_arg_kernel,
                                                   "sample_test");
        test_error(error, "Creating kernel failed");
        return CL_SUCCESS;
    }

    cl_int SetUpKernelArgs() override
    {
        MTdataHolder d(gRandomSeed);
        src_data.resize(num_elements);
        for (size_t i = 0; i < num_elements; i++)
            src_data[i] = (cl_int)genrand_int32(d);

        cl_int error = CL_SUCCESS;
        in_mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                num_elements * sizeof(cl_int), src_data.data(),
                                &error);
        test_error(error, "Creating src buffer");

        out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating dst buffer failed");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
        test_error(error, "Unable to set src kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
        test_error(error, "Unable to set initial dst kernel argument");

        return CL_SUCCESS;
    }

    cl_int Run() override
    {

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        cl_int error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the initial execution
        std::vector<cl_int> dst_data_0(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dst_data_0.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for initial dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (src_data[i] != dst_data_0[i])
            {
                log_error("Initial data failed to verify: src[%zu]=%d != "
                          "dst[%zu]=%d\n",
                          i, src_data[i], i, dst_data_0[i]);
                return TEST_FAIL;
            }
        }

        // Modify and execute the command buffer
        cl_mutable_dispatch_arg_khr arg{ 0, sizeof(cl_mem), nullptr };
        cl_mutable_dispatch_config_khr dispatch_config{
            command,
            1 /* num_args */,
            0 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            &arg /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            nullptr /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution
        std::vector<cl_int> dst_data_1(num_elements);
        error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dst_data_1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for modified dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (12345 != dst_data_1[i])
            {
                log_error("Modified data failed to verify: %d != dst[%zu]=%d\n",
                          12345, i, dst_data_1[i]);
                return TEST_FAIL;
            }
        }

        return TEST_PASS;
    }

    const cl_ulong max_size = 16;

    std::vector<cl_int> src_data;
};

struct MutableDispatchSVMArguments : public MutableDispatchArgumentsTest
{
    MutableDispatchSVMArguments(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : MutableDispatchArgumentsTest(device, context, queue)
    {}

    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;

        cl_mutable_dispatch_fields_khr mutable_capabilities;
        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        cl_device_svm_capabilities svm_caps;
        bool svm_capabilities =
            !clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                             sizeof(svm_caps), &svm_caps, NULL)
            && svm_caps != 0;

        // require mutable arguments capabillity
        return !svm_capabilities || !mutable_support;
    }

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        const char *svm_arguments_kernel =
            R"(
            typedef struct {
                global int* ptr;
            } wrapper;
            __kernel void test_svm_arguments(__global wrapper* pWrapper)
            {
                size_t i = get_global_id(0);
                pWrapper->ptr[i]++;
            })";

        create_single_kernel_helper(context, &program, &kernel, 1,
                                    &svm_arguments_kernel,
                                    "test_svm_arguments");

        return 0;
    }

    cl_int Run() override
    {
        const cl_int zero = 0;

        // Allocate and initialize SVM for initial execution
        cl_int *init_wrapper = (cl_int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                                    sizeof(cl_int *), 0);
        cl_int *init_buffer = (cl_int *)clSVMAlloc(
            context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), 0);
        test_assert_error(init_wrapper != nullptr && init_buffer != nullptr,
                          "clSVMAlloc failed for initial execution");

        cl_int error =
            clEnqueueSVMMemcpy(queue, CL_TRUE, init_wrapper, &init_buffer,
                               sizeof(cl_int *), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed for init_wrapper");

        error = clEnqueueSVMMemFill(queue, init_buffer, &zero, sizeof(zero),
                                    num_elements * sizeof(cl_int), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed for init_buffer");

        // Allocate and initialize SVM for modified execution

        cl_int *new_wrapper = (cl_int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                                   sizeof(cl_int *), 0);
        cl_int *new_buffer = (cl_int *)clSVMAlloc(
            context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), 0);
        test_assert_error(new_wrapper != nullptr && new_buffer != nullptr,
                          "clSVMAlloc failed for modified execution");

        error = clEnqueueSVMMemcpy(queue, CL_TRUE, new_wrapper, &new_buffer,
                                   sizeof(cl_int *), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed for new_wrapper");

        error = clEnqueueSVMMemFill(queue, new_buffer, &zero, sizeof(zero),
                                    num_elements * sizeof(cl_int), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed for newB");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArgSVMPointer(kernel, 0, init_wrapper);
        test_error(error, "clSetKernelArg failed for init_wrapper");

        error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                    sizeof(init_buffer), &init_buffer);
        test_error(error, "clSetKernelExecInfo failed for init_buffer");

        cl_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR
                | CL_MUTABLE_DISPATCH_EXEC_INFO_KHR,
            0
        };
        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the initial execution
        error =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, init_buffer,
                            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMap failed for init_buffer");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (init_buffer[i] != 1)
            {
                log_error("Initial verification failed at index %zu: Got %d, "
                          "wanted 1\n",
                          i, init_buffer[i]);
                return TEST_FAIL;
            }
        }

        error = clEnqueueSVMUnmap(queue, init_buffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMUnmap failed for init_buffer");

        // Modify and execute the command buffer

        cl_mutable_dispatch_arg_khr arg_svm{};
        arg_svm.arg_index = 0;
        arg_svm.arg_value = new_wrapper;

        cl_mutable_dispatch_exec_info_khr exec_info{};
        exec_info.param_name = CL_KERNEL_EXEC_INFO_SVM_PTRS;
        exec_info.param_value_size = sizeof(new_buffer);
        exec_info.param_value = &new_buffer;

        cl_mutable_dispatch_config_khr dispatch_config{};
        dispatch_config.command = command;
        dispatch_config.num_svm_args = 1;
        dispatch_config.arg_svm_list = &arg_svm;
        dispatch_config.num_exec_infos = 1;
        dispatch_config.exec_info_list = &exec_info;

        cl_uint num_configs = 1;
        cl_command_buffer_update_type_khr config_types[1] = {
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR
        };
        const void *configs[1] = { &dispatch_config };
        error = clUpdateMutableCommandsKHR(command_buffer, num_configs,
                                           config_types, configs);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution
        error =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, new_buffer,
                            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMap failed for new_buffer");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (new_buffer[i] != 1)
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted 1\n",
                          i, new_buffer[i]);
                return TEST_FAIL;
            }
        }

        error = clEnqueueSVMUnmap(queue, new_buffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMUnmap failed for new_buffer");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Clean up
        clSVMFree(context, init_wrapper);
        clSVMFree(context, init_buffer);
        clSVMFree(context, new_wrapper);
        clSVMFree(context, new_buffer);

        return TEST_PASS;
    }
};

}

int test_mutable_dispatch_local_arguments(cl_device_id device,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    return MakeAndRunTest<MutableDispatchLocalArguments>(device, context, queue,
                                                         num_elements);
}

int test_mutable_dispatch_global_arguments(cl_device_id device,
                                           cl_context context,
                                           cl_command_queue queue,
                                           int num_elements)
{
    return MakeAndRunTest<MutableDispatchGlobalArguments>(device, context,
                                                          queue, num_elements);
}

int test_mutable_dispatch_pod_arguments(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return MakeAndRunTest<MutableDispatchPODArguments>(device, context, queue,
                                                       num_elements);
}

int test_mutable_dispatch_null_arguments(cl_device_id device,
                                         cl_context context,
                                         cl_command_queue queue,
                                         int num_elements)
{
    return MakeAndRunTest<MutableDispatchNullArguments>(device, context, queue,
                                                        num_elements);
}

int test_mutable_dispatch_svm_arguments(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return MakeAndRunTest<MutableDispatchSVMArguments>(device, context, queue,
                                                       num_elements);
}
