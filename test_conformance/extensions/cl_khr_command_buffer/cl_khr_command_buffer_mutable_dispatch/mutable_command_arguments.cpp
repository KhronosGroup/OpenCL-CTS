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
#include "imageHelpers.h"
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases for
// CL_MUTABLE_DISPATCH_ARGUMENTS_KHR:
// - __global arguments
// - __local arguments
// - plain-old-data arguments
// - NULL arguments
// - SVM arguments

struct MutableDispatchGlobalArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchGlobalArguments(cl_device_id device, cl_context context,
                                   cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        return 0;
    }

    cl_int Run() override
    {
        cl_int error;

        // Create kernel

        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                dst[tid] = src[tid];
            })";

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        // Create and initialize buffers

        MTdataHolder d(gRandomSeed);

        std::vector<cl_int> srcData(num_elements);
        for (size_t i = 0; i < num_elements; i++)
            srcData[i] = (cl_int)genrand_int32(d);

        clMemWrapper srcBuf = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                             num_elements * sizeof(cl_int),
                                             srcData.data(), &error);
        test_error(error, "Creating src buffer");

        clMemWrapper dstBuf0 =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating initial dst buffer failed");

        clMemWrapper dstBuf1 =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating updated dst buffer failed");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArg(kernel, 0, sizeof(srcBuf), &srcBuf);
        test_error(error, "Unable to set src kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(dstBuf0), &dstBuf0);
        test_error(error, "Unable to set initial dst kernel argument");

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
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

        std::vector<cl_int> dstData0(num_elements);
        error = clEnqueueReadBuffer(queue, dstBuf0, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dstData0.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for initial dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (srcData[i] != dstData0[i])
            {
                log_error("Initial data failed to verify: src[%zu]=%d != "
                          "dst[%zu]=%d\n",
                          i, srcData[i], i, dstData0[i]);
                return TEST_FAIL;
            }
        }

        // Modify and execute the command buffer

        cl_mutable_dispatch_arg_khr arg{ 1, sizeof(dstBuf1), &dstBuf1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
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

        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution

        std::vector<cl_int> dstData1(num_elements);
        error = clEnqueueReadBuffer(queue, dstBuf1, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dstData1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for modified dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (srcData[i] != dstData1[i])
            {
                log_error("Initial data failed to verify: src[%zu]=%d != "
                          "dst[%zu]=%d\n",
                          i, srcData[i], i, dstData1[i]);
                return TEST_FAIL;
            }
        }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
};

struct MutableDispatchLocalArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchLocalArguments(cl_device_id device, cl_context context,
                                  cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        return 0;
    }

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src1, __local int
            *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                src[tid] = src1[tid];
                dst[tid] = src[tid];
            })";

        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;
        size_t threads[1], localThreads[1];
        std::vector<cl_int> constantData;
        std::vector<cl_int> resultData;

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)max_size / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (cl_int)genrand_int32(d);

        clMemWrapper streams[2];
        streams[0] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeToAllocate,
                           constantData.data(), &error);
        test_error(error, "Creating test array failed");
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                    nullptr, &error);
        test_error(error, "Creating test array failed");

        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &streams[0]);
        test_error(error, "Unable to set indexed kernel arguments");
        error =
            clSetKernelArg(kernel, 1, numberOfInts * sizeof(cl_int), nullptr);
        test_error(error, "Unable to set indexed kernel arguments");
        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &streams[1]);
        test_error(error, "Unable to set indexed kernel arguments");

        threads[0] = numberOfInts;
        localThreads[0] = 1;

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, threads,
            localThreads, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(cl_mem), nullptr };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
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
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error =
            clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, sizeToAllocate,
                                resultData.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < numberOfInts; i++)
            if (constantData[i] != resultData[i])
            {
                log_error("Data failed to verify: constantData[%d]=%d != "
                          "resultData[%d]=%d\n",
                          i, constantData[i], i, resultData[i]);
                return TEST_FAIL;
            }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const cl_ulong max_size = 16;
};

struct MutableDispatchPODArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchPODArguments(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(
                __kernel void sample_test(__constant int *src, int dst)
            {
                size_t  tid = get_global_id(0);
                dst = src[tid];
            })";

        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;
        size_t threads[1], localThreads[1];
        std::vector<cl_int> constantData;
        std::vector<cl_int> resultData;

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)max_size / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (cl_int)genrand_int32(d);

        clMemWrapper stream;
        stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeToAllocate,
                                constantData.data(), &error);
        test_error(error, "Creating test array failed");


        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");
        cl_int intarg = 10;
        error = clSetKernelArg(kernel, 1, sizeof(cl_int), &intarg);
        test_error(error, "Unable to set indexed kernel arguments");

        threads[0] = numberOfInts;
        localThreads[0] = 1;

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, threads,
            localThreads, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        intarg = 20;
        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(cl_int), &intarg };
        cl_mutable_dispatch_arg_khr args[] = { arg_1 };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
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
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueReadBuffer(queue, stream, CL_TRUE, 0, sizeToAllocate,
                                    resultData.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < numberOfInts; i++)
            if (constantData[i] != resultData[i])
            {
                log_error("Data failed to verify: constantData[%d]=%d != "
                          "resultData[%d]=%d\n",
                          i, constantData[i], i, resultData[i]);
                return TEST_FAIL;
            }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const cl_ulong max_size = 16;
};

struct MutableDispatchNullArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchNullArguments(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    cl_int Run() override
    {
        cl_int error;

        // Create kernel

        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                dst[tid] = src ? src[tid] : 12345;
            })";

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        MTdataHolder d(gRandomSeed);

        std::vector<cl_int> srcData(num_elements);
        for (size_t i = 0; i < num_elements; i++)
            srcData[i] = (cl_int)genrand_int32(d);

        clMemWrapper srcBuf = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                             num_elements * sizeof(cl_int),
                                             srcData.data(), &error);
        test_error(error, "Creating src buffer");

        clMemWrapper dstBuf =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           num_elements * sizeof(cl_int), NULL, &error);
        test_error(error, "Creating dst buffer failed");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArg(kernel, 0, sizeof(srcBuf), &srcBuf);
        test_error(error, "Unable to set src kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(dstBuf), &dstBuf);
        test_error(error, "Unable to set initial dst kernel argument");

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
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

        std::vector<cl_int> dstData0(num_elements);
        error = clEnqueueReadBuffer(queue, dstBuf, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dstData0.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for initial dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (srcData[i] != dstData0[i])
            {
                log_error("Initial data failed to verify: src[%zu]=%d != "
                          "dst[%zu]=%d\n",
                          i, srcData[i], i, dstData0[i]);
                return TEST_FAIL;
            }
        }

        // Modify and execute the command buffer

        cl_mutable_dispatch_arg_khr arg{ 0, sizeof(cl_mem), nullptr };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
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

        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution

        std::vector<cl_int> dstData1(num_elements);
        error = clEnqueueReadBuffer(queue, dstBuf, CL_TRUE, 0,
                                    num_elements * sizeof(cl_int),
                                    dstData1.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer for modified dst failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (12345 != dstData1[i])
            {
                log_error("Modified data failed to verify: %d != dst[%zu]=%d\n",
                          12345, i, dstData1[i]);
                return TEST_FAIL;
            }
        }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const cl_ulong max_size = 16;
};

struct MutableDispatchSVMArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchSVMArguments(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    bool Skip() override
    {
        cl_device_svm_capabilities svm_caps;
        bool svm_capabilities =
            !clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                             sizeof(svm_caps), &svm_caps, NULL)
            && svm_caps != 0;

        return !svm_capabilities || BasicMutableCommandBufferTest::Skip();
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
        cl_int error;

        // Allocate and initialize SVM for initial execution

        cl_int *initWrapper = (cl_int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                                   sizeof(cl_int *), 0);
        cl_int *initBuffer = (cl_int *)clSVMAlloc(
            context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), 0);
        test_assert_error(initWrapper != nullptr && initBuffer != nullptr,
                          "clSVMAlloc failed for initial execution");

        error = clEnqueueSVMMemcpy(queue, CL_TRUE, initWrapper, &initBuffer,
                                   sizeof(cl_int *), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed for initWrapper");

        error = clEnqueueSVMMemFill(queue, initBuffer, &zero, sizeof(zero),
                                    num_elements * sizeof(cl_int), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed for initBuffer");

        // Allocate and initialize SVM for modified execution

        cl_int *newWrapper = (cl_int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                                  sizeof(cl_int *), 0);
        cl_int *newBuffer = (cl_int *)clSVMAlloc(
            context, CL_MEM_READ_WRITE, num_elements * sizeof(cl_int), 0);
        test_assert_error(newWrapper != nullptr && newBuffer != nullptr,
                          "clSVMAlloc failed for modified execution");

        error = clEnqueueSVMMemcpy(queue, CL_TRUE, newWrapper, &newBuffer,
                                   sizeof(cl_int *), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMemcpy failed for newWrapper");

        error = clEnqueueSVMMemFill(queue, newBuffer, &zero, sizeof(zero),
                                    num_elements * sizeof(cl_int), 0, nullptr,
                                    nullptr);
        test_error(error, "clEnqueueSVMMemFill failed for newB");

        // Build and execute the command buffer for the initial execution

        error = clSetKernelArgSVMPointer(kernel, 0, initWrapper);
        test_error(error, "clSetKernelArg failed for initWrapper");

        error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                    sizeof(initBuffer), &initBuffer);
        test_error(error, "clSetKernelExecInfo failed for initBuffer");

        cl_ndrange_kernel_command_properties_khr props[] = {
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

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Check the results of the initial execution

        error =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, initBuffer,
                            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMap failed for initBuffer");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (initBuffer[i] != 1)
            {
                log_error("Initial verification failed at index %zu: Got %d, "
                          "wanted 1\n",
                          i, initBuffer[i]);
                return TEST_FAIL;
            }
        }

        error = clEnqueueSVMUnmap(queue, initBuffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMUnmap failed for initBuffer");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Modify and execute the command buffer

        cl_mutable_dispatch_arg_khr arg_svm{};
        arg_svm.arg_index = 0;
        arg_svm.arg_value = newWrapper;

        cl_mutable_dispatch_exec_info_khr exec_info{};
        exec_info.param_name = CL_KERNEL_EXEC_INFO_SVM_PTRS;
        exec_info.param_value_size = sizeof(newBuffer);
        exec_info.param_value = &newBuffer;

        cl_mutable_dispatch_config_khr dispatch_config{};
        dispatch_config.type = CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR;
        dispatch_config.command = command;
        dispatch_config.num_svm_args = 1;
        dispatch_config.arg_svm_list = &arg_svm;
        dispatch_config.num_exec_infos = 1;
        dispatch_config.exec_info_list = &exec_info;

        cl_mutable_base_config_khr mutable_config{};
        mutable_config.type = CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR;
        mutable_config.num_mutable_dispatch = 1;
        mutable_config.mutable_dispatch_list = &dispatch_config;

        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        // Check the results of the modified execution

        error =
            clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, newBuffer,
                            num_elements * sizeof(cl_int), 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMMap failed for newBuffer");

        for (size_t i = 0; i < num_elements; i++)
        {
            if (newBuffer[i] != 1)
            {
                log_error("Modified verification failed at index %zu: Got %d, "
                          "wanted 1\n",
                          i, newBuffer[i]);
                return TEST_FAIL;
            }
        }

        error = clEnqueueSVMUnmap(queue, newBuffer, 0, nullptr, nullptr);
        test_error(error, "clEnqueueSVMUnmap failed for newBuffer");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        // Clean up

        clSVMFree(context, initWrapper);
        clSVMFree(context, initBuffer);
        clSVMFree(context, newWrapper);
        clSVMFree(context, newBuffer);

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
};


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