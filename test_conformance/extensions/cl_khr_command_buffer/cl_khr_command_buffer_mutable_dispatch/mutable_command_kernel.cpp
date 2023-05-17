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
// mutable dispatch tests which handle following cases for
// CL_MUTABLE_DISPATCH_ARGUMENTS_KHR:
// - __global arguments
// - __local arguments
// - plain-old-data arguments
// - NULL arguments
// - image 1d arguments
// - image 2d arguments
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
        const char *sample_const_arg_kernel =
            R"(
            __kernel void sample_test(__constant int *src1, __global int *dst)
            {
                size_t  tid = get_global_id(0);
                dst[tid] = src1[tid];
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

        currentSize = max_size;
        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)currentSize / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (int)genrand_int32(d);

        clMemWrapper streams[2];
        streams[0] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeToAllocate,
                           constantData.data(), &error);
        test_error(error, "Creating test array failed");
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                    NULL, &error);
        test_error(error, "Creating test array failed");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &streams[0]);
        test_error(error, "Unable to set indexed kernel arguments");
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem) * 2, &streams[1]);
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

        clMemWrapper newBuffer;
        newBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                   NULL, &error);
        test_error(error, "Creating buffer failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(cl_mem), &newBuffer };
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
        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error =
            clEnqueueReadBuffer(queue, newBuffer, CL_TRUE, 0, sizeToAllocate,
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
    cl_ulong currentSize;
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

        currentSize = max_size;
        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)currentSize / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (int)genrand_int32(d);

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
        error = clSetKernelArg(kernel, 1, sizeof(cl_mem) * 2, nullptr);
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
    cl_ulong currentSize;
};

struct MutableDispatchPODArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchPODArguments(cl_device_id device, cl_context context,
                                cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        return 0;
    }

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

        currentSize = max_size;
        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)currentSize / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (int)genrand_int32(d);

        clMemWrapper stream;
        stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeToAllocate,
                                constantData.data(), &error);
        test_error(error, "Creating test array failed");


        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");
        int intarg = 10;
        error = clSetKernelArg(kernel, 1, sizeof(int), &intarg);
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
        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(int), &intarg };
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
    cl_ulong currentSize;
};

struct MutableDispatchNullArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchNullArguments(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        return 0;
    }

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(
                __kernel void sample_test(__constant int *src,
            __global int *dst)
            {
                size_t  tid = get_global_id(0);
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

        currentSize = max_size;
        MTdataHolder d(gRandomSeed);

        size_t sizeToAllocate =
            ((size_t)currentSize / sizeof(cl_int)) * sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate / sizeof(cl_int);
        constantData.resize(sizeToAllocate / sizeof(cl_int));
        resultData.resize(sizeToAllocate / sizeof(cl_int));

        for (size_t i = 0; i < numberOfInts; i++)
            constantData[i] = (int)genrand_int32(d);

        clMemWrapper streams[3];
        streams[0] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, sizeToAllocate,
                           constantData.data(), &error);
        test_error(error, "Creating test array failed");
        streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                    nullptr, &error);
        test_error(error, "Creating test array failed");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &streams[0]);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &streams[1]);
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

        streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                    NULL, &error);
        test_error(error, "Creating test array failed");

        clMemWrapper newBuffer;
        newBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeToAllocate,
                                   NULL, &error);
        test_error(error, "Creating buffer failed");

        cl_mutable_dispatch_arg_khr arg_1{ 1, sizeof(streams[2]) * 2, nullptr };
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
    cl_ulong currentSize;
};

struct MutableDispatchImage1DArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchImage1DArguments(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(__kernel void sample_test( read_only image1d_t source, sampler_t
            sampler, __global int4 *results )
            {
               int offset = get_global_id(0);
               results[ offset ] = read_imagei( source, sampler, offset);
            })";

        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;
        size_t threads[1], localThreads[1];

        cl_image_desc image_desc;
        memset(&image_desc, 0x0, sizeof(cl_image_desc));
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        image_desc.image_width = 100;
        image_desc.image_height = 100;
        image_desc.image_row_pitch = 0;
        image_desc.num_mip_levels = 0;

        size_t data_size =
            image_desc.image_width * image_desc.image_height * sizeof(cl_int);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

        image_descriptor imageInfo = { 0 };
        imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
        imageInfo.format = &formats;
        imageInfo.width = 100;
        imageInfo.depth = 0;

        BufferOwningPtr<char> imageValues, outputData;
        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues, d);
        generate_random_image_data(&imageInfo, outputData, d);

        clMemWrapper image = create_image_1d(
            context, CL_MEM_READ_WRITE, &formats, image_desc.image_width,
            image_desc.image_width, 0, nullptr, &error);
        test_error(error, "create_image_2d failed");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        clSamplerWrapper sampler = clCreateSampler(
            context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
        test_error(error, "Unable to create sampler");

        clMemWrapper stream;
        stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, data_size,
                                imageValues, &error);
        test_error(error, "Creating test array failed");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        threads[0] = image_desc.image_width;
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
        test_error(error, "clEnqueueCommandBufferKHR failed")

            clMemWrapper new_image =
                create_image_1d(context, CL_MEM_READ_WRITE, &formats,
                                image_desc.image_width, image_desc.image_width,
                                imageValues, nullptr, &error);
        test_error(error, "create_image_2d failed");

        cl_mutable_dispatch_arg_khr arg_0{ 0, sizeof(cl_mem), &new_image };
        cl_mutable_dispatch_arg_khr args[] = { arg_0 };

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
        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueReadBuffer(queue, stream, CL_TRUE, 0, data_size,
                                    outputData, 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < imageInfo.width; ++i)
        {
            if (imageValues[i] != outputData[i])
            {
                log_error("Data failed to verify: imageValues[%d]=%d != "
                          "outputData[%d]=%d\n",
                          i, imageValues[i], i, outputData[i]);

                return TEST_FAIL;
            }
        }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const cl_ulong max_size = 16;
    cl_ulong currentSize;
};

struct MutableDispatchImage2DArguments : public BasicMutableCommandBufferTest
{
    using BasicMutableCommandBufferTest::BasicMutableCommandBufferTest;

    MutableDispatchImage2DArguments(cl_device_id device, cl_context context,
                                    cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue)
    {}

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        return CL_SUCCESS;
    }

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(__kernel void sample_test( read_only image2d_t source, sampler_t
            sampler, int width, __global int4 *results )
            {
               int x = get_global_id(0);
               int y = get_global_id(1);

               int offset = width * y + x;
               results[ offset ] = read_imagei( source, sampler, (int2) (x, y) );
            })";


        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;
        size_t threads[1], localThreads[1];

        cl_image_desc image_desc;
        memset(&image_desc, 0x0, sizeof(cl_image_desc));
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = 100;
        image_desc.image_height = 100;
        image_desc.image_row_pitch = 0;
        image_desc.num_mip_levels = 0;

        size_t data_size =
            image_desc.image_width * image_desc.image_height * sizeof(cl_int);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

        image_descriptor imageInfo = { 0 };
        imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
        imageInfo.width = 100;
        imageInfo.height = 100;
        imageInfo.format = &formats;
        imageInfo.depth = 0;

        BufferOwningPtr<char> imageValues;
        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues, d);

        size_t image_size = imageInfo.width * imageInfo.height * sizeof(cl_int);
        BufferOwningPtr<int> outputData(malloc(image_size));
        memset(outputData, 0xff, image_size);

        clMemWrapper image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, image_desc.image_width,
            image_desc.image_width, 0, NULL, &error);
        test_error(error, "create_image_2d failed");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        clSamplerWrapper sampler = clCreateSampler(
            context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
        test_error(error, "Unable to create sampler");

        clMemWrapper stream;
        stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, data_size,
                                imageValues, &error);
        test_error(error, "Creating test array failed");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(int), &image_desc.image_width);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        threads[0] = image_desc.image_width;
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

        clMemWrapper new_image = create_image_2d(
            context, CL_MEM_READ_WRITE, &formats, image_desc.image_width,
            image_desc.image_width, 0, imageValues, &error);
        test_error(error, "create_image_2d failed");

        cl_mutable_dispatch_arg_khr arg_0{ 0, sizeof(cl_mem), &new_image };
        cl_mutable_dispatch_arg_khr args[] = { arg_0 };

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
        error = clUpdateMutableCommandsKHR(command_buffer, &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        error = clEnqueueReadBuffer(queue, stream, CL_TRUE, 0, data_size,
                                    outputData, 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        for (size_t i = 0; i < imageInfo.width; ++i)
        {
            if (imageValues[i] != outputData[i])
            {
                log_error("Data failed to verify: imageValues[%d]=%d != "
                          "outputData[%d]=%d\n",
                          i, imageValues[i], i, outputData[i]);
            }
            return TEST_FAIL;
        }

        return TEST_PASS;
    }

    cl_mutable_command_khr command = nullptr;
    const cl_ulong max_size = 16;
    cl_ulong currentSize;
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
        cl_mutable_dispatch_fields_khr mutable_capabilities;

        cl_device_svm_capabilities svm_caps;
        bool svm_capabilities =
            !clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES,
                             sizeof(svm_caps), &svm_caps, NULL)
            && svm_capabilities != 0;

        return !svm_capabilities || BasicMutableCommandBufferTest::Skip();
    }

    virtual cl_int SetUp(int elements) override
    {
        BasicMutableCommandBufferTest::SetUp(elements);

        const char *set_kernel_exec_info_svm_ptrs_kernel =
            R"(struct BufPtrs;

            typedef struct {
                __global int *pA;
                __global int *pB;
                __global int *pC;
            } BufPtrs;
            __kernel void set_kernel_exec_info_test(__global BufPtrs* pBufs)
                size_t i;
               i = get_global_id(0);
                pBufs->pA[i]++;
                pBufs->pB[i]++;
                pBufs->pC[i]++;
            })";

        create_single_kernel_helper(context, &program, &kernel, 1,
                                    &set_kernel_exec_info_svm_ptrs_kernel,
                                    "set_kernel_exec_info_test");

        return 0;
    }

    cl_int Run() override
    {
        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        size_t s = num_elements * sizeof(int);
        BufPtrs *pBuf = (BufPtrs *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                              sizeof(BufPtrs), 0);

        pBuf->pA = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);
        pBuf->pB = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);
        pBuf->pC = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);

        BufPtrs *newBuf = (BufPtrs *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                                sizeof(BufPtrs), 0);

        newBuf->pA = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);
        newBuf->pB = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);
        newBuf->pC = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE, s, 0);

        cl_int error = clSetKernelArgSVMPointer(kernel, 0, pBuf);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                    sizeof(BufPtrs), pBuf);
        test_error(error, "clSetKernelExecInfo failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr,
            &global_work_size, nullptr, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        cl_mutable_dispatch_exec_info_khr exec_info_list{
            CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(BufPtrs), pBuf
        };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            1 /* num_svm_arg */,
            0 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            &exec_info_list /* exec_info_list */,
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

        free(pBuf->pA);
        free(pBuf->pB);
        free(pBuf->pC);
        clSVMFree(context, pBuf);

        free(newBuf->pA);
        free(newBuf->pB);
        free(newBuf->pC);
        clSVMFree(context, newBuf);

        return CL_SUCCESS;
    }

    cl_mutable_command_khr command = nullptr;
    clCommandQueueWrapper queue_1, queue_2;

    typedef struct
    {
        cl_int *pA;
        cl_int *pB;
        cl_int *pC;
    } BufPtrs;
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

int test_mutable_dispatch_image_1d_arguments(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    return MakeAndRunTest<MutableDispatchImage1DArguments>(device, context,
                                                           queue, num_elements);
}

int test_mutable_dispatch_image_2d_arguments(cl_device_id device,
                                             cl_context context,
                                             cl_command_queue queue,
                                             int num_elements)
{
    return MakeAndRunTest<MutableDispatchImage2DArguments>(device, context,
                                                           queue, num_elements);
}

int test_mutable_dispatch_svm_arguments(cl_device_id device, cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return MakeAndRunTest<MutableDispatchSVMArguments>(device, context, queue,
                                                       num_elements);
}