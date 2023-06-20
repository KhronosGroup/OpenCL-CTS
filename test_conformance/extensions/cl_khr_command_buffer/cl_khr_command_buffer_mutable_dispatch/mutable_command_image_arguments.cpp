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
// - image arguments

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
            sampler, int width, write_only image1d_t dest, __global int4
        *results )
            {
               int offset = get_global_id(0);

               int4 color = read_imagei( source, sampler, offset );

               write_imagei( dest, offset, color );
            })";

        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;

        cl_image_desc image_desc;
        memset(&image_desc, 0x0, sizeof(cl_image_desc));
        image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
        image_desc.image_width = 4;
        image_desc.image_row_pitch = 0;
        image_desc.num_mip_levels = 0;

        size_t data_size = image_desc.image_width * sizeof(cl_int);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

        image_descriptor imageInfo = { 0 };
        imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
        imageInfo.format = &formats;
        imageInfo.width = 4;

        BufferOwningPtr<char> imageValues, outputData;
        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues, d);
        generate_random_image_data(&imageInfo, outputData, d);

        char *host_ptr = (char *)imageValues;

        clMemWrapper src_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr, nullptr, &error);
        test_error(error, "create_image_1d failed");

        clMemWrapper dst_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr, nullptr, &error);
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

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(int), &image_desc.image_width);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        size_t globalDim[3] = { 4, 1, 1 }, localDim[3] = { 1, 1, 1 };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, globalDim,
            localDim, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed")

            error = clFinish(queue);
        test_error(error, "clFinish failed.");

        clMemWrapper new_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr, nullptr, &error);
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

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed")

            size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { image_desc.image_width, 1, 1 };

        error = clEnqueueReadImage(queue, new_image, CL_TRUE, origin, region, 0,
                                   0, outputData, 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

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
            sampler, int width, write_only image2d_t dest, __global int4
        *results )
            {
               int x = get_global_id(0);
               int y = get_global_id(1);

               int4 color = read_imagei( source, sampler, (int2) (x, y) );

               write_imagei( dest, (int2) (x, y), color );
            })";

        cl_int error;
        clProgramWrapper program;
        clKernelWrapper kernel;

        cl_image_desc image_desc;
        memset(&image_desc, 0x0, sizeof(cl_image_desc));
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = 4;
        image_desc.image_height = 4;
        image_desc.image_row_pitch = 0;
        image_desc.num_mip_levels = 0;

        size_t data_size =
            image_desc.image_width * image_desc.image_height * sizeof(cl_int);

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

        image_descriptor imageInfo = { 0 };
        imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
        imageInfo.width = 4;
        imageInfo.height = 4;
        imageInfo.format = &formats;

        BufferOwningPtr<char> imageValues;

        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues, d);

        char *host_ptr = (char *)imageValues;
        BufferOwningPtr<char> outputData(malloc(data_size));

        clMemWrapper src_image =
            create_image_2d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            &formats, image_desc.image_width,
                            image_desc.image_height, 0, host_ptr, &error);
        test_error(error, "create_image_2d failed");

        clMemWrapper dst_image =
            create_image_2d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            &formats, image_desc.image_width,
                            image_desc.image_height, 0, host_ptr, &error);
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

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(int), &image_desc.image_width);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &stream);
        test_error(error, "Unable to set indexed kernel arguments");

        size_t globalDim[3] = { 4, 4, 1 }, localDim[3] = { 1, 1, 1 };

        cl_ndrange_kernel_command_properties_khr props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0
        };

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, props, kernel, 1, nullptr, globalDim,
            localDim, 0, nullptr, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        clMemWrapper new_image =
            create_image_2d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            &formats, image_desc.image_width,
                            image_desc.image_height, 0, imageValues, &error);
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

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                          nullptr, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { image_desc.image_width, image_desc.image_height,
                             1 };

        error = clEnqueueReadImage(queue, new_image, CL_TRUE, origin, region, 0,
                                   0, outputData, 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < imageInfo.width * imageInfo.height; ++i)
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
};

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
