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

#include <vector>
#include "imageHelpers.h"
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

    bool Skip() override
    {
        cl_bool image_support;

        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(image_support), &image_support, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        return (!mutable_support || !image_support)
            || BasicMutableCommandBufferTest::Skip();
    }

    cl_int Run() override
    {
        const char *sample_const_arg_kernel =
            R"(__kernel void sample_test( read_only image1d_t source, sampler_t
            sampler, write_only image1d_t dest)
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

        const cl_image_format formats = { CL_RGBA, CL_UNSIGNED_INT8 };

        image_descriptor imageInfo = { 0 };
        imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
        imageInfo.format = &formats;
        imageInfo.width = 4;
        imageInfo.rowPitch = imageInfo.width * get_pixel_size(imageInfo.format);

        BufferOwningPtr<char> imageValues_input, imageValues_output, outputData;
        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues_input, d);
        generate_random_image_data(&imageInfo, imageValues_output, d);
        generate_random_image_data(&imageInfo, outputData, d);

        char *host_ptr_input = (char *)imageValues_input;
        char *host_ptr_output = (char *)imageValues_output;

        clMemWrapper src_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr_input, nullptr, &error);
        test_error(error, "create_image_1d failed");

        clMemWrapper dst_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr_output, nullptr, &error);
        test_error(error, "create_image_2d failed");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        clSamplerWrapper sampler = clCreateSampler(
            context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
        test_error(error, "Unable to create sampler");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_image);
        test_error(error, "Unable to set indexed kernel arguments");

        cl_command_properties_khr props[] = {
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
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clFinish(queue);
        test_error(error, "clFinish failed.");

        clMemWrapper new_image = create_image_1d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, 0, host_ptr_output, nullptr, &error);
        test_error(error, "create_image_1d failed");

        cl_mutable_dispatch_arg_khr arg_2{ 2, sizeof(cl_mem), &new_image };
        cl_mutable_dispatch_arg_khr args[] = { arg_2 };

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

        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { image_desc.image_width, 1, 1 };

        error = clEnqueueReadImage(queue, new_image, CL_TRUE, origin, region, 0,
                                   0, outputData, 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < imageInfo.width; ++i)
        {
            if (imageValues_input[i] != outputData[i])
            {
                log_error("Data failed to verify: imageValues[%zu]=%d != "
                          "outputData[%zu]=%d\n",
                          i, imageValues_input[i], i, outputData[i]);

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

    bool Skip() override
    {
        cl_bool image_support;

        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(image_support), &image_support, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

        cl_mutable_dispatch_fields_khr mutable_capabilities;

        bool mutable_support =
            !clGetDeviceInfo(
                device, CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                sizeof(mutable_capabilities), &mutable_capabilities, nullptr)
            && mutable_capabilities & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;

        return (!mutable_support || !image_support)
            || BasicMutableCommandBufferTest::Skip();
    }

    cl_int Run() override
    {

        const char *sample_const_arg_kernel =
            R"(__kernel void sample_test( read_only image2d_t source, sampler_t
            sampler, write_only image2d_t dest)
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
        imageInfo.rowPitch = imageInfo.width * get_pixel_size(imageInfo.format);

        BufferOwningPtr<char> imageValues_input, imageValues_output;

        MTdataHolder d(gRandomSeed);
        generate_random_image_data(&imageInfo, imageValues_input, d);
        generate_random_image_data(&imageInfo, imageValues_output, d);

        char *host_ptr_input = (char *)imageValues_input;
        char *host_ptr_output = (char *)imageValues_output;
        std::vector<char> outputData(data_size);

        clMemWrapper src_image =
            create_image_2d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            &formats, image_desc.image_width,
                            image_desc.image_height, 0, host_ptr_input, &error);
        test_error(error, "create_image_2d failed");

        clMemWrapper dst_image = create_image_2d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, image_desc.image_height, 0, host_ptr_output,
            &error);
        test_error(error, "create_image_2d failed");

        error = create_single_kernel_helper(context, &program, &kernel, 1,
                                            &sample_const_arg_kernel,
                                            "sample_test");
        test_error(error, "Creating kernel failed");

        clSamplerWrapper sampler = clCreateSampler(
            context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
        test_error(error, "Unable to create sampler");

        error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_image);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 1, sizeof(cl_sampler), &sampler);
        test_error(error, "Unable to set indexed kernel arguments");

        error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_image);
        test_error(error, "Unable to set indexed kernel arguments");

        size_t globalDim[3] = { 4, 4, 1 }, localDim[3] = { 1, 1, 1 };

        cl_command_properties_khr props[] = {
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

        clMemWrapper new_image = create_image_2d(
            context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &formats,
            image_desc.image_width, image_desc.image_height, 0,
            imageValues_output, &error);
        test_error(error, "create_image_2d failed");

        cl_mutable_dispatch_arg_khr arg_2{ 2, sizeof(cl_mem), &new_image };
        cl_mutable_dispatch_arg_khr args[] = { arg_2 };

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

        size_t origin[3] = { 0, 0, 0 };
        size_t region[3] = { image_desc.image_width, image_desc.image_height,
                             1 };

        error = clEnqueueReadImage(queue, new_image, CL_TRUE, origin, region, 0,
                                   0, outputData.data(), 0, nullptr, nullptr);
        test_error(error, "clEnqueueReadImage failed");

        for (size_t i = 0; i < imageInfo.width * imageInfo.height; ++i)
        {
            if (imageValues_input[i] != outputData[i])
            {
                log_error("Data failed to verify: imageValues[%zu]=%d != "
                          "outputData[%zu]=%d\n",
                          i, imageValues_input[i], i, outputData[i]);
                return TEST_FAIL;
            }
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
