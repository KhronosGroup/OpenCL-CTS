//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "harness/compat.h"
#include "harness/kernelHelpers.h"
#include "harness/imageHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"
#include "harness/extensionHelpers.h"
#include <android/hardware_buffer.h>
#include "debug_ahb.h"

static bool isAHBUsageReadable(const AHardwareBuffer_UsageFlags usage)
{
    return (AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE & usage) != 0;
}

struct ahb_format_table
{
    AHardwareBuffer_Format aHardwareBufferFormat;
    cl_image_format clImageFormat;
    cl_mem_object_type clMemObjectType;
};

struct ahb_usage_table
{
    AHardwareBuffer_UsageFlags usageFlags;
};

struct ahb_image_size_table
{
    uint32_t width;
    uint32_t height;
};

ahb_image_size_table test_sizes[] = {
    { 64, 64 }, { 128, 128 }, { 256, 256 }, { 512, 512 }
};

ahb_usage_table test_usages[] = {
    { static_cast<AHardwareBuffer_UsageFlags>(
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
        | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN
        | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
        | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER) },
    { static_cast<AHardwareBuffer_UsageFlags>(
        AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
        | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
        | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN) },
    { static_cast<AHardwareBuffer_UsageFlags>(
        AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER
        | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
        | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN) },
};

ahb_format_table test_formats[] = {
    { AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT,
      { CL_RGBA, CL_HALF_FLOAT },
      CL_MEM_OBJECT_IMAGE2D },
    { AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
      { CL_RGBA, CL_UNORM_INT8 },
      CL_MEM_OBJECT_IMAGE2D },

    { AHARDWAREBUFFER_FORMAT_R8_UNORM,
      { CL_R, CL_UNORM_INT8 },
      CL_MEM_OBJECT_IMAGE2D },
};

static const char *diff_images_kernel_source = {
    R"(
        #define PIXEL_FORMAT %s4
       __kernel void verify_image( read_only image2d_t ahb_image , read_only image2d_t ocl_image, global PIXEL_FORMAT *ocl_pixel, global PIXEL_FORMAT *ahb_pixel)
        {
            int tidX = get_global_id(0);
            int tidY = get_global_id(1);
            int idx = tidY * get_global_size(0) + tidX;

            sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
            PIXEL_FORMAT a = read_image%s(ahb_image, sampler, (int2)( tidX, tidY ) );
            PIXEL_FORMAT o = read_image%s(ocl_image, sampler, (int2)( tidX, tidY ) );
            ahb_pixel[idx] = a;
            ocl_pixel[idx] = o;
        })"
};

// Checks that the inferred image format is correct
REGISTER_TEST(test_images)
{
    cl_int err = CL_SUCCESS;

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    for (const auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());
        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                const cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                cl_mem image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_WRITE, nullptr, nullptr,
                    nullptr, &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                cl_image_format imageFormat = { 0 };
                err = clGetImageInfo(image, CL_IMAGE_FORMAT,
                                     sizeof(cl_image_format), &imageFormat,
                                     nullptr);
                test_error(err, "Failed to query image format");

                if (imageFormat.image_channel_order
                    != format.clImageFormat.image_channel_order)
                {
                    log_error("Expected channel order %d, got %d\n",
                              format.clImageFormat.image_channel_order,
                              imageFormat.image_channel_order);
                    return TEST_FAIL;
                }

                if (imageFormat.image_channel_data_type
                    != format.clImageFormat.image_channel_data_type)
                {
                    log_error("Expected image_channel_data_type %d, got %d\n",
                              format.clImageFormat.image_channel_data_type,
                              imageFormat.image_channel_data_type);
                    return TEST_FAIL;
                }

                test_error(clReleaseMemObject(image),
                           "Failed to release image");
                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_images_read)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                // Populate AHB with random data
                size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = hardware_buffer_desc.stride * pixelSize;
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                memcpy(hardware_buffer_data, srcData, srcBytes);

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                cl_image_desc imageDesc = { 0 };
                imageDesc.image_type = imageInfo.type;
                imageDesc.image_width = imageInfo.width;
                imageDesc.image_height = imageInfo.height;
                imageDesc.image_row_pitch = imageInfo.rowPitch;

                clMemWrapper opencl_image = clCreateImage(
                    context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    imageInfo.format, &imageDesc, srcData, &err);
                test_error(err, "Failed to create CL image");

                ExplicitTypes outputType;
                const char *readFormat;

                if (format.clImageFormat.image_channel_data_type
                    == CL_UNSIGNED_INT8)
                {
                    readFormat = "ui";
                    outputType = kUInt;
                }
                else
                {
                    readFormat = "f";
                    outputType = kFloat;
                }

                size_t verify_buffer_size = imageInfo.width * imageInfo.height
                    * get_explicit_type_size(outputType) * 4;

                clMemWrapper ocl_pixel_buffer =
                    clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   verify_buffer_size, nullptr, &err);
                test_error(err, "Failed to create ocl pixel buffer");

                clMemWrapper ahb_pixel_buffer =
                    clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   verify_buffer_size, nullptr, &err);
                test_error(err, "Failed to crete ahb pixel buffer");

                // Populate kernel
                std::vector<char> programSrc(
                    2 * strlen(diff_images_kernel_source));
                const char *outputTypeName = get_explicit_type_name(outputType);

                sprintf(programSrc.data(), diff_images_kernel_source,
                        outputTypeName, // Read image format 1
                        readFormat, // Read image return type 1
                        readFormat // Read image return type 2
                );
                const char *ptr = programSrc.data();
                clProgramWrapper program;
                clKernelWrapper kernel;
                err = create_single_kernel_helper(context, &program, &kernel, 1,
                                                  &ptr, "verify_image");

                // Set kernel args

                err =
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &imported_image);
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &opencl_image);
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                                     &ocl_pixel_buffer);
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 3, sizeof(cl_mem),
                                     &ahb_pixel_buffer);
                test_error(err, "clSetKernelArg failed");

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjectsKHR failed");

                size_t global_work_size[] = { imageInfo.width,
                                              imageInfo.height };
                err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                             global_work_size, nullptr, 0,
                                             nullptr, nullptr);
                test_error(err, "clEnqueueNDRangeKernel failed");

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &opencl_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjectsKHR failed");

                // Read buffer and verify
                std::vector<char> ocl_verify_data(verify_buffer_size);
                err = clEnqueueReadBuffer(
                    queue, ocl_pixel_buffer, CL_BLOCKING, 0, verify_buffer_size,
                    ocl_verify_data.data(), 0, nullptr, nullptr);
                test_error(err, "clEnqueueReadBuffer failed");

                std::vector<char> ahb_verify_data(verify_buffer_size);
                err = clEnqueueReadBuffer(
                    queue, ahb_pixel_buffer, CL_BLOCKING, 0, verify_buffer_size,
                    ahb_verify_data.data(), 0, nullptr, nullptr);
                test_error(err, "clEnqueueReadBuffer failed");

                for (unsigned row = 0; row < imageInfo.height; row++)
                {
                    for (unsigned col = 0; col < imageInfo.width; col++)
                    {
                        unsigned pixel_index = row * imageInfo.width + col;
                        switch (outputType)
                        {
                            case kFloat: {
                                auto *cl_ptr = reinterpret_cast<cl_float4 *>(
                                    ocl_verify_data.data());
                                auto *ahb_ptr = reinterpret_cast<cl_float4 *>(
                                    ahb_verify_data.data());

                                if ((cl_ptr[pixel_index].s0
                                     != ahb_ptr[pixel_index].s0)
                                    || (cl_ptr[pixel_index].s1
                                        != ahb_ptr[pixel_index].s1)
                                    || (cl_ptr[pixel_index].s2
                                        != ahb_ptr[pixel_index].s2)
                                    || (cl_ptr[pixel_index].s3
                                        != ahb_ptr[pixel_index].s3))
                                {
                                    log_error(
                                        "At coord (%u, %u) expected "
                                        "(%f,%f,%f,%f), got (%f,%f,%f,%f)",
                                        col, row, cl_ptr[pixel_index].s0,
                                        cl_ptr[pixel_index].s1,
                                        cl_ptr[pixel_index].s2,
                                        cl_ptr[pixel_index].s3,
                                        ahb_ptr[pixel_index].s0,
                                        ahb_ptr[pixel_index].s1,
                                        ahb_ptr[pixel_index].s2,
                                        ahb_ptr[pixel_index].s3);

                                    return TEST_FAIL;
                                }
                            }
                            break;
                            case kUInt: {
                                auto *cl_ptr = reinterpret_cast<cl_uint4 *>(
                                    ocl_verify_data.data());
                                auto *ahb_ptr = reinterpret_cast<cl_uint4 *>(
                                    ahb_verify_data.data());

                                if ((cl_ptr[pixel_index].s0
                                     != ahb_ptr[pixel_index].s0)
                                    || (cl_ptr[pixel_index].s1
                                        != ahb_ptr[pixel_index].s1)
                                    || (cl_ptr[pixel_index].s2
                                        != ahb_ptr[pixel_index].s2)
                                    || (cl_ptr[pixel_index].s3
                                        != ahb_ptr[pixel_index].s3))
                                {
                                    log_error(
                                        "At coord (%u, %u) expected "
                                        "(%u,%u,%u,%u), got (%u,%u,%u,%u)",
                                        col, row, cl_ptr[pixel_index].s0,
                                        cl_ptr[pixel_index].s1,
                                        cl_ptr[pixel_index].s2,
                                        cl_ptr[pixel_index].s3,
                                        ahb_ptr[pixel_index].s0,
                                        ahb_ptr[pixel_index].s1,
                                        ahb_ptr[pixel_index].s2,
                                        ahb_ptr[pixel_index].s3);
                                    return TEST_FAIL;
                                }
                            }
                            break;
                            default: test_fail("Unknown output type");
                        }
                    }
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_read_image)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                // Populate AHB with random data
                const size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = hardware_buffer_desc.stride * pixelSize;
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                memcpy(hardware_buffer_data, srcData, srcBytes);

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                const cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjectsKHR failed");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };

                std::vector<char> out_image(srcBytes);
                err = clEnqueueReadImage(queue, imported_image, CL_TRUE, origin,
                                         region, imageInfo.rowPitch, 0,
                                         out_image.data(), 0, nullptr, nullptr);
                test_error(err, "clEnqueueCopyImage failed");

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjectsKHR failed");

                const char *out_image_ptr = out_image.data();
                auto srcData_ptr = static_cast<const char *>(srcData);

                const size_t scanlineSize =
                    imageInfo.width * get_pixel_size(imageInfo.format);

                // Count the number of bytes successfully matched
                size_t total_matched = 0;
                for (size_t line = 0; line < imageInfo.height; line++)
                {

                    if (memcmp(srcData_ptr, out_image_ptr, scanlineSize) != 0)
                    {
                        // Find the first differing pixel
                        const size_t pixel_size =
                            get_pixel_size(imageInfo.format);
                        size_t where = compare_scanlines(
                            &imageInfo, srcData_ptr, out_image_ptr);
                        if (where < imageInfo.width)
                        {
                            print_first_pixel_difference_error(
                                where, srcData_ptr + pixel_size * where,
                                out_image_ptr + pixel_size * where, &imageInfo,
                                line, 1);
                            return TEST_FAIL;
                        }
                    }

                    total_matched += scanlineSize;
                    srcData_ptr += imageInfo.rowPitch;
                    out_image_ptr += imageInfo.rowPitch;
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;

                if (total_matched == 0)
                {
                    test_fail("Zero bytes matched");
                }
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_copy_image)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                // Populate AHB with random data
                size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = resolution.width * pixelSize;
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                memcpy(hardware_buffer_data, srcData, srcBytes);

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                cl_image_desc imageDesc = { 0 };
                imageDesc.image_type = imageInfo.type;
                imageDesc.image_width = imageInfo.width;
                imageDesc.image_height = imageInfo.height;

                clMemWrapper opencl_image =
                    clCreateImage(context, CL_MEM_READ_WRITE, imageInfo.format,
                                  &imageDesc, nullptr, &err);
                test_error(err, "Failed to create CL image");

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjectsKHR failed");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };
                err = clEnqueueCopyImage(queue, imported_image, opencl_image,
                                         origin, origin, region, 0, nullptr,
                                         nullptr);
                test_error(err, "Failed calling clEnqueueCopyImage");

                ExplicitTypes outputType;
                const char *readFormat;

                if (format.clImageFormat.image_channel_data_type
                    == CL_UNSIGNED_INT8)
                {
                    readFormat = "ui";
                    outputType = kUInt;
                }
                else
                {
                    readFormat = "f";
                    outputType = kFloat;
                }

                size_t verify_buffer_size = imageInfo.width * imageInfo.height
                    * get_explicit_type_size(outputType) * 4;

                clMemWrapper ocl_pixel_buffer =
                    clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   verify_buffer_size, nullptr, &err);
                test_error(err, "Failed to create ocl pixel buffer");

                clMemWrapper ahb_pixel_buffer =
                    clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   verify_buffer_size, nullptr, &err);
                test_error(err, "Failed to crete ahb pixel buffer");

                // sprintf the kernel
                std::vector<char> programSrc(
                    2 * strlen(diff_images_kernel_source));
                const char *outputTypeName = get_explicit_type_name(outputType);

                sprintf(programSrc.data(), diff_images_kernel_source,
                        outputTypeName, /*read image format 1 */
                        readFormat, /*read image return type 1 */
                        readFormat /*read image return type 2 */
                );
                const char *ptr = programSrc.data();
                clProgramWrapper program;
                clKernelWrapper kernel;
                err = create_single_kernel_helper(context, &program, &kernel, 1,
                                                  &ptr, "verify_image");

                // set kernel args

                err = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                                     &imported_image); /*imported image */
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 1, sizeof(cl_mem),
                                     &opencl_image); /*image made in opencl*/
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 2, sizeof(cl_mem),
                                     &ocl_pixel_buffer); /*verification buffer*/
                test_error(err, "clSetKernelArg failed");

                err = clSetKernelArg(kernel, 3, sizeof(cl_mem),
                                     &ahb_pixel_buffer); /*verification buffer*/
                test_error(err, "clSetKernelArg failed");

                size_t global_work_size[] = { (imageInfo.width),
                                              (imageInfo.height) };
                err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                             global_work_size, nullptr, 0,
                                             nullptr, nullptr);

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjectsKHR failed");

                // Read buffer and verify
                std::vector<char> ocl_verify_data(verify_buffer_size);
                err = clEnqueueReadBuffer(
                    queue, ocl_pixel_buffer, CL_BLOCKING, 0, verify_buffer_size,
                    ocl_verify_data.data(), 0, nullptr, nullptr);
                test_error(err, "clEnqueueReadBuffer failed");

                std::vector<char> ahb_verify_data(verify_buffer_size);
                err = clEnqueueReadBuffer(
                    queue, ahb_pixel_buffer, CL_BLOCKING, 0, verify_buffer_size,
                    ahb_verify_data.data(), 0, nullptr, nullptr);
                test_error(err, "clEnqueueReadBuffer failed");

                for (unsigned row = 0; row < imageInfo.height; row++)
                {
                    for (unsigned col = 0; col < imageInfo.width; col++)
                    {
                        unsigned pixel_index = row * imageInfo.width + col;
                        switch (outputType)
                        {
                            case kFloat: {
                                auto *cl_ptr = reinterpret_cast<cl_float4 *>(
                                    ocl_verify_data.data());
                                auto *ahb_ptr = reinterpret_cast<cl_float4 *>(
                                    ahb_verify_data.data());

                                if ((cl_ptr[pixel_index].s0
                                     != ahb_ptr[pixel_index].s0)
                                    || (cl_ptr[pixel_index].s1
                                        != ahb_ptr[pixel_index].s1)
                                    || (cl_ptr[pixel_index].s2
                                        != ahb_ptr[pixel_index].s2)
                                    || (cl_ptr[pixel_index].s3
                                        != ahb_ptr[pixel_index].s3))
                                {
                                    printf("At %u\n", pixel_index);
                                    printf("Expected %f,%f,%f,%f\n",
                                           cl_ptr[pixel_index].s0,
                                           cl_ptr[pixel_index].s1,
                                           cl_ptr[pixel_index].s2,
                                           cl_ptr[pixel_index].s3);
                                    printf("Got %f,%f,%f,%f\n",
                                           ahb_ptr[pixel_index].s0,
                                           ahb_ptr[pixel_index].s1,
                                           ahb_ptr[pixel_index].s2,
                                           ahb_ptr[pixel_index].s3);

                                    return TEST_FAIL;
                                }
                            }
                            break;
                            case kUInt: {
                                auto *cl_ptr = reinterpret_cast<cl_uint4 *>(
                                    ocl_verify_data.data());
                                auto *ahb_ptr = reinterpret_cast<cl_uint4 *>(
                                    ahb_verify_data.data());

                                if ((cl_ptr[pixel_index].s0
                                     != ahb_ptr[pixel_index].s0)
                                    || (cl_ptr[pixel_index].s1
                                        != ahb_ptr[pixel_index].s1)
                                    || (cl_ptr[pixel_index].s2
                                        != ahb_ptr[pixel_index].s2)
                                    || (cl_ptr[pixel_index].s3
                                        != ahb_ptr[pixel_index].s3))
                                {
                                    printf("At %u\n", pixel_index);
                                    printf("Expected %u,%u,%u,%u\n",
                                           cl_ptr[pixel_index].s0,
                                           cl_ptr[pixel_index].s1,
                                           cl_ptr[pixel_index].s2,
                                           cl_ptr[pixel_index].s3);
                                    printf("Got %u,%u,%u,%u\n",
                                           ahb_ptr[pixel_index].s0,
                                           ahb_ptr[pixel_index].s1,
                                           ahb_ptr[pixel_index].s2,
                                           ahb_ptr[pixel_index].s3);

                                    return TEST_FAIL;
                                }
                            }
                            break;
                            default: test_fail("Unknown output type");
                        }
                    }
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_copy_image_to_buffer)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                // Populate AHB with random data
                size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = hardware_buffer_desc.stride * pixelSize;
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                memcpy(hardware_buffer_data, srcData, srcBytes);

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                clMemWrapper opencl_buffer = clCreateBuffer(
                    context, CL_MEM_READ_WRITE, srcBytes, nullptr, &err);
                test_error(err, "Failed to create CL buffer");

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjectsKHR failed");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };

                err = clEnqueueCopyImageToBuffer(queue, imported_image,
                                                 opencl_buffer, origin, region,
                                                 0, 0, nullptr, nullptr);
                test_error(
                    err, "Failed to copy imported AHB image to opencl buffer");

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjectsKHR failed");

                std::vector<char> out_buffer(srcBytes);
                err = clEnqueueReadBuffer(queue, opencl_buffer, CL_TRUE, 0,
                                          srcBytes, out_buffer.data(), 0,
                                          nullptr, nullptr);
                test_error(err, "clEnqueueReadBuffer failed");

                char *out_buffer_ptr = out_buffer.data();
                auto srcData_ptr = static_cast<char *>(srcData);

                const size_t scanlineSize =
                    imageInfo.width * get_pixel_size(imageInfo.format);

                // Count the number of bytes successfully matched
                size_t total_matched = 0;
                for (size_t line = 0; line < imageInfo.height; line++)
                {

                    if (memcmp(srcData_ptr, out_buffer_ptr, scanlineSize) != 0)
                    {
                        // Find the first differing pixel
                        const size_t pixel_size =
                            get_pixel_size(imageInfo.format);
                        size_t where = compare_scanlines(
                            &imageInfo, srcData_ptr, out_buffer_ptr);
                        if (where < imageInfo.width)
                        {
                            print_first_pixel_difference_error(
                                where, srcData_ptr + pixel_size * where,
                                out_buffer_ptr + pixel_size * where, &imageInfo,
                                line, 1);
                            return TEST_FAIL;
                        }
                    }

                    total_matched += scanlineSize;
                    srcData_ptr += imageInfo.rowPitch;
                    out_buffer_ptr += scanlineSize;
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;

                if (total_matched == 0)
                {
                    test_fail("Zero bytes matched");
                }
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_copy_buffer_to_image)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                // Generate random data for opencl buffer
                const size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = resolution.width * resolution.height
                    * pixelSize; // data is tightly packed in buffer
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                const size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                clMemWrapper opencl_buffer = clCreateBuffer(
                    context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, srcBytes,
                    srcData, &err);
                test_error(err, "Failed to create CL buffer");

                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_WRITE, nullptr, nullptr,
                    nullptr, &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjects failed");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };

                err = clEnqueueCopyBufferToImage(queue, opencl_buffer,
                                                 imported_image, 0, origin,
                                                 region, 0, nullptr, nullptr);
                test_error(
                    err, "Failed to copy opencl buffer to imported AHB image");

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjects failed");

                clFinish(queue);

                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                auto out_image_ptr = static_cast<char *>(hardware_buffer_data);
                auto srcData_ptr = static_cast<char *>(srcData);

                const size_t scanlineSize =
                    imageInfo.width * get_pixel_size(imageInfo.format);

                // Count the number of bytes successfully matched
                size_t total_matched = 0;
                for (size_t line = 0; line < imageInfo.height; line++)
                {

                    if (memcmp(srcData_ptr, out_image_ptr, scanlineSize) != 0)
                    {
                        // Find the first differing pixel
                        size_t where = compare_scanlines(
                            &imageInfo, srcData_ptr, out_image_ptr);
                        if (where < imageInfo.width)
                        {
                            print_first_pixel_difference_error(
                                where, srcData_ptr + pixelSize * where,
                                out_image_ptr + pixelSize * where, &imageInfo,
                                line, 1);
                            ahb_result = AHardwareBuffer_unlock(aHardwareBuffer,
                                                                nullptr);
                            if (ahb_result != 0)
                            {
                                log_error("AHardwareBuffer_unlock failed with "
                                          "code %d\n",
                                          ahb_result);
                                return TEST_FAIL;
                            }
                            return TEST_FAIL;
                        }
                    }

                    total_matched += scanlineSize;
                    srcData_ptr +=
                        scanlineSize; // image data is tightly packed in buffer
                    out_image_ptr += hardware_buffer_desc.stride * pixelSize;
                }

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;

                if (total_matched == 0)
                {
                    test_fail("Zero bytes matched");
                }
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_write_image)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");


                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                // Generate data to write to image
                const size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = resolution.width * resolution.height
                    * pixelSize; // Data is tightly packed
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                const size_t srcBytes = get_image_size(&imageInfo);
                test_assert_error(srcBytes > 0, "Image cannot have zero size");

                BufferOwningPtr<char> srcData;
                generate_random_image_data(&imageInfo, srcData, seed);

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjects failed");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };

                err = clEnqueueWriteImage(queue, imported_image, CL_TRUE,
                                          origin, region, 0, 0, srcData, 0,
                                          nullptr, nullptr);
                test_error(err, "Failed calling clEnqueueWriteImage");

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clReleaseExternalMemObject failed");

                clFinish(queue);

                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                auto out_image_ptr = static_cast<char *>(hardware_buffer_data);
                auto srcData_ptr = static_cast<char *>(srcData);

                const size_t scanlineSize =
                    imageInfo.width * get_pixel_size(imageInfo.format);

                // Count the number of bytes successfully matched
                size_t total_matched = 0;
                for (size_t line = 0; line < imageInfo.height; line++)
                {

                    if (memcmp(srcData_ptr, out_image_ptr, scanlineSize) != 0)
                    {
                        // Find the first differing pixel
                        const size_t pixel_size =
                            get_pixel_size(imageInfo.format);
                        size_t where = compare_scanlines(
                            &imageInfo, srcData_ptr, out_image_ptr);
                        if (where < imageInfo.width)
                        {
                            print_first_pixel_difference_error(
                                where, srcData_ptr + pixel_size * where,
                                out_image_ptr + pixel_size * where, &imageInfo,
                                line, 1);

                            ahb_result = AHardwareBuffer_unlock(aHardwareBuffer,
                                                                nullptr);
                            if (ahb_result != 0)
                            {
                                log_error("AHardwareBuffer_unlock failed with "
                                          "code %d\n",
                                          ahb_result);
                                return TEST_FAIL;
                            }
                            return TEST_FAIL;
                        }
                    }

                    total_matched += scanlineSize;
                    srcData_ptr += scanlineSize; // Data is tightly packed
                    out_image_ptr += hardware_buffer_desc.stride * pixelSize;
                }

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;

                if (total_matched == 0)
                {
                    test_fail("Zero bytes matched");
                }
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_enqueue_fill_image)
{
    cl_int err = CL_SUCCESS;
    RandomSeed seed(gRandomSeed);

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    GET_PFN(device, clEnqueueAcquireExternalMemObjectsKHR);
    GET_PFN(device, clEnqueueReleaseExternalMemObjectsKHR);

    for (auto format : test_formats)
    {
        log_info("Testing %s\n",
                 ahardwareBufferFormatToString(format.aHardwareBufferFormat)
                     .c_str());

        AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
        aHardwareBufferDesc.format = format.aHardwareBufferFormat;
        for (auto usage : test_usages)
        {
            // Filter out usage flags that are not readable on device
            if (!isAHBUsageReadable(usage.usageFlags))
            {
                continue;
            }

            aHardwareBufferDesc.usage = usage.usageFlags;
            for (auto resolution : test_sizes)
            {
                aHardwareBufferDesc.width = resolution.width;
                aHardwareBufferDesc.height = resolution.height;
                aHardwareBufferDesc.layers = 1;

                CHECK_AHARDWARE_BUFFER_SUPPORT(aHardwareBufferDesc, format);

                AHardwareBuffer *aHardwareBuffer = nullptr;
                int ahb_result = AHardwareBuffer_allocate(&aHardwareBufferDesc,
                                                          &aHardwareBuffer);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_allocate failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                // Determine AHB memory layout
                AHardwareBuffer_Desc hardware_buffer_desc = {};
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                test_assert_error(hardware_buffer_desc.width
                                      == resolution.width,
                                  "AHB has unexpected width");
                test_assert_error(hardware_buffer_desc.height
                                      == resolution.height,
                                  "AHB has unexpected height");

                cl_mem_properties props[] = {
                    CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
                    reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
                };

                clMemWrapper imported_image = clCreateImageWithProperties(
                    context, props, CL_MEM_READ_ONLY, nullptr, nullptr, nullptr,
                    &err);
                test_error(err,
                           "Failed to create CL image from AHardwareBuffer");

                // Create image info struct
                size_t pixelSize = get_pixel_size(&format.clImageFormat);
                image_descriptor imageInfo = { 0 };
                imageInfo.format = &format.clImageFormat;
                imageInfo.type = format.clMemObjectType;
                imageInfo.width = resolution.width;
                imageInfo.height = resolution.height;
                imageInfo.rowPitch = resolution.width * resolution.height
                    * pixelSize; // Data is tightly packed
                test_assert_error(imageInfo.rowPitch
                                      >= pixelSize * imageInfo.width,
                                  "Row pitch is smaller than width");

                size_t origin[] = { 0, 0, 0 };
                size_t region[] = { imageInfo.width, imageInfo.height, 1 };

                auto verificationValue = static_cast<char *>(malloc(pixelSize));
                if (!verificationValue)
                {
                    log_error(
                        "Unable to malloc %zu bytes for verificationValue",
                        pixelSize);
                    return TEST_FAIL;
                }

                err = clEnqueueAcquireExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueAcquireExternalMemObjects failed");

                // Generate pixel color and fill image
                switch (format.clImageFormat.image_channel_data_type)
                {
                    case CL_HALF_FLOAT:
                        DetectFloatToHalfRoundingMode(
                            queue); // Intentional drop-through
                    case CL_UNORM_INT8: {
                        auto pattern_decimal =
                            static_cast<float>(genrand_real1(seed));
                        cl_float fillColor[4] = { pattern_decimal,
                                                  pattern_decimal,
                                                  pattern_decimal,
                                                  pattern_decimal };

                        err = clEnqueueFillImage(queue, imported_image,
                                                 fillColor, origin, region, 0,
                                                 nullptr, nullptr);
                        test_error(err, "Failed calling clEnqueueFillImage");

                        pack_image_pixel(fillColor, &format.clImageFormat,
                                         verificationValue);
                        break;
                    }
                    case CL_UNSIGNED_INT16: {
                        const cl_uint pattern_whole = genrand_int32(seed);
                        cl_uint fillColor[4] = { pattern_whole, pattern_whole,
                                                 pattern_whole, pattern_whole };

                        err = clEnqueueFillImage(queue, imported_image,
                                                 fillColor, origin, region, 0,
                                                 nullptr, nullptr);
                        test_error(err, "Failed calling clEnqueueFillImage");

                        pack_image_pixel(fillColor, &format.clImageFormat,
                                         verificationValue);
                        break;
                    }
                    default:
                        log_info("Unsupported image channel data type");
                        continue;
                }

                err = clEnqueueReleaseExternalMemObjectsKHR(
                    queue, 1, &imported_image, 0, nullptr, nullptr);
                test_error(err, "clEnqueueReleaseExternalMemObjects failed");

                clFinish(queue);
                AHardwareBuffer_describe(aHardwareBuffer,
                                         &hardware_buffer_desc);

                void *hardware_buffer_data = nullptr;
                ahb_result = AHardwareBuffer_lock(
                    aHardwareBuffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1,
                    nullptr, &hardware_buffer_data);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_lock failed with code %d\n",
                              ahb_result);
                    return TEST_FAIL;
                }

                auto out_image_ptr = static_cast<char *>(hardware_buffer_data);
                const size_t scanlineSize = imageInfo.width * pixelSize;


                auto verificationLine =
                    static_cast<char *>(malloc(pixelSize * scanlineSize));
                if (!verificationLine)
                {
                    free(verificationValue);
                    log_error("Unable to malloc %zu bytes for verificationLine",
                              pixelSize * scanlineSize);
                    return TEST_FAIL;
                }
                char *index = verificationLine;
                for (size_t x = 0; x < imageInfo.width; x++)
                {
                    memcpy(index, verificationValue, pixelSize);
                    index += pixelSize;
                }

                free(verificationValue);

                // Count the number of bytes successfully matched
                size_t total_matched = 0;
                for (size_t line = 0; line < imageInfo.height; line++)
                {

                    if (memcmp(verificationLine, out_image_ptr, scanlineSize)
                        != 0)
                    {
                        // Find the first differing pixel
                        const size_t pixel_size =
                            get_pixel_size(imageInfo.format);
                        size_t where = compare_scanlines(
                            &imageInfo, verificationLine, out_image_ptr);
                        if (where < imageInfo.width)
                        {
                            print_first_pixel_difference_error(
                                where, verificationLine + pixel_size * where,
                                out_image_ptr + pixel_size * where, &imageInfo,
                                line, 1);

                            ahb_result = AHardwareBuffer_unlock(aHardwareBuffer,
                                                                nullptr);
                            if (ahb_result != 0)
                            {
                                log_error("AHardwareBuffer_unlock failed with "
                                          "code %d\n",
                                          ahb_result);
                                free(verificationLine);
                                return TEST_FAIL;
                            }
                            free(verificationLine);
                            return TEST_FAIL;
                        }
                    }

                    total_matched += scanlineSize;
                    out_image_ptr += hardware_buffer_desc.stride * pixelSize;
                }

                ahb_result = AHardwareBuffer_unlock(aHardwareBuffer, nullptr);
                if (ahb_result != 0)
                {
                    log_error("AHardwareBuffer_unlock failed with code %d\n",
                              ahb_result);
                    free(verificationLine);
                    return TEST_FAIL;
                }

                AHardwareBuffer_release(aHardwareBuffer);
                aHardwareBuffer = nullptr;
                free(verificationLine);

                if (total_matched == 0)
                {
                    test_fail("Zero bytes matched");
                }
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(test_blob)
{
    cl_int err = CL_SUCCESS;

    if (!is_extension_available(device, "cl_khr_external_memory"))
    {
        log_info("cl_khr_external_memory is not supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }
    if (!is_extension_available(
            device, "cl_khr_external_memory_android_hardware_buffer"))
    {
        log_info("cl_khr_external_memory_android_hardware_buffer is not "
                 "supported on this platform. "
                 "Skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    AHardwareBuffer_Desc aHardwareBufferDesc = { 0 };
    aHardwareBufferDesc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    aHardwareBufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;

    log_info("Testing %s\n",
             ahardwareBufferFormatToString(static_cast<AHardwareBuffer_Format>(
                                               aHardwareBufferDesc.format))
                 .c_str());

    for (auto resolution : test_sizes)
    {
        aHardwareBufferDesc.width = resolution.width * resolution.height;
        aHardwareBufferDesc.height = 1;
        aHardwareBufferDesc.layers = 1;
        aHardwareBufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;

        if (!AHardwareBuffer_isSupported(&aHardwareBufferDesc))
        {
            std::string usage_string = ahardwareBufferDecodeUsageFlagsToString(
                static_cast<AHardwareBuffer_UsageFlags>(
                    aHardwareBufferDesc.usage));
            log_info("Unsupported format %s, usage flags %s\n",
                     ahardwareBufferFormatToString(
                         static_cast<AHardwareBuffer_Format>(
                             aHardwareBufferDesc.format))
                         .c_str(),
                     usage_string.c_str());
            continue;
        }

        AHardwareBuffer *aHardwareBuffer = nullptr;
        int ahb_result =
            AHardwareBuffer_allocate(&aHardwareBufferDesc, &aHardwareBuffer);
        if (ahb_result != 0)
        {
            log_error("AHardwareBuffer_allocate failed with code %d\n",
                      ahb_result);
            return TEST_FAIL;
        }

        cl_mem_properties props[] = {
            CL_EXTERNAL_MEMORY_HANDLE_AHB_KHR,
            reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
        };

        cl_mem buffer = clCreateBufferWithProperties(
            context, props, CL_MEM_READ_WRITE, 0, nullptr, &err);
        test_error(err, "Failed to create CL buffer from AHardwareBuffer");

        test_error(clReleaseMemObject(buffer), "Failed to release buffer");
        AHardwareBuffer_release(aHardwareBuffer);
        aHardwareBuffer = nullptr;
    }

    return TEST_PASS;
}
