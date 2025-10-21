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
#include "harness/errorHelpers.h"
#include <android/hardware_buffer.h>
#include "debug_ahb.h"

REGISTER_TEST(test_buffer_format_negative)
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
    aHardwareBufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    aHardwareBufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    aHardwareBufferDesc.width = 64;
    aHardwareBufferDesc.height = 1;
    aHardwareBufferDesc.layers = 1;
    aHardwareBufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;

    if (!AHardwareBuffer_isSupported(&aHardwareBufferDesc))
    {
        const std::string usage_string =
            ahardwareBufferDecodeUsageFlagsToString(
                static_cast<AHardwareBuffer_UsageFlags>(
                    aHardwareBufferDesc.usage));
        log_info(
            "Unsupported format %s, usage flags %s\n",
            ahardwareBufferFormatToString(
                static_cast<AHardwareBuffer_Format>(aHardwareBufferDesc.format))
                .c_str(),
            usage_string.c_str());
        return TEST_SKIPPED_ITSELF;
    }

    AHardwareBuffer *aHardwareBuffer = nullptr;
    const int ahb_result =
        AHardwareBuffer_allocate(&aHardwareBufferDesc, &aHardwareBuffer);
    if (ahb_result != 0)
    {
        log_error("AHardwareBuffer_allocate failed with code %d\n", ahb_result);
        return TEST_FAIL;
    }
    log_info("Testing %s\n",
             ahardwareBufferFormatToString(static_cast<AHardwareBuffer_Format>(
                                               aHardwareBufferDesc.format))
                 .c_str());

    cl_mem_properties props[] = {
        CL_EXTERNAL_MEMORY_HANDLE_ANDROID_HARDWARE_BUFFER_KHR,
        reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
    };

    cl_mem buffer = clCreateBufferWithProperties(
        context, props, CL_MEM_READ_WRITE, 0, nullptr, &err);
    test_assert_error(err == CL_INVALID_OPERATION,
                      "To create a buffer the aHardwareFormat must be "
                      "AHARDWAREBUFFER_FORMAT_BLOB");

    if (buffer != nullptr)
    {
        test_error(clReleaseMemObject(buffer), "Failed to release buffer");
    }

    AHardwareBuffer_release(aHardwareBuffer);
    aHardwareBuffer = nullptr;

    return TEST_PASS;
}

REGISTER_TEST(test_buffer_size_negative)
{
    cl_int err = CL_SUCCESS;
    constexpr size_t buffer_size = 64;

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
    aHardwareBufferDesc.width = buffer_size;
    aHardwareBufferDesc.height = 1;
    aHardwareBufferDesc.layers = 1;
    aHardwareBufferDesc.usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;

    if (!AHardwareBuffer_isSupported(&aHardwareBufferDesc))
    {
        const std::string usage_string =
            ahardwareBufferDecodeUsageFlagsToString(
                static_cast<AHardwareBuffer_UsageFlags>(
                    aHardwareBufferDesc.usage));
        log_info(
            "Unsupported format %s, usage flags %s\n",
            ahardwareBufferFormatToString(
                static_cast<AHardwareBuffer_Format>(aHardwareBufferDesc.format))
                .c_str(),
            usage_string.c_str());
        return TEST_SKIPPED_ITSELF;
    }

    AHardwareBuffer *aHardwareBuffer = nullptr;
    const int ahb_result =
        AHardwareBuffer_allocate(&aHardwareBufferDesc, &aHardwareBuffer);
    if (ahb_result != 0)
    {
        log_error("AHardwareBuffer_allocate failed with code %d\n", ahb_result);
        return TEST_FAIL;
    }
    log_info("Testing %s\n",
             ahardwareBufferFormatToString(static_cast<AHardwareBuffer_Format>(
                                               aHardwareBufferDesc.format))
                 .c_str());

    cl_mem_properties props[] = {
        CL_EXTERNAL_MEMORY_HANDLE_ANDROID_HARDWARE_BUFFER_KHR,
        reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
    };

    cl_mem buffer = clCreateBufferWithProperties(
        context, props, CL_MEM_READ_WRITE, buffer_size / 2, nullptr, &err);
    test_assert_error(err == CL_INVALID_BUFFER_SIZE,
                      "Wrong error value returned");

    if (buffer != nullptr)
    {
        test_error(clReleaseMemObject(buffer), "Failed to release buffer");
    }

    AHardwareBuffer_release(aHardwareBuffer);
    aHardwareBuffer = nullptr;

    return TEST_PASS;
}

REGISTER_TEST(test_images_negative)
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
    aHardwareBufferDesc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    aHardwareBufferDesc.usage = static_cast<AHardwareBuffer_UsageFlags>(
        AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
        | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN
        | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
        | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER);
    aHardwareBufferDesc.width = 64;
    aHardwareBufferDesc.height = 64;
    aHardwareBufferDesc.layers = 1;

    AHardwareBuffer *aHardwareBuffer = nullptr;
    int ahb_result =
        AHardwareBuffer_allocate(&aHardwareBufferDesc, &aHardwareBuffer);
    if (ahb_result != 0)
    {
        log_error("AHardwareBuffer_allocate failed with code %d\n", ahb_result);
        return TEST_FAIL;
    }

    const cl_mem_properties props[] = {
        CL_EXTERNAL_MEMORY_HANDLE_ANDROID_HARDWARE_BUFFER_KHR,
        reinterpret_cast<cl_mem_properties>(aHardwareBuffer), 0
    };

    constexpr cl_image_format image_format = { CL_RGBA, CL_UNORM_INT8 };
    cl_mem image =
        clCreateImageWithProperties(context, props, CL_MEM_READ_WRITE,
                                    &image_format, nullptr, nullptr, &err);
    test_assert_error(err == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                      "Wrong error value returned");
    if (image != nullptr)
    {
        test_error(clReleaseMemObject(image), "Failed to release image");
    }

    constexpr cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, 64, 64 };
    image = clCreateImageWithProperties(context, props, CL_MEM_READ_WRITE,
                                        nullptr, &image_desc, nullptr, &err);
    test_assert_error(err == CL_INVALID_IMAGE_DESCRIPTOR,
                      "Wrong error value returned");
    if (image != nullptr)
    {
        test_error(clReleaseMemObject(image), "Failed to release image");
    }
    AHardwareBuffer_release(aHardwareBuffer);
    aHardwareBuffer = nullptr;

    return TEST_PASS;
}
