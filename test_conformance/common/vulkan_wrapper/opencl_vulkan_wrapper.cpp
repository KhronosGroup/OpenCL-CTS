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
#include <CL/cl_ext.h>
#include "opencl_vulkan_wrapper.hpp"
#include "vulkan_wrapper.hpp"
#include "harness/errorHelpers.h"
#include "harness/deviceInfo.h"
#include <assert.h>
#include <algorithm>
#include <stdexcept>

#define ASSERT(x) assert((x))
#define GB(x) ((unsigned long long)(x) << 30)

pfnclCreateSemaphoreWithPropertiesKHR clCreateSemaphoreWithPropertiesKHRptr;
pfnclEnqueueWaitSemaphoresKHR clEnqueueWaitSemaphoresKHRptr;
pfnclEnqueueSignalSemaphoresKHR clEnqueueSignalSemaphoresKHRptr;
pfnclEnqueueAcquireExternalMemObjectsKHR
    clEnqueueAcquireExternalMemObjectsKHRptr;
pfnclEnqueueReleaseExternalMemObjectsKHR
    clEnqueueReleaseExternalMemObjectsKHRptr;
pfnclReleaseSemaphoreKHR clReleaseSemaphoreKHRptr;
pfnclGetSemaphoreHandleForTypeKHR clGetSemaphoreHandleForTypeKHRptr;
pfnclReImportSemaphoreSyncFdKHR clReImportSemaphoreSyncFdKHRptr;

void init_cl_vk_ext(cl_platform_id opencl_platform, cl_uint num_devices,
                    cl_device_id *deviceIds)
{
    clEnqueueWaitSemaphoresKHRptr =
        (pfnclEnqueueWaitSemaphoresKHR)clGetExtensionFunctionAddressForPlatform(
            opencl_platform, "clEnqueueWaitSemaphoresKHR");
    if (NULL == clEnqueueWaitSemaphoresKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clEnqueueWaitSemaphoresKHRptr!");
    }
    clEnqueueSignalSemaphoresKHRptr = (pfnclEnqueueSignalSemaphoresKHR)
        clGetExtensionFunctionAddressForPlatform(
            opencl_platform, "clEnqueueSignalSemaphoresKHR");
    if (NULL == clEnqueueSignalSemaphoresKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clEnqueueSignalSemaphoresKHRptr!");
    }
    clReleaseSemaphoreKHRptr =
        (pfnclReleaseSemaphoreKHR)clGetExtensionFunctionAddressForPlatform(
            opencl_platform, "clReleaseSemaphoreKHR");
    if (NULL == clReleaseSemaphoreKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clReleaseSemaphoreKHRptr!");
    }
    clCreateSemaphoreWithPropertiesKHRptr =
        (pfnclCreateSemaphoreWithPropertiesKHR)
            clGetExtensionFunctionAddressForPlatform(
                opencl_platform, "clCreateSemaphoreWithPropertiesKHR");
    if (NULL == clCreateSemaphoreWithPropertiesKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clCreateSemaphoreWithPropertiesKHRptr!");
    }

    clGetSemaphoreHandleForTypeKHRptr = (pfnclGetSemaphoreHandleForTypeKHR)
        clGetExtensionFunctionAddressForPlatform(
            opencl_platform, "clGetSemaphoreHandleForTypeKHR");
    if (NULL == clGetSemaphoreHandleForTypeKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clGetSemaphoreHandleForTypeKHRptr!");
    }

    // Required only if cl_khr_external_semaphore_sync_fd is reported
    clReImportSemaphoreSyncFdKHRptr = (pfnclReImportSemaphoreSyncFdKHR)
        clGetExtensionFunctionAddressForPlatform(
            opencl_platform, "clReImportSemaphoreSyncFdKHR");
    for (cl_uint i = 0; i < num_devices; i++)
    {
        if (is_extension_available(deviceIds[i],
                                   "cl_khr_external_semaphore_sync_fd")
            && (NULL == clReImportSemaphoreSyncFdKHRptr))
        {
            throw std::runtime_error("Failed to get the function pointer of "
                                     "clReImportSemaphoreSyncFdKHR!");
        }
    }

    clEnqueueAcquireExternalMemObjectsKHRptr =
        (pfnclEnqueueAcquireExternalMemObjectsKHR)
            clGetExtensionFunctionAddressForPlatform(
                opencl_platform, "clEnqueueAcquireExternalMemObjectsKHR");
    if (nullptr == clEnqueueAcquireExternalMemObjectsKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clEnqueueAcquireExternalMemObjectsKHR!");
    }

    clEnqueueReleaseExternalMemObjectsKHRptr =
        (pfnclEnqueueReleaseExternalMemObjectsKHR)
            clGetExtensionFunctionAddressForPlatform(
                opencl_platform, "clEnqueueReleaseExternalMemObjectsKHR");
    if (nullptr == clEnqueueReleaseExternalMemObjectsKHRptr)
    {
        throw std::runtime_error("Failed to get the function pointer of "
                                 "clEnqueueReleaseExternalMemObjectsKHR!");
    }
}

cl_int setMaxImageDimensions(cl_device_id deviceID, size_t &max_width,
                             size_t &max_height)
{
    cl_int result = CL_SUCCESS;
    cl_ulong val;
    size_t paramSize;

    result = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE,
                             sizeof(cl_ulong), &val, &paramSize);

    if (result != CL_SUCCESS)
    {
        return result;
    }

    if (val < GB(4))
    {
        max_width = 256;
        max_height = 256;
    }
    else if (val < GB(8))
    {
        max_width = 512;
        max_height = 256;
    }
    else
    {
        max_width = 1024;
        max_height = 512;
    }

    return result;
}

cl_int getCLFormatFromVkFormat(VkFormat vkFormat,
                               cl_image_format *clImageFormat)
{
    cl_int result = CL_SUCCESS;
    switch (vkFormat)
    {
        case VK_FORMAT_R8G8B8A8_UNORM:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNORM_INT8;
            break;
        case VK_FORMAT_B8G8R8A8_UNORM:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_UNORM_INT8;
            break;
        case VK_FORMAT_R16G16B16A16_UNORM:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNORM_INT16;
            break;
        case VK_FORMAT_R8G8B8A8_SINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT8;
            break;
        case VK_FORMAT_R16G16B16A16_SINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT16;
            break;
        case VK_FORMAT_R32G32B32A32_SINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT32;
            break;
        case VK_FORMAT_R8G8B8A8_UINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT8;
            break;
        case VK_FORMAT_R16G16B16A16_UINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT16;
            break;
        case VK_FORMAT_R32G32B32A32_UINT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
            break;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_HALF_FLOAT;
            break;
        case VK_FORMAT_R32G32B32A32_SFLOAT:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_FLOAT;
            break;
        case VK_FORMAT_R8_SNORM:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_SNORM_INT8;
            break;
        case VK_FORMAT_R16_SNORM:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_SNORM_INT16;
            break;
        case VK_FORMAT_R8_UNORM:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_UNORM_INT8;
            break;
        case VK_FORMAT_R16_UNORM:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_UNORM_INT16;
            break;
        case VK_FORMAT_R8_SINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT8;
            break;
        case VK_FORMAT_R16_SINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT16;
            break;
        case VK_FORMAT_R32_SINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT32;
            break;
        case VK_FORMAT_R8_UINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT8;
            break;
        case VK_FORMAT_R16_UINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT16;
            break;
        case VK_FORMAT_R32_UINT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
            break;
        case VK_FORMAT_R16_SFLOAT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_HALF_FLOAT;
            break;
        case VK_FORMAT_R32_SFLOAT:
            clImageFormat->image_channel_order = CL_R;
            clImageFormat->image_channel_data_type = CL_FLOAT;
            break;
        case VK_FORMAT_R8G8_SNORM:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_SNORM_INT8;
            break;
        case VK_FORMAT_R16G16_SNORM:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_SNORM_INT16;
            break;
        case VK_FORMAT_R8G8_UNORM:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_UNORM_INT8;
            break;
        case VK_FORMAT_R16G16_UNORM:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_UNORM_INT16;
            break;
        case VK_FORMAT_R8G8_SINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT8;
            break;
        case VK_FORMAT_R16G16_SINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT16;
            break;
        case VK_FORMAT_R32G32_SINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT32;
            break;
        case VK_FORMAT_R8G8_UINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT8;
            break;
        case VK_FORMAT_R16G16_UINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT16;
            break;
        case VK_FORMAT_R32G32_UINT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT32;
            break;
        case VK_FORMAT_R16G16_SFLOAT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_HALF_FLOAT;
            break;
        case VK_FORMAT_R32G32_SFLOAT:
            clImageFormat->image_channel_order = CL_RG;
            clImageFormat->image_channel_data_type = CL_FLOAT;
            break;
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNORM_SHORT_565;
            break;
        case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_UNORM_SHORT_555;
            break;
        case VK_FORMAT_R8G8B8A8_SNORM:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_SNORM_INT8;
            break;
        case VK_FORMAT_R16G16B16A16_SNORM:
            clImageFormat->image_channel_order = CL_RGBA;
            clImageFormat->image_channel_data_type = CL_SNORM_INT16;
            break;
        case VK_FORMAT_B8G8R8A8_SNORM:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_SNORM_INT8;
            break;
        case VK_FORMAT_B5G6R5_UNORM_PACK16:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_UNORM_SHORT_565;
            break;
        case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_UNORM_SHORT_555;
            break;
        case VK_FORMAT_B8G8R8A8_SINT:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_SIGNED_INT8;
            break;
        case VK_FORMAT_B8G8R8A8_UINT:
            clImageFormat->image_channel_order = CL_BGRA;
            clImageFormat->image_channel_data_type = CL_UNSIGNED_INT8;
            break;
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32: result = CL_INVALID_VALUE; break;
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32: result = CL_INVALID_VALUE; break;
        case VK_FORMAT_A8B8G8R8_SINT_PACK32: result = CL_INVALID_VALUE; break;
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: result = CL_INVALID_VALUE; break;
        default:
            log_error("Unsupported format\n");
            ASSERT(0);
            break;
    }
    return result;
}

cl_mem_object_type getImageTypeFromVk(VkImageType imageType)
{
    cl_mem_object_type cl_image_type = CL_INVALID_VALUE;
    switch (imageType)
    {
        case VK_IMAGE_TYPE_1D: cl_image_type = CL_MEM_OBJECT_IMAGE1D; break;
        case VK_IMAGE_TYPE_2D: cl_image_type = CL_MEM_OBJECT_IMAGE2D; break;
        case VK_IMAGE_TYPE_3D: cl_image_type = CL_MEM_OBJECT_IMAGE3D; break;
        default: break;
    }
    return cl_image_type;
}

size_t GetElementNBytes(const cl_image_format *format)
{
    size_t result;

    switch (format->image_channel_order)
    {
        case CL_R:
        case CL_A:
        case CL_INTENSITY:
        case CL_LUMINANCE:
        case CL_DEPTH: result = 1; break;
        case CL_RG:
        case CL_RA: result = 2; break;
        case CL_RGB: result = 3; break;
        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
        case CL_sRGBA: result = 4; break;
        default: result = 0; break;
    }

    switch (format->image_channel_data_type)
    {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
            // result *= 1;
            break;

        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_HALF_FLOAT: result *= 2; break;

        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
        case CL_FLOAT: result *= 4; break;

        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
            if (result == 3)
            {
                result = 2;
            }
            else
            {
                result = 0;
            }
            break;

        case CL_UNORM_INT_101010:
            if (result == 3)
            {
                result = 4;
            }
            else
            {
                result = 0;
            }
            break;

        default: result = 0; break;
    }

    return result;
}

cl_int get2DImageDimensions(const VkImageCreateInfo *VulkanImageCreateInfo,
                            cl_image_format *img_fmt, size_t totalImageSize,
                            size_t &width, size_t &height)
{
    cl_int result = CL_SUCCESS;
    if (totalImageSize == 0)
    {
        result = CL_INVALID_VALUE;
    }
    size_t element_size = GetElementNBytes(img_fmt);
    size_t row_pitch = element_size * VulkanImageCreateInfo->extent.width;
    row_pitch = row_pitch % 64 == 0 ? row_pitch : ((row_pitch / 64) + 1) * 64;

    width = row_pitch / element_size;
    height = totalImageSize / row_pitch;

    return result;
}

cl_int
getCLImageInfoFromVkImageInfo(const VkImageCreateInfo *VulkanImageCreateInfo,
                              size_t totalImageSize, cl_image_format *img_fmt,
                              cl_image_desc *img_desc)
{
    cl_int result = CL_SUCCESS;

    cl_image_format clImgFormat = { 0 };
    result =
        getCLFormatFromVkFormat(VulkanImageCreateInfo->format, &clImgFormat);
    if (CL_SUCCESS != result)
    {
        return result;
    }
    memcpy(img_fmt, &clImgFormat, sizeof(cl_image_format));

    img_desc->image_type = getImageTypeFromVk(VulkanImageCreateInfo->imageType);
    if (CL_INVALID_VALUE == static_cast<cl_int>(img_desc->image_type))
    {
        return CL_INVALID_VALUE;
    }

    result =
        get2DImageDimensions(VulkanImageCreateInfo, img_fmt, totalImageSize,
                             img_desc->image_width, img_desc->image_height);
    if (CL_SUCCESS != result)
    {
        throw std::runtime_error("get2DImageDimensions failed!!!");
    }

    img_desc->image_depth =
        static_cast<size_t>(VulkanImageCreateInfo->extent.depth);
    img_desc->image_array_size = 0;
    img_desc->image_row_pitch = 0; // Row pitch set to zero as host_ptr is NULL
    img_desc->image_slice_pitch =
        img_desc->image_row_pitch * img_desc->image_height;
    img_desc->num_mip_levels = 0;
    img_desc->num_samples = 0;
    img_desc->buffer = NULL;

    return result;
}

cl_int check_external_memory_handle_type(
    cl_device_id deviceID,
    cl_external_memory_handle_type_khr requiredHandleType)
{
    unsigned int i;
    cl_external_memory_handle_type_khr *handle_type;
    size_t handle_type_size = 0;

    cl_int errNum = CL_SUCCESS;

    errNum = clGetDeviceInfo(deviceID,
                             CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR,
                             0, NULL, &handle_type_size);
    test_error(errNum, "clGetDeviceInfo failed");

    handle_type =
        (cl_external_memory_handle_type_khr *)malloc(handle_type_size);

    errNum = clGetDeviceInfo(deviceID,
                             CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR,
                             handle_type_size, handle_type, NULL);

    test_error(
        errNum,
        "Unable to query CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR \n");

    for (i = 0; i < handle_type_size; i++)
    {
        if (requiredHandleType == handle_type[i])
        {
            return CL_SUCCESS;
        }
    }
    log_error("cl_khr_external_memory extension is missing support for %d\n",
              requiredHandleType);

    return CL_INVALID_VALUE;
}

cl_int check_external_semaphore_handle_type(
    cl_device_id deviceID,
    cl_external_semaphore_handle_type_khr requiredHandleType,
    cl_device_info queryParamName)
{
    unsigned int i;
    cl_external_semaphore_handle_type_khr *handle_type;
    size_t handle_type_size = 0;
    cl_int errNum = CL_SUCCESS;

    errNum =
        clGetDeviceInfo(deviceID, queryParamName, 0, NULL, &handle_type_size);
    test_error(errNum, "clGetDeviceInfo failed");

    if (handle_type_size == 0)
    {
        log_error("Device does not support %s semaphore\n",
                  queryParamName == CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR
                      ? "importing"
                      : "exporting");
        return CL_INVALID_VALUE;
    }

    handle_type =
        (cl_external_semaphore_handle_type_khr *)malloc(handle_type_size);

    errNum = clGetDeviceInfo(deviceID, queryParamName, handle_type_size,
                             handle_type, NULL);

    test_error(
        errNum,
        "Unable to query supported device semaphore handle types list\n");

    for (i = 0; i < handle_type_size; i++)
    {
        if (requiredHandleType == handle_type[i])
        {
            return CL_SUCCESS;
        }
    }
    log_error("cl_khr_external_semaphore extension is missing support for %d\n",
              requiredHandleType);

    return CL_INVALID_VALUE;
}
clExternalMemory::clExternalMemory() {}

clExternalMemory::clExternalMemory(const clExternalMemory &externalMemory)
    : m_externalMemory(externalMemory.m_externalMemory)
{}

clExternalMemory::clExternalMemory(
    const VulkanDeviceMemory *deviceMemory,
    VulkanExternalMemoryHandleType externalMemoryHandleType, uint64_t size,
    cl_context context, cl_device_id deviceId)
{
    int err = 0;
    m_externalMemory = NULL;
    cl_device_id devList[] = { deviceId, NULL };
    std::vector<cl_mem_properties> extMemProperties;
#ifdef _WIN32
    if (!is_extension_available(devList[0], "cl_khr_external_memory_win32"))
    {
        throw std::runtime_error(
            "Device does not support cl_khr_external_memory_win32 extension\n");
    }
#else
    if (!is_extension_available(devList[0], "cl_khr_external_memory_opaque_fd"))
    {
        throw std::runtime_error(
            "Device does not support cl_khr_external_memory_opaque_fd "
            "extension \n");
    }
#endif

    switch (externalMemoryHandleType)
    {
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
#ifdef _WIN32
            log_info("Opaque file descriptors are not supported on Windows\n");
            ASSERT(0);
#else
            fd = (int)deviceMemory->getHandle(externalMemoryHandleType);
            err = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
            extMemProperties.push_back(
                (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
            extMemProperties.push_back((cl_mem_properties)fd);
#endif
            break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
#ifndef _WIN32
            log_info("Opaque NT handles are only supported on Windows\n");
            ASSERT(0);
#else
            handle = deviceMemory->getHandle(externalMemoryHandleType);
            err = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties.push_back(
                (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties.push_back((cl_mem_properties)handle);
#endif
            break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT_NAME: {
#ifndef _WIN32
            log_info("Opaque NT handles are only supported on Windows\n");
            ASSERT(0);
#else
            const std::wstring &name = deviceMemory->getName();
            if (name.size())
            {
                err = check_external_memory_handle_type(
                    devList[0],
                    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR);
                extMemProperties.push_back(
                    (cl_mem_properties)
                        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR);
                extMemProperties.push_back((cl_mem_properties)name.c_str());
            }
            else
            {
                throw std::runtime_error("Unsupported operation: import via "
                                         "name but no name provided\n");
            }
#endif
        }
        break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
#ifndef _WIN32
            log_info("Opaque D3DKMT handles are only supported on Windows\n");
            ASSERT(0);
#else
            handle = deviceMemory->getHandle(externalMemoryHandleType);
            err = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties.push_back(
                (cl_mem_properties)
                    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties.push_back((cl_mem_properties)handle);
#endif
            break;
        default:
            log_error("Unsupported external memory handle type\n");
            ASSERT(0);
            break;
    }
    if (CL_SUCCESS != err)
    {
        throw std::runtime_error("Unsupported external memory type\n");
    }

    extMemProperties.push_back(
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_KHR);
    extMemProperties.push_back((cl_mem_properties)devList[0]);
    extMemProperties.push_back(
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_END_KHR);
    extMemProperties.push_back(0);

    m_externalMemory = clCreateBufferWithProperties(
        context, extMemProperties.data(), 1, size, NULL, &err);
    if (CL_SUCCESS != err)
    {
        log_error("clCreateBufferWithProperties failed with %d\n", err);
        throw std::runtime_error("clCreateBufferWithProperties failed\n");
    }
}
clExternalMemoryImage::clExternalMemoryImage(
    const VulkanDeviceMemory &deviceMemory,
    VulkanExternalMemoryHandleType externalMemoryHandleType, cl_context context,
    size_t totalImageMemSize, size_t imageWidth, size_t imageHeight,
    size_t totalSize, const VulkanImage2D &image2D, cl_device_id deviceId)
{
    cl_int errcode_ret = 0;
    std::vector<cl_mem_properties> extMemProperties1;
    cl_device_id devList[] = { deviceId, NULL };

    VulkanImageTiling vulkanImageTiling =
        vkClExternalMemoryHandleTilingAssumption(
            deviceId, externalMemoryHandleType, &errcode_ret);
    if (CL_SUCCESS != errcode_ret)
    {
        throw std::runtime_error("Failed to query OpenCL tiling mode");
    }

#ifdef _WIN32
    if (!is_extension_available(devList[0], "cl_khr_external_memory_win32"))
    {
        throw std::runtime_error("Device does not support "
                                 "cl_khr_external_memory_win32 extension \n");
    }
#elif !defined(__APPLE__)
    if (!is_extension_available(devList[0], "cl_khr_external_memory_opaque_fd"))
    {
        throw std::runtime_error(
            "Device does not support cl_khr_external_memory_opaque_fd "
            "extension\n");
    }
#endif

    switch (externalMemoryHandleType)
    {
#ifdef _WIN32
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
            handle = deviceMemory.getHandle(externalMemoryHandleType);
            errcode_ret = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties1.push_back(
                (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties1.push_back((cl_mem_properties)handle);
            break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT_NAME: {
            const std::wstring &name = deviceMemory.getName();
            if (name.size())
            {
                errcode_ret = check_external_memory_handle_type(
                    devList[0],
                    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR);
                extMemProperties1.push_back(
                    (cl_mem_properties)
                        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_NAME_KHR);
                extMemProperties1.push_back((cl_mem_properties)name.c_str());
            }
            else
            {
                throw std::runtime_error("Unsupported operation: import via "
                                         "name but no name provided\n");
            }
        }
        break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            handle = deviceMemory.getHandle(externalMemoryHandleType);
            errcode_ret = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties1.push_back(
                (cl_mem_properties)
                    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties1.push_back((cl_mem_properties)handle);
            break;
#elif !defined(__APPLE__)
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            fd = (int)deviceMemory.getHandle(externalMemoryHandleType);
            errcode_ret = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
            extMemProperties1.push_back(
                (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
            extMemProperties1.push_back((cl_mem_properties)fd);
            break;
#endif
        default:
            log_error("Unsupported external memory handle type\n");
            ASSERT(0);
            break;
    }
    if (CL_SUCCESS != errcode_ret)
    {
        throw std::runtime_error("Unsupported external memory type\n");
    }

    // Set cl_image_desc
    size_t clImageFormatSize;
    cl_image_desc image_desc;
    memset(&image_desc, 0x0, sizeof(cl_image_desc));
    cl_image_format img_format = { 0 };
    const VkImageCreateInfo VulkanImageCreateInfo =
        image2D.getVkImageCreateInfo();

    errcode_ret = getCLImageInfoFromVkImageInfo(
        &VulkanImageCreateInfo, image2D.getSize(), &img_format, &image_desc);
    if (CL_SUCCESS != errcode_ret)
    {
        throw std::runtime_error("getCLImageInfoFromVkImageInfo failed\n");
    }

    // If OpenCL will assume linear, query the Vulkan image's row pitch,
    // otherwise it may not match OpenCL's assumption of the row pitch.
    if (vulkanImageTiling == VULKAN_IMAGE_TILING_LINEAR)
    {
        VkSubresourceLayout subresourceLayout = image2D.getSubresourceLayout();
        image_desc.image_row_pitch = subresourceLayout.rowPitch;
        image_desc.image_slice_pitch = subresourceLayout.depthPitch;
    }

    extMemProperties1.push_back(
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_KHR);
    extMemProperties1.push_back((cl_mem_properties)devList[0]);
    extMemProperties1.push_back(
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_END_KHR);
    extMemProperties1.push_back(0);
    m_externalMemory = clCreateImageWithProperties(
        context, extMemProperties1.data(), CL_MEM_READ_WRITE, &img_format,
        &image_desc, NULL, &errcode_ret);
    if (CL_SUCCESS != errcode_ret)
    {
        throw std::runtime_error("clCreateImageWithProperties failed\n");
    }
}

cl_mem clExternalMemory::getExternalMemoryBuffer() { return m_externalMemory; }

cl_mem clExternalMemoryImage::getExternalMemoryImage()
{
    return m_externalMemory;
}

clExternalMemoryImage::~clExternalMemoryImage()
{
    clReleaseMemObject(m_externalMemory);
}

clExternalMemory::~clExternalMemory() { clReleaseMemObject(m_externalMemory); }

clExternalMemoryImage::clExternalMemoryImage() {}


//////////////////////////////////////////
// clExternalSemaphore implementation //
//////////////////////////////////////////

clExternalSemaphore::~clExternalSemaphore() = default;

clExternalImportableSemaphore::clExternalImportableSemaphore(
    const VulkanSemaphore &semaphore, cl_context context,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
    cl_device_id deviceId)
    : m_deviceSemaphore(semaphore)
{

    cl_int err = 0;
    cl_device_id devList[] = { deviceId, NULL };
    m_externalHandleType = externalSemaphoreHandleType;
    m_externalSemaphore = nullptr;
    m_device = deviceId;
    m_context = context;

    std::vector<cl_semaphore_properties_khr> sema_props{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
    };
    switch (externalSemaphoreHandleType)
    {
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD:
#ifdef _WIN32
            log_info("Opaque file descriptors are not supported on Windows\n");
            ASSERT(0);
#else
            fd = (int)semaphore.getHandle(externalSemaphoreHandleType);
            err = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
            sema_props.push_back(
                (cl_semaphore_properties_khr)CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
            sema_props.push_back((cl_semaphore_properties_khr)fd);
#endif
            break;
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT:
#ifndef _WIN32
            log_info("Opaque NT handles are only supported on Windows\n");
            ASSERT(0);
#else
            handle = semaphore.getHandle(externalSemaphoreHandleType);
            err = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
            sema_props.push_back((cl_semaphore_properties_khr)
                                     CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
            sema_props.push_back((cl_semaphore_properties_khr)handle);
#endif
            break;
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT_NAME: {
#ifndef _WIN32
            log_info("Opaque NT handles are only supported on Windows\n");
            ASSERT(0);
#else
            const std::wstring &name = semaphore.getName();
            if (name.size())
            {
                err = check_external_semaphore_handle_type(
                    devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_NAME_KHR);
                sema_props.push_back(
                    (cl_semaphore_properties_khr)
                        CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_NAME_KHR);
                sema_props.push_back((cl_semaphore_properties_khr)name.c_str());
            }
            else
            {
                throw std::runtime_error("Unsupported operation: import via "
                                         "name but no name provided\n");
            }
#endif
        }
        break;
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT:
#ifndef _WIN32
            log_info("Opaque D3DKMT handles are only supported on Windows\n");
            ASSERT(0);
#else
            handle = semaphore.getHandle(externalSemaphoreHandleType);
            err = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
            sema_props.push_back((cl_semaphore_properties_khr)
                                     CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
            sema_props.push_back((cl_semaphore_properties_khr)handle);
#endif
            break;
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD:
            err = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_SYNC_FD_KHR);

            sema_props.push_back(static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_HANDLE_SYNC_FD_KHR));
            sema_props.push_back(static_cast<cl_semaphore_properties_khr>(-1));
            break;
        default:
            log_error("Unsupported external memory handle type\n");
            ASSERT(0);
            break;
    }
    if (CL_SUCCESS != err)
    {
        throw std::runtime_error(
            "Unsupported external sempahore handle type\n ");
    }

    sema_props.push_back(
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR);
    sema_props.push_back((cl_semaphore_properties_khr)devList[0]);
    sema_props.push_back(
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR);
    sema_props.push_back(0);
    m_externalSemaphore =
        clCreateSemaphoreWithPropertiesKHRptr(context, sema_props.data(), &err);
    if (CL_SUCCESS != err)
    {
        log_error("clCreateSemaphoreWithPropertiesKHRptr failed with %d\n",
                  err);
        throw std::runtime_error(
            "clCreateSemaphoreWithPropertiesKHRptr failed! ");
    }
}

clExternalImportableSemaphore::~clExternalImportableSemaphore()
{
    cl_int err = clReleaseSemaphoreKHRptr(m_externalSemaphore);
    if (err != CL_SUCCESS)
    {
        log_error("clReleaseSemaphoreKHR failed with %d\n", err);
    }
}

int clExternalImportableSemaphore::wait(cl_command_queue cmd_queue)
{
    int err = CL_SUCCESS;
    if (m_externalHandleType == VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD)
    {
        cl_int err = 0;
        fd = (int)m_deviceSemaphore.getHandle(m_externalHandleType);
        err = clReImportSemaphoreSyncFdKHRptr(m_externalSemaphore, nullptr, fd);
        if (err != CL_SUCCESS)
        {
            return err;
        }
    }

    err = clEnqueueWaitSemaphoresKHRptr(cmd_queue, 1, &m_externalSemaphore,
                                        NULL, 0, NULL, NULL);
    return err;
}

int clExternalImportableSemaphore::signal(cl_command_queue cmd_queue)
{
    return clEnqueueSignalSemaphoresKHRptr(cmd_queue, 1, &m_externalSemaphore,
                                           NULL, 0, NULL, NULL);
}

cl_semaphore_khr &clExternalImportableSemaphore::getCLSemaphore()
{
    return m_externalSemaphore;
}


clExternalExportableSemaphore::clExternalExportableSemaphore(
    const VulkanSemaphore &semaphore, cl_context context,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
    cl_device_id deviceId)
    : m_deviceSemaphore(semaphore)
{

    cl_int err = 0;
    cl_device_id devList[] = { deviceId, NULL };
    cl_external_semaphore_handle_type_khr clSemaphoreHandleType =
        getCLSemaphoreTypeFromVulkanType(externalSemaphoreHandleType);
    m_externalHandleType = externalSemaphoreHandleType;
    m_externalSemaphore = nullptr;
    m_device = deviceId;
    m_context = context;

    std::vector<cl_semaphore_properties_khr> sema_props{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
    };
    sema_props.push_back(
        (cl_semaphore_properties_khr)CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR);
    sema_props.push_back((cl_semaphore_properties_khr)clSemaphoreHandleType);
    sema_props.push_back((cl_semaphore_properties_khr)
                             CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR);
    sema_props.push_back(
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR);
    sema_props.push_back((cl_semaphore_properties_khr)devList[0]);
    sema_props.push_back(
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR);
    sema_props.push_back(0);
    m_externalSemaphore =
        clCreateSemaphoreWithPropertiesKHRptr(context, sema_props.data(), &err);
    if (CL_SUCCESS != err)
    {
        log_error("clCreateSemaphoreWithPropertiesKHRptr failed with %d\n",
                  err);
        throw std::runtime_error(
            "clCreateSemaphoreWithPropertiesKHRptr failed! ");
    }

    switch (m_externalHandleType)
    {
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD: {
            err = clGetSemaphoreHandleForTypeKHRptr(
                m_externalSemaphore, m_device, clSemaphoreHandleType,
                sizeof(int), &fd, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Failed to export OpenCL semaphore\n");
            }

            VkImportSemaphoreFdInfoKHR vkImportSemaphoreFdInfoKHR = {};
            vkImportSemaphoreFdInfoKHR.sType =
                VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
            vkImportSemaphoreFdInfoKHR.semaphore = m_deviceSemaphore;
            vkImportSemaphoreFdInfoKHR.fd = fd;
            vkImportSemaphoreFdInfoKHR.pNext = nullptr;
            vkImportSemaphoreFdInfoKHR.handleType =
                VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
            vkImportSemaphoreFdInfoKHR.flags = 0;

            if (vkImportSemaphoreFdKHR(m_deviceSemaphore.getDevice(),
                                       &vkImportSemaphoreFdInfoKHR)
                != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to import semaphore in Vulkan\n");
            }
            break;
        }
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT:
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT: {
            err = clGetSemaphoreHandleForTypeKHRptr(
                m_externalSemaphore, m_device, clSemaphoreHandleType,
                sizeof(void *), (void *)&handle, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Failed to export OpenCL semaphore\n");
            }

#ifdef _WIN32
            VkImportSemaphoreWin32HandleInfoKHR
                vkImportSemaphoreWin32HandleInfoKHR = {};
            vkImportSemaphoreWin32HandleInfoKHR.sType =
                VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
            vkImportSemaphoreWin32HandleInfoKHR.pNext = nullptr;
            vkImportSemaphoreWin32HandleInfoKHR.semaphore = m_deviceSemaphore;
            vkImportSemaphoreWin32HandleInfoKHR.flags = 0;
            vkImportSemaphoreWin32HandleInfoKHR.handleType =
                (VkExternalSemaphoreHandleTypeFlagBits)m_externalHandleType;
            vkImportSemaphoreWin32HandleInfoKHR.handle = (HANDLE)handle;
            vkImportSemaphoreWin32HandleInfoKHR.name = nullptr;

            if (vkImportSemaphoreWin32HandleKHR(
                    m_deviceSemaphore.getDevice(),
                    &vkImportSemaphoreWin32HandleInfoKHR)
                != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to import semaphore in Vulkan\n");
            }
#else
            log_error(
                "Opaque D3DKMT and NT handles are only supported on Windows\n");
            ASSERT(0);
#endif
            break;
        }
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD:
            // Do nothing, imported after each signal from OpenCL
            break;
        default:
            log_error("Unsupported external semaphore handle type\n");
            ASSERT(0);
            break;
    }
}

clExternalExportableSemaphore::~clExternalExportableSemaphore()
{
    cl_int err = clReleaseSemaphoreKHRptr(m_externalSemaphore);
    if (err != CL_SUCCESS)
    {
        log_error("clReleaseSemaphoreKHR failed with %d\n", err);
    }
}

int clExternalExportableSemaphore::signal(cl_command_queue cmd_queue)
{
    int err = clEnqueueSignalSemaphoresKHRptr(
        cmd_queue, 1, &m_externalSemaphore, NULL, 0, NULL, nullptr);
    if (err != CL_SUCCESS)
    {
        return err;
    }

    if (m_externalHandleType == VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD)
    {
        err = clGetSemaphoreHandleForTypeKHRptr(m_externalSemaphore, m_device,
                                                CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                                                sizeof(int), &fd, nullptr);
        if (err != CL_SUCCESS)
        {
            log_error("Failed to export fd from semaphore\n");
            return err;
        }

        VkImportSemaphoreFdInfoKHR import = {};
        import.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        import.semaphore = m_deviceSemaphore;
        import.fd = fd;
        import.pNext = nullptr;
        import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT_KHR;
        import.flags = 0;

        VkResult res =
            vkImportSemaphoreFdKHR(m_deviceSemaphore.getDevice(), &import);
        ASSERT(res == VK_SUCCESS);
        if (res != VK_SUCCESS)
        {
            err = CL_INVALID_OPERATION;
        }
    }

    return err;
}

int clExternalExportableSemaphore::wait(cl_command_queue command_queue)
{
    return clEnqueueWaitSemaphoresKHRptr(command_queue, 1, &m_externalSemaphore,
                                         NULL, 0, NULL, nullptr);
}

cl_semaphore_khr &clExternalExportableSemaphore::getCLSemaphore()
{
    return m_externalSemaphore;
}

cl_external_memory_handle_type_khr vkToOpenCLExternalMemoryHandleType(
    VulkanExternalMemoryHandleType vkExternalMemoryHandleType)
{
    switch (vkExternalMemoryHandleType)
    {
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            return CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
            return CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT_KMT:
            return CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE: return 0;
    }
    return 0;
}

VulkanImageTiling vkClExternalMemoryHandleTilingAssumption(
    cl_device_id deviceId,
    VulkanExternalMemoryHandleType vkExternalMemoryHandleType, int *error_ret)
{
    size_t size = 0;
    VulkanImageTiling mode = VULKAN_IMAGE_TILING_OPTIMAL;

    assert(error_ret
           != nullptr); // errcode_ret is not optional, it must be checked

    *error_ret = clGetDeviceInfo(
        deviceId,
        CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR,
        0, nullptr, &size);
    if (*error_ret != CL_SUCCESS)
    {
        return mode;
    }

    if (size == 0)
    {
        return mode;
    }

    std::vector<cl_external_memory_handle_type_khr> assume_linear_types(
        size / sizeof(cl_external_memory_handle_type_khr));

    *error_ret = clGetDeviceInfo(
        deviceId,
        CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR,
        size, assume_linear_types.data(), nullptr);
    if (*error_ret != CL_SUCCESS)
    {
        return mode;
    }

    if (std::find(
            assume_linear_types.begin(), assume_linear_types.end(),
            vkToOpenCLExternalMemoryHandleType(vkExternalMemoryHandleType))
        != assume_linear_types.end())
    {
        mode = VULKAN_IMAGE_TILING_LINEAR;
    }

    return mode;
}
