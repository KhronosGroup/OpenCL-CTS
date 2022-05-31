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

#include "vulkan_utility.hpp"
#include "vulkan_wrapper.hpp"
#include <assert.h>
#include <iostream>
#include <set>
#include <string>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#if defined(_WIN32) || defined(_WIN64)
#include <versionhelpers.h>
#endif
#define ASSERT(x) assert((x))
#define BUFFERSIZE 3000


const VulkanInstance &getVulkanInstance()
{
    static VulkanInstance instance;
    return instance;
}

const VulkanPhysicalDevice &getVulkanPhysicalDevice()
{
    size_t pdIdx;
    cl_int errNum = 0;
    cl_platform_id platform = NULL;
    cl_uchar uuid[CL_UUID_SIZE_KHR];
    cl_device_id *devices;
    char *extensions = NULL;
    size_t extensionSize = 0;
    cl_uint num_devices = 0;
    cl_uint device_no = 0;
    const size_t bufsize = BUFFERSIZE;
    char buf[BUFFERSIZE];
    const VulkanInstance &instance = getVulkanInstance();
    const VulkanPhysicalDeviceList &physicalDeviceList =
        instance.getPhysicalDeviceList();

    // get the platform ID
    errNum = clGetPlatformIDs(1, &platform, NULL);
    if (errNum != CL_SUCCESS)
    {
        printf("Error: Failed to get platform\n");
        throw std::runtime_error("Error: Failed to get number of platform\n");
    }

    errNum =
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (CL_SUCCESS != errNum)
    {
        throw std::runtime_error(
            "Error: clGetDeviceIDs failed in returning of devices\n");
    }
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    if (NULL == devices)
    {
        throw std::runtime_error(
            "Error: Unable to allocate memory for devices\n");
    }
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices,
                            NULL);
    if (CL_SUCCESS != errNum)
    {
        throw std::runtime_error("Error: Failed to get deviceID.\n");
    }
    bool is_selected = false;
    for (device_no = 0; device_no < num_devices; device_no++)
    {
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS, 0,
                                 NULL, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            throw std::runtime_error("Error in clGetDeviceInfo for getting "
                                     "device_extension size....\n");
        }
        extensions = (char *)malloc(extensionSize);
        if (NULL == extensions)
        {
            throw std::runtime_error(
                "Unable to allocate memory for extensions\n");
        }
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS,
                                 extensionSize, extensions, NULL);
        if (CL_SUCCESS != errNum)
        {
            throw std::runtime_error("Error: Error in clGetDeviceInfo for "
                                     "getting device_extension\n");
        }
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_UUID_KHR,
                                 CL_UUID_SIZE_KHR, uuid, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            throw std::runtime_error(
                "Error: clGetDeviceInfo failed with error\n");
        }
        free(extensions);
        for (pdIdx = 0; pdIdx < physicalDeviceList.size(); pdIdx++)
        {
            if (!memcmp(&uuid, physicalDeviceList[pdIdx].getUUID(),
                        VK_UUID_SIZE))
            {
                std::cout << "Selected physical device = "
                          << physicalDeviceList[pdIdx] << std::endl;
                is_selected = true;
                break;
            }
        }
        if (is_selected)
        {
            break;
        }
    }

    if ((pdIdx >= physicalDeviceList.size())
        || (physicalDeviceList[pdIdx] == (VkPhysicalDevice)VK_NULL_HANDLE))
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
    std::cout << "Selected physical device is: " << physicalDeviceList[pdIdx]
              << std::endl;
    return physicalDeviceList[pdIdx];
}

const VulkanQueueFamily &getVulkanQueueFamily(uint32_t queueFlags)
{
    size_t qfIdx;
    const VulkanPhysicalDevice &physicalDevice = getVulkanPhysicalDevice();
    const VulkanQueueFamilyList &queueFamilyList =
        physicalDevice.getQueueFamilyList();

    for (qfIdx = 0; qfIdx < queueFamilyList.size(); qfIdx++)
    {
        if ((queueFamilyList[qfIdx].getQueueFlags() & queueFlags) == queueFlags)
        {
            break;
        }
    }

    return queueFamilyList[qfIdx];
}

const VulkanMemoryType &
getVulkanMemoryType(const VulkanDevice &device,
                    VulkanMemoryTypeProperty memoryTypeProperty)
{
    size_t mtIdx;
    const VulkanMemoryTypeList &memoryTypeList =
        device.getPhysicalDevice().getMemoryTypeList();

    for (mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
    {
        if ((memoryTypeList[mtIdx].getMemoryTypeProperty() & memoryTypeProperty)
            == memoryTypeProperty)
        {
            break;
        }
    }

    // CHECK_LT(mtIdx, memoryTypeList.size());
    return memoryTypeList[mtIdx];
}

bool checkVkSupport()
{
    bool result = true;
    const VulkanInstance &instance = getVulkanInstance();
    const VulkanPhysicalDeviceList &physicalDeviceList =
        instance.getPhysicalDeviceList();
    if (physicalDeviceList == NULL)
    {
        std::cout << "physicalDeviceList is null, No GPUs found with "
                     "Vulkan support !!!\n";
        result = false;
    }
    return result;
}

const VulkanQueueFamilyList &getEmptyVulkanQueueFamilyList()
{
    static VulkanQueueFamilyList queueFamilyList;
    return queueFamilyList;
}

const VulkanDescriptorSetLayoutList &getEmptyVulkanDescriptorSetLayoutList()
{
    static VulkanDescriptorSetLayoutList descriptorSetLayoutList;

    return descriptorSetLayoutList;
}

const VulkanQueueFamilyToQueueCountMap &
getDefaultVulkanQueueFamilyToQueueCountMap()
{
    static VulkanQueueFamilyToQueueCountMap queueFamilyToQueueCountMap(1);

    return queueFamilyToQueueCountMap;
}

const std::vector<VulkanExternalMemoryHandleType>
getSupportedVulkanExternalMemoryHandleTypeList()
{
    std::vector<VulkanExternalMemoryHandleType> externalMemoryHandleTypeList;

#if _WIN32
    if (IsWindows8OrGreater())
    {
        externalMemoryHandleTypeList.push_back(
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT);
    }
    externalMemoryHandleTypeList.push_back(
        VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT);
#else
    externalMemoryHandleTypeList.push_back(
        VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD);
#endif

    return externalMemoryHandleTypeList;
}

const std::vector<VulkanExternalSemaphoreHandleType>
getSupportedVulkanExternalSemaphoreHandleTypeList()
{
    std::vector<VulkanExternalSemaphoreHandleType>
        externalSemaphoreHandleTypeList;

#if _WIN32
    if (IsWindows8OrGreater())
    {
        externalSemaphoreHandleTypeList.push_back(
            VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT);
    }
    externalSemaphoreHandleTypeList.push_back(
        VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT);
#else
    externalSemaphoreHandleTypeList.push_back(
        VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD);
#endif

    return externalSemaphoreHandleTypeList;
}

const std::vector<VulkanFormat> getSupportedVulkanFormatList()
{
    std::vector<VulkanFormat> formatList;

    formatList.push_back(VULKAN_FORMAT_R8_UINT);
    formatList.push_back(VULKAN_FORMAT_R8_SINT);
    formatList.push_back(VULKAN_FORMAT_R8G8_UINT);
    formatList.push_back(VULKAN_FORMAT_R8G8_SINT);
    formatList.push_back(VULKAN_FORMAT_R8G8B8A8_UINT);
    formatList.push_back(VULKAN_FORMAT_R8G8B8A8_SINT);
    formatList.push_back(VULKAN_FORMAT_R16_UINT);
    formatList.push_back(VULKAN_FORMAT_R16_SINT);
    formatList.push_back(VULKAN_FORMAT_R16G16_UINT);
    formatList.push_back(VULKAN_FORMAT_R16G16_SINT);
    formatList.push_back(VULKAN_FORMAT_R16G16B16A16_UINT);
    formatList.push_back(VULKAN_FORMAT_R16G16B16A16_SINT);
    formatList.push_back(VULKAN_FORMAT_R32_UINT);
    formatList.push_back(VULKAN_FORMAT_R32_SINT);
    formatList.push_back(VULKAN_FORMAT_R32_SFLOAT);
    formatList.push_back(VULKAN_FORMAT_R32G32_UINT);
    formatList.push_back(VULKAN_FORMAT_R32G32_SINT);
    formatList.push_back(VULKAN_FORMAT_R32G32_SFLOAT);
    formatList.push_back(VULKAN_FORMAT_R32G32B32A32_UINT);
    formatList.push_back(VULKAN_FORMAT_R32G32B32A32_SINT);
    formatList.push_back(VULKAN_FORMAT_R32G32B32A32_SFLOAT);

    for (size_t fIdx = 0; fIdx < formatList.size(); fIdx++)
    {
        switch (formatList[fIdx])
        {
            case VULKAN_FORMAT_R8_UINT:
            case VULKAN_FORMAT_R8_SINT:
            case VULKAN_FORMAT_R8G8_UINT:
            case VULKAN_FORMAT_R8G8_SINT:
            case VULKAN_FORMAT_R8G8B8A8_UINT:
            case VULKAN_FORMAT_R8G8B8A8_SINT:
            case VULKAN_FORMAT_R16_UINT:
            case VULKAN_FORMAT_R16_SINT:
            case VULKAN_FORMAT_R16G16_UINT:
            case VULKAN_FORMAT_R16G16_SINT:
            case VULKAN_FORMAT_R16G16B16A16_UINT:
            case VULKAN_FORMAT_R16G16B16A16_SINT:
            case VULKAN_FORMAT_R32_UINT:
            case VULKAN_FORMAT_R32_SINT:
            case VULKAN_FORMAT_R32_SFLOAT:
            case VULKAN_FORMAT_R32G32_UINT:
            case VULKAN_FORMAT_R32G32_SINT:
            case VULKAN_FORMAT_R32G32_SFLOAT:
            case VULKAN_FORMAT_R32G32B32A32_UINT:
            case VULKAN_FORMAT_R32G32B32A32_SINT:
            case VULKAN_FORMAT_R32G32B32A32_SFLOAT: break;

            case VULKAN_FORMAT_UNDEFINED:
            case VULKAN_FORMAT_R4G4_UNORM_PACK8:
            case VULKAN_FORMAT_R4G4B4A4_UNORM_PACK16:
            case VULKAN_FORMAT_B4G4R4A4_UNORM_PACK16:
            case VULKAN_FORMAT_R5G6B5_UNORM_PACK16:
            case VULKAN_FORMAT_B5G6R5_UNORM_PACK16:
            case VULKAN_FORMAT_R5G5B5A1_UNORM_PACK16:
            case VULKAN_FORMAT_B5G5R5A1_UNORM_PACK16:
            case VULKAN_FORMAT_A1R5G5B5_UNORM_PACK16:
            case VULKAN_FORMAT_R8_UNORM:
            case VULKAN_FORMAT_R8_SNORM:
            case VULKAN_FORMAT_R8_USCALED:
            case VULKAN_FORMAT_R8_SSCALED:
            case VULKAN_FORMAT_R8_SRGB:
            case VULKAN_FORMAT_R8G8_SNORM:
            case VULKAN_FORMAT_R8G8_UNORM:
            case VULKAN_FORMAT_R8G8_USCALED:
            case VULKAN_FORMAT_R8G8_SSCALED:
            case VULKAN_FORMAT_R8G8_SRGB:
            case VULKAN_FORMAT_R8G8B8_UNORM:
            case VULKAN_FORMAT_R8G8B8_SNORM:
            case VULKAN_FORMAT_R8G8B8_USCALED:
            case VULKAN_FORMAT_R8G8B8_SSCALED:
            case VULKAN_FORMAT_R8G8B8_UINT:
            case VULKAN_FORMAT_R8G8B8_SINT:
            case VULKAN_FORMAT_R8G8B8_SRGB:
            case VULKAN_FORMAT_B8G8R8_UNORM:
            case VULKAN_FORMAT_B8G8R8_SNORM:
            case VULKAN_FORMAT_B8G8R8_USCALED:
            case VULKAN_FORMAT_B8G8R8_SSCALED:
            case VULKAN_FORMAT_B8G8R8_UINT:
            case VULKAN_FORMAT_B8G8R8_SINT:
            case VULKAN_FORMAT_B8G8R8_SRGB:
            case VULKAN_FORMAT_R8G8B8A8_UNORM:
            case VULKAN_FORMAT_R8G8B8A8_SNORM:
            case VULKAN_FORMAT_R8G8B8A8_USCALED:
            case VULKAN_FORMAT_R8G8B8A8_SSCALED:
            case VULKAN_FORMAT_R8G8B8A8_SRGB:
            case VULKAN_FORMAT_B8G8R8A8_UNORM:
            case VULKAN_FORMAT_B8G8R8A8_SNORM:
            case VULKAN_FORMAT_B8G8R8A8_USCALED:
            case VULKAN_FORMAT_B8G8R8A8_SSCALED:
            case VULKAN_FORMAT_B8G8R8A8_UINT:
            case VULKAN_FORMAT_B8G8R8A8_SINT:
            case VULKAN_FORMAT_B8G8R8A8_SRGB:
            case VULKAN_FORMAT_A8B8G8R8_UNORM_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_SNORM_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_USCALED_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_SSCALED_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_UINT_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_SINT_PACK32:
            case VULKAN_FORMAT_A8B8G8R8_SRGB_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_UNORM_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_SNORM_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_USCALED_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_SSCALED_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_UINT_PACK32:
            case VULKAN_FORMAT_A2R10G10B10_SINT_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_UNORM_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_SNORM_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_USCALED_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_SSCALED_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_UINT_PACK32:
            case VULKAN_FORMAT_A2B10G10R10_SINT_PACK32:
            case VULKAN_FORMAT_R16_UNORM:
            case VULKAN_FORMAT_R16_SNORM:
            case VULKAN_FORMAT_R16_USCALED:
            case VULKAN_FORMAT_R16_SSCALED:
            case VULKAN_FORMAT_R16_SFLOAT:
            case VULKAN_FORMAT_R16G16_UNORM:
            case VULKAN_FORMAT_R16G16_SNORM:
            case VULKAN_FORMAT_R16G16_USCALED:
            case VULKAN_FORMAT_R16G16_SSCALED:
            case VULKAN_FORMAT_R16G16_SFLOAT:
            case VULKAN_FORMAT_R16G16B16_UNORM:
            case VULKAN_FORMAT_R16G16B16_SNORM:
            case VULKAN_FORMAT_R16G16B16_USCALED:
            case VULKAN_FORMAT_R16G16B16_SSCALED:
            case VULKAN_FORMAT_R16G16B16_UINT:
            case VULKAN_FORMAT_R16G16B16_SINT:
            case VULKAN_FORMAT_R16G16B16_SFLOAT:
            case VULKAN_FORMAT_R16G16B16A16_UNORM:
            case VULKAN_FORMAT_R16G16B16A16_SNORM:
            case VULKAN_FORMAT_R16G16B16A16_USCALED:
            case VULKAN_FORMAT_R16G16B16A16_SSCALED:
            case VULKAN_FORMAT_R16G16B16A16_SFLOAT:
            case VULKAN_FORMAT_R32G32B32_UINT:
            case VULKAN_FORMAT_R32G32B32_SINT:
            case VULKAN_FORMAT_R32G32B32_SFLOAT:
            case VULKAN_FORMAT_R64_UINT:
            case VULKAN_FORMAT_R64_SINT:
            case VULKAN_FORMAT_R64_SFLOAT:
            case VULKAN_FORMAT_R64G64_UINT:
            case VULKAN_FORMAT_R64G64_SINT:
            case VULKAN_FORMAT_R64G64_SFLOAT:
            case VULKAN_FORMAT_R64G64B64_UINT:
            case VULKAN_FORMAT_R64G64B64_SINT:
            case VULKAN_FORMAT_R64G64B64_SFLOAT:
            case VULKAN_FORMAT_R64G64B64A64_UINT:
            case VULKAN_FORMAT_R64G64B64A64_SINT:
            case VULKAN_FORMAT_R64G64B64A64_SFLOAT:
            case VULKAN_FORMAT_B10G11R11_UFLOAT_PACK32:
            case VULKAN_FORMAT_E5B9G9R9_UFLOAT_PACK32:
            case VULKAN_FORMAT_D16_UNORM:
            case VULKAN_FORMAT_X8_D24_UNORM_PACK32:
            case VULKAN_FORMAT_D32_SFLOAT:
            case VULKAN_FORMAT_S8_UINT:
            case VULKAN_FORMAT_D16_UNORM_S8_UINT:
            case VULKAN_FORMAT_D24_UNORM_S8_UINT:
            case VULKAN_FORMAT_D32_SFLOAT_S8_UINT:
            case VULKAN_FORMAT_BC1_RGB_UNORM_BLOCK:
            case VULKAN_FORMAT_BC1_RGB_SRGB_BLOCK:
            case VULKAN_FORMAT_BC1_RGBA_UNORM_BLOCK:
            case VULKAN_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VULKAN_FORMAT_BC2_UNORM_BLOCK:
            case VULKAN_FORMAT_BC2_SRGB_BLOCK:
            case VULKAN_FORMAT_BC3_UNORM_BLOCK:
            case VULKAN_FORMAT_BC3_SRGB_BLOCK:
            case VULKAN_FORMAT_BC4_UNORM_BLOCK:
            case VULKAN_FORMAT_BC4_SNORM_BLOCK:
            case VULKAN_FORMAT_BC5_UNORM_BLOCK:
            case VULKAN_FORMAT_BC5_SNORM_BLOCK:
            case VULKAN_FORMAT_BC6H_UFLOAT_BLOCK:
            case VULKAN_FORMAT_BC6H_SFLOAT_BLOCK:
            case VULKAN_FORMAT_BC7_UNORM_BLOCK:
            case VULKAN_FORMAT_BC7_SRGB_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
            case VULKAN_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
            case VULKAN_FORMAT_EAC_R11_UNORM_BLOCK:
            case VULKAN_FORMAT_EAC_R11_SNORM_BLOCK:
            case VULKAN_FORMAT_EAC_R11G11_UNORM_BLOCK:
            case VULKAN_FORMAT_EAC_R11G11_SNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_4x4_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_4x4_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_5x4_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_5x4_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_5x5_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_5x5_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_6x5_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_6x5_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_6x6_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_6x6_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_8x5_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_8x5_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_8x6_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_8x6_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_8x8_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_8x8_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_10x5_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_10x5_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_10x6_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_10x6_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_10x8_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_10x8_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_10x10_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_10x10_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_12x10_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_12x10_SRGB_BLOCK:
            case VULKAN_FORMAT_ASTC_12x12_UNORM_BLOCK:
            case VULKAN_FORMAT_ASTC_12x12_SRGB_BLOCK:
                ASSERT(0);
                std::cout << "Unsupport texture format";
        }
    }

    return formatList;
}

uint32_t getVulkanFormatElementSize(VulkanFormat format)
{
    switch (format)
    {
        case VULKAN_FORMAT_R8_UINT: return uint32_t(1);
        case VULKAN_FORMAT_R8_SINT: return uint32_t(1);
        case VULKAN_FORMAT_R8G8_UINT: return uint32_t(2);
        case VULKAN_FORMAT_R8G8_SINT: return uint32_t(2);
        case VULKAN_FORMAT_R8G8B8A8_UINT: return uint32_t(4);
        case VULKAN_FORMAT_R8G8B8A8_SINT: return uint32_t(4);
        case VULKAN_FORMAT_R16_UINT: return uint32_t(2);
        case VULKAN_FORMAT_R16_SINT: return uint32_t(2);
        case VULKAN_FORMAT_R16G16_UINT: return uint32_t(4);
        case VULKAN_FORMAT_R16G16_SINT: return uint32_t(4);
        case VULKAN_FORMAT_R16G16B16A16_UINT: return uint32_t(8);
        case VULKAN_FORMAT_R16G16B16A16_SINT: return uint32_t(8);
        case VULKAN_FORMAT_R32_UINT: return uint32_t(4);
        case VULKAN_FORMAT_R32_SINT: return uint32_t(4);
        case VULKAN_FORMAT_R32_SFLOAT: return uint32_t(4);
        case VULKAN_FORMAT_R32G32_UINT: return uint32_t(8);
        case VULKAN_FORMAT_R32G32_SINT: return uint32_t(8);
        case VULKAN_FORMAT_R32G32_SFLOAT: return uint32_t(8);
        case VULKAN_FORMAT_R32G32B32A32_UINT: return uint32_t(16);
        case VULKAN_FORMAT_R32G32B32A32_SINT: return uint32_t(16);
        case VULKAN_FORMAT_R32G32B32A32_SFLOAT: return uint32_t(16);
        default: ASSERT(0); std::cout << "Unknown format";
    }

    return uint32_t(0);
}

const char *getVulkanFormatGLSLFormat(VulkanFormat format)
{
    switch (format)
    {
        case VULKAN_FORMAT_R8_UINT: return "r8ui";
        case VULKAN_FORMAT_R8_SINT: return "r8i";
        case VULKAN_FORMAT_R8G8_UINT: return "rg8ui";
        case VULKAN_FORMAT_R8G8_SINT: return "rg8i";
        case VULKAN_FORMAT_R8G8B8A8_UINT: return "rgba8ui";
        case VULKAN_FORMAT_R8G8B8A8_SINT: return "rgba8i";
        case VULKAN_FORMAT_R16_UINT: return "r16ui";
        case VULKAN_FORMAT_R16_SINT: return "r16i";
        case VULKAN_FORMAT_R16G16_UINT: return "rg16ui";
        case VULKAN_FORMAT_R16G16_SINT: return "rg16i";
        case VULKAN_FORMAT_R16G16B16A16_UINT: return "rgba16ui";
        case VULKAN_FORMAT_R16G16B16A16_SINT: return "rgba16i";
        case VULKAN_FORMAT_R32_UINT: return "r32ui";
        case VULKAN_FORMAT_R32_SINT: return "r32i";
        case VULKAN_FORMAT_R32_SFLOAT: return "r32f";
        case VULKAN_FORMAT_R32G32_UINT: return "rg32ui";
        case VULKAN_FORMAT_R32G32_SINT: return "rg32i";
        case VULKAN_FORMAT_R32G32_SFLOAT: return "rg32f";
        case VULKAN_FORMAT_R32G32B32A32_UINT: return "rgba32ui";
        case VULKAN_FORMAT_R32G32B32A32_SINT: return "rgba32i";
        case VULKAN_FORMAT_R32G32B32A32_SFLOAT: return "rgba32f";
        default: ASSERT(0); std::cout << "Unknown format";
    }

    return (const char *)size_t(0);
}

const char *getVulkanFormatGLSLTypePrefix(VulkanFormat format)
{
    switch (format)
    {
        case VULKAN_FORMAT_R8_UINT:
        case VULKAN_FORMAT_R8G8_UINT:
        case VULKAN_FORMAT_R8G8B8A8_UINT:
        case VULKAN_FORMAT_R16_UINT:
        case VULKAN_FORMAT_R16G16_UINT:
        case VULKAN_FORMAT_R16G16B16A16_UINT:
        case VULKAN_FORMAT_R32_UINT:
        case VULKAN_FORMAT_R32G32_UINT:
        case VULKAN_FORMAT_R32G32B32A32_UINT: return "u";

        case VULKAN_FORMAT_R8_SINT:
        case VULKAN_FORMAT_R8G8_SINT:
        case VULKAN_FORMAT_R8G8B8A8_SINT:
        case VULKAN_FORMAT_R16_SINT:
        case VULKAN_FORMAT_R16G16_SINT:
        case VULKAN_FORMAT_R16G16B16A16_SINT:
        case VULKAN_FORMAT_R32_SINT:
        case VULKAN_FORMAT_R32G32_SINT:
        case VULKAN_FORMAT_R32G32B32A32_SINT: return "i";

        case VULKAN_FORMAT_R32_SFLOAT:
        case VULKAN_FORMAT_R32G32_SFLOAT:
        case VULKAN_FORMAT_R32G32B32A32_SFLOAT: return "";

        default: ASSERT(0); std::cout << "Unknown format";
    }

    return "";
}

std::string prepareVulkanShader(
    std::string shaderCode,
    const std::map<std::string, std::string> &patternToSubstituteMap)
{
    for (std::map<std::string, std::string>::const_iterator psIt =
             patternToSubstituteMap.begin();
         psIt != patternToSubstituteMap.end(); ++psIt)
    {
        std::string::size_type pos = 0u;
        while ((pos = shaderCode.find(psIt->first, pos)) != std::string::npos)
        {
            shaderCode.replace(pos, psIt->first.length(), psIt->second);
            pos += psIt->second.length();
        }
    }

    return shaderCode;
}

std::ostream &operator<<(std::ostream &os,
                         VulkanMemoryTypeProperty memoryTypeProperty)
{
    switch (memoryTypeProperty)
    {
        case VULKAN_MEMORY_TYPE_PROPERTY_NONE: return os << "None";
        case VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL:
            return os << "Device local";
        case VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT:
            return os << "Host visible and coherent";
        case VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_CACHED:
            return os << "Host visible and cached";
        case VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_CACHED_COHERENT:
            return os << "Host visible, cached and coherent";
        case VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_COHERENT:
            return os << "Device local, Host visible and coherent";
        case VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_CACHED:
            return os << "Device local, Host visible and cached";
        case VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_CACHED_COHERENT:
            return os << "Device local, Host visible, cached and coherent";
    }

    return os;
}

std::ostream &
operator<<(std::ostream &os,
           VulkanExternalMemoryHandleType externalMemoryHandleType)
{
    switch (externalMemoryHandleType)
    {
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE: return os << "None";
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            return os << "Opaque file descriptor";
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
            return os << "Opaque NT handle";
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            return os << "Opaque D3DKMT handle";
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT_KMT:
            return os << "Opaque NT and D3DKMT handle";
    }

    return os;
}

std::ostream &
operator<<(std::ostream &os,
           VulkanExternalSemaphoreHandleType externalSemaphoreHandleType)
{
    switch (externalSemaphoreHandleType)
    {
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE: return os << "None";
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD:
            return os << "Opaque file descriptor";
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT:
            return os << "Opaque NT handle";
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            return os << "Opaque D3DKMT handle";
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT_KMT:
            return os << "Opaque NT and D3DKMT handle";
    }

    return os;
}

std::ostream &operator<<(std::ostream &os, VulkanFormat format)
{
    switch (format)
    {
        case VULKAN_FORMAT_R8_UINT: return os << "R8_UINT";
        case VULKAN_FORMAT_R8_SINT: return os << "R8_SINT";
        case VULKAN_FORMAT_R8G8_UINT: return os << "R8G8_UINT";
        case VULKAN_FORMAT_R8G8_SINT: return os << "R8G8_SINT";
        case VULKAN_FORMAT_R8G8B8A8_UINT: return os << "R8G8B8A8_UINT";
        case VULKAN_FORMAT_R8G8B8A8_SINT: return os << "R8G8B8A8_SINT";
        case VULKAN_FORMAT_R16_UINT: return os << "R16_UINT";
        case VULKAN_FORMAT_R16_SINT: return os << "R16_SINT";
        case VULKAN_FORMAT_R16G16_UINT: return os << "R16G16_UINT";
        case VULKAN_FORMAT_R16G16_SINT: return os << "R16G16_SINT";
        case VULKAN_FORMAT_R16G16B16A16_UINT: return os << "R16G16B16A16_UINT";
        case VULKAN_FORMAT_R16G16B16A16_SINT: return os << "R16G16B16A16_SINT";
        case VULKAN_FORMAT_R32_UINT: return os << "R32_UINT";
        case VULKAN_FORMAT_R32_SINT: return os << "R32_SINT";
        case VULKAN_FORMAT_R32_SFLOAT: return os << "R32_SFLOAT";
        case VULKAN_FORMAT_R32G32_UINT: return os << "R32G32_UINT";
        case VULKAN_FORMAT_R32G32_SINT: return os << "R32G32_SINT";
        case VULKAN_FORMAT_R32G32_SFLOAT: return os << "R32G32_SFLOAT";
        case VULKAN_FORMAT_R32G32B32A32_UINT: return os << "R32G32B32A32_UINT";
        case VULKAN_FORMAT_R32G32B32A32_SINT: return os << "R32G32B32A32_SINT";
        case VULKAN_FORMAT_R32G32B32A32_SFLOAT:
            return os << "R32G32B32A32_SFLOAT";
            break;
        default: ASSERT(0); std::cout << "Unknown format";
    }

    return os;
}
