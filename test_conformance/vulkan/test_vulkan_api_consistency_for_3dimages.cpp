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

#include <vulkan_interop_common.hpp>
#include <opencl_vulkan_wrapper.hpp>
#include <vulkan_wrapper.hpp>
#if !defined(__APPLE__)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#endif

#include <assert.h>
#include <vector>
#include <iostream>
#include <string.h>
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/deviceInfo.h"
#include <string>

#include "vulkan_test_base.h"
#include "opencl_vulkan_wrapper.hpp"

namespace {

struct ConsistencyExternalImage3DTest : public VulkanTestBase
{
    ConsistencyExternalImage3DTest(cl_device_id device, cl_context context,
                                   cl_command_queue queue, cl_int nelems)
        : VulkanTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int errNum;

#ifdef _WIN32
        if (!is_extension_available(device, "cl_khr_external_memory_win32"))
        {
            throw std::runtime_error(
                "Device does not support"
                "cl_khr_external_memory_win32 extension \n");
        }
#else
        if (!is_extension_available(device, "cl_khr_external_memory_opaque_fd"))
        {
            throw std::runtime_error(
                "Device does not support cl_khr_external_memory_opaque_fd "
                "extension \n");
        }
#endif
        uint32_t width = 256;
        uint32_t height = 16;
        uint32_t depth = 10;
        cl_image_desc image_desc;
        memset(&image_desc, 0x0, sizeof(cl_image_desc));
        cl_image_format img_format = { 0 };

        VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
            getSupportedVulkanExternalMemoryHandleTypeList()[0];

        VulkanImageTiling vulkanImageTiling =
            vkClExternalMemoryHandleTilingAssumption(
                device, vkExternalMemoryHandleType, &errNum);
        ASSERT_SUCCESS(errNum, "Failed to query OpenCL tiling mode");

        VulkanImage3D vkImage3D = VulkanImage3D(
            *vkDevice, VULKAN_FORMAT_R8G8B8A8_UNORM, width, height, depth,
            vulkanImageTiling, 1, vkExternalMemoryHandleType);

        const VulkanMemoryTypeList& memoryTypeList =
            vkImage3D.getMemoryTypeList();
        uint64_t totalImageMemSize = vkImage3D.getSize();

        log_info("Memory type index: %u\n", (uint32_t)memoryTypeList[0]);
        log_info("Memory type property: %d\n",
                 memoryTypeList[0].getMemoryTypeProperty());
        log_info("Image size : %lu\n", totalImageMemSize);

        VulkanDeviceMemory* vkDeviceMem =
            new VulkanDeviceMemory(*vkDevice, vkImage3D, memoryTypeList[0],
                                   vkExternalMemoryHandleType);
        vkDeviceMem->bindImage(vkImage3D, 0);

        void* handle = NULL;
        int fd;
        std::vector<cl_mem_properties> extMemProperties{
            (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_KHR,
            (cl_mem_properties)device,
            (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
        };
        switch (vkExternalMemoryHandleType)
        {
#ifdef _WIN32
            case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
                handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
                errNum = check_external_memory_handle_type(
                    device, CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
                extMemProperties.push_back(
                    (cl_mem_properties)
                        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
                extMemProperties.push_back((cl_mem_properties)handle);
                break;
            case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
                handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
                errNum = check_external_memory_handle_type(
                    device, CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
                extMemProperties.push_back(
                    (cl_mem_properties)
                        CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
                extMemProperties.push_back((cl_mem_properties)handle);
                break;
#else
            case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
                fd = (int)vkDeviceMem->getHandle(vkExternalMemoryHandleType);
                errNum = check_external_memory_handle_type(
                    device, CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
                extMemProperties.push_back(
                    (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
                extMemProperties.push_back((cl_mem_properties)fd);
                break;
#endif
            default:
                errNum = TEST_FAIL;
                log_error("Unsupported external memory handle type \n");
                break;
        }
        if (errNum != CL_SUCCESS)
        {
            log_error("Checks failed for "
                      "CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR\n");
            return TEST_FAIL;
        }
        extMemProperties.push_back(0);

        const VkImageCreateInfo VulkanImageCreateInfo =
            vkImage3D.getVkImageCreateInfo();

        errNum = getCLImageInfoFromVkImageInfo(&VulkanImageCreateInfo,
                                               totalImageMemSize, &img_format,
                                               &image_desc);
        if (errNum != CL_SUCCESS)
        {
            log_error("getCLImageInfoFromVkImageInfo failed!!!");
            return TEST_FAIL;
        }

        clMemWrapper image;

        // Pass valid properties, image_desc and image_format
        image = clCreateImageWithProperties(
            context, extMemProperties.data(), CL_MEM_READ_WRITE, &img_format,
            &image_desc, NULL /* host_ptr */, &errNum);
        test_error(errNum, "Unable to create Image with Properties");
        image.reset();

        // Passing NULL properties and a valid image_format and image_desc
        image = clCreateImageWithProperties(context, NULL, CL_MEM_READ_WRITE,
                                            &img_format, &image_desc, NULL,
                                            &errNum);
        test_error(errNum,
                   "Unable to create image with NULL properties "
                   "with valid image format and image desc");

        image.reset();

        // Passing image_format as NULL
        image = clCreateImageWithProperties(context, extMemProperties.data(),
                                            CL_MEM_READ_WRITE, NULL,
                                            &image_desc, NULL, &errNum);
        test_failure_error(errNum, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                           "Image creation must fail with "
                           "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"
                           "when image desc passed as NULL");

        image.reset();

        // Passing image_desc as NULL
        image = clCreateImageWithProperties(context, extMemProperties.data(),
                                            CL_MEM_READ_WRITE, &img_format,
                                            NULL, NULL, &errNum);
        test_failure_error(errNum, CL_INVALID_IMAGE_DESCRIPTOR,
                           "Image creation must fail with "
                           "CL_INVALID_IMAGE_DESCRIPTOR "
                           "when image desc passed as NULL");
        image.reset();

        return TEST_PASS;
    }
};

} // anonymous namespace

int test_consistency_external_for_3dimage(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue defaultQueue,
                                          int num_elements)
{
    return MakeAndRunTest<ConsistencyExternalImage3DTest>(
        deviceID, context, defaultQueue, num_elements);
}
