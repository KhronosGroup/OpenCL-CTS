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

int test_consistency_external_buffer(cl_device_id deviceID, cl_context _context,
                                     cl_command_queue _queue, int num_elements)
{
    cl_int errNum;
    VulkanDevice vkDevice;
    // Context and command queue creation
    cl_platform_id platform = NULL;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    errNum = clGetPlatformIDs(1, &platform, NULL);
    test_error(errNum, "Failed to get platform Id");

    contextProperties[1] = (cl_context_properties)platform;

    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    test_error(errNum, "Unable to create context with properties");

    cmd_queue = clCreateCommandQueue(context, deviceID, 0, &errNum);
    test_error(errNum, "Unable to create command queue");

    uint32_t bufferSize = 32;
    cl_device_id devList[] = { deviceID, NULL };

#ifdef _WIN32
    if (!is_extension_available(devList[0], "cl_khr_external_memory_win32"))
    {
        throw std::runtime_error("Device does not support "
                                 "cl_khr_external_memory_win32 extension \n");
    }
#else
    if (!is_extension_available(devList[0], "cl_khr_external_memory_opaque_fd"))
    {
        throw std::runtime_error(
            "Device does not support "
            "cl_khr_external_memory_opaque_fd extension \n");
    }
#endif

    VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
        getSupportedVulkanExternalMemoryHandleTypeList()[0];

    VulkanBuffer vkDummyBuffer(vkDevice, 4 * 1024, vkExternalMemoryHandleType);
    const VulkanMemoryTypeList& memoryTypeList =
        vkDummyBuffer.getMemoryTypeList();

    VulkanDeviceMemory* vkDeviceMem = new VulkanDeviceMemory(
        vkDevice, bufferSize, memoryTypeList[0], vkExternalMemoryHandleType);
    VulkanBufferList vkBufferList(1, vkDevice, bufferSize,
                                  vkExternalMemoryHandleType);

    vkDeviceMem->bindBuffer(vkBufferList[0], 0);

    void* handle = NULL;
    int fd;

    std::vector<cl_mem_properties> extMemProperties{
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_KHR,
        (cl_mem_properties)devList[0],
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_END_KHR,
    };
    cl_external_memory_handle_type_khr type;
    switch (vkExternalMemoryHandleType)
    {
#ifdef _WIN32
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
            handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            type = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
            errNum = check_external_memory_handle_type(devList[0], type);
            extMemProperties.push_back((cl_mem_properties)type);
            extMemProperties.push_back((cl_mem_properties)handle);
            break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            type = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR;
            errNum = check_external_memory_handle_type(devList[0], type);
            extMemProperties.push_back((cl_mem_properties)type);
            extMemProperties.push_back((cl_mem_properties)handle);
            break;
#else
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            fd = (int)vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            type = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
            errNum = check_external_memory_handle_type(devList[0], type);
            extMemProperties.push_back((cl_mem_properties)type);
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

    clMemWrapper buffer;

    // Passing NULL properties and a valid extMem_desc size
    buffer = clCreateBufferWithProperties(context, NULL, 1, bufferSize, NULL,
                                          &errNum);
    test_error(errNum, "Unable to create buffer with NULL properties");

    buffer.reset();

    // Passing valid extMemProperties and buffersize
    buffer = clCreateBufferWithProperties(context, extMemProperties.data(), 1,
                                          bufferSize, NULL, &errNum);
    test_error(errNum, "Unable to create buffer with Properties");

    buffer.reset();

    // Not passing external memory handle
    std::vector<cl_mem_properties> extMemProperties2{
#ifdef _WIN32
        (cl_mem_properties)type,
        NULL, // Passing NULL handle
#else
        (cl_mem_properties)type,
        (cl_mem_properties)-64, // Passing random invalid fd
#endif
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_KHR,
        (cl_mem_properties)devList[0],
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_END_KHR,
        0
    };
    buffer = clCreateBufferWithProperties(context, extMemProperties2.data(), 1,
                                          bufferSize, NULL, &errNum);
    test_failure_error(errNum, CL_INVALID_VALUE,
                       "Should return CL_INVALID_VALUE ");

    buffer.reset();

    // Passing extMem_desc size = 0 but valid memProperties, CL_INVALID_SIZE
    // should be returned.
    buffer = clCreateBufferWithProperties(context, extMemProperties.data(), 1,
                                          0, NULL, &errNum);
    test_failure_error(errNum, CL_INVALID_BUFFER_SIZE,
                       "Should return CL_INVALID_BUFFER_SIZE");

    return TEST_PASS;
}

int test_consistency_external_image(cl_device_id deviceID, cl_context _context,
                                    cl_command_queue _queue, int num_elements)
{
    cl_int errNum;
    VulkanDevice vkDevice;

    // Context and command queue creation
    cl_platform_id platform = NULL;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    errNum = clGetPlatformIDs(1, &platform, NULL);
    test_error(errNum, "Failed to get platform id");

    contextProperties[1] = (cl_context_properties)platform;

    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    test_error(errNum, "Unable to create context with properties");

    cmd_queue = clCreateCommandQueue(context, deviceID, 0, &errNum);
    test_error(errNum, "Unable to create command queue");

    cl_device_id devList[] = { deviceID, NULL };

#ifdef _WIN32
    if (!is_extension_available(devList[0], "cl_khr_external_memory_win32"))
    {
        throw std::runtime_error("Device does not support"
                                 "cl_khr_external_memory_win32 extension \n");
    }
#else
    if (!is_extension_available(devList[0], "cl_khr_external_memory_opaque_fd"))
    {
        throw std::runtime_error(
            "Device does not support cl_khr_external_memory_opaque_fd "
            "extension \n");
    }
#endif
    uint32_t width = 256;
    uint32_t height = 16;
    cl_image_desc image_desc;
    memset(&image_desc, 0x0, sizeof(cl_image_desc));
    cl_image_format img_format = { 0 };

    VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
        getSupportedVulkanExternalMemoryHandleTypeList()[0];
    VulkanImage2D* vkImage2D =
        new VulkanImage2D(vkDevice, VULKAN_FORMAT_R8G8B8A8_UNORM, width, height,
                          1, vkExternalMemoryHandleType);

    const VulkanMemoryTypeList& memoryTypeList = vkImage2D->getMemoryTypeList();
    uint64_t totalImageMemSize = vkImage2D->getSize();

    log_info("Memory type index: %lu\n", (uint32_t)memoryTypeList[0]);
    log_info("Memory type property: %d\n",
             memoryTypeList[0].getMemoryTypeProperty());
    log_info("Image size : %d\n", totalImageMemSize);

    VulkanDeviceMemory* vkDeviceMem =
        new VulkanDeviceMemory(vkDevice, totalImageMemSize, memoryTypeList[0],
                               vkExternalMemoryHandleType);
    vkDeviceMem->bindImage(*vkImage2D, 0);

    void* handle = NULL;
    int fd;
    std::vector<cl_mem_properties> extMemProperties{
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_KHR,
        (cl_mem_properties)devList[0],
        (cl_mem_properties)CL_DEVICE_HANDLE_LIST_END_KHR,
    };
    switch (vkExternalMemoryHandleType)
    {
#ifdef _WIN32
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT:
            handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            errNum = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties.push_back(
                (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
            extMemProperties.push_back((cl_mem_properties)handle);
            break;
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            handle = vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            errNum = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties.push_back(
                (cl_mem_properties)
                    CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
            extMemProperties.push_back((cl_mem_properties)handle);
            break;
#else
        case VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            fd = (int)vkDeviceMem->getHandle(vkExternalMemoryHandleType);
            errNum = check_external_memory_handle_type(
                devList[0], CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
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
        vkImage2D->getVkImageCreateInfo();

    errNum = getCLImageInfoFromVkImageInfo(
        &VulkanImageCreateInfo, totalImageMemSize, &img_format, &image_desc);
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

    // Passing properties, image_desc and image_format all as NULL
    image = clCreateImageWithProperties(context, NULL, CL_MEM_READ_WRITE, NULL,
                                        NULL, NULL, &errNum);
    test_failure_error(
        errNum, CL_INVALID_IMAGE_DESCRIPTOR,
        "Image creation must fail with CL_INVALID_IMAGE_DESCRIPTOR "
        "when all are passed as NULL");

    image.reset();

    // Passing NULL properties and a valid image_format and image_desc
    image =
        clCreateImageWithProperties(context, NULL, CL_MEM_READ_WRITE,
                                    &img_format, &image_desc, NULL, &errNum);
    test_error(errNum,
               "Unable to create image with NULL properties "
               "with valid image format and image desc");

    image.reset();

    // Passing image_format as NULL
    image = clCreateImageWithProperties(context, extMemProperties.data(),
                                        CL_MEM_READ_WRITE, NULL, &image_desc,
                                        NULL, &errNum);
    test_failure_error(errNum, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                       "Image creation must fail with "
                       "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"
                       "when image desc passed as NULL");

    image.reset();

    // Passing image_desc as NULL
    image = clCreateImageWithProperties(context, extMemProperties.data(),
                                        CL_MEM_READ_WRITE, &img_format, NULL,
                                        NULL, &errNum);
    test_failure_error(errNum, CL_INVALID_IMAGE_DESCRIPTOR,
                       "Image creation must fail with "
                       "CL_INVALID_IMAGE_DESCRIPTOR "
                       "when image desc passed as NULL");
    image.reset();

    return TEST_PASS;
}

int test_consistency_external_semaphore(cl_device_id deviceID,
                                        cl_context _context,
                                        cl_command_queue _queue,
                                        int num_elements)
{
    cl_int errNum;
    VulkanDevice vkDevice;
    // Context and command queue creation
    cl_platform_id platform = NULL;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;

    errNum = clGetPlatformIDs(1, &platform, NULL);
    test_error(errNum, "Failed to get platform Id");

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, 0, 0 };

    contextProperties[1] = (cl_context_properties)platform;

    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    test_error(errNum, "Unable to create context with properties");

    cmd_queue = clCreateCommandQueue(context, deviceID, 0, &errNum);
    test_error(errNum, "Unable to create command queue");

    cl_device_id devList[] = { deviceID, NULL };

#ifdef _WIN32
    if (!is_extension_available(devList[0], "cl_khr_external_semaphore_win32"))
    {
        throw std::runtime_error(
            "Device does not support cl_khr_external_semaphore_win32 "
            "extension \n");
    }
#else
    if (!is_extension_available(devList[0],
                                "cl_khr_external_semaphore_opaque_fd"))
    {
        throw std::runtime_error(
            "Device does not support "
            "cl_khr_external_semaphore_opaque_fd extension \n");
    }
#endif
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2Clsemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2Vksemaphore(vkDevice, vkExternalSemaphoreHandleType);
    cl_semaphore_khr clCl2Vksemaphore;
    cl_semaphore_khr clVk2Clsemaphore;

    void* handle1 = NULL;
    void* handle2 = NULL;
    int fd1, fd2;
    std::vector<cl_semaphore_properties_khr> sema_props1{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
    };
    std::vector<cl_semaphore_properties_khr> sema_props2{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
    };
    switch (vkExternalSemaphoreHandleType)
    {
#ifdef _WIN32
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT:
            log_info(" Opaque NT handles are only supported on Windows\n");
            handle1 = vkVk2Clsemaphore.getHandle(vkExternalSemaphoreHandleType);
            handle2 = vkCl2Vksemaphore.getHandle(vkExternalSemaphoreHandleType);
            errNum = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
            sema_props1.push_back((cl_semaphore_properties_khr)
                                      CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
            sema_props1.push_back((cl_semaphore_properties_khr)handle1);
            sema_props2.push_back((cl_semaphore_properties_khr)
                                      CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR);
            sema_props2.push_back((cl_semaphore_properties_khr)handle2);
            break;
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT:
            log_info(" Opaque D3DKMT handles are only supported on Windows\n");
            handle1 = vkVk2Clsemaphore.getHandle(vkExternalSemaphoreHandleType);
            handle2 = vkCl2Vksemaphore.getHandle(vkExternalSemaphoreHandleType);
            errNum = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
            sema_props1.push_back((cl_semaphore_properties_khr)
                                      CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
            sema_props1.push_back((cl_semaphore_properties_khr)handle1);
            sema_props2.push_back((cl_semaphore_properties_khr)
                                      CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR);
            sema_props2.push_back((cl_semaphore_properties_khr)handle2);
            break;
#else
        case VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD:
            log_info(" Opaque file descriptors are not supported on Windows\n");
            fd1 =
                (int)vkVk2Clsemaphore.getHandle(vkExternalSemaphoreHandleType);
            fd2 =
                (int)vkCl2Vksemaphore.getHandle(vkExternalSemaphoreHandleType);
            errNum = check_external_semaphore_handle_type(
                devList[0], CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
            sema_props1.push_back(
                (cl_semaphore_properties_khr)CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
            sema_props1.push_back((cl_semaphore_properties_khr)fd1);
            sema_props2.push_back(
                (cl_semaphore_properties_khr)CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR);
            sema_props2.push_back((cl_semaphore_properties_khr)fd2);
            break;
#endif
        default: log_error("Unsupported external memory handle type\n"); break;
    }
    if (CL_SUCCESS != errNum)
    {
        throw std::runtime_error(
            "Unsupported external sempahore handle type\n ");
    }
    sema_props1.push_back(
        (cl_semaphore_properties_khr)CL_DEVICE_HANDLE_LIST_KHR);
    sema_props1.push_back((cl_semaphore_properties_khr)devList[0]);
    sema_props1.push_back(
        (cl_semaphore_properties_khr)CL_DEVICE_HANDLE_LIST_END_KHR);
    sema_props2.push_back(
        (cl_semaphore_properties_khr)CL_DEVICE_HANDLE_LIST_KHR);
    sema_props2.push_back((cl_semaphore_properties_khr)devList[0]);
    sema_props2.push_back(
        (cl_semaphore_properties_khr)CL_DEVICE_HANDLE_LIST_END_KHR);
    sema_props1.push_back(0);
    sema_props2.push_back(0);

    // Pass NULL properties
    cl_semaphore_khr cl_ext_semaphore =
        clCreateSemaphoreWithPropertiesKHRptr(context, NULL, &errNum);
    test_failure_error(errNum, CL_INVALID_VALUE,
                       "Semaphore creation must fail with CL_INVALID_VALUE "
                       " when properties are passed as NULL");


    // Pass invalid semaphore object to wait
    errNum =
        clEnqueueWaitSemaphoresKHRptr(cmd_queue, 1, NULL, NULL, 0, NULL, NULL);
    test_failure_error(errNum, CL_INVALID_VALUE,
                       "clEnqueueWaitSemaphoresKHR fails with CL_INVALID_VALUE "
                       "when invalid semaphore object is passed");


    // Pass invalid semaphore object to signal
    errNum = clEnqueueSignalSemaphoresKHRptr(cmd_queue, 1, NULL, NULL, 0, NULL,
                                             NULL);
    test_failure_error(
        errNum, CL_INVALID_VALUE,
        "clEnqueueSignalSemaphoresKHR fails with CL_INVALID_VALUE"
        "when invalid semaphore object is passed");


    // Create two semaphore objects
    clVk2Clsemaphore = clCreateSemaphoreWithPropertiesKHRptr(
        context, sema_props1.data(), &errNum);
    test_error(errNum,
               "Unable to create semaphore with valid semaphore properties");

    clCl2Vksemaphore = clCreateSemaphoreWithPropertiesKHRptr(
        context, sema_props2.data(), &errNum);
    test_error(errNum,
               "Unable to create semaphore with valid semaphore properties");


    // Call Signal twice consecutively
    errNum = clEnqueueSignalSemaphoresKHRptr(cmd_queue, 1, &clVk2Clsemaphore,
                                             NULL, 0, NULL, NULL);
    test_error(errNum, "clEnqueueSignalSemaphoresKHRptr failed");

    errNum = clEnqueueSignalSemaphoresKHRptr(cmd_queue, 1, &clCl2Vksemaphore,
                                             NULL, 0, NULL, NULL);
    test_error(errNum,
               "clEnqueueSignalSemaphoresKHRptr failed for two "
               "consecutive wait events");


    // Call Wait twice consecutively
    errNum = clEnqueueWaitSemaphoresKHRptr(cmd_queue, 1, &clVk2Clsemaphore,
                                           NULL, 0, NULL, NULL);
    test_error(errNum, "clEnqueueWaitSemaphoresKHRptr failed");

    errNum = clEnqueueWaitSemaphoresKHRptr(cmd_queue, 1, &clCl2Vksemaphore,
                                           NULL, 0, NULL, NULL);
    test_error(errNum,
               "clEnqueueWaitSemaphoresKHRptr failed for two "
               " consecutive wait events");


    // Pass invalid object to release call
    errNum = clReleaseSemaphoreKHRptr(NULL);
    test_failure_error(errNum, CL_INVALID_VALUE,
                       "clReleaseSemaphoreKHRptr fails with "
                       "CL_INVALID_VALUE when NULL semaphore object is passed");

    // Release both semaphore objects
    errNum = clReleaseSemaphoreKHRptr(clVk2Clsemaphore);
    test_error(errNum, "clReleaseSemaphoreKHRptr failed");

    errNum = clReleaseSemaphoreKHRptr(clCl2Vksemaphore);
    test_error(errNum, "clReleaseSemaphoreKHRptr failed");

    return TEST_PASS;
}
