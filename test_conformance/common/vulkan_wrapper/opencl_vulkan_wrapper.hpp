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
#ifndef _opencl_vulkan_wrapper_hpp_
#define _opencl_vulkan_wrapper_hpp_

#include "vulkan_wrapper.hpp"

#if !defined(__APPLE__)
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#endif

typedef cl_semaphore_khr (*pfnclCreateSemaphoreWithPropertiesKHR)(
    cl_context context, cl_semaphore_properties_khr *sema_props,
    cl_int *errcode_ret);
typedef cl_int (*pfnclEnqueueWaitSemaphoresKHR)(
    cl_command_queue command_queue, cl_uint num_semaphores,
    const cl_semaphore_khr *sema_list,
    const cl_semaphore_payload_khr *sema_payload_list,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);
typedef cl_int (*pfnclEnqueueSignalSemaphoresKHR)(
    cl_command_queue command_queue, cl_uint num_semaphores,
    const cl_semaphore_khr *sema_list,
    const cl_semaphore_payload_khr *sema_payload_list,
    cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *event);
typedef cl_int (*pfnclEnqueueAcquireExternalMemObjectsKHR)(
    cl_command_queue command_queue, cl_uint num_mem_objects,
    const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);
typedef cl_int (*pfnclEnqueueReleaseExternalMemObjectsKHR)(
    cl_command_queue command_queue, cl_uint num_mem_objects,
    const cl_mem *mem_objects, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event);
typedef cl_int (*pfnclReleaseSemaphoreKHR)(cl_semaphore_khr sema_object);
typedef cl_int (*pfnclGetSemaphoreHandleForTypeKHR)(
    cl_semaphore_khr sema_object, cl_device_id device,
    cl_external_semaphore_handle_type_khr handleType, size_t handle_size,
    void *handle, size_t *handleSize);
typedef cl_int (*pfnclReImportSemaphoreSyncFdKHR)(
    cl_semaphore_khr sema_object,
    cl_semaphore_reimport_properties_khr *reimport_props, int fd);

extern pfnclCreateSemaphoreWithPropertiesKHR
    clCreateSemaphoreWithPropertiesKHRptr;
extern pfnclEnqueueWaitSemaphoresKHR clEnqueueWaitSemaphoresKHRptr;
extern pfnclEnqueueSignalSemaphoresKHR clEnqueueSignalSemaphoresKHRptr;
extern pfnclEnqueueAcquireExternalMemObjectsKHR
    clEnqueueAcquireExternalMemObjectsKHRptr;
extern pfnclEnqueueReleaseExternalMemObjectsKHR
    clEnqueueReleaseExternalMemObjectsKHRptr;
extern pfnclReleaseSemaphoreKHR clReleaseSemaphoreKHRptr;
extern pfnclReImportSemaphoreSyncFdKHR pfnclReImportSemaphoreSyncFdKHRptr;

cl_int getCLImageInfoFromVkImageInfo(const VkImageCreateInfo *, size_t,
                                     cl_image_format *, cl_image_desc *);
cl_int check_external_memory_handle_type(
    cl_device_id deviceID,
    cl_external_memory_handle_type_khr requiredHandleType);
cl_int check_external_semaphore_handle_type(
    cl_device_id deviceID,
    cl_external_semaphore_handle_type_khr requiredHandleType);
cl_int setMaxImageDimensions(cl_device_id deviceID, size_t &width,
                             size_t &height);

class clExternalMemory {
protected:
    cl_mem m_externalMemory;
    int fd;
    void *handle;
    clExternalMemory(const clExternalMemory &externalMemory);

public:
    clExternalMemory();
    clExternalMemory(const VulkanDeviceMemory *deviceMemory,
                     VulkanExternalMemoryHandleType externalMemoryHandleType,
                     uint64_t size, cl_context context, cl_device_id deviceId);

    virtual ~clExternalMemory();
    cl_mem getExternalMemoryBuffer();
};
class clExternalMemoryImage {
protected:
    cl_mem m_externalMemory;
    int fd;
    void *handle;
    clExternalMemoryImage();

public:
    clExternalMemoryImage(
        const VulkanDeviceMemory &deviceMemory,
        VulkanExternalMemoryHandleType externalMemoryHandleType,
        cl_context context, size_t totalImageMemSize, size_t imageWidth,
        size_t imageHeight, size_t totalSize, const VulkanImage2D &image2D,
        cl_device_id deviceId);
    virtual ~clExternalMemoryImage();
    cl_mem getExternalMemoryImage();
};

class clExternalSemaphore {
public:
    virtual int signal(cl_command_queue command_queue) = 0;
    virtual int wait(cl_command_queue command_queue) = 0;
    virtual cl_semaphore_khr &getCLSemaphore() = 0;
    virtual ~clExternalSemaphore() = 0;
};


class clExternalImportableSemaphore : public virtual clExternalSemaphore {
protected:
    cl_semaphore_khr m_externalSemaphore;
    VulkanExternalSemaphoreHandleType m_externalHandleType;
    cl_device_id m_device;
    cl_context m_context;
    const VulkanSemaphore &m_deviceSemaphore;
    int fd;
    void *handle;

public:
    clExternalImportableSemaphore(
        const VulkanSemaphore &deviceSemaphore, cl_context context,
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
        cl_device_id deviceId);
    ~clExternalImportableSemaphore() override;
    int wait(cl_command_queue command_queue) override;
    int signal(cl_command_queue command_queue) override;
    cl_semaphore_khr &getCLSemaphore() override;
};

class clExternalExportableSemaphore : public virtual clExternalSemaphore {
protected:
    cl_semaphore_khr m_externalSemaphore;
    VulkanExternalSemaphoreHandleType m_externalHandleType;
    cl_device_id m_device;
    cl_context m_context;
    const VulkanSemaphore &m_deviceSemaphore;
    int fd;
    void *handle;

public:
    clExternalExportableSemaphore(
        const VulkanSemaphore &deviceSemaphore, cl_context context,
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
        cl_device_id deviceId);
    ~clExternalExportableSemaphore() override;
    int signal(cl_command_queue command_queue) override;
    int wait(cl_command_queue command_queue) override;
    cl_semaphore_khr &getCLSemaphore() override;
};

extern void init_cl_vk_ext(cl_platform_id, cl_uint num_devices,
                           cl_device_id *deviceIds);

VulkanImageTiling vkClExternalMemoryHandleTilingAssumption(
    cl_device_id deviceId,
    VulkanExternalMemoryHandleType vkExternalMemoryHandleType, int *error_ret);

#endif // _opencl_vulkan_wrapper_hpp_