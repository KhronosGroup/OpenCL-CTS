#ifndef _opencl_vulkan_wrapper_hpp_
#define _opencl_vulkan_wrapper_hpp_

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <vector>

enum VulkanExternalSemaphoreHandleType
{
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE = 0,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT_NAME
};

class VulkanDevice {
public:
    VulkanDevice() {}
};

class VulkanSemaphore {
public:
    VulkanSemaphore(const VulkanDevice &device,
                    VulkanExternalSemaphoreHandleType type)
    {}
};

std::vector<VulkanExternalSemaphoreHandleType>
getSupportedInteropExternalSemaphoreHandleTypes(cl_device_id device,
                                                const VulkanDevice &vkDevice);

class clExternalSemaphore {
public:
    virtual int signal(cl_command_queue command_queue) = 0;
    virtual int wait(cl_command_queue command_queue) = 0;
    virtual cl_semaphore_khr &getCLSemaphore() = 0;
    virtual ~clExternalSemaphore() {}
};

class clExternalImportableSemaphore : public virtual clExternalSemaphore {
protected:
    cl_semaphore_khr m_externalSemaphore;

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

extern void init_cl_vk_ext(cl_platform_id, cl_uint num_devices,
                           cl_device_id *deviceIds);

#endif // _opencl_vulkan_wrapper_hpp_
