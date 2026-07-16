#include "opencl_vulkan_wrapper.hpp"
#include "harness/errorHelpers.h"

std::vector<VulkanExternalSemaphoreHandleType>
getSupportedInteropExternalSemaphoreHandleTypes(cl_device_id device,
                                                const VulkanDevice &vkDevice)
{
    log_error("Stub getSupportedInteropExternalSemaphoreHandleTypes called\n");
    return std::vector<VulkanExternalSemaphoreHandleType>();
}

void init_cl_vk_ext(cl_platform_id platform, cl_uint num_devices,
                    cl_device_id *deviceIds)
{
    log_info("Stub init_cl_vk_ext called\n");
}

clExternalImportableSemaphore::clExternalImportableSemaphore(
    const VulkanSemaphore &deviceSemaphore, cl_context context,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
    cl_device_id deviceId)
    : m_externalSemaphore(nullptr)
{
    log_error("Stub clExternalImportableSemaphore constructor called\n");
}

clExternalImportableSemaphore::~clExternalImportableSemaphore()
{
    log_info("Stub clExternalImportableSemaphore destructor called\n");
}

int clExternalImportableSemaphore::wait(cl_command_queue command_queue)
{
    log_error("Stub clExternalImportableSemaphore::wait called\n");
    return CL_INVALID_OPERATION;
}

int clExternalImportableSemaphore::signal(cl_command_queue command_queue)
{
    log_error("Stub clExternalImportableSemaphore::signal called\n");
    return CL_INVALID_OPERATION;
}

cl_semaphore_khr &clExternalImportableSemaphore::getCLSemaphore()
{
    log_error("Stub clExternalImportableSemaphore::getCLSemaphore called\n");
    return m_externalSemaphore;
}
