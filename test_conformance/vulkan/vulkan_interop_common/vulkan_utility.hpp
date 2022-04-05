#ifndef _vulkan_utility_hpp_
#define _vulkan_utility_hpp_

#include "vulkan_wrapper_types.hpp"
#include <vector>
#include <ostream>
#include <string.h>
#include <map>

#define ARRAY_SIZE(a)  (sizeof(a) / sizeof(a[0]))

#define STRING_(str)  #str
#define STRING(str)   STRING_(str)

#define ROUND_UP(n, multiple)  (((n) + (multiple) - 1) - ((((n) + (multiple) - 1)) % (multiple)))

const VulkanInstance & getVulkanInstance();
const VulkanPhysicalDevice & getVulkanPhysicalDevice();
const VulkanQueueFamily & getVulkanQueueFamily(
    uint32_t queueFlags = VULKAN_QUEUE_FLAG_MASK_ALL);
const VulkanMemoryType & getVulkanMemoryType(
    const VulkanDevice & device,
    VulkanMemoryTypeProperty memoryTypeProperty);
bool checkVkSupport();
const VulkanQueueFamilyList & getEmptyVulkanQueueFamilyList();
const VulkanDescriptorSetLayoutList & getEmptyVulkanDescriptorSetLayoutList();
const VulkanQueueFamilyToQueueCountMap & getDefaultVulkanQueueFamilyToQueueCountMap();
const std::vector<VulkanExternalMemoryHandleType> getSupportedVulkanExternalMemoryHandleTypeList();
const std::vector<VulkanExternalSemaphoreHandleType> getSupportedVulkanExternalSemaphoreHandleTypeList();
const std::vector<VulkanFormat> getSupportedVulkanFormatList();

uint32_t getVulkanFormatElementSize(
    VulkanFormat format);
const char * getVulkanFormatGLSLFormat(
    VulkanFormat format);
const char * getVulkanFormatGLSLTypePrefix(
    VulkanFormat format);

std::string prepareVulkanShader(
    std::string shaderCode,
    const std::map<std::string, std::string> & patternToSubstituteMap);

std::ostream & operator<<(
    std::ostream & os,
    VulkanMemoryTypeProperty memoryTypeProperty);
std::ostream & operator<<(
    std::ostream & os,
    VulkanExternalMemoryHandleType externalMemoryHandleType);
std::ostream & operator<<(
    std::ostream & os,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType);
std::ostream & operator<<(
    std::ostream & os,
    VulkanFormat format);

#endif // _vulkan_utility_hpp_
