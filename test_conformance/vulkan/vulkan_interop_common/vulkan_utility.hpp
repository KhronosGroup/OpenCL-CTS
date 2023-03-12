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

#ifndef _vulkan_utility_hpp_
#define _vulkan_utility_hpp_

#include "vulkan_wrapper_types.hpp"
#include <vector>
#include <ostream>
#include <string.h>
#include <map>
#include "../../../test_common/harness/testHarness.h"

#define STRING_(str) #str
#define STRING(str) STRING_(str)

#define ROUND_UP(n, multiple)                                                  \
    (((n) + (multiple)-1) - ((((n) + (multiple)-1)) % (multiple)))

const VulkanInstance& getVulkanInstance();
const VulkanPhysicalDevice& getVulkanPhysicalDevice();
const VulkanQueueFamily&
getVulkanQueueFamily(uint32_t queueFlags = VULKAN_QUEUE_FLAG_MASK_ALL);
const VulkanMemoryType&
getVulkanMemoryType(const VulkanDevice& device,
                    VulkanMemoryTypeProperty memoryTypeProperty);
bool checkVkSupport();
const VulkanQueueFamilyList& getEmptyVulkanQueueFamilyList();
const VulkanDescriptorSetLayoutList& getEmptyVulkanDescriptorSetLayoutList();
const VulkanQueueFamilyToQueueCountMap&
getDefaultVulkanQueueFamilyToQueueCountMap();
const std::vector<VulkanExternalMemoryHandleType>
getSupportedVulkanExternalMemoryHandleTypeList();
const std::vector<VulkanExternalSemaphoreHandleType>
getSupportedVulkanExternalSemaphoreHandleTypeList();
const std::vector<VulkanFormat> getSupportedVulkanFormatList();

uint32_t getVulkanFormatElementSize(VulkanFormat format);
const char* getVulkanFormatGLSLFormat(VulkanFormat format);
const char* getVulkanFormatGLSLTypePrefix(VulkanFormat format);

std::string prepareVulkanShader(
    std::string shaderCode,
    const std::map<std::string, std::string>& patternToSubstituteMap);

std::ostream& operator<<(std::ostream& os,
                         VulkanMemoryTypeProperty memoryTypeProperty);
std::ostream&
operator<<(std::ostream& os,
           VulkanExternalMemoryHandleType externalMemoryHandleType);
std::ostream&
operator<<(std::ostream& os,
           VulkanExternalSemaphoreHandleType externalSemaphoreHandleType);
std::ostream& operator<<(std::ostream& os, VulkanFormat format);

std::vector<char> readFile(const std::string& filename);
#endif // _vulkan_utility_hpp_
