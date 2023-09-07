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


#include <algorithm>
#include "vulkan_list_map.hpp"
#include "vulkan_utility.hpp"
#include "vulkan_wrapper.hpp"

/////////////////////////////////////////////
// VulkanPhysicalDeviceList implementation //
/////////////////////////////////////////////

VulkanPhysicalDeviceList::VulkanPhysicalDeviceList(
    const VulkanPhysicalDeviceList &physicalDeviceList)
{}

VulkanPhysicalDeviceList::VulkanPhysicalDeviceList() {}

VulkanPhysicalDeviceList::~VulkanPhysicalDeviceList() {}

/////////////////////////////////////////
// VulkanMemoryHeapList implementation //
/////////////////////////////////////////

VulkanMemoryHeapList::VulkanMemoryHeapList(
    const VulkanMemoryHeapList &memoryHeapList)
{}

VulkanMemoryHeapList::VulkanMemoryHeapList() {}

VulkanMemoryHeapList::~VulkanMemoryHeapList() {}

/////////////////////////////////////////
// VulkanMemoryTypeList implementation //
/////////////////////////////////////////

VulkanMemoryTypeList::VulkanMemoryTypeList(
    const VulkanMemoryTypeList &memoryTypeList)
{}

VulkanMemoryTypeList::VulkanMemoryTypeList() {}

VulkanMemoryTypeList::~VulkanMemoryTypeList() {}

//////////////////////////////////////////
// VulkanQueueFamilyList implementation //
//////////////////////////////////////////

VulkanQueueFamilyList::VulkanQueueFamilyList(
    const VulkanQueueFamilyList &queueFamilyList)
{}

VulkanQueueFamilyList::VulkanQueueFamilyList() {}

VulkanQueueFamilyList::~VulkanQueueFamilyList() {}

/////////////////////////////////////////////////////
// VulkanQueueFamilyToQueueCountMap implementation //
/////////////////////////////////////////////////////

VulkanQueueFamilyToQueueCountMap::VulkanQueueFamilyToQueueCountMap(
    const VulkanQueueFamilyToQueueCountMap &queueFamilyToQueueCountMap)
{}

VulkanQueueFamilyToQueueCountMap::VulkanQueueFamilyToQueueCountMap(
    uint32_t numQueuesPerFamily)
{
    uint32_t maxQueueFamilyCount = 0;
    const VulkanPhysicalDeviceList &physicalDeviceList =
        getVulkanInstance().getPhysicalDeviceList();
    for (size_t pdIdx = 0; pdIdx < physicalDeviceList.size(); pdIdx++)
    {
        maxQueueFamilyCount = std::max(
            maxQueueFamilyCount,
            (uint32_t)physicalDeviceList[pdIdx].getQueueFamilyList().size());
    }

    for (uint32_t qfIdx = 0; qfIdx < maxQueueFamilyCount; qfIdx++)
    {
        insert(qfIdx, numQueuesPerFamily);
    }
}

VulkanQueueFamilyToQueueCountMap::~VulkanQueueFamilyToQueueCountMap() {}

////////////////////////////////////////////////////
// VulkanQueueFamilyToQueueListMap implementation //
////////////////////////////////////////////////////

VulkanQueueFamilyToQueueListMap::VulkanQueueFamilyToQueueListMap(
    const VulkanQueueFamilyToQueueListMap &queueFamilyToQueueMap)
{}

VulkanQueueFamilyToQueueListMap::VulkanQueueFamilyToQueueListMap() {}

VulkanQueueFamilyToQueueListMap::~VulkanQueueFamilyToQueueListMap() {}

void VulkanQueueFamilyToQueueListMap::insert(uint32_t key,
                                             VulkanQueueList &queueList)
{
    m_map.insert(std::pair<uint32_t, std::reference_wrapper<VulkanQueueList>>(
        key, std::reference_wrapper<VulkanQueueList>(queueList)));
}

VulkanQueueList &VulkanQueueFamilyToQueueListMap::operator[](uint32_t key)
{
    return m_map.at(key).get();
}

////////////////////////////////////
// VulkanQueueList implementation //
////////////////////////////////////

VulkanQueueList::VulkanQueueList(const VulkanQueueList &queueList) {}

VulkanQueueList::VulkanQueueList() {}

VulkanQueueList::~VulkanQueueList() {}

/////////////////////////////////////////////////////////
// VulkanDescriptorSetLayoutBindingList implementation //
/////////////////////////////////////////////////////////

VulkanDescriptorSetLayoutBindingList::VulkanDescriptorSetLayoutBindingList(
    const VulkanDescriptorSetLayoutBindingList &descriptorSetLayoutBindingList)
{}

VulkanDescriptorSetLayoutBindingList::VulkanDescriptorSetLayoutBindingList() {}

void VulkanDescriptorSetLayoutBindingList::addBinding(
    size_t binding, VulkanDescriptorType descriptorType,
    uint32_t descriptorCount, VulkanShaderStage shaderStage)
{
    VulkanDescriptorSetLayoutBinding *descriptorSetLayoutBinding =
        new VulkanDescriptorSetLayoutBinding(binding, descriptorType,
                                             descriptorCount, shaderStage);
    add(*descriptorSetLayoutBinding);
}

VulkanDescriptorSetLayoutBindingList::VulkanDescriptorSetLayoutBindingList(
    size_t numDescriptorSetLayoutBindings, VulkanDescriptorType descriptorType,
    uint32_t descriptorCount, VulkanShaderStage shaderStage)
{
    for (size_t idx = 0; idx < numDescriptorSetLayoutBindings; idx++)
    {
        VulkanDescriptorSetLayoutBinding *descriptorSetLayoutBinding =
            new VulkanDescriptorSetLayoutBinding((uint32_t)idx, descriptorType,
                                                 descriptorCount, shaderStage);
        add(*descriptorSetLayoutBinding);
    }
}

VulkanDescriptorSetLayoutBindingList::VulkanDescriptorSetLayoutBindingList(
    VulkanDescriptorType descriptorType0, uint32_t descriptorCount0,
    VulkanDescriptorType descriptorType1, uint32_t descriptorCount1,
    VulkanShaderStage shaderStage)
{
    for (uint32_t idx = 0; idx < descriptorCount0; idx++)
    {
        VulkanDescriptorSetLayoutBinding *descriptorSetLayoutBinding0 =
            new VulkanDescriptorSetLayoutBinding(idx, descriptorType0, 1,
                                                 shaderStage);
        add(*descriptorSetLayoutBinding0);
    }
    for (uint32_t idx = 0; idx < descriptorCount1; idx++)
    {
        VulkanDescriptorSetLayoutBinding *descriptorSetLayoutBinding1 =
            new VulkanDescriptorSetLayoutBinding(
                descriptorCount0 + idx, descriptorType1, 1, shaderStage);
        add(*descriptorSetLayoutBinding1);
    }
}

VulkanDescriptorSetLayoutBindingList::~VulkanDescriptorSetLayoutBindingList()
{
    for (size_t idx = 0; idx < m_wrapperList.size(); idx++)
    {
        VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding =
            m_wrapperList[idx];
        delete &descriptorSetLayoutBinding;
    }
}

//////////////////////////////////////////////////
// VulkanDescriptorSetLayoutList implementation //
//////////////////////////////////////////////////

VulkanDescriptorSetLayoutList::VulkanDescriptorSetLayoutList(
    const VulkanDescriptorSetLayoutList &descriptorSetLayoutList)
{}

VulkanDescriptorSetLayoutList::VulkanDescriptorSetLayoutList() {}

VulkanDescriptorSetLayoutList::~VulkanDescriptorSetLayoutList() {}

////////////////////////////////////////////
// VulkanCommandBufferList implementation //
////////////////////////////////////////////

VulkanCommandBufferList::VulkanCommandBufferList(
    const VulkanCommandBufferList &commandBufferList)
{}

VulkanCommandBufferList::VulkanCommandBufferList() {}

VulkanCommandBufferList::VulkanCommandBufferList(
    size_t numCommandBuffers, const VulkanDevice &device,
    const VulkanCommandPool &commandPool)
{
    for (size_t idx = 0; idx < numCommandBuffers; idx++)
    {
        VulkanCommandBuffer *commandBuffer =
            new VulkanCommandBuffer(device, commandPool);
        add(*commandBuffer);
    }
}

VulkanCommandBufferList::~VulkanCommandBufferList()
{
    for (size_t idx = 0; idx < m_wrapperList.size(); idx++)
    {
        VulkanCommandBuffer &commandBuffer = m_wrapperList[idx];
        delete &commandBuffer;
    }
}

/////////////////////////////////////
// VulkanBufferList implementation //
/////////////////////////////////////

VulkanBufferList::VulkanBufferList(const VulkanBufferList &bufferList) {}

VulkanBufferList::VulkanBufferList(
    size_t numBuffers, const VulkanDevice &device, uint64_t size,
    VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanBufferUsage bufferUsage, VulkanSharingMode sharingMode,
    const VulkanQueueFamilyList &queueFamilyList)
{
    for (size_t bIdx = 0; bIdx < numBuffers; bIdx++)
    {
        VulkanBuffer *buffer =
            new VulkanBuffer(device, size, externalMemoryHandleType,
                             bufferUsage, sharingMode, queueFamilyList);
        add(*buffer);
    }
}

VulkanBufferList::~VulkanBufferList()
{
    for (size_t bIdx = 0; bIdx < m_wrapperList.size(); bIdx++)
    {
        VulkanBuffer &buffer = m_wrapperList[bIdx];
        delete &buffer;
    }
}

//////////////////////////////////////
// VulkanImage2DList implementation //
//////////////////////////////////////

VulkanImage2DList::VulkanImage2DList(const VulkanImage2DList &image2DList) {}

VulkanImage2DList::VulkanImage2DList(
    size_t numImages, std::vector<VulkanDeviceMemory *> &deviceMemory,
    uint64_t baseOffset, uint64_t interImageOffset, const VulkanDevice &device,
    VulkanFormat format, uint32_t width, uint32_t height, uint32_t mipLevels,
    VulkanImageTiling vulkanImageTiling,
    VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanImageCreateFlag imageCreateFlag, VulkanImageUsage imageUsage,
    VulkanSharingMode sharingMode)
{
    for (size_t i2DIdx = 0; i2DIdx < numImages; i2DIdx++)
    {
        VulkanImage2D *image2D = new VulkanImage2D(
            device, format, width, height, vulkanImageTiling, mipLevels,
            externalMemoryHandleType, imageCreateFlag, imageUsage, sharingMode);
        add(*image2D);
        deviceMemory[i2DIdx]->bindImage(
            *image2D, baseOffset + (i2DIdx * interImageOffset));
    }
}

VulkanImage2DList::VulkanImage2DList(
    size_t numImages, const VulkanDevice &device, VulkanFormat format,
    uint32_t width, uint32_t height, VulkanImageTiling vulkanImageTiling,
    uint32_t mipLevels, VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanImageCreateFlag imageCreateFlag, VulkanImageUsage imageUsage,
    VulkanSharingMode sharingMode)
{
    for (size_t bIdx = 0; bIdx < numImages; bIdx++)
    {
        VulkanImage2D *image2D = new VulkanImage2D(
            device, format, width, height, vulkanImageTiling, mipLevels,
            externalMemoryHandleType, imageCreateFlag, imageUsage, sharingMode);
        add(*image2D);
    }
}

VulkanImage2DList::~VulkanImage2DList()
{
    for (size_t i2DIdx = 0; i2DIdx < m_wrapperList.size(); i2DIdx++)
    {
        VulkanImage2D &image2D = m_wrapperList[i2DIdx];
        delete &image2D;
    }
}

////////////////////////////////////////
// VulkanImageViewList implementation //
////////////////////////////////////////

VulkanImageViewList::VulkanImageViewList(const VulkanImageViewList &image2DList)
{}

VulkanImageViewList::VulkanImageViewList(const VulkanDevice &device,
                                         const VulkanImage2DList &image2DList,
                                         bool createImageViewPerMipLevel)
{
    for (size_t i2DIdx = 0; i2DIdx < image2DList.size(); i2DIdx++)
    {
        if (createImageViewPerMipLevel)
        {
            for (uint32_t mipLevel = 0;
                 mipLevel < image2DList[i2DIdx].getNumMipLevels(); mipLevel++)
            {
                VulkanImageView *image2DView =
                    new VulkanImageView(device, image2DList[i2DIdx],
                                        VULKAN_IMAGE_VIEW_TYPE_2D, mipLevel, 1);
                add(*image2DView);
            }
        }
        else
        {
            VulkanImageView *image2DView = new VulkanImageView(
                device, image2DList[i2DIdx], VULKAN_IMAGE_VIEW_TYPE_2D);
            add(*image2DView);
        }
    }
}

VulkanImageViewList::~VulkanImageViewList()
{
    for (size_t ivIdx = 0; ivIdx < m_wrapperList.size(); ivIdx++)
    {
        VulkanImageView &imageView = m_wrapperList[ivIdx];
        delete &imageView;
    }
}

///////////////////////////////////////////
// VulkanDeviceMemoryList implementation //
///////////////////////////////////////////

VulkanDeviceMemoryList::VulkanDeviceMemoryList(
    const VulkanDeviceMemoryList &deviceMemoryList)
{}

VulkanDeviceMemoryList::VulkanDeviceMemoryList(
    size_t numImages, const VulkanImage2DList &image2DList,
    const VulkanDevice &device, const VulkanMemoryType &memoryType,
    VulkanExternalMemoryHandleType externalMemoryHandleType)
{
    for (size_t i2DIdx = 0; i2DIdx < image2DList.size(); i2DIdx++)
    {
        VulkanDeviceMemory *deviceMemory = new VulkanDeviceMemory(
            device, image2DList[i2DIdx], memoryType, externalMemoryHandleType);
        add(*deviceMemory);
        deviceMemory->bindImage(image2DList[i2DIdx]);
    }
}

VulkanDeviceMemoryList::~VulkanDeviceMemoryList()
{
    for (size_t dmIdx = 0; dmIdx < m_wrapperList.size(); dmIdx++)
    {
        VulkanDeviceMemory &deviceMemory = m_wrapperList[dmIdx];
        delete &deviceMemory;
    }
}

////////////////////////////////////////
// VulkanSemaphoreList implementation //
////////////////////////////////////////

VulkanSemaphoreList::VulkanSemaphoreList(
    const VulkanSemaphoreList &semaphoreList)
{}

VulkanSemaphoreList::VulkanSemaphoreList() {}

VulkanSemaphoreList::VulkanSemaphoreList(
    size_t numSemaphores, const VulkanDevice &device,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
    const std::wstring namePrefix)
{
    std::wstring name = L"";
    for (size_t idx = 0; idx < numSemaphores; idx++)
    {
        if (namePrefix.size())
        {
            const size_t maxNameSize = 256;
            wchar_t tempName[maxNameSize];
            swprintf(tempName, maxNameSize, L"%s%d", namePrefix.c_str(),
                     (int)idx);
            name = tempName;
        }
        VulkanSemaphore *semaphore =
            new VulkanSemaphore(device, externalSemaphoreHandleType, name);
        add(*semaphore);
    }
}

VulkanSemaphoreList::~VulkanSemaphoreList()
{
    for (size_t idx = 0; idx < m_wrapperList.size(); idx++)
    {
        VulkanSemaphore &Semaphore = m_wrapperList[idx];
        delete &Semaphore;
    }
}
