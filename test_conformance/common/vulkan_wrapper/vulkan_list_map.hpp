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

#ifndef _vulkan_list_map_hpp_
#define _vulkan_list_map_hpp_

#include <functional>
#include "vulkan_wrapper_types.hpp"
#include "vulkan_utility.hpp"
#include <iostream>
template <class VulkanWrapper, class VulkanNative> class VulkanList {
protected:
    std::vector<std::reference_wrapper<VulkanWrapper>> m_wrapperList;
    std::vector<std::reference_wrapper<const VulkanWrapper>> m_constWrapperList;
    std::vector<VulkanNative> m_nativeList;

    VulkanList(const VulkanList &list);
    VulkanList();
    virtual ~VulkanList();
    virtual void add(VulkanWrapper &wrapper);

public:
    virtual void add(const VulkanWrapper &wrapper);
    virtual size_t size() const;
    virtual const VulkanWrapper &operator[](size_t idx) const;
    virtual VulkanWrapper &operator[](size_t idx);
    virtual const VulkanNative *operator()() const;
};

template <class VulkanKey, class VulkanValue> class VulkanMap {
protected:
    std::map<VulkanKey, VulkanValue> m_map;

    VulkanMap(const VulkanMap &map);
    VulkanMap();
    virtual ~VulkanMap();

public:
    void insert(const VulkanKey &key, VulkanValue &value);
    const VulkanValue &operator[](const VulkanKey &key) const;
    VulkanValue &operator[](const VulkanKey &key);
};

class VulkanPhysicalDeviceList
    : public VulkanList<VulkanPhysicalDevice, VkPhysicalDevice> {
    friend class VulkanInstance;

protected:
    VulkanPhysicalDeviceList(
        const VulkanPhysicalDeviceList &physicalDeviceList);

public:
    VulkanPhysicalDeviceList();
    virtual ~VulkanPhysicalDeviceList();
};

class VulkanQueueFamilyList : public VulkanList<VulkanQueueFamily, uint32_t> {
    friend class VulkanPhysicalDevice;

protected:
    VulkanQueueFamilyList(const VulkanQueueFamilyList &queueFamilyList);

public:
    VulkanQueueFamilyList();
    virtual ~VulkanQueueFamilyList();
};

class VulkanMemoryHeapList : public VulkanList<VulkanMemoryHeap, uint32_t> {
    friend class VulkanPhysicalDevice;

protected:
    VulkanMemoryHeapList(const VulkanMemoryHeapList &memoryHeapList);

public:
    VulkanMemoryHeapList();
    virtual ~VulkanMemoryHeapList();
};

class VulkanMemoryTypeList : public VulkanList<VulkanMemoryType, uint32_t> {
    friend class VulkanPhysicalDevice;
    friend class VulkanBuffer;
    friend class VulkanImage;

protected:
    VulkanMemoryTypeList(const VulkanMemoryTypeList &memoryTypeList);

public:
    VulkanMemoryTypeList();
    virtual ~VulkanMemoryTypeList();
};

class VulkanQueueFamilyToQueueCountMap : public VulkanMap<uint32_t, uint32_t> {
protected:
    VulkanQueueFamilyToQueueCountMap(
        const VulkanQueueFamilyToQueueCountMap &queueFamilyToQueueCountMap);

public:
    VulkanQueueFamilyToQueueCountMap(uint32_t numQueuesPerFamily = 0);
    virtual ~VulkanQueueFamilyToQueueCountMap();
};

class VulkanQueueList : public VulkanList<VulkanQueue, VkQueue> {
    friend class VulkanDevice;

protected:
    VulkanQueueList(const VulkanQueueList &queueList);

public:
    VulkanQueueList();
    virtual ~VulkanQueueList();
};

class VulkanQueueFamilyToQueueListMap
    : public VulkanMap<uint32_t, std::reference_wrapper<VulkanQueueList>> {
protected:
    VulkanQueueFamilyToQueueListMap(
        const VulkanQueueFamilyToQueueListMap &queueFamilyToQueueMap);

public:
    VulkanQueueFamilyToQueueListMap();
    virtual ~VulkanQueueFamilyToQueueListMap();
    void insert(uint32_t key, VulkanQueueList &queueList);
    VulkanQueueList &operator[](uint32_t key);
};

class VulkanDescriptorSetLayoutBindingList
    : public VulkanList<VulkanDescriptorSetLayoutBinding,
                        VkDescriptorSetLayoutBinding> {
protected:
    VulkanDescriptorSetLayoutBindingList(
        const VulkanDescriptorSetLayoutBindingList
            &descriptorSetLayoutBindingList);

public:
    VulkanDescriptorSetLayoutBindingList();
    VulkanDescriptorSetLayoutBindingList(
        size_t numDescriptorSetLayoutBindings,
        VulkanDescriptorType descriptorType, uint32_t descriptorCount = 1,
        VulkanShaderStage shaderStage = VULKAN_SHADER_STAGE_COMPUTE);
    VulkanDescriptorSetLayoutBindingList(
        VulkanDescriptorType descriptorType0, uint32_t descriptorCount0,
        VulkanDescriptorType descriptorType1, uint32_t descriptorCount1,
        VulkanShaderStage shaderStage = VULKAN_SHADER_STAGE_COMPUTE);
    virtual ~VulkanDescriptorSetLayoutBindingList();
};

class VulkanDescriptorSetLayoutList
    : public VulkanList<VulkanDescriptorSetLayout, VkDescriptorSetLayout> {
protected:
    VulkanDescriptorSetLayoutList(
        const VulkanDescriptorSetLayoutList &descriptorSetLayoutList);

public:
    VulkanDescriptorSetLayoutList();
    virtual ~VulkanDescriptorSetLayoutList();
};

class VulkanCommandBufferList
    : public VulkanList<VulkanCommandBuffer, VkCommandBuffer> {
protected:
    VulkanCommandBufferList(const VulkanCommandBufferList &commandBufferList);

public:
    VulkanCommandBufferList();
    VulkanCommandBufferList(size_t numCommandBuffers,
                            const VulkanDevice &device,
                            const VulkanCommandPool &commandPool);
    virtual ~VulkanCommandBufferList();
};

class VulkanBufferList : public VulkanList<VulkanBuffer, VkBuffer> {
protected:
    VulkanBufferList(const VulkanBufferList &bufferList);

public:
    VulkanBufferList(
        size_t numBuffers, const VulkanDevice &device, uint64_t size,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanBufferUsage bufferUsage =
            VULKAN_BUFFER_USAGE_STORAGE_BUFFER_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE,
        const VulkanQueueFamilyList &queueFamilyList =
            getEmptyVulkanQueueFamilyList());
    virtual ~VulkanBufferList();
};

class VulkanImage2DList : public VulkanList<VulkanImage2D, VkImage> {
protected:
    VulkanImage2DList(const VulkanImage2DList &image2DList);

public:
    VulkanImage2DList(
        size_t numImages, std::vector<VulkanDeviceMemory *> &deviceMemory,
        uint64_t baseOffset, uint64_t interImageOffset,
        const VulkanDevice &device, VulkanFormat format, uint32_t width,
        uint32_t height, uint32_t mipLevels,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlag = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    VulkanImage2DList(
        size_t numImages, const VulkanDevice &device, VulkanFormat format,
        uint32_t width, uint32_t height, uint32_t mipLevels = 1,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlag = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    virtual ~VulkanImage2DList();
};

class VulkanImageViewList : public VulkanList<VulkanImageView, VkImageView> {
protected:
    VulkanImageViewList(const VulkanImageViewList &imageViewList);

public:
    VulkanImageViewList(const VulkanDevice &device,
                        const VulkanImage2DList &image2DList,
                        bool createImageViewPerMipLevel = true);
    virtual ~VulkanImageViewList();
};

class VulkanDeviceMemoryList
    : public VulkanList<VulkanDeviceMemory, VkDeviceMemory> {
protected:
    VulkanDeviceMemoryList(const VulkanDeviceMemoryList &deviceMemoryList);

public:
    VulkanDeviceMemoryList(
        size_t numImages, const VulkanImage2DList &image2DList,
        const VulkanDevice &device, const VulkanMemoryType &memoryType,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE);
    virtual ~VulkanDeviceMemoryList();
};

class VulkanSemaphoreList : public VulkanList<VulkanSemaphore, VkSemaphore> {
protected:
    VulkanSemaphoreList(const VulkanSemaphoreList &semaphoreList);

public:
    VulkanSemaphoreList();
    VulkanSemaphoreList(
        size_t numSemaphores, const VulkanDevice &device,
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType =
            VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE,
        const std::wstring namePrefix = L"");
    virtual ~VulkanSemaphoreList();
};

///////////////////////////////
// VulkanList implementation //
///////////////////////////////

template <class VulkanWrapper, class VulkanNative>
VulkanList<VulkanWrapper, VulkanNative>::VulkanList(const VulkanList &list)
    : m_wrapperList(list.m_wrapperList),
      m_constWrapperList(list.m_constWrapperList),
      m_nativeList(list.m_nativeList)
{}

template <class VulkanWrapper, class VulkanNative>
VulkanList<VulkanWrapper, VulkanNative>::VulkanList()
{}

template <class VulkanWrapper, class VulkanNative>
VulkanList<VulkanWrapper, VulkanNative>::~VulkanList()
{}

template <class VulkanWrapper, class VulkanNative>
void VulkanList<VulkanWrapper, VulkanNative>::add(VulkanWrapper &wrapper)
{

    if (m_constWrapperList.size() != size_t(0))
    {
        std::cout << "This list can only contain externally allocated objects"
                  << std::endl;
        return;
    }
    m_wrapperList.push_back(std::reference_wrapper<VulkanWrapper>(wrapper));
    m_nativeList.push_back((VulkanNative)wrapper);
}

template <class VulkanWrapper, class VulkanNative>
void VulkanList<VulkanWrapper, VulkanNative>::add(const VulkanWrapper &wrapper)
{
    if (m_wrapperList.size() != size_t(0))
    {
        std::cout << "This list cannot contain externally allocated objects"
                  << std::endl;
        return;
    }

    m_constWrapperList.push_back(
        std::reference_wrapper<const VulkanWrapper>(wrapper));
    m_nativeList.push_back((VulkanNative)wrapper);
}

template <class VulkanWrapper, class VulkanNative>
size_t VulkanList<VulkanWrapper, VulkanNative>::size() const
{
    return (m_wrapperList.size() > 0) ? m_wrapperList.size()
                                      : m_constWrapperList.size();
}

template <class VulkanWrapper, class VulkanNative>
const VulkanWrapper &
    VulkanList<VulkanWrapper, VulkanNative>::operator[](size_t idx) const
{
    if (idx < size())
    {
        // CHECK_LT(idx, size());
        return (m_wrapperList.size() > 0) ? m_wrapperList[idx].get()
                                          : m_constWrapperList[idx].get();
    }
}

template <class VulkanWrapper, class VulkanNative>
VulkanWrapper &VulkanList<VulkanWrapper, VulkanNative>::operator[](size_t idx)
{
    // CHECK_LT(idx, m_wrapperList.size());
    return m_wrapperList[idx].get();
}

template <class VulkanWrapper, class VulkanNative>
const VulkanNative *VulkanList<VulkanWrapper, VulkanNative>::operator()() const
{
    return m_nativeList.data();
}

//////////////////////////////
// VulkanMap implementation //
//////////////////////////////

template <class VulkanKey, class VulkanValue>
VulkanMap<VulkanKey, VulkanValue>::VulkanMap(const VulkanMap &map)
    : m_map(map.m_map)
{}

template <class VulkanKey, class VulkanValue>
VulkanMap<VulkanKey, VulkanValue>::VulkanMap()
{}

template <class VulkanKey, class VulkanValue>
VulkanMap<VulkanKey, VulkanValue>::~VulkanMap()
{}

template <class VulkanKey, class VulkanValue>
void VulkanMap<VulkanKey, VulkanValue>::insert(const VulkanKey &key,
                                               VulkanValue &value)
{
    m_map.insert(std::pair<VulkanKey, std::reference_wrapper<VulkanValue>>(
        key, std::reference_wrapper<VulkanValue>(value)));
}

template <class VulkanKey, class VulkanValue>
const VulkanValue &
    VulkanMap<VulkanKey, VulkanValue>::operator[](const VulkanKey &key) const
{
    return m_map.at(key);
}

template <class VulkanKey, class VulkanValue>
VulkanValue &VulkanMap<VulkanKey, VulkanValue>::operator[](const VulkanKey &key)
{
    return m_map.at(key);
}

#endif // _vulkan_list_map_hpp_
