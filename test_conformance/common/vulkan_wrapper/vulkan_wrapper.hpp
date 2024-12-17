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

#ifndef _vulkan_wrapper_hpp_
#define _vulkan_wrapper_hpp_

#include <vulkan/vulkan.h>
#include "vulkan_wrapper_types.hpp"
#include "vulkan_list_map.hpp"
#include "vulkan_api_list.hpp"
#include <memory>
#include <cassert>

class VulkanInstance {
    friend const VulkanInstance &getVulkanInstance();

protected:
    VkInstance m_vkInstance;
    VulkanPhysicalDeviceList m_physicalDeviceList;

    VulkanInstance();
    VulkanInstance(const VulkanInstance &);
    virtual ~VulkanInstance();

public:
    const VulkanPhysicalDeviceList &getPhysicalDeviceList() const;
    operator VkInstance() const;
};

class VulkanPhysicalDevice {
    friend class VulkanInstance;

protected:
    VkPhysicalDevice m_vkPhysicalDevice;
    VkPhysicalDeviceProperties m_vkPhysicalDeviceProperties;
    uint8_t m_vkDeviceUUID[VK_UUID_SIZE];
    uint8_t m_vkDeviceLUID[VK_LUID_SIZE];
    uint32_t m_vkDeviceNodeMask;
    VkPhysicalDeviceFeatures m_vkPhysicalDeviceFeatures;
    VkPhysicalDeviceMemoryProperties m_vkPhysicalDeviceMemoryProperties;
    VulkanQueueFamilyList m_queueFamilyList;
    VulkanMemoryHeapList m_memoryHeapList;
    VulkanMemoryTypeList m_memoryTypeList;
    std::vector<VkExtensionProperties> m_extensions;


    VulkanPhysicalDevice(const VulkanPhysicalDevice &physicalDevice);
    VulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice);
    virtual ~VulkanPhysicalDevice();

public:
    bool hasExtension(const char *extension_name) const;
    const VulkanQueueFamilyList &getQueueFamilyList() const;
    const VulkanMemoryHeapList &getMemoryHeapList() const;
    const VulkanMemoryTypeList &getMemoryTypeList() const;
    const uint8_t *getUUID() const;
    const uint8_t *getLUID() const;
    uint32_t getNodeMask() const;
    operator VkPhysicalDevice() const;
};

class VulkanMemoryHeap {
    friend class VulkanPhysicalDevice;

protected:
    uint32_t m_memoryHeapIndex;
    uint64_t m_size;
    VulkanMemoryHeapFlag m_memoryHeapFlag;

    VulkanMemoryHeap(const VulkanMemoryHeap &memoryHeap);
    VulkanMemoryHeap(uint32_t m_memoryHeapIndex, uint64_t m_size,
                     VulkanMemoryHeapFlag m_memoryHeapFlag);
    virtual ~VulkanMemoryHeap();

public:
    uint64_t getSize() const;
    VulkanMemoryHeapFlag getMemoryHeapFlag() const;
    operator uint32_t() const;
};

class VulkanMemoryType {
    friend class VulkanPhysicalDevice;

protected:
    uint32_t m_memoryTypeIndex;
    const VulkanMemoryTypeProperty m_memoryTypeProperty;
    const VulkanMemoryHeap &m_memoryHeap;

    VulkanMemoryType(const VulkanMemoryType &memoryType);
    VulkanMemoryType(uint32_t memoryTypeIndex,
                     VulkanMemoryTypeProperty memoryTypeProperty,
                     const VulkanMemoryHeap &memoryHeap);
    virtual ~VulkanMemoryType();

public:
    VulkanMemoryTypeProperty getMemoryTypeProperty() const;
    const VulkanMemoryHeap &getMemoryHeap() const;
    operator uint32_t() const;
};

class VulkanQueueFamily {
    friend class VulkanPhysicalDevice;

protected:
    uint32_t m_queueFamilyIndex;
    VkQueueFamilyProperties m_vkQueueFamilyProperties;

    VulkanQueueFamily(const VulkanQueueFamily &queueFamily);
    VulkanQueueFamily(uint32_t queueFamilyIndex,
                      VkQueueFamilyProperties vkQueueFamilyProperties);
    virtual ~VulkanQueueFamily();

public:
    uint32_t getQueueFlags() const;
    uint32_t getQueueCount() const;
    operator uint32_t() const;
};

class VulkanDevice {
protected:
    const VulkanPhysicalDevice &m_physicalDevice;
    VkDevice m_vkDevice;
    VulkanQueueFamilyToQueueListMap m_queueFamilyIndexToQueueListMap;

    VulkanDevice(const VulkanDevice &device);

public:
    VulkanDevice(
        const VulkanPhysicalDevice &physicalDevice = getVulkanPhysicalDevice(),
        const VulkanQueueFamilyToQueueCountMap &queueFamilyToQueueCountMap =
            getDefaultVulkanQueueFamilyToQueueCountMap());
    virtual ~VulkanDevice();
    const VulkanPhysicalDevice &getPhysicalDevice() const;
    VulkanQueue &
    getQueue(const VulkanQueueFamily &queueFamily /* = getVulkanQueueFamily()*/,
             uint32_t queueIndex = 0);
    operator VkDevice() const;
};

class VulkanFence {
    friend class VulkanQueue;

protected:
    VkFence fence;
    VkDevice device;

public:
    VulkanFence(const VulkanDevice &device);
    virtual ~VulkanFence();
    void reset();
    void wait();
};

class VulkanQueue {
    friend class VulkanDevice;

protected:
    VkQueue m_vkQueue;

    VulkanQueue(VkQueue vkQueue);
    VulkanQueue(const VulkanQueue &queue);
    virtual ~VulkanQueue();

public:
    const VulkanQueueFamily &getQueueFamily();
    void submit(const VulkanCommandBuffer &commandBuffer,
                const std::shared_ptr<VulkanFence> &fence);
    void submit(const VulkanSemaphoreList &waitSemaphoreList,
                const VulkanCommandBufferList &commandBufferList,
                const VulkanSemaphoreList &signalSemaphoreList);
    void submit(const VulkanSemaphore &waitSemaphore,
                const VulkanCommandBuffer &commandBuffer,
                const VulkanSemaphore &signalSemaphore);
    void submit(const VulkanCommandBuffer &commandBuffer,
                const VulkanSemaphore &signalSemaphore);
    void submit(const VulkanCommandBuffer &commandBuffer);
    void waitIdle();
    operator VkQueue() const;
};

class VulkanDescriptorSetLayoutBinding {
protected:
    VkDescriptorSetLayoutBinding m_vkDescriptorSetLayoutBinding;

    VulkanDescriptorSetLayoutBinding(
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding);

public:
    VulkanDescriptorSetLayoutBinding(
        uint32_t binding, VulkanDescriptorType descriptorType,
        uint32_t descriptorCount = 1,
        VulkanShaderStage shaderStage = VULKAN_SHADER_STAGE_COMPUTE);
    virtual ~VulkanDescriptorSetLayoutBinding();
    operator VkDescriptorSetLayoutBinding() const;
};

class VulkanDescriptorSetLayout {
protected:
    const VulkanDevice &m_device;
    VkDescriptorSetLayout m_vkDescriptorSetLayout;

    VulkanDescriptorSetLayout(
        const VulkanDescriptorSetLayout &descriptorSetLayout);
    void
    VulkanDescriptorSetLayoutCommon(const VulkanDescriptorSetLayoutBindingList
                                        &descriptorSetLayoutBindingList);

public:
    VulkanDescriptorSetLayout(
        const VulkanDevice &device,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding);
    VulkanDescriptorSetLayout(
        const VulkanDevice &device,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding0,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding1);
    VulkanDescriptorSetLayout(const VulkanDevice &device,
                              const VulkanDescriptorSetLayoutBindingList
                                  &descriptorSetLayoutBindingList);
    virtual ~VulkanDescriptorSetLayout();
    operator VkDescriptorSetLayout() const;
};

class VulkanPipelineLayout {
protected:
    const VulkanDevice &m_device;
    VkPipelineLayout m_vkPipelineLayout;

    VulkanPipelineLayout(const VulkanPipelineLayout &pipelineLayout);
    void VulkanPipelineLayoutCommon(
        const VulkanDescriptorSetLayoutList &descriptorSetLayoutList);

public:
    VulkanPipelineLayout(const VulkanDevice &device,
                         const VulkanDescriptorSetLayout &descriptorSetLayout);
    VulkanPipelineLayout(
        const VulkanDevice &device,
        const VulkanDescriptorSetLayoutList &descriptorSetLayoutList =
            getEmptyVulkanDescriptorSetLayoutList());
    virtual ~VulkanPipelineLayout();
    operator VkPipelineLayout() const;
};

class VulkanShaderModule {
protected:
    const VulkanDevice &m_device;
    VkShaderModule m_vkShaderModule;

    VulkanShaderModule(const VulkanShaderModule &shaderModule);

public:
    VulkanShaderModule(const VulkanDevice &device,
                       const std::vector<char> &code);
    virtual ~VulkanShaderModule();
    operator VkShaderModule() const;
};

class VulkanPipeline {
protected:
    const VulkanDevice &m_device;
    VkPipeline m_vkPipeline;

    VulkanPipeline(const VulkanPipeline &pipeline);

public:
    VulkanPipeline(const VulkanDevice &device);
    virtual ~VulkanPipeline();
    virtual VulkanPipelineBindPoint getPipelineBindPoint() const = 0;
    operator VkPipeline() const;
};

class VulkanComputePipeline : public VulkanPipeline {
protected:
    VulkanComputePipeline(const VulkanComputePipeline &computePipeline);

public:
    VulkanComputePipeline(const VulkanDevice &device,
                          const VulkanPipelineLayout &pipelineLayout,
                          const VulkanShaderModule &shaderModule,
                          const std::string &entryFuncName = "main");
    virtual ~VulkanComputePipeline();
    VulkanPipelineBindPoint getPipelineBindPoint() const;
};

class VulkanDescriptorPool {
protected:
    const VulkanDevice &m_device;
    VkDescriptorPool m_vkDescriptorPool;

    VulkanDescriptorPool(const VulkanDescriptorPool &descriptorPool);
    void VulkanDescriptorPoolCommon(const VulkanDescriptorSetLayoutBindingList
                                        &descriptorSetLayoutBindingList);

public:
    VulkanDescriptorPool(
        const VulkanDevice &device,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding);
    VulkanDescriptorPool(
        const VulkanDevice &device,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding0,
        const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding1);
    VulkanDescriptorPool(const VulkanDevice &device,
                         const VulkanDescriptorSetLayoutBindingList
                             &descriptorSetLayoutBindingList);
    virtual ~VulkanDescriptorPool();
    operator VkDescriptorPool() const;
};

class VulkanDescriptorSet {
protected:
    const VulkanDevice &m_device;
    const VulkanDescriptorPool &m_descriptorPool;
    VkDescriptorSet m_vkDescriptorSet;

    VulkanDescriptorSet(const VulkanDescriptorSet &descriptorSet);

public:
    VulkanDescriptorSet(const VulkanDevice &device,
                        const VulkanDescriptorPool &descriptorPool,
                        const VulkanDescriptorSetLayout &descriptorSetLayout);
    virtual ~VulkanDescriptorSet();
    void update(uint32_t binding, const VulkanBuffer &buffer);
    void updateArray(uint32_t binding, unsigned numBuffers,
                     const VulkanBufferList &buffers);
    void update(uint32_t binding, const VulkanImageView &imageView);
    void updateArray(uint32_t binding,
                     const VulkanImageViewList &imageViewList);
    operator VkDescriptorSet() const;
};

class VulkanOffset3D {
protected:
    VkOffset3D m_vkOffset3D;

public:
    VulkanOffset3D(const VulkanOffset3D &extent3D);
    VulkanOffset3D(uint32_t x = 0, uint32_t y = 0, uint32_t z = 0);
    virtual ~VulkanOffset3D();
    uint32_t getX() const;
    uint32_t getY() const;
    uint32_t getZ() const;
    operator VkOffset3D() const;
};

class VulkanExtent3D {
protected:
    VkExtent3D m_vkExtent3D;

public:
    VulkanExtent3D(const VulkanExtent3D &extent3D);
    VulkanExtent3D(uint32_t width, uint32_t height = 1, uint32_t depth = 1);
    virtual ~VulkanExtent3D();
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getDepth() const;
    operator VkExtent3D() const;
};

class VulkanCommandPool {
protected:
    const VulkanDevice &m_device;
    VkCommandPool m_vkCommandPool;

    VulkanCommandPool(const VulkanCommandPool &commandPool);

public:
    VulkanCommandPool(
        const VulkanDevice &device,
        const VulkanQueueFamily &queueFamily = getVulkanQueueFamily());
    virtual ~VulkanCommandPool();
    operator VkCommandPool() const;
};

class VulkanCommandBuffer {
protected:
    const VulkanDevice &m_device;
    const VulkanCommandPool &m_commandPool;
    VkCommandBuffer m_vkCommandBuffer;

    VulkanCommandBuffer(const VulkanCommandBuffer &commandBuffer);

public:
    VulkanCommandBuffer(const VulkanDevice &device,
                        const VulkanCommandPool &commandPool);
    virtual ~VulkanCommandBuffer();
    void begin();
    void bindPipeline(const VulkanPipeline &pipeline);
    void bindDescriptorSets(const VulkanPipeline &pipeline,
                            const VulkanPipelineLayout &pipelineLayout,
                            const VulkanDescriptorSet &descriptorSet);
    void pipelineBarrier(const VulkanImage2DList &image2DList,
                         VulkanImageLayout oldImageLayout,
                         VulkanImageLayout newImageLayout);
    void dispatch(uint32_t groupCountX, uint32_t groupCountY,
                  uint32_t groupCountZ);
    void fillBuffer(const VulkanBuffer &buffer, uint32_t data,
                    uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);
    void updateBuffer(const VulkanBuffer &buffer, void *pdata,
                      uint64_t offset = 0, uint64_t size = VK_WHOLE_SIZE);
    void copyBufferToImage(const VulkanBuffer &buffer, const VulkanImage &image,
                           VulkanImageLayout imageLayout =
                               VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    void copyBufferToImage(const VulkanBuffer &buffer, const VulkanImage &image,
                           uint64_t bufferOffset = 0, uint32_t mipLevel = 0,
                           uint32_t baseArrayLayer = 0, uint32_t layerCount = 1,
                           VulkanOffset3D offset3D = VulkanOffset3D(0, 0, 0),
                           VulkanExtent3D extent3D = VulkanExtent3D(0, 0, 0));
    void copyImageToBuffer(const VulkanImage &image, const VulkanBuffer &buffer,
                           uint64_t bufferOffset = 0, uint32_t mipLevel = 0,
                           uint32_t baseArrayLayer = 0, uint32_t layerCount = 1,
                           VulkanOffset3D offset3D = VulkanOffset3D(0, 0, 0),
                           VulkanExtent3D extent3D = VulkanExtent3D(0, 0, 0));
    void end();
    operator VkCommandBuffer() const;
};

class VulkanBuffer {
protected:
    const VulkanDevice &m_device;
    VkBuffer m_vkBuffer;
    uint64_t m_size;
    uint64_t m_alignment;
    bool m_dedicated;
    VulkanMemoryTypeList m_memoryTypeList;

    VulkanBuffer(const VulkanBuffer &buffer);

public:
    VulkanBuffer(const VulkanDevice &device, uint64_t size,
                 VulkanExternalMemoryHandleType externalMemoryHandleType =
                     VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
                 VulkanBufferUsage bufferUsage =
                     VULKAN_BUFFER_USAGE_STORAGE_BUFFER_TRANSFER_SRC_DST,
                 VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE,
                 const VulkanQueueFamilyList &queueFamilyList =
                     getEmptyVulkanQueueFamilyList());
    virtual ~VulkanBuffer();
    uint64_t getSize() const;
    uint64_t getAlignment() const;
    const VulkanMemoryTypeList &getMemoryTypeList() const;
    bool isDedicated() const;
    operator VkBuffer() const;
};

class VulkanImage {
protected:
    const VulkanDevice &m_device;
    const VulkanImageType m_imageType;
    const VulkanExtent3D m_extent3D;
    const VulkanFormat m_format;
    const uint32_t m_numMipLevels;
    const uint32_t m_numLayers;
    bool m_dedicated;
    VkImage m_vkImage;
    uint64_t m_size;
    uint64_t m_alignment;
    VulkanMemoryTypeList m_memoryTypeList;
    VkImageCreateInfo VulkanImageCreateInfo;
    VulkanImage(const VulkanImage &image);

public:
    VulkanImage(
        const VulkanDevice &device, VulkanImageType imageType,
        VulkanFormat format, const VulkanExtent3D &extent3D,
        uint32_t numMipLevels = 1, uint32_t arrayLayers = 1,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlags = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageTiling imageTiling = VULKAN_IMAGE_TILING_OPTIMAL,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    virtual ~VulkanImage();
    virtual VulkanExtent3D getExtent3D(uint32_t mipLevel = 0) const;
    VulkanFormat getFormat() const;
    uint32_t getNumMipLevels() const;
    uint32_t getNumLayers() const;
    uint64_t getSize() const;
    uint64_t getAlignment() const;
    bool isDedicated() const;
    const VulkanMemoryTypeList &getMemoryTypeList() const;
    VkImageCreateInfo getVkImageCreateInfo() const;
    operator VkImage() const;
};

class VulkanImage1D : public VulkanImage {
protected:
    VkImageView m_vkImageView;

public:
    VulkanImage1D(
        const VulkanDevice &device, VulkanFormat format, uint32_t width,
        VulkanImageTiling imageTiling, uint32_t numMipLevels = 1,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlag = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    virtual ~VulkanImage1D();
    virtual VulkanExtent3D getExtent3D(uint32_t mipLevel = 0) const;

    VulkanImage1D(const VulkanImage1D &image1D);
};

class VulkanImage3D : public VulkanImage {
protected:
    VkImageView m_vkImageView;

public:
    VulkanImage3D(
        const VulkanDevice &device, VulkanFormat format, uint32_t width,
        uint32_t height, uint32_t depth, VulkanImageTiling imageTiling,
        uint32_t numMipLevels = 1,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlag = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    virtual ~VulkanImage3D();
    virtual VulkanExtent3D getExtent3D(uint32_t mipLevel = 0) const;

    VulkanImage3D(const VulkanImage3D &image3D);
};

class VulkanImage2D : public VulkanImage {
protected:
    VkImageView m_vkImageView;

public:
    VulkanImage2D(
        const VulkanDevice &device, VulkanFormat format, uint32_t width,
        uint32_t height, VulkanImageTiling imageTiling,
        uint32_t numMipLevels = 1,
        VulkanExternalMemoryHandleType externalMemoryHandleType =
            VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
        VulkanImageCreateFlag imageCreateFlag = VULKAN_IMAGE_CREATE_FLAG_NONE,
        VulkanImageUsage imageUsage =
            VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST,
        VulkanSharingMode sharingMode = VULKAN_SHARING_MODE_EXCLUSIVE);
    virtual ~VulkanImage2D();
    virtual VulkanExtent3D getExtent3D(uint32_t mipLevel = 0) const;
    virtual VkSubresourceLayout getSubresourceLayout() const;

    VulkanImage2D(const VulkanImage2D &image2D);
};

class VulkanImageView {
protected:
    const VulkanDevice &m_device;
    VkImageView m_vkImageView;

    VulkanImageView(const VulkanImageView &imageView);

public:
    VulkanImageView(const VulkanDevice &device, const VulkanImage &image,
                    VulkanImageViewType imageViewType,
                    uint32_t baseMipLevel = 0,
                    uint32_t mipLevelCount = VULKAN_REMAINING_MIP_LEVELS,
                    uint32_t baseArrayLayer = 0,
                    uint32_t layerCount = VULKAN_REMAINING_ARRAY_LAYERS);
    virtual ~VulkanImageView();
    operator VkImageView() const;
};

class VulkanDeviceMemory {
protected:
    const VulkanDevice &m_device;
    VkDeviceMemory m_vkDeviceMemory;
    uint64_t m_size;
    bool m_isDedicated;
    const std::wstring m_name;

    VulkanDeviceMemory(const VulkanDeviceMemory &deviceMemory);

public:
    VulkanDeviceMemory(const VulkanDevice &device, uint64_t size,
                       const VulkanMemoryType &memoryType,
                       VulkanExternalMemoryHandleType externalMemoryHandleType =
                           VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
                       const std::wstring name = L"");
    VulkanDeviceMemory(const VulkanDevice &device, const VulkanImage &image,
                       const VulkanMemoryType &memoryType,
                       VulkanExternalMemoryHandleType externalMemoryHandleType =
                           VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
                       const std::wstring name = L"");
    VulkanDeviceMemory(const VulkanDevice &device, const VulkanBuffer &buffer,
                       const VulkanMemoryType &memoryType,
                       VulkanExternalMemoryHandleType externalMemoryHandleType =
                           VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE,
                       const std::wstring name = L"");
    virtual ~VulkanDeviceMemory();
    uint64_t getSize() const;
#ifdef _WIN32
    HANDLE
    getHandle(VulkanExternalMemoryHandleType externalMemoryHandleType) const;
#else
    int
    getHandle(VulkanExternalMemoryHandleType externalMemoryHandleType) const;
#endif
    bool isDedicated() const;
    void *map(size_t offset = 0, size_t size = VK_WHOLE_SIZE);
    void unmap();
    void bindBuffer(const VulkanBuffer &buffer, uint64_t offset = 0);
    void bindImage(const VulkanImage &image, uint64_t offset = 0);
    const std::wstring &getName() const;
    operator VkDeviceMemory() const;
};

class VulkanSemaphore {
    friend class VulkanQueue;

protected:
    const VulkanDevice &m_device;
    VkSemaphore m_vkSemaphore;
    const std::wstring m_name;

    VulkanSemaphore(const VulkanSemaphore &semaphore);

public:
    VulkanSemaphore(
        const VulkanDevice &device,
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType =
            VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE,
        const std::wstring name = L"");
    const VulkanDevice &getDevice() const;
    virtual ~VulkanSemaphore();
#ifdef _WIN32
    HANDLE getHandle(
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType) const;
#else
    int getHandle(
        VulkanExternalSemaphoreHandleType externalSemaphoreHandleType) const;
#endif
    const std::wstring &getName() const;
    operator VkSemaphore() const;
};

#define VK_FUNC_DECL(name) extern "C" PFN_##name _##name;
VK_FUNC_LIST
#if defined(_WIN32) || defined(_WIN64)
VK_WINDOWS_FUNC_LIST
#endif
#undef VK_FUNC_DECL

#endif // _vulkan_wrapper_hpp_
