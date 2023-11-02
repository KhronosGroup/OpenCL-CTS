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

#ifdef _WIN32
#include <Windows.h>
#include <dxgi1_2.h>
#include <aclapi.h>
#include <algorithm>
#endif
#include <vulkan/vulkan.h>
#include "vulkan_wrapper.hpp"
#if defined(__linux__) && !defined(__ANDROID__)
#include <gnu/libc-version.h>
#include <dlfcn.h>
#elif defined(__ANDROID__)
#include <dlfcn.h>
#endif
#if defined _WIN32
#define LoadFunction GetProcAddress
#elif defined __linux
#define LoadFunction dlsym
#endif

extern "C" {
#define VK_FUNC_DECL(name) PFN_##name _##name = NULL;
VK_FUNC_LIST
#if defined(_WIN32) || defined(_WIN64)
VK_WINDOWS_FUNC_LIST
#endif
#undef VK_FUNC_DECL
}

#define WAIVED 2
#define HANDLE_ERROR -1

#define CHECK_VK(call)                                                         \
    if (call != VK_SUCCESS) return call;
///////////////////////////////////
// VulkanInstance implementation //
///////////////////////////////////

VulkanInstance::VulkanInstance(const VulkanInstance &instance)
    : m_vkInstance(instance.m_vkInstance),
      m_physicalDeviceList(instance.m_physicalDeviceList)
{}

VulkanInstance::VulkanInstance(): m_vkInstance(VK_NULL_HANDLE)
{
#if defined(__linux__) && !defined(__ANDROID__)
    char *glibcVersion = strdup(gnu_get_libc_version());
    int majNum = (int)atoi(strtok(glibcVersion, "."));
    int minNum = (int)atoi(strtok(NULL, "."));
    free(glibcVersion);
    if ((majNum < 2) || (majNum == 2 && minNum < 17))
    {
        // WAIVE_TEST() << "Insufficient GLIBC version. Test waived!";
    }
#endif

#if defined(_WIN32) || defined(_WIN64)
    const char *vulkanLoaderLibraryName = "vulkan-1.dll";
#elif defined(__ANDROID__)
    const char *vulkanLoaderLibraryName = "libvulkan.so";
#else
    const char *vulkanLoaderLibraryName = "libvulkan.so.1";
#endif
#ifdef _WIN32
    HINSTANCE hDLL;
    hDLL = LoadLibrary(vulkanLoaderLibraryName);
    if (hDLL == NULL)
    {
        throw std::runtime_error("LoadLibrary failed!");
    }
    vkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)LoadFunction(hDLL, "vkGetInstanceProcAddr");
#else
#if !defined(__APPLE__)
    void *handle;
    handle = dlopen(vulkanLoaderLibraryName, RTLD_LAZY);
    if (!handle)
    {
        fputs(dlerror(), stderr);
        throw std::runtime_error("dlopen failed !!!");
    }
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)LoadFunction(
        handle, "vkGetInstanceProcAddr");
#endif
#endif
    if ((unsigned long long)vkGetInstanceProcAddr == (unsigned long long)NULL)
    {
        throw std::runtime_error("vkGetInstanceProcAddr() not found!");
    }
#define VK_GET_NULL_INSTANCE_PROC_ADDR(name)                                   \
    _##name = (PFN_##name)vkGetInstanceProcAddr(NULL, #name);

    if ((unsigned long long)vkGetInstanceProcAddr == (unsigned long long)NULL)
    {
        throw std::runtime_error("Couldn't obtain address for function");
    }
    VK_GET_NULL_INSTANCE_PROC_ADDR(vkEnumerateInstanceExtensionProperties);
    uint32_t instanceExtensionPropertiesCount;
    VkResult vkStatus = VK_SUCCESS;
    vkStatus = vkEnumerateInstanceExtensionProperties(
        NULL, &instanceExtensionPropertiesCount, NULL);
    // Something went wrong in vulkan initialization (most likely incompatible
    // device/driver combination)
    if (vkStatus == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        throw std::runtime_error(
            "Waiving vulkan test because "
            "vkEnumerateInstanceExtensionProperties failed.");
        // return WAIVED;
    }

    VK_GET_NULL_INSTANCE_PROC_ADDR(vkEnumerateInstanceVersion);
    VK_GET_NULL_INSTANCE_PROC_ADDR(vkEnumerateInstanceLayerProperties);
    VK_GET_NULL_INSTANCE_PROC_ADDR(vkCreateInstance);
#undef VK_GET_NULL_INSTANCE_PROC_ADDR

    VkApplicationInfo vkApplicationInfo = {};
    vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    vkApplicationInfo.pNext = NULL;
    vkApplicationInfo.pApplicationName = "Default app";
    vkApplicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    vkApplicationInfo.pEngineName = "No engine";
    vkApplicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    vkApplicationInfo.apiVersion = VK_API_VERSION_1_0;

    std::vector<const char *> enabledExtensionNameList;
    enabledExtensionNameList.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

    std::vector<VkExtensionProperties> vkExtensionPropertiesList(
        instanceExtensionPropertiesCount);
    vkEnumerateInstanceExtensionProperties(NULL,
                                           &instanceExtensionPropertiesCount,
                                           vkExtensionPropertiesList.data());

    for (size_t eenIdx = 0; eenIdx < enabledExtensionNameList.size(); eenIdx++)
    {
        bool isSupported = false;
        for (size_t epIdx = 0; epIdx < vkExtensionPropertiesList.size();
             epIdx++)
        {
            if (!strcmp(enabledExtensionNameList[eenIdx],
                        vkExtensionPropertiesList[epIdx].extensionName))
            {
                isSupported = true;
                break;
            }
        }
        if (!isSupported)
        {
            return;
        }
    }

    VkInstanceCreateInfo vkInstanceCreateInfo = {};
    vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkInstanceCreateInfo.pNext = NULL;
    vkInstanceCreateInfo.flags = 0;
    vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
    vkInstanceCreateInfo.enabledLayerCount = 0;
    vkInstanceCreateInfo.ppEnabledLayerNames = NULL;
    vkInstanceCreateInfo.enabledExtensionCount =
        (uint32_t)enabledExtensionNameList.size();
    vkInstanceCreateInfo.ppEnabledExtensionNames =
        enabledExtensionNameList.data();

    vkCreateInstance(&vkInstanceCreateInfo, NULL, &m_vkInstance);

#define VK_FUNC_DECL(name)                                                     \
    _##name = (PFN_##name)vkGetInstanceProcAddr(m_vkInstance, #name);          \
    // ASSERT_NEQ((unsigned long long)name, 0ULL) << "Couldn't obtain address
    // for function" << #name;

    VK_FUNC_LIST
#if defined(_WIN32) || defined(_WIN64)
    VK_WINDOWS_FUNC_LIST
#endif
#undef VK_FUNC_DECL

    uint32_t physicalDeviceCount = 0;
    vkEnumeratePhysicalDevices(m_vkInstance, &physicalDeviceCount, NULL);
    // CHECK_NEQ(physicalDeviceCount, uint32_t(0));

    if (physicalDeviceCount == uint32_t(0))
    {
        std::cout << "failed to find GPUs with Vulkan support!\n";
        return;
    }

    std::vector<VkPhysicalDevice> vkPhysicalDeviceList(physicalDeviceCount,
                                                       VK_NULL_HANDLE);
    vkEnumeratePhysicalDevices(m_vkInstance, &physicalDeviceCount,
                               vkPhysicalDeviceList.data());

    for (size_t ppdIdx = 0; ppdIdx < vkPhysicalDeviceList.size(); ppdIdx++)
    {
        VulkanPhysicalDevice *physicalDevice =
            new VulkanPhysicalDevice(vkPhysicalDeviceList[ppdIdx]);
        m_physicalDeviceList.add(*physicalDevice);
    }
}

VulkanInstance::~VulkanInstance()
{
    for (size_t pdIdx = 0; pdIdx < m_physicalDeviceList.size(); pdIdx++)
    {
        const VulkanPhysicalDevice &physicalDevice =
            m_physicalDeviceList[pdIdx];
        delete &physicalDevice;
    }
    if (m_vkInstance)
    {
        vkDestroyInstance(m_vkInstance, NULL);
    }
}

const VulkanPhysicalDeviceList &VulkanInstance::getPhysicalDeviceList() const
{
    return m_physicalDeviceList;
}

VulkanInstance::operator VkInstance() const { return m_vkInstance; }

/////////////////////////////////////////
// VulkanPhysicalDevice implementation //
/////////////////////////////////////////

VulkanPhysicalDevice::VulkanPhysicalDevice(
    const VulkanPhysicalDevice &physicalDevice)
    : m_vkPhysicalDevice(physicalDevice.m_vkPhysicalDevice),
      m_vkPhysicalDeviceProperties(physicalDevice.m_vkPhysicalDeviceProperties),
      m_vkDeviceNodeMask(physicalDevice.m_vkDeviceNodeMask),
      m_vkPhysicalDeviceFeatures(physicalDevice.m_vkPhysicalDeviceFeatures),
      m_vkPhysicalDeviceMemoryProperties(
          physicalDevice.m_vkPhysicalDeviceMemoryProperties),
      m_queueFamilyList(physicalDevice.m_queueFamilyList)
{
    memcpy(m_vkDeviceUUID, physicalDevice.m_vkDeviceUUID, VK_UUID_SIZE);
}

VulkanPhysicalDevice::VulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice)
    : m_vkPhysicalDevice(vkPhysicalDevice)
{
    if (m_vkPhysicalDevice == (VkPhysicalDevice)VK_NULL_HANDLE)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    vkGetPhysicalDeviceProperties(m_vkPhysicalDevice,
                                  &m_vkPhysicalDeviceProperties);
    vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice,
                                &m_vkPhysicalDeviceFeatures);

    VkPhysicalDeviceIDPropertiesKHR vkPhysicalDeviceIDPropertiesKHR = {};
    vkPhysicalDeviceIDPropertiesKHR.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;
    vkPhysicalDeviceIDPropertiesKHR.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDPropertiesKHR;

    vkGetPhysicalDeviceProperties2(m_vkPhysicalDevice,
                                   &vkPhysicalDeviceProperties2);

    memcpy(m_vkDeviceUUID, vkPhysicalDeviceIDPropertiesKHR.deviceUUID,
           sizeof(m_vkDeviceUUID));
    memcpy(m_vkDeviceLUID, vkPhysicalDeviceIDPropertiesKHR.deviceLUID,
           sizeof(m_vkDeviceLUID));
    m_vkDeviceNodeMask = vkPhysicalDeviceIDPropertiesKHR.deviceNodeMask;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice,
                                             &queueFamilyCount, NULL);

    std::vector<VkQueueFamilyProperties> vkQueueFamilyPropertiesList(
        queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        m_vkPhysicalDevice, &queueFamilyCount,
        vkQueueFamilyPropertiesList.data());

    for (size_t qfpIdx = 0; qfpIdx < vkQueueFamilyPropertiesList.size();
         qfpIdx++)
    {
        VulkanQueueFamily *queueFamily = new VulkanQueueFamily(
            uint32_t(qfpIdx), vkQueueFamilyPropertiesList[qfpIdx]);
        m_queueFamilyList.add(*queueFamily);
    }

    vkGetPhysicalDeviceMemoryProperties(m_vkPhysicalDevice,
                                        &m_vkPhysicalDeviceMemoryProperties);

    for (uint32_t mhIdx = 0;
         mhIdx < m_vkPhysicalDeviceMemoryProperties.memoryHeapCount; mhIdx++)
    {
        VulkanMemoryHeap *memoryHeap = new VulkanMemoryHeap(
            mhIdx, m_vkPhysicalDeviceMemoryProperties.memoryHeaps[mhIdx].size,
            (VulkanMemoryHeapFlag)m_vkPhysicalDeviceMemoryProperties
                .memoryHeaps[mhIdx]
                .flags);
        m_memoryHeapList.add(*memoryHeap);
    }

    for (uint32_t mtIdx = 0;
         mtIdx < m_vkPhysicalDeviceMemoryProperties.memoryTypeCount; mtIdx++)
    {
        const VulkanMemoryHeap &memoryHeap = m_memoryHeapList
            [m_vkPhysicalDeviceMemoryProperties.memoryTypes[mtIdx].heapIndex];
        VulkanMemoryType *memoryType = new VulkanMemoryType(
            mtIdx,
            (VulkanMemoryTypeProperty)m_vkPhysicalDeviceMemoryProperties
                .memoryTypes[mtIdx]
                .propertyFlags,
            memoryHeap);
        m_memoryTypeList.add(*memoryType);
    }
}

VulkanPhysicalDevice::~VulkanPhysicalDevice()
{
    for (size_t mtIdx = 0; mtIdx < m_memoryTypeList.size(); mtIdx++)
    {
        const VulkanMemoryType &memoryType = m_memoryTypeList[mtIdx];
        delete &memoryType;
    }

    for (size_t mhIdx = 0; mhIdx < m_memoryHeapList.size(); mhIdx++)
    {
        const VulkanMemoryHeap &memoryHeap = m_memoryHeapList[mhIdx];
        delete &memoryHeap;
    }

    for (size_t qfIdx = 0; qfIdx < m_queueFamilyList.size(); qfIdx++)
    {
        const VulkanQueueFamily &queueFamily = m_queueFamilyList[qfIdx];
        delete &queueFamily;
    }
}


const VulkanQueueFamilyList &VulkanPhysicalDevice::getQueueFamilyList() const
{
    return m_queueFamilyList;
}

const VulkanMemoryHeapList &VulkanPhysicalDevice::getMemoryHeapList() const
{
    return m_memoryHeapList;
}

const VulkanMemoryTypeList &VulkanPhysicalDevice::getMemoryTypeList() const
{
    return m_memoryTypeList;
}

const uint8_t *VulkanPhysicalDevice::getUUID() const { return m_vkDeviceUUID; }

const uint8_t *VulkanPhysicalDevice::getLUID() const { return m_vkDeviceLUID; }

uint32_t VulkanPhysicalDevice::getNodeMask() const
{
    return m_vkDeviceNodeMask;
}

VulkanPhysicalDevice::operator VkPhysicalDevice() const
{
    return m_vkPhysicalDevice;
}

bool operator<(const VulkanQueueFamily &queueFamilyA,
               const VulkanQueueFamily &queueFamilyB)
{
    return (uint32_t)queueFamilyA < (uint32_t)queueFamilyB;
}

/////////////////////////////////////
// VulkanMemoryHeap implementation //
/////////////////////////////////////

VulkanMemoryHeap::VulkanMemoryHeap(const VulkanMemoryHeap &memoryHeap)
    : m_memoryHeapIndex(memoryHeap.m_memoryHeapIndex),
      m_size(memoryHeap.m_size), m_memoryHeapFlag(memoryHeap.m_memoryHeapFlag)
{}

VulkanMemoryHeap::VulkanMemoryHeap(uint32_t memoryHeapIndex, uint64_t size,
                                   VulkanMemoryHeapFlag memoryHeapFlag)
    : m_memoryHeapIndex(memoryHeapIndex), m_size(size),
      m_memoryHeapFlag(memoryHeapFlag)
{}

VulkanMemoryHeap::~VulkanMemoryHeap() {}

uint64_t VulkanMemoryHeap::getSize() const { return m_size; }


VulkanMemoryHeapFlag VulkanMemoryHeap::getMemoryHeapFlag() const
{
    return m_memoryHeapFlag;
}

VulkanMemoryHeap::operator uint32_t() const { return m_memoryHeapIndex; }

/////////////////////////////////////
// VulkanMemoryType implementation //
/////////////////////////////////////

VulkanMemoryType::VulkanMemoryType(const VulkanMemoryType &memoryType)
    : m_memoryTypeIndex(memoryType.m_memoryTypeIndex),
      m_memoryTypeProperty(memoryType.m_memoryTypeProperty),
      m_memoryHeap(memoryType.m_memoryHeap)
{}

VulkanMemoryType::VulkanMemoryType(uint32_t memoryTypeIndex,
                                   VulkanMemoryTypeProperty memoryTypeProperty,
                                   const VulkanMemoryHeap &memoryHeap)
    : m_memoryTypeIndex(memoryTypeIndex),
      m_memoryTypeProperty(memoryTypeProperty), m_memoryHeap(memoryHeap)
{}

VulkanMemoryType::~VulkanMemoryType() {}

VulkanMemoryTypeProperty VulkanMemoryType::getMemoryTypeProperty() const
{
    return m_memoryTypeProperty;
}

const VulkanMemoryHeap &VulkanMemoryType::getMemoryHeap() const
{
    return m_memoryHeap;
}

VulkanMemoryType::operator uint32_t() const { return m_memoryTypeIndex; }

//////////////////////////////////////
// VulkanQueueFamily implementation //
//////////////////////////////////////

VulkanQueueFamily::VulkanQueueFamily(const VulkanQueueFamily &queueFamily)
    : m_queueFamilyIndex(queueFamily.m_queueFamilyIndex),
      m_vkQueueFamilyProperties(queueFamily.m_vkQueueFamilyProperties)
{}

VulkanQueueFamily::VulkanQueueFamily(
    uint32_t queueFamilyIndex, VkQueueFamilyProperties vkQueueFamilyProperties)
    : m_queueFamilyIndex(queueFamilyIndex),
      m_vkQueueFamilyProperties(vkQueueFamilyProperties)
{}

VulkanQueueFamily::~VulkanQueueFamily() {}

uint32_t VulkanQueueFamily::getQueueFlags() const
{
    return m_vkQueueFamilyProperties.queueFlags
        & (uint32_t)VULKAN_QUEUE_FLAG_MASK_ALL;
}

uint32_t VulkanQueueFamily::getQueueCount() const
{
    return m_vkQueueFamilyProperties.queueCount;
}

VulkanQueueFamily::operator uint32_t() const { return m_queueFamilyIndex; }

/////////////////////////////////
// VulkanDevice implementation //
/////////////////////////////////

VulkanDevice::VulkanDevice(const VulkanDevice &device)
    : m_physicalDevice(device.m_physicalDevice), m_vkDevice(device.m_vkDevice)
{}

VulkanDevice::VulkanDevice(
    const VulkanPhysicalDevice &physicalDevice,
    const VulkanQueueFamilyToQueueCountMap &queueFamilyToQueueCountMap)
    : m_physicalDevice(physicalDevice), m_vkDevice(NULL)
{
    uint32_t maxQueueCount = 0;
    for (uint32_t qfIdx = 0;
         qfIdx < (uint32_t)physicalDevice.getQueueFamilyList().size(); qfIdx++)
    {
        maxQueueCount =
            std::max(maxQueueCount, queueFamilyToQueueCountMap[qfIdx]);
    }

    std::vector<VkDeviceQueueCreateInfo> vkDeviceQueueCreateInfoList;
    std::vector<float> queuePriorities(maxQueueCount);
    for (uint32_t qfIdx = 0;
         qfIdx < (uint32_t)physicalDevice.getQueueFamilyList().size(); qfIdx++)
    {
        if (queueFamilyToQueueCountMap[qfIdx])
        {
            VkDeviceQueueCreateInfo vkDeviceQueueCreateInfo = {};
            vkDeviceQueueCreateInfo.sType =
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            vkDeviceQueueCreateInfo.pNext = NULL;
            vkDeviceQueueCreateInfo.flags = 0;
            vkDeviceQueueCreateInfo.queueFamilyIndex = qfIdx;
            vkDeviceQueueCreateInfo.queueCount =
                queueFamilyToQueueCountMap[qfIdx];
            vkDeviceQueueCreateInfo.pQueuePriorities = queuePriorities.data();

            vkDeviceQueueCreateInfoList.push_back(vkDeviceQueueCreateInfo);
        }
    }

    std::vector<const char *> enabledExtensionNameList;
    enabledExtensionNameList.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32) || defined(_WIN64)
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    enabledExtensionNameList.push_back(
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif


    VkDeviceCreateInfo vkDeviceCreateInfo = {};
    vkDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    vkDeviceCreateInfo.pNext = NULL;
    vkDeviceCreateInfo.flags = 0;
    vkDeviceCreateInfo.queueCreateInfoCount =
        (uint32_t)vkDeviceQueueCreateInfoList.size();
    vkDeviceCreateInfo.pQueueCreateInfos = vkDeviceQueueCreateInfoList.data();
    vkDeviceCreateInfo.enabledLayerCount = 0;
    vkDeviceCreateInfo.ppEnabledLayerNames = NULL;
    vkDeviceCreateInfo.enabledExtensionCount =
        (uint32_t)enabledExtensionNameList.size();
    vkDeviceCreateInfo.ppEnabledExtensionNames =
        enabledExtensionNameList.data();
    vkDeviceCreateInfo.pEnabledFeatures = NULL;

    vkCreateDevice(physicalDevice, &vkDeviceCreateInfo, NULL, &m_vkDevice);

    for (uint32_t qfIdx = 0;
         qfIdx < (uint32_t)m_physicalDevice.getQueueFamilyList().size();
         qfIdx++)
    {
        VulkanQueueList *queueList = new VulkanQueueList();
        m_queueFamilyIndexToQueueListMap.insert(qfIdx, *queueList);
        for (uint32_t qIdx = 0; qIdx < queueFamilyToQueueCountMap[qfIdx];
             qIdx++)
        {
            VkQueue vkQueue;
            vkGetDeviceQueue(m_vkDevice, qfIdx, qIdx, &vkQueue);
            VulkanQueue *queue = new VulkanQueue(vkQueue);
            m_queueFamilyIndexToQueueListMap[qfIdx].add(*queue);
        }
    }
}

VulkanDevice::~VulkanDevice()
{
    for (uint32_t qfIdx = 0;
         qfIdx < (uint32_t)m_physicalDevice.getQueueFamilyList().size();
         qfIdx++)
    {
        for (size_t qIdx = 0;
             qIdx < m_queueFamilyIndexToQueueListMap[qfIdx].size(); qIdx++)
        {
            VulkanQueue &queue = m_queueFamilyIndexToQueueListMap[qfIdx][qIdx];
            delete &queue;
        }
        VulkanQueueList &queueList = m_queueFamilyIndexToQueueListMap[qfIdx];
        delete &queueList;
    }
    vkDestroyDevice(m_vkDevice, NULL);
}

const VulkanPhysicalDevice &VulkanDevice::getPhysicalDevice() const
{
    return m_physicalDevice;
}

VulkanQueue &VulkanDevice::getQueue(const VulkanQueueFamily &queueFamily,
                                    uint32_t queueIndex)
{
    return m_queueFamilyIndexToQueueListMap[queueFamily][queueIndex];
}

VulkanDevice::operator VkDevice() const { return m_vkDevice; }

////////////////////////////////
// VulkanFence implementation //
////////////////////////////////

VulkanFence::VulkanFence(const VulkanDevice &vkDevice)
{

    device = vkDevice;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = nullptr;
    fenceInfo.flags = 0;

    VkResult vkStatus = vkCreateFence(device, &fenceInfo, nullptr, &fence);

    if (vkStatus != VK_SUCCESS)
    {
        throw std::runtime_error("Error: Failed create fence.");
    }
}

VulkanFence::~VulkanFence() { vkDestroyFence(device, fence, nullptr); }

void VulkanFence::reset() { vkResetFences(device, 1, &fence); }

void VulkanFence::wait()
{
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
}

////////////////////////////////
// VulkanQueue implementation //
////////////////////////////////

VulkanQueue::VulkanQueue(const VulkanQueue &queue): m_vkQueue(queue.m_vkQueue)
{}

VulkanQueue::VulkanQueue(VkQueue vkQueue): m_vkQueue(vkQueue) {}

VulkanQueue::~VulkanQueue() {}

void VulkanQueue::submit(const VulkanCommandBuffer &commandBuffer,
                         const std::shared_ptr<VulkanFence> &vkFence)
{
    VulkanCommandBufferList commandBufferList;
    commandBufferList.add(commandBuffer);

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.pNext = NULL;
    vkSubmitInfo.waitSemaphoreCount = (uint32_t)0;
    vkSubmitInfo.commandBufferCount = (uint32_t)commandBufferList.size();
    vkSubmitInfo.pCommandBuffers = commandBufferList();

    vkQueueSubmit(m_vkQueue, 1, &vkSubmitInfo, vkFence->fence);
}

void VulkanQueue::submit(const VulkanSemaphoreList &waitSemaphoreList,
                         const VulkanCommandBufferList &commandBufferList,
                         const VulkanSemaphoreList &signalSemaphoreList)
{
    std::vector<VkPipelineStageFlags> vkPipelineStageFlagsList(
        waitSemaphoreList.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    VkSubmitInfo vkSubmitInfo = {};
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.pNext = NULL;
    vkSubmitInfo.waitSemaphoreCount = (uint32_t)waitSemaphoreList.size();
    vkSubmitInfo.pWaitSemaphores = waitSemaphoreList();
    vkSubmitInfo.pWaitDstStageMask = vkPipelineStageFlagsList.data();
    vkSubmitInfo.commandBufferCount = (uint32_t)commandBufferList.size();
    vkSubmitInfo.pCommandBuffers = commandBufferList();
    vkSubmitInfo.signalSemaphoreCount = (uint32_t)signalSemaphoreList.size();
    vkSubmitInfo.pSignalSemaphores = signalSemaphoreList();

    vkQueueSubmit(m_vkQueue, 1, &vkSubmitInfo, NULL);
}

void VulkanQueue::submit(const VulkanSemaphore &waitSemaphore,
                         const VulkanCommandBuffer &commandBuffer,
                         const VulkanSemaphore &signalSemaphore)
{
    VulkanSemaphoreList waitSemaphoreList;
    VulkanCommandBufferList commandBufferList;
    VulkanSemaphoreList signalSemaphoreList;

    waitSemaphoreList.add(waitSemaphore);
    commandBufferList.add(commandBuffer);
    signalSemaphoreList.add(signalSemaphore);

    submit(waitSemaphoreList, commandBufferList, signalSemaphoreList);
}

void VulkanQueue::submit(const VulkanCommandBuffer &commandBuffer,
                         const VulkanSemaphore &signalSemaphore)
{
    VulkanSemaphoreList waitSemaphoreList;
    VulkanCommandBufferList commandBufferList;
    VulkanSemaphoreList signalSemaphoreList;

    commandBufferList.add(commandBuffer);
    signalSemaphoreList.add(signalSemaphore);

    submit(waitSemaphoreList, commandBufferList, signalSemaphoreList);
}

void VulkanQueue::submit(const VulkanCommandBuffer &commandBuffer)
{
    VulkanSemaphoreList waitSemaphoreList;
    VulkanCommandBufferList commandBufferList;
    VulkanSemaphoreList signalSemaphoreList;

    commandBufferList.add(commandBuffer);

    submit(waitSemaphoreList, commandBufferList, signalSemaphoreList);
}

void VulkanQueue::waitIdle() { vkQueueWaitIdle(m_vkQueue); }

VulkanQueue::operator VkQueue() const { return m_vkQueue; }

/////////////////////////////////////////////////////
// VulkanDescriptorSetLayoutBinding implementation //
/////////////////////////////////////////////////////

VulkanDescriptorSetLayoutBinding::VulkanDescriptorSetLayoutBinding(
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding)
    : m_vkDescriptorSetLayoutBinding(
        descriptorSetLayoutBinding.m_vkDescriptorSetLayoutBinding)
{}

VulkanDescriptorSetLayoutBinding::VulkanDescriptorSetLayoutBinding(
    uint32_t binding, VulkanDescriptorType descriptorType,
    uint32_t descriptorCount, VulkanShaderStage shaderStage)
{
    m_vkDescriptorSetLayoutBinding.binding = binding;
    m_vkDescriptorSetLayoutBinding.descriptorType =
        (VkDescriptorType)descriptorType;
    m_vkDescriptorSetLayoutBinding.descriptorCount = descriptorCount;
    m_vkDescriptorSetLayoutBinding.stageFlags =
        (VkShaderStageFlags)(VkShaderStageFlagBits)shaderStage;
    m_vkDescriptorSetLayoutBinding.pImmutableSamplers = NULL;
}

VulkanDescriptorSetLayoutBinding::~VulkanDescriptorSetLayoutBinding() {}

VulkanDescriptorSetLayoutBinding::operator VkDescriptorSetLayoutBinding() const
{
    return m_vkDescriptorSetLayoutBinding;
}

//////////////////////////////////////////////
// VulkanDescriptorSetLayout implementation //
//////////////////////////////////////////////

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(
    const VulkanDescriptorSetLayout &descriptorSetLayout)
    : m_device(descriptorSetLayout.m_device),
      m_vkDescriptorSetLayout(descriptorSetLayout.m_vkDescriptorSetLayout)
{}

void VulkanDescriptorSetLayout::VulkanDescriptorSetLayoutCommon(
    const VulkanDescriptorSetLayoutBindingList &descriptorSetLayoutBindingList)
{
    VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo = {};
    vkDescriptorSetLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vkDescriptorSetLayoutCreateInfo.pNext = NULL;
    vkDescriptorSetLayoutCreateInfo.flags = 0;
    vkDescriptorSetLayoutCreateInfo.bindingCount =
        (uint32_t)descriptorSetLayoutBindingList.size();
    vkDescriptorSetLayoutCreateInfo.pBindings =
        descriptorSetLayoutBindingList();

    vkCreateDescriptorSetLayout(m_device, &vkDescriptorSetLayoutCreateInfo,
                                NULL, &m_vkDescriptorSetLayout);
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding)
    : m_device(device), m_vkDescriptorSetLayout(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutBindingList descriptorSetLayoutBindingList;
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding);

    VulkanDescriptorSetLayoutCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding0,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding1)
    : m_device(device), m_vkDescriptorSetLayout(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutBindingList descriptorSetLayoutBindingList;
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding0);
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding1);

    VulkanDescriptorSetLayoutCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBindingList &descriptorSetLayoutBindingList)
    : m_device(device), m_vkDescriptorSetLayout(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
{
    if (m_vkDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_vkDescriptorSetLayout, NULL);
    }
}

VulkanDescriptorSetLayout::operator VkDescriptorSetLayout() const
{
    return m_vkDescriptorSetLayout;
}

/////////////////////////////////////////
// VulkanPipelineLayout implementation //
/////////////////////////////////////////

VulkanPipelineLayout::VulkanPipelineLayout(
    const VulkanPipelineLayout &pipelineLayout)
    : m_device(pipelineLayout.m_device),
      m_vkPipelineLayout(pipelineLayout.m_vkPipelineLayout)
{}

void VulkanPipelineLayout::VulkanPipelineLayoutCommon(
    const VulkanDescriptorSetLayoutList &descriptorSetLayoutList)
{
    VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo = {};
    vkPipelineLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkPipelineLayoutCreateInfo.pNext = NULL;
    vkPipelineLayoutCreateInfo.flags = 0;
    vkPipelineLayoutCreateInfo.setLayoutCount =
        (uint32_t)descriptorSetLayoutList.size();
    vkPipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayoutList();
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = NULL;

    vkCreatePipelineLayout(m_device, &vkPipelineLayoutCreateInfo, NULL,
                           &m_vkPipelineLayout);
}

VulkanPipelineLayout::VulkanPipelineLayout(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayout &descriptorSetLayout)
    : m_device(device), m_vkPipelineLayout(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutList descriptorSetLayoutList;
    descriptorSetLayoutList.add(descriptorSetLayout);

    VulkanPipelineLayoutCommon(descriptorSetLayoutList);
}

VulkanPipelineLayout::VulkanPipelineLayout(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutList &descriptorSetLayoutList)
    : m_device(device), m_vkPipelineLayout(VK_NULL_HANDLE)
{
    VulkanPipelineLayoutCommon(descriptorSetLayoutList);
}

VulkanPipelineLayout::~VulkanPipelineLayout()
{
    vkDestroyPipelineLayout(m_device, m_vkPipelineLayout, NULL);
}

VulkanPipelineLayout::operator VkPipelineLayout() const
{
    return m_vkPipelineLayout;
}

///////////////////////////////////////
// VulkanShaderModule implementation //
///////////////////////////////////////

VulkanShaderModule::VulkanShaderModule(const VulkanShaderModule &shaderModule)
    : m_device(shaderModule.m_device),
      m_vkShaderModule(shaderModule.m_vkShaderModule)
{}

VulkanShaderModule::VulkanShaderModule(const VulkanDevice &device,
                                       const std::vector<char> &code)
    : m_device(device)
{

    VkShaderModuleCreateInfo vkShaderModuleCreateInfo = {};
    vkShaderModuleCreateInfo.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0;
    vkShaderModuleCreateInfo.codeSize = code.size();
    vkShaderModuleCreateInfo.pCode =
        reinterpret_cast<const uint32_t *>(code.data());

    vkCreateShaderModule(m_device, &vkShaderModuleCreateInfo, NULL,
                         &m_vkShaderModule);
}

VulkanShaderModule::~VulkanShaderModule()
{
    vkDestroyShaderModule(m_device, m_vkShaderModule, NULL);
}

VulkanShaderModule::operator VkShaderModule() const { return m_vkShaderModule; }

///////////////////////////////////
// VulkanPipeline implementation //
///////////////////////////////////

VulkanPipeline::VulkanPipeline(const VulkanPipeline &pipeline)
    : m_device(pipeline.m_device), m_vkPipeline(pipeline.m_vkPipeline)
{}

VulkanPipeline::VulkanPipeline(const VulkanDevice &device)
    : m_device(device), m_vkPipeline(VK_NULL_HANDLE)
{}

VulkanPipeline::~VulkanPipeline()
{
    vkDestroyPipeline(m_device, m_vkPipeline, NULL);
}

VulkanPipeline::operator VkPipeline() const { return m_vkPipeline; }

//////////////////////////////////////////
// VulkanComputePipeline implementation //
//////////////////////////////////////////

VulkanComputePipeline::VulkanComputePipeline(
    const VulkanComputePipeline &computePipeline)
    : VulkanPipeline(computePipeline)
{}

VulkanComputePipeline::VulkanComputePipeline(
    const VulkanDevice &device, const VulkanPipelineLayout &pipelineLayout,
    const VulkanShaderModule &shaderModule, const std::string &entryFuncName)
    : VulkanPipeline(device)
{
    VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo = {};
    vkPipelineShaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo.pNext = NULL;
    vkPipelineShaderStageCreateInfo.flags = 0;
    vkPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkPipelineShaderStageCreateInfo.module = shaderModule;
    vkPipelineShaderStageCreateInfo.pName = entryFuncName.c_str();
    vkPipelineShaderStageCreateInfo.pSpecializationInfo = NULL;

    VkComputePipelineCreateInfo vkComputePipelineCreateInfo = {};
    vkComputePipelineCreateInfo.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    vkComputePipelineCreateInfo.pNext = NULL;
    vkComputePipelineCreateInfo.flags = 0;
    vkComputePipelineCreateInfo.stage = vkPipelineShaderStageCreateInfo;
    vkComputePipelineCreateInfo.layout = pipelineLayout;
    vkComputePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkComputePipelineCreateInfo.basePipelineIndex = 0;

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                             &vkComputePipelineCreateInfo, NULL, &m_vkPipeline);
}

VulkanComputePipeline::~VulkanComputePipeline() {}

VulkanPipelineBindPoint VulkanComputePipeline::getPipelineBindPoint() const
{
    return VULKAN_PIPELINE_BIND_POINT_COMPUTE;
}

/////////////////////////////////////////
// VulkanDescriptorPool implementation //
/////////////////////////////////////////

VulkanDescriptorPool::VulkanDescriptorPool(
    const VulkanDescriptorPool &descriptorPool)
    : m_device(descriptorPool.m_device),
      m_vkDescriptorPool(descriptorPool.m_vkDescriptorPool)
{}

void VulkanDescriptorPool::VulkanDescriptorPoolCommon(
    const VulkanDescriptorSetLayoutBindingList &descriptorSetLayoutBindingList)
{
    if (descriptorSetLayoutBindingList.size())
    {
        std::map<VkDescriptorType, uint32_t>
            vkDescriptorTypeToDescriptorCountMap;

        for (size_t dslbIdx = 0;
             dslbIdx < descriptorSetLayoutBindingList.size(); dslbIdx++)
        {
            VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding =
                descriptorSetLayoutBindingList[dslbIdx];
            if (vkDescriptorTypeToDescriptorCountMap.find(
                    vkDescriptorSetLayoutBinding.descriptorType)
                == vkDescriptorTypeToDescriptorCountMap.end())
            {
                vkDescriptorTypeToDescriptorCountMap
                    [vkDescriptorSetLayoutBinding.descriptorType] =
                        vkDescriptorSetLayoutBinding.descriptorCount;
            }
            else
            {
                vkDescriptorTypeToDescriptorCountMap
                    [vkDescriptorSetLayoutBinding.descriptorType] +=
                    vkDescriptorSetLayoutBinding.descriptorCount;
            }
        }

        std::vector<VkDescriptorPoolSize> vkDescriptorPoolSizeList;
        std::map<VkDescriptorType, uint32_t>::iterator dtdcIt;
        for (dtdcIt = vkDescriptorTypeToDescriptorCountMap.begin();
             dtdcIt != vkDescriptorTypeToDescriptorCountMap.end(); ++dtdcIt)
        {
            VkDescriptorPoolSize vkDescriptorPoolSize = {};
            vkDescriptorPoolSize.type = dtdcIt->first;
            vkDescriptorPoolSize.descriptorCount = dtdcIt->second;

            vkDescriptorPoolSizeList.push_back(vkDescriptorPoolSize);
        }

        VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo = {};
        vkDescriptorPoolCreateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        vkDescriptorPoolCreateInfo.pNext = NULL;
        vkDescriptorPoolCreateInfo.flags =
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        vkDescriptorPoolCreateInfo.maxSets = 1;
        vkDescriptorPoolCreateInfo.poolSizeCount =
            (uint32_t)vkDescriptorPoolSizeList.size();
        vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSizeList.data();

        vkCreateDescriptorPool(m_device, &vkDescriptorPoolCreateInfo, NULL,
                               &m_vkDescriptorPool);
    }
}

VulkanDescriptorPool::VulkanDescriptorPool(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding)
    : m_device(device), m_vkDescriptorPool(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutBindingList descriptorSetLayoutBindingList;
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding);

    VulkanDescriptorPoolCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorPool::VulkanDescriptorPool(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding0,
    const VulkanDescriptorSetLayoutBinding &descriptorSetLayoutBinding1)
    : m_device(device), m_vkDescriptorPool(VK_NULL_HANDLE)
{
    VulkanDescriptorSetLayoutBindingList descriptorSetLayoutBindingList;
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding0);
    descriptorSetLayoutBindingList.add(descriptorSetLayoutBinding1);

    VulkanDescriptorPoolCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorPool::VulkanDescriptorPool(
    const VulkanDevice &device,
    const VulkanDescriptorSetLayoutBindingList &descriptorSetLayoutBindingList)
    : m_device(device), m_vkDescriptorPool(VK_NULL_HANDLE)
{
    VulkanDescriptorPoolCommon(descriptorSetLayoutBindingList);
}

VulkanDescriptorPool::~VulkanDescriptorPool()
{
    if (m_vkDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_vkDescriptorPool, NULL);
    }
}

VulkanDescriptorPool::operator VkDescriptorPool() const
{
    return m_vkDescriptorPool;
}

////////////////////////////////////////
// VulkanDescriptorSet implementation //
////////////////////////////////////////

VulkanDescriptorSet::VulkanDescriptorSet(
    const VulkanDescriptorSet &descriptorSet)
    : m_device(descriptorSet.m_device),
      m_descriptorPool(descriptorSet.m_descriptorPool),
      m_vkDescriptorSet(descriptorSet.m_vkDescriptorSet)
{}

VulkanDescriptorSet::VulkanDescriptorSet(
    const VulkanDevice &device, const VulkanDescriptorPool &descriptorPool,
    const VulkanDescriptorSetLayout &descriptorSetLayout)
    : m_device(device), m_descriptorPool(descriptorPool),
      m_vkDescriptorSet(VK_NULL_HANDLE)
{
    VkDescriptorSetLayout vkDescriptorSetLayout = descriptorSetLayout;

    if ((VkDescriptorPool)m_descriptorPool)
    {
        VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo = {};
        vkDescriptorSetAllocateInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        vkDescriptorSetAllocateInfo.pNext = NULL;
        vkDescriptorSetAllocateInfo.descriptorPool = descriptorPool;
        vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
        vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout;

        vkAllocateDescriptorSets(m_device, &vkDescriptorSetAllocateInfo,
                                 &m_vkDescriptorSet);
    }
}

VulkanDescriptorSet::~VulkanDescriptorSet()
{
    if ((VkDescriptorPool)m_descriptorPool)
    {
        vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &m_vkDescriptorSet);
    }
}

void VulkanDescriptorSet::update(uint32_t binding, const VulkanBuffer &buffer)
{
    VkDescriptorBufferInfo vkDescriptorBufferInfo = {};
    vkDescriptorBufferInfo.buffer = buffer;
    vkDescriptorBufferInfo.offset = 0;
    vkDescriptorBufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet vkWriteDescriptorSet = {};
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = m_vkDescriptorSet;
    vkWriteDescriptorSet.dstBinding = binding;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = 1;
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkWriteDescriptorSet.pImageInfo = NULL;
    vkWriteDescriptorSet.pBufferInfo = &vkDescriptorBufferInfo;
    vkWriteDescriptorSet.pTexelBufferView = NULL;

    vkUpdateDescriptorSets(m_device, 1, &vkWriteDescriptorSet, 0, NULL);
}

void VulkanDescriptorSet::updateArray(uint32_t binding, unsigned numBuffers,
                                      const VulkanBufferList &buffers)
{
    VkDescriptorBufferInfo *vkDescriptorBufferInfo =
        (VkDescriptorBufferInfo *)calloc(numBuffers,
                                         sizeof(VkDescriptorBufferInfo));
    for (unsigned i = 0; i < numBuffers; i++)
    {
        vkDescriptorBufferInfo[i].buffer = buffers[i];
        vkDescriptorBufferInfo[i].offset = 0;
        vkDescriptorBufferInfo[i].range = VK_WHOLE_SIZE;
    }

    VkWriteDescriptorSet vkWriteDescriptorSet = {};
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = m_vkDescriptorSet;
    vkWriteDescriptorSet.dstBinding = binding;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = numBuffers;
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkWriteDescriptorSet.pImageInfo = NULL;
    vkWriteDescriptorSet.pBufferInfo = vkDescriptorBufferInfo;
    vkWriteDescriptorSet.pTexelBufferView = NULL;

    vkUpdateDescriptorSets(m_device, 1, &vkWriteDescriptorSet, 0, NULL);
    free(vkDescriptorBufferInfo);
}

void VulkanDescriptorSet::update(uint32_t binding,
                                 const VulkanImageView &imageView)
{
    VkDescriptorImageInfo vkDescriptorImageInfo = {};
    vkDescriptorImageInfo.sampler = VK_NULL_HANDLE;
    vkDescriptorImageInfo.imageView = imageView;
    vkDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet vkWriteDescriptorSet = {};
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = m_vkDescriptorSet;
    vkWriteDescriptorSet.dstBinding = binding;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = 1;
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    vkWriteDescriptorSet.pImageInfo = &vkDescriptorImageInfo;
    vkWriteDescriptorSet.pBufferInfo = NULL;
    vkWriteDescriptorSet.pTexelBufferView = NULL;

    vkUpdateDescriptorSets(m_device, 1, &vkWriteDescriptorSet, 0, NULL);
}

void VulkanDescriptorSet::updateArray(uint32_t binding,
                                      const VulkanImageViewList &imageViewList)
{
    VkDescriptorImageInfo *vkDescriptorImageInfo =
        new VkDescriptorImageInfo[imageViewList.size()];
    for (size_t i = 0; i < imageViewList.size(); i++)
    {
        vkDescriptorImageInfo[i].sampler = VK_NULL_HANDLE;
        vkDescriptorImageInfo[i].imageView = imageViewList[i];
        vkDescriptorImageInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkWriteDescriptorSet vkWriteDescriptorSet = {};
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = m_vkDescriptorSet;
    vkWriteDescriptorSet.dstBinding = binding;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = imageViewList.size();
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    vkWriteDescriptorSet.pImageInfo = vkDescriptorImageInfo;
    vkWriteDescriptorSet.pBufferInfo = NULL;
    vkWriteDescriptorSet.pTexelBufferView = NULL;

    vkUpdateDescriptorSets(m_device, 1, &vkWriteDescriptorSet, 0, NULL);
    delete[] vkDescriptorImageInfo;
}

VulkanDescriptorSet::operator VkDescriptorSet() const
{
    return m_vkDescriptorSet;
}

///////////////////////////////////
// VulkanOffset3D implementation //
///////////////////////////////////

VulkanOffset3D::VulkanOffset3D(const VulkanOffset3D &offset3D)
    : m_vkOffset3D(offset3D.m_vkOffset3D)
{}

VulkanOffset3D::VulkanOffset3D(uint32_t x, uint32_t y, uint32_t z)
{
    m_vkOffset3D.x = x;
    m_vkOffset3D.y = y;
    m_vkOffset3D.z = z;
}

VulkanOffset3D::~VulkanOffset3D() {}

uint32_t VulkanOffset3D::getX() const { return m_vkOffset3D.x; }

uint32_t VulkanOffset3D::getY() const { return m_vkOffset3D.y; }

uint32_t VulkanOffset3D::getZ() const { return m_vkOffset3D.z; }

VulkanOffset3D::operator VkOffset3D() const { return m_vkOffset3D; }

///////////////////////////////////
// VulkanExtent3D implementation //
///////////////////////////////////

VulkanExtent3D::VulkanExtent3D(const VulkanExtent3D &extent3D)
    : m_vkExtent3D(extent3D.m_vkExtent3D)
{}

VulkanExtent3D::VulkanExtent3D(uint32_t width, uint32_t height, uint32_t depth)
{
    m_vkExtent3D.width = width;
    m_vkExtent3D.height = height;
    m_vkExtent3D.depth = depth;
}

VulkanExtent3D::~VulkanExtent3D() {}

uint32_t VulkanExtent3D::getWidth() const { return m_vkExtent3D.width; }

uint32_t VulkanExtent3D::getHeight() const { return m_vkExtent3D.height; }

uint32_t VulkanExtent3D::getDepth() const { return m_vkExtent3D.depth; }

VulkanExtent3D::operator VkExtent3D() const { return m_vkExtent3D; }

//////////////////////////////////////
// VulkanCommandPool implementation //
//////////////////////////////////////

VulkanCommandPool::VulkanCommandPool(const VulkanCommandPool &commandPool)
    : m_device(commandPool.m_device),
      m_vkCommandPool(commandPool.m_vkCommandPool)
{}

VulkanCommandPool::VulkanCommandPool(const VulkanDevice &device,
                                     const VulkanQueueFamily &queueFamily)
    : m_device(device)
{
    VkCommandPoolCreateInfo vkCommandPoolCreateInfo = {};
    vkCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    vkCommandPoolCreateInfo.pNext = NULL;
    vkCommandPoolCreateInfo.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCommandPoolCreateInfo.queueFamilyIndex = queueFamily;

    vkCreateCommandPool(m_device, &vkCommandPoolCreateInfo, NULL,
                        &m_vkCommandPool);
}

VulkanCommandPool::~VulkanCommandPool()
{
    vkDestroyCommandPool(m_device, m_vkCommandPool, NULL);
}

VulkanCommandPool::operator VkCommandPool() const { return m_vkCommandPool; }

////////////////////////////////////////
// VulkanCommandBuffer implementation //
////////////////////////////////////////

VulkanCommandBuffer::VulkanCommandBuffer(
    const VulkanCommandBuffer &commandBuffer)
    : m_device(commandBuffer.m_device),
      m_commandPool(commandBuffer.m_commandPool),
      m_vkCommandBuffer(commandBuffer.m_vkCommandBuffer)
{}

VulkanCommandBuffer::VulkanCommandBuffer(const VulkanDevice &device,
                                         const VulkanCommandPool &commandPool)
    : m_device(device), m_commandPool(commandPool)
{
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo = {};
    vkCommandBufferAllocateInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo.pNext = NULL;
    vkCommandBufferAllocateInfo.commandPool = commandPool;
    vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkCommandBufferAllocateInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(m_device, &vkCommandBufferAllocateInfo,
                             &m_vkCommandBuffer);
}

VulkanCommandBuffer::~VulkanCommandBuffer()
{
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_vkCommandBuffer);
}

void VulkanCommandBuffer::begin()
{
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo = {};
    vkCommandBufferBeginInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo.pNext = NULL;
    vkCommandBufferBeginInfo.flags =
        VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkCommandBufferBeginInfo.pInheritanceInfo = NULL;

    vkBeginCommandBuffer(m_vkCommandBuffer, &vkCommandBufferBeginInfo);
}

void VulkanCommandBuffer::bindPipeline(const VulkanPipeline &pipeline)
{
    VkPipelineBindPoint vkPipelineBindPoint =
        (VkPipelineBindPoint)pipeline.getPipelineBindPoint();

    vkCmdBindPipeline(m_vkCommandBuffer, vkPipelineBindPoint, pipeline);
}

void VulkanCommandBuffer::bindDescriptorSets(
    const VulkanPipeline &pipeline, const VulkanPipelineLayout &pipelineLayout,
    const VulkanDescriptorSet &descriptorSet)
{
    VkPipelineBindPoint vkPipelineBindPoint =
        (VkPipelineBindPoint)pipeline.getPipelineBindPoint();
    VkDescriptorSet vkDescriptorSet = descriptorSet;

    vkCmdBindDescriptorSets(m_vkCommandBuffer, vkPipelineBindPoint,
                            pipelineLayout, 0, 1, &vkDescriptorSet, 0, NULL);
}

void VulkanCommandBuffer::pipelineBarrier(const VulkanImage2DList &image2DList,
                                          VulkanImageLayout oldImageLayout,
                                          VulkanImageLayout newImageLayout)
{
    std::vector<VkImageMemoryBarrier> vkImageMemoryBarrierList;
    for (size_t i2DIdx = 0; i2DIdx < image2DList.size(); i2DIdx++)
    {
        VkImageSubresourceRange vkImageSubresourceRange = {};
        vkImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkImageSubresourceRange.baseMipLevel = 0;
        vkImageSubresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        vkImageSubresourceRange.baseArrayLayer = 0;
        vkImageSubresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        VkImageMemoryBarrier vkImageMemoryBarrier = {};
        vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        vkImageMemoryBarrier.pNext = NULL;
        vkImageMemoryBarrier.srcAccessMask = 0;
        vkImageMemoryBarrier.dstAccessMask = 0;
        vkImageMemoryBarrier.oldLayout = (VkImageLayout)oldImageLayout;
        vkImageMemoryBarrier.newLayout = (VkImageLayout)newImageLayout;
        vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkImageMemoryBarrier.image = image2DList[i2DIdx];
        vkImageMemoryBarrier.subresourceRange = vkImageSubresourceRange;

        vkImageMemoryBarrierList.push_back(vkImageMemoryBarrier);
    }

    vkCmdPipelineBarrier(m_vkCommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0,
                         NULL, (uint32_t)vkImageMemoryBarrierList.size(),
                         vkImageMemoryBarrierList.data());
}

void VulkanCommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY,
                                   uint32_t groupCountZ)
{
    vkCmdDispatch(m_vkCommandBuffer, groupCountX, groupCountY, groupCountZ);
}

void VulkanCommandBuffer::fillBuffer(const VulkanBuffer &buffer, uint32_t data,
                                     uint64_t offset, uint64_t size)
{
    vkCmdFillBuffer(m_vkCommandBuffer, buffer, offset, size, data);
}

void VulkanCommandBuffer::updateBuffer(const VulkanBuffer &buffer, void *pdata,
                                       uint64_t offset, uint64_t size)
{
    vkCmdUpdateBuffer(m_vkCommandBuffer, buffer, offset, size, pdata);
}

void VulkanCommandBuffer::copyBufferToImage(const VulkanBuffer &buffer,
                                            const VulkanImage &image,
                                            VulkanImageLayout imageLayout)
{
    VkDeviceSize bufferOffset = 0;

    std::vector<VkBufferImageCopy> vkBufferImageCopyList;
    for (uint32_t mipLevel = 0; mipLevel < image.getNumMipLevels(); mipLevel++)
    {
        VulkanExtent3D extent3D = image.getExtent3D(mipLevel);
        size_t elementSize = getVulkanFormatElementSize(image.getFormat());

        VkImageSubresourceLayers vkImageSubresourceLayers = {};
        vkImageSubresourceLayers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkImageSubresourceLayers.mipLevel = mipLevel;
        vkImageSubresourceLayers.baseArrayLayer = 0;
        vkImageSubresourceLayers.layerCount = image.getNumLayers();

        VkBufferImageCopy vkBufferImageCopy = {};
        vkBufferImageCopy.bufferOffset = bufferOffset;
        vkBufferImageCopy.bufferRowLength = 0;
        vkBufferImageCopy.bufferImageHeight = 0;
        vkBufferImageCopy.imageSubresource = vkImageSubresourceLayers;
        vkBufferImageCopy.imageOffset = VulkanOffset3D(0, 0, 0);
        vkBufferImageCopy.imageExtent = extent3D;

        vkBufferImageCopyList.push_back(vkBufferImageCopy);

        bufferOffset += extent3D.getWidth() * extent3D.getHeight()
            * extent3D.getDepth() * elementSize;
        bufferOffset =
            ROUND_UP(bufferOffset,
                     std::max(elementSize,
                              (size_t)VULKAN_MIN_BUFFER_OFFSET_COPY_ALIGNMENT));
    }

    vkCmdCopyBufferToImage(
        m_vkCommandBuffer, buffer, image, (VkImageLayout)imageLayout,
        (uint32_t)vkBufferImageCopyList.size(), vkBufferImageCopyList.data());
}

void VulkanCommandBuffer::copyBufferToImage(
    const VulkanBuffer &buffer, const VulkanImage &image, uint64_t bufferOffset,
    uint32_t mipLevel, uint32_t baseArrayLayer, uint32_t layerCount,
    VulkanOffset3D offset3D, VulkanExtent3D extent3D)
{
    VkImageSubresourceLayers vkImageSubresourceLayers = {};
    vkImageSubresourceLayers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageSubresourceLayers.mipLevel = mipLevel;
    vkImageSubresourceLayers.baseArrayLayer = baseArrayLayer;
    vkImageSubresourceLayers.layerCount = layerCount;

    VkExtent3D vkExtent3D = extent3D;
    if ((extent3D.getWidth() == 0) && (extent3D.getHeight() == 0)
        && (extent3D.getDepth() == 0))
    {
        vkExtent3D = image.getExtent3D(mipLevel);
    }

    VkBufferImageCopy vkBufferImageCopy = {};
    vkBufferImageCopy.bufferOffset = bufferOffset;
    vkBufferImageCopy.bufferRowLength = 0;
    vkBufferImageCopy.bufferImageHeight = 0;
    vkBufferImageCopy.imageSubresource = vkImageSubresourceLayers;
    vkBufferImageCopy.imageOffset = offset3D;
    vkBufferImageCopy.imageExtent = vkExtent3D;

    vkCmdCopyBufferToImage(m_vkCommandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &vkBufferImageCopy);
}

void VulkanCommandBuffer::copyImageToBuffer(
    const VulkanImage &image, const VulkanBuffer &buffer, uint64_t bufferOffset,
    uint32_t mipLevel, uint32_t baseArrayLayer, uint32_t layerCount,
    VulkanOffset3D offset3D, VulkanExtent3D extent3D)
{
    VkImageSubresourceLayers vkImageSubresourceLayers = {};
    vkImageSubresourceLayers.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageSubresourceLayers.mipLevel = mipLevel;
    vkImageSubresourceLayers.baseArrayLayer = baseArrayLayer;
    vkImageSubresourceLayers.layerCount = layerCount;

    VkExtent3D vkExtent3D = extent3D;
    if ((extent3D.getWidth() == 0) && (extent3D.getHeight() == 0)
        && (extent3D.getDepth() == 0))
    {
        vkExtent3D = image.getExtent3D(mipLevel);
    }

    VkBufferImageCopy vkBufferImageCopy = {};
    vkBufferImageCopy.bufferOffset = bufferOffset;
    vkBufferImageCopy.bufferRowLength = 0;
    vkBufferImageCopy.bufferImageHeight = 0;
    vkBufferImageCopy.imageSubresource = vkImageSubresourceLayers;
    vkBufferImageCopy.imageOffset = offset3D;
    vkBufferImageCopy.imageExtent = vkExtent3D;

    vkCmdCopyImageToBuffer(m_vkCommandBuffer, image, VK_IMAGE_LAYOUT_GENERAL,
                           buffer, 1, &vkBufferImageCopy);
}

void VulkanCommandBuffer::end() { vkEndCommandBuffer(m_vkCommandBuffer); }

VulkanCommandBuffer::operator VkCommandBuffer() const
{
    return m_vkCommandBuffer;
}

/////////////////////////////////
// VulkanBuffer implementation //
/////////////////////////////////

VulkanBuffer::VulkanBuffer(const VulkanBuffer &buffer)
    : m_device(buffer.m_device), m_vkBuffer(buffer.m_vkBuffer),
      m_size(buffer.m_size), m_alignment(buffer.m_alignment),
      m_memoryTypeList(buffer.m_memoryTypeList)
{}

bool VulkanBuffer::isDedicated() const { return m_dedicated; }

VulkanBuffer::VulkanBuffer(
    const VulkanDevice &device, uint64_t size,
    VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanBufferUsage bufferUsage, VulkanSharingMode sharingMode,
    const VulkanQueueFamilyList &queueFamilyList)
    : m_device(device), m_vkBuffer(VK_NULL_HANDLE), m_dedicated(false)
{
    std::vector<uint32_t> queueFamilyIndexList;
    if (queueFamilyList.size() == 0)
    {
        for (size_t qfIdx = 0;
             qfIdx < device.getPhysicalDevice().getQueueFamilyList().size();
             qfIdx++)
        {
            queueFamilyIndexList.push_back(
                device.getPhysicalDevice().getQueueFamilyList()[qfIdx]);
        }
    }
    else
    {
        for (size_t qfIdx = 0; qfIdx < queueFamilyList.size(); qfIdx++)
        {
            queueFamilyIndexList.push_back(queueFamilyList[qfIdx]);
        }
    }

    VkBufferCreateInfo vkBufferCreateInfo = {};
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.size = (VkDeviceSize)size;
    vkBufferCreateInfo.usage = (VkBufferUsageFlags)bufferUsage;
    vkBufferCreateInfo.sharingMode = (VkSharingMode)sharingMode;
    vkBufferCreateInfo.queueFamilyIndexCount =
        (uint32_t)queueFamilyIndexList.size();
    vkBufferCreateInfo.pQueueFamilyIndices = queueFamilyIndexList.data();

    VkExternalMemoryBufferCreateInfo vkExternalMemoryBufferCreateInfo = {};
    if (externalMemoryHandleType != VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE)
    {
        vkExternalMemoryBufferCreateInfo.sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR;
        vkExternalMemoryBufferCreateInfo.pNext = NULL;
        vkExternalMemoryBufferCreateInfo.handleTypes =
            (VkExternalMemoryHandleTypeFlags)externalMemoryHandleType;

        vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;
    }

    vkCreateBuffer(m_device, &vkBufferCreateInfo, NULL, &m_vkBuffer);

    VkMemoryDedicatedRequirements vkMemoryDedicatedRequirements = {};
    vkMemoryDedicatedRequirements.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    vkMemoryDedicatedRequirements.pNext = NULL;

    VkMemoryRequirements2 vkMemoryRequirements = {};
    vkMemoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    vkMemoryRequirements.pNext = &vkMemoryDedicatedRequirements;

    VkBufferMemoryRequirementsInfo2 vkMemoryRequirementsInfo = {};

    vkMemoryRequirementsInfo.sType =
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    vkMemoryRequirementsInfo.buffer = m_vkBuffer;
    vkMemoryRequirementsInfo.pNext = NULL;

    vkGetBufferMemoryRequirements2(m_device, &vkMemoryRequirementsInfo,
                                   &vkMemoryRequirements);

    m_dedicated = vkMemoryDedicatedRequirements.requiresDedicatedAllocation;

    m_size = vkMemoryRequirements.memoryRequirements.size;
    m_alignment = vkMemoryRequirements.memoryRequirements.alignment;
    const VulkanMemoryTypeList &memoryTypeList =
        m_device.getPhysicalDevice().getMemoryTypeList();
    for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
    {
        uint32_t memoryTypeIndex = memoryTypeList[mtIdx];
        if ((1 << memoryTypeIndex)
            & vkMemoryRequirements.memoryRequirements.memoryTypeBits)
        {
            m_memoryTypeList.add(memoryTypeList[mtIdx]);
        }
    }
}

VulkanBuffer::~VulkanBuffer() { vkDestroyBuffer(m_device, m_vkBuffer, NULL); }

uint64_t VulkanBuffer::getSize() const { return m_size; }

uint64_t VulkanBuffer::getAlignment() const { return m_alignment; }

const VulkanMemoryTypeList &VulkanBuffer::getMemoryTypeList() const
{
    return m_memoryTypeList;
}

VulkanBuffer::operator VkBuffer() const { return m_vkBuffer; }

////////////////////////////////
// VulkanImage implementation //
////////////////////////////////

VulkanImage::VulkanImage(const VulkanImage &image)
    : m_device(image.m_device), m_imageType(image.m_imageType),
      m_extent3D(image.m_extent3D), m_format(image.m_format),
      m_numMipLevels(image.m_numMipLevels), m_numLayers(image.m_numLayers),
      m_vkImage(image.m_vkImage), m_size(image.m_size),
      m_alignment(image.m_alignment), m_memoryTypeList(image.m_memoryTypeList)
{}

VulkanImage::VulkanImage(
    const VulkanDevice &device, VulkanImageType imageType, VulkanFormat format,
    const VulkanExtent3D &extent3D, uint32_t numMipLevels, uint32_t arrayLayers,
    VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanImageCreateFlag imageCreateFlag, VulkanImageTiling imageTiling,
    VulkanImageUsage imageUsage, VulkanSharingMode sharingMode)
    : m_device(device), m_imageType(imageType), m_extent3D(extent3D),
      m_format(format), m_numMipLevels(numMipLevels), m_numLayers(arrayLayers),
      m_vkImage(VK_NULL_HANDLE)
{
    VkImageCreateInfo vkImageCreateInfo = {};
    vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vkImageCreateInfo.pNext = NULL;
    vkImageCreateInfo.flags = (VkImageCreateFlags)imageCreateFlag;
    vkImageCreateInfo.imageType = (VkImageType)imageType;
    vkImageCreateInfo.format = (VkFormat)format;
    vkImageCreateInfo.extent = extent3D;
    vkImageCreateInfo.mipLevels = numMipLevels;
    vkImageCreateInfo.arrayLayers = arrayLayers;
    vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vkImageCreateInfo.tiling = (VkImageTiling)imageTiling;
    vkImageCreateInfo.usage = (VkImageUsageFlags)imageUsage;
    vkImageCreateInfo.sharingMode = (VkSharingMode)sharingMode;
    vkImageCreateInfo.queueFamilyIndexCount =
        (uint32_t)m_device.getPhysicalDevice().getQueueFamilyList().size();
    vkImageCreateInfo.pQueueFamilyIndices =
        m_device.getPhysicalDevice().getQueueFamilyList()();
    vkImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkExternalMemoryImageCreateInfo vkExternalMemoryImageCreateInfo = {};
    if (externalMemoryHandleType != VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE)
    {
        vkExternalMemoryImageCreateInfo.sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        vkExternalMemoryImageCreateInfo.pNext = NULL;
        vkExternalMemoryImageCreateInfo.handleTypes =
            (VkExternalMemoryHandleTypeFlags)externalMemoryHandleType;

        vkImageCreateInfo.pNext = &vkExternalMemoryImageCreateInfo;
    }

    vkCreateImage(m_device, &vkImageCreateInfo, NULL, &m_vkImage);
    VulkanImageCreateInfo = vkImageCreateInfo;

    VkMemoryDedicatedRequirements vkMemoryDedicatedRequirements = {};
    vkMemoryDedicatedRequirements.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    vkMemoryDedicatedRequirements.pNext = NULL;

    VkMemoryRequirements2 vkMemoryRequirements = {};
    vkMemoryRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    vkMemoryRequirements.pNext = &vkMemoryDedicatedRequirements;

    VkImageMemoryRequirementsInfo2 vkMemoryRequirementsInfo = {};

    vkMemoryRequirementsInfo.sType =
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    vkMemoryRequirementsInfo.image = m_vkImage;
    vkMemoryRequirementsInfo.pNext = NULL;

    vkGetImageMemoryRequirements2(m_device, &vkMemoryRequirementsInfo,
                                  &vkMemoryRequirements);
    m_size = vkMemoryRequirements.memoryRequirements.size;
    m_alignment = vkMemoryRequirements.memoryRequirements.alignment;
    m_dedicated = vkMemoryDedicatedRequirements.requiresDedicatedAllocation;

    const VulkanMemoryTypeList &memoryTypeList =
        m_device.getPhysicalDevice().getMemoryTypeList();
    for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
    {
        uint32_t memoryTypeIndex = memoryTypeList[mtIdx];
        if ((1 << memoryTypeIndex)
            & vkMemoryRequirements.memoryRequirements.memoryTypeBits)
        {
            m_memoryTypeList.add(memoryTypeList[mtIdx]);
        }
    }
}

VulkanImage::~VulkanImage() { vkDestroyImage(m_device, m_vkImage, NULL); }

VulkanExtent3D VulkanImage::getExtent3D(uint32_t mipLevel) const
{
    return VulkanExtent3D(0, 0, 0);
}

VulkanFormat VulkanImage::getFormat() const { return m_format; }

VkImageCreateInfo VulkanImage::getVkImageCreateInfo() const
{
    return VulkanImageCreateInfo;
}

uint32_t VulkanImage::getNumMipLevels() const { return m_numMipLevels; }

uint32_t VulkanImage::getNumLayers() const { return m_numLayers; }

uint64_t VulkanImage::getSize() const { return m_size; }

uint64_t VulkanImage::getAlignment() const { return m_alignment; }

bool VulkanImage::isDedicated() const { return m_dedicated; }

const VulkanMemoryTypeList &VulkanImage::getMemoryTypeList() const
{
    return m_memoryTypeList;
}

VulkanImage::operator VkImage() const { return m_vkImage; }

//////////////////////////////////
// VulkanImage2D implementation //
//////////////////////////////////

VulkanImage2D::VulkanImage2D(const VulkanImage2D &image2D): VulkanImage(image2D)
{}

VulkanImage2D::VulkanImage2D(
    const VulkanDevice &device, VulkanFormat format, uint32_t width,
    uint32_t height, VulkanImageTiling imageTiling, uint32_t numMipLevels,
    VulkanExternalMemoryHandleType externalMemoryHandleType,
    VulkanImageCreateFlag imageCreateFlag, VulkanImageUsage imageUsage,
    VulkanSharingMode sharingMode)
    : VulkanImage(device, VULKAN_IMAGE_TYPE_2D, format,
                  VulkanExtent3D(width, height, 1), numMipLevels, 1,
                  externalMemoryHandleType, imageCreateFlag, imageTiling,
                  imageUsage, sharingMode)
{}

VulkanImage2D::~VulkanImage2D() {}

VulkanExtent3D VulkanImage2D::getExtent3D(uint32_t mipLevel) const
{
    uint32_t width = std::max(m_extent3D.getWidth() >> mipLevel, uint32_t(1));
    uint32_t height = std::max(m_extent3D.getHeight() >> mipLevel, uint32_t(1));
    uint32_t depth = 1;

    return VulkanExtent3D(width, height, depth);
}

////////////////////////////////////
// VulkanImageView implementation //
////////////////////////////////////

VulkanImageView::VulkanImageView(const VulkanImageView &imageView)
    : m_device(imageView.m_device), m_vkImageView(imageView.m_vkImageView)
{}

VulkanImageView::VulkanImageView(const VulkanDevice &device,
                                 const VulkanImage &image,
                                 VulkanImageViewType imageViewType,
                                 uint32_t baseMipLevel, uint32_t levelCount,
                                 uint32_t baseArrayLayer, uint32_t layerCount)
    : m_device(device), m_vkImageView(VK_NULL_HANDLE)
{
    VkComponentMapping vkComponentMapping = {};
    vkComponentMapping.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    vkComponentMapping.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    vkComponentMapping.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    vkComponentMapping.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    VkImageSubresourceRange vkImageSubresourceRange = {};
    vkImageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageSubresourceRange.baseMipLevel = baseMipLevel;
    vkImageSubresourceRange.levelCount = levelCount;
    vkImageSubresourceRange.baseArrayLayer = baseArrayLayer;
    vkImageSubresourceRange.layerCount = layerCount;

    VkImageViewCreateInfo vkImageViewCreateInfo = {};
    vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vkImageViewCreateInfo.pNext = NULL;
    vkImageViewCreateInfo.flags = 0;
    vkImageViewCreateInfo.image = image;
    vkImageViewCreateInfo.viewType = (VkImageViewType)imageViewType;
    vkImageViewCreateInfo.format = (VkFormat)image.getFormat();
    vkImageViewCreateInfo.components = vkComponentMapping;
    vkImageViewCreateInfo.subresourceRange = vkImageSubresourceRange;

    vkCreateImageView(m_device, &vkImageViewCreateInfo, NULL, &m_vkImageView);
}

VulkanImageView::~VulkanImageView()
{
    vkDestroyImageView(m_device, m_vkImageView, NULL);
}

VulkanImageView::operator VkImageView() const { return m_vkImageView; }

///////////////////////////////////////
// VulkanDeviceMemory implementation //
///////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)

class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES *operator&();
    ~WindowsSecurityAttributes();
};


WindowsSecurityAttributes::WindowsSecurityAttributes()
{
    m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
        1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void **));
    // CHECK_NEQ(m_winPSecurityDescriptor, (PSECURITY_DESCRIPTOR)NULL);
    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor
                           + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
    InitializeSecurityDescriptor(m_winPSecurityDescriptor,
                                 SECURITY_DESCRIPTOR_REVISION);
    SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
        SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0,
                             0, 0, 0, 0, 0, 0, ppSID);
    EXPLICIT_ACCESS explicitAccess;
    ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
    explicitAccess.grfAccessPermissions =
        STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicitAccess.grfAccessMode = SET_ACCESS;
    explicitAccess.grfInheritance = INHERIT_ONLY;
    explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;
    SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);
    SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);
    m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
    m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
    m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES *WindowsSecurityAttributes::operator&()
{
    return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes()
{
    PSID *ppSID = (PSID *)((PBYTE)m_winPSecurityDescriptor
                           + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL *ppACL = (PACL *)((PBYTE)ppSID + sizeof(PSID *));
    if (*ppSID)
    {
        FreeSid(*ppSID);
    }
    if (*ppACL)
    {
        LocalFree(*ppACL);
    }
    free(m_winPSecurityDescriptor);
}

#endif

VulkanDeviceMemory::VulkanDeviceMemory(const VulkanDeviceMemory &deviceMemory)
    : m_device(deviceMemory.m_device),
      m_vkDeviceMemory(deviceMemory.m_vkDeviceMemory),
      m_size(deviceMemory.m_size), m_isDedicated(deviceMemory.m_isDedicated)
{}

VulkanDeviceMemory::VulkanDeviceMemory(
    const VulkanDevice &device, uint64_t size,
    const VulkanMemoryType &memoryType,
    VulkanExternalMemoryHandleType externalMemoryHandleType, const void *name)
    : m_device(device), m_size(size), m_isDedicated(false)
{
#if defined(_WIN32) || defined(_WIN64)
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportMemoryWin32HandleInfoKHR vkExportMemoryWin32HandleInfoKHR = {};
    vkExportMemoryWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vkExportMemoryWin32HandleInfoKHR.pNext = NULL;
    vkExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vkExportMemoryWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vkExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)name;

#endif

    VkExportMemoryAllocateInfoKHR vkExportMemoryAllocateInfoKHR = {};
    vkExportMemoryAllocateInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#if defined(_WIN32) || defined(_WIN64)
    vkExportMemoryAllocateInfoKHR.pNext = externalMemoryHandleType
            & VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT
        ? &vkExportMemoryWin32HandleInfoKHR
        : NULL;
#else
    vkExportMemoryAllocateInfoKHR.pNext = NULL;
#endif
    vkExportMemoryAllocateInfoKHR.handleTypes =
        (VkExternalMemoryHandleTypeFlagsKHR)externalMemoryHandleType;

    VkMemoryAllocateInfo vkMemoryAllocateInfo = {};
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext =
        externalMemoryHandleType ? &vkExportMemoryAllocateInfoKHR : NULL;
    vkMemoryAllocateInfo.allocationSize = m_size;
    vkMemoryAllocateInfo.memoryTypeIndex = (uint32_t)memoryType;

    vkAllocateMemory(m_device, &vkMemoryAllocateInfo, NULL, &m_vkDeviceMemory);
}

VulkanDeviceMemory::VulkanDeviceMemory(
    const VulkanDevice &device, const VulkanImage &image,
    const VulkanMemoryType &memoryType,
    VulkanExternalMemoryHandleType externalMemoryHandleType, const void *name)
    : m_device(device), m_size(image.getSize()),
      m_isDedicated(image.isDedicated())
{
#if defined(_WIN32) || defined(_WIN64)
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportMemoryWin32HandleInfoKHR vkExportMemoryWin32HandleInfoKHR = {};
    vkExportMemoryWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vkExportMemoryWin32HandleInfoKHR.pNext = NULL;
    vkExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vkExportMemoryWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vkExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)name;

#endif

    VkExportMemoryAllocateInfoKHR vkExportMemoryAllocateInfoKHR = {};
    vkExportMemoryAllocateInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#if defined(_WIN32) || defined(_WIN64)
    vkExportMemoryAllocateInfoKHR.pNext = externalMemoryHandleType
            & VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT
        ? &vkExportMemoryWin32HandleInfoKHR
        : NULL;
#else
    vkExportMemoryAllocateInfoKHR.pNext = NULL;
#endif
    vkExportMemoryAllocateInfoKHR.handleTypes =
        (VkExternalMemoryHandleTypeFlagsKHR)externalMemoryHandleType;

    VkMemoryDedicatedAllocateInfo vkMemoryDedicatedAllocateInfo = {};
    vkMemoryDedicatedAllocateInfo.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    vkMemoryDedicatedAllocateInfo.pNext = NULL;
    vkMemoryDedicatedAllocateInfo.image = image;
    vkMemoryDedicatedAllocateInfo.buffer = VK_NULL_HANDLE;

    VkMemoryAllocateInfo vkMemoryAllocateInfo = {};
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.allocationSize = m_size;
    vkMemoryAllocateInfo.memoryTypeIndex = (uint32_t)memoryType;

    if (m_isDedicated)
    {
        vkMemoryAllocateInfo.pNext = &vkMemoryDedicatedAllocateInfo;
        vkMemoryDedicatedAllocateInfo.pNext =
            externalMemoryHandleType ? &vkExportMemoryAllocateInfoKHR : NULL;
    }
    else
    {
        vkMemoryAllocateInfo.pNext =
            externalMemoryHandleType ? &vkExportMemoryAllocateInfoKHR : NULL;
    }

    vkAllocateMemory(m_device, &vkMemoryAllocateInfo, NULL, &m_vkDeviceMemory);
}

VulkanDeviceMemory::VulkanDeviceMemory(
    const VulkanDevice &device, const VulkanBuffer &buffer,
    const VulkanMemoryType &memoryType,
    VulkanExternalMemoryHandleType externalMemoryHandleType, const void *name)
    : m_device(device), m_size(buffer.getSize()),
      m_isDedicated(buffer.isDedicated())
{
#if defined(_WIN32) || defined(_WIN64)
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportMemoryWin32HandleInfoKHR vkExportMemoryWin32HandleInfoKHR = {};
    vkExportMemoryWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vkExportMemoryWin32HandleInfoKHR.pNext = NULL;
    vkExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vkExportMemoryWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vkExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)name;

#endif

    VkExportMemoryAllocateInfoKHR vkExportMemoryAllocateInfoKHR = {};
    vkExportMemoryAllocateInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#if defined(_WIN32) || defined(_WIN64)
    vkExportMemoryAllocateInfoKHR.pNext = externalMemoryHandleType
            & VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT
        ? &vkExportMemoryWin32HandleInfoKHR
        : NULL;
#else
    vkExportMemoryAllocateInfoKHR.pNext = NULL;
#endif
    vkExportMemoryAllocateInfoKHR.handleTypes =
        (VkExternalMemoryHandleTypeFlagsKHR)externalMemoryHandleType;

    VkMemoryDedicatedAllocateInfo vkMemoryDedicatedAllocateInfo = {};
    vkMemoryDedicatedAllocateInfo.sType =
        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    vkMemoryDedicatedAllocateInfo.pNext = NULL;
    vkMemoryDedicatedAllocateInfo.image = VK_NULL_HANDLE;
    vkMemoryDedicatedAllocateInfo.buffer = buffer;

    VkMemoryAllocateInfo vkMemoryAllocateInfo = {};
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.allocationSize = m_size;
    vkMemoryAllocateInfo.memoryTypeIndex = (uint32_t)memoryType;

    if (m_isDedicated)
    {
        vkMemoryAllocateInfo.pNext = &vkMemoryDedicatedAllocateInfo;
        vkMemoryDedicatedAllocateInfo.pNext =
            externalMemoryHandleType ? &vkExportMemoryAllocateInfoKHR : NULL;
    }
    else
    {
        vkMemoryAllocateInfo.pNext =
            externalMemoryHandleType ? &vkExportMemoryAllocateInfoKHR : NULL;
    }


    VkResult res = vkAllocateMemory(m_device, &vkMemoryAllocateInfo, NULL,
                                    &m_vkDeviceMemory);
    ASSERT_SUCCESS(res, "Failed to allocate device memory");
}

VulkanDeviceMemory::~VulkanDeviceMemory()
{
    vkFreeMemory(m_device, m_vkDeviceMemory, NULL);
}

uint64_t VulkanDeviceMemory::getSize() const { return m_size; }

#ifdef _WIN32
HANDLE VulkanDeviceMemory::getHandle(
    VulkanExternalMemoryHandleType externalMemoryHandleType) const
{
    HANDLE handle;

    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
    vkMemoryGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.memory = m_vkDeviceMemory;
    vkMemoryGetWin32HandleInfoKHR.handleType =
        (VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

    vkGetMemoryWin32HandleKHR(m_device, &vkMemoryGetWin32HandleInfoKHR,
                              &handle);

    return handle;
}
#else
int VulkanDeviceMemory::getHandle(
    VulkanExternalMemoryHandleType externalMemoryHandleType) const
{
    if (externalMemoryHandleType
        == VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD)
    {
        int fd;

        VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
        vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        vkMemoryGetFdInfoKHR.pNext = NULL;
        vkMemoryGetFdInfoKHR.memory = m_vkDeviceMemory;
        vkMemoryGetFdInfoKHR.handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

        vkGetMemoryFdKHR(m_device, &vkMemoryGetFdInfoKHR, &fd);

        return fd;
    }
    return HANDLE_ERROR;
}
#endif

bool VulkanDeviceMemory::isDedicated() const { return m_isDedicated; }

void *VulkanDeviceMemory::map(size_t offset, size_t size)
{
    void *pData;

    vkMapMemory(m_device, m_vkDeviceMemory, (VkDeviceSize)offset,
                (VkDeviceSize)size, 0, &pData);

    return pData;
}

void VulkanDeviceMemory::unmap() { vkUnmapMemory(m_device, m_vkDeviceMemory); }

void VulkanDeviceMemory::bindBuffer(const VulkanBuffer &buffer, uint64_t offset)
{
    if (buffer.isDedicated() && !m_isDedicated)
    {
        throw std::runtime_error(
            "Buffer requires dedicated memory.  Failed to bind");
    }
    vkBindBufferMemory(m_device, buffer, m_vkDeviceMemory, offset);
}

void VulkanDeviceMemory::bindImage(const VulkanImage &image, uint64_t offset)
{
    if (image.isDedicated() && !m_isDedicated)
    {
        throw std::runtime_error(
            "Image requires dedicated memory.  Failed to bind");
    }
    vkBindImageMemory(m_device, image, m_vkDeviceMemory, offset);
}

VulkanDeviceMemory::operator VkDeviceMemory() const { return m_vkDeviceMemory; }

////////////////////////////////////
// VulkanSemaphore implementation //
////////////////////////////////////

VulkanSemaphore::VulkanSemaphore(const VulkanSemaphore &semaphore)
    : m_device(semaphore.m_device), m_vkSemaphore(semaphore.m_vkSemaphore)
{}

VulkanSemaphore::VulkanSemaphore(
    const VulkanDevice &device,
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType,
    const std::wstring name)
    : m_device(device), m_name(name)
{
#if defined(_WIN32) || defined(_WIN64)
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportSemaphoreWin32HandleInfoKHR
        vkExportSemaphoreWin32HandleInfoKHR = {};
    vkExportSemaphoreWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    vkExportSemaphoreWin32HandleInfoKHR.pNext = NULL;
    vkExportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    vkExportSemaphoreWin32HandleInfoKHR.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vkExportSemaphoreWin32HandleInfoKHR.name =
        m_name.size() ? (LPCWSTR)m_name.c_str() : NULL;
#endif

    VkExportSemaphoreCreateInfoKHR vkExportSemaphoreCreateInfoKHR = {};
    vkExportSemaphoreCreateInfoKHR.sType =
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#if defined(_WIN32) || defined(_WIN64)
    vkExportSemaphoreCreateInfoKHR.pNext =
        (externalSemaphoreHandleType
         & VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT)
        ? &vkExportSemaphoreWin32HandleInfoKHR
        : NULL;
#else
    vkExportSemaphoreCreateInfoKHR.pNext = NULL;
#endif
    vkExportSemaphoreCreateInfoKHR.handleTypes =
        (VkExternalSemaphoreHandleTypeFlagsKHR)externalSemaphoreHandleType;

    VkSemaphoreCreateInfo vkSemaphoreCreateInfo = {};
    vkSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkSemaphoreCreateInfo.pNext =
        (externalSemaphoreHandleType
         != VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE)
        ? &vkExportSemaphoreCreateInfoKHR
        : NULL;
    vkSemaphoreCreateInfo.flags = 0;

    vkCreateSemaphore(m_device, &vkSemaphoreCreateInfo, NULL, &m_vkSemaphore);
}

VulkanSemaphore::~VulkanSemaphore()
{
    vkDestroySemaphore(m_device, m_vkSemaphore, NULL);
}

#if defined(_WIN32) || defined(_WIN64)
HANDLE VulkanSemaphore::getHandle(
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType) const
{
    HANDLE handle;

    VkSemaphoreGetWin32HandleInfoKHR vkSemaphoreGetWin32HandleInfoKHR = {};
    vkSemaphoreGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    vkSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
    vkSemaphoreGetWin32HandleInfoKHR.semaphore = m_vkSemaphore;
    vkSemaphoreGetWin32HandleInfoKHR.handleType =
        (VkExternalSemaphoreHandleTypeFlagBitsKHR)externalSemaphoreHandleType;

    vkGetSemaphoreWin32HandleKHR(m_device, &vkSemaphoreGetWin32HandleInfoKHR,
                                 &handle);

    return handle;
}
#else
int VulkanSemaphore::getHandle(
    VulkanExternalSemaphoreHandleType externalSemaphoreHandleType) const
{
    if (externalSemaphoreHandleType
        == VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD)
    {
        int fd;

        VkSemaphoreGetFdInfoKHR vkSemaphoreGetFdInfoKHR = {};
        vkSemaphoreGetFdInfoKHR.sType =
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        vkSemaphoreGetFdInfoKHR.pNext = NULL;
        vkSemaphoreGetFdInfoKHR.semaphore = m_vkSemaphore;
        vkSemaphoreGetFdInfoKHR.handleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

        vkGetSemaphoreFdKHR(m_device, &vkSemaphoreGetFdInfoKHR, &fd);

        return fd;
    }
    return HANDLE_ERROR;
}
#endif

const std::wstring &VulkanSemaphore::getName() const { return m_name; }

VulkanSemaphore::operator VkSemaphore() const { return m_vkSemaphore; }
