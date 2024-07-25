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

#ifndef _vulkan_api_list_hpp_
#define _vulkan_api_list_hpp_

#define VK_FUNC_LIST                                                           \
    VK_FUNC_DECL(vkEnumerateInstanceVersion)                                   \
    VK_FUNC_DECL(vkEnumerateInstanceExtensionProperties)                       \
    VK_FUNC_DECL(vkEnumerateInstanceLayerProperties)                           \
    VK_FUNC_DECL(vkCreateInstance)                                             \
    VK_FUNC_DECL(vkGetInstanceProcAddr)                                        \
    VK_FUNC_DECL(vkGetDeviceProcAddr)                                          \
    VK_FUNC_DECL(vkEnumeratePhysicalDevices)                                   \
    VK_FUNC_DECL(vkGetPhysicalDeviceProperties)                                \
    VK_FUNC_DECL(vkCreateDevice)                                               \
    VK_FUNC_DECL(vkDestroyDevice)                                              \
    VK_FUNC_DECL(vkGetDeviceQueue)                                             \
    VK_FUNC_DECL(vkQueueWaitIdle)                                              \
    VK_FUNC_DECL(vkCreateDescriptorSetLayout)                                  \
    VK_FUNC_DECL(vkCreatePipelineLayout)                                       \
    VK_FUNC_DECL(vkCreateShaderModule)                                         \
    VK_FUNC_DECL(vkCreateComputePipelines)                                     \
    VK_FUNC_DECL(vkCreateDescriptorPool)                                       \
    VK_FUNC_DECL(vkAllocateDescriptorSets)                                     \
    VK_FUNC_DECL(vkFreeDescriptorSets)                                         \
    VK_FUNC_DECL(vkAllocateCommandBuffers)                                     \
    VK_FUNC_DECL(vkBeginCommandBuffer)                                         \
    VK_FUNC_DECL(vkCmdBindPipeline)                                            \
    VK_FUNC_DECL(vkCmdBindDescriptorSets)                                      \
    VK_FUNC_DECL(vkCmdPipelineBarrier)                                         \
    VK_FUNC_DECL(vkCmdDispatch)                                                \
    VK_FUNC_DECL(vkCmdFillBuffer)                                              \
    VK_FUNC_DECL(vkCmdCopyBuffer)                                              \
    VK_FUNC_DECL(vkCmdUpdateBuffer)                                            \
    VK_FUNC_DECL(vkCmdCopyBufferToImage)                                       \
    VK_FUNC_DECL(vkCmdCopyImageToBuffer)                                       \
    VK_FUNC_DECL(vkEndCommandBuffer)                                           \
    VK_FUNC_DECL(vkCreateBuffer)                                               \
    VK_FUNC_DECL(vkCreateImageView)                                            \
    VK_FUNC_DECL(vkAllocateMemory)                                             \
    VK_FUNC_DECL(vkMapMemory)                                                  \
    VK_FUNC_DECL(vkBindBufferMemory)                                           \
    VK_FUNC_DECL(vkBindImageMemory)                                            \
    VK_FUNC_DECL(vkUnmapMemory)                                                \
    VK_FUNC_DECL(vkFreeMemory)                                                 \
    VK_FUNC_DECL(vkCreateCommandPool)                                          \
    VK_FUNC_DECL(vkResetCommandPool)                                           \
    VK_FUNC_DECL(vkDestroyCommandPool)                                         \
    VK_FUNC_DECL(vkResetCommandBuffer)                                         \
    VK_FUNC_DECL(vkFreeCommandBuffers)                                         \
    VK_FUNC_DECL(vkQueueSubmit)                                                \
    VK_FUNC_DECL(vkCmdExecuteCommands)                                         \
    VK_FUNC_DECL(vkCreateFence)                                                \
    VK_FUNC_DECL(vkDestroyFence)                                               \
    VK_FUNC_DECL(vkGetFenceStatus)                                             \
    VK_FUNC_DECL(vkResetFences)                                                \
    VK_FUNC_DECL(vkWaitForFences)                                              \
    VK_FUNC_DECL(vkCreateSemaphore)                                            \
    VK_FUNC_DECL(vkDestroySemaphore)                                           \
    VK_FUNC_DECL(vkCreateEvent)                                                \
    VK_FUNC_DECL(vkDestroyImageView)                                           \
    VK_FUNC_DECL(vkCreateImage)                                                \
    VK_FUNC_DECL(vkGetImageMemoryRequirements)                                 \
    VK_FUNC_DECL(vkGetImageMemoryRequirements2)                                \
    VK_FUNC_DECL(vkDestroyImage)                                               \
    VK_FUNC_DECL(vkDestroyBuffer)                                              \
    VK_FUNC_DECL(vkDestroyPipeline)                                            \
    VK_FUNC_DECL(vkDestroyShaderModule)                                        \
    VK_FUNC_DECL(vkGetPhysicalDeviceMemoryProperties)                          \
    VK_FUNC_DECL(vkDestroyInstance)                                            \
    VK_FUNC_DECL(vkUpdateDescriptorSets)                                       \
    VK_FUNC_DECL(vkDestroyDescriptorPool)                                      \
    VK_FUNC_DECL(vkDestroyPipelineLayout)                                      \
    VK_FUNC_DECL(vkDestroyDescriptorSetLayout)                                 \
    VK_FUNC_DECL(vkGetPhysicalDeviceQueueFamilyProperties)                     \
    VK_FUNC_DECL(vkGetPhysicalDeviceFeatures)                                  \
    VK_FUNC_DECL(vkGetPhysicalDeviceProperties2)                               \
    VK_FUNC_DECL(vkGetBufferMemoryRequirements)                                \
    VK_FUNC_DECL(vkGetBufferMemoryRequirements2)                               \
    VK_FUNC_DECL(vkGetMemoryFdKHR)                                             \
    VK_FUNC_DECL(vkGetSemaphoreFdKHR)                                          \
    VK_FUNC_DECL(vkEnumeratePhysicalDeviceGroups)                              \
    VK_FUNC_DECL(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)                    \
    VK_FUNC_DECL(vkGetPhysicalDeviceSurfaceFormatsKHR)                         \
    VK_FUNC_DECL(vkGetPhysicalDeviceSurfacePresentModesKHR)                    \
    VK_FUNC_DECL(vkEnumerateDeviceExtensionProperties)                         \
    VK_FUNC_DECL(vkGetPhysicalDeviceSurfaceSupportKHR)                         \
    VK_FUNC_DECL(vkImportSemaphoreFdKHR)                                       \
    VK_FUNC_DECL(vkGetPhysicalDeviceExternalSemaphorePropertiesKHR)
#define VK_WINDOWS_FUNC_LIST                                                   \
    VK_FUNC_DECL(vkGetMemoryWin32HandleKHR)                                    \
    VK_FUNC_DECL(vkGetSemaphoreWin32HandleKHR)                                 \
    VK_FUNC_DECL(vkImportSemaphoreWin32HandleKHR)

#define vkEnumerateInstanceVersion _vkEnumerateInstanceVersion
#define vkEnumerateInstanceExtensionProperties                                 \
    _vkEnumerateInstanceExtensionProperties
#define vkEnumerateInstanceLayerProperties _vkEnumerateInstanceLayerProperties
#define vkCreateInstance _vkCreateInstance
#define vkGetInstanceProcAddr _vkGetInstanceProcAddr
#define vkGetDeviceProcAddr _vkGetDeviceProcAddr
#define vkEnumeratePhysicalDevices _vkEnumeratePhysicalDevices
#define vkGetPhysicalDeviceProperties _vkGetPhysicalDeviceProperties
#define vkCreateDevice _vkCreateDevice
#define vkDestroyDevice _vkDestroyDevice
#define vkGetDeviceQueue _vkGetDeviceQueue
#define vkQueueWaitIdle _vkQueueWaitIdle
#define vkCreateDescriptorSetLayout _vkCreateDescriptorSetLayout
#define vkCreatePipelineLayout _vkCreatePipelineLayout
#define vkCreateShaderModule _vkCreateShaderModule
#define vkCreateComputePipelines _vkCreateComputePipelines
#define vkCreateDescriptorPool _vkCreateDescriptorPool
#define vkAllocateDescriptorSets _vkAllocateDescriptorSets
#define vkFreeDescriptorSets _vkFreeDescriptorSets
#define vkAllocateCommandBuffers _vkAllocateCommandBuffers
#define vkBeginCommandBuffer _vkBeginCommandBuffer
#define vkCmdBindPipeline _vkCmdBindPipeline
#define vkCmdBindDescriptorSets _vkCmdBindDescriptorSets
#define vkCmdPipelineBarrier _vkCmdPipelineBarrier
#define vkCmdDispatch _vkCmdDispatch
#define vkCmdFillBuffer _vkCmdFillBuffer
#define vkCmdCopyBuffer _vkCmdCopyBuffer
#define vkCmdUpdateBuffer _vkCmdUpdateBuffer
#define vkCmdCopyBufferToImage _vkCmdCopyBufferToImage
#define vkCmdCopyImageToBuffer _vkCmdCopyImageToBuffer
#define vkEndCommandBuffer _vkEndCommandBuffer
#define vkCreateBuffer _vkCreateBuffer
#define vkCreateImageView _vkCreateImageView
#define vkAllocateMemory _vkAllocateMemory
#define vkMapMemory _vkMapMemory
#define vkBindBufferMemory _vkBindBufferMemory
#define vkBindImageMemory _vkBindImageMemory
#define vkUnmapMemory _vkUnmapMemory
#define vkFreeMemory _vkFreeMemory
#define vkCreateCommandPool _vkCreateCommandPool
#define vkResetCommandPool _vkResetCommandPool
#define vkDestroyCommandPool _vkDestroyCommandPool
#define vkResetCommandBuffer _vkResetCommandBuffer
#define vkFreeCommandBuffers _vkFreeCommandBuffers
#define vkQueueSubmit _vkQueueSubmit
#define vkCmdExecuteCommands _vkCmdExecuteCommands
#define vkCreateFence _vkCreateFence
#define vkDestroyFence _vkDestroyFence
#define vkGetFenceStatus _vkGetFenceStatus
#define vkResetFences _vkResetFences
#define vkWaitForFences _vkWaitForFences
#define vkCreateSemaphore _vkCreateSemaphore
#define vkDestroySemaphore _vkDestroySemaphore
#define vkCreateEvent _vkCreateEvent
#define vkDestroyImageView _vkDestroyImageView
#define vkCreateImage _vkCreateImage
#define vkGetImageMemoryRequirements _vkGetImageMemoryRequirements
#define vkGetImageMemoryRequirements2 _vkGetImageMemoryRequirements2
#define vkDestroyImage _vkDestroyImage
#define vkDestroyBuffer _vkDestroyBuffer
#define vkDestroyPipeline _vkDestroyPipeline
#define vkDestroyShaderModule _vkDestroyShaderModule
#define vkGetPhysicalDeviceMemoryProperties _vkGetPhysicalDeviceMemoryProperties
#define vkDestroyInstance _vkDestroyInstance
#define vkUpdateDescriptorSets _vkUpdateDescriptorSets
#define vkDestroyDescriptorPool _vkDestroyDescriptorPool
#define vkDestroyPipelineLayout _vkDestroyPipelineLayout
#define vkDestroyDescriptorSetLayout _vkDestroyDescriptorSetLayout
#define vkGetPhysicalDeviceQueueFamilyProperties                               \
    _vkGetPhysicalDeviceQueueFamilyProperties
#define vkGetPhysicalDeviceFeatures _vkGetPhysicalDeviceFeatures
#define vkGetPhysicalDeviceProperties2 _vkGetPhysicalDeviceProperties2
#define vkGetBufferMemoryRequirements _vkGetBufferMemoryRequirements
#define vkGetBufferMemoryRequirements2 _vkGetBufferMemoryRequirements2
#define vkGetMemoryFdKHR _vkGetMemoryFdKHR
#define vkGetSemaphoreFdKHR _vkGetSemaphoreFdKHR
#define vkEnumeratePhysicalDeviceGroups _vkEnumeratePhysicalDeviceGroups
#define vkGetPhysicalDeviceSurfaceCapabilitiesKHR                              \
    _vkGetPhysicalDeviceSurfaceCapabilitiesKHR
#define vkGetPhysicalDeviceSurfaceFormatsKHR                                   \
    _vkGetPhysicalDeviceSurfaceFormatsKHR
#define vkGetPhysicalDeviceSurfacePresentModesKHR                              \
    _vkGetPhysicalDeviceSurfacePresentModesKHR
#define vkEnumerateDeviceExtensionProperties                                   \
    _vkEnumerateDeviceExtensionProperties
#define vkGetPhysicalDeviceSurfaceSupportKHR                                   \
    _vkGetPhysicalDeviceSurfaceSupportKHR
#define vkImportSemaphoreFdKHR _vkImportSemaphoreFdKHR
#define vkGetPhysicalDeviceExternalSemaphorePropertiesKHR                      \
    _vkGetPhysicalDeviceExternalSemaphorePropertiesKHR
#define vkGetMemoryWin32HandleKHR _vkGetMemoryWin32HandleKHR
#define vkGetSemaphoreWin32HandleKHR _vkGetSemaphoreWin32HandleKHR
#define vkImportSemaphoreWin32HandleKHR _vkImportSemaphoreWin32HandleKHR

#endif //_vulkan_api_list_hpp_
