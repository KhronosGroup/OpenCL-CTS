#ifndef _vulkan_wrapper_types_hpp_
#define _vulkan_wrapper_types_hpp_

#include <vulkan/vulkan.h>

#define VULKAN_MIN_BUFFER_OFFSET_COPY_ALIGNMENT   4
#define VULKAN_REMAINING_MIP_LEVELS               VK_REMAINING_MIP_LEVELS
#define VULKAN_REMAINING_ARRAY_LAYERS             VK_REMAINING_ARRAY_LAYERS

class VulkanInstance;
class VulkanPhysicalDevice;
class VulkanMemoryHeap;
class VulkanMemoryType;
class VulkanQueueFamily;
class VulkanDevice;
class VulkanQueue;
class VulkanDescriptorSetLayoutBinding;
class VulkanDescriptorSetLayout;
class VulkanPipelineLayout;
class VulkanShaderModule;
class VulkanPipeline;
class VulkanComputePipeline;
class VulkanDescriptorPool;
class VulkanDescriptorSet;
class VulkanCommandPool;
class VulkanCommandBuffer;
class VulkanBuffer;
class VulkanOffset3D;
class VulkanExtent3D;
class VulkanImage;
class VulkanImage2D;
class VulkanImageView;
class VulkanDeviceMemory;
class VulkanSemaphore;

class VulkanPhysicalDeviceList;
class VulkanMemoryHeapList;
class VulkanMemoryTypeList;
class VulkanQueueFamilyList;
class VulkanQueueFamilyToQueueCountMap;
class VulkanQueueFamilyToQueueListMap;
class VulkanQueueList;
class VulkanCommandBufferList;
class VulkanDescriptorSetLayoutList;
class VulkanBufferList;
class VulkanImage2DList;
class VulkanImageViewList;
class VulkanDeviceMemoryList;
class VulkanSemaphoreList;

enum VulkanQueueFlag
{
    VULKAN_QUEUE_FLAG_GRAPHICS = VK_QUEUE_GRAPHICS_BIT,
    VULKAN_QUEUE_FLAG_COMPUTE  = VK_QUEUE_COMPUTE_BIT,
    VULKAN_QUEUE_FLAG_TRANSFER = VK_QUEUE_TRANSFER_BIT,
    VULKAN_QUEUE_FLAG_MASK_ALL = VULKAN_QUEUE_FLAG_GRAPHICS | VULKAN_QUEUE_FLAG_COMPUTE | VULKAN_QUEUE_FLAG_TRANSFER
};

enum VulkanDescriptorType
{
    VULKAN_DESCRIPTOR_TYPE_SAMPLER                = VK_DESCRIPTOR_TYPE_SAMPLER,
    VULKAN_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    VULKAN_DESCRIPTOR_TYPE_SAMPLED_IMAGE          = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    VULKAN_DESCRIPTOR_TYPE_STORAGE_IMAGE          = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    VULKAN_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER   = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    VULKAN_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER   = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    VULKAN_DESCRIPTOR_TYPE_UNIFORM_BUFFER         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER         = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VULKAN_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    VULKAN_DESCRIPTOR_TYPE_INPUT_ATTACHMENT       = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
};

enum VulkanShaderStage
{
    VULKAN_SHADER_STAGE_VERTEX       = VK_SHADER_STAGE_VERTEX_BIT,
    VULKAN_SHADER_STAGE_FRAGMENT     = VK_SHADER_STAGE_FRAGMENT_BIT,
    VULKAN_SHADER_STAGE_COMPUTE      = VK_SHADER_STAGE_COMPUTE_BIT,
    VULKAN_SHADER_STAGE_ALL_GRAPHICS = VK_SHADER_STAGE_ALL_GRAPHICS,
    VULKAN_SHADER_STAGE_ALL          = VK_SHADER_STAGE_ALL
};

enum VulkanPipelineBindPoint
{
    VULKAN_PIPELINE_BIND_POINT_GRAPHICS = VK_PIPELINE_BIND_POINT_GRAPHICS,
    VULKAN_PIPELINE_BIND_POINT_COMPUTE  = VK_PIPELINE_BIND_POINT_COMPUTE
};

enum VulkanMemoryTypeProperty
{
    VULKAN_MEMORY_TYPE_PROPERTY_NONE                                      = 0,
    VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL                              = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT                     = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_CACHED                       = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_CACHED_COHERENT              = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_COHERENT        = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_CACHED          = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL_HOST_VISIBLE_CACHED_COHERENT = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
};

enum VulkanMemoryHeapFlag
{
    VULKAN_MEMORY_HEAP_FLAG_NONE         = 0,
    VULKAN_MEMORY_HEAP_FLAG_DEVICE_LOCAL = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
};

enum VulkanExternalMemoryHandleType
{
    VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_NONE                = 0,
    VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD           = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT     = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR,
    VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT_KMT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR | VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR
};

enum VulkanExternalSemaphoreHandleType
{
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NONE                = 0,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD           = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT     = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT    = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR,
    VULKAN_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_NT_KMT = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR | VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR
};

enum VulkanBufferUsage
{
    VULKAN_BUFFER_USAGE_TRANSFER_SRC                    = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VULKAN_BUFFER_USAGE_TRANSFER_DST                    = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VULKAN_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER            = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_STORAGE_TEXEL_BUFFER            = VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_UNIFORM_BUFFER                  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_STORAGE_BUFFER                  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_INDEX_BUFFER                    = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_VERTEX_BUFFER                   = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_INDIRECT_BUFFER                 = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
    VULKAN_BUFFER_USAGE_STORAGE_BUFFER_TRANSFER_SRC_DST = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VULKAN_BUFFER_USAGE_UNIFORM_BUFFER_TRANSFER_SRC_DST = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
};

enum VulkanSharingMode
{
    VULKAN_SHARING_MODE_EXCLUSIVE  = VK_SHARING_MODE_EXCLUSIVE,
    VULKAN_SHARING_MODE_CONCURRENT = VK_SHARING_MODE_CONCURRENT
};

enum VulkanImageType
{
    VULKAN_IMAGE_TYPE_1D = VK_IMAGE_TYPE_1D,
    VULKAN_IMAGE_TYPE_2D = VK_IMAGE_TYPE_2D,
    VULKAN_IMAGE_TYPE_3D = VK_IMAGE_TYPE_3D
};

enum VulkanFormat
{
    VULKAN_FORMAT_UNDEFINED                  = VK_FORMAT_UNDEFINED,
    VULKAN_FORMAT_R4G4_UNORM_PACK8           = VK_FORMAT_R4G4_UNORM_PACK8,
    VULKAN_FORMAT_R4G4B4A4_UNORM_PACK16      = VK_FORMAT_R4G4B4A4_UNORM_PACK16,
    VULKAN_FORMAT_B4G4R4A4_UNORM_PACK16      = VK_FORMAT_B4G4R4A4_UNORM_PACK16,
    VULKAN_FORMAT_R5G6B5_UNORM_PACK16        = VK_FORMAT_R5G6B5_UNORM_PACK16,
    VULKAN_FORMAT_B5G6R5_UNORM_PACK16        = VK_FORMAT_B5G6R5_UNORM_PACK16,
    VULKAN_FORMAT_R5G5B5A1_UNORM_PACK16      = VK_FORMAT_R5G5B5A1_UNORM_PACK16,
    VULKAN_FORMAT_B5G5R5A1_UNORM_PACK16      = VK_FORMAT_B5G5R5A1_UNORM_PACK16,
    VULKAN_FORMAT_A1R5G5B5_UNORM_PACK16      = VK_FORMAT_A1R5G5B5_UNORM_PACK16,
    VULKAN_FORMAT_R8_UNORM                   = VK_FORMAT_R8_UNORM,
    VULKAN_FORMAT_R8_SNORM                   = VK_FORMAT_R8_SNORM,
    VULKAN_FORMAT_R8_USCALED                 = VK_FORMAT_R8_USCALED,
    VULKAN_FORMAT_R8_SSCALED                 = VK_FORMAT_R8_SSCALED,
    VULKAN_FORMAT_R8_UINT                    = VK_FORMAT_R8_UINT,
    VULKAN_FORMAT_R8_SINT                    = VK_FORMAT_R8_SINT,
    VULKAN_FORMAT_R8_SRGB                    = VK_FORMAT_R8_SRGB,
    VULKAN_FORMAT_R8G8_SNORM                 = VK_FORMAT_R8G8_SNORM,
    VULKAN_FORMAT_R8G8_UNORM                 = VK_FORMAT_R8G8_UNORM,
    VULKAN_FORMAT_R8G8_USCALED               = VK_FORMAT_R8G8_USCALED,
    VULKAN_FORMAT_R8G8_SSCALED               = VK_FORMAT_R8G8_SSCALED,
    VULKAN_FORMAT_R8G8_UINT                  = VK_FORMAT_R8G8_UINT,
    VULKAN_FORMAT_R8G8_SINT                  = VK_FORMAT_R8G8_SINT,
    VULKAN_FORMAT_R8G8_SRGB                  = VK_FORMAT_R8G8_SRGB,
    VULKAN_FORMAT_R8G8B8_UNORM               = VK_FORMAT_R8G8B8_UNORM,
    VULKAN_FORMAT_R8G8B8_SNORM               = VK_FORMAT_R8G8B8_SNORM,
    VULKAN_FORMAT_R8G8B8_USCALED             = VK_FORMAT_R8G8B8_USCALED,
    VULKAN_FORMAT_R8G8B8_SSCALED             = VK_FORMAT_R8G8B8_SSCALED,
    VULKAN_FORMAT_R8G8B8_UINT                = VK_FORMAT_R8G8B8_UINT,
    VULKAN_FORMAT_R8G8B8_SINT                = VK_FORMAT_R8G8B8_SINT,
    VULKAN_FORMAT_R8G8B8_SRGB                = VK_FORMAT_R8G8B8_SRGB,
    VULKAN_FORMAT_B8G8R8_UNORM               = VK_FORMAT_B8G8R8_UNORM,
    VULKAN_FORMAT_B8G8R8_SNORM               = VK_FORMAT_B8G8R8_SNORM,
    VULKAN_FORMAT_B8G8R8_USCALED             = VK_FORMAT_B8G8R8_USCALED,
    VULKAN_FORMAT_B8G8R8_SSCALED             = VK_FORMAT_B8G8R8_SSCALED,
    VULKAN_FORMAT_B8G8R8_UINT                = VK_FORMAT_B8G8R8_UINT,
    VULKAN_FORMAT_B8G8R8_SINT                = VK_FORMAT_B8G8R8_SINT,
    VULKAN_FORMAT_B8G8R8_SRGB                = VK_FORMAT_B8G8R8_SRGB,
    VULKAN_FORMAT_R8G8B8A8_UNORM             = VK_FORMAT_R8G8B8A8_UNORM,
    VULKAN_FORMAT_R8G8B8A8_SNORM             = VK_FORMAT_R8G8B8A8_SNORM,
    VULKAN_FORMAT_R8G8B8A8_USCALED           = VK_FORMAT_R8G8B8A8_USCALED,
    VULKAN_FORMAT_R8G8B8A8_SSCALED           = VK_FORMAT_R8G8B8A8_SSCALED,
    VULKAN_FORMAT_R8G8B8A8_UINT              = VK_FORMAT_R8G8B8A8_UINT,
    VULKAN_FORMAT_R8G8B8A8_SINT              = VK_FORMAT_R8G8B8A8_SINT,
    VULKAN_FORMAT_R8G8B8A8_SRGB              = VK_FORMAT_R8G8B8A8_SRGB,
    VULKAN_FORMAT_B8G8R8A8_UNORM             = VK_FORMAT_B8G8R8A8_UNORM,
    VULKAN_FORMAT_B8G8R8A8_SNORM             = VK_FORMAT_B8G8R8A8_SNORM,
    VULKAN_FORMAT_B8G8R8A8_USCALED           = VK_FORMAT_B8G8R8A8_USCALED,
    VULKAN_FORMAT_B8G8R8A8_SSCALED           = VK_FORMAT_B8G8R8A8_SSCALED,
    VULKAN_FORMAT_B8G8R8A8_UINT              = VK_FORMAT_B8G8R8A8_UINT,
    VULKAN_FORMAT_B8G8R8A8_SINT              = VK_FORMAT_B8G8R8A8_SINT,
    VULKAN_FORMAT_B8G8R8A8_SRGB              = VK_FORMAT_B8G8R8A8_SRGB,
    VULKAN_FORMAT_A8B8G8R8_UNORM_PACK32      = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
    VULKAN_FORMAT_A8B8G8R8_SNORM_PACK32      = VK_FORMAT_A8B8G8R8_SNORM_PACK32,
    VULKAN_FORMAT_A8B8G8R8_USCALED_PACK32    = VK_FORMAT_A8B8G8R8_USCALED_PACK32,
    VULKAN_FORMAT_A8B8G8R8_SSCALED_PACK32    = VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
    VULKAN_FORMAT_A8B8G8R8_UINT_PACK32       = VK_FORMAT_A8B8G8R8_UINT_PACK32,
    VULKAN_FORMAT_A8B8G8R8_SINT_PACK32       = VK_FORMAT_A8B8G8R8_SINT_PACK32,
    VULKAN_FORMAT_A8B8G8R8_SRGB_PACK32       = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
    VULKAN_FORMAT_A2R10G10B10_UNORM_PACK32   = VK_FORMAT_A2R10G10B10_UNORM_PACK32,
    VULKAN_FORMAT_A2R10G10B10_SNORM_PACK32   = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
    VULKAN_FORMAT_A2R10G10B10_USCALED_PACK32 = VK_FORMAT_A2R10G10B10_USCALED_PACK32,
    VULKAN_FORMAT_A2R10G10B10_SSCALED_PACK32 = VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
    VULKAN_FORMAT_A2R10G10B10_UINT_PACK32    = VK_FORMAT_A2R10G10B10_UINT_PACK32,
    VULKAN_FORMAT_A2R10G10B10_SINT_PACK32    = VK_FORMAT_A2R10G10B10_SINT_PACK32,
    VULKAN_FORMAT_A2B10G10R10_UNORM_PACK32   = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VULKAN_FORMAT_A2B10G10R10_SNORM_PACK32   = VK_FORMAT_A2B10G10R10_SNORM_PACK32,
    VULKAN_FORMAT_A2B10G10R10_USCALED_PACK32 = VK_FORMAT_A2B10G10R10_USCALED_PACK32,
    VULKAN_FORMAT_A2B10G10R10_SSCALED_PACK32 = VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
    VULKAN_FORMAT_A2B10G10R10_UINT_PACK32    = VK_FORMAT_A2B10G10R10_UINT_PACK32,
    VULKAN_FORMAT_A2B10G10R10_SINT_PACK32    = VK_FORMAT_A2B10G10R10_SINT_PACK32,
    VULKAN_FORMAT_R16_UNORM                  = VK_FORMAT_R16_UNORM,
    VULKAN_FORMAT_R16_SNORM                  = VK_FORMAT_R16_SNORM,
    VULKAN_FORMAT_R16_USCALED                = VK_FORMAT_R16_USCALED,
    VULKAN_FORMAT_R16_SSCALED                = VK_FORMAT_R16_SSCALED,
    VULKAN_FORMAT_R16_UINT                   = VK_FORMAT_R16_UINT,
    VULKAN_FORMAT_R16_SINT                   = VK_FORMAT_R16_SINT,
    VULKAN_FORMAT_R16_SFLOAT                 = VK_FORMAT_R16_SFLOAT,
    VULKAN_FORMAT_R16G16_UNORM               = VK_FORMAT_R16G16_UNORM,
    VULKAN_FORMAT_R16G16_SNORM               = VK_FORMAT_R16G16_SNORM,
    VULKAN_FORMAT_R16G16_USCALED             = VK_FORMAT_R16G16_USCALED,
    VULKAN_FORMAT_R16G16_SSCALED             = VK_FORMAT_R16G16_SSCALED,
    VULKAN_FORMAT_R16G16_UINT                = VK_FORMAT_R16G16_UINT,
    VULKAN_FORMAT_R16G16_SINT                = VK_FORMAT_R16G16_SINT,
    VULKAN_FORMAT_R16G16_SFLOAT              = VK_FORMAT_R16G16_SFLOAT,
    VULKAN_FORMAT_R16G16B16_UNORM            = VK_FORMAT_R16G16B16_UNORM,
    VULKAN_FORMAT_R16G16B16_SNORM            = VK_FORMAT_R16G16B16_SNORM,
    VULKAN_FORMAT_R16G16B16_USCALED          = VK_FORMAT_R16G16B16_USCALED,
    VULKAN_FORMAT_R16G16B16_SSCALED          = VK_FORMAT_R16G16B16_SSCALED,
    VULKAN_FORMAT_R16G16B16_UINT             = VK_FORMAT_R16G16B16_UINT,
    VULKAN_FORMAT_R16G16B16_SINT             = VK_FORMAT_R16G16B16_SINT,
    VULKAN_FORMAT_R16G16B16_SFLOAT           = VK_FORMAT_R16G16B16_SFLOAT,
    VULKAN_FORMAT_R16G16B16A16_UNORM         = VK_FORMAT_R16G16B16A16_UNORM,
    VULKAN_FORMAT_R16G16B16A16_SNORM         = VK_FORMAT_R16G16B16A16_SNORM,
    VULKAN_FORMAT_R16G16B16A16_USCALED       = VK_FORMAT_R16G16B16A16_USCALED,
    VULKAN_FORMAT_R16G16B16A16_SSCALED       = VK_FORMAT_R16G16B16A16_SSCALED,
    VULKAN_FORMAT_R16G16B16A16_UINT          = VK_FORMAT_R16G16B16A16_UINT,
    VULKAN_FORMAT_R16G16B16A16_SINT          = VK_FORMAT_R16G16B16A16_SINT,
    VULKAN_FORMAT_R16G16B16A16_SFLOAT        = VK_FORMAT_R16G16B16A16_SFLOAT,
    VULKAN_FORMAT_R32_UINT                   = VK_FORMAT_R32_UINT,
    VULKAN_FORMAT_R32_SINT                   = VK_FORMAT_R32_SINT,
    VULKAN_FORMAT_R32_SFLOAT                 = VK_FORMAT_R32_SFLOAT,
    VULKAN_FORMAT_R32G32_UINT                = VK_FORMAT_R32G32_UINT,
    VULKAN_FORMAT_R32G32_SINT                = VK_FORMAT_R32G32_SINT,
    VULKAN_FORMAT_R32G32_SFLOAT              = VK_FORMAT_R32G32_SFLOAT,
    VULKAN_FORMAT_R32G32B32_UINT             = VK_FORMAT_R32G32B32_UINT,
    VULKAN_FORMAT_R32G32B32_SINT             = VK_FORMAT_R32G32B32_SINT,
    VULKAN_FORMAT_R32G32B32_SFLOAT           = VK_FORMAT_R32G32B32_SFLOAT,
    VULKAN_FORMAT_R32G32B32A32_UINT          = VK_FORMAT_R32G32B32A32_UINT,
    VULKAN_FORMAT_R32G32B32A32_SINT          = VK_FORMAT_R32G32B32A32_SINT,
    VULKAN_FORMAT_R32G32B32A32_SFLOAT        = VK_FORMAT_R32G32B32A32_SFLOAT,
    VULKAN_FORMAT_R64_UINT                   = VK_FORMAT_R64_UINT,
    VULKAN_FORMAT_R64_SINT                   = VK_FORMAT_R64_SINT,
    VULKAN_FORMAT_R64_SFLOAT                 = VK_FORMAT_R64_SFLOAT,
    VULKAN_FORMAT_R64G64_UINT                = VK_FORMAT_R64G64_UINT,
    VULKAN_FORMAT_R64G64_SINT                = VK_FORMAT_R64G64_SINT,
    VULKAN_FORMAT_R64G64_SFLOAT              = VK_FORMAT_R64G64_SFLOAT,
    VULKAN_FORMAT_R64G64B64_UINT             = VK_FORMAT_R64G64B64_UINT,
    VULKAN_FORMAT_R64G64B64_SINT             = VK_FORMAT_R64G64B64_SINT,
    VULKAN_FORMAT_R64G64B64_SFLOAT           = VK_FORMAT_R64G64B64_SFLOAT,
    VULKAN_FORMAT_R64G64B64A64_UINT          = VK_FORMAT_R64G64B64A64_UINT,
    VULKAN_FORMAT_R64G64B64A64_SINT          = VK_FORMAT_R64G64B64A64_SINT,
    VULKAN_FORMAT_R64G64B64A64_SFLOAT        = VK_FORMAT_R64G64B64A64_SFLOAT,
    VULKAN_FORMAT_B10G11R11_UFLOAT_PACK32    = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VULKAN_FORMAT_E5B9G9R9_UFLOAT_PACK32     = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
    VULKAN_FORMAT_D16_UNORM                  = VK_FORMAT_D16_UNORM,
    VULKAN_FORMAT_X8_D24_UNORM_PACK32        = VK_FORMAT_X8_D24_UNORM_PACK32,
    VULKAN_FORMAT_D32_SFLOAT                 = VK_FORMAT_D32_SFLOAT,
    VULKAN_FORMAT_S8_UINT                    = VK_FORMAT_S8_UINT,
    VULKAN_FORMAT_D16_UNORM_S8_UINT          = VK_FORMAT_D16_UNORM_S8_UINT,
    VULKAN_FORMAT_D24_UNORM_S8_UINT          = VK_FORMAT_D24_UNORM_S8_UINT,
    VULKAN_FORMAT_D32_SFLOAT_S8_UINT         = VK_FORMAT_D32_SFLOAT_S8_UINT,
    VULKAN_FORMAT_BC1_RGB_UNORM_BLOCK        = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
    VULKAN_FORMAT_BC1_RGB_SRGB_BLOCK         = VK_FORMAT_BC1_RGB_SRGB_BLOCK,
    VULKAN_FORMAT_BC1_RGBA_UNORM_BLOCK       = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    VULKAN_FORMAT_BC1_RGBA_SRGB_BLOCK        = VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    VULKAN_FORMAT_BC2_UNORM_BLOCK            = VK_FORMAT_BC2_UNORM_BLOCK,
    VULKAN_FORMAT_BC2_SRGB_BLOCK             = VK_FORMAT_BC2_SRGB_BLOCK,
    VULKAN_FORMAT_BC3_UNORM_BLOCK            = VK_FORMAT_BC3_UNORM_BLOCK,
    VULKAN_FORMAT_BC3_SRGB_BLOCK             = VK_FORMAT_BC3_SRGB_BLOCK,
    VULKAN_FORMAT_BC4_UNORM_BLOCK            = VK_FORMAT_BC4_UNORM_BLOCK,
    VULKAN_FORMAT_BC4_SNORM_BLOCK            = VK_FORMAT_BC4_SNORM_BLOCK,
    VULKAN_FORMAT_BC5_UNORM_BLOCK            = VK_FORMAT_BC5_UNORM_BLOCK,
    VULKAN_FORMAT_BC5_SNORM_BLOCK            = VK_FORMAT_BC5_SNORM_BLOCK,
    VULKAN_FORMAT_BC6H_UFLOAT_BLOCK          = VK_FORMAT_BC6H_UFLOAT_BLOCK,
    VULKAN_FORMAT_BC6H_SFLOAT_BLOCK          = VK_FORMAT_BC6H_SFLOAT_BLOCK,
    VULKAN_FORMAT_BC7_UNORM_BLOCK            = VK_FORMAT_BC7_UNORM_BLOCK,
    VULKAN_FORMAT_BC7_SRGB_BLOCK             = VK_FORMAT_BC7_SRGB_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8_UNORM_BLOCK    = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8_SRGB_BLOCK     = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK  = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK   = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK  = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    VULKAN_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK   = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    VULKAN_FORMAT_EAC_R11_UNORM_BLOCK        = VK_FORMAT_EAC_R11_UNORM_BLOCK,
    VULKAN_FORMAT_EAC_R11_SNORM_BLOCK        = VK_FORMAT_EAC_R11_SNORM_BLOCK,
    VULKAN_FORMAT_EAC_R11G11_UNORM_BLOCK     = VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
    VULKAN_FORMAT_EAC_R11G11_SNORM_BLOCK     = VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
    VULKAN_FORMAT_ASTC_4x4_UNORM_BLOCK       = VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_4x4_SRGB_BLOCK        = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_5x4_UNORM_BLOCK       = VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_5x4_SRGB_BLOCK        = VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_5x5_UNORM_BLOCK       = VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_5x5_SRGB_BLOCK        = VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_6x5_UNORM_BLOCK       = VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_6x5_SRGB_BLOCK        = VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_6x6_UNORM_BLOCK       = VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_6x6_SRGB_BLOCK        = VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_8x5_UNORM_BLOCK       = VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_8x5_SRGB_BLOCK        = VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_8x6_UNORM_BLOCK       = VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_8x6_SRGB_BLOCK        = VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_8x8_UNORM_BLOCK       = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_8x8_SRGB_BLOCK        = VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_10x5_UNORM_BLOCK      = VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_10x5_SRGB_BLOCK       = VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_10x6_UNORM_BLOCK      = VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_10x6_SRGB_BLOCK       = VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_10x8_UNORM_BLOCK      = VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_10x8_SRGB_BLOCK       = VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_10x10_UNORM_BLOCK     = VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_10x10_SRGB_BLOCK      = VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_12x10_UNORM_BLOCK     = VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_12x10_SRGB_BLOCK      = VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
    VULKAN_FORMAT_ASTC_12x12_UNORM_BLOCK     = VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
    VULKAN_FORMAT_ASTC_12x12_SRGB_BLOCK      = VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
};

enum VulkanImageLayout
{
    VULKAN_IMAGE_LAYOUT_UNDEFINED            = VK_IMAGE_LAYOUT_UNDEFINED,
    VULKAN_IMAGE_LAYOUT_GENERAL              = VK_IMAGE_LAYOUT_GENERAL,
    VULKAN_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
};

enum VulkanImageUsage
{
    VULKAN_IMAGE_USAGE_TRANSFER_SRC                     = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    VULKAN_IMAGE_USAGE_TRANSFER_DST                     = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VULKAN_IMAGE_USAGE_SAMPLED                          = VK_IMAGE_USAGE_SAMPLED_BIT,
    VULKAN_IMAGE_USAGE_STORAGE                          = VK_IMAGE_USAGE_STORAGE_BIT,
    VULKAN_IMAGE_USAGE_COLOR_ATTACHMENT                 = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    VULKAN_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    VULKAN_IMAGE_USAGE_TRANSIENT_ATTACHMENT             = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
    VULKAN_IMAGE_USAGE_INPUT_ATTACHMENT                 = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
    VULKAN_IMAGE_USAGE_TRANSFER_SRC_DST                 = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
    VULKAN_IMAGE_USAGE_STORAGE_TRANSFER_SRC_DST         = VULKAN_IMAGE_USAGE_STORAGE | VULKAN_IMAGE_USAGE_TRANSFER_SRC | VULKAN_IMAGE_USAGE_TRANSFER_DST,
    VULKAN_IMAGE_USAGE_SAMPLED_STORAGE_TRANSFER_SRC_DST = VK_IMAGE_USAGE_SAMPLED_BIT | VULKAN_IMAGE_USAGE_STORAGE | VULKAN_IMAGE_USAGE_TRANSFER_SRC | VULKAN_IMAGE_USAGE_TRANSFER_DST
};

enum VulkanImageTiling
{
    VULKAN_IMAGE_TILING_OPTIMAL = VK_IMAGE_TILING_OPTIMAL,
    VULKAN_IMAGE_TILING_LINEAR  = VK_IMAGE_TILING_LINEAR
};

enum VulkanImageCreateFlag
{
    VULKAN_IMAGE_CREATE_FLAG_NONE                           = 0,
    VULKAN_IMAGE_CREATE_FLAG_MUTABLE_FORMAT                 = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
    VULKAN_IMAGE_CREATE_FLAG_CUBE_COMPATIBLE                = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
    VULKAN_IMAGE_CREATE_FLAG_CUBE_COMPATIBLE_MUTABLE_FORMAT = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
};

enum VulkanImageViewType
{
    VULKAN_IMAGE_VIEW_TYPE_1D         = VK_IMAGE_VIEW_TYPE_1D,
    VULKAN_IMAGE_VIEW_TYPE_2D         = VK_IMAGE_VIEW_TYPE_2D,
    VULKAN_IMAGE_VIEW_TYPE_3D         = VK_IMAGE_VIEW_TYPE_3D,
    VULKAN_IMAGE_VIEW_TYPE_CUBE       = VK_IMAGE_VIEW_TYPE_CUBE,
    VULKAN_IMAGE_VIEW_TYPE_1D_ARRAY   = VK_IMAGE_VIEW_TYPE_1D_ARRAY,
    VULKAN_IMAGE_VIEW_TYPE_2D_ARRAY   = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
    VULKAN_IMAGE_VIEW_TYPE_CUBE_ARRAY = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
};

#endif // _vulkan_wrapper_types_hpp_
