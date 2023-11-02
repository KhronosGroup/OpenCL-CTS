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

#include <vulkan_interop_common.hpp>
#include <string>
#include "harness/errorHelpers.h"
#include <algorithm>

#define MAX_2D_IMAGES 5
#define MAX_2D_IMAGE_WIDTH 1024
#define MAX_2D_IMAGE_HEIGHT 1024
#define MAX_2D_IMAGE_ELEMENT_SIZE 16
#define MAX_2D_IMAGE_MIP_LEVELS 11
#define MAX_2D_IMAGE_DESCRIPTORS MAX_2D_IMAGES *MAX_2D_IMAGE_MIP_LEVELS
#define NUM_THREADS_PER_GROUP_X 32
#define NUM_THREADS_PER_GROUP_Y 32
#define NUM_BLOCKS(size, blockSize)                                            \
    (ROUND_UP((size), (blockSize)) / (blockSize))

#define ASSERT(x)                                                              \
    if (!(x))                                                                  \
    {                                                                          \
        fprintf(stderr, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__,    \
                __LINE__);                                                     \
        exit(1);                                                               \
    }

#define ASSERT_LEQ(x, y)                                                       \
    if (x > y)                                                                 \
    {                                                                          \
        ASSERT(0);                                                             \
    }

namespace {
struct Params
{
    uint32_t numImage2DDescriptors;
};
}
static cl_uchar uuid[CL_UUID_SIZE_KHR];
static cl_device_id deviceId = NULL;
size_t max_width = MAX_2D_IMAGE_WIDTH;
size_t max_height = MAX_2D_IMAGE_HEIGHT;

const char *kernel_text_numImage_1 = " \
__constant sampler_t smpImg = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_NONE|CLK_FILTER_NEAREST;\n\
__kernel void image2DKernel(read_only image2d_t InputImage, write_only image2d_t OutImage, int num2DImages, int baseWidth, int baseHeight, int numMipLevels)\n\
{\n\
    int threadIdxX = get_global_id(0);\n\
    int threadIdxY = get_global_id(1);\n\
    int numThreadsX = get_global_size(0);                                                                                                  \n\
    int numThreadsY = get_global_size(1);\n\
    if (threadIdxX >= baseWidth || threadIdxY >= baseHeight)\n\
    {\n\
        return;\n\
    }\n\
    %s dataA =  read_image%s(InputImage, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataB =  read_image%s(InputImage, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    write_image%s(OutImage, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataA);\n\
    write_image%s(OutImage, (int2)( threadIdxX, threadIdxY), dataB);\n\
\n\
}";

const char *kernel_text_numImage_2 = " \
__constant sampler_t smpImg = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_NONE|CLK_FILTER_NEAREST;\n\
__kernel void image2DKernel(read_only image2d_t InputImage_1, write_only image2d_t OutImage_1, read_only image2d_t InputImage_2,write_only image2d_t OutImage_2,int num2DImages, int baseWidth, int baseHeight, int numMipLevels)    \n\
{\n\
    int threadIdxX = get_global_id(0);\n\
    int threadIdxY = get_global_id(1);\n\
    int numThreadsX = get_global_size(0);\n\
    int numThreadsY = get_global_size(1);\n\
    if (threadIdxX >= baseWidth || threadIdxY >= baseHeight) \n\
    {\n\
        return;\n\
    }\n\
    %s dataA =  read_image%s(InputImage_1, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataB =  read_image%s(InputImage_1, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    %s dataC =  read_image%s(InputImage_2, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataD =  read_image%s(InputImage_2, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    write_image%s(OutImage_1, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataA);\n\
    write_image%s(OutImage_1, (int2)(threadIdxX, threadIdxY), dataB);\n\
    write_image%s(OutImage_2, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataC);\n\
    write_image%s(OutImage_2, (int2)(threadIdxX, threadIdxY), dataD);\n\
\n\
}";

const char *kernel_text_numImage_4 = " \
__constant sampler_t smpImg = CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_NONE|CLK_FILTER_NEAREST;\n\
__kernel void image2DKernel(read_only image2d_t InputImage_1, write_only image2d_t OutImage_1, read_only image2d_t InputImage_2, write_only image2d_t OutImage_2, read_only image2d_t InputImage_3, write_only image2d_t OutImage_3, read_only image2d_t InputImage_4, write_only image2d_t OutImage_4, int num2DImages, int baseWidth, int baseHeight, int numMipLevels)    \n\
{\n\
    int threadIdxX = get_global_id(0);\n\
    int threadIdxY = get_global_id(1);\n\
    int numThreadsX = get_global_size(0);\n\
    int numThreadsY = get_global_size(1);\n\
    if (threadIdxX >= baseWidth || threadIdxY >= baseHeight) \n\
    {\n\
        return;\n\
    }\n\
    %s dataA =  read_image%s(InputImage_1, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataB =  read_image%s(InputImage_1, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    %s dataC =  read_image%s(InputImage_2, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataD =  read_image%s(InputImage_2, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    %s dataE =  read_image%s(InputImage_3, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataF =  read_image%s(InputImage_3, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    %s dataG =  read_image%s(InputImage_4, smpImg, (int2)(threadIdxX, threadIdxY)); \n\
    %s dataH =  read_image%s(InputImage_4, smpImg, (int2)(threadIdxX, baseHeight-threadIdxY-1)); \n\
    write_image%s(OutImage_1, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataA);\n\
    write_image%s(OutImage_1, (int2)(threadIdxX, threadIdxY), dataB);\n\
    write_image%s(OutImage_2, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataC);\n\
    write_image%s(OutImage_2, (int2)(threadIdxX, threadIdxY), dataD);\n\
    write_image%s(OutImage_3, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataE);\n\
    write_image%s(OutImage_3, (int2)(threadIdxX, threadIdxY), dataF);\n\
    write_image%s(OutImage_4, (int2)(threadIdxX, baseHeight-threadIdxY-1), dataG);\n\
    write_image%s(OutImage_4, (int2)(threadIdxX, threadIdxY), dataH);\n\
\n\
}";

const uint32_t num2DImagesList[] = { 1, 2, 4 };
const uint32_t widthList[] = { 4, 64, 183, 1024 };
const uint32_t heightList[] = { 4, 64, 365 };

const cl_kernel getKernelType(VulkanFormat format, cl_kernel kernel_float,
                              cl_kernel kernel_signed,
                              cl_kernel kernel_unsigned)
{
    cl_kernel kernel;
    switch (format)
    {
        case VULKAN_FORMAT_R32G32B32A32_SFLOAT: kernel = kernel_float; break;

        case VULKAN_FORMAT_R32G32B32A32_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R32G32B32A32_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R16G16B16A16_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R16G16B16A16_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R8G8B8A8_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R8G8B8A8_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R32G32_SFLOAT: kernel = kernel_float; break;

        case VULKAN_FORMAT_R32G32_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R32G32_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R16G16_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R16G16_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R8G8_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R8G8_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R32_SFLOAT: kernel = kernel_float; break;

        case VULKAN_FORMAT_R32_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R32_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R16_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R16_SINT: kernel = kernel_signed; break;

        case VULKAN_FORMAT_R8_UINT: kernel = kernel_unsigned; break;

        case VULKAN_FORMAT_R8_SINT: kernel = kernel_signed; break;

        default:
            log_error(" Unsupported format");
            ASSERT(0);
            break;
    }
    return kernel;
}

int run_test_with_two_queue(cl_context &context, cl_command_queue &cmd_queue1,
                            cl_command_queue &cmd_queue2,
                            cl_kernel *kernel_unsigned,
                            cl_kernel *kernel_signed, cl_kernel *kernel_float,
                            VulkanDevice &vkDevice)
{
    cl_int err = CL_SUCCESS;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { 1, 1, 1 };

    cl_kernel updateKernelCQ1, updateKernelCQ2;
    std::vector<VulkanFormat> vkFormatList = getSupportedVulkanFormatList();
    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    char magicValue = 0;

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);

    uint64_t maxImage2DSize =
        max_width * max_height * MAX_2D_IMAGE_ELEMENT_SIZE * 2;
    VulkanBuffer vkSrcBuffer(vkDevice, maxImage2DSize);
    VulkanDeviceMemory vkSrcBufferDeviceMemory(
        vkDevice, vkSrcBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkSrcBufferDeviceMemory.bindBuffer(vkSrcBuffer);

    char *srcBufferPtr, *dstBufferPtr;
    srcBufferPtr = (char *)malloc(maxImage2DSize);
    dstBufferPtr = (char *)malloc(maxImage2DSize);

    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList;
    vkDescriptorSetLayoutBindingList.addBinding(
        0, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    vkDescriptorSetLayoutBindingList.addBinding(
        1, VULKAN_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_2D_IMAGE_DESCRIPTORS);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCopyCommandBuffer(vkDevice, vkCommandPool);
    VulkanCommandBuffer vkShaderCommandBuffer(vkDevice, vkCommandPool);
    VulkanQueue &vkQueue = vkDevice.getQueue();

    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);

    std::vector<VulkanDeviceMemory *> vkImage2DListDeviceMemory1;
    std::vector<VulkanDeviceMemory *> vkImage2DListDeviceMemory2;
    std::vector<clExternalMemoryImage *> externalMemory1;
    std::vector<clExternalMemoryImage *> externalMemory2;
    std::vector<char> vkImage2DShader;

    for (size_t fIdx = 0; fIdx < vkFormatList.size(); fIdx++)
    {
        VulkanFormat vkFormat = vkFormatList[fIdx];
        log_info("Format: %d\n", vkFormat);
        uint32_t elementSize = getVulkanFormatElementSize(vkFormat);
        ASSERT_LEQ(elementSize, (uint32_t)MAX_2D_IMAGE_ELEMENT_SIZE);
        log_info("elementSize= %d\n", elementSize);

        std::string fileName = "image2D_"
            + std::string(getVulkanFormatGLSLFormat(vkFormat)) + ".spv";
        log_info("Load %s file", fileName.c_str());
        vkImage2DShader = readFile(fileName);
        VulkanShaderModule vkImage2DShaderModule(vkDevice, vkImage2DShader);

        VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                                vkImage2DShaderModule);

        for (size_t wIdx = 0; wIdx < ARRAY_SIZE(widthList); wIdx++)
        {
            uint32_t width = widthList[wIdx];
            log_info("Width: %d\n", width);
            if (width > max_width) continue;
            region[0] = width;
            for (size_t hIdx = 0; hIdx < ARRAY_SIZE(heightList); hIdx++)
            {
                uint32_t height = heightList[hIdx];
                log_info("Height: %d", height);
                if (height > max_height) continue;
                region[1] = height;

                uint32_t numMipLevels = 1;
                log_info("Number of mipmap levels: %d\n", numMipLevels);

                magicValue++;
                char *vkSrcBufferDeviceMemoryPtr =
                    (char *)vkSrcBufferDeviceMemory.map();
                uint64_t srcBufSize = 0;
                memset(vkSrcBufferDeviceMemoryPtr, 0, maxImage2DSize);
                memset(srcBufferPtr, 0, maxImage2DSize);
                uint32_t mipLevel = 0;
                for (uint32_t row = 0;
                     row < std::max(height >> mipLevel, uint32_t(1)); row++)
                {
                    for (uint32_t col = 0;
                         col < std::max(width >> mipLevel, uint32_t(1)); col++)
                    {
                        for (uint32_t elementByte = 0;
                             elementByte < elementSize; elementByte++)
                        {
                            vkSrcBufferDeviceMemoryPtr[srcBufSize] =
                                (char)(magicValue + mipLevel + row + col);
                            srcBufferPtr[srcBufSize] =
                                (char)(magicValue + mipLevel + row + col);
                            srcBufSize++;
                        }
                    }
                }
                srcBufSize = ROUND_UP(
                    srcBufSize,
                    std::max(
                        elementSize,
                        (uint32_t)VULKAN_MIN_BUFFER_OFFSET_COPY_ALIGNMENT));
                vkSrcBufferDeviceMemory.unmap();

                for (size_t niIdx = 0; niIdx < ARRAY_SIZE(num2DImagesList);
                     niIdx++)
                {
                    uint32_t num2DImages = num2DImagesList[niIdx] + 1;
                    // added one image for cross-cq case for updateKernelCQ2
                    log_info("Number of images: %d\n", num2DImages);
                    ASSERT_LEQ(num2DImages, (uint32_t)MAX_2D_IMAGES);
                    uint32_t num_2D_image;
                    if (useSingleImageKernel)
                    {
                        num_2D_image = 1;
                    }
                    else
                    {
                        num_2D_image = num2DImages;
                    }
                    Params *params = (Params *)vkParamsDeviceMemory.map();
                    params->numImage2DDescriptors = num_2D_image * numMipLevels;
                    vkParamsDeviceMemory.unmap();
                    vkDescriptorSet.update(0, vkParamsBuffer);
                    for (size_t emhtIdx = 0;
                         emhtIdx < vkExternalMemoryHandleTypeList.size();
                         emhtIdx++)
                    {
                        VulkanExternalMemoryHandleType
                            vkExternalMemoryHandleType =
                                vkExternalMemoryHandleTypeList[emhtIdx];
                        if ((true == disableNTHandleType)
                            && (VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT
                                == vkExternalMemoryHandleType))
                        {
                            // Skip running for WIN32 NT handle.
                            continue;
                        }
                        log_info("External memory handle type: %d \n",
                                 vkExternalMemoryHandleType);
                        VulkanImageTiling vulkanImageTiling =
                            vkClExternalMemoryHandleTilingAssumption(
                                deviceId,
                                vkExternalMemoryHandleTypeList[emhtIdx], &err);
                        ASSERT_SUCCESS(err,
                                       "Failed to query OpenCL tiling mode");

                        VulkanImage2D vkDummyImage2D(
                            vkDevice, vkFormatList[0], widthList[0],
                            heightList[0], vulkanImageTiling, 1,
                            vkExternalMemoryHandleType);
                        const VulkanMemoryTypeList &memoryTypeList =
                            vkDummyImage2D.getMemoryTypeList();

                        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size();
                             mtIdx++)
                        {
                            const VulkanMemoryType &memoryType =
                                memoryTypeList[mtIdx];
                            log_info("Memory type index: %d\n",
                                     (uint32_t)memoryType);
                            log_info("Memory type property: %d\n",
                                     memoryType.getMemoryTypeProperty());
                            if (!useDeviceLocal)
                            {
                                if (VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL
                                    == memoryType.getMemoryTypeProperty())
                                {
                                    continue;
                                }
                            }

                            size_t totalImageMemSize = 0;
                            uint64_t interImageOffset = 0;
                            {
                                VulkanImage2D vkImage2D(
                                    vkDevice, vkFormat, width, height,
                                    vulkanImageTiling, numMipLevels,
                                    vkExternalMemoryHandleType);
                                ASSERT_LEQ(vkImage2D.getSize(), maxImage2DSize);
                                totalImageMemSize =
                                    ROUND_UP(vkImage2D.getSize(),
                                             vkImage2D.getAlignment());
                            }
                            VulkanImage2DList vkImage2DList(
                                num2DImages, vkDevice, vkFormat, width, height,
                                vulkanImageTiling, numMipLevels,
                                vkExternalMemoryHandleType);
                            for (size_t bIdx = 0; bIdx < num2DImages; bIdx++)
                            {
                                vkImage2DListDeviceMemory1.push_back(
                                    new VulkanDeviceMemory(
                                        vkDevice, vkImage2DList[bIdx],
                                        memoryType,
                                        vkExternalMemoryHandleType));
                                vkImage2DListDeviceMemory1[bIdx]->bindImage(
                                    vkImage2DList[bIdx], 0);
                                externalMemory1.push_back(
                                    new clExternalMemoryImage(
                                        *vkImage2DListDeviceMemory1[bIdx],
                                        vkExternalMemoryHandleType, context,
                                        totalImageMemSize, width, height, 0,
                                        vkImage2DList[bIdx], deviceId));
                            }
                            VulkanImageViewList vkImage2DViewList(
                                vkDevice, vkImage2DList);
                            VulkanImage2DList vkImage2DList2(
                                num2DImages, vkDevice, vkFormat, width, height,
                                vulkanImageTiling, numMipLevels,
                                vkExternalMemoryHandleType);
                            for (size_t bIdx = 0; bIdx < num2DImages; bIdx++)
                            {
                                vkImage2DListDeviceMemory2.push_back(
                                    new VulkanDeviceMemory(
                                        vkDevice, vkImage2DList2[bIdx],
                                        memoryType,
                                        vkExternalMemoryHandleType));
                                vkImage2DListDeviceMemory2[bIdx]->bindImage(
                                    vkImage2DList2[bIdx], 0);
                                externalMemory2.push_back(
                                    new clExternalMemoryImage(
                                        *vkImage2DListDeviceMemory2[bIdx],
                                        vkExternalMemoryHandleType, context,
                                        totalImageMemSize, width, height, 0,
                                        vkImage2DList2[bIdx], deviceId));
                            }

                            cl_mem external_mem_image1[5];
                            cl_mem external_mem_image2[5];
                            for (int i = 0; i < num2DImages; i++)
                            {
                                external_mem_image1[i] =
                                    externalMemory1[i]
                                        ->getExternalMemoryImage();
                                external_mem_image2[i] =
                                    externalMemory2[i]
                                        ->getExternalMemoryImage();
                            }

                            clCl2VkExternalSemaphore->signal(cmd_queue1);
                            if (!useSingleImageKernel)
                            {
                                vkDescriptorSet.updateArray(1,
                                                            vkImage2DViewList);
                                vkCopyCommandBuffer.begin();
                                vkCopyCommandBuffer.pipelineBarrier(
                                    vkImage2DList,
                                    VULKAN_IMAGE_LAYOUT_UNDEFINED,
                                    VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                for (size_t i2DIdx = 0;
                                     i2DIdx < vkImage2DList.size(); i2DIdx++)
                                {
                                    vkCopyCommandBuffer.copyBufferToImage(
                                        vkSrcBuffer, vkImage2DList[i2DIdx],
                                        VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                }
                                vkCopyCommandBuffer.pipelineBarrier(
                                    vkImage2DList,
                                    VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VULKAN_IMAGE_LAYOUT_GENERAL);
                                vkCopyCommandBuffer.end();
                                memset(dstBufferPtr, 0, srcBufSize);
                                vkQueue.submit(vkCopyCommandBuffer);
                                vkShaderCommandBuffer.begin();
                                vkShaderCommandBuffer.bindPipeline(
                                    vkComputePipeline);
                                vkShaderCommandBuffer.bindDescriptorSets(
                                    vkComputePipeline, vkPipelineLayout,
                                    vkDescriptorSet);
                                vkShaderCommandBuffer.dispatch(
                                    NUM_BLOCKS(width, NUM_THREADS_PER_GROUP_X),
                                    NUM_BLOCKS(height,
                                               NUM_THREADS_PER_GROUP_Y / 2),
                                    1);
                                vkShaderCommandBuffer.end();
                            }
                            for (uint32_t iter = 0; iter < innerIterations;
                                 iter++)
                            {
                                if (useSingleImageKernel)
                                {
                                    for (size_t i2DIdx = 0;
                                         i2DIdx < vkImage2DList.size();
                                         i2DIdx++)
                                    {
                                        vkDescriptorSet.update(
                                            1, vkImage2DViewList[i2DIdx]);
                                        vkCopyCommandBuffer.begin();
                                        vkCopyCommandBuffer.pipelineBarrier(
                                            vkImage2DList,
                                            VULKAN_IMAGE_LAYOUT_UNDEFINED,
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                                        vkCopyCommandBuffer.copyBufferToImage(
                                            vkSrcBuffer, vkImage2DList[i2DIdx],
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                        vkCopyCommandBuffer.pipelineBarrier(
                                            vkImage2DList,
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VULKAN_IMAGE_LAYOUT_GENERAL);
                                        vkCopyCommandBuffer.end();
                                        memset(dstBufferPtr, 0, srcBufSize);
                                        vkQueue.submit(vkCopyCommandBuffer);
                                        vkShaderCommandBuffer.begin();
                                        vkShaderCommandBuffer.bindPipeline(
                                            vkComputePipeline);
                                        vkShaderCommandBuffer
                                            .bindDescriptorSets(
                                                vkComputePipeline,
                                                vkPipelineLayout,
                                                vkDescriptorSet);
                                        vkShaderCommandBuffer.dispatch(
                                            NUM_BLOCKS(width,
                                                       NUM_THREADS_PER_GROUP_X),
                                            NUM_BLOCKS(height,
                                                       NUM_THREADS_PER_GROUP_Y
                                                           / 2),
                                            1);
                                        vkShaderCommandBuffer.end();
                                        if (i2DIdx < vkImage2DList.size() - 1)
                                        {
                                            vkQueue.submit(
                                                vkShaderCommandBuffer);
                                        }
                                    }
                                }
                                vkQueue.submit(vkCl2VkSemaphore,
                                               vkShaderCommandBuffer,
                                               vkVk2CLSemaphore);
                                clVk2CLExternalSemaphore->wait(cmd_queue1);
                                switch (num2DImages)
                                {
                                    case 2:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[0],
                                            kernel_signed[0],
                                            kernel_unsigned[0]);
                                        break;
                                    case 3:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[1],
                                            kernel_signed[1],
                                            kernel_unsigned[1]);
                                        break;
                                    case 5:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[2],
                                            kernel_signed[2],
                                            kernel_unsigned[2]);
                                        break;
                                }
                                updateKernelCQ2 = getKernelType(
                                    vkFormat, kernel_float[3], kernel_signed[3],
                                    kernel_unsigned[3]);
                                // similar kernel-type based on vkFormat
                                int j = 0;
                                // Setting arguments of updateKernelCQ2

                                err = clSetKernelArg(updateKernelCQ2, 0,
                                                     sizeof(cl_mem),
                                                     &external_mem_image1[0]);
                                err |= clSetKernelArg(updateKernelCQ2, 1,
                                                      sizeof(cl_mem),
                                                      &external_mem_image2[0]);
                                err |= clSetKernelArg(
                                    updateKernelCQ2, 2, sizeof(cl_mem),
                                    &external_mem_image1[num2DImages - 1]);
                                err |= clSetKernelArg(
                                    updateKernelCQ2, 3, sizeof(cl_mem),
                                    &external_mem_image2[num2DImages - 1]);
                                err |= clSetKernelArg(updateKernelCQ2, 4,
                                                      sizeof(unsigned int),
                                                      &num2DImages);
                                err |= clSetKernelArg(updateKernelCQ2, 5,
                                                      sizeof(unsigned int),
                                                      &width);
                                err |= clSetKernelArg(updateKernelCQ2, 6,
                                                      sizeof(unsigned int),
                                                      &height);
                                err |= clSetKernelArg(updateKernelCQ2, 7,
                                                      sizeof(unsigned int),
                                                      &numMipLevels);
                                for (int i = 0; i < num2DImages - 1; i++, ++j)
                                {
                                    err = clSetKernelArg(
                                        updateKernelCQ1, j, sizeof(cl_mem),
                                        &external_mem_image1[i]);
                                    err |= clSetKernelArg(
                                        updateKernelCQ1, ++j, sizeof(cl_mem),
                                        &external_mem_image2[i]);
                                }
                                err |= clSetKernelArg(updateKernelCQ1, j,
                                                      sizeof(unsigned int),
                                                      &num2DImages);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &width);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &height);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &numMipLevels);

                                if (err != CL_SUCCESS)
                                {
                                    print_error(
                                        err,
                                        "Error: Failed to set arg values \n");
                                    goto CLEANUP;
                                }
                                // clVk2CLExternalSemaphore->wait(cmd_queue1);
                                size_t global_work_size[3] = { width, height,
                                                               1 };
                                cl_event first_launch;
                                err = clEnqueueNDRangeKernel(
                                    cmd_queue1, updateKernelCQ1, 2, NULL,
                                    global_work_size, NULL, 0, NULL,
                                    &first_launch);
                                if (err != CL_SUCCESS)
                                {
                                    goto CLEANUP;
                                }
                                err = clEnqueueNDRangeKernel(
                                    cmd_queue2, updateKernelCQ2, 2, NULL,
                                    global_work_size, NULL, 1, &first_launch,
                                    NULL);
                                if (err != CL_SUCCESS)
                                {
                                    goto CLEANUP;
                                }

                                clFinish(cmd_queue2);
                                clCl2VkExternalSemaphore->signal(cmd_queue2);
                            }

                            unsigned int flags = 0;
                            size_t mipmapLevelOffset = 0;
                            cl_event eventReadImage = NULL;
                            clFinish(cmd_queue2);
                            for (int i = 0; i < num2DImages; i++)
                            {
                                err = clEnqueueReadImage(
                                    cmd_queue1, external_mem_image2[i], CL_TRUE,
                                    origin, region, 0, 0, dstBufferPtr, 0, NULL,
                                    &eventReadImage);

                                if (err != CL_SUCCESS)
                                {
                                    print_error(err,
                                                "clEnqueueReadImage failed with"
                                                "error\n");
                                }

                                if (memcmp(srcBufferPtr, dstBufferPtr,
                                           srcBufSize))
                                {
                                    log_info("Source and destination buffers "
                                             "don't match\n");
                                    if (debug_trace)
                                    {
                                        log_info("Source buffer contents: \n");
                                        for (uint64_t sIdx = 0;
                                             sIdx < srcBufSize; sIdx++)
                                        {
                                            log_info(
                                                "%d ",
                                                (int)vkSrcBufferDeviceMemoryPtr
                                                    [sIdx]);
                                        }
                                        log_info("Destination buffer contents:"
                                                 "\n");
                                        for (uint64_t dIdx = 0;
                                             dIdx < srcBufSize; dIdx++)
                                        {
                                            log_info("%d ",
                                                     (int)dstBufferPtr[dIdx]);
                                        }
                                    }
                                    err = -1;
                                    break;
                                }
                            }
                            for (int i = 0; i < num2DImages; i++)
                            {
                                delete vkImage2DListDeviceMemory1[i];
                                delete vkImage2DListDeviceMemory2[i];
                                delete externalMemory1[i];
                                delete externalMemory2[i];
                            }
                            vkImage2DListDeviceMemory1.erase(
                                vkImage2DListDeviceMemory1.begin(),
                                vkImage2DListDeviceMemory1.begin()
                                    + num2DImages);
                            vkImage2DListDeviceMemory2.erase(
                                vkImage2DListDeviceMemory2.begin(),
                                vkImage2DListDeviceMemory2.begin()
                                    + num2DImages);
                            externalMemory1.erase(externalMemory1.begin(),
                                                  externalMemory1.begin()
                                                      + num2DImages);
                            externalMemory2.erase(externalMemory2.begin(),
                                                  externalMemory2.begin()
                                                      + num2DImages);
                            if (CL_SUCCESS != err)
                            {
                                goto CLEANUP;
                            }
                        }
                    }
                }
            }
        }

        vkImage2DShader.clear();
    }
CLEANUP:
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;

    if (srcBufferPtr) free(srcBufferPtr);
    if (dstBufferPtr) free(dstBufferPtr);
    return err;
}

int run_test_with_one_queue(cl_context &context, cl_command_queue &cmd_queue1,
                            cl_kernel *kernel_unsigned,
                            cl_kernel *kernel_signed, cl_kernel *kernel_float,
                            VulkanDevice &vkDevice)
{
    cl_int err = CL_SUCCESS;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { 1, 1, 1 };
    cl_kernel updateKernelCQ1;
    std::vector<VulkanFormat> vkFormatList = getSupportedVulkanFormatList();
    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    char magicValue = 0;

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);

    uint64_t maxImage2DSize =
        max_width * max_height * MAX_2D_IMAGE_ELEMENT_SIZE * 2;
    VulkanBuffer vkSrcBuffer(vkDevice, maxImage2DSize);
    VulkanDeviceMemory vkSrcBufferDeviceMemory(
        vkDevice, vkSrcBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkSrcBufferDeviceMemory.bindBuffer(vkSrcBuffer);

    char *srcBufferPtr, *dstBufferPtr;
    srcBufferPtr = (char *)malloc(maxImage2DSize);
    dstBufferPtr = (char *)malloc(maxImage2DSize);

    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList;
    vkDescriptorSetLayoutBindingList.addBinding(
        0, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1);
    vkDescriptorSetLayoutBindingList.addBinding(
        1, VULKAN_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_2D_IMAGE_DESCRIPTORS);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCopyCommandBuffer(vkDevice, vkCommandPool);
    VulkanCommandBuffer vkShaderCommandBuffer(vkDevice, vkCommandPool);
    VulkanQueue &vkQueue = vkDevice.getQueue();

    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);

    std::vector<VulkanDeviceMemory *> vkImage2DListDeviceMemory1;
    std::vector<VulkanDeviceMemory *> vkImage2DListDeviceMemory2;
    std::vector<clExternalMemoryImage *> externalMemory1;
    std::vector<clExternalMemoryImage *> externalMemory2;
    std::vector<char> vkImage2DShader;

    for (size_t fIdx = 0; fIdx < vkFormatList.size(); fIdx++)
    {
        VulkanFormat vkFormat = vkFormatList[fIdx];
        log_info("Format: %d\n", vkFormat);
        uint32_t elementSize = getVulkanFormatElementSize(vkFormat);
        ASSERT_LEQ(elementSize, (uint32_t)MAX_2D_IMAGE_ELEMENT_SIZE);
        log_info("elementSize= %d\n", elementSize);

        std::string fileName = "image2D_"
            + std::string(getVulkanFormatGLSLFormat(vkFormat)) + ".spv";
        log_info("Load %s file", fileName.c_str());
        vkImage2DShader = readFile(fileName);
        VulkanShaderModule vkImage2DShaderModule(vkDevice, vkImage2DShader);

        VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                                vkImage2DShaderModule);

        for (size_t wIdx = 0; wIdx < ARRAY_SIZE(widthList); wIdx++)
        {
            uint32_t width = widthList[wIdx];
            log_info("Width: %d\n", width);
            if (width > max_width) continue;
            region[0] = width;
            for (size_t hIdx = 0; hIdx < ARRAY_SIZE(heightList); hIdx++)
            {
                uint32_t height = heightList[hIdx];
                log_info("Height: %d\n", height);
                if (height > max_height) continue;
                region[1] = height;

                uint32_t numMipLevels = 1;
                log_info("Number of mipmap levels: %d\n", numMipLevels);

                magicValue++;
                char *vkSrcBufferDeviceMemoryPtr =
                    (char *)vkSrcBufferDeviceMemory.map();
                uint64_t srcBufSize = 0;
                memset(vkSrcBufferDeviceMemoryPtr, 0, maxImage2DSize);
                memset(srcBufferPtr, 0, maxImage2DSize);
                uint32_t mipLevel = 0;
                for (uint32_t row = 0;
                     row < std::max(height >> mipLevel, uint32_t(1)); row++)
                {
                    for (uint32_t col = 0;
                         col < std::max(width >> mipLevel, uint32_t(1)); col++)
                    {
                        for (uint32_t elementByte = 0;
                             elementByte < elementSize; elementByte++)
                        {
                            vkSrcBufferDeviceMemoryPtr[srcBufSize] =
                                (char)(magicValue + mipLevel + row + col);
                            srcBufferPtr[srcBufSize] =
                                (char)(magicValue + mipLevel + row + col);
                            srcBufSize++;
                        }
                    }
                }
                srcBufSize = ROUND_UP(
                    srcBufSize,
                    std::max(
                        elementSize,
                        (uint32_t)VULKAN_MIN_BUFFER_OFFSET_COPY_ALIGNMENT));
                vkSrcBufferDeviceMemory.unmap();

                for (size_t niIdx = 0; niIdx < ARRAY_SIZE(num2DImagesList);
                     niIdx++)
                {
                    uint32_t num2DImages = num2DImagesList[niIdx];
                    log_info("Number of images: %d\n", num2DImages);
                    ASSERT_LEQ(num2DImages, (uint32_t)MAX_2D_IMAGES);

                    Params *params = (Params *)vkParamsDeviceMemory.map();
                    uint32_t num_2D_image;
                    if (useSingleImageKernel)
                    {
                        num_2D_image = 1;
                    }
                    else
                    {
                        num_2D_image = num2DImages;
                    }
                    params->numImage2DDescriptors = num_2D_image * numMipLevels;
                    vkParamsDeviceMemory.unmap();
                    vkDescriptorSet.update(0, vkParamsBuffer);
                    for (size_t emhtIdx = 0;
                         emhtIdx < vkExternalMemoryHandleTypeList.size();
                         emhtIdx++)
                    {
                        VulkanExternalMemoryHandleType
                            vkExternalMemoryHandleType =
                                vkExternalMemoryHandleTypeList[emhtIdx];
                        log_info("External memory handle type: %d \n",
                                 vkExternalMemoryHandleType);
                        if ((true == disableNTHandleType)
                            && (VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_NT
                                == vkExternalMemoryHandleType))
                        {
                            // Skip running for WIN32 NT handle.
                            continue;
                        }

                        VulkanImageTiling vulkanImageTiling =
                            vkClExternalMemoryHandleTilingAssumption(
                                deviceId,
                                vkExternalMemoryHandleTypeList[emhtIdx], &err);
                        ASSERT_SUCCESS(err,
                                       "Failed to query OpenCL tiling mode");

                        VulkanImage2D vkDummyImage2D(
                            vkDevice, vkFormatList[0], widthList[0],
                            heightList[0], vulkanImageTiling, 1,
                            vkExternalMemoryHandleType);
                        const VulkanMemoryTypeList &memoryTypeList =
                            vkDummyImage2D.getMemoryTypeList();

                        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size();
                             mtIdx++)
                        {
                            const VulkanMemoryType &memoryType =
                                memoryTypeList[mtIdx];
                            log_info("Memory type index: %d\n",
                                     (uint32_t)memoryType);
                            log_info("Memory type property: %d\n",
                                     memoryType.getMemoryTypeProperty());
                            if (!useDeviceLocal)
                            {
                                if (VULKAN_MEMORY_TYPE_PROPERTY_DEVICE_LOCAL
                                    == memoryType.getMemoryTypeProperty())
                                {
                                    continue;
                                }
                            }
                            size_t totalImageMemSize = 0;
                            uint64_t interImageOffset = 0;
                            {
                                VulkanImage2D vkImage2D(
                                    vkDevice, vkFormat, width, height,
                                    vulkanImageTiling, numMipLevels,
                                    vkExternalMemoryHandleType);
                                ASSERT_LEQ(vkImage2D.getSize(), maxImage2DSize);
                                totalImageMemSize =
                                    ROUND_UP(vkImage2D.getSize(),
                                             vkImage2D.getAlignment());
                            }
                            VulkanImage2DList vkImage2DList(
                                num2DImages, vkDevice, vkFormat, width, height,
                                vulkanImageTiling, numMipLevels,
                                vkExternalMemoryHandleType);
                            for (size_t bIdx = 0; bIdx < vkImage2DList.size();
                                 bIdx++)
                            {
                                // Create list of Vulkan device memories and
                                // bind the list of Vulkan images.
                                vkImage2DListDeviceMemory1.push_back(
                                    new VulkanDeviceMemory(
                                        vkDevice, vkImage2DList[bIdx],
                                        memoryType,
                                        vkExternalMemoryHandleType));
                                vkImage2DListDeviceMemory1[bIdx]->bindImage(
                                    vkImage2DList[bIdx], 0);
                                externalMemory1.push_back(
                                    new clExternalMemoryImage(
                                        *vkImage2DListDeviceMemory1[bIdx],
                                        vkExternalMemoryHandleType, context,
                                        totalImageMemSize, width, height, 0,
                                        vkImage2DList[bIdx], deviceId));
                            }
                            VulkanImageViewList vkImage2DViewList(
                                vkDevice, vkImage2DList);

                            VulkanImage2DList vkImage2DList2(
                                num2DImages, vkDevice, vkFormat, width, height,
                                vulkanImageTiling, numMipLevels,
                                vkExternalMemoryHandleType);
                            for (size_t bIdx = 0; bIdx < vkImage2DList2.size();
                                 bIdx++)
                            {
                                vkImage2DListDeviceMemory2.push_back(
                                    new VulkanDeviceMemory(
                                        vkDevice, vkImage2DList2[bIdx],
                                        memoryType,
                                        vkExternalMemoryHandleType));
                                vkImage2DListDeviceMemory2[bIdx]->bindImage(
                                    vkImage2DList2[bIdx], 0);
                                externalMemory2.push_back(
                                    new clExternalMemoryImage(
                                        *vkImage2DListDeviceMemory2[bIdx],
                                        vkExternalMemoryHandleType, context,
                                        totalImageMemSize, width, height, 0,
                                        vkImage2DList2[bIdx], deviceId));
                            }

                            cl_mem external_mem_image1[4];
                            cl_mem external_mem_image2[4];
                            for (int i = 0; i < num2DImages; i++)
                            {
                                external_mem_image1[i] =
                                    externalMemory1[i]
                                        ->getExternalMemoryImage();
                                external_mem_image2[i] =
                                    externalMemory2[i]
                                        ->getExternalMemoryImage();
                            }

                            clCl2VkExternalSemaphore->signal(cmd_queue1);
                            if (!useSingleImageKernel)
                            {
                                vkDescriptorSet.updateArray(1,
                                                            vkImage2DViewList);
                                vkCopyCommandBuffer.begin();
                                vkCopyCommandBuffer.pipelineBarrier(
                                    vkImage2DList,
                                    VULKAN_IMAGE_LAYOUT_UNDEFINED,
                                    VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                for (size_t i2DIdx = 0;
                                     i2DIdx < vkImage2DList.size(); i2DIdx++)
                                {
                                    vkCopyCommandBuffer.copyBufferToImage(
                                        vkSrcBuffer, vkImage2DList[i2DIdx],
                                        VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                }
                                vkCopyCommandBuffer.pipelineBarrier(
                                    vkImage2DList,
                                    VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VULKAN_IMAGE_LAYOUT_GENERAL);
                                vkCopyCommandBuffer.end();
                                memset(dstBufferPtr, 0, srcBufSize);
                                vkQueue.submit(vkCopyCommandBuffer);
                                vkShaderCommandBuffer.begin();
                                vkShaderCommandBuffer.bindPipeline(
                                    vkComputePipeline);
                                vkShaderCommandBuffer.bindDescriptorSets(
                                    vkComputePipeline, vkPipelineLayout,
                                    vkDescriptorSet);
                                vkShaderCommandBuffer.dispatch(
                                    NUM_BLOCKS(width, NUM_THREADS_PER_GROUP_X),
                                    NUM_BLOCKS(height,
                                               NUM_THREADS_PER_GROUP_Y / 2),
                                    1);
                                vkShaderCommandBuffer.end();
                            }
                            for (uint32_t iter = 0; iter < innerIterations;
                                 iter++)
                            {
                                if (useSingleImageKernel)
                                {
                                    for (size_t i2DIdx = 0;
                                         i2DIdx < vkImage2DList.size();
                                         i2DIdx++)
                                    {
                                        vkDescriptorSet.update(
                                            1, vkImage2DViewList[i2DIdx]);
                                        vkCopyCommandBuffer.begin();
                                        vkCopyCommandBuffer.pipelineBarrier(
                                            vkImage2DList,
                                            VULKAN_IMAGE_LAYOUT_UNDEFINED,
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

                                        vkCopyCommandBuffer.copyBufferToImage(
                                            vkSrcBuffer, vkImage2DList[i2DIdx],
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                                        vkCopyCommandBuffer.pipelineBarrier(
                                            vkImage2DList,
                                            VULKAN_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            VULKAN_IMAGE_LAYOUT_GENERAL);
                                        vkCopyCommandBuffer.end();
                                        memset(dstBufferPtr, 0, srcBufSize);
                                        vkQueue.submit(vkCopyCommandBuffer);
                                        vkShaderCommandBuffer.begin();
                                        vkShaderCommandBuffer.bindPipeline(
                                            vkComputePipeline);
                                        vkShaderCommandBuffer
                                            .bindDescriptorSets(
                                                vkComputePipeline,
                                                vkPipelineLayout,
                                                vkDescriptorSet);
                                        vkShaderCommandBuffer.dispatch(
                                            NUM_BLOCKS(width,
                                                       NUM_THREADS_PER_GROUP_X),
                                            NUM_BLOCKS(height,
                                                       NUM_THREADS_PER_GROUP_Y
                                                           / 2),
                                            1);
                                        vkShaderCommandBuffer.end();
                                        if (i2DIdx < vkImage2DList.size() - 1)
                                        {
                                            vkQueue.submit(
                                                vkShaderCommandBuffer);
                                        }
                                    }
                                }
                                vkQueue.submit(vkCl2VkSemaphore,
                                               vkShaderCommandBuffer,
                                               vkVk2CLSemaphore);
                                clVk2CLExternalSemaphore->wait(cmd_queue1);
                                switch (num2DImages)
                                {
                                    case 1:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[0],
                                            kernel_signed[0],
                                            kernel_unsigned[0]);
                                        break;
                                    case 2:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[1],
                                            kernel_signed[1],
                                            kernel_unsigned[1]);
                                        break;
                                    case 4:
                                        updateKernelCQ1 = getKernelType(
                                            vkFormat, kernel_float[2],
                                            kernel_signed[2],
                                            kernel_unsigned[2]);
                                        break;
                                }
                                int j = 0;
                                for (int i = 0; i < num2DImages; i++, ++j)
                                {
                                    err = clSetKernelArg(
                                        updateKernelCQ1, j, sizeof(cl_mem),
                                        &external_mem_image1[i]);
                                    err |= clSetKernelArg(
                                        updateKernelCQ1, ++j, sizeof(cl_mem),
                                        &external_mem_image2[i]);
                                }
                                err |= clSetKernelArg(updateKernelCQ1, j,
                                                      sizeof(unsigned int),
                                                      &num2DImages);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &width);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &height);
                                err |= clSetKernelArg(updateKernelCQ1, ++j,
                                                      sizeof(unsigned int),
                                                      &numMipLevels);

                                if (err != CL_SUCCESS)
                                {
                                    print_error(err,
                                                "Error: Failed to set arg "
                                                "values for kernel-1\n");
                                    goto CLEANUP;
                                }

                                size_t global_work_size[3] = { width, height,
                                                               1 };
                                err = clEnqueueNDRangeKernel(
                                    cmd_queue1, updateKernelCQ1, 2, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);
                                if (err != CL_SUCCESS)
                                {
                                    goto CLEANUP;
                                }
                                clCl2VkExternalSemaphore->signal(cmd_queue1);
                            }

                            unsigned int flags = 0;
                            size_t mipmapLevelOffset = 0;
                            cl_event eventReadImage = NULL;
                            for (int i = 0; i < num2DImages; i++)
                            {
                                err = clEnqueueReadImage(
                                    cmd_queue1, external_mem_image2[i], CL_TRUE,
                                    origin, region, 0, 0, dstBufferPtr, 0, NULL,
                                    &eventReadImage);

                                if (err != CL_SUCCESS)
                                {
                                    print_error(err,
                                                "clEnqueueReadImage failed with"
                                                "error\n");
                                }

                                if (memcmp(srcBufferPtr, dstBufferPtr,
                                           srcBufSize))
                                {
                                    log_info("Source and destination buffers "
                                             "don't match\n");
                                    if (debug_trace)
                                    {
                                        log_info("Source buffer contents: \n");
                                        for (uint64_t sIdx = 0;
                                             sIdx < srcBufSize; sIdx++)
                                        {
                                            log_info(
                                                "%d",
                                                (int)vkSrcBufferDeviceMemoryPtr
                                                    [sIdx]);
                                        }
                                        log_info(
                                            "Destination buffer contents:");
                                        for (uint64_t dIdx = 0;
                                             dIdx < srcBufSize; dIdx++)
                                        {
                                            log_info("%d",
                                                     (int)dstBufferPtr[dIdx]);
                                        }
                                    }
                                    err = -1;
                                    break;
                                }
                            }
                            for (int i = 0; i < num2DImages; i++)
                            {
                                delete vkImage2DListDeviceMemory1[i];
                                delete vkImage2DListDeviceMemory2[i];
                                delete externalMemory1[i];
                                delete externalMemory2[i];
                            }
                            vkImage2DListDeviceMemory1.erase(
                                vkImage2DListDeviceMemory1.begin(),
                                vkImage2DListDeviceMemory1.begin()
                                    + num2DImages);
                            vkImage2DListDeviceMemory2.erase(
                                vkImage2DListDeviceMemory2.begin(),
                                vkImage2DListDeviceMemory2.begin()
                                    + num2DImages);
                            externalMemory1.erase(externalMemory1.begin(),
                                                  externalMemory1.begin()
                                                      + num2DImages);
                            externalMemory2.erase(externalMemory2.begin(),
                                                  externalMemory2.begin()
                                                      + num2DImages);
                            if (CL_SUCCESS != err)
                            {
                                goto CLEANUP;
                            }
                        }
                    }
                }
            }
        }
        vkImage2DShader.clear();
    }
CLEANUP:
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;

    if (srcBufferPtr) free(srcBufferPtr);
    if (dstBufferPtr) free(dstBufferPtr);
    return err;
}

int test_image_common(cl_device_id device_, cl_context context_,
                      cl_command_queue queue_, int numElements_)
{
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;
    cl_int err = CL_SUCCESS;
    cl_platform_id platform = NULL;
    size_t extensionSize = 0;
    cl_uint num_devices = 0;
    cl_uint device_no = 0;
    cl_device_id *devices;
    char *extensions = NULL;
    const char *program_source_const;
    cl_command_queue cmd_queue1 = NULL;
    cl_command_queue cmd_queue2 = NULL;
    cl_context context = NULL;
    const uint32_t num_kernels = ARRAY_SIZE(num2DImagesList) + 1;
    // One kernel for Cross-CQ case
    const uint32_t num_kernel_types = 3;
    const char *kernel_source[num_kernels] = { kernel_text_numImage_1,
                                               kernel_text_numImage_2,
                                               kernel_text_numImage_4 };
    char source_1[4096];
    char source_2[4096];
    char source_3[4096];
    size_t program_source_length;
    cl_program program[num_kernel_types];
    cl_kernel kernel_float[num_kernels] = { NULL, NULL, NULL, NULL };
    cl_kernel kernel_signed[num_kernels] = { NULL, NULL, NULL, NULL };
    cl_kernel kernel_unsigned[num_kernels] = { NULL, NULL, NULL, NULL };
    cl_mem external_mem_image1;
    cl_mem external_mem_image2;

    VulkanDevice vkDevice;

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    // get the platform ID
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to get platform\n");
        goto CLEANUP;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (CL_SUCCESS != err)
    {
        print_error(err, "clGetDeviceIDs failed in returning no. of devices\n");
        goto CLEANUP;
    }
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    if (NULL == devices)
    {
        err = CL_OUT_OF_HOST_MEMORY;
        print_error(err, "Unable to allocate memory for devices\n");
        goto CLEANUP;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices,
                         NULL);
    if (CL_SUCCESS != err)
    {
        print_error(err, "Failed to get deviceID.\n");
        goto CLEANUP;
    }
    contextProperties[1] = (cl_context_properties)platform;
    log_info("Assigned contextproperties for platform\n");
    for (device_no = 0; device_no < num_devices; device_no++)
    {
        err = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS, 0, NULL,
                              &extensionSize);
        if (CL_SUCCESS != err)
        {
            print_error(
                err,
                "Error in clGetDeviceInfo for getting device_extension size\n");
            goto CLEANUP;
        }
        extensions = (char *)malloc(extensionSize);
        if (NULL == extensions)
        {
            err = CL_OUT_OF_HOST_MEMORY;
            print_error(err, "Unable to allocate memory for extensions\n");
            goto CLEANUP;
        }
        err = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS,
                              extensionSize, extensions, NULL);
        if (CL_SUCCESS != err)
        {
            print_error(
                err, "Error in clGetDeviceInfo for getting device_extension\n");
            goto CLEANUP;
        }
        err = clGetDeviceInfo(devices[device_no], CL_DEVICE_UUID_KHR,
                              CL_UUID_SIZE_KHR, uuid, &extensionSize);
        if (CL_SUCCESS != err)
        {
            print_error(err, "clGetDeviceInfo failed with error");
            goto CLEANUP;
        }
        err =
            memcmp(uuid, vkDevice.getPhysicalDevice().getUUID(), VK_UUID_SIZE);
        if (err == 0)
        {
            break;
        }
    }
    if (device_no >= num_devices)
    {
        err = EXIT_FAILURE;
        print_error(err,
                    "OpenCL error:"
                    "No Vulkan-OpenCL Interop capable GPU found.\n");
        goto CLEANUP;
    }
    deviceId = devices[device_no];
    err = setMaxImageDimensions(deviceId, max_width, max_height);
    if (CL_SUCCESS != err)
    {
        print_error(err, "error setting max image dimensions");
        goto CLEANUP;
    }
    log_info("Set max_width to %lu and max_height to %lu\n", max_width,
             max_height);
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &err);
    if (CL_SUCCESS != err)
    {
        print_error(err, "error creating context");
        goto CLEANUP;
    }
    log_info("Successfully created context !!!\n");

    cmd_queue1 = clCreateCommandQueue(context, devices[device_no], 0, &err);
    if (CL_SUCCESS != err)
    {
        err = CL_INVALID_COMMAND_QUEUE;
        print_error(err, "Error: Failed to create command queue!\n");
        goto CLEANUP;
    }
    log_info("clCreateCommandQueue successfull \n");

    cmd_queue2 = clCreateCommandQueue(context, devices[device_no], 0, &err);
    if (CL_SUCCESS != err)
    {
        err = CL_INVALID_COMMAND_QUEUE;
        print_error(err, "Error: Failed to create command queue!\n");
        goto CLEANUP;
    }
    log_info("clCreateCommandQueue2 successful \n");

    for (int i = 0; i < num_kernels; i++)
    {
        switch (i)
        {
            case 0:
                sprintf(source_1, kernel_source[i], "float4", "f", "float4",
                        "f", "f", "f");
                sprintf(source_2, kernel_source[i], "int4", "i", "int4", "i",
                        "i", "i");
                sprintf(source_3, kernel_source[i], "uint4", "ui", "uint4",
                        "ui", "ui", "ui");
                break;
            case 1:
                sprintf(source_1, kernel_source[i], "float4", "f", "float4",
                        "f", "float4", "f", "float4", "f", "f", "f", "f", "f");
                sprintf(source_2, kernel_source[i], "int4", "i", "int4", "i",
                        "int4", "i", "int4", "i", "i", "i", "i", "i");
                sprintf(source_3, kernel_source[i], "uint4", "ui", "uint4",
                        "ui", "uint4", "ui", "uint4", "ui", "ui", "ui", "ui",
                        "ui");
                break;
            case 2:
                sprintf(source_1, kernel_source[i], "float4", "f", "float4",
                        "f", "float4", "f", "float4", "f", "float4", "f",
                        "float4", "f", "float4", "f", "float4", "f", "f", "f",
                        "f", "f", "f", "f", "f", "f");
                sprintf(source_2, kernel_source[i], "int4", "i", "int4", "i",
                        "int4", "i", "int4", "i", "int4", "i", "int4", "i",
                        "int4", "i", "int4", "i", "i", "i", "i", "i", "i", "i",
                        "i", "i");
                sprintf(source_3, kernel_source[i], "uint4", "ui", "uint4",
                        "ui", "uint4", "ui", "uint4", "ui", "uint4", "ui",
                        "uint4", "ui", "uint4", "ui", "uint4", "ui", "ui", "ui",
                        "ui", "ui", "ui", "ui", "ui", "ui");
                break;
            case 3:
                // Addtional case for creating updateKernelCQ2 which takes two
                // images
                sprintf(source_1, kernel_source[1], "float4", "f", "float4",
                        "f", "float4", "f", "float4", "f", "f", "f", "f", "f");
                sprintf(source_2, kernel_source[1], "int4", "i", "int4", "i",
                        "int4", "i", "int4", "i", "i", "i", "i", "i");
                sprintf(source_3, kernel_source[1], "uint4", "ui", "uint4",
                        "ui", "uint4", "ui", "uint4", "ui", "ui", "ui", "ui",
                        "ui");
                break;
        }
        const char *sourceTexts[num_kernel_types] = { source_1, source_2,
                                                      source_3 };
        for (int k = 0; k < num_kernel_types; k++)
        {
            program_source_length = strlen(sourceTexts[k]);
            program[k] = clCreateProgramWithSource(
                context, 1, &sourceTexts[k], &program_source_length, &err);
            err |= clBuildProgram(program[k], 0, NULL, NULL, NULL, NULL);
        }

        if (err != CL_SUCCESS)
        {
            print_error(err, "Error: Failed to build program");
            goto CLEANUP;
        }
        // create the kernel
        kernel_float[i] = clCreateKernel(program[0], "image2DKernel", &err);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clCreateKernel failed");
            goto CLEANUP;
        }
        kernel_signed[i] = clCreateKernel(program[1], "image2DKernel", &err);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clCreateKernel failed");
            goto CLEANUP;
        }
        kernel_unsigned[i] = clCreateKernel(program[2], "image2DKernel", &err);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clCreateKernel failed ");
            goto CLEANUP;
        }
    }
    if (numCQ == 2)
    {
        err = run_test_with_two_queue(context, cmd_queue1, cmd_queue2,
                                      kernel_unsigned, kernel_signed,
                                      kernel_float, vkDevice);
    }
    else
    {
        err = run_test_with_one_queue(context, cmd_queue1, kernel_unsigned,
                                      kernel_signed, kernel_float, vkDevice);
    }
CLEANUP:
    for (int i = 0; i < num_kernels; i++)
    {
        if (kernel_float[i])
        {
            clReleaseKernel(kernel_float[i]);
        }
        if (kernel_unsigned[i])
        {
            clReleaseKernel(kernel_unsigned[i]);
        }
        if (kernel_signed[i])
        {
            clReleaseKernel(kernel_signed[i]);
        }
    }
    for (int i = 0; i < num_kernel_types; i++)
    {
        if (program[i])
        {
            clReleaseProgram(program[i]);
        }
    }
    if (cmd_queue1) clReleaseCommandQueue(cmd_queue1);
    if (cmd_queue2) clReleaseCommandQueue(cmd_queue2);
    if (context) clReleaseContext(context);

    if (extensions) free(extensions);
    if (devices) free(devices);

    return err;
}
