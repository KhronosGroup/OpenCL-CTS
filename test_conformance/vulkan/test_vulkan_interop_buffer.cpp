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
#include <vulkan_wrapper.hpp>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <string.h>
#include "harness/errorHelpers.h"

#define MAX_BUFFERS 5
#define MAX_IMPORTS 5
#define BUFFERSIZE 3000
static cl_uchar uuid[CL_UUID_SIZE_KHR];
static cl_device_id deviceId = NULL;

namespace {
struct Params
{
    uint32_t numBuffers;
    uint32_t bufferSize;
    uint32_t interBufferOffset;
};
}

const char *kernel_text_numbuffer_1 = " \
__kernel void clUpdateBuffer(int bufferSize, __global unsigned char *a) {  \n\
    int gid = get_global_id(0); \n\
    if (gid < bufferSize) { \n\
        a[gid]++; \n\
    } \n\
}";

const char *kernel_text_numbuffer_2 = " \
__kernel void clUpdateBuffer(int bufferSize, __global unsigned char *a, __global unsigned char *b) {  \n\
    int gid = get_global_id(0); \n\
    if (gid < bufferSize) { \n\
        a[gid]++; \n\
        b[gid]++;\n\
    } \n\
}";

const char *kernel_text_numbuffer_4 = " \
__kernel void clUpdateBuffer(int bufferSize, __global unsigned char *a, __global unsigned char *b, __global unsigned char *c, __global unsigned char *d) {  \n\
    int gid = get_global_id(0); \n\
    if (gid < bufferSize) { \n\
        a[gid]++;\n\
        b[gid]++; \n\
        c[gid]++; \n\
        d[gid]++; \n\
    } \n\
}";


const char *kernel_text_verify = " \
__kernel void checkKernel(__global unsigned char *ptr, int size, int expVal, __global unsigned char *err)     \n\
{                                                                                         \n\
    int idx = get_global_id(0);                                                           \n\
    if ((idx < size) && (*err == 0)) { \n\
    if (ptr[idx] != expVal){           \n\
            *err = 1;                  \n\
           }                           \n\
        }                              \n\
}";

int run_test_with_two_queue(cl_context &context, cl_command_queue &cmd_queue1,
                            cl_command_queue &cmd_queue2, cl_kernel *kernel,
                            cl_kernel &verify_kernel, VulkanDevice &vkDevice,
                            uint32_t numBuffers, uint32_t bufferSize)
{
    int err = CL_SUCCESS;
    size_t global_work_size[1];
    uint8_t *error_2;
    cl_mem error_1;
    cl_kernel update_buffer_kernel;
    cl_kernel kernel_cq;
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;
    const char *program_source_const = kernel_text_numbuffer_2;
    size_t program_source_length = strlen(program_source_const);
    cl_program program = clCreateProgramWithSource(
        context, 1, &program_source_const, &program_source_length, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        print_error(err, "Error: Failed to build program \n");
        return err;
    }
    // create the kernel
    kernel_cq = clCreateKernel(program, "clUpdateBuffer", &err);
    if (err != CL_SUCCESS)
    {
        print_error(err, "clCreateKernel failed \n");
        return err;
    }

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    VulkanQueue &vkQueue = vkDevice.getQueue();

    std::vector<char> vkBufferShader = readFile("buffer.spv");

    VulkanShaderModule vkBufferShaderModule(vkDevice, vkBufferShader);
    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList(
        MAX_BUFFERS + 1, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);
    VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                            vkBufferShaderModule);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);

    const uint32_t maxIter = innerIterations;
    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCommandBuffer(vkDevice, vkCommandPool);

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);
    std::vector<VulkanDeviceMemory *> vkBufferListDeviceMemory;
    std::vector<clExternalMemory *> externalMemory;
    for (size_t emhtIdx = 0; emhtIdx < vkExternalMemoryHandleTypeList.size();
         emhtIdx++)
    {
        VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
            vkExternalMemoryHandleTypeList[emhtIdx];
        log_info("External memory handle type: %d\n",
                 vkExternalMemoryHandleType);

        VulkanBuffer vkDummyBuffer(vkDevice, 4 * 1024,
                                   vkExternalMemoryHandleType);
        const VulkanMemoryTypeList &memoryTypeList =
            vkDummyBuffer.getMemoryTypeList();

        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
        {
            const VulkanMemoryType &memoryType = memoryTypeList[mtIdx];

            log_info("Memory type index: %d\n", (uint32_t)memoryType);
            log_info("Memory type property: %d\n",
                     memoryType.getMemoryTypeProperty());

            VulkanBufferList vkBufferList(numBuffers, vkDevice, bufferSize,
                                          vkExternalMemoryHandleType);

            for (size_t bIdx = 0; bIdx < numBuffers; bIdx++)
            {
                vkBufferListDeviceMemory.push_back(
                    new VulkanDeviceMemory(vkDevice, bufferSize, memoryType,
                                           vkExternalMemoryHandleType));
                externalMemory.push_back(new clExternalMemory(
                    vkBufferListDeviceMemory[bIdx], vkExternalMemoryHandleType,
                    0, bufferSize, context, deviceId));
            }
            cl_mem buffers[MAX_BUFFERS];
            clFinish(cmd_queue1);
            Params *params = (Params *)vkParamsDeviceMemory.map();
            params->numBuffers = numBuffers;
            params->bufferSize = bufferSize;
            params->interBufferOffset = 0;
            vkParamsDeviceMemory.unmap();
            vkDescriptorSet.update(0, vkParamsBuffer);
            for (size_t bIdx = 0; bIdx < vkBufferList.size(); bIdx++)
            {
                size_t buffer_size = vkBufferList[bIdx].getSize();
                vkBufferListDeviceMemory[bIdx]->bindBuffer(vkBufferList[bIdx],
                                                           0);
                buffers[bIdx] = externalMemory[bIdx]->getExternalMemoryBuffer();
                vkDescriptorSet.update((uint32_t)bIdx + 1, vkBufferList[bIdx]);
            }
            vkCommandBuffer.begin();
            vkCommandBuffer.bindPipeline(vkComputePipeline);
            vkCommandBuffer.bindDescriptorSets(
                vkComputePipeline, vkPipelineLayout, vkDescriptorSet);
            vkCommandBuffer.dispatch(512, 1, 1);
            vkCommandBuffer.end();

            if (vkBufferList.size() == 2)
            {
                update_buffer_kernel = kernel[0];
            }
            else if (vkBufferList.size() == 3)
            {
                update_buffer_kernel = kernel[1];
            }
            else if (vkBufferList.size() == 5)
            {
                update_buffer_kernel = kernel[2];
            }
            // global work size should be less than or equal to
            // bufferSizeList[i]
            global_work_size[0] = bufferSize;
            for (uint32_t iter = 0; iter < maxIter; iter++)
            {

                if (iter == 0)
                {
                    vkQueue.submit(vkCommandBuffer, vkVk2CLSemaphore);
                }
                else
                {
                    vkQueue.submit(vkCl2VkSemaphore, vkCommandBuffer,
                                   vkVk2CLSemaphore);
                }
                clVk2CLExternalSemaphore->wait(cmd_queue1);

                err = clSetKernelArg(update_buffer_kernel, 0, sizeof(uint32_t),
                                     (void *)&bufferSize);
                err |= clSetKernelArg(kernel_cq, 0, sizeof(uint32_t),
                                      (void *)&bufferSize);
                err |= clSetKernelArg(kernel_cq, 1, sizeof(cl_mem),
                                      (void *)&(buffers[0]));

                for (int i = 0; i < vkBufferList.size() - 1; i++)
                {
                    err |=
                        clSetKernelArg(update_buffer_kernel, i + 1,
                                       sizeof(cl_mem), (void *)&(buffers[i]));
                }

                err |=
                    clSetKernelArg(kernel_cq, 2, sizeof(cl_mem),
                                   (void *)&(buffers[vkBufferList.size() - 1]));

                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to set arg values for kernel\n");
                    goto CLEANUP;
                }
                cl_event first_launch;

                err = clEnqueueNDRangeKernel(cmd_queue1, update_buffer_kernel,
                                             1, NULL, global_work_size, NULL, 0,
                                             NULL, &first_launch);
                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to launch update_buffer_kernel,"
                                "error\n");
                    goto CLEANUP;
                }

                err = clEnqueueNDRangeKernel(cmd_queue2, kernel_cq, 1, NULL,
                                             global_work_size, NULL, 1,
                                             &first_launch, NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to launch update_buffer_kernel,"
                                "error\n");
                    goto CLEANUP;
                }

                if (iter != (maxIter - 1))
                {
                    clCl2VkExternalSemaphore->signal(cmd_queue2);
                }
            }
            error_2 = (uint8_t *)malloc(sizeof(uint8_t));
            if (NULL == error_2)
            {
                log_error("Not able to allocate memory\n");
                goto CLEANUP;
            }
            clFinish(cmd_queue2);
            error_1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(uint8_t), NULL, &err);
            if (CL_SUCCESS != err)
            {
                print_error(err, "Error: clCreateBuffer \n");
                goto CLEANUP;
            }
            uint8_t val = 0;
            err = clEnqueueWriteBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                       sizeof(uint8_t), &val, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                print_error(err, "Error: Failed read output, error\n");
                goto CLEANUP;
            }

            int calc_max_iter;
            for (int i = 0; i < vkBufferList.size(); i++)
            {
                if (i == 0)
                    calc_max_iter = (maxIter * 3);
                else
                    calc_max_iter = (maxIter * 2);
                err = clSetKernelArg(verify_kernel, 0, sizeof(cl_mem),
                                     (void *)&(buffers[i]));
                err |=
                    clSetKernelArg(verify_kernel, 1, sizeof(int), &bufferSize);
                err |= clSetKernelArg(verify_kernel, 2, sizeof(int),
                                      &calc_max_iter);
                err |= clSetKernelArg(verify_kernel, 3, sizeof(cl_mem),
                                      (void *)&error_1);
                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to set arg values for "
                                "verify_kernel \n");
                    goto CLEANUP;
                }
                err = clEnqueueNDRangeKernel(cmd_queue1, verify_kernel, 1, NULL,
                                             global_work_size, NULL, 0, NULL,
                                             NULL);

                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to launch verify_kernel,"
                                "error \n");
                    goto CLEANUP;
                }
                err = clEnqueueReadBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                          sizeof(uint8_t), error_2, 0, NULL,
                                          NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err, "Error: Failed read output, error \n ");
                    goto CLEANUP;
                }
                if (*error_2 == 1)
                {
                    log_error("&&&& vulkan_opencl_buffer test FAILED\n");
                    goto CLEANUP;
                }
            }
            for (size_t i = 0; i < vkBufferList.size(); i++)
            {
                delete vkBufferListDeviceMemory[i];
                delete externalMemory[i];
            }
            vkBufferListDeviceMemory.erase(vkBufferListDeviceMemory.begin(),
                                           vkBufferListDeviceMemory.begin()
                                               + numBuffers);
            externalMemory.erase(externalMemory.begin(),
                                 externalMemory.begin() + numBuffers);
        }
    }
CLEANUP:
    for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
    {
        if (vkBufferListDeviceMemory[i])
        {
            delete vkBufferListDeviceMemory[i];
        }
        if (externalMemory[i])
        {
            delete externalMemory[i];
        }
    }
    if (program) clReleaseProgram(program);
    if (kernel_cq) clReleaseKernel(kernel_cq);
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;
    if (error_2) free(error_2);
    if (error_1) clReleaseMemObject(error_1);

    return err;
}

int run_test_with_one_queue(cl_context &context, cl_command_queue &cmd_queue1,
                            cl_kernel *kernel, cl_kernel &verify_kernel,
                            VulkanDevice &vkDevice, uint32_t numBuffers,
                            uint32_t bufferSize)
{
    log_info("RUNNING TEST WITH ONE QUEUE...... \n\n");
    size_t global_work_size[1];
    uint8_t *error_2;
    cl_mem error_1;
    cl_kernel update_buffer_kernel;
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;
    int err = CL_SUCCESS;

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    VulkanQueue &vkQueue = vkDevice.getQueue();

    std::vector<char> vkBufferShader = readFile("buffer.spv");
    VulkanShaderModule vkBufferShaderModule(vkDevice, vkBufferShader);
    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList(
        MAX_BUFFERS + 1, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);
    VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                            vkBufferShaderModule);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    const uint32_t maxIter = innerIterations;
    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCommandBuffer(vkDevice, vkCommandPool);

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);
    std::vector<VulkanDeviceMemory *> vkBufferListDeviceMemory;
    std::vector<clExternalMemory *> externalMemory;

    for (size_t emhtIdx = 0; emhtIdx < vkExternalMemoryHandleTypeList.size();
         emhtIdx++)
    {
        VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
            vkExternalMemoryHandleTypeList[emhtIdx];
        log_info("External memory handle type: %d\n",
                 vkExternalMemoryHandleType);

        VulkanBuffer vkDummyBuffer(vkDevice, 4 * 1024,
                                   vkExternalMemoryHandleType);
        const VulkanMemoryTypeList &memoryTypeList =
            vkDummyBuffer.getMemoryTypeList();

        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
        {
            const VulkanMemoryType &memoryType = memoryTypeList[mtIdx];

            log_info("Memory type index: %d\n", (uint32_t)memoryType);
            log_info("Memory type property: %d\n",
                     memoryType.getMemoryTypeProperty());

            VulkanBufferList vkBufferList(numBuffers, vkDevice, bufferSize,
                                          vkExternalMemoryHandleType);

            for (size_t bIdx = 0; bIdx < numBuffers; bIdx++)
            {
                vkBufferListDeviceMemory.push_back(
                    new VulkanDeviceMemory(vkDevice, bufferSize, memoryType,
                                           vkExternalMemoryHandleType));
                externalMemory.push_back(new clExternalMemory(
                    vkBufferListDeviceMemory[bIdx], vkExternalMemoryHandleType,
                    0, bufferSize, context, deviceId));
            }
            cl_mem buffers[4];
            clFinish(cmd_queue1);
            Params *params = (Params *)vkParamsDeviceMemory.map();
            params->numBuffers = numBuffers;
            params->bufferSize = bufferSize;
            params->interBufferOffset = 0;
            vkParamsDeviceMemory.unmap();
            vkDescriptorSet.update(0, vkParamsBuffer);
            for (size_t bIdx = 0; bIdx < vkBufferList.size(); bIdx++)
            {
                size_t buffer_size = vkBufferList[bIdx].getSize();
                vkBufferListDeviceMemory[bIdx]->bindBuffer(vkBufferList[bIdx],
                                                           0);
                buffers[bIdx] = externalMemory[bIdx]->getExternalMemoryBuffer();
                vkDescriptorSet.update((uint32_t)bIdx + 1, vkBufferList[bIdx]);
            }
            vkCommandBuffer.begin();
            vkCommandBuffer.bindPipeline(vkComputePipeline);
            vkCommandBuffer.bindDescriptorSets(
                vkComputePipeline, vkPipelineLayout, vkDescriptorSet);
            vkCommandBuffer.dispatch(512, 1, 1);
            vkCommandBuffer.end();

            if (vkBufferList.size() == 1)
            {
                update_buffer_kernel = kernel[0];
            }
            else if (vkBufferList.size() == 2)
            {
                update_buffer_kernel = kernel[1];
            }
            else if (vkBufferList.size() == 4)
            {
                update_buffer_kernel = kernel[2];
            }

            // global work size should be less than or equal to
            // bufferSizeList[i]
            global_work_size[0] = bufferSize;

            for (uint32_t iter = 0; iter < maxIter; iter++)
            {
                if (iter == 0)
                {
                    vkQueue.submit(vkCommandBuffer, vkVk2CLSemaphore);
                }
                else
                {
                    vkQueue.submit(vkCl2VkSemaphore, vkCommandBuffer,
                                   vkVk2CLSemaphore);
                }
                clVk2CLExternalSemaphore->wait(cmd_queue1);

                err = clSetKernelArg(update_buffer_kernel, 0, sizeof(uint32_t),
                                     (void *)&bufferSize);
                for (int i = 0; i < vkBufferList.size(); i++)
                {
                    err |=
                        clSetKernelArg(update_buffer_kernel, i + 1,
                                       sizeof(cl_mem), (void *)&(buffers[i]));
                }

                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to set arg values for kernel\n");
                    goto CLEANUP;
                }
                err = clEnqueueNDRangeKernel(cmd_queue1, update_buffer_kernel,
                                             1, NULL, global_work_size, NULL, 0,
                                             NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err,
                                "Error: Failed to launch update_buffer_kernel,"
                                " error\n");
                    goto CLEANUP;
                }
                if (iter != (maxIter - 1))
                {
                    clCl2VkExternalSemaphore->signal(cmd_queue1);
                }
            }
            error_2 = (uint8_t *)malloc(sizeof(uint8_t));
            if (NULL == error_2)
            {
                log_error("Not able to allocate memory\n");
                goto CLEANUP;
            }

            error_1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(uint8_t), NULL, &err);
            if (CL_SUCCESS != err)
            {
                print_error(err, "Error: clCreateBuffer \n");
                goto CLEANUP;
            }
            uint8_t val = 0;
            err = clEnqueueWriteBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                       sizeof(uint8_t), &val, 0, NULL, NULL);
            if (CL_SUCCESS != err)
            {
                print_error(err, "Error: clEnqueueWriteBuffer \n");
                goto CLEANUP;
            }

            int calc_max_iter = (maxIter * 2);
            for (int i = 0; i < vkBufferList.size(); i++)
            {
                err = clSetKernelArg(verify_kernel, 0, sizeof(cl_mem),
                                     (void *)&(buffers[i]));
                err |=
                    clSetKernelArg(verify_kernel, 1, sizeof(int), &bufferSize);
                err |= clSetKernelArg(verify_kernel, 2, sizeof(int),
                                      &calc_max_iter);
                err |= clSetKernelArg(verify_kernel, 3, sizeof(cl_mem),
                                      (void *)&error_1);
                if (err != CL_SUCCESS)
                {
                    print_error(
                        err,
                        "Error: Failed to set arg values for verify_kernel \n");
                    goto CLEANUP;
                }
                err = clEnqueueNDRangeKernel(cmd_queue1, verify_kernel, 1, NULL,
                                             global_work_size, NULL, 0, NULL,
                                             NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(
                        err, "Error: Failed to launch verify_kernel, error\n");
                    goto CLEANUP;
                }

                err = clEnqueueReadBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                          sizeof(uint8_t), error_2, 0, NULL,
                                          NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err, "Error: Failed read output, error  \n");
                    goto CLEANUP;
                }
                if (*error_2 == 1)
                {
                    log_error("&&&& vulkan_opencl_buffer test FAILED\n");
                    goto CLEANUP;
                }
            }
            for (size_t i = 0; i < vkBufferList.size(); i++)
            {
                delete vkBufferListDeviceMemory[i];
                delete externalMemory[i];
            }
            vkBufferListDeviceMemory.erase(vkBufferListDeviceMemory.begin(),
                                           vkBufferListDeviceMemory.begin()
                                               + numBuffers);
            externalMemory.erase(externalMemory.begin(),
                                 externalMemory.begin() + numBuffers);
        }
    }
CLEANUP:
    for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
    {
        if (vkBufferListDeviceMemory[i])
        {
            delete vkBufferListDeviceMemory[i];
        }
        if (externalMemory[i])
        {
            delete externalMemory[i];
        }
    }
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;
    if (error_2) free(error_2);
    if (error_1) clReleaseMemObject(error_1);
    return err;
}

int run_test_with_multi_import_same_ctx(
    cl_context &context, cl_command_queue &cmd_queue1, cl_kernel *kernel,
    cl_kernel &verify_kernel, VulkanDevice &vkDevice, uint32_t numBuffers,
    uint32_t bufferSize, uint32_t bufferSizeForOffset)
{
    size_t global_work_size[1];
    uint8_t *error_2;
    cl_mem error_1;
    int numImports = numBuffers;
    cl_kernel update_buffer_kernel[MAX_IMPORTS];
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;
    int err = CL_SUCCESS;
    int calc_max_iter;
    bool withOffset;
    uint32_t pBufferSize;

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    VulkanQueue &vkQueue = vkDevice.getQueue();

    std::vector<char> vkBufferShader = readFile("buffer.spv");

    VulkanShaderModule vkBufferShaderModule(vkDevice, vkBufferShader);
    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList(
        MAX_BUFFERS + 1, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);
    VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                            vkBufferShaderModule);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    const uint32_t maxIter = innerIterations;
    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCommandBuffer(vkDevice, vkCommandPool);

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);
    std::vector<VulkanDeviceMemory *> vkBufferListDeviceMemory;
    std::vector<std::vector<clExternalMemory *>> externalMemory;


    for (size_t emhtIdx = 0; emhtIdx < vkExternalMemoryHandleTypeList.size();
         emhtIdx++)
    {
        VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
            vkExternalMemoryHandleTypeList[emhtIdx];
        log_info("External memory handle type: %d\n",
                 vkExternalMemoryHandleType);

        VulkanBuffer vkDummyBuffer(vkDevice, 4 * 1024,
                                   vkExternalMemoryHandleType);
        const VulkanMemoryTypeList &memoryTypeList =
            vkDummyBuffer.getMemoryTypeList();

        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
        {
            const VulkanMemoryType &memoryType = memoryTypeList[mtIdx];

            log_info("Memory type index: %d\n", (uint32_t)memoryType);
            log_info("Memory type property: %d\n",
                     memoryType.getMemoryTypeProperty());
            for (unsigned int withOffset = 0;
                 withOffset <= (unsigned int)enableOffset; withOffset++)
            {
                log_info("Running withOffset case %d\n", (uint32_t)withOffset);
                if (withOffset)
                {
                    pBufferSize = bufferSizeForOffset;
                }
                else
                {
                    pBufferSize = bufferSize;
                }
                cl_mem buffers[MAX_BUFFERS][MAX_IMPORTS];
                VulkanBufferList vkBufferList(numBuffers, vkDevice, pBufferSize,
                                              vkExternalMemoryHandleType);
                uint32_t interBufferOffset =
                    (uint32_t)(vkBufferList[0].getSize());

                for (size_t bIdx = 0; bIdx < numBuffers; bIdx++)
                {
                    if (withOffset == 0)
                    {
                        vkBufferListDeviceMemory.push_back(
                            new VulkanDeviceMemory(vkDevice, pBufferSize,
                                                   memoryType,
                                                   vkExternalMemoryHandleType));
                    }
                    if (withOffset == 1)
                    {
                        uint32_t totalSize =
                            (uint32_t)(vkBufferList.size() * interBufferOffset);
                        vkBufferListDeviceMemory.push_back(
                            new VulkanDeviceMemory(vkDevice, totalSize,
                                                   memoryType,
                                                   vkExternalMemoryHandleType));
                    }
                    std::vector<clExternalMemory *> pExternalMemory;
                    for (size_t cl_bIdx = 0; cl_bIdx < numImports; cl_bIdx++)
                    {
                        pExternalMemory.push_back(new clExternalMemory(
                            vkBufferListDeviceMemory[bIdx],
                            vkExternalMemoryHandleType,
                            withOffset * bIdx * interBufferOffset, pBufferSize,
                            context, deviceId));
                    }
                    externalMemory.push_back(pExternalMemory);
                }

                clFinish(cmd_queue1);
                Params *params = (Params *)vkParamsDeviceMemory.map();
                params->numBuffers = numBuffers;
                params->bufferSize = pBufferSize;
                params->interBufferOffset = interBufferOffset * withOffset;
                vkParamsDeviceMemory.unmap();
                vkDescriptorSet.update(0, vkParamsBuffer);
                for (size_t bIdx = 0; bIdx < vkBufferList.size(); bIdx++)
                {
                    size_t buffer_size = vkBufferList[bIdx].getSize();
                    vkBufferListDeviceMemory[bIdx]->bindBuffer(
                        vkBufferList[bIdx],
                        bIdx * interBufferOffset * withOffset);
                    for (size_t cl_bIdx = 0; cl_bIdx < numImports; cl_bIdx++)
                    {
                        buffers[bIdx][cl_bIdx] =
                            externalMemory[bIdx][cl_bIdx]
                                ->getExternalMemoryBuffer();
                    }
                    vkDescriptorSet.update((uint32_t)bIdx + 1,
                                           vkBufferList[bIdx]);
                }
                vkCommandBuffer.begin();
                vkCommandBuffer.bindPipeline(vkComputePipeline);
                vkCommandBuffer.bindDescriptorSets(
                    vkComputePipeline, vkPipelineLayout, vkDescriptorSet);
                vkCommandBuffer.dispatch(512, 1, 1);
                vkCommandBuffer.end();
                for (int i = 0; i < numImports; i++)
                {
                    update_buffer_kernel[i] = (numBuffers == 1)
                        ? kernel[0]
                        : ((numBuffers == 2) ? kernel[1] : kernel[2]);
                }
                // global work size should be less than or equal to
                // bufferSizeList[i]
                global_work_size[0] = pBufferSize;

                for (uint32_t iter = 0; iter < maxIter; iter++)
                {
                    if (iter == 0)
                    {
                        vkQueue.submit(vkCommandBuffer, vkVk2CLSemaphore);
                    }
                    else
                    {
                        vkQueue.submit(vkCl2VkSemaphore, vkCommandBuffer,
                                       vkVk2CLSemaphore);
                    }
                    clVk2CLExternalSemaphore->wait(cmd_queue1);
                    for (uint8_t launchIter = 0; launchIter < numImports;
                         launchIter++)
                    {
                        err = clSetKernelArg(update_buffer_kernel[launchIter],
                                             0, sizeof(uint32_t),
                                             (void *)&pBufferSize);
                        for (int i = 0; i < numBuffers; i++)
                        {
                            err |= clSetKernelArg(
                                update_buffer_kernel[launchIter], i + 1,
                                sizeof(cl_mem),
                                (void *)&(buffers[i][launchIter]));
                        }

                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to set arg values for "
                                        "kernel\n ");
                            goto CLEANUP;
                        }
                        err = clEnqueueNDRangeKernel(
                            cmd_queue1, update_buffer_kernel[launchIter], 1,
                            NULL, global_work_size, NULL, 0, NULL, NULL);
                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to launch "
                                        "update_buffer_kernel, error\n ");
                            goto CLEANUP;
                        }
                    }
                    if (iter != (maxIter - 1))
                    {
                        clCl2VkExternalSemaphore->signal(cmd_queue1);
                    }
                }
                error_2 = (uint8_t *)malloc(sizeof(uint8_t));
                if (NULL == error_2)
                {
                    log_error("Not able to allocate memory\n");
                    goto CLEANUP;
                }

                error_1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(uint8_t), NULL, &err);
                if (CL_SUCCESS != err)
                {
                    print_error(err, "Error: clCreateBuffer \n");
                    goto CLEANUP;
                }
                uint8_t val = 0;
                err =
                    clEnqueueWriteBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                         sizeof(uint8_t), &val, 0, NULL, NULL);
                if (CL_SUCCESS != err)
                {
                    print_error(err, "Error: clEnqueueWriteBuffer \n");
                    goto CLEANUP;
                }
                calc_max_iter = maxIter * (numBuffers + 1);

                for (int i = 0; i < vkBufferList.size(); i++)
                {
                    err = clSetKernelArg(verify_kernel, 0, sizeof(cl_mem),
                                         (void *)&(buffers[i][0]));
                    err |= clSetKernelArg(verify_kernel, 1, sizeof(int),
                                          &pBufferSize);
                    err |= clSetKernelArg(verify_kernel, 2, sizeof(int),
                                          &calc_max_iter);
                    err |= clSetKernelArg(verify_kernel, 3, sizeof(cl_mem),
                                          (void *)&error_1);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err,
                                    "Error: Failed to set arg values for "
                                    "verify_kernel \n");
                        goto CLEANUP;
                    }
                    err = clEnqueueNDRangeKernel(cmd_queue1, verify_kernel, 1,
                                                 NULL, global_work_size, NULL,
                                                 0, NULL, NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(
                            err,
                            "Error: Failed to launch verify_kernel, error\n");
                        goto CLEANUP;
                    }

                    err = clEnqueueReadBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                              sizeof(uint8_t), error_2, 0, NULL,
                                              NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err, "Error: Failed read output, error \n");
                        goto CLEANUP;
                    }
                    if (*error_2 == 1)
                    {
                        log_error("&&&& vulkan_opencl_buffer test FAILED\n");
                        goto CLEANUP;
                    }
                }
                for (size_t i = 0; i < vkBufferList.size(); i++)
                {
                    for (size_t j = 0; j < numImports; j++)
                    {
                        delete externalMemory[i][j];
                    }
                }
                for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
                {
                    delete vkBufferListDeviceMemory[i];
                }
                vkBufferListDeviceMemory.erase(vkBufferListDeviceMemory.begin(),
                                               vkBufferListDeviceMemory.end());
                for (size_t i = 0; i < externalMemory.size(); i++)
                {
                    externalMemory[i].erase(externalMemory[i].begin(),
                                            externalMemory[i].begin()
                                                + numBuffers);
                }
                externalMemory.clear();
            }
        }
    }
CLEANUP:
    for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
    {
        if (vkBufferListDeviceMemory[i])
        {
            delete vkBufferListDeviceMemory[i];
        }
    }
    for (size_t i = 0; i < externalMemory.size(); i++)
    {
        for (size_t j = 0; j < externalMemory[i].size(); j++)
        {
            if (externalMemory[i][j])
            {
                delete externalMemory[i][j];
            }
        }
    }
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;
    if (error_2) free(error_2);
    if (error_1) clReleaseMemObject(error_1);
    return err;
}

int run_test_with_multi_import_diff_ctx(
    cl_context &context, cl_context &context2, cl_command_queue &cmd_queue1,
    cl_command_queue &cmd_queue2, cl_kernel *kernel1, cl_kernel *kernel2,
    cl_kernel &verify_kernel, cl_kernel verify_kernel2, VulkanDevice &vkDevice,
    uint32_t numBuffers, uint32_t bufferSize, uint32_t bufferSizeForOffset)
{
    size_t global_work_size[1];
    uint8_t *error_3;
    cl_mem error_1;
    cl_mem error_2;
    int numImports = numBuffers;
    cl_kernel update_buffer_kernel1[MAX_IMPORTS];
    cl_kernel update_buffer_kernel2[MAX_IMPORTS];
    clExternalSemaphore *clVk2CLExternalSemaphore = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore = NULL;
    clExternalSemaphore *clVk2CLExternalSemaphore2 = NULL;
    clExternalSemaphore *clCl2VkExternalSemaphore2 = NULL;
    int err = CL_SUCCESS;
    int calc_max_iter;
    bool withOffset;
    uint32_t pBufferSize;

    const std::vector<VulkanExternalMemoryHandleType>
        vkExternalMemoryHandleTypeList =
            getSupportedVulkanExternalMemoryHandleTypeList();
    VulkanExternalSemaphoreHandleType vkExternalSemaphoreHandleType =
        getSupportedVulkanExternalSemaphoreHandleTypeList()[0];
    VulkanSemaphore vkVk2CLSemaphore(vkDevice, vkExternalSemaphoreHandleType);
    VulkanSemaphore vkCl2VkSemaphore(vkDevice, vkExternalSemaphoreHandleType);

    VulkanQueue &vkQueue = vkDevice.getQueue();

    std::vector<char> vkBufferShader = readFile("buffer.spv");

    VulkanShaderModule vkBufferShaderModule(vkDevice, vkBufferShader);
    VulkanDescriptorSetLayoutBindingList vkDescriptorSetLayoutBindingList(
        MAX_BUFFERS + 1, VULKAN_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    VulkanDescriptorSetLayout vkDescriptorSetLayout(
        vkDevice, vkDescriptorSetLayoutBindingList);
    VulkanPipelineLayout vkPipelineLayout(vkDevice, vkDescriptorSetLayout);
    VulkanComputePipeline vkComputePipeline(vkDevice, vkPipelineLayout,
                                            vkBufferShaderModule);

    VulkanDescriptorPool vkDescriptorPool(vkDevice,
                                          vkDescriptorSetLayoutBindingList);
    VulkanDescriptorSet vkDescriptorSet(vkDevice, vkDescriptorPool,
                                        vkDescriptorSetLayout);

    clVk2CLExternalSemaphore = new clExternalSemaphore(
        vkVk2CLSemaphore, context, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore = new clExternalSemaphore(
        vkCl2VkSemaphore, context, vkExternalSemaphoreHandleType, deviceId);

    clVk2CLExternalSemaphore2 = new clExternalSemaphore(
        vkVk2CLSemaphore, context2, vkExternalSemaphoreHandleType, deviceId);
    clCl2VkExternalSemaphore2 = new clExternalSemaphore(
        vkCl2VkSemaphore, context2, vkExternalSemaphoreHandleType, deviceId);

    const uint32_t maxIter = innerIterations;
    VulkanCommandPool vkCommandPool(vkDevice);
    VulkanCommandBuffer vkCommandBuffer(vkDevice, vkCommandPool);

    VulkanBuffer vkParamsBuffer(vkDevice, sizeof(Params));
    VulkanDeviceMemory vkParamsDeviceMemory(
        vkDevice, vkParamsBuffer.getSize(),
        getVulkanMemoryType(vkDevice,
                            VULKAN_MEMORY_TYPE_PROPERTY_HOST_VISIBLE_COHERENT));
    vkParamsDeviceMemory.bindBuffer(vkParamsBuffer);
    std::vector<VulkanDeviceMemory *> vkBufferListDeviceMemory;
    std::vector<std::vector<clExternalMemory *>> externalMemory1;
    std::vector<std::vector<clExternalMemory *>> externalMemory2;

    for (size_t emhtIdx = 0; emhtIdx < vkExternalMemoryHandleTypeList.size();
         emhtIdx++)
    {
        VulkanExternalMemoryHandleType vkExternalMemoryHandleType =
            vkExternalMemoryHandleTypeList[emhtIdx];
        log_info("External memory handle type:%d\n",
                 vkExternalMemoryHandleType);

        VulkanBuffer vkDummyBuffer(vkDevice, 4 * 1024,
                                   vkExternalMemoryHandleType);
        const VulkanMemoryTypeList &memoryTypeList =
            vkDummyBuffer.getMemoryTypeList();

        for (size_t mtIdx = 0; mtIdx < memoryTypeList.size(); mtIdx++)
        {
            const VulkanMemoryType &memoryType = memoryTypeList[mtIdx];

            log_info("Memory type index: %d\n", (uint32_t)memoryType);
            log_info("Memory type property: %d\n",
                     memoryType.getMemoryTypeProperty());

            for (unsigned int withOffset = 0;
                 withOffset <= (unsigned int)enableOffset; withOffset++)
            {
                log_info("Running withOffset case %d\n", (uint32_t)withOffset);
                cl_mem buffers1[MAX_BUFFERS][MAX_IMPORTS];
                cl_mem buffers2[MAX_BUFFERS][MAX_IMPORTS];
                if (withOffset)
                {
                    pBufferSize = bufferSizeForOffset;
                }
                else
                {
                    pBufferSize = bufferSize;
                }
                VulkanBufferList vkBufferList(numBuffers, vkDevice, pBufferSize,
                                              vkExternalMemoryHandleType);
                uint32_t interBufferOffset =
                    (uint32_t)(vkBufferList[0].getSize());

                for (size_t bIdx = 0; bIdx < numBuffers; bIdx++)
                {
                    if (withOffset == 0)
                    {
                        vkBufferListDeviceMemory.push_back(
                            new VulkanDeviceMemory(vkDevice, pBufferSize,
                                                   memoryType,
                                                   vkExternalMemoryHandleType));
                    }
                    if (withOffset == 1)
                    {
                        uint32_t totalSize =
                            (uint32_t)(vkBufferList.size() * interBufferOffset);
                        vkBufferListDeviceMemory.push_back(
                            new VulkanDeviceMemory(vkDevice, totalSize,
                                                   memoryType,
                                                   vkExternalMemoryHandleType));
                    }
                    std::vector<clExternalMemory *> pExternalMemory1;
                    std::vector<clExternalMemory *> pExternalMemory2;
                    for (size_t cl_bIdx = 0; cl_bIdx < numImports; cl_bIdx++)
                    {
                        pExternalMemory1.push_back(new clExternalMemory(
                            vkBufferListDeviceMemory[bIdx],
                            vkExternalMemoryHandleType,
                            withOffset * bIdx * interBufferOffset, pBufferSize,
                            context, deviceId));
                        pExternalMemory2.push_back(new clExternalMemory(
                            vkBufferListDeviceMemory[bIdx],
                            vkExternalMemoryHandleType,
                            withOffset * bIdx * interBufferOffset, pBufferSize,
                            context2, deviceId));
                    }
                    externalMemory1.push_back(pExternalMemory1);
                    externalMemory2.push_back(pExternalMemory2);
                }

                clFinish(cmd_queue1);
                Params *params = (Params *)vkParamsDeviceMemory.map();
                params->numBuffers = numBuffers;
                params->bufferSize = pBufferSize;
                params->interBufferOffset = interBufferOffset * withOffset;
                vkParamsDeviceMemory.unmap();
                vkDescriptorSet.update(0, vkParamsBuffer);
                for (size_t bIdx = 0; bIdx < vkBufferList.size(); bIdx++)
                {
                    size_t buffer_size = vkBufferList[bIdx].getSize();
                    vkBufferListDeviceMemory[bIdx]->bindBuffer(
                        vkBufferList[bIdx],
                        bIdx * interBufferOffset * withOffset);
                    for (size_t cl_bIdx = 0; cl_bIdx < numImports; cl_bIdx++)
                    {
                        buffers1[bIdx][cl_bIdx] =
                            externalMemory1[bIdx][cl_bIdx]
                                ->getExternalMemoryBuffer();
                        buffers2[bIdx][cl_bIdx] =
                            externalMemory2[bIdx][cl_bIdx]
                                ->getExternalMemoryBuffer();
                    }
                    vkDescriptorSet.update((uint32_t)bIdx + 1,
                                           vkBufferList[bIdx]);
                }

                vkCommandBuffer.begin();
                vkCommandBuffer.bindPipeline(vkComputePipeline);
                vkCommandBuffer.bindDescriptorSets(
                    vkComputePipeline, vkPipelineLayout, vkDescriptorSet);
                vkCommandBuffer.dispatch(512, 1, 1);
                vkCommandBuffer.end();

                for (int i = 0; i < numImports; i++)
                {
                    update_buffer_kernel1[i] = (numBuffers == 1)
                        ? kernel1[0]
                        : ((numBuffers == 2) ? kernel1[1] : kernel1[2]);
                    update_buffer_kernel2[i] = (numBuffers == 1)
                        ? kernel2[0]
                        : ((numBuffers == 2) ? kernel2[1] : kernel2[2]);
                }

                // global work size should be less than or equal
                // to bufferSizeList[i]
                global_work_size[0] = pBufferSize;

                for (uint32_t iter = 0; iter < maxIter; iter++)
                {
                    if (iter == 0)
                    {
                        vkQueue.submit(vkCommandBuffer, vkVk2CLSemaphore);
                    }
                    else
                    {
                        vkQueue.submit(vkCl2VkSemaphore, vkCommandBuffer,
                                       vkVk2CLSemaphore);
                    }
                    clVk2CLExternalSemaphore->wait(cmd_queue1);

                    for (uint8_t launchIter = 0; launchIter < numImports;
                         launchIter++)
                    {
                        err = clSetKernelArg(update_buffer_kernel1[launchIter],
                                             0, sizeof(uint32_t),
                                             (void *)&pBufferSize);
                        for (int i = 0; i < numBuffers; i++)
                        {
                            err |= clSetKernelArg(
                                update_buffer_kernel1[launchIter], i + 1,
                                sizeof(cl_mem),
                                (void *)&(buffers1[i][launchIter]));
                        }

                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to set arg values for "
                                        "kernel\n ");
                            goto CLEANUP;
                        }
                        err = clEnqueueNDRangeKernel(
                            cmd_queue1, update_buffer_kernel1[launchIter], 1,
                            NULL, global_work_size, NULL, 0, NULL, NULL);
                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to launch "
                                        "update_buffer_kernel, error\n");
                            goto CLEANUP;
                        }
                    }
                    if (iter != (maxIter - 1))
                    {
                        clCl2VkExternalSemaphore->signal(cmd_queue1);
                    }
                }
                clFinish(cmd_queue1);
                for (uint32_t iter = 0; iter < maxIter; iter++)
                {
                    if (iter == 0)
                    {
                        vkQueue.submit(vkCommandBuffer, vkVk2CLSemaphore);
                    }
                    else
                    {
                        vkQueue.submit(vkCl2VkSemaphore, vkCommandBuffer,
                                       vkVk2CLSemaphore);
                    }
                    clVk2CLExternalSemaphore2->wait(cmd_queue2);

                    for (uint8_t launchIter = 0; launchIter < numImports;
                         launchIter++)
                    {
                        err = clSetKernelArg(update_buffer_kernel2[launchIter],
                                             0, sizeof(uint32_t),
                                             (void *)&bufferSize);
                        for (int i = 0; i < numBuffers; i++)
                        {
                            err |= clSetKernelArg(
                                update_buffer_kernel2[launchIter], i + 1,
                                sizeof(cl_mem),
                                (void *)&(buffers2[i][launchIter]));
                        }

                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to set arg values for "
                                        "kernel\n ");
                            goto CLEANUP;
                        }
                        err = clEnqueueNDRangeKernel(
                            cmd_queue2, update_buffer_kernel2[launchIter], 1,
                            NULL, global_work_size, NULL, 0, NULL, NULL);
                        if (err != CL_SUCCESS)
                        {
                            print_error(err,
                                        "Error: Failed to launch "
                                        "update_buffer_kernel, error\n ");
                            goto CLEANUP;
                        }
                    }
                    if (iter != (maxIter - 1))
                    {
                        clCl2VkExternalSemaphore2->signal(cmd_queue2);
                    }
                }
                clFinish(cmd_queue2);
                error_3 = (uint8_t *)malloc(sizeof(uint8_t));
                if (NULL == error_3)
                {
                    log_error("Not able to allocate memory\n");
                    goto CLEANUP;
                }

                error_1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(uint8_t), NULL, &err);
                if (CL_SUCCESS != err)
                {
                    print_error(err, "Error: clCreateBuffer \n");
                    goto CLEANUP;
                }
                error_2 = clCreateBuffer(context2, CL_MEM_WRITE_ONLY,
                                         sizeof(uint8_t), NULL, &err);
                if (CL_SUCCESS != err)
                {
                    print_error(err, "Error: clCreateBuffer \n");
                    goto CLEANUP;
                }
                uint8_t val = 0;
                err =
                    clEnqueueWriteBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                         sizeof(uint8_t), &val, 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err, "Error: Failed read output, error  \n");
                    goto CLEANUP;
                }

                err =
                    clEnqueueWriteBuffer(cmd_queue2, error_2, CL_TRUE, 0,
                                         sizeof(uint8_t), &val, 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    print_error(err, "Error: Failed read output, error  \n");
                    goto CLEANUP;
                }

                calc_max_iter = maxIter * 2 * (numBuffers + 1);
                for (int i = 0; i < numBuffers; i++)
                {
                    err = clSetKernelArg(verify_kernel, 0, sizeof(cl_mem),
                                         (void *)&(buffers1[i][0]));
                    err |= clSetKernelArg(verify_kernel, 1, sizeof(int),
                                          &pBufferSize);
                    err |= clSetKernelArg(verify_kernel, 2, sizeof(int),
                                          &calc_max_iter);
                    err |= clSetKernelArg(verify_kernel, 3, sizeof(cl_mem),
                                          (void *)&error_1);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err,
                                    "Error: Failed to set arg values for "
                                    "verify_kernel \n");
                        goto CLEANUP;
                    }
                    err = clEnqueueNDRangeKernel(cmd_queue1, verify_kernel, 1,
                                                 NULL, global_work_size, NULL,
                                                 0, NULL, NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err,
                                    "Error: Failed to launch verify_kernel,"
                                    "error\n");
                        goto CLEANUP;
                    }

                    err = clEnqueueReadBuffer(cmd_queue1, error_1, CL_TRUE, 0,
                                              sizeof(uint8_t), error_3, 0, NULL,
                                              NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err, "Error: Failed read output, error\n");
                        goto CLEANUP;
                    }
                    if (*error_3 == 1)
                    {
                        log_error("&&&& vulkan_opencl_buffer test FAILED\n");
                        goto CLEANUP;
                    }
                }
                *error_3 = 0;
                for (int i = 0; i < vkBufferList.size(); i++)
                {
                    err = clSetKernelArg(verify_kernel2, 0, sizeof(cl_mem),
                                         (void *)&(buffers2[i][0]));
                    err |= clSetKernelArg(verify_kernel2, 1, sizeof(int),
                                          &pBufferSize);
                    err |= clSetKernelArg(verify_kernel2, 2, sizeof(int),
                                          &calc_max_iter);
                    err |= clSetKernelArg(verify_kernel2, 3, sizeof(cl_mem),
                                          (void *)&error_2);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err,
                                    "Error: Failed to set arg values for "
                                    "verify_kernel \n");
                        goto CLEANUP;
                    }
                    err = clEnqueueNDRangeKernel(cmd_queue2, verify_kernel2, 1,
                                                 NULL, global_work_size, NULL,
                                                 0, NULL, NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err,
                                    "Error: Failed to launch verify_kernel,"
                                    "error\n");
                        goto CLEANUP;
                    }

                    err = clEnqueueReadBuffer(cmd_queue2, error_2, CL_TRUE, 0,
                                              sizeof(uint8_t), error_3, 0, NULL,
                                              NULL);
                    if (err != CL_SUCCESS)
                    {
                        print_error(err, "Error: Failed read output, error\n");
                        goto CLEANUP;
                    }
                    if (*error_3 == 1)
                    {
                        log_error("&&&& vulkan_opencl_buffer test FAILED\n");
                        goto CLEANUP;
                    }
                }
                for (size_t i = 0; i < vkBufferList.size(); i++)
                {
                    for (size_t j = 0; j < numImports; j++)
                    {
                        delete externalMemory1[i][j];
                        delete externalMemory2[i][j];
                    }
                }
                for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
                {
                    delete vkBufferListDeviceMemory[i];
                }
                vkBufferListDeviceMemory.erase(vkBufferListDeviceMemory.begin(),
                                               vkBufferListDeviceMemory.end());
                for (size_t i = 0; i < externalMemory1.size(); i++)
                {
                    externalMemory1[i].erase(externalMemory1[i].begin(),
                                             externalMemory1[i].begin()
                                                 + numBuffers);
                    externalMemory2[i].erase(externalMemory2[i].begin(),
                                             externalMemory2[i].begin()
                                                 + numBuffers);
                }
                externalMemory1.clear();
                externalMemory2.clear();
            }
        }
    }
CLEANUP:
    for (size_t i = 0; i < vkBufferListDeviceMemory.size(); i++)
    {
        if (vkBufferListDeviceMemory[i])
        {
            delete vkBufferListDeviceMemory[i];
        }
    }
    for (size_t i = 0; i < externalMemory1.size(); i++)
    {
        for (size_t j = 0; j < externalMemory1[i].size(); j++)
        {
            if (externalMemory1[i][j])
            {
                delete externalMemory1[i][j];
            }
        }
    }
    for (size_t i = 0; i < externalMemory2.size(); i++)
    {
        for (size_t j = 0; j < externalMemory2[i].size(); j++)
        {
            if (externalMemory2[i][j])
            {
                delete externalMemory2[i][j];
            }
        }
    }
    if (clVk2CLExternalSemaphore) delete clVk2CLExternalSemaphore;
    if (clCl2VkExternalSemaphore) delete clCl2VkExternalSemaphore;
    if (clVk2CLExternalSemaphore2) delete clVk2CLExternalSemaphore2;
    if (clCl2VkExternalSemaphore2) delete clCl2VkExternalSemaphore2;
    if (error_3) free(error_3);
    if (error_1) clReleaseMemObject(error_1);
    if (error_2) clReleaseMemObject(error_2);
    return err;
}

int test_buffer_common(cl_device_id device_, cl_context context_,
                       cl_command_queue queue_, int numElements_)
{

    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;
    cl_int errNum = CL_SUCCESS;
    cl_platform_id platform = NULL;
    size_t extensionSize = 0;
    cl_uint num_devices = 0;
    cl_uint device_no = 0;
    const size_t bufsize = BUFFERSIZE;
    char buf[BUFFERSIZE];
    cl_device_id *devices;
    char *extensions = NULL;
    cl_kernel verify_kernel;
    cl_kernel verify_kernel2;
    cl_kernel kernel[3] = { NULL, NULL, NULL };
    cl_kernel kernel2[3] = { NULL, NULL, NULL };
    const char *program_source_const[3] = { kernel_text_numbuffer_1,
                                            kernel_text_numbuffer_2,
                                            kernel_text_numbuffer_4 };
    const char *program_source_const_verify;
    size_t program_source_length;
    cl_command_queue cmd_queue1 = NULL;
    cl_command_queue cmd_queue2 = NULL;
    cl_command_queue cmd_queue3 = NULL;
    cl_context context = NULL;
    cl_program program[3] = { NULL, NULL, NULL };
    cl_program program_verify, program_verify2;
    cl_context context2 = NULL;


    VulkanDevice vkDevice;
    uint32_t numBuffersList[] = { 1, 2, 4 };
    uint32_t bufferSizeList[] = { 4 * 1024, 64 * 1024, 2 * 1024 * 1024 };
    uint32_t bufferSizeListforOffset[] = { 256, 512, 1024 };

    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, 0, 0 };
    errNum = clGetPlatformIDs(1, &platform, NULL);
    if (errNum != CL_SUCCESS)
    {
        print_error(errNum, "Error: Failed to get platform\n");
        goto CLEANUP;
    }

    errNum =
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (CL_SUCCESS != errNum)
    {
        print_error(errNum, "clGetDeviceIDs failed in returning of devices\n");
        goto CLEANUP;
    }
    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    if (NULL == devices)
    {
        errNum = CL_OUT_OF_HOST_MEMORY;
        print_error(errNum, "Unable to allocate memory for devices\n");
        goto CLEANUP;
    }
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices,
                            NULL);
    if (CL_SUCCESS != errNum)
    {
        print_error(errNum, "Failed to get deviceID.\n");
        goto CLEANUP;
    }
    contextProperties[1] = (cl_context_properties)platform;
    log_info("Assigned contextproperties for platform\n");
    for (device_no = 0; device_no < num_devices; device_no++)
    {
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS, 0,
                                 NULL, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum,
                        "Error in clGetDeviceInfo for getting device_extension "
                        "size....\n");
            goto CLEANUP;
        }
        extensions = (char *)malloc(extensionSize);
        if (NULL == extensions)
        {
            print_error(errNum, "Unable to allocate memory for extensions\n");
            errNum = CL_OUT_OF_HOST_MEMORY;
            goto CLEANUP;
        }
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_EXTENSIONS,
                                 extensionSize, extensions, NULL);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum,
                        "Error in clGetDeviceInfo for device_extension\n");
            goto CLEANUP;
        }
        errNum = clGetDeviceInfo(devices[device_no], CL_DEVICE_UUID_KHR,
                                 CL_UUID_SIZE_KHR, uuid, &extensionSize);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum, "clGetDeviceInfo failed\n");
            goto CLEANUP;
        }
        errNum =
            memcmp(uuid, vkDevice.getPhysicalDevice().getUUID(), VK_UUID_SIZE);
        if (errNum == 0)
        {
            break;
        }
    }
    if (device_no >= num_devices)
    {
        errNum = EXIT_FAILURE;
        print_error(errNum,
                    "OpenCL error: "
                    "No Vulkan-OpenCL Interop capable GPU found.\n");
        goto CLEANUP;
    }
    deviceId = devices[device_no];
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (CL_SUCCESS != errNum)
    {
        print_error(errNum, "error creating context\n");
        goto CLEANUP;
    }
    log_info("Successfully created context !!!\n");

    cmd_queue1 = clCreateCommandQueue(context, devices[device_no], 0, &errNum);
    if (CL_SUCCESS != errNum)
    {
        errNum = CL_INVALID_COMMAND_QUEUE;
        print_error(errNum, "Error: Failed to create command queue!\n");
        goto CLEANUP;
    }
    cmd_queue2 = clCreateCommandQueue(context, devices[device_no], 0, &errNum);
    if (CL_SUCCESS != errNum)
    {
        errNum = CL_INVALID_COMMAND_QUEUE;
        print_error(errNum, "Error: Failed to create command queue!\n");
        goto CLEANUP;
    }
    log_info("clCreateCommandQueue successful\n");
    for (int i = 0; i < 3; i++)
    {
        program_source_length = strlen(program_source_const[i]);
        program[i] =
            clCreateProgramWithSource(context, 1, &program_source_const[i],
                                      &program_source_length, &errNum);
        errNum = clBuildProgram(program[i], 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            print_error(errNum, "Error: Failed to build program \n");
            return errNum;
        }
        // create the kernel
        kernel[i] = clCreateKernel(program[i], "clUpdateBuffer", &errNum);
        if (errNum != CL_SUCCESS)
        {
            print_error(errNum, "clCreateKernel failed \n");
            return errNum;
        }
    }

    program_source_const_verify = kernel_text_verify;
    program_source_length = strlen(program_source_const_verify);
    program_verify =
        clCreateProgramWithSource(context, 1, &program_source_const_verify,
                                  &program_source_length, &errNum);
    errNum = clBuildProgram(program_verify, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        log_error("Error: Failed to build program2\n");
        return errNum;
    }
    verify_kernel = clCreateKernel(program_verify, "checkKernel", &errNum);
    if (errNum != CL_SUCCESS)
    {
        print_error(errNum, "clCreateKernel failed \n");
        return errNum;
    }

    if (multiCtx) // different context guard
    {
        context2 = clCreateContextFromType(
            contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
        if (CL_SUCCESS != errNum)
        {
            print_error(errNum, "error creating context\n");
            goto CLEANUP;
        }
        cmd_queue3 =
            clCreateCommandQueue(context2, devices[device_no], 0, &errNum);
        if (CL_SUCCESS != errNum)
        {
            errNum = CL_INVALID_COMMAND_QUEUE;
            print_error(errNum, "Error: Failed to create command queue!\n");
            goto CLEANUP;
        }
        for (int i = 0; i < 3; i++)
        {
            program_source_length = strlen(program_source_const[i]);
            program[i] =
                clCreateProgramWithSource(context2, 1, &program_source_const[i],
                                          &program_source_length, &errNum);
            errNum = clBuildProgram(program[i], 0, NULL, NULL, NULL, NULL);
            if (errNum != CL_SUCCESS)
            {
                print_error(errNum, "Error: Failed to build program \n");
                return errNum;
            }
            // create the kernel
            kernel2[i] = clCreateKernel(program[i], "clUpdateBuffer", &errNum);
            if (errNum != CL_SUCCESS)
            {
                print_error(errNum, "clCreateKernel failed \n");
                return errNum;
            }
        }
        program_source_length = strlen(program_source_const_verify);
        program_verify =
            clCreateProgramWithSource(context2, 1, &program_source_const_verify,
                                      &program_source_length, &errNum);
        errNum = clBuildProgram(program_verify, 0, NULL, NULL, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            log_error("Error: Failed to build program2\n");
            return errNum;
        }
        verify_kernel2 = clCreateKernel(program_verify, "checkKernel", &errNum);
        if (errNum != CL_SUCCESS)
        {
            print_error(errNum, "clCreateKernel failed \n");
            return errNum;
        }
    }

    for (size_t numBuffersIdx = 0; numBuffersIdx < ARRAY_SIZE(numBuffersList);
         numBuffersIdx++)
    {
        uint32_t numBuffers = numBuffersList[numBuffersIdx];
        log_info("Number of buffers: %d\n", numBuffers);
        for (size_t sizeIdx = 0; sizeIdx < ARRAY_SIZE(bufferSizeList);
             sizeIdx++)
        {
            uint32_t bufferSize = bufferSizeList[sizeIdx];
            uint32_t bufferSizeForOffset = bufferSizeListforOffset[sizeIdx];
            log_info("&&&& RUNNING vulkan_opencl_buffer test for Buffer size: "
                     "%d\n",
                     bufferSize);
            if (multiImport && !multiCtx)
            {
                errNum = run_test_with_multi_import_same_ctx(
                    context, cmd_queue1, kernel, verify_kernel, vkDevice,
                    numBuffers, bufferSize, bufferSizeForOffset);
            }
            else if (multiImport && multiCtx)
            {
                errNum = run_test_with_multi_import_diff_ctx(
                    context, context2, cmd_queue1, cmd_queue3, kernel, kernel2,
                    verify_kernel, verify_kernel2, vkDevice, numBuffers,
                    bufferSize, bufferSizeForOffset);
            }
            else if (numCQ == 2)
            {
                errNum = run_test_with_two_queue(
                    context, cmd_queue1, cmd_queue2, kernel, verify_kernel,
                    vkDevice, numBuffers + 1, bufferSize);
            }
            else
            {
                errNum = run_test_with_one_queue(context, cmd_queue1, kernel,
                                                 verify_kernel, vkDevice,
                                                 numBuffers, bufferSize);
            }
            if (errNum != CL_SUCCESS)
            {
                print_error(errNum, "func_name failed \n");
                goto CLEANUP;
            }
        }
    }

CLEANUP:
    for (int i = 0; i < 3; i++)
    {
        if (program[i]) clReleaseProgram(program[i]);
        if (kernel[i]) clReleaseKernel(kernel[i]);
    }
    if (cmd_queue1) clReleaseCommandQueue(cmd_queue1);
    if (cmd_queue2) clReleaseCommandQueue(cmd_queue2);
    if (cmd_queue3) clReleaseCommandQueue(cmd_queue3);
    if (context) clReleaseContext(context);
    if (context2) clReleaseContext(context2);

    if (devices) free(devices);
    if (extensions) free(extensions);

    return errNum;
}
