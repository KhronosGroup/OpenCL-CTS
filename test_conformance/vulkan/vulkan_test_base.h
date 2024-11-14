//
// Copyright (c) 2024 The Khronos Group Inc.
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

#ifndef CL_VULKAN_TEST_BASE_H
#define CL_VULKAN_TEST_BASE_H

#include <CL/cl_ext.h>

#include <memory>
#include <vector>

#include "vulkan_interop_common.hpp"

#include "harness/deviceInfo.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

inline void params_reset()
{
    numCQ = 1;
    multiImport = false;
    multiCtx = false;
}

struct VulkanTestBase
{
    VulkanTestBase(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
        : device(device), context(context), num_elems(nelems)
    {
        vkDevice.reset(
            new VulkanDevice(getAssociatedVulkanPhysicalDevice(device)));

        if (!(is_extension_available(device, "cl_khr_external_memory")
              && is_extension_available(device, "cl_khr_external_semaphore")))
        {
            log_info("Device does not support cl_khr_external_memory "
                     "or cl_khr_external_semaphore\n");
            log_info(" TEST SKIPPED\n");
            throw std::runtime_error("VulkanTestBase not supported");
        }

        cl_platform_id platform;
        cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                       sizeof(cl_platform_id), &platform, NULL);
        if (error != CL_SUCCESS)
            throw std::runtime_error(
                "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");


        // verify whether selected device is one of the type CL_DEVICE_TYPE_GPU
        cl_uint num_devices = 0;
        error =
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (CL_SUCCESS != error)
            throw std::runtime_error(
                "clGetDeviceIDs failed in returning of devices");

        std::vector<cl_device_id> devices(num_devices);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices,
                               devices.data(), NULL);

        bool found_gpu_match = false;
        for (cl_uint i = 0; i < num_devices; i++)
            if (devices[i] == device)
            {
                found_gpu_match = true;
                break;
            }

        if (!found_gpu_match)
            throw std::runtime_error(
                "Vulkan tests can only run on a GPU device.");

        init_cl_vk_ext(platform, 1, &device);
    }

    virtual cl_int Run() = 0;

protected:
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    clCommandQueueWrapper queue = nullptr;
    cl_int num_elems = 0;
    std::unique_ptr<VulkanDevice> vkDevice;
};

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
{
    if (!checkVkSupport())
    {
        log_info("Vulkan supported GPU not found \n");
        log_info("TEST SKIPPED \n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int status = TEST_PASS;
    try
    {
        // moved from original test - do we want to stick to that ?
        cl_int numElementsToUse = 1024;

        auto test_fixture =
            T(device, context, queue, /*nelems*/ numElementsToUse);
        status = test_fixture.Run();
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return status;
}

#endif // CL_VULKAN_TEST_BASE_H
