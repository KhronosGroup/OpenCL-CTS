//
// Copyright (c) 2021 The Khronos Group Inc.
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

#ifndef _CL_KHR_COMMAND_BUFFER_TEST_BASE_H
#define _CL_KHR_COMMAND_BUFFER_TEST_BASE_H

#include <CL/cl_ext.h>
#include "harness/deviceInfo.h"
#include "harness/testHarness.h"


// Base class for setting function pointers to new extension entry points
struct CommandBufferTestBase
{
    CommandBufferTestBase(cl_device_id device): device(device) {}

    cl_int init_extension_functions()
    {
        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        // If it is supported get the addresses of all the APIs here.
#define GET_EXTENSION_ADDRESS(FUNC)                                            \
    FUNC = reinterpret_cast<FUNC##_fn>(                                        \
        clGetExtensionFunctionAddressForPlatform(platform, #FUNC));            \
    if (FUNC == nullptr)                                                       \
    {                                                                          \
        log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed"     \
                  " with " #FUNC "\n");                                        \
        return TEST_FAIL;                                                      \
    }

        GET_EXTENSION_ADDRESS(clCreateCommandBufferKHR);
        GET_EXTENSION_ADDRESS(clReleaseCommandBufferKHR);
        GET_EXTENSION_ADDRESS(clRetainCommandBufferKHR);
        GET_EXTENSION_ADDRESS(clFinalizeCommandBufferKHR);
        GET_EXTENSION_ADDRESS(clEnqueueCommandBufferKHR);
        GET_EXTENSION_ADDRESS(clCommandBarrierWithWaitListKHR);
        GET_EXTENSION_ADDRESS(clCommandCopyBufferKHR);
        GET_EXTENSION_ADDRESS(clCommandCopyBufferRectKHR);
        GET_EXTENSION_ADDRESS(clCommandCopyBufferToImageKHR);
        GET_EXTENSION_ADDRESS(clCommandCopyImageKHR);
        GET_EXTENSION_ADDRESS(clCommandCopyImageToBufferKHR);
        GET_EXTENSION_ADDRESS(clCommandFillBufferKHR);
        GET_EXTENSION_ADDRESS(clCommandFillImageKHR);
        GET_EXTENSION_ADDRESS(clCommandNDRangeKernelKHR);
        GET_EXTENSION_ADDRESS(clGetCommandBufferInfoKHR);
#undef GET_EXTENSION_ADDRESS
        return CL_SUCCESS;
    }

    clCreateCommandBufferKHR_fn clCreateCommandBufferKHR = nullptr;
    clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR = nullptr;
    clRetainCommandBufferKHR_fn clRetainCommandBufferKHR = nullptr;
    clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR = nullptr;
    clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR = nullptr;
    clCommandBarrierWithWaitListKHR_fn clCommandBarrierWithWaitListKHR =
        nullptr;
    clCommandCopyBufferKHR_fn clCommandCopyBufferKHR = nullptr;
    clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR = nullptr;
    clCommandCopyBufferToImageKHR_fn clCommandCopyBufferToImageKHR = nullptr;
    clCommandCopyImageKHR_fn clCommandCopyImageKHR = nullptr;
    clCommandCopyImageToBufferKHR_fn clCommandCopyImageToBufferKHR = nullptr;
    clCommandFillBufferKHR_fn clCommandFillBufferKHR = nullptr;
    clCommandFillImageKHR_fn clCommandFillImageKHR = nullptr;
    clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR = nullptr;
    clGetCommandBufferInfoKHR_fn clGetCommandBufferInfoKHR = nullptr;

    cl_device_id device = nullptr;
};

#define CHECK_COMMAND_BUFFER_EXTENSION_AVAILABLE(device)                       \
    {                                                                          \
        if (!is_extension_available(device, "cl_khr_command_buffer"))          \
        {                                                                      \
            log_info(                                                          \
                "Device does not support 'cl_khr_command_buffer'. Skipping "   \
                "the test.\n");                                                \
            return TEST_SKIPPED_ITSELF;                                        \
        }                                                                      \
    }


#endif // _CL_KHR_COMMAND_BUFFER_TEST_BASE_H
