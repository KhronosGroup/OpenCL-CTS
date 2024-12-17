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
//

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "harness/deviceInfo.h"
#include "harness/testHarness.h"
#include <iostream>
#include <string>
#include <vector>

#include "vulkan_test_base.h"

namespace {

typedef struct
{
    cl_uint info;
    const char *name;
} _info;

_info platform_info_table[] = {
#define PLATFORM_INFO_STRING(x)                                                \
    {                                                                          \
        x, #x                                                                  \
    }
    PLATFORM_INFO_STRING(CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR),
    PLATFORM_INFO_STRING(CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
    PLATFORM_INFO_STRING(CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR)
#undef PLATFORM_INFO_STRING
};

_info device_info_table[] = {
#define DEVICE_INFO_STRING(x)                                                  \
    {                                                                          \
        x, #x                                                                  \
    }
    DEVICE_INFO_STRING(CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR),
    DEVICE_INFO_STRING(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
    DEVICE_INFO_STRING(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
#undef DEVICE_INFO_STRING
};

struct PlatformInfoTest : public VulkanTestBase
{
    PlatformInfoTest(cl_device_id device, cl_context context,
                     cl_command_queue queue, cl_int nelems)
        : VulkanTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_uint i;
        cl_platform_id platform = getPlatformFromDevice(device);
        cl_int errNum;
        cl_uint *handle_type;
        size_t handle_type_size = 0;
        cl_uint num_handles = 0;
        cl_bool external_mem_extn_available = is_platform_extension_available(
            platform, "cl_khr_external_semaphore");
        cl_bool external_sema_extn_available =
            is_platform_extension_available(platform, "cl_khr_external_memory");
        cl_bool supports_atleast_one_sema_query = false;

        if (!external_mem_extn_available && !external_sema_extn_available)
        {
            log_info("Platform does not support 'cl_khr_external_semaphore' "
                     "and 'cl_khr_external_memory'. Skipping the test.\n");
            return TEST_SKIPPED_ITSELF;
        }

        log_info("Platform (id %lu) info:\n", (unsigned long)platform);

        for (i = 0;
             i < sizeof(platform_info_table) / sizeof(platform_info_table[0]);
             i++)
        {
            errNum = clGetPlatformInfo(platform, platform_info_table[i].info, 0,
                                       NULL, &handle_type_size);
            test_error(errNum, "clGetPlatformInfo failed");

            if (handle_type_size == 0)
            {
                if (platform_info_table[i].info
                        == CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR
                    && external_mem_extn_available)
                {
                    test_fail("External memory import handle types should be "
                              "reported if "
                              "cl_khr_external_memory is available.\n");
                }
                log_info("%s not supported. Skipping the query.\n",
                         platform_info_table[i].name);
                continue;
            }

            if ((platform_info_table[i].info
                 == CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
                || (platform_info_table[i].info
                    == CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR))
            {
                supports_atleast_one_sema_query = true;
            }

            num_handles = handle_type_size / sizeof(cl_uint);
            handle_type = (cl_uint *)malloc(handle_type_size);
            errNum = clGetPlatformInfo(platform, platform_info_table[i].info,
                                       handle_type_size, handle_type, NULL);
            test_error(errNum, "clGetPlatformInfo failed");

            log_info("%s: \n", platform_info_table[i].name);
            while (num_handles--)
            {
                log_info("%x \n", handle_type[num_handles]);
            }
            if (handle_type)
            {
                free(handle_type);
            }
        }

        if (external_sema_extn_available && !supports_atleast_one_sema_query)
        {
            log_info(
                "External semaphore import/export or both should be supported "
                "if cl_khr_external_semaphore is available.\n");
            return TEST_FAIL;
        }

        return TEST_PASS;
    }
};

struct DeviceInfoTest : public VulkanTestBase
{
    DeviceInfoTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, cl_int nelems)
        : VulkanTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_uint j;
        cl_uint *handle_type;
        size_t handle_type_size = 0;
        cl_uint num_handles = 0;
        cl_int errNum = CL_SUCCESS;
        cl_bool external_mem_extn_available =
            is_extension_available(device, "cl_khr_external_memory");
        cl_bool external_sema_extn_available =
            is_extension_available(device, "cl_khr_external_semaphore");
        cl_bool supports_atleast_one_sema_query = false;

        if (!external_mem_extn_available && !external_sema_extn_available)
        {
            log_info("Device does not support 'cl_khr_external_semaphore' "
                     "and 'cl_khr_external_memory'. Skipping the test.\n");
            return TEST_SKIPPED_ITSELF;
        }

        for (j = 0;
             j < sizeof(device_info_table) / sizeof(device_info_table[0]); j++)
        {
            errNum = clGetDeviceInfo(device, device_info_table[j].info, 0, NULL,
                                     &handle_type_size);
            test_error(errNum, "clGetDeviceInfo failed");

            if (handle_type_size == 0)
            {
                if (device_info_table[j].info
                        == CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR
                    && external_mem_extn_available)
                {
                    test_fail("External memory import handle types should be "
                              "reported if "
                              "cl_khr_external_memory is available.\n");
                }
                log_info("%s not supported. Skipping the query.\n",
                         device_info_table[j].name);
                continue;
            }

            if ((device_info_table[j].info
                 == CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
                || (device_info_table[j].info
                    == CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR))
            {
                supports_atleast_one_sema_query = true;
            }

            num_handles = handle_type_size / sizeof(cl_uint);
            handle_type = (cl_uint *)malloc(handle_type_size);

            errNum = clGetDeviceInfo(device, device_info_table[j].info,
                                     handle_type_size, handle_type, NULL);
            test_error(errNum, "clGetDeviceInfo failed");

            log_info("%s: \n", device_info_table[j].name);
            while (num_handles--)
            {
                log_info("%x \n", handle_type[num_handles]);
            }
            if (handle_type)
            {
                free(handle_type);
            }
        }

        if (external_sema_extn_available && !supports_atleast_one_sema_query)
        {
            log_info(
                "External semaphore import/export or both should be supported "
                "if cl_khr_external_semaphore is available.\n");
            return TEST_FAIL;
        }

        return TEST_PASS;
    }
};

} // anonymous namespace

int test_platform_info(cl_device_id deviceID, cl_context context,
                       cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<PlatformInfoTest>(deviceID, context, defaultQueue,
                                            num_elements);
}

int test_device_info(cl_device_id deviceID, cl_context context,
                     cl_command_queue defaultQueue, int num_elements)
{
    return MakeAndRunTest<DeviceInfoTest>(deviceID, context, defaultQueue,
                                          num_elements);
}
