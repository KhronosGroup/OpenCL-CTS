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

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "harness/testHarness.h"
#include <iostream>
#include <string>

typedef struct
{
    cl_uint info;
    const char *name;
} _info;

_info platform_info_table[] = {
#define STRING(x)                                                              \
    {                                                                          \
        x, #x                                                                  \
    }
    STRING(CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR),
    STRING(CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
    STRING(CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR)
#undef STRING
};

_info device_info_table[] = {
#define STRING(x)                                                              \
    {                                                                          \
        x, #x                                                                  \
    }
    STRING(CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR),
    STRING(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR),
    STRING(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
#undef STRING
};

int test_platform_info(cl_device_id deviceID, cl_context _context,
                       cl_command_queue _queue, int num_elements)
{
    cl_uint num_platforms;
    cl_uint i, j;
    cl_platform_id *platforms;
    cl_int errNum;
    cl_uint *handle_type;
    size_t handle_type_size = 0;
    cl_uint num_handles = 0;

    // get total # of platforms
    errNum = clGetPlatformIDs(0, NULL, &num_platforms);
    test_error(errNum, "clGetPlatformIDs (getting count) failed");

    platforms =
        (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms)
    {
        printf("error allocating memory\n");
        exit(1);
    }
    log_info("%d platforms available\n", num_platforms);
    errNum = clGetPlatformIDs(num_platforms, platforms, NULL);
    test_error(errNum, "clGetPlatformIDs (getting IDs) failed");

    for (i = 0; i < num_platforms; i++)
    {
        log_info("Platform%d (id %lu) info:\n", i, (unsigned long)platforms[i]);
        for (j = 0;
             j < sizeof(platform_info_table) / sizeof(platform_info_table[0]);
             j++)
        {
            errNum =
                clGetPlatformInfo(platforms[i], platform_info_table[j].info, 0,
                                  NULL, &handle_type_size);
            test_error(errNum, "clGetPlatformInfo failed");
            num_handles = handle_type_size / sizeof(cl_uint);
            handle_type = (cl_uint *)malloc(handle_type_size);
            errNum =
                clGetPlatformInfo(platforms[i], platform_info_table[j].info,
                                  handle_type_size, handle_type, NULL);
            test_error(errNum, "clGetPlatformInfo failed");

            log_info("%s: \n", platform_info_table[j].name);
            while (num_handles--)
            {
                log_info("%x \n", handle_type[num_handles]);
            }
            if (handle_type)
            {
                free(handle_type);
            }
        }
    }
    if (platforms)
    {
        free(platforms);
    }
    return TEST_PASS;
}

int test_device_info(cl_device_id deviceID, cl_context _context,
                     cl_command_queue _queue, int num_elements)
{
    cl_uint j;
    cl_uint *handle_type;
    size_t handle_type_size = 0;
    cl_uint num_handles = 0;
    cl_int errNum = CL_SUCCESS;
    for (j = 0; j < sizeof(device_info_table) / sizeof(device_info_table[0]);
         j++)
    {
        errNum = clGetDeviceInfo(deviceID, device_info_table[j].info, 0, NULL,
                                 &handle_type_size);
        test_error(errNum, "clGetDeviceInfo failed");

        num_handles = handle_type_size / sizeof(cl_uint);
        handle_type = (cl_uint *)malloc(handle_type_size);

        errNum = clGetDeviceInfo(deviceID, device_info_table[j].info,
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
    return TEST_PASS;
}
