//
// Copyright (c) 2017-2024 The Khronos Group Inc.
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
#include "testBase.h"

#include <string.h>

#include <algorithm>
#include <sstream>

int test_platform_extensions(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    // These extensions are special extensions that may be reported by the
    // platform and not by the devices in the platform.
    // clang-format off
    const std::vector<std::string> cPlatformExtensions = {
        "cl_khr_icd",
        "cl_amd_offline_devices",
    };
    // clang-format on

    cl_platform_id platformID;
    cl_int err;

    err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                          (void *)(&platformID), NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM");

    // First, check that all platform extensions are reported by the device.
    {
        std::string platformExtensions =
            get_platform_info_string(platformID, CL_PLATFORM_EXTENSIONS);

        std::istringstream is(platformExtensions);
        std::string token;
        while (std::getline(is, token, ' '))
        {
            // log_info("Checking platform extension: %s\n", token.c_str());
            bool isPlatformExtension =
                std::find(cPlatformExtensions.begin(),
                          cPlatformExtensions.end(), token)
                != cPlatformExtensions.end();
            if (!isPlatformExtension
                && !is_extension_available(deviceID, token.c_str()))
            {
                test_fail(
                    "%s is not a platform extension and is supported by the "
                    "platform but not by the device\n",
                    token.c_str());
            }
        }
    }

    // Next, check that device extensions reported by all devices are reported
    // by the platform.
    {
        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, 0, NULL,
                             &numDevices);
        test_error(err, "clGetDeviceIDs failed to get number of devices");

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_ALL, numDevices,
                             devices.data(), NULL);
        test_error(err, "clGetDeviceIDs failed to get device IDs");

        std::string deviceExtensions =
            get_device_info_string(deviceID, CL_DEVICE_EXTENSIONS);

        std::istringstream is(deviceExtensions);
        std::string token;
        while (std::getline(is, token, ' '))
        {
            // log_info("Checking device extension: %s\n", token.c_str());
            bool supportedByAllDevices = std::all_of(
                devices.begin(), devices.end(), [&](cl_device_id device) {
                    return is_extension_available(device, token.c_str());
                });
            if (supportedByAllDevices
                && !is_platform_extension_available(platformID, token.c_str()))
            {
                test_fail(
                    "%s is supported by all devices but not by the platform\n",
                    token.c_str());
            }
        }
    }

    return TEST_PASS;
}

int test_get_platform_ids(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    cl_platform_id platforms[16];
    cl_uint num_platforms;
    char *string_returned;

    string_returned = (char *)malloc(8192);

    int total_errors = 0;
    int err = CL_SUCCESS;


    err = clGetPlatformIDs(16, platforms, &num_platforms);
    test_error(err, "clGetPlatformIDs failed");

    if (num_platforms <= 16)
    {
        // Try with NULL
        err = clGetPlatformIDs(num_platforms, platforms, NULL);
        test_error(err, "clGetPlatformIDs failed with NULL for return size");
    }

    if (num_platforms < 1)
    {
        log_error("Found 0 platforms.\n");
        return -1;
    }
    log_info("Found %d platforms.\n", num_platforms);


    for (int p = 0; p < (int)num_platforms; p++)
    {
        cl_device_id *devices;
        cl_uint num_devices;
        size_t size;


        log_info("Platform %d (%p):\n", p, platforms[p]);

        memset(string_returned, 0, 8192);
        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_PROFILE, 8192,
                                string_returned, &size);
        test_error(err, "clGetPlatformInfo for CL_PLATFORM_PROFILE failed");
        log_info("\tCL_PLATFORM_PROFILE: %s\n", string_returned);
        if (strlen(string_returned) + 1 != size)
        {
            log_error(
                "Returned string length %zu does not equal reported one %zu.\n",
                strlen(string_returned) + 1, size);
            total_errors++;
        }

        memset(string_returned, 0, 8192);
        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VERSION, 8192,
                                string_returned, &size);
        test_error(err, "clGetPlatformInfo for CL_PLATFORM_VERSION failed");
        log_info("\tCL_PLATFORM_VERSION: %s\n", string_returned);
        if (strlen(string_returned) + 1 != size)
        {
            log_error(
                "Returned string length %zu does not equal reported one %zu.\n",
                strlen(string_returned) + 1, size);
            total_errors++;
        }

        memset(string_returned, 0, 8192);
        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 8192,
                                string_returned, &size);
        test_error(err, "clGetPlatformInfo for CL_PLATFORM_NAME failed");
        log_info("\tCL_PLATFORM_NAME: %s\n", string_returned);
        if (strlen(string_returned) + 1 != size)
        {
            log_error(
                "Returned string length %zu does not equal reported one %zu.\n",
                strlen(string_returned) + 1, size);
            total_errors++;
        }

        memset(string_returned, 0, 8192);
        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 8192,
                                string_returned, &size);
        test_error(err, "clGetPlatformInfo for CL_PLATFORM_VENDOR failed");
        log_info("\tCL_PLATFORM_VENDOR: %s\n", string_returned);
        if (strlen(string_returned) + 1 != size)
        {
            log_error(
                "Returned string length %zu does not equal reported one %zu.\n",
                strlen(string_returned) + 1, size);
            total_errors++;
        }

        memset(string_returned, 0, 8192);
        err = clGetPlatformInfo(platforms[p], CL_PLATFORM_EXTENSIONS, 8192,
                                string_returned, &size);
        test_error(err, "clGetPlatformInfo for CL_PLATFORM_EXTENSIONS failed");
        log_info("\tCL_PLATFORM_EXTENSIONS: %s\n", string_returned);
        if (strlen(string_returned) + 1 != size)
        {
            log_error(
                "Returned string length %zu does not equal reported one %zu.\n",
                strlen(string_returned) + 1, size);
            total_errors++;
        }

        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL,
                             &num_devices);
        test_error(err, "clGetDeviceIDs failed.\n");
        if (num_devices == 0)
        {
            log_error("clGetDeviceIDs must return at least one device\n");
            total_errors++;
        }

        devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
        memset(devices, 0, sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices,
                             devices, NULL);
        test_error(err, "clGetDeviceIDs failed.\n");

        log_info("\tPlatform has %d devices.\n", (int)num_devices);
        for (int d = 0; d < (int)num_devices; d++)
        {
            size_t returned_size;
            cl_platform_id returned_platform;
            cl_context context;
            cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[p], 0
            };

            err = clGetDeviceInfo(devices[d], CL_DEVICE_PLATFORM,
                                  sizeof(cl_platform_id), &returned_platform,
                                  &returned_size);
            test_error(err, "clGetDeviceInfo failed for CL_DEVICE_PLATFORM\n");
            if (returned_size != sizeof(cl_platform_id))
            {
                log_error(
                    "Reported return size (%zu) does not match expected size "
                    "(%zu).\n",
                    returned_size, sizeof(cl_platform_id));
                total_errors++;
            }

            memset(string_returned, 0, 8192);
            err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 8192,
                                  string_returned, NULL);
            test_error(err, "clGetDeviceInfo failed for CL_DEVICE_NAME\n");

            log_info("\t\tPlatform for device %d (%s) is %p.\n", d,
                     string_returned, returned_platform);

            log_info(
                "\t\t\tTesting clCreateContext for the platform/device...\n");
            // Try creating a context for the platform
            context =
                clCreateContext(properties, 1, &devices[d], NULL, NULL, &err);
            test_error(err,
                       "\t\tclCreateContext failed for device with platform "
                       "properties\n");

            memset(properties, 0, sizeof(cl_context_properties) * 3);

            err = clGetContextInfo(context, CL_CONTEXT_PROPERTIES,
                                   sizeof(cl_context_properties) * 3,
                                   properties, &returned_size);
            test_error(err,
                       "clGetContextInfo for CL_CONTEXT_PROPERTIES failed");
            if (returned_size != sizeof(cl_context_properties) * 3)
            {
                log_error("Invalid size returned from clGetContextInfo for "
                          "CL_CONTEXT_PROPERTIES. Got %zu, expected %zu.\n",
                          returned_size, sizeof(cl_context_properties) * 3);
                total_errors++;
            }

            if (properties[0] != (cl_context_properties)CL_CONTEXT_PLATFORM
                || properties[1] != (cl_context_properties)platforms[p])
            {
                log_error("Wrong properties returned. Expected: [%p %p], got "
                          "[%p %p]\n",
                          (void *)CL_CONTEXT_PLATFORM, platforms[p],
                          (void *)properties[0], (void *)properties[1]);
                total_errors++;
            }

            err = clReleaseContext(context);
            test_error(err, "clReleaseContext failed");
        }
        free(devices);

        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_DEFAULT, 0, NULL,
                             &num_devices);
        test_error(err, "clGetDeviceIDs failed.\n");
        if (num_devices != 1)
        {
            log_error("clGetDeviceIDs must return exactly one device\n");
            total_errors++;
        }
    }

    free(string_returned);

    return total_errors;
}
