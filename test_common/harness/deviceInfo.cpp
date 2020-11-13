//
// Copyright (c) 2017-2019 The Khronos Group Inc.
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

#include <sstream>
#include <stdexcept>
#include <vector>

#include "deviceInfo.h"
#include "errorHelpers.h"
#include "typeWrappers.h"

/* Helper to return a string containing device information for the specified
 * device info parameter. */
std::string get_device_info_string(cl_device_id device,
                                   cl_device_info param_name)
{
    size_t size = 0;
    int err;

    if ((err = clGetDeviceInfo(device, param_name, 0, NULL, &size))
            != CL_SUCCESS
        || size == 0)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }

    std::vector<char> info(size);

    if ((err = clGetDeviceInfo(device, param_name, size, info.data(), NULL))
        != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }

    /* The returned string does not include the null terminator. */
    return std::string(info.data(), size - 1);
}

/* Determines if an extension is supported by a device. */
int is_extension_available(cl_device_id device, const char *extensionName)
{
    std::string extString = get_device_extensions_string(device);
    std::istringstream ss(extString);
    while (ss)
    {
        std::string found;
        ss >> found;
        if (found == extensionName) return true;
    }
    return false;
}

/* Returns a string containing the supported extensions list for a device. */
std::string get_device_extensions_string(cl_device_id device)
{
    return get_device_info_string(device, CL_DEVICE_EXTENSIONS);
}

/* Returns a string containing the supported IL version(s) for a device. */
std::string get_device_il_version_string(cl_device_id device)
{
    return get_device_info_string(device, CL_DEVICE_IL_VERSION);
}

/* Returns a string containing the supported OpenCL version for a device. */
std::string get_device_version_string(cl_device_id device)
{
    return get_device_info_string(device, CL_DEVICE_VERSION);
}

/* Returns a string containing the device name. */
std::string get_device_name(cl_device_id device)
{
    return get_device_info_string(device, CL_DEVICE_NAME);
}

cl_ulong get_device_info_max_size(cl_device_id device, cl_device_info info, unsigned int divisor)
{
    int err;
    cl_ulong max_size;

    if (divisor == 0)
    {
        throw std::runtime_error("Allocation divisor should not be 0\n");
    }

    if ((err = clGetDeviceInfo(device, info, sizeof(max_size), &max_size, NULL))
        != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }
    return max_size / divisor;
}

cl_ulong get_device_info_max_mem_alloc_size(cl_device_id device, unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, divisor);
}

cl_ulong get_device_info_global_mem_size(cl_device_id device, unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_GLOBAL_MEM_SIZE, divisor);
}

cl_ulong get_device_info_max_constant_buffer_size(cl_device_id device, unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, divisor);
}