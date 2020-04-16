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
static std::string get_device_info_string(cl_device_id device,
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

    return std::string(info.begin(), info.end());
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
