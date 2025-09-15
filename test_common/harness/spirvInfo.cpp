//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "spirvInfo.h"
#include "deviceInfo.h"
#include "errorHelpers.h"

#include <string>

bool gVersionSkip = false;

bool is_spirv_version_supported(cl_device_id deviceID, const char* version)
{
    std::string ilVersions = get_device_il_version_string(deviceID);

    if (gVersionSkip)
    {
        log_info("    Skipping version check for %s.\n", version);
        return true;
    }
    else if (ilVersions.find(version) == std::string::npos)
    {
        return false;
    }

    return true;
}

int get_device_spirv_queries(cl_device_id device,
                             std::vector<const char*>& extendedInstructionSets,
                             std::vector<const char*>& extensions,
                             std::vector<cl_uint>& capabilities)
{
    cl_int error = CL_SUCCESS;

    size_t size = 0;
    error =
        clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR,
                        0, nullptr, &size);
    test_error(error,
               "clGetDeviceInfo failed for "
               "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR size\n");

    extendedInstructionSets.resize(size / sizeof(const char*));
    error =
        clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR,
                        size, extendedInstructionSets.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for "
               "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR\n");

    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, 0, nullptr,
                            &size);
    test_error(
        error,
        "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR size\n");

    extensions.resize(size / sizeof(const char*));
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, size,
                            extensions.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR\n");

    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR, 0,
                            nullptr, &size);
    test_error(
        error,
        "clGetDeviceInfo failed for CL_DEVICE_SPIRV_CAPABILITIES_KHR size\n");

    capabilities.resize(size / sizeof(cl_uint));
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_CAPABILITIES_KHR, size,
                            capabilities.data(), nullptr);
    test_error(error,
               "clGetDeviceInfo failed for CL_DEVICE_SPIRV_CAPABILITIES_KHR\n");

    return CL_SUCCESS;
}
