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
#include <stdexcept>
#include "deviceInfo.h"
#include "errorHelpers.h"
#include "typeWrappers.h"

/* Helper to allocate and return a buffer containing device information for the specified device info parameter. */
static void *alloc_and_get_device_info( cl_device_id device, cl_device_info param_name, const char *param_description )
{
    size_t size = 0;
    int err;

    if ((err = clGetDeviceInfo(device, param_name, 0, NULL, &size)) != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }

    if (0 == size)
        return NULL;

    auto buffer = new uint8_t[size];

    if ((err = clGetDeviceInfo(device, param_name, size, buffer, NULL)) != CL_SUCCESS)
    {
        delete [] buffer;
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }

    return buffer;
}

/* Determines if an extension is supported by a device. */
int is_extension_available(cl_device_id device, const char *extensionName)
{
    char *extString = alloc_and_get_device_extensions_string(device);

    BufferOwningPtr<char> extStringBuf(extString);

    return strstr(extString, extensionName) != NULL;
}

/* Returns a newly allocated C string containing the supported extensions list for a device. */
char *alloc_and_get_device_extensions_string(cl_device_id device)
{
    return (char *) alloc_and_get_device_info(device, CL_DEVICE_EXTENSIONS, "extensions string");
}

/* Returns a newly allocated C string containing the supported IL version(s) for a device. */
char *alloc_and_get_device_il_version_string(cl_device_id device)
{
    return (char *) alloc_and_get_device_info(device, CL_DEVICE_IL_VERSION, "IL version string");
}

/* Returns a newly allocated C string containing the supported OpenCL version for a device. */
char *alloc_and_get_device_version_string(cl_device_id device)
{
    return (char *) alloc_and_get_device_info(device, CL_DEVICE_VERSION, "version string");
}
