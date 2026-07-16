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
#include "testHarness.h"
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
bool is_extension_available(cl_device_id device, const char *extensionName)
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

cl_version get_extension_version(cl_device_id device, const char *extensionName)
{
    cl_int err;
    size_t size;

    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS_WITH_VERSION, 0, nullptr,
                          &size);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo(CL_DEVICE_EXTENSIONS_WITH_"
                                 "VERSION) failed to return size\n");
    }

    std::vector<cl_name_version> extensions(size / sizeof(cl_name_version));
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS_WITH_VERSION, size,
                          extensions.data(), &size);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo(CL_DEVICE_EXTENSIONS_WITH_"
                                 "VERSION) failed to return value\n");
    }

    for (auto &ext : extensions)
    {
        if (!strcmp(extensionName, ext.name))
        {
            return ext.version;
        }
    }

    throw std::runtime_error("Extension " + std::string(extensionName)
                             + " not supported by device!");
}

/* Returns a string containing the supported extensions list for a device. */
std::string get_device_extensions_string(cl_device_id device)
{
    return get_device_info_string(device, CL_DEVICE_EXTENSIONS);
}

/* Determines if an IL is supported by a device. */
bool is_il_available(cl_device_id device, const char *ilNameVersion)
{
    std::string ilString = get_device_il_version_string(device);
    std::istringstream ss(ilString);
    while (ss)
    {
        std::string found;
        ss >> found;
        if (found == ilNameVersion) return true;
    }
    return false;
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

size_t get_max_param_size(cl_device_id device)
{
    size_t ret(0);
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(ret), &ret,
                        nullptr)
        != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }
    return ret;
}

static cl_ulong get_device_info_max_size(cl_device_id device,
                                         cl_device_info info,
                                         unsigned int divisor)
{
    cl_ulong max_size;

    if (divisor == 0)
    {
        throw std::runtime_error("Allocation divisor should not be 0\n");
    }

    if (clGetDeviceInfo(device, info, sizeof(max_size), &max_size, NULL)
        != CL_SUCCESS)
    {
        throw std::runtime_error("clGetDeviceInfo failed\n");
    }
    return max_size / divisor;
}

cl_ulong get_device_info_max_mem_alloc_size(cl_device_id device,
                                            unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                    divisor);
}

cl_ulong get_device_info_global_mem_size(cl_device_id device,
                                         unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_GLOBAL_MEM_SIZE, divisor);
}

cl_ulong get_device_info_max_constant_buffer_size(cl_device_id device,
                                                  unsigned int divisor)
{
    return get_device_info_max_size(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                    divisor);
}

static std::vector<std::string> g_stub_required_extensions;

static int test_stub_check_extensions(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int numElements)
{
    std::vector<std::string> supported;
    for (const auto& ext : g_stub_required_extensions)
    {
        if (is_extension_available(device, ext.c_str()))
        {
            supported.push_back(ext);
        }
    }

    if (!supported.empty())
    {
        std::string extList;
        for (const auto& ext : supported)
        {
            extList += ext + " ";
        }
        log_error("ERROR: Extensions [ %s] are supported by the device, but "
                  "this test was built as a stub!\n",
                  extList.c_str());
        return TEST_FAIL;
    }

    log_info("Required extensions are not supported. Skipping stub.\n");
    return TEST_SKIPPED_ITSELF;
}

int run_extension_stub(int argc, const char* argv[],
                       const std::vector<std::string>& extensionNames)
{
    g_stub_required_extensions = extensionNames;

    test_definition test_list[] = { { test_stub_check_extensions,
                                      "stub_check_extensions" } };

    return runTestHarness(argc, argv, 1, test_list, false, 0);
}

bool is_queue_properties_available(cl_device_id device,
                                   cl_command_queue_properties properties)
{
    cl_command_queue_properties supported_properties;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                                 sizeof(supported_properties),
                                 &supported_properties, NULL);
    return err == CL_SUCCESS
        && ((supported_properties & properties) == properties);
}
