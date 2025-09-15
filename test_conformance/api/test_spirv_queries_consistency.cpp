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
#include "testBase.h"
#include <vector>

static int doQueries(cl_device_id device,
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

// Performs basic consistency checks for the cl_khr_spirv_queries extension.
// When SPIR-V is supported, more in-depth testing is performed by the spirv_new
// test.
REGISTER_TEST(spirv_query_consistency)
{
    REQUIRE_EXTENSION("cl_khr_spirv_queries");

    std::vector<const char*> queriedExtendedInstructionSets;
    std::vector<const char*> queriedExtensions;
    std::vector<cl_uint> queriedCapabilities;

    cl_int error = doQueries(device, queriedExtendedInstructionSets,
                                queriedExtensions, queriedCapabilities);
    test_error_fail(error, "Unable to perform SPIR-V queries");

    auto ilVersions = get_device_il_version_string(device);
    if (ilVersions.find("SPIR-V") == std::string::npos)
    {
        test_assert_error(
            queriedExtendedInstructionSets.empty(),
            "No SPIR-V versions supported, but "
            "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR is not empty");
        test_assert_error(queriedExtensions.empty(),
                          "No SPIR-V versions supported, but "
                          "CL_DEVICE_SPIRV_EXTENSIONS_KHR is not empty");
        test_assert_error(queriedCapabilities.empty(),
                          "No SPIR-V versions supported, but "
                          "CL_DEVICE_SPIRV_CAPABILITIES_KHR is not empty");
    }
    else
    {
        test_assert_error(
            !queriedExtendedInstructionSets.empty(),
            "SPIR-V is supported, but "
            "CL_DEVICE_SPIRV_EXTENDED_INSTRUCTION_SETS_KHR is empty");
        test_assert_error(
            !queriedCapabilities.empty(),
            "SPIR-V is supported, but "
            "CL_DEVICE_SPIRV_CAPABILITIES_KHR is empty");
    }

    return TEST_PASS;
}
