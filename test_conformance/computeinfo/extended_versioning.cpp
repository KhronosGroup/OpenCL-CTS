//
// Copyright (c) 2019-2020 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <vector>
#include <set>
#include <algorithm>
#include <cstring>
#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

using name_version_set = std::set<cl_name_version_khr>;

static bool operator<(const cl_name_version_khr& lhs,
                      const cl_name_version_khr& rhs)
{
    const int cmp = strcmp(lhs.name, rhs.name);
    if (0 == cmp)
    {
        return lhs.version < rhs.version;
    }

    return cmp < 0;
}

static bool operator==(const cl_name_version_khr& lhs,
                       const cl_name_version_khr& rhs)
{
    return (0 == strcmp(lhs.name, rhs.name)) && (lhs.version == rhs.version);
}

/* Parse major and minor version numbers out of version_string according to
 * format, which is a scanf-format with two %u specifiers, then compare the
 * version to the major and minor versions of version_numeric */
static bool is_same_version(const char* const format,
                            const char* const version_string,
                            const cl_version_khr version_numeric)
{
    unsigned int string_major = 0;
    unsigned int string_minor = 0;
    const int matched =
        sscanf(version_string, format, &string_major, &string_minor);

    if (2 != matched)
    {
        log_error("sscanf() fail on version string \"%s\", format=\"%s\"\n",
                  version_string, format);
        return false;
    }

    const unsigned int numeric_major = CL_VERSION_MAJOR_KHR(version_numeric);
    const unsigned int numeric_minor = CL_VERSION_MINOR_KHR(version_numeric);

    return (string_major == numeric_major) && (string_minor == numeric_minor);
}

static std::vector<char> get_platform_string(cl_platform_id platform,
                                             cl_platform_info name)
{
    size_t size{};
    cl_int err = clGetPlatformInfo(platform, name, 0, nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetPlatformInfo failed\n");
        return {};
    }

    std::vector<char> result(size);

    err = clGetPlatformInfo(platform, name, size, result.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetPlatformInfo failed\n");
        return {};
    }

    return result;
}

static std::vector<char> get_device_string(cl_device_id device,
                                           cl_device_info name)
{
    size_t size{};
    cl_int err = clGetDeviceInfo(device, name, 0, nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return {};
    }

    std::vector<char> result(size);

    err = clGetDeviceInfo(device, name, size, result.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return {};
    }

    return result;
}

/* Parse an extension string into a cl_name_version_khr set. Error out if we
 * have an invalid extension string */
static bool name_version_set_from_extension_string(char* const src,
                                                   name_version_set& dest)
{
    for (char* token = strtok(src, " "); nullptr != token;
         token = strtok(nullptr, " "))
    {
        if (CL_NAME_VERSION_MAX_NAME_SIZE_KHR <= strlen(token))
        {
            log_error("Extension name is longer than allowed\n");
            return false;
        }

        cl_name_version_khr name_version{};
        strncpy(name_version.name, token, CL_NAME_VERSION_MAX_NAME_SIZE_KHR);

        if (dest.find(name_version) != dest.cend())
        {
            log_error("Duplicate extension in extension string\n");
            return false;
        }

        dest.insert(name_version);
    }

    return true;
}

/* Parse a built-in kernels string into a cl_name_version_khr set. Error out if
 * we have an invalid built-in kernels string */
static bool name_version_set_from_built_in_kernel_string(char* const src,
                                                         name_version_set& dest)
{
    for (char* token = strtok(src, ";"); nullptr != token;
         token = strtok(nullptr, ";"))
    {
        if (CL_NAME_VERSION_MAX_NAME_SIZE_KHR <= strlen(token))
        {
            log_error("Kernel name is longer than allowed\n");
            return false;
        }

        cl_name_version_khr name_version{};
        strncpy(name_version.name, token, CL_NAME_VERSION_MAX_NAME_SIZE_KHR);

        if (dest.find(name_version) != dest.cend())
        {
            log_error("Duplicate kernel name in kernel string\n");
            return false;
        }

        dest.insert(name_version);
    }

    return true;
}

/* Helper to log the names of elements of the set difference of two
 * cl_name_version_khr sets */
static void log_name_only_set_difference(const name_version_set& lhs,
                                         const name_version_set& rhs)
{
    std::vector<cl_name_version_khr> difference;
    std::set_difference(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(),
                        std::back_inserter(difference));

    for (const cl_name_version_khr& il : difference)
    {
        log_info(" %s", il.name);
    }
}

/* Helper to log as IL versions the elements of the set difference of two
 * cl_name_version_khr sets */
static void log_il_set_difference(const name_version_set& lhs,
                                  const name_version_set& rhs)
{
    std::vector<cl_name_version_khr> difference;
    std::set_difference(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(),
                        std::back_inserter(difference));

    for (const cl_name_version_khr& il : difference)
    {
        const unsigned int major = CL_VERSION_MAJOR_KHR(il.version);
        const unsigned int minor = CL_VERSION_MINOR_KHR(il.version);
        log_info(" %s_%u.%u", il.name, major, minor);
    }
}

/* Check that CL_PLATFORM_NUMERIC_VERSION_KHR returns the same version as
 * CL_PLATFORM_VERSION */
static int test_extended_versioning_platform_version(cl_platform_id platform)
{
    log_info("Platform versions:\n");

    const std::vector<char> version_string(
        get_platform_string(platform, CL_PLATFORM_VERSION));
    if (version_string.empty())
    {
        log_error("Could not get CL platform version string\n");
        return 1;
    }

    cl_version_khr version_numeric{};
    cl_int err =
        clGetPlatformInfo(platform, CL_PLATFORM_NUMERIC_VERSION_KHR,
                          sizeof(version_numeric), &version_numeric, nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetPlatformInfo failed\n");
        return 1;
    }

    if (!is_same_version("OpenCL %u.%u", version_string.data(),
                         version_numeric))
    {
        log_error(
            "Numeric platform version does not match the version string\n");
        return 1;
    }

    log_info("\tMatched the platform version\n");

    return 0;
}

/* Check that CL_DEVICE{,_OPENCL_C}_NUMERIC_VERSION_KHR return the same versions
 * as CL_DEVICE{,_OPENCL_C}_VERSION */
static int test_extended_versioning_device_versions(cl_device_id deviceID)
{
    log_info("Device versions:\n");

    static constexpr struct
    {
        cl_platform_info param_name_numeric;
        cl_platform_info param_name_string;
        const char* format;
    } device_version_queries[]{
        { CL_DEVICE_NUMERIC_VERSION_KHR, CL_DEVICE_VERSION, "OpenCL %u.%u" },
        { CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, CL_DEVICE_OPENCL_C_VERSION,
          "OpenCL C %u.%u" },
    };

    for (const auto& query : device_version_queries)
    {
        const std::vector<char> version_string(
            get_device_string(deviceID, query.param_name_string));
        if (version_string.empty())
        {
            log_error("Could not get CL platform version string\n");
            return 1;
        }

        cl_version_khr version_numeric{};
        cl_int err =
            clGetDeviceInfo(deviceID, query.param_name_numeric,
                            sizeof(version_numeric), &version_numeric, nullptr);
        if (err != CL_SUCCESS)
        {
            log_error("clGetDeviceInfo failed\n");
            return 1;
        }

        if (!is_same_version(query.format, version_string.data(),
                             version_numeric))
        {
            log_error(
                "Numeric device version does not match the version string\n");
            return 1;
        }
    }

    log_info("\tMatched the device OpenCL and OpenCL C versions\n");

    return 0;
}

/* Check that the platform extension string and name_version queries return the
 * same set */
static int test_extended_versioning_platform_extensions(cl_platform_id platform)
{
    log_info("Platform extensions:\n");
    std::vector<char> extension_string{ get_platform_string(
        platform, CL_PLATFORM_EXTENSIONS) };
    if (extension_string.empty())
    {
        log_error("Could not get CL platform extensions string\n");
        return 1;
    }

    size_t size{};
    cl_int err = clGetPlatformInfo(
        platform, CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR, 0, nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetPlatformInfo failed\n");
        return 1;
    }

    if ((size % sizeof(cl_name_version_khr)) != 0)
    {
        log_error("CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR return size not a "
                  "multiple of sizeof(cl_name_version_khr)\n");
        return 1;
    }

    const size_t extension_name_vers_count = size / sizeof(cl_name_version_khr);
    std::vector<cl_name_version_khr> extension_name_vers(
        extension_name_vers_count);

    err = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR,
                            size, extension_name_vers.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetPlatformInfo failed\n");
        return 1;
    }

    name_version_set extension_name_vers_set;
    for (const auto& extension : extension_name_vers)
    {
        /* Extension string doesn't have versions, so set it to all zeroes for
         * matching */
        cl_name_version_khr name_version = extension;
        name_version.version = CL_MAKE_VERSION_KHR(0, 0, 0);

        if (extension_name_vers_set.find(name_version)
            != extension_name_vers_set.cend())
        {
            log_error("Duplicate extension in extension name-version array\n");
            return 1;
        }

        extension_name_vers_set.insert(name_version);
    }

    name_version_set extension_string_set;
    if (!name_version_set_from_extension_string(extension_string.data(),
                                                extension_string_set))
    {
        log_error("Failed to parse platform extension string\n");
        return 1;
    }

    if (extension_string_set != extension_name_vers_set)
    {
        log_error("Platform extension mismatch\n");

        log_info("\tExtensions only in numeric:");
        log_name_only_set_difference(extension_name_vers_set,
                                     extension_string_set);
        log_info("\n\tExtensions only in string:");
        log_name_only_set_difference(extension_string_set,
                                     extension_name_vers_set);
        log_info("\n");

        return 1;
    }

    log_info("\tMatched %zu extensions\n", extension_name_vers_set.size());

    return 0;
}

/* Check that the device extension string and name_version queries return the
 * same set */
static int test_extended_versioning_device_extensions(cl_device_id device)
{
    log_info("Device extensions:\n");
    std::vector<char> extension_string{ get_device_string(
        device, CL_DEVICE_EXTENSIONS) };
    if (extension_string.empty())
    {
        log_error("Could not get CL device extensions string\n");
        return 1;
    }

    size_t size{};
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR,
                                 0, nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    if ((size % sizeof(cl_name_version_khr)) != 0)
    {
        log_error("CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR return size not a "
                  "multiple of sizeof(cl_name_version_khr)\n");
        return 1;
    }

    const size_t extension_name_vers_count = size / sizeof(cl_name_version_khr);
    std::vector<cl_name_version_khr> extension_name_vers(
        extension_name_vers_count);

    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR, size,
                          extension_name_vers.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    name_version_set extension_name_vers_set;
    for (const auto& extension : extension_name_vers)
    {
        /* Extension string doesn't have versions, so set it to all zeroes for
         * matching */
        cl_name_version_khr name_version = extension;
        name_version.version = CL_MAKE_VERSION_KHR(0, 0, 0);

        if (extension_name_vers_set.find(name_version)
            != extension_name_vers_set.cend())
        {
            log_error("Duplicate extension in extension name-version array\n");
            return 1;
        }

        extension_name_vers_set.insert(name_version);
    }

    name_version_set extension_string_set;
    if (!name_version_set_from_extension_string(extension_string.data(),
                                                extension_string_set))
    {
        log_error("Failed to parse device extension string\n");
        return 1;
    }

    if (extension_string_set != extension_name_vers_set)
    {
        log_error("Device extension mismatch\n");

        log_info("\tExtensions only in numeric:");
        log_name_only_set_difference(extension_name_vers_set,
                                     extension_string_set);
        log_info("\n\tExtensions only in string:");
        log_name_only_set_difference(extension_string_set,
                                     extension_name_vers_set);
        log_info("\n");

        return 1;
    }

    log_info("\tMatched %zu extensions\n", extension_name_vers_set.size());

    return 0;
}

/* Check that the device ILs string and numeric queries return the same set */
static int test_extended_versioning_device_il(cl_device_id device)
{
    log_info("Device ILs:\n");

    size_t size{};
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION_KHR, 0,
                                 nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    if ((size % sizeof(cl_name_version_khr)) != 0)
    {
        log_error("CL_DEVICE_ILS_WITH_VERSION_KHR return size not a multiple "
                  "of sizeof(cl_name_version_khr)\n");
        return 1;
    }

    const size_t il_name_vers_count = size / sizeof(cl_name_version_khr);
    std::vector<cl_name_version_khr> il_name_vers(il_name_vers_count);

    err = clGetDeviceInfo(device, CL_DEVICE_ILS_WITH_VERSION_KHR, size,
                          il_name_vers.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    const bool has_khr_il_program =
        is_extension_available(device, "cl_khr_il_program");
    const bool has_core_il_program =
        get_device_cl_version(device) > Version(2, 0);

    // IL query should return an empty list if the device does not support IL
    // programs
    if (!(has_khr_il_program || has_core_il_program))
    {
        const bool success = il_name_vers.empty();
        if (!success)
        {
            log_error(
                "No il_program support, but CL_DEVICE_ILS_WITH_VERSION_KHR "
                "returned a non-empty list\n");
            return 1;
        }
        else
        {
            log_info(
                "\tNo il_program support, and CL_DEVICE_ILS_WITH_VERSION_KHR "
                "correctly returned the empty list\n");
            return 0;
        }
    }

    // Pick the core or extension version of the query parameter name
    const cl_device_info il_version_param_name =
        has_core_il_program ? CL_DEVICE_IL_VERSION : CL_DEVICE_IL_VERSION_KHR;

    std::vector<char> il_string{ get_device_string(device,
                                                   il_version_param_name) };
    if (il_string.empty())
    {
        log_error("Couldn't get device IL string\n");
        return 1;
    }

    name_version_set il_string_set;
    char* saveptr_outer = nullptr;
    for (char* token = strtok_r(il_string.data(), " ", &saveptr_outer);
         nullptr != token; token = strtok_r(nullptr, " ", &saveptr_outer))
    {
        char* saveptr_inner = nullptr;
        const char* const prefix = strtok_r(token, "_", &saveptr_inner);
        const char* const version = strtok_r(nullptr, "", &saveptr_inner);

        unsigned major = 0;
        unsigned minor = 0;
        const int matched = sscanf(version, "%u.%u", &major, &minor);
        if (2 != matched)
        {
            log_error("IL version string scan mismatch\n");
            return 1;
        }
        if (CL_NAME_VERSION_MAX_NAME_SIZE_KHR <= strlen(prefix))
        {
            log_error("IL name longer than allowed\n");
            return 1;
        }

        cl_name_version_khr name_version{};
        strncpy(name_version.name, prefix, CL_NAME_VERSION_MAX_NAME_SIZE_KHR);
        name_version.version = CL_MAKE_VERSION_KHR(major, minor, 0);

        if (il_string_set.find(name_version) != il_string_set.end())
        {
            log_error("Duplicate IL version in IL string\n");
            return 1;
        }

        il_string_set.insert(name_version);
    }

    name_version_set il_name_vers_set;
    for (const auto& il : il_name_vers)
    {
        const unsigned major = CL_VERSION_MAJOR_KHR(il.version);
        const unsigned minor = CL_VERSION_MINOR_KHR(il.version);

        cl_name_version_khr name_version = il;
        name_version.version = CL_MAKE_VERSION_KHR(major, minor, 0);

        if (il_name_vers_set.find(name_version) != il_name_vers_set.cend())
        {
            log_error("Duplicate IL in name-version array\n");
            return 1;
        }

        il_name_vers_set.insert(name_version);
    }

    if (il_string_set != il_name_vers_set)
    {
        log_error("Device IL mismatch\n");

        log_info("\tILs only in numeric:");
        log_il_set_difference(il_name_vers_set, il_string_set);
        log_info("\n\tILs only in string:");
        log_il_set_difference(il_string_set, il_name_vers_set);
        log_info("\n");

        return 1;
    }

    log_info("\tMatched %zu ILs\n", il_name_vers_set.size());

    return 0;
}

static int test_extended_versioning_device_built_in_kernels(cl_device_id device)
{
    log_info("Device built-in kernels:\n");
    std::vector<char> kernel_string{ get_device_string(
        device, CL_DEVICE_BUILT_IN_KERNELS) };
    if (kernel_string.empty())
    {
        log_error("Could not get CL device extensions string\n");
        return 1;
    }

    size_t size{};
    cl_int err = clGetDeviceInfo(
        device, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR, 0, nullptr, &size);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    if ((size % sizeof(cl_name_version_khr)) != 0)
    {
        log_error("CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR return size not "
                  "a multiple of sizeof(cl_name_version_khr)\n");
        return 1;
    }

    const size_t kernel_name_vers_count = size / sizeof(cl_name_version_khr);
    std::vector<cl_name_version_khr> kernel_name_vers(kernel_name_vers_count);

    err = clGetDeviceInfo(device, CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR,
                          size, kernel_name_vers.data(), nullptr);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return 1;
    }

    name_version_set kernel_name_vers_set;
    for (const auto& kernel : kernel_name_vers)
    {
        cl_name_version_khr name_version = kernel;
        name_version.version = CL_MAKE_VERSION_KHR(0, 0, 0);

        if (kernel_name_vers_set.find(name_version)
            != kernel_name_vers_set.cend())
        {
            log_error("Duplicate kernel in kernel name-version array\n");
            return 1;
        }

        kernel_name_vers_set.insert(name_version);
    }

    name_version_set kernel_string_set;
    if (!name_version_set_from_built_in_kernel_string(kernel_string.data(),
                                                      kernel_string_set))
    {
        log_error("Failed to parse device kernel string\n");
        return 1;
    }

    if (kernel_string_set != kernel_name_vers_set)
    {
        log_error("Device kernel mismatch\n");

        log_info("\tKernels only in numeric:");
        log_name_only_set_difference(kernel_name_vers_set, kernel_string_set);
        log_info("\n\tKernels only in string:");
        log_name_only_set_difference(kernel_string_set, kernel_name_vers_set);
        log_info("\n");

        return 1;
    }

    log_info("\tMatched %zu kernels\n", kernel_name_vers_set.size());

    return 0;
}

int test_extended_versioning(cl_device_id deviceID, cl_context context,
                             cl_command_queue ignoreQueue, int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_extended_versioning"))
    {
        log_info(
            "cl_khr_extended_versioning not supported. Skipping test...\n");
        return 0;
    }

    cl_platform_id platform;
    cl_int err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform),
                                 &platform, nullptr);
    test_error(err, "clGetDeviceInfo failed\n");

    int total_errors = 0;
    total_errors += test_extended_versioning_platform_version(platform);
    total_errors += test_extended_versioning_platform_extensions(platform);
    total_errors += test_extended_versioning_device_versions(deviceID);
    total_errors += test_extended_versioning_device_extensions(deviceID);
    total_errors += test_extended_versioning_device_il(deviceID);
    total_errors += test_extended_versioning_device_built_in_kernels(deviceID);

    return total_errors;
}
