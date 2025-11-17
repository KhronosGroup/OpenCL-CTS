//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "harness/featureHelpers.h"

#include <vector>

static const char* test_kernel = R"CLC(
__kernel void test(__global int* dst) {
    dst[0] = 0;
}
)CLC";

// This sub-test checks that CL_DEVICE_OPENCL_C_VERSION meets any API
// requirements and that programs can be built for the reported OpenCL C version
// and all previous versions.
static int test_CL_DEVICE_OPENCL_C_VERSION(cl_device_id device,
                                           cl_context context)
{
    const Version latest_version = Version(3, 0);

    const Version api_version = get_device_cl_version(device);
    const Version clc_version = get_device_cl_c_version(device);

    if (api_version > latest_version)
    {
        log_info("CL_DEVICE_VERSION is %s, which is bigger than %s.\n"
                 "Need to update the opencl_c_versions test!\n",
                 api_version.to_string().c_str(),
                 latest_version.to_string().c_str());
    }

    if (clc_version > latest_version)
    {
        log_info("CL_DEVICE_OPENCL_C_VERSION is %s, which is bigger than %s.\n"
                 "Need to update the opencl_c_versions test!\n",
                 clc_version.to_string().c_str(),
                 latest_version.to_string().c_str());
    }

    // For OpenCL 3.0, the minimum required OpenCL C version is OpenCL C 1.2.
    // For OpenCL 2.x, the minimum required OpenCL C version is OpenCL C 2.0.
    // For other OpenCL versions, the minimum required OpenCL C version is
    // the same as the API version.
    const Version min_clc_version = api_version == Version(3, 0) ? Version(1, 2)
        : api_version >= Version(2, 0)                           ? Version(2, 0)
                                                                 : api_version;
    if (clc_version < min_clc_version)
    {
        log_error("The minimum required OpenCL C version for API version %s is "
                  "%s (got %s)!\n",
                  api_version.to_string().c_str(),
                  min_clc_version.to_string().c_str(),
                  clc_version.to_string().c_str());
        return TEST_FAIL;
    }

    log_info("  testing compilation based on CL_DEVICE_OPENCL_C_VERSION\n");

    struct TestCase
    {
        Version version;
        const char* buildOptions;
    };

    std::vector<TestCase> tests;
    tests.push_back({ Version(1, 1), "-cl-std=CL1.1" });
    tests.push_back({ Version(1, 2), "-cl-std=CL1.2" });
    tests.push_back({ Version(2, 0), "-cl-std=CL2.0" });
    tests.push_back({ Version(3, 0), "-cl-std=CL3.0" });

    for (const auto& testcase : tests)
    {
        if (clc_version >= testcase.version)
        {
            clProgramWrapper program;
            clKernelWrapper kernel;
            cl_int error = create_single_kernel_helper(
                context, &program, &kernel, 1, &test_kernel, "test",
                testcase.buildOptions);
            test_error(error, "Unable to build program!");

            log_info("    successfully built program with build options '%s'\n",
                     testcase.buildOptions);
        }
    }

    return TEST_PASS;
}

// This sub-test checks that CL_DEVICE_OPENCL_C_ALL_VERSIONS includes any
// requirements for the API version, and that programs can be built for all
// reported versions.
static int test_CL_DEVICE_OPENCL_C_ALL_VERSIONS(cl_device_id device,
                                                cl_context context)
{
    // For now, the required OpenCL C version is the same as the API version.
    const Version api_version = get_device_cl_version(device);
    bool found_api_version = false;

    log_info(
        "  testing compilation based on CL_DEVICE_OPENCL_C_ALL_VERSIONS\n");

    cl_int error = CL_SUCCESS;

    size_t sz = 0;
    error =
        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, NULL, &sz);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_ALL_VERSIONS size");

    std::vector<cl_name_version> clc_versions(sz / sizeof(cl_name_version));
    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, sz,
                            clc_versions.data(), NULL);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_FEATURES");

    for (const auto& clc_version : clc_versions)
    {
        const unsigned major = CL_VERSION_MAJOR(clc_version.version);
        const unsigned minor = CL_VERSION_MINOR(clc_version.version);

        if (strcmp(clc_version.name, "OpenCL C") == 0)
        {
            if (api_version == Version(major, minor))
            {
                found_api_version = true;
            }

            if (major == 1 && minor == 0)
            {
                log_info(
                    "    skipping OpenCL C 1.0, there is no -cl-std=CL1.0.\n");
                continue;
            }

            std::string buildOptions = "-cl-std=CL";
            buildOptions += std::to_string(major);
            buildOptions += ".";
            buildOptions += std::to_string(minor);

            clProgramWrapper program;
            clKernelWrapper kernel;
            error = create_single_kernel_helper(context, &program, &kernel, 1,
                                                &test_kernel, "test",
                                                buildOptions.c_str());
            test_error(error, "Unable to build program!");

            log_info("    successfully built program with build options '%s'\n",
                     buildOptions.c_str());
        }
        else
        {
            log_error("    unknown OpenCL C name '%s'.\n", clc_version.name);
            return TEST_FAIL;
        }
    }

    if (!found_api_version)
    {
        log_error("    didn't find required OpenCL C version '%s'!\n",
                  api_version.to_string().c_str());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

// This sub-test checks that any required features are present for a specific
// CL_DEVICE_OPENCL_C_VERSION.
static int test_CL_DEVICE_OPENCL_C_VERSION_features(cl_device_id device,
                                                    cl_context context)
{
    log_info("  testing for OPENCL_C_VERSION required features\n");

    OpenCLCFeatures features;
    int error = get_device_cl_c_features(device, features);
    if (error)
    {
        log_error("Couldn't query OpenCL C features for the device!\n");
        return TEST_FAIL;
    }

    const Version clc_version = get_device_cl_c_version(device);
    if (clc_version >= Version(2, 0))
    {
        bool has_all_OpenCL_C_20_features =
            features.supports__opencl_c_atomic_order_acq_rel
            && features.supports__opencl_c_atomic_order_seq_cst
            && features.supports__opencl_c_atomic_scope_device
            && features.supports__opencl_c_atomic_scope_all_devices
            && features.supports__opencl_c_device_enqueue
            && features.supports__opencl_c_generic_address_space
            && features.supports__opencl_c_pipes
            && features.supports__opencl_c_program_scope_global_variables
            && features.supports__opencl_c_work_group_collective_functions;

        if (features.supports__opencl_c_images)
        {
            has_all_OpenCL_C_20_features = has_all_OpenCL_C_20_features
                && features.supports__opencl_c_3d_image_writes
                && features.supports__opencl_c_read_write_images;
        }

        test_assert_error(
            has_all_OpenCL_C_20_features,
            "At least one required OpenCL C 2.0 feature is missing!");
    }

    return TEST_PASS;
}

// This sub-test checks that all required OpenCL C versions are present for a
// specific CL_DEVICE_OPENCL_C_VERSION.
static int test_CL_DEVICE_OPENCL_C_VERSION_versions(cl_device_id device,
                                                    cl_context context)
{
    log_info("  testing for OPENCL_C_VERSION required versions\n");

    const Version device_clc_version = get_device_cl_c_version(device);

    std::vector<Version> test_clc_versions;
    test_clc_versions.push_back(Version(1, 0));
    test_clc_versions.push_back(Version(1, 1));
    test_clc_versions.push_back(Version(1, 2));
    test_clc_versions.push_back(Version(2, 0));
    test_clc_versions.push_back(Version(3, 0));

    cl_int error = CL_SUCCESS;

    size_t sz = 0;
    error =
        clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, NULL, &sz);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_ALL_VERSIONS size");

    std::vector<cl_name_version> device_clc_versions(sz
                                                     / sizeof(cl_name_version));
    error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, sz,
                            device_clc_versions.data(), NULL);
    test_error(error, "Unable to query CL_DEVICE_OPENCL_C_FEATURES");

    for (const auto& test_clc_version : test_clc_versions)
    {
        if (device_clc_version >= test_clc_version)
        {
            bool found = false;
            for (const auto& check : device_clc_versions)
            {
                const unsigned major = CL_VERSION_MAJOR(check.version);
                const unsigned minor = CL_VERSION_MINOR(check.version);

                if (strcmp(check.name, "OpenCL C") == 0
                    && test_clc_version == Version(major, minor))
                {
                    found = true;
                    break;
                }
            }

            if (found)
            {
                log_info("    found OpenCL C version '%s'\n",
                         test_clc_version.to_string().c_str());
            }
            else
            {
                log_error("Didn't find OpenCL C version '%s'!\n",
                          test_clc_version.to_string().c_str());
                return TEST_FAIL;
            }
        }
    }


    return TEST_PASS;
}

REGISTER_TEST(opencl_c_versions)
{
    check_compiler_available(device);

    const Version version = get_device_cl_version(device);

    int result = TEST_PASS;

    result |= test_CL_DEVICE_OPENCL_C_VERSION(device, context);

    if (version >= Version(3, 0))
    {
        result |= test_CL_DEVICE_OPENCL_C_ALL_VERSIONS(device, context);
        result |= test_CL_DEVICE_OPENCL_C_VERSION_features(device, context);
        result |= test_CL_DEVICE_OPENCL_C_VERSION_versions(device, context);
    }

    return result;
}
