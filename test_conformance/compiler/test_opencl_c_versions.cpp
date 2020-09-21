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

static int test_CL_DEVICE_OPENCL_C_VERSION(cl_device_id device,
                                           cl_context context)
{
    log_info("  testing compilation based on CL_DEVICE_OPENCL_C_VERSION\n");

    const Version clc_version = get_device_cl_c_version(device);
    if (clc_version > Version(3, 0))
    {
        log_info("CL_DEVICE_OPENCL_C_VERSION is %s, which is bigger than %s.\n"
                 "Need to update the opencl_c_versions test!\n",
                 clc_version.to_string().c_str(),
                 Version(3, 0).to_string().c_str());
    }

    if (clc_version < Version(1, 0))
    {
        log_error("CL_DEVICE_OPENCL_C_VERSION must be at least 1.0 (got %s)!\n",
                  clc_version.to_string().c_str());
        return TEST_FAIL;
    }

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
            cl_int error =
                create_single_kernel_helper_create_program_for_device(
                    context, device, &program, 1, &test_kernel,
                    testcase.buildOptions);
            test_error(error, "Unable to build program!");

            log_info("    successfully built program with build options '%s'\n",
                     testcase.buildOptions);
        }
    }

    return TEST_PASS;
}

static int test_CL_DEVICE_OPENCL_C_ALL_VERSIONS(cl_device_id device,
                                                cl_context context)
{
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
            error = create_single_kernel_helper_create_program_for_device(
                context, device, &program, 1, &test_kernel,
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

    return TEST_PASS;
}

// This sub-test checks that any required features are present for a specific
// CL_DEVICE_OPENCL_C_VERSION.
static int test_CL_DEVICE_OPENCL_C_VERSION_features(cl_device_id device,
                                                    cl_context context)
{
    log_info("  testing for OPENCL_C_VERSION required features\n");

    OpenCLCFeatures features;
    int error = get_device_cl_c_features(device, &features);
    if (error)
    {
        log_error("Couldn't query OpenCL C features for the device!\n");
        return TEST_FAIL;
    }

    const Version clc_version = get_device_cl_c_version(device);
    if (clc_version >= Version(2, 0))
    {
        bool has_all_OpenCL_C_20_features =
            features.__opencl_c_atomic_order_acq_rel
            && features.__opencl_c_atomic_order_seq_cst
            && features.__opencl_c_atomic_scope_device
            && features.__opencl_c_atomic_scope_all_devices
            && features.__opencl_c_device_enqueue
            && features.__opencl_c_generic_address_space
            && features.__opencl_c_pipes
            && features.__opencl_c_program_scope_global_variables
            && features.__opencl_c_work_group_collective_functions;

        if (features.__opencl_c_images)
        {
            has_all_OpenCL_C_20_features = has_all_OpenCL_C_20_features
                && features.__opencl_c_read_write_images;
        }

        test_assert_error(
            has_all_OpenCL_C_20_features,
            "At least one required OpenCL C 2.0 feature is missing!");
    }

    return TEST_PASS;
}

// This sub-test checks that all required OpenCL C versions are present for a
// specific C_DEVICE_OPENCL_C_VERSION.
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

int test_opencl_c_versions(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    cl_bool compilerAvailable = CL_FALSE;
    cl_int error =
        clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE,
                        sizeof(compilerAvailable), &compilerAvailable, NULL);
    if (compilerAvailable == CL_FALSE)
    {
        log_info("Skipping test - no compiler is available.\n");
        return TEST_SKIPPED_ITSELF;
    }

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
