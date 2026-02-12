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

// clang-format off
// These are OpenCL C extensions that should (or should not) have OpenCL C
// defines, depending whether the extension is supported.
static const std::vector<std::string> opencl_c_extensions = {
    "cl_khr_3d_image_writes",
    "cl_khr_byte_addressable_store",
    "cl_khr_depth_images",
    "cl_khr_device_enqueue_local_arg_types",
    "cl_khr_extended_async_copies",
    "cl_khr_extended_bit_ops",
    "cl_khr_fp16",
    "cl_khr_fp64",
    "cl_khr_gl_depth_images",
    "cl_khr_gl_msaa_sharing",
    "cl_khr_global_int32_base_atomics",
    "cl_khr_global_int32_extended_atomics",
    "cl_khr_int64_base_atomics",
    "cl_khr_int64_extended_atomics",
    "cl_khr_integer_dot_product",
    "cl_khr_kernel_clock",
    "cl_khr_local_int32_base_atomics",
    "cl_khr_local_int32_extended_atomics",
    "cl_khr_mipmap_image",
    "cl_khr_mipmap_image_writes",
    "cl_khr_select_fprounding_mode",
    "cl_khr_srgb_image_writes",
    "cl_khr_subgroup_ballot",
    "cl_khr_subgroup_clustered_reduce",
    "cl_khr_subgroup_extended_types",
    "cl_khr_subgroup_named_barrier",
    "cl_khr_subgroup_non_uniform_arithmetic",
    "cl_khr_subgroup_non_uniform_vote",
    "cl_khr_subgroup_rotate",
    "cl_khr_subgroup_shuffle",
    "cl_khr_subgroup_shuffle_relative",
    "cl_khr_subgroups",
    "cles_khr_int64",
};
// clang-format on

static const std::string kernel_prolog = R"(
__kernel void test_defines_for_extensions(__global int *supported)
{
)";

static const std::string kernel_epilog = R"(
}
)";

REGISTER_TEST(compiler_defines_for_extensions_new)
{
    cl_int error = CL_SUCCESS;

    std::vector<Version> test_clc_versions;
    if (get_device_cl_version(device) < Version(3, 0))
    {
        test_clc_versions.push_back(get_device_cl_c_version(device));
    }
    else
    {
        size_t sz = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0,
                                NULL, &sz);
        test_error(error,
                   "Unable to query CL_DEVICE_OPENCL_C_ALL_VERSIONS size");

        std::vector<cl_name_version> device_clc_versions(
            sz / sizeof(cl_name_version));
        error = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, sz,
                                device_clc_versions.data(), NULL);
        test_error(error, "Unable to query CL_DEVICE_OPENCL_C_ALL_VERSIONS");

        for (const auto& version : device_clc_versions)
        {
            const unsigned major = CL_VERSION_MAJOR(version.version);
            const unsigned minor = CL_VERSION_MINOR(version.version);
            test_clc_versions.push_back(Version(major, minor));
        }
    }

    size_t mismatches = 0;
    for (const auto& test_clc_version : test_clc_versions)
    {
        // Note: there is no -cl-std=CL1.0, so skip it, unless it is the only
        // supported OpenCL C version.
        if (test_clc_versions.size() > 1 && test_clc_version == Version(1, 0))
        {
            continue;
        }

        log_info("    testing OpenCL C version %s\n",
                 test_clc_version.to_string().c_str());

        std::string kernel_string(kernel_prolog);
        for (size_t i = 0; i < opencl_c_extensions.size(); ++i)
        {
            kernel_string += "  #ifdef " + opencl_c_extensions[i] + "\n";
            kernel_string += "    supported[" + std::to_string(i) + "] = 1;\n";
            kernel_string += "  #else\n";
            kernel_string += "    supported[" + std::to_string(i) + "] = 0;\n";
            kernel_string += "  #endif\n";
        }
        kernel_string += kernel_epilog;

        std::string options_string;
        if (!(test_clc_version == Version(1, 0)))
        {
            options_string = "-cl-std=CL";
            options_string += test_clc_version.to_string();
        }

        clProgramWrapper program;
        clKernelWrapper kernel;

        const char* source = kernel_string.c_str();
        const char* options = options_string.c_str();
        error = create_single_kernel_helper(
            context, &program, &kernel, 1, (const char**)&source,
            "test_defines_for_extensions", options);
        test_error(error, "Unable to create test kernel");

        clMemWrapper dst = clCreateBuffer(
            context, 0, opencl_c_extensions.size() * sizeof(cl_int), NULL,
            &error);
        test_error(error, "Unable to create dst buffer");

        error = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
        test_error(error, "Unable to set dst buffer kernel arg");

        size_t one = 1;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &one, NULL, 0,
                                       NULL, NULL);
        test_error(error, "Unable to enqueue test kernel");

        std::vector<cl_int> results(opencl_c_extensions.size(), 99);
        error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                    results.size() * sizeof(cl_int),
                                    results.data(), 0, NULL, NULL);
        test_error(error, "Unable to read data after test kernel");

        for (size_t i = 0; i < opencl_c_extensions.size(); ++i)
        {
            const char* extension = opencl_c_extensions[i].c_str();
            if (results[i] == 1)
            {
                if (!is_extension_available(device, extension))
                {
                    log_error(
                        "Extension %s is defined but not supported by the "
                        "device.\n",
                        extension);
                    mismatches++;
                }
            }
            else if (results[i] == 0)
            {
                if (is_extension_available(device, extension))
                {
                    log_error(
                        "Extension %s is not defined but is supported by the "
                        "device.\n",
                        extension);
                    mismatches++;
                }
            }
            else
            {
                test_fail("Unexpected result at index %zu: %d\n", i,
                          results[i]);
            }
        }
    }

    return mismatches == 0 ? TEST_PASS : TEST_FAIL;
}
