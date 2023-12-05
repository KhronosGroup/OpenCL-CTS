/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed h_in whole or h_in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited h_in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
h_in whole or h_in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include "testBase.h"
#include "types.hpp"

#include <map>
#include <sstream>
#include <string>

extern bool gVersionSkip;

TEST_SPIRV_FUNC(basic_versions)
{
    cl_int error = CL_SUCCESS;

    MTdataHolder d(gRandomSeed);

    std::vector<cl_int> h_src(num_elements);
    generate_random_data(kInt, h_src.size(), d, h_src.data());

    clMemWrapper src = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, h_src.size() * sizeof(cl_int), h_src.data(), &error);
    test_error(error, "Unable to create source buffer");

    clMemWrapper dst = clCreateBuffer(context, 0, h_src.size() * sizeof(cl_int), NULL, &error);
    test_error(error, "Unable to create destination buffer");

    std::map<std::string, std::string> mapILtoSubdir({
        {"SPIR-V_1.0", ""}, // SPIR-V 1.0 files are in the base directory
        {"SPIR-V_1.1", "spv1.1"},
        {"SPIR-V_1.2", "spv1.2"},
        {"SPIR-V_1.3", "spv1.3"},
        {"SPIR-V_1.4", "spv1.4"},
        {"SPIR-V_1.5", "spv1.5"},
        {"SPIR-V_1.6", "spv1.6"},
    });

    size_t sz = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IL_VERSION, 0, NULL, &sz);
    test_error(error, "Unable to query device IL versions size");

    std::string ilVersions;
    ilVersions.resize(sz);
    error = clGetDeviceInfo(deviceID, CL_DEVICE_IL_VERSION, sz, &ilVersions[0], NULL);
    test_error(error, "Unable to query device IL versions string");

    for (auto& testCase : mapILtoSubdir)
    {
        if (gVersionSkip) {
            log_info("Skipping version check for %s.\n", testCase.first.c_str());
        } else if (ilVersions.find(testCase.first) == std::string::npos) {
            log_info("Version %s is not supported; skipping test.\n", testCase.first.c_str());
            continue;
        }

        const cl_int zero = 0;
        error = clEnqueueFillBuffer(queue, dst, &zero, sizeof(zero), 0, h_src.size() * sizeof(cl_int), 0, NULL, NULL);
        test_error(error, "Unable to initialize destination buffer");

        std::string filename = testCase.second + "/basic";

        clProgramWrapper prog;
        error = get_program_with_il(prog, deviceID, context, filename.c_str());
        test_error(error, "Unable to build SPIR-V program");

        clKernelWrapper kernel = clCreateKernel(prog, "test_basic", &error);
        test_error(error, "Unable to create SPIR-V kernel");

        error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
        error |= clSetKernelArg(kernel, 1, sizeof(src), &src);
        test_error(error, "Unable to set kernel arguments");

        size_t global = num_elements;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                       NULL, NULL);
        test_error(error, "Unable to enqueue kernel");

        std::vector<cl_int> h_dst(num_elements);
        error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                    h_dst.size() * sizeof(cl_int), h_dst.data(),
                                    0, NULL, NULL);
        test_error(error, "Unable to read destination buffer");

        for (int i = 0; i < num_elements; i++)
        {
            if (h_dst[i] != h_src[i])
            {
                log_error("Values do not match at location %d\n", i);
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}
