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

#include <sstream>
#include <string>

extern bool gVersionSkip;

TEST_SPIRV_FUNC(basic_1_3)
{
    Version spirv_version = get_device_spirv_il_version(deviceID);
    if (gVersionSkip) {
        log_info("Skipping SPIR-V 1.3 version check.\n");
    } else if(spirv_version < Version(1, 3)) {
        log_info("SPIR-V 1.3 is not supported; skipping test.\n");
        return TEST_PASS;   // can't return TEST_SKIP here
    }

    cl_int error = CL_SUCCESS;
    MTdataHolder d(gRandomSeed);

    std::vector<cl_int> h_src(num_elements);
    generate_random_data(kInt, h_src.size(), d, h_src.data());

    clMemWrapper src = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, h_src.size() * sizeof(cl_int), h_src.data(), &error);
    test_error(error, "Unable to create source buffer");

    clMemWrapper dst = clCreateBuffer(context, 0, h_src.size() * sizeof(cl_int), NULL, &error);
    test_error(error, "Unable to create destination buffer");

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, "spv1.3/basic");
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel = clCreateKernel(prog, "test_basic_1_3", &error);
    test_error(error, "Unable to create SPIR-V kernel");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(src), &src);
    test_error(error, "Unable to set kernel arguments");

    size_t global = num_elements;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue kernel");

    std::vector<cl_int> h_dst(num_elements);
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, h_dst.size() * sizeof(cl_int), h_dst.data(), 0, NULL, NULL);
    test_error(error, "Unable to read destination buffer");

    for (int i = 0; i < num_elements; i++) {
        if (h_dst[i] != h_src[i]) {
            log_error("Values do not match at location %d\n", i);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}
