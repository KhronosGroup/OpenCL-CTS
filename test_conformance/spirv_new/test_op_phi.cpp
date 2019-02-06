/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
in whole or in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#include "testBase.h"
#include "types.hpp"

#include <sstream>
#include <string>


template<typename T>
int test_phi(cl_device_id deviceID,
             cl_context context,
             cl_command_queue queue,
             const char *name,
             const std::vector<T> &h_lhs,
             const std::vector<T> &h_rhs,
             const std::vector<T> &h_ref)
{

    cl_int err = CL_SUCCESS;
    int num = (int)h_lhs.size();
    size_t bytes = num * sizeof(T);

    clMemWrapper lhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create lhs buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, bytes, &h_lhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper rhs = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create rhs buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, bytes, &h_rhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to rhs buffer");

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create res buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<T> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_res[i] != h_ref[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

TEST_SPIRV_FUNC(op_phi_2_blocks)
{
    const int num = 1 << 10;
    RandomSeed seed(gRandomSeed);

    std::vector<cl_int> lhs(num);
    std::vector<cl_int> rhs(num);
    std::vector<cl_int> out(num);

    for (int i = 0; i < num; i++) {
        lhs[i] = genrand<cl_int>(seed);
        rhs[i] = genrand<cl_int>(seed);
        out[i] = lhs[i] < rhs[i] ? (rhs[i] - lhs[i]) : (lhs[i] - rhs[i]);
    }

    return test_phi(deviceID, context, queue, "phi_2", lhs, rhs, out);
}

TEST_SPIRV_FUNC(op_phi_3_blocks)
{
    const int num = 1 << 10;
    RandomSeed seed(gRandomSeed);

    std::vector<cl_int> lhs(num);
    std::vector<cl_int> rhs(num);
    std::vector<cl_int> out(num);

    for (int i = 0; i < num; i++) {
        lhs[i] = genrand<cl_int>(seed);
        rhs[i] = genrand<cl_int>(seed);
        if (lhs[i] < rhs[i]) {
            out[i] = lhs[i] < 0 ? -lhs[i] : lhs[i];
        } else {
            out[i] = lhs[i] - rhs[i];
        }
    }

    return test_phi(deviceID, context, queue, "phi_3", lhs, rhs, out);
}

TEST_SPIRV_FUNC(op_phi_4_blocks)
{
    const int num = 1 << 10;
    RandomSeed seed(gRandomSeed);

    std::vector<cl_int> lhs(num);
    std::vector<cl_int> rhs(num);
    std::vector<cl_int> out(num);

    for (int i = 0; i < num; i++) {
        lhs[i] = genrand<cl_int>(seed);
        rhs[i] = genrand<cl_int>(seed);
        if (lhs[i] < rhs[i]) {
            out[i] = lhs[i] < 0 ? -lhs[i] : lhs[i];
        } else {
            out[i] = rhs[i] < 0 ? -rhs[i] : rhs[i];
        }
    }

    return test_phi(deviceID, context, queue, "phi_4", lhs, rhs, out);
}
