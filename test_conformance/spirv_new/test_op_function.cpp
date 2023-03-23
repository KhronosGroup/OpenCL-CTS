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

int test_function(cl_device_id deviceID,
                  cl_context context,
                  cl_command_queue queue,
                  const char *funcType,
                  const std::vector<float> &h_in)
{
    cl_int err = CL_SUCCESS;
    int num = (int)h_in.size();
    size_t bytes = sizeof(float) * num;

    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    std::string spvStr = std::string("op_function") + "_" + std::string(funcType);
    const char *spvName = spvStr.c_str();

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, spvName);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, spvName, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<cl_float> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != -h_in[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}


#define TEST_FUNCTION(TYPE)                     \
    TEST_SPIRV_FUNC(function_##TYPE)            \
    {                                           \
        int num = 1 << 20;                      \
        std::vector<cl_float> in(num);          \
        RandomSeed seed(gRandomSeed);           \
        for (int i = 0; i < num; i++) {         \
            in[i] = genrand<cl_float>(seed);    \
        }                                       \
        return test_function(deviceID,          \
                             context,           \
                             queue,             \
                             #TYPE,             \
                             in);               \
    }                                           \

TEST_FUNCTION(none)
TEST_FUNCTION(inline)
TEST_FUNCTION(noinline)
TEST_FUNCTION(pure)
TEST_FUNCTION(const)
TEST_FUNCTION(pure_ptr)
