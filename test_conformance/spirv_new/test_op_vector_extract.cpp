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

template<typename Tv, typename Ts>
int test_extract(cl_device_id deviceID, cl_context context,
                 cl_command_queue queue, const char *name,
                 const std::vector<Tv> &h_in, const int n)
{
    if(std::string(name).find("double") != std::string::npos) {
        if(!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return 0;
        }
    }
    cl_int err = CL_SUCCESS;

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    int num = (int)h_in.size();
    std::vector<Ts> h_out(num);

    size_t in_bytes = num * sizeof(Tv);
    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, in_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, in_bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to rhs buffer");

    size_t out_bytes = num * sizeof(Ts);
    clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    for (int k = 0; k < n; k++) {

        err = clSetKernelArg(kernel, 2, sizeof(int), &k);
        SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

        size_t global = num;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, out_bytes, &h_out[0], 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

        for (int i = 0; i < num; i++) {
            if (h_out[i] != h_in[i].s[k]) {
                log_error("Values do not match at location %d for vector position %d\n", i, k);
                return -1;
            }
        }
    }
    return 0;
}

#define TEST_VECTOR_EXTRACT(TYPE, N)                        \
    TEST_SPIRV_FUNC(op_vector_##TYPE##N##_extract)          \
    {                                                       \
        typedef cl_##TYPE##N Tv;                            \
        typedef cl_##TYPE Ts;                               \
        const int num = 1 << 20;                            \
        std::vector<Tv> in(num);                            \
        const char *name = "vector_" #TYPE #N "_extract";   \
                                                            \
        RandomSeed seed(gRandomSeed);                       \
                                                            \
        for (int i = 0; i < num; i++) {                     \
            in[i] = genrand<Tv>(seed);                      \
        }                                                   \
                                                            \
        return test_extract<Tv, Ts>(deviceID,               \
                                    context, queue,         \
                                    name,                   \
                                    in, N);                 \
    }

TEST_VECTOR_EXTRACT(int, 4)
TEST_VECTOR_EXTRACT(float, 4)
TEST_VECTOR_EXTRACT(long, 2)
TEST_VECTOR_EXTRACT(double, 2)
TEST_VECTOR_EXTRACT(char, 16)
