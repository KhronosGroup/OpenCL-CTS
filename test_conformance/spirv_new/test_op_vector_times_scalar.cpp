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

template<typename Tv, typename Ts>
int test_vector_times_scalar(cl_device_id deviceID,
                             cl_context context,
                             cl_command_queue queue,
                             const char *Tname,
                             std::vector<Tv> &h_lhs,
                             std::vector<Ts> &h_rhs)
{
    if(std::string(Tname).find("double") != std::string::npos) {
        if(!is_extension_available(deviceID, "cl_khr_fp64")) {
            log_info("Extension cl_khr_fp64 not supported; skipping double tests.\n");
            return 0;
        }
    }

    cl_int err = CL_SUCCESS;
    int num = (int)h_lhs.size();
    size_t lhs_bytes = num * sizeof(Tv);
    size_t rhs_bytes = num * sizeof(Ts);
    size_t res_bytes = lhs_bytes;
    int vec_size = sizeof(Tv) / sizeof(Ts);

    clMemWrapper lhs = clCreateBuffer(context, CL_MEM_READ_ONLY, lhs_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create lhs buffer");

    err = clEnqueueWriteBuffer(queue, lhs, CL_TRUE, 0, lhs_bytes, &h_lhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper rhs = clCreateBuffer(context, CL_MEM_READ_ONLY, rhs_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create rhs buffer");

    err = clEnqueueWriteBuffer(queue, rhs, CL_TRUE, 0, rhs_bytes, &h_rhs[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to rhs buffer");

    std::string kernelStr;

    {
        std::stringstream kernelStream;

        if (is_double<Ts>::value) {
            kernelStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        } else if (sizeof(Ts) == sizeof(cl_half)) {
            kernelStream << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }

        kernelStream << "#define Ts " << Tname             << "\n";
        kernelStream << "#define Tv " << Tname << vec_size << "\n";
        kernelStream << "__kernel void vector_times_scalar(    \n";
        kernelStream << "              __global Tv *out,       \n";
        kernelStream << "               const __global Tv *lhs,\n";
        kernelStream << "               const __global Ts *rhs)\n";
        kernelStream << "{                                     \n";
        kernelStream << "    int id = get_global_id(0);        \n";
        kernelStream << "    out[id] = lhs[id] * rhs[id];      \n";
        kernelStream << "}                                     \n";
        kernelStr = kernelStream.str();
    }

    size_t kernelLen = kernelStr.size();
    const char *kernelBuf = kernelStr.c_str();

    std::vector<Tv> h_ref(num);
    {
        // Run the cl kernel for reference results
        clProgramWrapper prog;
        err = create_single_kernel_helper_create_program(context, &prog, 1, &kernelBuf, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to create cl program");

        err = clBuildProgram(prog, 1, &deviceID, NULL, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to build program");

        clKernelWrapper kernel = clCreateKernel(prog, "vector_times_scalar", &err);
        SPIRV_CHECK_ERROR(err, "Failed to create cl kernel");

        clMemWrapper ref = clCreateBuffer(context, CL_MEM_READ_WRITE, res_bytes, NULL, &err);
        SPIRV_CHECK_ERROR(err, "Failed to create ref buffer");

        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ref);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 0");

        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &lhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 1");

        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &rhs);
        SPIRV_CHECK_ERROR(err, "Failed to set arg 2");

        size_t global = num;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

        err = clEnqueueReadBuffer(queue, ref, CL_TRUE, 0, res_bytes, &h_ref[0], 0, NULL, NULL);
        SPIRV_CHECK_ERROR(err, "Failed to read from ref");
    }

    cl_uint bits = sizeof(void *) * 8;
    std::string ref = "vector_times_scalar_";
    ref += Tname;
    const char *spvName = ref.c_str();

    clProgramWrapper prog;
    err = get_program_with_il(prog, deviceID, context, spvName);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, "vector_times_scalar", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    clMemWrapper res = clCreateBuffer(context, CL_MEM_READ_WRITE, res_bytes, NULL, &err);
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

    std::vector<Tv> h_res(num);
    err = clEnqueueReadBuffer(queue, res, CL_TRUE, 0, res_bytes, &h_res[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read from ref");

    for (int i = 0; i < num; i++) {
        if (h_res[i] != h_ref[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

#define TEST_VECTOR_TIMES_SCALAR(TYPE, N)                       \
    TEST_SPIRV_FUNC(op_vector_times_scalar_##TYPE)              \
    {                                                           \
        if (sizeof(cl_##TYPE) == 2) {                           \
            PASSIVE_REQUIRE_FP16_SUPPORT(deviceID);             \
        }                                                       \
        typedef cl_##TYPE##N Tv;                                \
        typedef cl_##TYPE Ts;                                   \
        const int num = 1 << 20;                                \
        std::vector<Tv> lhs(num);                               \
        std::vector<Ts> rhs(num);                               \
                                                                \
        RandomSeed seed(gRandomSeed);                           \
                                                                \
        for (int i = 0; i < num; i++) {                         \
            lhs[i] = genrandReal<cl_##TYPE##N>(seed);           \
            rhs[i] = genrandReal<cl_##TYPE>(seed);              \
        }                                                       \
                                                                \
        return test_vector_times_scalar<Tv, Ts>(deviceID,       \
                                                context, queue, \
                                                #TYPE,          \
                                                lhs, rhs);      \
    }

TEST_VECTOR_TIMES_SCALAR(float, 4)
TEST_VECTOR_TIMES_SCALAR(double, 4)
