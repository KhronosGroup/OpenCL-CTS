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

template<typename T>
int test_atomic(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, const char *name,
                const int num,
                bool is_inc)
{
    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    size_t bytes = num * sizeof(T);
    clMemWrapper ctr_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T), NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    T initial, result;

    if (is_inc) {
        initial = 0;
         result = num;
    } else {
        initial = num;
         result = 0;
    }

    err = clEnqueueWriteBuffer(queue, ctr_mem, CL_TRUE, 0, sizeof(T), &initial, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to lhs buffer");

    clMemWrapper val_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &val_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &ctr_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    T count = 0;
    err = clEnqueueReadBuffer(queue, ctr_mem, CL_TRUE, 0, sizeof(T), &count, 0, NULL, NULL);

    if (count !=  result) {
        log_error("Counter value does not match. Expected: %d, Found: %d\n",  result, (int)count);
        return -1;
    }

    std::vector<cl_int> flags(num, 0);
    std::vector<cl_int> locs(num, -1);
    std::vector<T> host(num);
    err = clEnqueueReadBuffer(queue, val_mem, CL_TRUE, 0, bytes, &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    for (int i = 0; i < num; i++) {
        T idx = host[i] - (is_inc ? 0 : 1);
        if (flags[idx] == 1) {
            log_error("Atomic inc value is repeated at %d and %d\n", locs[idx], i);
            return -1;
        } else {
            flags[idx] = 1;
            locs[idx] = i;
        }
    }
    return 0;
}

TEST_SPIRV_FUNC(op_atomic_inc_global)
{
    int num = 1 << 16;
    return test_atomic<cl_int>(deviceID, context, queue,
                               "atomic_inc_global", num, true);
}

TEST_SPIRV_FUNC(op_atomic_dec_global)
{
    int num = 1 << 16;
    return test_atomic<cl_int>(deviceID, context, queue,
                               "atomic_dec_global", num, false);
}
