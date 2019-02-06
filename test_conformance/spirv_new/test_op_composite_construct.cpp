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
int test_composite_construct(cl_device_id deviceID, cl_context context,
                cl_command_queue queue, const char *name,
                std::vector<T> &results,
                bool (*notEqual)(const T&, const T&) = isNotEqual<T>)
{
    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    int num = (int)results.size();

    size_t bytes = num * sizeof(T);
    clMemWrapper mem = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument");

    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    std::vector<T> host(num);
    err = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, bytes, &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    for (int i = 0; i < num; i++) {
        if (notEqual(host[i], results[i])) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }
    return 0;
}

TEST_SPIRV_FUNC(op_composite_construct_int4)
{
    cl_int4 value = {123, 122, 121, 119};
    std::vector<cl_int4> results(256, value);
    return test_composite_construct(deviceID, context, queue, "composite_construct_int4", results);
}

TEST_SPIRV_FUNC(op_composite_construct_struct)
{
    typedef AbstractStruct2<int, char> CustomType1;
    typedef AbstractStruct2<cl_int2, CustomType1> CustomType2;

    CustomType1 value1 = {2100483600, 128};
    cl_int2 intvals = {2100480000, 2100480000};
    CustomType2 value2 = {intvals, value1};

    std::vector<CustomType2> results(256, value2);
    return test_composite_construct(deviceID, context, queue, "composite_construct_struct", results);
}
