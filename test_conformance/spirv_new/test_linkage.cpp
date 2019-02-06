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

static int test_linkage_compile(cl_device_id deviceID,
                                cl_context context,
                                cl_command_queue queue,
                                const char *fname,
                                clProgramWrapper &prog)
{
    cl_int err = CL_SUCCESS;
    std::vector<unsigned char> buffer_vec = readSPIRV(fname);

    int file_bytes = buffer_vec.size();
    if (file_bytes == 0) {
        log_error("File not found\n");
        return -1;
    }
    unsigned char *buffer = &buffer_vec[0];

    prog = clCreateProgramWithIL(context, buffer, file_bytes, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create program with clCreateProgramWithIL");

    err = clCompileProgram(prog, 1, &deviceID,
                           NULL, // options
                           0,    // num headers
                           NULL, // input headers
                           NULL, // header include names
                           NULL, // callback
                           NULL  // User data
        );
    SPIRV_CHECK_ERROR(err, "Failed to compile spv program");
    return 0;
}

TEST_SPIRV_FUNC(linkage_export_function_compile)
{
    clProgramWrapper prog;
    return test_linkage_compile(deviceID, context, queue, "linkage_export", prog);
}

TEST_SPIRV_FUNC(linkage_import_function_compile)
{
    clProgramWrapper prog;
    return test_linkage_compile(deviceID, context, queue, "linkage_import", prog);
}

TEST_SPIRV_FUNC(linkage_import_function_link)
{
    int err = 0;

    clProgramWrapper prog_export;
    err = test_linkage_compile(deviceID, context, queue, "linkage_export", prog_export);
    SPIRV_CHECK_ERROR(err, "Failed to compile export program");

    clProgramWrapper prog_import;
    err = test_linkage_compile(deviceID, context, queue, "linkage_import", prog_import);
    SPIRV_CHECK_ERROR(err, "Failed to compile import program");

    cl_program progs[] = {prog_export, prog_import};

    clProgramWrapper prog = clLinkProgram(context, 1, &deviceID, NULL, 2, progs, NULL, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to link programs");

    clKernelWrapper kernel = clCreateKernel(prog, "test_linkage", &err);
    SPIRV_CHECK_ERROR(err, "Failed to create spv kernel");

    const int num = 1 << 20;
    std::vector<cl_float> h_in(num);
    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num; i++) {
        h_in[i] = genrand<cl_float>(seed);
    }

    size_t bytes = sizeof(cl_float) * num;
    clMemWrapper in = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create  in buffer");

    err = clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, &h_in[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy to in buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
    SPIRV_CHECK_ERROR(err, "Failed to set arg 1");


    size_t global = num;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue cl kernel");

    std::vector<cl_float> h_out(num);
    err = clEnqueueReadBuffer(queue, in, CL_TRUE, 0, bytes, &h_out[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to read to output");

    for (int i = 0; i < num; i++) {
        if (h_out[i] != -h_in[i]) {
            log_error("Values do not match at location %d\n", i);
            return -1;
        }
    }

    return 0;
}
