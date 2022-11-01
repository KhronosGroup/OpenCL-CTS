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

TEST_SPIRV_FUNC(op_type_opaque_simple)
{
    const char *name = "opaque";
    cl_int err = CL_SUCCESS;
    std::vector<unsigned char> buffer_vec = readSPIRV(name);

    int file_bytes = buffer_vec.size();
    if (file_bytes == 0) {
        log_error("File not found\n");
        return -1;
    }
    unsigned char *buffer = &buffer_vec[0];

    clProgramWrapper prog;

    if (gCoreILProgram)
    {
        prog = clCreateProgramWithIL(context, buffer, file_bytes, &err);
        SPIRV_CHECK_ERROR(
            err, "Failed to create program with clCreateProgramWithIL");
    }
    else
    {
        cl_platform_id platform;
        err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM,
                              sizeof(cl_platform_id), &platform, NULL);
        SPIRV_CHECK_ERROR(err,
                          "Failed to get platform info with clGetDeviceInfo");
        clCreateProgramWithILKHR_fn clCreateProgramWithILKHR = NULL;

        clCreateProgramWithILKHR = (clCreateProgramWithILKHR_fn)
            clGetExtensionFunctionAddressForPlatform(
                platform, "clCreateProgramWithILKHR");
        if (clCreateProgramWithILKHR == NULL)
        {
            log_error(
                "ERROR: clGetExtensionFunctionAddressForPlatform failed\n");
            return -1;
        }
        prog = clCreateProgramWithILKHR(context, buffer, file_bytes, &err);
        SPIRV_CHECK_ERROR(
            err, "Failed to create program with clCreateProgramWithILKHR");
    }

    err = clCompileProgram(prog, 1, &deviceID,
                           NULL, // options
                           0, // num headers
                           NULL, // input headers
                           NULL, // header include names
                           NULL, // callback
                           NULL // User data
    );
    SPIRV_CHECK_ERROR(err, "Failed to compile spv program");
    return 0;
}
