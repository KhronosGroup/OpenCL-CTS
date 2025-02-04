//
// Copyright (c) 2016-2023 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "testBase.h"
#include "types.hpp"

REGISTER_TEST(op_type_opaque_simple)
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
        err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
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

    err = clCompileProgram(prog, 1, &device,
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
