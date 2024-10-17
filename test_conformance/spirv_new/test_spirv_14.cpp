//
// Copyright (c) 2024 The Khronos Group Inc.
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
#include "spirvInfo.hpp"
#include "types.hpp"

#include <string>

static int test_usersemantic_decoration(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        bool test_memberdecoratestring)
{
    cl_int error = CL_SUCCESS;

    const char* filename = test_memberdecoratestring
        ? "spv1.4/usersemantic_memberdecoratestring"
        : "spv1.4/usersemantic_decoratestring";

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, filename);
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "usersemantic_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    int h_dst = -1;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                      sizeof(h_dst), &h_dst, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(h_dst), &h_dst,
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (h_dst != 0)
    {
        log_error("Mismatch! Got: %i, Wanted: %i\n", h_dst, 0);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

TEST_SPIRV_FUNC(spirv14_usersemantic_decoratestring)
{
    if (!is_spirv_version_supported(deviceID, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_usersemantic_decoration(deviceID, context, queue, false);
}

TEST_SPIRV_FUNC(spirv14_usersemantic_memberdecoratestring)
{
    if (!is_spirv_version_supported(deviceID, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_usersemantic_decoration(deviceID, context, queue, true);
}
