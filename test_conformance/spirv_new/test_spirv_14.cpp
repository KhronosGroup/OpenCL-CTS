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

TEST_SPIRV_FUNC(spirv14_nonwriteable_decoration)
{
    if (!is_spirv_version_supported(deviceID, "SPIR-V_1.4"))
    {
        log_info("SPIR-V 1.4 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(
        prog, deviceID, context,
        "spv1.4/nonwriteable_decoration_function_storage_class");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "nonwriteable_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    int result = 0;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(result), NULL, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(result), &result,
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    int expected = 42;
    if (result != expected)
    {
        log_error("Result mismatch!  Got %d, Wanted %d\n", result, expected);
        return TEST_FAIL;
    }

    return TEST_PASS;
}
