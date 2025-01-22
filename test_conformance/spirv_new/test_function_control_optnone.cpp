//
// Copyright (c) 2025 The Khronos Group Inc.
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

REGISTER_TEST(function_control_optnone)
{
    // TODO: need to define an extension or some other mechanism to advertise
    // support for SPV_EXT_optnone.
    if (false)
    {
        log_info("SPV_EXT_optnone is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error =
        get_program_with_il(prog, device, context, "function_control_OptNone");
    test_error(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel = clCreateKernel(prog, "test_optnone", &error);
    test_error(error, "Unable to create SPIR-V kernel");

    clMemWrapper dst = clCreateBuffer(context, 0, sizeof(cl_int), NULL, &error);
    test_error(error, "Unable to create destination buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    test_error(error, "Unable to set kernel arguments");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    test_error(error, "Unable to enqueue kernel");

    cl_int value = 0;
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(value), &value,
                                0, NULL, NULL);
    test_error(error, "Unable to read destination buffer");

    test_assert_error(value == 300, "Value does not match");

    return TEST_PASS;
}
