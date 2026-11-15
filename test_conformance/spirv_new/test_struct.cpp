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
#include "types.hpp"

#include <string>

static int test_struct_helper(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, const char* filename,
                              const char* kernelname)
{
    cl_int error = CL_SUCCESS;

    clProgramWrapper prog;
    error = get_program_with_il(prog, deviceID, context, filename);
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, kernelname, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    cl_uint result = 0;
    clMemWrapper dst = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                      sizeof(result), &result, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    // Unconditionally pass the data as an int2, which should work both for
    // structure and int2 arguments.
    const cl_uint2 arg = { 123, 456 };
    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(arg), &arg);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(result), &result,
                                0, NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    if (result != arg.s[0] + arg.s[1])
    {
        log_error("Mismatch! Got: %u, Wanted: %u\n", result,
                  arg.s[0] + arg.s[1]);
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST(struct_handling_function_arg_ptr)
{
    return test_struct_helper(device, context, queue,
                              "struct_function_argument_ptr", "func_arg_ptr");
}

REGISTER_TEST(struct_handling_function_arg_val)
{
    return test_struct_helper(device, context, queue,
                              "struct_function_argument_val", "func_arg_val");
}

REGISTER_TEST(struct_handling_function_ret_ptr)
{
    return test_struct_helper(device, context, queue, "struct_function_ret_ptr",
                              "func_ret_ptr");
}

REGISTER_TEST(struct_handling_function_ret_val)
{
    return test_struct_helper(device, context, queue, "struct_function_ret_val",
                              "func_ret_val");
}

REGISTER_TEST(struct_handling_kernel_arg_ptr)
{
    return test_struct_helper(device, context, queue,
                              "struct_kernel_argument_ptr", "kernel_arg_ptr");
}

REGISTER_TEST(struct_handling_kernel_arg_val)
{
    return test_struct_helper(device, context, queue,
                              "struct_kernel_argument_val", "kernel_arg_val");
}
