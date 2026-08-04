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

#include <sstream>
#include <string>

// TODO: move this to a shared location
static bool is_spirv_extension_supported(cl_device_id device,
                                         const char* extension)
{
    if (!is_extension_available(device, "cl_khr_spirv_queries"))
    {
        return false;
    }


    cl_int error = CL_SUCCESS;

    size_t size{};
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, 0, nullptr,
                            &size);
    test_error_ret(
        error,
        "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR size\n",
        false);

    std::vector<const char*> spirvExtensions(size / sizeof(const char*));
    error = clGetDeviceInfo(device, CL_DEVICE_SPIRV_EXTENSIONS_KHR, size,
                            spirvExtensions.data(), nullptr);
    test_error_ret(
        error, "clGetDeviceInfo failed for CL_DEVICE_SPIRV_EXTENSIONS_KHR\n",
        false);

    for (auto check : spirvExtensions)
    {
        if (!strcmp(check, extension))
        {
            return true;
        }
    }

    return false;
}

static int test_nonsemantic_helper(cl_device_id device, cl_context context,
                                   cl_command_queue queue, const char* filename)
{
    cl_int error = 0;

    clProgramWrapper program;
    error = get_program_with_il(program, device, context, filename);
    test_error_fail(error, "Unable to build SPIR-V program");

    clKernelWrapper kernel =
        clCreateKernel(program, "non_semantic_info_test", &error);
    test_error_fail(error, "Unable to create SPIR-V kernel");

    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(cl_int), nullptr, &error);
    test_error_fail(error, "Unable to create dst buffer");

    const cl_int zero = 0;
    error = clEnqueueFillBuffer(queue, dst, &zero, sizeof(zero), 0,
                                sizeof(cl_int), 0, NULL, NULL);
    test_error_fail(error, "Unable to initialize destination buffer");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    test_error_fail(error, "Unable to set kernel arguments");

    const size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    test_error_fail(error, "Unable to enqueue kernel");

    cl_int result = 0;
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(cl_int), &result,
                                0, NULL, NULL);
    test_error_fail(error, "Unable to read destination buffer");

    if (result != 42)
    {
        test_fail("Unxpected result: wanted 42, got %d!\n", result);
    }

    return TEST_PASS;
}

REGISTER_TEST(extinstimport_nonsemantic)
{
    if (!is_spirv_extension_supported(device, "SPV_KHR_non_semantic_info") && false)
    {
        log_info("SPIR-V extension SPV_KHR_non_semantic_info not supported; "
                 "skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int result = TEST_PASS;

    // Test with a SPIR-V module that only declares the
    // SPV_KHR_non_semantic_info extension, but does not import any non-semantic
    // extended instruction sets.
    result |= test_nonsemantic_helper(device, context, queue,
                                      "extinstimport_nonsemantic_none");

    // Test with a SPIR-V module that declares the SPV_KHR_non_semantic_info and
    // imports a few unknown non-semantic extended instruction sets.
    result |= test_nonsemantic_helper(device, context, queue,
                                      "extinstimport_nonsemantic_unknown");

    return result;
}
