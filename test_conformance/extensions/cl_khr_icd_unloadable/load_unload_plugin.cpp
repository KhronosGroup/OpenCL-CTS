// Copyright (c) 2026 The Khronos Group Inc.
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
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define PLUGIN_API __attribute__((dllexport))
#else
#define PLUGIN_API __declspec(dllexport)
#endif
#else
#if __GNUC__ >= 4
#define PLUGIN_API __attribute__((visibility("default")))
#else
#define PLUGIN_API
#endif
#endif

REGISTER_TEST(execute_kernel)
{
    REQUIRE_EXTENSION("cl_khr_icd_unloadable");

    cl_int error = CL_SUCCESS;

    const char* source = R"CLC(
        __kernel void test_kernel(__global int* dst)
        {
            size_t id = get_global_id(0);
            dst[id] = id;
        }
    )CLC";

    clProgramWrapper program;
    clKernelWrapper kernel;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                        "test_kernel");
    test_error(error, "Unable to create test kernel");

    std::array<cl_int, 4> data = { -1, -1, -1, -1 };
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(data), data.data(), &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dst);
    test_error(error, "clSetKernelArg failed");

    size_t global_work_size = data.size();
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(data),
                                data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i] != static_cast<cl_int>(i))
        {
            test_fail("Data mismatch at index %zu: expected %d, got %d", i,
                      static_cast<cl_int>(i), data[i]);
        }
    }

    return TEST_PASS;
}

#ifdef __cplusplus
extern "C" {
#endif

PLUGIN_API int do_test(int argc, const char* argv[])
{
    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}

#ifdef __cplusplus
}
#endif
