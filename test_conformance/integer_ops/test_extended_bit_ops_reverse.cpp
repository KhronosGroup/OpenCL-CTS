//
// Copyright (c) 2022 The Khronos Group Inc.
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

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "procs.h"
#include "harness/integer_ops_test_info.h"
#include "harness/testHarness.h"

template <typename T> static T cpu_bit_reverse(T base)
{
    T result = 0;

    const size_t count = sizeof(T) * 8;
    for (size_t i = 0; i < count; i++)
    {
        if (base & ((T)1 << i))
        {
            result |= ((T)1 << (count - i - 1));
        }
    }
    return result;
}

template <typename T>
static void calculate_reference(std::vector<T>& ref, const std::vector<T>& base)
{
    ref.resize(base.size());
    for (size_t i = 0; i < base.size(); i++)
    {
        ref[i] = cpu_bit_reverse(base[i]);
    }
}

static constexpr const char* kernel_source = R"CLC(
__kernel void test_bit_reverse(__global TYPE* dst, __global TYPE* base)
{
    int index = get_global_id(0);
    dst[index] = bit_reverse(base[index]);
}
)CLC";

static constexpr const char* kernel_source_vec3 = R"CLC(
__kernel void test_bit_reverse(__global BASETYPE* dst, __global BASETYPE* base)
{
    int index = get_global_id(0);
    TYPE s = vload3(index, base);
    TYPE d = bit_reverse(s);
    vstore3(d, index, dst);
}
)CLC";

template <typename T, size_t N>
static int test_vectype(cl_device_id device, cl_context context,
                        cl_command_queue queue)
{
    cl_int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string buildOptions{ "-DTYPE=" };
    buildOptions += TestInfo<T>::deviceTypeName;
    if (N > 1)
    {
        buildOptions += std::to_string(N);
    }
    buildOptions += " -DBASETYPE=";
    buildOptions += TestInfo<T>::deviceTypeName;

    const size_t ELEMENTS_TO_TEST = 65536;
    std::vector<T> base(ELEMENTS_TO_TEST * N);
    fill_vector_with_random_data(base);

    std::vector<T> reference;
    calculate_reference(reference, base);

    const char* source = (N == 3) ? kernel_source_vec3 : kernel_source;
    error =
        create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                    "test_bit_reverse", buildOptions.c_str());
    test_error(error, "Unable to create test_bit_reverse kernel");

    clMemWrapper src;
    clMemWrapper dst;

    dst =
        clCreateBuffer(context, 0, reference.size() * sizeof(T), NULL, &error);
    test_error(error, "Unable to create output buffer");

    src = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, base.size() * sizeof(T),
                         base.data(), &error);
    test_error(error, "Unable to create base buffer");

    error = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    test_error(error, "Unable to set output buffer kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(src), &src);
    test_error(error, "Unable to set base buffer kernel arg");

    size_t global_work_size[] = { reference.size() / N };
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                   NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue test kernel");

    error = clFinish(queue);
    test_error(error, "clFinish failed after test kernel");

    std::vector<T> results(reference.size(), 99);
    error =
        clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, results.size() * sizeof(T),
                            results.data(), 0, NULL, NULL);
    test_error(error, "Unable to read data after test kernel");

    if (results != reference)
    {
        log_error("Result buffer did not match reference buffer!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
static int test_type(cl_device_id device, cl_context context,
                     cl_command_queue queue)
{
    log_info("    testing type %s\n", TestInfo<T>::deviceTypeName);

    return test_vectype<T, 1>(device, context, queue)
        | test_vectype<T, 2>(device, context, queue)
        | test_vectype<T, 3>(device, context, queue)
        | test_vectype<T, 4>(device, context, queue)
        | test_vectype<T, 8>(device, context, queue)
        | test_vectype<T, 16>(device, context, queue);
}

int test_extended_bit_ops_reverse(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    if (is_extension_available(device, "cl_khr_extended_bit_ops"))
    {
        int result = TEST_PASS;

        result |= test_type<cl_char>(device, context, queue);
        result |= test_type<cl_uchar>(device, context, queue);
        result |= test_type<cl_short>(device, context, queue);
        result |= test_type<cl_ushort>(device, context, queue);
        result |= test_type<cl_int>(device, context, queue);
        result |= test_type<cl_uint>(device, context, queue);
        if (gHasLong)
        {
            result |= test_type<cl_long>(device, context, queue);
            result |= test_type<cl_ulong>(device, context, queue);
        }
        return result;
    }

    log_info("cl_khr_extended_bit_ops is not supported\n");
    return TEST_SKIPPED_ITSELF;
}
