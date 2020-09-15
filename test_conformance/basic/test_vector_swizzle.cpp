//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include <iostream> // deleteme
#include <numeric>
#include <string>
#include <vector>

#include "procs.h"
#include "harness/testHarness.h"

template<int N>
struct TestInfo
{
    static const size_t vector_size = N;

    static constexpr const char* kernel_source_xyzw = "";
    static constexpr const char* kernel_source_rgba = "";
    static constexpr const char* kernel_source_sN = "";
};

template <>
struct TestInfo<2>
{
    static const size_t vector_size = 2;

    static constexpr const char* kernel_source_xyzw = R"CLC(
__kernel void test_vector_swizzle_xyzw(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].x = value.x;
    dst[index++].y = value.x;
    dst[index++].xy = value;
    dst[index++].yx = value;

    // rvalue swizzles
    dst[index++] = value.xx;
    dst[index++] = value.yy;
    dst[index++] = value.xy;
    dst[index++] = value.yx;
}
)CLC";

    static constexpr const char* kernel_source_rgba = R"CLC(
__kernel void test_vector_swizzle_rgba(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].r = value.r;
    dst[index++].g = value.r;
    dst[index++].rg = value;
    dst[index++].gr = value;

    // rvalue swizzles
    dst[index++] = value.rr;
    dst[index++] = value.gg;
    dst[index++] = value.rg;
    dst[index++] = value.gr;
}
)CLC";

    static constexpr const char* kernel_source_sN = R"CLC(
__kernel void test_vector_swizzle_sN(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].s0 = value.s0;
    dst[index++].s1 = value.s0;
    dst[index++].s01 = value;
    dst[index++].s10 = value;

    // rvalue swizzles
    dst[index++] = value.s00;
    dst[index++] = value.s11;
    dst[index++] = value.s01;
    dst[index++] = value.s10;
}
)CLC";
};

template <>
struct TestInfo<4>
{
    static const size_t vector_size = 4;

    static constexpr const char* kernel_source_xyzw = R"CLC(
__kernel void test_vector_swizzle_xyzw(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].x = value.x;
    dst[index++].y = value.x;
    dst[index++].z = value.x;
    dst[index++].w = value.x;
    dst[index++].xyzw = value;
    dst[index++].wzyx = value;

    // rvalue swizzles
    dst[index++] = value.xxxx;
    dst[index++] = value.yyyy;
    dst[index++] = value.zzzz;
    dst[index++] = value.wwww;
    dst[index++] = value.xyzw;
    dst[index++] = value.wzyx;
}
)CLC";

    static constexpr const char* kernel_source_rgba = R"CLC(
__kernel void test_vector_swizzle_rgba(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].r = value.r;
    dst[index++].g = value.r;
    dst[index++].b = value.r;
    dst[index++].a = value.r;
    dst[index++].rgba = value;
    dst[index++].abgr = value;

    // rvalue swizzles
    dst[index++] = value.rrrr;
    dst[index++] = value.gggg;
    dst[index++] = value.bbbb;
    dst[index++] = value.aaaa;
    dst[index++] = value.rgba;
    dst[index++] = value.abgr;
}
)CLC";

    static constexpr const char* kernel_source_sN = R"CLC(
__kernel void test_vector_swizzle_sN(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    dst[index++].s0 = value.s0;
    dst[index++].s1 = value.s0;
    dst[index++].s2 = value.s0;
    dst[index++].s3 = value.s0;
    dst[index++].s0123 = value;
    dst[index++].s3210 = value;

    // rvalue swizzles
    dst[index++] = value.s0000;
    dst[index++] = value.s1111;
    dst[index++] = value.s2222;
    dst[index++] = value.s3333;
    dst[index++] = value.s0123;
    dst[index++] = value.s3210;
}
)CLC";
};

template <typename T, size_t N, size_t S>
void makeReference(std::vector<T>& ref)
{
    // N single channel lvalue tests
    // 2 multi-value lvalue tests
    // N single channel rvalue tests
    // 2 multi-value rvalue tests
    const size_t refSize = (N + 2 + N + 2) * S;

    ref.resize(refSize);
    std::fill(ref.begin(), ref.end(), 99);

    size_t dstIndex = 0;

    // single channel lvalue
    for (size_t i = 0; i < N; i++) {
        ref[dstIndex * S + i] = 0;
        ++dstIndex;
    }

    // normal lvalue
    for (size_t c = 0; c < N; c++) {
        ref[dstIndex * S + c] = c;
    }
    ++dstIndex;

    // reverse lvalue
    for (size_t c = 0; c < N; c++) {
        ref[dstIndex * S + c] = N - c - 1;
    }
    ++dstIndex;

    // single channel rvalue
    for (size_t i = 0; i < N; i++) {
        for (size_t c = 0; c < N; c++) {
            ref[dstIndex * S + c] = i;
        }
        ++dstIndex;
    }

    // normal rvalue
    for (size_t c = 0; c < N; c++) {
        ref[dstIndex * S + c] = c;
    }
    ++dstIndex;

    // reverse rvalue
    for (size_t c = 0; c < N; c++) {
        ref[dstIndex * S + c] = N - c - 1;
    }
    ++dstIndex;

    assert(dstIndex * S == refSize);
}

template <typename T>
int test_vector_swizzle_vec_case(const std::vector<T>& value, const std::vector<T>& reference, cl_context context, cl_kernel kernel, cl_command_queue queue)
{
    cl_int error = CL_SUCCESS;

    clMemWrapper mem;

    std::vector<T> buffer(reference.size(), 99);
    mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, buffer.size() * sizeof(T), buffer.data(), &error);
    test_error(error, "Unable to create test buffer");

    error = clSetKernelArg(kernel, 0, value.size() * sizeof(T), value.data());
    test_error(error, "Unable to set value kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(mem), &mem);
    test_error(error, "Unable to set destination buffer kernel arg");

    size_t global_work_size[] = { 1 };
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue test kernel");

    error = clFinish(queue);
    test_error(error, "clFinish failed after test kernel");

    error = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, buffer.size() * sizeof(T), buffer.data(), 0, NULL, NULL);
    test_error(error, "Unable to read data after test kernel");

    if (buffer != reference) {
        log_error("Result buffer did not match reference buffer!\n");
        log_error("buffer size = %d, reference size = %d\n", buffer.size(), reference.size());
        for (int i = 0; i < buffer.size(); i++) {
            std::cout << i << ": buffer = " << buffer[i] << "  reference = " << reference[i] << std::endl;
        }
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T, size_t N>
int test_vector_swizzle_vectype(cl_device_id device, cl_context context, cl_command_queue queue, const char* type_name)
{
    cl_int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string buildOptions{"-DTYPE="};
    buildOptions += type_name;
    buildOptions += std::to_string(N);

    std::vector<T> value(N);
    std::iota(value.begin(), value.end(), 0);

    std::vector<T> reference;
    constexpr size_t S = TestInfo<N>::vector_size;
    makeReference<T, N, S>(reference);

    if (N <= 4) {
        // Test XYZW swizzles for vector sizes less than four:
        const char* xyzw_source = TestInfo<N>::kernel_source_xyzw;
        error = create_single_kernel_helper(
            context, &program, &kernel, 1, &xyzw_source,
            "test_vector_swizzle_xyzw", buildOptions.c_str());
        test_error(error, "Unable to create xyzw test kernel");

        error = test_vector_swizzle_vec_case(value, reference, context, kernel, queue);

        // Test RGBA swizzles for OpenCL 3.0 and newer:
        const Version device_version = get_device_cl_version(device);
        if (false && device_version >= Version(3, 0)) {
            const char* rgba_source =TestInfo<N>::kernel_source_rgba;
            error = create_single_kernel_helper(
                context, &program, &kernel, 1, &xyzw_source,
                "test_vector_swizzle_rgba", buildOptions.c_str());
            test_error(error, "Unable to create rgba test kernel");

            error = test_vector_swizzle_vec_case(value, reference, context, kernel, queue);
        }
    }

    // Test sN swizzles:
    const char* sN_source =TestInfo<N>::kernel_source_sN;
    error = create_single_kernel_helper(
        context, &program, &kernel, 1, &sN_source,
        "test_vector_swizzle_sN", buildOptions.c_str());
    test_error(error, "Unable to create sN test kernel");

    error = test_vector_swizzle_vec_case(value, reference, context, kernel, queue);

    return TEST_PASS;
}

template <typename T>
int test_vector_swizzle_type(cl_device_id device, cl_context context, cl_command_queue queue, const char* type_name)
{
    return test_vector_swizzle_vectype<T, 2>(device, context, queue, type_name) |
        test_vector_swizzle_vectype<T, 4>(device, context, queue, type_name);
}

int test_vector_swizzle_int(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_vector_swizzle_type<cl_int>(device, context, queue, "int");
}

int test_vector_swizzle(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int result = TEST_PASS;
    result |= test_vector_swizzle_int(device, context, queue, num_elements);
    return result;
}