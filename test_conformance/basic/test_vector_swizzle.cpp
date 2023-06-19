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
#include <numeric>
#include <string>
#include <vector>

#include "procs.h"
#include "harness/testHarness.h"

static std::string pragma_extension;

template <int N> struct TestInfo
{
};

template <> struct TestInfo<2>
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
    dst[index++] = value.x;
    dst[index++] = value.y;
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
    dst[index++] = value.r;
    dst[index++] = value.g;
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
    dst[index++] = value.s0;
    dst[index++] = value.s1;
    dst[index++] = value.s01;
    dst[index++] = value.s10;
}
)CLC";
};

template <> struct TestInfo<3>
{
    static const size_t vector_size = 4; // sizeof(vec3) is four elements

    static constexpr const char* kernel_source_xyzw = R"CLC(
__kernel void test_vector_swizzle_xyzw(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    TYPE t;
    t = dst[index]; t.x = value.x;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.y = value.x;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.z = value.x;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.xyz = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.zyx = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));

    // rvalue swizzles
    vstore3(value.x, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.y, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.z, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.xyz, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.zyx, 0, (__global BASETYPE*)(dst + index++));
}
)CLC";

    static constexpr const char* kernel_source_rgba = R"CLC(
__kernel void test_vector_swizzle_rgba(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    TYPE t;
    t = dst[index]; t.r = value.r;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.g = value.r;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.b = value.r;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.rgb = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.bgr = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));

    // rvalue swizzles
    vstore3(value.r, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.g, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.b, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.rgb, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.bgr, 0, (__global BASETYPE*)(dst + index++));
}
)CLC";

    static constexpr const char* kernel_source_sN = R"CLC(
__kernel void test_vector_swizzle_sN(TYPE value, __global TYPE* dst) {
    int index = 0;

    // lvalue swizzles
    TYPE t;
    t = dst[index]; t.s0 = value.s0;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.s1 = value.s0;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.s2 = value.s0;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.s012 = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));
    t = dst[index]; t.s210 = value;
    vstore3(t, 0, (__global BASETYPE*)(dst + index++));

    // rvalue swizzles
    vstore3(value.s0, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.s1, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.s2, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.s012, 0, (__global BASETYPE*)(dst + index++));
    vstore3(value.s210, 0, (__global BASETYPE*)(dst + index++));
}
)CLC";
};

template <> struct TestInfo<4>
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
    dst[index++] = value.x;
    dst[index++] = value.y;
    dst[index++] = value.z;
    dst[index++] = value.w;
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
    dst[index++] = value.r;
    dst[index++] = value.g;
    dst[index++] = value.b;
    dst[index++] = value.a;
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
    dst[index++] = value.s0;
    dst[index++] = value.s1;
    dst[index++] = value.s2;
    dst[index++] = value.s3;
    dst[index++] = value.s0123;
    dst[index++] = value.s3210;
}
)CLC";
};

template <> struct TestInfo<8>
{
    static const size_t vector_size = 8;

    static constexpr const char* kernel_source_xyzw = R"CLC(
__kernel void test_vector_swizzle_xyzw(TYPE value, __global TYPE* dst) {
    int index = 0;

    // xwzw only for first four components!

    // lvalue swizzles
    dst[index++].x = value.x;
    dst[index++].y = value.x;
    dst[index++].z = value.x;
    dst[index++].w = value.x;
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index].xyzw = value.s0123;
    dst[index++].s4567 = value.s4567;
    dst[index].s7654 = value.s0123;
    dst[index++].wzyx = value.s4567;

    // rvalue swizzles
    dst[index++] = value.x;
    dst[index++] = value.y;
    dst[index++] = value.z;
    dst[index++] = value.w;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = (TYPE)(value.xyzw, value.s4567);
    dst[index++] = (TYPE)(value.s7654, value.wzyx);
}
)CLC";
    static constexpr const char* kernel_source_rgba = R"CLC(
__kernel void test_vector_swizzle_rgba(TYPE value, __global TYPE* dst) {
    int index = 0;

    // rgba only for first four components!

    // lvalue swizzles
    dst[index++].r = value.r;
    dst[index++].g = value.r;
    dst[index++].b = value.r;
    dst[index++].a = value.r;
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index].rgba = value.s0123;
    dst[index++].s4567 = value.s4567;
    dst[index].s7654 = value.s0123;
    dst[index++].abgr = value.s4567;

    // rvalue swizzles
    dst[index++] = value.r;
    dst[index++] = value.g;
    dst[index++] = value.b;
    dst[index++] = value.a;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = (TYPE)(value.rgba, value.s4567);
    dst[index++] = (TYPE)(value.s7654, value.abgr);
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
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index++].s01234567 = value;
    dst[index++].s76543210 = value;

    // rvalue swizzles
    dst[index++] = value.s0;
    dst[index++] = value.s1;
    dst[index++] = value.s2;
    dst[index++] = value.s3;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = value.s01234567;
    dst[index++] = value.s76543210;
}
)CLC";
};

template <> struct TestInfo<16>
{
    static const size_t vector_size = 16;

    static constexpr const char* kernel_source_xyzw = R"CLC(
__kernel void test_vector_swizzle_xyzw(TYPE value, __global TYPE* dst) {
    int index = 0;

    // xwzw only for first four components!

    // lvalue swizzles
    dst[index++].x = value.x;
    dst[index++].y = value.x;
    dst[index++].z = value.x;
    dst[index++].w = value.x;
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index++].s8 = value.s0;
    dst[index++].s9 = value.s0;
    dst[index++].sa = value.s0;
    dst[index++].sb = value.s0;
    dst[index++].sc = value.s0;
    dst[index++].sd = value.s0;
    dst[index++].se = value.s0;
    dst[index++].sf = value.s0;
    dst[index].xyzw = value.s0123;
    dst[index].s4567 = value.s4567;
    dst[index].s89ab = value.s89ab;
    dst[index++].scdef = value.scdef;
    dst[index].sfedc = value.s0123;
    dst[index].sba98 = value.s4567;
    dst[index].s7654 = value.s89ab;
    dst[index++].wzyx = value.scdef;

    // rvalue swizzles
    dst[index++] = value.x;
    dst[index++] = value.y;
    dst[index++] = value.z;
    dst[index++] = value.w;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = value.s8;
    dst[index++] = value.s9;
    dst[index++] = value.sa;
    dst[index++] = value.sb;
    dst[index++] = value.sc;
    dst[index++] = value.sd;
    dst[index++] = value.se;
    dst[index++] = value.sf;
    dst[index++] = (TYPE)(value.xyzw, value.s4567, value.s89abcdef);
    dst[index++] = (TYPE)(value.sfedcba98, value.s7654, value.wzyx);
}
)CLC";
    static constexpr const char* kernel_source_rgba = R"CLC(
__kernel void test_vector_swizzle_rgba(TYPE value, __global TYPE* dst) {
    int index = 0;

    // rgba only for first four components!

    // lvalue swizzles
    dst[index++].r = value.r;
    dst[index++].g = value.r;
    dst[index++].b = value.r;
    dst[index++].a = value.r;
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index++].s8 = value.s0;
    dst[index++].s9 = value.s0;
    dst[index++].sa = value.s0;
    dst[index++].sb = value.s0;
    dst[index++].sc = value.s0;
    dst[index++].sd = value.s0;
    dst[index++].se = value.s0;
    dst[index++].sf = value.s0;
    dst[index].rgba = value.s0123;
    dst[index].s4567 = value.s4567;
    dst[index].s89ab = value.s89ab;
    dst[index++].scdef = value.scdef;
    dst[index].sfedc = value.s0123;
    dst[index].sba98 = value.s4567;
    dst[index].s7654 = value.s89ab;
    dst[index++].abgr = value.scdef;

    // rvalue swizzles
    dst[index++] = value.r;
    dst[index++] = value.g;
    dst[index++] = value.b;
    dst[index++] = value.a;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = value.s8;
    dst[index++] = value.s9;
    dst[index++] = value.sa;
    dst[index++] = value.sb;
    dst[index++] = value.sc;
    dst[index++] = value.sd;
    dst[index++] = value.se;
    dst[index++] = value.sf;
    dst[index++] = (TYPE)(value.rgba, value.s4567, value.s89abcdef);
    dst[index++] = (TYPE)(value.sfedcba98, value.s7654, value.abgr);
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
    dst[index++].s4 = value.s0;
    dst[index++].s5 = value.s0;
    dst[index++].s6 = value.s0;
    dst[index++].s7 = value.s0;
    dst[index++].s8 = value.s0;
    dst[index++].s9 = value.s0;
    dst[index++].sa = value.s0;
    dst[index++].sb = value.s0;
    dst[index++].sc = value.s0;
    dst[index++].sd = value.s0;
    dst[index++].se = value.s0;
    dst[index++].sf = value.s0;
    dst[index++].s0123456789abcdef = value; // lower-case
    dst[index++].sFEDCBA9876543210 = value; // upper-case

    // rvalue swizzles
    dst[index++] = value.s0;
    dst[index++] = value.s1;
    dst[index++] = value.s2;
    dst[index++] = value.s3;
    dst[index++] = value.s4;
    dst[index++] = value.s5;
    dst[index++] = value.s6;
    dst[index++] = value.s7;
    dst[index++] = value.s8;
    dst[index++] = value.s9;
    dst[index++] = value.sa;
    dst[index++] = value.sb;
    dst[index++] = value.sc;
    dst[index++] = value.sd;
    dst[index++] = value.se;
    dst[index++] = value.sf;
    dst[index++] = value.s0123456789abcdef; // lower-case
    dst[index++] = value.sFEDCBA9876543210; // upper-case
}
)CLC";
};

template <typename T, size_t N, size_t S>
static void makeReference(std::vector<T>& ref)
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
    for (size_t i = 0; i < N; i++)
    {
        ref[dstIndex * S + i] = 0;
        ++dstIndex;
    }

    // normal lvalue
    for (size_t c = 0; c < N; c++)
    {
        ref[dstIndex * S + c] = c;
    }
    ++dstIndex;

    // reverse lvalue
    for (size_t c = 0; c < N; c++)
    {
        ref[dstIndex * S + c] = N - c - 1;
    }
    ++dstIndex;

    // single channel rvalue
    for (size_t i = 0; i < N; i++)
    {
        for (size_t c = 0; c < N; c++)
        {
            ref[dstIndex * S + c] = i;
        }
        ++dstIndex;
    }

    // normal rvalue
    for (size_t c = 0; c < N; c++)
    {
        ref[dstIndex * S + c] = c;
    }
    ++dstIndex;

    // reverse rvalue
    for (size_t c = 0; c < N; c++)
    {
        ref[dstIndex * S + c] = N - c - 1;
    }
    ++dstIndex;

    assert(dstIndex * S == refSize);
}

template <typename T>
static int
test_vectype_case(const std::vector<T>& value, const std::vector<T>& reference,
                  cl_context context, cl_kernel kernel, cl_command_queue queue)
{
    cl_int error = CL_SUCCESS;

    clMemWrapper mem;

    std::vector<T> buffer(reference.size(), 99);
    mem = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                         buffer.size() * sizeof(T), buffer.data(), &error);
    test_error(error, "Unable to create test buffer");

    error = clSetKernelArg(kernel, 0, value.size() * sizeof(T), value.data());
    test_error(error, "Unable to set value kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(mem), &mem);
    test_error(error, "Unable to set destination buffer kernel arg");

    size_t global_work_size[] = { 1 };
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                   NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue test kernel");

    error = clFinish(queue);
    test_error(error, "clFinish failed after test kernel");

    error =
        clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, buffer.size() * sizeof(T),
                            buffer.data(), 0, NULL, NULL);
    test_error(error, "Unable to read data after test kernel");

    if (buffer != reference)
    {
        log_error("Result buffer did not match reference buffer!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T, size_t N>
static int test_vectype(const char* type_name, cl_device_id device,
                        cl_context context, cl_command_queue queue)
{
    log_info("    testing type %s%d\n", type_name, N);

    cl_int error = CL_SUCCESS;
    int result = TEST_PASS;

    std::string buildOptions{ "-DTYPE=" };
    buildOptions += type_name;
    buildOptions += std::to_string(N);
    buildOptions += " -DBASETYPE=";
    buildOptions += type_name;

    constexpr size_t S = TestInfo<N>::vector_size;

    std::vector<T> value(S);
    std::iota(value.begin(), value.end(), 0);

    std::vector<T> reference;
    makeReference<T, N, S>(reference);

    // XYZW swizzles:
    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        std::string program_src =
            pragma_extension + std::string(TestInfo<N>::kernel_source_xyzw);
        const char* xyzw_source = program_src.c_str();
        error = create_single_kernel_helper(
            context, &program, &kernel, 1, &xyzw_source,
            "test_vector_swizzle_xyzw", buildOptions.c_str());
        test_error(error, "Unable to create xyzw test kernel");

        result |= test_vectype_case(value, reference, context, kernel, queue);
    }

    // sN swizzles:
    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        std::string program_src =
            pragma_extension + std::string(TestInfo<N>::kernel_source_sN);
        const char* sN_source = program_src.c_str();
        error = create_single_kernel_helper(
            context, &program, &kernel, 1, &sN_source, "test_vector_swizzle_sN",
            buildOptions.c_str());
        test_error(error, "Unable to create sN test kernel");

        result |= test_vectype_case(value, reference, context, kernel, queue);
    }

    // RGBA swizzles for OpenCL 3.0 and newer:
    {
        clProgramWrapper program;
        clKernelWrapper kernel;

        const Version device_version = get_device_cl_version(device);
        if (device_version >= Version(3, 0))
        {
            std::string program_src =
                pragma_extension + std::string(TestInfo<N>::kernel_source_rgba);
            const char* rgba_source = program_src.c_str();
            error = create_single_kernel_helper(
                context, &program, &kernel, 1, &rgba_source,
                "test_vector_swizzle_rgba", buildOptions.c_str());
            test_error(error, "Unable to create rgba test kernel");

            result |=
                test_vectype_case(value, reference, context, kernel, queue);
        }
    }

    return result;
}

template <typename T>
static int test_type(const char* type_name, cl_device_id device,
                     cl_context context, cl_command_queue queue)
{
    return test_vectype<T, 2>(type_name, device, context, queue)
        | test_vectype<T, 3>(type_name, device, context, queue)
        | test_vectype<T, 4>(type_name, device, context, queue)
        | test_vectype<T, 8>(type_name, device, context, queue)
        | test_vectype<T, 16>(type_name, device, context, queue);
}

int test_vector_swizzle(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    int hasDouble = is_extension_available(device, "cl_khr_fp64");
    int hasHalf = is_extension_available(device, "cl_khr_fp16");

    int result = TEST_PASS;
    result |= test_type<cl_char>("char", device, context, queue);
    result |= test_type<cl_uchar>("uchar", device, context, queue);
    result |= test_type<cl_short>("short", device, context, queue);
    result |= test_type<cl_ushort>("ushort", device, context, queue);
    result |= test_type<cl_int>("int", device, context, queue);
    result |= test_type<cl_uint>("uint", device, context, queue);
    if (gHasLong)
    {
        result |= test_type<cl_long>("long", device, context, queue);
        result |= test_type<cl_ulong>("ulong", device, context, queue);
    }
    result |= test_type<cl_float>("float", device, context, queue);
    if (hasHalf)
    {
        pragma_extension = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        result |= test_type<cl_half>("half", device, context, queue);
    }
    if (hasDouble)
    {
        pragma_extension = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        result |= test_type<cl_double>("double", device, context, queue);
    }
    return result;
}
