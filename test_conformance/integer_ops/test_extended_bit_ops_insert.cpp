//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "extended_bit_ops.h"
#include "harness/testHarness.h"

template <typename T>
static T cpu_bit_insert(T tbase, T tinsert, cl_uint offset, cl_uint count)
{
    assert(offset <= sizeof(T) * 8);
    assert(count <= sizeof(T) * 8);
    assert(offset + count <= sizeof(T) * 8);

    cl_ulong base = static_cast<cl_ulong>(tbase);
    cl_ulong insert = static_cast<cl_ulong>(tinsert);

    cl_ulong mask = (count < 64) ? ((1ULL << count) - 1) << offset : ~0ULL;
    cl_ulong result = ((insert << offset) & mask) | (base & ~mask);

    return static_cast<T>(result);
}

template <typename T, size_t N>
static void calculate_reference(std::vector<T>& ref, const std::vector<T>& base,
                                const std::vector<T>& insert)
{
    ref.resize(base.size());
    for (size_t i = 0; i < base.size(); i++)
    {
        cl_uint offset = (i / N) / (sizeof(T) * 8 + 1);
        cl_uint count = (i / N) % (sizeof(T) * 8 + 1);
        if (offset + count > sizeof(T) * 8)
        {
            count = (sizeof(T) * 8) - offset;
        }
        ref[i] = cpu_bit_insert(base[i], insert[i], offset, count);
    }
}

static constexpr const char* kernel_source = R"CLC(
#define OVLD __attribute__((overloadable))

char OVLD intel_bfi(char base, char insert, uint offset, uint count) { return as_char(intel_bfi(as_uchar(base), as_uchar(insert), offset, count)); }
char2 OVLD intel_bfi(char2 base, char2 insert, uint offset, uint count) { return as_char2(intel_bfi(as_uchar2(base), as_uchar2(insert), offset, count)); }
char4 OVLD intel_bfi(char4 base, char4 insert, uint offset, uint count) { return as_char4(intel_bfi(as_uchar4(base), as_uchar4(insert), offset, count)); }
char8 OVLD intel_bfi(char8 base, char8 insert, uint offset, uint count) { return as_char8(intel_bfi(as_uchar8(base), as_uchar8(insert), offset, count)); }
char16 OVLD intel_bfi(char16 base, char16 insert, uint offset, uint count) { return as_char16(intel_bfi(as_uchar16(base), as_uchar16(insert), offset, count)); }

short OVLD intel_bfi(short base, short insert, uint offset, uint count) { return as_short(intel_bfi(as_ushort(base), as_ushort(insert), offset, count)); }
short2 OVLD intel_bfi(short2 base, short2 insert, uint offset, uint count) { return as_short2(intel_bfi(as_ushort2(base), as_ushort2(insert), offset, count)); }
short4 OVLD intel_bfi(short4 base, short4 insert, uint offset, uint count) { return as_short4(intel_bfi(as_ushort4(base), as_ushort4(insert), offset, count)); }
short8 OVLD intel_bfi(short8 base, short8 insert, uint offset, uint count) { return as_short8(intel_bfi(as_ushort8(base), as_ushort8(insert), offset, count)); }
short16 OVLD intel_bfi(short16 base, short16 insert, uint offset, uint count) { return as_short16(intel_bfi(as_ushort16(base), as_ushort16(insert), offset, count)); }

int OVLD intel_bfi(int base, int insert, uint offset, uint count) { return as_int(intel_bfi(as_uint(base), as_uint(insert), offset, count)); }
int2 OVLD intel_bfi(int2 base, int2 insert, uint offset, uint count) { return as_int2(intel_bfi(as_uint2(base), as_uint2(insert), offset, count)); }
int4 OVLD intel_bfi(int4 base, int4 insert, uint offset, uint count) { return as_int4(intel_bfi(as_uint4(base), as_uint4(insert), offset, count)); }
int8 OVLD intel_bfi(int8 base, int8 insert, uint offset, uint count) { return as_int8(intel_bfi(as_uint8(base), as_uint8(insert), offset, count)); }
int16 OVLD intel_bfi(int16 base, int16 insert, uint offset, uint count) { return as_int16(intel_bfi(as_uint16(base), as_uint16(insert), offset, count)); }

long OVLD intel_bfi(long base, long insert, uint offset, uint count) { return as_long(intel_bfi(as_ulong(base), as_ulong(insert), offset, count)); }
long2 OVLD intel_bfi(long2 base, long2 insert, uint offset, uint count) { return as_long2(intel_bfi(as_ulong2(base), as_ulong2(insert), offset, count)); }
long4 OVLD intel_bfi(long4 base, long4 insert, uint offset, uint count) { return as_long4(intel_bfi(as_ulong4(base), as_ulong4(insert), offset, count)); }
long8 OVLD intel_bfi(long8 base, long8 insert, uint offset, uint count) { return as_long8(intel_bfi(as_ulong8(base), as_ulong8(insert), offset, count)); }
long16 OVLD intel_bfi(long16 base, long16 insert, uint offset, uint count) { return as_long16(intel_bfi(as_ulong16(base), as_ulong16(insert), offset, count)); }

__kernel void test_bitfield_insert(__global TYPE* dst, __global TYPE* base, __global TYPE* insert)
{
    int index = get_global_id(0);
    uint offset = index / (sizeof(BASETYPE) * 8 + 1);
    uint count = index % (sizeof(BASETYPE) * 8 + 1);
    if (offset + count > sizeof(BASETYPE) * 8) {
        count = (sizeof(BASETYPE) * 8) - offset;
    }
    dst[index] = bitfield_insert(base[index], insert[index], offset, count);
}
)CLC";

static constexpr const char* kernel_source_vec3 = R"CLC(
#define OVLD __attribute__((overloadable))

char3 OVLD intel_bfi(char3 base, char3 insert, uint offset, uint count) { return as_char3(intel_bfi(as_uchar3(base), as_uchar3(insert), offset, count)); }
short3 OVLD intel_bfi(short3 base, short3 insert, uint offset, uint count) { return as_short3(intel_bfi(as_ushort3(base), as_ushort3(insert), offset, count)); }
int3 OVLD intel_bfi(int3 base, int3 insert, uint offset, uint count) { return as_int3(intel_bfi(as_uint3(base), as_uint3(insert), offset, count)); }
long3 OVLD intel_bfi(long3 base, long3 insert, uint offset, uint count) { return as_long3(intel_bfi(as_ulong3(base), as_ulong3(insert), offset, count)); }

__kernel void test_bitfield_insert(__global BASETYPE* dst, __global BASETYPE* base, __global BASETYPE* insert)
{
    int index = get_global_id(0);
    uint offset = index / (sizeof(BASETYPE) * 8 + 1);
    uint count = index % (sizeof(BASETYPE) * 8 + 1);
    if (offset + count > sizeof(BASETYPE) * 8) {
        count = (sizeof(BASETYPE) * 8) - offset;
    }
    TYPE b = vload3(index, base);
    TYPE i = vload3(index, insert);
    TYPE d = bitfield_insert(b, i, offset, count);
    vstore3(d, index, dst);
}
)CLC";

template <typename T, size_t N>
static int test_vectype(cl_device_id device, cl_context context,
                        cl_command_queue queue)
{
    cl_int error = CL_SUCCESS;
    int result = TEST_PASS;

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
    // TEMP: delete this when we've switched names!
    buildOptions += " -Dcl_intel_bit_instructions -Dbitfield_insert=intel_bfi";

    const size_t ELEMENTS_TO_TEST = (sizeof(T) * 8 + 1) * (sizeof(T) * 8 + 1);

    std::vector<T> base(ELEMENTS_TO_TEST * N);
    std::fill(base.begin(), base.end(), static_cast<T>(0xA5A5A5A5A5A5A5A5ULL));

    std::vector<T> insert(ELEMENTS_TO_TEST * N);
    generate_input(insert);

    std::vector<T> reference;
    calculate_reference<T, N>(reference, base, insert);

    const char* source = (N == 3) ? kernel_source_vec3 : kernel_source;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                        "test_bitfield_insert",
                                        buildOptions.c_str());
    test_error(error, "Unable to create test_bitfield_insert kernel");

    clMemWrapper dst =
        clCreateBuffer(context, 0, reference.size() * sizeof(T), NULL, &error);
    test_error(error, "Unable to create output buffer");

    clMemWrapper src_base =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, base.size() * sizeof(T),
                       base.data(), &error);
    test_error(error, "Unable to create base buffer");

    clMemWrapper src_insert =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, insert.size() * sizeof(T),
                       insert.data(), &error);
    test_error(error, "Unable to create insert buffer");

    error = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    test_error(error, "Unable to set output buffer kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(src_base), &src_base);
    test_error(error, "Unable to set base buffer kernel arg");

    error = clSetKernelArg(kernel, 2, sizeof(src_insert), &src_insert);
    test_error(error, "Unable to set insert buffer kernel arg");

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

int test_extended_bit_ops_insert(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    // TODO: add back this check!
    if (true || is_extension_available(device, "cl_khr_extended_bit_ops"))
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
