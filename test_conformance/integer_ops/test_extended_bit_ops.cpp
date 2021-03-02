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
#include "harness/testHarness.h"
#include "harness/conversions.h"

// TODO: Move this to a common location?
template <typename T> struct TestInfo
{
};
template <> struct TestInfo<cl_char>
{
    static const ExplicitType explicitType = kChar;
};
template <> struct TestInfo<cl_uchar>
{
    static const ExplicitType explicitType = kUChar;
};
template <> struct TestInfo<cl_short>
{
    static const ExplicitType explicitType = kShort;
};
template <> struct TestInfo<cl_ushort>
{
    static const ExplicitType explicitType = kUShort;
};
template <> struct TestInfo<cl_int>
{
    static const ExplicitType explicitType = kInt;
};
template <> struct TestInfo<cl_uint>
{
    static const ExplicitType explicitType = kUInt;
};
template <> struct TestInfo<cl_long>
{
    static const ExplicitType explicitType = kLong;
};
template <> struct TestInfo<cl_ulong>
{
    static const ExplicitType explicitType = kULong;
};

template <typename T>
static void generate_input(std::vector<T>& base)
{
    // TODO: Should we generate the random data once and reuse it?
    MTdata d = init_genrand(gRandomSeed);
    generate_random_data(TestInfo<T>::explicitType, base.size(), d, base.data());
    free_mtdata(d);
    d = NULL;
}

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
static void calculate_reference(std::vector<T>& ref, const std::vector<T>& base, const std::vector<T>& insert)
{
    ref.resize(base.size());
    for (size_t i = 0; i < base.size(); i++) {
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
static int test_vectype(const char* type_name, cl_device_id device,
                        cl_context context, cl_command_queue queue)
{
    cl_int error = CL_SUCCESS;
    int result = TEST_PASS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string buildOptions{ "-DTYPE=" };
    buildOptions += type_name;
    if (N > 1)
    {
        buildOptions += std::to_string(N);
    }
    buildOptions += " -DBASETYPE=";
    buildOptions += type_name;
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
    error = create_single_kernel_helper(
        context, &program, &kernel, 1, &source, "test_bitfield_insert",
        buildOptions.c_str());
    test_error(error, "Unable to create test_bitfield_insert kernel");

    clMemWrapper dst;
    clMemWrapper src_base;
    clMemWrapper src_insert;

    dst =
        clCreateBuffer(context, 0, reference.size() * sizeof(T), NULL, &error);
    test_error(error, "Unable to create output buffer");

    src_base = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              base.size() * sizeof(T), base.data(), &error);
    test_error(error, "Unable to create base buffer");

    src_insert =
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
#if 1
        for (size_t i = 0; i < reference.size(); i++) {
            if (results[i] != reference[i]) {
                cl_uint offset = (i / N) / (sizeof(T) * 8 + 1);
                cl_uint count = (i / N) % (sizeof(T) * 8 + 1);
                if (offset + count > sizeof(T) * 8)
                {
                    count = (sizeof(T) * 8) - offset;
                }
                printf("At index %zu: wanted %llX, got %llX: base = %llX, insert = %llX, offset = %u, count = %u.\n", i, reference[i], results[i], base[i], insert[i], offset, count);
            }
        }
#endif
        log_error("Result buffer did not match reference buffer!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename T>
static int test_type(const char* type_name, cl_device_id device,
                     cl_context context, cl_command_queue queue)
{
    log_info("    testing type %s\n", type_name);

    return test_vectype<T, 1>(type_name, device, context, queue)
        | test_vectype<T, 2>(type_name, device, context, queue)
        | test_vectype<T, 3>(type_name, device, context, queue)
        | test_vectype<T, 4>(type_name, device, context, queue)
        | test_vectype<T, 8>(type_name, device, context, queue)
        | test_vectype<T, 16>(type_name, device, context, queue);
}

int test_extended_bit_ops_insert(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    // TODO: add back this check!
    if (true || is_extension_available(device, "cl_khr_extended_bit_ops"))
    {
        int result = TEST_PASS;

        //result |= test_type<cl_char>("char", device, context, queue);
        result |= test_type<cl_uchar>("uchar", device, context, queue);
        //result |= test_type<cl_short>("short", device, context, queue);
        result |= test_type<cl_ushort>("ushort", device, context, queue);
        //result |= test_type<cl_int>("int", device, context, queue);
        result |= test_type<cl_uint>("uint", device, context, queue);
        if (gHasLong)
        {
        //    result |= test_type<cl_long>("long", device, context, queue);
            result |= test_type<cl_ulong>("ulong", device, context, queue);
        }
        return result;
    }

    log_info("cl_khr_extended_bit_ops is not supported\n");
    return TEST_SKIPPED_ITSELF;
}

int test_extended_bit_ops_extract(cl_device_id device, cl_context context,
                                  cl_command_queue queue, int num_elements)
{
    if (is_extension_available(device, "cl_khr_extended_bit_ops"))
    {
        int result = TEST_PASS;

#if 0
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
#endif
        return result;
    }

    log_info("cl_khr_extended_bit_ops is not supported\n");
    return TEST_SKIPPED_ITSELF;
}
