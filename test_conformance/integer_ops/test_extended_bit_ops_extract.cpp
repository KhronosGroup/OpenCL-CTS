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
#include <type_traits>
#include <vector>

#include "procs.h"
#include "harness/integer_ops_test_info.h"
#include "harness/testHarness.h"

template <typename T>
static typename std::make_unsigned<T>::type
arithmetic_shift_right(T tx, cl_uint count)
{
    typedef typename std::make_unsigned<T>::type unsigned_t;
    unsigned_t x = static_cast<unsigned_t>(tx);

    // To implement an arithmetic shift right:
    // - If the sign bit is not set, shift as usual.
    // - Otherwise, flip all of the bits, shift, then flip back.
    unsigned_t s = -(x >> (sizeof(x) * 8 - 1));
    unsigned_t result = (s ^ x) >> count ^ s;

    return result;
}

template <typename T>
static typename std::make_unsigned<T>::type
cpu_bit_extract_signed(T tbase, cl_uint offset, cl_uint count)
{
    typedef typename std::make_signed<T>::type unsigned_t;

    assert(offset <= sizeof(T) * 8);
    assert(count <= sizeof(T) * 8);
    assert(offset + count <= sizeof(T) * 8);

    unsigned_t base = static_cast<unsigned_t>(tbase);
    unsigned_t result;

    if (count == 0)
    {
        result = 0;
    }
    else
    {
        result = base << (sizeof(T) * 8 - offset - count);
        result = arithmetic_shift_right(result, sizeof(T) * 8 - count);
    }

    return result;
}

template <typename T>
static typename std::make_unsigned<T>::type
cpu_bit_extract_unsigned(T tbase, cl_uint offset, cl_uint count)
{
    typedef typename std::make_unsigned<T>::type unsigned_t;

    assert(offset <= sizeof(T) * 8);
    assert(count <= sizeof(T) * 8);
    assert(offset + count <= sizeof(T) * 8);

    unsigned_t base = static_cast<unsigned_t>(tbase);
    unsigned_t result;

    if (count == 0)
    {
        result = 0;
    }
    else
    {
        result = base << (sizeof(T) * 8 - offset - count);
        result = result >> (sizeof(T) * 8 - count);
    }

    return result;
}

template <typename T, size_t N>
static void
calculate_reference(std::vector<typename std::make_unsigned<T>::type>& sref,
                    std::vector<typename std::make_unsigned<T>::type>& uref,
                    const std::vector<T>& base)
{
    sref.resize(base.size());
    uref.resize(base.size());
    for (size_t i = 0; i < base.size(); i++)
    {
        cl_uint offset = (i / N) / (sizeof(T) * 8 + 1);
        cl_uint count = (i / N) % (sizeof(T) * 8 + 1);
        if (offset + count > sizeof(T) * 8)
        {
            count = (sizeof(T) * 8) - offset;
        }
        sref[i] = cpu_bit_extract_signed(base[i], offset, count);
        uref[i] = cpu_bit_extract_unsigned(base[i], offset, count);
    }
}

static constexpr const char* kernel_source = R"CLC(
__kernel void test_bitfield_extract(__global SIGNED_TYPE* sdst, __global UNSIGNED_TYPE* udst, __global TYPE* base)
{
    int index = get_global_id(0);
    uint offset = index / (sizeof(BASETYPE) * 8 + 1);
    uint count = index % (sizeof(BASETYPE) * 8 + 1);
    if (offset + count > sizeof(BASETYPE) * 8) {
        count = (sizeof(BASETYPE) * 8) - offset;
    }
    sdst[index] = bitfield_extract_signed(base[index], offset, count);
    udst[index] = bitfield_extract_unsigned(base[index], offset, count);
}
)CLC";

static constexpr const char* kernel_source_vec3 = R"CLC(
__kernel void test_bitfield_extract(__global SIGNED_BASETYPE* sdst, __global UNSIGNED_BASETYPE* udst, __global BASETYPE* base)
{
    int index = get_global_id(0);
    uint offset = index / (sizeof(BASETYPE) * 8 + 1);
    uint count = index % (sizeof(BASETYPE) * 8 + 1);
    if (offset + count > sizeof(BASETYPE) * 8) {
        count = (sizeof(BASETYPE) * 8) - offset;
    }
    TYPE b = vload3(index, base);
    SIGNED_TYPE s = bitfield_extract_signed(b, offset, count);
    UNSIGNED_TYPE u = bitfield_extract_unsigned(b, offset, count);
    vstore3(s, index, sdst);
    vstore3(u, index, udst);
}
)CLC";

template <typename T, size_t N>
static int test_vectype(cl_device_id device, cl_context context,
                        cl_command_queue queue)
{
    // Because converting from an unsigned type to a signed type is
    // implementation-defined if the most significant bit is set until C++ 20,
    // compute all reference results using unsigned types.
    typedef typename std::make_unsigned<T>::type unsigned_t;

    cl_int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string buildOptions;
    buildOptions += " -DTYPE=";
    buildOptions +=
        TestInfo<T>::deviceTypeName + ((N > 1) ? std::to_string(N) : "");
    buildOptions += " -DSIGNED_TYPE=";
    buildOptions +=
        TestInfo<T>::deviceTypeNameSigned + ((N > 1) ? std::to_string(N) : "");
    buildOptions += " -DUNSIGNED_TYPE=";
    buildOptions += TestInfo<T>::deviceTypeNameUnsigned
        + ((N > 1) ? std::to_string(N) : "");
    buildOptions += " -DBASETYPE=";
    buildOptions += TestInfo<T>::deviceTypeName;
    buildOptions += " -DSIGNED_BASETYPE=";
    buildOptions += TestInfo<T>::deviceTypeNameSigned;
    buildOptions += " -DUNSIGNED_BASETYPE=";
    buildOptions += TestInfo<T>::deviceTypeNameUnsigned;

    const size_t ELEMENTS_TO_TEST = (sizeof(T) * 8 + 1) * (sizeof(T) * 8 + 1);

    std::vector<T> base(ELEMENTS_TO_TEST * N);
    fill_vector_with_random_data(base);

    std::vector<unsigned_t> sreference;
    std::vector<unsigned_t> ureference;
    calculate_reference<T, N>(sreference, ureference, base);

    const char* source = (N == 3) ? kernel_source_vec3 : kernel_source;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                        "test_bitfield_extract",
                                        buildOptions.c_str());
    test_error(error, "Unable to create test_bitfield_insert kernel");

    clMemWrapper sdst =
        clCreateBuffer(context, 0, sreference.size() * sizeof(T), NULL, &error);
    test_error(error, "Unable to create signed output buffer");

    clMemWrapper udst =
        clCreateBuffer(context, 0, ureference.size() * sizeof(T), NULL, &error);
    test_error(error, "Unable to create unsigned output buffer");

    clMemWrapper src_base =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, base.size() * sizeof(T),
                       base.data(), &error);
    test_error(error, "Unable to create base buffer");

    error = clSetKernelArg(kernel, 0, sizeof(sdst), &sdst);
    test_error(error, "Unable to set signed output buffer kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(udst), &udst);
    test_error(error, "Unable to set unsigned output buffer kernel arg");

    error = clSetKernelArg(kernel, 2, sizeof(src_base), &src_base);
    test_error(error, "Unable to set base buffer kernel arg");

    size_t global_work_size[] = { sreference.size() / N };
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                   NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue test kernel");

    error = clFinish(queue);
    test_error(error, "clFinish failed after test kernel");

    std::vector<unsigned_t> sresults(sreference.size(), 99);
    error = clEnqueueReadBuffer(queue, sdst, CL_TRUE, 0,
                                sresults.size() * sizeof(T), sresults.data(), 0,
                                NULL, NULL);
    test_error(error, "Unable to read signed data after test kernel");

    if (sresults != sreference)
    {
        log_error("Signed result buffer did not match reference buffer!\n");
        return TEST_FAIL;
    }

    std::vector<unsigned_t> uresults(ureference.size(), 99);
    error = clEnqueueReadBuffer(queue, udst, CL_TRUE, 0,
                                uresults.size() * sizeof(T), uresults.data(), 0,
                                NULL, NULL);
    test_error(error, "Unable to read unsigned data after test kernel");

    if (uresults != ureference)
    {
        log_error("Unsigned result buffer did not match reference buffer!\n");
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

int test_extended_bit_ops_extract(cl_device_id device, cl_context context,
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
