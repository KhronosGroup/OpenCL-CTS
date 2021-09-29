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
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "procs.h"
#include "harness/integer_ops_test_info.h"
#include "harness/testHarness.h"

template <size_t N, typename DstType, typename SrcTypeA, typename SrcTypeB>
static void
calculate_reference(std::vector<DstType>& ref, const std::vector<SrcTypeA>& a,
                    const std::vector<SrcTypeB>& b, const bool AccSat = false,
                    const std::vector<DstType>& acc = {})
{
    assert(a.size() == b.size());
    assert(AccSat == false || acc.size() == a.size() / N);

    ref.resize(a.size() / N);
    for (size_t r = 0; r < ref.size(); r++)
    {
        cl_long result = AccSat ? acc[r] : 0;
        for (size_t c = 0; c < N; c++)
        {
            // OK to assume no overflow?
            result += a[r * N + c] * b[r * N + c];
        }
        if (AccSat && result > std::numeric_limits<DstType>::max())
        {
            result = std::numeric_limits<DstType>::max();
        }
        ref[r] = static_cast<DstType>(result);
    }
}

template <typename SrcTypeA, typename SrcTypeB>
void generate_inputs_with_special_values(std::vector<SrcTypeA>& a,
                                         std::vector<SrcTypeB>& b)
{
    const std::vector<SrcTypeA> specialValuesA(
        { static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::min()),
          static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::min() + 1),
          static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::min() / 2), 0,
          static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::max() / 2),
          static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::max() - 1),
          static_cast<SrcTypeA>(std::numeric_limits<SrcTypeA>::max()) });
    const std::vector<SrcTypeB> specialValuesB(
        { static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::min()),
          static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::min() + 1),
          static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::min() / 2), 0,
          static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::max() / 2),
          static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::max() - 1),
          static_cast<SrcTypeB>(std::numeric_limits<SrcTypeB>::max()) });

    size_t count = 0;
    for (auto svA : specialValuesA)
    {
        for (auto svB : specialValuesB)
        {
            a[count] = svA;
            b[count] = svB;
            ++count;
        }
    }

    // Generate random data for the rest of the inputs:
    MTdataHolder d(gRandomSeed);
    generate_random_data(TestInfo<SrcTypeA>::explicitType, a.size() - count, d,
                         a.data() + count);
    generate_random_data(TestInfo<SrcTypeB>::explicitType, b.size() - count, d,
                         b.data() + count);
}

template <typename SrcType>
void generate_acc_sat_inputs(std::vector<SrcType>& acc)
{
    // First generate random data:
    fill_vector_with_random_data(acc);

    // Now go through the generated data, and make every other element large.
    // This ensures we have some elements that need saturation.
    for (size_t i = 0; i < acc.size(); i += 2)
    {
        acc[i] = std::numeric_limits<SrcType>::max() - acc[i];
    }
}

template <typename T> struct PackedTestInfo
{
    static constexpr const char* deviceTypeName = "UNSUPPORTED";
};
template <> struct PackedTestInfo<cl_char>
{
    static constexpr const char* deviceTypeName = "int";
};
template <> struct PackedTestInfo<cl_uchar>
{
    static constexpr const char* deviceTypeName = "uint";
};

static constexpr const char* kernel_source_dot = R"CLC(
__kernel void test_dot(__global DSTTYPE* dst, __global SRCTYPEA* a, __global SRCTYPEB* b)
{
    int index = get_global_id(0);
    dst[index] = DOT(a[index], b[index]);
}
)CLC";

static constexpr const char* kernel_source_dot_acc_sat = R"CLC(
__kernel void test_dot_acc_sat(
    __global DSTTYPE* dst,
    __global SRCTYPEA* a, __global SRCTYPEB* b, __global DSTTYPE* acc)
{
    int index = get_global_id(0);
    dst[index] = DOT_ACC_SAT(a[index], b[index], acc[index]);
}
)CLC";

template <typename DstType, typename SrcTypeA, typename SrcTypeB, size_t N>
static int test_case_dot(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements, bool packed,
                         bool sat)
{
    log_info("    testing %s = dot%s%s(%s, %s)\n",
             std::numeric_limits<DstType>::is_signed ? "signed" : "unsigned",
             sat ? "_acc_sat" : "", packed ? "_packed" : "",
             std::numeric_limits<SrcTypeA>::is_signed ? "signed" : "unsigned",
             std::numeric_limits<SrcTypeB>::is_signed ? "signed" : "unsigned");

    cl_int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string buildOptions;
    buildOptions += " -DDSTTYPE=";
    buildOptions += TestInfo<DstType>::deviceTypeName;
    buildOptions += " -DSRCTYPEA=";
    buildOptions += packed
        ? PackedTestInfo<SrcTypeA>::deviceTypeName
        : TestInfo<SrcTypeA>::deviceTypeName + std::to_string(N);
    buildOptions += " -DSRCTYPEB=";
    buildOptions += packed
        ? PackedTestInfo<SrcTypeB>::deviceTypeName
        : TestInfo<SrcTypeB>::deviceTypeName + std::to_string(N);
    std::string packedSuffix;
    packedSuffix += std::numeric_limits<SrcTypeA>::is_signed ? "s" : "u";
    packedSuffix += std::numeric_limits<SrcTypeB>::is_signed ? "s" : "u";
    packedSuffix += std::numeric_limits<DstType>::is_signed ? "_int" : "_uint";
    if (sat)
    {
        buildOptions += packed
            ? " -DDOT_ACC_SAT=dot_acc_sat_4x8packed_" + packedSuffix
            : " -DDOT_ACC_SAT=dot_acc_sat";
    }
    else
    {
        buildOptions +=
            packed ? " -DDOT=dot_4x8packed_" + packedSuffix : " -DDOT=dot";
    }

    std::vector<SrcTypeA> a(N * num_elements);
    std::vector<SrcTypeB> b(N * num_elements);
    generate_inputs_with_special_values(a, b);

    std::vector<DstType> acc;
    if (sat)
    {
        acc.resize(num_elements);
        generate_acc_sat_inputs(acc);
    }

    std::vector<DstType> reference(num_elements);
    calculate_reference<N>(reference, a, b, sat, acc);

    const char* source = sat ? kernel_source_dot_acc_sat : kernel_source_dot;
    const char* name = sat ? "test_dot_acc_sat" : "test_dot";
    error = create_single_kernel_helper(context, &program, &kernel, 1, &source,
                                        name, buildOptions.c_str());
    test_error(error, "Unable to create test kernel");

    clMemWrapper dst = clCreateBuffer(
        context, 0, reference.size() * sizeof(DstType), NULL, &error);
    test_error(error, "Unable to create output buffer");

    clMemWrapper srcA =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       a.size() * sizeof(SrcTypeA), a.data(), &error);
    test_error(error, "Unable to create srcA buffer");

    clMemWrapper srcB =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       b.size() * sizeof(SrcTypeB), b.data(), &error);
    test_error(error, "Unable to create srcB buffer");

    clMemWrapper srcAcc;
    if (sat)
    {
        srcAcc =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                           acc.size() * sizeof(DstType), acc.data(), &error);
        test_error(error, "Unable to create acc buffer");
    }

    error = clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    test_error(error, "Unable to set output buffer kernel arg");

    error = clSetKernelArg(kernel, 1, sizeof(srcA), &srcA);
    test_error(error, "Unable to set srcA buffer kernel arg");

    error = clSetKernelArg(kernel, 2, sizeof(srcB), &srcB);
    test_error(error, "Unable to set srcB buffer kernel arg");

    if (sat)
    {
        error = clSetKernelArg(kernel, 3, sizeof(srcAcc), &srcAcc);
        test_error(error, "Unable to set acc buffer kernel arg");
    }

    size_t global_work_size[] = { reference.size() };
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                   NULL, 0, NULL, NULL);
    test_error(error, "Unable to enqueue test kernel");

    error = clFinish(queue);
    test_error(error, "clFinish failed after test kernel");

    std::vector<DstType> results(reference.size(), 99);
    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                results.size() * sizeof(DstType),
                                results.data(), 0, NULL, NULL);
    test_error(error, "Unable to read data after test kernel");

    if (results != reference)
    {
        log_error("Result buffer did not match reference buffer!\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

template <typename SrcType, typename DstType, size_t N>
static int test_vectype(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    int result = TEST_PASS;

    typedef typename std::make_signed<SrcType>::type SSrcType;
    typedef typename std::make_signed<DstType>::type SDstType;

    typedef typename std::make_unsigned<SrcType>::type USrcType;
    typedef typename std::make_unsigned<DstType>::type UDstType;

    // dot testing:
    result |= test_case_dot<UDstType, USrcType, USrcType, N>(
        deviceID, context, queue, num_elements, false, false);
    result |= test_case_dot<SDstType, SSrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, false, false);
    result |= test_case_dot<SDstType, USrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, false, false);
    result |= test_case_dot<SDstType, SSrcType, USrcType, N>(
        deviceID, context, queue, num_elements, false, false);

    // dot_acc_sat testing:
    result |= test_case_dot<UDstType, USrcType, USrcType, N>(
        deviceID, context, queue, num_elements, false, true);
    result |= test_case_dot<SDstType, SSrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, false, true);
    result |= test_case_dot<SDstType, USrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, false, true);
    result |= test_case_dot<SDstType, SSrcType, USrcType, N>(
        deviceID, context, queue, num_elements, false, true);

    return result;
}

template <typename SrcType, typename DstType, size_t N>
static int test_vectype_packed(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    int result = TEST_PASS;

    typedef typename std::make_signed<SrcType>::type SSrcType;
    typedef typename std::make_signed<DstType>::type SDstType;

    typedef typename std::make_unsigned<SrcType>::type USrcType;
    typedef typename std::make_unsigned<DstType>::type UDstType;

    // packed dot testing:
    result |= test_case_dot<UDstType, USrcType, USrcType, N>(
        deviceID, context, queue, num_elements, true, false);
    result |= test_case_dot<SDstType, SSrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, true, false);
    result |= test_case_dot<SDstType, USrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, true, false);
    result |= test_case_dot<SDstType, SSrcType, USrcType, N>(
        deviceID, context, queue, num_elements, true, false);

    // packed dot_acc_sat testing:
    result |= test_case_dot<UDstType, USrcType, USrcType, N>(
        deviceID, context, queue, num_elements, true, true);
    result |= test_case_dot<SDstType, SSrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, true, true);
    result |= test_case_dot<SDstType, USrcType, SSrcType, N>(
        deviceID, context, queue, num_elements, true, true);
    result |= test_case_dot<SDstType, SSrcType, USrcType, N>(
        deviceID, context, queue, num_elements, true, true);

    return result;
}

int test_integer_dot_product(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_integer_dot_product"))
    {
        log_info("cl_khr_integer_dot_product is not supported\n");
        return TEST_SKIPPED_ITSELF;
    }

    Version deviceVersion = get_device_cl_version(deviceID);
    cl_version extensionVersion;

    if ((deviceVersion >= Version(3, 0))
        || is_extension_available(deviceID, "cl_khr_extended_versioning"))
    {
        extensionVersion =
            get_extension_version(deviceID, "cl_khr_integer_dot_product");
    }
    else
    {
        // Assume 1.0.0 is supported if the version can't be queried
        extensionVersion = CL_MAKE_VERSION(1, 0, 0);
    }

    cl_int error = CL_SUCCESS;
    int result = TEST_PASS;

    cl_device_integer_dot_product_capabilities_khr dotCaps = 0;
    error = clGetDeviceInfo(deviceID,
                            CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                            sizeof(dotCaps), &dotCaps, NULL);
    test_error(
        error,
        "Unable to query CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR");

    // Check that the required capabilities are reported
    test_assert_error(
        dotCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR,
        "When cl_khr_integer_dot_product is supported "
        "CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR must be "
        "supported");

    if (extensionVersion >= CL_MAKE_VERSION(2, 0, 0))
    {
        test_assert_error(
            dotCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR,
            "When cl_khr_integer_dot_product is supported with version >= 2.0.0"
            "CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR must be "
            "supported");
    }

    // Check that acceleration properties can be queried
    if (extensionVersion >= CL_MAKE_VERSION(2, 0, 0))
    {
        size_t size_ret;
        error = clGetDeviceInfo(
            deviceID,
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR, 0,
            nullptr, &size_ret);
        test_error(
            error,
            "Unable to query size of data returned by "
            "CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR");

        cl_device_integer_dot_product_acceleration_properties_khr
            accelerationProperties;
        error = clGetDeviceInfo(
            deviceID,
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR,
            sizeof(accelerationProperties), &accelerationProperties, nullptr);
        test_error(error, "Unable to query 8-bit acceleration properties");

        error = clGetDeviceInfo(
            deviceID,
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR,
            0, nullptr, &size_ret);
        test_error(
            error,
            "Unable to query size of data returned by "
            "CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_"
            "PACKED_KHR");

        error = clGetDeviceInfo(
            deviceID,
            CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR,
            sizeof(accelerationProperties), &accelerationProperties, nullptr);
        test_error(error,
                   "Unable to query 4x8-bit packed acceleration properties");
    }

    // Report when unknown capabilities are found
    if (dotCaps
        & ~(CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR
            | CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR))
    {
        log_info("NOTE: found an unknown / untested capability!\n");
    }

    // Test built-in functions
    if (dotCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR)
    {
        result |= test_vectype<cl_uchar, cl_uint, 4>(deviceID, context, queue,
                                                     num_elements);
    }

    if (dotCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR)
    {
        result |= test_vectype_packed<cl_uchar, cl_uint, 4>(
            deviceID, context, queue, num_elements);
    }

    return result;
}
