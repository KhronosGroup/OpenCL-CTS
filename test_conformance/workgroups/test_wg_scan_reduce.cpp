//
// Copyright (c) 2017-2022 The Khronos Group Inc.
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
#include "harness/compat.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "testBase.h"

cl_half_rounding_mode gHalfRoundingMode = CL_HALF_RTE;
constexpr cl_half g_half_min = 0xfbff;
constexpr cl_half g_half_max = 0x7bff;

static std::string make_kernel_string(const std::string &type,
                                      const std::string &kernelName,
                                      const std::string &func)
{
    // Build a kernel string of the form:
    // __kernel void KERNEL_NAME(global TYPE *input, global TYPE *output) {
    //     int  tid = get_global_id(0);
    //     output[tid] = FUNC(input[tid]);
    // }

    std::ostringstream os;
    os << "__kernel void " << kernelName << "(global " << type
       << " *input, global " << type << " *output) {\n";
    os << "    int tid = get_global_id(0);\n";
    os << "    output[tid] = " << func << "(input[tid]);\n";
    os << "}\n";
    return os.str();
}

template <typename T> struct TestTypeInfo
{
};

template <> struct TestTypeInfo<cl_int>
{
    static constexpr const char *deviceName = "int";
};

template <> struct TestTypeInfo<cl_uint>
{
    static constexpr const char *deviceName = "uint";
};

template <> struct TestTypeInfo<cl_long>
{
    static constexpr const char *deviceName = "long";
};

template <> struct TestTypeInfo<cl_ulong>
{
    static constexpr const char *deviceName = "ulong";
};

template <> struct TestTypeInfo<cl_double>
{
    static constexpr const char *deviceName = "double";
};

template <> struct TestTypeInfo<cl_float>
{
    static constexpr const char *deviceName = "float";
};

// please keep in mind cl_half type on host side is the same as uint16_t,
// therefore, if you will add below 16-bit unsigned int type support it will be
// likely confused with cl_half

template <> struct TestTypeInfo<cl_half>
{
    static constexpr const char *deviceName = "half";
};

template <typename T> struct Add
{
    using Type = T;
    static constexpr const char *opName = "add";
    static constexpr T identityValue = 0;
    static T combine(T a, T b)
    {
        if (std::is_same_v<T, cl_half>)
            return cl_half_from_float(cl_half_to_float(a) + cl_half_to_float(b),
                                      gHalfRoundingMode);
        else
            return a + b;
    }
};

template <typename T> struct Max
{
    using Type = T;
    static constexpr const char *opName = "max";
    static constexpr T identityValue = std::is_same_v<T, cl_half>
        ? g_half_min
        : (std::is_integral_v<T> ? std::numeric_limits<T>::min()
                                 : -std::numeric_limits<T>::max());
    static T combine(T a, T b)
    {
        if (std::is_same_v<T, cl_half>)
            return cl_half_from_float(
                std::max(cl_half_to_float(a), cl_half_to_float(b)),
                gHalfRoundingMode);
        else
            return std::max(a, b);
    }
};

template <typename T> struct Min
{
    using Type = T;
    static constexpr const char *opName = "min";
    static constexpr T identityValue =
        std::is_same_v<T, cl_half> ? g_half_max : std::numeric_limits<T>::max();
    static T combine(T a, T b)
    {
        if (std::is_same_v<T, cl_half>)
            return cl_half_from_float(
                std::min(cl_half_to_float(a), cl_half_to_float(b)),
                gHalfRoundingMode);
        else
            return std::min(a, b);
    }
};

template <typename C> struct Reduce
{
    using Type = typename C::Type;
    using Operation = C;

    static constexpr const char *testName = "work_group_reduce";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_reduce";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size, const Type &max_err = 0)
    {
        for (size_t i = 0; i < n_elems; i += max_wg_size)
        {
            size_t wg_size = std::min(max_wg_size, n_elems - i);

            Type result = C::identityValue;
            for (size_t j = 0; j < wg_size; j++)
            {
                result = C::combine(result, inptr[i + j]);
            }

            for (size_t j = 0; j < wg_size; j++)
            {
                if constexpr (std::is_floating_point_v<Type>)
                {
                    if (fabs(result - outptr[i + j]) > max_err)
                    {
                        log_info("%s_%s: Error at %zu\n", testName, testOpName,
                                 i + j);
                        return -1;
                    }
                }
                else if (std::is_same_v<Type, cl_half>)
                {
                    if (fabs(cl_half_to_float(result)
                             - cl_half_to_float(outptr[i + j]))
                        > cl_half_to_float(max_err))
                    {
                        log_info("%s_%s: Error at %zu\n", testName, testOpName,
                                 i + j);
                        return -1;
                    }
                }
                else
                {
                    if (result != outptr[i + j])
                    {
                        log_info("%s_%s: Error at %zu\n", testName, testOpName,
                                 i + j);
                        return -1;
                    }
                }
            }
        }
        return 0;
    }

    static void generate_reference_values(Type *inptr, size_t n_elems,
                                          size_t max_wg_size, Type &max_err = 0)
    {
        MTdataHolder d(gRandomSeed);
        if constexpr (std::is_floating_point_v<
                          Type> || std::is_same_v<Type, cl_half>)
        {
            std::vector<Type> ref_vals(max_wg_size, 0);
            if (std::is_same_v<Type, cl_half>)
            {
                // to prevent overflow limit range of randomization
                float max_range = 99.0;
                float min_range = -99.0;
                // generate reference values for one work group
                for (size_t j = 0; j < max_wg_size; j++)
                    ref_vals[j] = cl_half_from_float(
                        get_random_float(min_range, max_range, d),
                        gHalfRoundingMode);

                // populate reference data across all work groups
                for (size_t i = 0; i < (size_t)n_elems; i += max_wg_size)
                {
                    size_t wg_size = std::min(max_wg_size, n_elems - i);
                    memcpy(&inptr[i], ref_vals.data(), sizeof(Type) * wg_size);
                }

                if constexpr (std::is_same_v<Operation, Add<Type>>)
                {
                    // compute maximal summation error
                    std::sort(ref_vals.begin(), ref_vals.end(),
                              [](cl_half a, cl_half b) {
                                  return std::abs(cl_half_to_float(a))
                                      < std::abs(cl_half_to_float(b));
                              });

                    float s = 0.f;
                    for (auto it = ref_vals.begin(); it != ref_vals.end(); ++it)
                        s += std::abs(cl_half_to_float(*it));
                    max_err = cl_half_from_float(
                        fabs((max_wg_size - 1) * CL_HALF_EPSILON * s),
                        gHalfRoundingMode);
                }
            }
            else
            {
                double max_range = 999.0;
                double min_range = -999.0;
                for (size_t j = 0; j < max_wg_size; j++)
                    ref_vals[j] = get_random_float(min_range, max_range, d);

                for (size_t i = 0; i < (size_t)n_elems; i += max_wg_size)
                {
                    size_t work_group_size = std::min(max_wg_size, n_elems - i);
                    memcpy(&inptr[i], ref_vals.data(),
                           sizeof(Type) * work_group_size);
                }

                if constexpr (std::is_same_v<Operation, Add<Type>>)
                {
                    // compute maximal summation error
                    std::sort(ref_vals.begin(), ref_vals.end());
                    Type abs_sum = 0;
                    for (auto elem : ref_vals) abs_sum += fabs(elem);
                    // Higham, N. J. (2002). Accuracy and Stability of Numerical
                    // Algorithms (2nd ed.), Chapter 4: Summation, Section 2:
                    // Error Analysis (worst case error summation)
                    max_err = (max_wg_size - 1)
                        * (std::is_same_v<Type, cl_float> ? CL_FLT_EPSILON
                                                          : CL_DBL_EPSILON)
                        * abs_sum;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n_elems; i++)
                inptr[i] = (Type)genrand_int64(d);
        }
    }
};

template <typename C> struct ScanInclusive
{
    using Type = typename C::Type;
    using Operation = C;

    static constexpr const char *testName = "work_group_scan_inclusive";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_scan_inclusive";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size, const Type &max_err = 0)
    {
        for (size_t i = 0; i < n_elems; i += max_wg_size)
        {
            size_t wg_size = std::min(max_wg_size, n_elems - i);

            Type result = C::identityValue;
            for (size_t j = 0; j < wg_size; ++j)
            {
                result = C::combine(result, inptr[i + j]);
                if (result != outptr[i + j])
                {
                    log_info("%s_%s: Error at %zu\n", testName, testOpName,
                             i + j);
                    return -1;
                }
            }
        }
        return 0;
    }

    static void generate_reference_values(Type *inptr, size_t n_elems,
                                          size_t max_wg_size, Type &max_err = 0)
    {
        MTdataHolder d(gRandomSeed);
        for (size_t i = 0; i < n_elems; i++) inptr[i] = (Type)genrand_int64(d);
    }
};

template <typename C> struct ScanExclusive
{
    using Type = typename C::Type;
    using Operation = C;

    static constexpr const char *testName = "work_group_scan_exclusive";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_scan_exclusive";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size, const Type &max_err = 0)
    {
        for (size_t i = 0; i < n_elems; i += max_wg_size)
        {
            size_t wg_size = std::min(max_wg_size, n_elems - i);

            Type result = C::identityValue;
            for (size_t j = 0; j < wg_size; ++j)
            {
                if (result != outptr[i + j])
                {
                    log_info("%s_%s: Error at %zu\n", testName, testOpName,
                             i + j);
                    return -1;
                }
                result = C::combine(result, inptr[i + j]);
            }
        }
        return 0;
    }

    static void generate_reference_values(Type *inptr, size_t n_elems,
                                          size_t max_wg_size, Type &max_err = 0)
    {
        MTdataHolder d(gRandomSeed);
        for (size_t i = 0; i < n_elems; i++) inptr[i] = (Type)genrand_int64(d);
    }
};

template <typename TestInfo>
static int run_test(cl_device_id device, cl_context context,
                    cl_command_queue queue, int n_elems)
{
    using T = typename TestInfo::Type;
    cl_int err = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string funcName = TestInfo::testName;
    funcName += "_";
    funcName += TestInfo::testOpName;

    std::string kernelName = TestInfo::kernelName;
    kernelName += "_";
    kernelName += TestInfo::testOpName;
    kernelName += "_";
    kernelName += TestInfo::deviceTypeName;

    std::string kernelString =
        make_kernel_string(TestInfo::deviceTypeName, kernelName, funcName);

    const char *kernel_source = kernelString.c_str();
    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &kernel_source, kernelName.c_str());
    test_error(err, "Unable to create test kernel");

    size_t wg_size[1];
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    clMemWrapper src = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(T) * n_elems, NULL, &err);
    test_error(err, "Unable to create source buffer");

    clMemWrapper dst = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(T) * n_elems, NULL, &err);
    test_error(err, "Unable to create destination buffer");

    std::vector<T> input_ptr(n_elems);

    T max_err = 0;
    TestInfo::generate_reference_values(input_ptr.data(), n_elems, wg_size[0],
                                        max_err);

    err = clEnqueueWriteBuffer(queue, src, CL_TRUE, 0, sizeof(T) * n_elems,
                               input_ptr.data(), 0, NULL, NULL);
    test_error(err, "clWriteBuffer to initialize src buffer failed");

    err = clSetKernelArg(kernel, 0, sizeof(src), &src);
    test_error(err, "Unable to set src buffer kernel arg");
    err |= clSetKernelArg(kernel, 1, sizeof(dst), &dst);
    test_error(err, "Unable to set dst buffer kernel arg");

    size_t global_work_size[] = { (size_t)n_elems };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 wg_size, 0, NULL, NULL);
    test_error(err, "Unable to enqueue test kernel");

    std::vector<T> output_ptr(n_elems);

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr.data(), &dead, sizeof(T) * n_elems);
    err = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0, sizeof(T) * n_elems,
                              output_ptr.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueReadBuffer to read read dst buffer failed");

    if (TestInfo::verify(input_ptr.data(), output_ptr.data(), n_elems,
                         wg_size[0], max_err))
    {
        log_error("%s_%s %s verify failed\n", TestInfo::testName,
                  TestInfo::testOpName, TestInfo::deviceTypeName);
        return TEST_FAIL;
    }

    log_info("%s_%s %s passed\n", TestInfo::testName, TestInfo::testOpName,
             TestInfo::deviceTypeName);
    return TEST_PASS;
}

REGISTER_TEST_VERSION(work_group_reduce_add, Version(2, 0))
{
    int result = TEST_PASS;

    result |=
        run_test<Reduce<Add<cl_int>>>(device, context, queue, num_elements);
    result |=
        run_test<Reduce<Add<cl_uint>>>(device, context, queue, num_elements);

    if (gHasLong)
    {
        result |= run_test<Reduce<Add<cl_long>>>(device, context, queue,
                                                 num_elements);
        result |= run_test<Reduce<Add<cl_ulong>>>(device, context, queue,
                                                  num_elements);
    }

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode\n");
            return TEST_FAIL;
        }

        result |= run_test<Reduce<Add<cl_half>>>(device, context, queue,
                                                 num_elements);
    }

    result |=
        run_test<Reduce<Add<cl_float>>>(device, context, queue, num_elements);

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        result |= run_test<Reduce<Add<cl_double>>>(device, context, queue,
                                                   num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_reduce_max, Version(2, 0))
{
    int result = TEST_PASS;

    result |=
        run_test<Reduce<Max<cl_int>>>(device, context, queue, num_elements);
    result |=
        run_test<Reduce<Max<cl_uint>>>(device, context, queue, num_elements);

    if (gHasLong)
    {
        result |= run_test<Reduce<Max<cl_long>>>(device, context, queue,
                                                 num_elements);
        result |= run_test<Reduce<Max<cl_ulong>>>(device, context, queue,
                                                  num_elements);
    }

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode\n");
            return TEST_FAIL;
        }

        result |= run_test<Reduce<Max<cl_half>>>(device, context, queue,
                                                 num_elements);
    }

    result |=
        run_test<Reduce<Max<cl_float>>>(device, context, queue, num_elements);

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        result |= run_test<Reduce<Max<cl_double>>>(device, context, queue,
                                                   num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_reduce_min, Version(2, 0))
{
    int result = TEST_PASS;

    result |=
        run_test<Reduce<Min<cl_int>>>(device, context, queue, num_elements);
    result |=
        run_test<Reduce<Min<cl_uint>>>(device, context, queue, num_elements);

    if (gHasLong)
    {
        result |= run_test<Reduce<Min<cl_long>>>(device, context, queue,
                                                 num_elements);
        result |= run_test<Reduce<Min<cl_ulong>>>(device, context, queue,
                                                  num_elements);
    }

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode\n");
            return TEST_FAIL;
        }

        result |= run_test<Reduce<Min<cl_half>>>(device, context, queue,
                                                 num_elements);
    }

    result |=
        run_test<Reduce<Min<cl_float>>>(device, context, queue, num_elements);

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        result |= run_test<Reduce<Min<cl_double>>>(device, context, queue,
                                                   num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_inclusive_add, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanInclusive<Add<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanInclusive<Add<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Add<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanInclusive<Add<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_inclusive_max, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanInclusive<Max<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanInclusive<Max<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Max<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanInclusive<Max<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_inclusive_min, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanInclusive<Min<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanInclusive<Min<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Min<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanInclusive<Min<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_exclusive_add, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanExclusive<Add<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanExclusive<Add<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Add<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanExclusive<Add<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_exclusive_max, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanExclusive<Max<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanExclusive<Max<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Max<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanExclusive<Max<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}

REGISTER_TEST_VERSION(work_group_scan_exclusive_min, Version(2, 0))
{
    int result = TEST_PASS;

    result |= run_test<ScanExclusive<Min<cl_int>>>(device, context, queue,
                                                   num_elements);
    result |= run_test<ScanExclusive<Min<cl_uint>>>(device, context, queue,
                                                    num_elements);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Min<cl_long>>>(device, context, queue,
                                                        num_elements);
        result |= run_test<ScanExclusive<Min<cl_ulong>>>(device, context, queue,
                                                         num_elements);
    }

    return result;
}
