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

#include "procs.h"

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

template <typename T> struct Add
{
    using Type = T;
    static constexpr const char *opName = "add";
    static constexpr T identityValue = 0;
    static T combine(T a, T b) { return a + b; }
};

template <typename T> struct Max
{
    using Type = T;
    static constexpr const char *opName = "max";
    static constexpr T identityValue = std::numeric_limits<T>::min();
    static T combine(T a, T b) { return std::max(a, b); }
};

template <typename T> struct Min
{
    using Type = T;
    static constexpr const char *opName = "min";
    static constexpr T identityValue = std::numeric_limits<T>::max();
    static T combine(T a, T b) { return std::min(a, b); }
};

template <typename C> struct Reduce
{
    using Type = typename C::Type;

    static constexpr const char *testName = "work_group_reduce";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_reduce";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size)
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
};

template <typename C> struct ScanInclusive
{
    using Type = typename C::Type;

    static constexpr const char *testName = "work_group_scan_inclusive";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_scan_inclusive";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size)
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
};

template <typename C> struct ScanExclusive
{
    using Type = typename C::Type;

    static constexpr const char *testName = "work_group_scan_exclusive";
    static constexpr const char *testOpName = C::opName;
    static constexpr const char *deviceTypeName =
        TestTypeInfo<Type>::deviceName;
    static constexpr const char *kernelName = "test_wg_scan_exclusive";
    static int verify(Type *inptr, Type *outptr, size_t n_elems,
                      size_t max_wg_size)
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

    MTdataHolder d(gRandomSeed);
    for (int i = 0; i < n_elems; i++)
    {
        input_ptr[i] = (T)genrand_int64(d);
    }

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
                         wg_size[0]))
    {
        log_error("%s_%s %s failed\n", TestInfo::testName, TestInfo::testOpName,
                  TestInfo::deviceTypeName);
        return TEST_FAIL;
    }

    log_info("%s_%s %s passed\n", TestInfo::testName, TestInfo::testOpName,
             TestInfo::deviceTypeName);
    return TEST_PASS;
}

int test_work_group_reduce_add(cl_device_id device, cl_context context,
                               cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |= run_test<Reduce<Add<cl_int>>>(device, context, queue, n_elems);
    result |= run_test<Reduce<Add<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |=
            run_test<Reduce<Add<cl_long>>>(device, context, queue, n_elems);
        result |=
            run_test<Reduce<Add<cl_ulong>>>(device, context, queue, n_elems);
    }

    return result;
}

int test_work_group_reduce_max(cl_device_id device, cl_context context,
                               cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |= run_test<Reduce<Max<cl_int>>>(device, context, queue, n_elems);
    result |= run_test<Reduce<Max<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |=
            run_test<Reduce<Max<cl_long>>>(device, context, queue, n_elems);
        result |=
            run_test<Reduce<Max<cl_ulong>>>(device, context, queue, n_elems);
    }

    return result;
}

int test_work_group_reduce_min(cl_device_id device, cl_context context,
                               cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |= run_test<Reduce<Min<cl_int>>>(device, context, queue, n_elems);
    result |= run_test<Reduce<Min<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |=
            run_test<Reduce<Min<cl_long>>>(device, context, queue, n_elems);
        result |=
            run_test<Reduce<Min<cl_ulong>>>(device, context, queue, n_elems);
    }

    return result;
}

int test_work_group_scan_inclusive_add(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanInclusive<Add<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanInclusive<Add<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Add<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanInclusive<Add<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}

int test_work_group_scan_inclusive_max(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanInclusive<Max<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanInclusive<Max<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Max<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanInclusive<Max<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}

int test_work_group_scan_inclusive_min(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanInclusive<Min<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanInclusive<Min<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanInclusive<Min<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanInclusive<Min<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}

int test_work_group_scan_exclusive_add(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanExclusive<Add<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanExclusive<Add<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Add<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanExclusive<Add<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}

int test_work_group_scan_exclusive_max(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanExclusive<Max<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanExclusive<Max<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Max<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanExclusive<Max<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}

int test_work_group_scan_exclusive_min(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int n_elems)
{
    int result = TEST_PASS;

    result |=
        run_test<ScanExclusive<Min<cl_int>>>(device, context, queue, n_elems);
    result |=
        run_test<ScanExclusive<Min<cl_uint>>>(device, context, queue, n_elems);

    if (gHasLong)
    {
        result |= run_test<ScanExclusive<Min<cl_long>>>(device, context, queue,
                                                        n_elems);
        result |= run_test<ScanExclusive<Min<cl_ulong>>>(device, context, queue,
                                                         n_elems);
    }

    return result;
}
