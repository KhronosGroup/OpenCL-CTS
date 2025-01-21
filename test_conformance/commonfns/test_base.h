// Copyright (c) 2023 The Khronos Group Inc.
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

#ifndef TEST_COMMONFNS_BASE_H
#define TEST_COMMONFNS_BASE_H

#include <vector>
#include <map>
#include <memory>
#include <cmath>

#include <CL/cl_half.h>
#include <CL/cl_ext.h>

#include "harness/conversions.h"
#include "harness/mt19937.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"

#define kVectorSizeCount 5
#define kStrangeVectorSizeCount 1
#define kTotalVecCount (kVectorSizeCount + kStrangeVectorSizeCount)

extern int g_arrVecSizes[kVectorSizeCount + kStrangeVectorSizeCount];

template <typename T>
using VerifyFuncBinary = int (*)(const T *const, const T *const, const T *const,
                                 const int num, const int vs, const int vp);

template <typename T>
using VerifyFuncUnary = int (*)(const T *const, const T *const, const int num);

using half = cl_half;

struct BaseFunctionTest
{
    BaseFunctionTest(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elems, const char *fn,
                     bool vsp)
        : device(device), context(context), queue(queue), num_elems(num_elems),
          fnName(fn), vecParam(vsp)
    {}

    // Test body returning an OpenCL error code
    virtual cl_int Run() = 0;

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    int num_elems;
    std::string fnName;
    bool vecParam;

    static std::map<size_t, std::string> type2name;
    static cl_half_rounding_mode halfRoundingMode;
};

struct MinTest : BaseFunctionTest
{
    MinTest(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct MaxTest : BaseFunctionTest
{
    MaxTest(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct ClampTest : BaseFunctionTest
{
    ClampTest(cl_device_id device, cl_context context, cl_command_queue queue,
              int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct DegreesTest : BaseFunctionTest
{
    DegreesTest(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct RadiansTest : BaseFunctionTest
{
    RadiansTest(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct SignTest : BaseFunctionTest
{
    SignTest(cl_device_id device, cl_context context, cl_command_queue queue,
             int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct SmoothstepTest : BaseFunctionTest
{
    SmoothstepTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elems, const char *fn,
                   bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct StepTest : BaseFunctionTest
{
    StepTest(cl_device_id device, cl_context context, cl_command_queue queue,
             int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

struct MixTest : BaseFunctionTest
{
    MixTest(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elems, const char *fn, bool vsp)
        : BaseFunctionTest(device, context, queue, num_elems, fn, vsp)
    {}

    cl_int Run() override;
};

template <typename T> inline double conv_to_dbl(const T &val)
{
    if (std::is_same<T, half>::value)
        return (double)cl_half_to_float(val);
    else
        return (double)val;
}

template <typename T> inline double conv_to_flt(const T &val)
{
    if (std::is_same<T, half>::value)
        return (float)cl_half_to_float(val);
    else
        return (float)val;
}

template <typename T> inline half conv_to_half(const T &val)
{
    if (std::is_floating_point<T>::value)
        return cl_half_from_float(val, BaseFunctionTest::halfRoundingMode);
    return 0;
}

template <typename T> bool isfinite_fp(const T &v)
{
    if (std::is_same<T, half>::value)
    {
        // Extract FP16 exponent and mantissa
        uint16_t h_exp = (((half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
        uint16_t h_mant = ((half)v) & 0x3FF;

        // !Inf test
        return !(h_exp == 0x1F && h_mant == 0);
    }
    else
    {
#if !defined(_WIN32)
        return std::isfinite(v);
#else
        return isfinite(v);
#endif
    }
}

template <typename T> float UlpFn(const T &val, const double &r)
{
    if (std::is_same<T, half>::value)
    {
        if (conv_to_half(r) == val)
        {
            return 0.0f;
        }

        return Ulp_Error_Half(val, r);
    }
    else if (std::is_same<T, float>::value)
    {
        return Ulp_Error(val, r);
    }
    else if (std::is_same<T, double>::value)
    {
        return Ulp_Error_Double(val, r);
    }
    else
    {
        log_error("UlpFn: unsupported data type\n");
    }

    return -1.f; // wrong val
}

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements,
                   const char *fn = "", bool vsp = false)
{
    auto test_fixture = T(device, context, queue, num_elements, fn, vsp);

    cl_int error = test_fixture.Run();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}

#endif // TEST_COMMONFNS_BASE_H
