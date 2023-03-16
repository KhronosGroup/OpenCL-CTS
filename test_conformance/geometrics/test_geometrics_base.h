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

#ifndef _TEST_GEOMETRICS_BASE_H
#define _TEST_GEOMETRICS_BASE_H

#include <vector>
#include <map>
#include <memory>
#include <CL/cl_half.h>

#include "testBase.h"
#include <limits>
#include <cmath>

#define HALF_P_NAN 0x7e00
#define HALF_N_NAN 0xfe00
#define HALF_P_INF 0x7c00
#define HALF_N_INF 0xfc00

struct GeometricsFPTest;

//--------------------------------------------------------------------------/

using half = cl_half;

//--------------------------------------------------------------------------

struct GeomTestBase
{
    GeomTestBase(const ExplicitTypes &dt, const std::string &name,
                 const float &ulp)
        : dataType(dt), fnName(name), ulpLimit(ulp)
    {}

    ExplicitTypes dataType;
    std::string fnName;
    float ulpLimit;

    static cl_half_rounding_mode halfRoundingMode;

    static char ftype[32];
    static char extension[128];
    static char load_store[128];
};

//--------------------------------------------------------------------------/

#define TEST_SIZE (1 << 10)
#define HFF(num) cl_half_from_float(num, GeomTestBase::halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

//--------------------------------------------------------------------------

template <typename T> struct GeomTestParams : public GeomTestBase
{
    GeomTestParams(const ExplicitTypes &dt, const std::string &name,
                   const float &ulp)
        : GeomTestBase(dt, name, ulp)
    {

        if (std::is_same<T, half>::value)
        {
            std::vector<half> trickyValuesInit = {
                HFF(-CL_HALF_EPSILON),
                HFF(CL_HALF_EPSILON),
                HFF(MAKE_HEX_FLOAT(0x1.0p7f, 0x1L, 7)),
                HFF(MAKE_HEX_FLOAT(0x1.8p7f, 0x18L, 3)),
                HFF(MAKE_HEX_FLOAT(0x1.0p8f, 0x1L, 8)),
                HFF(MAKE_HEX_FLOAT(-0x1.0p7f, -0x1L, 7)),
                HFF(MAKE_HEX_FLOAT(-0x1.8p-7f, -0x18L, -11)),
                HFF(MAKE_HEX_FLOAT(-0x1.0p8f, -0x1L, 8)),
                HFF(MAKE_HEX_FLOAT(0x1.0p-7f, 0x1L, -7)),
                HFF(MAKE_HEX_FLOAT(0x1.8p-7f, 0x18L, -11)),
                HFF(MAKE_HEX_FLOAT(0x1.0p-8f, 0x1L, -8)),
                HFF(MAKE_HEX_FLOAT(-0x1.0p-7f, -0x1L, -7)),
                HFF(MAKE_HEX_FLOAT(-0x1.8p-7f, -0x18L, -11)),
                HFF(MAKE_HEX_FLOAT(-0x1.0p-8f, -0x1L, -8)),
                HFF(HTF(CL_HALF_MAX) / 2.f),
                HFF(-HTF(CL_HALF_MAX) / 2.f),
                HALF_P_INF,
                HALF_N_INF,
                HFF(0.f),
                HFF(-0.f)
            };
            trickyValues.assign(trickyValuesInit.begin(),
                                trickyValuesInit.end());
        }
        else if (std::is_same<T, float>::value)
        {
            std::vector<float> trickyValuesInit = {
                -FLT_EPSILON,
                FLT_EPSILON,
                MAKE_HEX_FLOAT(0x1.0p63f, 0x1L, 63),
                MAKE_HEX_FLOAT(0x1.8p63f, 0x18L, 59),
                MAKE_HEX_FLOAT(0x1.0p64f, 0x1L, 64),
                MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63),
                MAKE_HEX_FLOAT(-0x1.8p-63f, -0x18L, -67),
                MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64),
                MAKE_HEX_FLOAT(0x1.0p-63f, 0x1L, -63),
                MAKE_HEX_FLOAT(0x1.8p-63f, 0x18L, -67),
                MAKE_HEX_FLOAT(0x1.0p-64f, 0x1L, -64),
                MAKE_HEX_FLOAT(-0x1.0p-63f, -0x1L, -63),
                MAKE_HEX_FLOAT(-0x1.8p-63f, -0x18L, -67),
                MAKE_HEX_FLOAT(-0x1.0p-64f, -0x1L, -64),
                FLT_MAX / 2.f,
                -FLT_MAX / 2.f,
                INFINITY,
                -INFINITY,
                0.f,
                -0.f
            };
            trickyValues.assign(trickyValuesInit.begin(),
                                trickyValuesInit.end());
        }
        else if (std::is_same<T, double>::value)
        {
            std::vector<double> trickyValuesInit = {
                -FLT_EPSILON,
                FLT_EPSILON,
                MAKE_HEX_DOUBLE(0x1.0p511, 0x1L, 511),
                MAKE_HEX_DOUBLE(0x1.8p511, 0x18L, 507),
                MAKE_HEX_DOUBLE(0x1.0p512, 0x1L, 512),
                MAKE_HEX_DOUBLE(-0x1.0p511, -0x1L, 511),
                MAKE_HEX_DOUBLE(-0x1.8p-511, -0x18L, -515),
                MAKE_HEX_DOUBLE(-0x1.0p512, -0x1L, 512),
                MAKE_HEX_DOUBLE(0x1.0p-511, 0x1L, -511),
                MAKE_HEX_DOUBLE(0x1.8p-511, 0x18L, -515),
                MAKE_HEX_DOUBLE(0x1.0p-512, 0x1L, -512),
                MAKE_HEX_DOUBLE(-0x1.0p-511, -0x1L, -511),
                MAKE_HEX_DOUBLE(-0x1.8p-511, -0x18L, -515),
                MAKE_HEX_DOUBLE(-0x1.0p-512, -0x1L, -512),
                DBL_MAX / 2.,
                -DBL_MAX / 2.,
                INFINITY,
                -INFINITY,
                0.,
                -0.
            };
            trickyValues.assign(trickyValuesInit.begin(),
                                trickyValuesInit.end());
        }
    }

    std::vector<T> trickyValues;
};

//--------------------------------------------------------------------------

// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct GeometricsFPTest
{
    GeometricsFPTest(cl_device_id device, cl_context context,
                     cl_command_queue queue);

    virtual cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    virtual cl_int Run();

    virtual cl_int RunSingleTest(const GeomTestBase *p) = 0;

    template <typename T>
    void FillWithTrickyNums(T *const, T *const, const size_t, const size_t,
                            const MTdata &, const GeomTestParams<T> &);

    template <typename T> float UlpError(const T &, const double &);

    template <typename T> double ToDouble(const T &);

    cl_int VerifyTestSize(size_t &test_size, const size_t &max_alloc,
                          const size_t &total_buf_size);

public:
    std::string static concat_kernel(const char *sstr[], int num);

    template <typename... Args>
    static std::string string_format(const std::string &format, Args... args);

    template <typename T>
    void vector2string(std::stringstream &sstr, T *vector, size_t elements);

    template <typename T> bool isnan_fp(const T &v);

    template <typename T> static bool isfinite_fp(const T &v);

    template <typename T> T max_fp(const T &lhs, const T &rhs);

    template <typename T> T abs_fp(const T &val);

    template <typename T> T get_random(float low, float high, const MTdata &d);

    template <typename T> T any_value(MTdata d);


protected:
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    cl_device_fp_config floatRoundingMode;
    cl_device_fp_config halfRoundingMode;

    bool floatHasInfNan;
    bool halfHasInfNan;

    int halfFlushDenormsToZero;

    size_t num_elements;

    std::vector<std::unique_ptr<GeomTestBase>> params;

    static cl_ulong maxAllocSize;
    static cl_ulong maxGlobalMemSize;
};

//--------------------------------------------------------------------------

template <typename T>
void GeometricsFPTest::FillWithTrickyNums(T *const aVectors, T *const bVectors,
                                          const size_t num_elems,
                                          const size_t vecSize, const MTdata &d,
                                          const GeomTestParams<T> &p)
{
    const size_t trickyCount = p.trickyValues.size();
    size_t copySize = vecSize * vecSize * trickyCount;

    if (copySize * 2 > num_elems)
    {
        log_error(
            "Buffer size not sufficient for test purposes, please verify!");
        return;
    }

    for (int j = 0; j < vecSize; j++)
        for (int k = 0; k < vecSize; k++)
            for (int i = 0; i < trickyCount; i++)
                aVectors[i + k * trickyCount + j * vecSize * trickyCount] =
                    p.trickyValues[i];

    if (bVectors)
    {
        memset(bVectors, 0, sizeof(T) * copySize);
        memset(aVectors + copySize, 0, sizeof(T) * copySize);
        memcpy(bVectors + copySize, aVectors, sizeof(T) * copySize);

        /* Clamp values to be in range for fast_ functions */
        // fast_* calls available only for single precision fp
        if (std::is_same<T, float>::value)
        {
            if (p.fnName.find("fast_") != std::string::npos)
            {
                for (int i = 0; i < num_elems; i++)
                {
                    if (abs_fp<T>(bVectors[i])
                            > MAKE_HEX_FLOAT(0x1.0p62f, 0x1L, 62)
                        || abs_fp<T>(bVectors[i])
                            < MAKE_HEX_FLOAT(0x1.0p-62f, 0x1L, -62))
                        bVectors[i] = get_random_float(-512.f, 512.f, d);
                }
            }
        }
    }

    if (std::is_same<T, float>::value)
    {
        if (p.fnName.find("fast_") != std::string::npos)
        {
            for (int i = 0; i < num_elems; i++)
            {
                if (abs_fp<T>(aVectors[i]) > MAKE_HEX_FLOAT(0x1.0p62f, 0x1L, 62)
                    || abs_fp<T>(aVectors[i])
                        < MAKE_HEX_FLOAT(0x1.0p-62f, 0x1L, -62))
                    aVectors[i] = get_random_float(-512.f, 512.f, d);
            }
        }
    }
}

//--------------------------------------------------------------------------

template <typename T>
float GeometricsFPTest::UlpError(const T &val, const double &ref)
{
    if (std::is_same<T, half>::value)
    {
        return Ulp_Error_Half(val, ref);
    }
    else if (std::is_same<T, float>::value)
    {
        return Ulp_Error(val, ref);
    }
    else if (std::is_same<T, double>::value)
    {
        return Ulp_Error_Double(val, ref);
    }
    else
    {
        log_error("GeometricsFPTest::UlpError: unsupported data type\n");
    }

    return -1.f; // wrong val
}

//--------------------------------------------------------------------------

template <typename T> double GeometricsFPTest::ToDouble(const T &val)
{
    if (std::is_same<T, half>::value)
        return (double)HTF(val);
    else
        return (double)val;
}

//--------------------------------------------------------------------------/

template <typename... Args>
std::string GeometricsFPTest::string_format(const std::string &format,
                                            Args... args)
{
    int sformat = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
    if (sformat <= 0)
        throw std::runtime_error(
            "GeometricsFPTest::string_format: string processing error.");
    auto format_size = static_cast<size_t>(sformat);
    std::unique_ptr<char[]> buffer(new char[format_size]);
    std::snprintf(buffer.get(), format_size, format.c_str(), args...);
    return std::string(buffer.get(), buffer.get() + format_size - 1);
}

//--------------------------------------------------------------------------

template <typename T>
void GeometricsFPTest::vector2string(std::stringstream &sstr, T *vector,
                                     size_t elements)
{
    sstr << "{ ";
    if (std::is_same<T, half>::value == 0)
        for (auto i = 0; i < elements; i++) sstr << vector[i] << ", ";
    else
        for (auto i = 0; i < elements; i++) sstr << HTF(vector[i]) << ", ";
    sstr.seekp(-1, sstr.cur);
    sstr << '}';
}

//--------------------------------------------------------------------------

template <typename T> bool GeometricsFPTest::isnan_fp(const T &v)
{
    if (std::is_same<T, half>::value)
    {
        // Extract FP16 exponent and mantissa
        uint16_t h_exp = (((half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
        uint16_t h_mant = ((half)v) & 0x3FF;

        // NaN test
        return (h_exp == 0x1F && h_mant != 0);
    }
    else
    {
#if !defined(_WIN32)
        return std::isnan(v);
#else
        return _isnan(v);
#endif
    }
}

//--------------------------------------------------------------------------

template <typename T> bool GeometricsFPTest::isfinite_fp(const T &v)
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

//--------------------------------------------------------------------------

template <typename T> T GeometricsFPTest::max_fp(const T &lhs, const T &rhs)
{
    if (std::is_same<T, half>::value)
        return cl_half_to_float(lhs) > cl_half_to_float(rhs) ? lhs : rhs;
    else
        return std::fmax(lhs, rhs);
}

//--------------------------------------------------------------------------

template <typename T> T GeometricsFPTest::abs_fp(const T &val)
{
    if (std::is_same<T, half>::value)
        return static_cast<half>(val) & 0x7FFF;
    else
        return std::fabs(val);
}

//--------------------------------------------------------------------------

template <typename T>
T GeometricsFPTest::get_random(float low, float high, const MTdata &d)
{
    if (std::is_same<T, half>::value)
    {
        float t = (float)((double)genrand_int32(d) / (double)0xFFFFFFFF);
        return HFF((1.0f - t) * low + t * high);
    }
    else if (std::is_same<T, float>::value)
    {
        return get_random_float(low, high, d);
    }
    else if (std::is_same<T, double>::value)
    {
        return get_random_double(low, high, d);
    }
}

//--------------------------------------------------------------------------

template <typename T> T GeometricsFPTest::any_value(MTdata d)
{
    if (std::is_same<T, half>::value)
    {
        float t = (float)((double)genrand_int32(d) / (double)0xFFFFFFFF);
        return HFF((1.0f - t) * std::numeric_limits<T>::lowest()
                   + t * std::numeric_limits<T>::max());
    }
    else
    {
        return any_float(d);
    }
}

//--------------------------------------------------------------------------

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    auto test_fixture = T(device, context, queue);

    cl_int error = test_fixture.SetUp(num_elements);
    test_error_ret(error, "Error in test initialization", TEST_FAIL);

    error = test_fixture.Run();
    test_error_ret(error, "Test Failed", TEST_FAIL);

    return TEST_PASS;
}

//--------------------------------------------------------------------------

#endif // _TEST_GEOMETRICS_FP_H
