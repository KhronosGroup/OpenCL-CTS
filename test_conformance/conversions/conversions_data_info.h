//
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
#ifndef CONVERSIONS_DATA_INFO_H
#define CONVERSIONS_DATA_INFO_H

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
extern bool qcom_sat;
extern roundingMode qcom_rm;
#endif

#include "harness/mt19937.h"
#include "harness/rounding_mode.h"

#include <vector>

#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

extern size_t gTypeSizes[kTypeCount];
extern void *gIn;


typedef enum
{
    kUnsaturated = 0,
    kSaturated,

    kSaturationModeCount
} SaturationMode;

struct DataInitInfo
{
    cl_ulong start;
    cl_uint size;
    Type outType;
    Type inType;
    SaturationMode sat;
    RoundingMode round;
    cl_uint threads;

    static std::vector<uint32_t> specialValuesUInt;
    static std::vector<float> specialValuesFloat;
    static std::vector<double> specialValuesDouble;
};

struct DataInitBase : public DataInitInfo
{
    virtual ~DataInitBase() = default;

    explicit DataInitBase(const DataInitInfo &agg): DataInitInfo(agg) {}
    virtual void conv_array(void *out, void *in, size_t n) {}
    virtual void conv_array_sat(void *out, void *in, size_t n) {}
    virtual void init(const cl_uint &, const cl_uint &) {}
};

template <typename InType, typename OutType>
struct DataInfoSpec : public DataInitBase
{
    explicit DataInfoSpec(const DataInitInfo &agg);

    // helpers
    float round_to_int(float f);
    long long round_to_int_and_clamp(double d);

    OutType absolute(const OutType &x);

    // actual conversion of reference values
    void conv(OutType *out, InType *in);
    void conv_sat(OutType *out, InType *in);

    // min/max ranges for output type of data
    std::pair<OutType, OutType> ranges;

    // matrix of clamping ranges for each rounding type
    std::vector<std::pair<InType, InType>> clamp_ranges;

    std::vector<MTdataHolder> mdv;

    void conv_array(void *out, void *in, size_t n) override
    {
        for (size_t i = 0; i < n; i++)
            conv(&((OutType *)out)[i], &((InType *)in)[i]);
    }

    void conv_array_sat(void *out, void *in, size_t n) override
    {
        for (size_t i = 0; i < n; i++)
            conv_sat(&((OutType *)out)[i], &((InType *)in)[i]);
    }

    void init(const cl_uint &, const cl_uint &) override;
    InType clamp(const InType &);
    inline float fclamp(float lo, float v, float hi)
    {
        v = v < lo ? lo : v;
        return v < hi ? v : hi;
    }

    inline double dclamp(double lo, double v, double hi)
    {
        v = v < lo ? lo : v;
        return v < hi ? v : hi;
    }
};

template <typename InType, typename OutType>
DataInfoSpec<InType, OutType>::DataInfoSpec(const DataInitInfo &agg)
    : DataInitBase(agg), mdv(0)
{
    if (std::is_same<cl_float, OutType>::value)
        ranges = std::make_pair(CL_FLT_MIN, CL_FLT_MAX);
    else if (std::is_same<cl_double, OutType>::value)
        ranges = std::make_pair(CL_DBL_MIN, CL_DBL_MAX);
    else if (std::is_same<cl_uchar, OutType>::value)
        ranges = std::make_pair(0, CL_UCHAR_MAX);
    else if (std::is_same<cl_char, OutType>::value)
        ranges = std::make_pair(CL_CHAR_MIN, CL_CHAR_MAX);
    else if (std::is_same<cl_ushort, OutType>::value)
        ranges = std::make_pair(0, CL_USHRT_MAX);
    else if (std::is_same<cl_short, OutType>::value)
        ranges = std::make_pair(CL_SHRT_MIN, CL_SHRT_MAX);
    else if (std::is_same<cl_uint, OutType>::value)
        ranges = std::make_pair(0, CL_UINT_MAX);
    else if (std::is_same<cl_int, OutType>::value)
        ranges = std::make_pair(CL_INT_MIN, CL_INT_MAX);
    else if (std::is_same<cl_ulong, OutType>::value)
        ranges = std::make_pair(0, CL_ULONG_MAX);
    else if (std::is_same<cl_long, OutType>::value)
        ranges = std::make_pair(CL_LONG_MIN, CL_LONG_MAX);

    // clang-format off
    // for readability sake keep this section unformatted
    if (std::is_floating_point<InType>::value)
    { // from float/double
        InType outMin = static_cast<InType>(ranges.first);
        InType outMax = static_cast<InType>(ranges.second);

        InType eps = std::is_same<InType, cl_float>::value ? (InType) FLT_EPSILON : (InType) DBL_EPSILON;
        if (std::is_integral<OutType>::value)
        { // to char/uchar/short/ushort/int/uint/long/ulong
            if (sizeof(OutType)<=sizeof(cl_short))
            { // to char/uchar/short/ushort
                clamp_ranges=
                {{outMin-0.5f, outMax + 0.5f - outMax * 0.5f * eps},
                  {outMin-0.5f, outMax + 0.5f - outMax * 0.5f * eps},
                  {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, outMax-1.f},
                  {outMin-0.0f, outMax - outMax * 0.5f * eps },
                  {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, outMax - outMax * 0.5f * eps}};
            }
            else if (std::is_same<InType, cl_float>::value)
            { // from float
                if (std::is_same<OutType, cl_uint>::value)
                { // to uint
                    clamp_ranges=
                    { {outMin-0.5f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)},
                      {outMin-0.5f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)},
                      {outMin-1.0f+0.5f*eps, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)},
                      {outMin-0.0f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7) },
                      {outMin-1.0f+0.5f*eps, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)}};
                }
                else if (std::is_same<OutType, cl_int>::value)
                { // to int
                    clamp_ranges=
                    { {outMin, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6)},
                      {outMin, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6)},
                      {outMin, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6)},
                      {outMin, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) },
                      {outMin, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6)}};
                }
                else if (std::is_same<OutType, cl_ulong>::value)
                { // to ulong
                    clamp_ranges=
                    {{outMin-0.5f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39)},
                      {outMin-0.5f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39)},
                      {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39)},
                      {outMin-0.0f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) },
                      {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39)}};
                }
                else if (std::is_same<OutType, cl_long>::value)
                { // to long
                    clamp_ranges=
                    { {MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38)},
                      {MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38)},
                      {MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38)},
                      {MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38)},
                      {MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38)}};
                }
            }
            else
            { // from double
                if (std::is_same<OutType, cl_uint>::value)
                { // to uint
                    clamp_ranges=
                    { {outMin-0.5f, outMax + 0.5 - MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31) * eps},
                      {outMin-0.5f, outMax + 0.5 - MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31) * eps},
                      {outMin-1.0f+0.5f*eps, outMax},
                      {outMin-0.0f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp31, 0x1fffffffffffffLL, -21) },
                      {outMin-1.0f+0.5f*eps, MAKE_HEX_DOUBLE(0x1.fffffffffffffp31, 0x1fffffffffffffLL, -21)}};
                }
                else if (std::is_same<OutType, cl_int>::value)
                { // to int
                    clamp_ranges=
                    { {outMin-0.5f, outMax + 0.5 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * eps},
                      {outMin-0.5f, outMax + 0.5 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * eps},
                      {outMin-1.0f+outMax*eps, outMax},
                      {outMin-0.0f, outMax + 1.0 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * eps },
                      {outMin-1.0f+outMax*eps, outMax + 1.0 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * eps}};
                }
                else if (std::is_same<OutType, cl_ulong>::value)
                { // to ulong
                    clamp_ranges=
                    {{outMin-0.5f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11)},
                      {outMin-0.5f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11)},
                      {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11)},
                      {outMin-0.0f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) },
                      {outMin-1.0f+(std::is_signed<OutType>::value?outMax:0.5f)*eps, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11)}};
                }
                else if (std::is_same<OutType, cl_long>::value)
                { // to long
                    clamp_ranges=
                    { {MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10)},
                      {MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10)},
                      {MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10)},
                      {MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10)},
                      {MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10)}};
                }
            }
        }
    }
    // clang-format on
}

template <typename InType, typename OutType>
float DataInfoSpec<InType, OutType>::round_to_int(float f)
{
    static const float magic[2] = { MAKE_HEX_FLOAT(0x1.0p23f, 0x1, 23),
                                    -MAKE_HEX_FLOAT(0x1.0p23f, 0x1, 23) };

    // Round fractional values to integer in round towards nearest mode
    if (fabsf(f) < MAKE_HEX_FLOAT(0x1.0p23f, 0x1, 23))
    {
        volatile float x = f;
        float magicVal = magic[f < 0];

#if defined(__SSE__)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this
        // incorrectly
        __m128 v = _mm_set_ss(x);
        __m128 m = _mm_set_ss(magicVal);
        v = _mm_add_ss(v, m);
        v = _mm_sub_ss(v, m);
        _mm_store_ss((float *)&x, v);
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }
    return f;
}

template <typename InType, typename OutType>
long long DataInfoSpec<InType, OutType>::round_to_int_and_clamp(double f)
{
    static const double magic[2] = { MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52),
                                     MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52) };

    if (f >= -(double)LLONG_MIN) return LLONG_MAX;

    if (f <= (double)LLONG_MIN) return LLONG_MIN;

    // Round fractional values to integer in round towards nearest mode
    if (fabs(f) < MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52))
    {
        volatile double x = f;
        double magicVal = magic[f < 0];
#if defined(__SSE2__) || defined(_MSC_VER)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this
        // incorrectly
        __m128d v = _mm_set_sd(x);
        __m128d m = _mm_set_sd(magicVal);
        v = _mm_add_sd(v, m);
        v = _mm_sub_sd(v, m);
        _mm_store_sd((double *)&x, v);
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }
    return (long long)f;
}

template <typename InType, typename OutType>
OutType DataInfoSpec<InType, OutType>::absolute(const OutType &x)
{
    union {
        cl_uint u;
        OutType f;
    } u;
    u.f = x;
    if (std::is_same<OutType, float>::value)
        u.u &= 0x7fffffff;
    else if (std::is_same<OutType, double>::value)
        u.u &= 0x7fffffffffffffffULL;
    else
        log_error("Unexpected argument type of DataInfoSpec::absolute");

    return u.f;
}

template <typename InType, typename OutType>
void DataInfoSpec<InType, OutType>::conv(OutType *out, InType *in)
{
    if (std::is_same<cl_float, InType>::value)
    {
        cl_float inVal = *in;

        if (std::is_floating_point<OutType>::value)
        {
            *out = (OutType)inVal;
        }
        else if (std::is_same<cl_ulong, OutType>::value)
        {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
            // VS2005 (at least) on x86 uses fistp to store the float as a
            // 64-bit int. However, fistp stores it as a signed int, and some of
            // the test values won't fit into a signed int. (These test values
            // are >= 2^63.) The result on VS2005 is that these end up silently
            // (at least by default settings) clamped to the max lowest ulong.
            cl_float x = round_to_int(inVal);
            if (x >= 9223372036854775808.0f)
            {
                x -= 9223372036854775808.0f;
                ((cl_ulong *)out)[0] = x;
                ((cl_ulong *)out)[0] += 9223372036854775808ULL;
            }
            else
            {
                ((cl_ulong *)out)[0] = x;
            }
#else
            *out = round_to_int(inVal);
#endif
        }
        else if (std::is_same<cl_long, OutType>::value)
        {
            *out = round_to_int_and_clamp(inVal);
        }
        else
            *out = round_to_int(inVal);
    }
    else if (std::is_same<cl_double, InType>::value)
    {
        if (std::is_same<cl_float, OutType>::value)
            *out = (OutType)*in;
        else
            *out = rint(*in);
    }
    else if (std::is_same<cl_ulong, InType>::value
             || std::is_same<cl_long, InType>::value)
    {
        if (std::is_same<cl_double, OutType>::value)
        {
#if defined(_MSC_VER)
            double result;

            if (std::is_same<cl_ulong, InType>::value)
            {
                cl_ulong l = ((cl_ulong *)in)[0];
                cl_long sl = ((cl_long)l < 0) ? (cl_long)((l >> 1) | (l & 1))
                                              : (cl_long)l;
#if defined(_M_X64)
                _mm_store_sd(&result, _mm_cvtsi64_sd(_mm_setzero_pd(), sl));
#else
                result = sl;
#endif
                ((double *)out)[0] =
                    (l == 0 ? 0.0 : (((cl_long)l < 0) ? result * 2.0 : result));
            }
            else
            {
                cl_long l = ((cl_long *)in)[0];
#if defined(_M_X64)
                _mm_store_sd(&result, _mm_cvtsi64_sd(_mm_setzero_pd(), l));
#else
                result = l;
#endif
                ((double *)out)[0] =
                    (l == 0 ? 0.0 : result); // Per IEEE-754-2008 5.4.1, 0's
                                             // always convert to +0.0
            }
#else
            // Use volatile to prevent optimization by Clang compiler
            volatile InType vi = *in;
            *out = (vi == 0 ? 0.0 : static_cast<OutType>(vi));
#endif
        }
        else if (std::is_same<cl_float, OutType>::value)
        {
            cl_float outVal = 0.f;

#if defined(_MSC_VER) && defined(_M_X64)
            float result;
            if (std::is_same<cl_ulong, InType>::value)
            {
                cl_ulong l = ((cl_ulong *)in)[0];
                cl_long sl = ((cl_long)l < 0) ? (cl_long)((l >> 1) | (l & 1))
                                              : (cl_long)l;
                _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), sl));
                outVal = (l == 0 ? 0.0f
                                 : (((cl_long)l < 0) ? result * 2.0f : result));
            }
            else
            {
                cl_long l = ((cl_long *)in)[0];
                _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), l));
                outVal = (l == 0 ? 0.0f : result); // Per IEEE-754-2008 5.4.1,
                                                   // 0's always convert to +0.0
            }
#else
            InType l = ((InType *)in)[0];
#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
            /* ARM VFP doesn't have hardware instruction for converting from
             * 64-bit integer to float types, hence GCC ARM uses the
             * floating-point emulation code despite which -mfloat-abi setting
             * it is. But the emulation code in libgcc.a has only one rounding
             * mode (round to nearest even in this case) and ignores the user
             * rounding mode setting in hardware. As a result setting rounding
             * modes in hardware won't give correct rounding results for type
             * covert from 64-bit integer to float using GCC for ARM compiler so
             * for testing different rounding modes, we need to use alternative
             * reference function. ARM64 does have an instruction, however we
             * cannot guarantee the compiler will use it.  On all ARM
             * architechures use emulation to calculate reference.*/
            if (std::is_same<cl_ulong, InType>::value)
                outVal = qcom_u64_2_f32(l, qcom_sat, qcom_rm);
            else
                outVal = (l == 0 ? 0.0f : qcom_s64_2_f32(l, qcom_sat, qcom_rm));
#else
            outVal = (l == 0 ? 0.0f : (float)l); // Per IEEE-754-2008 5.4.1, 0's
                                                 // always convert to +0.0
#endif
#endif

            *out = outVal;
        }
        else
        {
            *out = (OutType)*in;
        }
    }
    else
    {
        if (std::is_same<cl_float, OutType>::value)
        {
            // Use volatile to prevent optimization by Clang compiler
            volatile InType vi = *in;
            // Per IEEE-754-2008 5.4.1, 0 always converts to +0.0
            *out = (vi == 0 ? 0.0f : vi);
        }
        else if (std::is_same<cl_double, OutType>::value)
        {
            // Per IEEE-754-2008 5.4.1, 0 always converts to +0.0
            *out = (*in == 0 ? 0.0 : *in);
        }
        else
        {
            *out = (OutType)*in;
        }
    }
}

#define CLAMP(_lo, _x, _hi)                                                    \
    ((_x) < (_lo) ? (_lo) : ((_x) > (_hi) ? (_hi) : (_x)))

template <typename InType, typename OutType>
void DataInfoSpec<InType, OutType>::conv_sat(OutType *out, InType *in)
{
    if (std::is_floating_point<InType>::value)
    {
        if (std::is_floating_point<OutType>::value)
        { // in float/double, out float/double
            *out = (OutType)(*in);
        }
        else if ((std::is_same<InType, cl_float>::value)
                 && std::is_same<cl_ulong, OutType>::value)
        {
            cl_float x = round_to_int(*in);

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
            // VS2005 (at least) on x86 uses fistp to store the float as a
            // 64-bit int. However, fistp stores it as a signed int, and some of
            // the test values won't fit into a signed int. (These test values
            // are >= 2^63.) The result on VS2005 is that these end up silently
            // (at least by default settings) clamped to the max lowest ulong.
            if (x >= 18446744073709551616.0f)
            { // 2^64
                *out = 0xFFFFFFFFFFFFFFFFULL;
            }
            else if (x < 0)
            {
                *out = 0;
            }
            else if (x >= 9223372036854775808.0f)
            { // 2^63
                x -= 9223372036854775808.0f;
                *out = x;
                *out += 9223372036854775808ULL;
            }
            else
            {
                *out = x;
            }
#else
            *out = x >= MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64)
                ? 0xFFFFFFFFFFFFFFFFULL
                : x < 0 ? 0 : (OutType)x;
#endif
        }
        else if ((std::is_same<InType, cl_float>::value)
                 && std::is_same<cl_long, OutType>::value)
        {
            cl_float f = round_to_int(*in);
            *out = f >= MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63)
                ? 0x7FFFFFFFFFFFFFFFULL
                : f < MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63)
                    ? 0x8000000000000000LL
                    : (OutType)f;
        }
        else if (std::is_same<InType, cl_double>::value
                 && std::is_same<cl_ulong, OutType>::value)
        {
            InType f = rint(*in);
            *out = f >= MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64)
                ? 0xFFFFFFFFFFFFFFFFULL
                : f < 0 ? 0 : (OutType)f;
        }
        else if (std::is_same<InType, cl_double>::value
                 && std::is_same<cl_long, OutType>::value)
        {
            InType f = rint(*in);
            *out = f >= MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63)
                ? 0x7FFFFFFFFFFFFFFFULL
                : f < MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63)
                    ? 0x8000000000000000LL
                    : (OutType)f;
        }
        else
        { // in float/double, out char/uchar/short/ushort/int/uint
            *out =
                CLAMP(ranges.first, round_to_int_and_clamp(*in), ranges.second);
        }
    }
    else if (std::is_integral<InType>::value
             && std::is_integral<OutType>::value)
    {
        {
            if ((std::is_signed<InType>::value
                 && std::is_signed<OutType>::value)
                || (!std::is_signed<InType>::value
                    && !std::is_signed<OutType>::value))
            {
                if (sizeof(InType) <= sizeof(OutType))
                {
                    *out = (OutType)*in;
                }
                else
                {
                    *out = CLAMP(ranges.first, *in, ranges.second);
                }
            }
            else
            { // mixed signed/unsigned types
                if (sizeof(InType) < sizeof(OutType))
                {
                    *out = (!std::is_signed<InType>::value)
                        ? (OutType)*in
                        : CLAMP(0, *in, ranges.second); // *in < 0 ? 0 : *in
                }
                else
                { // bigger/equal mixed signed/unsigned types - always clamp
                    *out = CLAMP(0, *in, ranges.second);
                }
            }
        }
    }
    else
    { // InType integral, OutType floating
        *out = std::is_signed<InType>::value ? (OutType)*in
                                             : absolute((OutType)*in);
    }
}

template <typename InType, typename OutType>
void DataInfoSpec<InType, OutType>::init(const cl_uint &job_id,
                                         const cl_uint &thread_id)
{
    uint64_t ulStart = start;
    void *pIn = (char *)gIn + job_id * size * gTypeSizes[inType];

    if (std::is_integral<InType>::value)
    {
        InType *o = (InType *)pIn;
        if (sizeof(InType) <= sizeof(cl_short))
        { // char/uchar/ushort/short
            for (int i = 0; i < size; i++) o[i] = ulStart++;
        }
        else if (sizeof(InType) <= sizeof(cl_int))
        { // int/uint
            int i = 0;
            if (gIsEmbedded)
                for (i = 0; i < size; i++)
                    o[i] = (InType)genrand_int32(mdv[thread_id]);
            else
                for (i = 0; i < size; i++) o[i] = (InType)i + ulStart;

            if (0 == ulStart)
            {
                size_t tableSize = specialValuesUInt.size()
                    * sizeof(decltype(specialValuesUInt)::value_type);
                if (sizeof(InType) * size < tableSize)
                    tableSize = sizeof(InType) * size;
                memcpy((char *)(o + i) - tableSize, &specialValuesUInt.front(),
                       tableSize);
            }
        }
        else
        { // long/ulong
            cl_ulong *o = (cl_ulong *)pIn;
            cl_ulong i, j, k;

            i = 0;
            if (ulStart == 0)
            {
                // Try various powers of two
                for (j = 0; j < (cl_ulong)size && j < 8 * sizeof(cl_ulong); j++)
                    o[j] = (cl_ulong)1 << j;
                i = j;

                // try the complement of those
                for (j = 0; i < (cl_ulong)size && j < 8 * sizeof(cl_ulong); j++)
                    o[i++] = ~((cl_ulong)1 << j);

                // Try various negative powers of two
                for (j = 0; i < (cl_ulong)size && j < 8 * sizeof(cl_ulong); j++)
                    o[i++] = (cl_ulong)0xFFFFFFFFFFFFFFFEULL << j;

                // try various powers of two plus 1, shifted by various amounts
                for (j = 0; i < (cl_ulong)size && j < 8 * sizeof(cl_ulong); j++)
                    for (k = 0;
                         i < (cl_ulong)size && k < 8 * sizeof(cl_ulong) - j;
                         k++)
                        o[i++] = (((cl_ulong)1 << j) + 1) << k;

                // try various powers of two minus 1
                for (j = 0; i < (cl_ulong)size && j < 8 * sizeof(cl_ulong); j++)
                    for (k = 0;
                         i < (cl_ulong)size && k < 8 * sizeof(cl_ulong) - j;
                         k++)
                        o[i++] = (((cl_ulong)1 << j) - 1) << k;

                // Other patterns
                cl_ulong pattern[] = {
                    0x3333333333333333ULL, 0x5555555555555555ULL,
                    0x9999999999999999ULL, 0x6666666666666666ULL,
                    0xccccccccccccccccULL, 0xaaaaaaaaaaaaaaaaULL
                };
                cl_ulong mask[] = { 0xffffffffffffffffULL,
                                    0xff00ff00ff00ff00ULL,
                                    0xffff0000ffff0000ULL,
                                    0xffffffff00000000ULL };
                for (j = 0; i < (cl_ulong)size
                     && j < sizeof(pattern) / sizeof(pattern[0]);
                     j++)
                    for (k = 0; i + 2 <= (cl_ulong)size
                         && k < sizeof(mask) / sizeof(mask[0]);
                         k++)
                    {
                        o[i++] = pattern[j] & mask[k];
                        o[i++] = pattern[j] & ~mask[k];
                    }
            }

            auto &md = mdv[thread_id];
            for (; i < (cl_ulong)size; i++)
                o[i] = (cl_ulong)genrand_int32(md)
                    | ((cl_ulong)genrand_int32(md) << 32);
        }
    } // integrals
    else if (std::is_same<InType, cl_float>::value)
    {
        cl_uint *o = (cl_uint *)pIn;
        int i;

        if (gIsEmbedded)
            for (i = 0; i < size; i++)
                o[i] = (cl_uint)genrand_int32(mdv[thread_id]);
        else
            for (i = 0; i < size; i++) o[i] = (cl_uint)i + ulStart;

        if (0 == ulStart)
        {
            size_t tableSize = specialValuesFloat.size()
                * sizeof(decltype(specialValuesFloat)::value_type);
            if (sizeof(InType) * size < tableSize)
                tableSize = sizeof(InType) * size;
            memcpy((char *)(o + i) - tableSize, &specialValuesFloat.front(),
                   tableSize);
        }

        if (kUnsaturated == sat)
        {
            InType *f = (InType *)pIn;
            for (i = 0; i < size; i++) f[i] = clamp(f[i]);
        }
    }
    else if (std::is_same<InType, cl_double>::value)
    {
        InType *o = (InType *)pIn;
        int i = 0;

        union {
            uint64_t u;
            InType d;
        } u;

        for (i = 0; i < size; i++)
        {
            uint64_t z = i + ulStart;

            uint32_t bits = ((uint32_t)z ^ (uint32_t)(z >> 32));
            // split 0x89abcdef to 0x89abc00000000def
            u.u = bits & 0xfffU;
            u.u |= (uint64_t)(bits & ~0xfffU) << 32;
            // sign extend the leading bit of def segment as sign bit so that
            // the middle region consists of either all 1s or 0s
            u.u -= (bits & 0x800U) << 1;
            o[i] = u.d;
        }

        if (0 == ulStart)
        {
            size_t tableSize = specialValuesDouble.size()
                * sizeof(decltype(specialValuesDouble)::value_type);
            if (sizeof(InType) * size < tableSize)
                tableSize = sizeof(InType) * size;
            memcpy((char *)(o + i) - tableSize, &specialValuesDouble.front(),
                   tableSize);
        }

        if (0 == sat)
            for (i = 0; i < size; i++) o[i] = clamp(o[i]);
    }
}

template <typename InType, typename OutType>
InType DataInfoSpec<InType, OutType>::clamp(const InType &in)
{
    if (std::is_integral<OutType>::value)
    {
        if (std::is_same<InType, cl_float>::value)
        {
            return fclamp(clamp_ranges[round].first, in,
                          clamp_ranges[round].second);
        }
        else if (std::is_same<InType, cl_double>::value)
        {
            return dclamp(clamp_ranges[round].first, in,
                          clamp_ranges[round].second);
        }
    }
    return in;
}

#endif /* CONVERSIONS_DATA_INFO_H */
