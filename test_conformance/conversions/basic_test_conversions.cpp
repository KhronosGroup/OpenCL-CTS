//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include "harness/rounding_mode.h"
#include "harness/ThreadPool.h"
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/mt19937.h"
#include "harness/kernelHelpers.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

#if defined(__MINGW32__)
#include <sys/param.h>
#endif

#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include <algorithm>

#include <vector>
#include <type_traits>

#include "basic_test_conversions.h"

#if (defined(_WIN32) && defined(_MSC_VER))
// need for _controlfp_s and rouinding modes in RoundingMode
#include "harness/testHarness.h"
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
extern bool qcom_sat;
extern roundingMode qcom_rm;
#endif

cl_context gContext = NULL;
cl_command_queue gQueue = NULL;
int gStartTestNumber = -1;
int gEndTestNumber = 0;
#if defined(__APPLE__)
int gTimeResults = 1;
#else
int gTimeResults = 0;
#endif
int gReportAverageTimes = 0;
void *gIn = NULL;
void *gRef = NULL;
void *gAllowZ = NULL;
void *gOut[kCallStyleCount] = { NULL };
cl_mem gInBuffer;
cl_mem gOutBuffers[kCallStyleCount];
size_t gComputeDevices = 0;
uint32_t gDeviceFrequency = 0;
int gWimpyMode = 0;
int gWimpyReductionFactor = 128;
int gSkipTesting = 0;
int gForceFTZ = 0;
int gIsRTZ = 0;
uint32_t gSimdSize = 1;
int gHasDouble = 0;
int gTestDouble = 1;
int gHasHalfs = 0;
int gTestHalfs = 1;
const char *sizeNames[] = { "", "", "2", "3", "4", "8", "16" };
int vectorSizes[] = { 1, 1, 2, 3, 4, 8, 16 };
int gMinVectorSize = 0;
int gMaxVectorSize = sizeof(vectorSizes) / sizeof(vectorSizes[0]);
MTdata gMTdata;
const char **argList = NULL;
int argCount = 0;

////////////////////////////////////////////////////////////////////////////////////////

static cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                              RoundingMode round, int vectorSize,
                              cl_kernel *outKernel);
static int RunKernel(cl_kernel kernel, void *inBuf, void *outBuf,
                     size_t blockCount);

static int GetTestCase(const char *name, Type *outType, Type *inType,
                       SaturationMode *sat, RoundingMode *round);

////////////////////////////////////////////////////////////////////////////////////////

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p);
cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p);
uint64_t GetTime(void);
double SubtractTime(uint64_t endTime, uint64_t startTime);
void WriteInputBufferComplete(void *);
void *FlushToZero(void);
void UnFlushToZero(void *);

////////////////////////////////////////////////////////////////////////////////////////

cl_half_rounding_mode ConversionsTest::halfRoundingMode = CL_HALF_RTE;
cl_half_rounding_mode ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTE;
#define HFF(num) cl_half_from_float(num, ConversionsTest::halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

////////////////////////////////////////////////////////////////////////////////////////

// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
static inline void Force64BitFPUPrecision(void)
{
#if __MINGW32__
    // The usual method is to use _controlfp as follows:
    //     #include <float.h>
    //     _controlfp(_PC_64, _MCW_PC);
    //
    // _controlfp is available on MinGW32 but not on MinGW64. Instead of having
    // divergent code just use inline assembly which works for both.
    unsigned short int orig_cw = 0;
    unsigned short int new_cw = 0;
    __asm__ __volatile__("fstcw %0" : "=m"(orig_cw));
    new_cw = orig_cw | 0x0300; // set precision to 64-bit
    __asm__ __volatile__("fldcw  %0" ::"m"(new_cw));
#else
    /* Implement for other platforms if needed */
#endif
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
DataInfoSpec<InType, OutType, InFP, OutFP>::DataInfoSpec(
    const DataInitInfo &agg)
    : DataInitBase(agg)
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

    InType outMin = ((InType)ranges.first);
    InType outMax = ((InType)ranges.second);

    // clang-format off
    // for readability sake keep this section unformatted
    if (std::is_floating_point<InType>::value)
    { // from float/double
        InType eps = std::is_same<InType, cl_float>::value ? (InType) FLT_EPSILON : (InType) DBL_EPSILON;
        if (std::is_integral<OutType>::value)
        { // to char/uchar/short/ushort/half/int/uint/long/ulong
            if (sizeof(OutType)<=sizeof(cl_short))
            { // to char/uchar/short/ushort/half
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

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
float DataInfoSpec<InType, OutType, InFP, OutFP>::round_to_int(float f)
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

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
long long
DataInfoSpec<InType, OutType, InFP, OutFP>::round_to_int_and_clamp(double f)
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

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
OutType DataInfoSpec<InType, OutType, InFP, OutFP>::absolute(const OutType &x)
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

//////////////////////////////////////////////////////////////////////////////////////////

template <typename T, bool fp> constexpr bool is_half()
{
    return (std::is_same<cl_half, T>::value && fp);
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
void DataInfoSpec<InType, OutType, InFP, OutFP>::conv(OutType *out, InType *in)
{
    if (std::is_same<cl_float, InType>::value || is_in_half())
    {
        cl_float inVal = *in;
        if (std::is_same<cl_half, InType>::value)
        {
            inVal = HTF(*in);
        }

        if (std::is_same<cl_double, OutType>::value)
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
        else if (is_out_half())
            *out = HFF(*in);
        else
            *out = rint(*in);
    }
    else if (std::is_same<cl_ulong, InType>::value
             || std::is_same<cl_long, InType>::value)
    {
        if (std::is_same<cl_double, OutType>::value)
        {
#if defined(_MSC_VER)
            InType l = ((InType *)in)[0];
            double result;

            if (std::is_same<cl_ulong, InType>::value)
            {
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
                _mm_store_sd(&result, _mm_cvtsi64_sd(_mm_setzero_pd(), l));
                ((double *)out)[0] =
                    (l == 0 ? 0.0 : result); // Per IEEE-754-2008 5.4.1, 0's
                                             // always convert to +0.0
            }
#else
            *out = (*in == 0 ? 0.0 : (OutType)*in);
#endif
        }
        else if (std::is_same<cl_float, OutType>::value || is_out_half())
        {
            InType l = ((InType *)in)[0];
            cl_float outVal = 0.f;

#if defined(_MSC_VER) && defined(_M_X64)
            float result;
            if (std::is_same<cl_ulong, InType>::value)
            {
                cl_long sl = ((cl_long)l < 0) ? (cl_long)((l >> 1) | (l & 1))
                                              : (cl_long)l;
                _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), sl));
                outVal = (l == 0 ? 0.0f
                                 : (((cl_long)l < 0) ? result * 2.0f : result));
            }
            else
            {
                _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), l));
                outVal = (l == 0 ? 0.0f : result); // Per IEEE-754-2008 5.4.1,
                                                   // 0's always convert to +0.0
            }
#else
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

            *out = std::is_same<cl_half, OutType>::value ? HFF(outVal) : outVal;
        }
        else
        {
            *out = (OutType)*in;
        }
    }
    else
    {
        if (std::is_same<cl_float, OutType>::value)
            *out = (*in == 0 ? 0.f : *in); // Per IEEE-754-2008 5.4.1, 0's
                                           // always convert to +0.0
        else if (std::is_same<cl_double, OutType>::value)
            *out = (*in == 0 ? 0.0 : *in);
        else if (is_out_half())
            *out = HFF(*in == 0 ? 0.f : *in);
        else
            *out = (OutType)*in;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

#define CLAMP(_lo, _x, _hi)                                                    \
    ((_x) < (_lo) ? (_lo) : ((_x) > (_hi) ? (_hi) : (_x)))

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
void DataInfoSpec<InType, OutType, InFP, OutFP>::conv_sat(OutType *out,
                                                          InType *in)
{
    if (std::is_floating_point<InType>::value || is_in_half())
    {
        cl_float inVal = *in;
        if (is_in_half()) inVal = HTF(*in);

        if (std::is_floating_point<OutType>::value || is_out_half())
        { // in half/float/double, out half/float/double
            if (is_out_half())
                *out = HFF(inVal);
            else
                *out = (OutType)(is_in_half() ? inVal : *in);
        }
        else if ((std::is_same<InType, cl_float>::value || is_in_half())
                 && std::is_same<cl_ulong, OutType>::value)
        {
            cl_float x = round_to_int(is_in_half() ? HTF(*in) : *in);

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
        else if ((std::is_same<InType, cl_float>::value || is_in_half())
                 && std::is_same<cl_long, OutType>::value)
        {
            cl_float f = round_to_int(is_in_half() ? HTF(*in) : *in);
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
        { // in half/float/double, out char/uchar/short/ushort/int/uint
            *out = CLAMP(ranges.first,
                         round_to_int_and_clamp(is_in_half() ? inVal : *in),
                         ranges.second);
        }
    }
    else if (std::is_integral<InType>::value
             && std::is_integral<OutType>::value)
    {
        if (is_out_half())
        {
            *out = std::is_signed<InType>::value
                ? HFF((cl_float)*in)
                : absolute((OutType)HFF((cl_float)*in));
        }
        else
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

////////////////////////////////////////////////////////////////////////////////////////

// clang-format off
// for readability sake keep this section unformatted
static const unsigned int specialValuesUInt[] = {
    INT_MIN, INT_MIN + 1, INT_MIN + 2,
    -(1<<30)-3,-(1<<30)-2,-(1<<30)-1, -(1<<30), -(1<<30)+1, -(1<<30)+2, -(1<<30)+3,
    -(1<<24)-3,-(1<<24)-2,-(1<<24)-1, -(1<<24), -(1<<24)+1, -(1<<24)+2, -(1<<24)+3,
    -(1<<23)-3,-(1<<23)-2,-(1<<23)-1, -(1<<23), -(1<<23)+1, -(1<<23)+2, -(1<<23)+3,
    -(1<<22)-3,-(1<<22)-2,-(1<<22)-1, -(1<<22), -(1<<22)+1, -(1<<22)+2, -(1<<22)+3,
    -(1<<21)-3,-(1<<21)-2,-(1<<21)-1, -(1<<21), -(1<<21)+1, -(1<<21)+2, -(1<<21)+3,
    -(1<<16)-3,-(1<<16)-2,-(1<<16)-1, -(1<<16), -(1<<16)+1, -(1<<16)+2, -(1<<16)+3,
    -(1<<15)-3,-(1<<15)-2,-(1<<15)-1, -(1<<15), -(1<<15)+1, -(1<<15)+2, -(1<<15)+3,
    -(1<<8)-3,-(1<<8)-2,-(1<<8)-1, -(1<<8), -(1<<8)+1, -(1<<8)+2, -(1<<8)+3,
    -(1<<7)-3,-(1<<7)-2,-(1<<7)-1, -(1<<7), -(1<<7)+1, -(1<<7)+2, -(1<<7)+3,
    -4, -3, -2, -1, 0, 1, 2, 3, 4,
    (1<<7)-3,(1<<7)-2,(1<<7)-1, (1<<7), (1<<7)+1, (1<<7)+2, (1<<7)+3,
    (1<<8)-3,(1<<8)-2,(1<<8)-1, (1<<8), (1<<8)+1, (1<<8)+2, (1<<8)+3,
    (1<<15)-3,(1<<15)-2,(1<<15)-1, (1<<15), (1<<15)+1, (1<<15)+2, (1<<15)+3,
    (1<<16)-3,(1<<16)-2,(1<<16)-1, (1<<16), (1<<16)+1, (1<<16)+2, (1<<16)+3,
    (1<<21)-3,(1<<21)-2,(1<<21)-1, (1<<21), (1<<21)+1, (1<<21)+2, (1<<21)+3,
    (1<<22)-3,(1<<22)-2,(1<<22)-1, (1<<22), (1<<22)+1, (1<<22)+2, (1<<22)+3,
    (1<<23)-3,(1<<23)-2,(1<<23)-1, (1<<23), (1<<23)+1, (1<<23)+2, (1<<23)+3,
    (1<<24)-3,(1<<24)-2,(1<<24)-1, (1<<24), (1<<24)+1, (1<<24)+2, (1<<24)+3,
    (1<<30)-3,(1<<30)-2,(1<<30)-1, (1<<30), (1<<30)+1, (1<<30)+2, (1<<30)+3,
    INT_MAX-3, INT_MAX-2, INT_MAX-1, INT_MAX, // 0x80000000, 0x80000001 0x80000002 already covered above
    UINT_MAX-3, UINT_MAX-2, UINT_MAX-1, UINT_MAX
};

static const float specialValuesFloat[] = {
    -NAN, -INFINITY, -FLT_MAX,
    MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64), MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),
    MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8), MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32), MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7),
    MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7), MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31), MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6),
    -1000.f, -100.f, -4.0f, -3.5f, -3.0f,
    MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23), -2.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23), -2.0f,
    MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24), -1.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24), MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24), -1.0f,
    MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25), MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25), -0.5f,
    MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26), MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26), -0.25f,
    MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27), MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150), -FLT_MIN,
    MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150),
    MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150), MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150), MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150), MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150),
    MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150), MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150),
    MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150), -0.0f, +NAN, +INFINITY, +FLT_MAX,
    MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40), MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64), MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39),
    MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39), MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63), MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8), MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32), MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7),
    MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7), MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31), MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6),
    +1000.f, +100.f, +4.0f, +3.5f, +3.0f,
    MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23), 2.5f, MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23), +2.0f,
    MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24), 1.5f, MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24),
    MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24), +1.0f, MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25), +0.5f, MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26),
    MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26), +0.25f, MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150), +FLT_MIN, MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150),
    MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150), MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150), MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150), MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150),
    MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150), MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150),
    MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150), +0.0f
};

// A table of more difficult cases to get right
static const double specialValuesDouble[] = {
    -NAN, -INFINITY, -DBL_MAX,
    MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12), MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11), MAKE_HEX_DOUBLE(-0x1.80000000000001p64, -0x180000000000001LL, 8),
    MAKE_HEX_DOUBLE(-0x1.8p64, -0x18LL, 60), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp64, -0x17ffffffffffffLL, 12),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p63, -0x180000000000001LL, 7), MAKE_HEX_DOUBLE(-0x1.8p63, -0x18LL, 59),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp63, -0x17ffffffffffffLL, 11), MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p32, -0x180000000000001LL, -24), MAKE_HEX_DOUBLE(-0x1.8p32, -0x18LL, 28),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp32, -0x17ffffffffffffLL, -20), MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8),
    MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p31, -0x180000000000001LL, -25), MAKE_HEX_DOUBLE(-0x1.8p31, -0x18LL, 27),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp31, -0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22),
    -1000., -100., -4.0, -3.5, -3.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51), -2.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51), -2.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52), -1.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52), MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), -1.0,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53), MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53), -0.5,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54), MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54), -0.25,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55), MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074),
    -DBL_MIN,
    MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074),
    -0.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(0x1.80000000000001p63, 0x180000000000001LL, 7), MAKE_HEX_DOUBLE(0x1.8p63, 0x18LL, 59),
    MAKE_HEX_DOUBLE(0x1.7ffffffffffffp63, 0x17ffffffffffffLL, 11), MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.80000000000001p32, +0x180000000000001LL, -24), MAKE_HEX_DOUBLE(+0x1.8p32, +0x18LL, 28),
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp32, +0x17ffffffffffffLL, -20), MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8),
    MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(+0x1.80000000000001p31, +0x180000000000001LL, -25), MAKE_HEX_DOUBLE(+0x1.8p31, +0x18LL, 27),
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp31, +0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22),
    +1000., +100., +4.0, +3.5, +3.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51), +2.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51), +2.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52),
    +1.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52), MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    +1.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53), MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53),
    +0.5, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54), MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54),
    +0.25, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55), MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074),
    +DBL_MIN, MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074), +0.0, MAKE_HEX_DOUBLE(-0x1.ffffffffffffep62, -0x1ffffffffffffeLL, 10),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp62, -0x1ffffffffffffcLL, 10), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep62, +0x1ffffffffffffeLL, 10), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp62, +0x1ffffffffffffcLL, 10),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep51, -0x1ffffffffffffeLL, -1),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp51, -0x1ffffffffffffcLL, -1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp51, -0x1fffffffffffffLL, -1),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep51, +0x1ffffffffffffeLL, -1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp51, +0x1ffffffffffffcLL, -1),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp51, +0x1fffffffffffffLL, -1), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep52, -0x1ffffffffffffeLL, 0),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp52, -0x1ffffffffffffcLL, 0), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp52, -0x1fffffffffffffLL, 0),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep52, +0x1ffffffffffffeLL, 0), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp52, +0x1ffffffffffffcLL, 0),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp52, +0x1fffffffffffffLL, 0), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep53, -0x1ffffffffffffeLL, 1),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp53, -0x1ffffffffffffcLL, 1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp53, -0x1fffffffffffffLL, 1),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep53, +0x1ffffffffffffeLL, 1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp53, +0x1ffffffffffffcLL, 1),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp53, +0x1fffffffffffffLL, 1), MAKE_HEX_DOUBLE(-0x1.0000000000002p52, -0x10000000000002LL, 0),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p52, -0x10000000000001LL, 0), MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p52, +0x10000000000002LL, 0), MAKE_HEX_DOUBLE(+0x1.0000000000001p52, +0x10000000000001LL, 0),
    MAKE_HEX_DOUBLE(+0x1.0p52, +0x1LL, 52), MAKE_HEX_DOUBLE(-0x1.0000000000002p53, -0x10000000000002LL, 1),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p53, -0x10000000000001LL, 1), MAKE_HEX_DOUBLE(-0x1.0p53, -0x1LL, 53),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p53, +0x10000000000002LL, 1), MAKE_HEX_DOUBLE(+0x1.0000000000001p53, +0x10000000000001LL, 1),
    MAKE_HEX_DOUBLE(+0x1.0p53, +0x1LL, 53), MAKE_HEX_DOUBLE(-0x1.0000000000002p54, -0x10000000000002LL, 2),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p54, -0x10000000000001LL, 2), MAKE_HEX_DOUBLE(-0x1.0p54, -0x1LL, 54),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p54, +0x10000000000002LL, 2), MAKE_HEX_DOUBLE(+0x1.0000000000001p54, +0x10000000000001LL, 2),
    MAKE_HEX_DOUBLE(+0x1.0p54, +0x1LL, 54), MAKE_HEX_DOUBLE(-0x1.fffffffefffffp62, -0x1fffffffefffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.ffffffffp62, -0x1ffffffffLL, 30), MAKE_HEX_DOUBLE(-0x1.ffffffff00001p62, -0x1ffffffff00001LL, 10),
    MAKE_HEX_DOUBLE(0x1.fffffffefffffp62, 0x1fffffffefffffLL, 10), MAKE_HEX_DOUBLE(0x1.ffffffffp62, 0x1ffffffffLL, 30),
    MAKE_HEX_DOUBLE(0x1.ffffffff00001p62, 0x1ffffffff00001LL, 10),
};
// clang-format on

////////////////////////////////////////////////////////////////////////////////////////

cl_ulong random64(MTdata d)
{
    return (cl_ulong)genrand_int32(d) | ((cl_ulong)genrand_int32(d) << 32);
}

////////////////////////////////////////////////////////////////////////////////////////
template <typename InType, typename OutType, bool InFP, bool OutFP>
void DataInfoSpec<InType, OutType, InFP, OutFP>::init(const cl_uint &job_id,
                                                      const cl_uint &thread_id)
{
    uint64_t start = start + job_id * size;
    void *pIn = (char *)gIn + job_id * size * gTypeSizes[inType];

    if (std::is_integral<InType>::value || is_in_half())
    {
        InType *o = (InType *)pIn;
        if (sizeof(InType) <= sizeof(cl_short))
        { // char/uchar/ushort/short/half
            for (int i = 0; i < size; i++) o[i] = start++;
        }
        else if (sizeof(InType) <= sizeof(cl_int))
        { // int/uint
            int i = 0;
            if (gIsEmbedded)
                for (i = 0; i < size; i++)
                    o[i] = (InType)genrand_int32(d[thread_id]);
            else
                for (i = 0; i < size; i++) o[i] = (InType)i + start;

            if (0 == start)
            {
                size_t tableSize = sizeof(specialValuesUInt);
                if (sizeof(InType) * size < tableSize)
                    tableSize = sizeof(InType) * size;
                memcpy((char *)(o + i) - tableSize, specialValuesUInt,
                       tableSize);
            }
        }
        else
        { // long/ulong
            cl_ulong *o = (cl_ulong *)pIn;
            cl_ulong i, j, k;

            i = 0;
            if (start == 0)
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

            for (; i < (cl_ulong)size; i++) o[i] = random64(d[thread_id]);
        }
    } // integrals
    else if (std::is_same<InType, cl_float>::value)
    {
        cl_uint *o = (cl_uint *)pIn;
        int i;

        if (gIsEmbedded)
            for (i = 0; i < size; i++)
                o[i] = (cl_uint)genrand_int32(d[thread_id]);
        else
            for (i = 0; i < size; i++) o[i] = (cl_uint)i + start;

        if (0 == start)
        {
            size_t tableSize = sizeof(specialValuesFloat);
            if (sizeof(InType) * size < tableSize)
                tableSize = sizeof(InType) * size;
            memcpy((char *)(o + i) - tableSize, specialValuesFloat, tableSize);
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
            uint64_t z = i + start;

            uint32_t bits = ((uint32_t)z ^ (uint32_t)(z >> 32));
            // split 0x89abcdef to 0x89abc00000000def
            u.u = bits & 0xfffU;
            u.u |= (uint64_t)(bits & ~0xfffU) << 32;
            // sign extend the leading bit of def segment as sign bit so that
            // the middle region consists of either all 1s or 0s
            u.u -= (bits & 0x800U) << 1;
            o[i] = u.d;
        }

        if (0 == start)
        {
            size_t tableSize = sizeof(specialValuesDouble);
            if (sizeof(InType) * size < tableSize)
                tableSize = sizeof(InType) * size;
            memcpy((char *)(o + i) - tableSize, specialValuesDouble, tableSize);
        }

        if (0 == sat)
            for (i = 0; i < size; i++) o[i] = clamp(o[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////

static inline float fclamp(float lo, float v, float hi)
    __attribute__((always_inline));
static inline double dclamp(double lo, double v, double hi)
    __attribute__((always_inline));

static inline float fclamp(float lo, float v, float hi)
{
    v = v < lo ? lo : v;
    return v < hi ? v : hi;
}
static inline double dclamp(double lo, double v, double hi)
{
    v = v < lo ? lo : v;
    return v < hi ? v : hi;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
InType DataInfoSpec<InType, OutType, InFP, OutFP>::clamp(const InType &in)
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

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
int CalcRefValsPat<InType, OutType, InFP, OutFP>::check_result(void *test,
                                                               uint32_t count,
                                                               int vectorSize)
{
    const cl_uchar *a = (const cl_uchar *)gAllowZ;

    if (std::is_integral<OutType>::value || is_half<OutType, OutFP>())
    { // char/uchar/short/ushort/half/int/uint/long/ulong
        const OutType *t = (const OutType *)test;
        const OutType *c = (const OutType *)gRef;
        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (OutType)0))
            {
                size_t s = sizeof(OutType) * 2;
                std::stringstream sstr;
                sstr << "\nError for vector size %d found at 0x%8.8x:  *0x%"
                     << s << "." << s << "x vs 0x%" << s << "." << s << "x\n";
                vlog(sstr.str().c_str(), vectorSize, i, c[i], t[i]);
                return i + 1;
            }
    }
    else if (std::is_same<OutType, cl_float>::value)
    {
        // cast to integral - from original test
        const cl_uint *t = (const cl_uint *)test;
        const cl_uint *c = (const cl_uint *)gRef;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !((t[i] & 0x7fffffffU) > 0x7f800000U
                  && (c[i] & 0x7fffffffU) > 0x7f800000U)
                && !(a[i] != (cl_uchar)0 && t[i] == (c[i] & 0x80000000U)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)gRef)[i], ((OutType *)test)[i]);
                return i + 1;
            }
    }
    else
    {
        const cl_ulong *t = (const cl_ulong *)test;
        const cl_ulong *c = (const cl_ulong *)gRef;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !((t[i] & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL
                  && (c[i] & 0x7fffffffffffffffULL) > 0x7f80000000000000ULL)
                && !(a[i] != (cl_uchar)0
                     && t[i] == (c[i] & 0x8000000000000000ULL)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)gRef)[i], ((OutType *)test)[i]);
                return i + 1;
            }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////

cl_uint RoundUpToNextPowerOfTwo(cl_uint x)
{
    if (0 == (x & (x - 1))) return x;

    while (x & (x - 1)) x &= x - 1;

    return x + x;
}

////////////////////////////////////////////////////////////////////////////////////////

cl_int CustomConversionsTest::Run()
{
    int startMinVectorSize = gMinVectorSize;
    Type inType, outType;
    RoundingMode round;
    SaturationMode sat;

    for (int i = 0; i < argCount; i++)
    {
        if (GetTestCase(argList[i], &outType, &inType, &sat, &round))
        {
            vlog_error("\n\t\t**** ERROR:  Unable to parse function name "
                       "%s.  Skipping....  *****\n\n",
                       argList[i]);
            continue;
        }

        // skip double if we don't have it
        if (!gTestDouble && (inType == kdouble || outType == kdouble))
        {
            if (gHasDouble)
            {
                vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                           gTypeNames[outType], gSaturationNames[sat],
                           gRoundingModeNames[round], gTypeNames[inType]);
                vlog("\t\tcl_khr_fp64 enabled, but double testing turned "
                     "off.\n");
            }
            continue;
        }


        // skip double if we don't have it
        if (!gTestHalfs && (inType == khalf || outType == khalf))
        {
            if (gHasHalfs)
            {
                vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                           gTypeNames[outType], gSaturationNames[sat],
                           gRoundingModeNames[round], gTypeNames[inType]);
                vlog("\t\tcl_khr_fp64 enabled, but double testing turned "
                     "off.\n");
            }
            continue;
        }

        // skip longs on embedded
        if (!gHasLong
            && (inType == klong || outType == klong || inType == kulong
                || outType == kulong))
        {
            continue;
        }

        // Skip the implicit converts if the rounding mode is not default or
        // test is saturated
        if (0 == startMinVectorSize)
        {
            if (sat || round != kDefaultRoundingMode)
                gMinVectorSize = 1;
            else
                gMinVectorSize = 0;
        }

        IterOverSelectedTypes iter(typeIterator, *this, inType, outType);

        iter.Run();

        if (iter.GetFailCount())
        {
            vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                       gTypeNames[outType], gSaturationNames[sat],
                       gRoundingModeNames[round], gTypeNames[inType]);
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

////////////////////////////////////////////////////////////////////////////////////////

ConversionsTest::ConversionsTest(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
    : device(device), context(context), queue(queue),
      typeIterator({ cl_uchar(0), cl_char(0), cl_ushort(0), cl_short(0),
                     cl_uint(0), cl_int(0), cl_half(0), cl_float(0),
                     cl_double(0), cl_ulong(0), cl_long(0) })
{}

////////////////////////////////////////////////////////////////////////////////////////

cl_int ConversionsTest::Run()
{
    IterOverTypes iter(typeIterator, *this);

    iter.Run();

    return iter.GetFailCount();
}

////////////////////////////////////////////////////////////////////////////////////////

cl_int ConversionsTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            ConversionsTest::halfRoundingMode = CL_HALF_RTE;
            ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            ConversionsTest::halfRoundingMode = CL_HALF_RTZ;
            ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTZ;
        }
        else // CL_FP_ROUND_TO_INF ??
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }
    }

    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
cl_int ConversionsTest::TestTypesConversion(const Type &inType,
                                            const Type &outType,
                                            int &testNumber)
{
    SaturationMode sat;
    RoundingMode round;
    int error;
    int startMinVectorSize = gMinVectorSize;

    // skip longs on embedded
    if (!gHasLong
        && (inType == klong || outType == klong || inType == kulong
            || outType == kulong))
    {
        return CL_SUCCESS;
    }

    for (sat = (SaturationMode)0; sat < kSaturationModeCount;
         sat = (SaturationMode)(sat + 1))
    {
        // skip illegal saturated conversions to float type
        if (kSaturated == sat && (outType == kfloat || outType == kdouble))
        {
            continue;
        }

        for (round = (RoundingMode)0; round < kRoundingModeCount;
             round = (RoundingMode)(round + 1))
        {
            if (++testNumber < gStartTestNumber)
            {
                continue;
            }
            else
            {
                if (gEndTestNumber > 0 && testNumber >= gEndTestNumber)
                    return gFailCount;
            }

            vlog("%d) Testing convert_%sn%s%s( %sn ):\n", testNumber,
                 gTypeNames[outType], gSaturationNames[sat],
                 gRoundingModeNames[round], gTypeNames[inType]);

            // skip double if we don't have it
            if (!gTestDouble && (inType == kdouble || outType == kdouble))
            {
                if (gHasDouble)
                {
                    vlog_error("\t *** %d) convert_%sn%s%s( %sn ) "
                               "FAILED ** \n",
                               testNumber, gTypeNames[outType],
                               gSaturationNames[sat], gRoundingModeNames[round],
                               gTypeNames[inType]);
                    vlog("\t\tcl_khr_fp64 enabled, but double "
                         "testing turned off.\n");
                }
                continue;
            }

            // skip double if we don't have it
            if (!gTestHalfs && (inType == khalf || outType == khalf))
            {
                if (gHasHalfs)
                {
                    vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                               gTypeNames[outType], gSaturationNames[sat],
                               gRoundingModeNames[round], gTypeNames[inType]);
                    vlog("\t\tcl_khr_fp64 enabled, but double testing turned "
                         "off.\n");
                }
                continue;
            }

            // Skip the implicit converts if the rounding mode is
            // not default or test is saturated
            if (0 == startMinVectorSize)
            {
                if (sat || round != kDefaultRoundingMode)
                    gMinVectorSize = 1;
                else
                    gMinVectorSize = 0;
            }

            if ((error = DoTest<InType, OutType, InFP, OutFP>(
                     outType, inType, sat, round, gMTdata)))
            {
                vlog_error("\t *** %d) convert_%sn%s%s( %sn ) "
                           "FAILED ** \n",
                           testNumber, gTypeNames[outType],
                           gSaturationNames[sat], gRoundingModeNames[round],
                           gTypeNames[inType]);
            }
        }
    }
    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType, bool InFP, bool OutFP>
int ConversionsTest::DoTest(Type outType, Type inType, SaturationMode sat,
                            RoundingMode round, MTdata d)
{
#ifdef __APPLE__
    cl_ulong wall_start = mach_absolute_time();
#endif

    DataInitInfo info = { 0, 0, outType, inType, sat, round, NULL };
    DataInfoSpec<InType, OutType, InFP, OutFP> init_info(info);
    WriteInputBufferInfo writeInputBufferInfo;
    int vectorSize;
    int error = 0;
    cl_uint threads = GetThreadCount();
    uint64_t i;

    gTestCount++;
    size_t blockCount =
        BUFFER_SIZE / std::max(gTypeSizes[inType], gTypeSizes[outType]);
    size_t step = blockCount;
    // uint64_t lastCase = 1ULL << (8 * gTypeSizes[inType]);
    uint64_t lastCase = 1000000ULL;

    init_info.d = (MTdata *)malloc(threads * sizeof(MTdata));
    if (NULL == init_info.d)
    {
        vlog_error(
            "ERROR: Unable to allocate storage for random number generator!\n");
        return -1;
    }
    for (i = 0; i < threads; i++)
    {
        init_info.d[i] = init_genrand(genrand_int32(d));
        if (NULL == init_info.d[i])
        {
            vlog_error("ERROR: Unable to allocate storage for random number "
                       "generator!\n");
            return -1;
        }
    }

    writeInputBufferInfo.outType = outType;
    writeInputBufferInfo.inType = inType;

    writeInputBufferInfo.calcInfo.resize(gMaxVectorSize);
    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        writeInputBufferInfo.calcInfo[vectorSize].reset(
            new CalcRefValsPat<InType, OutType, InFP, OutFP>());
        writeInputBufferInfo.calcInfo[vectorSize]->program =
            MakeProgram(outType, inType, sat, round, vectorSize,
                        &writeInputBufferInfo.calcInfo[vectorSize]->kernel);
        if (NULL == writeInputBufferInfo.calcInfo[vectorSize]->program)
        {
            gFailCount++;
            return -1;
        }
        if (NULL == writeInputBufferInfo.calcInfo[vectorSize]->kernel)
        {
            gFailCount++;
            vlog_error("\t\tFAILED -- Failed to create kernel.\n");
            return -2;
        }

        writeInputBufferInfo.calcInfo[vectorSize]->parent =
            &writeInputBufferInfo;
        writeInputBufferInfo.calcInfo[vectorSize]->vectorSize = vectorSize;
        writeInputBufferInfo.calcInfo[vectorSize]->result = -1;
    }

    if (gSkipTesting) goto exit;

    // Patch up rounding mode if default is RTZ
    // We leave the part above in default rounding mode so that the right kernel
    // is compiled.
    if (round == kDefaultRoundingMode && gIsRTZ && (outType == kfloat))
        init_info.round = round = kRoundTowardZero;

    // Figure out how many elements are in a work block

    // we handle 64-bit types a bit differently.
    // if (8 * gTypeSizes[inType] > 32) lastCase = 0x100000000ULL;

    if (!gWimpyMode && gIsEmbedded)
        step = blockCount * EMBEDDED_REDUCTION_FACTOR;

    if (gWimpyMode) step = (size_t)blockCount * (size_t)gWimpyReductionFactor;
    vlog("Testing... ");
    fflush(stdout);
    for (i = 0; i < (uint64_t)lastCase; i += step)
    {

        if (0 == (i & ((lastCase >> 3) - 1)))
        {
            vlog(".");
            fflush(stdout);
        }

        cl_uint count = (uint32_t)std::min((uint64_t)blockCount, lastCase - i);
        writeInputBufferInfo.count = count;

        // Crate a user event to represent the status of the reference value
        // computation completion
        writeInputBufferInfo.calcReferenceValues =
            clCreateUserEvent(gContext, &error);
        if (error || NULL == writeInputBufferInfo.calcReferenceValues)
        {
            vlog_error("ERROR: Unable to create user event. (%d)\n", error);
            gFailCount++;
            goto exit;
        }

        // retain for consumption by MapOutputBufferComplete
        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            if ((error =
                     clRetainEvent(writeInputBufferInfo.calcReferenceValues)))
            {
                vlog_error("ERROR: Unable to retain user event. (%d)\n", error);
                gFailCount++;
                goto exit;
            }
        }

        // Crate a user event to represent when the callbacks are done verifying
        // correctness
        writeInputBufferInfo.doneBarrier = clCreateUserEvent(gContext, &error);
        if (error || NULL == writeInputBufferInfo.calcReferenceValues)
        {
            vlog_error("ERROR: Unable to create user event for barrier. (%d)\n",
                       error);
            gFailCount++;
            goto exit;
        }

        // retain for use by the callback that calls this
        if ((error = clRetainEvent(writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("ERROR: Unable to retain user event doneBarrier. (%d)\n",
                       error);
            gFailCount++;
            goto exit;
        }

        //      Call this in a multithreaded manner
        cl_uint chunks = RoundUpToNextPowerOfTwo(threads) * 2;
        init_info.start = i;
        init_info.size = count / chunks;
        if (init_info.size < 16384)
        {
            chunks = RoundUpToNextPowerOfTwo(threads);
            init_info.size = count / chunks;
            if (init_info.size < 16384)
            {
                init_info.size = count;
                chunks = 1;
            }
        }

        ThreadPool_Do(InitData, chunks, &init_info);

        // Copy the results to the device
        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_TRUE, 0,
                                          count * gTypeSizes[inType], gIn, 0,
                                          NULL, NULL)))
        {
            vlog_error("ERROR: clEnqueueWriteBuffer failed. (%d)\n", error);
            gFailCount++;
            goto exit;
        }

        // Call completion callback for the write, which will enqueue the rest
        // of the work.
        WriteInputBufferComplete((void *)&writeInputBufferInfo);

        // Make sure the work is actually running, so we don't deadlock
        if ((error = clFlush(gQueue)))
        {
            vlog_error("clFlush failed with error %d\n", error);
            gFailCount++;
            goto exit;
        }

        ThreadPool_Do(PrepareReference, chunks, &init_info);

        // signal we are done calculating the reference results
        if ((error = clSetUserEventStatus(
                 writeInputBufferInfo.calcReferenceValues, CL_COMPLETE)))
        {
            vlog_error(
                "Error:  Failed to set user event status to CL_COMPLETE:  %d\n",
                error);
            gFailCount++;
            goto exit;
        }

        // Wait for the event callbacks to finish verifying correctness.
        if ((error = clWaitForEvents(
                 1, (cl_event *)&writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("Error:  Failed to wait for barrier:  %d\n", error);
            gFailCount++;
            goto exit;
        }

        if ((error = clReleaseEvent(writeInputBufferInfo.calcReferenceValues)))
        {
            vlog_error("Error:  Failed to release calcReferenceValues:  %d\n",
                       error);
            gFailCount++;
            goto exit;
        }

        if ((error = clReleaseEvent(writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("Error:  Failed to release done barrier:  %d\n", error);
            gFailCount++;
            goto exit;
        }

        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            if ((error = writeInputBufferInfo.calcInfo[vectorSize]->result))
            {
                switch (inType)
                {
                    case kuchar:
                    case kchar:
                        vlog("Input value: 0x%2.2x ",
                             ((unsigned char *)gIn)[error - 1]);
                        break;
                    case kushort:
                    case kshort:
                        vlog("Input value: 0x%4.4x ",
                             ((unsigned short *)gIn)[error - 1]);
                        break;
                    case kuint:
                    case kint:
                        vlog("Input value: 0x%8.8x ",
                             ((unsigned int *)gIn)[error - 1]);
                        break;
                    case khalf:
                        vlog("Input value: %a ",
                             HTF(((cl_half *)gIn)[error - 1]));
                        break;
                    case kfloat:
                        vlog("Input value: %a ", ((float *)gIn)[error - 1]);
                        break;
                    case kulong:
                    case klong:
                        vlog("Input value: 0x%16.16llx ",
                             ((unsigned long long *)gIn)[error - 1]);
                        break;
                    case kdouble:
                        vlog("Input value: %a ", ((double *)gIn)[error - 1]);
                        break;
                    default:
                        vlog_error("Internal error at %s: %d\n", __FILE__,
                                   __LINE__);
                        abort();
                        break;
                }

                // tell the user which conversion it was.
                if (0 == vectorSize)
                    vlog(" (implicit scalar conversion from %s to %s)\n",
                         gTypeNames[inType], gTypeNames[outType]);
                else
                    vlog(" (convert_%s%s%s%s( %s%s ))\n", gTypeNames[outType],
                         sizeNames[vectorSize], gSaturationNames[sat],
                         gRoundingModeNames[round], gTypeNames[inType],
                         sizeNames[vectorSize]);

                gFailCount++;
                goto exit;
            }
        }
    }

    log_info("done.\n");

    if (gTimeResults)
    {
        // Kick off tests for the various vector lengths
        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            size_t workItemCount = blockCount / vectorSizes[vectorSize];
            if (vectorSizes[vectorSize] * gTypeSizes[outType] < 4)
                workItemCount /=
                    4 / (vectorSizes[vectorSize] * gTypeSizes[outType]);

            double sum = 0.0;
            double bestTime = INFINITY;
            cl_uint k;
            for (k = 0; k < PERF_LOOP_COUNT; k++)
            {
                uint64_t startTime = GetTime();
                if ((error = RunKernel(
                         writeInputBufferInfo.calcInfo[vectorSize]->kernel,
                         gInBuffer, gOutBuffers[vectorSize], workItemCount)))
                {
                    gFailCount++;
                    goto exit;
                }

                // Make sure OpenCL is done
                if ((error = clFinish(gQueue)))
                {
                    vlog_error("Error %d at clFinish\n", error);
                    goto exit;
                }

                uint64_t endTime = GetTime();
                double time = SubtractTime(endTime, startTime);
                sum += time;
                if (time < bestTime) bestTime = time;
            }

            if (gReportAverageTimes) bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double)gDeviceFrequency
                * gComputeDevices * gSimdSize * 1e6
                / (workItemCount * vectorSizes[vectorSize]);
            if (0 == vectorSize)
                vlog_perf(clocksPerOp, LOWER_IS_BETTER, "clocks / element",
                          "implicit convert %s -> %s", gTypeNames[inType],
                          gTypeNames[outType]);
            else
                vlog_perf(clocksPerOp, LOWER_IS_BETTER, "clocks / element",
                          "convert_%s%s%s%s( %s%s )", gTypeNames[outType],
                          sizeNames[vectorSize], gSaturationNames[sat],
                          gRoundingModeNames[round], gTypeNames[inType],
                          sizeNames[vectorSize]);
        }
    }

    if (gWimpyMode)
        vlog("\tWimp pass");
    else
        vlog("\tpassed");

#ifdef __APPLE__
    // record the run time
    vlog("\t(%f s)", 1e-9 * (mach_absolute_time() - wall_start));
#endif
    vlog("\n\n");
    fflush(stdout);


exit:
    // clean up
    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        clReleaseProgram(writeInputBufferInfo.calcInfo[vectorSize]->program);
        clReleaseKernel(writeInputBufferInfo.calcInfo[vectorSize]->kernel);
    }

    if (init_info.d)
    {
        for (i = 0; i < threads; i++) free_mtdata(init_info.d[i]);
        free(init_info.d);
    }

    return error;
}

//////////////////////////////////////////////////////////////////////////////////////////

static int RunKernel(cl_kernel kernel, void *inBuf, void *outBuf,
                     size_t blockCount)
{
    // The global dimensions are just the blockCount to execute since we haven't
    // set up multiple queues for multiple devices.
    int error;

    error = clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf);
    error |= clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf);

    if (error)
    {
        vlog_error("FAILED -- could not set kernel args (%d)\n", error);
        return error;
    }

    if ((error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &blockCount,
                                        NULL, 0, NULL, NULL)))
    {
        vlog_error("FAILED -- could not execute kernel (%d)\n", error);
        return error;
    }

    return 0;
}

#if !defined(__APPLE__)
void memset_pattern4(void *dest, const void *src_pattern, size_t bytes);
#endif

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

uint64_t GetTime(void)
{
#if defined(__APPLE__)
    return mach_absolute_time();
#elif defined(_MSC_VER)
    return ReadTime();
#else
    // mach_absolute_time is a high precision timer with precision < 1
    // microsecond.
#warning need accurate clock here.  Times are invalid.
    return 0;
#endif
}


#if defined(_MSC_VER)
/* function is defined in "compat.h" */
#else
double SubtractTime(uint64_t endTime, uint64_t startTime)
{
    uint64_t diff = endTime - startTime;
    static double conversion = 0.0;

    if (0.0 == conversion)
    {
#if defined(__APPLE__)
        mach_timebase_info_data_t info = { 0, 0 };
        kern_return_t err = mach_timebase_info(&info);
        if (0 == err)
            conversion = 1e-9 * (double)info.numer / (double)info.denom;
#else
        // This function consumes output from GetTime() above, and converts the
        // time to secionds.
#warning need accurate ticks to seconds conversion factor here. Times are invalid.
#endif
    }

    // strictly speaking we should also be subtracting out timer latency here
    return conversion * (double)diff;
}
#endif

////////////////////////////////////////////////////////////////////////////////

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    info->init(job_id, thread_id);

    return CL_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
static void setAllowZ(uint8_t *allow, uint32_t *x, cl_uint count)
{
    cl_uint i;
    for (i = 0; i < count; ++i)
        allow[i] |= (uint8_t)((x[i] & 0x7f800000U) == 0);
}

////////////////////////////////////////////////////////////////////////////////
cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    cl_uint count = info->size;
    Type inType = info->inType;
    Type outType = info->outType;
    size_t j;

    Force64BitFPUPrecision();

    void *s = (cl_uchar *)gIn + job_id * count * gTypeSizes[info->inType];
    void *a = (cl_uchar *)gAllowZ + job_id * count;
    void *d = (cl_uchar *)gRef + job_id * count * gTypeSizes[info->outType];


    if (outType != inType)
    {


        // create the reference while we wait

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
        /* ARM VFP doesn't have hardware instruction for converting from 64-bit
         * integer to float types, hence GCC ARM uses the floating-point
         * emulation code despite which -mfloat-abi setting it is. But the
         * emulation code in libgcc.a has only one rounding mode (round to
         * nearest even in this case) and ignores the user rounding mode setting
         * in hardware. As a result setting rounding modes in hardware won't
         * give correct rounding results for type covert from 64-bit integer to
         * float using GCC for ARM compiler so for testing different rounding
         * modes, we need to use alternative reference function. ARM64 does have
         * an instruction, however we cannot guarantee the compiler will use it.
         * On all ARM architechures use emulation to calculate reference.*/
        switch (round)
        {
            /* conversions to floating-point type use the current rounding mode.
             * The only default floating-point rounding mode supported is round
             * to nearest even i.e the current rounding mode will be _rte for
             * floating-point types. */
            case kDefaultRoundingMode: qcom_rm = qcomRTE; break;
            case kRoundToNearestEven: qcom_rm = qcomRTE; break;
            case kRoundUp: qcom_rm = qcomRTP; break;
            case kRoundDown: qcom_rm = qcomRTN; break;
            case kRoundTowardZero: qcom_rm = qcomRTZ; break;
            default:
                vlog_error("ERROR: undefined rounding mode %d\n", round);
                break;
        }
        qcom_sat = info->sat;
#endif


        RoundingMode oldRound, round = info->round;

        if (/*is_half<OutType, OutFP>()*/ outType == khalf)
        {
            oldRound = set_round(kRoundToNearestEven, kfloat);
            switch (round)
            {
                default:
                case kDefaultRoundingMode:
                    ConversionsTest::halfRoundingMode =
                        ConversionsTest::defaultHalfRoundingMode;
                    break;
                case kRoundToNearestEven:
                    ConversionsTest::halfRoundingMode = CL_HALF_RTE;
                    break;
                case kRoundUp:
                    ConversionsTest::halfRoundingMode = CL_HALF_RTP;
                    break;
                case kRoundDown:
                    ConversionsTest::halfRoundingMode = CL_HALF_RTN;
                    break;
                case kRoundTowardZero:
                    ConversionsTest::halfRoundingMode = CL_HALF_RTZ;
                    break;
            }
        }
        else
            oldRound = set_round(round, outType);


        if (info->sat)
            info->conv_array_sat(d, s, count);
        else
            info->conv_array(d, s, count);


        set_round(oldRound, outType);


        // Decide if we allow a zero result in addition to the correctly rounded
        // one
        memset(a, 0, count);
        if (gForceFTZ)
        {
            if (inType == kfloat) setAllowZ((uint8_t *)a, (uint32_t *)s, count);
            if (outType == kfloat)
                setAllowZ((uint8_t *)a, (uint32_t *)d, count);
        }
    }
    else
    {
        // Copy the input to the reference
        memcpy(d, s, info->size * gTypeSizes[inType]);
    }

    // Patch up NaNs conversions to integer to zero -- these can be converted to
    // any integer
    if (info->outType != kfloat && info->outType != kdouble)
    {
        if (inType == kfloat)
        {
            float *inp = (float *)s;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]))
                    memset((char *)d + j * gTypeSizes[outType], 0,
                           gTypeSizes[outType]);
            }
        }
        if (inType == kdouble)
        {
            double *inp = (double *)s;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]))
                    memset((char *)d + j * gTypeSizes[outType], 0,
                           gTypeSizes[outType]);
            }
        }
    }
    else if (inType == kfloat || inType == kdouble)
    { // outtype and intype is float or double.  NaN conversions for float <->
      // double can be any NaN
        if (inType == kfloat && outType == kdouble)
        {
            float *inp = (float *)s;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j])) ((double *)d)[j] = NAN;
            }
        }
        if (inType == kdouble && outType == kfloat)
        {
            double *inp = (double *)s;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j])) ((float *)d)[j] = NAN;
            }
        }
    }

    return CL_SUCCESS;
}


void MapResultValuesComplete(const std::unique_ptr<CalcRefValsBase> &ptr);

// Note: not called reentrantly
void WriteInputBufferComplete(void *data)
{
    cl_int status;
    WriteInputBufferInfo *info = (WriteInputBufferInfo *)data;
    cl_uint count = info->count;
    int vectorSize;

    info->barrierCount = gMaxVectorSize - gMinVectorSize;

    // now that we know that the write buffer is complete, enqueue callbacks to
    // wait for the main thread to finish calculating the reference results.
    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        size_t workItemCount =
            (count + vectorSizes[vectorSize] - 1) / (vectorSizes[vectorSize]);

        if ((status = RunKernel(info->calcInfo[vectorSize]->kernel, gInBuffer,
                                gOutBuffers[vectorSize], workItemCount)))
        {
            gFailCount++;
            return;
        }

        info->calcInfo[vectorSize]->p = clEnqueueMapBuffer(
            gQueue, gOutBuffers[vectorSize], CL_TRUE,
            CL_MAP_READ | CL_MAP_WRITE, 0, count * gTypeSizes[info->outType], 0,
            NULL, NULL, &status);
        {
            if (status)
            {
                vlog_error("ERROR: WriteInputBufferComplete calback failed "
                           "with status: %d\n",
                           status);
                gFailCount++;
                return;
            }
        }
    }

    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        MapResultValuesComplete(info->calcInfo[vectorSize]);
    }

    // Make sure the work starts moving -- otherwise we may deadlock
    if ((status = clFlush(gQueue)))
    {
        vlog_error(
            "ERROR: WriteInputBufferComplete calback failed with status: %d\n",
            status);
        gFailCount++;
        return;
    }

    // e was already released by the main thread. It should be destroyed
    // automatically soon after we exit.
}

void CL_CALLBACK CalcReferenceValuesComplete(cl_event e, cl_int status,
                                             void *data);

// Note: May be called reentrantly
void MapResultValuesComplete(const std::unique_ptr<CalcRefValsBase> &info)
{
    cl_int status;
    // CalcRefValsBase *info = (CalcRefValsBase *)data;
    cl_event calcReferenceValues = info->parent->calcReferenceValues;

    // we know that the map is done, wait for the main thread to finish
    // calculating the reference values
    if ((status =
             clSetEventCallback(calcReferenceValues, CL_COMPLETE,
                                CalcReferenceValuesComplete, (void *)&info)))
    {
        vlog_error("ERROR: clSetEventCallback failed in "
                   "MapResultValuesComplete with status: %d\n",
                   status);
        gFailCount++; // not thread safe -- being lazy here
    }

    // this thread no longer needs its reference to info->calcReferenceValues,
    // so release it
    if ((status = clReleaseEvent(calcReferenceValues)))
    {
        vlog_error("ERROR: clReleaseEvent(info->calcReferenceValues) failed "
                   "with status: %d\n",
                   status);
        gFailCount++; // not thread safe -- being lazy here
    }

    // no need to flush since we didn't enqueue anything

    // e was already released by WriteInputBufferComplete. It should be
    // destroyed automatically soon after we exit.
}


void CL_CALLBACK CalcReferenceValuesComplete(cl_event e, cl_int status,
                                             void *data)
{
    std::unique_ptr<CalcRefValsBase> &info =
        *(std::unique_ptr<CalcRefValsBase> *)data;

    cl_uint vectorSize = info->vectorSize;
    cl_uint count = info->parent->count;
    Type outType =
        info->parent->outType; // the data type of the conversion result
    Type inType = info->parent->inType; // the data type of the conversion input
    size_t j;
    cl_int error;
    cl_event doneBarrier = info->parent->doneBarrier;

    // report spurious error condition
    if (CL_SUCCESS != status)
    {
        vlog_error("ERROR: CalcReferenceValuesComplete did not succeed! (%d)\n",
                   status);
        gFailCount++; // lazy about thread safety here
        return;
    }

    // Now we know that both results have been mapped back from the device, and
    // the main thread is done calculating the reference results. It is now time
    // to check the results.

    // verify results
    void *mapped = info->p;

    // Patch up NaNs conversions to integer to zero -- these can be converted to
    // any integer
    if (outType != kfloat && outType != kdouble)
    {
        if (inType == kfloat)
        {
            float *inp = (float *)gIn;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]))
                    memset((char *)mapped + j * gTypeSizes[outType], 0,
                           gTypeSizes[outType]);
            }
        }
        if (inType == kdouble)
        {
            double *inp = (double *)gIn;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]))
                    memset((char *)mapped + j * gTypeSizes[outType], 0,
                           gTypeSizes[outType]);
            }
        }
    }
    else if (inType == kfloat || inType == kdouble)
    { // outtype and intype is float or double.  NaN conversions for float <->
      // double can be any NaN
        if (inType == kfloat && outType == kdouble)
        {
            float *inp = (float *)gIn;
            double *outp = (double *)mapped;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]) && isnan(outp[j])) outp[j] = NAN;
            }
        }
        if (inType == kdouble && outType == kfloat)
        {
            double *inp = (double *)gIn;
            float *outp = (float *)mapped;
            for (j = 0; j < count; j++)
            {
                if (isnan(inp[j]) && isnan(outp[j])) outp[j] = NAN;
            }
        }
    }

    if (memcmp(mapped, gRef, count * gTypeSizes[outType]))
        info->result =
            info->check_result(mapped, count, vectorSizes[vectorSize]);
    else
        info->result = 0;

    // Fill the output buffer with junk and release it
    {
        cl_uint pattern = 0xffffdead;
        memset_pattern4(mapped, &pattern, count * gTypeSizes[outType]);
        if ((error = clEnqueueUnmapMemObject(gQueue, gOutBuffers[vectorSize],
                                             mapped, 0, NULL, NULL)))
        {
            vlog_error("ERROR: clEnqueueUnmapMemObject failed in "
                       "CalcReferenceValuesComplete  (%d)\n",
                       error);
            gFailCount++;
        }
    }

    if (1 == ThreadPool_AtomicAdd(&info->parent->barrierCount, -1))
    {
        if ((status = clSetUserEventStatus(doneBarrier, CL_COMPLETE)))
        {
            vlog_error("ERROR: clSetUserEventStatus failed in "
                       "CalcReferenceValuesComplete (err: %d). We're probably "
                       "going to deadlock.\n",
                       status);
            gFailCount++;
            return;
        }

        if ((status = clReleaseEvent(doneBarrier)))
        {
            vlog_error("ERROR: clReleaseEvent failed in "
                       "CalcReferenceValuesComplete (err: %d).\n",
                       status);
            gFailCount++;
            return;
        }
    }
    // e was already released by WriteInputBufferComplete. It should be
    // destroyed automatically soon after all the calls to
    // CalcReferenceValuesComplete exit.
}

static cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                              RoundingMode round, int vectorSize,
                              cl_kernel *outKernel)
{
    cl_program program;
    char testName[256];
    int error = 0;

    std::ostringstream source;
    if (outType == kdouble || inType == kdouble)
        source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    // Create the program. This is a bit complicated because we are trying to
    // avoid byte and short stores.
    if (0 == vectorSize)
    {
        // Create the type names.
        char inName[32];
        char outName[32];
        strncpy(inName, gTypeNames[inType], sizeof(inName));
        strncpy(outName, gTypeNames[outType], sizeof(outName));
        sprintf(testName, "test_implicit_%s_%s", outName, inName);

        source << "__kernel void " << testName << "( __global " << inName
               << " *src, __global " << outName << " *dest )\n";
        source << "{\n";
        source << "   size_t i = get_global_id(0);\n";
        source << "   dest[i] =  src[i];\n";
        source << "}\n";

        vlog("Building implicit %s -> %s conversion test\n", gTypeNames[inType],
             gTypeNames[outType]);
        fflush(stdout);
    }
    else
    {
        int vectorSizetmp = vectorSizes[vectorSize];

        // Create the type names.
        char convertString[128];
        char inName[32];
        char outName[32];
        switch (vectorSizetmp)
        {
            case 1:
                strncpy(inName, gTypeNames[inType], sizeof(inName));
                strncpy(outName, gTypeNames[outType], sizeof(outName));
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
            case 3:
                strncpy(inName, gTypeNames[inType], sizeof(inName));
                strncpy(outName, gTypeNames[outType], sizeof(outName));
                snprintf(convertString, sizeof(convertString),
                         "convert_%s3%s%s", outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s3", convertString, inName);
                vlog("Building %s( %s3 ) test\n", convertString, inName);
                break;
            default:
                snprintf(inName, sizeof(inName), "%s%d", gTypeNames[inType],
                         vectorSizetmp);
                snprintf(outName, sizeof(outName), "%s%d", gTypeNames[outType],
                         vectorSizetmp);
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
        }
        fflush(stdout);

        if (vectorSizetmp == 3)
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   if( i + 1 < get_global_size(0))\n";
            source << "       vstore3( " << convertString
                   << "( vload3( i, src)), i, dest );\n";
            source << "   else\n";
            source << "   {\n";
            source << "       " << inName << "3 in;\n";
            source << "       " << outName << "3 out;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           in.y = src[3*i+1];\n";
            source << "       in.x = src[3*i];\n";
            source << "       out = " << convertString << "( in ); \n";
            source << "       dest[3*i] = out.x;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           dest[3*i+1] = out.y;\n";
            source << "   }\n";
            source << "}\n";
        }
        else
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   dest[i] = " << convertString << "( src[i] );\n";
            source << "}\n";
        }
    }
    *outKernel = NULL;

    const char *flags = NULL;
    if (gForceFTZ) flags = "-cl-denorms-are-zero";

    // build it
    std::string sourceString = source.str();
    const char *programSource = sourceString.c_str();
    error = create_single_kernel_helper(gContext, &program, outKernel, 1,
                                        &programSource, testName, flags);
    if (error)
    {
        vlog_error("Failed to build kernel/program (err = %d).\n", error);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

////////////////////////////////////////////////////////////////////////////////////////

static int GetTestCase(const char *name, Type *outType, Type *inType,
                       SaturationMode *sat, RoundingMode *round)
{
    int i;

    // Find the return type
    for (i = 0; i < kTypeCount; i++)
        if (name == strstr(name, gTypeNames[i]))
        {
            *outType = (Type)i;
            name += strlen(gTypeNames[i]);

            break;
        }

    if (i == kTypeCount) return -1;

    // Check to see if _sat appears next
    *sat = (SaturationMode)0;
    for (i = 1; i < kSaturationModeCount; i++)
        if (name == strstr(name, gSaturationNames[i]))
        {
            *sat = (SaturationMode)i;
            name += strlen(gSaturationNames[i]);
            break;
        }

    *round = (RoundingMode)0;
    for (i = 1; i < kRoundingModeCount; i++)
        if (name == strstr(name, gRoundingModeNames[i]))
        {
            *round = (RoundingMode)i;
            name += strlen(gRoundingModeNames[i]);
            break;
        }

    if (*name != '_') return -2;
    name++;

    for (i = 0; i < kTypeCount; i++)
        if (name == strstr(name, gTypeNames[i]))
        {
            *inType = (Type)i;
            name += strlen(gTypeNames[i]);

            break;
        }

    if (i == kTypeCount) return -3;

    if (*name != '\0') return -4;

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////
