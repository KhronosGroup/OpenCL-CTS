//
// Copyright (c) 2017-2024 The Khronos Group Inc.
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
#ifndef UTILITY_H
#define UTILITY_H

#include "harness/compat.h"
#include "harness/rounding_mode.h"
#include "harness/fpcontrol.h"
#include "harness/testHarness.h"
#include "harness/ThreadPool.h"
#include "harness/conversions.h"
#include "CL/cl_half.h"

#define BUFFER_SIZE (1024 * 1024 * 2)
#define EMBEDDED_REDUCTION_FACTOR (64)

#if defined(__GNUC__)
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

struct Func;

extern int gWimpyReductionFactor;

#define VECTOR_SIZE_COUNT 6
extern const char *sizeNames[VECTOR_SIZE_COUNT];
extern const int sizeValues[VECTOR_SIZE_COUNT];

extern cl_device_id gDevice;
extern cl_context gContext;
extern cl_command_queue gQueue;
extern void *gIn;
extern void *gIn2;
extern void *gIn3;
extern void *gOut_Ref;
extern void *gOut_Ref2;
extern void *gOut[VECTOR_SIZE_COUNT];
extern void *gOut2[VECTOR_SIZE_COUNT];
extern cl_mem gInBuffer;
extern cl_mem gInBuffer2;
extern cl_mem gInBuffer3;
extern cl_mem gOutBuffer[VECTOR_SIZE_COUNT];
extern cl_mem gOutBuffer2[VECTOR_SIZE_COUNT];
extern int gSkipCorrectnessTesting;
extern int gForceFTZ;
extern int gFastRelaxedDerived;
extern int gWimpyMode;
extern int gHostFill;
extern int gIsInRTZMode;
extern int gHasHalf;
extern int gInfNanSupport;
extern int gIsEmbedded;
extern int gVerboseBruteForce;
extern uint32_t gMaxVectorSizeIndex;
extern uint32_t gMinVectorSizeIndex;
extern cl_device_fp_config gFloatCapabilities;
extern cl_device_fp_config gHalfCapabilities;
extern RoundingMode gFloatToHalfRoundingMode;

extern cl_half_rounding_mode gHalfRoundingMode;

#define HFF(num) cl_half_from_float(num, gHalfRoundingMode)
#define HFD(num) cl_half_from_double(num, gHalfRoundingMode)
#define HTF(num) cl_half_to_float(num)

#define LOWER_IS_BETTER 0
#define HIGHER_IS_BETTER 1

#include "harness/errorHelpers.h"

#if defined(_MSC_VER)
// Deal with missing scalbn on windows
#define scalbnf(_a, _i) ldexpf(_a, _i)
#define scalbn(_a, _i) ldexp(_a, _i)
#define scalbnl(_a, _i) ldexpl(_a, _i)
#endif

float Abs_Error(float test, double reference);
float Ulp_Error(float test, double reference);
float Bruteforce_Ulp_Error_Double(double test, long double reference);

// used to convert a bucket of bits into a search pattern through double
inline double DoubleFromUInt32(uint32_t bits)
{
    union {
        uint64_t u;
        double d;
    } u;

    // split 0x89abcdef to 0x89abc00000000def
    u.u = bits & 0xfffU;
    u.u |= (uint64_t)(bits & ~0xfffU) << 32;

    // sign extend the leading bit of def segment as sign bit so that the middle
    // region consists of either all 1s or 0s
    u.u -= (bits & 0x800U) << 1;

    // return result
    return u.d;
}

void _LogBuildError(cl_program p, int line, const char *file);
#define LogBuildError(program) _LogBuildError(program, __LINE__, __FILE__)

// The spec is fairly clear that we may enforce a hard cutoff to prevent
// premature flushing to zero.
// However, to avoid conflict for 1.0, we are letting results at TYPE_MIN +
// ulp_limit to be flushed to zero.
inline int IsFloatResultSubnormal(double x, float ulps)
{
    x = fabs(x) - MAKE_HEX_DOUBLE(0x1.0p-149, 0x1, -149) * (double)ulps;
    return x < MAKE_HEX_DOUBLE(0x1.0p-126, 0x1, -126);
}

inline int IsHalfResultSubnormal(float x, float ulps)
{
    x = fabs(x) - MAKE_HEX_FLOAT(0x1.0p-24, 0x1, -24) * ulps;
    return x < MAKE_HEX_FLOAT(0x1.0p-14, 0x1, -14);
}

inline int IsFloatResultSubnormalAbsError(double x, float abs_err)
{
    x = x - abs_err;
    return x < MAKE_HEX_DOUBLE(0x1.0p-126, 0x1, -126);
}

inline int IsDoubleResultSubnormal(long double x, float ulps)
{
    x = fabsl(x) - MAKE_HEX_LONG(0x1.0p-1074, 0x1, -1074) * (long double)ulps;
    return x < MAKE_HEX_LONG(0x1.0p-1022, 0x1, -1022);
}

inline int IsFloatInfinity(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) == 0x7F800000U);
}

inline int IsFloatMaxFloat(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) == 0x7F7FFFFFU);
}

inline int IsFloatNaN(double x)
{
    union {
        cl_float d;
        cl_uint u;
    } u;
    u.d = (cl_float)x;
    return ((u.u & 0x7fffffffU) > 0x7F800000U);
}

inline bool IsHalfNaN(const cl_half v)
{
    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = ((cl_half)v) & 0x3FF;

    // NaN test
    return (h_exp == 0x1F && h_mant != 0);
}

inline bool IsHalfInfinity(const cl_half v)
{
    // Extract FP16 exponent and mantissa
    uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
    uint16_t h_mant = ((cl_half)v) & 0x3FF;

    // Inf test
    return (h_exp == 0x1F && h_mant == 0);
}

cl_uint RoundUpToNextPowerOfTwo(cl_uint x);

// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
inline void Force64BitFPUPrecision(void)
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
#elif defined(_WIN32)                                                          \
    && (defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))
    // Unfortunately, usual method (`_controlfp( _PC_64, _MCW_PC );') does *not*
    // work on win.x64: > On the x64 architecture, changing the floating point
    // precision is not supported. (Taken from
    // http://msdn.microsoft.com/en-us/library/e9b52ceh%28v=vs.100%29.aspx)
    int cw;
    __asm { fnstcw cw }
    ; // Get current value of FPU control word.
    cw = cw & 0xfffffcff
        | (3 << 8); // Set Precision Control to Double Extended Precision.
    __asm { fldcw cw }
    ; // Set new value of FPU control word.
#else
    /* Implement for other platforms if needed */
#endif
}

void memset_pattern4(void *dest, const void *src_pattern, size_t bytes);

union int32f_t {
    int32_t i;
    float f;
};

union int64d_t {
    int64_t l;
    double d;
};

void MulD(double *rhi, double *rlo, double u, double v);
void AddD(double *rhi, double *rlo, double a, double b);
void MulDD(double *rhi, double *rlo, double xh, double xl, double yh,
           double yl);
void AddDD(double *rhi, double *rlo, double xh, double xl, double yh,
           double yl);
void DivideDD(double *chi, double *clo, double a, double b);
int compareFloats(float x, float y);
int compareDoubles(double x, double y);

void logFunctionInfo(const char *fname, unsigned int float_size,
                     unsigned int isFastRelaxed);

float getAllowedUlpError(const Func *f, Type t, const bool relaxed);

inline cl_uint getTestScale(size_t typeSize)
{
    if (gWimpyMode)
    {
        return (cl_uint)typeSize * 2 * gWimpyReductionFactor;
    }
    else if (gIsEmbedded)
    {
        return EMBEDDED_REDUCTION_FACTOR;
    }
    else
    {
        return 1;
    }
}

inline uint64_t getTestStep(size_t typeSize, size_t bufferSize)
{
    if (gWimpyMode)
    {
        return (1ULL << 32) * gWimpyReductionFactor / (512);
    }
    else if (gIsEmbedded)
    {
        return (BUFFER_SIZE / typeSize) * EMBEDDED_REDUCTION_FACTOR;
    }
    else
    {
        return bufferSize / typeSize;
    }
}

#endif /* UTILITY_H */
