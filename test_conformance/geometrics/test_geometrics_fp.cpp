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
#include "test_geometrics_fp.h"
#include "harness/compat.h"

#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include <float.h>
#include <limits>
#include <cmath>

#include <CL/cl_half.h>

//--------------------------------------------------------------------------/

cl_half_rounding_mode GeomTestBase::halfRoundingMode = CL_HALF_RTE;
cl_ulong GeometricsFPTest::maxAllocSize = 0;
cl_ulong GeometricsFPTest::maxGlobalMemSize = 0;

//--------------------------------------------------------------------------/

char ftype[32] = { 0 };
char extension[128] = { 0 };
char load_store[128] = { 0 };

//--------------------------------------------------------------------------/
// clang-format off
// for readability sake keep this section unformatted

static const char *twoArgsToScalarV1 = "dst[tid] = %s( srcA[tid], srcB[tid] );\n";
static const char *twoArgsToScalarVn = "dst[tid] = %s( vload%d( tid, srcA), vload%d( tid, srcB) );\n";

static const char *oneArgToScalarV1 = "dst[tid] = %s( srcA[tid] );\n";
static const char *oneArgToScalarVn = "dst[tid] = %s( vload%d( tid, srcA) );\n";

static const char *oneArgToVecV1 = "dst[tid] = %s( srcA[tid] );\n";
static const char *oneArgToVecVn = "vstore%d( %s( vload%d( tid, srcA) ), tid, dst );\n";

//--------------------------------------------------------------------------/

static const char *crossKernelSource[] = {
    extension,
    "__kernel void sample_test(__global ", ftype, " *srcA, __global ", ftype, " *srcB, __global ", ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    vstore%d( cross( vload%d( tid, srcA), vload%d( tid,  srcB) ), tid, dst );\n"
    "\n"
    "}\n"
};

static const char *twoArgsKernelPattern[] = {
    extension,
    "__kernel void sample_test(__global ", ftype, " *srcA, __global ", ftype, " *srcB, __global ", ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    , load_store,
    "}\n"
};

static const char *oneToFloatKernelPattern[] = {
    extension,
    "__kernel void sample_test(__global ", ftype, " *srcA, __global ", ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    , load_store,
    "\n"
    "}\n"
};

static const char *oneToOneKernelPattern[] = {
    extension,
    "__kernel void sample_test(__global ", ftype, " *srcA, __global ", ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    , load_store,
    "\n"
    "}\n"
};

// clang-format on
//--------------------------------------------------------------------------/

#define TEST_SIZE (1 << 10)
#define HFF(num) cl_half_from_float(num, GeomTestBase::halfRoundingMode)
#define HTF(num) cl_half_to_float(num)

//--------------------------------------------------------------------------/

std::string concat_kernel(const char *sstr[], int num)
{
    std::string res;
    for (int i = 0; i < num; i++) res += std::string(sstr[i]);
    return res;
}

//--------------------------------------------------------------------------/

template <typename... Args>
std::string string_format(const std::string &format, Args... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...)
        + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(),
                       buf.get() + size - 1); // We don't want the '\0' inside
}

//--------------------------------------------------------------------------

template <typename T>
void vector2string(std::stringstream &sstr, T *vector, size_t elements)
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

template <typename T> static bool isnan_fp(const T &v)
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
        return std::isnan(v);
    }
}

//--------------------------------------------------------------------------

template <typename T> static bool isfinite_fp(const T &v)
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
        return std::isfinite(v);
    }
}

//--------------------------------------------------------------------------

template <typename T> static T max_fp(const T &lhs, const T &rhs)
{
    if (std::is_same<T, half>::value)
        return cl_half_to_float(lhs) > cl_half_to_float(rhs) ? lhs : rhs;
    else
        return std::fmax(lhs, rhs);
}

//--------------------------------------------------------------------------

template <typename T> static T abs_fp(const T &val)
{
    if (std::is_same<T, half>::value)
        return static_cast<half>(val) & 0x7FFF;
    else
        return std::fabs(val);
}

//--------------------------------------------------------------------------

template <typename T>
static T get_random(float low, float high, const MTdata &d)
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

template <typename T> static T any_value(MTdata d)
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
template <typename T>
static bool verify_cross(int i, const T *const lhs, const T *const rhs,
                         const T *const inA, const T *const inB,
                         const double *const etol)
{
    if (std::is_same<T, half>::value == false)
    {
        double errs[] = { fabs(lhs[0] - rhs[0]), fabs(lhs[1] - rhs[1]),
                          fabs(lhs[2] - rhs[2]) };

        if (errs[0] > etol[0] || errs[1] > etol[1] || errs[2] > etol[2])
        {
            char printout[256];
            std::snprintf(printout, sizeof(printout),
                          "ERROR: Data sample %d does not validate! Expected "
                          "(%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1],
                          rhs[2], rhs[3]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Input: (%a %a %a) and (%a %a %a)\n", inA[0],
                          inA[1], inA[2], inB[0], inB[1], inB[2]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Errors: (%a out of %a), (%a out of %a), (%a out "
                          "of %a)\n",
                          errs[0], etol[0], errs[1], etol[1], errs[2], etol[2]);
            log_error(printout);

            return true;
        }
    }
    else
    {
        float errs[] = { fabsf(HTF(lhs[0]) - HTF(rhs[0])),
                         fabsf(HTF(lhs[1]) - HTF(rhs[1])),
                         fabsf(HTF(lhs[2]) - HTF(rhs[2])) };

        if (errs[0] > etol[0] || errs[1] > etol[1] || errs[2] > etol[2])
        {
            char printout[256];
            std::snprintf(printout, sizeof(printout),
                          "ERROR: Data sample %d does not validate! Expected "
                          "(%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, HTF(lhs[0]), HTF(lhs[1]), HTF(lhs[2]), HTF(lhs[3]),
                          rhs[0], rhs[1], rhs[2], rhs[3]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Input: (%a %a %a) and (%a %a %a)\n", HTF(inA[0]),
                          HTF(inA[1]), HTF(inA[2]), HTF(inB[0]), HTF(inB[1]),
                          HTF(inB[2]));
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Errors: (%a out of %a), (%a out of %a), (%a out "
                          "of %a)\n",
                          errs[0], etol[0], errs[1], etol[1], errs[2], etol[2]);
            log_error(printout);
            return true;
        }
    }
    return false;
}

//--------------------------------------------------------------------------

template <typename T>
void cross_product(const T *const vecA, const T *const vecB, T *const outVector,
                   double *const errorTolerances, double ulpTolerance)
{
    if (std::is_same<T, half>::value == false)
    {
        outVector[0] = (vecA[1] * vecB[2]) - (vecA[2] * vecB[1]);
        outVector[1] = (vecA[2] * vecB[0]) - (vecA[0] * vecB[2]);
        outVector[2] = (vecA[0] * vecB[1]) - (vecA[1] * vecB[0]);
        outVector[3] = 0.0f;

        errorTolerances[0] =
            fmax(fabs((double)vecA[1]),
                 fmax(fabs((double)vecB[2]),
                      fmax(fabs((double)vecA[2]), fabs((double)vecB[1]))));
        errorTolerances[1] =
            fmax(fabs((double)vecA[2]),
                 fmax(fabs((double)vecB[0]),
                      fmax(fabs((double)vecA[0]), fabs((double)vecB[2]))));
        errorTolerances[2] =
            fmax(fabs((double)vecA[0]),
                 fmax(fabs((double)vecB[1]),
                      fmax(fabs((double)vecA[1]), fabs((double)vecB[0]))));

        // This gives us max squared times ulp tolerance, i.e. the worst-case
        // expected variance we could expect from this result
        errorTolerances[0] = errorTolerances[0] * errorTolerances[0]
            * (ulpTolerance * FLT_EPSILON);
        errorTolerances[1] = errorTolerances[1] * errorTolerances[1]
            * (ulpTolerance * FLT_EPSILON);
        errorTolerances[2] = errorTolerances[2] * errorTolerances[2]
            * (ulpTolerance * FLT_EPSILON);
    }
    else
    {
        const float fvA[] = { HTF(vecA[0]), HTF(vecA[1]), HTF(vecA[2]) };
        const float fvB[] = { HTF(vecB[0]), HTF(vecB[1]), HTF(vecB[2]) };

        outVector[0] = HFF((fvA[1] * fvB[2]) - (fvA[2] * fvB[1]));
        outVector[1] = HFF((fvA[2] * fvB[0]) - (fvA[0] * fvB[2]));
        outVector[2] = HFF((fvA[0] * fvB[1]) - (fvA[1] * fvB[0]));
        outVector[3] = 0.0f;

        errorTolerances[0] =
            fmax(fabs(fvA[1]),
                 fmax(fabs(fvB[2]), fmaxf(fabs(fvA[2]), fabs(fvB[1]))));
        errorTolerances[1] =
            fmax(fabs(fvA[2]),
                 fmax(fabs(fvB[0]), fmaxf(fabs(fvA[0]), fabs(fvB[2]))));
        errorTolerances[2] =
            fmax(fabs(fvA[0]),
                 fmax(fabs(fvB[1]), fmaxf(fabs(fvA[1]), fabs(fvB[0]))));

        errorTolerances[0] = errorTolerances[0] * errorTolerances[0]
            * (ulpTolerance * CL_HALF_EPSILON);
        errorTolerances[1] = errorTolerances[1] * errorTolerances[1]
            * (ulpTolerance * CL_HALF_EPSILON);
        errorTolerances[2] = errorTolerances[2] * errorTolerances[2]
            * (ulpTolerance * CL_HALF_EPSILON);
    }
}

//--------------------------------------------------------------------------

template <typename T> bool signbit_fp(const T &a)
{
    if (std::is_same<T, half>::value)
        return static_cast<half>(a) & 0x8000 ? 1 : 0;
    else
        return std::signbit(a);
}

//--------------------------------------------------------------------------

template <typename T> double mad_fp(const T &a, const T &b, const T &c)
{
    if (!isfinite_fp<T>(a)) return signbit_fp<T>(a) ? -INFINITY : INFINITY;
    if (!isfinite_fp<T>(b)) return signbit_fp<T>(b) ? -INFINITY : INFINITY;
    if (!isfinite_fp<T>(c)) return signbit_fp<T>(c) ? -INFINITY : INFINITY;
    if (std::is_same<T, half>::value)
        return HTF(a) * HTF(b) + HTF(c);
    else
        return a * b + c;
}

//--------------------------------------------------------------------------

template <typename T>
double verifyDot(const T *srcA, const T *srcB, size_t vecSize)
{
    double total = 0.f;
    if (std::is_same<T, half>::value)
    {
        for (unsigned int i = vecSize; i--;)
            total = mad_fp<T>(srcA[i], srcB[i], HFF(total));
        return (!isfinite_fp<T>(HFF(total)))
            ? (signbit_fp<T>(total) ? -INFINITY : INFINITY)
            : total;
    }
    else
    {
        for (unsigned int i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcB[i];
        return total;
    }
}

//--------------------------------------------------------------------------

double verifyFastDistance(const float *srcA, const float *srcB, size_t vecSize)
{
    double total = 0, value;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    for (i = 0; i < vecSize; i++)
    {
        value = (double)srcA[i] - (double)srcB[i];
        total += value * value;
    }

    return sqrt(total);
}

//--------------------------------------------------------------------------

template <typename T> double verifyLength(const T *srcA, size_t vecSize)
{
    double total = 0;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    if (std::is_same<T, half>::value)
    {
        for (i = 0; i < vecSize; i++)
            total += (double)HTF(srcA[i]) * (double)HTF(srcA[i]);
    }
    else
    {
        for (i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcA[i];
    }


    if (std::is_same<T, double>::value)
    {
        // Deal with spurious overflow
        if (total == INFINITY)
        {
            total = 0.0;
            for (i = 0; i < vecSize; i++)
            {
                double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p-600, 0x1LL, -600);
                total += f * f;
            }

            return sqrt(total) * MAKE_HEX_DOUBLE(0x1.0p600, 0x1LL, 600);
        }

        // Deal with spurious underflow
        if (total < 4 /*max vector length*/ * DBL_MIN / DBL_EPSILON)
        {
            total = 0.0;
            for (i = 0; i < vecSize; i++)
            {
                double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700);
                total += f * f;
            }

            return sqrt(total) * MAKE_HEX_DOUBLE(0x1.0p-700, 0x1LL, -700);
        }
    }

    return sqrt(total);
}

//--------------------------------------------------------------------------

template <typename T>
double verifyDistance(const T *srcA, const T *srcB, size_t vecSize)
{
    if (std::is_same<T, double>::value)
    {
        double diff[4];
        for (unsigned i = 0; i < vecSize; i++) diff[i] = srcA[i] - srcB[i];

        return verifyLength<T>((T *)diff, vecSize);
    }
    else
    {
        double total = 0, value;
        if (std::is_same<T, half>::value)
        {
            // We calculate the distance as a double, to try and make up for the
            // fact that the GPU has better precision distance since it's a
            // single op
            for (unsigned i = 0; i < vecSize; i++)
            {
                value = (double)HTF(srcA[i]) - (double)HTF(srcB[i]);
                total += value * value;
            }
        }
        else
        {
            for (unsigned i = 0; i < vecSize; i++)
            {
                value = (double)srcA[i] - (double)srcB[i];
                total += value * value;
            }
        }
        return sqrt(total);
    }
}

//--------------------------------------------------------------------------

double verifyFastLength(const float *srcA, size_t vecSize)
{
    double total = 0;
    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    for (unsigned i = 0; i < vecSize; i++)
        total += (double)srcA[i] * (double)srcA[i];

    return sqrt(total);
}

//--------------------------------------------------------------------------

template <typename T>
void verifyNormalize(const T *srcA, T *dst, size_t vecSize)
{
    double total = 0, value;
    // We calculate everything as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op

    if (std::is_same<T, half>::value)
    {
        for (unsigned i = 0; i < vecSize; i++)
            total += (double)HTF(srcA[i]) * (double)HTF(srcA[i]);

        if (total == 0.f)
        {
            for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
            return;
        }

        if (total == INFINITY)
        {
            total = 0.0f;
            for (unsigned i = 0; i < vecSize; i++)
            {
                if (fabsf(HTF(srcA[i])) == INFINITY)
                    dst[i] = HFF(copysignf(1.0f, HTF(srcA[i])));
                else
                    dst[i] = HFF(copysignf(0.0f, HTF(srcA[i])));
                total += (double)HTF(dst[i]) * (double)HTF(dst[i]);
            }
            srcA = dst;
        }

        value = sqrt(total);
        for (unsigned i = 0; i < vecSize; i++)
            dst[i] = HFF(HTF(srcA[i]) / value);
    }
    else
    {
        for (unsigned i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcA[i];

        if (std::is_same<T, float>::value)
        {
            if (total == 0.f)
            {
                // Special edge case: copy vector over without change
                for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
                return;
            }

            // Deal with infinities
            if (total == INFINITY)
            {
                total = 0.0f;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    if (fabsf((float)srcA[i]) == INFINITY)
                        dst[i] = copysignf(1.0f, srcA[i]);
                    else
                        dst[i] = copysignf(0.0f, srcA[i]);
                    total += (double)dst[i] * (double)dst[i];
                }

                srcA = dst;
            }
        }
        else if (std::is_same<T, double>::value)
        {
            if (total < vecSize * DBL_MIN / DBL_EPSILON)
            { // we may have incurred denormalization loss -- rescale
                total = 0;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    dst[i] = srcA[i]
                        * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700); // exact
                    total += dst[i] * dst[i];
                }

                // If still zero
                if (total == 0.0)
                {
                    // Special edge case: copy vector over without change
                    for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
                    return;
                }

                srcA = dst;
            }
            else if (total == INFINITY)
            { // we may have incurred spurious overflow
                double scale =
                    MAKE_HEX_DOUBLE(0x1.0p-512, 0x1LL, -512) / vecSize;
                total = 0;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    dst[i] = srcA[i] * scale; // exact
                    total += dst[i] * dst[i];
                }

                // If there are infinities here, handle those
                if (total == INFINITY)
                {
                    total = 0;
                    for (unsigned i = 0; i < vecSize; i++)
                    {
                        if (isinf(dst[i]))
                        {
                            dst[i] = copysign(1.0, srcA[i]);
                            total += 1.0;
                        }
                        else
                            dst[i] = copysign(0.0, srcA[i]);
                    }
                }
                srcA = dst;
            }
        }
        value = sqrt(total);
        for (unsigned i = 0; i < vecSize; i++)
            dst[i] = (T)((double)srcA[i] / value);
    }
}

//--------------------------------------------------------------------------

template <typename T>
GeomTestParams<T>::GeomTestParams(const ExplicitTypes &dt,
                                  const std::string &name, const float &ulp)
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
        trickyValues.assign(trickyValuesInit.begin(), trickyValuesInit.end());
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
        trickyValues.assign(trickyValuesInit.begin(), trickyValuesInit.end());
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
        trickyValues.assign(trickyValuesInit.begin(), trickyValuesInit.end());
    }
}

//--------------------------------------------------------------------------

GeometricsFPTest::GeometricsFPTest(cl_device_id device, cl_context context,
                                   cl_command_queue queue)
    : context(context), device(device), queue(queue), floatRoundingMode(0),
      halfRoundingMode(0), floatHasInfNan(false), halfHasInfNan(false),
      halfFlushDenormsToZero(0), num_elements(0)
{}

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::SetUp(int elements)
{
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            GeomTestBase::halfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            GeomTestBase::halfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }

        halfRoundingMode =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);

        cl_device_fp_config config = 0;
        cl_int error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                       sizeof(config), &config, NULL);
        test_error(error, "Unable to get device CL_DEVICE_HALF_FP_CONFIG");
        halfHasInfNan = (0 != (config & CL_FP_INF_NAN));

        halfFlushDenormsToZero = (0 == (config & CL_FP_DENORM));
        log_info("Supports half precision denormals: %s\n",
                 halfFlushDenormsToZero ? "NO" : "YES");
    }

    floatRoundingMode =
        get_default_rounding_mode(device, CL_DEVICE_SINGLE_FP_CONFIG);

    cl_device_fp_config config = 0;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                                   sizeof(config), &config, NULL);
    test_error(error, "Unable to get device CL_DEVICE_SINGLE_FP_CONFIG");
    floatHasInfNan = (0 != (config & CL_FP_INF_NAN));

    return CL_SUCCESS;
}

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

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::Run()
{
    cl_int error = CL_SUCCESS;
    for (auto &&p : params)
    {
        std::snprintf(ftype, sizeof(ftype), "%s",
                      get_explicit_type_name(p->dataType));

        if (p->dataType == kDouble)
            strcpy(extension,
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
        else if (p->dataType == kHalf)
            strcpy(extension,
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n");
        else
            extension[0] = '\0';

        error = RunSingleTest(p.get());
        test_error(error, "GeometricsFPTest::Run: test_relational failed");
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int GeometricsFPTest::VerifyTestSize(size_t &test_size,
                                        const size_t &max_alloc,
                                        const size_t &total_buf_size)
{
    if (maxAllocSize == 0 || maxGlobalMemSize == 0)
    {
        cl_int error = CL_SUCCESS;

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                sizeof(maxAllocSize), &maxAllocSize, NULL);
        error |=
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
        test_error(error, "Unable to get device config");

        log_info("Device supports:\nCL_DEVICE_MAX_MEM_ALLOC_SIZE: "
                 "%gMB\nCL_DEVICE_GLOBAL_MEM_SIZE: %gMB\n",
                 maxGlobalMemSize / (1024.0 * 1024.0),
                 maxAllocSize / (1024.0 * 1024.0));

        if (maxGlobalMemSize > (cl_ulong)SIZE_MAX)
        {
            maxGlobalMemSize = (cl_ulong)SIZE_MAX;
        }
    }

    /* Try to allocate a bit less than the limits */
    unsigned int adjustment = 32 * 1024 * 1024;

    while ((max_alloc > (maxAllocSize - adjustment))
           || (total_buf_size > (maxGlobalMemSize - adjustment)))
    {
        test_size /= 2;
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int CrossFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new GeomTestParams<half>(kHalf, "", 3.f));

    params.emplace_back(new GeomTestParams<float>(kFloat, "", 3.f));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new GeomTestParams<double>(kDouble, "", 3.f));
    return GeometricsFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int CrossFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = CrossKernel<cl_half>(*((GeomTestParams<half> *)p));
            break;
        case kFloat:
            error = CrossKernel<cl_float>(*((GeomTestParams<float> *)p));
            break;
        case kDouble:
            error = CrossKernel<cl_double>(*((GeomTestParams<double> *)p));
            break;
        default:
            test_error(-1, "CrossFPTest::RunSingleTest: incorrect fp type");
            break;
    }
    test_error(error, "CrossFPTest::RunSingleTest: test_relational failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T> int CrossFPTest::CrossKernel(const GeomTestParams<T> &p)
{
    RandomSeed seed(gRandomSeed);

    cl_int error;
    /* Check the default rounding mode */
    if (std::is_same<T, float>::value && floatRoundingMode == 0
        || std::is_same<T, half>::value && halfRoundingMode == 0)
        return -1;

    for (int vecsize = 3; vecsize <= 4; ++vecsize)
    {
        /* Make sure we adhere to the maximum individual allocation size and
         * global memory size limits. */
        size_t test_size = TEST_SIZE;
        size_t maxAllocSize = sizeof(T) * TEST_SIZE * vecsize;
        size_t totalBufSize = maxAllocSize * 3;

        error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
        test_error(error, "VerifyTestSize failed");

        size_t bufSize = sizeof(T) * test_size * vecsize;

        clProgramWrapper program;
        clKernelWrapper kernel;
        clMemWrapper streams[3];
        BufferOwningPtr<T> A(malloc(bufSize));
        BufferOwningPtr<T> B(malloc(bufSize));
        BufferOwningPtr<T> C(malloc(bufSize));
        T testVec[4];
        T *inDataA = A, *inDataB = B, *outData = C;
        size_t threads[1], localThreads[1];

        std::string str =
            concat_kernel(crossKernelSource,
                          sizeof(crossKernelSource) / sizeof(const char *));
        std::string kernelSource =
            string_format(str, vecsize, vecsize, vecsize);

        /* Create kernels */
        const char *programPtr = kernelSource.c_str();
        if (create_single_kernel_helper(context, &program, &kernel, 1,
                                        &programPtr, "sample_test"))
            return -1;

        /* Generate some streams. Note: deliberately do some random data in w to
         * verify that it gets ignored */
        for (int i = 0; i < test_size * vecsize; i++)
        {
            inDataA[i] = get_random<T>(-512, 512, seed);
            inDataB[i] = get_random<T>(-512, 512, seed);
        }
        FillWithTrickyNums(inDataA, inDataB, test_size * vecsize, vecsize, seed,
                           p);

        streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataA, &error);
        if (streams[0] == NULL)
        {
            print_error(error, "ERROR: Creating input array A failed!\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataB, &error);
        if (streams[1] == NULL)
        {
            print_error(error, "ERROR: Creating input array B failed!\n");
            return -1;
        }
        streams[2] =
            clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &error);
        if (streams[2] == NULL)
        {
            print_error(error, "ERROR: Creating output array failed!\n");
            return -1;
        }

        /* Assign streams and execute */
        for (int i = 0; i < 3; i++)
        {
            error = clSetKernelArg(kernel, i, sizeof(streams[i]), &streams[i]);
            test_error(error, "Unable to set indexed kernel arguments");
        }

        /* Run the kernel */
        threads[0] = test_size;
        error = get_max_common_work_group_size(context, kernel, threads[0],
                                               &localThreads[0]);
        test_error(error, "Unable to get work group size to use");

        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                       localThreads, 0, NULL, NULL);
        test_error(error, "Unable to execute test kernel");

        /* Now get the results */
        error = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, bufSize,
                                    outData, 0, NULL, NULL);
        test_error(error, "Unable to read output array!");

        /* And verify! */
        for (int i = 0; i < test_size; i++)
        {
            double errorTolerances[4];
            // On an embedded device w/ round-to-zero, 3 ulps is the worst-case
            // tolerance for cross product
            cross_product<T>(&inDataA[i * vecsize], &inDataB[i * vecsize],
                             testVec, errorTolerances, p.ulpLimit);

            // RTZ devices accrue approximately double the amount of error per
            // operation.  Allow for that.
            if ((std::is_same<T, float>::value
                 && floatRoundingMode == CL_FP_ROUND_TO_ZERO)
                || (std::is_same<T, half>::value
                    && halfRoundingMode == CL_FP_ROUND_TO_ZERO))
            {
                errorTolerances[0] *= 2.0;
                errorTolerances[1] *= 2.0;
                errorTolerances[2] *= 2.0;
                errorTolerances[3] *= 2.0;
            }

            if (verify_cross<T>(i, testVec, &outData[i * vecsize],
                                &inDataA[i * vecsize], &inDataB[i * vecsize],
                                errorTolerances))
            {
                char printout[64];
                std::snprintf(printout, sizeof(printout), "     ulp %f\n",
                              UlpError<T>(outData[i * vecsize + 1],
                                          ToDouble<T>(testVec[1])));
                log_error(printout);
                return -1;
            }
        }
    } // for(vecsize=...
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int TwoArgsFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = TwoArgs<cl_half>(*((TwoArgsTestParams<half> *)p));
            break;
        case kFloat:
            error = TwoArgs<cl_float>(*((TwoArgsTestParams<float> *)p));
            break;
        case kDouble:
            error = TwoArgs<cl_double>(*((TwoArgsTestParams<double> *)p));
            break;
        default:
            test_error(-1, "TwoArgsFPTest::RunSingleTest: incorrect fp type");
            break;
    }
    test_error(error, "TwoArgsFPTest::RunSingleTest: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T> int TwoArgsFPTest::TwoArgs(const TwoArgsTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        cl_int error = TwoArgsKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   geom_two_args vector size %d FAILED\n",
                      (int)sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
T TwoArgsFPTest::GetMaxValue(const T *const vecA, const T *const vecB,
                             const size_t &vecSize,
                             const TwoArgsTestParams<T> &p)
{
    T a = max_fp<T>(abs_fp<T>(vecA[0]), abs_fp<T>(vecB[0]));
    for (size_t i = 1; i < vecSize; i++)
        a = max_fp<T>(abs_fp<T>(vecA[i]), max_fp<T>(abs_fp<T>(vecB[i]), a));
    return a;
}

//--------------------------------------------------------------------------

template <typename T>
int TwoArgsFPTest::TwoArgsKernel(const size_t &vecSize, const MTdata &d,
                                 const TwoArgsTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    int error;
    size_t i, threads[1], localThreads[1];

    // error in sqrt
    float ulpLimit = p.ulpLimit;

    if ((std::is_same<T, float>::value
         && floatRoundingMode == CL_FP_ROUND_TO_ZERO)
        || (std::is_same<T, half>::value
            && halfRoundingMode == CL_FP_ROUND_TO_ZERO))
    {
        char dev_profile[256];
        error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(dev_profile),
                                dev_profile, NULL);
        test_error(error, "Unable to get device profile");
        if (0 == strcmp(dev_profile, "EMBEDDED_PROFILE"))
        {
            // rtz operations average twice the accrued error of rte operations
            ulpLimit *= 2.0f;
        }
    }

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * TEST_SIZE * vecSize;
    size_t totalBufSize = maxAllocSize * 2 + sizeof(T) * TEST_SIZE;

    error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t srcBufSize = sizeof(T) * test_size * vecSize;
    size_t dstBufSize = sizeof(T) * test_size;

    BufferOwningPtr<T> A(malloc(srcBufSize));
    BufferOwningPtr<T> B(malloc(srcBufSize));
    BufferOwningPtr<T> C(malloc(dstBufSize));

    T *inDataA = A, *inDataB = B, *outData = C;

    /* Create the source */
    if (vecSize == 1)
    {
        std::snprintf(load_store, sizeof(load_store), twoArgsToScalarV1,
                      p.fnName.c_str());
    }
    else
    {
        std::snprintf(load_store, sizeof(load_store), twoArgsToScalarVn,
                      p.fnName.c_str(), vecSize, vecSize);
    }
    std::string str =
        concat_kernel(twoArgsKernelPattern,
                      sizeof(twoArgsKernelPattern) / sizeof(const char *));

    /* Create kernels */
    const char *programPtr = str.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
    {
        return -1;
    }
    /* Generate some streams */
    for (i = 0; i < test_size * vecSize; i++)
    {
        inDataA[i] = get_random<T>(-512, 512, d);
        inDataB[i] = get_random<T>(-512, 512, d);
    }
    FillWithTrickyNums(inDataA, inDataB, test_size * vecSize, vecSize, d, p);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataB, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, dstBufSize, NULL, &error);
    if (streams[2] == NULL)
    {
        print_error(error, "ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    for (i = 0; i < 3; i++)
    {
        error = clSetKernelArg(kernel, (int)i, sizeof(streams[i]), &streams[i]);
        test_error(error, "Unable to set indexed kernel arguments");
    }

    /* Run the kernel */
    threads[0] = test_size;
    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[2], true, 0, dstBufSize, outData,
                                0, NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    int skipCount = 0;
    for (i = 0; i < test_size; i++)
    {
        T *src1 = inDataA + i * vecSize;
        T *src2 = inDataB + i * vecSize;
        double expected = p.verifyFunc(src1, src2, vecSize);
        bool isDif = (std::is_same<T, half>::value)
            ? (HFF(expected) != outData[i])
            : ((T)expected != outData[i]);
        if (isDif)
        {
            if (std::is_same<T, half>::value)
            {
                T expv = HFF(expected);
                if (isnan_fp<T>(expv) && isnan_fp<T>(outData[i])
                    || (!isfinite_fp<T>(expv) && !isfinite_fp<T>(outData[i])))
                    continue;
            }
            else if (isnan(expected) && isnan_fp<T>(outData[i]))
                continue;

            if (std::is_same<T, float>::value && !floatHasInfNan
                || std::is_same<T, half>::value && !halfHasInfNan)
            {
                for (size_t ii = 0; ii < vecSize; ii++)
                {
                    if (!isfinite_fp<T>(src1[ii]) || !isfinite_fp<T>(src2[ii]))
                    {
                        skipCount++;
                        continue;
                    }
                }
                if (!std::isfinite(expected))
                {
                    skipCount++;
                    continue;
                }
            }

            std::stringstream sstr;
            std::string printout;
            if (ulpLimit < 0)
            {
                // Limit below zero means we need to test via a computed error
                // (like cross product does)
                T maxValue = GetMaxValue<T>(inDataA + i * vecSize,
                                            inDataB + i * vecSize, vecSize, p);

                double error = 0.0, errorTolerance = 0.0;
                if (std::is_same<T, half>::value == false)
                {
                    // In this case (dot is the only one that gets here), the
                    // ulp is 2*vecSize - 1 (n + n-1 max # of errors)
                    errorTolerance = maxValue * maxValue
                        * (2.f * (float)vecSize - 1.f) * FLT_EPSILON;
                }
                else
                {
                    float mxv = HTF(maxValue);
                    errorTolerance = mxv * mxv * (2.f * (float)vecSize - 1.f)
                        * CL_HALF_EPSILON;
                }

                // Limit below zero means test via epsilon instead
                error = fabs(expected - ToDouble<T>(outData[i]));
                if (error > errorTolerance)
                {
                    printout = string_format(
                        "ERROR: Data sample %d at size %d does not "
                        "validate! Expected (%a), got (%a), sources (%a "
                        "and %a) error of %g against tolerance %g\n",
                        (int)i, (int)vecSize, expected, ToDouble<T>(outData[i]),
                        ToDouble<T>(inDataA[i * vecSize]),
                        ToDouble<T>(inDataB[i * vecSize]), (float)error,
                        (float)errorTolerance);

                    sstr << printout;
                    sstr << "\tvector A: ";
                    vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                    sstr << ", vector B: ";
                    vector2string<T>(sstr, inDataB + i * vecSize, vecSize);
                    log_error(sstr.str().c_str());
                    return -1;
                }
            }
            else
            {
                float error = UlpError<T>(outData[i], expected);
                if (fabsf(error) > ulpLimit)
                {
                    printout = string_format(
                        "ERROR: Data sample %d at size %d does not "
                        "validate! Expected (%a), got (%a), sources (%a "
                        "and %a) ulp of %f\n",
                        (int)i, (int)vecSize, expected, ToDouble<T>(outData[i]),
                        ToDouble<T>(inDataA[i * vecSize]),
                        ToDouble<T>(inDataB[i * vecSize]), error);

                    sstr << printout;
                    sstr << "\tvector A: ";
                    vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                    sstr << ", vector B: ";
                    vector2string<T>(sstr, inDataB + i * vecSize, vecSize);
                    log_error(sstr.str().c_str());
                    return -1;
                }
            }
        }
    }

    if (skipCount)
        log_info(
            "Skipped %d tests out of %d because they contained Infs or "
            "NaNs\n\tEMBEDDED_PROFILE Device does not support CL_FP_INF_NAN\n",
            skipCount, test_size);
    return 0;
}

//--------------------------------------------------------------------------

cl_int DotProdFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(
            new TwoArgsTestParams<half>(&verifyDot, kHalf, "dot", -1.f));

    params.emplace_back(
        new TwoArgsTestParams<float>(&verifyDot, kFloat, "dot", -1.f));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(
            new TwoArgsTestParams<double>(&verifyDot, kDouble, "dot", -1.f));
    return TwoArgsFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastDistanceFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((TwoArgsTestParams<float> *)param);

    float ulpConst = p.ulpLimit;
    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + 2.f * sizes[size]);

        cl_int error = TwoArgsKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   FastDistanceFPTest::RunSingleTest vector size %d FAILED\n",
                (int)sizes[size]);
            return error;
        }
    }

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastDistanceFPTest::SetUp(int elements)
{
    // only float supports fast_distance
    params.emplace_back(new TwoArgsTestParams<float>(
        &verifyFastDistance, kFloat, "fast_distance",
        8191.5f)); // 8191.5 + 2n ulp
    return TwoArgsFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int DistanceFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = DistTest<half>(*((TwoArgsTestParams<half> *)p));
            break;
        case kFloat:
            error = DistTest<float>(*((TwoArgsTestParams<float> *)p));
            break;
        case kDouble:
            error = DistTest<double>(*((TwoArgsTestParams<double> *)p));
            break;
        default: test_error(-1, "TwoArgsFPTest::Run: incorrect fp type"); break;
    }
    test_error(error, "TwoArgsFPTest::Run: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T> int DistanceFPTest::DistTest(TwoArgsTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + 2.f * sizes[size]);

        cl_int error = TwoArgsKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   geom_two_args vector size %d FAILED\n",
                      (int)sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int DistanceFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new TwoArgsTestParams<half>(
            &verifyDistance, kHalf, "distance", 0.f)); // 2n ulp

    params.emplace_back(new TwoArgsTestParams<float>(
        &verifyDistance, kFloat, "distance", 2.5f)); // 2.5 + 2n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new TwoArgsTestParams<double>(
            &verifyDistance, kDouble, "distance", 5.5f)); // 5.5 + 2n ulp
    return TwoArgsFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

template <typename T>
int OneArgFPTest::OneArgKernel(const size_t &vecSize, const MTdata &d,
                               const OneArgTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * test_size * vecSize;
    size_t totalBufSize = maxAllocSize + sizeof(T) * test_size;

    cl_int error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t srcBufSize = sizeof(T) * test_size * vecSize;
    size_t dstBufSize = sizeof(T) * test_size;

    BufferOwningPtr<T> A(malloc(srcBufSize));
    BufferOwningPtr<T> B(malloc(dstBufSize));

    size_t i, threads[1], localThreads[1];
    T *inDataA = A;
    T *outData = B;

    /* Create the source */
    if (vecSize == 1)
        std::snprintf(load_store, sizeof(load_store), oneArgToScalarV1,
                      p.fnName.c_str());
    else
        std::snprintf(load_store, sizeof(load_store), oneArgToScalarVn,
                      p.fnName.c_str(), vecSize);
    std::string str =
        concat_kernel(oneToFloatKernelPattern,
                      sizeof(oneToFloatKernelPattern) / sizeof(const char *));

    /* Create kernels */
    const char *programPtr = str.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
    {
        return -1;
    }

    /* Generate some streams */
    for (i = 0; i < test_size * vecSize; i++)
    {
        inDataA[i] = get_random<T>(-512, 512, d);
    }
    FillWithTrickyNums(inDataA, (T *)nullptr, test_size * vecSize, vecSize, d,
                       p);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, dstBufSize, NULL, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    /* Run the kernel */
    threads[0] = test_size;
    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[1], true, 0, dstBufSize, outData,
                                0, NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    for (i = 0; i < test_size; i++)
    {
        double expected = p.verifyFunc(inDataA + i * vecSize, vecSize);
        bool isDif = (std::is_same<T, half>::value)
            ? (HFF(expected) != outData[i])
            : ((T)expected != outData[i]);
        if (isDif)
        {
            float ulps = UlpError<T>(outData[i], expected);
            if (fabsf(ulps) <= p.ulpLimit) continue;

            // We have to special case NAN
            if (isnan_fp<T>(outData[i]) && isnan(expected)) continue;

            if (!(fabsf(ulps) < p.ulpLimit))
            {
                std::stringstream sstr;
                std::string printout = string_format(
                    "ERROR: Data sample %d at size %d does not validate! "
                    "Expected (%a), got (%a), source (%a), ulp %f\n",
                    (int)i, (int)vecSize, expected, ToDouble<T>(outData[i]),
                    ToDouble<T>(inDataA[i * vecSize]), ulps);

                sstr << printout;
                sstr << "\tvector: ";
                vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                log_error(sstr.str().c_str());
                return -1;
            }
        }
    }
    return 0;
}

//--------------------------------------------------------------------------

cl_int LengthFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = LenghtTest<half>(*((OneArgTestParams<half> *)p));
            break;
        case kFloat:
            error = LenghtTest<float>(*((OneArgTestParams<float> *)p));
            break;
        case kDouble:
            error = LenghtTest<double>(*((OneArgTestParams<double> *)p));
            break;
        default: test_error(-1, "LengthFPTest::Run: incorrect fp type"); break;
    }
    test_error(error, "LengthFPTest::RunSingleTest: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T> int LengthFPTest::LenghtTest(OneArgTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + p.ulpMult * sizes[size]);

        cl_int error = OneArgKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   LenghtTest vector size %d FAILED\n",
                      (int)sizes[size]);
            return error;
        }
    }

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int LengthFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new OneArgTestParams<half>(
            &verifyLength, kHalf, "length", 0.25f, 0.5f)); // 0.25 + 0.5n ulp

    params.emplace_back(new OneArgTestParams<float>(
        &verifyLength, kFloat, "length", 2.75f, 0.5f)); // 2.75 + 0.5n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new OneArgTestParams<double>(
            &verifyLength, kDouble, "length", 5.5f, 1.f)); // 5.5 + n ulp
    return OneArgFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastLengthFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((OneArgTestParams<float> *)param);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + sizes[size]);

        cl_int error = OneArgKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   FastLengthFPTest::RunSingleTest vector size %d FAILED\n",
                (int)sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastLengthFPTest::SetUp(int elements)
{
    // only float supports fast_distance
    params.emplace_back(new OneArgTestParams<float>(&verifyFastLength, kFloat,
                                                    "fast_length", 8191.5f,
                                                    1.f)); // 8191.5 + n ulp
    return OneArgFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

template <typename T>
int OneToOneArgFPTest::VerifySubnormals(int &fail, const size_t &vecSize,
                                        const T *const inA, T *const out,
                                        T *const expected,
                                        const OneToOneTestParams<T> &p)
{
    if (std::is_same<T, half>::value)
    {
        if (fail && halfFlushDenormsToZero)
        {
            T temp[4], expected2[4];
            for (size_t j = 0; j < vecSize; j++)
            {
                if (IsHalfSubnormal(inA[j]))
                    temp[j] = HFF(copysignf(0.0f, HTF(inA[j])));
                else
                    temp[j] = inA[j];
            }

            p.verifyFunc(temp, expected2, vecSize);
            fail = 0;

            for (size_t j = 0; j < vecSize; j++)
            {
                // We have to special case NAN
                if (isnan_fp<T>(out[j]) && isnan_fp<T>(expected[j])) continue;

                if (expected2[j] != out[j])
                {
                    float ulp_error =
                        UlpError<T>(out[j], ToDouble<T>(expected[j]));
                    if (fabsf(ulp_error) > p.ulpLimit
                        && IsHalfSubnormal(expected2[j]))
                    {
                        expected2[j] = 0.0f;
                        if (expected2[j] != out[j])
                        {
                            ulp_error =
                                UlpError<T>(out[j], ToDouble<T>(expected[j]));
                            if (fabsf(ulp_error) > p.ulpLimit)
                            {
                                fail = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (std::is_same<T, float>::value)
    {
        if (fail && gFlushDenormsToZero)
        {
            T temp[4], expected2[4];
            for (size_t j = 0; j < vecSize; j++)
            {
                if (IsFloatSubnormal(inA[j]))
                    temp[j] = copysignf(0.0f, inA[j]);
                else
                    temp[j] = inA[j];
            }

            p.verifyFunc(temp, expected2, vecSize);
            fail = 0;

            for (size_t j = 0; j < vecSize; j++)
            {
                // We have to special case NAN
                if (isnan_fp<T>(out[j]) && isnan_fp<T>(expected[j])) continue;

                if (expected2[j] != out[j])
                {
                    float ulp_error =
                        UlpError<T>(out[j], ToDouble<T>(expected[j]));
                    if (fabsf(ulp_error) > p.ulpLimit
                        && IsFloatSubnormal(expected2[j]))
                    {
                        expected2[j] = 0.0f;
                        if (expected2[j] != out[j])
                        {
                            ulp_error =
                                UlpError<T>(out[j], ToDouble<T>(expected[j]));
                            if (fabsf(ulp_error) > p.ulpLimit)
                            {
                                fail = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
int OneToOneArgFPTest::OneToOneArgKernel(const size_t &vecSize, const MTdata &d,
                                         const OneToOneTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * test_size * vecSize;
    size_t totalBufSize = maxAllocSize * 2;

    cl_int error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t bufSize = sizeof(T) * test_size * vecSize;

    BufferOwningPtr<T> A(malloc(bufSize));
    BufferOwningPtr<T> B(malloc(bufSize));
    size_t i, j, threads[1], localThreads[1];

    T *inDataA = A, *outData = B;
    float ulp_error = 0;

    /* Create the source */
    if (vecSize == 1)
        std::snprintf(load_store, sizeof(load_store), oneArgToVecV1,
                      p.fnName.c_str());
    else
        std::snprintf(load_store, sizeof(load_store), oneArgToVecVn, vecSize,
                      p.fnName.c_str(), vecSize);
    std::string str =
        concat_kernel(oneToOneKernelPattern,
                      sizeof(oneToOneKernelPattern) / sizeof(const char *));

    /* Create kernels */
    const char *programPtr = str.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
        return -1;

    /* Initialize data.  First element always 0. */
    memset(inDataA, 0, sizeof(T) * vecSize);
    if (std::is_same<T, float>::value
        && p.fnName.find("fast_") != std::string::npos)
    {
        // keep problematic cases out of the fast function
        for (i = vecSize; i < test_size * vecSize; i++)
        {
            float z = get_random_float(-MAKE_HEX_FLOAT(0x1.0p60f, 1, 60),
                                       MAKE_HEX_FLOAT(0x1.0p60f, 1, 60), d);
            if (fabsf(z) < MAKE_HEX_FLOAT(0x1.0p-60f, 1, -60))
                z = copysignf(0.0f, z);
            inDataA[i] = z;
        }
    }
    else
    {
        for (i = vecSize; i < test_size * vecSize; i++)
            inDataA[i] = any_value<T>(d);
    }

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize, inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    /* Run the kernel */
    threads[0] = test_size;

    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[1], true, 0, bufSize, outData, 0,
                                NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    for (i = 0; i < test_size; i++)
    {
        T expected[4];
        int fail = 0;
        p.verifyFunc(inDataA + i * vecSize, expected, vecSize);
        for (j = 0; j < vecSize; j++)
        {
            // We have to special case NAN
            if (isnan_fp<T>(outData[i * vecSize + j])
                && isnan_fp<T>(expected[j]))
                continue;

            if (expected[j] != outData[i * vecSize + j])
            {
                ulp_error = UlpError<T>(outData[i * vecSize + j],
                                        ToDouble<T>(expected[j]));
                if (fabsf(ulp_error) > p.ulpLimit)
                {
                    fail = 1;
                    break;
                }
            }
        }

        // try again with subnormals flushed to zero if the platform flushes
        error = VerifySubnormals<T>(fail, vecSize, &inDataA[i * vecSize],
                                    &outData[i * vecSize], expected, p);
        test_error(error, "VerifySubnormals failed");

        if (fail)
        {
            std::stringstream sstr;
            std::string printout = string_format(
                "ERROR: Data sample {%d,%d} at size %d does not validate! "
                "Expected %12.24f (%a), got %12.24f (%a), ulp %f\n",
                (int)i, (int)j, (int)vecSize, ToDouble<T>(expected[j]),
                ToDouble<T>(expected[j]), ToDouble<T>(outData[i * vecSize + j]),
                ToDouble<T>(outData[i * vecSize + j]), ulp_error);
            sstr << printout;

            sstr << "       Source: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(inDataA[i * vecSize + q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(inDataA[i * vecSize + q])
                     << " ";
            sstr << "\n";
            sstr << "       Result: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(outData[i * vecSize + q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(outData[i * vecSize + q])
                     << " ";
            sstr << "\n";
            sstr << "       Expected: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(expected[q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(expected[q]) << " ";
            sstr << "\n";

            log_error(sstr.str().c_str());
            return -1;
        }
    }
    return 0;
}

//--------------------------------------------------------------------------

cl_int NormalizeFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = NormalizeTest<half>(*((OneToOneTestParams<half> *)p));
            break;
        case kFloat:
            error = NormalizeTest<float>(*((OneToOneTestParams<float> *)p));
            break;
        case kDouble:
            error = NormalizeTest<double>(*((OneToOneTestParams<double> *)p));
            break;
        default:
            test_error(-1, "NormalizeFPTest::Run: incorrect fp type");
            break;
    }
    test_error(error, "NormalizeFPTest::Run: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
int NormalizeFPTest::NormalizeTest(OneToOneTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + sizes[size]);

        cl_int error = OneToOneArgKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   NormalizeFPTest::NormalizeTest vector size %d FAILED\n",
                (int)sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int NormalizeFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new OneToOneTestParams<half>(
            &verifyNormalize, kHalf, "normalize", 1.f)); // 1 + n ulp

    params.emplace_back(new OneToOneTestParams<float>(
        &verifyNormalize, kFloat, "normalize", 2.f)); // 2 + n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new OneToOneTestParams<double>(
            &verifyNormalize, kDouble, "normalize", 4.5f)); // 4.5 + n ulp
    return OneToOneArgFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastNormalizeFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((OneToOneTestParams<float> *)param);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + sizes[size]);

        cl_int error = OneToOneArgKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   FastNormalizeFPTest::RunSingleTest vector size %d FAILED\n",
                (int)sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastNormalizeFPTest::SetUp(int elements)
{
    // only float supports fast_distance
    params.emplace_back(new OneToOneTestParams<float>(
        &verifyNormalize, kFloat, "fast_normalize", 8192.f)); // 8192 + n ulp
    return OneToOneArgFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

int test_geom_cross(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CrossFPTest>(device, context, queue, num_elements);
}

//--------------------------------------------------------------------------

int test_geom_dot(cl_device_id device, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<DotProdFPTest>(device, context, queue, num_elements);
}

//--------------------------------------------------------------------------

int test_geom_fast_distance(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastDistanceFPTest>(device, context, queue,
                                              num_elems);
}

//--------------------------------------------------------------------------

int test_geom_distance(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<DistanceFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_length(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<LengthFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_fast_length(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastLengthFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_normalize(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<NormalizeFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_fast_normalize(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastNormalizeFPTest>(device, context, queue,
                                               num_elems);
}
