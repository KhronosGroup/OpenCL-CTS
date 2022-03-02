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

#include "reference_math.h"
#include "harness/compat.h"

#include <climits>

#if !defined(_WIN32)
#include <cstring>
#endif

#include "utility.h"

#if defined(__SSE__)                                                           \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
#include <xmmintrin.h>
#endif
#if defined(__SSE2__)                                                          \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
#include <emmintrin.h>
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI / 4)
#endif

#pragma STDC FP_CONTRACT OFF
static void __log2_ep(double *hi, double *lo, double x);

union uint64d_t {
    uint64_t i;
    double d;
};

static const uint64d_t _CL_NAN = { 0x7ff8000000000000ULL };

#define cl_make_nan() _CL_NAN.d

static double reduce1(double x)
{
    if (fabs(x) >= HEX_DBL(+, 1, 0, +, 53))
    {
        if (fabs(x) == INFINITY) return cl_make_nan();

        return 0.0; // we patch up the sign for sinPi and cosPi later, since
                    // they need different signs
    }

    // Find the nearest multiple of 2
    const double r = copysign(HEX_DBL(+, 1, 0, +, 53), x);
    double z = x + r;
    z -= r;

    // subtract it from x. Value is now in the range -1 <= x <= 1
    return x - z;
}

double reference_acospi(double x) { return reference_acos(x) / M_PI; }
double reference_asinpi(double x) { return reference_asin(x) / M_PI; }
double reference_atanpi(double x) { return reference_atan(x) / M_PI; }
double reference_atan2pi(double y, double x)
{
    return reference_atan2(y, x) / M_PI;
}
double reference_cospi(double x)
{
    if (reference_fabs(x) >= HEX_DBL(+, 1, 0, +, 52))
    {
        if (reference_fabs(x) == INFINITY) return cl_make_nan();

        // Note this probably fails for odd values between 0x1.0p52 and
        // 0x1.0p53. However, when starting with single precision inputs, there
        // will be no odd values.

        return 1.0;
    }

    x = reduce1(x + 0.5);

    // reduce to [-0.5, 0.5]
    if (x < -0.5)
        x = -1 - x;
    else if (x > 0.5)
        x = 1 - x;

    // cosPi zeros are all +0
    if (x == 0.0) return 0.0;

    return reference_sin(x * M_PI);
}

double reference_relaxed_cospi(double x) { return reference_cospi(x); }

double reference_relaxed_divide(double x, double y)
{
    return (float)(((float)x) / ((float)y));
}

double reference_divide(double x, double y) { return x / y; }

// Add a + b. If the result modulo overflowed, write 1 to *carry, otherwise 0
static inline cl_ulong add_carry(cl_ulong a, cl_ulong b, cl_ulong *carry)
{
    cl_ulong result = a + b;
    *carry = result < a;
    return result;
}

// Subtract a - b. If the result modulo overflowed, write 1 to *carry, otherwise
// 0
static inline cl_ulong sub_carry(cl_ulong a, cl_ulong b, cl_ulong *carry)
{
    cl_ulong result = a - b;
    *carry = result > a;
    return result;
}

static float fallback_frexpf(float x, int *iptr)
{
    cl_uint u, v;
    float fu, fv;

    memcpy(&u, &x, sizeof(u));

    cl_uint exponent = u & 0x7f800000U;
    cl_uint mantissa = u & ~0x7f800000U;

    // add 1 to the exponent
    exponent += 0x00800000U;

    if ((cl_int)exponent < (cl_int)0x01000000)
    { // subnormal, NaN, Inf
        mantissa |= 0x3f000000U;

        v = mantissa & 0xff800000U;
        u = mantissa;
        memcpy(&fv, &v, sizeof(v));
        memcpy(&fu, &u, sizeof(u));

        fu -= fv;

        memcpy(&v, &fv, sizeof(v));
        memcpy(&u, &fu, sizeof(u));

        exponent = u & 0x7f800000U;
        mantissa = u & ~0x7f800000U;

        *iptr = (exponent >> 23) + (-126 + 1 - 126);
        u = mantissa | 0x3f000000U;
        memcpy(&fu, &u, sizeof(u));
        return fu;
    }

    *iptr = (exponent >> 23) - 127;
    u = mantissa | 0x3f000000U;
    memcpy(&fu, &u, sizeof(u));
    return fu;
}

static inline int extractf(float x, cl_uint *mant)
{
    static float (*frexppf)(float, int *) = NULL;
    int e;

    // verify that frexp works properly
    if (NULL == frexppf)
    {
        if (0.5f == frexpf(HEX_FLT(+, 1, 0, -, 130), &e) && e == -129)
            frexppf = frexpf;
        else
            frexppf = fallback_frexpf;
    }

    *mant = (cl_uint)(HEX_FLT(+, 1, 0, +, 32) * fabsf(frexppf(x, &e)));
    return e - 1;
}

// Shift right by shift bits. Any bits lost on the right side are bitwise OR'd
// together and ORd into the LSB of the result
static inline void shift_right_sticky_64(cl_ulong *p, int shift)
{
    cl_ulong sticky = 0;
    cl_ulong r = *p;

    // C doesn't handle shifts greater than the size of the variable dependably
    if (shift >= 64)
    {
        sticky |= (0 != r);
        r = 0;
    }
    else
    {
        sticky |= (0 != (r << (64 - shift)));
        r >>= shift;
    }

    *p = r | sticky;
}

// Add two 64 bit mantissas. Bits that are below the LSB of the result are OR'd
// into the LSB of the result
static inline void add64(cl_ulong *p, cl_ulong c, int *exponent)
{
    cl_ulong carry;
    c = add_carry(c, *p, &carry);
    if (carry)
    {
        carry = c & 1; // set aside sticky bit
        c >>= 1; // right shift to deal with overflow
        c |= carry
            | 0x8000000000000000ULL; // or in carry bit, and sticky bit. The
                                     // latter is to prevent rounding from
                                     // believing we are exact half way case
        *exponent = *exponent + 1; // adjust exponent
    }

    *p = c;
}

// IEEE-754 round to nearest, ties to even rounding
static float round_to_nearest_even_float(cl_ulong p, int exponent)
{
    union {
        cl_uint u;
        cl_float d;
    } u;

    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if (exponent > 127)
    {
        volatile float r = exponent * CL_FLT_MAX; // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if (r > CL_FLT_MAX) return INFINITY;

        return r;
    }
    if (exponent == -150 && p > 0x8000000000000000ULL)
        return HEX_FLT(+, 1, 0, -, 149);
    if (exponent <= -150) return 0.0f;

    // Figure out which bits go where
    int shift = 8 + 32;
    if (exponent < -126)
    {
        shift -= 126 + exponent; // subnormal: shift is not 52
        exponent = -127; //            set exponent to 0
    }
    else
        p &= 0x7fffffffffffffffULL; // normal: leading bit is implicit. Remove
                                    // it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(p >> shift) | ((cl_uint)(exponent + 127) << 23);

    // put a representation of the residual bits into hi
    p <<= (64 - shift);

    // round to nearest, ties to even  based on the unused portion of p
    if (p < 0x8000000000000000ULL) return u.d;
    if (p == 0x8000000000000000ULL)
        u.u += u.u & 1U;
    else
        u.u++;

    return u.d;
}

static float round_to_nearest_even_float_ftz(cl_ulong p, int exponent)
{
    extern int gCheckTininessBeforeRounding;

    union {
        cl_uint u;
        cl_float d;
    } u;
    int shift = 8 + 32;

    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if (exponent > 127)
    {
        volatile float r = exponent * CL_FLT_MAX; // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if (r > CL_FLT_MAX) return INFINITY;

        return r;
    }

    // Deal with FTZ for gCheckTininessBeforeRounding
    if (exponent < (gCheckTininessBeforeRounding - 127)) return 0.0f;

    if (exponent
        == -127) // only happens for machines that check tininess after rounding
        p = (p & 1) | (p >> 1);
    else
        p &= 0x7fffffffffffffffULL; // normal: leading bit is implicit. Remove
                                    // it.

    cl_ulong q = p;


    // Assemble the double (round toward zero)
    u.u = (cl_uint)(q >> shift) | ((cl_uint)(exponent + 127) << 23);

    // put a representation of the residual bits into hi
    q <<= (64 - shift);

    // round to nearest, ties to even  based on the unused portion of p
    if (q > 0x8000000000000000ULL)
        u.u++;
    else if (q == 0x8000000000000000ULL)
        u.u += u.u & 1U;

    // Deal with FTZ for ! gCheckTininessBeforeRounding
    if (0 == (u.u & 0x7f800000U)) return 0.0f;

    return u.d;
}


// IEEE-754 round toward zero.
static float round_toward_zero_float(cl_ulong p, int exponent)
{
    union {
        cl_uint u;
        cl_float d;
    } u;

    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if (exponent > 127)
    {
        volatile float r = exponent * CL_FLT_MAX; // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if (r > CL_FLT_MAX) return CL_FLT_MAX;

        return r;
    }

    if (exponent <= -149) return 0.0f;

    // Figure out which bits go where
    int shift = 8 + 32;
    if (exponent < -126)
    {
        shift -= 126 + exponent; // subnormal: shift is not 52
        exponent = -127; //            set exponent to 0
    }
    else
        p &= 0x7fffffffffffffffULL; // normal: leading bit is implicit. Remove
                                    // it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(p >> shift) | ((cl_uint)(exponent + 127) << 23);

    return u.d;
}

static float round_toward_zero_float_ftz(cl_ulong p, int exponent)
{
    union {
        cl_uint u;
        cl_float d;
    } u;
    int shift = 8 + 32;

    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if (exponent > 127)
    {
        volatile float r = exponent * CL_FLT_MAX; // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if (r > CL_FLT_MAX) return CL_FLT_MAX;

        return r;
    }

    // Deal with FTZ for gCheckTininessBeforeRounding
    if (exponent < -126) return 0.0f;

    cl_ulong q = p &=
        0x7fffffffffffffffULL; // normal: leading bit is implicit. Remove it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(q >> shift) | ((cl_uint)(exponent + 127) << 23);

    // put a representation of the residual bits into hi
    q <<= (64 - shift);

    return u.d;
}

// Subtract two significands.
static inline void sub64(cl_ulong *c, cl_ulong p, cl_uint *signC, int *expC)
{
    cl_ulong carry;
    p = sub_carry(*c, p, &carry);

    if (carry)
    {
        *signC ^= 0x80000000U;
        p = -p;
    }

    // normalize
    if (p)
    {
        int shift = 32;
        cl_ulong test = 1ULL << 32;
        while (0 == (p & 0x8000000000000000ULL))
        {
            if (p < test)
            {
                p <<= shift;
                *expC = *expC - shift;
            }
            shift >>= 1;
            test <<= shift;
        }
    }
    else
    {
        // zero result.
        *expC = -200;
        *signC =
            0; // IEEE rules say a - a = +0 for all rounding modes except -inf
    }

    *c = p;
}


float reference_fma(float a, float b, float c, int shouldFlush)
{
    static const cl_uint kMSB = 0x80000000U;

    // Make bits accessible
    union {
        cl_uint u;
        cl_float d;
    } ua;
    ua.d = a;
    union {
        cl_uint u;
        cl_float d;
    } ub;
    ub.d = b;
    union {
        cl_uint u;
        cl_float d;
    } uc;
    uc.d = c;

    // deal with Nans, infinities and zeros
    if (isnan(a) || isnan(b) || isnan(c) || isinf(a) || isinf(b) || isinf(c)
        || 0 == (ua.u & ~kMSB) || // a == 0, defeat host FTZ behavior
        0 == (ub.u & ~kMSB) || // b == 0, defeat host FTZ behavior
        0 == (uc.u & ~kMSB)) // c == 0, defeat host FTZ behavior
    {
        FPU_mode_type oldMode;
        RoundingMode oldRoundMode = kRoundToNearestEven;
        if (isinf(c) && !isinf(a) && !isinf(b)) return (c + a) + b;

        if (gIsInRTZMode) oldRoundMode = set_round(kRoundTowardZero, kfloat);

        memset(&oldMode, 0, sizeof(oldMode));
        if (shouldFlush) ForceFTZ(&oldMode);

        a = (float)reference_multiply(
            a, b); // some risk that the compiler will insert a non-compliant
                   // fma here on some platforms.
        a = (float)reference_add(
            a,
            c); // We use STDC FP_CONTRACT OFF above to attempt to defeat that.

        if (shouldFlush) RestoreFPState(&oldMode);

        if (gIsInRTZMode) set_round(oldRoundMode, kfloat);
        return a;
    }

    // extract exponent and mantissa
    //   exponent is a standard unbiased signed integer
    //   mantissa is a cl_uint, with leading non-zero bit positioned at the MSB
    cl_uint mantA, mantB, mantC;
    int expA = extractf(a, &mantA);
    int expB = extractf(b, &mantB);
    int expC = extractf(c, &mantC);
    cl_uint signC = uc.u & kMSB; // We'll need the sign bit of C later to decide
                                 // if we are adding or subtracting

    // exact product of A and B
    int exponent = expA + expB;
    cl_uint sign = (ua.u ^ ub.u) & kMSB;
    cl_ulong product = (cl_ulong)mantA * (cl_ulong)mantB;

    // renormalize -- 1.m * 1.n yields a number between 1.0 and 3.99999..
    //  The MSB might not be set. If so, fix that. Otherwise, reflect the fact
    //  that we got another power of two from the multiplication
    if (0 == (0x8000000000000000ULL & product))
        product <<= 1;
    else
        exponent++; // 2**31 * 2**31 gives 2**62. If the MSB was set, then our
                    // exponent increased.

    // infinite precision add
    cl_ulong addend = (cl_ulong)mantC << 32;
    if (exponent >= expC)
    {
        // Shift C relative to the product so that their exponents match
        if (exponent > expC) shift_right_sticky_64(&addend, exponent - expC);

        // Add
        if (sign ^ signC)
            sub64(&product, addend, &sign, &exponent);
        else
            add64(&product, addend, &exponent);
    }
    else
    {
        // Shift the product relative to C so that their exponents match
        shift_right_sticky_64(&product, expC - exponent);

        // add
        if (sign ^ signC)
            sub64(&addend, product, &signC, &expC);
        else
            add64(&addend, product, &expC);

        product = addend;
        exponent = expC;
        sign = signC;
    }

    // round to IEEE result -- we do not do flushing to zero here. That part is
    // handled manually in ternary.c.
    if (gIsInRTZMode)
    {
        if (shouldFlush)
            ua.d = round_toward_zero_float_ftz(product, exponent);
        else
            ua.d = round_toward_zero_float(product, exponent);
    }
    else
    {
        if (shouldFlush)
            ua.d = round_to_nearest_even_float_ftz(product, exponent);
        else
            ua.d = round_to_nearest_even_float(product, exponent);
    }

    // Set the sign
    ua.u |= sign;

    return ua.d;
}

double reference_relaxed_exp10(double x) { return reference_exp10(x); }

double reference_exp10(double x)
{
    return reference_exp2(x * HEX_DBL(+, 1, a934f0979a371, +, 1));
}


int reference_ilogb(double x)
{
    extern int gDeviceILogb0, gDeviceILogbNaN;
    union {
        cl_double f;
        cl_ulong u;
    } u;

    u.f = (float)x;
    cl_int exponent = (cl_int)(u.u >> 52) & 0x7ff;
    if (exponent == 0x7ff)
    {
        if (u.u & 0x000fffffffffffffULL) return gDeviceILogbNaN;

        return CL_INT_MAX;
    }

    if (exponent == 0)
    { // deal with denormals
        u.f = x * HEX_DBL(+, 1, 0, +, 64);
        exponent = (cl_int)(u.u >> 52) & 0x7ff;
        if (exponent == 0) return gDeviceILogb0;

        return exponent - (1023 + 64);
    }

    return exponent - 1023;
}

double reference_nan(cl_uint x)
{
    union {
        cl_uint u;
        cl_float f;
    } u;
    u.u = x | 0x7fc00000U;
    return (double)u.f;
}

double reference_maxmag(double x, double y)
{
    double fabsx = fabs(x);
    double fabsy = fabs(y);

    if (fabsx < fabsy) return y;

    if (fabsy < fabsx) return x;

    return reference_fmax(x, y);
}

double reference_minmag(double x, double y)
{
    double fabsx = fabs(x);
    double fabsy = fabs(y);

    if (fabsx > fabsy) return y;

    if (fabsy > fabsx) return x;

    return reference_fmin(x, y);
}

double reference_relaxed_mad(double a, double b, double c)
{
    return ((float)a) * ((float)b) + (float)c;
}

double reference_mad(double a, double b, double c) { return a * b + c; }

double reference_recip(double x) { return 1.0 / x; }
double reference_rootn(double x, int i)
{

    // rootn ( x, 0 )  returns a NaN.
    if (0 == i) return cl_make_nan();

    // rootn ( x, n )  returns a NaN for x < 0 and n is even.
    if (x < 0 && 0 == (i & 1)) return cl_make_nan();

    if (x == 0.0)
    {
        switch (i & 0x80000001)
        {
            // rootn ( +-0,  n ) is +0 for even n > 0.
            case 0: return 0.0f;

            // rootn ( +-0,  n ) is +-0 for odd n > 0.
            case 1: return x;

            // rootn ( +-0,  n ) is +inf for even n < 0.
            case 0x80000000: return INFINITY;

            // rootn ( +-0,  n ) is +-inf for odd n < 0.
            case 0x80000001: return copysign(INFINITY, x);
        }
    }

    double sign = x;
    x = reference_fabs(x);
    x = reference_exp2(reference_log2(x) / (double)i);
    return reference_copysignd(x, sign);
}

double reference_rsqrt(double x) { return 1.0 / reference_sqrt(x); }

double reference_sinpi(double x)
{
    double r = reduce1(x);

    // reduce to [-0.5, 0.5]
    if (r < -0.5)
        r = -1 - r;
    else if (r > 0.5)
        r = 1 - r;

    // sinPi zeros have the same sign as x
    if (r == 0.0) return reference_copysignd(0.0, x);

    return reference_sin(r * M_PI);
}

double reference_relaxed_sinpi(double x) { return reference_sinpi(x); }

double reference_tanpi(double x)
{
    // set aside the sign  (allows us to preserve sign of -0)
    double sign = reference_copysignd(1.0, x);
    double z = reference_fabs(x);

    // if big and even  -- caution: only works if x only has single precision
    if (z >= HEX_DBL(+, 1, 0, +, 24))
    {
        if (z == INFINITY) return x - x; // nan

        return reference_copysignd(
            0.0, x); // tanpi ( n ) is copysign( 0.0, n)  for even integers n.
    }

    // reduce to the range [ -0.5, 0.5 ]
    double nearest = reference_rint(z); // round to nearest even places n + 0.5
                                        // values in the right place for us
    int i = (int)nearest; // test above against 0x1.0p24 avoids overflow here
    z -= nearest;

    // correction for odd integer x for the right sign of zero
    if ((i & 1) && z == 0.0) sign = -sign;

    // track changes to the sign
    sign *= reference_copysignd(1.0, z); // really should just be an xor
    z = reference_fabs(z); // remove the sign again

    // reduce once more
    // If we don't do this, rounding error in z * M_PI will cause us not to
    // return infinities properly
    if (z > 0.25)
    {
        z = 0.5 - z;
        return sign
            / reference_tan(z * M_PI); // use system tan to get the right result
    }

    //
    return sign
        * reference_tan(z * M_PI); // use system tan to get the right result
}

double reference_pown(double x, int i) { return reference_pow(x, (double)i); }
double reference_powr(double x, double y)
{
    // powr ( x, y ) returns NaN for x < 0.
    if (x < 0.0) return cl_make_nan();

    // powr ( x, NaN ) returns the NaN for x >= 0.
    // powr ( NaN, y ) returns the NaN.
    if (isnan(x) || isnan(y))
        return x + y; // Note: behavior different here than for pow(1,NaN),
                      // pow(NaN, 0)

    if (x == 1.0)
    {
        // powr ( +1, +-inf ) returns NaN.
        if (reference_fabs(y) == INFINITY) return cl_make_nan();

        // powr ( +1, y ) is 1 for finite y.    (NaN handled above)
        return 1.0;
    }

    if (y == 0.0)
    {
        // powr ( +inf, +-0 ) returns NaN.
        // powr ( +-0, +-0 ) returns NaN.
        if (x == 0.0 || x == INFINITY) return cl_make_nan();

        // powr ( x, +-0 ) is 1 for finite x > 0.  (x <= 0, NaN, INF already
        // handled above)
        return 1.0;
    }

    if (x == 0.0)
    {
        // powr ( +-0, -inf) is +inf.
        // powr ( +-0, y ) is +inf for finite y < 0.
        if (y < 0.0) return INFINITY;

        // powr ( +-0, y ) is +0 for y > 0.    (NaN, y==0 handled above)
        return 0.0;
    }

    // x = +inf
    if (isinf(x))
    {
        if (y < 0) return 0;
        return INFINITY;
    }

    double fabsx = reference_fabs(x);
    double fabsy = reference_fabs(y);

    // y = +-inf cases
    if (isinf(fabsy))
    {
        if (y < 0)
        {
            if (fabsx < 1) return INFINITY;
            return 0;
        }
        if (fabsx < 1) return 0;
        return INFINITY;
    }

    double hi, lo;
    __log2_ep(&hi, &lo, x);
    double prod = y * hi;
    double result = reference_exp2(prod);

    return result;
}

double reference_fract(double x, double *ip)
{
    if (isnan(x))
    {
        *ip = cl_make_nan();
        return cl_make_nan();
    }

    float i;
    float f = modff((float)x, &i);
    if (f < 0.0)
    {
        f = 1.0f + f;
        i -= 1.0f;
        if (f == 1.0f) f = HEX_FLT(+, 1, fffffe, -, 1);
    }
    *ip = i;
    return f;
}


double reference_add(double x, double y)
{
    volatile float a = (float)x;
    volatile float b = (float)y;

#if defined(__SSE__)                                                           \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss((float)a);
    __m128 vb = _mm_set_ss((float)b);
    va = _mm_add_ss(va, vb);
    _mm_store_ss((float *)&a, va);
#elif defined(__PPC__)
    // Most Power host CPUs do not support the non-IEEE mode (NI) which flushes
    // denorm's to zero. As such, the reference add with FTZ must be emulated in
    // sw.
    if (fpu_control & _FPU_MASK_NI)
    {
        union {
            cl_uint u;
            cl_float d;
        } ua;
        ua.d = a;
        union {
            cl_uint u;
            cl_float d;
        } ub;
        ub.d = b;
        cl_uint mantA, mantB;
        cl_ulong addendA, addendB, sum;
        int expA = extractf(a, &mantA);
        int expB = extractf(b, &mantB);
        cl_uint signA = ua.u & 0x80000000U;
        cl_uint signB = ub.u & 0x80000000U;

        // Force matching exponents if an operand is 0
        if (a == 0.0f)
        {
            expA = expB;
        }
        else if (b == 0.0f)
        {
            expB = expA;
        }

        addendA = (cl_ulong)mantA << 32;
        addendB = (cl_ulong)mantB << 32;

        if (expA >= expB)
        {
            // Shift B relative to the A so that their exponents match
            if (expA > expB) shift_right_sticky_64(&addendB, expA - expB);

            // add
            if (signA ^ signB)
                sub64(&addendA, addendB, &signA, &expA);
            else
                add64(&addendA, addendB, &expA);
        }
        else
        {
            // Shift the A relative to B so that their exponents match
            shift_right_sticky_64(&addendA, expB - expA);

            // add
            if (signA ^ signB)
                sub64(&addendB, addendA, &signB, &expB);
            else
                add64(&addendB, addendA, &expB);

            addendA = addendB;
            expA = expB;
            signA = signB;
        }

        // round to IEEE result
        if (gIsInRTZMode)
        {
            ua.d = round_toward_zero_float_ftz(addendA, expA);
        }
        else
        {
            ua.d = round_to_nearest_even_float_ftz(addendA, expA);
        }
        // Set the sign
        ua.u |= signA;
        a = ua.d;
    }
    else
    {
        a += b;
    }
#else
    a += b;
#endif
    return (double)a;
}


double reference_subtract(double x, double y)
{
    volatile float a = (float)x;
    volatile float b = (float)y;
#if defined(__SSE__)                                                           \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss((float)a);
    __m128 vb = _mm_set_ss((float)b);
    va = _mm_sub_ss(va, vb);
    _mm_store_ss((float *)&a, va);
#else
    a -= b;
#endif
    return a;
}

double reference_multiply(double x, double y)
{
    volatile float a = (float)x;
    volatile float b = (float)y;
#if defined(__SSE__)                                                           \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss((float)a);
    __m128 vb = _mm_set_ss((float)b);
    va = _mm_mul_ss(va, vb);
    _mm_store_ss((float *)&a, va);
#elif defined(__PPC__)
    // Most Power host CPUs do not support the non-IEEE mode (NI) which flushes
    // denorm's to zero. As such, the reference multiply with FTZ must be
    // emulated in sw.
    if (fpu_control & _FPU_MASK_NI)
    {
        // extract exponent and mantissa
        //   exponent is a standard unbiased signed integer
        //   mantissa is a cl_uint, with leading non-zero bit positioned at the
        //   MSB
        union {
            cl_uint u;
            cl_float d;
        } ua;
        ua.d = a;
        union {
            cl_uint u;
            cl_float d;
        } ub;
        ub.d = b;
        cl_uint mantA, mantB;
        int expA = extractf(a, &mantA);
        int expB = extractf(b, &mantB);

        // exact product of A and B
        int exponent = expA + expB;
        cl_uint sign = (ua.u ^ ub.u) & 0x80000000U;
        cl_ulong product = (cl_ulong)mantA * (cl_ulong)mantB;

        // renormalize -- 1.m * 1.n yields a number between 1.0 and 3.99999..
        //  The MSB might not be set. If so, fix that. Otherwise, reflect the
        //  fact that we got another power of two from the multiplication
        if (0 == (0x8000000000000000ULL & product))
            product <<= 1;
        else
            exponent++; // 2**31 * 2**31 gives 2**62. If the MSB was set, then
                        // our exponent increased.

        // round to IEEE result -- we do not do flushing to zero here. That part
        // is handled manually in ternary.c.
        if (gIsInRTZMode)
        {
            ua.d = round_toward_zero_float_ftz(product, exponent);
        }
        else
        {
            ua.d = round_to_nearest_even_float_ftz(product, exponent);
        }
        // Set the sign
        ua.u |= sign;
        a = ua.d;
    }
    else
    {
        a *= b;
    }
#else
    a *= b;
#endif
    return a;
}

double reference_lgamma_r(double x, int *signp)
{
    // This is not currently tested
    *signp = 0;
    return x;
}


int reference_isequal(double x, double y) { return x == y; }
int reference_isfinite(double x) { return 0 != isfinite(x); }
int reference_isgreater(double x, double y) { return x > y; }
int reference_isgreaterequal(double x, double y) { return x >= y; }
int reference_isinf(double x) { return 0 != isinf(x); }
int reference_isless(double x, double y) { return x < y; }
int reference_islessequal(double x, double y) { return x <= y; }
int reference_islessgreater(double x, double y)
{
    return 0 != islessgreater(x, y);
}
int reference_isnan(double x) { return 0 != isnan(x); }
int reference_isnormal(double x) { return 0 != isnormal((float)x); }
int reference_isnotequal(double x, double y) { return x != y; }
int reference_isordered(double x, double y) { return x == x && y == y; }
int reference_isunordered(double x, double y) { return isnan(x) || isnan(y); }
int reference_signbit(float x) { return 0 != signbit(x); }

#if 1 // defined( _MSC_VER )

// Missing functions for win32


float reference_copysign(float x, float y)
{
    union {
        float f;
        cl_uint u;
    } ux, uy;
    ux.f = x;
    uy.f = y;
    ux.u &= 0x7fffffffU;
    ux.u |= uy.u & 0x80000000U;
    return ux.f;
}


double reference_copysignd(double x, double y)
{
    union {
        double f;
        cl_ulong u;
    } ux, uy;
    ux.f = x;
    uy.f = y;
    ux.u &= 0x7fffffffffffffffULL;
    ux.u |= uy.u & 0x8000000000000000ULL;
    return ux.f;
}


double reference_round(double x)
{
    double absx = reference_fabs(x);
    if (absx < 0.5) return reference_copysignd(0.0, x);

    if (absx < HEX_DBL(+, 1, 0, +, 53))
        x = reference_trunc(x + reference_copysignd(0.5, x));

    return x;
}

double reference_trunc(double x)
{
    if (fabs(x) < HEX_DBL(+, 1, 0, +, 53))
    {
        cl_long l = (cl_long)x;

        return reference_copysignd((double)l, x);
    }

    return x;
}

#ifndef FP_ILOGB0
#define FP_ILOGB0 INT_MIN
#endif

#ifndef FP_ILOGBNAN
#define FP_ILOGBNAN INT_MAX
#endif


double reference_cbrt(double x)
{
    return reference_copysignd(reference_pow(reference_fabs(x), 1.0 / 3.0), x);
}

double reference_rint(double x)
{
    if (reference_fabs(x) < HEX_DBL(+, 1, 0, +, 52))
    {
        double magic = reference_copysignd(HEX_DBL(+, 1, 0, +, 52), x);
        double rounded = (x + magic) - magic;
        x = reference_copysignd(rounded, x);
    }

    return x;
}

double reference_acosh(double x)
{ // not full precision. Sufficient precision to cover float
    if (isnan(x)) return x + x;

    if (x < 1.0) return cl_make_nan();

    return reference_log(x + reference_sqrt(x + 1) * reference_sqrt(x - 1));
}

double reference_asinh(double x)
{
    /*
     * ====================================================
     * This function is from fdlibm: http://www.netlib.org
     *   It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     */
    if (isnan(x) || isinf(x)) return x + x;

    double absx = reference_fabs(x);
    if (absx < HEX_DBL(+, 1, 0, -, 28)) return x;

    double sign = reference_copysignd(1.0, x);

    if (absx > HEX_DBL(+, 1, 0, +, 28))
        return sign
            * (reference_log(absx)
               + 0.693147180559945309417232121458176568); // log(2)

    if (absx > 2.0)
        return sign
            * reference_log(2.0 * absx
                            + 1.0 / (reference_sqrt(x * x + 1.0) + absx));

    return sign
        * reference_log1p(absx + x * x / (1.0 + reference_sqrt(1.0 + x * x)));
}


double reference_atanh(double x)
{
    /*
     * ====================================================
     * This function is from fdlibm: http://www.netlib.org
     *   It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     */
    if (isnan(x)) return x + x;

    double signed_half = reference_copysignd(0.5, x);
    x = reference_fabs(x);
    if (x > 1.0) return cl_make_nan();

    if (x < 0.5)
        return signed_half * reference_log1p(2.0 * (x + x * x / (1 - x)));

    return signed_half * reference_log1p(2.0 * x / (1 - x));
}

double reference_relaxed_atan(double x) { return reference_atan(x); }

double reference_relaxed_exp2(double x) { return reference_exp2(x); }

double reference_exp2(double x)
{ // Note: only suitable for verifying single precision. Doesn't have range of a
  // full double exp2 implementation.
    if (x == 0.0) return 1.0;

    // separate x into fractional and integer parts
    double i = reference_rint(x); // round to nearest integer

    if (i < -150) return 0.0;

    if (i > 129) return INFINITY;

    double f = x - i; // -0.5 <= f <= 0.5

    // find exp2(f)
    // calculate as p(f) = (exp2(f)-1)/f
    //              exp2(f) = f * p(f) + 1
    // p(f) is a minimax polynomial with error within 0x1.c1fd80f0d1ab7p-50

    double p = 0.693147180560184539289
        + (0.240226506955902863183
           + (0.055504108656833424373
              + (0.009618129212846484796
                 + (0.001333355902958566035
                    + (0.000154034191902497930
                       + (0.000015252317761038105
                          + (0.000001326283129417092
                             + 0.000000102593187638680 * f)
                              * f)
                           * f)
                        * f)
                     * f)
                  * f)
               * f)
            * f;
    f *= p;
    f += 1.0;

    // scale by 2 ** i
    union {
        cl_ulong u;
        double d;
    } u;
    int exponent = (int)i + 1023;
    u.u = (cl_ulong)exponent << 52;

    return f * u.d;
}


double reference_expm1(double x)
{ // Note: only suitable for verifying single precision. Doesn't have range of a
  // full double expm1 implementation. It is only accurate to 47 bits or less.

    // early out for small numbers and NaNs
    if (!(reference_fabs(x) > HEX_DBL(+, 1, 0, -, 24))) return x;

    // early out for large negative numbers
    if (x < -130.0) return -1.0;

    // early out for large positive numbers
    if (x > 100.0) return INFINITY;

    // separate x into fractional and integer parts
    double i = reference_rint(x); // round to nearest integer
    double f = x - i; // -0.5 <= f <= 0.5

    // reduce f to the range -0.0625 .. f.. 0.0625
    int index = (int)(f * 16.0) + 8; // 0...16

    static const double reduction[17] = { -0.5,  -0.4375, -0.375, -0.3125,
                                          -0.25, -0.1875, -0.125, -0.0625,
                                          0.0,   +0.0625, +0.125, +0.1875,
                                          +0.25, +0.3125, +0.375, +0.4375,
                                          +0.5 };


    // exponentials[i] = expm1(reduction[i])
    static const double exponentials[17] = {
        HEX_DBL(-, 1, 92e9a0720d3ec, -, 2),
        HEX_DBL(-, 1, 6adb1cd9205ee, -, 2),
        HEX_DBL(-, 1, 40373d42ce2e3, -, 2),
        HEX_DBL(-, 1, 12d35a41ba104, -, 2),
        HEX_DBL(-, 1, c5041854df7d4, -, 3),
        HEX_DBL(-, 1, 5e25fb4fde211, -, 3),
        HEX_DBL(-, 1, e14aed893eef4, -, 4),
        HEX_DBL(-, 1, f0540438fd5c3, -, 5),
        HEX_DBL(+, 0, 0, +, 0),
        HEX_DBL(+, 1, 082b577d34ed8, -, 4),
        HEX_DBL(+, 1, 10b022db7ae68, -, 3),
        HEX_DBL(+, 1, a65c0b85ac1a9, -, 3),
        HEX_DBL(+, 1, 22d78f0fa061a, -, 2),
        HEX_DBL(+, 1, 77a45d8117fd5, -, 2),
        HEX_DBL(+, 1, d1e944f6fbdaa, -, 2),
        HEX_DBL(+, 1, 190048ef6002, -, 1),
        HEX_DBL(+, 1, 4c2531c3c0d38, -, 1),
    };


    f -= reduction[index];

    // find expm1(f)
    // calculate as p(f) = (exp(f)-1)/f
    //              expm1(f) = f * p(f)
    // p(f) is a minimax polynomial with error within 0x1.1d7693618d001p-48 over
    // the range +- 0.0625
    double p = 0.999999999999998001599
        + (0.499999999999839628284
           + (0.166666666672817459505
              + (0.041666666612283048687
                 + (0.008333330214567431435
                    + (0.001389005319303770070 + 0.000198833381525156667 * f)
                        * f)
                     * f)
                  * f)
               * f)
            * f;
    f *= p; // expm1( reduced f )

    // expm1(f) = (exmp1( reduced_f) + 1.0) * ( exponentials[index] + 1 ) - 1
    //          =  exmp1( reduced_f) * exponentials[index] + exmp1( reduced_f) +
    //          exponentials[index] + 1 -1 =  exmp1( reduced_f) *
    //          exponentials[index] + exmp1( reduced_f) + exponentials[index]
    f += exponentials[index] + f * exponentials[index];

    // scale by e ** i
    int exponent = (int)i;
    if (0 == exponent) return f; // precise answer for x near 1

    // table of e**(i-150)
    static const double exp_table[128 + 150 + 1] = {
        HEX_DBL(+, 1, 82e16284f5ec5, -, 217),
        HEX_DBL(+, 1, 06e9996332ba1, -, 215),
        HEX_DBL(+, 1, 6555cb289e44b, -, 214),
        HEX_DBL(+, 1, e5ab364643354, -, 213),
        HEX_DBL(+, 1, 4a0bd18e64df7, -, 211),
        HEX_DBL(+, 1, c094499cc578e, -, 210),
        HEX_DBL(+, 1, 30d759323998c, -, 208),
        HEX_DBL(+, 1, 9e5278ab1d4cf, -, 207),
        HEX_DBL(+, 1, 198fa3f30be25, -, 205),
        HEX_DBL(+, 1, 7eae636d6144e, -, 204),
        HEX_DBL(+, 1, 040f1036f4863, -, 202),
        HEX_DBL(+, 1, 6174e477a895f, -, 201),
        HEX_DBL(+, 1, e065b82dd95a, -, 200),
        HEX_DBL(+, 1, 4676be491d129, -, 198),
        HEX_DBL(+, 1, bbb5da5f7c823, -, 197),
        HEX_DBL(+, 1, 2d884eef5fdcb, -, 195),
        HEX_DBL(+, 1, 99d3397ab8371, -, 194),
        HEX_DBL(+, 1, 1681497ed15b3, -, 192),
        HEX_DBL(+, 1, 7a870f597fdbd, -, 191),
        HEX_DBL(+, 1, 013c74edba307, -, 189),
        HEX_DBL(+, 1, 5d9ec4ada7938, -, 188),
        HEX_DBL(+, 1, db2edfd20fa7c, -, 187),
        HEX_DBL(+, 1, 42eb9f39afb0b, -, 185),
        HEX_DBL(+, 1, b6e4f282b43f4, -, 184),
        HEX_DBL(+, 1, 2a42764857b19, -, 182),
        HEX_DBL(+, 1, 9560792d19314, -, 181),
        HEX_DBL(+, 1, 137b6ce8e052c, -, 179),
        HEX_DBL(+, 1, 766b45dd84f18, -, 178),
        HEX_DBL(+, 1, fce362fe6e7d, -, 177),
        HEX_DBL(+, 1, 59d34dd8a5473, -, 175),
        HEX_DBL(+, 1, d606847fc727a, -, 174),
        HEX_DBL(+, 1, 3f6a58b795de3, -, 172),
        HEX_DBL(+, 1, b2216c6efdac1, -, 171),
        HEX_DBL(+, 1, 2705b5b153fb8, -, 169),
        HEX_DBL(+, 1, 90fa1509bd50d, -, 168),
        HEX_DBL(+, 1, 107df698da211, -, 166),
        HEX_DBL(+, 1, 725ae6e7b9d35, -, 165),
        HEX_DBL(+, 1, f75d6040aeff6, -, 164),
        HEX_DBL(+, 1, 56126259e093c, -, 162),
        HEX_DBL(+, 1, d0ec7df4f7bd4, -, 161),
        HEX_DBL(+, 1, 3bf2cf6722e46, -, 159),
        HEX_DBL(+, 1, ad6b22f55db42, -, 158),
        HEX_DBL(+, 1, 23d1f3e5834a, -, 156),
        HEX_DBL(+, 1, 8c9feab89b876, -, 155),
        HEX_DBL(+, 1, 0d88cf37f00dd, -, 153),
        HEX_DBL(+, 1, 6e55d2bf838a7, -, 152),
        HEX_DBL(+, 1, f1e6b68529e33, -, 151),
        HEX_DBL(+, 1, 525be4e4e601d, -, 149),
        HEX_DBL(+, 1, cbe0a45f75eb1, -, 148),
        HEX_DBL(+, 1, 3884e838aea68, -, 146),
        HEX_DBL(+, 1, a8c1f14e2af5d, -, 145),
        HEX_DBL(+, 1, 20a717e64a9bd, -, 143),
        HEX_DBL(+, 1, 8851d84118908, -, 142),
        HEX_DBL(+, 1, 0a9bdfb02d24, -, 140),
        HEX_DBL(+, 1, 6a5bea046b42e, -, 139),
        HEX_DBL(+, 1, ec7f3b269efa8, -, 138),
        HEX_DBL(+, 1, 4eafb87eab0f2, -, 136),
        HEX_DBL(+, 1, c6e2d05bbc, -, 135),
        HEX_DBL(+, 1, 35208867c2683, -, 133),
        HEX_DBL(+, 1, a425b317eeacd, -, 132),
        HEX_DBL(+, 1, 1d8508fa8246a, -, 130),
        HEX_DBL(+, 1, 840fbc08fdc8a, -, 129),
        HEX_DBL(+, 1, 07b7112bc1ffe, -, 127),
        HEX_DBL(+, 1, 666d0dad2961d, -, 126),
        HEX_DBL(+, 1, e726c3f64d0fe, -, 125),
        HEX_DBL(+, 1, 4b0dc07cabf98, -, 123),
        HEX_DBL(+, 1, c1f2daf3b6a46, -, 122),
        HEX_DBL(+, 1, 31c5957a47de2, -, 120),
        HEX_DBL(+, 1, 9f96445648b9f, -, 119),
        HEX_DBL(+, 1, 1a6baeadb4fd1, -, 117),
        HEX_DBL(+, 1, 7fd974d372e45, -, 116),
        HEX_DBL(+, 1, 04da4d1452919, -, 114),
        HEX_DBL(+, 1, 62891f06b345, -, 113),
        HEX_DBL(+, 1, e1dd273aa8a4a, -, 112),
        HEX_DBL(+, 1, 4775e0840bfdd, -, 110),
        HEX_DBL(+, 1, bd109d9d94bda, -, 109),
        HEX_DBL(+, 1, 2e73f53fba844, -, 107),
        HEX_DBL(+, 1, 9b138170d6bfe, -, 106),
        HEX_DBL(+, 1, 175af0cf60ec5, -, 104),
        HEX_DBL(+, 1, 7baee1bffa80b, -, 103),
        HEX_DBL(+, 1, 02057d1245ceb, -, 101),
        HEX_DBL(+, 1, 5eafffb34ba31, -, 100),
        HEX_DBL(+, 1, dca23bae16424, -, 99),
        HEX_DBL(+, 1, 43e7fc88b8056, -, 97),
        HEX_DBL(+, 1, b83bf23a9a9eb, -, 96),
        HEX_DBL(+, 1, 2b2b8dd05b318, -, 94),
        HEX_DBL(+, 1, 969d47321e4cc, -, 93),
        HEX_DBL(+, 1, 1452b7723aed2, -, 91),
        HEX_DBL(+, 1, 778fe2497184c, -, 90),
        HEX_DBL(+, 1, fe7116182e9cc, -, 89),
        HEX_DBL(+, 1, 5ae191a99585a, -, 87),
        HEX_DBL(+, 1, d775d87da854d, -, 86),
        HEX_DBL(+, 1, 4063f8cc8bb98, -, 84),
        HEX_DBL(+, 1, b374b315f87c1, -, 83),
        HEX_DBL(+, 1, 27ec458c65e3c, -, 81),
        HEX_DBL(+, 1, 923372c67a074, -, 80),
        HEX_DBL(+, 1, 1152eaeb73c08, -, 78),
        HEX_DBL(+, 1, 737c5645114b5, -, 77),
        HEX_DBL(+, 1, f8e6c24b5592e, -, 76),
        HEX_DBL(+, 1, 571db733a9d61, -, 74),
        HEX_DBL(+, 1, d257d547e083f, -, 73),
        HEX_DBL(+, 1, 3ce9b9de78f85, -, 71),
        HEX_DBL(+, 1, aebabae3a41b5, -, 70),
        HEX_DBL(+, 1, 24b6031b49bda, -, 68),
        HEX_DBL(+, 1, 8dd5e1bb09d7e, -, 67),
        HEX_DBL(+, 1, 0e5b73d1ff53d, -, 65),
        HEX_DBL(+, 1, 6f741de1748ec, -, 64),
        HEX_DBL(+, 1, f36bd37f42f3e, -, 63),
        HEX_DBL(+, 1, 536452ee2f75c, -, 61),
        HEX_DBL(+, 1, cd480a1b7482, -, 60),
        HEX_DBL(+, 1, 39792499b1a24, -, 58),
        HEX_DBL(+, 1, aa0de4bf35b38, -, 57),
        HEX_DBL(+, 1, 2188ad6ae3303, -, 55),
        HEX_DBL(+, 1, 898471fca6055, -, 54),
        HEX_DBL(+, 1, 0b6c3afdde064, -, 52),
        HEX_DBL(+, 1, 6b7719a59f0e, -, 51),
        HEX_DBL(+, 1, ee001eed62aa, -, 50),
        HEX_DBL(+, 1, 4fb547c775da8, -, 48),
        HEX_DBL(+, 1, c8464f7616468, -, 47),
        HEX_DBL(+, 1, 36121e24d3bba, -, 45),
        HEX_DBL(+, 1, a56e0c2ac7f75, -, 44),
        HEX_DBL(+, 1, 1e642baeb84a, -, 42),
        HEX_DBL(+, 1, 853f01d6d53ba, -, 41),
        HEX_DBL(+, 1, 0885298767e9a, -, 39),
        HEX_DBL(+, 1, 67852a7007e42, -, 38),
        HEX_DBL(+, 1, e8a37a45fc32e, -, 37),
        HEX_DBL(+, 1, 4c1078fe9228a, -, 35),
        HEX_DBL(+, 1, c3527e433fab1, -, 34),
        HEX_DBL(+, 1, 32b48bf117da2, -, 32),
        HEX_DBL(+, 1, a0db0d0ddb3ec, -, 31),
        HEX_DBL(+, 1, 1b48655f37267, -, 29),
        HEX_DBL(+, 1, 81056ff2c5772, -, 28),
        HEX_DBL(+, 1, 05a628c699fa1, -, 26),
        HEX_DBL(+, 1, 639e3175a689d, -, 25),
        HEX_DBL(+, 1, e355bbaee85cb, -, 24),
        HEX_DBL(+, 1, 4875ca227ec38, -, 22),
        HEX_DBL(+, 1, be6c6fdb01612, -, 21),
        HEX_DBL(+, 1, 2f6053b981d98, -, 19),
        HEX_DBL(+, 1, 9c54c3b43bc8b, -, 18),
        HEX_DBL(+, 1, 18354238f6764, -, 16),
        HEX_DBL(+, 1, 7cd79b5647c9b, -, 15),
        HEX_DBL(+, 1, 02cf22526545a, -, 13),
        HEX_DBL(+, 1, 5fc21041027ad, -, 12),
        HEX_DBL(+, 1, de16b9c24a98f, -, 11),
        HEX_DBL(+, 1, 44e51f113d4d6, -, 9),
        HEX_DBL(+, 1, b993fe00d5376, -, 8),
        HEX_DBL(+, 1, 2c155b8213cf4, -, 6),
        HEX_DBL(+, 1, 97db0ccceb0af, -, 5),
        HEX_DBL(+, 1, 152aaa3bf81cc, -, 3),
        HEX_DBL(+, 1, 78b56362cef38, -, 2),
        HEX_DBL(+, 1, 0, +, 0),
        HEX_DBL(+, 1, 5bf0a8b145769, +, 1),
        HEX_DBL(+, 1, d8e64b8d4ddae, +, 2),
        HEX_DBL(+, 1, 415e5bf6fb106, +, 4),
        HEX_DBL(+, 1, b4c902e273a58, +, 5),
        HEX_DBL(+, 1, 28d389970338f, +, 7),
        HEX_DBL(+, 1, 936dc5690c08f, +, 8),
        HEX_DBL(+, 1, 122885aaeddaa, +, 10),
        HEX_DBL(+, 1, 749ea7d470c6e, +, 11),
        HEX_DBL(+, 1, fa7157c470f82, +, 12),
        HEX_DBL(+, 1, 5829dcf95056, +, 14),
        HEX_DBL(+, 1, d3c4488ee4f7f, +, 15),
        HEX_DBL(+, 1, 3de1654d37c9a, +, 17),
        HEX_DBL(+, 1, b00b5916ac955, +, 18),
        HEX_DBL(+, 1, 259ac48bf05d7, +, 20),
        HEX_DBL(+, 1, 8f0ccafad2a87, +, 21),
        HEX_DBL(+, 1, 0f2ebd0a8002, +, 23),
        HEX_DBL(+, 1, 709348c0ea4f9, +, 24),
        HEX_DBL(+, 1, f4f22091940bd, +, 25),
        HEX_DBL(+, 1, 546d8f9ed26e1, +, 27),
        HEX_DBL(+, 1, ceb088b68e804, +, 28),
        HEX_DBL(+, 1, 3a6e1fd9eecfd, +, 30),
        HEX_DBL(+, 1, ab5adb9c436, +, 31),
        HEX_DBL(+, 1, 226af33b1fdc1, +, 33),
        HEX_DBL(+, 1, 8ab7fb5475fb7, +, 34),
        HEX_DBL(+, 1, 0c3d3920962c9, +, 36),
        HEX_DBL(+, 1, 6c932696a6b5d, +, 37),
        HEX_DBL(+, 1, ef822f7f6731d, +, 38),
        HEX_DBL(+, 1, 50bba3796379a, +, 40),
        HEX_DBL(+, 1, c9aae4631c056, +, 41),
        HEX_DBL(+, 1, 370470aec28ed, +, 43),
        HEX_DBL(+, 1, a6b765d8cdf6d, +, 44),
        HEX_DBL(+, 1, 1f43fcc4b662c, +, 46),
        HEX_DBL(+, 1, 866f34a725782, +, 47),
        HEX_DBL(+, 1, 0953e2f3a1ef7, +, 49),
        HEX_DBL(+, 1, 689e221bc8d5b, +, 50),
        HEX_DBL(+, 1, ea215a1d20d76, +, 51),
        HEX_DBL(+, 1, 4d13fbb1a001a, +, 53),
        HEX_DBL(+, 1, c4b334617cc67, +, 54),
        HEX_DBL(+, 1, 33a43d282a519, +, 56),
        HEX_DBL(+, 1, a220d397972eb, +, 57),
        HEX_DBL(+, 1, 1c25c88df6862, +, 59),
        HEX_DBL(+, 1, 8232558201159, +, 60),
        HEX_DBL(+, 1, 0672a3c9eb871, +, 62),
        HEX_DBL(+, 1, 64b41c6d37832, +, 63),
        HEX_DBL(+, 1, e4cf766fe49be, +, 64),
        HEX_DBL(+, 1, 49767bc0483e3, +, 66),
        HEX_DBL(+, 1, bfc951eb8bb76, +, 67),
        HEX_DBL(+, 1, 304d6aeca254b, +, 69),
        HEX_DBL(+, 1, 9d97010884251, +, 70),
        HEX_DBL(+, 1, 19103e4080b45, +, 72),
        HEX_DBL(+, 1, 7e013cd114461, +, 73),
        HEX_DBL(+, 1, 03996528e074c, +, 75),
        HEX_DBL(+, 1, 60d4f6fdac731, +, 76),
        HEX_DBL(+, 1, df8c5af17ba3b, +, 77),
        HEX_DBL(+, 1, 45e3076d61699, +, 79),
        HEX_DBL(+, 1, baed16a6e0da7, +, 80),
        HEX_DBL(+, 1, 2cffdfebde1a1, +, 82),
        HEX_DBL(+, 1, 9919cabefcb69, +, 83),
        HEX_DBL(+, 1, 160345c9953e3, +, 85),
        HEX_DBL(+, 1, 79dbc9dc53c66, +, 86),
        HEX_DBL(+, 1, 00c810d464097, +, 88),
        HEX_DBL(+, 1, 5d009394c5c27, +, 89),
        HEX_DBL(+, 1, da57de8f107a8, +, 90),
        HEX_DBL(+, 1, 425982cf597cd, +, 92),
        HEX_DBL(+, 1, b61e5ca3a5e31, +, 93),
        HEX_DBL(+, 1, 29bb825dfcf87, +, 95),
        HEX_DBL(+, 1, 94a90db0d6fe2, +, 96),
        HEX_DBL(+, 1, 12fec759586fd, +, 98),
        HEX_DBL(+, 1, 75c1dc469e3af, +, 99),
        HEX_DBL(+, 1, fbfd219c43b04, +, 100),
        HEX_DBL(+, 1, 5936d44e1a146, +, 102),
        HEX_DBL(+, 1, d531d8a7ee79c, +, 103),
        HEX_DBL(+, 1, 3ed9d24a2d51b, +, 105),
        HEX_DBL(+, 1, b15cfe5b6e17b, +, 106),
        HEX_DBL(+, 1, 268038c2c0e, +, 108),
        HEX_DBL(+, 1, 9044a73545d48, +, 109),
        HEX_DBL(+, 1, 1002ab6218b38, +, 111),
        HEX_DBL(+, 1, 71b3540cbf921, +, 112),
        HEX_DBL(+, 1, f6799ea9c414a, +, 113),
        HEX_DBL(+, 1, 55779b984f3eb, +, 115),
        HEX_DBL(+, 1, d01a210c44aa4, +, 116),
        HEX_DBL(+, 1, 3b63da8e9121, +, 118),
        HEX_DBL(+, 1, aca8d6b0116b8, +, 119),
        HEX_DBL(+, 1, 234de9e0c74e9, +, 121),
        HEX_DBL(+, 1, 8bec7503ca477, +, 122),
        HEX_DBL(+, 1, 0d0eda9796b9, +, 124),
        HEX_DBL(+, 1, 6db0118477245, +, 125),
        HEX_DBL(+, 1, f1056dc7bf22d, +, 126),
        HEX_DBL(+, 1, 51c2cc3433801, +, 128),
        HEX_DBL(+, 1, cb108ffbec164, +, 129),
        HEX_DBL(+, 1, 37f780991b584, +, 131),
        HEX_DBL(+, 1, a801c0ea8ac4d, +, 132),
        HEX_DBL(+, 1, 20247cc4c46c1, +, 134),
        HEX_DBL(+, 1, 87a0553328015, +, 135),
        HEX_DBL(+, 1, 0a233dee4f9bb, +, 137),
        HEX_DBL(+, 1, 69b7f55b808ba, +, 138),
        HEX_DBL(+, 1, eba064644060a, +, 139),
        HEX_DBL(+, 1, 4e184933d9364, +, 141),
        HEX_DBL(+, 1, c614fe2531841, +, 142),
        HEX_DBL(+, 1, 3494a9b171bf5, +, 144),
        HEX_DBL(+, 1, a36798b9d969b, +, 145),
        HEX_DBL(+, 1, 1d03d8c0c04af, +, 147),
        HEX_DBL(+, 1, 836026385c974, +, 148),
        HEX_DBL(+, 1, 073fbe9ac901d, +, 150),
        HEX_DBL(+, 1, 65cae0969f286, +, 151),
        HEX_DBL(+, 1, e64a58639cae8, +, 152),
        HEX_DBL(+, 1, 4a77f5f9b50f9, +, 154),
        HEX_DBL(+, 1, c12744a3a28e3, +, 155),
        HEX_DBL(+, 1, 313b3b6978e85, +, 157),
        HEX_DBL(+, 1, 9eda3a31e587e, +, 158),
        HEX_DBL(+, 1, 19ebe56b56453, +, 160),
        HEX_DBL(+, 1, 7f2bc6e599b7e, +, 161),
        HEX_DBL(+, 1, 04644610df2ff, +, 163),
        HEX_DBL(+, 1, 61e8b490ac4e6, +, 164),
        HEX_DBL(+, 1, e103201f299b3, +, 165),
        HEX_DBL(+, 1, 46e1b637beaf5, +, 167),
        HEX_DBL(+, 1, bc473cfede104, +, 168),
        HEX_DBL(+, 1, 2deb1b9c85e2d, +, 170),
        HEX_DBL(+, 1, 9a5981ca67d1, +, 171),
        HEX_DBL(+, 1, 16dc8a9ef670b, +, 173),
        HEX_DBL(+, 1, 7b03166942309, +, 174),
        HEX_DBL(+, 1, 0190be03150a7, +, 176),
        HEX_DBL(+, 1, 5e1152f9a8119, +, 177),
        HEX_DBL(+, 1, dbca9263f8487, +, 178),
        HEX_DBL(+, 1, 43556dee93bee, +, 180),
        HEX_DBL(+, 1, b774c12967dfa, +, 181),
        HEX_DBL(+, 1, 2aa4306e922c2, +, 183),
        HEX_DBL(+, 1, 95e54c5dd4217, +, 184)
    };

    // scale by e**i --  (expm1(f) + 1)*e**i - 1  = expm1(f) * e**i + e**i - 1 =
    // e**i
    return exp_table[exponent + 150] + (f * exp_table[exponent + 150] - 1.0);
}


double reference_fmax(double x, double y)
{
    if (isnan(y)) return x;

    return x >= y ? x : y;
}

double reference_fmin(double x, double y)
{
    if (isnan(y)) return x;

    return x <= y ? x : y;
}

double reference_hypot(double x, double y)
{
    // Since the inputs are actually floats, we don't have to worry about range
    // here
    if (isinf(x) || isinf(y)) return INFINITY;

    return sqrt(x * x + y * y);
}

int reference_ilogbl(long double x)
{
    extern int gDeviceILogb0, gDeviceILogbNaN;

    // Since we are just using this to verify double precision, we can
    // use the double precision ilogb here
    union {
        double f;
        cl_ulong u;
    } u;
    u.f = (double)x;

    int exponent = (int)(u.u >> 52) & 0x7ff;
    if (exponent == 0x7ff)
    {
        if (u.u & 0x000fffffffffffffULL) return gDeviceILogbNaN;

        return CL_INT_MAX;
    }

    if (exponent == 0)
    { // deal with denormals
        u.f = x * HEX_DBL(+, 1, 0, +, 64);
        exponent = (cl_uint)(u.u >> 52) & 0x7ff;
        if (exponent == 0) return gDeviceILogb0;

        exponent -= 1023 + 64;
        return exponent;
    }

    return exponent - 1023;
}

double reference_relaxed_log2(double x) { return reference_log2(x); }

double reference_log2(double x)
{
    if (isnan(x) || x < 0.0 || x == -INFINITY) return cl_make_nan();

    if (x == 0.0f) return -INFINITY;

    if (x == INFINITY) return INFINITY;

    double hi, lo;
    __log2_ep(&hi, &lo, x);
    return hi;
}

double reference_log1p(double x)
{ // This function is suitable only for verifying log1pf(). It produces several
  // double precision ulps of error.

    // Handle small and NaN
    if (!(reference_fabs(x) > HEX_DBL(+, 1, 0, -, 53))) return x;

    // deal with special values
    if (x <= -1.0)
    {
        if (x < -1.0) return cl_make_nan();
        return -INFINITY;
    }

    // infinity
    if (x == INFINITY) return INFINITY;

    // High precision result for when near 0, to avoid problems with the
    // reference result falling in the wrong binade.
    if (reference_fabs(x) < HEX_DBL(+, 1, 0, -, 28)) return (1.0 - 0.5 * x) * x;

    // Our polynomial is only good in the region +-2**-4.
    // If we aren't in that range then we need to reduce to be in that range
    double correctionLo =
        -0.0; // correction down stream to compensate for the reduction, if any
    double correctionHi =
        -0.0; // correction down stream to compensate for the exponent, if any
    if (reference_fabs(x) > HEX_DBL(+, 1, 0, -, 4))
    {
        x += 1.0; // double should cover any loss of precision here

        // separate x into (1+f) * 2**i
        union {
            double d;
            cl_ulong u;
        } u;
        u.d = x;
        int i = (int)((u.u >> 52) & 0x7ff) - 1023;
        u.u &= 0x000fffffffffffffULL;
        int index = (int)(u.u >> 48);
        u.u |= 0x3ff0000000000000ULL;
        double f = u.d;

        // further reduce f to be within 1/16 of 1.0
        static const double scale_table[16] = {
            1.0,
            HEX_DBL(+, 1, d2d2d2d6e3f79, -, 1),
            HEX_DBL(+, 1, b8e38e42737a1, -, 1),
            HEX_DBL(+, 1, a1af28711adf3, -, 1),
            HEX_DBL(+, 1, 8cccccd88dd65, -, 1),
            HEX_DBL(+, 1, 79e79e810ec8f, -, 1),
            HEX_DBL(+, 1, 68ba2e94df404, -, 1),
            HEX_DBL(+, 1, 590b216defb29, -, 1),
            HEX_DBL(+, 1, 4aaaaab1500ed, -, 1),
            HEX_DBL(+, 1, 3d70a3e0d6f73, -, 1),
            HEX_DBL(+, 1, 313b13bb39f4f, -, 1),
            HEX_DBL(+, 1, 25ed09823f1cc, -, 1),
            HEX_DBL(+, 1, 1b6db6e77457b, -, 1),
            HEX_DBL(+, 1, 11a7b96a3a34f, -, 1),
            HEX_DBL(+, 1, 0888888e46fea, -, 1),
            HEX_DBL(+, 1, 00000038e9862, -, 1)
        };

        // correction_table[i] = -log( scale_table[i] )
        // All entries have >= 64 bits of precision (rather than the expected
        // 53)
        static const double correction_table[16] = {
            -0.0,
            HEX_DBL(+, 1, 7a5c722c16058, -, 4),
            HEX_DBL(+, 1, 323db16c89ab1, -, 3),
            HEX_DBL(+, 1, a0f87d180629, -, 3),
            HEX_DBL(+, 1, 050279324e17c, -, 2),
            HEX_DBL(+, 1, 36f885bb270b0, -, 2),
            HEX_DBL(+, 1, 669b771b5cc69, -, 2),
            HEX_DBL(+, 1, 94203a6292a05, -, 2),
            HEX_DBL(+, 1, bfb4f9cb333a4, -, 2),
            HEX_DBL(+, 1, e982376ddb80e, -, 2),
            HEX_DBL(+, 1, 08d5d8769b2b2, -, 1),
            HEX_DBL(+, 1, 1c288bc00e0cf, -, 1),
            HEX_DBL(+, 1, 2ec7535b31ecb, -, 1),
            HEX_DBL(+, 1, 40bed0adc63fb, -, 1),
            HEX_DBL(+, 1, 521a5c0330615, -, 1),
            HEX_DBL(+, 1, 62e42f7dd092c, -, 1)
        };

        f *= scale_table[index];
        correctionLo = correction_table[index];

        // log( 2**(i) ) = i * log(2)
        correctionHi = (double)i * 0.693147180559945309417232121458176568;

        x = f - 1.0;
    }


    // minmax polynomial for p(x) = (log(x+1) - x)/x valid over the range x =
    // [-1/16, 1/16]
    //          max error HEX_DBL( +, 1, 048f61f9a5eca, -, 52 )
    double p = HEX_DBL(-, 1, cc33de97a9d7b, -, 46)
        + (HEX_DBL(-, 1, fffffffff3eb7, -, 2)
           + (HEX_DBL(+, 1, 5555555633ef7, -, 2)
              + (HEX_DBL(-, 1, 00000062c78, -, 2)
                 + (HEX_DBL(+, 1, 9999958a3321, -, 3)
                    + (HEX_DBL(-, 1, 55534ce65c347, -, 3)
                       + (HEX_DBL(+, 1, 24957208391a5, -, 3)
                          + (HEX_DBL(-, 1, 02287b9a5b4a1, -, 3)
                             + HEX_DBL(+, 1, c757d922180ed, -, 4) * x)
                              * x)
                           * x)
                        * x)
                     * x)
                  * x)
               * x)
            * x;

    // log(x+1) = x * p(x) + x
    x += x * p;

    return correctionHi + (correctionLo + x);
}

double reference_logb(double x)
{
    union {
        float f;
        cl_uint u;
    } u;
    u.f = (float)x;

    cl_int exponent = (u.u >> 23) & 0xff;
    if (exponent == 0xff) return x * x;

    if (exponent == 0)
    { // deal with denormals
        u.u = (u.u & 0x007fffff) | 0x3f800000;
        u.f -= 1.0f;
        exponent = (u.u >> 23) & 0xff;
        if (exponent == 0) return -INFINITY;

        return exponent - (127 + 126);
    }

    return exponent - 127;
}

double reference_relaxed_reciprocal(double x) { return 1.0f / ((float)x); }

double reference_reciprocal(double x) { return 1.0 / x; }

double reference_remainder(double x, double y)
{
    int i;
    return reference_remquo(x, y, &i);
}

double reference_lgamma(double x)
{
    /*
     * ====================================================
     * This function is from fdlibm. http://www.netlib.org
     * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     *
     */

    static const double // two52 = 4.50359962737049600000e+15, /* 0x43300000,
                        // 0x00000000 */
        half = 5.00000000000000000000e-01, /* 0x3FE00000,
                                              0x00000000 */
        one = 1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
        pi = 3.14159265358979311600e+00, /* 0x400921FB, 0x54442D18 */
        a0 = 7.72156649015328655494e-02, /* 0x3FB3C467, 0xE37DB0C8 */
        a1 = 3.22467033424113591611e-01, /* 0x3FD4A34C, 0xC4A60FAD */
        a2 = 6.73523010531292681824e-02, /* 0x3FB13E00, 0x1A5562A7 */
        a3 = 2.05808084325167332806e-02, /* 0x3F951322, 0xAC92547B */
        a4 = 7.38555086081402883957e-03, /* 0x3F7E404F, 0xB68FEFE8 */
        a5 = 2.89051383673415629091e-03, /* 0x3F67ADD8, 0xCCB7926B */
        a6 = 1.19270763183362067845e-03, /* 0x3F538A94, 0x116F3F5D */
        a7 = 5.10069792153511336608e-04, /* 0x3F40B6C6, 0x89B99C00 */
        a8 = 2.20862790713908385557e-04, /* 0x3F2CF2EC, 0xED10E54D */
        a9 = 1.08011567247583939954e-04, /* 0x3F1C5088, 0x987DFB07 */
        a10 = 2.52144565451257326939e-05, /* 0x3EFA7074, 0x428CFA52 */
        a11 = 4.48640949618915160150e-05, /* 0x3F07858E, 0x90A45837 */
        tc = 1.46163214496836224576e+00, /* 0x3FF762D8, 0x6356BE3F */
        tf = -1.21486290535849611461e-01, /* 0xBFBF19B9, 0xBCC38A42 */
        /* tt = -(tail of tf) */
        tt = -3.63867699703950536541e-18, /* 0xBC50C7CA, 0xA48A971F */
        t0 = 4.83836122723810047042e-01, /* 0x3FDEF72B, 0xC8EE38A2 */
        t1 = -1.47587722994593911752e-01, /* 0xBFC2E427, 0x8DC6C509 */
        t2 = 6.46249402391333854778e-02, /* 0x3FB08B42, 0x94D5419B */
        t3 = -3.27885410759859649565e-02, /* 0xBFA0C9A8, 0xDF35B713 */
        t4 = 1.79706750811820387126e-02, /* 0x3F9266E7, 0x970AF9EC */
        t5 = -1.03142241298341437450e-02, /* 0xBF851F9F, 0xBA91EC6A */
        t6 = 6.10053870246291332635e-03, /* 0x3F78FCE0, 0xE370E344 */
        t7 = -3.68452016781138256760e-03, /* 0xBF6E2EFF, 0xB3E914D7 */
        t8 = 2.25964780900612472250e-03, /* 0x3F6282D3, 0x2E15C915 */
        t9 = -1.40346469989232843813e-03, /* 0xBF56FE8E, 0xBF2D1AF1 */
        t10 = 8.81081882437654011382e-04, /* 0x3F4CDF0C, 0xEF61A8E9 */
        t11 = -5.38595305356740546715e-04, /* 0xBF41A610, 0x9C73E0EC */
        t12 = 3.15632070903625950361e-04, /* 0x3F34AF6D, 0x6C0EBBF7 */
        t13 = -3.12754168375120860518e-04, /* 0xBF347F24, 0xECC38C38 */
        t14 = 3.35529192635519073543e-04, /* 0x3F35FD3E, 0xE8C2D3F4 */
        u0 = -7.72156649015328655494e-02, /* 0xBFB3C467, 0xE37DB0C8 */
        u1 = 6.32827064025093366517e-01, /* 0x3FE4401E, 0x8B005DFF */
        u2 = 1.45492250137234768737e+00, /* 0x3FF7475C, 0xD119BD6F */
        u3 = 9.77717527963372745603e-01, /* 0x3FEF4976, 0x44EA8450 */
        u4 = 2.28963728064692451092e-01, /* 0x3FCD4EAE, 0xF6010924 */
        u5 = 1.33810918536787660377e-02, /* 0x3F8B678B, 0xBF2BAB09 */
        v1 = 2.45597793713041134822e+00, /* 0x4003A5D7, 0xC2BD619C */
        v2 = 2.12848976379893395361e+00, /* 0x40010725, 0xA42B18F5 */
        v3 = 7.69285150456672783825e-01, /* 0x3FE89DFB, 0xE45050AF */
        v4 = 1.04222645593369134254e-01, /* 0x3FBAAE55, 0xD6537C88 */
        v5 = 3.21709242282423911810e-03, /* 0x3F6A5ABB, 0x57D0CF61 */
        s0 = -7.72156649015328655494e-02, /* 0xBFB3C467, 0xE37DB0C8 */
        s1 = 2.14982415960608852501e-01, /* 0x3FCB848B, 0x36E20878 */
        s2 = 3.25778796408930981787e-01, /* 0x3FD4D98F, 0x4F139F59 */
        s3 = 1.46350472652464452805e-01, /* 0x3FC2BB9C, 0xBEE5F2F7 */
        s4 = 2.66422703033638609560e-02, /* 0x3F9B481C, 0x7E939961 */
        s5 = 1.84028451407337715652e-03, /* 0x3F5E26B6, 0x7368F239 */
        s6 = 3.19475326584100867617e-05, /* 0x3F00BFEC, 0xDD17E945 */
        r1 = 1.39200533467621045958e+00, /* 0x3FF645A7, 0x62C4AB74 */
        r2 = 7.21935547567138069525e-01, /* 0x3FE71A18, 0x93D3DCDC */
        r3 = 1.71933865632803078993e-01, /* 0x3FC601ED, 0xCCFBDF27 */
        r4 = 1.86459191715652901344e-02, /* 0x3F9317EA, 0x742ED475 */
        r5 = 7.77942496381893596434e-04, /* 0x3F497DDA, 0xCA41A95B */
        r6 = 7.32668430744625636189e-06, /* 0x3EDEBAF7, 0xA5B38140 */
        w0 = 4.18938533204672725052e-01, /* 0x3FDACFE3, 0x90C97D69 */
        w1 = 8.33333333333329678849e-02, /* 0x3FB55555, 0x5555553B */
        w2 = -2.77777777728775536470e-03, /* 0xBF66C16C, 0x16B02E5C */
        w3 = 7.93650558643019558500e-04, /* 0x3F4A019F, 0x98CF38B6 */
        w4 = -5.95187557450339963135e-04, /* 0xBF4380CB, 0x8C0FE741 */
        w5 = 8.36339918996282139126e-04, /* 0x3F4B67BA, 0x4CDAD5D1 */
        w6 = -1.63092934096575273989e-03; /* 0xBF5AB89D, 0x0B9E43E4 */

    static const double zero = 0.00000000000000000000e+00;
    double t, y, z, nadj, p, p1, p2, p3, q, r, w;
    cl_int i, hx, lx, ix;

    union {
        double d;
        cl_ulong u;
    } u;
    u.d = x;

    hx = (cl_int)(u.u >> 32);
    lx = (cl_int)(u.u & 0xffffffffULL);

    /* purge off +-inf, NaN, +-0, and negative arguments */
    //    *signgamp = 1;
    ix = hx & 0x7fffffff;
    if (ix >= 0x7ff00000) return x * x;
    if ((ix | lx) == 0) return INFINITY;
    if (ix < 0x3b900000)
    { /* |x|<2**-70, return -log(|x|) */
        if (hx < 0)
        {
            //            *signgamp = -1;
            return -reference_log(-x);
        }
        else
            return -reference_log(x);
    }
    if (hx < 0)
    {
        if (ix >= 0x43300000) /* |x|>=2**52, must be -integer */
            return INFINITY;
        t = reference_sinpi(x);
        if (t == zero) return INFINITY; /* -integer */
        nadj = reference_log(pi / reference_fabs(t * x));
        //        if(t<zero) *signgamp = -1;
        x = -x;
    }

    /* purge off 1 and 2 */
    if ((((ix - 0x3ff00000) | lx) == 0) || (((ix - 0x40000000) | lx) == 0))
        r = 0;
    /* for x < 2.0 */
    else if (ix < 0x40000000)
    {
        if (ix <= 0x3feccccc)
        { /* lgamma(x) = lgamma(x+1)-log(x) */
            r = -reference_log(x);
            if (ix >= 0x3FE76944)
            {
                y = 1.0 - x;
                i = 0;
            }
            else if (ix >= 0x3FCDA661)
            {
                y = x - (tc - one);
                i = 1;
            }
            else
            {
                y = x;
                i = 2;
            }
        }
        else
        {
            r = zero;
            if (ix >= 0x3FFBB4C3)
            {
                y = 2.0 - x;
                i = 0;
            } /* [1.7316,2] */
            else if (ix >= 0x3FF3B4C4)
            {
                y = x - tc;
                i = 1;
            } /* [1.23,1.73] */
            else
            {
                y = x - one;
                i = 2;
            }
        }
        switch (i)
        {
            case 0:
                z = y * y;
                p1 = a0 + z * (a2 + z * (a4 + z * (a6 + z * (a8 + z * a10))));
                p2 = z
                    * (a1
                       + z * (a3 + z * (a5 + z * (a7 + z * (a9 + z * a11)))));
                p = y * p1 + p2;
                r += (p - 0.5 * y);
                break;
            case 1:
                z = y * y;
                w = z * y;
                p1 = t0
                    + w
                        * (t3
                           + w * (t6 + w * (t9 + w * t12))); /* parallel comp */
                p2 = t1 + w * (t4 + w * (t7 + w * (t10 + w * t13)));
                p3 = t2 + w * (t5 + w * (t8 + w * (t11 + w * t14)));
                p = z * p1 - (tt - w * (p2 + y * p3));
                r += (tf + p);
                break;
            case 2:
                p1 = y
                    * (u0 + y * (u1 + y * (u2 + y * (u3 + y * (u4 + y * u5)))));
                p2 = one + y * (v1 + y * (v2 + y * (v3 + y * (v4 + y * v5))));
                r += (-0.5 * y + p1 / p2);
        }
    }
    else if (ix < 0x40200000)
    { /* x < 8.0 */
        i = (int)x;
        t = zero;
        y = x - (double)i;
        p = y
            * (s0
               + y * (s1 + y * (s2 + y * (s3 + y * (s4 + y * (s5 + y * s6))))));
        q = one + y * (r1 + y * (r2 + y * (r3 + y * (r4 + y * (r5 + y * r6)))));
        r = half * y + p / q;
        z = one; /* lgamma(1+s) = log(s) + lgamma(s) */
        switch (i)
        {
            case 7: z *= (y + 6.0); /* FALLTHRU */
            case 6: z *= (y + 5.0); /* FALLTHRU */
            case 5: z *= (y + 4.0); /* FALLTHRU */
            case 4: z *= (y + 3.0); /* FALLTHRU */
            case 3:
                z *= (y + 2.0); /* FALLTHRU */
                r += reference_log(z);
                break;
        }
        /* 8.0 <= x < 2**58 */
    }
    else if (ix < 0x43900000)
    {
        t = reference_log(x);
        z = one / x;
        y = z * z;
        w = w0 + z * (w1 + y * (w2 + y * (w3 + y * (w4 + y * (w5 + y * w6)))));
        r = (x - half) * (t - one) + w;
    }
    else
        /* 2**58 <= x <= inf */
        r = x * (reference_log(x) - one);
    if (hx < 0) r = nadj - r;
    return r;
}

#endif // _MSC_VER

double reference_assignment(double x) { return x; }

int reference_not(double x)
{
    int r = !x;
    return r;
}

#pragma mark -
#pragma mark Double testing

#ifndef M_PIL
#define M_PIL                                                                  \
    3.14159265358979323846264338327950288419716939937510582097494459230781640628620899L
#endif

static long double reduce1l(long double x);

#ifdef __PPC__
// Since long double on PPC is really extended precision double arithmetic
// consisting of two doubles (a high and low). This form of long double has
// the potential of representing a number with more than LDBL_MANT_DIG digits
// such that reduction algorithm used for other architectures will not work.
// Instead and alternate reduction method is used.

static long double reduce1l(long double x)
{
    union {
        long double ld;
        double d[2];
    } u;

    // Reduce the high and low halfs separately.
    u.ld = x;
    return ((long double)reduce1(u.d[0]) + reduce1(u.d[1]));
}

#else // !__PPC__

static long double reduce1l(long double x)
{
    static long double unit_exp = 0;
    if (0.0L == unit_exp) unit_exp = scalbnl(1.0L, LDBL_MANT_DIG);

    if (reference_fabsl(x) >= unit_exp)
    {
        if (reference_fabsl(x) == INFINITY) return cl_make_nan();

        return 0.0L; // we patch up the sign for sinPi and cosPi later, since
                     // they need different signs
    }

    // Find the nearest multiple of 2
    const long double r = reference_copysignl(unit_exp, x);
    long double z = x + r;
    z -= r;

    // subtract it from x. Value is now in the range -1 <= x <= 1
    return x - z;
}
#endif // __PPC__

long double reference_acospil(long double x)
{
    return reference_acosl(x) / M_PIL;
}
long double reference_asinpil(long double x)
{
    return reference_asinl(x) / M_PIL;
}
long double reference_atanpil(long double x)
{
    return reference_atanl(x) / M_PIL;
}
long double reference_atan2pil(long double y, long double x)
{
    return reference_atan2l(y, x) / M_PIL;
}
long double reference_cospil(long double x)
{
    if (reference_fabsl(x) >= HEX_LDBL(+, 1, 0, +, 54))
    {
        if (reference_fabsl(x) == INFINITY) return cl_make_nan();

        // Note this probably fails for odd values between 0x1.0p52 and
        // 0x1.0p53. However, when starting with single precision inputs, there
        // will be no odd values.

        return 1.0L;
    }

    x = reduce1l(x);

#if DBL_MANT_DIG >= LDBL_MANT_DIG

    // phase adjust
    double xhi = 0.0;
    double xlo = 0.0;
    xhi = (double)x + 0.5;

    if (reference_fabsl(x) > 0.5L)
    {
        xlo = xhi - x;
        xlo = 0.5 - xlo;
    }
    else
    {
        xlo = xhi - 0.5;
        xlo = x - xlo;
    }

    // reduce to [-0.5, 0.5]
    if (xhi < -0.5)
    {
        xhi = -1.0 - xhi;
        xlo = -xlo;
    }
    else if (xhi > 0.5)
    {
        xhi = 1.0 - xhi;
        xlo = -xlo;
    }

    // cosPi zeros are all +0
    if (xhi == 0.0 && xlo == 0.0) return 0.0;

    xhi *= M_PI;
    xlo *= M_PI;

    xhi += xlo;

    return reference_sinl(xhi);

#else
    // phase adjust
    x += 0.5L;

    // reduce to [-0.5, 0.5]
    if (x < -0.5L)
        x = -1.0L - x;
    else if (x > 0.5L)
        x = 1.0L - x;

    // cosPi zeros are all +0
    if (x == 0.0L) return 0.0L;

    return reference_sinl(x * M_PIL);
#endif
}

long double reference_dividel(long double x, long double y)
{
    double dx = x;
    double dy = y;
    return dx / dy;
}

struct double_double
{
    double hi, lo;
};

// Split doubles_double into a series of consecutive 26-bit precise doubles and
// a remainder. Note for later -- for multiplication, it might be better to
// split each double into a power of two and two 26 bit portions
//                      multiplication of a double double by a known power of
//                      two is cheap. The current approach causes some inexact
//                      arithmetic in mul_dd.
static inline void split_dd(double_double x, double_double *hi,
                            double_double *lo)
{
    union {
        double d;
        cl_ulong u;
    } u;
    u.d = x.hi;
    u.u &= 0xFFFFFFFFF8000000ULL;
    hi->hi = u.d;
    x.hi -= u.d;

    u.d = x.hi;
    u.u &= 0xFFFFFFFFF8000000ULL;
    hi->lo = u.d;
    x.hi -= u.d;

    double temp = x.hi;
    x.hi += x.lo;
    x.lo -= x.hi - temp;
    u.d = x.hi;
    u.u &= 0xFFFFFFFFF8000000ULL;
    lo->hi = u.d;
    x.hi -= u.d;

    lo->lo = x.hi + x.lo;
}

static inline double_double accum_d(double_double a, double b)
{
    double temp;
    if (fabs(b) > fabs(a.hi))
    {
        temp = a.hi;
        a.hi += b;
        a.lo += temp - (a.hi - b);
    }
    else
    {
        temp = a.hi;
        a.hi += b;
        a.lo += b - (a.hi - temp);
    }

    if (isnan(a.lo)) a.lo = 0.0;

    return a;
}

static inline double_double add_dd(double_double a, double_double b)
{
    double_double r = { -0.0 - 0.0 };

    if (isinf(a.hi) || isinf(b.hi) || isnan(a.hi) || isnan(b.hi) || 0.0 == a.hi
        || 0.0 == b.hi)
    {
        r.hi = a.hi + b.hi;
        r.lo = a.lo + b.lo;
        if (isnan(r.lo)) r.lo = 0.0;
        return r;
    }

    // merge sort terms by magnitude -- here we assume that |a.hi| > |a.lo|,
    // |b.hi| > |b.lo|, so we don't have to do the first merge pass
    double terms[4] = { a.hi, b.hi, a.lo, b.lo };
    double temp;

    // Sort hi terms
    if (fabs(terms[0]) < fabs(terms[1]))
    {
        temp = terms[0];
        terms[0] = terms[1];
        terms[1] = temp;
    }
    // sort lo terms
    if (fabs(terms[2]) < fabs(terms[3]))
    {
        temp = terms[2];
        terms[2] = terms[3];
        terms[3] = temp;
    }
    // Fix case where small high term is less than large low term
    if (fabs(terms[1]) < fabs(terms[2]))
    {
        temp = terms[1];
        terms[1] = terms[2];
        terms[2] = temp;
    }

    // accumulate the results
    r.hi = terms[2] + terms[3];
    r.lo = terms[3] - (r.hi - terms[2]);

    temp = r.hi;
    r.hi += terms[1];
    r.lo += temp - (r.hi - terms[1]);

    temp = r.hi;
    r.hi += terms[0];
    r.lo += temp - (r.hi - terms[0]);

    // canonicalize the result
    temp = r.hi;
    r.hi += r.lo;
    r.lo = r.lo - (r.hi - temp);
    if (isnan(r.lo)) r.lo = 0.0;

    return r;
}

static inline double_double mul_dd(double_double a, double_double b)
{
    double_double result = { -0.0, -0.0 };

    // Inf, nan and 0
    if (isnan(a.hi) || isnan(b.hi) || isinf(a.hi) || isinf(b.hi) || 0.0 == a.hi
        || 0.0 == b.hi)
    {
        result.hi = a.hi * b.hi;
        return result;
    }

    double_double ah, al, bh, bl;
    split_dd(a, &ah, &al);
    split_dd(b, &bh, &bl);

    double p0 = ah.hi * bh.hi; // exact    (52 bits in product) 0
    double p1 = ah.hi * bh.lo; // exact    (52 bits in product) 26
    double p2 = ah.lo * bh.hi; // exact    (52 bits in product) 26
    double p3 = ah.lo * bh.lo; // exact    (52 bits in product) 52
    double p4 = al.hi * bh.hi; // exact    (52 bits in product) 52
    double p5 = al.hi * bh.lo; // exact    (52 bits in product) 78
    double p6 = al.lo * bh.hi; // inexact  (54 bits in product) 78
    double p7 = al.lo * bh.lo; // inexact  (54 bits in product) 104
    double p8 = ah.hi * bl.hi; // exact    (52 bits in product) 52
    double p9 = ah.hi * bl.lo; // inexact  (54 bits in product) 78
    double pA = ah.lo * bl.hi; // exact    (52 bits in product) 78
    double pB = ah.lo * bl.lo; // inexact  (54 bits in product) 104
    double pC = al.hi * bl.hi; // exact    (52 bits in product) 104
    // the last 3 terms are two low to appear in the result


    // take advantage of the known relative magnitudes of the partial products
    // to avoid some sorting Combine 2**-78 and 2**-104 terms. Here we are a bit
    // sloppy about canonicalizing the double_doubles
    double_double t0 = { pA, pC };
    double_double t1 = { p9, pB };
    double_double t2 = { p6, p7 };
    double temp0, temp1, temp2;

    t0 = accum_d(t0, p5); // there is an extra 2**-78 term to deal with

    // Add in 2**-52 terms. Here we are a bit sloppy about canonicalizing the
    // double_doubles
    temp0 = t0.hi;
    temp1 = t1.hi;
    temp2 = t2.hi;
    t0.hi += p3;
    t1.hi += p4;
    t2.hi += p8;
    temp0 -= t0.hi - p3;
    temp1 -= t1.hi - p4;
    temp2 -= t2.hi - p8;
    t0.lo += temp0;
    t1.lo += temp1;
    t2.lo += temp2;

    // Add in 2**-26 terms. Here we are a bit sloppy about canonicalizing the
    // double_doubles
    temp1 = t1.hi;
    temp2 = t2.hi;
    t1.hi += p1;
    t2.hi += p2;
    temp1 -= t1.hi - p1;
    temp2 -= t2.hi - p2;
    t1.lo += temp1;
    t2.lo += temp2;

    // Combine accumulators to get the low bits of result
    t1 = add_dd(t1, add_dd(t2, t0));

    // Add in MSB's, and round to precision
    return accum_d(t1, p0); // canonicalizes
}


long double reference_exp10l(long double z)
{
    const double_double log2_10 = { HEX_DBL(+, 1, a934f0979a371, +, 1),
                                    HEX_DBL(+, 1, 7f2495fb7fa6d, -, 53) };
    double_double x;
    int j;

    // Handle NaNs
    if (isnan(z)) return z;

    // init x
    x.hi = z;
    x.lo = z - x.hi;


    // 10**x = exp2( x * log2(10) )

    x = mul_dd(x, log2_10); // x * log2(10)

    // Deal with overflow and underflow for exp2(x) stage next
    if (x.hi >= 1025) return INFINITY;

    if (x.hi < -1075 - 24) return +0.0;

    // find nearest integer to x
    int i = (int)rint(x.hi);

    // x now holds fractional part.  The result would be then 2**i  * exp2( x )
    x.hi -= i;

    // We could attempt to find a minimax polynomial for exp2(x) over the range
    // x = [-0.5, 0.5]. However, this would converge very slowly near the
    // extrema, where 0.5**n is not a lot different from 0.5**(n+1), thereby
    // requiring something like a 20th order polynomial to get 53 + 24 bits of
    // precision. Instead we further reduce the range to [-1/32, 1/32] by
    // observing that
    //
    //  2**(a+b) = 2**a * 2**b
    //
    // We can thus build a table of 2**a values for a = n/16, n = [-8, 8], and
    // reduce the range of x to [-1/32, 1/32] by subtracting away the nearest
    // value of n/16 from x.
    const double_double corrections[17] = {
        { HEX_DBL(+, 1, 6a09e667f3bcd, -, 1),
          HEX_DBL(-, 1, bdd3413b26456, -, 55) },
        { HEX_DBL(+, 1, 7a11473eb0187, -, 1),
          HEX_DBL(-, 1, 41577ee04992f, -, 56) },
        { HEX_DBL(+, 1, 8ace5422aa0db, -, 1),
          HEX_DBL(+, 1, 6e9f156864b27, -, 55) },
        { HEX_DBL(+, 1, 9c49182a3f09, -, 1),
          HEX_DBL(+, 1, c7c46b071f2be, -, 57) },
        { HEX_DBL(+, 1, ae89f995ad3ad, -, 1),
          HEX_DBL(+, 1, 7a1cd345dcc81, -, 55) },
        { HEX_DBL(+, 1, c199bdd85529c, -, 1),
          HEX_DBL(+, 1, 11065895048dd, -, 56) },
        { HEX_DBL(+, 1, d5818dcfba487, -, 1),
          HEX_DBL(+, 1, 2ed02d75b3707, -, 56) },
        { HEX_DBL(+, 1, ea4afa2a490da, -, 1),
          HEX_DBL(-, 1, e9c23179c2893, -, 55) },
        { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
        { HEX_DBL(+, 1, 0b5586cf9890f, +, 0),
          HEX_DBL(+, 1, 8a62e4adc610b, -, 54) },
        { HEX_DBL(+, 1, 172b83c7d517b, +, 0),
          HEX_DBL(-, 1, 19041b9d78a76, -, 55) },
        { HEX_DBL(+, 1, 2387a6e756238, +, 0),
          HEX_DBL(+, 1, 9b07eb6c70573, -, 54) },
        { HEX_DBL(+, 1, 306fe0a31b715, +, 0),
          HEX_DBL(+, 1, 6f46ad23182e4, -, 55) },
        { HEX_DBL(+, 1, 3dea64c123422, +, 0),
          HEX_DBL(+, 1, ada0911f09ebc, -, 55) },
        { HEX_DBL(+, 1, 4bfdad5362a27, +, 0),
          HEX_DBL(+, 1, d4397afec42e2, -, 56) },
        { HEX_DBL(+, 1, 5ab07dd485429, +, 0),
          HEX_DBL(+, 1, 6324c054647ad, -, 54) },
        { HEX_DBL(+, 1, 6a09e667f3bcd, +, 0),
          HEX_DBL(-, 1, bdd3413b26456, -, 54) }

    };
    int index = (int)rint(x.hi * 16.0);
    x.hi -= (double)index * 0.0625;

    // canonicalize x
    double temp = x.hi;
    x.hi += x.lo;
    x.lo -= x.hi - temp;

    // Minimax polynomial for (exp2(x)-1)/x, over the range [-1/32, 1/32].  Max
    // Error: 2 * 0x1.e112p-87
    const double_double c[] = { { HEX_DBL(+, 1, 62e42fefa39ef, -, 1),
                                  HEX_DBL(+, 1, abc9e3ac1d244, -, 56) },
                                { HEX_DBL(+, 1, ebfbdff82c58f, -, 3),
                                  HEX_DBL(-, 1, 5e4987a631846, -, 57) },
                                { HEX_DBL(+, 1, c6b08d704a0c, -, 5),
                                  HEX_DBL(-, 1, d323200a05713, -, 59) },
                                { HEX_DBL(+, 1, 3b2ab6fba4e7a, -, 7),
                                  HEX_DBL(+, 1, c5ee8f8b9f0c1, -, 63) },
                                { HEX_DBL(+, 1, 5d87fe78a672a, -, 10),
                                  HEX_DBL(+, 1, 884e5e5cc7ecc, -, 64) },
                                { HEX_DBL(+, 1, 430912f7e8373, -, 13),
                                  HEX_DBL(+, 1, 4f1b59514a326, -, 67) },
                                { HEX_DBL(+, 1, ffcbfc5985e71, -, 17),
                                  HEX_DBL(-, 1, db7d6a0953b78, -, 71) },
                                { HEX_DBL(+, 1, 62c150eb16465, -, 20),
                                  HEX_DBL(+, 1, e0767c2d7abf5, -, 80) },
                                { HEX_DBL(+, 1, b52502b5e953, -, 24),
                                  HEX_DBL(+, 1, 6797523f944bc, -, 78) } };
    size_t count = sizeof(c) / sizeof(c[0]);

    // Do polynomial
    double_double r = c[count - 1];
    for (j = (int)count - 2; j >= 0; j--) r = add_dd(c[j], mul_dd(r, x));

    // unwind approximation
    r = mul_dd(r, x); // before: r =(exp2(x)-1)/x;   after: r = exp2(x) - 1

    // correct for [-0.5, 0.5] -> [-1/32, 1/32] reduction above
    //  exp2(x) = (r + 1) * correction = r * correction + correction
    r = mul_dd(r, corrections[index + 8]);
    r = add_dd(r, corrections[index + 8]);

    // Format result for output:

    // Get mantissa
    long double m = ((long double)r.hi + (long double)r.lo);

    // Handle a pesky overflow cases when long double = double
    if (i > 512)
    {
        m *= HEX_DBL(+, 1, 0, +, 512);
        i -= 512;
    }
    else if (i < -512)
    {
        m *= HEX_DBL(+, 1, 0, -, 512);
        i += 512;
    }

    return m * ldexpl(1.0L, i);
}


static double fallback_frexp(double x, int *iptr)
{
    cl_ulong u, v;
    double fu, fv;

    memcpy(&u, &x, sizeof(u));

    cl_ulong exponent = u & 0x7ff0000000000000ULL;
    cl_ulong mantissa = u & ~0x7ff0000000000000ULL;

    // add 1 to the exponent
    exponent += 0x0010000000000000ULL;

    if ((cl_long)exponent < (cl_long)0x0020000000000000LL)
    { // subnormal, NaN, Inf
        mantissa |= 0x3fe0000000000000ULL;

        v = mantissa & 0xfff0000000000000ULL;
        u = mantissa;
        memcpy(&fv, &v, sizeof(v));
        memcpy(&fu, &u, sizeof(u));

        fu -= fv;

        memcpy(&v, &fv, sizeof(v));
        memcpy(&u, &fu, sizeof(u));

        exponent = u & 0x7ff0000000000000ULL;
        mantissa = u & ~0x7ff0000000000000ULL;

        *iptr = (exponent >> 52) + (-1022 + 1 - 1022);
        u = mantissa | 0x3fe0000000000000ULL;
        memcpy(&fu, &u, sizeof(u));
        return fu;
    }

    *iptr = (exponent >> 52) - 1023;
    u = mantissa | 0x3fe0000000000000ULL;
    memcpy(&fu, &u, sizeof(u));
    return fu;
}

// Assumes zeros, infinities and NaNs handed elsewhere
static inline int extract(double x, cl_ulong *mant)
{
    static double (*frexpp)(double, int *) = NULL;
    int e;

    // verify that frexp works properly
    if (NULL == frexpp)
    {
        if (0.5 == frexp(HEX_DBL(+, 1, 0, -, 1030), &e) && e == -1029)
            frexpp = frexp;
        else
            frexpp = fallback_frexp;
    }

    *mant = (cl_ulong)(HEX_DBL(+, 1, 0, +, 64) * fabs(frexpp(x, &e)));
    return e - 1;
}

// Return 128-bit product of a*b  as (hi << 64) + lo
static inline void mul128(cl_ulong a, cl_ulong b, cl_ulong *hi, cl_ulong *lo)
{
    cl_ulong alo = a & 0xffffffffULL;
    cl_ulong ahi = a >> 32;
    cl_ulong blo = b & 0xffffffffULL;
    cl_ulong bhi = b >> 32;
    cl_ulong aloblo = alo * blo;
    cl_ulong alobhi = alo * bhi;
    cl_ulong ahiblo = ahi * blo;
    cl_ulong ahibhi = ahi * bhi;

    alobhi += (aloblo >> 32)
        + (ahiblo
           & 0xffffffffULL); // cannot overflow: (2^32-1)^2 + 2 * (2^32-1)   =
                             // (2^64 - 2^33 + 1) + (2^33 - 2) = 2^64 - 1
    *hi = ahibhi + (alobhi >> 32)
        + (ahiblo >> 32); // cannot overflow: (2^32-1)^2 + 2 * (2^32-1)   =
                          // (2^64 - 2^33 + 1) + (2^33 - 2) = 2^64 - 1
    *lo = (aloblo & 0xffffffffULL) | (alobhi << 32);
}

static double round_to_nearest_even_double(cl_ulong hi, cl_ulong lo,
                                           int exponent)
{
    union {
        cl_ulong u;
        cl_double d;
    } u;

    // edges
    if (exponent > 1023) return INFINITY;
    if (exponent == -1075 && (hi | (lo != 0)) > 0x8000000000000000ULL)
        return HEX_DBL(+, 1, 0, -, 1074);
    if (exponent <= -1075) return 0.0;

    // Figure out which bits go where
    int shift = 11;
    if (exponent < -1022)
    {
        shift -= 1022 + exponent; // subnormal: shift is not 52
        exponent = -1023; //              set exponent to 0
    }
    else
        hi &= 0x7fffffffffffffffULL; // normal: leading bit is implicit. Remove
                                     // it.

    // Assemble the double (round toward zero)
    u.u = (hi >> shift) | ((cl_ulong)(exponent + 1023) << 52);

    // put a representation of the residual bits into hi
    hi <<= (64 - shift);
    hi |= lo >> shift;
    lo <<= (64 - shift);
    hi |= lo != 0;

    // round to nearest, ties to even
    if (hi < 0x8000000000000000ULL) return u.d;
    if (hi == 0x8000000000000000ULL)
        u.u += u.u & 1ULL;
    else
        u.u++;

    return u.d;
}

// Shift right.  Bits lost on the right will be OR'd together and OR'd with the
// LSB
static inline void shift_right_sticky_128(cl_ulong *hi, cl_ulong *lo, int shift)
{
    cl_ulong sticky = 0;
    cl_ulong h = *hi;
    cl_ulong l = *lo;

    if (shift >= 64)
    {
        shift -= 64;
        sticky = 0 != lo;
        l = h;
        h = 0;
        if (shift >= 64)
        {
            sticky |= (0 != l);
            l = 0;
        }
        else
        {
            sticky |= (0 != (l << (64 - shift)));
            l >>= shift;
        }
    }
    else
    {
        sticky |= (0 != (l << (64 - shift)));
        l >>= shift;
        l |= h << (64 - shift);
        h >>= shift;
    }

    *lo = l | sticky;
    *hi = h;
}

// 128-bit add  of ((*hi << 64) + *lo) + ((chi << 64) + clo)
// If the 129 bit result doesn't fit, bits lost off the right end will be OR'd
// with the LSB
static inline void add128(cl_ulong *hi, cl_ulong *lo, cl_ulong chi,
                          cl_ulong clo, int *exponent)
{
    cl_ulong carry, carry2;
    // extended precision add
    clo = add_carry(*lo, clo, &carry);
    chi = add_carry(*hi, chi, &carry2);
    chi = add_carry(chi, carry, &carry);

    // If we overflowed the 128 bit result
    if (carry || carry2)
    {
        carry = clo & 1; // set aside low bit
        clo >>= 1; // right shift low 1
        clo |= carry; // or back in the low bit, so we don't come to believe
                      // this is an exact half way case for rounding
        clo |= chi << 63; // move lowest high bit into highest bit of lo
        chi >>= 1; // right shift hi
        chi |= 0x8000000000000000ULL; // move the carry bit into hi.
        *exponent = *exponent + 1;
    }

    *hi = chi;
    *lo = clo;
}

// 128-bit subtract  of ((chi << 64) + clo)  - ((*hi << 64) + *lo)
static inline void sub128(cl_ulong *chi, cl_ulong *clo, cl_ulong hi,
                          cl_ulong lo, cl_ulong *signC, int *expC)
{
    cl_ulong rHi = *chi;
    cl_ulong rLo = *clo;
    cl_ulong carry, carry2;

    // extended precision subtract
    rLo = sub_carry(rLo, lo, &carry);
    rHi = sub_carry(rHi, hi, &carry2);
    rHi = sub_carry(rHi, carry, &carry);

    // Check for sign flip
    if (carry || carry2)
    {
        *signC ^= 0x8000000000000000ULL;

        // negate rLo, rHi:   -x = (x ^ -1) + 1
        rLo ^= -1ULL;
        rHi ^= -1ULL;
        rLo++;
        rHi += 0 == rLo;
    }

    // normalize -- move the most significant non-zero bit to the MSB, and
    // adjust exponent accordingly
    if (rHi == 0)
    {
        rHi = rLo;
        *expC = *expC - 64;
        rLo = 0;
    }

    if (rHi)
    {
        int shift = 32;
        cl_ulong test = 1ULL << 32;
        while (0 == (rHi & 0x8000000000000000ULL))
        {
            if (rHi < test)
            {
                rHi <<= shift;
                rHi |= rLo >> (64 - shift);
                rLo <<= shift;
                *expC = *expC - shift;
            }
            shift >>= 1;
            test <<= shift;
        }
    }
    else
    {
        // zero
        *expC = INT_MIN;
        *signC = 0;
    }


    *chi = rHi;
    *clo = rLo;
}

long double reference_fmal(long double x, long double y, long double z)
{
    static const cl_ulong kMSB = 0x8000000000000000ULL;

    // cast values back to double. This is an exact function, so
    double a = x;
    double b = y;
    double c = z;

    // Make bits accessible
    union {
        cl_ulong u;
        cl_double d;
    } ua;
    ua.d = a;
    union {
        cl_ulong u;
        cl_double d;
    } ub;
    ub.d = b;
    union {
        cl_ulong u;
        cl_double d;
    } uc;
    uc.d = c;

    // deal with Nans, infinities and zeros
    if (isnan(a) || isnan(b) || isnan(c) || isinf(a) || isinf(b) || isinf(c)
        || 0 == (ua.u & ~kMSB) || // a == 0, defeat host FTZ behavior
        0 == (ub.u & ~kMSB) || // b == 0, defeat host FTZ behavior
        0 == (uc.u & ~kMSB)) // c == 0, defeat host FTZ behavior
    {
        if (isinf(c) && !isinf(a) && !isinf(b)) return (c + a) + b;

        a = (double)reference_multiplyl(
            a, b); // some risk that the compiler will insert a non-compliant
                   // fma here on some platforms.
        return reference_addl(
            a,
            c); // We use STDC FP_CONTRACT OFF above to attempt to defeat that.
    }

    // extract exponent and mantissa
    //   exponent is a standard unbiased signed integer
    //   mantissa is a cl_uint, with leading non-zero bit positioned at the MSB
    cl_ulong mantA, mantB, mantC;
    int expA = extract(a, &mantA);
    int expB = extract(b, &mantB);
    int expC = extract(c, &mantC);
    cl_ulong signC = uc.u & kMSB; // We'll need the sign bit of C later to
                                  // decide if we are adding or subtracting

    // exact product of A and B
    int exponent = expA + expB;
    cl_ulong sign = (ua.u ^ ub.u) & kMSB;
    cl_ulong hi, lo;
    mul128(mantA, mantB, &hi, &lo);

    // renormalize
    if (0 == (kMSB & hi))
    {
        hi <<= 1;
        hi |= lo >> 63;
        lo <<= 1;
    }
    else
        exponent++; // 2**63 * 2**63 gives 2**126. If the MSB was set, then our
                    // exponent increased.

    // infinite precision add
    cl_ulong chi = mantC;
    cl_ulong clo = 0;

    if (exponent >= expC)
    {
        // Normalize C relative to the product
        if (exponent > expC)
            shift_right_sticky_128(&chi, &clo, exponent - expC);

        // Add
        if (sign ^ signC)
            sub128(&hi, &lo, chi, clo, &sign, &exponent);
        else
            add128(&hi, &lo, chi, clo, &exponent);
    }
    else
    {
        // Shift the product relative to C so that their exponents match
        shift_right_sticky_128(&hi, &lo, expC - exponent);

        // add
        if (sign ^ signC)
            sub128(&chi, &clo, hi, lo, &signC, &expC);
        else
            add128(&chi, &clo, hi, lo, &expC);

        hi = chi;
        lo = clo;
        exponent = expC;
        sign = signC;
    }

    // round
    ua.d = round_to_nearest_even_double(hi, lo, exponent);

    // Set the sign
    ua.u |= sign;

    return ua.d;
}


long double reference_madl(long double a, long double b, long double c)
{
    return a * b + c;
}

long double reference_recipl(long double x) { return 1.0L / x; }

long double reference_rootnl(long double x, int i)
{
    // rootn ( x, 0 )  returns a NaN.
    if (0 == i) return cl_make_nan();

    // rootn ( x, n )  returns a NaN for x < 0 and n is even.
    if (x < 0.0L && 0 == (i & 1)) return cl_make_nan();

    if (isinf(x))
    {
        if (i < 0) return reference_copysignl(0.0L, x);

        return x;
    }

    if (x == 0.0)
    {
        switch (i & 0x80000001)
        {
            // rootn ( +-0,  n ) is +0 for even n > 0.
            case 0: return 0.0L;

            // rootn ( +-0,  n ) is +-0 for odd n > 0.
            case 1: return x;

            // rootn ( +-0,  n ) is +inf for even n < 0.
            case 0x80000000: return INFINITY;

            // rootn ( +-0,  n ) is +-inf for odd n < 0.
            case 0x80000001: return copysign(INFINITY, x);
        }
    }

    if (i == 1) return x;

    if (i == -1) return 1.0 / x;

    long double sign = x;
    x = reference_fabsl(x);
    double iHi, iLo;
    DivideDD(&iHi, &iLo, 1.0, i);
    x = reference_powl(x, iHi) * reference_powl(x, iLo);

    return reference_copysignl(x, sign);
}

long double reference_rsqrtl(long double x) { return 1.0L / sqrtl(x); }

long double reference_sinpil(long double x)
{
    double r = reduce1l(x);

    // reduce to [-0.5, 0.5]
    if (r < -0.5L)
        r = -1.0L - r;
    else if (r > 0.5L)
        r = 1.0L - r;

    // sinPi zeros have the same sign as x
    if (r == 0.0L) return reference_copysignl(0.0L, x);

    return reference_sinl(r * M_PIL);
}

long double reference_tanpil(long double x)
{
    // set aside the sign  (allows us to preserve sign of -0)
    long double sign = reference_copysignl(1.0L, x);
    long double z = reference_fabsl(x);

    // if big and even  -- caution: only works if x only has single precision
    if (z >= HEX_LDBL(+, 1, 0, +, 53))
    {
        if (z == INFINITY) return x - x; // nan

        return reference_copysignl(
            0.0L, x); // tanpi ( n ) is copysign( 0.0, n)  for even integers n.
    }

    // reduce to the range [ -0.5, 0.5 ]
    long double nearest =
        reference_rintl(z); // round to nearest even places n + 0.5 values in
                            // the right place for us
    int64_t i =
        (int64_t)nearest; // test above against 0x1.0p53 avoids overflow here
    z -= nearest;

    // correction for odd integer x for the right sign of zero
    if ((i & 1) && z == 0.0L) sign = -sign;

    // track changes to the sign
    sign *= reference_copysignl(1.0L, z); // really should just be an xor
    z = reference_fabsl(z); // remove the sign again

    // reduce once more
    // If we don't do this, rounding error in z * M_PI will cause us not to
    // return infinities properly
    if (z > 0.25L)
    {
        z = 0.5L - z;
        return sign
            / reference_tanl(z
                             * M_PIL); // use system tan to get the right result
    }

    //
    return sign
        * reference_tanl(z * M_PIL); // use system tan to get the right result
}

long double reference_pownl(long double x, int i)
{
    return reference_powl(x, (long double)i);
}

long double reference_powrl(long double x, long double y)
{
    // powr ( x, y ) returns NaN for x < 0.
    if (x < 0.0L) return cl_make_nan();

    // powr ( x, NaN ) returns the NaN for x >= 0.
    // powr ( NaN, y ) returns the NaN.
    if (isnan(x) || isnan(y))
        return x + y; // Note: behavior different here than for pow(1,NaN),
                      // pow(NaN, 0)

    if (x == 1.0L)
    {
        // powr ( +1, +-inf ) returns NaN.
        if (reference_fabsl(y) == INFINITY) return cl_make_nan();

        // powr ( +1, y ) is 1 for finite y.    (NaN handled above)
        return 1.0L;
    }

    if (y == 0.0L)
    {
        // powr ( +inf, +-0 ) returns NaN.
        // powr ( +-0, +-0 ) returns NaN.
        if (x == 0.0L || x == INFINITY) return cl_make_nan();

        // powr ( x, +-0 ) is 1 for finite x > 0.  (x <= 0, NaN, INF already
        // handled above)
        return 1.0L;
    }

    if (x == 0.0L)
    {
        // powr ( +-0, -inf) is +inf.
        // powr ( +-0, y ) is +inf for finite y < 0.
        if (y < 0.0L) return INFINITY;

        // powr ( +-0, y ) is +0 for y > 0.    (NaN, y==0 handled above)
        return 0.0L;
    }

    return reference_powl(x, y);
}

long double reference_addl(long double x, long double y)
{
    volatile double a = (double)x;
    volatile double b = (double)y;

#if defined(__SSE2__)
    // defeat x87
    __m128d va = _mm_set_sd((double)a);
    __m128d vb = _mm_set_sd((double)b);
    va = _mm_add_sd(va, vb);
    _mm_store_sd((double *)&a, va);
#else
    a += b;
#endif
    return (long double)a;
}

long double reference_subtractl(long double x, long double y)
{
    volatile double a = (double)x;
    volatile double b = (double)y;

#if defined(__SSE2__)
    // defeat x87
    __m128d va = _mm_set_sd((double)a);
    __m128d vb = _mm_set_sd((double)b);
    va = _mm_sub_sd(va, vb);
    _mm_store_sd((double *)&a, va);
#else
    a -= b;
#endif
    return (long double)a;
}

long double reference_multiplyl(long double x, long double y)
{
    volatile double a = (double)x;
    volatile double b = (double)y;

#if defined(__SSE2__)
    // defeat x87
    __m128d va = _mm_set_sd((double)a);
    __m128d vb = _mm_set_sd((double)b);
    va = _mm_mul_sd(va, vb);
    _mm_store_sd((double *)&a, va);
#else
    a *= b;
#endif
    return (long double)a;
}

long double reference_lgamma_rl(long double x, int *signp)
{
    *signp = 0;
    return x;
}

int reference_isequall(long double x, long double y) { return x == y; }
int reference_isfinitel(long double x) { return 0 != isfinite(x); }
int reference_isgreaterl(long double x, long double y) { return x > y; }
int reference_isgreaterequall(long double x, long double y) { return x >= y; }
int reference_isinfl(long double x) { return 0 != isinf(x); }
int reference_islessl(long double x, long double y) { return x < y; }
int reference_islessequall(long double x, long double y) { return x <= y; }
#if defined(__INTEL_COMPILER)
int reference_islessgreaterl(long double x, long double y)
{
    return 0 != islessgreaterl(x, y);
}
#else
int reference_islessgreaterl(long double x, long double y)
{
    return 0 != islessgreater(x, y);
}
#endif
int reference_isnanl(long double x) { return 0 != isnan(x); }
int reference_isnormall(long double x) { return 0 != isnormal((double)x); }
int reference_isnotequall(long double x, long double y) { return x != y; }
int reference_isorderedl(long double x, long double y)
{
    return x == x && y == y;
}
int reference_isunorderedl(long double x, long double y)
{
    return isnan(x) || isnan(y);
}
#if defined(__INTEL_COMPILER)
int reference_signbitl(long double x) { return 0 != signbitl(x); }
#else
int reference_signbitl(long double x) { return 0 != signbit(x); }
#endif
long double reference_copysignl(long double x, long double y);
long double reference_roundl(long double x);
long double reference_cbrtl(long double x);

long double reference_copysignl(long double x, long double y)
{
    // We hope that the long double to double conversion proceeds with sign
    // fidelity, even for zeros and NaNs
    union {
        double d;
        cl_ulong u;
    } u;
    u.d = (double)y;

    x = reference_fabsl(x);
    if (u.u >> 63) x = -x;

    return x;
}

long double reference_roundl(long double x)
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here

#if defined(__MINGW32__) && defined(__x86_64__)
    long double absx = reference_fabsl(x);
    if (absx < 0.5L) return reference_copysignl(0.0L, x);
#endif
    return round((double)x);
}

long double reference_truncl(long double x)
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here
    return trunc((double)x);
}

static long double reference_scalblnl(long double x, long n);

long double reference_cbrtl(long double x)
{
    double yhi = HEX_DBL(+, 1, 5555555555555, -, 2);
    double ylo = HEX_DBL(+, 1, 558, -, 56);

    double fabsx = reference_fabs(x);

    if (isnan(x) || fabsx == 1.0 || fabsx == 0.0 || isinf(x)) return x;

    double log2x_hi, log2x_lo;

    // extended precision log .... accurate to at least 64-bits + couple of
    // guard bits
    __log2_ep(&log2x_hi, &log2x_lo, fabsx);

    double ylog2x_hi, ylog2x_lo;

    double y_hi = yhi;
    double y_lo = ylo;

    // compute product of y*log2(x)
    MulDD(&ylog2x_hi, &ylog2x_lo, log2x_hi, log2x_lo, y_hi, y_lo);

    long double powxy;
    if (isinf(ylog2x_hi) || (reference_fabs(ylog2x_hi) > 2200))
    {
        powxy =
            reference_signbit(ylog2x_hi) ? HEX_DBL(+, 0, 0, +, 0) : INFINITY;
    }
    else
    {
        // separate integer + fractional part
        long int m = lrint(ylog2x_hi);
        AddDD(&ylog2x_hi, &ylog2x_lo, ylog2x_hi, ylog2x_lo, -m, 0.0);

        // revert to long double arithemtic
        long double ylog2x = (long double)ylog2x_hi + (long double)ylog2x_lo;
        powxy = reference_exp2l(ylog2x);
        powxy = reference_scalblnl(powxy, m);
    }

    return reference_copysignl(powxy, x);
}

long double reference_rintl(long double x)
{
#if defined(__PPC__)
    // On PPC, long doubles are maintained as 2 doubles. Therefore, the combined
    // mantissa can represent more than LDBL_MANT_DIG binary digits.
    x = rintl(x);
#else
    static long double magic[2] = { 0.0L, 0.0L };

    if (0.0L == magic[0])
    {
        magic[0] = scalbnl(0.5L, LDBL_MANT_DIG);
        magic[1] = scalbnl(-0.5L, LDBL_MANT_DIG);
    }

    if (reference_fabsl(x) < magic[0] && x != 0.0L)
    {
        long double m = magic[x < 0];
        x += m;
        x -= m;
    }
#endif // __PPC__
    return x;
}

// extended precision sqrt using newton iteration on 1/sqrt(x).
// Final result is computed as x * 1/sqrt(x)
static void __sqrt_ep(double *rhi, double *rlo, double xhi, double xlo)
{
    // approximate reciprocal sqrt
    double thi = 1.0 / sqrt(xhi);
    double tlo = 0.0;

    // One newton iteration in double-double
    double yhi, ylo;
    MulDD(&yhi, &ylo, thi, tlo, thi, tlo);
    MulDD(&yhi, &ylo, yhi, ylo, xhi, xlo);
    AddDD(&yhi, &ylo, -yhi, -ylo, 3.0, 0.0);
    MulDD(&yhi, &ylo, yhi, ylo, thi, tlo);
    MulDD(&yhi, &ylo, yhi, ylo, 0.5, 0.0);

    MulDD(rhi, rlo, yhi, ylo, xhi, xlo);
}

long double reference_acoshl(long double x)
{
    /*
     * ====================================================
     * This function derived from fdlibm http://www.netlib.org
     * It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     *
     */
    if (isnan(x) || isinf(x)) return x + fabsl(x);

    if (x < 1.0L) return cl_make_nan();

    if (x == 1.0L) return 0.0L;

    if (x > HEX_LDBL(+, 1, 0, +, 60))
        return reference_logl(x) + 0.693147180559945309417232121458176568L;

    if (x > 2.0L)
        return reference_logl(2.0L * x - 1.0L / (x + sqrtl(x * x - 1.0L)));

    double hi, lo;
    MulD(&hi, &lo, x, x);
    AddDD(&hi, &lo, hi, lo, -1.0, 0.0);
    __sqrt_ep(&hi, &lo, hi, lo);
    AddDD(&hi, &lo, hi, lo, x, 0.0);
    double correction = lo / hi;
    __log2_ep(&hi, &lo, hi);
    double log2Hi = HEX_DBL(+, 1, 62e42fefa39ef, -, 1);
    double log2Lo = HEX_DBL(+, 1, abc9e3b39803f, -, 56);
    MulDD(&hi, &lo, hi, lo, log2Hi, log2Lo);
    AddDD(&hi, &lo, hi, lo, correction, 0.0);

    return hi + lo;
}

long double reference_asinhl(long double x)
{
    long double cutoff = 0.0L;
    const long double ln2 = HEX_LDBL(+, b, 17217f7d1cf79ab, -, 4);

    if (cutoff == 0.0L) cutoff = reference_ldexpl(1.0L, -LDBL_MANT_DIG);

    if (isnan(x) || isinf(x)) return x + x;

    long double absx = reference_fabsl(x);
    if (absx < cutoff) return x;

    long double sign = reference_copysignl(1.0L, x);

    if (absx <= 4.0 / 3.0)
    {
        return sign
            * reference_log1pl(absx + x * x / (1.0 + sqrtl(1.0 + x * x)));
    }
    else if (absx <= HEX_LDBL(+, 1, 0, +, 27))
    {
        return sign
            * reference_logl(2.0L * absx + 1.0L / (sqrtl(x * x + 1.0) + absx));
    }
    else
    {
        return sign * (reference_logl(absx) + ln2);
    }
}

long double reference_atanhl(long double x)
{
    /*
     * ====================================================
     * This function is from fdlibm: http://www.netlib.org
     *   It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     */
    if (isnan(x)) return x + x;

    long double signed_half = reference_copysignl(0.5L, x);
    x = reference_fabsl(x);
    if (x > 1.0L) return cl_make_nan();

    if (x < 0.5L)
        return signed_half * reference_log1pl(2.0L * (x + x * x / (1 - x)));

    return signed_half * reference_log1pl(2.0L * x / (1 - x));
}

long double reference_exp2l(long double z)
{
    double_double x;
    int j;

    // Handle NaNs
    if (isnan(z)) return z;

    // init x
    x.hi = z;
    x.lo = z - x.hi;

    // Deal with overflow and underflow for exp2(x) stage next
    if (x.hi >= 1025) return INFINITY;

    if (x.hi < -1075 - 24) return +0.0;

    // find nearest integer to x
    int i = (int)rint(x.hi);

    // x now holds fractional part.  The result would be then 2**i  * exp2( x )
    x.hi -= i;

    // We could attempt to find a minimax polynomial for exp2(x) over the range
    // x = [-0.5, 0.5]. However, this would converge very slowly near the
    // extrema, where 0.5**n is not a lot different from 0.5**(n+1), thereby
    // requiring something like a 20th order polynomial to get 53 + 24 bits of
    // precision. Instead we further reduce the range to [-1/32, 1/32] by
    // observing that
    //
    //  2**(a+b) = 2**a * 2**b
    //
    // We can thus build a table of 2**a values for a = n/16, n = [-8, 8], and
    // reduce the range of x to [-1/32, 1/32] by subtracting away the nearest
    // value of n/16 from x.
    const double_double corrections[17] = {
        { HEX_DBL(+, 1, 6a09e667f3bcd, -, 1),
          HEX_DBL(-, 1, bdd3413b26456, -, 55) },
        { HEX_DBL(+, 1, 7a11473eb0187, -, 1),
          HEX_DBL(-, 1, 41577ee04992f, -, 56) },
        { HEX_DBL(+, 1, 8ace5422aa0db, -, 1),
          HEX_DBL(+, 1, 6e9f156864b27, -, 55) },
        { HEX_DBL(+, 1, 9c49182a3f09, -, 1),
          HEX_DBL(+, 1, c7c46b071f2be, -, 57) },
        { HEX_DBL(+, 1, ae89f995ad3ad, -, 1),
          HEX_DBL(+, 1, 7a1cd345dcc81, -, 55) },
        { HEX_DBL(+, 1, c199bdd85529c, -, 1),
          HEX_DBL(+, 1, 11065895048dd, -, 56) },
        { HEX_DBL(+, 1, d5818dcfba487, -, 1),
          HEX_DBL(+, 1, 2ed02d75b3707, -, 56) },
        { HEX_DBL(+, 1, ea4afa2a490da, -, 1),
          HEX_DBL(-, 1, e9c23179c2893, -, 55) },
        { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
        { HEX_DBL(+, 1, 0b5586cf9890f, +, 0),
          HEX_DBL(+, 1, 8a62e4adc610b, -, 54) },
        { HEX_DBL(+, 1, 172b83c7d517b, +, 0),
          HEX_DBL(-, 1, 19041b9d78a76, -, 55) },
        { HEX_DBL(+, 1, 2387a6e756238, +, 0),
          HEX_DBL(+, 1, 9b07eb6c70573, -, 54) },
        { HEX_DBL(+, 1, 306fe0a31b715, +, 0),
          HEX_DBL(+, 1, 6f46ad23182e4, -, 55) },
        { HEX_DBL(+, 1, 3dea64c123422, +, 0),
          HEX_DBL(+, 1, ada0911f09ebc, -, 55) },
        { HEX_DBL(+, 1, 4bfdad5362a27, +, 0),
          HEX_DBL(+, 1, d4397afec42e2, -, 56) },
        { HEX_DBL(+, 1, 5ab07dd485429, +, 0),
          HEX_DBL(+, 1, 6324c054647ad, -, 54) },
        { HEX_DBL(+, 1, 6a09e667f3bcd, +, 0),
          HEX_DBL(-, 1, bdd3413b26456, -, 54) }
    };
    int index = (int)rint(x.hi * 16.0);
    x.hi -= (double)index * 0.0625;

    // canonicalize x
    double temp = x.hi;
    x.hi += x.lo;
    x.lo -= x.hi - temp;

    // Minimax polynomial for (exp2(x)-1)/x, over the range [-1/32, 1/32].  Max
    // Error: 2 * 0x1.e112p-87
    const double_double c[] = { { HEX_DBL(+, 1, 62e42fefa39ef, -, 1),
                                  HEX_DBL(+, 1, abc9e3ac1d244, -, 56) },
                                { HEX_DBL(+, 1, ebfbdff82c58f, -, 3),
                                  HEX_DBL(-, 1, 5e4987a631846, -, 57) },
                                { HEX_DBL(+, 1, c6b08d704a0c, -, 5),
                                  HEX_DBL(-, 1, d323200a05713, -, 59) },
                                { HEX_DBL(+, 1, 3b2ab6fba4e7a, -, 7),
                                  HEX_DBL(+, 1, c5ee8f8b9f0c1, -, 63) },
                                { HEX_DBL(+, 1, 5d87fe78a672a, -, 10),
                                  HEX_DBL(+, 1, 884e5e5cc7ecc, -, 64) },
                                { HEX_DBL(+, 1, 430912f7e8373, -, 13),
                                  HEX_DBL(+, 1, 4f1b59514a326, -, 67) },
                                { HEX_DBL(+, 1, ffcbfc5985e71, -, 17),
                                  HEX_DBL(-, 1, db7d6a0953b78, -, 71) },
                                { HEX_DBL(+, 1, 62c150eb16465, -, 20),
                                  HEX_DBL(+, 1, e0767c2d7abf5, -, 80) },
                                { HEX_DBL(+, 1, b52502b5e953, -, 24),
                                  HEX_DBL(+, 1, 6797523f944bc, -, 78) } };
    size_t count = sizeof(c) / sizeof(c[0]);

    // Do polynomial
    double_double r = c[count - 1];
    for (j = (int)count - 2; j >= 0; j--) r = add_dd(c[j], mul_dd(r, x));

    // unwind approximation
    r = mul_dd(r, x); // before: r =(exp2(x)-1)/x;   after: r = exp2(x) - 1

    // correct for [-0.5, 0.5] -> [-1/32, 1/32] reduction above
    //  exp2(x) = (r + 1) * correction = r * correction + correction
    r = mul_dd(r, corrections[index + 8]);
    r = add_dd(r, corrections[index + 8]);

    // Format result for output:

    // Get mantissa
    long double m = ((long double)r.hi + (long double)r.lo);

    // Handle a pesky overflow cases when long double = double
    if (i > 512)
    {
        m *= HEX_DBL(+, 1, 0, +, 512);
        i -= 512;
    }
    else if (i < -512)
    {
        m *= HEX_DBL(+, 1, 0, -, 512);
        i += 512;
    }

    return m * ldexpl(1.0L, i);
}

long double reference_expm1l(long double x)
{
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // unimplemented
    return x;
#else
    if (reference_isnanl(x)) return x;

    if (x > 710) return INFINITY;

    long double y = expm1l(x);

    // Range of expm1l is -1.0L to +inf. Negative inf
    // on a few Linux platforms is clearly the wrong sign.
    if (reference_isinfl(y)) y = INFINITY;

    return y;
#endif
}

long double reference_fmaxl(long double x, long double y)
{
    if (isnan(y)) return x;

    return x >= y ? x : y;
}

long double reference_fminl(long double x, long double y)
{
    if (isnan(y)) return x;

    return x <= y ? x : y;
}

long double reference_hypotl(long double x, long double y)
{
    static const double tobig = HEX_DBL(+, 1, 0, +, 511);
    static const double big = HEX_DBL(+, 1, 0, +, 513);
    static const double rbig = HEX_DBL(+, 1, 0, -, 513);
    static const double tosmall = HEX_DBL(+, 1, 0, -, 511);
    static const double smalll = HEX_DBL(+, 1, 0, -, 607);
    static const double rsmall = HEX_DBL(+, 1, 0, +, 607);

    long double max, min;

    if (isinf(x) || isinf(y)) return INFINITY;

    if (isnan(x) || isnan(y)) return x + y;

    x = reference_fabsl(x);
    y = reference_fabsl(y);

    max = reference_fmaxl(x, y);
    min = reference_fminl(x, y);

    if (max > tobig)
    {
        max *= rbig;
        min *= rbig;
        return big * sqrtl(max * max + min * min);
    }

    if (max < tosmall)
    {
        max *= rsmall;
        min *= rsmall;
        return smalll * sqrtl(max * max + min * min);
    }
    return sqrtl(x * x + y * y);
}

long double reference_log2l(long double x)
{
    if (isnan(x) || x < 0.0 || x == -INFINITY) return NAN;

    if (x == 0.0f) return -INFINITY;

    if (x == INFINITY) return INFINITY;

    double hi, lo;
    __log2_ep(&hi, &lo, x);

    return (long double)hi + (long double)lo;
}

long double reference_log1pl(long double x)
{
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // unimplemented
    return x;
#elif defined(__PPC__)
    // log1pl on PPC inadvertantly returns NaN for very large values. Work
    // around this limitation by returning logl for large values.
    return ((x > (long double)(0x1.0p+1022)) ? logl(x) : log1pl(x));
#else
    return log1pl(x);
#endif
}

long double reference_logbl(long double x)
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here
    union {
        double f;
        cl_ulong u;
    } u;
    u.f = (double)x;

    cl_int exponent = (cl_uint)(u.u >> 52) & 0x7ff;
    if (exponent == 0x7ff) return x * x;

    if (exponent == 0)
    { // deal with denormals
        u.f = x * HEX_DBL(+, 1, 0, +, 64);
        exponent = (cl_int)(u.u >> 52) & 0x7ff;
        if (exponent == 0) return -INFINITY;

        return exponent - (1023 + 64);
    }

    return exponent - 1023;
}

long double reference_maxmagl(long double x, long double y)
{
    long double fabsx = fabsl(x);
    long double fabsy = fabsl(y);

    if (fabsx < fabsy) return y;

    if (fabsy < fabsx) return x;

    return reference_fmaxl(x, y);
}

long double reference_minmagl(long double x, long double y)
{
    long double fabsx = fabsl(x);
    long double fabsy = fabsl(y);

    if (fabsx > fabsy) return y;

    if (fabsy > fabsx) return x;

    return reference_fminl(x, y);
}

long double reference_nanl(cl_ulong x)
{
    union {
        cl_ulong u;
        cl_double f;
    } u;
    u.u = x | 0x7ff8000000000000ULL;
    return (long double)u.f;
}


long double reference_reciprocall(long double x) { return 1.0L / x; }

long double reference_remainderl(long double x, long double y)
{
    int i;
    return reference_remquol(x, y, &i);
}

long double reference_lgammal(long double x)
{
    // lgamma is currently not tested
    return reference_lgamma(x);
}

static uint32_t two_over_pi[] = {
    0x0,        0x28be60db, 0x24e44152, 0x27f09d5f, 0x11f534dd, 0x3036d8a5,
    0x1993c439, 0x107f945,  0x23abdebb, 0x31586dc9, 0x6e3a424,  0x374b8019,
    0x92eea09,  0x3464873f, 0x21deb1cb, 0x4a69cfb,  0x288235f5, 0xbaed121,
    0xe99c702,  0x1ad17df9, 0x13991d6,  0xe60d4ce,  0x1f49c845, 0x3e2ef7e4,
    0x283b1ff8, 0x25fff781, 0x1980fef2, 0x3c462d68, 0xa6d1f6d,  0xd9fb3c9,
    0x3cb09b74, 0x3d18fd9a, 0x1e5fea2d, 0x1d49eeb1, 0x3ebe5f17, 0x2cf41ce7,
    0x378a5292, 0x3a9afed7, 0x3b11f8d5, 0x3421580c, 0x3046fc7b, 0x1aeafc33,
    0x3bc209af, 0x10d876a7, 0x2391615e, 0x3986c219, 0x199855f1, 0x1281a102,
    0xdffd880,  0x135cc9cc, 0x10606155
};

static uint32_t pi_over_two[] = { 0x1,        0x2487ed51, 0x42d1846,
                                  0x26263314, 0x1701b839, 0x28948127 };

union d_ui64_t {
    uint64_t u;
    double d;
};

// radix or base of representation
#define RADIX (30)
#define DIGITS 6

d_ui64_t two_pow_pradix = { (uint64_t)(1023 + RADIX) << 52 };
d_ui64_t two_pow_mradix = { (uint64_t)(1023 - RADIX) << 52 };
d_ui64_t two_pow_two_mradix = { (uint64_t)(1023 - 2 * RADIX) << 52 };

#define tp_pradix two_pow_pradix.d
#define tp_mradix two_pow_mradix.d

// extended fixed point representation of double precision
// floating point number.
// x = sign * [ sum_{i = 0 to 2} ( X[i] * 2^(index - i)*RADIX ) ]
struct eprep_t
{
    uint32_t X[3]; // three 32 bit integers are sufficient to represnt double in
                   // base_30
    int index; // exponent bias
    int sign; // sign of double
};

static eprep_t double_to_eprep(double x)
{
    eprep_t result;

    result.sign = (signbit(x) == 0) ? 1 : -1;
    x = fabs(x);

    int index = 0;
    while (x > tp_pradix)
    {
        index++;
        x *= tp_mradix;
    }
    while (x < 1)
    {
        index--;
        x *= tp_pradix;
    }

    result.index = index;
    int i = 0;
    result.X[0] = result.X[1] = result.X[2] = 0;
    while (x != 0.0)
    {
        result.X[i] = (uint32_t)x;
        x = (x - (double)result.X[i]) * tp_pradix;
        i++;
    }
    return result;
}

static double eprep_to_double(eprep_t epx)
{
    double res = 0.0;

    res += ldexp((double)epx.X[0], (epx.index - 0) * RADIX);
    res += ldexp((double)epx.X[1], (epx.index - 1) * RADIX);
    res += ldexp((double)epx.X[2], (epx.index - 2) * RADIX);

    return copysign(res, epx.sign);
}

static int payne_hanek(double *y, int *exception)
{
    double x = *y;

    // exception cases .. no reduction required
    if (isnan(x) || isinf(x) || (fabs(x) <= M_PI_4))
    {
        *exception = 1;
        return 0;
    }

    *exception = 0;

    // After computation result[0] contains integer part while
    // result[1]....result[DIGITS-1] contain fractional part. So we are doing
    // computation with (DIGITS-1)*RADIX precision. Default DIGITS=6 and
    // RADIX=30 so default precision is 150 bits. Kahan-McDonald algorithm shows
    // that a double precision x, closest to pi/2 is 6381956970095103 x 2^797
    // which can cause 61 digits of cancellation in computation of f = x*2/pi -
    // floor(x*2/pi) ... thus we need at least 114 bits (61 leading zeros + 53
    // bits of mentissa of f) of precision to accurately compute f in double
    // precision. Since we are using 150 bits (still an overkill), we should be
    // safe. Extra bits can act as guard bits for correct rounding.
    uint64_t result[DIGITS + 2];

    // compute extended precision representation of x
    eprep_t epx = double_to_eprep(x);
    int index = epx.index;
    int i, j;
    // extended precision multiplication of 2/pi*x .... we will loose at max two
    // RADIX=30 bit digits in the worst case
    for (i = 0; i < (DIGITS + 2); i++)
    {
        result[i] = 0;
        result[i] += ((index + i - 0) >= 0)
            ? ((uint64_t)two_over_pi[index + i - 0] * (uint64_t)epx.X[0])
            : 0;
        result[i] += ((index + i - 1) >= 0)
            ? ((uint64_t)two_over_pi[index + i - 1] * (uint64_t)epx.X[1])
            : 0;
        result[i] += ((index + i - 2) >= 0)
            ? ((uint64_t)two_over_pi[index + i - 2] * (uint64_t)epx.X[2])
            : 0;
    }

    // Carry propagation.
    uint64_t tmp;
    for (i = DIGITS + 2 - 1; i > 0; i--)
    {
        tmp = result[i] >> RADIX;
        result[i - 1] += tmp;
        result[i] -= (tmp << RADIX);
    }

    // we dont ned to normalize the integer part since only last two bits of
    // this will be used subsequently algorithm which remain unaltered by this
    // normalization. tmp = result[0] >> RADIX; result[0] -= (tmp << RADIX);
    unsigned int N = (unsigned int)result[0];

    // if the result is > pi/4, bring it to (-pi/4, pi/4] range. Note that
    // testing if the final x_star = pi/2*(x*2/pi - k) > pi/4 is equivalent to
    // testing, at this stage, if r[1] (the first fractional digit) is greater
    // than (2^RADIX)/2 and substracting pi/4 from x_star to bring it to
    // mentioned range is equivalent to substracting fractional part at this
    // stage from one and changing the sign.
    int sign = 1;
    if (result[1] > (uint64_t)(1 << (RADIX - 1)))
    {
        for (i = 1; i < (DIGITS + 2); i++)
            result[i] = (~((unsigned int)result[i]) & 0x3fffffff);
        N += 1;
        sign = -1;
    }

    // Again as per Kahan-McDonald algorithim there may be 61 leading zeros in
    // the worst case (when x is multiple of 2/pi very close to an integer) so
    // we need to get rid of these zeros and adjust the index of final result.
    // So in the worst case, precision of comupted result is 90 bits (150 bits
    // original bits - 60 lost in cancellation).
    int ind = 1;
    for (i = 1; i < (DIGITS + 2); i++)
    {
        if (result[i] != 0)
            break;
        else
            ind++;
    }

    uint64_t r[DIGITS - 1];
    for (i = 0; i < (DIGITS - 1); i++)
    {
        r[i] = 0;
        for (j = 0; j <= i; j++)
        {
            r[i] += (result[ind + i - j] * (uint64_t)pi_over_two[j]);
        }
    }
    for (i = (DIGITS - 2); i > 0; i--)
    {
        tmp = r[i] >> RADIX;
        r[i - 1] += tmp;
        r[i] -= (tmp << RADIX);
    }
    tmp = r[0] >> RADIX;
    r[0] -= (tmp << RADIX);

    eprep_t epr;
    epr.sign = epx.sign * sign;
    if (tmp != 0)
    {
        epr.index = -ind + 1;
        epr.X[0] = (uint32_t)tmp;
        epr.X[1] = (uint32_t)r[0];
        epr.X[2] = (uint32_t)r[1];
    }
    else
    {
        epr.index = -ind;
        epr.X[0] = (uint32_t)r[0];
        epr.X[1] = (uint32_t)r[1];
        epr.X[2] = (uint32_t)r[2];
    }

    *y = eprep_to_double(epr);
    return epx.sign * N;
}

double reference_relaxed_cos(double x)
{
    if (isnan(x)) return NAN;
    return (float)cos((float)x);
}

double reference_cos(double x)
{
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return cos(x);
    unsigned int c = N & 3;
    switch (c)
    {
        case 0: return cos(x);
        case 1: return -sin(x);
        case 2: return -cos(x);
        case 3: return sin(x);
    }
    return 0.0;
}

double reference_relaxed_sin(double x) { return (float)sin((float)x); }

double reference_sin(double x)
{
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return sin(x);
    int c = N & 3;
    switch (c)
    {
        case 0: return sin(x);
        case 1: return cos(x);
        case 2: return -sin(x);
        case 3: return -cos(x);
    }
    return 0.0;
}

double reference_relaxed_sincos(double x, double *y)
{
    *y = reference_relaxed_cos(x);
    return reference_relaxed_sin(x);
}

double reference_sincos(double x, double *y)
{
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception)
    {
        *y = cos(x);
        return sin(x);
    }
    int c = N & 3;
    switch (c)
    {
        case 0: *y = cos(x); return sin(x);
        case 1: *y = -sin(x); return cos(x);
        case 2: *y = -cos(x); return -sin(x);
        case 3: *y = sin(x); return -cos(x);
    }
    return 0.0;
}

double reference_relaxed_tan(double x)
{
    return ((float)reference_relaxed_sin((float)x))
        / ((float)reference_relaxed_cos((float)x));
}

double reference_tan(double x)
{
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return tan(x);
    int c = N & 3;
    switch (c)
    {
        case 0: return tan(x);
        case 1: return -1.0 / tan(x);
        case 2: return tan(x);
        case 3: return -1.0 / tan(x);
    }
    return 0.0;
}

long double reference_cosl(long double xx)
{
    double x = (double)xx;
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return cosl(x);
    unsigned int c = N & 3;
    switch (c)
    {
        case 0: return cosl(x);
        case 1: return -sinl(x);
        case 2: return -cosl(x);
        case 3: return sinl(x);
    }
    return 0.0;
}

long double reference_sinl(long double xx)
{
    // we use system tanl after reduction which
    // can flush denorm input to zero so
    // take care of it here.
    if (reference_fabsl(xx) < HEX_DBL(+, 1, 0, -, 1022)) return xx;

    double x = (double)xx;
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return sinl(x);
    int c = N & 3;
    switch (c)
    {
        case 0: return sinl(x);
        case 1: return cosl(x);
        case 2: return -sinl(x);
        case 3: return -cosl(x);
    }
    return 0.0;
}

long double reference_sincosl(long double xx, long double *y)
{
    // we use system tanl after reduction which
    // can flush denorm input to zero so
    // take care of it here.
    if (reference_fabsl(xx) < HEX_DBL(+, 1, 0, -, 1022))
    {
        *y = cosl(xx);
        return xx;
    }

    double x = (double)xx;
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception)
    {
        *y = cosl(x);
        return sinl(x);
    }
    int c = N & 3;
    switch (c)
    {
        case 0: *y = cosl(x); return sinl(x);
        case 1: *y = -sinl(x); return cosl(x);
        case 2: *y = -cosl(x); return -sinl(x);
        case 3: *y = sinl(x); return -cosl(x);
    }
    return 0.0;
}

long double reference_tanl(long double xx)
{
    // we use system tanl after reduction which
    // can flush denorm input to zero so
    // take care of it here.
    if (reference_fabsl(xx) < HEX_DBL(+, 1, 0, -, 1022)) return xx;

    double x = (double)xx;
    int exception;
    int N = payne_hanek(&x, &exception);
    if (exception) return tanl(x);
    int c = N & 3;
    switch (c)
    {
        case 0: return tanl(x);
        case 1: return -1.0 / tanl(x);
        case 2: return tanl(x);
        case 3: return -1.0 / tanl(x);
    }
    return 0.0;
}

static double __loglTable1[64][3] = {
    { HEX_DBL(+, 1, 5390948f40fea, +, 0), HEX_DBL(-, 1, a152f142a, -, 2),
      HEX_DBL(+, 1, f93e27b43bd2c, -, 40) },
    { HEX_DBL(+, 1, 5015015015015, +, 0), HEX_DBL(-, 1, 921800925, -, 2),
      HEX_DBL(+, 1, 162432a1b8df7, -, 41) },
    { HEX_DBL(+, 1, 4cab88725af6e, +, 0), HEX_DBL(-, 1, 8304d90c18, -, 2),
      HEX_DBL(+, 1, 80bb749056fe7, -, 40) },
    { HEX_DBL(+, 1, 49539e3b2d066, +, 0), HEX_DBL(-, 1, 7418acebc, -, 2),
      HEX_DBL(+, 1, ceac7f0607711, -, 43) },
    { HEX_DBL(+, 1, 460cbc7f5cf9a, +, 0), HEX_DBL(-, 1, 6552b49988, -, 2),
      HEX_DBL(+, 1, d8913d0e89fa, -, 42) },
    { HEX_DBL(+, 1, 42d6625d51f86, +, 0), HEX_DBL(-, 1, 56b22e6b58, -, 2),
      HEX_DBL(+, 1, c7eaf515033a1, -, 44) },
    { HEX_DBL(+, 1, 3fb013fb013fb, +, 0), HEX_DBL(-, 1, 48365e696, -, 2),
      HEX_DBL(+, 1, 434adcde7edc7, -, 41) },
    { HEX_DBL(+, 1, 3c995a47babe7, +, 0), HEX_DBL(-, 1, 39de8e156, -, 2),
      HEX_DBL(+, 1, 8246f8e527754, -, 40) },
    { HEX_DBL(+, 1, 3991c2c187f63, +, 0), HEX_DBL(-, 1, 2baa0c34c, -, 2),
      HEX_DBL(+, 1, e1513c28e180d, -, 42) },
    { HEX_DBL(+, 1, 3698df3de0747, +, 0), HEX_DBL(-, 1, 1d982c9d58, -, 2),
      HEX_DBL(+, 1, 63ea3fed4b8a2, -, 40) },
    { HEX_DBL(+, 1, 33ae45b57bcb1, +, 0), HEX_DBL(-, 1, 0fa848045, -, 2),
      HEX_DBL(+, 1, 32ccbacf1779b, -, 40) },
    { HEX_DBL(+, 1, 30d190130d19, +, 0), HEX_DBL(-, 1, 01d9bbcfa8, -, 2),
      HEX_DBL(+, 1, e2bfeb2b884aa, -, 42) },
    { HEX_DBL(+, 1, 2e025c04b8097, +, 0), HEX_DBL(-, 1, e857d3d37, -, 3),
      HEX_DBL(+, 1, d9309b4d2ea85, -, 40) },
    { HEX_DBL(+, 1, 2b404ad012b4, +, 0), HEX_DBL(-, 1, cd3c712d4, -, 3),
      HEX_DBL(+, 1, ddf360962d7ab, -, 40) },
    { HEX_DBL(+, 1, 288b01288b012, +, 0), HEX_DBL(-, 1, b2602497e, -, 3),
      HEX_DBL(+, 1, 597f8a121640f, -, 40) },
    { HEX_DBL(+, 1, 25e22708092f1, +, 0), HEX_DBL(-, 1, 97c1cb13d, -, 3),
      HEX_DBL(+, 1, 02807d15580dc, -, 40) },
    { HEX_DBL(+, 1, 23456789abcdf, +, 0), HEX_DBL(-, 1, 7d60496d, -, 3),
      HEX_DBL(+, 1, 12ce913d7a827, -, 41) },
    { HEX_DBL(+, 1, 20b470c67c0d8, +, 0), HEX_DBL(-, 1, 633a8bf44, -, 3),
      HEX_DBL(+, 1, 0648bca9c96bd, -, 40) },
    { HEX_DBL(+, 1, 1e2ef3b3fb874, +, 0), HEX_DBL(-, 1, 494f863b9, -, 3),
      HEX_DBL(+, 1, 066fceb89b0eb, -, 42) },
    { HEX_DBL(+, 1, 1bb4a4046ed29, +, 0), HEX_DBL(-, 1, 2f9e32d5c, -, 3),
      HEX_DBL(+, 1, 17b8b6c4f846b, -, 46) },
    { HEX_DBL(+, 1, 19453808ca29c, +, 0), HEX_DBL(-, 1, 162593187, -, 3),
      HEX_DBL(+, 1, 2c83506452154, -, 42) },
    { HEX_DBL(+, 1, 16e0689427378, +, 0), HEX_DBL(-, 1, f9c95dc1e, -, 4),
      HEX_DBL(+, 1, dd5d2183150f3, -, 41) },
    { HEX_DBL(+, 1, 1485f0e0acd3b, +, 0), HEX_DBL(-, 1, c7b528b72, -, 4),
      HEX_DBL(+, 1, 0e43c4f4e619d, -, 40) },
    { HEX_DBL(+, 1, 12358e75d3033, +, 0), HEX_DBL(-, 1, 960caf9ac, -, 4),
      HEX_DBL(+, 1, 20fbfd5902a1e, -, 42) },
    { HEX_DBL(+, 1, 0fef010fef01, +, 0), HEX_DBL(-, 1, 64ce26c08, -, 4),
      HEX_DBL(+, 1, 8ebeefb4ac467, -, 40) },
    { HEX_DBL(+, 1, 0db20a88f4695, +, 0), HEX_DBL(-, 1, 33f7cde16, -, 4),
      HEX_DBL(+, 1, 30b3312da7a7d, -, 40) },
    { HEX_DBL(+, 1, 0b7e6ec259dc7, +, 0), HEX_DBL(-, 1, 0387efbcc, -, 4),
      HEX_DBL(+, 1, 796f1632949c3, -, 40) },
    { HEX_DBL(+, 1, 0953f39010953, +, 0), HEX_DBL(-, 1, a6f9c378, -, 5),
      HEX_DBL(+, 1, 1687e151172cc, -, 40) },
    { HEX_DBL(+, 1, 073260a47f7c6, +, 0), HEX_DBL(-, 1, 47aa07358, -, 5),
      HEX_DBL(+, 1, 1f87e4a9cc778, -, 42) },
    { HEX_DBL(+, 1, 05197f7d73404, +, 0), HEX_DBL(-, 1, d23afc498, -, 6),
      HEX_DBL(+, 1, b183a6b628487, -, 40) },
    { HEX_DBL(+, 1, 03091b51f5e1a, +, 0), HEX_DBL(-, 1, 16a21e21, -, 6),
      HEX_DBL(+, 1, 7d75c58973ce5, -, 40) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, f44659e4a4271, -, 1), HEX_DBL(+, 1, 11cd1d51, -, 5),
      HEX_DBL(+, 1, 9a0d857e2f4b2, -, 40) },
    { HEX_DBL(+, 1, ecc07b301ecc, -, 1), HEX_DBL(+, 1, c4dfab908, -, 5),
      HEX_DBL(+, 1, 55b53fce557fd, -, 40) },
    { HEX_DBL(+, 1, e573ac901e573, -, 1), HEX_DBL(+, 1, 3aa2fdd26, -, 4),
      HEX_DBL(+, 1, f1cb0c9532089, -, 40) },
    { HEX_DBL(+, 1, de5d6e3f8868a, -, 1), HEX_DBL(+, 1, 918a16e46, -, 4),
      HEX_DBL(+, 1, 9af0dcd65a6e1, -, 43) },
    { HEX_DBL(+, 1, d77b654b82c33, -, 1), HEX_DBL(+, 1, e72ec117e, -, 4),
      HEX_DBL(+, 1, a5b93c4ebe124, -, 40) },
    { HEX_DBL(+, 1, d0cb58f6ec074, -, 1), HEX_DBL(+, 1, 1dcd19755, -, 3),
      HEX_DBL(+, 1, 5be50e71ddc6c, -, 42) },
    { HEX_DBL(+, 1, ca4b3055ee191, -, 1), HEX_DBL(+, 1, 476a9f983, -, 3),
      HEX_DBL(+, 1, ee9a798719e7f, -, 40) },
    { HEX_DBL(+, 1, c3f8f01c3f8f, -, 1), HEX_DBL(+, 1, 70742d4ef, -, 3),
      HEX_DBL(+, 1, 3ff1352c1219c, -, 46) },
    { HEX_DBL(+, 1, bdd2b899406f7, -, 1), HEX_DBL(+, 1, 98edd077e, -, 3),
      HEX_DBL(+, 1, c383cd11362f4, -, 41) },
    { HEX_DBL(+, 1, b7d6c3dda338b, -, 1), HEX_DBL(+, 1, c0db6cdd9, -, 3),
      HEX_DBL(+, 1, 37bd85b1a824e, -, 41) },
    { HEX_DBL(+, 1, b2036406c80d9, -, 1), HEX_DBL(+, 1, e840be74e, -, 3),
      HEX_DBL(+, 1, a9334d525e1ec, -, 41) },
    { HEX_DBL(+, 1, ac5701ac5701a, -, 1), HEX_DBL(+, 1, 0790adbb, -, 2),
      HEX_DBL(+, 1, 8060bfb6a491, -, 41) },
    { HEX_DBL(+, 1, a6d01a6d01a6d, -, 1), HEX_DBL(+, 1, 1ac05b2918, -, 2),
      HEX_DBL(+, 1, c1c161471580a, -, 40) },
    { HEX_DBL(+, 1, a16d3f97a4b01, -, 1), HEX_DBL(+, 1, 2db10fc4d8, -, 2),
      HEX_DBL(+, 1, ab1aa62214581, -, 42) },
    { HEX_DBL(+, 1, 9c2d14ee4a101, -, 1), HEX_DBL(+, 1, 406463b1b, -, 2),
      HEX_DBL(+, 1, 12e95dbda6611, -, 44) },
    { HEX_DBL(+, 1, 970e4f80cb872, -, 1), HEX_DBL(+, 1, 52dbdfc4c8, -, 2),
      HEX_DBL(+, 1, 6b53fee511af, -, 42) },
    { HEX_DBL(+, 1, 920fb49d0e228, -, 1), HEX_DBL(+, 1, 6518fe467, -, 2),
      HEX_DBL(+, 1, eea7d7d7d1764, -, 40) },
    { HEX_DBL(+, 1, 8d3018d3018d3, -, 1), HEX_DBL(+, 1, 771d2ba7e8, -, 2),
      HEX_DBL(+, 1, ecefa8d4fab97, -, 40) },
    { HEX_DBL(+, 1, 886e5f0abb049, -, 1), HEX_DBL(+, 1, 88e9c72e08, -, 2),
      HEX_DBL(+, 1, 913ea3d33fd14, -, 41) },
    { HEX_DBL(+, 1, 83c977ab2bedd, -, 1), HEX_DBL(+, 1, 9a802391e, -, 2),
      HEX_DBL(+, 1, 197e845877c94, -, 41) },
    { HEX_DBL(+, 1, 7f405fd017f4, -, 1), HEX_DBL(+, 1, abe18797f, -, 2),
      HEX_DBL(+, 1, f4a52f8e8a81, -, 42) },
    { HEX_DBL(+, 1, 7ad2208e0ecc3, -, 1), HEX_DBL(+, 1, bd0f2e9e78, -, 2),
      HEX_DBL(+, 1, 031f4336644cc, -, 42) },
    { HEX_DBL(+, 1, 767dce434a9b1, -, 1), HEX_DBL(+, 1, ce0a4923a, -, 2),
      HEX_DBL(+, 1, 61f33c897020c, -, 40) },
    { HEX_DBL(+, 1, 724287f46debc, -, 1), HEX_DBL(+, 1, ded3fd442, -, 2),
      HEX_DBL(+, 1, b2632e830632, -, 41) },
    { HEX_DBL(+, 1, 6e1f76b4337c6, -, 1), HEX_DBL(+, 1, ef6d673288, -, 2),
      HEX_DBL(+, 1, 888ec245a0bf, -, 40) },
    { HEX_DBL(+, 1, 6a13cd153729, -, 1), HEX_DBL(+, 1, ffd799a838, -, 2),
      HEX_DBL(+, 1, fe6f3b2f5fc8e, -, 40) },
    { HEX_DBL(+, 1, 661ec6a5122f9, -, 1), HEX_DBL(+, 1, 0809cf27f4, -, 1),
      HEX_DBL(+, 1, 81eaa9ef284dd, -, 40) },
    { HEX_DBL(+, 1, 623fa7701623f, -, 1), HEX_DBL(+, 1, 10113b153c, -, 1),
      HEX_DBL(+, 1, 1d7b07d6b1143, -, 42) },
    { HEX_DBL(+, 1, 5e75bb8d015e7, -, 1), HEX_DBL(+, 1, 18028cf728, -, 1),
      HEX_DBL(+, 1, 76b100b1f6c6, -, 41) },
    { HEX_DBL(+, 1, 5ac056b015ac, -, 1), HEX_DBL(+, 1, 1fde3d30e8, -, 1),
      HEX_DBL(+, 1, 26faeb9870945, -, 45) },
    { HEX_DBL(+, 1, 571ed3c506b39, -, 1), HEX_DBL(+, 1, 27a4c0585c, -, 1),
      HEX_DBL(+, 1, 7f2c5344d762b, -, 42) }
};

static double __loglTable2[64][3] = {
    { HEX_DBL(+, 1, 01fbe7f0a1be6, +, 0), HEX_DBL(-, 1, 6cf6ddd26112a, -, 7),
      HEX_DBL(+, 1, 0725e5755e314, -, 60) },
    { HEX_DBL(+, 1, 01eba93a97b12, +, 0), HEX_DBL(-, 1, 6155b1d99f603, -, 7),
      HEX_DBL(+, 1, 4bcea073117f4, -, 60) },
    { HEX_DBL(+, 1, 01db6c9029cd1, +, 0), HEX_DBL(-, 1, 55b54153137ff, -, 7),
      HEX_DBL(+, 1, 21e8faccad0ec, -, 61) },
    { HEX_DBL(+, 1, 01cb31f0f534c, +, 0), HEX_DBL(-, 1, 4a158c27245bd, -, 7),
      HEX_DBL(+, 1, 1a5b7bfbf35d3, -, 60) },
    { HEX_DBL(+, 1, 01baf95c9723c, +, 0), HEX_DBL(-, 1, 3e76923e3d678, -, 7),
      HEX_DBL(+, 1, eee400eb5fe34, -, 62) },
    { HEX_DBL(+, 1, 01aac2d2acee6, +, 0), HEX_DBL(-, 1, 32d85380ce776, -, 7),
      HEX_DBL(+, 1, cbf7a513937bd, -, 61) },
    { HEX_DBL(+, 1, 019a8e52d401e, +, 0), HEX_DBL(-, 1, 273acfd74be72, -, 7),
      HEX_DBL(+, 1, 5c64599efa5e6, -, 60) },
    { HEX_DBL(+, 1, 018a5bdca9e42, +, 0), HEX_DBL(-, 1, 1b9e072a2e65, -, 7),
      HEX_DBL(+, 1, 364180e0a5d37, -, 60) },
    { HEX_DBL(+, 1, 017a2b6fcc33e, +, 0), HEX_DBL(-, 1, 1001f961f3243, -, 7),
      HEX_DBL(+, 1, 63d795746f216, -, 60) },
    { HEX_DBL(+, 1, 0169fd0bd8a8a, +, 0), HEX_DBL(-, 1, 0466a6671bca4, -, 7),
      HEX_DBL(+, 1, 4c99ff1907435, -, 60) },
    { HEX_DBL(+, 1, 0159d0b06d129, +, 0), HEX_DBL(-, 1, f1981c445cd05, -, 8),
      HEX_DBL(+, 1, 4bfff6366b723, -, 62) },
    { HEX_DBL(+, 1, 0149a65d275a6, +, 0), HEX_DBL(-, 1, da6460f76ab8c, -, 8),
      HEX_DBL(+, 1, 9c5404f47589c, -, 61) },
    { HEX_DBL(+, 1, 01397e11a581b, +, 0), HEX_DBL(-, 1, c3321ab87f4ef, -, 8),
      HEX_DBL(+, 1, c0da537429cea, -, 61) },
    { HEX_DBL(+, 1, 012957cd85a28, +, 0), HEX_DBL(-, 1, ac014958c112c, -, 8),
      HEX_DBL(+, 1, 000c2a1b595e3, -, 64) },
    { HEX_DBL(+, 1, 0119339065ef7, +, 0), HEX_DBL(-, 1, 94d1eca95f67a, -, 8),
      HEX_DBL(+, 1, d8d20b0564d5, -, 61) },
    { HEX_DBL(+, 1, 01091159e4b3d, +, 0), HEX_DBL(-, 1, 7da4047b92b3e, -, 8),
      HEX_DBL(+, 1, 6194a5d68cf2, -, 66) },
    { HEX_DBL(+, 1, 00f8f129a0535, +, 0), HEX_DBL(-, 1, 667790a09bf77, -, 8),
      HEX_DBL(+, 1, ca230e0bea645, -, 61) },
    { HEX_DBL(+, 1, 00e8d2ff374a1, +, 0), HEX_DBL(-, 1, 4f4c90e9c4ead, -, 8),
      HEX_DBL(+, 1, 1de3e7f350c1, -, 61) },
    { HEX_DBL(+, 1, 00d8b6da482ce, +, 0), HEX_DBL(-, 1, 3823052860649, -, 8),
      HEX_DBL(+, 1, 5789b4c5891b8, -, 64) },
    { HEX_DBL(+, 1, 00c89cba71a8c, +, 0), HEX_DBL(-, 1, 20faed2dc9a9e, -, 8),
      HEX_DBL(+, 1, 9e7c40f9839fd, -, 62) },
    { HEX_DBL(+, 1, 00b8849f52834, +, 0), HEX_DBL(-, 1, 09d448cb65014, -, 8),
      HEX_DBL(+, 1, 387e3e9b6d02, -, 62) },
    { HEX_DBL(+, 1, 00a86e88899a4, +, 0), HEX_DBL(-, 1, e55e2fa53ebf1, -, 9),
      HEX_DBL(+, 1, cdaa71fddfddf, -, 62) },
    { HEX_DBL(+, 1, 00985a75b5e3f, +, 0), HEX_DBL(-, 1, b716b429dce0f, -, 9),
      HEX_DBL(+, 1, 2f2af081367bf, -, 63) },
    { HEX_DBL(+, 1, 00884866766ee, +, 0), HEX_DBL(-, 1, 88d21ec7a16d7, -, 9),
      HEX_DBL(+, 1, fb95c228d6f16, -, 62) },
    { HEX_DBL(+, 1, 0078385a6a61d, +, 0), HEX_DBL(-, 1, 5a906f219a9e8, -, 9),
      HEX_DBL(+, 1, 18aff10a89f29, -, 64) },
    { HEX_DBL(+, 1, 00682a5130fbe, +, 0), HEX_DBL(-, 1, 2c51a4dae87f1, -, 9),
      HEX_DBL(+, 1, bcc7e33ddde3, -, 63) },
    { HEX_DBL(+, 1, 00581e4a69944, +, 0), HEX_DBL(-, 1, fc2b7f2d782b1, -, 10),
      HEX_DBL(+, 1, fe3ef3300a9fa, -, 64) },
    { HEX_DBL(+, 1, 00481445b39a8, +, 0), HEX_DBL(-, 1, 9fb97df0b0b83, -, 10),
      HEX_DBL(+, 1, 0d9a601f2f324, -, 65) },
    { HEX_DBL(+, 1, 00380c42ae963, +, 0), HEX_DBL(-, 1, 434d4546227ae, -, 10),
      HEX_DBL(+, 1, 0b9b6a5868f33, -, 63) },
    { HEX_DBL(+, 1, 00280640fa271, +, 0), HEX_DBL(-, 1, cdcda8e930c19, -, 11),
      HEX_DBL(+, 1, 3d424ab39f789, -, 64) },
    { HEX_DBL(+, 1, 0018024036051, +, 0), HEX_DBL(-, 1, 150c558601261, -, 11),
      HEX_DBL(+, 1, 285bb90327a0f, -, 64) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, ffa011fca0a1e, -, 1), HEX_DBL(+, 1, 14e5640c4197b, -, 10),
      HEX_DBL(+, 1, 95728136ae401, -, 63) },
    { HEX_DBL(+, 1, ff6031f064e07, -, 1), HEX_DBL(+, 1, cd61806bf532d, -, 10),
      HEX_DBL(+, 1, 568a4f35d8538, -, 63) },
    { HEX_DBL(+, 1, ff2061d532b9c, -, 1), HEX_DBL(+, 1, 42e34af550eda, -, 9),
      HEX_DBL(+, 1, 8f69cee55fec, -, 62) },
    { HEX_DBL(+, 1, fee0a1a513253, -, 1), HEX_DBL(+, 1, 9f0a5523902ea, -, 9),
      HEX_DBL(+, 1, daec734b11615, -, 63) },
    { HEX_DBL(+, 1, fea0f15a12139, -, 1), HEX_DBL(+, 1, fb25e19f11b26, -, 9),
      HEX_DBL(+, 1, 8bafca62941da, -, 62) },
    { HEX_DBL(+, 1, fe6150ee3e6d4, -, 1), HEX_DBL(+, 1, 2b9af9a28e282, -, 8),
      HEX_DBL(+, 1, 0fd3674e1dc5b, -, 61) },
    { HEX_DBL(+, 1, fe21c05baa109, -, 1), HEX_DBL(+, 1, 599d4678f24b9, -, 8),
      HEX_DBL(+, 1, dafce1f09937b, -, 61) },
    { HEX_DBL(+, 1, fde23f9c69cf9, -, 1), HEX_DBL(+, 1, 8799d8c046eb, -, 8),
      HEX_DBL(+, 1, ffa0ce0bdd217, -, 65) },
    { HEX_DBL(+, 1, fda2ceaa956e8, -, 1), HEX_DBL(+, 1, b590b1e5951ee, -, 8),
      HEX_DBL(+, 1, 645a769232446, -, 62) },
    { HEX_DBL(+, 1, fd636d8047a1f, -, 1), HEX_DBL(+, 1, e381d3555dbcf, -, 8),
      HEX_DBL(+, 1, 882320d368331, -, 61) },
    { HEX_DBL(+, 1, fd241c179e0cc, -, 1), HEX_DBL(+, 1, 08b69f3dccde, -, 7),
      HEX_DBL(+, 1, 01ad5065aba9e, -, 61) },
    { HEX_DBL(+, 1, fce4da6ab93e8, -, 1), HEX_DBL(+, 1, 1fa97a61dd298, -, 7),
      HEX_DBL(+, 1, 84cd1f931ae34, -, 60) },
    { HEX_DBL(+, 1, fca5a873bcb19, -, 1), HEX_DBL(+, 1, 36997bcc54a3f, -, 7),
      HEX_DBL(+, 1, 1485e97eaee03, -, 60) },
    { HEX_DBL(+, 1, fc66862ccec93, -, 1), HEX_DBL(+, 1, 4d86a43264a4f, -, 7),
      HEX_DBL(+, 1, c75e63370988b, -, 61) },
    { HEX_DBL(+, 1, fc27739018cfe, -, 1), HEX_DBL(+, 1, 6470f448fb09d, -, 7),
      HEX_DBL(+, 1, d7361eeaed0a1, -, 65) },
    { HEX_DBL(+, 1, fbe87097c6f5a, -, 1), HEX_DBL(+, 1, 7b586cc4c2523, -, 7),
      HEX_DBL(+, 1, b3df952cc473c, -, 61) },
    { HEX_DBL(+, 1, fba97d3e084dd, -, 1), HEX_DBL(+, 1, 923d0e5a21e06, -, 7),
      HEX_DBL(+, 1, cf56c7b64ae5d, -, 62) },
    { HEX_DBL(+, 1, fb6a997d0ecdc, -, 1), HEX_DBL(+, 1, a91ed9bd3df9a, -, 7),
      HEX_DBL(+, 1, b957bdcd89e43, -, 61) },
    { HEX_DBL(+, 1, fb2bc54f0f4ab, -, 1), HEX_DBL(+, 1, bffdcfa1f7fbb, -, 7),
      HEX_DBL(+, 1, ea8cad9a21771, -, 62) },
    { HEX_DBL(+, 1, faed00ae41783, -, 1), HEX_DBL(+, 1, d6d9f0bbee6f6, -, 7),
      HEX_DBL(+, 1, 5762a9af89c82, -, 60) },
    { HEX_DBL(+, 1, faae4b94dfe64, -, 1), HEX_DBL(+, 1, edb33dbe7d335, -, 7),
      HEX_DBL(+, 1, 21e24fc245697, -, 62) },
    { HEX_DBL(+, 1, fa6fa5fd27ff8, -, 1), HEX_DBL(+, 1, 0244dbae5ed05, -, 6),
      HEX_DBL(+, 1, 12ef51b967102, -, 60) },
    { HEX_DBL(+, 1, fa310fe15a078, -, 1), HEX_DBL(+, 1, 0daeaf24c3529, -, 6),
      HEX_DBL(+, 1, 10d3cfca60b45, -, 59) },
    { HEX_DBL(+, 1, f9f2893bb9192, -, 1), HEX_DBL(+, 1, 1917199bb66bc, -, 6),
      HEX_DBL(+, 1, 6cf6034c32e19, -, 60) },
    { HEX_DBL(+, 1, f9b412068b247, -, 1), HEX_DBL(+, 1, 247e1b6c615d5, -, 6),
      HEX_DBL(+, 1, 42f0fffa229f7, -, 61) },
    { HEX_DBL(+, 1, f975aa3c18ed6, -, 1), HEX_DBL(+, 1, 2fe3b4efcc5ad, -, 6),
      HEX_DBL(+, 1, 70106136a8919, -, 60) },
    { HEX_DBL(+, 1, f93751d6ae09b, -, 1), HEX_DBL(+, 1, 3b47e67edea93, -, 6),
      HEX_DBL(+, 1, 38dd5a4f6959a, -, 59) },
    { HEX_DBL(+, 1, f8f908d098df6, -, 1), HEX_DBL(+, 1, 46aab0725ea6c, -, 6),
      HEX_DBL(+, 1, 821fc1e799e01, -, 60) },
    { HEX_DBL(+, 1, f8bacf242aa2c, -, 1), HEX_DBL(+, 1, 520c1322f1e4e, -, 6),
      HEX_DBL(+, 1, 129dcda3ad563, -, 60) },
    { HEX_DBL(+, 1, f87ca4cbb755, -, 1), HEX_DBL(+, 1, 5d6c0ee91d2ab, -, 6),
      HEX_DBL(+, 1, c5b190c04606e, -, 62) },
    { HEX_DBL(+, 1, f83e89c195c25, -, 1), HEX_DBL(+, 1, 68caa41d448c3, -, 6),
      HEX_DBL(+, 1, 4723441195ac9, -, 59) }
};

static double __loglTable3[8][3] = {
    { HEX_DBL(+, 1, 000e00c40ab89, +, 0), HEX_DBL(-, 1, 4332be0032168, -, 12),
      HEX_DBL(+, 1, a1003588d217a, -, 65) },
    { HEX_DBL(+, 1, 000a006403e82, +, 0), HEX_DBL(-, 1, cdb2987366fcc, -, 13),
      HEX_DBL(+, 1, 5c86001294bbc, -, 67) },
    { HEX_DBL(+, 1, 0006002400d8, +, 0), HEX_DBL(-, 1, 150297c90fa6f, -, 13),
      HEX_DBL(+, 1, 01fb4865fae32, -, 66) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, 0, +, 0), HEX_DBL(+, 0, 0, +, 0), HEX_DBL(+, 0, 0, +, 0) },
    { HEX_DBL(+, 1, ffe8011ff280a, -, 1), HEX_DBL(+, 1, 14f8daf5e3d3b, -, 12),
      HEX_DBL(+, 1, 3c933b4b6b914, -, 68) },
    { HEX_DBL(+, 1, ffd8031fc184e, -, 1), HEX_DBL(+, 1, cd978c38042bb, -, 12),
      HEX_DBL(+, 1, 10f8e642e66fd, -, 65) },
    { HEX_DBL(+, 1, ffc8061f5492b, -, 1), HEX_DBL(+, 1, 43183c878274e, -, 11),
      HEX_DBL(+, 1, 5885dd1eb6582, -, 65) }
};

static void __log2_ep(double *hi, double *lo, double x)
{
    union {
        uint64_t i;
        double d;
    } uu;

    int m;
    double f = reference_frexp(x, &m);

    // bring f in [0.75, 1.5)
    if (f < 0.75)
    {
        f *= 2.0;
        m -= 1;
    }

    // index first table .... brings down to [1-2^-7, 1+2^6)
    uu.d = f;
    int index =
        (int)(((uu.i + ((uint64_t)1 << 51)) & 0x000fc00000000000ULL) >> 46);
    double r1 = __loglTable1[index][0];
    double logr1hi = __loglTable1[index][1];
    double logr1lo = __loglTable1[index][2];
    // since log1rhi has 39 bits of precision, we have 14 bit in hand ... since
    // |m| <= 1023 which needs 10bits at max, we can directly add m to log1hi
    // without spilling
    logr1hi += m;

    // argument reduction needs to be in double-double since reduced argument
    // will form the leading term of polynomial approximation which sets the
    // precision we eventually achieve
    double zhi, zlo;
    MulD(&zhi, &zlo, r1, uu.d);

    // second index table .... brings down to [1-2^-12, 1+2^-11)
    uu.d = zhi;
    index = (int)(((uu.i + ((uint64_t)1 << 46)) & 0x00007e0000000000ULL) >> 41);
    double r2 = __loglTable2[index][0];
    double logr2hi = __loglTable2[index][1];
    double logr2lo = __loglTable2[index][2];

    // reduce argument
    MulDD(&zhi, &zlo, zhi, zlo, r2, 0.0);

    // third index table .... brings down to [1-2^-14, 1+2^-13)
    // Actually reduction to 2^-11 would have been sufficient to calculate
    // second order term in polynomial in double rather than double-double, I
    // reduced it a bit more to make sure other systematic arithmetic errors
    // are guarded against .... also this allow lower order product of leading
    // polynomial term i.e. Ao_hi*z_lo + Ao_lo*z_hi to be done in double rather
    // than double-double ... hence only term that needs to be done in
    // double-double is Ao_hi*z_hi
    uu.d = zhi;
    index = (int)(((uu.i + ((uint64_t)1 << 41)) & 0x0000038000000000ULL) >> 39);
    double r3 = __loglTable3[index][0];
    double logr3hi = __loglTable3[index][1];
    double logr3lo = __loglTable3[index][2];

    // log2(x) = m + log2(r1) + log2(r1) + log2(1 + (zh + zlo))
    // calculate sum of first three terms ... note that m has already
    // been added to log2(r1)_hi
    double log2hi, log2lo;
    AddDD(&log2hi, &log2lo, logr1hi, logr1lo, logr2hi, logr2lo);
    AddDD(&log2hi, &log2lo, logr3hi, logr3lo, log2hi, log2lo);

    // final argument reduction .... zhi will be in [1-2^-14, 1+2^-13) after
    // this
    MulDD(&zhi, &zlo, zhi, zlo, r3, 0.0);
    // we dont need to do full double-double substract here. substracting 1.0
    // for higher term is exact
    zhi = zhi - 1.0;
    // normalize
    AddD(&zhi, &zlo, zhi, zlo);

    // polynomail fitting to compute log2(1 + z) ... forth order polynomial fit
    // to log2(1 + z)/z gives minimax absolute error of O(2^-76) with z in
    // [-2^-14, 2^-13] log2(1 + z)/z = Ao + A1*z + A2*z^2 + A3*z^3 + A4*z^4
    // => log2(1 + z) = Ao*z + A1*z^2 + A2*z^3 + A3*z^4 + A4*z^5
    // => log2(1 + z) = (Aohi + Aolo)*(zhi + zlo) + z^2*(A1 + A2*z + A3*z^2 +
    // A4*z^3) since we are looking for at least 64 digits of precision and z in
    // [-2^-14, 2^-13], final term can be done in double .... also Aolo*zhi +
    // Aohi*zlo can be done in double .... Aohi*zhi needs to be done in
    // double-double

    double Aohi = HEX_DBL(+, 1, 71547652b82fe, +, 0);
    double Aolo = HEX_DBL(+, 1, 777c9cbb675c, -, 56);
    double y;
    y = HEX_DBL(+, 1, 276d2736fade7, -, 2);
    y = HEX_DBL(-, 1, 7154765782df1, -, 2) + y * zhi;
    y = HEX_DBL(+, 1, ec709dc3a0f67, -, 2) + y * zhi;
    y = HEX_DBL(-, 1, 71547652b82fe, -, 1) + y * zhi;
    double zhisq = zhi * zhi;
    y = y * zhisq;
    y = y + zhi * Aolo;
    y = y + zlo * Aohi;

    MulD(&zhi, &zlo, Aohi, zhi);
    AddDD(&zhi, &zlo, zhi, zlo, y, 0.0);
    AddDD(&zhi, &zlo, zhi, zlo, log2hi, log2lo);

    *hi = zhi;
    *lo = zlo;
}

long double reference_powl(long double x, long double y)
{
    // this will be used for testing doubles i.e. arguments will
    // be doubles so cast the input back to double ... returned
    // result will be long double though .... > 53 bits of precision
    // if platform allows.
    // ===========
    // New finding.
    // ===========
    // this function is getting used for computing reference cube root (cbrt)
    // as follows __powl( x, 1.0L/3.0L ) so if the y are assumed to
    // be double and is converted from long double to double, truncation
    // causes errors. So we need to tread y as long double and convert it
    // to hi, lo doubles when performing y*log2(x).

    static const double neg_epsilon = HEX_DBL(+, 1, 0, +, 53);

    // if x = 1, return x for any y, even NaN
    if (x == 1.0) return x;

    // if y == 0, return 1 for any x, even NaN
    if (y == 0.0) return 1.0L;

    // get NaNs out of the way
    if (x != x || y != y) return x + y;

    // do the work required to sort out edge cases
    double fabsy = (double)reference_fabsl(y);
    double fabsx = (double)reference_fabsl(x);
    double iy = reference_rint(
        fabsy); // we do round to nearest here so that |fy| <= 0.5
    if (iy > fabsy) // convert nearbyint to floor
        iy -= 1.0;
    int isOddInt = 0;
    if (fabsy == iy && !reference_isinf(fabsy) && iy < neg_epsilon)
        isOddInt = (int)(iy - 2.0 * rint(0.5 * iy)); // might be 0, -1, or 1

    /// test a few more edge cases
    // deal with x == 0 cases
    if (x == 0.0)
    {
        if (!isOddInt) x = 0.0;

        if (y < 0) x = 1.0 / x;

        return x;
    }

    // x == +-Inf cases
    if (isinf(fabsx))
    {
        if (x < 0)
        {
            if (isOddInt)
            {
                if (y < 0)
                    return -0.0;
                else
                    return -INFINITY;
            }
            else
            {
                if (y < 0)
                    return 0.0;
                else
                    return INFINITY;
            }
        }

        if (y < 0) return 0;
        return INFINITY;
    }

    // y = +-inf cases
    if (isinf(fabsy))
    {
        if (x == -1) return 1;

        if (y < 0)
        {
            if (fabsx < 1) return INFINITY;
            return 0;
        }
        if (fabsx < 1) return 0;
        return INFINITY;
    }

    // x < 0 and y non integer case
    if (x < 0 && iy != fabsy)
    {
        // return nan;
        return cl_make_nan();
    }

    // speedy resolution of sqrt and reciprocal sqrt
    if (fabsy == 0.5)
    {
        long double xl = sqrtl(x);
        if (y < 0) xl = 1.0 / xl;
        return xl;
    }

    double log2x_hi, log2x_lo;

    // extended precision log .... accurate to at least 64-bits + couple of
    // guard bits
    __log2_ep(&log2x_hi, &log2x_lo, fabsx);

    double ylog2x_hi, ylog2x_lo;

    double y_hi = (double)y;
    double y_lo = (double)(y - (long double)y_hi);

    // compute product of y*log2(x)
    // scale to avoid overflow in double-double multiplication
    if (fabsy > HEX_DBL(+, 1, 0, +, 970))
    {
        y_hi = reference_ldexp(y_hi, -53);
        y_lo = reference_ldexp(y_lo, -53);
    }
    MulDD(&ylog2x_hi, &ylog2x_lo, log2x_hi, log2x_lo, y_hi, y_lo);
    if (fabsy > HEX_DBL(+, 1, 0, +, 970))
    {
        ylog2x_hi = reference_ldexp(ylog2x_hi, 53);
        ylog2x_lo = reference_ldexp(ylog2x_lo, 53);
    }

    long double powxy;
    if (isinf(ylog2x_hi) || (reference_fabs(ylog2x_hi) > 2200))
    {
        powxy =
            reference_signbit(ylog2x_hi) ? HEX_DBL(+, 0, 0, +, 0) : INFINITY;
    }
    else
    {
        // separate integer + fractional part
        long int m = lrint(ylog2x_hi);
        AddDD(&ylog2x_hi, &ylog2x_lo, ylog2x_hi, ylog2x_lo, -m, 0.0);

        // revert to long double arithemtic
        long double ylog2x = (long double)ylog2x_hi + (long double)ylog2x_lo;
        long double tmp = reference_exp2l(ylog2x);
        powxy = reference_scalblnl(tmp, m);
    }

    // if y is odd integer and x is negative, reverse sign
    if (isOddInt & reference_signbit(x)) powxy = -powxy;
    return powxy;
}

double reference_nextafter(double xx, double yy)
{
    float x = (float)xx;
    float y = (float)yy;

    // take care of nans
    if (x != x) return x;

    if (y != y) return y;

    if (x == y) return y;

    int32f_t a, b;

    a.f = x;
    b.f = y;

    if (a.i & 0x80000000) a.i = 0x80000000 - a.i;
    if (b.i & 0x80000000) b.i = 0x80000000 - b.i;

    a.i += (a.i < b.i) ? 1 : -1;
    a.i = (a.i < 0) ? (cl_int)0x80000000 - a.i : a.i;

    return a.f;
}


long double reference_nextafterl(long double xx, long double yy)
{
    double x = (double)xx;
    double y = (double)yy;

    // take care of nans
    if (x != x) return x;

    if (y != y) return y;

    int64d_t a, b;

    a.d = x;
    b.d = y;

    int64_t tmp = 0x8000000000000000LL;

    if (a.l & tmp) a.l = tmp - a.l;
    if (b.l & tmp) b.l = tmp - b.l;

    // edge case. if (x == y) or (x = 0.0f and y = -0.0f) or (x = -0.0f and y =
    // 0.0f) test needs to be done using integer rep because subnormals may be
    // flushed to zero on some platforms
    if (a.l == b.l) return y;

    a.l += (a.l < b.l) ? 1 : -1;
    a.l = (a.l < 0) ? tmp - a.l : a.l;

    return a.d;
}

double reference_fdim(double xx, double yy)
{
    float x = (float)xx;
    float y = (float)yy;

    if (x != x) return x;

    if (y != y) return y;

    float r = (x > y) ? (float)reference_subtract(x, y) : 0.0f;
    return r;
}


long double reference_fdiml(long double xx, long double yy)
{
    double x = (double)xx;
    double y = (double)yy;

    if (x != x) return x;

    if (y != y) return y;

    double r = (x > y) ? (double)reference_subtractl(x, y) : 0.0;
    return r;
}

double reference_remquo(double xd, double yd, int *n)
{
    float xx = (float)xd;
    float yy = (float)yd;

    if (isnan(xx) || isnan(yy) || fabsf(xx) == INFINITY || yy == 0.0)
    {
        *n = 0;
        return cl_make_nan();
    }

    if (fabsf(yy) == INFINITY || xx == 0.0f)
    {
        *n = 0;
        return xd;
    }

    if (fabsf(xx) == fabsf(yy))
    {
        *n = (xx == yy) ? 1 : -1;
        return reference_signbit(xx) ? -0.0 : 0.0;
    }

    int signx = reference_signbit(xx) ? -1 : 1;
    int signy = reference_signbit(yy) ? -1 : 1;
    int signn = (signx == signy) ? 1 : -1;
    float x = fabsf(xx);
    float y = fabsf(yy);

    int ex, ey;
    ex = reference_ilogb(x);
    ey = reference_ilogb(y);
    float xr = x;
    float yr = y;
    uint32_t q = 0;

    if (ex - ey >= -1)
    {

        yr = (float)reference_ldexp(y, -ey);
        xr = (float)reference_ldexp(x, -ex);

        if (ex - ey >= 0)
        {
            int i;
            for (i = ex - ey; i > 0; i--)
            {
                q <<= 1;
                if (xr >= yr)
                {
                    xr -= yr;
                    q += 1;
                }
                xr += xr;
            }
            q <<= 1;
            if (xr > yr)
            {
                xr -= yr;
                q += 1;
            }
        }
        else // ex-ey = -1
            xr = reference_ldexp(xr, ex - ey);
    }

    if ((yr < 2.0f * xr) || ((yr == 2.0f * xr) && (q & 0x00000001)))
    {
        xr -= yr;
        q += 1;
    }

    if (ex - ey >= -1) xr = reference_ldexp(xr, ey);

    int qout = q & 0x0000007f;
    if (signn < 0) qout = -qout;
    if (xx < 0.0) xr = -xr;

    *n = qout;

    return xr;
}

long double reference_remquol(long double xd, long double yd, int *n)
{
    double xx = (double)xd;
    double yy = (double)yd;

    if (isnan(xx) || isnan(yy) || fabs(xx) == INFINITY || yy == 0.0)
    {
        *n = 0;
        return cl_make_nan();
    }

    if (reference_fabs(yy) == INFINITY || xx == 0.0)
    {
        *n = 0;
        return xd;
    }

    if (reference_fabs(xx) == reference_fabs(yy))
    {
        *n = (xx == yy) ? 1 : -1;
        return reference_signbit(xx) ? -0.0 : 0.0;
    }

    int signx = reference_signbit(xx) ? -1 : 1;
    int signy = reference_signbit(yy) ? -1 : 1;
    int signn = (signx == signy) ? 1 : -1;
    double x = reference_fabs(xx);
    double y = reference_fabs(yy);

    int ex, ey;
    ex = reference_ilogbl(x);
    ey = reference_ilogbl(y);
    double xr = x;
    double yr = y;
    uint32_t q = 0;

    if (ex - ey >= -1)
    {
        yr = reference_ldexp(y, -ey);
        xr = reference_ldexp(x, -ex);
        int i;

        if (ex - ey >= 0)
        {
            for (i = ex - ey; i > 0; i--)
            {
                q <<= 1;
                if (xr >= yr)
                {
                    xr -= yr;
                    q += 1;
                }
                xr += xr;
            }
            q <<= 1;
            if (xr > yr)
            {
                xr -= yr;
                q += 1;
            }
        }
        else
            xr = reference_ldexp(xr, ex - ey);
    }

    if ((yr < 2.0 * xr) || ((yr == 2.0 * xr) && (q & 0x00000001)))
    {
        xr -= yr;
        q += 1;
    }

    if (ex - ey >= -1) xr = reference_ldexp(xr, ey);

    int qout = q & 0x0000007f;
    if (signn < 0) qout = -qout;
    if (xx < 0.0) xr = -xr;

    *n = qout;
    return xr;
}

static double reference_scalbn(double x, int n)
{
    if (reference_isinf(x) || reference_isnan(x) || x == 0.0) return x;

    int bias = 1023;
    union {
        double d;
        cl_long l;
    } u;
    u.d = (double)x;
    int e = (int)((u.l & 0x7ff0000000000000LL) >> 52);
    if (e == 0)
    {
        u.l |= ((cl_long)1023 << 52);
        u.d -= 1.0;
        e = (int)((u.l & 0x7ff0000000000000LL) >> 52) - 1022;
    }
    e += n;
    if (e >= 2047 || n >= 2098) return reference_copysign(INFINITY, x);
    if (e < -51 || n < -2097) return reference_copysign(0.0, x);
    if (e <= 0)
    {
        bias += (e - 1);
        e = 1;
    }
    u.l &= 0x800fffffffffffffLL;
    u.l |= ((cl_long)e << 52);
    x = u.d;
    u.l = ((cl_long)bias << 52);
    return x * u.d;
}

static long double reference_scalblnl(long double x, long n)
{
#if defined(__i386__) || defined(__x86_64__) // INTEL
    union {
        long double d;
        struct
        {
            cl_ulong m;
            cl_ushort sexp;
        } u;
    } u;
    u.u.m = CL_LONG_MIN;

    if (reference_isinf(x)) return x;

    if (x == 0.0L || n < -2200) return reference_copysignl(0.0L, x);

    if (n > 2200) return reference_copysignl(INFINITY, x);

    if (n < 0)
    {
        u.u.sexp = 0x3fff - 1022;
        while (n <= -1022)
        {
            x *= u.d;
            n += 1022;
        }
        u.u.sexp = 0x3fff + n;
        x *= u.d;
        return x;
    }

    if (n > 0)
    {
        u.u.sexp = 0x3fff + 1023;
        while (n >= 1023)
        {
            x *= u.d;
            n -= 1023;
        }
        u.u.sexp = 0x3fff + n;
        x *= u.d;
        return x;
    }

    return x;

#elif defined(__arm__) // ARM .. sizeof(long double) == sizeof(double)

#if __DBL_MAX_EXP__ >= __LDBL_MAX_EXP__
    if (reference_isinfl(x) || reference_isnanl(x)) return x;

    int bias = 1023;
    union {
        double d;
        cl_long l;
    } u;
    u.d = (double)x;
    int e = (int)((u.l & 0x7ff0000000000000LL) >> 52);
    if (e == 0)
    {
        u.l |= ((cl_long)1023 << 52);
        u.d -= 1.0;
        e = (int)((u.l & 0x7ff0000000000000LL) >> 52) - 1022;
    }
    e += n;
    if (e >= 2047) return reference_copysignl(INFINITY, x);
    if (e < -51) return reference_copysignl(0.0, x);
    if (e <= 0)
    {
        bias += (e - 1);
        e = 1;
    }
    u.l &= 0x800fffffffffffffLL;
    u.l |= ((cl_long)e << 52);
    x = u.d;
    u.l = ((cl_long)bias << 52);
    return x * u.d;
#endif

#else // PPC
    return scalblnl(x, n);
#endif
}

double reference_relaxed_exp(double x) { return reference_exp(x); }

double reference_exp(double x)
{
    return reference_exp2(x * HEX_DBL(+, 1, 71547652b82fe, +, 0));
}

long double reference_expl(long double x)
{
#if defined(__PPC__)
    long double scale, bias;

    // The PPC double long version of expl fails to produce denorm results
    // and instead generates a 0.0. Compensate for this limitation by
    // computing expl as:
    //     expl(x + 40) * expl(-40)
    // Likewise, overflows can prematurely produce an infinity, so we
    // compute expl as:
    //     expl(x - 40) * expl(40)
    scale = 1.0L;
    bias = 0.0L;
    if (x < -708.0L)
    {
        bias = 40.0;
        scale = expl(-40.0L);
    }
    else if (x > 708.0L)
    {
        bias = -40.0L;
        scale = expl(40.0L);
    }
    return expl(x + bias) * scale;
#else
    return expl(x);
#endif
}

double reference_sinh(double x) { return sinh(x); }

long double reference_sinhl(long double x) { return sinhl(x); }

double reference_fmod(double x, double y)
{
    if (x == 0.0 && fabs(y) > 0.0) return x;

    if (fabs(x) == INFINITY || y == 0) return cl_make_nan();

    if (fabs(y) == INFINITY) // we know x is finite from above
        return x;
#if defined(_MSC_VER) && defined(_M_X64)
    return fmod(x, y);
#else
    return fmodf((float)x, (float)y);
#endif
}

long double reference_fmodl(long double x, long double y)
{
    if (x == 0.0L && fabsl(y) > 0.0L) return x;

    if (fabsl(x) == INFINITY || y == 0.0L) return cl_make_nan();

    if (fabsl(y) == INFINITY) // we know x is finite from above
        return x;

    return fmod((double)x, (double)y);
}

double reference_modf(double x, double *n)
{
    if (isnan(x))
    {
        *n = cl_make_nan();
        return cl_make_nan();
    }
    float nr;
    float yr = modff((float)x, &nr);
    *n = nr;
    return yr;
}

long double reference_modfl(long double x, long double *n)
{
    if (isnan(x))
    {
        *n = cl_make_nan();
        return cl_make_nan();
    }
    double nr;
    double yr = modf((double)x, &nr);
    *n = nr;
    return yr;
}

long double reference_fractl(long double x, long double *ip)
{
    if (isnan(x))
    {
        *ip = cl_make_nan();
        return cl_make_nan();
    }

    double i;
    double f = modf((double)x, &i);
    if (f < 0.0)
    {
        f = 1.0 + f;
        i -= 1.0;
        if (f == 1.0) f = HEX_DBL(+, 1, fffffffffffff, -, 1);
    }
    *ip = i;
    return f;
}

long double reference_fabsl(long double x) { return fabsl(x); }

double reference_relaxed_log(double x)
{
    return (float)reference_log((float)x);
}

double reference_log(double x)
{
    if (x == 0.0) return -INFINITY;

    if (x < 0.0) return cl_make_nan();

    if (isinf(x)) return INFINITY;

    double log2Hi = HEX_DBL(+, 1, 62e42fefa39ef, -, 1);
    double logxHi, logxLo;
    __log2_ep(&logxHi, &logxLo, x);
    return logxHi * log2Hi;
}

long double reference_logl(long double x)
{
    if (x == 0.0) return -INFINITY;

    if (x < 0.0) return cl_make_nan();

    if (isinf(x)) return INFINITY;

    double log2Hi = HEX_DBL(+, 1, 62e42fefa39ef, -, 1);
    double log2Lo = HEX_DBL(+, 1, abc9e3b39803f, -, 56);
    double logxHi, logxLo;
    __log2_ep(&logxHi, &logxLo, x);

    long double lg2 = (long double)log2Hi + (long double)log2Lo;
    long double logx = (long double)logxHi + (long double)logxLo;
    return logx * lg2;
}

double reference_relaxed_pow(double x, double y)
{
    return (float)reference_exp2(((float)y) * (float)reference_log2((float)x));
}

double reference_pow(double x, double y)
{
    static const double neg_epsilon = HEX_DBL(+, 1, 0, +, 53);

    // if x = 1, return x for any y, even NaN
    if (x == 1.0) return x;

    // if y == 0, return 1 for any x, even NaN
    if (y == 0.0) return 1.0;

    // get NaNs out of the way
    if (x != x || y != y) return x + y;

    // do the work required to sort out edge cases
    double fabsy = reference_fabs(y);
    double fabsx = reference_fabs(x);
    double iy = reference_rint(
        fabsy); // we do round to nearest here so that |fy| <= 0.5
    if (iy > fabsy) // convert nearbyint to floor
        iy -= 1.0;
    int isOddInt = 0;
    if (fabsy == iy && !reference_isinf(fabsy) && iy < neg_epsilon)
        isOddInt = (int)(iy - 2.0 * rint(0.5 * iy)); // might be 0, -1, or 1

    /// test a few more edge cases
    // deal with x == 0 cases
    if (x == 0.0)
    {
        if (!isOddInt) x = 0.0;

        if (y < 0) x = 1.0 / x;

        return x;
    }

    // x == +-Inf cases
    if (isinf(fabsx))
    {
        if (x < 0)
        {
            if (isOddInt)
            {
                if (y < 0)
                    return -0.0;
                else
                    return -INFINITY;
            }
            else
            {
                if (y < 0)
                    return 0.0;
                else
                    return INFINITY;
            }
        }

        if (y < 0) return 0;
        return INFINITY;
    }

    // y = +-inf cases
    if (isinf(fabsy))
    {
        if (x == -1) return 1;

        if (y < 0)
        {
            if (fabsx < 1) return INFINITY;
            return 0;
        }
        if (fabsx < 1) return 0;
        return INFINITY;
    }

    // x < 0 and y non integer case
    if (x < 0 && iy != fabsy)
    {
        // return nan;
        return cl_make_nan();
    }

    // speedy resolution of sqrt and reciprocal sqrt
    if (fabsy == 0.5)
    {
        long double xl = reference_sqrt(x);
        if (y < 0) xl = 1.0 / xl;
        return xl;
    }

    double hi, lo;
    __log2_ep(&hi, &lo, fabsx);
    double prod = y * hi;
    double result = reference_exp2(prod);
    return isOddInt ? reference_copysignd(result, x) : result;
}

double reference_sqrt(double x) { return sqrt(x); }

double reference_floor(double x) { return floorf((float)x); }

double reference_ldexp(double value, int exponent)
{
#ifdef __MINGW32__
    /*
     * ====================================================
     * This function is from fdlibm: http://www.netlib.org
     *   It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
     *
     * Developed at SunSoft, a Sun Microsystems, Inc. business.
     * Permission to use, copy, modify, and distribute this
     * software is freely granted, provided that this notice
     * is preserved.
     * ====================================================
     */
    if (!finite(value) || value == 0.0) return value;
    return scalbn(value, exponent);
#else
    return reference_scalbn(value, exponent);
#endif
}

long double reference_ldexpl(long double x, int n) { return ldexpl(x, n); }

long double reference_coshl(long double x) { return coshl(x); }

double reference_ceil(double x) { return ceilf((float)x); }

long double reference_ceill(long double x)
{
    if (x == 0.0 || reference_isinfl(x) || reference_isnanl(x)) return x;

    long double absx = reference_fabsl(x);
    if (absx >= HEX_LDBL(+, 1, 0, +, 52)) return x;

    if (absx < 1.0)
    {
        if (x < 0.0)
            return 0.0;
        else
            return 1.0;
    }

    long double r = (long double)((cl_long)x);

    if (x > 0.0 && r < x) r += 1.0;

    return r;
}


long double reference_acosl(long double x)
{
    long double x2 = x * x;
    int i;

    // Prepare a head + tail representation of PI in long double.  A good
    // compiler should get rid of all of this work.
    static const cl_ulong pi_bits[2] = {
        0x3243F6A8885A308DULL, 0x313198A2E0370734ULL
    }; // first 126 bits of pi
       // http://www.super-computing.org/pi-hexa_current.html
    long double head, tail, temp;
#if __LDBL_MANT_DIG__ >= 64
    // long double has 64-bits of precision or greater
    temp = (long double)pi_bits[0] * 0x1.0p64L;
    head = temp + (long double)pi_bits[1];
    temp -= head; // rounding err rounding pi_bits[1] into head
    tail = (long double)pi_bits[1] + temp;
    head *= HEX_LDBL(+, 1, 0, -, 125);
    tail *= HEX_LDBL(+, 1, 0, -, 125);
#else
    head = (long double)pi_bits[0];
    tail =
        (long double)((cl_long)pi_bits[0]
                      - (cl_long)
                          head); // residual part of pi_bits[0] after rounding
    tail = tail * HEX_LDBL(+, 1, 0, +, 64) + (long double)pi_bits[1];
    head *= HEX_LDBL(+, 1, 0, -, 61);
    tail *= HEX_LDBL(+, 1, 0, -, 125);
#endif

    // oversize values and NaNs go to NaN
    if (!(x2 <= 1.0)) return sqrtl(1.0L - x2);

    //
    // deal with large |x|:
    //                                                      sqrt( 1 - x**2)
    // acos(|x| > sqrt(0.5)) = 2 * atan( z );       z = -------------------- ;
    // z in [0, sqrt(0.5)/(1+sqrt(0.5) = .4142135...]
    //                                                          1 + x
    if (x2 > 0.5)
    {
        // we handle the x < 0 case as pi - acos(|x|)

        long double sign = reference_copysignl(1.0L, x);
        long double fabsx = reference_fabsl(x);
        head -= head * sign; // x > 0 ? 0 : pi.hi
        tail -= tail * sign; // x > 0 ? 0 : pi.low

        // z = sqrt( 1-x**2 ) / (1+x) = sqrt( (1-x)(1+x) / (1+x)**2 ) = sqrt(
        // (1-x)/(1+x) )
        long double z2 = (1.0L - fabsx) / (1.0L + fabsx); // z**2
        long double z = sign * sqrtl(z2);

        //                     atan(sqrt(q))
        // Minimax fit p(x) = ---------------- - 1
        //                        sqrt(q)
        //
        // Define q = r*r, and solve for atan(r):
        //
        //  atan(r) = (p(r) + 1) * r = rp(r) + r
        static long double atan_coeffs[] = {
            HEX_LDBL(-, b, 3f52e0c278293b3, -, 67),
            HEX_LDBL(-, a, aaaaaaaaaaa95b8, -, 5),
            HEX_LDBL(+, c, ccccccccc992407, -, 6),
            HEX_LDBL(-, 9, 24924923024398, -, 6),
            HEX_LDBL(+, e, 38e38d6f92c98f3, -, 7),
            HEX_LDBL(-, b, a2e89bfb8393ec6, -, 7),
            HEX_LDBL(+, 9, d89a9f574d412cb, -, 7),
            HEX_LDBL(-, 8, 88580517884c547, -, 7),
            HEX_LDBL(+, f, 0ab6756abdad408, -, 8),
            HEX_LDBL(-, d, 56a5b07a2f15b49, -, 8),
            HEX_LDBL(+, b, 72ab587e46d80b2, -, 8),
            HEX_LDBL(-, 8, 62ea24bb5b2e636, -, 8),
            HEX_LDBL(+, e, d67c16582123937, -, 10)
        }; // minimax fit over [ 0x1.0p-52, 0.18]   Max error:
           // 0x1.67ea5c184e5d9p-64

        // Calculate y = p(r)
        const size_t atan_coeff_count =
            sizeof(atan_coeffs) / sizeof(atan_coeffs[0]);
        long double y = atan_coeffs[atan_coeff_count - 1];
        for (i = (int)atan_coeff_count - 2; i >= 0; i--)
            y = atan_coeffs[i] + y * z2;

        z *= 2.0L; // fold in 2.0 for 2.0 * atan(z)
        y *= z; // rp(r)

        return head + ((y + tail) + z);
    }

    // do |x| <= sqrt(0.5) here
    //                                                     acos( sqrt(z) ) -
    //                                                     PI/2
    //  Piecewise minimax polynomial fits for p(z) = 1 +
    //  ------------------------;
    //                                                            sqrt(z)
    //
    //  Define z = x*x, and solve for acos(x) over x in  x >= 0:
    //
    //      acos( sqrt(z) ) = acos(x) = x*(p(z)-1) + PI/2 = xp(x**2) - x + PI/2
    //
    const long double coeffs[4][14] = {
        { HEX_LDBL(-, a, fa7382e1f347974, -, 10),
          HEX_LDBL(-, b, 4d5a992de1ac4da, -, 6),
          HEX_LDBL(-, a, c526184bd558c17, -, 7),
          HEX_LDBL(-, d, 9ed9b0346ec092a, -, 8),
          HEX_LDBL(-, 9, dca410c1f04b1f, -, 8),
          HEX_LDBL(-, f, 76e411ba9581ee5, -, 9),
          HEX_LDBL(-, c, c71b00479541d8e, -, 9),
          HEX_LDBL(-, a, f527a3f9745c9de, -, 9),
          HEX_LDBL(-, 9, a93060051f48d14, -, 9),
          HEX_LDBL(-, 8, b3d39ad70e06021, -, 9),
          HEX_LDBL(-, f, f2ab95ab84f79c, -, 10),
          HEX_LDBL(-, e, d1af5f5301ccfe4, -, 10),
          HEX_LDBL(-, e, 1b53ba562f0f74a, -, 10),
          HEX_LDBL(-, d, 6a3851330e15526, -,
                   10) }, // x - 0.0625 in [ -0x1.fffffffffp-5, 0x1.0p-4 ]
                          // Error: 0x1.97839bf07024p-76

        { HEX_LDBL(-, 8, c2f1d638e4c1b48, -, 8),
          HEX_LDBL(-, c, d47ac903c311c2c, -, 6),
          HEX_LDBL(-, d, e020b2dabd5606a, -, 7),
          HEX_LDBL(-, a, 086fafac220f16b, -, 7),
          HEX_LDBL(-, 8, 55b5efaf6b86c3e, -, 7),
          HEX_LDBL(-, f, 05c9774fed2f571, -, 8),
          HEX_LDBL(-, e, 484a93f7f0fc772, -, 8),
          HEX_LDBL(-, e, 1a32baef01626e4, -, 8),
          HEX_LDBL(-, e, 528e525b5c9c73d, -, 8),
          HEX_LDBL(-, e, ddd5d27ad49b2c8, -, 8),
          HEX_LDBL(-, f, b3259e7ae10c6f, -, 8),
          HEX_LDBL(-, 8, 68998170d5b19b7, -, 7),
          HEX_LDBL(-, 9, 4468907f007727, -, 7),
          HEX_LDBL(-, a, 2ad5e4906a8e7b3, -,
                   7) }, // x - 0.1875 in [ -0x1.0p-4, 0x1.0p-4 ]    Error:
                         // 0x1.647af70073457p-73

        { HEX_LDBL(-, f, a76585ad399e7ac, -, 8),
          HEX_LDBL(-, e, d665b7dd504ca7c, -, 6),
          HEX_LDBL(-, 9, 4c7c2402bd4bc33, -, 6),
          HEX_LDBL(-, f, ba76b69074ff71c, -, 7),
          HEX_LDBL(-, f, 58117784bdb6d5f, -, 7),
          HEX_LDBL(-, 8, 22ddd8eef53227d, -, 6),
          HEX_LDBL(-, 9, 1d1d3b57a63cdb4, -, 6),
          HEX_LDBL(-, a, 9c4bdc40cca848, -, 6),
          HEX_LDBL(-, c, b673b12794edb24, -, 6),
          HEX_LDBL(-, f, 9290a06e31575bf, -, 6),
          HEX_LDBL(-, 9, b4929c16aeb3d1f, -, 5),
          HEX_LDBL(-, c, 461e725765a7581, -, 5),
          HEX_LDBL(-, 8, 0a59654c98d9207, -, 4),
          HEX_LDBL(-, a, 6de6cbd96c80562, -,
                   4) }, // x - 0.3125 in [ -0x1.0p-4, 0x1.0p-4 ]   Error:
                         // 0x1.b0246c304ce1ap-70

        { HEX_LDBL(-, b, dca8b0359f96342, -, 7),
          HEX_LDBL(-, 8, cd2522fcde9823, -, 5),
          HEX_LDBL(-, d, 2af9397b27ff74d, -, 6),
          HEX_LDBL(-, d, 723f2c2c2409811, -, 6),
          HEX_LDBL(-, f, ea8f8481ecc3cd1, -, 6),
          HEX_LDBL(-, a, 43fd8a7a646b0b2, -, 5),
          HEX_LDBL(-, e, 01b0bf63a4e8d76, -, 5),
          HEX_LDBL(-, 9, f0b7096a2a7b4d, -, 4),
          HEX_LDBL(-, e, 872e7c5a627ab4c, -, 4),
          HEX_LDBL(-, a, dbd760a1882da48, -, 3),
          HEX_LDBL(-, 8, 424e4dea31dd273, -, 2),
          HEX_LDBL(-, c, c05d7730963e793, -, 2),
          HEX_LDBL(-, a, 523d97197cd124a, -, 1),
          HEX_LDBL(-, 8, 307ba943978aaee, +,
                   0) } // x - 0.4375 in [ -0x1.0p-4, 0x1.0p-4 ]  Error:
                        // 0x1.9ecff73da69c9p-66
    };

    const long double offsets[4] = { 0.0625, 0.1875, 0.3125, 0.4375 };
    const size_t coeff_count = sizeof(coeffs[0]) / sizeof(coeffs[0][0]);

    // reduce the incoming values a bit so that they are in the range
    // [-0x1.0p-4, 0x1.0p-4]
    const long double *c;
    i = x2 * 8.0L;
    c = coeffs[i];
    x2 -= offsets[i]; // exact

    // calcualte p(x2)
    long double y = c[coeff_count - 1];
    for (i = (int)coeff_count - 2; i >= 0; i--) y = c[i] + y * x2;

    // xp(x2)
    y *= x;

    // return xp(x2) - x + PI/2
    return head + ((y + tail) - x);
}

double reference_relaxed_acos(double x) { return reference_acos(x); }

double reference_log10(double x)
{
    if (x == 0.0) return -INFINITY;

    if (x < 0.0) return cl_make_nan();

    if (isinf(x)) return INFINITY;

    double log2Hi = HEX_DBL(+, 1, 34413509f79fe, -, 2);
    double logxHi, logxLo;
    __log2_ep(&logxHi, &logxLo, x);
    return logxHi * log2Hi;
}

double reference_relaxed_log10(double x) { return reference_log10(x); }

long double reference_log10l(long double x)
{
    if (x == 0.0) return -INFINITY;

    if (x < 0.0) return cl_make_nan();

    if (isinf(x)) return INFINITY;

    double log2Hi = HEX_DBL(+, 1, 34413509f79fe, -, 2);
    double log2Lo = HEX_DBL(+, 1, e623e2566b02d, -, 55);
    double logxHi, logxLo;
    __log2_ep(&logxHi, &logxLo, x);

    long double lg2 = (long double)log2Hi + (long double)log2Lo;
    long double logx = (long double)logxHi + (long double)logxLo;
    return logx * lg2;
}

double reference_acos(double x) { return acos(x); }

double reference_atan2(double x, double y)
{
#if defined(_WIN32)
    // fix edge cases for Windows
    if (isinf(x) && isinf(y))
    {
        double retval = (y > 0) ? M_PI_4 : 3.f * M_PI_4;
        return (x > 0) ? retval : -retval;
    }
#endif // _WIN32
    return atan2(x, y);
}

long double reference_atan2l(long double x, long double y)
{
#if defined(_WIN32)
    // fix edge cases for Windows
    if (isinf(x) && isinf(y))
    {
        long double retval = (y > 0) ? M_PI_4 : 3.f * M_PI_4;
        return (x > 0) ? retval : -retval;
    }
#endif // _WIN32
    return atan2l(x, y);
}

double reference_frexp(double a, int *exp)
{
    if (isnan(a) || isinf(a) || a == 0.0)
    {
        *exp = 0;
        return a;
    }

    union {
        cl_double d;
        cl_ulong l;
    } u;

    u.d = a;

    // separate out sign
    cl_ulong s = u.l & 0x8000000000000000ULL;
    u.l &= 0x7fffffffffffffffULL;
    int bias = -1022;

    if ((u.l & 0x7ff0000000000000ULL) == 0)
    {
        double d = u.l;
        u.d = d;
        bias -= 1074;
    }

    int e = (int)((u.l & 0x7ff0000000000000ULL) >> 52);
    u.l &= 0x000fffffffffffffULL;
    e += bias;
    u.l |= ((cl_ulong)1022 << 52);
    u.l |= s;

    *exp = e;
    return u.d;
}

long double reference_frexpl(long double a, int *exp)
{
    if (isnan(a) || isinf(a) || a == 0.0)
    {
        *exp = 0;
        return a;
    }

    if (sizeof(long double) == sizeof(double))
    {
        return reference_frexp(a, exp);
    }
    else
    {
        return frexpl(a, exp);
    }
}


double reference_atan(double x) { return atan(x); }

long double reference_atanl(long double x) { return atanl(x); }

long double reference_asinl(long double x) { return asinl(x); }

double reference_asin(double x) { return asin(x); }

double reference_relaxed_asin(double x) { return reference_asin(x); }

double reference_fabs(double x) { return fabs(x); }

double reference_cosh(double x) { return cosh(x); }

long double reference_sqrtl(long double x)
{
#if defined(__SSE2__)                                                          \
    || (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)))
    __m128d result128 = _mm_set_sd((double)x);
    result128 = _mm_sqrt_sd(result128, result128);
    return _mm_cvtsd_f64(result128);
#else
    volatile double dx = x;
    return sqrt(dx);
#endif
}

long double reference_tanhl(long double x) { return tanhl(x); }

long double reference_floorl(long double x)
{
    if (x == 0.0 || reference_isinfl(x) || reference_isnanl(x)) return x;

    long double absx = reference_fabsl(x);
    if (absx >= HEX_LDBL(+, 1, 0, +, 52)) return x;

    if (absx < 1.0)
    {
        if (x < 0.0)
            return -1.0;
        else
            return 0.0;
    }

    long double r = (long double)((cl_long)x);

    if (x < 0.0 && r > x) r -= 1.0;

    return r;
}


double reference_tanh(double x) { return tanh(x); }

long double reference_assignmentl(long double x) { return x; }

int reference_notl(long double x)
{
    int r = !x;
    return r;
}
