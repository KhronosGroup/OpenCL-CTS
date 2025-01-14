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
#include "compat.h"

#if defined(_MSC_VER)

#include <limits.h>
#include <stdlib.h>

#include <CL/cl.h>

#include <windows.h>

#if _MSC_VER < 1900 && !defined(__INTEL_COMPILER)

///////////////////////////////////////////////////////////////////
//
//                   rint, rintf
//
///////////////////////////////////////////////////////////////////

float copysignf(float x, float y)
{
    union {
        cl_uint u;
        float f;
    } ux, uy;

    ux.f = x;
    uy.f = y;

    ux.u = (ux.u & 0x7fffffffU) | (uy.u & 0x80000000U);

    return ux.f;
}

double copysign(double x, double y)
{
    union {
        cl_ulong u;
        double f;
    } ux, uy;

    ux.f = x;
    uy.f = y;

    ux.u = (ux.u & 0x7fffffffffffffffULL) | (uy.u & 0x8000000000000000ULL);

    return ux.f;
}

long double copysignl(long double x, long double y)
{
    union {
        long double f;
        struct
        {
            cl_ulong m;
            cl_ushort sexp;
        } u;
    } ux, uy;

    ux.f = x;
    uy.f = y;

    ux.u.sexp = (ux.u.sexp & 0x7fff) | (uy.u.sexp & 0x8000);

    return ux.f;
}

float rintf(float x)
{
    float absx = fabsf(x);

    if (absx < 8388608.0f /* 0x1.0p23f */)
    {
        float magic = copysignf(8388608.0f /* 0x1.0p23f */, x);
        float rounded = x + magic;
        rounded -= magic;
        x = copysignf(rounded, x);
    }

    return x;
}

double rint(double x)
{
    double absx = fabs(x);

    if (absx < 4503599627370496.0 /* 0x1.0p52f */)
    {
        double magic = copysign(4503599627370496.0 /* 0x1.0p52 */, x);
        double rounded = x + magic;
        rounded -= magic;
        x = copysign(rounded, x);
    }

    return x;
}

long double rintl(long double x)
{
    double absx = fabs(x);

    if (absx < 9223372036854775808.0L /* 0x1.0p64f */)
    {
        long double magic =
            copysignl(9223372036854775808.0L /* 0x1.0p63L */, x);
        long double rounded = x + magic;
        rounded -= magic;
        x = copysignl(rounded, x);
    }

    return x;
}

#if _MSC_VER < 1800

///////////////////////////////////////////////////////////////////
//
//                   ilogb, ilogbf, ilogbl
//
///////////////////////////////////////////////////////////////////
#ifndef FP_ILOGB0
#define FP_ILOGB0 INT_MIN
#endif

#ifndef FP_ILOGBNAN
#define FP_ILOGBNAN INT_MIN
#endif

int ilogb(double x)
{
    union {
        double f;
        cl_ulong u;
    } u;
    u.f = x;

    cl_ulong absx = u.u & CL_LONG_MAX;
    if (absx - 0x0001000000000000ULL
        >= 0x7ff0000000000000ULL - 0x0001000000000000ULL)
    {
        switch (absx)
        {
            case 0: return FP_ILOGB0;
            case 0x7ff0000000000000ULL: return INT_MAX;
            default:
                if (absx > 0x7ff0000000000000ULL) return FP_ILOGBNAN;

                // subnormal
                u.u = absx | 0x3ff0000000000000ULL;
                u.f -= 1.0;
                return (u.u >> 52) - (1023 + 1022);
        }
    }

    return (absx >> 52) - 1023;
}


int ilogbf(float x)
{
    union {
        float f;
        cl_uint u;
    } u;
    u.f = x;

    cl_uint absx = u.u & 0x7fffffff;
    if (absx - 0x00800000U >= 0x7f800000U - 0x00800000U)
    {
        switch (absx)
        {
            case 0: return FP_ILOGB0;
            case 0x7f800000U: return INT_MAX;
            default:
                if (absx > 0x7f800000) return FP_ILOGBNAN;

                // subnormal
                u.u = absx | 0x3f800000U;
                u.f -= 1.0f;
                return (u.u >> 23) - (127 + 126);
        }
    }

    return (absx >> 23) - 127;
}

int ilogbl(long double x)
{
    union {
        long double f;
        struct
        {
            cl_ulong m;
            cl_ushort sexp;
        } u;
    } u;
    u.f = x;

    int exp = u.u.sexp & 0x7fff;
    if (0 == exp)
    {
        if (0 == u.u.m) return FP_ILOGB0;

        // subnormal
        u.u.sexp = 0x3fff;
        u.f -= 1.0f;
        exp = u.u.sexp & 0x7fff;

        return exp - (0x3fff + 0x3ffe);
    }
    else if (0x7fff == exp)
    {
        if (u.u.m & CL_LONG_MAX) return FP_ILOGBNAN;

        return INT_MAX;
    }

    return exp - 0x3fff;
}

#endif // _MSC_VER < 1800

///////////////////////////////////////////////////////////////////
//
//                 fmax, fmin, fmaxf, fminf
//
///////////////////////////////////////////////////////////////////

static void GET_BITS_SP32(float fx, unsigned int* ux)
{
    volatile union {
        float f;
        unsigned int u;
    } _bitsy;
    _bitsy.f = (fx);
    *ux = _bitsy.u;
}
/* static void GET_BITS_SP32(float fx, unsigned int* ux) */
/* { */
/*     volatile union {float f; unsigned int i;} _bitsy; */
/*     _bitsy.f = (fx); */
/*     *ux = _bitsy.i; */
/* } */
static void PUT_BITS_SP32(unsigned int ux, float* fx)
{
    volatile union {
        float f;
        unsigned int u;
    } _bitsy;
    _bitsy.u = (ux);
    *fx = _bitsy.f;
}
/* static void PUT_BITS_SP32(unsigned int ux, float* fx) */
/* { */
/*     volatile union {float f; unsigned int i;} _bitsy; */
/*     _bitsy.i = (ux); */
/*     *fx = _bitsy.f; */
/* } */
static void GET_BITS_DP64(double dx, unsigned __int64* lx)
{
    volatile union {
        double d;
        unsigned __int64 l;
    } _bitsy;
    _bitsy.d = (dx);
    *lx = _bitsy.l;
}
static void PUT_BITS_DP64(unsigned __int64 lx, double* dx)
{
    volatile union {
        double d;
        unsigned __int64 l;
    } _bitsy;
    _bitsy.l = (lx);
    *dx = _bitsy.d;
}

#if 0
int SIGNBIT_DP64(double x )
{
    int hx;
    _GET_HIGH_WORD(hx,x);
    return((hx>>31));
}
#endif

#if _MSC_VER < 1900

/* fmax(x, y) returns the larger (more positive) of x and y.
   NaNs are treated as missing values: if one argument is NaN,
   the other argument is returned. If both arguments are NaN,
   the first argument is returned. */

/* This works so long as the compiler knows that (x != x) means
   that x is NaN; gcc does. */
double fmax(double x, double y)
{
    if (isnan(y)) return x;

    return x >= y ? x : y;
}


/* fmin(x, y) returns the smaller (more negative) of x and y.
   NaNs are treated as missing values: if one argument is NaN,
   the other argument is returned. If both arguments are NaN,
   the first argument is returned. */

double fmin(double x, double y)
{
    if (isnan(y)) return x;

    return x <= y ? x : y;
}


float fmaxf(float x, float y)
{
    if (isnan(y)) return x;

    return x >= y ? x : y;
}

/* fminf(x, y) returns the smaller (more negative) of x and y.
   NaNs are treated as missing values: if one argument is NaN,
   the other argument is returned. If both arguments are NaN,
   the first argument is returned. */

float fminf(float x, float y)
{
    if (isnan(y)) return x;

    return x <= y ? x : y;
}

long double scalblnl(long double x, long n)
{
    union {
        long double d;
        struct
        {
            cl_ulong m;
            cl_ushort sexp;
        } u;
    } u;
    u.u.m = CL_LONG_MIN;

    if (x == 0.0L || n < -2200) return copysignl(0.0L, x);

    if (n > 2200) return INFINITY;

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
}

///////////////////////////////////////////////////////////////////
//
//                          log2
//
///////////////////////////////////////////////////////////////////
const static cl_double log_e_base2 = 1.4426950408889634074;
const static cl_double log_10_base2 = 3.3219280948873623478;

// double log10(double x);

double log2(double x) { return 1.44269504088896340735992468100189214 * log(x); }

long double log2l(long double x)
{
    return 1.44269504088896340735992468100189214L * log(x);
}

double trunc(double x)
{
    double absx = fabs(x);

    if (absx < 4503599627370496.0 /* 0x1.0p52f */)
    {
        cl_long rounded = x;
        x = copysign((double)rounded, x);
    }

    return x;
}

float truncf(float x)
{
    float absx = fabsf(x);

    if (absx < 8388608.0f /* 0x1.0p23f */)
    {
        cl_int rounded = x;
        x = copysignf((float)rounded, x);
    }

    return x;
}

long lround(double x)
{
    double absx = fabs(x);

    if (absx < 0.5) return 0;

    if (absx < 4503599627370496.0 /* 0x1.0p52 */)
    {
        absx += 0.5;
        cl_long rounded = absx;
        absx = rounded;
        x = copysign(absx, x);
    }

    if (x >= (double)LONG_MAX) return LONG_MAX;

    return (long)x;
}

long lroundf(float x)
{
    float absx = fabsf(x);

    if (absx < 0.5f) return 0;

    if (absx < 8388608.0f)
    {
        absx += 0.5f;
        cl_int rounded = absx;
        absx = rounded;
        x = copysignf(absx, x);
    }

    if (x >= (float)LONG_MAX) return LONG_MAX;

    return (long)x;
}

double round(double x)
{
    double absx = fabs(x);

    if (absx < 0.5) return copysign(0.0, x);

    if (absx < 4503599627370496.0 /* 0x1.0p52 */)
    {
        absx += 0.5;
        cl_long rounded = absx;
        absx = rounded;
        x = copysign(absx, x);
    }

    return x;
}

float roundf(float x)
{
    float absx = fabsf(x);

    if (absx < 0.5f) return copysignf(0.0f, x);

    if (absx < 8388608.0f)
    {
        absx += 0.5f;
        cl_int rounded = absx;
        absx = rounded;
        x = copysignf(absx, x);
    }

    return x;
}

long double roundl(long double x)
{
    long double absx = fabsl(x);

    if (absx < 0.5L) return copysignl(0.0L, x);

    if (absx < 9223372036854775808.0L /*0x1.0p63L*/)
    {
        absx += 0.5L;
        cl_ulong rounded = absx;
        absx = rounded;
        x = copysignl(absx, x);
    }

    return x;
}

float cbrtf(float x)
{
    float z = pow(fabs((double)x), 1.0 / 3.0);
    return copysignf(z, x);
}

double cbrt(double x) { return copysign(pow(fabs(x), 1.0 / 3.0), x); }

long int lrint(double x)
{
    double absx = fabs(x);

    if (x >= (double)LONG_MAX) return LONG_MAX;

    if (absx < 4503599627370496.0 /* 0x1.0p52 */)
    {
        double magic = copysign(4503599627370496.0 /* 0x1.0p52 */, x);
        double rounded = x + magic;
        rounded -= magic;
        return (long int)rounded;
    }

    return (long int)x;
}

long int lrintf(float x)
{
    float absx = fabsf(x);

    if (x >= (float)LONG_MAX) return LONG_MAX;

    if (absx < 8388608.0f /* 0x1.0p23f */)
    {
        float magic = copysignf(8388608.0f /* 0x1.0p23f */, x);
        float rounded = x + magic;
        rounded -= magic;
        return (long int)rounded;
    }

    return (long int)x;
}

#endif // _MSC_VER < 1900

///////////////////////////////////////////////////////////////////
//
//                  fenv functions
//
///////////////////////////////////////////////////////////////////

#if _MSC_VER < 1800
int fetestexcept(int excepts)
{
    unsigned int status = _statusfp();
    return excepts
        & (((status & _SW_INEXACT) ? FE_INEXACT : 0)
           | ((status & _SW_UNDERFLOW) ? FE_UNDERFLOW : 0)
           | ((status & _SW_OVERFLOW) ? FE_OVERFLOW : 0)
           | ((status & _SW_ZERODIVIDE) ? FE_DIVBYZERO : 0)
           | ((status & _SW_INVALID) ? FE_INVALID : 0));
}

int feclearexcept(int excepts)
{
    _clearfp();
    return 0;
}
#endif

#endif // __INTEL_COMPILER

#if _MSC_VER < 1900 && (!defined(__INTEL_COMPILER) || __INTEL_COMPILER < 1300)

float nanf(const char* str)
{
    cl_uint u = atoi(str);
    u |= 0x7fc00000U;
    return *(float*)(&u);
}


double nan(const char* str)
{
    cl_ulong u = atoi(str);
    u |= 0x7ff8000000000000ULL;
    return *(double*)(&u);
}

// double check this implementatation
long double nanl(const char* str)
{
    union {
        long double f;
        struct
        {
            cl_ulong m;
            cl_ushort sexp;
        } u;
    } u;
    u.u.sexp = 0x7fff;
    u.u.m = 0x8000000000000000ULL | atoi(str);

    return u.f;
}

#endif

///////////////////////////////////////////////////////////////////
//
//                  misc functions
//
///////////////////////////////////////////////////////////////////

/*
// This function is commented out because the Windows implementation should
never call munmap.
// If it is calling it, we have a bug. Please file a bugzilla.
int munmap(void *addr, size_t len)
{
// FIXME: this is not correct.  munmap is like free()
// http://www.opengroup.org/onlinepubs/7990989775/xsh/munmap.html

    return (int)VirtualAlloc( (LPVOID)addr, len,
                  MEM_COMMIT|MEM_RESERVE, PAGE_NOACCESS );
}
*/

uint64_t ReadTime(void)
{
    LARGE_INTEGER current;
    QueryPerformanceCounter(&current);
    return (uint64_t)current.QuadPart;
}

double SubtractTime(uint64_t endTime, uint64_t startTime)
{
    static double PerformanceFrequency = 0.0;

    if (PerformanceFrequency == 0.0)
    {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        PerformanceFrequency = (double)frequency.QuadPart;
    }

    return (double)(endTime - startTime) / PerformanceFrequency * 1e9;
}

int cf_signbit(double x)
{
    union {
        double f;
        cl_ulong u;
    } u;
    u.f = x;
    return u.u >> 63;
}

int cf_signbitf(float x)
{
    union {
        float f;
        cl_uint u;
    } u;
    u.f = x;
    return u.u >> 31;
}

float int2float(int32_t ix)
{
    union {
        float f;
        int32_t i;
    } u;
    u.i = ix;
    return u.f;
}

int32_t float2int(float fx)
{
    union {
        float f;
        int32_t i;
    } u;
    u.f = fx;
    return u.i;
}

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if !__has_builtin(__builtin_clz)
#if !defined(_WIN64)
/** Returns the number of leading 0-bits in x,
    starting at the most significant bit position.
    If x is 0, the result is undefined.
*/
int __builtin_clz(unsigned int pattern)
{
#if 0
    int res;
    __asm {
        mov eax, pattern
        bsr eax, eax
        mov res, eax
    }
    return 31 - res;
#endif
    unsigned long index;
    unsigned char res = _BitScanReverse(&index, pattern);
    if (res)
    {
        return 8 * sizeof(int) - 1 - index;
    }
    else
    {
        return 8 * sizeof(int);
    }
}
#else
int __builtin_clz(unsigned int pattern)
{
    int count;
    if (pattern == 0u)
    {
        return 32;
    }
    count = 31;
    if (pattern >= 1u << 16)
    {
        pattern >>= 16;
        count -= 16;
    }
    if (pattern >= 1u << 8)
    {
        pattern >>= 8;
        count -= 8;
    }
    if (pattern >= 1u << 4)
    {
        pattern >>= 4;
        count -= 4;
    }
    if (pattern >= 1u << 2)
    {
        pattern >>= 2;
        count -= 2;
    }
    if (pattern >= 1u << 1)
    {
        count -= 1;
    }
    return count;
}

#endif // !defined(_WIN64)
#endif // !__has_builtin(__builtin_clz)

#include <intrin.h>
#include <emmintrin.h>

int usleep(int usec)
{
    Sleep((usec + 999) / 1000);
    return 0;
}

unsigned int sleep(unsigned int sec)
{
    Sleep(sec * 1000);
    return 0;
}

#endif // defined( _MSC_VER )
