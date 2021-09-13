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
#ifndef _COMPAT_H_
#define _COMPAT_H_

#if defined(_WIN32) && defined(_MSC_VER)
#include <Windows.h>
#else
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
#endif


//
// stdlib.h
//

#include <stdlib.h> // On Windows, _MAX_PATH defined there.

// llabs appeared in MS C v16 (VS 10/2010).
#if defined(_MSC_VER) && _MSC_VER <= 1500
EXTERN_C inline long long llabs(long long __x) { return __x >= 0 ? __x : -__x; }
#endif


//
// stdbool.h
//

// stdbool.h appeared in MS C v18 (VS 12/2013).
#if defined(_MSC_VER) && MSC_VER <= 1700
#if !defined(__cplusplus)
typedef char bool;
#define true 1
#define false 0
#endif
#else
#include <stdbool.h>
#endif // defined(_MSC_VER) && MSC_VER <= 1700


//
// stdint.h
//

// stdint.h appeared in MS C v16 (VS 10/2010) and Intel C v12.
#if defined(_MSC_VER)                                                          \
    && (!defined(__INTEL_COMPILER) && _MSC_VER <= 1500                         \
        || defined(__INTEL_COMPILER) && __INTEL_COMPILER < 1200)
typedef unsigned char uint8_t;
typedef char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
#else
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>
#endif


//
// float.h
//

#include <float.h>


//
// fenv.h
//

// fenv.h appeared in MS C v18 (VS 12/2013).
#if defined(_MSC_VER) && _MSC_VER <= 1700 && !defined(__INTEL_COMPILER)
// reimplement fenv.h because windows doesn't have it
#define FE_INEXACT 0x0020
#define FE_UNDERFLOW 0x0010
#define FE_OVERFLOW 0x0008
#define FE_DIVBYZERO 0x0004
#define FE_INVALID 0x0001
#define FE_ALL_EXCEPT 0x003D
int fetestexcept(int excepts);
int feclearexcept(int excepts);
#else
#include <fenv.h>
#endif


//
// math.h
//

#if defined(__INTEL_COMPILER)
#include <mathimf.h>
#else
#include <math.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#if defined(_MSC_VER)

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NAN
#define NAN (INFINITY - INFINITY)
#endif

#ifndef HUGE_VALF
#define HUGE_VALF (float)HUGE_VAL
#endif

#ifndef INFINITY
#define INFINITY (FLT_MAX + FLT_MAX)
#endif

#ifndef isfinite
#define isfinite(x) _finite(x)
#endif

#ifndef isnan
#define isnan(x) ((x) != (x))
#endif

#ifndef isinf
#define isinf(_x) ((_x) == INFINITY || (_x) == -INFINITY)
#endif

#if _MSC_VER < 1900 && !defined(__INTEL_COMPILER)

double rint(double x);
float rintf(float x);
long double rintl(long double x);

float cbrtf(float);
double cbrt(double);

int ilogb(double x);
int ilogbf(float x);
int ilogbl(long double x);

double fmax(double x, double y);
double fmin(double x, double y);
float fmaxf(float x, float y);
float fminf(float x, float y);

double log2(double x);
long double log2l(long double x);

double exp2(double x);
long double exp2l(long double x);

double fdim(double x, double y);
float fdimf(float x, float y);
long double fdiml(long double x, long double y);

double remquo(double x, double y, int* quo);
float remquof(float x, float y, int* quo);
long double remquol(long double x, long double y, int* quo);

long double scalblnl(long double x, long n);

float hypotf(float x, float y);
long double hypotl(long double x, long double y);
double lgamma(double x);
float lgammaf(float x);

double trunc(double x);
float truncf(float x);

double log1p(double x);
float log1pf(float x);
long double log1pl(long double x);

double copysign(double x, double y);
float copysignf(float x, float y);
long double copysignl(long double x, long double y);

long lround(double x);
long lroundf(float x);
// long lroundl(long double x)

double round(double x);
float roundf(float x);
long double roundl(long double x);

int cf_signbit(double x);
int cf_signbitf(float x);

// Added in _MSC_VER == 1800 (Visual Studio 2013)
#if _MSC_VER < 1800
static int signbit(double x) { return cf_signbit(x); }
#endif
static int signbitf(float x) { return cf_signbitf(x); }

long int lrint(double flt);
long int lrintf(float flt);

float int2float(int32_t ix);
int32_t float2int(float fx);

#endif // _MSC_VER < 1900 && ! defined( __INTEL_COMPILER )

#if _MSC_VER < 1900 && (!defined(__INTEL_COMPILER) || __INTEL_COMPILER < 1300)
// These functions appeared in Intel C v13 and Visual Studio 2015
float nanf(const char* str);
double nan(const char* str);
long double nanl(const char* str);
#endif

#ifdef __cplusplus
}
#endif

#endif // defined(_MSC_VER)

#if defined(__ANDROID__)
#define log2(X) (log(X) / log(2))
#endif


//
// stdio.h
//

#if defined(_MSC_VER)
// snprintf added in _MSC_VER == 1900 (Visual Studio 2015)
#if _MSC_VER < 1900
#define snprintf sprintf_s
#endif
#endif // defined(_MSC_VER)


//
// string.h
//

#if defined(_MSC_VER)
#define strtok_r strtok_s
#endif


//
// unistd.h
//

#if defined(_MSC_VER)
EXTERN_C unsigned int sleep(unsigned int sec);
EXTERN_C int usleep(int usec);
#endif


//
// syscall.h
//

#if defined(__ANDROID__)
// Android bionic's isn't providing SYS_sysctl wrappers.
#define SYS__sysctl __NR__sysctl
#endif


// Some tests use _malloca which defined in malloc.h.
#if !defined(__APPLE__)
#include <malloc.h>
#endif


//
// ???
//

#if defined(_MSC_VER)

#define MAXPATHLEN _MAX_PATH

EXTERN_C uint64_t ReadTime(void);
EXTERN_C double SubtractTime(uint64_t endTime, uint64_t startTime);

/** Returns the number of leading 0-bits in x,
    starting at the most significant bit position.
    If x is 0, the result is undefined.
*/
EXTERN_C int __builtin_clz(unsigned int pattern);

#endif


/*-----------------------------------------------------------------------------
   WARNING: DO NOT USE THESE MACROS:
        MAKE_HEX_FLOAT, MAKE_HEX_DOUBLE, MAKE_HEX_LONG.

   This is a typical usage of the macros:

     double yhi = MAKE_HEX_DOUBLE(0x1.5555555555555p-2,0x15555555555555LL,-2);

   (taken from math_brute_force/reference_math.c). There are two problems:

     1. There is an error here. On Windows in will produce incorrect result
        `0x1.5555555555555p+50'.
        To have a correct result it should be written as:
           MAKE_HEX_DOUBLE(0x1.5555555555555p-2, 0x15555555555555LL, -54)
        A proper value of the third argument is not obvious -- sometimes it
        should be the same as exponent of the first argument, but sometimes
        not.

     2. Information is duplicated. It is easy to make a mistake.

   Use HEX_FLT, HEX_DBL, HEX_LDBL macros instead
   (see them in the bottom of the file).
-----------------------------------------------------------------------------*/
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)

#define MAKE_HEX_FLOAT(x, y, z) ((float)ldexp((float)(y), z))
#define MAKE_HEX_DOUBLE(x, y, z) ldexp((double)(y), z)
#define MAKE_HEX_LONG(x, y, z) ((long double)ldexp((long double)(y), z))

#else

// Do not use these macros in new code, use HEX_FLT, HEX_DBL, HEX_LDBL instead.
#define MAKE_HEX_FLOAT(x, y, z) x
#define MAKE_HEX_DOUBLE(x, y, z) x
#define MAKE_HEX_LONG(x, y, z) x

#endif


/*-----------------------------------------------------------------------------
   HEX_FLT, HEXT_DBL, HEX_LDBL -- Create hex floating point literal of type
   float, double, long double respectively. Arguments:

      sm    -- sign of number,
      int   -- integer part of mantissa (without `0x' prefix),
      fract -- fractional part of mantissa (without decimal point and `L' or
            `LL' suffixes),
      se    -- sign of exponent,
      exp   -- absolute value of (binary) exponent.

   Example:

      double yhi = HEX_DBL(+, 1, 5555555555555, -, 2); // 0x1.5555555555555p-2

   Note:

      We have to pass signs as separate arguments because gcc pass negative
   integer values (e. g. `-2') into a macro as two separate tokens, so
   `HEX_FLT(1, 0, -2)' produces result `0x1.0p- 2' (note a space between minus
   and two) which is not a correct floating point literal.
-----------------------------------------------------------------------------*/
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
// If compiler does not support hex floating point literals:
#define HEX_FLT(sm, int, fract, se, exp)                                       \
    sm ldexpf((float)(0x##int##fract##UL),                                     \
              se exp + ilogbf((float)0x##int)                                  \
                  - ilogbf((float)(0x##int##fract##UL)))
#define HEX_DBL(sm, int, fract, se, exp)                                       \
    sm ldexp((double)(0x##int##fract##ULL),                                    \
             se exp + ilogb((double)0x##int)                                   \
                 - ilogb((double)(0x##int##fract##ULL)))
#define HEX_LDBL(sm, int, fract, se, exp)                                      \
    sm ldexpl((long double)(0x##int##fract##ULL),                              \
              se exp + ilogbl((long double)0x##int)                            \
                  - ilogbl((long double)(0x##int##fract##ULL)))
#else
// If compiler supports hex floating point literals: just concatenate all the
// parts into a literal.
#define HEX_FLT(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##F
#define HEX_DBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp
#define HEX_LDBL(sm, int, fract, se, exp) sm 0x##int##.##fract##p##se##exp##L
#endif

#if defined(__MINGW32__)
#include <Windows.h>
#define sleep(sec) Sleep((sec)*1000)
#endif

#endif // _COMPAT_H_
