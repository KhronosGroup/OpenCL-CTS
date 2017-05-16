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

#if defined(_WIN32) && defined (_MSC_VER)

#include <Windows.h>
#include <Winbase.h>
#include <CL/cl.h>
#include <float.h>
#include <xmmintrin.h>
#include <math.h>

#define MAKE_HEX_FLOAT(x,y,z)  ((float)ldexp( (float)(y), z))
#define MAKE_HEX_DOUBLE(x,y,z) ldexp( (double)(y), z)
#define MAKE_HEX_LONG(x,y,z)   ((long double) ldexp( (long double)(y), z))

#define isfinite(x) _finite(x)

#if !defined(__cplusplus)
typedef char bool;
#define inline

#else
extern "C" {
#endif

typedef unsigned char       uint8_t;
typedef char                int8_t;
typedef unsigned short      uint16_t;
typedef short               int16_t;
typedef unsigned int        uint32_t;
typedef int                 int32_t;
typedef unsigned long long  uint64_t;
typedef long long           int64_t;

#define MAXPATHLEN MAX_PATH

typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;


#define INFINITY    (FLT_MAX + FLT_MAX)
//#define NAN (INFINITY | 1)
//const static int PINFBITPATT_SP32  = INFINITY;

#ifndef M_PI
    #define M_PI    3.14159265358979323846264338327950288
#endif


#define    isnan( x )       ((x) != (x))
#define     isinf( _x)      ((_x) == INFINITY || (_x) == -INFINITY)

double rint( double x);
float  rintf( float x);
long double rintl( long double x);

float cbrtf( float );
double cbrt( double );

int    ilogb( double x);
int    ilogbf (float x);
int    ilogbl(long double x);

double fmax(double x, double y);
double fmin(double x, double y);
float  fmaxf( float x, float y );
float  fminf(float x, float y);

double      log2(double x);
long double log2l(long double x);

double      exp2(double x);
long double exp2l(long double x);

double      fdim(double x, double y);
float       fdimf(float x, float y);
long double fdiml(long double x, long double y);

double      remquo( double x, double y, int *quo);
float       remquof( float x, float y, int *quo);
long double remquol( long double x, long double y, int *quo);

long double scalblnl(long double x, long n);

inline long long
llabs(long long __x) { return __x >= 0 ? __x : -__x; }


// end of math functions

uint64_t ReadTime( void );
double SubtractTime( uint64_t endTime, uint64_t startTime );

#define sleep(X)   Sleep(1000*X)
// snprintf added in _MSC_VER == 1900 (Visual Studio 2015)
#if _MSC_VER < 1900
	#define snprintf   sprintf_s
#endif
//#define hypotl     _hypot

float   make_nan();
float nanf( const char* str);
double  nan( const char* str);
long double nanl( const char* str);

//#if defined USE_BOOST
//#include <boost/math/tr1.hpp>
//double hypot(double x, double y);
float hypotf(float x, float y);
long double hypotl(long double x, long double y) ;
double lgamma(double x);
float  lgammaf(float x);

double trunc(double x);
float  truncf(float x);

double log1p(double x);
float  log1pf(float x);
long double log1pl(long double x);

double copysign(double x, double y);
float  copysignf(float x, float y);
long double copysignl(long double x, long double y);

long lround(double x);
long lroundf(float x);
//long lroundl(long double x)

double round(double x);
float  roundf(float x);
long double roundl(long double x);

// Added in _MSC_VER == 1800 (Visual Studio 2013)
#if _MSC_VER < 1800
	int signbit(double x);
#endif
int signbitf(float x);

//bool signbitl(long double x)         { return boost::math::tr1::signbit<long double>(x); }
//#endif // USE_BOOST

long int lrint (double flt);
long int lrintf (float flt);


float   int2float (int32_t ix);
int32_t float2int (float   fx);

/** Returns the number of leading 0-bits in x,
    starting at the most significant bit position.
    If x is 0, the result is undefined.
*/
int __builtin_clz(unsigned int pattern);


static const double zero=  0.00000000000000000000e+00;
#define NAN  (INFINITY - INFINITY)
#define HUGE_VALF (float)HUGE_VAL

int usleep(int usec);

// reimplement fenv.h because windows doesn't have it
#define FE_INEXACT          0x0020
#define FE_UNDERFLOW        0x0010
#define FE_OVERFLOW         0x0008
#define FE_DIVBYZERO        0x0004
#define FE_INVALID          0x0001
#define FE_ALL_EXCEPT       0x003D

int fetestexcept(int excepts);
int feclearexcept(int excepts);

#ifdef __cplusplus
}
#endif

#else // !((defined(_WIN32) && defined(_MSC_VER)
#if defined(__MINGW32__)
#include <windows.h>
#define sleep(X)   Sleep(1000*X)

#endif
#if defined(__linux__) || defined(__MINGW32__) || defined(__APPLE__)
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <fenv.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#endif
#define MAKE_HEX_FLOAT(x,y,z) x
#define MAKE_HEX_DOUBLE(x,y,z) x
#define MAKE_HEX_LONG(x,y,z) x

#endif // !((defined(_WIN32) && defined(_MSC_VER)


#endif // _COMPAT_H_
