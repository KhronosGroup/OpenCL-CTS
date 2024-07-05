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
#ifndef REFERENCE_MATH_H
#define REFERENCE_MATH_H

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#include "CL/cl_half.h"
#endif

// --  for testing float --
double reference_sinh(double x);
double reference_sqrt(double x);
double reference_tanh(double x);
double reference_acos(double);
double reference_asin(double);
double reference_atan(double);
double reference_atan2(double, double);
double reference_ceil(double);
double reference_cosh(double);
double reference_exp(double);
double reference_fabs(double);
double reference_acospi(double);
double reference_asinpi(double);
double reference_atanpi(double);
double reference_atan2pi(double, double);
double reference_cospi(double);
double reference_divide(double, double);
double reference_fract(double, double*);
float reference_fma(float, float, float, int);
double reference_mad(double, double, double);
double reference_nextafter(double, double);
double reference_recip(double);
double reference_rootn(double, int);
double reference_rsqrt(double);
double reference_sincos(double, double*);
double reference_sinpi(double);
double reference_tanpi(double);
double reference_pow(double x, double y);
double reference_pown(double, int);
double reference_powr(double, double);
double reference_cos(double);
double reference_sin(double);
double reference_tan(double);
double reference_log(double);
double reference_log10(double);
double reference_modf(double, double* n);

double reference_fdim(double, double);
double reference_add(double, double);
double reference_subtract(double, double);
double reference_divide(double, double);
double reference_multiply(double, double);
double reference_remquo(double, double, int*);
double reference_lgamma_r(double, int*);

int reference_isequal(double, double);
int reference_isfinite(double);
int reference_isgreater(double, double);
int reference_isgreaterequal(double, double);
int reference_isinf(double);
int reference_isless(double, double);
int reference_islessequal(double, double);
int reference_islessgreater(double, double);
int reference_isnan(double);
int reference_isnormal(double);
int reference_isnotequal(double, double);
int reference_isordered(double, double);
int reference_isunordered(double, double);
int reference_signbit(float);

double reference_acosh(double x);
double reference_asinh(double x);
double reference_atanh(double x);
double reference_cbrt(double x);
float reference_copysignf(float x, float y);
double reference_copysign(double x, double y);
double reference_exp10(double);
double reference_exp2(double x);
double reference_expm1(double x);
double reference_fmax(double x, double y);
double reference_fmin(double x, double y);
double reference_hypot(double x, double y);
double reference_lgamma(double x);
int reference_ilogb(double);
double reference_log2(double x);
double reference_log1p(double x);
double reference_logb(double x);
double reference_maxmag(double x, double y);
double reference_minmag(double x, double y);
double reference_nan(cl_uint x);
double reference_reciprocal(double x);
double reference_remainder(double x, double y);
double reference_rint(double x);
double reference_round(double x);
double reference_trunc(double x);
double reference_floor(double x);
double reference_fmod(double x, double y);
double reference_frexp(double x, int* n);
double reference_ldexp(double x, int n);

double reference_assignment(double x);
int reference_not(double x);
// -- for testing fast-relaxed

double reference_relaxed_acos(double);
double reference_relaxed_asin(double);
double reference_relaxed_atan(double);
double reference_relaxed_mad(double, double, double);
double reference_relaxed_divide(double x, double y);
double reference_relaxed_sin(double x);
double reference_relaxed_sinpi(double x);
double reference_relaxed_cos(double x);
double reference_relaxed_cospi(double x);
double reference_relaxed_sincos(double x, double* y);
double reference_relaxed_tan(double x);
double reference_relaxed_exp(double x);
double reference_relaxed_exp2(double x);
double reference_relaxed_exp10(double x);
double reference_relaxed_log(double x);
double reference_relaxed_log2(double x);
double reference_relaxed_log10(double x);
double reference_relaxed_pow(double x, double y);
double reference_relaxed_reciprocal(double x);

// -- for testing double --

long double reference_sinhl(long double x);
long double reference_sqrtl(long double x);
long double reference_tanhl(long double x);
long double reference_acosl(long double);
long double reference_asinl(long double);
long double reference_atanl(long double);
long double reference_atan2l(long double, long double);
long double reference_ceill(long double);
long double reference_coshl(long double);
long double reference_expl(long double);
long double reference_fabsl(long double);
long double reference_acospil(long double);
long double reference_asinpil(long double);
long double reference_atanpil(long double);
long double reference_atan2pil(long double, long double);
long double reference_cospil(long double);
long double reference_dividel(long double, long double);
long double reference_fractl(long double, long double*);
long double reference_fmal(long double, long double, long double);
long double reference_madl(long double, long double, long double);
long double reference_nextafterl(long double, long double);
float reference_nextafterh(float, float, bool allow_denormals = true);
cl_half reference_nanh(cl_ushort);
long double reference_recipl(long double);
long double reference_rootnl(long double, int);
long double reference_rsqrtl(long double);
long double reference_sincosl(long double, long double*);
long double reference_sinpil(long double);
long double reference_tanpil(long double);
long double reference_powl(long double x, long double y);
long double reference_pownl(long double, int);
long double reference_powrl(long double, long double);
long double reference_cosl(long double);
long double reference_sinl(long double);
long double reference_tanl(long double);
long double reference_logl(long double);
long double reference_log10l(long double);
long double reference_modfl(long double, long double* n);


long double reference_fdiml(long double, long double);
long double reference_addl(long double, long double);
long double reference_subtractl(long double, long double);
long double reference_dividel(long double, long double);
long double reference_multiplyl(long double, long double);
long double reference_remquol(long double, long double, int*);
long double reference_lgamma_rl(long double, int*);


int reference_isequall(long double, long double);
int reference_isfinitel(long double);
int reference_isgreaterl(long double, long double);
int reference_isgreaterequall(long double, long double);
int reference_isinfl(long double);
int reference_islessl(long double, long double);
int reference_islessequall(long double, long double);
int reference_islessgreaterl(long double, long double);
int reference_isnanl(long double);
int reference_isnormall(long double);
int reference_isnotequall(long double, long double);
int reference_isorderedl(long double, long double);
int reference_isunorderedl(long double, long double);
int reference_signbitl(long double);

long double reference_acoshl(long double x);
long double reference_asinhl(long double x);
long double reference_atanhl(long double x);
long double reference_cbrtl(long double x);
long double reference_copysignl(long double x, long double y);
long double reference_exp10l(long double);
long double reference_exp2l(long double x);
long double reference_expm1l(long double x);
long double reference_fmaxl(long double x, long double y);
long double reference_fminl(long double x, long double y);
long double reference_hypotl(long double x, long double y);
long double reference_lgammal(long double x);
int reference_ilogbl(long double);
long double reference_log2l(long double x);
long double reference_log1pl(long double x);
long double reference_logbl(long double x);
long double reference_maxmagl(long double x, long double y);
long double reference_minmagl(long double x, long double y);
long double reference_nanl(cl_ulong x);
long double reference_reciprocall(long double x);
long double reference_remainderl(long double x, long double y);
long double reference_rintl(long double x);
long double reference_roundl(long double x);
long double reference_truncl(long double x);
long double reference_floorl(long double x);
long double reference_fmodl(long double x, long double y);
long double reference_frexpl(long double x, int* n);
long double reference_ldexpl(long double x, int n);

long double reference_assignmentl(long double x);
int reference_notl(long double x);

#endif
