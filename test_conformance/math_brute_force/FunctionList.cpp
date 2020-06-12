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
#include "FunctionList.h"
#include "reference_math.h"

#define FTZ_ON  1
#define FTZ_OFF 0
#define EXACT    0.0f
#define RELAXED_ON 1
#define RELAXED_OFF 0

#define STRINGIFY( _s)                  #_s

// Only use ulps information in spir test
#ifdef FUNCTION_LIST_ULPS_ONLY

#define ENTRY(      _name, _ulp, _embedded_ulp, _rmode, _type )                 { STRINGIFY(_name), STRINGIFY(_name),                 {NULL}, {NULL}, {NULL}, _ulp, _ulp, _embedded_ulp, INFINITY,     _rmode, RELAXED_OFF, _type }
#define ENTRY_EXT(  _name, _ulp, _embedded_ulp, _relaxed_ulp, _rmode, _type )   { STRINGIFY(_name), STRINGIFY(_name),                 {NULL}, {NULL}, {NULL}, _ulp, _ulp, _embedded_ulp, _relaxed_ulp, _rmode, RELAXED_ON,  _type }
#define HALF_ENTRY( _name, _ulp, _embedded_ulp, _rmode, _type )                 { "half_" STRINGIFY(_name), "half_" STRINGIFY(_name), {NULL}, {NULL}, {NULL}, _ulp, _ulp, _embedded_ulp, INFINITY,     _rmode, RELAXED_OFF, _type }
#define OPERATOR_ENTRY(_name, _operator, _ulp, _embedded_ulp, _rmode, _type)    { STRINGIFY(_name), _operator,                        {NULL}, {NULL}, {NULL}, _ulp, _ulp, _embedded_ulp, INFINITY,     _rmode, RELAXED_OFF, _type }
#define unaryF                NULL
#define i_unaryF              NULL
#define unaryF_u              NULL
#define macro_unaryF          NULL
#define binaryF               NULL
#define binaryF_nextafter     NULL
#define binaryOperatorF       NULL
#define binaryF_i             NULL
#define macro_binaryF         NULL
#define ternaryF              NULL
#define unaryF_two_results    NULL
#define unaryF_two_results_i  NULL
#define binaryF_two_results_i NULL
#define mad_function          NULL

#define reference_sqrt        NULL
#define reference_sqrtl       NULL
#define reference_divide      NULL
#define reference_dividel     NULL
#define reference_relaxed_divide NULL

#else // FUNCTION_LIST_ULPS_ONLY

#define ENTRY(      _name, _ulp, _embedded_ulp, _rmode, _type )                 { STRINGIFY(_name), STRINGIFY(_name),                 {(void*)reference_##_name}, {(void*)reference_##_name##l}, {(void*)reference_##_name},           _ulp, _ulp, _embedded_ulp, INFINITY,     _rmode, RELAXED_OFF, _type }
#define ENTRY_EXT(  _name, _ulp, _embedded_ulp, _relaxed_ulp, _rmode, _type )   { STRINGIFY(_name), STRINGIFY(_name),                 {(void*)reference_##_name}, {(void*)reference_##_name##l}, {(void*)reference_##relaxed_##_name}, _ulp, _ulp, _embedded_ulp, _relaxed_ulp, _rmode, RELAXED_ON,  _type }
#define HALF_ENTRY( _name, _ulp, _embedded_ulp, _rmode, _type )                 { "half_" STRINGIFY(_name), "half_" STRINGIFY(_name), {(void*)reference_##_name}, {NULL}, {NULL},                   _ulp, _ulp, _embedded_ulp, INFINITY, _rmode, RELAXED_OFF, _type }
#define OPERATOR_ENTRY(_name, _operator, _ulp, _embedded_ulp, _rmode, _type)    { STRINGIFY(_name), _operator,                        {(void*)reference_##_name}, {(void*)reference_##_name##l}, {NULL},                               _ulp, _ulp, _embedded_ulp, INFINITY,     _rmode, RELAXED_OFF, _type }

extern const vtbl _unary;               // float foo( float )
extern const vtbl _unary_u;             // float foo( uint ),  double foo( ulong )
extern const vtbl _i_unary;             // int foo( float )
extern const vtbl _macro_unary;         // int foo( float ),  returns {0,1} for scalar, { 0, -1 } for vector
extern const vtbl _binary;              // float foo( float, float )
extern const vtbl _binary_nextafter;    // float foo( float, float ), special handling for nextafter
extern const vtbl _binary_operator;     // float .op. float
extern const vtbl _macro_binary;        // int foo( float, float ), returns {0,1} for scalar, { 0, -1 } for vector
extern const vtbl _binary_i;            // float foo( float, int )
extern const vtbl _ternary;             // float foo( float, float, float )
extern const vtbl _unary_two_results;   // float foo( float, float * )
extern const vtbl _unary_two_results_i; // float foo( float, int * )
extern const vtbl _binary_two_results_i; // float foo( float, float, int * )
extern const vtbl _mad_tbl;             // float mad( float, float, float )

#define unaryF &_unary
#define i_unaryF &_i_unary
#define unaryF_u  &_unary_u
#define macro_unaryF &_macro_unary
#define binaryF &_binary
#define binaryF_nextafter &_binary_nextafter
#define binaryOperatorF &_binary_operator
#define binaryF_i &_binary_i
#define macro_binaryF &_macro_binary
#define ternaryF &_ternary
#define unaryF_two_results  &_unary_two_results
#define unaryF_two_results_i  &_unary_two_results_i
#define binaryF_two_results_i  &_binary_two_results_i
#define mad_function        &_mad_tbl

#endif // FUNCTION_LIST_ULPS_ONLY

const Func  functionList[] = {
                                    ENTRY( acos,                  4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( acosh,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( acospi,                5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( asin,                  4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( asinh,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( asinpi,                5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( atan,                  5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( atanh,                 5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( atanpi,                5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( atan2,                 6.0f,         6.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( atan2pi,               6.0f,         6.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( cbrt,                  2.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( ceil,                  0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( copysign,              0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY_EXT( cos,               4.0f,         4.0f,        0.00048828125f,        FTZ_OFF,     unaryF), //relaxed ulp 2^-11
                                    ENTRY( cosh,                  4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( cospi,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
//                                  ENTRY( erfc,                  16.0f,         16.0f,         FTZ_OFF,     unaryF), //disabled for 1.0 due to lack of reference implementation
//                                  ENTRY( erf,                   16.0f,         16.0f,         FTZ_OFF,     unaryF), //disabled for 1.0 due to lack of reference implementation
                                    ENTRY_EXT( exp,               3.0f,         4.0f,       3.0f,       FTZ_OFF,    unaryF), //relaxed error is actually overwritten in unary.c as it is 3+floor(fabs(2*x))
                                    ENTRY_EXT( exp2,              3.0f,         4.0f,       3.0f,       FTZ_OFF,    unaryF), //relaxed error is actually overwritten in unary.c as it is 3+floor(fabs(2*x))
                                    ENTRY_EXT( exp10,             3.0f,         4.0f,       8192.0f,    FTZ_OFF,    unaryF), //relaxed error is actually overwritten in unary.c as it is 3+floor(fabs(2*x)) in derived mode,
                                    // in non-derived mode it uses the ulp error for half_exp10.
                                    ENTRY( expm1,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( fabs,                  0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( fdim,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( floor,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( fma,                   0.0f,         0.0f,         FTZ_OFF,     ternaryF),
                                    ENTRY( fmax,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( fmin,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( fmod,                  0.0f,         0.0f,         FTZ_OFF,     binaryF ),
                                    ENTRY( fract,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results),
                                    ENTRY( frexp,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results_i),
                                    ENTRY( hypot,                 4.0f,         4.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( ilogb,                 0.0f,         0.0f,         FTZ_OFF,     i_unaryF),
                                    ENTRY( isequal,               0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isfinite,              0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isgreater,             0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isgreaterequal,        0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isinf,                 0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isless,                0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( islessequal,           0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( islessgreater,         0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isnan,                 0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isnormal,              0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isnotequal,            0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isordered,             0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isunordered,           0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( ldexp,                 0.0f,         0.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( lgamma,            INFINITY,     INFINITY,         FTZ_OFF,     unaryF),
                                    ENTRY( lgamma_r,          INFINITY,     INFINITY,         FTZ_OFF,     unaryF_two_results_i),
                                    ENTRY_EXT( log,               3.0f,         4.0f,       4.76837158203125e-7f,   FTZ_OFF,    unaryF), //relaxed ulp 2^-21
                                    ENTRY_EXT( log2,              3.0f,         4.0f,       4.76837158203125e-7f,   FTZ_OFF,    unaryF), //relaxed ulp 2^-21
                                    ENTRY( log10,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( log1p,                 2.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( logb,                  0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY_EXT( mad,           INFINITY,     INFINITY,        INFINITY,    FTZ_OFF,    mad_function), //in fast-relaxed-math mode it has to be either exactly rounded fma or exactly rounded a*b+c
                                    ENTRY( maxmag,                0.0f,         0.0f,         FTZ_OFF,    binaryF ),
                                    ENTRY( minmag,                0.0f,         0.0f,         FTZ_OFF,    binaryF ),
                                    ENTRY( modf,                  0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results ),
                                    ENTRY( nan,                   0.0f,         0.0f,         FTZ_OFF,     unaryF_u),
                                    ENTRY( nextafter,             0.0f,         0.0f,         FTZ_OFF,     binaryF_nextafter),
                                    ENTRY_EXT( pow,              16.0f,        16.0f,         8192.0f,     FTZ_OFF,    binaryF), //in derived mode the ulp error is calculated as exp2(y*log2(x)) and in non-derived it is the same as half_pow
                                    ENTRY( pown,                 16.0f,        16.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( powr,                 16.0f,        16.0f,         FTZ_OFF,     binaryF),
//                                  ENTRY( reciprocal,            1.0f,         1.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( remainder,             0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( remquo,                0.0f,         0.0f,         FTZ_OFF,     binaryF_two_results_i),
                                    ENTRY( rint,                  0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( rootn,                16.0f,        16.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( round,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( rsqrt,                 2.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( signbit,               0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY_EXT( sin,               4.0f,         4.0f,  0.00048828125f,     FTZ_OFF,    unaryF), //relaxed ulp 2^-11
                                    ENTRY_EXT( sincos,            4.0f,         4.0f,  0.00048828125f,     FTZ_OFF,    unaryF_two_results), //relaxed ulp 2^-11
                                    ENTRY( sinh,                  4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( sinpi,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    { "sqrt", "sqrt",     {(void*)reference_sqrt}, {(void*)reference_sqrtl}, {NULL}, 3.0f, 0.0f,    4.0f, INFINITY, FTZ_OFF, RELAXED_OFF, unaryF },
                                    { "sqrt_cr", "sqrt",  {(void*)reference_sqrt}, {(void*)reference_sqrtl}, {NULL}, 0.0f, 0.0f,    0.0f, INFINITY, FTZ_OFF, RELAXED_OFF, unaryF },
                                    ENTRY_EXT( tan,               5.0f,         5.0f,         8192.0f,    FTZ_OFF,     unaryF), //in derived mode it the ulp error is calculated as sin/cos and in non-derived mode it is the same as half_tan.
                                    ENTRY( tanh,                  5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( tanpi,                 6.0f,         6.0f,         FTZ_OFF,     unaryF),
//                                    ENTRY( tgamma,                 16.0f,         16.0f,         FTZ_OFF,     unaryF), // Commented this out until we can be sure this requirement is realistic
                                    ENTRY( trunc,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),

                                    HALF_ENTRY( cos,           8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( divide,        8192.0f,      8192.0f,          FTZ_ON,     binaryF),
                                    HALF_ENTRY( exp,           8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( exp2,          8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( exp10,         8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( log,           8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( log2,          8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( log10,         8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( powr,          8192.0f,      8192.0f,          FTZ_ON,     binaryF),
                                    HALF_ENTRY( recip,         8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( rsqrt,         8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( sin,           8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( sqrt,          8192.0f,      8192.0f,          FTZ_ON,     unaryF),
                                    HALF_ENTRY( tan,           8192.0f,      8192.0f,          FTZ_ON,     unaryF),

                                    // basic operations
                                    OPERATOR_ENTRY( add, "+",         0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    OPERATOR_ENTRY( subtract, "-",     0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    { "divide", "/",  {(void*)reference_divide}, {(void*)reference_dividel}, {(void*)reference_relaxed_divide}, 2.5f, 0.0f,         3.0f, 2.5f, FTZ_OFF, RELAXED_ON, binaryOperatorF },
                                    { "divide_cr", "/",  {(void*)reference_divide}, {(void*)reference_dividel}, {(void*)reference_relaxed_divide}, 0.0f, 0.0f,         0.0f, 0.f, FTZ_OFF, RELAXED_OFF, binaryOperatorF },
                                    OPERATOR_ENTRY( multiply, "*",     0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    OPERATOR_ENTRY( assignment, "", 0.0f,       0.0f,     FTZ_OFF,     unaryF),        // A simple copy operation
                                    OPERATOR_ENTRY( not, "!",       0.0f,       0.0f,   FTZ_OFF,    macro_unaryF),
                                };

const size_t functionListCount = sizeof( functionList ) / sizeof( functionList[0] );
