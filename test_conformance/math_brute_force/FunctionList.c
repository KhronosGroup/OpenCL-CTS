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

#define STRINGIFY( _s)                  #_s

#define ENTRY( _name, _ulp, _embedded_ulp, _rmode, _type )                           { STRINGIFY(_name), STRINGIFY(_name),                 {reference_##_name}, {reference_##_name##l}, _ulp, _ulp, _embedded_ulp, _rmode, _type }
#define HALF_ENTRY( _name, _ulp, _embedded_ulp, _rmode, _type )                        { "half_" STRINGIFY(_name), "half_" STRINGIFY(_name), {reference_##_name}, {NULL},                    _ulp, _ulp, _embedded_ulp, _rmode, _type }
#define OPERATOR_ENTRY(_name, _operator, _ulp, _embedded_ulp, _rmode, _type)        { STRINGIFY(_name), _operator,                        {reference_##_name}, {reference_##_name##l}, _ulp, _ulp, _embedded_ulp, _rmode, _type }

#if defined( __cplusplus )
    extern "C" {
#endif
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
#if defined( __cplusplus)
    }
#endif

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
                                    ENTRY( cos,                   4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( cosh,                  4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( cospi,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
//                                  ENTRY( erfc,                  16.0f,         16.0f,         FTZ_OFF,     unaryF), //disabled for 1.0 due to lack of reference implementation
//                                  ENTRY( erf,                   16.0f,         16.0f,         FTZ_OFF,     unaryF), //disabled for 1.0 due to lack of reference implementation
                                    ENTRY( exp,                   3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( exp2,                  3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( exp10,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( expm1,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( fabs,                  0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( fdim,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( floor,                 0.0f,         0.0f,       FTZ_OFF,     unaryF),
                                    ENTRY( fma,                   0.0f,         0.0f,         FTZ_OFF,     ternaryF),
                                    ENTRY( fmax,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( fmin,                  0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( fmod,                  0.0f,         0.0f,         FTZ_OFF,     binaryF ),
                                    ENTRY( fract,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results),
                                    ENTRY( frexp,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results_i),
                                    ENTRY( hypot,                 4.0f,         4.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( ilogb,                 0.0f,         0.0f,         FTZ_OFF,     i_unaryF),
                                    ENTRY( isequal,               0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isfinite,               0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isgreater,              0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isgreaterequal,         0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isinf,                 0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isless,                 0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( islessequal,         0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( islessgreater,         0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isnan,                 0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isnormal,             0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( isnotequal,             0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isordered,             0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( isunordered,         0.0f,         0.0f,         FTZ_OFF,     macro_binaryF),
                                    ENTRY( ldexp,                 0.0f,         0.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( lgamma,                 INFINITY,    INFINITY,     FTZ_OFF,     unaryF),
                                    ENTRY( lgamma_r,             INFINITY,    INFINITY,     FTZ_OFF,     unaryF_two_results_i),
                                    ENTRY( log,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( log2,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( log10,                 3.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( log1p,                 2.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( logb,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( mad,                 INFINITY,     INFINITY,     FTZ_OFF,     mad_function),
                                    ENTRY( maxmag,              0.0f,       0.0f,       FTZ_OFF,    binaryF ),
                                    ENTRY( minmag,              0.0f,       0.0f,       FTZ_OFF,    binaryF ),
                                    ENTRY( modf,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_two_results ),
                                    ENTRY( nan,                 0.0f,         0.0f,         FTZ_OFF,     unaryF_u),
                                    ENTRY( nextafter,             0.0f,         0.0f,         FTZ_OFF,     binaryF_nextafter),
                                    ENTRY( pow,                 16.0f,         16.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( pown,                 16.0f,         16.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( powr,                 16.0f,         16.0f,         FTZ_OFF,     binaryF),
//                                  ENTRY( reciprocal,             1.0f,         1.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( remainder,             0.0f,         0.0f,         FTZ_OFF,     binaryF),
                                    ENTRY( remquo,                 0.0f,         0.0f,         FTZ_OFF,     binaryF_two_results_i),
                                    ENTRY( rint,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( rootn,                 16.0f,         16.0f,         FTZ_OFF,     binaryF_i),
                                    ENTRY( round,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( rsqrt,                 2.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( signbit,             0.0f,         0.0f,         FTZ_OFF,     macro_unaryF),
                                    ENTRY( sin,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( sincos,                 4.0f,         4.0f,         FTZ_OFF,     unaryF_two_results),
                                    ENTRY( sinh,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( sinpi,                 4.0f,         4.0f,         FTZ_OFF,     unaryF),
                                    { "sqrt", "sqrt", {reference_sqrt}, {reference_sqrtl}, 3.0f, 0.0f,         4.0f, FTZ_OFF, unaryF },
                                    { "sqrt_cr", "sqrt", {reference_sqrt}, {reference_sqrtl}, 0.0f, 0.0f,         0.0f, FTZ_OFF, unaryF },
                                    ENTRY( tan,                 5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( tanh,                 5.0f,         5.0f,         FTZ_OFF,     unaryF),
                                    ENTRY( tanpi,                 6.0f,         6.0f,         FTZ_OFF,     unaryF),
//                                    ENTRY( tgamma,                 16.0f,         16.0f,         FTZ_OFF,     unaryF), // Commented this out until we can be sure this requirement is realistic
                                    ENTRY( trunc,                 0.0f,         0.0f,         FTZ_OFF,     unaryF),

                                    HALF_ENTRY( cos,            8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( divide,         8192.0f,     8192.0f,     FTZ_ON,     binaryF),
                                    HALF_ENTRY( exp,            8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( exp2,           8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( exp10,          8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( log,            8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( log2,           8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( log10,          8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( powr,           8192.0f,     8192.0f,     FTZ_ON,     binaryF),
                                    HALF_ENTRY( recip,          8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( rsqrt,          8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( sin,            8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( sqrt,           8192.0f,     8192.0f,     FTZ_ON,     unaryF),
                                    HALF_ENTRY( tan,            8192.0f,     8192.0f,     FTZ_ON,     unaryF),

                                    // basic operations
                                    OPERATOR_ENTRY( add, "+",         0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    OPERATOR_ENTRY( subtract, "-",     0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    { "divide", "/", {reference_divide}, {reference_dividel}, 2.5f, 0.0f,         3.0f, FTZ_OFF, binaryOperatorF },
                                    { "divide_cr", "/", {reference_divide}, {reference_dividel}, 0.0f, 0.0f,         0.0f, FTZ_OFF, binaryOperatorF },
                                    OPERATOR_ENTRY( multiply, "*",     0.0f,         0.0f,     FTZ_OFF,     binaryOperatorF),
                                    OPERATOR_ENTRY( assignment, "", 0.0f,       0.0f,     FTZ_OFF,     unaryF),        // A simple copy operation
                                    OPERATOR_ENTRY( not, "!",       0.0f,       0.0f,   FTZ_OFF,    macro_unaryF),
                                };

const size_t functionListCount = sizeof( functionList ) / sizeof( functionList[0] );


