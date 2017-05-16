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
#include <math.h>
#include <limits.h>

#if !defined(_WIN32)
#include <string.h>
#include <stdint.h>
#endif

#include <float.h>
#include "Utility.h"

#if defined( __SSE__ ) || (defined( _MSC_VER ) && (defined(_M_IX86) || defined(_M_X64)))
    #include <xmmintrin.h>
#endif
#if defined( __SSE2__ ) || (defined( _MSC_VER ) && (defined(_M_IX86) || defined(_M_X64)))
    #include <emmintrin.h>
#endif

#ifndef M_PI
    #define M_PI    3.14159265358979323846264338327950288
#endif

#ifndef M_PI_4
    #define M_PI_4 (M_PI/4)
#endif

#define EVALUATE( x )       x
#define CONCATENATE(x, y)  x ## EVALUATE(y)


// Declare Classification macros for non-C99 platforms
#ifndef isinf
    #define isinf(x)    (	sizeof (x) == sizeof(float )	?	fabsf(x) == INFINITY  	\
                        :	sizeof (x) == sizeof(double)	?	fabs(x) == INFINITY  	\
                        :	fabsl(x) == INFINITY)
#endif

#ifndef isfinite
    #define isfinite(x) (	sizeof (x) == sizeof(float )	?	fabsf(x) < INFINITY  	\
                        :	sizeof (x) == sizeof(double)	?	fabs(x) < INFINITY  	\
                        :	fabsl(x) < INFINITY)
#endif

#ifndef isnan
    #define isnan(_a)       ( (_a) != (_a) )
#endif

#ifdef __MINGW32__
    #undef isnormal
#endif

#ifndef isnormal
    #define isnormal(x) (	sizeof (x) == sizeof(float )	?	(fabsf(x) < INFINITY && fabsf(x) >= FLT_MIN) 	\
                        :	sizeof (x) == sizeof(double)	?	(fabs(x) < INFINITY && fabs(x) >= DBL_MIN) 	\
                        :	(fabsl(x) < INFINITY && fabsl(x) >= LDBL_MIN)   )
#endif

#ifndef islessgreater
    // Note: Non-C99 conformant. This will trigger floating point exceptions. We don't care about that here.
    #define islessgreater( _x, _y )     ( (_x) < (_y) || (_x) > (_y) )
#endif

#pragma STDC FP_CONTRACT OFF
static void __log2_ep(double *hi, double *lo, double x);

typedef union
{
	uint64_t i;
	double d;
}uint64d_t;

static const uint64d_t _CL_NAN = { 0x7ff8000000000000ULL };

#define cl_make_nan() _CL_NAN.d

static double reduce1( double x );
static double reduce1( double x )
{
    if( fabs(x) >= MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53) )
    {
        if( fabs(x) == INFINITY )
            return cl_make_nan();

        return 0.0; //we patch up the sign for sinPi and cosPi later, since they need different signs
    }

    // Find the nearest multiple of 2
    const double r = copysign( MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53), x );
    double z = x + r;
    z -= r;

    // subtract it from x. Value is now in the range -1 <= x <= 1
    return x - z;    
}

/*
static double reduceHalf( double x );
static double reduceHalf( double x )
{
    if( fabs(x) >= MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52) ) 
    {
        if( fabs(x) == INFINITY )
            return cl_make_nan();
            
        return 0.0; //we patch up the sign for sinPi and cosPi later, since they need different signs
    }

    // Find the nearest multiple of 1
    const double r = copysign( MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52), x );
    double z = x + r;
    z -= r;

    // subtract it from x. Value is now in the range -0.5 <= x <= 0.5
    return x - z;
}
*/

double reference_acospi( double x) {  return reference_acos( x ) / M_PI;    }
double reference_asinpi( double x) {  return reference_asin( x ) / M_PI;    }
double reference_atanpi( double x) {  return reference_atan( x ) / M_PI;    }
double reference_atan2pi( double y, double x ) { return reference_atan2( y, x) / M_PI; }
double reference_cospi( double x) 
{   
    if( reference_fabs(x) >= MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52) )
    {
        if( reference_fabs(x) == INFINITY )
            return cl_make_nan();

        //Note this probably fails for odd values between 0x1.0p52 and 0x1.0p53.
        //However, when starting with single precision inputs, there will be no odd values.

        return 1.0; 
    }

    x = reduce1(x+0.5);     
        
    // reduce to [-0.5, 0.5]
    if( x < -0.5 )
        x = -1 - x;
    else if ( x > 0.5 )
        x = 1 - x;

    // cosPi zeros are all +0
    if( x == 0.0 )
        return 0.0;

    return reference_sin( x * M_PI );   
}

double reference_divide( double x, double y ) { return x / y; }

// Add a + b. If the result modulo overflowed, write 1 to *carry, otherwise 0
static inline cl_ulong  add_carry( cl_ulong a, cl_ulong b, cl_ulong *carry )
{
    cl_ulong result = a + b;
    *carry = result < a;
    return result;
}

// Subtract a - b. If the result modulo overflowed, write 1 to *carry, otherwise 0
static inline cl_ulong  sub_carry( cl_ulong a, cl_ulong b, cl_ulong *carry )
{
    cl_ulong result = a - b;
    *carry = result > a;
    return result;
}

static float fallback_frexpf( float x, int *iptr )
{
    cl_uint u, v;
    float fu, fv;

    memcpy( &u, &x, sizeof(u));

    cl_uint exponent = u &  0x7f800000U;
    cl_uint mantissa = u & ~0x7f800000U;

    // add 1 to the exponent
    exponent += 0x00800000U;
    
    if( (cl_int) exponent < (cl_int) 0x01000000 )
    { // subnormal, NaN, Inf
        mantissa |= 0x3f000000U;
        
        v = mantissa & 0xff800000U;
        u = mantissa;
        memcpy( &fv, &v, sizeof(v));
        memcpy( &fu, &u, sizeof(u));
        
        fu -= fv;

        memcpy( &v, &fv, sizeof(v));
        memcpy( &u, &fu, sizeof(u));
        
        exponent = u &  0x7f800000U;
        mantissa = u & ~0x7f800000U;
        
        *iptr = (exponent >> 23) + (-126 + 1 -126);
        u = mantissa | 0x3f000000U;
        memcpy( &fu, &u, sizeof(u));
        return fu;
    }
    
    *iptr = (exponent >> 23) - 127;
    u = mantissa | 0x3f000000U;
    memcpy( &fu, &u, sizeof(u));
    return fu;
}

static inline int extractf( float, cl_uint * );
static inline int extractf( float x, cl_uint *mant )
{
    static float (*frexppf)(float, int*) = NULL;
    int e;
    
    // verify that frexp works properly
    if( NULL == frexppf )
    {
        if( 0.5f == frexpf( MAKE_HEX_FLOAT(0x1.0p-130f, 0x1L, -130), &e ) && e == -129 )
            frexppf = frexpf;
        else
            frexppf = fallback_frexpf;
    }

    *mant = (cl_uint) (MAKE_HEX_FLOAT(0x1.0p32f, 0x1L, 32) * fabsf( frexppf( x, &e )));         
    return e - 1;
}

// Shift right by shift bits. Any bits lost on the right side are bitwise OR'd together and ORd into the LSB of the result
static inline void shift_right_sticky_64( cl_ulong *p, int shift );
static inline void shift_right_sticky_64( cl_ulong *p, int shift )
{
    cl_ulong sticky = 0;
    cl_ulong r = *p;
    
    // C doesn't handle shifts greater than the size of the variable dependably
    if( shift >= 64 )
    {
        sticky |= (0 != r);
        r = 0;
    }
    else
    {
        sticky |= (0 != (r << (64-shift)));
        r >>= shift;
    }

    *p = r | sticky;
}

// Add two 64 bit mantissas. Bits that are below the LSB of the result are OR'd into the LSB of the result
static inline void add64( cl_ulong *p, cl_ulong c, int *exponent );
static inline void add64( cl_ulong *p, cl_ulong c, int *exponent )
{
    cl_ulong carry;
    c = add_carry(c, *p, &carry);
    if( carry )
    {
        carry = c & 1;                              // set aside sticky bit
        c >>= 1;                                    // right shift to deal with overflow
        c |= carry | 0x8000000000000000ULL;         // or in carry bit, and sticky bit. The latter is to prevent rounding from believing we are exact half way case
        *exponent = *exponent + 1;                  // adjust exponent
    }
    
    *p = c;
}

// IEEE-754 round to nearest, ties to even rounding
static float round_to_nearest_even_float( cl_ulong p, int exponent );
static float round_to_nearest_even_float( cl_ulong p, int exponent )
{
    union{ cl_uint u; cl_float d;} u;
    
    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if( exponent > 127 )
    {
        volatile float r = exponent * CL_FLT_MAX;       // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if( r > CL_FLT_MAX )
            return INFINITY;

        return r;
    }
    if( exponent == -150 && p > 0x8000000000000000ULL)      
        return MAKE_HEX_FLOAT(0x1.0p-149f, 0x1L, -149);
    if( exponent <= -150 )       return 0.0f;
        
    //Figure out which bits go where
    int shift = 8 + 32;
    if( exponent < -126 )
    {
        shift -= 126 + exponent;                    // subnormal: shift is not 52
        exponent = -127;                            //            set exponent to 0
    }
    else
        p &= 0x7fffffffffffffffULL;                 // normal: leading bit is implicit. Remove it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(p >> shift) | ((cl_uint) (exponent + 127) << 23);

    // put a representation of the residual bits into hi
    p <<= (64-shift);     
    
    //round to nearest, ties to even  based on the unused portion of p
    if( p < 0x8000000000000000ULL )        return u.d;        
    if( p == 0x8000000000000000ULL )       u.u += u.u & 1U;
    else                                   u.u++;
    
    return u.d;
}

static float round_to_nearest_even_float_ftz( cl_ulong p, int exponent );
static float round_to_nearest_even_float_ftz( cl_ulong p, int exponent )
{
    extern int gCheckTininessBeforeRounding;

    union{ cl_uint u; cl_float d;} u;
    int shift = 8 + 32;
    
    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if( exponent > 127 )        
    {
        volatile float r = exponent * CL_FLT_MAX;       // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if( r > CL_FLT_MAX )
        return INFINITY;

        return r;
    }

    // Deal with FTZ for gCheckTininessBeforeRounding
    if( exponent < (gCheckTininessBeforeRounding - 127) )   
        return 0.0f;
    
    if( exponent == -127 ) // only happens for machines that check tininess after rounding
        p = (p&1) | (p>>1);
    else
        p &= 0x7fffffffffffffffULL;     // normal: leading bit is implicit. Remove it.

    cl_ulong q = p;


    // Assemble the double (round toward zero)
    u.u = (cl_uint)(q >> shift) | ((cl_uint) (exponent + 127) << 23);

    // put a representation of the residual bits into hi
    q <<= (64-shift);     
        
    //round to nearest, ties to even  based on the unused portion of p
    if( q > 0x8000000000000000ULL )        
        u.u++;
    else if( q == 0x8000000000000000ULL )       
        u.u += u.u & 1U;

    // Deal with FTZ for ! gCheckTininessBeforeRounding
    if( 0 == (u.u & 0x7f800000U )  )
        return 0.0f;

    return u.d;
}


// IEEE-754 round toward zero.
static float round_toward_zero_float( cl_ulong p, int exponent );
static float round_toward_zero_float( cl_ulong p, int exponent )
{
    union{ cl_uint u; cl_float d;} u;

    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;
    
    // edges
    if( exponent > 127 )
    {
        volatile float r = exponent * CL_FLT_MAX;       // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if( r > CL_FLT_MAX )
            return CL_FLT_MAX;

        return r;
    }

    if( exponent <= -149 )       
        return 0.0f;
        
    //Figure out which bits go where
    int shift = 8 + 32;
    if( exponent < -126 )
    {
        shift -= 126 + exponent;                    // subnormal: shift is not 52
        exponent = -127;                            //            set exponent to 0
    }
    else
        p &= 0x7fffffffffffffffULL;                 // normal: leading bit is implicit. Remove it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(p >> shift) | ((cl_uint) (exponent + 127) << 23);

    return u.d;
}

static float round_toward_zero_float_ftz( cl_ulong p, int exponent );
static float round_toward_zero_float_ftz( cl_ulong p, int exponent )
{
    extern int gCheckTininessBeforeRounding;

    union{ cl_uint u; cl_float d;} u;
    int shift = 8 + 32;
    
    // If mantissa is zero, return 0.0f
    if (p == 0) return 0.0f;

    // edges
    if( exponent > 127 )        
    {
        volatile float r = exponent * CL_FLT_MAX;       // signal overflow

        // attempt to fool the compiler into not optimizing the above line away
        if( r > CL_FLT_MAX )
            return CL_FLT_MAX;

        return r;
    }

    // Deal with FTZ for gCheckTininessBeforeRounding
    if( exponent < -126 )
        return 0.0f;
            
    cl_ulong q = p &= 0x7fffffffffffffffULL;     // normal: leading bit is implicit. Remove it.

    // Assemble the double (round toward zero)
    u.u = (cl_uint)(q >> shift) | ((cl_uint) (exponent + 127) << 23);

    // put a representation of the residual bits into hi
    q <<= (64-shift);     
        
    return u.d;
}

// Subtract two significands. 
static inline void sub64( cl_ulong *c, cl_ulong p, cl_uint *signC, int *expC );
static inline void sub64( cl_ulong *c, cl_ulong p, cl_uint *signC, int *expC )
{
    cl_ulong carry;
    p = sub_carry( *c, p, &carry );
    
    if( carry )
    {
        *signC ^= 0x80000000U;
        p = -p;
    }
    
    // normalize
    if( p )
    {
        int shift = 32;
        cl_ulong test = 1ULL << 32;
        while( 0 == (p & 0x8000000000000000ULL))
        {
            if( p < test )
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
        *signC = 0;     // IEEE rules say a - a = +0 for all rounding modes except -inf
    }

    *c = p;
}


float reference_fma( float a, float b, float c, int shouldFlush )
{
    static const cl_uint kMSB = 0x80000000U;
    
    // Make bits accessible
    union{ cl_uint u; cl_float d; } ua; ua.d = a;
    union{ cl_uint u; cl_float d; } ub; ub.d = b;
    union{ cl_uint u; cl_float d; } uc; uc.d = c;
    
    // deal with Nans, infinities and zeros
    if( isnan( a ) || isnan( b ) || isnan(c)    || 
        isinf( a ) || isinf( b ) || isinf(c)    || 
        0 == ( ua.u & ~kMSB)                ||  // a == 0, defeat host FTZ behavior
        0 == ( ub.u & ~kMSB)                ||  // b == 0, defeat host FTZ behavior
        0 == ( uc.u & ~kMSB)                )   // c == 0, defeat host FTZ behavior
    {
        FPU_mode_type oldMode;
        RoundingMode oldRoundMode = kRoundToNearestEven;
        if( isinf( c ) && !isinf(a) && !isinf(b) )
            return (c + a) + b;

        if (gIsInRTZMode) 
            oldRoundMode = set_round(kRoundTowardZero, kfloat);

        memset( &oldMode, 0, sizeof( oldMode ) );
        if( shouldFlush )
            ForceFTZ( &oldMode );

        a = (float) reference_multiply( a, b );    // some risk that the compiler will insert a non-compliant fma here on some platforms.
        a = (float) reference_add( a, c );           // We use STDC FP_CONTRACT OFF above to attempt to defeat that.
    
        if( shouldFlush )
            RestoreFPState( &oldMode );

        if( gIsInRTZMode )
            set_round(oldRoundMode, kfloat);
        return a;
    }
    
    // extract exponent and mantissa 
    //   exponent is a standard unbiased signed integer
    //   mantissa is a cl_uint, with leading non-zero bit positioned at the MSB
    cl_uint mantA, mantB, mantC;
    int expA = extractf( a, &mantA );
    int expB = extractf( b, &mantB );
    int expC = extractf( c, &mantC );
    cl_uint signC = uc.u & kMSB;                // We'll need the sign bit of C later to decide if we are adding or subtracting
        
// exact product of A and B
    int exponent = expA + expB;
    cl_uint sign = (ua.u ^ ub.u) & kMSB;
    cl_ulong product = (cl_ulong) mantA * (cl_ulong) mantB;
    
    // renormalize -- 1.m * 1.n yields a number between 1.0 and 3.99999.. 
    //  The MSB might not be set. If so, fix that. Otherwise, reflect the fact that we got another power of two from the multiplication
    if( 0 == (0x8000000000000000ULL & product) )
        product <<= 1;
    else
        exponent++;         // 2**31 * 2**31 gives 2**62. If the MSB was set, then our exponent increased.
    
//infinite precision add 
    cl_ulong addend = (cl_ulong) mantC << 32;
    if( exponent >= expC )
    {
        // Shift C relative to the product so that their exponents match
        if( exponent > expC )
            shift_right_sticky_64( &addend, exponent - expC );

        // Add
        if( sign ^ signC )
            sub64( &product, addend, &sign, &exponent );
        else
            add64( &product, addend, &exponent );
    }
    else 
    {
        // Shift the product relative to C so that their exponents match
        shift_right_sticky_64( &product, expC - exponent );

        // add
        if( sign ^ signC )
            sub64( &addend, product, &signC, &expC );
        else
            add64( &addend, product, &expC );
            
        product = addend;
        exponent = expC;
        sign = signC;
    }

    // round to IEEE result -- we do not do flushing to zero here. That part is handled manually in ternary.c.
    if (gIsInRTZMode)
    {
        if( shouldFlush )
            ua.d = round_toward_zero_float_ftz( product, exponent);
        else
            ua.d = round_toward_zero_float( product, exponent);
    }
    else
    {
        if( shouldFlush )
            ua.d = round_to_nearest_even_float_ftz( product, exponent);
        else
            ua.d = round_to_nearest_even_float( product, exponent);
    }
    
    // Set the sign
    ua.u |= sign;

    return ua.d;
}

double reference_exp10( double x) {   return reference_exp2( x * MAKE_HEX_DOUBLE(0x1.a934f0979a371p+1, 0x1a934f0979a371LL, -51) );    }


int   reference_ilogb( double x )
{
    extern int gDeviceILogb0, gDeviceILogbNaN;
    union { cl_double f; cl_ulong u;} u;

    u.f = (float) x;    
    cl_int exponent = (cl_int) (u.u >> 52) & 0x7ff;
    if( exponent == 0x7ff )
    {
        if( u.u & 0x000fffffffffffffULL )
            return gDeviceILogbNaN;
            
        return CL_INT_MAX;
    }
        
    if( exponent == 0 )
    {   // deal with denormals
        u.f = x * MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64);
        exponent = (cl_int) (u.u >> 52) & 0x7ff;
        if( exponent == 0 )
            return gDeviceILogb0;
        
        return exponent - (1023 + 64);
    }

    return exponent - 1023;
}

double reference_nan( cl_uint x )
{
    union{ cl_uint u; cl_float f; }u;
    u.u = x | 0x7fc00000U;
    return (double) u.f;
}

double reference_maxmag( double x, double y )
{
    double fabsx = fabs(x);
    double fabsy = fabs(y);

    if( fabsx < fabsy )
        return y;

    if( fabsy < fabsx )
        return x;
    
    return reference_fmax( x, y );
}

double reference_minmag( double x, double y )
{
    double fabsx = fabs(x);
    double fabsy = fabs(y);

    if( fabsx > fabsy )
        return y;

    if( fabsy > fabsx )
        return x;
    
    return reference_fmin( x, y );
}

//double my_nextafter( double x, double y ){  return (double) nextafterf( (float) x, (float) y ); }
double reference_mad( double a, double b, double c )
{
    return a * b + c;
}

double reference_recip( double x) {   return 1.0 / x; }
double reference_rootn( double x, int i ) 
{

    //rootn ( x, 0 )  returns a NaN. 
    if( 0 == i )
        return cl_make_nan();

    //rootn ( x, n )  returns a NaN for x < 0 and n is even. 
    if( x < 0 && 0 == (i&1) )
        return cl_make_nan();

    if( x == 0.0 )
    {
        switch( i & 0x80000001 )
        {
            //rootn ( +-0,  n ) is +0 for even n > 0. 
            case 0:
                return 0.0f;

            //rootn ( +-0,  n ) is +-0 for odd n > 0. 
            case 1:
                return x;

            //rootn ( +-0,  n ) is +inf for even n < 0. 
            case 0x80000000:
                return INFINITY;

            //rootn ( +-0,  n ) is +-inf for odd n < 0. 
            case 0x80000001:
                return copysign(INFINITY, x);
        }    
    }
    
    double sign = x;
    x = reference_fabs(x);
    x = reference_exp2( reference_log2(x) / (double) i ); 
    return reference_copysignd( x, sign );
}

double reference_rsqrt( double x) {   return 1.0 / reference_sqrt(x);   }
//double reference_sincos( double x, double *c ){ *c = cos(x); return sin(x); }
double reference_sinpi( double x) 
{   
    double r = reduce1(x); 
        
    // reduce to [-0.5, 0.5]
    if( r < -0.5 )
        r = -1 - r;
    else if ( r > 0.5 )
        r = 1 - r;

    // sinPi zeros have the same sign as x
    if( r == 0.0 )
        return reference_copysignd(0.0, x);

    return reference_sin( r * M_PI );   
}

double reference_tanpi( double x) 
{
    // set aside the sign  (allows us to preserve sign of -0)
    double sign = reference_copysignd( 1.0, x);
    double z = reference_fabs(x);

    // if big and even  -- caution: only works if x only has single precision
    if( z >= MAKE_HEX_DOUBLE(0x1.0p24, 0x1LL, 24) )
    {
        if( z == INFINITY )
            return x - x;       // nan
            
        return reference_copysignd( 0.0, x);   // tanpi ( n ) is copysign( 0.0, n)  for even integers n.
    }
    
    // reduce to the range [ -0.5, 0.5 ]
    double nearest = reference_rint( z );     // round to nearest even places n + 0.5 values in the right place for us
    int i = (int) nearest;          // test above against 0x1.0p24 avoids overflow here
    z -= nearest;                   
    
    //correction for odd integer x for the right sign of zero
    if( (i&1) && z == 0.0 )
        sign = -sign;
    
    // track changes to the sign
    sign *= reference_copysignd(1.0, z);       // really should just be an xor
    z = reference_fabs(z);                    // remove the sign again
    
    // reduce once more
    // If we don't do this, rounding error in z * M_PI will cause us not to return infinities properly
    if( z > 0.25 )
    {
        z = 0.5 - z;
        return sign / reference_tan( z * M_PI );      // use system tan to get the right result
    }
    
    //
    return sign * reference_tan( z * M_PI );          // use system tan to get the right result
}

double reference_pown( double x, int i) { return reference_pow( x, (double) i ); }
double reference_powr( double x, double y ) 
{  
    //powr ( x, y ) returns NaN for x < 0. 
    if( x < 0.0 )
        return cl_make_nan();

    //powr ( x, NaN ) returns the NaN for x >= 0. 
    //powr ( NaN, y ) returns the NaN. 
    if( isnan(x) || isnan(y) )
        return x + y;       // Note: behavior different here than for pow(1,NaN), pow(NaN, 0)
      
    if( x == 1.0 )
    {
        //powr ( +1, +-inf ) returns NaN. 
        if( reference_fabs(y) == INFINITY )
            return cl_make_nan();
        
        //powr ( +1, y ) is 1 for finite y.    (NaN handled above)
        return 1.0;
    }

    if( y == 0.0 )
    {
        //powr ( +inf, +-0 ) returns NaN. 
        //powr ( +-0, +-0 ) returns NaN. 
        if( x == 0.0 || x == INFINITY )
            return cl_make_nan(); 
    
        //powr ( x, +-0 ) is 1 for finite x > 0.  (x <= 0, NaN, INF already handled above)
        return 1.0;
    }
    
    if( x == 0.0 )
    {
        //powr ( +-0, -inf) is +inf. 
        //powr ( +-0, y ) is +inf for finite y < 0. 
        if( y < 0.0 )
            return INFINITY;
            
        //powr ( +-0, y ) is +0 for y > 0.    (NaN, y==0 handled above)
        return 0.0;
    }
    
    // x = +inf  
	if( isinf(x) )
	{		
		if( y < 0 )
			return 0;
		return INFINITY;
	}
	
	double fabsx = reference_fabs(x);
	double fabsy = reference_fabs(y);
	
	//y = +-inf cases
	if( isinf(fabsy) )
	{
		if( y < 0 )
		{
			if( fabsx < 1 )
				return INFINITY;
			return 0;
		}
		if( fabsx < 1 )
			return 0;
		return INFINITY;
	}            
    
	double hi, lo;
	__log2_ep(&hi, &lo, x);
	double prod = y * hi;
	double result = reference_exp2(prod);
	
    return result;
}

double reference_fract( double x, double *ip )
{
    float i;
    float f = modff((float) x, &i );
    if( f < 0.0 )
    {
        f = 1.0f + f;
        i -= 1.0f;
        if( f == 1.0f )
            f = MAKE_HEX_FLOAT(0x1.fffffep-1f, 0x1fffffeL, -25);
    }
    *ip = i;
    return f;
}


//double my_fdim( double x, double y){ return fdimf( (float) x, (float) y ); }
double reference_add( double x, double y )
{ 
    volatile float a = (float) x;
    volatile float b = (float) y;

#if defined( __SSE__ ) || (defined( _MSC_VER ) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss( (float) a );
    __m128 vb = _mm_set_ss( (float) b );
    va = _mm_add_ss( va, vb );
    _mm_store_ss( (float*) &a, va );
#elif defined(__PPC__)
    // Most Power host CPUs do not support the non-IEEE mode (NI) which flushes denorm's to zero.
    // As such, the reference add with FTZ must be emulated in sw.
    if (fpu_control & _FPU_MASK_NI) {
      union{ cl_uint u; cl_float d; } ua; ua.d = a;
      union{ cl_uint u; cl_float d; } ub; ub.d = b;
      cl_uint mantA, mantB;
      cl_ulong addendA, addendB, sum;
      int expA = extractf( a, &mantA );
      int expB = extractf( b, &mantB );
      cl_uint signA = ua.u & 0x80000000U;
      cl_uint signB = ub.u & 0x80000000U;

      // Force matching exponents if an operand is 0
      if (a == 0.0f) {
	expA = expB;
      } else if (b == 0.0f) {
	expB = expA;
      }

      addendA = (cl_ulong)mantA << 32;
      addendB = (cl_ulong)mantB << 32;

      if (expA >= expB) {
        // Shift B relative to the A so that their exponents match
        if( expA > expB )
	  shift_right_sticky_64( &addendB, expA - expB );

        // add
        if( signA ^ signB )
	  sub64( &addendA, addendB, &signA, &expA );
        else
	  add64( &addendA, addendB, &expA );
      } else  {
        // Shift the A relative to B so that their exponents match
        shift_right_sticky_64( &addendA, expB - expA );

        // add
        if( signA ^ signB )
	  sub64( &addendB, addendA, &signB, &expB );
        else
	  add64( &addendB, addendA, &expB );

        addendA = addendB;
        expA = expB;
        signA = signB;
      }

      // round to IEEE result
      if (gIsInRTZMode)	{
	ua.d = round_toward_zero_float_ftz( addendA, expA );
      } else {
	ua.d = round_to_nearest_even_float_ftz( addendA, expA );
      }
      // Set the sign
      ua.u |= signA;
      a = ua.d;
    } else {
      a += b;
    }
#else
    a += b;
#endif
    return (double) a;
 }


double reference_subtract( double x, double y )
{ 
    volatile float a = (float) x;
    volatile float b = (float) y;
#if defined( __SSE__ ) || (defined( _MSC_VER ) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss( (float) a );
    __m128 vb = _mm_set_ss( (float) b );
    va = _mm_sub_ss( va, vb );
    _mm_store_ss( (float*) &a, va );
#else
    a -= b;
#endif
    return a;
}

//double reference_divide( double x, double y ){ return (float) x / (float) y; }
double reference_multiply( double x, double y)
{ 
    volatile float a = (float) x;
    volatile float b = (float) y;
#if defined( __SSE__ ) || (defined( _MSC_VER ) && (defined(_M_IX86) || defined(_M_X64)))
    // defeat x87
    __m128 va = _mm_set_ss( (float) a );
    __m128 vb = _mm_set_ss( (float) b );
    va = _mm_mul_ss( va, vb );
    _mm_store_ss( (float*) &a, va );
#elif defined(__PPC__) 
    // Most Power host CPUs do not support the non-IEEE mode (NI) which flushes denorm's to zero.
    // As such, the reference multiply with FTZ must be emulated in sw.
    if (fpu_control & _FPU_MASK_NI) {
      // extract exponent and mantissa 
      //   exponent is a standard unbiased signed integer
      //   mantissa is a cl_uint, with leading non-zero bit positioned at the MSB
      union{ cl_uint u; cl_float d; } ua; ua.d = a;
      union{ cl_uint u; cl_float d; } ub; ub.d = b;
      cl_uint mantA, mantB;
      int expA = extractf( a, &mantA );
      int expB = extractf( b, &mantB );
        
      // exact product of A and B
      int exponent = expA + expB;
      cl_uint sign = (ua.u ^ ub.u) & 0x80000000U;
      cl_ulong product = (cl_ulong) mantA * (cl_ulong) mantB;
    
      // renormalize -- 1.m * 1.n yields a number between 1.0 and 3.99999.. 
      //  The MSB might not be set. If so, fix that. Otherwise, reflect the fact that we got another power of two from the multiplication
      if( 0 == (0x8000000000000000ULL & product) )
        product <<= 1;
      else
        exponent++;         // 2**31 * 2**31 gives 2**62. If the MSB was set, then our exponent increased.
    
      // round to IEEE result -- we do not do flushing to zero here. That part is handled manually in ternary.c.
      if (gIsInRTZMode)	{
	ua.d = round_toward_zero_float_ftz( product, exponent);
      } else {
	ua.d = round_to_nearest_even_float_ftz( product, exponent);
      }
      // Set the sign
      ua.u |= sign;
      a = ua.d;
    } else {
      a *= b;
    }
#else
    a *= b;
#endif
    return a;
}

/*double my_remquo( double x, double y, int *iptr )
{
    if( isnan(x) || isnan(y) ||
        fabs(x) == INFINITY  ||
        y == 0.0 )
    {
        *iptr = 0;
        return NAN;
    }

    return (double) remquof( (float) x, (float) y, iptr );
}*/
double reference_lgamma_r( double x, int *signp )
{
	// This is not currently tested
	*signp = 0;
	return x;
}


int reference_isequal( double x, double y ){ return x == y; }
int reference_isfinite( double x ){ return 0 != isfinite(x); }
int reference_isgreater( double x, double y ){ return x > y; }
int reference_isgreaterequal( double x, double y ){ return x >= y; }
int reference_isinf( double x ){ return 0 != isinf(x); }
int reference_isless( double x, double y ){ return x < y; }
int reference_islessequal( double x, double y ){ return x <= y; }
int reference_islessgreater( double x, double y ){  return 0 != islessgreater( x, y ); }
int reference_isnan( double x ){ return 0 != isnan( x ); }
int reference_isnormal( double x ){ return 0 != isnormal( (float) x ); }
int reference_isnotequal( double x, double y ){ return x != y; }
int reference_isordered( double x, double y){ return x == x && y == y; }
int reference_isunordered( double x, double y ){ return isnan(x) || isnan( y ); }
int reference_signbit( float x ){ return 0 != signbit( x ); } 

#if 1 // defined( _MSC_VER )

//Missing functions for win32


float reference_copysign( float x, float y )
{
    union { float f; cl_uint u;} ux, uy;
    ux.f = x; uy.f = y;
    ux.u &= 0x7fffffffU;
    ux.u |= uy.u & 0x80000000U;
    return ux.f;
}


double reference_copysignd( double x, double y )
{
    union { double f; cl_ulong u;} ux, uy;
    ux.f = x; uy.f = y;
    ux.u &= 0x7fffffffffffffffULL;
    ux.u |= uy.u & 0x8000000000000000ULL;
    return ux.f;
}

 
double reference_round( double x )
{
    double absx = reference_fabs(x);
    if( absx < 0.5 )
        return reference_copysignd( 0.0, x );

    if( absx < MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53) )
        x = reference_trunc( x + reference_copysignd( 0.5, x ) );
             
    return x;
}

double reference_trunc( double x )
{
    if( fabs(x) < MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53) )
    {
        cl_long l = (cl_long) x;
    
        return reference_copysignd( (double) l, x );
    }

    return x;
}

#ifndef FP_ILOGB0
    #define FP_ILOGB0   INT_MIN
#endif

#ifndef FP_ILOGBNAN
    #define FP_ILOGBNAN   INT_MAX
#endif



double reference_cbrt(double x){ return reference_copysignd( reference_pow( reference_fabs(x), 1.0/3.0 ), x ); }

/*
double reference_scalbn(double x, int i)
{ // suitable for checking single precision scalbnf only

    if( i > 300 )
        return copysign( INFINITY, x);
    if( i < -300 )
        return copysign( 0.0, x);
    
    union{ cl_ulong u; double d;} u;
    u.u = ((cl_ulong) i + 1023) << 52;

    return x * u.d;
}
*/

double reference_rint( double x )
{
    if( reference_fabs(x) < MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52)  )
    {
        double magic = reference_copysignd( MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52), x );
        double rounded = (x + magic) - magic;
        x = reference_copysignd( rounded, x ); 
    }

    return x;
}

double reference_acosh( double x )
{ // not full precision. Sufficient precision to cover float
    if( isnan(x) )
        return x + x;
        
    if( x < 1.0 )
        return cl_make_nan();

    return reference_log( x + reference_sqrt(x + 1) * reference_sqrt(x-1) );
}

double reference_asinh( double x )
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
    if( isnan(x) || isinf(x) )
        return x + x;
        
    double absx = reference_fabs(x);
    if( absx < MAKE_HEX_DOUBLE(0x1.0p-28, 0x1LL, -28) )
        return x;

    double sign = reference_copysignd(1.0, x);

    if( absx > MAKE_HEX_DOUBLE(0x1.0p+28, 0x1LL, 28) )
        return sign * (reference_log( absx ) + 0.693147180559945309417232121458176568);    // log(2)

    if( absx > 2.0 )
        return sign * reference_log( 2.0 * absx + 1.0 / (reference_sqrt( x * x + 1.0 ) + absx));

    return sign * reference_log1p( absx + x*x / (1.0 + reference_sqrt(1.0 + x*x)));
}


double reference_atanh( double x )
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
    if( isnan(x)  )
        return x + x;
    
    double signed_half = reference_copysignd( 0.5, x );
    x = reference_fabs(x);
    if( x > 1.0 )
        return cl_make_nan();
        
    if( x < 0.5 )
        return signed_half * reference_log1p( 2.0 * ( x + x*x / (1-x) ) );
    
    return signed_half * reference_log1p(2.0 * x / (1-x));
}


double reference_exp2( double x )
{ // Note: only suitable for verifying single precision. Doesn't have range of a full double exp2 implementation.
    if( x == 0.0 )
        return 1.0;

    // separate x into fractional and integer parts
    double i = reference_rint( x );        // round to nearest integer

    if( i < -150 )
        return 0.0;
        
    if( i > 129 )
        return INFINITY;

    double f = x - i;            // -0.5 <= f <= 0.5
    
    // find exp2(f)
    // calculate as p(f) = (exp2(f)-1)/f
    //              exp2(f) = f * p(f) + 1
    // p(f) is a minimax polynomial with error within 0x1.c1fd80f0d1ab7p-50
    
    double p = 0.693147180560184539289 + 
               (0.240226506955902863183 + 
               (0.055504108656833424373 + 
               (0.009618129212846484796 + 
               (0.001333355902958566035 + 
               (0.000154034191902497930 + 
               (0.000015252317761038105 + 
               (0.000001326283129417092 + 0.000000102593187638680 * f)*f)*f)*f)*f)*f)*f)*f;
    f *= p;
    f += 1.0;

    // scale by 2 ** i
    union{ cl_ulong u; double d; } u;
    int exponent = (int) i + 1023;
    u.u = (cl_ulong) exponent << 52;
 
    return f * u.d;   
}


double reference_expm1( double x )
{ // Note: only suitable for verifying single precision. Doesn't have range of a full double expm1 implementation. It is only accurate to 47 bits or less.

    // early out for small numbers and NaNs
    if( ! (reference_fabs(x) > MAKE_HEX_DOUBLE(0x1.0p-24, 0x1LL, -24)) )
        return x;

    // early out for large negative numbers
    if( x < -130.0 )
        return -1.0;

    // early out for large positive numbers
    if( x > 100.0 )
        return INFINITY;

    // separate x into fractional and integer parts
    double i = reference_rint( x );        // round to nearest integer
    double f = x - i;            // -0.5 <= f <= 0.5
    
    // reduce f to the range -0.0625 .. f.. 0.0625
    int index = (int) (f * 16.0) + 8;       // 0...16
    
    static const double reduction[17] = { -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 
                                           0.0,
                                          +0.0625, +0.125, +0.1875, +0.25, +0.3125, +0.375, +0.4375, +0.5  };
    
	
    // exponentials[i] = expm1(reduction[i])
    static const double exponentials[17] = {	MAKE_HEX_DOUBLE( -0x1.92e9a0720d3ecp-2, -0x192e9a0720d3ecLL, -54),	MAKE_HEX_DOUBLE( -0x1.6adb1cd9205eep-2, -0x16adb1cd9205eeLL, -54),	
												MAKE_HEX_DOUBLE( -0x1.40373d42ce2e3p-2, -0x140373d42ce2e3LL, -54),	MAKE_HEX_DOUBLE( -0x1.12d35a41ba104p-2, -0x112d35a41ba104LL, -54),	
												MAKE_HEX_DOUBLE( -0x1.c5041854df7d4p-3, -0x1c5041854df7d4LL, -55),	MAKE_HEX_DOUBLE( -0x1.5e25fb4fde211p-3, -0x15e25fb4fde211LL, -55),	
												MAKE_HEX_DOUBLE( -0x1.e14aed893eef4p-4, -0x1e14aed893eef4LL, -56),	MAKE_HEX_DOUBLE( -0x1.f0540438fd5c3p-5, -0x1f0540438fd5c3LL, -57),	
												MAKE_HEX_DOUBLE( 0x0p+0, +0, 0),	
												MAKE_HEX_DOUBLE( 0x1.082b577d34ed8p-4, +0x1082b577d34ed8LL, -56),	MAKE_HEX_DOUBLE( 0x1.10b022db7ae68p-3, +0x110b022db7ae68LL, -55),	
												MAKE_HEX_DOUBLE( 0x1.a65c0b85ac1a9p-3, +0x1a65c0b85ac1a9LL, -55),	MAKE_HEX_DOUBLE( 0x1.22d78f0fa061ap-2, +0x122d78f0fa061aLL, -54),	
												MAKE_HEX_DOUBLE( 0x1.77a45d8117fd5p-2, +0x177a45d8117fd5LL, -54),	MAKE_HEX_DOUBLE( 0x1.d1e944f6fbdaap-2, +0x1d1e944f6fbdaaLL, -54),	
												MAKE_HEX_DOUBLE( 0x1.190048ef6002p-1, +0x1190048ef60020LL, -53),	MAKE_HEX_DOUBLE( 0x1.4c2531c3c0d38p-1, +0x14c2531c3c0d38LL, -53),	
											};
											
    
    f -= reduction[index];
    
    // find expm1(f)
    // calculate as p(f) = (exp(f)-1)/f
    //              expm1(f) = f * p(f) 
    // p(f) is a minimax polynomial with error within 0x1.1d7693618d001p-48 over the range +- 0.0625    
    double p = 0.999999999999998001599 + 
               (0.499999999999839628284 + 
               (0.166666666672817459505 + 
               (0.041666666612283048687 + 
               (0.008333330214567431435 + 
               (0.001389005319303770070 + 0.000198833381525156667 * f)*f)*f)*f)*f)*f;
    f *= p; // expm1( reduced f )

    // expm1(f) = (exmp1( reduced_f) + 1.0) * ( exponentials[index] + 1 ) - 1
    //          =  exmp1( reduced_f) * exponentials[index] + exmp1( reduced_f) + exponentials[index] + 1 -1
    //          =  exmp1( reduced_f) * exponentials[index] + exmp1( reduced_f) + exponentials[index]
    f +=  exponentials[index] + f * exponentials[index];
    
    // scale by e ** i
    int exponent = (int) i;
    if( 0 == exponent )
        return f;       // precise answer for x near 1
        
    // table of e**(i-150)
    static const double exp_table[128+150+1] = 
    {
		MAKE_HEX_DOUBLE( 0x1.82e16284f5ec5p-217, +0x182e16284f5ec5LL, -269),	MAKE_HEX_DOUBLE( 0x1.06e9996332ba1p-215, +0x106e9996332ba1LL, -267),	
		MAKE_HEX_DOUBLE( 0x1.6555cb289e44bp-214, +0x16555cb289e44bLL, -266),	MAKE_HEX_DOUBLE( 0x1.e5ab364643354p-213, +0x1e5ab364643354LL, -265),	
		MAKE_HEX_DOUBLE( 0x1.4a0bd18e64df7p-211, +0x14a0bd18e64df7LL, -263),	MAKE_HEX_DOUBLE( 0x1.c094499cc578ep-210, +0x1c094499cc578eLL, -262),	
		MAKE_HEX_DOUBLE( 0x1.30d759323998cp-208, +0x130d759323998cLL, -260),	MAKE_HEX_DOUBLE( 0x1.9e5278ab1d4cfp-207, +0x19e5278ab1d4cfLL, -259),	
		MAKE_HEX_DOUBLE( 0x1.198fa3f30be25p-205, +0x1198fa3f30be25LL, -257),	MAKE_HEX_DOUBLE( 0x1.7eae636d6144ep-204, +0x17eae636d6144eLL, -256),	
		MAKE_HEX_DOUBLE( 0x1.040f1036f4863p-202, +0x1040f1036f4863LL, -254),	MAKE_HEX_DOUBLE( 0x1.6174e477a895fp-201, +0x16174e477a895fLL, -253),	
		MAKE_HEX_DOUBLE( 0x1.e065b82dd95ap-200, +0x1e065b82dd95a0LL, -252),	MAKE_HEX_DOUBLE( 0x1.4676be491d129p-198, +0x14676be491d129LL, -250),	
		MAKE_HEX_DOUBLE( 0x1.bbb5da5f7c823p-197, +0x1bbb5da5f7c823LL, -249),	MAKE_HEX_DOUBLE( 0x1.2d884eef5fdcbp-195, +0x12d884eef5fdcbLL, -247),	
		MAKE_HEX_DOUBLE( 0x1.99d3397ab8371p-194, +0x199d3397ab8371LL, -246),	MAKE_HEX_DOUBLE( 0x1.1681497ed15b3p-192, +0x11681497ed15b3LL, -244),	
		MAKE_HEX_DOUBLE( 0x1.7a870f597fdbdp-191, +0x17a870f597fdbdLL, -243),	MAKE_HEX_DOUBLE( 0x1.013c74edba307p-189, +0x1013c74edba307LL, -241),	
		MAKE_HEX_DOUBLE( 0x1.5d9ec4ada7938p-188, +0x15d9ec4ada7938LL, -240),	MAKE_HEX_DOUBLE( 0x1.db2edfd20fa7cp-187, +0x1db2edfd20fa7cLL, -239),	
		MAKE_HEX_DOUBLE( 0x1.42eb9f39afb0bp-185, +0x142eb9f39afb0bLL, -237),	MAKE_HEX_DOUBLE( 0x1.b6e4f282b43f4p-184, +0x1b6e4f282b43f4LL, -236),	
		MAKE_HEX_DOUBLE( 0x1.2a42764857b19p-182, +0x12a42764857b19LL, -234),	MAKE_HEX_DOUBLE( 0x1.9560792d19314p-181, +0x19560792d19314LL, -233),	
		MAKE_HEX_DOUBLE( 0x1.137b6ce8e052cp-179, +0x1137b6ce8e052cLL, -231),	MAKE_HEX_DOUBLE( 0x1.766b45dd84f18p-178, +0x1766b45dd84f18LL, -230),	
		MAKE_HEX_DOUBLE( 0x1.fce362fe6e7dp-177, +0x1fce362fe6e7d0LL, -229),	MAKE_HEX_DOUBLE( 0x1.59d34dd8a5473p-175, +0x159d34dd8a5473LL, -227),	
		MAKE_HEX_DOUBLE( 0x1.d606847fc727ap-174, +0x1d606847fc727aLL, -226),	MAKE_HEX_DOUBLE( 0x1.3f6a58b795de3p-172, +0x13f6a58b795de3LL, -224),	
		MAKE_HEX_DOUBLE( 0x1.b2216c6efdac1p-171, +0x1b2216c6efdac1LL, -223),	MAKE_HEX_DOUBLE( 0x1.2705b5b153fb8p-169, +0x12705b5b153fb8LL, -221),	
		MAKE_HEX_DOUBLE( 0x1.90fa1509bd50dp-168, +0x190fa1509bd50dLL, -220),	MAKE_HEX_DOUBLE( 0x1.107df698da211p-166, +0x1107df698da211LL, -218),	
		MAKE_HEX_DOUBLE( 0x1.725ae6e7b9d35p-165, +0x1725ae6e7b9d35LL, -217),	MAKE_HEX_DOUBLE( 0x1.f75d6040aeff6p-164, +0x1f75d6040aeff6LL, -216),	
		MAKE_HEX_DOUBLE( 0x1.56126259e093cp-162, +0x156126259e093cLL, -214),	MAKE_HEX_DOUBLE( 0x1.d0ec7df4f7bd4p-161, +0x1d0ec7df4f7bd4LL, -213),	
		MAKE_HEX_DOUBLE( 0x1.3bf2cf6722e46p-159, +0x13bf2cf6722e46LL, -211),	MAKE_HEX_DOUBLE( 0x1.ad6b22f55db42p-158, +0x1ad6b22f55db42LL, -210),	
		MAKE_HEX_DOUBLE( 0x1.23d1f3e5834ap-156, +0x123d1f3e5834a0LL, -208),	MAKE_HEX_DOUBLE( 0x1.8c9feab89b876p-155, +0x18c9feab89b876LL, -207),	
		MAKE_HEX_DOUBLE( 0x1.0d88cf37f00ddp-153, +0x10d88cf37f00ddLL, -205),	MAKE_HEX_DOUBLE( 0x1.6e55d2bf838a7p-152, +0x16e55d2bf838a7LL, -204),	
		MAKE_HEX_DOUBLE( 0x1.f1e6b68529e33p-151, +0x1f1e6b68529e33LL, -203),	MAKE_HEX_DOUBLE( 0x1.525be4e4e601dp-149, +0x1525be4e4e601dLL, -201),	
		MAKE_HEX_DOUBLE( 0x1.cbe0a45f75eb1p-148, +0x1cbe0a45f75eb1LL, -200),	MAKE_HEX_DOUBLE( 0x1.3884e838aea68p-146, +0x13884e838aea68LL, -198),	
		MAKE_HEX_DOUBLE( 0x1.a8c1f14e2af5dp-145, +0x1a8c1f14e2af5dLL, -197),	MAKE_HEX_DOUBLE( 0x1.20a717e64a9bdp-143, +0x120a717e64a9bdLL, -195),	
		MAKE_HEX_DOUBLE( 0x1.8851d84118908p-142, +0x18851d84118908LL, -194),	MAKE_HEX_DOUBLE( 0x1.0a9bdfb02d24p-140, +0x10a9bdfb02d240LL, -192),	
		MAKE_HEX_DOUBLE( 0x1.6a5bea046b42ep-139, +0x16a5bea046b42eLL, -191),	MAKE_HEX_DOUBLE( 0x1.ec7f3b269efa8p-138, +0x1ec7f3b269efa8LL, -190),	
		MAKE_HEX_DOUBLE( 0x1.4eafb87eab0f2p-136, +0x14eafb87eab0f2LL, -188),	MAKE_HEX_DOUBLE( 0x1.c6e2d05bbcp-135, +0x1c6e2d05bbc000LL, -187),	
		MAKE_HEX_DOUBLE( 0x1.35208867c2683p-133, +0x135208867c2683LL, -185),	MAKE_HEX_DOUBLE( 0x1.a425b317eeacdp-132, +0x1a425b317eeacdLL, -184),	
		MAKE_HEX_DOUBLE( 0x1.1d8508fa8246ap-130, +0x11d8508fa8246aLL, -182),	MAKE_HEX_DOUBLE( 0x1.840fbc08fdc8ap-129, +0x1840fbc08fdc8aLL, -181),	
		MAKE_HEX_DOUBLE( 0x1.07b7112bc1ffep-127, +0x107b7112bc1ffeLL, -179),	MAKE_HEX_DOUBLE( 0x1.666d0dad2961dp-126, +0x1666d0dad2961dLL, -178),	
		MAKE_HEX_DOUBLE( 0x1.e726c3f64d0fep-125, +0x1e726c3f64d0feLL, -177),	MAKE_HEX_DOUBLE( 0x1.4b0dc07cabf98p-123, +0x14b0dc07cabf98LL, -175),	
		MAKE_HEX_DOUBLE( 0x1.c1f2daf3b6a46p-122, +0x1c1f2daf3b6a46LL, -174),	MAKE_HEX_DOUBLE( 0x1.31c5957a47de2p-120, +0x131c5957a47de2LL, -172),	
		MAKE_HEX_DOUBLE( 0x1.9f96445648b9fp-119, +0x19f96445648b9fLL, -171),	MAKE_HEX_DOUBLE( 0x1.1a6baeadb4fd1p-117, +0x11a6baeadb4fd1LL, -169),	
		MAKE_HEX_DOUBLE( 0x1.7fd974d372e45p-116, +0x17fd974d372e45LL, -168),	MAKE_HEX_DOUBLE( 0x1.04da4d1452919p-114, +0x104da4d1452919LL, -166),	
		MAKE_HEX_DOUBLE( 0x1.62891f06b345p-113, +0x162891f06b3450LL, -165),	MAKE_HEX_DOUBLE( 0x1.e1dd273aa8a4ap-112, +0x1e1dd273aa8a4aLL, -164),	
		MAKE_HEX_DOUBLE( 0x1.4775e0840bfddp-110, +0x14775e0840bfddLL, -162),	MAKE_HEX_DOUBLE( 0x1.bd109d9d94bdap-109, +0x1bd109d9d94bdaLL, -161),	
		MAKE_HEX_DOUBLE( 0x1.2e73f53fba844p-107, +0x12e73f53fba844LL, -159),	MAKE_HEX_DOUBLE( 0x1.9b138170d6bfep-106, +0x19b138170d6bfeLL, -158),	
		MAKE_HEX_DOUBLE( 0x1.175af0cf60ec5p-104, +0x1175af0cf60ec5LL, -156),	MAKE_HEX_DOUBLE( 0x1.7baee1bffa80bp-103, +0x17baee1bffa80bLL, -155),	
		MAKE_HEX_DOUBLE( 0x1.02057d1245cebp-101, +0x102057d1245cebLL, -153),	MAKE_HEX_DOUBLE( 0x1.5eafffb34ba31p-100, +0x15eafffb34ba31LL, -152),	
		MAKE_HEX_DOUBLE( 0x1.dca23bae16424p-99, +0x1dca23bae16424LL, -151),	MAKE_HEX_DOUBLE( 0x1.43e7fc88b8056p-97, +0x143e7fc88b8056LL, -149),	
		MAKE_HEX_DOUBLE( 0x1.b83bf23a9a9ebp-96, +0x1b83bf23a9a9ebLL, -148),	MAKE_HEX_DOUBLE( 0x1.2b2b8dd05b318p-94, +0x12b2b8dd05b318LL, -146),	
		MAKE_HEX_DOUBLE( 0x1.969d47321e4ccp-93, +0x1969d47321e4ccLL, -145),	MAKE_HEX_DOUBLE( 0x1.1452b7723aed2p-91, +0x11452b7723aed2LL, -143),	
		MAKE_HEX_DOUBLE( 0x1.778fe2497184cp-90, +0x1778fe2497184cLL, -142),	MAKE_HEX_DOUBLE( 0x1.fe7116182e9ccp-89, +0x1fe7116182e9ccLL, -141),	
		MAKE_HEX_DOUBLE( 0x1.5ae191a99585ap-87, +0x15ae191a99585aLL, -139),	MAKE_HEX_DOUBLE( 0x1.d775d87da854dp-86, +0x1d775d87da854dLL, -138),	
		MAKE_HEX_DOUBLE( 0x1.4063f8cc8bb98p-84, +0x14063f8cc8bb98LL, -136),	MAKE_HEX_DOUBLE( 0x1.b374b315f87c1p-83, +0x1b374b315f87c1LL, -135),	
		MAKE_HEX_DOUBLE( 0x1.27ec458c65e3cp-81, +0x127ec458c65e3cLL, -133),	MAKE_HEX_DOUBLE( 0x1.923372c67a074p-80, +0x1923372c67a074LL, -132),	
		MAKE_HEX_DOUBLE( 0x1.1152eaeb73c08p-78, +0x11152eaeb73c08LL, -130),	MAKE_HEX_DOUBLE( 0x1.737c5645114b5p-77, +0x1737c5645114b5LL, -129),	
		MAKE_HEX_DOUBLE( 0x1.f8e6c24b5592ep-76, +0x1f8e6c24b5592eLL, -128),	MAKE_HEX_DOUBLE( 0x1.571db733a9d61p-74, +0x1571db733a9d61LL, -126),	
		MAKE_HEX_DOUBLE( 0x1.d257d547e083fp-73, +0x1d257d547e083fLL, -125),	MAKE_HEX_DOUBLE( 0x1.3ce9b9de78f85p-71, +0x13ce9b9de78f85LL, -123),	
		MAKE_HEX_DOUBLE( 0x1.aebabae3a41b5p-70, +0x1aebabae3a41b5LL, -122),	MAKE_HEX_DOUBLE( 0x1.24b6031b49bdap-68, +0x124b6031b49bdaLL, -120),	
		MAKE_HEX_DOUBLE( 0x1.8dd5e1bb09d7ep-67, +0x18dd5e1bb09d7eLL, -119),	MAKE_HEX_DOUBLE( 0x1.0e5b73d1ff53dp-65, +0x10e5b73d1ff53dLL, -117),	
		MAKE_HEX_DOUBLE( 0x1.6f741de1748ecp-64, +0x16f741de1748ecLL, -116),	MAKE_HEX_DOUBLE( 0x1.f36bd37f42f3ep-63, +0x1f36bd37f42f3eLL, -115),	
		MAKE_HEX_DOUBLE( 0x1.536452ee2f75cp-61, +0x1536452ee2f75cLL, -113),	MAKE_HEX_DOUBLE( 0x1.cd480a1b7482p-60, +0x1cd480a1b74820LL, -112),	
		MAKE_HEX_DOUBLE( 0x1.39792499b1a24p-58, +0x139792499b1a24LL, -110),	MAKE_HEX_DOUBLE( 0x1.aa0de4bf35b38p-57, +0x1aa0de4bf35b38LL, -109),	
		MAKE_HEX_DOUBLE( 0x1.2188ad6ae3303p-55, +0x12188ad6ae3303LL, -107),	MAKE_HEX_DOUBLE( 0x1.898471fca6055p-54, +0x1898471fca6055LL, -106),	
		MAKE_HEX_DOUBLE( 0x1.0b6c3afdde064p-52, +0x10b6c3afdde064LL, -104),	MAKE_HEX_DOUBLE( 0x1.6b7719a59f0ep-51, +0x16b7719a59f0e0LL, -103),	
		MAKE_HEX_DOUBLE( 0x1.ee001eed62aap-50, +0x1ee001eed62aa0LL, -102),	MAKE_HEX_DOUBLE( 0x1.4fb547c775da8p-48, +0x14fb547c775da8LL, -100),	
		MAKE_HEX_DOUBLE( 0x1.c8464f7616468p-47, +0x1c8464f7616468LL, -99),	MAKE_HEX_DOUBLE( 0x1.36121e24d3bbap-45, +0x136121e24d3bbaLL, -97),	
		MAKE_HEX_DOUBLE( 0x1.a56e0c2ac7f75p-44, +0x1a56e0c2ac7f75LL, -96),	MAKE_HEX_DOUBLE( 0x1.1e642baeb84ap-42, +0x11e642baeb84a0LL, -94),	
		MAKE_HEX_DOUBLE( 0x1.853f01d6d53bap-41, +0x1853f01d6d53baLL, -93),	MAKE_HEX_DOUBLE( 0x1.0885298767e9ap-39, +0x10885298767e9aLL, -91),	
		MAKE_HEX_DOUBLE( 0x1.67852a7007e42p-38, +0x167852a7007e42LL, -90),	MAKE_HEX_DOUBLE( 0x1.e8a37a45fc32ep-37, +0x1e8a37a45fc32eLL, -89),	
		MAKE_HEX_DOUBLE( 0x1.4c1078fe9228ap-35, +0x14c1078fe9228aLL, -87),	MAKE_HEX_DOUBLE( 0x1.c3527e433fab1p-34, +0x1c3527e433fab1LL, -86),	
		MAKE_HEX_DOUBLE( 0x1.32b48bf117da2p-32, +0x132b48bf117da2LL, -84),	MAKE_HEX_DOUBLE( 0x1.a0db0d0ddb3ecp-31, +0x1a0db0d0ddb3ecLL, -83),	
		MAKE_HEX_DOUBLE( 0x1.1b48655f37267p-29, +0x11b48655f37267LL, -81),	MAKE_HEX_DOUBLE( 0x1.81056ff2c5772p-28, +0x181056ff2c5772LL, -80),	
		MAKE_HEX_DOUBLE( 0x1.05a628c699fa1p-26, +0x105a628c699fa1LL, -78),	MAKE_HEX_DOUBLE( 0x1.639e3175a689dp-25, +0x1639e3175a689dLL, -77),	
		MAKE_HEX_DOUBLE( 0x1.e355bbaee85cbp-24, +0x1e355bbaee85cbLL, -76),	MAKE_HEX_DOUBLE( 0x1.4875ca227ec38p-22, +0x14875ca227ec38LL, -74),	
		MAKE_HEX_DOUBLE( 0x1.be6c6fdb01612p-21, +0x1be6c6fdb01612LL, -73),	MAKE_HEX_DOUBLE( 0x1.2f6053b981d98p-19, +0x12f6053b981d98LL, -71),	
		MAKE_HEX_DOUBLE( 0x1.9c54c3b43bc8bp-18, +0x19c54c3b43bc8bLL, -70),	MAKE_HEX_DOUBLE( 0x1.18354238f6764p-16, +0x118354238f6764LL, -68),	
		MAKE_HEX_DOUBLE( 0x1.7cd79b5647c9bp-15, +0x17cd79b5647c9bLL, -67),	MAKE_HEX_DOUBLE( 0x1.02cf22526545ap-13, +0x102cf22526545aLL, -65),	
		MAKE_HEX_DOUBLE( 0x1.5fc21041027adp-12, +0x15fc21041027adLL, -64),	MAKE_HEX_DOUBLE( 0x1.de16b9c24a98fp-11, +0x1de16b9c24a98fLL, -63),	
		MAKE_HEX_DOUBLE( 0x1.44e51f113d4d6p-9, +0x144e51f113d4d6LL, -61),	MAKE_HEX_DOUBLE( 0x1.b993fe00d5376p-8, +0x1b993fe00d5376LL, -60),	
		MAKE_HEX_DOUBLE( 0x1.2c155b8213cf4p-6, +0x12c155b8213cf4LL, -58),	MAKE_HEX_DOUBLE( 0x1.97db0ccceb0afp-5, +0x197db0ccceb0afLL, -57),	
		MAKE_HEX_DOUBLE( 0x1.152aaa3bf81ccp-3, +0x1152aaa3bf81ccLL, -55),	MAKE_HEX_DOUBLE( 0x1.78b56362cef38p-2, +0x178b56362cef38LL, -54),	
		MAKE_HEX_DOUBLE( 0x1p+0, +0x10000000000000LL, -52),					MAKE_HEX_DOUBLE( 0x1.5bf0a8b145769p+1, +0x15bf0a8b145769LL, -51),	
		MAKE_HEX_DOUBLE( 0x1.d8e64b8d4ddaep+2, +0x1d8e64b8d4ddaeLL, -50),	MAKE_HEX_DOUBLE( 0x1.415e5bf6fb106p+4, +0x1415e5bf6fb106LL, -48),	
		MAKE_HEX_DOUBLE( 0x1.b4c902e273a58p+5, +0x1b4c902e273a58LL, -47),	MAKE_HEX_DOUBLE( 0x1.28d389970338fp+7, +0x128d389970338fLL, -45),	
		MAKE_HEX_DOUBLE( 0x1.936dc5690c08fp+8, +0x1936dc5690c08fLL, -44),	MAKE_HEX_DOUBLE( 0x1.122885aaeddaap+10, +0x1122885aaeddaaLL, -42),	
		MAKE_HEX_DOUBLE( 0x1.749ea7d470c6ep+11, +0x1749ea7d470c6eLL, -41),	MAKE_HEX_DOUBLE( 0x1.fa7157c470f82p+12, +0x1fa7157c470f82LL, -40),	
		MAKE_HEX_DOUBLE( 0x1.5829dcf95056p+14, +0x15829dcf950560LL, -38),	MAKE_HEX_DOUBLE( 0x1.d3c4488ee4f7fp+15, +0x1d3c4488ee4f7fLL, -37),	
		MAKE_HEX_DOUBLE( 0x1.3de1654d37c9ap+17, +0x13de1654d37c9aLL, -35),	MAKE_HEX_DOUBLE( 0x1.b00b5916ac955p+18, +0x1b00b5916ac955LL, -34),	
		MAKE_HEX_DOUBLE( 0x1.259ac48bf05d7p+20, +0x1259ac48bf05d7LL, -32),	MAKE_HEX_DOUBLE( 0x1.8f0ccafad2a87p+21, +0x18f0ccafad2a87LL, -31),	
		MAKE_HEX_DOUBLE( 0x1.0f2ebd0a8002p+23, +0x10f2ebd0a80020LL, -29),	MAKE_HEX_DOUBLE( 0x1.709348c0ea4f9p+24, +0x1709348c0ea4f9LL, -28),	
		MAKE_HEX_DOUBLE( 0x1.f4f22091940bdp+25, +0x1f4f22091940bdLL, -27),	MAKE_HEX_DOUBLE( 0x1.546d8f9ed26e1p+27, +0x1546d8f9ed26e1LL, -25),	
		MAKE_HEX_DOUBLE( 0x1.ceb088b68e804p+28, +0x1ceb088b68e804LL, -24),	MAKE_HEX_DOUBLE( 0x1.3a6e1fd9eecfdp+30, +0x13a6e1fd9eecfdLL, -22),	
		MAKE_HEX_DOUBLE( 0x1.ab5adb9c436p+31, +0x1ab5adb9c43600LL, -21),	MAKE_HEX_DOUBLE( 0x1.226af33b1fdc1p+33, +0x1226af33b1fdc1LL, -19),	
		MAKE_HEX_DOUBLE( 0x1.8ab7fb5475fb7p+34, +0x18ab7fb5475fb7LL, -18),	MAKE_HEX_DOUBLE( 0x1.0c3d3920962c9p+36, +0x10c3d3920962c9LL, -16),	
		MAKE_HEX_DOUBLE( 0x1.6c932696a6b5dp+37, +0x16c932696a6b5dLL, -15),	MAKE_HEX_DOUBLE( 0x1.ef822f7f6731dp+38, +0x1ef822f7f6731dLL, -14),	
		MAKE_HEX_DOUBLE( 0x1.50bba3796379ap+40, +0x150bba3796379aLL, -12),	MAKE_HEX_DOUBLE( 0x1.c9aae4631c056p+41, +0x1c9aae4631c056LL, -11),	
		MAKE_HEX_DOUBLE( 0x1.370470aec28edp+43, +0x1370470aec28edLL, -9),	MAKE_HEX_DOUBLE( 0x1.a6b765d8cdf6dp+44, +0x1a6b765d8cdf6dLL, -8),	
		MAKE_HEX_DOUBLE( 0x1.1f43fcc4b662cp+46, +0x11f43fcc4b662cLL, -6),	MAKE_HEX_DOUBLE( 0x1.866f34a725782p+47, +0x1866f34a725782LL, -5),	
		MAKE_HEX_DOUBLE( 0x1.0953e2f3a1ef7p+49, +0x10953e2f3a1ef7LL, -3),	MAKE_HEX_DOUBLE( 0x1.689e221bc8d5bp+50, +0x1689e221bc8d5bLL, -2),	
		MAKE_HEX_DOUBLE( 0x1.ea215a1d20d76p+51, +0x1ea215a1d20d76LL, -1),	MAKE_HEX_DOUBLE( 0x1.4d13fbb1a001ap+53, +0x14d13fbb1a001aLL, 1),	
		MAKE_HEX_DOUBLE( 0x1.c4b334617cc67p+54, +0x1c4b334617cc67LL, 2),	MAKE_HEX_DOUBLE( 0x1.33a43d282a519p+56, +0x133a43d282a519LL, 4),	
		MAKE_HEX_DOUBLE( 0x1.a220d397972ebp+57, +0x1a220d397972ebLL, 5),	MAKE_HEX_DOUBLE( 0x1.1c25c88df6862p+59, +0x11c25c88df6862LL, 7),	
		MAKE_HEX_DOUBLE( 0x1.8232558201159p+60, +0x18232558201159LL, 8),	MAKE_HEX_DOUBLE( 0x1.0672a3c9eb871p+62, +0x10672a3c9eb871LL, 10),	
		MAKE_HEX_DOUBLE( 0x1.64b41c6d37832p+63, +0x164b41c6d37832LL, 11),	MAKE_HEX_DOUBLE( 0x1.e4cf766fe49bep+64, +0x1e4cf766fe49beLL, 12),	
		MAKE_HEX_DOUBLE( 0x1.49767bc0483e3p+66, +0x149767bc0483e3LL, 14),	MAKE_HEX_DOUBLE( 0x1.bfc951eb8bb76p+67, +0x1bfc951eb8bb76LL, 15),	
		MAKE_HEX_DOUBLE( 0x1.304d6aeca254bp+69, +0x1304d6aeca254bLL, 17),	MAKE_HEX_DOUBLE( 0x1.9d97010884251p+70, +0x19d97010884251LL, 18),	
		MAKE_HEX_DOUBLE( 0x1.19103e4080b45p+72, +0x119103e4080b45LL, 20),	MAKE_HEX_DOUBLE( 0x1.7e013cd114461p+73, +0x17e013cd114461LL, 21),	
		MAKE_HEX_DOUBLE( 0x1.03996528e074cp+75, +0x103996528e074cLL, 23),	MAKE_HEX_DOUBLE( 0x1.60d4f6fdac731p+76, +0x160d4f6fdac731LL, 24),	
		MAKE_HEX_DOUBLE( 0x1.df8c5af17ba3bp+77, +0x1df8c5af17ba3bLL, 25),	MAKE_HEX_DOUBLE( 0x1.45e3076d61699p+79, +0x145e3076d61699LL, 27),	
		MAKE_HEX_DOUBLE( 0x1.baed16a6e0da7p+80, +0x1baed16a6e0da7LL, 28),	MAKE_HEX_DOUBLE( 0x1.2cffdfebde1a1p+82, +0x12cffdfebde1a1LL, 30),	
		MAKE_HEX_DOUBLE( 0x1.9919cabefcb69p+83, +0x19919cabefcb69LL, 31),	MAKE_HEX_DOUBLE( 0x1.160345c9953e3p+85, +0x1160345c9953e3LL, 33),	
		MAKE_HEX_DOUBLE( 0x1.79dbc9dc53c66p+86, +0x179dbc9dc53c66LL, 34),	MAKE_HEX_DOUBLE( 0x1.00c810d464097p+88, +0x100c810d464097LL, 36),	
		MAKE_HEX_DOUBLE( 0x1.5d009394c5c27p+89, +0x15d009394c5c27LL, 37),	MAKE_HEX_DOUBLE( 0x1.da57de8f107a8p+90, +0x1da57de8f107a8LL, 38),	
		MAKE_HEX_DOUBLE( 0x1.425982cf597cdp+92, +0x1425982cf597cdLL, 40),	MAKE_HEX_DOUBLE( 0x1.b61e5ca3a5e31p+93, +0x1b61e5ca3a5e31LL, 41),	
		MAKE_HEX_DOUBLE( 0x1.29bb825dfcf87p+95, +0x129bb825dfcf87LL, 43),	MAKE_HEX_DOUBLE( 0x1.94a90db0d6fe2p+96, +0x194a90db0d6fe2LL, 44),	
		MAKE_HEX_DOUBLE( 0x1.12fec759586fdp+98, +0x112fec759586fdLL, 46),	MAKE_HEX_DOUBLE( 0x1.75c1dc469e3afp+99, +0x175c1dc469e3afLL, 47),	
		MAKE_HEX_DOUBLE( 0x1.fbfd219c43b04p+100, +0x1fbfd219c43b04LL, 48),	MAKE_HEX_DOUBLE( 0x1.5936d44e1a146p+102, +0x15936d44e1a146LL, 50),	
		MAKE_HEX_DOUBLE( 0x1.d531d8a7ee79cp+103, +0x1d531d8a7ee79cLL, 51),	MAKE_HEX_DOUBLE( 0x1.3ed9d24a2d51bp+105, +0x13ed9d24a2d51bLL, 53),	
		MAKE_HEX_DOUBLE( 0x1.b15cfe5b6e17bp+106, +0x1b15cfe5b6e17bLL, 54),	MAKE_HEX_DOUBLE( 0x1.268038c2c0ep+108, +0x1268038c2c0e00LL, 56),	
		MAKE_HEX_DOUBLE( 0x1.9044a73545d48p+109, +0x19044a73545d48LL, 57),	MAKE_HEX_DOUBLE( 0x1.1002ab6218b38p+111, +0x11002ab6218b38LL, 59),	
		MAKE_HEX_DOUBLE( 0x1.71b3540cbf921p+112, +0x171b3540cbf921LL, 60),	MAKE_HEX_DOUBLE( 0x1.f6799ea9c414ap+113, +0x1f6799ea9c414aLL, 61),	
		MAKE_HEX_DOUBLE( 0x1.55779b984f3ebp+115, +0x155779b984f3ebLL, 63),	MAKE_HEX_DOUBLE( 0x1.d01a210c44aa4p+116, +0x1d01a210c44aa4LL, 64),	
		MAKE_HEX_DOUBLE( 0x1.3b63da8e9121p+118, +0x13b63da8e91210LL, 66),	MAKE_HEX_DOUBLE( 0x1.aca8d6b0116b8p+119, +0x1aca8d6b0116b8LL, 67),	
		MAKE_HEX_DOUBLE( 0x1.234de9e0c74e9p+121, +0x1234de9e0c74e9LL, 69),	MAKE_HEX_DOUBLE( 0x1.8bec7503ca477p+122, +0x18bec7503ca477LL, 70),	
		MAKE_HEX_DOUBLE( 0x1.0d0eda9796b9p+124, +0x10d0eda9796b90LL, 72),	MAKE_HEX_DOUBLE( 0x1.6db0118477245p+125, +0x16db0118477245LL, 73),	
		MAKE_HEX_DOUBLE( 0x1.f1056dc7bf22dp+126, +0x1f1056dc7bf22dLL, 74),	MAKE_HEX_DOUBLE( 0x1.51c2cc3433801p+128, +0x151c2cc3433801LL, 76),	
		MAKE_HEX_DOUBLE( 0x1.cb108ffbec164p+129, +0x1cb108ffbec164LL, 77),	MAKE_HEX_DOUBLE( 0x1.37f780991b584p+131, +0x137f780991b584LL, 79),	
		MAKE_HEX_DOUBLE( 0x1.a801c0ea8ac4dp+132, +0x1a801c0ea8ac4dLL, 80),	MAKE_HEX_DOUBLE( 0x1.20247cc4c46c1p+134, +0x120247cc4c46c1LL, 82),	
		MAKE_HEX_DOUBLE( 0x1.87a0553328015p+135, +0x187a0553328015LL, 83),	MAKE_HEX_DOUBLE( 0x1.0a233dee4f9bbp+137, +0x10a233dee4f9bbLL, 85),	
		MAKE_HEX_DOUBLE( 0x1.69b7f55b808bap+138, +0x169b7f55b808baLL, 86),	MAKE_HEX_DOUBLE( 0x1.eba064644060ap+139, +0x1eba064644060aLL, 87),	
		MAKE_HEX_DOUBLE( 0x1.4e184933d9364p+141, +0x14e184933d9364LL, 89),	MAKE_HEX_DOUBLE( 0x1.c614fe2531841p+142, +0x1c614fe2531841LL, 90),	
		MAKE_HEX_DOUBLE( 0x1.3494a9b171bf5p+144, +0x13494a9b171bf5LL, 92),	MAKE_HEX_DOUBLE( 0x1.a36798b9d969bp+145, +0x1a36798b9d969bLL, 93),	
		MAKE_HEX_DOUBLE( 0x1.1d03d8c0c04afp+147, +0x11d03d8c0c04afLL, 95),	MAKE_HEX_DOUBLE( 0x1.836026385c974p+148, +0x1836026385c974LL, 96),	
		MAKE_HEX_DOUBLE( 0x1.073fbe9ac901dp+150, +0x1073fbe9ac901dLL, 98),	MAKE_HEX_DOUBLE( 0x1.65cae0969f286p+151, +0x165cae0969f286LL, 99),	
		MAKE_HEX_DOUBLE( 0x1.e64a58639cae8p+152, +0x1e64a58639cae8LL, 100),	MAKE_HEX_DOUBLE( 0x1.4a77f5f9b50f9p+154, +0x14a77f5f9b50f9LL, 102),	
		MAKE_HEX_DOUBLE( 0x1.c12744a3a28e3p+155, +0x1c12744a3a28e3LL, 103),	MAKE_HEX_DOUBLE( 0x1.313b3b6978e85p+157, +0x1313b3b6978e85LL, 105),	
		MAKE_HEX_DOUBLE( 0x1.9eda3a31e587ep+158, +0x19eda3a31e587eLL, 106),	MAKE_HEX_DOUBLE( 0x1.19ebe56b56453p+160, +0x119ebe56b56453LL, 108),	
		MAKE_HEX_DOUBLE( 0x1.7f2bc6e599b7ep+161, +0x17f2bc6e599b7eLL, 109),	MAKE_HEX_DOUBLE( 0x1.04644610df2ffp+163, +0x104644610df2ffLL, 111),	
		MAKE_HEX_DOUBLE( 0x1.61e8b490ac4e6p+164, +0x161e8b490ac4e6LL, 112),	MAKE_HEX_DOUBLE( 0x1.e103201f299b3p+165, +0x1e103201f299b3LL, 113),	
		MAKE_HEX_DOUBLE( 0x1.46e1b637beaf5p+167, +0x146e1b637beaf5LL, 115),	MAKE_HEX_DOUBLE( 0x1.bc473cfede104p+168, +0x1bc473cfede104LL, 116),	
		MAKE_HEX_DOUBLE( 0x1.2deb1b9c85e2dp+170, +0x12deb1b9c85e2dLL, 118),	MAKE_HEX_DOUBLE( 0x1.9a5981ca67d1p+171, +0x19a5981ca67d10LL, 119),	
		MAKE_HEX_DOUBLE( 0x1.16dc8a9ef670bp+173, +0x116dc8a9ef670bLL, 121),	MAKE_HEX_DOUBLE( 0x1.7b03166942309p+174, +0x17b03166942309LL, 122),	
		MAKE_HEX_DOUBLE( 0x1.0190be03150a7p+176, +0x10190be03150a7LL, 124),	MAKE_HEX_DOUBLE( 0x1.5e1152f9a8119p+177, +0x15e1152f9a8119LL, 125),	
		MAKE_HEX_DOUBLE( 0x1.dbca9263f8487p+178, +0x1dbca9263f8487LL, 126),	MAKE_HEX_DOUBLE( 0x1.43556dee93beep+180, +0x143556dee93beeLL, 128),	
		MAKE_HEX_DOUBLE( 0x1.b774c12967dfap+181, +0x1b774c12967dfaLL, 129),	MAKE_HEX_DOUBLE( 0x1.2aa4306e922c2p+183, +0x12aa4306e922c2LL, 131),	
		MAKE_HEX_DOUBLE( 0x1.95e54c5dd4217p+184, +0x195e54c5dd4217LL, 132)    };
    
    // scale by e**i --  (expm1(f) + 1)*e**i - 1  = expm1(f) * e**i + e**i - 1 = e**i 
    return exp_table[exponent+150] + (f * exp_table[exponent+150] - 1.0);
}


double reference_fmax( double x, double y )
{
    if( isnan(y) )
        return x;

    return x >= y ? x : y;
}

double reference_fmin( double x, double y )
{
    if( isnan(y) )
        return x;

    return x <= y ? x : y;
}

double reference_hypot( double x, double y )
{  
    // Since the inputs are actually floats, we don't have to worry about range here
    if( isinf(x) || isinf(y) )
        return INFINITY;
        
    return sqrt( x * x + y * y );
}

int    reference_ilogbl( long double x)
{
    extern int gDeviceILogb0, gDeviceILogbNaN;

    // Since we are just using this to verify double precision, we can
    // use the double precision ilogb here
    union { double f; cl_ulong u;} u;
    u.f = (double) x;
            
    int exponent = (int)(u.u >> 52) & 0x7ff;
    if( exponent == 0x7ff )
    {
        if( u.u & 0x000fffffffffffffULL )
            return gDeviceILogbNaN;
            
        return CL_INT_MAX;
    }
        
    if( exponent == 0 )
    {   // deal with denormals
        u.f =  x * MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64);
        exponent = (cl_uint)(u.u >> 52) & 0x7ff;
        if( exponent == 0 )
            return gDeviceILogb0;
        
        exponent -= 1023 + 64;
        return exponent;
    }

    return exponent - 1023;
}

//double reference_log2( double x )
//{
//    return log( x ) * 1.44269504088896340735992468100189214;
//}

double reference_log2( double x )
{
	if( isnan(x) || x < 0.0 || x == -INFINITY)
		return cl_make_nan();
	
	if( x == 0.0f) 
		return -INFINITY;
		
	if( x == INFINITY )
		return INFINITY;

	double hi, lo;
	__log2_ep( &hi, &lo, x );
	return hi;
}

double reference_log1p( double x )
{   // This function is suitable only for verifying log1pf(). It produces several double precision ulps of error.

    // Handle small and NaN
    if( ! ( reference_fabs(x) > MAKE_HEX_DOUBLE(0x1.0p-53, 0x1LL, -53) ) )
        return x;

    // deal with special values
    if( x <= -1.0 )
    {
        if( x < -1.0 )
            return cl_make_nan();
        return -INFINITY;
    }

    // infinity
    if( x == INFINITY )
        return INFINITY;
    
    // High precision result for when near 0, to avoid problems with the reference result falling in the wrong binade.
    if( reference_fabs(x) < MAKE_HEX_DOUBLE(0x1.0p-28, 0x1LL, -28) )
        return (1.0 - 0.5 * x) * x;
    
    // Our polynomial is only good in the region +-2**-4.  
    // If we aren't in that range then we need to reduce to be in that range
    double correctionLo = -0.0;           // correction down stream to compensate for the reduction, if any
    double correctionHi = -0.0;           // correction down stream to compensate for the exponent, if any
    if( reference_fabs(x) > MAKE_HEX_DOUBLE(0x1.0p-4, 0x1LL, -4) )
    {
        x += 1.0;   // double should cover any loss of precision here

        // separate x into (1+f) * 2**i
        union{ double d; cl_ulong u;} u;        u.d = x;
        int i = (int) ((u.u >> 52) & 0x7ff) - 1023;
        u.u &= 0x000fffffffffffffULL;
        int index = (int) (u.u >> 48 );
        u.u |= 0x3ff0000000000000ULL;
        double f = u.d;

        // further reduce f to be within 1/16 of 1.0
        static const double scale_table[16] = {                  1.0, MAKE_HEX_DOUBLE(0x1.d2d2d2d6e3f79p-1, 0x1d2d2d2d6e3f79LL, -53), MAKE_HEX_DOUBLE(0x1.b8e38e42737a1p-1, 0x1b8e38e42737a1LL, -53), MAKE_HEX_DOUBLE(0x1.a1af28711adf3p-1, 0x1a1af28711adf3LL, -53),
                                                MAKE_HEX_DOUBLE(0x1.8cccccd88dd65p-1, 0x18cccccd88dd65LL, -53), MAKE_HEX_DOUBLE(0x1.79e79e810ec8fp-1, 0x179e79e810ec8fLL, -53), MAKE_HEX_DOUBLE(0x1.68ba2e94df404p-1, 0x168ba2e94df404LL, -53), MAKE_HEX_DOUBLE(0x1.590b216defb29p-1, 0x1590b216defb29LL, -53),
                                                MAKE_HEX_DOUBLE(0x1.4aaaaab1500edp-1, 0x14aaaaab1500edLL, -53), MAKE_HEX_DOUBLE(0x1.3d70a3e0d6f73p-1, 0x13d70a3e0d6f73LL, -53), MAKE_HEX_DOUBLE(0x1.313b13bb39f4fp-1, 0x1313b13bb39f4fLL, -53), MAKE_HEX_DOUBLE(0x1.25ed09823f1ccp-1, 0x125ed09823f1ccLL, -53),
                                                MAKE_HEX_DOUBLE(0x1.1b6db6e77457bp-1, 0x11b6db6e77457bLL, -53), MAKE_HEX_DOUBLE(0x1.11a7b96a3a34fp-1, 0x111a7b96a3a34fLL, -53), MAKE_HEX_DOUBLE(0x1.0888888e46feap-1, 0x10888888e46feaLL, -53), MAKE_HEX_DOUBLE(0x1.00000038e9862p-1, 0x100000038e9862LL, -53) };
        
        // correction_table[i] = -log( scale_table[i] )
        // All entries have >= 64 bits of precision (rather than the expected 53)
        static const double correction_table[16] = {                   -0.0, MAKE_HEX_DOUBLE(0x1.7a5c722c16058p-4, 0x17a5c722c16058LL, -56), MAKE_HEX_DOUBLE(0x1.323db16c89ab1p-3, 0x1323db16c89ab1LL, -55), MAKE_HEX_DOUBLE(0x1.a0f87d180629p-3, 0x1a0f87d180629LL, -51), 
                                                       MAKE_HEX_DOUBLE(0x1.050279324e17cp-2, 0x1050279324e17cLL, -54), MAKE_HEX_DOUBLE(0x1.36f885bb270b0p-2, 0x136f885bb270bLL, -50), MAKE_HEX_DOUBLE(0x1.669b771b5cc69p-2, 0x1669b771b5cc69LL, -54), MAKE_HEX_DOUBLE(0x1.94203a6292a05p-2, 0x194203a6292a05LL, -54),
                                                       MAKE_HEX_DOUBLE(0x1.bfb4f9cb333a4p-2, 0x1bfb4f9cb333a4LL, -54), MAKE_HEX_DOUBLE(0x1.e982376ddb80ep-2, 0x1e982376ddb80eLL, -54), MAKE_HEX_DOUBLE(0x1.08d5d8769b2b2p-1, 0x108d5d8769b2b2LL, -53), MAKE_HEX_DOUBLE(0x1.1c288bc00e0cfp-1, 0x11c288bc00e0cfLL, -53),
                                                       MAKE_HEX_DOUBLE(0x1.2ec7535b31ecbp-1, 0x12ec7535b31ecbLL, -53), MAKE_HEX_DOUBLE(0x1.40bed0adc63fbp-1, 0x140bed0adc63fbLL, -53), MAKE_HEX_DOUBLE(0x1.521a5c0330615p-1, 0x1521a5c0330615LL, -53), MAKE_HEX_DOUBLE(0x1.62e42f7dd092cp-1, 0x162e42f7dd092cLL, -53) };  
        
        f *= scale_table[index];
        correctionLo = correction_table[index];

        // log( 2**(i) ) = i * log(2) 
        correctionHi = (double)i * 0.693147180559945309417232121458176568;
        
        x = f - 1.0;
    }


    // minmax polynomial for p(x) = (log(x+1) - x)/x valid over the range x = [-1/16, 1/16]
    //          max error MAKE_HEX_DOUBLE(0x1.048f61f9a5ecap-52, 0x1048f61f9a5ecaLL, -104)
    double p = MAKE_HEX_DOUBLE(-0x1.cc33de97a9d7bp-46, -0x1cc33de97a9d7bLL, -98) + 
               (MAKE_HEX_DOUBLE(-0x1.fffffffff3eb7p-2, -0x1fffffffff3eb7LL, -54) + 
               (MAKE_HEX_DOUBLE(0x1.5555555633ef7p-2, 0x15555555633ef7LL, -54) + 
               (MAKE_HEX_DOUBLE(-0x1.00000062c78p-2, -0x100000062c78LL, -46) + 
               (MAKE_HEX_DOUBLE(0x1.9999958a3321p-3, 0x19999958a3321LL, -51) + 
               (MAKE_HEX_DOUBLE(-0x1.55534ce65c347p-3, -0x155534ce65c347LL, -55) + 
               (MAKE_HEX_DOUBLE(0x1.24957208391a5p-3, 0x124957208391a5LL, -55) + 
               (MAKE_HEX_DOUBLE(-0x1.02287b9a5b4a1p-3, -0x102287b9a5b4a1LL, -55) + 
                MAKE_HEX_DOUBLE(0x1.c757d922180edp-4, 0x1c757d922180edLL, -56) * x)*x)*x)*x)*x)*x)*x)*x;

    // log(x+1) = x * p(x) + x
    x += x * p;

    return correctionHi + (correctionLo + x);
}

double reference_logb( double x )
{
    union { float f; cl_uint u;} u;
    u.f = (float) x;
        
    cl_int exponent = (u.u >> 23) & 0xff;
    if( exponent == 0xff )
        return x * x;
        
    if( exponent == 0 )
    {   // deal with denormals
        u.u = (u.u & 0x007fffff) | 0x3f800000;
        u.f -= 1.0f;
        exponent = (u.u >> 23) & 0xff;
        if( exponent == 0 )
            return -INFINITY;
        
        return exponent - (127 + 126);
    }

    return exponent - 127;
}

double reference_reciprocal( double x )
{
    return 1.0 / x;
}

double reference_remainder( double x, double y )
{
    int i;
    return reference_remquo( x, y, &i );
}

double reference_lgamma( double x)
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

static const double //two52 = 4.50359962737049600000e+15, /* 0x43300000, 0x00000000 */
                    half=  5.00000000000000000000e-01, /* 0x3FE00000, 0x00000000 */
                    one =  1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
                    pi  =  3.14159265358979311600e+00, /* 0x400921FB, 0x54442D18 */
                    a0  =  7.72156649015328655494e-02, /* 0x3FB3C467, 0xE37DB0C8 */
                    a1  =  3.22467033424113591611e-01, /* 0x3FD4A34C, 0xC4A60FAD */
                    a2  =  6.73523010531292681824e-02, /* 0x3FB13E00, 0x1A5562A7 */
                    a3  =  2.05808084325167332806e-02, /* 0x3F951322, 0xAC92547B */
                    a4  =  7.38555086081402883957e-03, /* 0x3F7E404F, 0xB68FEFE8 */
                    a5  =  2.89051383673415629091e-03, /* 0x3F67ADD8, 0xCCB7926B */
                    a6  =  1.19270763183362067845e-03, /* 0x3F538A94, 0x116F3F5D */
                    a7  =  5.10069792153511336608e-04, /* 0x3F40B6C6, 0x89B99C00 */
                    a8  =  2.20862790713908385557e-04, /* 0x3F2CF2EC, 0xED10E54D */
                    a9  =  1.08011567247583939954e-04, /* 0x3F1C5088, 0x987DFB07 */
                    a10 =  2.52144565451257326939e-05, /* 0x3EFA7074, 0x428CFA52 */
                    a11 =  4.48640949618915160150e-05, /* 0x3F07858E, 0x90A45837 */
                    tc  =  1.46163214496836224576e+00, /* 0x3FF762D8, 0x6356BE3F */
                    tf  = -1.21486290535849611461e-01, /* 0xBFBF19B9, 0xBCC38A42 */
                    /* tt = -(tail of tf) */
                    tt  = -3.63867699703950536541e-18, /* 0xBC50C7CA, 0xA48A971F */
                    t0  =  4.83836122723810047042e-01, /* 0x3FDEF72B, 0xC8EE38A2 */
                    t1  = -1.47587722994593911752e-01, /* 0xBFC2E427, 0x8DC6C509 */
                    t2  =  6.46249402391333854778e-02, /* 0x3FB08B42, 0x94D5419B */
                    t3  = -3.27885410759859649565e-02, /* 0xBFA0C9A8, 0xDF35B713 */
                    t4  =  1.79706750811820387126e-02, /* 0x3F9266E7, 0x970AF9EC */
                    t5  = -1.03142241298341437450e-02, /* 0xBF851F9F, 0xBA91EC6A */
                    t6  =  6.10053870246291332635e-03, /* 0x3F78FCE0, 0xE370E344 */
                    t7  = -3.68452016781138256760e-03, /* 0xBF6E2EFF, 0xB3E914D7 */
                    t8  =  2.25964780900612472250e-03, /* 0x3F6282D3, 0x2E15C915 */
                    t9  = -1.40346469989232843813e-03, /* 0xBF56FE8E, 0xBF2D1AF1 */
                    t10 =  8.81081882437654011382e-04, /* 0x3F4CDF0C, 0xEF61A8E9 */
                    t11 = -5.38595305356740546715e-04, /* 0xBF41A610, 0x9C73E0EC */
                    t12 =  3.15632070903625950361e-04, /* 0x3F34AF6D, 0x6C0EBBF7 */
                    t13 = -3.12754168375120860518e-04, /* 0xBF347F24, 0xECC38C38 */
                    t14 =  3.35529192635519073543e-04, /* 0x3F35FD3E, 0xE8C2D3F4 */
                    u0  = -7.72156649015328655494e-02, /* 0xBFB3C467, 0xE37DB0C8 */
                    u1  =  6.32827064025093366517e-01, /* 0x3FE4401E, 0x8B005DFF */
                    u2  =  1.45492250137234768737e+00, /* 0x3FF7475C, 0xD119BD6F */
                    u3  =  9.77717527963372745603e-01, /* 0x3FEF4976, 0x44EA8450 */
                    u4  =  2.28963728064692451092e-01, /* 0x3FCD4EAE, 0xF6010924 */
                    u5  =  1.33810918536787660377e-02, /* 0x3F8B678B, 0xBF2BAB09 */
                    v1  =  2.45597793713041134822e+00, /* 0x4003A5D7, 0xC2BD619C */
                    v2  =  2.12848976379893395361e+00, /* 0x40010725, 0xA42B18F5 */
                    v3  =  7.69285150456672783825e-01, /* 0x3FE89DFB, 0xE45050AF */
                    v4  =  1.04222645593369134254e-01, /* 0x3FBAAE55, 0xD6537C88 */
                    v5  =  3.21709242282423911810e-03, /* 0x3F6A5ABB, 0x57D0CF61 */
                    s0  = -7.72156649015328655494e-02, /* 0xBFB3C467, 0xE37DB0C8 */
                    s1  =  2.14982415960608852501e-01, /* 0x3FCB848B, 0x36E20878 */
                    s2  =  3.25778796408930981787e-01, /* 0x3FD4D98F, 0x4F139F59 */
                    s3  =  1.46350472652464452805e-01, /* 0x3FC2BB9C, 0xBEE5F2F7 */
                    s4  =  2.66422703033638609560e-02, /* 0x3F9B481C, 0x7E939961 */
                    s5  =  1.84028451407337715652e-03, /* 0x3F5E26B6, 0x7368F239 */
                    s6  =  3.19475326584100867617e-05, /* 0x3F00BFEC, 0xDD17E945 */
                    r1  =  1.39200533467621045958e+00, /* 0x3FF645A7, 0x62C4AB74 */
                    r2  =  7.21935547567138069525e-01, /* 0x3FE71A18, 0x93D3DCDC */
                    r3  =  1.71933865632803078993e-01, /* 0x3FC601ED, 0xCCFBDF27 */
                    r4  =  1.86459191715652901344e-02, /* 0x3F9317EA, 0x742ED475 */
                    r5  =  7.77942496381893596434e-04, /* 0x3F497DDA, 0xCA41A95B */
                    r6  =  7.32668430744625636189e-06, /* 0x3EDEBAF7, 0xA5B38140 */
                    w0  =  4.18938533204672725052e-01, /* 0x3FDACFE3, 0x90C97D69 */
                    w1  =  8.33333333333329678849e-02, /* 0x3FB55555, 0x5555553B */
                    w2  = -2.77777777728775536470e-03, /* 0xBF66C16C, 0x16B02E5C */
                    w3  =  7.93650558643019558500e-04, /* 0x3F4A019F, 0x98CF38B6 */
                    w4  = -5.95187557450339963135e-04, /* 0xBF4380CB, 0x8C0FE741 */
                    w5  =  8.36339918996282139126e-04, /* 0x3F4B67BA, 0x4CDAD5D1 */
                    w6  = -1.63092934096575273989e-03; /* 0xBF5AB89D, 0x0B9E43E4 */

    static const double zero=  0.00000000000000000000e+00;
	double t,y,z,nadj,p,p1,p2,p3,q,r,w;
	cl_int i,hx,lx,ix;

    union{ double d; cl_ulong u;}u; u.d = x;

	hx = (cl_int) (u.u >> 32);
	lx = (cl_int) (u.u & 0xffffffffULL);

    /* purge off +-inf, NaN, +-0, and negative arguments */
//	*signgamp = 1;
	ix = hx&0x7fffffff;
	if(ix>=0x7ff00000) return x*x;
	if((ix|lx)==0) return one/zero;
	if(ix<0x3b900000) {	/* |x|<2**-70, return -log(|x|) */
	    if(hx<0) {
//	        *signgamp = -1;
	        return -reference_log(-x);
	    } else return -reference_log(x);
	}
	if(hx<0) {
	    if(ix>=0x43300000) 	/* |x|>=2**52, must be -integer */
		return one/zero;
	    t = reference_sinpi(x);
	    if(t==zero) return one/zero; /* -integer */
	    nadj = reference_log(pi/reference_fabs(t*x));
//	    if(t<zero) *signgamp = -1;
	    x = -x;
	}

    /* purge off 1 and 2 */
	if((((ix-0x3ff00000)|lx)==0)||(((ix-0x40000000)|lx)==0)) r = 0;
    /* for x < 2.0 */
	else if(ix<0x40000000) {
	    if(ix<=0x3feccccc) { 	/* lgamma(x) = lgamma(x+1)-log(x) */
		r = -reference_log(x);
		if(ix>=0x3FE76944) {y = 1.0-x; i= 0;}
		else if(ix>=0x3FCDA661) {y= x-(tc-one); i=1;}
	  	else {y = x; i=2;}
	    } else {
	  	r = zero;
	        if(ix>=0x3FFBB4C3) {y=2.0-x;i=0;} /* [1.7316,2] */
	        else if(ix>=0x3FF3B4C4) {y=x-tc;i=1;} /* [1.23,1.73] */
		else {y=x-one;i=2;}
	    }
	    switch(i) {
	      case 0:
		z = y*y;
		p1 = a0+z*(a2+z*(a4+z*(a6+z*(a8+z*a10))));
		p2 = z*(a1+z*(a3+z*(a5+z*(a7+z*(a9+z*a11)))));
		p  = y*p1+p2;
		r  += (p-0.5*y); break;
	      case 1:
		z = y*y;
		w = z*y;
		p1 = t0+w*(t3+w*(t6+w*(t9 +w*t12)));	/* parallel comp */
		p2 = t1+w*(t4+w*(t7+w*(t10+w*t13)));
		p3 = t2+w*(t5+w*(t8+w*(t11+w*t14)));
		p  = z*p1-(tt-w*(p2+y*p3));
		r += (tf + p); break;
	      case 2:	
		p1 = y*(u0+y*(u1+y*(u2+y*(u3+y*(u4+y*u5)))));
		p2 = one+y*(v1+y*(v2+y*(v3+y*(v4+y*v5))));
		r += (-0.5*y + p1/p2);
	    }
	}
	else if(ix<0x40200000) { 			/* x < 8.0 */
	    i = (int)x;
	    t = zero;
	    y = x-(double)i;
	    p = y*(s0+y*(s1+y*(s2+y*(s3+y*(s4+y*(s5+y*s6))))));
	    q = one+y*(r1+y*(r2+y*(r3+y*(r4+y*(r5+y*r6)))));
	    r = half*y+p/q;
	    z = one;	/* lgamma(1+s) = log(s) + lgamma(s) */
	    switch(i) {
	    case 7: z *= (y+6.0);	/* FALLTHRU */
	    case 6: z *= (y+5.0);	/* FALLTHRU */
	    case 5: z *= (y+4.0);	/* FALLTHRU */
	    case 4: z *= (y+3.0);	/* FALLTHRU */
	    case 3: z *= (y+2.0);	/* FALLTHRU */
		    r += reference_log(z); break;
	    }
    /* 8.0 <= x < 2**58 */
	} else if (ix < 0x43900000) {
	    t = reference_log(x);
	    z = one/x;
	    y = z*z;
	    w = w0+z*(w1+y*(w2+y*(w3+y*(w4+y*(w5+y*w6)))));
	    r = (x-half)*(t-one)+w;
	} else 
    /* 2**58 <= x <= inf */
	    r =  x*(reference_log(x)-one);
	if(hx<0) r = nadj - r;
	return r;

}

#endif // _MSC_VER

double reference_assignment( double x ){ return x; }

int reference_not( double x )
{
  int r = !x;
  return r;
}

#pragma mark -
#pragma mark Double testing

#ifndef M_PIL
    #define M_PIL        3.14159265358979323846264338327950288419716939937510582097494459230781640628620899L
#endif

static long double reduce1l( long double x );

#ifdef __PPC__
// Since long double on PPC is really extended precision double arithmetic 
// consisting of two doubles (a high and low). This form of long double has
// the potential of representing a number with more than LDBL_MANT_DIG digits
// such that reduction algorithm used for other architectures will not work.
// Instead and alternate reduction method is used.

static long double reduce1l( long double x )
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

static long double reduce1l( long double x )
{
    static long double unit_exp = 0; 
    if( 0.0L == unit_exp )
        unit_exp = scalbnl( 1.0L, LDBL_MANT_DIG);

    if( reference_fabsl(x) >= unit_exp )
    {
        if( reference_fabsl(x) == INFINITY )
            return cl_make_nan();

        return 0.0L; //we patch up the sign for sinPi and cosPi later, since they need different signs
    }

    // Find the nearest multiple of 2
    const long double r = reference_copysignl( unit_exp, x );
    long double z = x + r;
    z -= r;

    // subtract it from x. Value is now in the range -1 <= x <= 1
    return x - z;    
}
#endif // __PPC__ 

long double reference_acospil( long double x){  return reference_acosl( x ) / M_PIL;    }
long double reference_asinpil( long double x){  return reference_asinl( x ) / M_PIL;    }
long double reference_atanpil( long double x){  return reference_atanl( x ) / M_PIL;    }
long double reference_atan2pil( long double y, long double x){ return reference_atan2l( y, x) / M_PIL; }
long double reference_cospil( long double x)
{   
    if( reference_fabsl(x) >= MAKE_HEX_LONG(0x1.0p54L, 0x1LL, 54) )
    {
        if( reference_fabsl(x) == INFINITY )
            return cl_make_nan();

        //Note this probably fails for odd values between 0x1.0p52 and 0x1.0p53.
        //However, when starting with single precision inputs, there will be no odd values.

        return 1.0L; 
    }

    x = reduce1l(x);     

#if DBL_MANT_DIG >= LDBL_MANT_DIG  

    // phase adjust
    double xhi = 0.0;
    double xlo = 0.0;
    xhi = (double) x + 0.5;
        
    if(reference_fabsl(x) > 0.5L)
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
    if( xhi < -0.5 )
    {
        xhi = -1.0 - xhi;
        xlo = -xlo;
    }
    else if ( xhi > 0.5 )
    {
        xhi = 1.0 - xhi;
        xlo = -xlo;
    } 
        
    // cosPi zeros are all +0
    if( xhi == 0.0 && xlo == 0.0 )
        return 0.0;
        
    xhi *= M_PI;
    xlo *= M_PI;
       
    xhi += xlo;
      
    return reference_sinl( xhi );

#else
	// phase adjust
	x += 0.5L;
	
    // reduce to [-0.5, 0.5]
    if( x < -0.5L )
        x = -1.0L - x;
    else if ( x > 0.5L )
        x = 1.0L - x;

    // cosPi zeros are all +0
    if( x == 0.0L )
        return 0.0L;

    return reference_sinl( x * M_PIL );   
#endif    
}

long double reference_dividel( long double x, long double y)
{ 
    double dx = x; 
    double dy = y; 
    return dx/dy; 
}

typedef struct{ double hi, lo; } double_double;

// Split doubles_double into a series of consecutive 26-bit precise doubles and a remainder.
// Note for later -- for multiplication, it might be better to split each double into a power of two and two 26 bit portions
//                      multiplication of a double double by a known power of two is cheap. The current approach causes some inexact arithmetic in mul_dd.
static inline void split_dd( double_double x, double_double *hi, double_double *lo )
{
    union{ double d; cl_ulong u;}u;
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

static inline double_double accum_d( double_double a, double b )
{
    double temp;
    if( fabs(b) > fabs(a.hi) )
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
    
    if( isnan( a.lo ) )
        a.lo = 0.0;
    
    return a;
}

static inline double_double add_dd( double_double a, double_double b )
{
    double_double r = {-0.0 -0.0 };
    
    if( isinf(a.hi) || isinf( b.hi )  ||
       isnan(a.hi) || isnan( b.hi )  ||
       0.0 == a.hi || 0.0 == b.hi )
    {
        r.hi = a.hi + b.hi;
        r.lo = a.lo + b.lo;
        if( isnan( r.lo ) )
            r.lo = 0.0;
        return r;
    }
    
    //merge sort terms by magnitude -- here we assume that |a.hi| > |a.lo|, |b.hi| > |b.lo|, so we don't have to do the first merge pass
    double terms[4] = { a.hi, b.hi, a.lo, b.lo };
    double temp;
    
    //Sort hi terms
    if( fabs(terms[0]) < fabs(terms[1]) )
    {
        temp = terms[0];
        terms[0] = terms[1];
        terms[1] = temp;
    }
    //sort lo terms
    if( fabs(terms[2]) < fabs(terms[3]) )
    {
        temp = terms[2];
        terms[2] = terms[3];
        terms[3] = temp;
    }
    // Fix case where small high term is less than large low term
    if( fabs(terms[1]) < fabs(terms[2]) )
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
    if( isnan( r.lo ) )
        r.lo = 0.0;
    
    return r;
}

static inline double_double mul_dd( double_double a, double_double b )
{
    double_double result = {-0.0,-0.0};
    
    // Inf, nan and 0
    if( isnan( a.hi ) || isnan( b.hi ) || 
       isinf( a.hi ) || isinf( b.hi ) || 
       0.0 == a.hi || 0.0 == b.hi )
    {
        result.hi = a.hi * b.hi;
        return result;
    }
    
    double_double ah, al, bh, bl;
    split_dd( a, &ah, &al );
    split_dd( b, &bh, &bl );
    
    double p0 = ah.hi * bh.hi;        // exact    (52 bits in product) 0
    double p1 = ah.hi * bh.lo;        // exact    (52 bits in product) 26
    double p2 = ah.lo * bh.hi;        // exact    (52 bits in product) 26
    double p3 = ah.lo * bh.lo;        // exact    (52 bits in product) 52
    double p4 = al.hi * bh.hi;        // exact    (52 bits in product) 52
    double p5 = al.hi * bh.lo;        // exact    (52 bits in product) 78
    double p6 = al.lo * bh.hi;        // inexact  (54 bits in product) 78
    double p7 = al.lo * bh.lo;        // inexact  (54 bits in product) 104
    double p8 = ah.hi * bl.hi;        // exact    (52 bits in product) 52
    double p9 = ah.hi * bl.lo;        // inexact  (54 bits in product) 78
    double pA = ah.lo * bl.hi;        // exact    (52 bits in product) 78
    double pB = ah.lo * bl.lo;        // inexact  (54 bits in product) 104
    double pC = al.hi * bl.hi;        // exact    (52 bits in product) 104
    // the last 3 terms are two low to appear in the result
    
    
    // accumulate from bottom up
#if 0
    // works but slow
    result.hi = pC;
    result = accum_d( result, pB );
    result = accum_d( result, p7 );
    result = accum_d( result, pA );
    result = accum_d( result, p9 );
    result = accum_d( result, p6 );
    result = accum_d( result, p5 );
    result = accum_d( result, p8 );
    result = accum_d( result, p4 );
    result = accum_d( result, p3 );
    result = accum_d( result, p2 );
    result = accum_d( result, p1 );
    result = accum_d( result, p0 );

    // canonicalize the result
    double temp = result.hi;
    result.hi += result.lo;
    result.lo -= (result.hi - temp);
    if( isnan( result.lo ) )
        result.lo = 0.0;

    return result;
#else
    // take advantage of the known relative magnitudes of the partial products to avoid some sorting
    // Combine 2**-78 and 2**-104 terms. Here we are a bit sloppy about canonicalizing the double_doubles
    double_double t0 = { pA, pC };
    double_double t1 = { p9, pB };
    double_double t2 = { p6, p7 };
    double temp0, temp1, temp2;
    
    t0 = accum_d( t0, p5 );  // there is an extra 2**-78 term to deal with
    
    // Add in 2**-52 terms. Here we are a bit sloppy about canonicalizing the double_doubles
    temp0 = t0.hi;      temp1 = t1.hi;      temp2 = t2.hi;
    t0.hi += p3;        t1.hi += p4;        t2.hi += p8;
    temp0 -= t0.hi-p3;  temp1 -= t1.hi-p4;  temp2 -= t2.hi - p8;
    t0.lo += temp0;     t1.lo += temp1;     t2.lo += temp2;

    // Add in 2**-26 terms. Here we are a bit sloppy about canonicalizing the double_doubles
    temp1 = t1.hi;      temp2 = t2.hi;
    t1.hi += p1;        t2.hi += p2;
    temp1 -= t1.hi-p1;  temp2 -= t2.hi - p2;
    t1.lo += temp1;     t2.lo += temp2;

    // Combine accumulators to get the low bits of result
    t1 = add_dd( t1, add_dd( t2, t0 ) );

    // Add in MSB's, and round to precision
    return accum_d( t1, p0 );  // canonicalizes
#endif
    
}


long double reference_exp10l( long double z )
{
    const double_double log2_10 = { MAKE_HEX_DOUBLE( 0x1.a934f0979a371p+1, 0x1a934f0979a371LL, 1), MAKE_HEX_DOUBLE( 0x1.7f2495fb7fa6dp-53, 0x17f2495fb7fa6dLL, -53) };
    double_double x;
    int j;
    
    // Handle NaNs
    if( isnan(z) )
        return z;
    
    // init x
    x.hi = z;
    x.lo = z - x.hi;
    
    
    // 10**x = exp2( x * log2(10) )
    
    x = mul_dd( x, log2_10);    // x * log2(10)
    
    //Deal with overflow and underflow for exp2(x) stage next
    if( x.hi >= 1025 )
        return INFINITY;
    
    if( x.hi < -1075-24 )
        return +0.0;
    
    // find nearest integer to x
    int i = (int) rint(x.hi);
    
    // x now holds fractional part.  The result would be then 2**i  * exp2( x )
    x.hi -= i;
    
    // We could attempt to find a minimax polynomial for exp2(x) over the range x = [-0.5, 0.5].
    // However, this would converge very slowly near the extrema, where 0.5**n is not a lot different
    // from 0.5**(n+1), thereby requiring something like a 20th order polynomial to get 53 + 24 bits 
    // of precision. Instead we further reduce the range to [-1/32, 1/32] by observing that 
    //
    //  2**(a+b) = 2**a * 2**b
    //
    // We can thus build a table of 2**a values for a = n/16, n = [-8, 8], and reduce the range
    // of x to [-1/32, 1/32] by subtracting away the nearest value of n/16 from x.
    const double_double corrections[17] = 
    {
        { MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp-1,0x16a09e667f3bcdLL,-1), MAKE_HEX_DOUBLE(-0x1.bdd3413b26456p-55,-0x1bdd3413b26456LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.7a11473eb0187p-1,0x17a11473eb0187LL,-1), MAKE_HEX_DOUBLE(-0x1.41577ee04992fp-56,-0x141577ee04992fLL,-56) },
        { MAKE_HEX_DOUBLE(0x1.8ace5422aa0dbp-1,0x18ace5422aa0dbLL,-1), MAKE_HEX_DOUBLE(0x1.6e9f156864b27p-55,0x16e9f156864b27LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.9c49182a3f09p-1,0x19c49182a3f09LL,-1), MAKE_HEX_DOUBLE(0x1.c7c46b071f2bep-57,0x1c7c46b071f2beLL,-57) },
        { MAKE_HEX_DOUBLE(0x1.ae89f995ad3adp-1,0x1ae89f995ad3adLL,-1), MAKE_HEX_DOUBLE(0x1.7a1cd345dcc81p-55,0x17a1cd345dcc81LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.c199bdd85529cp-1,0x1c199bdd85529cLL,-1), MAKE_HEX_DOUBLE(0x1.11065895048ddp-56,0x111065895048ddLL,-56) },
        { MAKE_HEX_DOUBLE(0x1.d5818dcfba487p-1,0x1d5818dcfba487LL,-1), MAKE_HEX_DOUBLE(0x1.2ed02d75b3707p-56,0x12ed02d75b3707LL,-56) },
        { MAKE_HEX_DOUBLE(0x1.ea4afa2a490dap-1,0x1ea4afa2a490daLL,-1), MAKE_HEX_DOUBLE(-0x1.e9c23179c2893p-55,-0x1e9c23179c2893LL,-55) },
        { MAKE_HEX_DOUBLE(0x1p+0,0x1LL,0), MAKE_HEX_DOUBLE(0x0p+0,0x0LL,0) },
        { MAKE_HEX_DOUBLE(0x1.0b5586cf9890fp+0,0x10b5586cf9890fLL,0), MAKE_HEX_DOUBLE(0x1.8a62e4adc610bp-54,0x18a62e4adc610bLL,-54) },
        { MAKE_HEX_DOUBLE(0x1.172b83c7d517bp+0,0x1172b83c7d517bLL,0), MAKE_HEX_DOUBLE(-0x1.19041b9d78a76p-55,-0x119041b9d78a76LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.2387a6e756238p+0,0x12387a6e756238LL,0), MAKE_HEX_DOUBLE(0x1.9b07eb6c70573p-54,0x19b07eb6c70573LL,-54) },
        { MAKE_HEX_DOUBLE(0x1.306fe0a31b715p+0,0x1306fe0a31b715LL,0), MAKE_HEX_DOUBLE(0x1.6f46ad23182e4p-55,0x16f46ad23182e4LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.3dea64c123422p+0,0x13dea64c123422LL,0), MAKE_HEX_DOUBLE(0x1.ada0911f09ebcp-55,0x1ada0911f09ebcLL,-55) },
        { MAKE_HEX_DOUBLE(0x1.4bfdad5362a27p+0,0x14bfdad5362a27LL,0), MAKE_HEX_DOUBLE(0x1.d4397afec42e2p-56,0x1d4397afec42e2LL,-56) },
        { MAKE_HEX_DOUBLE(0x1.5ab07dd485429p+0,0x15ab07dd485429LL,0), MAKE_HEX_DOUBLE(0x1.6324c054647adp-54,0x16324c054647adLL,-54) },
        { MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp+0,0x16a09e667f3bcdLL,0), MAKE_HEX_DOUBLE(-0x1.bdd3413b26456p-54,-0x1bdd3413b26456LL,-54) }

    };
    int index = (int) rint( x.hi * 16.0 );
    x.hi -= (double) index * 0.0625;
    
    // canonicalize x
    double temp = x.hi;
    x.hi += x.lo;
    x.lo -= x.hi - temp;
    
    // Minimax polynomial for (exp2(x)-1)/x, over the range [-1/32, 1/32].  Max Error: 2 * 0x1.e112p-87
    const double_double c[] = {
        {MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1,0x162e42fefa39efLL,-1), MAKE_HEX_DOUBLE(0x1.abc9e3ac1d244p-56,0x1abc9e3ac1d244LL,-56)}, 
        {MAKE_HEX_DOUBLE(0x1.ebfbdff82c58fp-3,0x1ebfbdff82c58fLL,-3), MAKE_HEX_DOUBLE(-0x1.5e4987a631846p-57,-0x15e4987a631846LL,-57)}, 
        {MAKE_HEX_DOUBLE(0x1.c6b08d704a0cp-5,0x1c6b08d704a0cLL,-5), MAKE_HEX_DOUBLE(-0x1.d323200a05713p-59,-0x1d323200a05713LL,-59)}, 
        {MAKE_HEX_DOUBLE(0x1.3b2ab6fba4e7ap-7,0x13b2ab6fba4e7aLL,-7), MAKE_HEX_DOUBLE(0x1.c5ee8f8b9f0c1p-63,0x1c5ee8f8b9f0c1LL,-63)}, 
        {MAKE_HEX_DOUBLE(0x1.5d87fe78a672ap-10,0x15d87fe78a672aLL,-10), MAKE_HEX_DOUBLE(0x1.884e5e5cc7eccp-64,0x1884e5e5cc7eccLL,-64)}, 
        {MAKE_HEX_DOUBLE(0x1.430912f7e8373p-13,0x1430912f7e8373LL,-13), MAKE_HEX_DOUBLE(0x1.4f1b59514a326p-67,0x14f1b59514a326LL,-67)}, 
        {MAKE_HEX_DOUBLE(0x1.ffcbfc5985e71p-17,0x1ffcbfc5985e71LL,-17), MAKE_HEX_DOUBLE(-0x1.db7d6a0953b78p-71,-0x1db7d6a0953b78LL,-71)}, 
        {MAKE_HEX_DOUBLE(0x1.62c150eb16465p-20,0x162c150eb16465LL,-20), MAKE_HEX_DOUBLE(0x1.e0767c2d7abf5p-80,0x1e0767c2d7abf5LL,-80)}, 
        {MAKE_HEX_DOUBLE(0x1.b52502b5e953p-24,0x1b52502b5e953LL,-24), MAKE_HEX_DOUBLE(0x1.6797523f944bcp-78,0x16797523f944bcLL,-78)}
    };
    size_t count = sizeof( c ) / sizeof( c[0] );
    
    // Do polynomial
    double_double r = c[count-1];
    for( j = (int) count-2; j >= 0; j-- )
        r = add_dd( c[j], mul_dd( r, x ) );
    
    // unwind approximation
    r = mul_dd( r, x );     // before: r =(exp2(x)-1)/x;   after: r = exp2(x) - 1
    
    // correct for [-0.5, 0.5] -> [-1/32, 1/32] reduction above
    //  exp2(x) = (r + 1) * correction = r * correction + correction
    r = mul_dd( r, corrections[index+8] );
    r = add_dd( r, corrections[index+8] );
    
// Format result for output:
    
    // Get mantissa
    long double m = ((long double) r.hi + (long double) r.lo );
    
    // Handle a pesky overflow cases when long double = double
    if( i > 512 )
    {
        m *=  MAKE_HEX_DOUBLE(0x1.0p512,0x1LL,512);
        i -= 512;
    }
    else if( i < -512 )
    {
        m *= MAKE_HEX_DOUBLE(0x1.0p-512,0x1LL,-512);
        i += 512;
    }
    
    return m * ldexpl( 1.0L, i );
}


static double fallback_frexp( double x, int *iptr )
{
    cl_ulong u, v;
    double fu, fv;

    memcpy( &u, &x, sizeof(u));

    cl_ulong exponent = u &  0x7ff0000000000000ULL;
    cl_ulong mantissa = u & ~0x7ff0000000000000ULL;

    // add 1 to the exponent
    exponent += 0x0010000000000000ULL;
    
    if( (cl_long) exponent < (cl_long) 0x0020000000000000LL )
    { // subnormal, NaN, Inf
        mantissa |= 0x3fe0000000000000ULL;
        
        v = mantissa & 0xfff0000000000000ULL;
        u = mantissa;
        memcpy( &fv, &v, sizeof(v));
        memcpy( &fu, &u, sizeof(u));
        
        fu -= fv;

        memcpy( &v, &fv, sizeof(v));
        memcpy( &u, &fu, sizeof(u));
        
        exponent = u &  0x7ff0000000000000ULL;
        mantissa = u & ~0x7ff0000000000000ULL;
        
        *iptr = (exponent >> 52) + (-1022 + 1 -1022);
        u = mantissa | 0x3fe0000000000000ULL;
        memcpy( &fu, &u, sizeof(u));
        return fu;
    }
    
    *iptr = (exponent >> 52) - 1023;
    u = mantissa | 0x3fe0000000000000ULL;
    memcpy( &fu, &u, sizeof(u));
    return fu;
}

// Assumes zeros, infinities and NaNs handed elsewhere
static inline int extract( double x, cl_ulong *mant );
static inline int extract( double x, cl_ulong *mant )
{
    static double (*frexpp)(double, int*) = NULL;
    int e;
    
    // verify that frexp works properly
    if( NULL == frexpp )
    {
        if( 0.5 == frexp( MAKE_HEX_DOUBLE(0x1.0p-1030, 0x1LL, -1030), &e ) && e == -1029 )
            frexpp = frexp;
        else
            frexpp = fallback_frexp;
    }

    *mant = (cl_ulong) (MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64) * fabs( frexpp( x, &e )));         
    return e - 1;
}

// Return 128-bit product of a*b  as (hi << 64) + lo
static inline void mul128( cl_ulong a, cl_ulong b, cl_ulong *hi, cl_ulong *lo );
static inline void mul128( cl_ulong a, cl_ulong b, cl_ulong *hi, cl_ulong *lo )
{
    cl_ulong alo = a & 0xffffffffULL;
    cl_ulong ahi = a >> 32;
    cl_ulong blo = b & 0xffffffffULL;
    cl_ulong bhi = b >> 32;
    cl_ulong aloblo = alo * blo;
    cl_ulong alobhi = alo * bhi;
    cl_ulong ahiblo = ahi * blo;
    cl_ulong ahibhi = ahi * bhi;

    alobhi += (aloblo >> 32) + (ahiblo & 0xffffffffULL);  // cannot overflow: (2^32-1)^2 + 2 * (2^32-1)   = (2^64 - 2^33 + 1) + (2^33 - 2) = 2^64 - 1
    *hi = ahibhi + (alobhi >> 32) + (ahiblo >> 32);       // cannot overflow: (2^32-1)^2 + 2 * (2^32-1)   = (2^64 - 2^33 + 1) + (2^33 - 2) = 2^64 - 1
    *lo = (aloblo & 0xffffffffULL) | (alobhi << 32);
}

// Move the most significant non-zero bit to the MSB
// Note: not general. Only works if the most significant non-zero bit is at MSB-1
static inline void renormalize( cl_ulong *hi, cl_ulong *lo, int *exponent )
{
    if( 0 == (0x8000000000000000ULL & *hi ))
    {
        *hi <<= 1;
        *hi |= *lo >> 63;
        *lo <<= 1;
        *exponent -= 1;
    }
}

static double round_to_nearest_even_double( cl_ulong hi, cl_ulong lo, int exponent );
static double round_to_nearest_even_double( cl_ulong hi, cl_ulong lo, int exponent )
{
    union{ cl_ulong u; cl_double d;} u;
    
    // edges
    if( exponent > 1023 )        return INFINITY;
    if( exponent == -1075 && (hi | (lo!=0)) > 0x8000000000000000ULL )  
        return MAKE_HEX_DOUBLE(0x1.0p-1074, 0x1LL, -1074);
    if( exponent <= -1075 )       return 0.0;
        
    //Figure out which bits go where
    int shift = 11;
    if( exponent < -1022 )
    {
        shift -= 1022 + exponent;               // subnormal: shift is not 52
        exponent = -1023;                       //              set exponent to 0
    }
    else
        hi &= 0x7fffffffffffffffULL;           // normal: leading bit is implicit. Remove it.

    // Assemble the double (round toward zero)
    u.u = (hi >> shift) | ((cl_ulong) (exponent + 1023) << 52);

    // put a representation of the residual bits into hi
    hi <<= (64-shift);     
    hi |= lo >> shift;
    lo <<= (64-shift );
    hi |= lo != 0;
    
    //round to nearest, ties to even
    if( hi < 0x8000000000000000ULL )    return u.d;        
    if( hi == 0x8000000000000000ULL )   u.u += u.u & 1ULL;
    else                                u.u++;
    
    return u.d;
}

// Shift right.  Bits lost on the right will be OR'd together and OR'd with the LSB
static inline void shift_right_sticky_128( cl_ulong *hi, cl_ulong *lo, int shift );
static inline void shift_right_sticky_128( cl_ulong *hi, cl_ulong *lo, int shift )
{
    cl_ulong sticky = 0;
    cl_ulong h = *hi;
    cl_ulong l = *lo;
    
    if( shift >= 64 )
    {
        shift -= 64;
        sticky = 0 != lo;
        l = h;
        h = 0;
        if( shift >= 64 )
        {
            sticky |= (0 != l);
            l = 0;
        }
        else
        {
            sticky |= (0 != (l << (64-shift)));
            l >>= shift;
        }
    }
    else
    {
        sticky |= (0 != (l << (64-shift)));
        l >>= shift;
        l |=  h << (64-shift);
        h >>= shift;
    }

    *lo = l | sticky;
    *hi = h;
}

// 128-bit add  of ((*hi << 64) + *lo) + ((chi << 64) + clo) 
// If the 129 bit result doesn't fit, bits lost off the right end will be OR'd with the LSB
static inline void add128( cl_ulong *hi, cl_ulong *lo, cl_ulong chi, cl_ulong clo, int *exp );
static inline void add128( cl_ulong *hi, cl_ulong *lo, cl_ulong chi, cl_ulong clo, int *exponent )
{
    cl_ulong carry, carry2;
    // extended precision add
    clo = add_carry(*lo, clo, &carry);
    chi = add_carry(*hi, chi, &carry2);
    chi = add_carry(chi, carry, &carry);
    
    //If we overflowed the 128 bit result
    if( carry || carry2 )
    {
        carry = clo & 1;                        // set aside low bit
        clo >>= 1;                              // right shift low 1
        clo |= carry;                           // or back in the low bit, so we don't come to believe this is an exact half way case for rounding
        clo |= chi << 63;                       // move lowest high bit into highest bit of lo
        chi >>= 1;                              // right shift hi
        chi |= 0x8000000000000000ULL;           // move the carry bit into hi.
        *exponent = *exponent + 1;
    }
    
    *hi = chi;
    *lo = clo;
}

// 128-bit subtract  of ((chi << 64) + clo)  - ((*hi << 64) + *lo) 
static inline void sub128( cl_ulong *chi, cl_ulong *clo, cl_ulong hi, cl_ulong lo, cl_ulong *signC, int *expC );
static inline void sub128( cl_ulong *chi, cl_ulong *clo, cl_ulong hi, cl_ulong lo, cl_ulong *signC, int *expC )
{
    cl_ulong rHi = *chi;
    cl_ulong rLo = *clo;
    cl_ulong carry, carry2;
    
    //extended precision subtract
    rLo = sub_carry(rLo, lo, &carry);
    rHi = sub_carry(rHi, hi, &carry2);
    rHi = sub_carry(rHi, carry, &carry);
    
    // Check for sign flip
    if( carry || carry2 )
    {   
        *signC ^= 0x8000000000000000ULL;

        //negate rLo, rHi:   -x = (x ^ -1) + 1
        rLo ^= -1ULL;
        rHi ^= -1ULL;
        rLo++;
        rHi += 0 == rLo;
    }

    // normalize -- move the most significant non-zero bit to the MSB, and adjust exponent accordingly
    if( rHi == 0 )  
    {
        rHi = rLo;
        *expC = *expC - 64;
        rLo = 0;
    }
    
    if( rHi )
    {
        int shift = 32;
        cl_ulong test = 1ULL << 32;
        while( 0 == (rHi & 0x8000000000000000ULL))
        {
            if( rHi < test )
            {
                rHi <<= shift;
                rHi |= rLo >> (64-shift);
                rLo <<= shift;
                *expC = *expC - shift;
            }
            shift >>= 1;
            test <<= shift;
        }
    }
    else 
    {
        //zero
        *expC = INT_MIN;
        *signC = 0;
    }


    *chi = rHi;
    *clo = rLo;
}

long double reference_fmal( long double x, long double y, long double z)
{
    static const cl_ulong kMSB = 0x8000000000000000ULL;

    // cast values back to double. This is an exact function, so 
    double a = x;
    double b = y; 
    double c = z;
    
    // Make bits accessible
    union{ cl_ulong u; cl_double d; } ua; ua.d = a;
    union{ cl_ulong u; cl_double d; } ub; ub.d = b;
    union{ cl_ulong u; cl_double d; } uc; uc.d = c;
    
    // deal with Nans, infinities and zeros
    if( isnan( a ) || isnan( b ) || isnan(c)    || 
        isinf( a ) || isinf( b ) || isinf(c)    || 
        0 == ( ua.u & ~kMSB)                ||  // a == 0, defeat host FTZ behavior
        0 == ( ub.u & ~kMSB)                ||  // b == 0, defeat host FTZ behavior
        0 == ( uc.u & ~kMSB)                )   // c == 0, defeat host FTZ behavior
    {
        if( isinf( c ) && !isinf(a) && !isinf(b) )
            return (c + a) + b;

        a = (double) reference_multiplyl( a, b );   // some risk that the compiler will insert a non-compliant fma here on some platforms.
        return reference_addl(a, c);                // We use STDC FP_CONTRACT OFF above to attempt to defeat that.
    }
    
    // extract exponent and mantissa 
    //   exponent is a standard unbiased signed integer
    //   mantissa is a cl_uint, with leading non-zero bit positioned at the MSB
    cl_ulong mantA, mantB, mantC;
    int expA = extract( a, &mantA );
    int expB = extract( b, &mantB );
    int expC = extract( c, &mantC );
    cl_ulong signC = uc.u & kMSB;               // We'll need the sign bit of C later to decide if we are adding or subtracting
        
// exact product of A and B
    int exponent = expA + expB;
    cl_ulong sign = (ua.u ^ ub.u) & kMSB;
    cl_ulong hi, lo;
    mul128( mantA, mantB, &hi, &lo );
    
    // renormalize
    if( 0 == (kMSB & hi) )
    {
        hi <<= 1;
        hi |= lo >> 63;
        lo <<= 1;
    }
    else
        exponent++;         // 2**63 * 2**63 gives 2**126. If the MSB was set, then our exponent increased.
    
//infinite precision add 
    cl_ulong chi = mantC;
    cl_ulong clo = 0;
    
    if( exponent >= expC )
    {
        // Normalize C relative to the product
        if( exponent > expC )
            shift_right_sticky_128( &chi, &clo, exponent - expC );

        // Add
        if( sign ^ signC )
            sub128( &hi, &lo, chi, clo, &sign, &exponent );
        else
            add128( &hi, &lo, chi, clo, &exponent );
    }
    else 
    {
        // Shift the product relative to C so that their exponents match
        shift_right_sticky_128( &hi, &lo, expC - exponent );

        // add
        if( sign ^ signC )
            sub128( &chi, &clo, hi, lo, &signC, &expC );
        else
            add128( &chi, &clo, hi, lo, &expC );
            
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




long double reference_madl( long double a, long double b, long double c) { return a * b + c; }

//long double my_nextafterl(long double x, long double y){  return (long double) nextafter( (double) x, (double) y ); }

long double reference_recipl( long double x){ return 1.0L / x; }

long double reference_rootnl( long double x, int i)
{
    double hi,  lo;
    long double l;
    //rootn ( x, 0 )  returns a NaN. 
    if( 0 == i )
        return cl_make_nan();

    //rootn ( x, n )  returns a NaN for x < 0 and n is even. 
    if( x < 0.0L && 0 == (i&1) )
        return cl_make_nan();

    if( isinf(x) )
    {
        if( i < 0 )
            return reference_copysignl(0.0L, x);

        return x;
    }

    if( x == 0.0 )
    {
        switch( i & 0x80000001 )
        {
            //rootn ( +-0,  n ) is +0 for even n > 0. 
            case 0:
                return 0.0L;

            //rootn ( +-0,  n ) is +-0 for odd n > 0. 
            case 1:
                return x;

            //rootn ( +-0,  n ) is +inf for even n < 0. 
            case 0x80000000:
                return INFINITY;

            //rootn ( +-0,  n ) is +-inf for odd n < 0. 
            case 0x80000001:
                return copysign(INFINITY, x);
        }    
    }
    
    if( i == 1 )
        return x;
    
    if( i == -1 )
        return 1.0 / x;
    
    long double sign = x;
    x = reference_fabsl(x);    
    double iHi, iLo;
    DivideDD(&iHi, &iLo, 1.0, i);
    x = reference_powl(x, iHi) * reference_powl(x, iLo);
    
    return reference_copysignl( x, sign );
    
}

long double reference_rsqrtl( long double x){ return 1.0L / sqrtl(x); }
//long double reference_sincosl( long double x, long double *c ){ *c = reference_cosl(x); return reference_sinl(x); }
long double reference_sinpil( long double x)
{   
    double r = reduce1l(x); 
        
    // reduce to [-0.5, 0.5]
    if( r < -0.5L )
        r = -1.0L - r;
    else if ( r > 0.5L )
        r = 1.0L - r;

    // sinPi zeros have the same sign as x
    if( r == 0.0L )
        return reference_copysignl(0.0L, x);

    return reference_sinl( r * M_PIL );   
}

long double reference_tanpil( long double x)
{
    // set aside the sign  (allows us to preserve sign of -0)
    long double sign = reference_copysignl( 1.0L, x);
    long double z = reference_fabsl(x);

    // if big and even  -- caution: only works if x only has single precision
    if( z >= MAKE_HEX_LONG(0x1.0p53L, 0x1LL, 53) )
    {
        if( z == INFINITY )
            return x - x;       // nan
            
        return reference_copysignl( 0.0L, x);   // tanpi ( n ) is copysign( 0.0, n)  for even integers n.
    }
    
    // reduce to the range [ -0.5, 0.5 ]
    long double nearest = reference_rintl( z );     // round to nearest even places n + 0.5 values in the right place for us
    int64_t i = (int64_t) nearest;          // test above against 0x1.0p53 avoids overflow here
    z -= nearest;                   
    
    //correction for odd integer x for the right sign of zero
    if( (i&1) && z == 0.0L )
        sign = -sign;
    
    // track changes to the sign
    sign *= reference_copysignl(1.0L, z);       // really should just be an xor
    z = reference_fabsl(z);                    // remove the sign again
    
    // reduce once more
    // If we don't do this, rounding error in z * M_PI will cause us not to return infinities properly
    if( z > 0.25L )
    {
        z = 0.5L - z;
        return sign / reference_tanl( z * M_PIL );      // use system tan to get the right result
    }
    
    //
    return sign * reference_tanl( z * M_PIL );          // use system tan to get the right result
}

long double reference_pownl( long double x, int i ){ return reference_powl( x, (long double) i ); }

long double reference_powrl( long double x, long double y )
{  
    //powr ( x, y ) returns NaN for x < 0. 
    if( x < 0.0L )
        return cl_make_nan();

    //powr ( x, NaN ) returns the NaN for x >= 0. 
    //powr ( NaN, y ) returns the NaN. 
    if( isnan(x) || isnan(y) )
        return x + y;   // Note: behavior different here than for pow(1,NaN), pow(NaN, 0)

    if( x == 1.0L )
    {
        //powr ( +1, +-inf ) returns NaN. 
        if( reference_fabsl(y) == INFINITY )
            return cl_make_nan();
        
        //powr ( +1, y ) is 1 for finite y.    (NaN handled above)
        return 1.0L;
    }

    if( y == 0.0L )
    {
        //powr ( +inf, +-0 ) returns NaN. 
        //powr ( +-0, +-0 ) returns NaN. 
        if( x == 0.0L || x == INFINITY )
            return cl_make_nan(); 
    
        //powr ( x, +-0 ) is 1 for finite x > 0.  (x <= 0, NaN, INF already handled above)
        return 1.0L;
    }
    
    if( x == 0.0L )
    {
        //powr ( +-0, -inf) is +inf. 
        //powr ( +-0, y ) is +inf for finite y < 0. 
        if( y < 0.0L )
            return INFINITY;
            
        //powr ( +-0, y ) is +0 for y > 0.    (NaN, y==0 handled above)
        return 0.0L;
    }
        
	return reference_powl( x, y );
}

//long double my_fdiml( long double x, long double y){ return fdim( (double) x, (double) y ); }
long double reference_addl( long double x, long double y)
{ 
    volatile double a = (double) x;
    volatile double b = (double) y;

#if defined( __SSE2__ )
    // defeat x87
    __m128d va = _mm_set_sd( (double) a );
    __m128d vb = _mm_set_sd( (double) b );
    va = _mm_add_sd( va, vb );
    _mm_store_sd( (double*) &a, va );
#else
    a += b;
#endif
    return (long double) a;
}

long double reference_subtractl( long double x, long double y)
{ 
    volatile double a = (double) x;
    volatile double b = (double) y;

#if defined( __SSE2__ )
    // defeat x87
    __m128d va = _mm_set_sd( (double) a );
    __m128d vb = _mm_set_sd( (double) b );
    va = _mm_sub_sd( va, vb );
    _mm_store_sd( (double*) &a, va );
#else
    a -= b;
#endif
    return (long double) a;
}

long double reference_multiplyl( long double x, long double y)
{ 
    volatile double a = (double) x;
    volatile double b = (double) y;

#if defined( __SSE2__ )
    // defeat x87
    __m128d va = _mm_set_sd( (double) a );
    __m128d vb = _mm_set_sd( (double) b );
    va = _mm_mul_sd( va, vb );
    _mm_store_sd( (double*) &a, va );
#else
    a *= b;
#endif
    return (long double) a;
}

/*long double my_remquol( long double x, long double y, int *iptr )
{
    if( isnan(x) || isnan(y) ||
        fabs(x) == INFINITY  ||
        y == 0.0 )
    {
        *iptr = 0;
        return NAN;
    }

    return remquo( (double) x, (double) y, iptr );
}*/
long double reference_lgamma_rl( long double x, int *signp )
{
//	long double lgamma_val = (long double)reference_lgamma( (double)x );
//	*signp = signgam;
    *signp = 0;
	return x;
}


int reference_isequall( long double x, long double y){ return x == y; }
int reference_isfinitel( long double x){ return 0 != isfinite(x); }
int reference_isgreaterl( long double x, long double y){ return x > y; }
int reference_isgreaterequall( long double x, long double y){ return x >= y; }
int reference_isinfl( long double x){ return 0 != isinf(x); }
int reference_islessl( long double x, long double y){ return x < y; }
int reference_islessequall( long double x, long double y){ return x <= y; }
int reference_islessgreaterl( long double x, long double y){  return 0 != islessgreater( x, y ); }
int reference_isnanl( long double x){ return 0 != isnan( x ); }
int reference_isnormall( long double x){ return 0 != isnormal( (double) x ); }
int reference_isnotequall( long double x, long double y){ return x != y; }
int reference_isorderedl( long double x, long double y){ return x == x && y == y; }
int reference_isunorderedl( long double x, long double y){ return isnan(x) || isnan( y ); }
int reference_signbitl( long double x){ return 0 != signbit( x ); }

long double reference_copysignl( long double x, long double y);
long double reference_roundl( long double x );
long double reference_cbrtl(long double x);

long double reference_copysignl( long double x, long double y )
{
    // We hope that the long double to double conversion proceeds with sign fidelity,
    // even for zeros and NaNs
    union{ double d; cl_ulong u;}u; u.d = (double) y;
    
    x = reference_fabsl(x);
    if( u.u >> 63 )
        x = -x;
        
    return x;
}

long double reference_roundl( long double x )
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here
    return round( (double) x );
}

long double reference_truncl( long double x )
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here
    return trunc( (double) x );
}

static long double reference_scalblnl(long double x, long n);

long double reference_cbrtl(long double x)
{
	double yhi = MAKE_HEX_DOUBLE(0x1.5555555555555p-2,0x15555555555555LL,-2);
	double ylo = MAKE_HEX_DOUBLE(0x1.558p-56,0x1558LL,-56);
	
	double fabsx = reference_fabs( x );
	
    if( isnan(x) || fabsx == 1.0 || fabsx == 0.0 || isinf(x) )
        return x; 
	
	double iy = 0.0;
	double log2x_hi, log2x_lo;
	
	// extended precision log .... accurate to at least 64-bits + couple of guard bits
	__log2_ep(&log2x_hi, &log2x_lo, fabsx);
	
	double ylog2x_hi, ylog2x_lo;
	
	double y_hi = yhi;
	double y_lo = ylo;
	
	// compute product of y*log2(x)
	MulDD(&ylog2x_hi, &ylog2x_lo, log2x_hi, log2x_lo, y_hi, y_lo);
	
	long double powxy;
	if(isinf(ylog2x_hi) || (reference_fabs(ylog2x_hi) > 2200)) {
		powxy = reference_signbit(ylog2x_hi) ? MAKE_HEX_DOUBLE(0x0p0, 0x0LL, 0) : INFINITY;
    } else {
        // separate integer + fractional part
        long int m = lrint(ylog2x_hi);
        AddDD(&ylog2x_hi, &ylog2x_lo, ylog2x_hi, ylog2x_lo, -m, 0.0);
        
        // revert to long double arithemtic
        long double ylog2x = (long double) ylog2x_hi + (long double) ylog2x_lo;
        powxy = reference_exp2l( ylog2x );
        powxy = reference_scalblnl(powxy, m); 
    }
				
	return reference_copysignl( powxy, x ); 
}

/*
long double scalbnl( long double x, int i )
{
    //suitable for checking double precision scalbn only
    
    if( i > 3000 )
        return copysignl( INFINITY, x);
    if( i < -3000 )
        return copysignl( 0.0L, x);

    if( i > 0 )
    {
        while( i >= 1000 )
        {
            x *= MAKE_HEX_LONG(0x1.0p1000L, 0x1LL, 1000);
            i -= 1000;
        }
        
        union{ cl_ulong u; double d;}u;
        u.u = (cl_ulong)( i + 1023 ) << 52;
        x *= (long double) u.d;
    }
    else if( i < 0 )
    {
        while( i <= -1000 )
        {
            x *= MAKE_HEX_LONG(0x1.0p-1000L, 0x1LL, -1000);
            i += 1000;
        }
        
        union{ cl_ulong u; double d;}u;
        u.u = (cl_ulong)( i + 1023 ) << 52;
        x *= (long double) u.d;
    }
    
    return x;
}
*/

long double reference_rintl( long double x )
{
#if defined(__PPC__)
  // On PPC, long doubles are maintained as 2 doubles. Therefore, the combined 
  // mantissa can represent more than LDBL_MANT_DIG binary digits.
  x = rintl(x);
#else
    static long double magic[2] = { 0.0L, 0.0L};
    
    if( 0.0L == magic[0] )
    {
        magic[0] = scalbnl(0.5L, LDBL_MANT_DIG);
        magic[1] = scalbnl(-0.5L, LDBL_MANT_DIG);
    }

    if( reference_fabsl(x) < magic[0] && x != 0.0L )
    {
        long double m = magic[ x < 0 ];
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
    double thi = 1.0 / sqrt( xhi );
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

long double reference_acoshl( long double x )
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
    if( isnan(x) || isinf(x))
        return x + fabsl(x);
        
    if( x < 1.0L )
        return cl_make_nan();

    if( x == 1.0L )
        return 0.0L;
    
    if( x > MAKE_HEX_LONG(0x1.0p60L, 0x1LL, 60) )
        return reference_logl(x) + 0.693147180559945309417232121458176568L;
        
    if( x > 2.0L )
        return reference_logl(2.0L * x - 1.0L / (x + sqrtl(x*x - 1.0L)));
    
    double hi, lo;
    MulD(&hi, &lo, x, x);
    AddDD(&hi, &lo, hi, lo, -1.0, 0.0);
    __sqrt_ep(&hi, &lo, hi, lo);
    AddDD(&hi, &lo, hi, lo, x, 0.0);
    double correction = lo / hi;
    __log2_ep(&hi, &lo, hi);
	double log2Hi = MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1,  0x162e42fefa39efLL, -53);
	double log2Lo = MAKE_HEX_DOUBLE(0x1.abc9e3b39803fp-56, 0x1abc9e3b39803fLL, -108);
    MulDD(&hi, &lo, hi, lo, log2Hi, log2Lo);
    AddDD(&hi, &lo, hi, lo, correction, 0.0);
    
    return hi + lo;
}

long double reference_asinhl( long double x )
{
    long double cutoff = 0.0L;
    const long double ln2 = MAKE_HEX_LONG( 0xb.17217f7d1cf79abp-4L, 0xb17217f7d1cf79abLL, -64);	
    
    if( cutoff == 0.0L )
        cutoff = reference_ldexpl(1.0L, -LDBL_MANT_DIG);

    if( isnan(x) || isinf(x) )
        return x + x;
        
    long double absx = reference_fabsl(x);
    if( absx < cutoff )
        return x;

    long double sign = reference_copysignl(1.0L, x);

	if( absx <= 4.0/3.0 ) {
		return sign * reference_log1pl( absx + x*x / (1.0 + sqrtl(1.0 + x*x)));
	}
	else if( absx <= MAKE_HEX_LONG(0x1.0p27L, 0x1LL, 27) ) {
		return sign * reference_logl( 2.0L * absx + 1.0L / (sqrtl( x * x + 1.0 ) + absx));
	}
    else {
        return sign * ( reference_logl( absx ) + ln2 );
    }
}

long double reference_atanhl( long double x )
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
    if( isnan(x)  )
        return x + x;
    
    long double signed_half = reference_copysignl( 0.5L, x );
    x = reference_fabsl(x);
    if( x > 1.0L )
        return cl_make_nan();
        
    if( x < 0.5L )
        return signed_half * reference_log1pl( 2.0L * ( x + x*x / (1-x) ) );
    
    return signed_half * reference_log1pl(2.0L * x / (1-x));
}

long double reference_exp2l(  long double z)
{
    double_double x;
    int j;
    
    // Handle NaNs
    if( isnan(z) )
        return z;
    
    // init x
    x.hi = z;
    x.lo = z - x.hi;
    
    //Deal with overflow and underflow for exp2(x) stage next
    if( x.hi >= 1025 )
        return INFINITY;
    
    if( x.hi < -1075-24 )
        return +0.0;
    
    // find nearest integer to x
    int i = (int) rint(x.hi);
    
    // x now holds fractional part.  The result would be then 2**i  * exp2( x )
    x.hi -= i;
    
    // We could attempt to find a minimax polynomial for exp2(x) over the range x = [-0.5, 0.5].
    // However, this would converge very slowly near the extrema, where 0.5**n is not a lot different
    // from 0.5**(n+1), thereby requiring something like a 20th order polynomial to get 53 + 24 bits 
    // of precision. Instead we further reduce the range to [-1/32, 1/32] by observing that 
    //
    //  2**(a+b) = 2**a * 2**b
    //
    // We can thus build a table of 2**a values for a = n/16, n = [-8, 8], and reduce the range
    // of x to [-1/32, 1/32] by subtracting away the nearest value of n/16 from x.
    const double_double corrections[17] = 
    {
        { MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp-1,0x16a09e667f3bcdLL,-1), MAKE_HEX_DOUBLE(-0x1.bdd3413b26456p-55,-0x1bdd3413b26456LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.7a11473eb0187p-1,0x17a11473eb0187LL,-1), MAKE_HEX_DOUBLE(-0x1.41577ee04992fp-56,-0x141577ee04992fLL,-56) },
        { MAKE_HEX_DOUBLE(0x1.8ace5422aa0dbp-1,0x18ace5422aa0dbLL,-1), MAKE_HEX_DOUBLE(0x1.6e9f156864b27p-55,0x16e9f156864b27LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.9c49182a3f09p-1,0x19c49182a3f09LL,-1), MAKE_HEX_DOUBLE(0x1.c7c46b071f2bep-57,0x1c7c46b071f2beLL,-57) },
        { MAKE_HEX_DOUBLE(0x1.ae89f995ad3adp-1,0x1ae89f995ad3adLL,-1), MAKE_HEX_DOUBLE(0x1.7a1cd345dcc81p-55,0x17a1cd345dcc81LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.c199bdd85529cp-1,0x1c199bdd85529cLL,-1), MAKE_HEX_DOUBLE(0x1.11065895048ddp-56,0x111065895048ddLL,-56) },
        { MAKE_HEX_DOUBLE(0x1.d5818dcfba487p-1,0x1d5818dcfba487LL,-1), MAKE_HEX_DOUBLE(0x1.2ed02d75b3707p-56,0x12ed02d75b3707LL,-56) },
        { MAKE_HEX_DOUBLE(0x1.ea4afa2a490dap-1,0x1ea4afa2a490daLL,-1), MAKE_HEX_DOUBLE(-0x1.e9c23179c2893p-55,-0x1e9c23179c2893LL,-55) },
        { MAKE_HEX_DOUBLE(0x1p+0,0x1LL,0), MAKE_HEX_DOUBLE(0x0p+0,0x0LL,0) },
        { MAKE_HEX_DOUBLE(0x1.0b5586cf9890fp+0,0x10b5586cf9890fLL,0), MAKE_HEX_DOUBLE(0x1.8a62e4adc610bp-54,0x18a62e4adc610bLL,-54) },
        { MAKE_HEX_DOUBLE(0x1.172b83c7d517bp+0,0x1172b83c7d517bLL,0), MAKE_HEX_DOUBLE(-0x1.19041b9d78a76p-55,-0x119041b9d78a76LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.2387a6e756238p+0,0x12387a6e756238LL,0), MAKE_HEX_DOUBLE(0x1.9b07eb6c70573p-54,0x19b07eb6c70573LL,-54) },
        { MAKE_HEX_DOUBLE(0x1.306fe0a31b715p+0,0x1306fe0a31b715LL,0), MAKE_HEX_DOUBLE(0x1.6f46ad23182e4p-55,0x16f46ad23182e4LL,-55) },
        { MAKE_HEX_DOUBLE(0x1.3dea64c123422p+0,0x13dea64c123422LL,0), MAKE_HEX_DOUBLE(0x1.ada0911f09ebcp-55,0x1ada0911f09ebcLL,-55) },
        { MAKE_HEX_DOUBLE(0x1.4bfdad5362a27p+0,0x14bfdad5362a27LL,0), MAKE_HEX_DOUBLE(0x1.d4397afec42e2p-56,0x1d4397afec42e2LL,-56) },
        { MAKE_HEX_DOUBLE(0x1.5ab07dd485429p+0,0x15ab07dd485429LL,0), MAKE_HEX_DOUBLE(0x1.6324c054647adp-54,0x16324c054647adLL,-54) },
        { MAKE_HEX_DOUBLE(0x1.6a09e667f3bcdp+0,0x16a09e667f3bcdLL,0), MAKE_HEX_DOUBLE(-0x1.bdd3413b26456p-54,-0x1bdd3413b26456LL,-54) }
    };
    int index = (int) rint( x.hi * 16.0 );
    x.hi -= (double) index * 0.0625;
    
    // canonicalize x
    double temp = x.hi;
    x.hi += x.lo;
    x.lo -= x.hi - temp;
    
    // Minimax polynomial for (exp2(x)-1)/x, over the range [-1/32, 1/32].  Max Error: 2 * 0x1.e112p-87
    const double_double c[] = {
        {MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1,0x162e42fefa39efLL,-1), MAKE_HEX_DOUBLE(0x1.abc9e3ac1d244p-56,0x1abc9e3ac1d244LL,-56)}, 
        {MAKE_HEX_DOUBLE(0x1.ebfbdff82c58fp-3,0x1ebfbdff82c58fLL,-3), MAKE_HEX_DOUBLE(-0x1.5e4987a631846p-57,-0x15e4987a631846LL,-57)}, 
        {MAKE_HEX_DOUBLE(0x1.c6b08d704a0cp-5,0x1c6b08d704a0cLL,-5), MAKE_HEX_DOUBLE(-0x1.d323200a05713p-59,-0x1d323200a05713LL,-59)}, 
        {MAKE_HEX_DOUBLE(0x1.3b2ab6fba4e7ap-7,0x13b2ab6fba4e7aLL,-7), MAKE_HEX_DOUBLE(0x1.c5ee8f8b9f0c1p-63,0x1c5ee8f8b9f0c1LL,-63)}, 
        {MAKE_HEX_DOUBLE(0x1.5d87fe78a672ap-10,0x15d87fe78a672aLL,-10), MAKE_HEX_DOUBLE(0x1.884e5e5cc7eccp-64,0x1884e5e5cc7eccLL,-64)}, 
        {MAKE_HEX_DOUBLE(0x1.430912f7e8373p-13,0x1430912f7e8373LL,-13), MAKE_HEX_DOUBLE(0x1.4f1b59514a326p-67,0x14f1b59514a326LL,-67)}, 
        {MAKE_HEX_DOUBLE(0x1.ffcbfc5985e71p-17,0x1ffcbfc5985e71LL,-17), MAKE_HEX_DOUBLE(-0x1.db7d6a0953b78p-71,-0x1db7d6a0953b78LL,-71)}, 
        {MAKE_HEX_DOUBLE(0x1.62c150eb16465p-20,0x162c150eb16465LL,-20), MAKE_HEX_DOUBLE(0x1.e0767c2d7abf5p-80,0x1e0767c2d7abf5LL,-80)}, 
        {MAKE_HEX_DOUBLE(0x1.b52502b5e953p-24,0x1b52502b5e953LL,-24), MAKE_HEX_DOUBLE(0x1.6797523f944bcp-78,0x16797523f944bcLL,-78)}
    };
    size_t count = sizeof( c ) / sizeof( c[0] );
    
    // Do polynomial
    double_double r = c[count-1];
    for( j = (int) count-2; j >= 0; j-- )
        r = add_dd( c[j], mul_dd( r, x ) );
    
    // unwind approximation
    r = mul_dd( r, x );     // before: r =(exp2(x)-1)/x;   after: r = exp2(x) - 1
    
    // correct for [-0.5, 0.5] -> [-1/32, 1/32] reduction above
    //  exp2(x) = (r + 1) * correction = r * correction + correction
    r = mul_dd( r, corrections[index+8] );
    r = add_dd( r, corrections[index+8] );
    
// Format result for output:
    
    // Get mantissa
    long double m = ((long double) r.hi + (long double) r.lo );
    
    // Handle a pesky overflow cases when long double = double
    if( i > 512 )
    {
        m *= MAKE_HEX_DOUBLE(0x1.0p512,0x1LL,512);
        i -= 512;
    }
    else if( i < -512 )
    {
        m *= MAKE_HEX_DOUBLE(0x1.0p-512,0x1LL,-512);
        i += 512;
    }
    
    return m * ldexpl( 1.0L, i );
}

long double reference_expm1l(  long double x)
{
#if defined( _MSC_VER )
    //unimplemented
    return x;
#else
    union { double f; cl_ulong u;} u;
    u.f = (double) x;

    if (reference_isnanl(x))
        return x;

    if ( x > 710 )
        return INFINITY;

    long double y = expm1l(x);

    // Range of expm1l is -1.0L to +inf. Negative inf 
    // on a few Linux platforms is clearly the wrong sign.
    if (reference_isinfl(y))
        y = INFINITY;

    return y;
#endif
}

long double reference_fmaxl( long double x, long double y )
{
    if( isnan(y) )
        return x;

    return x >= y ? x : y;
}

long double reference_fminl( long double x, long double y )
{
    if( isnan(y) )
        return x;

    return x <= y ? x : y;
}

long double reference_hypotl( long double x, long double y )
{
  static const double tobig = MAKE_HEX_DOUBLE( 0x1.0p511, 0x1LL, 511 );
  static const double big = MAKE_HEX_DOUBLE( 0x1.0p513, 0x1LL, 513 );
  static const double rbig = MAKE_HEX_DOUBLE( 0x1.0p-513, 0x1LL, -513 );
  static const double tosmall = MAKE_HEX_DOUBLE( 0x1.0p-511, 0x1LL, -511);
  static const double smalll = MAKE_HEX_DOUBLE( 0x1.0p-607, 0x1LL, -607);
  static const double rsmall = MAKE_HEX_DOUBLE( 0x1.0p+607, 0x1LL, 607);

    long double max, min;

    if( isinf(x) || isinf(y) )
        return INFINITY;

    if( isnan(x) || isnan(y) )
        return x + y;

    x = reference_fabsl(x);
    y = reference_fabsl(y);
    
    max = reference_fmaxl( x, y );
    min = reference_fminl( x, y );
    
  if( max > tobig )
    {
        max *= rbig;
        min *= rbig;
        return big * sqrtl( max * max + min * min );
    }

  if( max < tosmall )
    {
        max *= rsmall;
        min *= rsmall;
      return smalll * sqrtl( max * max + min * min );
    }
    return sqrtl( x * x + y * y );
}

//long double reference_log2l( long double x )
//{
//    return log( x ) * 1.44269504088896340735992468100189214L;
//}

long double reference_log2l( long double x )
{
	if( isnan(x) || x < 0.0 || x == -INFINITY)
		return NAN;
	
	if( x == 0.0f) 
		return -INFINITY;
		
	if( x == INFINITY )
		return INFINITY;
	
    double hi, lo;
    __log2_ep( &hi, &lo, x);
    
    return (long double) hi + (long double) lo;
}

long double reference_log1pl(  long double x)
{
#if defined( _MSC_VER )
    //unimplemented
    return x;
#elif defined(__PPC__)
    // log1pl on PPC inadvertantly returns NaN for very large values. Work
    // around this limitation by returning logl for large values.
    return ((x > (long double)(0x1.0p+1022)) ? logl(x) : log1pl(x));
#else
    return log1pl(x);
#endif
}

long double reference_logbl( long double x )
{
    // Since we are just using this to verify double precision, we can
    // use the double precision copysign here
    union { double f; cl_ulong u;} u;
    u.f = (double) x;
    
    cl_int exponent = (cl_uint)(u.u >> 52) & 0x7ff;
    if( exponent == 0x7ff )
        return x * x;
        
    if( exponent == 0 )
    {   // deal with denormals
        u.f =  x * MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64);
        exponent = (cl_int)(u.u >> 52) & 0x7ff;
        if( exponent == 0 )
            return -INFINITY;
        
        return exponent - (1023 + 64);
    }

    return exponent - 1023;
}

long double reference_maxmagl( long double x, long double y )
{
    long double fabsx = fabsl(x);
    long double fabsy = fabsl(y);

    if( fabsx < fabsy )
        return y;

    if( fabsy < fabsx )
        return x;
    
    return reference_fmaxl(x, y);
}

long double reference_minmagl( long double x, long double y )
{
    long double fabsx = fabsl(x);
    long double fabsy = fabsl(y);

    if( fabsx > fabsy )
        return y;

    if( fabsy > fabsx )
        return x;
    
    return reference_fminl(x, y);
}

long double reference_nanl( cl_ulong x )
{
    union{ cl_ulong u; cl_double f; }u;
    u.u = x | 0x7ff8000000000000ULL;
    return (long double) u.f;
}


long double reference_reciprocall( long double x )
{
    return 1.0L / x;
}

long double reference_remainderl( long double x, long double y );
long double reference_remainderl( long double x, long double y )
{
    int i;
    return reference_remquol( x, y, &i );
}

long double reference_lgammal( long double x);
long double reference_lgammal( long double x)
{
    // lgamma is currently not tested
    return reference_lgamma( x );
}

static uint32_t two_over_pi[] = { 0x0, 0x28be60db, 0x24e44152, 0x27f09d5f, 0x11f534dd, 0x3036d8a5, 0x1993c439, 0x107f945, 0x23abdebb, 0x31586dc9, 
0x6e3a424, 0x374b8019, 0x92eea09, 0x3464873f, 0x21deb1cb, 0x4a69cfb, 0x288235f5, 0xbaed121, 0xe99c702, 0x1ad17df9, 
0x13991d6, 0xe60d4ce, 0x1f49c845, 0x3e2ef7e4, 0x283b1ff8, 0x25fff781, 0x1980fef2, 0x3c462d68, 0xa6d1f6d, 0xd9fb3c9, 
0x3cb09b74, 0x3d18fd9a, 0x1e5fea2d, 0x1d49eeb1, 0x3ebe5f17, 0x2cf41ce7, 0x378a5292, 0x3a9afed7, 0x3b11f8d5, 0x3421580c, 
0x3046fc7b, 0x1aeafc33, 0x3bc209af, 0x10d876a7, 0x2391615e, 0x3986c219, 0x199855f1, 0x1281a102, 0xdffd880, 0x135cc9cc, 
0x10606155
};

static uint32_t pi_over_two[] = { 0x1, 0x2487ed51, 0x42d1846, 0x26263314, 0x1701b839, 0x28948127 }; 

typedef union 
	{
		uint64_t u;
		double   d;
	}d_ui64_t;

// radix or base of representation
#define RADIX (30)
#define DIGITS 6

d_ui64_t two_pow_pradix = { (uint64_t) (1023 + RADIX) << 52 };
d_ui64_t two_pow_mradix = { (uint64_t) (1023 - RADIX) << 52 };
d_ui64_t two_pow_two_mradix = { (uint64_t) (1023-2*RADIX) << 52 };

#define tp_pradix two_pow_pradix.d
#define tp_mradix two_pow_mradix.d

// extended fixed point representation of double precision
// floating point number.
// x = sign * [ sum_{i = 0 to 2} ( X[i] * 2^(index - i)*RADIX ) ]
typedef struct
	{
		uint32_t X[3];		// three 32 bit integers are sufficient to represnt double in base_30
		int index;			// exponent bias
		int sign;			// sign of double
	}eprep_t;

static eprep_t double_to_eprep(double x);

static eprep_t double_to_eprep(double x)
{
	eprep_t result;
	
	result.sign = (signbit( x ) == 0) ? 1 : -1;
	x = fabs( x );
	
	int index = 0;
	while( x > tp_pradix ) {
		index++;
		x *= tp_mradix;
	}
	while( x < 1 ) {
		index--;
		x *= tp_pradix;
	}
	
	result.index = index;
	int i = 0;
	result.X[0] = result.X[1] = result.X[2] = 0;
	while( x != 0.0 ) {
		result.X[i] = (uint32_t) x;
		x = (x - (double) result.X[i]) * tp_pradix;
		i++;
	}
	return result;
}

/*
 double eprep_to_double( uint32_t *R, int digits, int index, int sgn )
 {
 d_ui64_t nb, rndcorr;
 uint64_t lowpart, roundbits, t1;
 int expo, expofinal, shift;
 double res;
 
 nb.d = (double) R[0]; 
 
 t1   = R[1];
 lowpart  = (t1 << RADIX) + R[2];
 expo = ((nb.u & 0x7ff0000000000000ULL) >> 52) - 1023; 
 
 expofinal = expo + RADIX*index;
 
 if (expofinal >  1023) {
 d_ui64_t inf = { 0x7ff0000000000000ULL };
 res = inf.d;
 }
 
 else if (expofinal >= -1022){		
 shift = expo + 2*RADIX - 53;
 roundbits = lowpart << (64-shift);
 lowpart = lowpart >> shift;     
 if (lowpart & 0x0000000000000001ULL) {
 if(roundbits == 0) { 
 int i;
 for (i=3; i < digits; i++)
 roundbits = roundbits | R[i];
 }
 if(roundbits == 0) {
 if (lowpart & 0x0000000000000002ULL)
 rndcorr.u = (uint64_t) (expo - 52 + 1023) << 52;
 else
 rndcorr.d = 0.0;
 }
 else 
 rndcorr.u = (uint64_t) (expo - 52 + 1023) << 52;
 }
 else{
 rndcorr.d = 0.0;
 }
 
 lowpart = lowpart >> 1;
 nb.u = nb.u | lowpart;   
 res  = nb.d + rndcorr.d; 
 
 if(index*RADIX + 1023 > 0) {
 nb.u = 0;
 nb.u = (uint64_t) (index*RADIX + 1023) << 52;  		
 res *= nb.d;
 }
 else { 
 nb.u = 0;
 nb.u = (uint64_t) (index*RADIX + 1023 + 2*RADIX) << 52;  		
 res *= two_pow_two_mradix.d;  
 res *= nb.d;                  
 }
 } 
 else { 
 if (expofinal < -1022 - 53 ) {
 res = 0.0;
 }
 else {
 lowpart = lowpart >> (expo + (2*RADIX) - 52);     
 nb.u = nb.u | lowpart; 
 nb.u = (nb.u & 0x000FFFFFFFFFFFFFULL) | 0x0010000000000000ULL;
 nb.u = nb.u >> (-1023 - expofinal);
 if(nb.u & 0x0000000000000001ULL)
 rndcorr.u = 1;
 else
 rndcorr.d = 0.0;
 res  = 0.5*(nb.d + rndcorr.d); 
 }          
 } 
 
 return sgn*res;
 }
 */
static double eprep_to_double( eprep_t epx );

static double eprep_to_double( eprep_t epx )
{
	double res = 0.0;
	
	res += ldexp((double) epx.X[0], (epx.index - 0)*RADIX);
	res += ldexp((double) epx.X[1], (epx.index - 1)*RADIX);
	res += ldexp((double) epx.X[2], (epx.index - 2)*RADIX);
	
	return copysign(res, epx.sign);
}

static int payne_hanek( double *y, int *exception );

static int payne_hanek( double *y, int *exception )
{
	double x = *y;
	
	// exception cases .. no reduction required
	if( isnan( x ) || isinf( x ) || (fabs( x ) <= M_PI_4) ) {
		*exception = 1;
		return 0;
	}
	
	*exception = 0;
	
	// After computation result[0] contains integer part while result[1]....result[DIGITS-1] 
	// contain fractional part. So we are doing computation with (DIGITS-1)*RADIX precision.
	// Default DIGITS=6 and RADIX=30 so default precision is 150 bits. Kahan-McDonald algorithm 
	// shows that a double precision x, closest to pi/2 is 6381956970095103 x 2^797 which can 
	// cause 61 digits of cancellation in computation of f = x*2/pi - floor(x*2/pi) ... thus we need
	// at least 114 bits (61 leading zeros + 53 bits of mentissa of f) of precision to accurately compute
	// f in double precision. Since we are using 150 bits (still an overkill), we should be safe. Extra
	// bits can act as guard bits for correct rounding.
	uint64_t result[DIGITS+2];
	
	// compute extended precision representation of x
	eprep_t epx = double_to_eprep( x );
	int index = epx.index;
	int i, j;
	// extended precision multiplication of 2/pi*x .... we will loose at max two RADIX=30 bit digits in 
	// the worst case
	for(i = 0; i < (DIGITS+2); i++) {
		result[i] = 0;
		result[i] += ((index + i - 0) >= 0) ? ((uint64_t) two_over_pi[index + i - 0] * (uint64_t) epx.X[0]) : 0;
		result[i] += ((index + i - 1) >= 0) ? ((uint64_t) two_over_pi[index + i - 1] * (uint64_t) epx.X[1]) : 0;
		result[i] += ((index + i - 2) >= 0) ? ((uint64_t) two_over_pi[index + i - 2] * (uint64_t) epx.X[2]) : 0;
	}
	
	// Carry propagation.
	uint64_t tmp;
	for(i = DIGITS+2-1; i > 0; i--) {
		tmp = result[i] >> RADIX;
		result[i - 1] += tmp;
		result[i] -= (tmp << RADIX);
	}
	
	// we dont ned to normalize the integer part since only last two bits of this will be used
	// subsequently algorithm which remain unaltered by this normalization.
	// tmp = result[0] >> RADIX;
	// result[0] -= (tmp << RADIX);	
	unsigned int N = (unsigned int) result[0];
	
	// if the result is > pi/4, bring it to (-pi/4, pi/4] range. Note that testing if the final
	// x_star = pi/2*(x*2/pi - k) > pi/4 is equivalent to testing, at this stage, if r[1] (the first fractional
	// digit) is greater than (2^RADIX)/2 and substracting pi/4 from x_star to bring it to mentioned
	// range is equivalent to substracting fractional part at this stage from one and changing the sign.
	int sign = 1;
	if(result[1] > (uint64_t)(1 << (RADIX - 1))) {
		for(i = 1; i < (DIGITS + 2); i++)
			result[i] = (~((unsigned int)result[i]) & 0x3fffffff);
		N += 1;
		sign = -1;
	}
	
	// Again as per Kahan-McDonald algorithim there may be 61 leading zeros in the worst case
	// (when x is multiple of 2/pi very close to an integer) so we need to get rid of these zeros
	// and adjust the index of final result. So in the worst case, precision of comupted result is 
	// 90 bits (150 bits original bits - 60 lost in cancellation). 
	int ind = 1;
	for(i = 1; i < (DIGITS+2); i++) {
		if(result[i] != 0)
			break;
		else
			ind++;
	}
	
	uint64_t r[DIGITS-1];
	for(i = 0; i < (DIGITS-1); i++) {
		r[i] = 0;
		for(j = 0; j <= i; j++) {
			r[i] += (result[ind+i-j] * (uint64_t) pi_over_two[j]);
		}
	}
	for(i = (DIGITS-2); i > 0; i--) {
		tmp = r[i] >> RADIX;
		r[i - 1] += tmp;
		r[i] -= (tmp << RADIX);
	}
	tmp = r[0] >> RADIX;
	r[0] -= (tmp << RADIX);
	
	eprep_t epr;
	epr.sign = epx.sign*sign;
	if(tmp != 0) {
		epr.index = -ind + 1;
		epr.X[0] = (uint32_t) tmp;
		epr.X[1] = (uint32_t) r[0];
		epr.X[2] = (uint32_t) r[1];
	}
	else {
		epr.index = -ind;
		epr.X[0] = (uint32_t) r[0];
		epr.X[1] = (uint32_t) r[1];
		epr.X[2] = (uint32_t) r[2];		
	}
	
	*y = eprep_to_double( epr );
	return epx.sign*N;
}

double reference_cos(double x) 
{
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return cos( x );
	unsigned int c = N & 3;
	switch ( c ) {
		case 0:
			return  cos( x );
		case 1:
			return -sin( x );
		case 2:
			return -cos( x );
		case 3:
			return  sin( x );			
	}
	return 0.0;
}

double reference_sin(double x) 
{
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return sin( x );	
	int c = N & 3;
	switch ( c ) {
		case 0:
			return  sin( x );
		case 1:
			return  cos( x );
		case 2:
			return -sin( x );
		case 3:
			return -cos( x );			
	}
	return 0.0;
}

double reference_sincos(double x, double *y)
{
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception ) {
		*y = cos( x );
		return sin( x );
	}
	int c = N & 3;
	switch ( c ) {
		case 0:
			*y = cos( x );
			return  sin( x );
		case 1:
			*y = -sin( x );
			return  cos( x );
		case 2:
			*y = -cos( x );
			return -sin( x );
		case 3:
			*y = sin( x );
			return -cos( x );			
	}
	return 0.0;	
}

double reference_tan(double x)
{
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return tan( x );	
	int c = N & 3;
	switch ( c ) {
		case 0:
			return  tan( x );
		case 1:
			return -1.0 / tan( x );
		case 2:
			return tan( x );
		case 3:
			return -1.0 / tan( x );			
	}
	return 0.0;	
}

long double reference_cosl(long double xx) 
{
	double x = (double) xx;
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return cosl( x );
	unsigned int c = N & 3;
	switch ( c ) {
		case 0:
			return  cosl( x );
		case 1:
			return -sinl( x );
		case 2:
			return -cosl( x );
		case 3:
			return  sinl( x );			
	}
	return 0.0;
}

long double reference_sinl(long double xx) 
{
	// we use system tanl after reduction which 
	// can flush denorm input to zero so
	//take care of it here.
	if(reference_fabsl(xx) < MAKE_HEX_DOUBLE(0x1.0p-1022,0x1LL,-1022))
		return xx;
	
	double x = (double) xx;
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return sinl( x );	
	int c = N & 3;
	switch ( c ) {
		case 0:
			return  sinl( x );
		case 1:
			return  cosl( x );
		case 2:
			return -sinl( x );
		case 3:
			return -cosl( x );			
	}
	return 0.0;
}

long double reference_sincosl(long double xx, long double *y)
{
	// we use system tanl after reduction which 
	// can flush denorm input to zero so
	//take care of it here.
	if(reference_fabsl(xx) < MAKE_HEX_DOUBLE(0x1.0p-1022,0x1LL,-1022))
	{
		*y = cosl(xx);
		return xx;
	}
	
	double x = (double) xx;
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception ) {
		*y = cosl( x );
		return sinl( x );
	}
	int c = N & 3;
	switch ( c ) {
		case 0:
			*y = cosl( x );
			return  sinl( x );
		case 1:
			*y = -sinl( x );
			return  cosl( x );
		case 2:
			*y = -cosl( x );
			return -sinl( x );
		case 3:
			*y = sinl( x );
			return -cosl( x );			
	}
	return 0.0;	
}

long double reference_tanl(long double xx)
{
	// we use system tanl after reduction which 
	// can flush denorm input to zero so
	//take care of it here.
	if(reference_fabsl(xx) < MAKE_HEX_DOUBLE(0x1.0p-1022,0x1LL,-1022))
		return xx;
	
	double x = (double) xx;
	int exception;
	int N = payne_hanek( &x, &exception );
	if( exception )
		return tanl( x );	
	int c = N & 3;
	switch ( c ) {
		case 0:
			return  tanl( x );
		case 1:
			return -1.0 / tanl( x );
		case 2:
			return tanl( x );
		case 3:
			return -1.0 / tanl( x );			
	}
	return 0.0;	
}

static double __loglTable1[64][3] = {
{MAKE_HEX_DOUBLE(0x1.5390948f40feap+0, 0x15390948f40feaLL, -52), MAKE_HEX_DOUBLE(-0x1.a152f142ap-2, -0x1a152f142aLL, -38), MAKE_HEX_DOUBLE(0x1.f93e27b43bd2cp-40, 0x1f93e27b43bd2cLL, -92)},
{MAKE_HEX_DOUBLE(0x1.5015015015015p+0, 0x15015015015015LL, -52), MAKE_HEX_DOUBLE(-0x1.921800925p-2, -0x1921800925LL, -38), MAKE_HEX_DOUBLE(0x1.162432a1b8df7p-41, 0x1162432a1b8df7LL, -93)},
{MAKE_HEX_DOUBLE(0x1.4cab88725af6ep+0, 0x14cab88725af6eLL, -52), MAKE_HEX_DOUBLE(-0x1.8304d90c18p-2, -0x18304d90c18LL, -42), MAKE_HEX_DOUBLE(0x1.80bb749056fe7p-40, 0x180bb749056fe7LL, -92)},
{MAKE_HEX_DOUBLE(0x1.49539e3b2d066p+0, 0x149539e3b2d066LL, -52), MAKE_HEX_DOUBLE(-0x1.7418acebcp-2, -0x17418acebcLL, -38), MAKE_HEX_DOUBLE(0x1.ceac7f0607711p-43, 0x1ceac7f0607711LL, -95)},
{MAKE_HEX_DOUBLE(0x1.460cbc7f5cf9ap+0, 0x1460cbc7f5cf9aLL, -52), MAKE_HEX_DOUBLE(-0x1.6552b49988p-2, -0x16552b49988LL, -42), MAKE_HEX_DOUBLE(0x1.d8913d0e89fap-42, 0x1d8913d0e89faLL, -90)},
{MAKE_HEX_DOUBLE(0x1.42d6625d51f86p+0, 0x142d6625d51f86LL, -52), MAKE_HEX_DOUBLE(-0x1.56b22e6b58p-2, -0x156b22e6b58LL, -42), MAKE_HEX_DOUBLE(0x1.c7eaf515033a1p-44, 0x1c7eaf515033a1LL, -96)},
{MAKE_HEX_DOUBLE(0x1.3fb013fb013fbp+0, 0x13fb013fb013fbLL, -52), MAKE_HEX_DOUBLE(-0x1.48365e696p-2, -0x148365e696LL, -38), MAKE_HEX_DOUBLE(0x1.434adcde7edc7p-41, 0x1434adcde7edc7LL, -93)},
{MAKE_HEX_DOUBLE(0x1.3c995a47babe7p+0, 0x13c995a47babe7LL, -52), MAKE_HEX_DOUBLE(-0x1.39de8e156p-2, -0x139de8e156LL, -38), MAKE_HEX_DOUBLE(0x1.8246f8e527754p-40, 0x18246f8e527754LL, -92)},
{MAKE_HEX_DOUBLE(0x1.3991c2c187f63p+0, 0x13991c2c187f63LL, -52), MAKE_HEX_DOUBLE(-0x1.2baa0c34cp-2, -0x12baa0c34cLL, -38), MAKE_HEX_DOUBLE(0x1.e1513c28e180dp-42, 0x1e1513c28e180dLL, -94)},
{MAKE_HEX_DOUBLE(0x1.3698df3de0747p+0, 0x13698df3de0747LL, -52), MAKE_HEX_DOUBLE(-0x1.1d982c9d58p-2, -0x11d982c9d58LL, -42), MAKE_HEX_DOUBLE(0x1.63ea3fed4b8a2p-40, 0x163ea3fed4b8a2LL, -92)},
{MAKE_HEX_DOUBLE(0x1.33ae45b57bcb1p+0, 0x133ae45b57bcb1LL, -52), MAKE_HEX_DOUBLE(-0x1.0fa848045p-2, -0x10fa848045LL, -38), MAKE_HEX_DOUBLE(0x1.32ccbacf1779bp-40, 0x132ccbacf1779bLL, -92)},
{MAKE_HEX_DOUBLE(0x1.30d190130d19p+0, 0x130d190130d19LL, -48), MAKE_HEX_DOUBLE(-0x1.01d9bbcfa8p-2, -0x101d9bbcfa8LL, -42), MAKE_HEX_DOUBLE(0x1.e2bfeb2b884aap-42, 0x1e2bfeb2b884aaLL, -94)},
{MAKE_HEX_DOUBLE(0x1.2e025c04b8097p+0, 0x12e025c04b8097LL, -52), MAKE_HEX_DOUBLE(-0x1.e857d3d37p-3, -0x1e857d3d37LL, -39), MAKE_HEX_DOUBLE(0x1.d9309b4d2ea85p-40, 0x1d9309b4d2ea85LL, -92)},
{MAKE_HEX_DOUBLE(0x1.2b404ad012b4p+0, 0x12b404ad012b4LL, -48), MAKE_HEX_DOUBLE(-0x1.cd3c712d4p-3, -0x1cd3c712d4LL, -39), MAKE_HEX_DOUBLE(0x1.ddf360962d7abp-40, 0x1ddf360962d7abLL, -92)},
{MAKE_HEX_DOUBLE(0x1.288b01288b012p+0, 0x1288b01288b012LL, -52), MAKE_HEX_DOUBLE(-0x1.b2602497ep-3, -0x1b2602497eLL, -39), MAKE_HEX_DOUBLE(0x1.597f8a121640fp-40, 0x1597f8a121640fLL, -92)},
{MAKE_HEX_DOUBLE(0x1.25e22708092f1p+0, 0x125e22708092f1LL, -52), MAKE_HEX_DOUBLE(-0x1.97c1cb13dp-3, -0x197c1cb13dLL, -39), MAKE_HEX_DOUBLE(0x1.02807d15580dcp-40, 0x102807d15580dcLL, -92)},
{MAKE_HEX_DOUBLE(0x1.23456789abcdfp+0, 0x123456789abcdfLL, -52), MAKE_HEX_DOUBLE(-0x1.7d60496dp-3, -0x17d60496dLL, -35), MAKE_HEX_DOUBLE(0x1.12ce913d7a827p-41, 0x112ce913d7a827LL, -93)},
{MAKE_HEX_DOUBLE(0x1.20b470c67c0d8p+0, 0x120b470c67c0d8LL, -52), MAKE_HEX_DOUBLE(-0x1.633a8bf44p-3, -0x1633a8bf44LL, -39), MAKE_HEX_DOUBLE(0x1.0648bca9c96bdp-40, 0x10648bca9c96bdLL, -92)},
{MAKE_HEX_DOUBLE(0x1.1e2ef3b3fb874p+0, 0x11e2ef3b3fb874LL, -52), MAKE_HEX_DOUBLE(-0x1.494f863b9p-3, -0x1494f863b9LL, -39), MAKE_HEX_DOUBLE(0x1.066fceb89b0ebp-42, 0x1066fceb89b0ebLL, -94)},
{MAKE_HEX_DOUBLE(0x1.1bb4a4046ed29p+0, 0x11bb4a4046ed29LL, -52), MAKE_HEX_DOUBLE(-0x1.2f9e32d5cp-3, -0x12f9e32d5cLL, -39), MAKE_HEX_DOUBLE(0x1.17b8b6c4f846bp-46, 0x117b8b6c4f846bLL, -98)},
{MAKE_HEX_DOUBLE(0x1.19453808ca29cp+0, 0x119453808ca29cLL, -52), MAKE_HEX_DOUBLE(-0x1.162593187p-3, -0x1162593187LL, -39), MAKE_HEX_DOUBLE(0x1.2c83506452154p-42, 0x12c83506452154LL, -94)},
{MAKE_HEX_DOUBLE(0x1.16e0689427378p+0, 0x116e0689427378LL, -52), MAKE_HEX_DOUBLE(-0x1.f9c95dc1ep-4, -0x1f9c95dc1eLL, -40), MAKE_HEX_DOUBLE(0x1.dd5d2183150f3p-41, 0x1dd5d2183150f3LL, -93)},
{MAKE_HEX_DOUBLE(0x1.1485f0e0acd3bp+0, 0x11485f0e0acd3bLL, -52), MAKE_HEX_DOUBLE(-0x1.c7b528b72p-4, -0x1c7b528b72LL, -40), MAKE_HEX_DOUBLE(0x1.0e43c4f4e619dp-40, 0x10e43c4f4e619dLL, -92)},
{MAKE_HEX_DOUBLE(0x1.12358e75d3033p+0, 0x112358e75d3033LL, -52), MAKE_HEX_DOUBLE(-0x1.960caf9acp-4, -0x1960caf9acLL, -40), MAKE_HEX_DOUBLE(0x1.20fbfd5902a1ep-42, 0x120fbfd5902a1eLL, -94)},
{MAKE_HEX_DOUBLE(0x1.0fef010fef01p+0, 0x10fef010fef01LL, -48), MAKE_HEX_DOUBLE(-0x1.64ce26c08p-4, -0x164ce26c08LL, -40), MAKE_HEX_DOUBLE(0x1.8ebeefb4ac467p-40, 0x18ebeefb4ac467LL, -92)},
{MAKE_HEX_DOUBLE(0x1.0db20a88f4695p+0, 0x10db20a88f4695LL, -52), MAKE_HEX_DOUBLE(-0x1.33f7cde16p-4, -0x133f7cde16LL, -40), MAKE_HEX_DOUBLE(0x1.30b3312da7a7dp-40, 0x130b3312da7a7dLL, -92)},
{MAKE_HEX_DOUBLE(0x1.0b7e6ec259dc7p+0, 0x10b7e6ec259dc7LL, -52), MAKE_HEX_DOUBLE(-0x1.0387efbccp-4, -0x10387efbccLL, -40), MAKE_HEX_DOUBLE(0x1.796f1632949c3p-40, 0x1796f1632949c3LL, -92)},
{MAKE_HEX_DOUBLE(0x1.0953f39010953p+0, 0x10953f39010953LL, -52), MAKE_HEX_DOUBLE(-0x1.a6f9c378p-5, -0x1a6f9c378LL, -37), MAKE_HEX_DOUBLE(0x1.1687e151172ccp-40, 0x11687e151172ccLL, -92)},
{MAKE_HEX_DOUBLE(0x1.073260a47f7c6p+0, 0x1073260a47f7c6LL, -52), MAKE_HEX_DOUBLE(-0x1.47aa07358p-5, -0x147aa07358LL, -41), MAKE_HEX_DOUBLE(0x1.1f87e4a9cc778p-42, 0x11f87e4a9cc778LL, -94)},
{MAKE_HEX_DOUBLE(0x1.05197f7d73404p+0, 0x105197f7d73404LL, -52), MAKE_HEX_DOUBLE(-0x1.d23afc498p-6, -0x1d23afc498LL, -42), MAKE_HEX_DOUBLE(0x1.b183a6b628487p-40, 0x1b183a6b628487LL, -92)},
{MAKE_HEX_DOUBLE(0x1.03091b51f5e1ap+0, 0x103091b51f5e1aLL, -52), MAKE_HEX_DOUBLE(-0x1.16a21e21p-6, -0x116a21e21LL, -38), MAKE_HEX_DOUBLE(0x1.7d75c58973ce5p-40, 0x17d75c58973ce5LL, -92)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1.f44659e4a4271p-1, 0x1f44659e4a4271LL, -53), MAKE_HEX_DOUBLE(0x1.11cd1d51p-5, 0x111cd1d51LL, -37), MAKE_HEX_DOUBLE(0x1.9a0d857e2f4b2p-40, 0x19a0d857e2f4b2LL, -92)},
{MAKE_HEX_DOUBLE(0x1.ecc07b301eccp-1, 0x1ecc07b301eccLL, -49), MAKE_HEX_DOUBLE(0x1.c4dfab908p-5, 0x1c4dfab908LL, -41), MAKE_HEX_DOUBLE(0x1.55b53fce557fdp-40, 0x155b53fce557fdLL, -92)},
{MAKE_HEX_DOUBLE(0x1.e573ac901e573p-1, 0x1e573ac901e573LL, -53), MAKE_HEX_DOUBLE(0x1.3aa2fdd26p-4, 0x13aa2fdd26LL, -40), MAKE_HEX_DOUBLE(0x1.f1cb0c9532089p-40, 0x1f1cb0c9532089LL, -92)},
{MAKE_HEX_DOUBLE(0x1.de5d6e3f8868ap-1, 0x1de5d6e3f8868aLL, -53), MAKE_HEX_DOUBLE(0x1.918a16e46p-4, 0x1918a16e46LL, -40), MAKE_HEX_DOUBLE(0x1.9af0dcd65a6e1p-43, 0x19af0dcd65a6e1LL, -95)},
{MAKE_HEX_DOUBLE(0x1.d77b654b82c33p-1, 0x1d77b654b82c33LL, -53), MAKE_HEX_DOUBLE(0x1.e72ec117ep-4, 0x1e72ec117eLL, -40), MAKE_HEX_DOUBLE(0x1.a5b93c4ebe124p-40, 0x1a5b93c4ebe124LL, -92)},
{MAKE_HEX_DOUBLE(0x1.d0cb58f6ec074p-1, 0x1d0cb58f6ec074LL, -53), MAKE_HEX_DOUBLE(0x1.1dcd19755p-3, 0x11dcd19755LL, -39), MAKE_HEX_DOUBLE(0x1.5be50e71ddc6cp-42, 0x15be50e71ddc6cLL, -94)},
{MAKE_HEX_DOUBLE(0x1.ca4b3055ee191p-1, 0x1ca4b3055ee191LL, -53), MAKE_HEX_DOUBLE(0x1.476a9f983p-3, 0x1476a9f983LL, -39), MAKE_HEX_DOUBLE(0x1.ee9a798719e7fp-40, 0x1ee9a798719e7fLL, -92)},
{MAKE_HEX_DOUBLE(0x1.c3f8f01c3f8fp-1, 0x1c3f8f01c3f8fLL, -49), MAKE_HEX_DOUBLE(0x1.70742d4efp-3, 0x170742d4efLL, -39), MAKE_HEX_DOUBLE(0x1.3ff1352c1219cp-46, 0x13ff1352c1219cLL, -98)},
{MAKE_HEX_DOUBLE(0x1.bdd2b899406f7p-1, 0x1bdd2b899406f7LL, -53), MAKE_HEX_DOUBLE(0x1.98edd077ep-3, 0x198edd077eLL, -39), MAKE_HEX_DOUBLE(0x1.c383cd11362f4p-41, 0x1c383cd11362f4LL, -93)},
{MAKE_HEX_DOUBLE(0x1.b7d6c3dda338bp-1, 0x1b7d6c3dda338bLL, -53), MAKE_HEX_DOUBLE(0x1.c0db6cdd9p-3, 0x1c0db6cdd9LL, -39), MAKE_HEX_DOUBLE(0x1.37bd85b1a824ep-41, 0x137bd85b1a824eLL, -93)},
{MAKE_HEX_DOUBLE(0x1.b2036406c80d9p-1, 0x1b2036406c80d9LL, -53), MAKE_HEX_DOUBLE(0x1.e840be74ep-3, 0x1e840be74eLL, -39), MAKE_HEX_DOUBLE(0x1.a9334d525e1ecp-41, 0x1a9334d525e1ecLL, -93)},
{MAKE_HEX_DOUBLE(0x1.ac5701ac5701ap-1, 0x1ac5701ac5701aLL, -53), MAKE_HEX_DOUBLE(0x1.0790adbbp-2, 0x10790adbbLL, -34), MAKE_HEX_DOUBLE(0x1.8060bfb6a491p-41, 0x18060bfb6a491LL, -89)},
{MAKE_HEX_DOUBLE(0x1.a6d01a6d01a6dp-1, 0x1a6d01a6d01a6dLL, -53), MAKE_HEX_DOUBLE(0x1.1ac05b2918p-2, 0x11ac05b2918LL, -42), MAKE_HEX_DOUBLE(0x1.c1c161471580ap-40, 0x1c1c161471580aLL, -92)},
{MAKE_HEX_DOUBLE(0x1.a16d3f97a4b01p-1, 0x1a16d3f97a4b01LL, -53), MAKE_HEX_DOUBLE(0x1.2db10fc4d8p-2, 0x12db10fc4d8LL, -42), MAKE_HEX_DOUBLE(0x1.ab1aa62214581p-42, 0x1ab1aa62214581LL, -94)},
{MAKE_HEX_DOUBLE(0x1.9c2d14ee4a101p-1, 0x19c2d14ee4a101LL, -53), MAKE_HEX_DOUBLE(0x1.406463b1bp-2, 0x1406463b1bLL, -38), MAKE_HEX_DOUBLE(0x1.12e95dbda6611p-44, 0x112e95dbda6611LL, -96)},
{MAKE_HEX_DOUBLE(0x1.970e4f80cb872p-1, 0x1970e4f80cb872LL, -53), MAKE_HEX_DOUBLE(0x1.52dbdfc4c8p-2, 0x152dbdfc4c8LL, -42), MAKE_HEX_DOUBLE(0x1.6b53fee511afp-42, 0x16b53fee511afLL, -90)},
{MAKE_HEX_DOUBLE(0x1.920fb49d0e228p-1, 0x1920fb49d0e228LL, -53), MAKE_HEX_DOUBLE(0x1.6518fe467p-2, 0x16518fe467LL, -38), MAKE_HEX_DOUBLE(0x1.eea7d7d7d1764p-40, 0x1eea7d7d7d1764LL, -92)},
{MAKE_HEX_DOUBLE(0x1.8d3018d3018d3p-1, 0x18d3018d3018d3LL, -53), MAKE_HEX_DOUBLE(0x1.771d2ba7e8p-2, 0x1771d2ba7e8LL, -42), MAKE_HEX_DOUBLE(0x1.ecefa8d4fab97p-40, 0x1ecefa8d4fab97LL, -92)},
{MAKE_HEX_DOUBLE(0x1.886e5f0abb049p-1, 0x1886e5f0abb049LL, -53), MAKE_HEX_DOUBLE(0x1.88e9c72e08p-2, 0x188e9c72e08LL, -42), MAKE_HEX_DOUBLE(0x1.913ea3d33fd14p-41, 0x1913ea3d33fd14LL, -93)},
{MAKE_HEX_DOUBLE(0x1.83c977ab2beddp-1, 0x183c977ab2beddLL, -53), MAKE_HEX_DOUBLE(0x1.9a802391ep-2, 0x19a802391eLL, -38), MAKE_HEX_DOUBLE(0x1.197e845877c94p-41, 0x1197e845877c94LL, -93)},
{MAKE_HEX_DOUBLE(0x1.7f405fd017f4p-1, 0x17f405fd017f4LL, -49), MAKE_HEX_DOUBLE(0x1.abe18797fp-2, 0x1abe18797fLL, -38), MAKE_HEX_DOUBLE(0x1.f4a52f8e8a81p-42, 0x1f4a52f8e8a81LL, -90)},
{MAKE_HEX_DOUBLE(0x1.7ad2208e0ecc3p-1, 0x17ad2208e0ecc3LL, -53), MAKE_HEX_DOUBLE(0x1.bd0f2e9e78p-2, 0x1bd0f2e9e78LL, -42), MAKE_HEX_DOUBLE(0x1.031f4336644ccp-42, 0x1031f4336644ccLL, -94)},
{MAKE_HEX_DOUBLE(0x1.767dce434a9b1p-1, 0x1767dce434a9b1LL, -53), MAKE_HEX_DOUBLE(0x1.ce0a4923ap-2, 0x1ce0a4923aLL, -38), MAKE_HEX_DOUBLE(0x1.61f33c897020cp-40, 0x161f33c897020cLL, -92)},
{MAKE_HEX_DOUBLE(0x1.724287f46debcp-1, 0x1724287f46debcLL, -53), MAKE_HEX_DOUBLE(0x1.ded3fd442p-2, 0x1ded3fd442LL, -38), MAKE_HEX_DOUBLE(0x1.b2632e830632p-41, 0x1b2632e830632LL, -89)},
{MAKE_HEX_DOUBLE(0x1.6e1f76b4337c6p-1, 0x16e1f76b4337c6LL, -53), MAKE_HEX_DOUBLE(0x1.ef6d673288p-2, 0x1ef6d673288LL, -42), MAKE_HEX_DOUBLE(0x1.888ec245a0bfp-40, 0x1888ec245a0bfLL, -88)},
{MAKE_HEX_DOUBLE(0x1.6a13cd153729p-1, 0x16a13cd153729LL, -49), MAKE_HEX_DOUBLE(0x1.ffd799a838p-2, 0x1ffd799a838LL, -42), MAKE_HEX_DOUBLE(0x1.fe6f3b2f5fc8ep-40, 0x1fe6f3b2f5fc8eLL, -92)},
{MAKE_HEX_DOUBLE(0x1.661ec6a5122f9p-1, 0x1661ec6a5122f9LL, -53), MAKE_HEX_DOUBLE(0x1.0809cf27f4p-1, 0x10809cf27f4LL, -41), MAKE_HEX_DOUBLE(0x1.81eaa9ef284ddp-40, 0x181eaa9ef284ddLL, -92)},
{MAKE_HEX_DOUBLE(0x1.623fa7701623fp-1, 0x1623fa7701623fLL, -53), MAKE_HEX_DOUBLE(0x1.10113b153cp-1, 0x110113b153cLL, -41), MAKE_HEX_DOUBLE(0x1.1d7b07d6b1143p-42, 0x11d7b07d6b1143LL, -94)},
{MAKE_HEX_DOUBLE(0x1.5e75bb8d015e7p-1, 0x15e75bb8d015e7LL, -53), MAKE_HEX_DOUBLE(0x1.18028cf728p-1, 0x118028cf728LL, -41), MAKE_HEX_DOUBLE(0x1.76b100b1f6c6p-41, 0x176b100b1f6c6LL, -89)},
{MAKE_HEX_DOUBLE(0x1.5ac056b015acp-1, 0x15ac056b015acLL, -49), MAKE_HEX_DOUBLE(0x1.1fde3d30e8p-1, 0x11fde3d30e8LL, -41), MAKE_HEX_DOUBLE(0x1.26faeb9870945p-45, 0x126faeb9870945LL, -97)},
{MAKE_HEX_DOUBLE(0x1.571ed3c506b39p-1, 0x1571ed3c506b39LL, -53), MAKE_HEX_DOUBLE(0x1.27a4c0585cp-1, 0x127a4c0585cLL, -41), MAKE_HEX_DOUBLE(0x1.7f2c5344d762bp-42, 0x17f2c5344d762bLL, -94)}
};

static double __loglTable2[64][3] = {
{MAKE_HEX_DOUBLE(0x1.01fbe7f0a1be6p+0, 0x101fbe7f0a1be6LL, -52), MAKE_HEX_DOUBLE(-0x1.6cf6ddd26112ap-7, -0x16cf6ddd26112aLL, -59), MAKE_HEX_DOUBLE(0x1.0725e5755e314p-60, 0x10725e5755e314LL, -112)},
{MAKE_HEX_DOUBLE(0x1.01eba93a97b12p+0, 0x101eba93a97b12LL, -52), MAKE_HEX_DOUBLE(-0x1.6155b1d99f603p-7, -0x16155b1d99f603LL, -59), MAKE_HEX_DOUBLE(0x1.4bcea073117f4p-60, 0x14bcea073117f4LL, -112)},
{MAKE_HEX_DOUBLE(0x1.01db6c9029cd1p+0, 0x101db6c9029cd1LL, -52), MAKE_HEX_DOUBLE(-0x1.55b54153137ffp-7, -0x155b54153137ffLL, -59), MAKE_HEX_DOUBLE(0x1.21e8faccad0ecp-61, 0x121e8faccad0ecLL, -113)},
{MAKE_HEX_DOUBLE(0x1.01cb31f0f534cp+0, 0x101cb31f0f534cLL, -52), MAKE_HEX_DOUBLE(-0x1.4a158c27245bdp-7, -0x14a158c27245bdLL, -59), MAKE_HEX_DOUBLE(0x1.1a5b7bfbf35d3p-60, 0x11a5b7bfbf35d3LL, -112)},
{MAKE_HEX_DOUBLE(0x1.01baf95c9723cp+0, 0x101baf95c9723cLL, -52), MAKE_HEX_DOUBLE(-0x1.3e76923e3d678p-7, -0x13e76923e3d678LL, -59), MAKE_HEX_DOUBLE(0x1.eee400eb5fe34p-62, 0x1eee400eb5fe34LL, -114)},
{MAKE_HEX_DOUBLE(0x1.01aac2d2acee6p+0, 0x101aac2d2acee6LL, -52), MAKE_HEX_DOUBLE(-0x1.32d85380ce776p-7, -0x132d85380ce776LL, -59), MAKE_HEX_DOUBLE(0x1.cbf7a513937bdp-61, 0x1cbf7a513937bdLL, -113)},
{MAKE_HEX_DOUBLE(0x1.019a8e52d401ep+0, 0x1019a8e52d401eLL, -52), MAKE_HEX_DOUBLE(-0x1.273acfd74be72p-7, -0x1273acfd74be72LL, -59), MAKE_HEX_DOUBLE(0x1.5c64599efa5e6p-60, 0x15c64599efa5e6LL, -112)},
{MAKE_HEX_DOUBLE(0x1.018a5bdca9e42p+0, 0x1018a5bdca9e42LL, -52), MAKE_HEX_DOUBLE(-0x1.1b9e072a2e65p-7, -0x11b9e072a2e65LL, -55), MAKE_HEX_DOUBLE(0x1.364180e0a5d37p-60, 0x1364180e0a5d37LL, -112)},
{MAKE_HEX_DOUBLE(0x1.017a2b6fcc33ep+0, 0x1017a2b6fcc33eLL, -52), MAKE_HEX_DOUBLE(-0x1.1001f961f3243p-7, -0x11001f961f3243LL, -59), MAKE_HEX_DOUBLE(0x1.63d795746f216p-60, 0x163d795746f216LL, -112)},
{MAKE_HEX_DOUBLE(0x1.0169fd0bd8a8ap+0, 0x10169fd0bd8a8aLL, -52), MAKE_HEX_DOUBLE(-0x1.0466a6671bca4p-7, -0x10466a6671bca4LL, -59), MAKE_HEX_DOUBLE(0x1.4c99ff1907435p-60, 0x14c99ff1907435LL, -112)},
{MAKE_HEX_DOUBLE(0x1.0159d0b06d129p+0, 0x10159d0b06d129LL, -52), MAKE_HEX_DOUBLE(-0x1.f1981c445cd05p-8, -0x1f1981c445cd05LL, -60), MAKE_HEX_DOUBLE(0x1.4bfff6366b723p-62, 0x14bfff6366b723LL, -114)},
{MAKE_HEX_DOUBLE(0x1.0149a65d275a6p+0, 0x10149a65d275a6LL, -52), MAKE_HEX_DOUBLE(-0x1.da6460f76ab8cp-8, -0x1da6460f76ab8cLL, -60), MAKE_HEX_DOUBLE(0x1.9c5404f47589cp-61, 0x19c5404f47589cLL, -113)},
{MAKE_HEX_DOUBLE(0x1.01397e11a581bp+0, 0x101397e11a581bLL, -52), MAKE_HEX_DOUBLE(-0x1.c3321ab87f4efp-8, -0x1c3321ab87f4efLL, -60), MAKE_HEX_DOUBLE(0x1.c0da537429ceap-61, 0x1c0da537429ceaLL, -113)},
{MAKE_HEX_DOUBLE(0x1.012957cd85a28p+0, 0x1012957cd85a28LL, -52), MAKE_HEX_DOUBLE(-0x1.ac014958c112cp-8, -0x1ac014958c112cLL, -60), MAKE_HEX_DOUBLE(0x1.000c2a1b595e3p-64, 0x1000c2a1b595e3LL, -116)},
{MAKE_HEX_DOUBLE(0x1.0119339065ef7p+0, 0x10119339065ef7LL, -52), MAKE_HEX_DOUBLE(-0x1.94d1eca95f67ap-8, -0x194d1eca95f67aLL, -60), MAKE_HEX_DOUBLE(0x1.d8d20b0564d5p-61, 0x1d8d20b0564d5LL, -109)},
{MAKE_HEX_DOUBLE(0x1.01091159e4b3dp+0, 0x101091159e4b3dLL, -52), MAKE_HEX_DOUBLE(-0x1.7da4047b92b3ep-8, -0x17da4047b92b3eLL, -60), MAKE_HEX_DOUBLE(0x1.6194a5d68cf2p-66, 0x16194a5d68cf2LL, -114)},
{MAKE_HEX_DOUBLE(0x1.00f8f129a0535p+0, 0x100f8f129a0535LL, -52), MAKE_HEX_DOUBLE(-0x1.667790a09bf77p-8, -0x1667790a09bf77LL, -60), MAKE_HEX_DOUBLE(0x1.ca230e0bea645p-61, 0x1ca230e0bea645LL, -113)},
{MAKE_HEX_DOUBLE(0x1.00e8d2ff374a1p+0, 0x100e8d2ff374a1LL, -52), MAKE_HEX_DOUBLE(-0x1.4f4c90e9c4eadp-8, -0x14f4c90e9c4eadLL, -60), MAKE_HEX_DOUBLE(0x1.1de3e7f350c1p-61, 0x11de3e7f350c1LL, -109)},
{MAKE_HEX_DOUBLE(0x1.00d8b6da482cep+0, 0x100d8b6da482ceLL, -52), MAKE_HEX_DOUBLE(-0x1.3823052860649p-8, -0x13823052860649LL, -60), MAKE_HEX_DOUBLE(0x1.5789b4c5891b8p-64, 0x15789b4c5891b8LL, -116)},
{MAKE_HEX_DOUBLE(0x1.00c89cba71a8cp+0, 0x100c89cba71a8cLL, -52), MAKE_HEX_DOUBLE(-0x1.20faed2dc9a9ep-8, -0x120faed2dc9a9eLL, -60), MAKE_HEX_DOUBLE(0x1.9e7c40f9839fdp-62, 0x19e7c40f9839fdLL, -114)},
{MAKE_HEX_DOUBLE(0x1.00b8849f52834p+0, 0x100b8849f52834LL, -52), MAKE_HEX_DOUBLE(-0x1.09d448cb65014p-8, -0x109d448cb65014LL, -60), MAKE_HEX_DOUBLE(0x1.387e3e9b6d02p-62, 0x1387e3e9b6d02LL, -110)},
{MAKE_HEX_DOUBLE(0x1.00a86e88899a4p+0, 0x100a86e88899a4LL, -52), MAKE_HEX_DOUBLE(-0x1.e55e2fa53ebf1p-9, -0x1e55e2fa53ebf1LL, -61), MAKE_HEX_DOUBLE(0x1.cdaa71fddfddfp-62, 0x1cdaa71fddfddfLL, -114)},
{MAKE_HEX_DOUBLE(0x1.00985a75b5e3fp+0, 0x100985a75b5e3fLL, -52), MAKE_HEX_DOUBLE(-0x1.b716b429dce0fp-9, -0x1b716b429dce0fLL, -61), MAKE_HEX_DOUBLE(0x1.2f2af081367bfp-63, 0x12f2af081367bfLL, -115)},
{MAKE_HEX_DOUBLE(0x1.00884866766eep+0, 0x100884866766eeLL, -52), MAKE_HEX_DOUBLE(-0x1.88d21ec7a16d7p-9, -0x188d21ec7a16d7LL, -61), MAKE_HEX_DOUBLE(0x1.fb95c228d6f16p-62, 0x1fb95c228d6f16LL, -114)},
{MAKE_HEX_DOUBLE(0x1.0078385a6a61dp+0, 0x10078385a6a61dLL, -52), MAKE_HEX_DOUBLE(-0x1.5a906f219a9e8p-9, -0x15a906f219a9e8LL, -61), MAKE_HEX_DOUBLE(0x1.18aff10a89f29p-64, 0x118aff10a89f29LL, -116)},
{MAKE_HEX_DOUBLE(0x1.00682a5130fbep+0, 0x100682a5130fbeLL, -52), MAKE_HEX_DOUBLE(-0x1.2c51a4dae87f1p-9, -0x12c51a4dae87f1LL, -61), MAKE_HEX_DOUBLE(0x1.bcc7e33ddde3p-63, 0x1bcc7e33ddde3LL, -111)},
{MAKE_HEX_DOUBLE(0x1.00581e4a69944p+0, 0x100581e4a69944LL, -52), MAKE_HEX_DOUBLE(-0x1.fc2b7f2d782b1p-10, -0x1fc2b7f2d782b1LL, -62), MAKE_HEX_DOUBLE(0x1.fe3ef3300a9fap-64, 0x1fe3ef3300a9faLL, -116)},
{MAKE_HEX_DOUBLE(0x1.00481445b39a8p+0, 0x100481445b39a8LL, -52), MAKE_HEX_DOUBLE(-0x1.9fb97df0b0b83p-10, -0x19fb97df0b0b83LL, -62), MAKE_HEX_DOUBLE(0x1.0d9a601f2f324p-65, 0x10d9a601f2f324LL, -117)},
{MAKE_HEX_DOUBLE(0x1.00380c42ae963p+0, 0x100380c42ae963LL, -52), MAKE_HEX_DOUBLE(-0x1.434d4546227aep-10, -0x1434d4546227aeLL, -62), MAKE_HEX_DOUBLE(0x1.0b9b6a5868f33p-63, 0x10b9b6a5868f33LL, -115)},
{MAKE_HEX_DOUBLE(0x1.00280640fa271p+0, 0x100280640fa271LL, -52), MAKE_HEX_DOUBLE(-0x1.cdcda8e930c19p-11, -0x1cdcda8e930c19LL, -63), MAKE_HEX_DOUBLE(0x1.3d424ab39f789p-64, 0x13d424ab39f789LL, -116)},
{MAKE_HEX_DOUBLE(0x1.0018024036051p+0, 0x10018024036051LL, -52), MAKE_HEX_DOUBLE(-0x1.150c558601261p-11, -0x1150c558601261LL, -63), MAKE_HEX_DOUBLE(0x1.285bb90327a0fp-64, 0x1285bb90327a0fLL, -116)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1.ffa011fca0a1ep-1, 0x1ffa011fca0a1eLL, -53), MAKE_HEX_DOUBLE(0x1.14e5640c4197bp-10, 0x114e5640c4197bLL, -62), MAKE_HEX_DOUBLE(0x1.95728136ae401p-63, 0x195728136ae401LL, -115)},
{MAKE_HEX_DOUBLE(0x1.ff6031f064e07p-1, 0x1ff6031f064e07LL, -53), MAKE_HEX_DOUBLE(0x1.cd61806bf532dp-10, 0x1cd61806bf532dLL, -62), MAKE_HEX_DOUBLE(0x1.568a4f35d8538p-63, 0x1568a4f35d8538LL, -115)},
{MAKE_HEX_DOUBLE(0x1.ff2061d532b9cp-1, 0x1ff2061d532b9cLL, -53), MAKE_HEX_DOUBLE(0x1.42e34af550edap-9, 0x142e34af550edaLL, -61), MAKE_HEX_DOUBLE(0x1.8f69cee55fecp-62, 0x18f69cee55fecLL, -110)},
{MAKE_HEX_DOUBLE(0x1.fee0a1a513253p-1, 0x1fee0a1a513253LL, -53), MAKE_HEX_DOUBLE(0x1.9f0a5523902eap-9, 0x19f0a5523902eaLL, -61), MAKE_HEX_DOUBLE(0x1.daec734b11615p-63, 0x1daec734b11615LL, -115)},
{MAKE_HEX_DOUBLE(0x1.fea0f15a12139p-1, 0x1fea0f15a12139LL, -53), MAKE_HEX_DOUBLE(0x1.fb25e19f11b26p-9, 0x1fb25e19f11b26LL, -61), MAKE_HEX_DOUBLE(0x1.8bafca62941dap-62, 0x18bafca62941daLL, -114)},
{MAKE_HEX_DOUBLE(0x1.fe6150ee3e6d4p-1, 0x1fe6150ee3e6d4LL, -53), MAKE_HEX_DOUBLE(0x1.2b9af9a28e282p-8, 0x12b9af9a28e282LL, -60), MAKE_HEX_DOUBLE(0x1.0fd3674e1dc5bp-61, 0x10fd3674e1dc5bLL, -113)},
{MAKE_HEX_DOUBLE(0x1.fe21c05baa109p-1, 0x1fe21c05baa109LL, -53), MAKE_HEX_DOUBLE(0x1.599d4678f24b9p-8, 0x1599d4678f24b9LL, -60), MAKE_HEX_DOUBLE(0x1.dafce1f09937bp-61, 0x1dafce1f09937bLL, -113)},
{MAKE_HEX_DOUBLE(0x1.fde23f9c69cf9p-1, 0x1fde23f9c69cf9LL, -53), MAKE_HEX_DOUBLE(0x1.8799d8c046ebp-8, 0x18799d8c046ebLL, -56), MAKE_HEX_DOUBLE(0x1.ffa0ce0bdd217p-65, 0x1ffa0ce0bdd217LL, -117)},
{MAKE_HEX_DOUBLE(0x1.fda2ceaa956e8p-1, 0x1fda2ceaa956e8LL, -53), MAKE_HEX_DOUBLE(0x1.b590b1e5951eep-8, 0x1b590b1e5951eeLL, -60), MAKE_HEX_DOUBLE(0x1.645a769232446p-62, 0x1645a769232446LL, -114)},
{MAKE_HEX_DOUBLE(0x1.fd636d8047a1fp-1, 0x1fd636d8047a1fLL, -53), MAKE_HEX_DOUBLE(0x1.e381d3555dbcfp-8, 0x1e381d3555dbcfLL, -60), MAKE_HEX_DOUBLE(0x1.882320d368331p-61, 0x1882320d368331LL, -113)},
{MAKE_HEX_DOUBLE(0x1.fd241c179e0ccp-1, 0x1fd241c179e0ccLL, -53), MAKE_HEX_DOUBLE(0x1.08b69f3dccdep-7, 0x108b69f3dccdeLL, -55), MAKE_HEX_DOUBLE(0x1.01ad5065aba9ep-61, 0x101ad5065aba9eLL, -113)},
{MAKE_HEX_DOUBLE(0x1.fce4da6ab93e8p-1, 0x1fce4da6ab93e8LL, -53), MAKE_HEX_DOUBLE(0x1.1fa97a61dd298p-7, 0x11fa97a61dd298LL, -59), MAKE_HEX_DOUBLE(0x1.84cd1f931ae34p-60, 0x184cd1f931ae34LL, -112)},
{MAKE_HEX_DOUBLE(0x1.fca5a873bcb19p-1, 0x1fca5a873bcb19LL, -53), MAKE_HEX_DOUBLE(0x1.36997bcc54a3fp-7, 0x136997bcc54a3fLL, -59), MAKE_HEX_DOUBLE(0x1.1485e97eaee03p-60, 0x11485e97eaee03LL, -112)},
{MAKE_HEX_DOUBLE(0x1.fc66862ccec93p-1, 0x1fc66862ccec93LL, -53), MAKE_HEX_DOUBLE(0x1.4d86a43264a4fp-7, 0x14d86a43264a4fLL, -59), MAKE_HEX_DOUBLE(0x1.c75e63370988bp-61, 0x1c75e63370988bLL, -113)},
{MAKE_HEX_DOUBLE(0x1.fc27739018cfep-1, 0x1fc27739018cfeLL, -53), MAKE_HEX_DOUBLE(0x1.6470f448fb09dp-7, 0x16470f448fb09dLL, -59), MAKE_HEX_DOUBLE(0x1.d7361eeaed0a1p-65, 0x1d7361eeaed0a1LL, -117)},
{MAKE_HEX_DOUBLE(0x1.fbe87097c6f5ap-1, 0x1fbe87097c6f5aLL, -53), MAKE_HEX_DOUBLE(0x1.7b586cc4c2523p-7, 0x17b586cc4c2523LL, -59), MAKE_HEX_DOUBLE(0x1.b3df952cc473cp-61, 0x1b3df952cc473cLL, -113)},
{MAKE_HEX_DOUBLE(0x1.fba97d3e084ddp-1, 0x1fba97d3e084ddLL, -53), MAKE_HEX_DOUBLE(0x1.923d0e5a21e06p-7, 0x1923d0e5a21e06LL, -59), MAKE_HEX_DOUBLE(0x1.cf56c7b64ae5dp-62, 0x1cf56c7b64ae5dLL, -114)},
{MAKE_HEX_DOUBLE(0x1.fb6a997d0ecdcp-1, 0x1fb6a997d0ecdcLL, -53), MAKE_HEX_DOUBLE(0x1.a91ed9bd3df9ap-7, 0x1a91ed9bd3df9aLL, -59), MAKE_HEX_DOUBLE(0x1.b957bdcd89e43p-61, 0x1b957bdcd89e43LL, -113)},
{MAKE_HEX_DOUBLE(0x1.fb2bc54f0f4abp-1, 0x1fb2bc54f0f4abLL, -53), MAKE_HEX_DOUBLE(0x1.bffdcfa1f7fbbp-7, 0x1bffdcfa1f7fbbLL, -59), MAKE_HEX_DOUBLE(0x1.ea8cad9a21771p-62, 0x1ea8cad9a21771LL, -114)},
{MAKE_HEX_DOUBLE(0x1.faed00ae41783p-1, 0x1faed00ae41783LL, -53), MAKE_HEX_DOUBLE(0x1.d6d9f0bbee6f6p-7, 0x1d6d9f0bbee6f6LL, -59), MAKE_HEX_DOUBLE(0x1.5762a9af89c82p-60, 0x15762a9af89c82LL, -112)},
{MAKE_HEX_DOUBLE(0x1.faae4b94dfe64p-1, 0x1faae4b94dfe64LL, -53), MAKE_HEX_DOUBLE(0x1.edb33dbe7d335p-7, 0x1edb33dbe7d335LL, -59), MAKE_HEX_DOUBLE(0x1.21e24fc245697p-62, 0x121e24fc245697LL, -114)},
{MAKE_HEX_DOUBLE(0x1.fa6fa5fd27ff8p-1, 0x1fa6fa5fd27ff8LL, -53), MAKE_HEX_DOUBLE(0x1.0244dbae5ed05p-6, 0x10244dbae5ed05LL, -58), MAKE_HEX_DOUBLE(0x1.12ef51b967102p-60, 0x112ef51b967102LL, -112)},
{MAKE_HEX_DOUBLE(0x1.fa310fe15a078p-1, 0x1fa310fe15a078LL, -53), MAKE_HEX_DOUBLE(0x1.0daeaf24c3529p-6, 0x10daeaf24c3529LL, -58), MAKE_HEX_DOUBLE(0x1.10d3cfca60b45p-59, 0x110d3cfca60b45LL, -111)},
{MAKE_HEX_DOUBLE(0x1.f9f2893bb9192p-1, 0x1f9f2893bb9192LL, -53), MAKE_HEX_DOUBLE(0x1.1917199bb66bcp-6, 0x11917199bb66bcLL, -58), MAKE_HEX_DOUBLE(0x1.6cf6034c32e19p-60, 0x16cf6034c32e19LL, -112)},
{MAKE_HEX_DOUBLE(0x1.f9b412068b247p-1, 0x1f9b412068b247LL, -53), MAKE_HEX_DOUBLE(0x1.247e1b6c615d5p-6, 0x1247e1b6c615d5LL, -58), MAKE_HEX_DOUBLE(0x1.42f0fffa229f7p-61, 0x142f0fffa229f7LL, -113)},
{MAKE_HEX_DOUBLE(0x1.f975aa3c18ed6p-1, 0x1f975aa3c18ed6LL, -53), MAKE_HEX_DOUBLE(0x1.2fe3b4efcc5adp-6, 0x12fe3b4efcc5adLL, -58), MAKE_HEX_DOUBLE(0x1.70106136a8919p-60, 0x170106136a8919LL, -112)},
{MAKE_HEX_DOUBLE(0x1.f93751d6ae09bp-1, 0x1f93751d6ae09bLL, -53), MAKE_HEX_DOUBLE(0x1.3b47e67edea93p-6, 0x13b47e67edea93LL, -58), MAKE_HEX_DOUBLE(0x1.38dd5a4f6959ap-59, 0x138dd5a4f6959aLL, -111)},
{MAKE_HEX_DOUBLE(0x1.f8f908d098df6p-1, 0x1f8f908d098df6LL, -53), MAKE_HEX_DOUBLE(0x1.46aab0725ea6cp-6, 0x146aab0725ea6cLL, -58), MAKE_HEX_DOUBLE(0x1.821fc1e799e01p-60, 0x1821fc1e799e01LL, -112)},
{MAKE_HEX_DOUBLE(0x1.f8bacf242aa2cp-1, 0x1f8bacf242aa2cLL, -53), MAKE_HEX_DOUBLE(0x1.520c1322f1e4ep-6, 0x1520c1322f1e4eLL, -58), MAKE_HEX_DOUBLE(0x1.129dcda3ad563p-60, 0x1129dcda3ad563LL, -112)},
{MAKE_HEX_DOUBLE(0x1.f87ca4cbb755p-1, 0x1f87ca4cbb755LL, -49), MAKE_HEX_DOUBLE(0x1.5d6c0ee91d2abp-6, 0x15d6c0ee91d2abLL, -58), MAKE_HEX_DOUBLE(0x1.c5b190c04606ep-62, 0x1c5b190c04606eLL, -114)},
{MAKE_HEX_DOUBLE(0x1.f83e89c195c25p-1, 0x1f83e89c195c25LL, -53), MAKE_HEX_DOUBLE(0x1.68caa41d448c3p-6, 0x168caa41d448c3LL, -58), MAKE_HEX_DOUBLE(0x1.4723441195ac9p-59, 0x14723441195ac9LL, -111)}
};

static double __loglTable3[8][3] = {
{MAKE_HEX_DOUBLE(0x1.000e00c40ab89p+0, 0x1000e00c40ab89LL, -52), MAKE_HEX_DOUBLE(-0x1.4332be0032168p-12, -0x14332be0032168LL, -64), MAKE_HEX_DOUBLE(0x1.a1003588d217ap-65, 0x1a1003588d217aLL, -117)},
{MAKE_HEX_DOUBLE(0x1.000a006403e82p+0, 0x1000a006403e82LL, -52), MAKE_HEX_DOUBLE(-0x1.cdb2987366fccp-13, -0x1cdb2987366fccLL, -65), MAKE_HEX_DOUBLE(0x1.5c86001294bbcp-67, 0x15c86001294bbcLL, -119)},
{MAKE_HEX_DOUBLE(0x1.0006002400d8p+0, 0x10006002400d8LL, -48), MAKE_HEX_DOUBLE(-0x1.150297c90fa6fp-13, -0x1150297c90fa6fLL, -65), MAKE_HEX_DOUBLE(0x1.01fb4865fae32p-66, 0x101fb4865fae32LL, -118)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1p+0, 0x1LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0), MAKE_HEX_DOUBLE(0x0p+0, 0x0LL, 0)},
{MAKE_HEX_DOUBLE(0x1.ffe8011ff280ap-1, 0x1ffe8011ff280aLL, -53), MAKE_HEX_DOUBLE(0x1.14f8daf5e3d3bp-12, 0x114f8daf5e3d3bLL, -64), MAKE_HEX_DOUBLE(0x1.3c933b4b6b914p-68, 0x13c933b4b6b914LL, -120)},
{MAKE_HEX_DOUBLE(0x1.ffd8031fc184ep-1, 0x1ffd8031fc184eLL, -53), MAKE_HEX_DOUBLE(0x1.cd978c38042bbp-12, 0x1cd978c38042bbLL, -64), MAKE_HEX_DOUBLE(0x1.10f8e642e66fdp-65, 0x110f8e642e66fdLL, -117)},
{MAKE_HEX_DOUBLE(0x1.ffc8061f5492bp-1, 0x1ffc8061f5492bLL, -53), MAKE_HEX_DOUBLE(0x1.43183c878274ep-11, 0x143183c878274eLL, -63), MAKE_HEX_DOUBLE(0x1.5885dd1eb6582p-65, 0x15885dd1eb6582LL, -117)}
};

static void __log2_ep(double *hi, double *lo, double x)
{
	union { uint64_t i; double d; } uu;
	
	int m;
	double f = reference_frexp(x, &m);
	
	// bring f in [0.75, 1.5)
	if( f < 0.75 ) {
		f *= 2.0;
		m -= 1;
	}
	
	// index first table .... brings down to [1-2^-7, 1+2^6)
	uu.d = f;
	int index = (int) (((uu.i + ((uint64_t) 1 << 51)) & 0x000fc00000000000ULL) >> 46);
	double r1 = __loglTable1[index][0];
	double logr1hi = __loglTable1[index][1];
	double logr1lo = __loglTable1[index][2];
	// since log1rhi has 39 bits of precision, we have 14 bit in hand ... since |m| <= 1023
	// which needs 10bits at max, we can directly add m to log1hi without spilling
	logr1hi += m;
	
	// argument reduction needs to be in double-double since reduced argument will form the
	// leading term of polynomial approximation which sets the precision we eventually achieve
	double zhi, zlo;
	MulD(&zhi, &zlo, r1, uu.d);
	
	// second index table .... brings down to [1-2^-12, 1+2^-11)
	uu.d = zhi;
	index = (int) (((uu.i + ((uint64_t) 1 << 46)) & 0x00007e0000000000ULL) >> 41);
	double r2 = __loglTable2[index][0];
	double logr2hi = __loglTable2[index][1];
	double logr2lo = __loglTable2[index][2];
	
	// reduce argument
	MulDD(&zhi, &zlo, zhi, zlo, r2, 0.0);
	
	// third index table .... brings down to [1-2^-14, 1+2^-13)
	// Actually reduction to 2^-11 would have been sufficient to calculate
	// second order term in polynomial in double rather than double-double, I 
	// reduced it a bit more to make sure other systematic arithmetic errors 
	// are guarded against .... also this allow lower order product of leading polynomial
	// term i.e. Ao_hi*z_lo + Ao_lo*z_hi to be done in double rather than double-double ...
	// hence only term that needs to be done in double-double is Ao_hi*z_hi
	uu.d = zhi;
	index = (int) (((uu.i + ((uint64_t) 1 << 41)) & 0x0000038000000000ULL) >> 39);
	double r3 = __loglTable3[index][0];
	double logr3hi = __loglTable3[index][1];
	double logr3lo = __loglTable3[index][2];
	
	// log2(x) = m + log2(r1) + log2(r1) + log2(1 + (zh + zlo)) 
	// calculate sum of first three terms ... note that m has already
	// been added to log2(r1)_hi
	double log2hi, log2lo;
	AddDD(&log2hi, &log2lo, logr1hi, logr1lo, logr2hi, logr2lo);
	AddDD(&log2hi, &log2lo, logr3hi, logr3lo, log2hi, log2lo);
	
	// final argument reduction .... zhi will be in [1-2^-14, 1+2^-13) after this
	MulDD(&zhi, &zlo, zhi, zlo, r3, 0.0);
	// we dont need to do full double-double substract here. substracting 1.0 for higher
	// term is exact
	zhi = zhi - 1.0;
	// normalize
	AddD(&zhi, &zlo, zhi, zlo);
	
	// polynomail fitting to compute log2(1 + z) ... forth order polynomial fit
	// to log2(1 + z)/z gives minimax absolute error of O(2^-76) with z in [-2^-14, 2^-13]
	// log2(1 + z)/z = Ao + A1*z + A2*z^2 + A3*z^3 + A4*z^4
	// => log2(1 + z) = Ao*z + A1*z^2 + A2*z^3 + A3*z^4 + A4*z^5
	// => log2(1 + z) = (Aohi + Aolo)*(zhi + zlo) + z^2*(A1 + A2*z + A3*z^2 + A4*z^3)
	// since we are looking for at least 64 digits of precision and z in [-2^-14, 2^-13], final term 
	// can be done in double .... also Aolo*zhi + Aohi*zlo can be done in double .... 
	// Aohi*zhi needs to be done in double-double
	
	double Aohi = MAKE_HEX_DOUBLE(0x1.71547652b82fep+0, 0x171547652b82feLL, -52);
	double Aolo = MAKE_HEX_DOUBLE(0x1.777c9cbb675cp-56, 0x1777c9cbb675cLL, -104);
	double y;
	y = MAKE_HEX_DOUBLE(0x1.276d2736fade7p-2, 0x1276d2736fade7LL, -54);
	y = MAKE_HEX_DOUBLE(-0x1.7154765782df1p-2, -0x17154765782df1LL, -54) + y*zhi;
	y = MAKE_HEX_DOUBLE(0x1.ec709dc3a0f67p-2, 0x1ec709dc3a0f67LL, -54) + y*zhi;
	y = MAKE_HEX_DOUBLE(-0x1.71547652b82fep-1, -0x171547652b82feLL, -53) + y*zhi;
	double zhisq = zhi*zhi;
	y = y*zhisq;
	y = y + zhi*Aolo;
	y = y + zlo*Aohi;
	
	MulD(&zhi, &zlo, Aohi, zhi);
	AddDD(&zhi, &zlo, zhi, zlo, y, 0.0);
	AddDD(&zhi, &zlo, zhi, zlo, log2hi, log2lo);
	
	*hi = zhi;
	*lo = zlo;
}

long double reference_powl( long double x, long double y )
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
	
//	double x = (double) xx;
//	double y = (double) yy;
	
	static const double neg_epsilon = MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53);
	
	//if x = 1, return x for any y, even NaN
	if( x == 1.0 )
		return x;
	
	//if y == 0, return 1 for any x, even NaN
	if( y == 0.0 )
		return 1.0L;
	
	//get NaNs out of the way
	if( x != x  || y != y )
		return x + y;
	
	//do the work required to sort out edge cases
	double fabsy = reference_fabs( y );
	double fabsx = reference_fabs( x );
	double iy = reference_rint( fabsy );			//we do round to nearest here so that |fy| <= 0.5
	if( iy > fabsy )//convert nearbyint to floor
		iy -= 1.0;
	int isOddInt = 0;
	if( fabsy == iy && !reference_isinf(fabsy) && iy < neg_epsilon )
		isOddInt = 	(int) (iy - 2.0 * rint( 0.5 * iy ));		//might be 0, -1, or 1
	
	///test a few more edge cases
	//deal with x == 0 cases
	if( x == 0.0 )
	{
		if( ! isOddInt )
			x = 0.0;
		
		if( y < 0 )
			x = 1.0/ x;
		
		return x;
	}
	
	//x == +-Inf cases
	if( isinf(fabsx) )
	{
		if( x < 0 )
		{
			if( isOddInt )
			{
				if( y < 0 )
					return -0.0;
				else
					return -INFINITY;
			}
			else
			{
				if( y < 0 )
					return 0.0;
				else
					return INFINITY;
			}
		}
		
		if( y < 0 )
			return 0;
		return INFINITY;
	}
	
	//y = +-inf cases
	if( isinf(fabsy) )
	{
		if( x == -1 )
			return 1;
		
		if( y < 0 )
		{
			if( fabsx < 1 )
				return INFINITY;
			return 0;
		}
		if( fabsx < 1 )
			return 0;
		return INFINITY;
	}
	
	// x < 0 and y non integer case
	if( x < 0 && iy != fabsy )
	{
		//return nan;
		return cl_make_nan();
	}
	
	//speedy resolution of sqrt and reciprocal sqrt
	if( fabsy == 0.5 )
	{
		long double xl = sqrtl( x );
		if( y < 0 )
			xl = 1.0/ xl;
		return xl;
	}
	
	double log2x_hi, log2x_lo;
	
	// extended precision log .... accurate to at least 64-bits + couple of guard bits
	__log2_ep(&log2x_hi, &log2x_lo, fabsx);
	
	double ylog2x_hi, ylog2x_lo;
	
	double y_hi = (double) y;
	double y_lo = (double) ( y - (long double) y_hi);
	
	// compute product of y*log2(x)
	// scale to avoid overflow in double-double multiplication
	if( reference_fabs( y ) > MAKE_HEX_DOUBLE(0x1.0p970, 0x1LL, 970) ) {
		y_hi = reference_ldexp(y_hi, -53);
		y_lo = reference_ldexp(y_lo, -53);
	}
	MulDD(&ylog2x_hi, &ylog2x_lo, log2x_hi, log2x_lo, y_hi, y_lo);
	if( fabs( y ) > MAKE_HEX_DOUBLE(0x1.0p970, 0x1LL, 970) ) {
		ylog2x_hi = reference_ldexp(ylog2x_hi, 53);
		ylog2x_lo = reference_ldexp(ylog2x_lo, 53);
	}

	long double powxy;
	if(isinf(ylog2x_hi) || (reference_fabs(ylog2x_hi) > 2200)) {
		powxy = reference_signbit(ylog2x_hi) ? MAKE_HEX_DOUBLE(0x0p0, 0x0LL, 0) : INFINITY;
    } else {
        // separate integer + fractional part
        long int m = lrint(ylog2x_hi);
        AddDD(&ylog2x_hi, &ylog2x_lo, ylog2x_hi, ylog2x_lo, -m, 0.0);
        
        // revert to long double arithemtic
        long double ylog2x = (long double) ylog2x_hi + (long double) ylog2x_lo;
        long double tmp = reference_exp2l( ylog2x );
        powxy = reference_scalblnl(tmp, m); 
    }
	
	// if y is odd integer and x is negative, reverse sign
	if( isOddInt & reference_signbit(x))
		powxy = -powxy;
	return powxy;
}

double reference_nextafter(double xx, double yy)
{
	float x = (float) xx;
	float y = (float) yy;
	
	// take care of nans
	if( x != x )
		return x;
	
	if( y != y )
		return y;
		
	if( x == y )
		return y;
	
	int32f_t a, b;
	
	a.f  = x;
	b.f  = y;
	
	if( a.i & 0x80000000 )
		a.i = 0x80000000 - a.i;
	if(b.i & 0x80000000 )
		b.i = 0x80000000 - b.i;
		
	a.i += (a.i < b.i) ? 1 : -1;
	a.i = (a.i < 0) ? (cl_int) 0x80000000 - a.i : a.i;
	
	return a.f;	
}


long double reference_nextafterl(long double xx, long double yy)
{
	double x = (double) xx;
	double y = (double) yy;
	
	// take care of nans
	if( x != x )
		return x;
	
	if( y != y )
		return y;
	
	int64d_t a, b;
	
	a.d  = x;
	b.d  = y;
	
	int64_t tmp = 0x8000000000000000LL;
	
	if( a.l & tmp )
		a.l = tmp - a.l;
	if(b.l & tmp )
		b.l = tmp - b.l;
	
	// edge case. if (x == y) or (x = 0.0f and y = -0.0f) or (x = -0.0f and y = 0.0f)
	// test needs to be done using integer rep because 
	// subnormals may be flushed to zero on some platforms
	if( a.l == b.l )
		return y;
	
	a.l += (a.l < b.l) ? 1 : -1;
	a.l = (a.l < 0) ? tmp - a.l : a.l;
	
	return a.d;	
}

double reference_fdim(double xx, double yy)
{
	float x = (float) xx;
	float y = (float) yy;
	
	if( x != x )
		return x;
		
	if( y != y )
		return y;
		
	float r = ( x > y ) ? (float) reference_subtract( x, y) : 0.0f;
	return r;

}


long double reference_fdiml(long double xx, long double yy)
{
	double x = (double) xx;
	double y = (double) yy;
	
	if( x != x )
		return x;
		
	if( y != y )
		return y;
	
	double r = ( x > y ) ? (double) reference_subtractl(x, y) : 0.0;
	return r;
}

double reference_remquo(double xd, double yd, int *n)
{   
	float xx = (float) xd;
	float yy = (float) yd;
	
    if( isnan(xx) || isnan(yy) ||
        fabsf(xx) == INFINITY  ||
        yy == 0.0 )
    {
        *n = 0;
        return cl_make_nan();
    }

	if( fabsf(yy) == INFINITY || xx == 0.0f ) {
		*n = 0;
		return xd;
	}
	
	if( fabsf(xx) == fabsf(yy) ) {
		*n = (xx == yy) ? 1 : -1;
		return reference_signbit( xx ) ? -0.0 : 0.0;
	}

	int signx = reference_signbit( xx ) ? -1 : 1;
	int signy = reference_signbit( yy ) ? -1 : 1;
	int signn = (signx == signy) ? 1 : -1;
	float x = fabsf(xx);
	float y = fabsf(yy);

	int ex, ey;
	ex = reference_ilogb( x ); 
	ey = reference_ilogb( y );
	float xr = x; 
	float yr = y;
	uint32_t q = 0;
    
	if(ex-ey >= -1) {

        yr = (float) reference_ldexp( y, -ey );
        xr = (float) reference_ldexp( x, -ex );
        
        if(ex-ey >= 0) {
            
		
            int i;
            for(i = ex-ey; i > 0; i--) {
                q <<= 1;
                if(xr >= yr) {
                    xr -= yr;
                    q += 1;
                }
                xr += xr;
            }
            q <<= 1;
            if( xr > yr ) {
                xr -= yr;
                q += 1;
            }
        }
        else //ex-ey = -1
            xr = reference_ldexp(xr, ex-ey); 
    }
            
	if( (yr < 2.0f*xr) || ( (yr == 2.0f*xr) && (q & 0x00000001) ) ) {
		xr -= yr;
		q += 1;
	}
    
    if(ex-ey >= -1)
        xr = reference_ldexp(xr, ey);
	
	int qout = q & 0x0000007f;
	if( signn < 0)
		qout = -qout;
	if( xx < 0.0 )
		xr = -xr;
	
	*n = qout;
	
	return xr;
}

long double reference_remquol(long double xd, long double yd, int *n)
{

	double xx = (double) xd;
	double yy = (double) yd;
	
    if( isnan(xx) || isnan(yy) ||
        fabs(xx) == INFINITY  ||
        yy == 0.0 )
    {
        *n = 0;
        return cl_make_nan();
    }

	if( reference_fabs(yy) == INFINITY || xx == 0.0 ) {
		*n = 0;
		return xd;
	}
	
	if( reference_fabs(xx) == reference_fabs(yy) ) {
		*n = (xx == yy) ? 1 : -1;
		return reference_signbit( xx ) ? -0.0 : 0.0;
	}

	int signx = reference_signbit( xx ) ? -1 : 1;
	int signy = reference_signbit( yy ) ? -1 : 1;
	int signn = (signx == signy) ? 1 : -1;
	double x = reference_fabs(xx);
	double y = reference_fabs(yy);

	int ex, ey;
	ex = reference_ilogbl( x ); 
	ey = reference_ilogbl( y );
	double xr = x; 
	double yr = y;
	uint32_t q = 0;
	
	if(ex-ey >= -1) {
		
		yr = reference_ldexp( y, -ey );
		xr = reference_ldexp( x, -ex );
		int i;
		
        if(ex-ey >= 0) {
        
            for(i = ex-ey; i > 0; i--) {
                q <<= 1;
                if(xr >= yr) {
                    xr -= yr;
                    q += 1;
                }
                xr += xr;
            }
            q <<= 1;
            if( xr > yr ) {
                xr -= yr;
                q += 1;
            }
        }
        else
            xr = reference_ldexp(xr, ex-ey);
	}
	
	if( (yr < 2.0*xr) || ( (yr == 2.0*xr) && (q & 0x00000001) ) ) {
		xr -= yr;
		q += 1;
	}
	
    if(ex-ey >= -1)
        xr = reference_ldexp(xr, ey);
    
	int qout = q & 0x0000007f;
	if( signn < 0)
		qout = -qout;
	if( xx < 0.0 )
		xr = -xr;
	
	*n = qout;	
	return xr;
}

static double reference_scalbn(double x, int n)
{  
    if(reference_isinf(x) || reference_isnan(x) || x == 0.0)
        return x;
    
    int bias = 1023;
    union { double d; cl_long l; } u;
    u.d = (double) x;
    int e = (int)((u.l & 0x7ff0000000000000LL) >> 52);
    if(e == 0)
    {
    	u.l |= ((cl_long)1023 << 52);
    	u.d -= 1.0;
    	e = (int)((u.l & 0x7ff0000000000000LL) >> 52) - 1022;
    } 
    e += n;
    if(e >= 2047 || n >= 2098 ) 
        return reference_copysign(INFINITY, x);
    if(e < -51 || n <-2097 )
        return reference_copysign(0.0, x);
    if(e <= 0)
    {
        bias += (e-1);
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
    union
    {
        long double d;
        struct{ cl_ulong m; cl_ushort sexp;}u;
    }u;
    u.u.m = CL_LONG_MIN;

    if ( reference_isinf(x) )
        return x;

    if( x == 0.0L || n < -2200)
        return reference_copysignl( 0.0L, x );

    if( n > 2200 )
        return reference_copysignl( INFINITY, x );

    if( n < 0 )
    {
        u.u.sexp = 0x3fff - 1022;
        while( n <= -1022 )
        {
            x *= u.d;
            n += 1022;
        }
        u.u.sexp = 0x3fff + n;
        x *= u.d;
        return x;
    }

    if( n > 0 )
    {
        u.u.sexp = 0x3fff + 1023;
        while( n >= 1023 )
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
    if(reference_isinfl(x) || reference_isnanl(x))
        return x;

    int bias = 1023;
    union { double d; cl_long l; } u;
    u.d = (double) x;
    int e = (int)((u.l & 0x7ff0000000000000LL) >> 52);
    if(e == 0)
    {
    	u.l |= ((cl_long)1023 << 52);
    	u.d -= 1.0;
    	e = (int)((u.l & 0x7ff0000000000000LL) >> 52) - 1022;
    } 
    e += n;
    if(e >= 2047) 
        return reference_copysignl(INFINITY, x);
    if(e < -51)
        return reference_copysignl(0.0, x);
    if(e <= 0)
    {
        bias += (e-1);
        e = 1;
    }
    u.l &= 0x800fffffffffffffLL;
    u.l |= ((cl_long)e << 52);
    x = u.d;
    u.l = ((cl_long)bias << 52);
    return x * u.d;
#endif    
    
#else  // PPC
	return scalblnl(x, n);
#endif
}

double reference_exp(double x)
{
	return reference_exp2( x * MAKE_HEX_DOUBLE(0x1.71547652b82fep+0, 0x171547652b82feLL, -52) );
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
  if (x < -708.0L) {
    bias = 40.0;
    scale = expl(-40.0L);
  } else if (x > 708.0L) {
    bias = -40.0L;
    scale = expl(40.0L);
  }
  return expl(x + bias) * scale;
#else
	return expl( x );
#endif
}

double reference_sinh(double x)
{
	return sinh(x);
}

long double reference_sinhl(long double x)
{
	return sinhl(x);
}

double reference_fmod(double x, double y)
{
	if( x == 0.0 && fabs(y) > 0.0 )
		return x;

	if( fabs(x) == INFINITY || y == 0 )
		return cl_make_nan();

	if( fabs(y) == INFINITY )	// we know x is finite from above
		return x;
#if defined(_MSC_VER) && defined(_M_X64)
	return fmod( x, y );
#else
	return fmodf( (float) x, (float) y );
#endif
}

long double reference_fmodl(long double x, long double y)
{
	if( x == 0.0L && fabsl(y) > 0.0L )
		return x;

	if( fabsl(x) == INFINITY || y == 0.0L )
		return cl_make_nan();

	if( fabsl(y) == INFINITY )	// we know x is finite from above
		return x;

	return fmod( (double) x, (double) y );
}

double reference_modf(double x, double *n)
{
	float nr;
	float yr = modff((float) x, &nr);
	*n = nr;
	return yr;
}

long double reference_modfl(long double x, long double *n)
{
	double nr;
	double yr = modf((double) x, &nr);
	*n = nr;
	return yr;
}

long double reference_fractl(long double x, long double *ip )
{
	if(isnan(x)) {
		*ip = cl_make_nan();
		return cl_make_nan();
	}
	
    double i;
    double f = modf((double) x, &i );
    if( f < 0.0 )
    {
        f = 1.0 + f;
        i -= 1.0;
        if( f == 1.0 )
            f = MAKE_HEX_DOUBLE(0x1.fffffffffffffp-1, 0x1fffffffffffffLL, -53); 
    }
    *ip = i;
    return f;
}

long double reference_fabsl(long double x)
{
	return fabsl( x );
}

double reference_log(double x)
{
	if( x == 0.0 )
		return -INFINITY;
		
	if( x < 0.0 )
		return cl_make_nan();
		
	if( isinf(x) )
		return INFINITY;
		
	double log2Hi = MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1, 0x162e42fefa39efLL, -53);
	double logxHi, logxLo;
	__log2_ep(&logxHi, &logxLo, x);
	return logxHi*log2Hi;
}

long double reference_logl(long double x)
{
	if( x == 0.0 )
		return -INFINITY;
		
	if( x < 0.0 )
		return cl_make_nan();
	
	if( isinf(x) )
		return INFINITY;	
		
	double log2Hi = MAKE_HEX_DOUBLE(0x1.62e42fefa39efp-1,  0x162e42fefa39efLL, -53);
	double log2Lo = MAKE_HEX_DOUBLE(0x1.abc9e3b39803fp-56, 0x1abc9e3b39803fLL, -108);
	double logxHi, logxLo;
	__log2_ep(&logxHi, &logxLo, x);
	
	//double rhi, rlo;
	//MulDD(&rhi, &rlo, logxHi, logxLo, log2Hi, log2Lo);
	//return (long double) rhi + (long double) rlo;
	
	long double lg2 = (long double) log2Hi + (long double) log2Lo;
	long double logx = (long double) logxHi + (long double) logxLo;
	return logx*lg2;
}

double reference_pow( double x, double y )
{	
	static const double neg_epsilon = MAKE_HEX_DOUBLE(0x1.0p53, 0x1LL, 53);
	
	//if x = 1, return x for any y, even NaN
	if( x == 1.0 )
		return x;
	
	//if y == 0, return 1 for any x, even NaN
	if( y == 0.0 )
		return 1.0;
	
	//get NaNs out of the way
	if( x != x  || y != y )
		return x + y;
	
	//do the work required to sort out edge cases
	double fabsy = reference_fabs( y );
	double fabsx = reference_fabs( x );
	double iy = reference_rint( fabsy );			//we do round to nearest here so that |fy| <= 0.5
	if( iy > fabsy )//convert nearbyint to floor
		iy -= 1.0;
	int isOddInt = 0;
	if( fabsy == iy && !reference_isinf(fabsy) && iy < neg_epsilon )
		isOddInt = 	(int) (iy - 2.0 * rint( 0.5 * iy ));		//might be 0, -1, or 1
	
	///test a few more edge cases
	//deal with x == 0 cases
	if( x == 0.0 )
	{
		if( ! isOddInt )
			x = 0.0;
		
		if( y < 0 )
			x = 1.0/ x;
		
		return x;
	}
	
	//x == +-Inf cases
	if( isinf(fabsx) )
	{
		if( x < 0 )
		{
			if( isOddInt )
			{
				if( y < 0 )
					return -0.0;
				else
					return -INFINITY;
			}
			else
			{
				if( y < 0 )
					return 0.0;
				else
					return INFINITY;
			}
		}
		
		if( y < 0 )
			return 0;
		return INFINITY;
	}
	
	//y = +-inf cases
	if( isinf(fabsy) )
	{
		if( x == -1 )
			return 1;
		
		if( y < 0 )
		{
			if( fabsx < 1 )
				return INFINITY;
			return 0;
		}
		if( fabsx < 1 )
			return 0;
		return INFINITY;
	}
	
	// x < 0 and y non integer case
	if( x < 0 && iy != fabsy )
	{
		//return nan;
		return cl_make_nan();
	}
	
	//speedy resolution of sqrt and reciprocal sqrt
	if( fabsy == 0.5 )
	{
		long double xl = reference_sqrt( x );
		if( y < 0 )
			xl = 1.0/ xl;
		return xl;
	}
	
	double hi, lo;
	__log2_ep(&hi, &lo, fabsx);
	double prod = y * hi;
	double result = reference_exp2(prod);
	return isOddInt ? reference_copysignd(result, x) : result;
}

double reference_sqrt(double x)
{
	return sqrt(x);
}

double reference_floor(double x)
{
	return floorf((float) x);
}

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
	if(!finite(value)||value==0.0) return value;
	return scalbn(value,exponent);
#else
	return reference_scalbn(value, exponent);
#endif
}

long double reference_ldexpl(long double x, int n)
{
	return ldexpl( x, n);
}

long double reference_coshl(long double x)
{
	return coshl(x);
}

double reference_ceil(double x)
{
	return ceilf((float) x);
}

long double reference_ceill(long double x)
{
    if( x == 0.0 || reference_isinfl(x) || reference_isnanl(x) )
        return x;
    
    long double absx = reference_fabsl(x);
    if( absx >= MAKE_HEX_LONG(0x1.0p52L, 0x1LL, 52) )
        return x;
    
    if( absx < 1.0 )
    {
        if( x < 0.0 )
            return 0.0;
        else
            return 1.0;
    }
    
    long double r = (long double) ((cl_long) x);
    
    if( x > 0.0 && r < x )
        r += 1.0;
    
    return r;
}


long double reference_acosl(long double x)
{
    long double x2 = x * x;
    int i;
    
    //Prepare a head + tail representation of PI in long double.  A good compiler should get rid of all of this work.
    static const cl_ulong pi_bits[2] = { 0x3243F6A8885A308DULL, 0x313198A2E0370734ULL};  // first 126 bits of pi http://www.super-computing.org/pi-hexa_current.html
    long double head, tail, temp;
#if __LDBL_MANT_DIG__ >= 64 
    // long double has 64-bits of precision or greater
    temp = (long double) pi_bits[0] * 0x1.0p64L;
    head = temp + (long double) pi_bits[1];
    temp -= head;           // rounding err rounding pi_bits[1] into head
    tail = (long double) pi_bits[1] + temp; 
    head *= MAKE_HEX_LONG( 0x1.0p-125L, 1, -125 );
    tail *= MAKE_HEX_LONG( 0x1.0p-125L, 1, -125 );
#else
    head = (long double) pi_bits[0];
    tail = (long double) ((cl_long) pi_bits[0] - (cl_long) head );       // residual part of pi_bits[0] after rounding
    tail = tail * MAKE_HEX_LONG( 0x1.0p64L, 1, 64 ) + (long double) pi_bits[1];
    head *= MAKE_HEX_LONG( 0x1.0p-61, 1, -61 );
    tail *= MAKE_HEX_LONG( 0x1.0p-125, 1, -125 );
#endif
    
    // oversize values and NaNs go to NaN
    if( ! (x2 <= 1.0) )
        return sqrtl(1.0L - x2 );
    
    //
    // deal with large |x|:
    //                                                      sqrt( 1 - x**2)
    // acos(|x| > sqrt(0.5)) = 2 * atan( z );       z = -------------------- ;      z in [0, sqrt(0.5)/(1+sqrt(0.5) = .4142135...]
    //                                                          1 + x
    if( x2 > 0.5 )
    {
        // we handle the x < 0 case as pi - acos(|x|)

        long double sign = reference_copysignl( 1.0L, x );
        long double fabsx = reference_fabsl( x );
        head -= head * sign;        // x > 0 ? 0 : pi.hi
        tail -= tail * sign;        // x > 0 ? 0 : pi.low

        // z = sqrt( 1-x**2 ) / (1+x) = sqrt( (1-x)(1+x) / (1+x)**2 ) = sqrt( (1-x)/(1+x) ) 
        long double z2 = (1.0L - fabsx) / (1.0L + fabsx);   // z**2
        long double z = sign * sqrtl(z2);
        
        //                     atan(sqrt(q))
        // Minimax fit p(x) = ---------------- - 1
        //                        sqrt(q)
        //
        // Define q = r*r, and solve for atan(r):
        // 
        //  atan(r) = (p(r) + 1) * r = rp(r) + r
        static long double atan_coeffs[] = { -MAKE_HEX_LONG( 0xb.3f52e0c278293b3p-67L, 0xb3f52e0c278293b3ULL, -127 ), -MAKE_HEX_LONG( 0xa.aaaaaaaaaaa95b8p-5L, 0xaaaaaaaaaaaa95b8ULL, -65 ),
                                              MAKE_HEX_LONG( 0xc.ccccccccc992407p-6L, 0xcccccccccc992407ULL, -66 ), -MAKE_HEX_LONG( 0x9.24924923024398p-6L, 0x9249249230243980ULL, -66 ),
                                              MAKE_HEX_LONG( 0xe.38e38d6f92c98f3p-7L, 0xe38e38d6f92c98f3ULL, -67 ), -MAKE_HEX_LONG( 0xb.a2e89bfb8393ec6p-7L, 0xba2e89bfb8393ec6ULL, -67 ),
                                              MAKE_HEX_LONG( 0x9.d89a9f574d412cbp-7L, 0x9d89a9f574d412cbULL, -67 ), -MAKE_HEX_LONG( 0x8.88580517884c547p-7L, 0x888580517884c547ULL, -67 ),
                                              MAKE_HEX_LONG( 0xf.0ab6756abdad408p-8L, 0xf0ab6756abdad408ULL, -68 ), -MAKE_HEX_LONG( 0xd.56a5b07a2f15b49p-8L, 0xd56a5b07a2f15b49ULL, -68 ),
                                              MAKE_HEX_LONG( 0xb.72ab587e46d80b2p-8L, 0xb72ab587e46d80b2ULL, -68 ), -MAKE_HEX_LONG( 0x8.62ea24bb5b2e636p-8L, 0x862ea24bb5b2e636ULL, -68 ),
                                              MAKE_HEX_LONG( 0xe.d67c16582123937p-10L, 0xed67c16582123937ULL, -70 ) }; // minimax fit over [ 0x1.0p-52, 0.18]   Max error:  0x1.67ea5c184e5d9p-64
        
        // Calculate y = p(r)
        const size_t atan_coeff_count = sizeof( atan_coeffs ) / sizeof( atan_coeffs[0] );
        long double y = atan_coeffs[ atan_coeff_count - 1];
        for( i = (int)atan_coeff_count - 2; i >= 0; i-- )
            y = atan_coeffs[i] + y * z2;
        
        z *= 2.0L;   // fold in 2.0 for 2.0 * atan(z)
        y *= z;      // rp(r)
        
        return head + ((y + tail) + z);
    }

    // do |x| <= sqrt(0.5) here
    //                                                     acos( sqrt(z) ) - PI/2
    //  Piecewise minimax polynomial fits for p(z) = 1 + ------------------------;     
    //                                                            sqrt(z)
    //
    //  Define z = x*x, and solve for acos(x) over x in  x >= 0:
    //
    //      acos( sqrt(z) ) = acos(x) = x*(p(z)-1) + PI/2 = xp(x**2) - x + PI/2
    //
    const long double coeffs[4][14] = {
                                    { -MAKE_HEX_LONG( 0xa.fa7382e1f347974p-10L, 0xafa7382e1f347974ULL, -70 ), -MAKE_HEX_LONG( 0xb.4d5a992de1ac4dap-6L, 0xb4d5a992de1ac4daULL, -66 ),
                                      -MAKE_HEX_LONG( 0xa.c526184bd558c17p-7L, 0xac526184bd558c17ULL, -67 ), -MAKE_HEX_LONG( 0xd.9ed9b0346ec092ap-8L, 0xd9ed9b0346ec092aULL, -68 ),
                                      -MAKE_HEX_LONG( 0x9.dca410c1f04b1fp-8L, 0x9dca410c1f04b1f0ULL, -68 ), -MAKE_HEX_LONG( 0xf.76e411ba9581ee5p-9L, 0xf76e411ba9581ee5ULL, -69 ),
                                      -MAKE_HEX_LONG( 0xc.c71b00479541d8ep-9L, 0xcc71b00479541d8eULL, -69 ), -MAKE_HEX_LONG( 0xa.f527a3f9745c9dep-9L, 0xaf527a3f9745c9deULL, -69 ),
                                      -MAKE_HEX_LONG( 0x9.a93060051f48d14p-9L, 0x9a93060051f48d14ULL, -69 ), -MAKE_HEX_LONG( 0x8.b3d39ad70e06021p-9L, 0x8b3d39ad70e06021ULL, -69 ),
                                      -MAKE_HEX_LONG( 0xf.f2ab95ab84f79cp-10L, 0xff2ab95ab84f79c0ULL, -70 ), -MAKE_HEX_LONG( 0xe.d1af5f5301ccfe4p-10L, 0xed1af5f5301ccfe4ULL, -70 ),
                                      -MAKE_HEX_LONG( 0xe.1b53ba562f0f74ap-10L, 0xe1b53ba562f0f74aULL, -70 ), -MAKE_HEX_LONG( 0xd.6a3851330e15526p-10L, 0xd6a3851330e15526ULL, -70 ) },  // x - 0.0625 in [ -0x1.fffffffffp-5, 0x1.0p-4 ]    Error: 0x1.97839bf07024p-76
                                      
                                    { -MAKE_HEX_LONG( 0x8.c2f1d638e4c1b48p-8L, 0x8c2f1d638e4c1b48ULL, -68 ), -MAKE_HEX_LONG( 0xc.d47ac903c311c2cp-6L, 0xcd47ac903c311c2cULL, -66 ),
                                      -MAKE_HEX_LONG( 0xd.e020b2dabd5606ap-7L, 0xde020b2dabd5606aULL, -67 ), -MAKE_HEX_LONG( 0xa.086fafac220f16bp-7L, 0xa086fafac220f16bULL, -67 ),
                                      -MAKE_HEX_LONG( 0x8.55b5efaf6b86c3ep-7L, 0x855b5efaf6b86c3eULL, -67 ), -MAKE_HEX_LONG( 0xf.05c9774fed2f571p-8L, 0xf05c9774fed2f571ULL, -68 ),
                                      -MAKE_HEX_LONG( 0xe.484a93f7f0fc772p-8L, 0xe484a93f7f0fc772ULL, -68 ), -MAKE_HEX_LONG( 0xe.1a32baef01626e4p-8L, 0xe1a32baef01626e4ULL, -68 ),
                                      -MAKE_HEX_LONG( 0xe.528e525b5c9c73dp-8L, 0xe528e525b5c9c73dULL, -68 ), -MAKE_HEX_LONG( 0xe.ddd5d27ad49b2c8p-8L, 0xeddd5d27ad49b2c8ULL, -68 ),
                                      -MAKE_HEX_LONG( 0xf.b3259e7ae10c6fp-8L, 0xfb3259e7ae10c6f0ULL, -68 ), -MAKE_HEX_LONG( 0x8.68998170d5b19b7p-7L, 0x868998170d5b19b7ULL, -67 ),
                                      -MAKE_HEX_LONG( 0x9.4468907f007727p-7L, 0x94468907f0077270ULL, -67 ), -MAKE_HEX_LONG( 0xa.2ad5e4906a8e7b3p-7L, 0xa2ad5e4906a8e7b3ULL, -67 ) },// x - 0.1875 in [ -0x1.0p-4, 0x1.0p-4 ]    Error: 0x1.647af70073457p-73
                                     
                                    { -MAKE_HEX_LONG( 0xf.a76585ad399e7acp-8L, 0xfa76585ad399e7acULL, -68 ), -MAKE_HEX_LONG( 0xe.d665b7dd504ca7cp-6L, 0xed665b7dd504ca7cULL, -66 ),
                                      -MAKE_HEX_LONG( 0x9.4c7c2402bd4bc33p-6L, 0x94c7c2402bd4bc33ULL, -66 ), -MAKE_HEX_LONG( 0xf.ba76b69074ff71cp-7L, 0xfba76b69074ff71cULL, -67 ),
                                      -MAKE_HEX_LONG( 0xf.58117784bdb6d5fp-7L, 0xf58117784bdb6d5fULL, -67 ), -MAKE_HEX_LONG( 0x8.22ddd8eef53227dp-6L, 0x822ddd8eef53227dULL, -66 ),
                                      -MAKE_HEX_LONG( 0x9.1d1d3b57a63cdb4p-6L, 0x91d1d3b57a63cdb4ULL, -66 ), -MAKE_HEX_LONG( 0xa.9c4bdc40cca848p-6L, 0xa9c4bdc40cca8480ULL, -66 ),
                                      -MAKE_HEX_LONG( 0xc.b673b12794edb24p-6L, 0xcb673b12794edb24ULL, -66 ), -MAKE_HEX_LONG( 0xf.9290a06e31575bfp-6L, 0xf9290a06e31575bfULL, -66 ),
                                      -MAKE_HEX_LONG( 0x9.b4929c16aeb3d1fp-5L, 0x9b4929c16aeb3d1fULL, -65 ), -MAKE_HEX_LONG( 0xc.461e725765a7581p-5L, 0xc461e725765a7581ULL, -65 ),
                                      -MAKE_HEX_LONG( 0x8.0a59654c98d9207p-4L, 0x80a59654c98d9207ULL, -64 ), -MAKE_HEX_LONG( 0xa.6de6cbd96c80562p-4L, 0xa6de6cbd96c80562ULL, -64 ) }, // x - 0.3125 in [ -0x1.0p-4, 0x1.0p-4 ]   Error: 0x1.b0246c304ce1ap-70
                                       
                                    { -MAKE_HEX_LONG( 0xb.dca8b0359f96342p-7L, 0xbdca8b0359f96342ULL, -67 ), -MAKE_HEX_LONG( 0x8.cd2522fcde9823p-5L, 0x8cd2522fcde98230ULL, -65 ),
                                      -MAKE_HEX_LONG( 0xd.2af9397b27ff74dp-6L, 0xd2af9397b27ff74dULL, -66 ), -MAKE_HEX_LONG( 0xd.723f2c2c2409811p-6L, 0xd723f2c2c2409811ULL, -66 ),
                                      -MAKE_HEX_LONG( 0xf.ea8f8481ecc3cd1p-6L, 0xfea8f8481ecc3cd1ULL, -66 ), -MAKE_HEX_LONG( 0xa.43fd8a7a646b0b2p-5L, 0xa43fd8a7a646b0b2ULL, -65 ),
                                      -MAKE_HEX_LONG( 0xe.01b0bf63a4e8d76p-5L, 0xe01b0bf63a4e8d76ULL, -65 ), -MAKE_HEX_LONG( 0x9.f0b7096a2a7b4dp-4L, 0x9f0b7096a2a7b4d0ULL, -64 ),
                                      -MAKE_HEX_LONG( 0xe.872e7c5a627ab4cp-4L, 0xe872e7c5a627ab4cULL, -64 ), -MAKE_HEX_LONG( 0xa.dbd760a1882da48p-3L, 0xadbd760a1882da48ULL, -63 ),
                                      -MAKE_HEX_LONG( 0x8.424e4dea31dd273p-2L, 0x8424e4dea31dd273ULL, -62 ), -MAKE_HEX_LONG( 0xc.c05d7730963e793p-2L, 0xcc05d7730963e793ULL, -62 ),
                                      -MAKE_HEX_LONG( 0xa.523d97197cd124ap-1L, 0xa523d97197cd124aULL, -61 ), -MAKE_HEX_LONG( 0x8.307ba943978aaeep+0L, 0x8307ba943978aaeeULL, -60 ) } // x - 0.4375 in [ -0x1.0p-4, 0x1.0p-4 ]  Error: 0x1.9ecff73da69c9p-66
                                 };
    
    const long double offsets[4] = { 0.0625, 0.1875, 0.3125, 0.4375 };
    const size_t coeff_count = sizeof( coeffs[0] ) / sizeof( coeffs[0][0] );
    
    // reduce the incoming values a bit so that they are in the range [-0x1.0p-4, 0x1.0p-4]
    const long double *c;
    i = x2 * 8.0L;
    c = coeffs[i];
    x2 -= offsets[i];       // exact

    // calcualte p(x2)
    long double y = c[ coeff_count - 1];
    for( i = (int)coeff_count - 2; i >= 0; i-- )
        y = c[i] + y * x2;
    
    // xp(x2)
    y *= x;
    
    // return xp(x2) - x + PI/2
    return head + ((y + tail) - x);
}

double reference_log10(double x)
{
	if( x == 0.0 )
		return -INFINITY;
		
	if( x < 0.0 )
		return cl_make_nan();
		
	if( isinf(x) )
		return INFINITY;
		
	double log2Hi = MAKE_HEX_DOUBLE(0x1.34413509f79fep-2, 0x134413509f79feLL, -54);
	double logxHi, logxLo;
	__log2_ep(&logxHi, &logxLo, x);
	return logxHi*log2Hi;
}

long double reference_log10l(long double x)
{
	if( x == 0.0 )
		return -INFINITY;
		
	if( x < 0.0 )
		return cl_make_nan();
	
	if( isinf(x) )
		return INFINITY;	
		
	double log2Hi = MAKE_HEX_DOUBLE(0x1.34413509f79fep-2,  0x134413509f79feLL, -54);
	double log2Lo = MAKE_HEX_DOUBLE(0x1.e623e2566b02dp-55, 0x1e623e2566b02dLL, -107);
	double logxHi, logxLo;
	__log2_ep(&logxHi, &logxLo, x);
	
	//double rhi, rlo;
	//MulDD(&rhi, &rlo, logxHi, logxLo, log2Hi, log2Lo);
	//return (long double) rhi + (long double) rlo;
	
	long double lg2 = (long double) log2Hi + (long double) log2Lo;
	long double logx = (long double) logxHi + (long double) logxLo;
	return logx*lg2;
}

double reference_acos(double x)
{
	return acos( x );
}

double reference_atan2(double x, double y)
{
#if defined(_WIN32)
	// fix edge cases for Windows
	if (isinf(x) && isinf(y)) {
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
	if (isinf(x) && isinf(y)) {
		long double retval = (y > 0) ? M_PI_4 : 3.f * M_PI_4;
		return (x > 0) ? retval : -retval;
	}
#endif // _WIN32
	return atan2l(x, y);
}

double reference_frexp(double a, int *exp)
{
	if(isnan(a) || isinf(a) || a == 0.0)
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
	
	if((u.l & 0x7ff0000000000000ULL) == 0)
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
	if(isnan(a) || isinf(a) || a == 0.0)
	{
		*exp = 0;
		return a;
	}
	
	if(sizeof(long double) == sizeof(double))
	{
		return reference_frexp(a, exp);
	}
	else
	{
		return frexpl(a, exp);
	}
}


double reference_atan(double x)
{
	return atan( x );
}

long double reference_atanl(long double x)
{
	return atanl( x );
}

long double reference_asinl(long double x)
{
	return asinl( x );
}

double reference_asin(double x)
{
	return asin( x );
}

double reference_fabs(double x)
{
	return fabs( x);
}

double reference_cosh(double x)
{
	return cosh( x );
}

long double reference_sqrtl(long double x)
{
    volatile double dx = x;
	return sqrt( dx );
}

long double reference_tanhl(long double x)
{
	return tanhl( x );
}

long double reference_floorl(long double x)
{
    if( x == 0.0 || reference_isinfl(x) || reference_isnanl(x) )
        return x;
    
    long double absx = reference_fabsl(x);
    if( absx >= MAKE_HEX_LONG(0x1.0p52L, 0x1LL, 52) )
        return x;
    
    if( absx < 1.0 )
    {
        if( x < 0.0 )
            return -1.0;
        else
            return 0.0;
    }
        
    long double r = (long double) ((cl_long) x);
    
    if( x < 0.0 && r > x )
        r -= 1.0;
    
    return r;
}


double reference_tanh(double x)
{
	return tanh( x );
}

long double reference_assignmentl( long double x ){ return x; }

int reference_notl( long double x )
{
    int r = !x;
    return r;
}


