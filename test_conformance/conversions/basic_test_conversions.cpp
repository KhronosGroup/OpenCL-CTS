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
#include "harness/testHarness.h"
#include "harness/compat.h"

#include "basic_test_conversions.h"
#include <limits.h>
#include <string.h>

#include "harness/mt19937.h"

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
/* Rounding modes and saturation for use with qcom 64 bit to float conversion library */
    bool            qcom_sat;
    roundingMode    qcom_rm;
#endif

static inline cl_ulong random64( MTdata d );

#if defined (_WIN32)
    #include <mmintrin.h>
    #include <emmintrin.h>
#else // !_WIN32
#if defined (__SSE__ )
    #include <xmmintrin.h>
#endif
#if defined (__SSE2__ )
    #include <emmintrin.h>
#endif
#endif // _WIN32

const char *gTypeNames[ kTypeCount ] = {
                                            "uchar", "char",
                                            "ushort", "short",
                                            "uint",   "int",
                                            "float", "double",
                                            "ulong", "long"
                                        };

const char *gRoundingModeNames[ kRoundingModeCount ] = {
                                                            "",
                                                            "_rte",
                                                            "_rtp",
                                                            "_rtn",
                                                            "_rtz"
                                                        };

const char *gSaturationNames[ 2 ] = { "", "_sat" };

size_t gTypeSizes[ kTypeCount ] = {
                                    sizeof( cl_uchar ), sizeof( cl_char ),
                                    sizeof( cl_ushort ), sizeof( cl_short ),
                                    sizeof( cl_uint ), sizeof( cl_int ),
                                    sizeof( cl_float ), sizeof( cl_double ),
                                    sizeof( cl_ulong ), sizeof( cl_long ),
                                };

long lrintf_clamped( float f );
long lrintf_clamped( float f )
{
    static const float magic[2] = { MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23), - MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23) };

    if( f >= -(float) LONG_MIN )
        return LONG_MAX;

    if( f <= (float) LONG_MIN )
        return LONG_MIN;

    // Round fractional values to integer in round towards nearest mode
    if( fabsf(f) < MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23 ) )
    {
        volatile float x = f;
        float magicVal = magic[ f < 0 ];

#if defined( __SSE__ ) || defined (_WIN32)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this incorrectly
        __m128 v = _mm_set_ss( x );
        __m128 m = _mm_set_ss( magicVal );
        v = _mm_add_ss( v, m );
        v = _mm_sub_ss( v, m );
        _mm_store_ss( (float*) &x, v );
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }

    return (long) f;
}

long long llrintf_clamped( float f );
long long llrintf_clamped( float f )
{
    static const float magic[2] = { MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23), - MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23) };

    if( f >= -(float) LLONG_MIN )
        return LLONG_MAX;

    if( f <= (float) LLONG_MIN )
        return LLONG_MIN;

    // Round fractional values to integer in round towards nearest mode
    if( fabsf(f) < MAKE_HEX_FLOAT(0x1.0p23f, 0x1L, 23) )
    {
        volatile float x = f;
        float magicVal = magic[ f < 0 ];
#if defined( __SSE__ ) || defined (_WIN32)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this incorrectly
        __m128 v = _mm_set_ss( x );
        __m128 m = _mm_set_ss( magicVal );
        v = _mm_add_ss( v, m );
        v = _mm_sub_ss( v, m );
        _mm_store_ss( (float*) &x, v );
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }

    return (long long) f;
}

long lrint_clamped( double f );
long lrint_clamped( double f )
{
    static const double magic[2] = { MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52), MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52) };

    if( sizeof( long ) > 4 )
    {
        if( f >= -(double) LONG_MIN )
            return LONG_MAX;
    }
    else
    {
        if( f >= LONG_MAX )
            return LONG_MAX;
    }

    if( f <= (double) LONG_MIN )
        return LONG_MIN;

    // Round fractional values to integer in round towards nearest mode
    if( fabs(f) < MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52) )
    {
        volatile double x = f;
        double magicVal = magic[ f < 0 ];
#if defined( __SSE2__ ) || defined (_MSC_VER)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this incorrectly
        __m128d v = _mm_set_sd( x );
        __m128d m = _mm_set_sd( magicVal );
        v = _mm_add_sd( v, m );
        v = _mm_sub_sd( v, m );
        _mm_store_sd( (double*) &x, v );
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }

    return (long) f;
}

long long llrint_clamped( double f );
long long llrint_clamped( double f )
{
    static const double magic[2] = { MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52), MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52) };

    if( f >= -(double) LLONG_MIN )
        return LLONG_MAX;

    if( f <= (double) LLONG_MIN )
        return LLONG_MIN;

    // Round fractional values to integer in round towards nearest mode
    if( fabs(f) < MAKE_HEX_DOUBLE(0x1.0p52, 0x1LL, 52) )
    {
        volatile double x = f;
        double magicVal = magic[ f < 0 ];
#if defined( __SSE2__ ) || defined (_MSC_VER)
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this incorrectly
        __m128d v = _mm_set_sd( x );
        __m128d m = _mm_set_sd( magicVal );
        v = _mm_add_sd( v, m );
        v = _mm_sub_sd( v, m );
        _mm_store_sd( (double*) &x, v );
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }

    return (long long) f;
}


/*
    Names created as:

    #include <stdio.h>

    const char *names[] = { "uchar", "char", "ushort", "short", "uint", "int", "float", "double", "ulong", "long" };

    int main( void )
    {

        int i,j;

        for( i = 0; i < sizeof( names ) / sizeof( names[0] ); i++ )
            for( j = 0; j < sizeof( names ) / sizeof( names[0] ); j++ )
            {
                if( j == i )
                    continue;

                vlog( "void %s2%s( void *, void *);\n", names[i], names[j] );
            }


        return 0;
    }
*/

static float my_fabsf( float x );
static double my_fabs( double x );



static void uchar2char( void *, void *);
static void uchar2ushort( void *, void *);
static void uchar2short( void *, void *);
static void uchar2uint( void *, void *);
static void uchar2int( void *, void *);
static void uchar2float( void *, void *);
static void uchar2double( void *, void *);
static void uchar2ulong( void *, void *);
static void uchar2long( void *, void *);
static void char2uchar( void *, void *);
static void char2ushort( void *, void *);
static void char2short( void *, void *);
static void char2uint( void *, void *);
static void char2int( void *, void *);
static void char2float( void *, void *);
static void char2double( void *, void *);
static void char2ulong( void *, void *);
static void char2long( void *, void *);
static void ushort2uchar( void *, void *);
static void ushort2char( void *, void *);
static void ushort2short( void *, void *);
static void ushort2uint( void *, void *);
static void ushort2int( void *, void *);
static void ushort2float( void *, void *);
static void ushort2double( void *, void *);
static void ushort2ulong( void *, void *);
static void ushort2long( void *, void *);
static void short2uchar( void *, void *);
static void short2char( void *, void *);
static void short2ushort( void *, void *);
static void short2uint( void *, void *);
static void short2int( void *, void *);
static void short2float( void *, void *);
static void short2double( void *, void *);
static void short2ulong( void *, void *);
static void short2long( void *, void *);
static void uint2uchar( void *, void *);
static void uint2char( void *, void *);
static void uint2ushort( void *, void *);
static void uint2short( void *, void *);
static void uint2int( void *, void *);
static void uint2float( void *, void *);
static void uint2double( void *, void *);
static void uint2ulong( void *, void *);
static void uint2long( void *, void *);
static void int2uchar( void *, void *);
static void int2char( void *, void *);
static void int2ushort( void *, void *);
static void int2short( void *, void *);
static void int2uint( void *, void *);
static void int2float( void *, void *);
static void int2double( void *, void *);
static void int2ulong( void *, void *);
static void int2long( void *, void *);
static void float2uchar( void *, void *);
static void float2char( void *, void *);
static void float2ushort( void *, void *);
static void float2short( void *, void *);
static void float2uint( void *, void *);
static void float2int( void *, void *);
static void float2double( void *, void *);
static void float2ulong( void *, void *);
static void float2long( void *, void *);
static void double2uchar( void *, void *);
static void double2char( void *, void *);
static void double2ushort( void *, void *);
static void double2short( void *, void *);
static void double2uint( void *, void *);
static void double2int( void *, void *);
static void double2float( void *, void *);
static void double2ulong( void *, void *);
static void double2long( void *, void *);
static void ulong2uchar( void *, void *);
static void ulong2char( void *, void *);
static void ulong2ushort( void *, void *);
static void ulong2short( void *, void *);
static void ulong2uint( void *, void *);
static void ulong2int( void *, void *);
static void ulong2float( void *, void *);
static void ulong2double( void *, void *);
static void ulong2long( void *, void *);
static void long2uchar( void *, void *);
static void long2char( void *, void *);
static void long2ushort( void *, void *);
static void long2short( void *, void *);
static void long2uint( void *, void *);
static void long2int( void *, void *);
static void long2float( void *, void *);
static void long2double( void *, void *);
static void long2ulong( void *, void *);

/*
    Conversion list created as

    #include <stdio.h>

    const char *names[] = { "uchar", "char", "ushort", "short", "uint", "int", "float", "double", "ulong", "long" };

    int main( void )
    {

        int i,j;

        for( i = 0; i < sizeof( names ) / sizeof( names[0] ); i++ )
        {
            vlog( "{ " );
            for( j = 0; j < sizeof( names ) / sizeof( names[0] ); j++ )
            {
                if( j == i )
                    vlog( "          NULL, " );
                else
                {
                    char s[64];
                    sprintf( s, "%s2%s,", names[j], names[i] );
                    vlog( "%15s ", s );
                }
            }
            vlog( "},\n" );
        }

        return 0;
    }

 */
/*
Convert gConversions[kTypeCount][kTypeCount] = {
{           NULL,     char2uchar,   ushort2uchar,    short2uchar,     uint2uchar,      int2uchar,    float2uchar,   double2uchar,    ulong2uchar,     long2uchar, },
{     uchar2char,           NULL,    ushort2char,     short2char,      uint2char,       int2char,     float2char,    double2char,     ulong2char,      long2char, },
{   uchar2ushort,    char2ushort,           NULL,   short2ushort,    uint2ushort,     int2ushort,   float2ushort,  double2ushort,   ulong2ushort,    long2ushort, },
{    uchar2short,     char2short,   ushort2short,           NULL,     uint2short,      int2short,    float2short,   double2short,    ulong2short,     long2short, },
{     uchar2uint,      char2uint,    ushort2uint,     short2uint,           NULL,       int2uint,     float2uint,    double2uint,     ulong2uint,      long2uint, },
{      uchar2int,       char2int,     ushort2int,      short2int,       uint2int,           NULL,      float2int,     double2int,      ulong2int,       long2int, },
{    uchar2float,     char2float,   ushort2float,    short2float,     uint2float,      int2float,           NULL,   double2float,    ulong2float,     long2float, },
{   uchar2double,    char2double,  ushort2double,   short2double,    uint2double,     int2double,   float2double,           NULL,   ulong2double,    long2double, },
{    uchar2ulong,     char2ulong,   ushort2ulong,    short2ulong,     uint2ulong,      int2ulong,    float2ulong,   double2ulong,           NULL,     long2ulong, },
{     uchar2long,      char2long,    ushort2long,     short2long,      uint2long,       int2long,     float2long,    double2long,     ulong2long,           NULL, } };
*/

static void uchar2char_sat( void *, void *);
static void uchar2ushort_sat( void *, void *);
static void uchar2short_sat( void *, void *);
static void uchar2uint_sat( void *, void *);
static void uchar2int_sat( void *, void *);
static void uchar2float_sat( void *, void *);
static void uchar2double_sat( void *, void *);
static void uchar2ulong_sat( void *, void *);
static void uchar2long_sat( void *, void *);
static void char2uchar_sat( void *, void *);
static void char2ushort_sat( void *, void *);
static void char2short_sat( void *, void *);
static void char2uint_sat( void *, void *);
static void char2int_sat( void *, void *);
static void char2float_sat( void *, void *);
static void char2double_sat( void *, void *);
static void char2ulong_sat( void *, void *);
static void char2long_sat( void *, void *);
static void ushort2uchar_sat( void *, void *);
static void ushort2char_sat( void *, void *);
static void ushort2short_sat( void *, void *);
static void ushort2uint_sat( void *, void *);
static void ushort2int_sat( void *, void *);
static void ushort2float_sat( void *, void *);
static void ushort2double_sat( void *, void *);
static void ushort2ulong_sat( void *, void *);
static void ushort2long_sat( void *, void *);
static void short2uchar_sat( void *, void *);
static void short2char_sat( void *, void *);
static void short2ushort_sat( void *, void *);
static void short2uint_sat( void *, void *);
static void short2int_sat( void *, void *);
static void short2float_sat( void *, void *);
static void short2double_sat( void *, void *);
static void short2ulong_sat( void *, void *);
static void short2long_sat( void *, void *);
static void uint2uchar_sat( void *, void *);
static void uint2char_sat( void *, void *);
static void uint2ushort_sat( void *, void *);
static void uint2short_sat( void *, void *);
static void uint2int_sat( void *, void *);
static void uint2float_sat( void *, void *);
static void uint2double_sat( void *, void *);
static void uint2ulong_sat( void *, void *);
static void uint2long_sat( void *, void *);
static void int2uchar_sat( void *, void *);
static void int2char_sat( void *, void *);
static void int2ushort_sat( void *, void *);
static void int2short_sat( void *, void *);
static void int2uint_sat( void *, void *);
static void int2float_sat( void *, void *);
static void int2double_sat( void *, void *);
static void int2ulong_sat( void *, void *);
static void int2long_sat( void *, void *);
static void float2uchar_sat( void *, void *);
static void float2char_sat( void *, void *);
static void float2ushort_sat( void *, void *);
static void float2short_sat( void *, void *);
static void float2uint_sat( void *, void *);
static void float2int_sat( void *, void *);
static void float2double_sat( void *, void *);
static void float2ulong_sat( void *, void *);
static void float2long_sat( void *, void *);
static void double2uchar_sat( void *, void *);
static void double2char_sat( void *, void *);
static void double2ushort_sat( void *, void *);
static void double2short_sat( void *, void *);
static void double2uint_sat( void *, void *);
static void double2int_sat( void *, void *);
static void double2float_sat( void *, void *);
static void double2ulong_sat( void *, void *);
static void double2long_sat( void *, void *);
static void ulong2uchar_sat( void *, void *);
static void ulong2char_sat( void *, void *);
static void ulong2ushort_sat( void *, void *);
static void ulong2short_sat( void *, void *);
static void ulong2uint_sat( void *, void *);
static void ulong2int_sat( void *, void *);
static void ulong2float_sat( void *, void *);
static void ulong2double_sat( void *, void *);
static void ulong2long_sat( void *, void *);
static void long2uchar_sat( void *, void *);
static void long2char_sat( void *, void *);
static void long2ushort_sat( void *, void *);
static void long2short_sat( void *, void *);
static void long2uint_sat( void *, void *);
static void long2int_sat( void *, void *);
static void long2float_sat( void *, void *);
static void long2double_sat( void *, void *);
static void long2ulong_sat( void *, void *);
/*
    #include <stdio.h>

    const char *names[] = { "uchar", "char", "ushort", "short", "uint", "int", "float", "double", "ulong", "long" };

    int main( void )
    {

        int i,j;

        for( i = 0; i < sizeof( names ) / sizeof( names[0] ); i++ )
        {
            vlog( "{ " );
            for( j = 0; j < sizeof( names ) / sizeof( names[0] ); j++ )
            {
                if( j == i )
                    vlog( "             NULL, " );
                else
                {
                    char s[64];
                    sprintf( s, "%s2%s_sat,", names[j], names[i] );
                    vlog( "%18s ", s );
                }
            }
            vlog( "},\n" );
        }

        return 0;
    }

Convert gSaturatedConversions[kTypeCount][kTypeCount] = {
{              NULL,    char2uchar_sat,  ushort2uchar_sat,   short2uchar_sat,    uint2uchar_sat,     int2uchar_sat,   float2uchar_sat,  double2uchar_sat,   ulong2uchar_sat,    long2uchar_sat, },
{    uchar2char_sat,              NULL,   ushort2char_sat,    short2char_sat,     uint2char_sat,      int2char_sat,    float2char_sat,   double2char_sat,    ulong2char_sat,     long2char_sat, },
{  uchar2ushort_sat,   char2ushort_sat,              NULL,  short2ushort_sat,   uint2ushort_sat,    int2ushort_sat,  float2ushort_sat, double2ushort_sat,  ulong2ushort_sat,   long2ushort_sat, },
{   uchar2short_sat,    char2short_sat,  ushort2short_sat,              NULL,    uint2short_sat,     int2short_sat,   float2short_sat,  double2short_sat,   ulong2short_sat,    long2short_sat, },
{    uchar2uint_sat,     char2uint_sat,   ushort2uint_sat,    short2uint_sat,              NULL,      int2uint_sat,    float2uint_sat,   double2uint_sat,    ulong2uint_sat,     long2uint_sat, },
{     uchar2int_sat,      char2int_sat,    ushort2int_sat,     short2int_sat,      uint2int_sat,              NULL,     float2int_sat,    double2int_sat,     ulong2int_sat,      long2int_sat, },
{   uchar2float_sat,    char2float_sat,  ushort2float_sat,   short2float_sat,    uint2float_sat,     int2float_sat,              NULL,  double2float_sat,   ulong2float_sat,    long2float_sat, },
{  uchar2double_sat,   char2double_sat, ushort2double_sat,  short2double_sat,   uint2double_sat,    int2double_sat,  float2double_sat,              NULL,  ulong2double_sat,   long2double_sat, },
{   uchar2ulong_sat,    char2ulong_sat,  ushort2ulong_sat,   short2ulong_sat,    uint2ulong_sat,     int2ulong_sat,   float2ulong_sat,  double2ulong_sat,              NULL,    long2ulong_sat, },
{    uchar2long_sat,     char2long_sat,   ushort2long_sat,    short2long_sat,     uint2long_sat,      int2long_sat,    float2long_sat,   double2long_sat,    ulong2long_sat,              NULL, }
};
*/

/*
    #include <stdio.h>

    const char *names[] = { "uchar", "char", "ushort", "short", "uint", "int", "float", "double", "ulong", "long" };
    const char *types[] = { "uchar", "char", "ushort", "short", "uint", "int", "float", "double", "ulong", "llong" };

    int main( void )
    {

        int i,j;

        for( i = 0; i < sizeof( names ) / sizeof( names[0] ); i++ )
            for( j = 0; j < sizeof( names ) / sizeof( names[0] ); j++ )
            {
                if( j == i )
                    continue;

                switch( i )
                {
                    case 6: //float
                        if( j == 7 )
                            vlog( "void %s2%s( void *out, void *in){ ((%s*) out)[0] = (%s) ((%s*) in)[0]; }\n", names[i], names[i], names[j], types[j], types[i] );
                        else
                            vlog( "void %s2%s( void *out, void *in){ ((%s*) out)[0] = (%s) my_rintf(((%s*) in)[0]); }\n", names[i], names[i], names[j], types[j], types[i] );
                        break;
                    case 7: //double
                        if( j == 6 )
                            vlog( "void %s2%s( void *out, void *in){ ((%s*) out)[0] = (%s) ((%s*) in)[0]; }\n", names[i], names[i], names[j], types[j], types[i] );
                        else
                            vlog( "void %s2%s( void *out, void *in){ ((%s*) out)[0] = (%s) rint(((%s*) in)[0]); }\n", names[i], names[i], names[j], types[j], types[i] );
                        break;
                    default:
                        vlog( "void %s2%s( void *out, void *in){ ((%s*) out)[0] = (%s)
                        ((%s*) in)[0]; }\n", names[i], names[i], names[j], types[j], types[i] );
                        break;
                }
            }


        return 0;
    }
*/

float my_fabsf( float x )
{
    union{ cl_uint u; float f; }u;
    u.f = x;
    u.u &= 0x7fffffff;
    return u.f;
}

double my_fabs( double x )
{
    union{ cl_ulong u; double f; }u;
    u.f = x;
    u.u &= 0x7fffffffffffffffULL;
    return u.f;
}

static float my_rintf( float f );
static float my_rintf( float f )
{
    static const float magic[2] = { MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23), - MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23) };

    // Round fractional values to integer in round towards nearest mode
    if( fabsf(f) < MAKE_HEX_FLOAT( 0x1.0p23f, 0x1, 23 ) )
    {
        volatile float x = f;
        float magicVal = magic[ f < 0 ];

#if defined( __SSE__ )
        // Defeat x87 based arithmetic, which cant do FTZ, and will round this incorrectly
        __m128 v = _mm_set_ss( x );
        __m128 m = _mm_set_ss( magicVal );
        v = _mm_add_ss( v, m );
        v = _mm_sub_ss( v, m );
        _mm_store_ss( (float*) &x, v );
#else
        x += magicVal;
        x -= magicVal;
#endif
        f = x;
    }

    return f;
}

static void uchar2char( void *out, void *in){ ((char*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2short( void *out, void *in){ ((short*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2uint( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2int( void *out, void *in){ ((int*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2float( void *out, void *in)
{
    cl_uchar l = ((cl_uchar*) in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void uchar2double( void *out, void *in)
{
    cl_uchar l = ((cl_uchar*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void uchar2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_uchar*) in)[0]; }
static void char2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = ((cl_char*) in)[0]; }
static void char2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_char*) in)[0]; }
static void char2short( void *out, void *in){ ((short*) out)[0] = ((cl_char*) in)[0]; }
static void char2uint( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_char*) in)[0]; }
static void char2int( void *out, void *in){ ((int*) out)[0] = ((cl_char*) in)[0]; }
static void char2float( void *out, void *in)
{
    cl_char l = ((cl_char*) in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void char2double( void *out, void *in)
{
    cl_char l = ((cl_char*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void char2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_char*) in)[0]; }
static void char2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_char*) in)[0]; }
static void ushort2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2char( void *out, void *in){ ((char*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2short( void *out, void *in){ ((short*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2uint( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2int( void *out, void *in){ ((int*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2float( void *out, void *in)
{
    cl_ushort l = ((cl_ushort*) in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void ushort2double( void *out, void *in)
{
    cl_ushort l = ((cl_ushort*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void ushort2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_ushort*) in)[0]; }
static void short2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = ((cl_short*) in)[0]; }
static void short2char( void *out, void *in){ ((cl_char*) out)[0] = ((cl_short*) in)[0]; }
static void short2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_short*) in)[0]; }
static void short2uint( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_short*) in)[0]; }
static void short2int( void *out, void *in){ ((cl_int*) out)[0] = ((cl_short*) in)[0]; }
static void short2float( void *out, void *in)
{
    cl_short l = ((cl_short*) in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void short2double( void *out, void *in)
{
    cl_short l = ((cl_short*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void short2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_short*) in)[0]; }
static void short2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_short*) in)[0]; }
static void uint2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2char( void *out, void *in){ ((cl_char*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2short( void *out, void *in){ ((short*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2int( void *out, void *in){ ((cl_int*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2float( void *out, void *in)
{
    // Use volatile to prevent optimization by Clang compiler
    volatile cl_uint l = ((cl_uint *)in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void uint2double( void *out, void *in)
{
    cl_uint l = ((cl_uint*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void uint2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_uint*) in)[0]; }
static void int2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = ((cl_int*) in)[0]; }
static void int2char( void *out, void *in){ ((cl_char*) out)[0] = ((cl_int*) in)[0]; }
static void int2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_int*) in)[0]; }
static void int2short( void *out, void *in){ ((cl_short*) out)[0] = ((cl_int*) in)[0]; }
static void int2uint( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_int*) in)[0]; }
static void int2float( void *out, void *in)
{
    // Use volatile to prevent optimization by Clang compiler
    volatile cl_int l = ((cl_int *)in)[0];
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void int2double( void *out, void *in)
{
    cl_int l = ((cl_int*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
}
static void int2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_int*) in)[0]; }
static void int2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_int*) in)[0]; }
static void float2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2char( void *out, void *in){ ((cl_char*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2short( void *out, void *in){ ((cl_short*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2uint( void *out, void *in){ ((cl_uint*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2int( void *out, void *in){ ((cl_int*) out)[0] = my_rintf(((cl_float*) in)[0]); }
static void float2double( void *out, void *in){ ((cl_double*) out)[0] = ((cl_float*) in)[0]; }
static void float2ulong( void *out, void *in)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    // VS2005 (at least) on x86 uses fistp to store the float as a 64-bit int.
    // However, fistp stores it as a signed int, and some of the test values won't
    // fit into a signed int. (These test values are >= 2^63.) The result on VS2005
    // is that these end up silently (at least by default settings) clamped to
    // the max lowest ulong.
    cl_float x = my_rintf(((cl_float *)in)[0]);
    if (x >= 9223372036854775808.0f) {
        x -= 9223372036854775808.0f;
        ((cl_ulong*) out)[0] = x;
        ((cl_ulong*) out)[0] += 9223372036854775808ULL;
    } else {
        ((cl_ulong*) out)[0] = x;
    }
#else
    ((cl_ulong*) out)[0] = my_rintf(((cl_float*) in)[0]);
#endif
}

static void float2long( void *out, void *in){ ((cl_long*) out)[0] =  llrint_clamped( ((cl_float*) in)[0] ); }
static void double2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = rint(((cl_double*) in)[0]); }
static void double2char( void *out, void *in){ ((cl_char*) out)[0] = rint(((cl_double*) in)[0]); }
static void double2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = rint(((cl_double*) in)[0]); }
static void double2short( void *out, void *in){ ((cl_short*) out)[0] = rint(((cl_double*) in)[0]); }
static void double2uint( void *out, void *in){ ((cl_uint*) out)[0] = (cl_uint) rint(((cl_double*) in)[0]); }
static void double2int( void *out, void *in){ ((cl_int*) out)[0] = (int) rint(((cl_double*) in)[0]); }
static void double2float( void *out, void *in){ ((cl_float*) out)[0] = (float) ((cl_double*) in)[0]; }
static void double2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = (cl_ulong) rint(((cl_double*) in)[0]); }
static void double2long( void *out, void *in){ ((cl_long*) out)[0] = (cl_long) rint(((cl_double*) in)[0]); }
static void ulong2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = (cl_uchar) ((cl_ulong*) in)[0]; }
static void ulong2char( void *out, void *in){ ((cl_char*) out)[0] = (cl_char) ((cl_ulong*) in)[0]; }
static void ulong2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = (cl_ushort) ((cl_ulong*) in)[0]; }
static void ulong2short( void *out, void *in){ ((cl_short*) out)[0] = (cl_short)((cl_ulong*) in)[0]; }
static void ulong2uint( void *out, void *in){ ((cl_uint*) out)[0] = (cl_uint) ((cl_ulong*) in)[0]; }
static void ulong2int( void *out, void *in){ ((cl_int*) out)[0] = (cl_int) ((cl_ulong*) in)[0]; }
static void ulong2float( void *out, void *in)
{
#if defined(_MSC_VER) && defined(_M_X64)
    cl_ulong l = ((cl_ulong*) in)[0];
    float result;
    cl_long sl = ((cl_long)l < 0) ? (cl_long)((l >> 1) | (l & 1)) : (cl_long)l;
    _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), sl));
    ((float*) out)[0] = (l == 0 ? 0.0f : (((cl_long)l < 0) ? result * 2.0f : result));
#else
    cl_ulong l = ((cl_ulong*) in)[0];
#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
    /* ARM VFP doesn't have hardware instruction for converting from 64-bit
     * integer to float types, hence GCC ARM uses the floating-point emulation
     * code despite which -mfloat-abi setting it is. But the emulation code in
     * libgcc.a has only one rounding mode (round to nearest even in this case)
     * and ignores the user rounding mode setting in hardware.
     * As a result setting rounding modes in hardware won't give correct
     * rounding results for type covert from 64-bit integer to float using GCC
     * for ARM compiler so for testing different rounding modes, we need to use
     * alternative reference function. ARM64 does have an instruction, however
     * we cannot guarantee the compiler will use it.  On all ARM architechures
     * use emulation to calculate reference.*/
    ((float*) out)[0] = qcom_u64_2_f32(l, qcom_sat, qcom_rm);
#else
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#endif
#endif
}
static void ulong2double( void *out, void *in)
{
#if defined(_MSC_VER)
    cl_ulong l = ((cl_ulong*) in)[0];
    double result;

    cl_long sl = ((cl_long)l < 0) ? (cl_long)((l >> 1) | (l & 1)) : (cl_long)l;
#if defined(_M_X64)
    _mm_store_sd(&result, _mm_cvtsi64_sd(_mm_setzero_pd(), sl));
#else
    result = sl;
#endif
    ((double*) out)[0] = (l == 0 ? 0.0 : (((cl_long)l < 0) ? result * 2.0 : result));
#else
    // Use volatile to prevent optimization by Clang compiler
    volatile cl_ulong l = ((cl_ulong *)in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#endif
}
static void ulong2long( void *out, void *in){ ((cl_long*) out)[0] = ((cl_ulong*) in)[0]; }
static void long2uchar( void *out, void *in){ ((cl_uchar*) out)[0] = (cl_uchar) ((cl_long*) in)[0]; }
static void long2char( void *out, void *in){ ((cl_char*) out)[0] = (cl_char) ((cl_long*) in)[0]; }
static void long2ushort( void *out, void *in){ ((cl_ushort*) out)[0] = (cl_ushort) ((cl_long*) in)[0]; }
static void long2short( void *out, void *in){ ((cl_short*) out)[0] = (cl_short) ((cl_long*) in)[0]; }
static void long2uint( void *out, void *in){ ((cl_uint*) out)[0] = (cl_uint) ((cl_long*) in)[0]; }
static void long2int( void *out, void *in){ ((cl_int*) out)[0] = (cl_int) ((cl_long*) in)[0]; }
static void long2float( void *out, void *in)
{
#if defined(_MSC_VER) && defined(_M_X64)
    cl_long l = ((cl_long*) in)[0];
    float result;

    _mm_store_ss(&result, _mm_cvtsi64_ss(_mm_setzero_ps(), l));
    ((float*) out)[0] = (l == 0 ? 0.0f : result);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#else
    cl_long l = ((cl_long*) in)[0];
#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
    /* ARM VFP doesn't have hardware instruction for converting from 64-bit
     * integer to float types, hence GCC ARM uses the floating-point emulation
     * code despite which -mfloat-abi setting it is. But the emulation code in
     * libgcc.a has only one rounding mode (round to nearest even in this case)
     * and ignores the user rounding mode setting in hardware.
     * As a result setting rounding modes in hardware won't give correct
     * rounding results for type covert from 64-bit integer to float using GCC
     * for ARM compiler so for testing different rounding modes, we need to use
     * alternative reference function. ARM64 does have an instruction, however
     * we cannot guarantee the compiler will use it.  On all ARM architechures
     * use emulation to calculate reference.*/
    ((float*) out)[0] = (l == 0 ? 0.0f : qcom_s64_2_f32(l, qcom_sat, qcom_rm));
#else
    ((float*) out)[0] = (l == 0 ? 0.0f : (float) l);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#endif
#endif
}
static void long2double( void *out, void *in)
{
#if defined(_MSC_VER) && defined(_M_X64)
    cl_long l = ((cl_long*) in)[0];
    double result;

    _mm_store_sd(&result, _mm_cvtsi64_sd(_mm_setzero_pd(), l));
    ((double*) out)[0] = (l == 0 ? 0.0 : result);        // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#else
    cl_long l = ((cl_long*) in)[0];
    ((double*) out)[0] = (l == 0 ? 0.0 : (double) l);      // Per IEEE-754-2008 5.4.1, 0's always convert to +0.0
#endif
}
static void long2ulong( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_long*) in)[0]; }

#define CLAMP( _lo, _x, _hi )   ( (_x) < (_lo) ? (_lo) : ((_x) > (_hi) ? (_hi) : (_x)))

// Done by hand
static void uchar2char_sat( void *out, void *in){ cl_uchar c = ((cl_uchar*) in)[0]; ((cl_char*) out)[0] = c > 0x7f ? 0x7f : c; }
static void uchar2ushort_sat( void *out, void *in){ ((cl_ushort*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2short_sat( void *out, void *in){ ((cl_short*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2uint_sat( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2int_sat( void *out, void *in){ ((cl_int*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2float_sat( void *out, void *in){ ((cl_float*) out)[0] = my_fabsf( (cl_float) ((cl_uchar*) in)[0]); } // my_fabs workaround for <rdar://problem/5965527>
static void uchar2double_sat( void *out, void *in){ ((cl_double*) out)[0] = my_fabs( (cl_double) ((cl_uchar*) in)[0]); } // my_fabs workaround for <rdar://problem/5965527>
static void uchar2ulong_sat( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_uchar*) in)[0]; }
static void uchar2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_uchar*) in)[0]; }
static void char2uchar_sat( void *out, void *in){ cl_char c = ((cl_char*) in)[0]; ((cl_uchar*) out)[0] = c < 0 ? 0 : c; }
static void char2ushort_sat( void *out, void *in){ cl_char c = ((cl_char*) in)[0]; ((cl_ushort*) out)[0] = c < 0 ? 0 : c; }
static void char2short_sat( void *out, void *in){ ((cl_short*) out)[0] = ((cl_char*) in)[0]; }
static void char2uint_sat( void *out, void *in){ cl_char c = ((cl_char*) in)[0]; ((cl_uint*) out)[0] = c < 0 ? 0 : c; }
static void char2int_sat( void *out, void *in){ ((cl_int*) out)[0] = ((cl_char*) in)[0]; }
static void char2float_sat( void *out, void *in){ ((cl_float*) out)[0] = ((cl_char*) in)[0]; }
static void char2double_sat( void *out, void *in){ ((cl_double*) out)[0] = ((cl_char*) in)[0]; }
static void char2ulong_sat( void *out, void *in){ cl_char c = ((cl_char*) in)[0]; ((cl_ulong*) out)[0] = c < 0 ? 0 : c; }
static void char2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_char*) in)[0]; }
static void ushort2uchar_sat( void *out, void *in){ cl_ushort u = ((cl_ushort*) in)[0]; ((cl_uchar*) out)[0] = u > 0xff ? 0xFF : u; }
static void ushort2char_sat( void *out, void *in){ cl_ushort u = ((cl_ushort*) in)[0]; ((cl_char*) out)[0] = u > 0x7f ? 0x7F : u; }
static void ushort2short_sat( void *out, void *in){ cl_ushort u = ((cl_ushort*) in)[0]; ((cl_short*) out)[0] = u > 0x7fff ? 0x7fFF : u; }
static void ushort2uint_sat( void *out, void *in){ ((cl_uint*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2int_sat( void *out, void *in){ ((cl_int*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2float_sat( void *out, void *in){ ((cl_float*) out)[0] = my_fabsf((cl_float)((cl_ushort*) in)[0]); }     // my_fabs workaround for <rdar://problem/5965527>
static void ushort2double_sat( void *out, void *in){ ((cl_double*) out)[0] = my_fabs( (cl_double) ((cl_ushort*) in)[0]); } // my_fabs workaround for <rdar://problem/5965527>
static void ushort2ulong_sat( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_ushort*) in)[0]; }
static void ushort2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_ushort*) in)[0]; }
static void short2uchar_sat( void *out, void *in){ cl_short s = ((cl_short*) in)[0]; ((cl_uchar*) out)[0] = CLAMP( 0, s, CL_UCHAR_MAX ); }
static void short2char_sat( void *out, void *in){ cl_short s = ((cl_short*) in)[0]; ((cl_char*) out)[0] = CLAMP( CL_CHAR_MIN, s, CL_CHAR_MAX ); }
static void short2ushort_sat( void *out, void *in){ cl_short s = ((cl_short*) in)[0]; ((cl_ushort*) out)[0] = s < 0 ? 0 : s; }
static void short2uint_sat( void *out, void *in){ cl_short s = ((cl_short*) in)[0]; ((cl_uint*) out)[0] = s < 0 ? 0 : s; }
static void short2int_sat( void *out, void *in){ ((cl_int*) out)[0] = ((cl_short*) in)[0]; }
static void short2float_sat( void *out, void *in){ ((cl_float*) out)[0] = ((cl_short*) in)[0]; }
static void short2double_sat( void *out, void *in){ ((cl_double*) out)[0] = ((cl_short*) in)[0]; }
static void short2ulong_sat( void *out, void *in){ cl_short s = ((cl_short*) in)[0]; ((cl_ulong*) out)[0] = s < 0 ? 0 : s; }
static void short2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_short*) in)[0]; }
static void uint2uchar_sat( void *out, void *in){ cl_uint u = ((cl_uint*) in)[0]; ((cl_uchar*) out)[0] = CLAMP( 0, u, CL_UCHAR_MAX); }
static void uint2char_sat( void *out, void *in){  cl_uint u = ((cl_uint*) in)[0]; ((cl_char*) out)[0] = CLAMP( 0, u, CL_CHAR_MAX ); }
static void uint2ushort_sat( void *out, void *in){  cl_uint u = ((cl_uint*) in)[0]; ((cl_ushort*) out)[0] = CLAMP( 0, u, CL_USHRT_MAX); }
static void uint2short_sat( void *out, void *in){  cl_uint u = ((cl_uint*) in)[0]; ((cl_short*) out)[0] = CLAMP( 0, u, CL_SHRT_MAX); }
static void uint2int_sat( void *out, void *in){  cl_uint u = ((cl_uint*) in)[0]; ((cl_int*) out)[0] = CLAMP( 0, u, CL_INT_MAX); }
static void uint2float_sat( void *out, void *in){ ((cl_float*) out)[0] = my_fabsf( (cl_float) ((cl_uint*) in)[0] ); }  // my_fabs workaround for <rdar://problem/5965527>
static void uint2double_sat( void *out, void *in){ ((cl_double*) out)[0] = my_fabs( (cl_double) ((cl_uint*) in)[0]); } // my_fabs workaround for <rdar://problem/5965527>
static void uint2ulong_sat( void *out, void *in){ ((cl_ulong*) out)[0] = ((cl_uint*) in)[0]; }
static void uint2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_uint*) in)[0]; }
static void int2uchar_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_uchar*) out)[0] = CLAMP( 0, i, CL_UCHAR_MAX); }
static void int2char_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_char*) out)[0] = CLAMP( CL_CHAR_MIN, i, CL_CHAR_MAX); }
static void int2ushort_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_ushort*) out)[0] = CLAMP( 0, i, CL_USHRT_MAX); }
static void int2short_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_short*) out)[0] = CLAMP( CL_SHRT_MIN, i, CL_SHRT_MAX); }
static void int2uint_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_uint*) out)[0] = CLAMP( 0, i, CL_INT_MAX); }
static void int2float_sat( void *out, void *in){ ((cl_float*) out)[0] = ((cl_int*) in)[0]; }
static void int2double_sat( void *out, void *in){ ((cl_double*) out)[0] = ((cl_int*) in)[0]; }
static void int2ulong_sat( void *out, void *in){ cl_int i = ((cl_int*) in)[0]; ((cl_ulong*) out)[0] = i < 0 ? 0 : i; }
static void int2long_sat( void *out, void *in){ ((cl_long*) out)[0] = ((cl_int*) in)[0]; }
static void float2uchar_sat( void *out, void *in){ ((cl_uchar*) out)[0] = CLAMP( 0, lrintf_clamped(((cl_float*) in)[0]), CL_UCHAR_MAX ); }
static void float2char_sat( void *out, void *in){ ((cl_char*) out)[0] = CLAMP( CL_CHAR_MIN, lrintf_clamped(((cl_float*) in)[0]), CL_CHAR_MAX); }
static void float2ushort_sat( void *out, void *in){ ((cl_ushort*) out)[0] = CLAMP( 0, lrintf_clamped(((cl_float*) in)[0]), CL_USHRT_MAX ); }
static void float2short_sat( void *out, void *in){ ((cl_short*) out)[0] = CLAMP( CL_SHRT_MIN, lrintf_clamped(((cl_float*) in)[0]), CL_SHRT_MAX ); }
static void float2uint_sat( void *out, void *in){ ((cl_uint*) out)[0] = (cl_uint) CLAMP( 0, llrintf_clamped(((cl_float*) in)[0]), CL_UINT_MAX ); }
static void float2int_sat( void *out, void *in){ ((cl_int*) out)[0] = (cl_int) CLAMP( CL_INT_MIN, lrintf_clamped(((cl_float*) in)[0]), CL_INT_MAX ); }
static void float2double_sat( void *out, void *in){ ((cl_double*) out)[0] = ((cl_float*) in)[0]; }
static void float2ulong_sat( void *out, void *in)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    // VS2005 (at least) on x86 uses fistp to store the float as a 64-bit int.
    // However, fistp stores it as a signed int, and some of the test values won't
    // fit into a signed int. (These test values are >= 2^63.) The result on VS2005
    // is that these end up silently (at least by default settings) clamped to
    // the max lowest ulong.
    cl_float x = my_rintf(((cl_float *)in)[0]);
    if (x >= 18446744073709551616.0f) {         // 2^64
        ((cl_ulong*) out)[0] = 0xFFFFFFFFFFFFFFFFULL;
    } else if (x < 0) {
        ((cl_ulong*) out)[0] = 0;
    } else if (x >= 9223372036854775808.0f) {   // 2^63
        x -= 9223372036854775808.0f;
        ((cl_ulong*) out)[0] = x;
        ((cl_ulong*) out)[0] += 9223372036854775808ULL;
    } else {
        ((cl_ulong*) out)[0] = x;
    }
#else
    float f = my_rintf(((float*) in)[0]); ((cl_ulong*) out)[0] = f >= MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64) ? 0xFFFFFFFFFFFFFFFFULL : f < 0 ? 0 : (cl_ulong) f;
#endif
}
// The final cast used to be (cl_ulong) f, but on Linux (RHEL5 at least)
// if f = -1.0f, then (cl_ulong) f = 0xffffffff, which clearly isn't right.
// Switching it to (cl_long) f seems to fix that.
static void float2long_sat( void *out, void *in){ float f = my_rintf(((float*) in)[0]); ((cl_long*) out)[0] = f >= MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63) ? 0x7FFFFFFFFFFFFFFFULL : f < MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63) ? 0x8000000000000000LL : (cl_long) f; }
static void double2uchar_sat( void *out, void *in){ ((cl_uchar*) out)[0] = CLAMP( 0, lrint_clamped(((cl_double*) in)[0]), CL_UCHAR_MAX ); }
static void double2char_sat( void *out, void *in){ ((cl_char*) out)[0] = CLAMP( CL_CHAR_MIN, lrint_clamped(((cl_double*) in)[0]), CL_CHAR_MAX); }
static void double2ushort_sat( void *out, void *in){ ((cl_ushort*) out)[0] = CLAMP( 0, lrint_clamped(((cl_double*) in)[0]), CL_USHRT_MAX ); }
static void double2short_sat( void *out, void *in){ ((cl_short*) out)[0] = CLAMP( CL_SHRT_MIN, lrint_clamped(((cl_double*) in)[0]), CL_SHRT_MAX ); }
static void double2uint_sat( void *out, void *in){ ((cl_uint*) out)[0] = (cl_uint) CLAMP( 0, llrint_clamped(((cl_double*) in)[0]), CL_UINT_MAX ); }
static void double2int_sat( void *out, void *in){ ((cl_int*) out)[0] = (cl_int) CLAMP( CL_INT_MIN, lrint_clamped(((cl_double*) in)[0]), CL_INT_MAX ); }
static void double2float_sat( void *out, void *in){ ((cl_float*) out)[0] = (cl_float) ((double*) in)[0]; }
static void double2ulong_sat( void *out, void *in){ double f = rint(((double*) in)[0]); ((cl_ulong*) out)[0] = f >= MAKE_HEX_DOUBLE(0x1.0p64, 0x1LL, 64) ? 0xFFFFFFFFFFFFFFFFULL : f < 0 ? 0 : (cl_ulong) f; }
static void double2long_sat( void *out, void *in){ double f = rint(((double*) in)[0]); ((cl_long*) out)[0] = f >= MAKE_HEX_DOUBLE(0x1.0p63, 0x1LL, 63) ? 0x7FFFFFFFFFFFFFFFULL : f < MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63) ? 0x8000000000000000LL : (cl_long) f; }
static void ulong2uchar_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_uchar*) out)[0] = CLAMP( 0, u, CL_UCHAR_MAX ); }
static void ulong2char_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_char*) out)[0] = CLAMP( 0, u, CL_CHAR_MAX ); }
static void ulong2ushort_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_ushort*) out)[0] = CLAMP( 0, u, CL_USHRT_MAX ); }
static void ulong2short_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_short*) out)[0] = CLAMP( 0, u, CL_SHRT_MAX ); }
static void ulong2uint_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_uint*) out)[0] = (cl_uint) CLAMP( 0, u, CL_UINT_MAX ); }
static void ulong2int_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_int*) out)[0] = (cl_int) CLAMP( 0, u, CL_INT_MAX ); }
static void ulong2float_sat( void *out, void *in){ ((float*) out)[0] = my_fabsf((float) ((cl_ulong*) in)[0]); }  // my_fabs workaround for <rdar://problem/5965527>
static void ulong2double_sat( void *out, void *in){ ((double*) out)[0] = my_fabs( ((cl_ulong*) in)[0]); }        // my_fabs workaround for <rdar://problem/5965527>
static void ulong2long_sat( void *out, void *in){ cl_ulong u = ((cl_ulong*) in)[0]; ((cl_long*) out)[0] = CLAMP( 0, u, CL_LONG_MAX ); }
static void long2uchar_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_uchar*) out)[0] = CLAMP( 0, u, CL_UCHAR_MAX ); }
static void long2char_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_char*) out)[0] = CLAMP( CL_CHAR_MIN, u, CL_CHAR_MAX ); }
static void long2ushort_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_ushort*) out)[0] = CLAMP( 0, u, CL_USHRT_MAX ); }
static void long2short_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_short*) out)[0] = CLAMP( CL_SHRT_MIN, u, CL_SHRT_MAX ); }
static void long2uint_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_uint*) out)[0] = (cl_uint) CLAMP( 0, u, CL_UINT_MAX ); }
static void long2int_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_int*) out)[0] = (int) CLAMP( CL_INT_MIN, u, CL_INT_MAX ); }
static void long2float_sat( void *out, void *in){ ((float*) out)[0] = (float) ((cl_long*) in)[0]; }
static void long2double_sat( void *out, void *in){ ((double*) out)[0] = ((cl_long*) in)[0]; }
static void long2ulong_sat( void *out, void *in){ cl_long u = ((cl_long*) in)[0]; ((cl_ulong*) out)[0] = CLAMP( 0, u, CL_LONG_MAX ); }

/*
#include <stdio.h>

char *ground[] = {   "",
                                                            "_rte",
                                                            "_rtp",
                                                            "_rtn",
                                                            "_rtz"
                    };

const char *gTypeNames[  ] = {
                                            "uchar", "char",
                                            "ushort", "short",
                                            "uint",   "int",
                                            "float", "double",
                                            "ulong", "long"
                                        };


int main( void )
{
    int i, j;

    for( i = 0; i < sizeof( gTypeNames ) / sizeof( gTypeNames[0] ); i++ )
        for( j = 0; j < sizeof( ground ) / sizeof( ground[0] ); j++ )
        {
            vlog( "float clampf_%s%s( float );\n", gTypeNames[i], ground[j] );
            vlog( "double clampd_%s%s( double );\n", gTypeNames[i], ground[j] );
        }

    return 0;

}
*/


float clampf_uchar( float );
double clampd_uchar( double );
float clampf_uchar_rte( float );
double clampd_uchar_rte( double );
float clampf_uchar_rtp( float );
double clampd_uchar_rtp( double );
float clampf_uchar_rtn( float );
double clampd_uchar_rtn( double );
float clampf_uchar_rtz( float );
double clampd_uchar_rtz( double );
float clampf_char( float );
double clampd_char( double );
float clampf_char_rte( float );
double clampd_char_rte( double );
float clampf_char_rtp( float );
double clampd_char_rtp( double );
float clampf_char_rtn( float );
double clampd_char_rtn( double );
float clampf_char_rtz( float );
double clampd_char_rtz( double );
float clampf_ushort( float );
double clampd_ushort( double );
float clampf_ushort_rte( float );
double clampd_ushort_rte( double );
float clampf_ushort_rtp( float );
double clampd_ushort_rtp( double );
float clampf_ushort_rtn( float );
double clampd_ushort_rtn( double );
float clampf_ushort_rtz( float );
double clampd_ushort_rtz( double );
float clampf_short( float );
double clampd_short( double );
float clampf_short_rte( float );
double clampd_short_rte( double );
float clampf_short_rtp( float );
double clampd_short_rtp( double );
float clampf_short_rtn( float );
double clampd_short_rtn( double );
float clampf_short_rtz( float );
double clampd_short_rtz( double );
float clampf_uint( float );
double clampd_uint( double );
float clampf_uint_rte( float );
double clampd_uint_rte( double );
float clampf_uint_rtp( float );
double clampd_uint_rtp( double );
float clampf_uint_rtn( float );
double clampd_uint_rtn( double );
float clampf_uint_rtz( float );
double clampd_uint_rtz( double );
float clampf_int( float );
double clampd_int( double );
float clampf_int_rte( float );
double clampd_int_rte( double );
float clampf_int_rtp( float );
double clampd_int_rtp( double );
float clampf_int_rtn( float );
double clampd_int_rtn( double );
float clampf_int_rtz( float );
double clampd_int_rtz( double );
float clampf_float( float );
double clampd_float( double );
float clampf_float_rte( float );
double clampd_float_rte( double );
float clampf_float_rtp( float );
double clampd_float_rtp( double );
float clampf_float_rtn( float );
double clampd_float_rtn( double );
float clampf_float_rtz( float );
double clampd_float_rtz( double );
float clampf_double( float );
double clampd_double( double );
float clampf_double_rte( float );
double clampd_double_rte( double );
float clampf_double_rtp( float );
double clampd_double_rtp( double );
float clampf_double_rtn( float );
double clampd_double_rtn( double );
float clampf_double_rtz( float );
double clampd_double_rtz( double );
float clampf_ulong( float );
double clampd_ulong( double );
float clampf_ulong_rte( float );
double clampd_ulong_rte( double );
float clampf_ulong_rtp( float );
double clampd_ulong_rtp( double );
float clampf_ulong_rtn( float );
double clampd_ulong_rtn( double );
float clampf_ulong_rtz( float );
double clampd_ulong_rtz( double );
float clampf_long( float );
double clampd_long( double );
float clampf_long_rte( float );
double clampd_long_rte( double );
float clampf_long_rtp( float );
double clampd_long_rtp( double );
float clampf_long_rtn( float );
double clampd_long_rtn( double );
float clampf_long_rtz( float );
double clampd_long_rtz( double );

/*
#include <stdio.h>

char *ground[] = {   "",
                                                            "_rte",
                                                            "_rtp",
                                                            "_rtn",
                                                            "_rtz"
                    };

const char *gTypeNames[  ] = {
                                            "uchar", "char",
                                            "ushort", "short",
                                            "uint",   "int",
                                            "float", "double",
                                            "ulong", "long"
                                        };


int main( void )
{
    int i, j;

    for( i = 0; i < sizeof( gTypeNames ) / sizeof( gTypeNames[0] ); i++ )
    {
        vlog( "{\t" );
        for( j = 0; j < sizeof( ground ) / sizeof( ground[0] ); j++ )
            vlog( "clampf_%s%s,\t", gTypeNames[i], ground[j] );

        vlog( "\t},\n" );
    }

    return 0;

}
*/
clampf gClampFloat[ kTypeCount ][kRoundingModeCount] = {
    {    clampf_uchar,    clampf_uchar_rte,    clampf_uchar_rtp,    clampf_uchar_rtn,    clampf_uchar_rtz,        },
    {    clampf_char,    clampf_char_rte,    clampf_char_rtp,    clampf_char_rtn,    clampf_char_rtz,        },
    {    clampf_ushort,    clampf_ushort_rte,    clampf_ushort_rtp,    clampf_ushort_rtn,    clampf_ushort_rtz,        },
    {    clampf_short,    clampf_short_rte,    clampf_short_rtp,    clampf_short_rtn,    clampf_short_rtz,        },
    {    clampf_uint,    clampf_uint_rte,    clampf_uint_rtp,    clampf_uint_rtn,    clampf_uint_rtz,        },
    {    clampf_int,     clampf_int_rte,     clampf_int_rtp,     clampf_int_rtn,     clampf_int_rtz,         },
    {    clampf_float,    clampf_float_rte,    clampf_float_rtp,    clampf_float_rtn,    clampf_float_rtz,        },
    {    clampf_double,    clampf_double_rte,    clampf_double_rtp,    clampf_double_rtn,    clampf_double_rtz,        },
    {    clampf_ulong,    clampf_ulong_rte,    clampf_ulong_rtp,    clampf_ulong_rtn,    clampf_ulong_rtz,        },
    {    clampf_long,    clampf_long_rte,    clampf_long_rtp,    clampf_long_rtn,    clampf_long_rtz,        }
};

clampd gClampDouble[ kTypeCount ][kRoundingModeCount] = {
    {    clampd_uchar,    clampd_uchar_rte,    clampd_uchar_rtp,    clampd_uchar_rtn,    clampd_uchar_rtz,        },
    {    clampd_char,    clampd_char_rte,    clampd_char_rtp,    clampd_char_rtn,    clampd_char_rtz,        },
    {    clampd_ushort,    clampd_ushort_rte,    clampd_ushort_rtp,    clampd_ushort_rtn,    clampd_ushort_rtz,        },
    {    clampd_short,    clampd_short_rte,    clampd_short_rtp,    clampd_short_rtn,    clampd_short_rtz,        },
    {    clampd_uint,    clampd_uint_rte,    clampd_uint_rtp,    clampd_uint_rtn,    clampd_uint_rtz,        },
    {    clampd_int,     clampd_int_rte,     clampd_int_rtp,     clampd_int_rtn,     clampd_int_rtz,         },
    {    clampd_float,    clampd_float_rte,    clampd_float_rtp,    clampd_float_rtn,    clampd_float_rtz,        },
    {    clampd_double,    clampd_double_rte,    clampd_double_rtp,    clampd_double_rtn,    clampd_double_rtz,        },
    {    clampd_ulong,    clampd_ulong_rte,    clampd_ulong_rtp,    clampd_ulong_rtn,    clampd_ulong_rtz,        },
    {    clampd_long,    clampd_long_rte,    clampd_long_rtp,    clampd_long_rtn,    clampd_long_rtz,        }
};

#if defined (_WIN32)
#define __attribute__(X)
#endif

static inline float fclamp( float lo, float v, float hi ) __attribute__ ((always_inline));
static inline double dclamp( double lo, double v, double hi ) __attribute__ ((always_inline));

static inline float fclamp( float lo, float v, float hi ){ v = v < lo ? lo : v; return v < hi ? v : hi; }
static inline double dclamp( double lo, double v, double hi ){ v = v < lo ? lo : v; return v < hi ? v : hi; }

// Clamp unsaturated inputs into range so we don't get test errors:
float clampf_uchar( float f )       { return fclamp( -0.5f, f, 255.5f - 128.0f * FLT_EPSILON ); }
double clampd_uchar( double f )     { return dclamp( -0.5, f, 255.5 - 128.0 * DBL_EPSILON ); }
float clampf_uchar_rte( float f )   { return fclamp( -0.5f, f, 255.5f - 128.0f * FLT_EPSILON ); }
double clampd_uchar_rte( double f ) { return dclamp( -0.5, f, 255.5 - 128.0 * DBL_EPSILON ); }
float clampf_uchar_rtp( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, 255.0f ); }
double clampd_uchar_rtp( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, 255.0 ); }
float clampf_uchar_rtn( float f )   { return fclamp( -0.0f, f, 256.0f - 128.0f * FLT_EPSILON); }
double clampd_uchar_rtn( double f ) { return dclamp( -0.0, f, 256.0 - 128.0 * DBL_EPSILON); }
float clampf_uchar_rtz( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, 256.0f - 128.0f * FLT_EPSILON); }
double clampd_uchar_rtz( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, 256.0 - 128.0f * DBL_EPSILON); }

float clampf_char( float f )        { return fclamp( -128.5f, f, 127.5f - 64.f * FLT_EPSILON ); }
double clampd_char( double f )      { return dclamp( -128.5, f, 127.5 - 64. * DBL_EPSILON ); }
float clampf_char_rte( float f )    { return fclamp( -128.5f, f, 127.5f - 64.f * FLT_EPSILON ); }
double clampd_char_rte( double f )  { return dclamp( -128.5, f, 127.5 - 64. * DBL_EPSILON ); }
float clampf_char_rtp( float f )    { return fclamp( -129.0f + 128.f*FLT_EPSILON, f, 127.f ); }
double clampd_char_rtp( double f )  { return dclamp( -129.0 + 128.*DBL_EPSILON, f, 127. ); }
float clampf_char_rtn( float f )    { return fclamp( -128.0f, f, 128.f - 64.0f*FLT_EPSILON ); }
double clampd_char_rtn( double f )  { return dclamp( -128.0, f, 128. - 64.0*DBL_EPSILON ); }
float clampf_char_rtz( float f )    { return fclamp( -129.0f + 128.f*FLT_EPSILON, f, 128.f - 64.0f*FLT_EPSILON ); }
double clampd_char_rtz( double f )  { return dclamp( -129.0 + 128.*DBL_EPSILON, f, 128. - 64.0*DBL_EPSILON ); }

float clampf_ushort( float f )       { return fclamp( -0.5f, f, 65535.5f - 32768.0f * FLT_EPSILON ); }
double clampd_ushort( double f )     { return dclamp( -0.5, f, 65535.5 - 32768.0 * DBL_EPSILON ); }
float clampf_ushort_rte( float f )   { return fclamp( -0.5f, f, 65535.5f - 32768.0f * FLT_EPSILON ); }
double clampd_ushort_rte( double f ) { return dclamp( -0.5, f, 65535.5 - 32768.0 * DBL_EPSILON ); }
float clampf_ushort_rtp( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, 65535.0f ); }
double clampd_ushort_rtp( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, 65535.0 ); }
float clampf_ushort_rtn( float f )   { return fclamp( -0.0f, f, 65536.0f - 32768.0f * FLT_EPSILON); }
double clampd_ushort_rtn( double f ) { return dclamp( -0.0, f, 65536.0 - 32768.0 * DBL_EPSILON); }
float clampf_ushort_rtz( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, 65536.0f - 32768.0f * FLT_EPSILON); }
double clampd_ushort_rtz( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, 65536.0 - 32768.0f * DBL_EPSILON); }

float clampf_short( float f )        { return fclamp( -32768.5f, f, 32767.5f - 16384.f * FLT_EPSILON ); }
double clampd_short( double f )      { return dclamp( -32768.5, f, 32767.5 - 16384. * DBL_EPSILON ); }
float clampf_short_rte( float f )    { return fclamp( -32768.5f, f, 32767.5f - 16384.f * FLT_EPSILON ); }
double clampd_short_rte( double f )  { return dclamp( -32768.5, f, 32767.5 - 16384. * DBL_EPSILON ); }
float clampf_short_rtp( float f )    { return fclamp( -32769.0f + 32768.f*FLT_EPSILON, f, 32767.f ); }
double clampd_short_rtp( double f )  { return dclamp( -32769.0 + 32768.*DBL_EPSILON, f, 32767. ); }
float clampf_short_rtn( float f )    { return fclamp( -32768.0f, f, 32768.f - 16384.0f*FLT_EPSILON ); }
double clampd_short_rtn( double f )  { return dclamp( -32768.0, f, 32768. - 16384.0*DBL_EPSILON ); }
float clampf_short_rtz( float f )    { return fclamp( -32769.0f + 32768.f*FLT_EPSILON, f, 32768.f - 16384.0f*FLT_EPSILON ); }
double clampd_short_rtz( double f )  { return dclamp( -32769.0 + 32768.*DBL_EPSILON, f, 32768. - 16384.0*DBL_EPSILON ); }

float clampf_uint( float f )        { return fclamp( -0.5f, f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7) ); }
double clampd_uint( double f )      { return dclamp( -0.5, f, CL_UINT_MAX + 0.5 - MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31) * DBL_EPSILON ); }
float clampf_uint_rte( float f )    { return fclamp( -0.5f, f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7) ); }
double clampd_uint_rte( double f )  { return dclamp( -0.5, f, CL_UINT_MAX + 0.5 - MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31) * DBL_EPSILON ); }
float clampf_uint_rtp( float f )    { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7) ); }
double clampd_uint_rtp( double f )  { return dclamp( -1.0 + DBL_EPSILON/2.0, f, CL_UINT_MAX ); }
float clampf_uint_rtn( float f )    { return fclamp( -0.0f, f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)); }
double clampd_uint_rtn( double f )  { return dclamp( -0.0, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp31, 0x1fffffffffffffLL, -21) ); }
float clampf_uint_rtz( float f )    { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, MAKE_HEX_FLOAT(0x1.fffffep31f, 0x1fffffeL, 7)); }
double clampd_uint_rtz( double f )  { return dclamp( -1.0 + DBL_EPSILON/2.0, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp31, 0x1fffffffffffffLL, -21)); }

float clampf_int( float f )         { return fclamp( INT_MIN, f, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) ); }
double clampd_int( double f )       { return dclamp( INT_MIN - 0.5, f, CL_INT_MAX + 0.5 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * DBL_EPSILON ); }
float clampf_int_rte( float f )     { return fclamp( INT_MIN, f, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) ); }
double clampd_int_rte( double f )   { return dclamp( INT_MIN - 0.5, f, CL_INT_MAX + 0.5 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * DBL_EPSILON ); }
float clampf_int_rtp( float f )     { return fclamp( INT_MIN, f, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) ); }
double clampd_int_rtp( double f )   { return dclamp( INT_MIN - 1.0 + DBL_EPSILON * MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31), f, CL_INT_MAX ); }
float clampf_int_rtn( float f )     { return fclamp( INT_MIN, f, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) ); }
double clampd_int_rtn( double f )   { return dclamp( INT_MIN, f, CL_INT_MAX + 1.0 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * DBL_EPSILON ); }
float clampf_int_rtz( float f )     { return fclamp( INT_MIN, f, MAKE_HEX_FLOAT(0x1.fffffep30f, 0x1fffffeL, 6) ); }
double clampd_int_rtz( double f )   { return dclamp( INT_MIN - 1.0 + DBL_EPSILON * MAKE_HEX_DOUBLE(0x1.0p31, 0x1LL, 31), f, CL_INT_MAX + 1.0 - MAKE_HEX_DOUBLE(0x1.0p30, 0x1LL, 30) * DBL_EPSILON ); }

float clampf_float( float f ){ return f; }
double clampd_float( double f ){ return f; }
float clampf_float_rte( float f ){ return f; }
double clampd_float_rte( double f ){ return f; }
float clampf_float_rtp( float f ){ return f; }
double clampd_float_rtp( double f ){ return f; }
float clampf_float_rtn( float f ){ return f; }
double clampd_float_rtn( double f ){ return f; }
float clampf_float_rtz( float f ){ return f; }
double clampd_float_rtz( double f ){ return f; }

float clampf_double( float f ){ return f; }
double clampd_double( double f ){ return f; }
float clampf_double_rte( float f ){ return f; }
double clampd_double_rte( double f ){ return f; }
float clampf_double_rtp( float f ){ return f; }
double clampd_double_rtp( double f ){ return f; }
float clampf_double_rtn( float f ){ return f; }
double clampd_double_rtn( double f ){ return f; }
float clampf_double_rtz( float f ){ return f; }
double clampd_double_rtz( double f ){ return f; }

float clampf_ulong( float f )       { return fclamp( -0.5f, f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) ); }
double clampd_ulong( double f )     { return dclamp( -0.5, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) ); }
float clampf_ulong_rte( float f )   { return fclamp( -0.5f, f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) ); }
double clampd_ulong_rte( double f ) { return dclamp( -0.5, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) ); }
float clampf_ulong_rtp( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) ); }
double clampd_ulong_rtp( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) ); }
float clampf_ulong_rtn( float f )   { return fclamp( -0.0f, f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) ); }
double clampd_ulong_rtn( double f ) { return dclamp( -0.0, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) ); }
float clampf_ulong_rtz( float f )   { return fclamp( -1.0f + FLT_EPSILON/2.0f, f, MAKE_HEX_FLOAT(0x1.fffffep63f, 0x1fffffeL, 39) ); }
double clampd_ulong_rtz( double f ) { return dclamp( -1.0 + DBL_EPSILON/2.0, f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp63, 0x1fffffffffffffLL, 11) ); }

float clampf_long( float f )        { return fclamp( MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), f, MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38) ); }
double clampd_long( double f )      { return dclamp( MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10) ); }
float clampf_long_rte( float f )    { return fclamp( MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), f, MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38) ); }
double clampd_long_rte( double f )  { return dclamp( MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10) ); }
float clampf_long_rtp( float f )    { return fclamp( MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), f, MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38) ); }
double clampd_long_rtp( double f )  { return dclamp( MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10) ); }
float clampf_long_rtn( float f )    { return fclamp( MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), f, MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38) ); }
double clampd_long_rtn( double f )  { return dclamp( MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10) ); }
float clampf_long_rtz( float f )    { return fclamp( MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), f, MAKE_HEX_FLOAT(0x1.fffffep62f, 0x1fffffeL, 38) ); }
double clampd_long_rtz( double f )  { return dclamp( MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), f, MAKE_HEX_DOUBLE(0x1.fffffffffffffp62, 0x1fffffffffffffLL, 10) ); }

#pragma mark -

int alwaysPass( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int alwaysFail( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_uchar( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_char( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_ushort( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_short( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_uint( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_int( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_ulong( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_long( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_float( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
int check_double( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );

void init_uchar( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_char( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_ushort( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_short( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_uint( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_int( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_float( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_double( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_ulong( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
void init_long( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );

InitDataFunc gInitFunctions[ kTypeCount ] = {
                                                init_uchar, init_char,
                                                init_ushort, init_short,
                                                init_uint, init_int,
                                                init_float, init_double,
                                                init_ulong, init_long
                                            };


CheckResults gCheckResults[ kTypeCount ] = {
                                                check_uchar, check_char, check_ushort, check_short, check_uint,
                                                check_int, check_float, check_double, check_ulong, check_long
                                            };
#if !defined (__APPLE__)
#define UNUSED
#else
#define UNUSED  __attribute__((unused))
#endif

int alwaysPass( void UNUSED *out1, void UNUSED *out2, void UNUSED *allowZ, uint32_t UNUSED count, int UNUSED vectorSize){ return 0; }
int alwaysFail( void UNUSED *out1, void UNUSED *out2, void UNUSED *allowZ, uint32_t UNUSED count, int UNUSED vectorSize ){ return -1; }

int check_uchar( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_uchar *t = (const cl_uchar*)test;
    const cl_uchar *c = (const cl_uchar*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_uchar)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%2.2x vs 0x%2.2x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_char( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_char *t = (const cl_char*)test;
    const cl_char *c = (const cl_char*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_char)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%2.2x vs 0x%2.2x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_ushort( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_ushort *t = (const cl_ushort*)test;
    const cl_ushort *c = (const cl_ushort*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_ushort)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%4.4x vs 0x%4.4x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_short( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_short *t = (const cl_short*)test;
    const cl_short *c = (const cl_short*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_short)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%4.4x vs 0x%4.4x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_uint( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_uint *t = (const cl_uint*)test;
    const cl_uint *c = (const cl_uint*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_uint)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%8.8x vs 0x%8.8x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_int( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_int *t = (const cl_int*)test;
    const cl_int *c = (const cl_int*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_int)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%8.8x vs 0x%8.8x\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_ulong( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_ulong *t = (const cl_ulong*)test;
    const cl_ulong *c = (const cl_ulong*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_ulong)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%16.16llx vs 0x%16.16llx\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_long( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_long *t = (const cl_long*)test;
    const cl_long *c = (const cl_long*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if( t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (cl_long)0))
        {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *0x%16.16llx vs 0x%16.16llx\n", vectorSize, i, c[i], t[i] );
            return i + 1;
        }

    return 0;
}

int check_float( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_uint *t = (const cl_uint*)test;
    const cl_uint *c = (const cl_uint*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if (t[i] != c[i] &&
            // Allow nan's to be binary different
            !((t[i] & 0x7fffffffU) > 0x7f800000U &&
              (c[i] & 0x7fffffffU) > 0x7f800000U) &&
            !(a[i] != (cl_uchar)0 &&
              t[i] == (c[i] & 0x80000000U))) {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((float*)correct)[i], ((float*)test)[i] );
            return i + 1;
        }

    return 0;
}

int check_double( void *test, void *correct, void *allowZ, uint32_t count, int vectorSize )
{
    const cl_ulong *t = (const cl_ulong*)test;
    const cl_ulong *c = (const cl_ulong*)correct;
    const cl_uchar *a = (const cl_uchar*)allowZ;
    uint32_t i;

    for( i = 0; i < count; i++ )
        if (t[i] != c[i] &&
            // Allow nan's to be binary different
            !((t[i] & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL &&
              (c[i] & 0x7fffffffffffffffULL) > 0x7f80000000000000ULL) &&
            !(a[i] != (cl_uchar)0 &&
              t[i] == (c[i] & 0x8000000000000000ULL))) {
            vlog( "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                  vectorSize, i, ((double*)correct)[i], ((double*)test)[i] );
            return i + 1;
        }

    return 0;
}


void init_uchar( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata UNUSED d )
{
    cl_uchar *o = (cl_uchar *)out;
    int i;

    for( i = 0; i < count; i++ )
        o[i] = start++;
}

void init_char( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata UNUSED d )
{
    char *o = (char *)out;
    int i;

    for( i = 0; i < count; i++ )
        o[i] = start++;
}

void init_ushort( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata UNUSED d )
{
    cl_ushort *o = (cl_ushort *)out;
    int i;

    for( i = 0; i < count; i++ )
        o[i] = start++;
}

void init_short( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, UNUSED Type destType, uint64_t start, int count, MTdata UNUSED d )
{
    short *o = (short *)out;
    int i;

    for( i = 0; i < count; i++ )
        o[i] = start++;
}

void init_uint( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata d )
{
    static const unsigned int specialValuesUInt[] = {
    INT_MIN, INT_MIN + 1, INT_MIN + 2,
    -(1<<30)-3,-(1<<30)-2,-(1<<30)-1, -(1<<30), -(1<<30)+1, -(1<<30)+2, -(1<<30)+3,
    -(1<<24)-3,-(1<<24)-2,-(1<<24)-1, -(1<<24), -(1<<24)+1, -(1<<24)+2, -(1<<24)+3,
    -(1<<23)-3,-(1<<23)-2,-(1<<23)-1, -(1<<23), -(1<<23)+1, -(1<<23)+2, -(1<<23)+3,
    -(1<<22)-3,-(1<<22)-2,-(1<<22)-1, -(1<<22), -(1<<22)+1, -(1<<22)+2, -(1<<22)+3,
    -(1<<21)-3,-(1<<21)-2,-(1<<21)-1, -(1<<21), -(1<<21)+1, -(1<<21)+2, -(1<<21)+3,
    -(1<<16)-3,-(1<<16)-2,-(1<<16)-1, -(1<<16), -(1<<16)+1, -(1<<16)+2, -(1<<16)+3,
    -(1<<15)-3,-(1<<15)-2,-(1<<15)-1, -(1<<15), -(1<<15)+1, -(1<<15)+2, -(1<<15)+3,
    -(1<<8)-3,-(1<<8)-2,-(1<<8)-1, -(1<<8), -(1<<8)+1, -(1<<8)+2, -(1<<8)+3,
    -(1<<7)-3,-(1<<7)-2,-(1<<7)-1, -(1<<7), -(1<<7)+1, -(1<<7)+2, -(1<<7)+3,
    -4, -3, -2, -1, 0, 1, 2, 3, 4,
    (1<<7)-3,(1<<7)-2,(1<<7)-1, (1<<7), (1<<7)+1, (1<<7)+2, (1<<7)+3,
    (1<<8)-3,(1<<8)-2,(1<<8)-1, (1<<8), (1<<8)+1, (1<<8)+2, (1<<8)+3,
    (1<<15)-3,(1<<15)-2,(1<<15)-1, (1<<15), (1<<15)+1, (1<<15)+2, (1<<15)+3,
    (1<<16)-3,(1<<16)-2,(1<<16)-1, (1<<16), (1<<16)+1, (1<<16)+2, (1<<16)+3,
    (1<<21)-3,(1<<21)-2,(1<<21)-1, (1<<21), (1<<21)+1, (1<<21)+2, (1<<21)+3,
    (1<<22)-3,(1<<22)-2,(1<<22)-1, (1<<22), (1<<22)+1, (1<<22)+2, (1<<22)+3,
    (1<<23)-3,(1<<23)-2,(1<<23)-1, (1<<23), (1<<23)+1, (1<<23)+2, (1<<23)+3,
    (1<<24)-3,(1<<24)-2,(1<<24)-1, (1<<24), (1<<24)+1, (1<<24)+2, (1<<24)+3,
    (1<<30)-3,(1<<30)-2,(1<<30)-1, (1<<30), (1<<30)+1, (1<<30)+2, (1<<30)+3,
    INT_MAX-3, INT_MAX-2, INT_MAX-1, INT_MAX, // 0x80000000, 0x80000001 0x80000002 already covered above
    UINT_MAX-3, UINT_MAX-2, UINT_MAX-1, UINT_MAX
    };

    cl_uint *o = (cl_uint *)out;
    int i;

    for( i = 0; i < count; i++) {
    if( gIsEmbedded )
        o[i] = (cl_uint) genrand_int32(d);
    else
        o[i] = (cl_uint)i + start;
    }

    if( 0 == start )
    {
    size_t tableSize = sizeof( specialValuesUInt );
    if( sizeof( cl_uint) * count < tableSize )
        tableSize = sizeof( cl_uint) * count;
    memcpy( (char*)(o + i) - tableSize, specialValuesUInt, tableSize );
    }
}

void init_int( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata d )
{
    static const unsigned int specialValuesInt[] = {
    INT_MIN, INT_MIN + 1, INT_MIN + 2,
    -(1<<30)-3,-(1<<30)-2,-(1<<30)-1, -(1<<30), -(1<<30)+1, -(1<<30)+2, -(1<<30)+3,
    -(1<<24)-3,-(1<<24)-2,-(1<<24)-1, -(1<<24), -(1<<24)+1, -(1<<24)+2, -(1<<24)+3,
    -(1<<23)-3,-(1<<23)-2,-(1<<23)-1, -(1<<23), -(1<<23)+1, -(1<<23)+2, -(1<<23)+3,
    -(1<<22)-3,-(1<<22)-2,-(1<<22)-1, -(1<<22), -(1<<22)+1, -(1<<22)+2, -(1<<22)+3,
    -(1<<21)-3,-(1<<21)-2,-(1<<21)-1, -(1<<21), -(1<<21)+1, -(1<<21)+2, -(1<<21)+3,
    -(1<<16)-3,-(1<<16)-2,-(1<<16)-1, -(1<<16), -(1<<16)+1, -(1<<16)+2, -(1<<16)+3,
    -(1<<15)-3,-(1<<15)-2,-(1<<15)-1, -(1<<15), -(1<<15)+1, -(1<<15)+2, -(1<<15)+3,
    -(1<<8)-3,-(1<<8)-2,-(1<<8)-1, -(1<<8), -(1<<8)+1, -(1<<8)+2, -(1<<8)+3,
    -(1<<7)-3,-(1<<7)-2,-(1<<7)-1, -(1<<7), -(1<<7)+1, -(1<<7)+2, -(1<<7)+3,
    -4, -3, -2, -1, 0, 1, 2, 3, 4,
    (1<<7)-3,(1<<7)-2,(1<<7)-1, (1<<7), (1<<7)+1, (1<<7)+2, (1<<7)+3,
    (1<<8)-3,(1<<8)-2,(1<<8)-1, (1<<8), (1<<8)+1, (1<<8)+2, (1<<8)+3,
    (1<<15)-3,(1<<15)-2,(1<<15)-1, (1<<15), (1<<15)+1, (1<<15)+2, (1<<15)+3,
    (1<<16)-3,(1<<16)-2,(1<<16)-1, (1<<16), (1<<16)+1, (1<<16)+2, (1<<16)+3,
    (1<<21)-3,(1<<21)-2,(1<<21)-1, (1<<21), (1<<21)+1, (1<<21)+2, (1<<21)+3,
    (1<<22)-3,(1<<22)-2,(1<<22)-1, (1<<22), (1<<22)+1, (1<<22)+2, (1<<22)+3,
    (1<<23)-3,(1<<23)-2,(1<<23)-1, (1<<23), (1<<23)+1, (1<<23)+2, (1<<23)+3,
    (1<<24)-3,(1<<24)-2,(1<<24)-1, (1<<24), (1<<24)+1, (1<<24)+2, (1<<24)+3,
    (1<<30)-3,(1<<30)-2,(1<<30)-1, (1<<30), (1<<30)+1, (1<<30)+2, (1<<30)+3,
    INT_MAX-3, INT_MAX-2, INT_MAX-1, INT_MAX, // 0x80000000, 0x80000001 0x80000002 already covered above
    UINT_MAX-3, UINT_MAX-2, UINT_MAX-1, UINT_MAX
    };

    int *o = (int *)out;
    int i;

    for( i = 0; i < count; i++ ) {
    if( gIsEmbedded ) {
        o[i] = (int) genrand_int32(d);
    }
    else {
        o[i] = (int) i + start;
    }
    }

    if( 0 == start )
    {
    size_t tableSize = sizeof( specialValuesInt );
    if( sizeof( int) * count < tableSize )
        tableSize = sizeof( int) * count;
    memcpy( (char*)(o + i) - tableSize, specialValuesInt, tableSize );
    }
}

void init_float( void *out, SaturationMode sat, RoundingMode round, Type destType, uint64_t start, int count, MTdata d )
{
    static const float specialValuesFloat[] = {
    -NAN, -INFINITY, -FLT_MAX, MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64), MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39), MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8), MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32), MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7), MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7), MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31), MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6), -1000.f, -100.f, -4.0f, -3.5f,
    -3.0f, MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23), -2.5f, MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23), -2.0f, MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24), -1.5f, MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24),MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24), -1.0f, MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25), -0.5f, MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26), MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26), -0.25f, MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150), -FLT_MIN, MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150), MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150), MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150), MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150), MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150), MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150), MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150), MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150), MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150), -0.0f,
    +NAN, +INFINITY, +FLT_MAX, MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40), MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64), MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39), MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39), MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63), MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8), MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32), MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7), MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7), MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31), MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6), +1000.f, +100.f, +4.0f, +3.5f,
    +3.0f, MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23), 2.5f, MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),+2.0f, MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24), 1.5f, MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24), MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24), +1.0f, MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25), +0.5f, MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26), MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26), +0.25f, MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150), +FLT_MIN, MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150), MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150), MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150), MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150), MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150), MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150), MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150), MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150), MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150), +0.0f
    };

    cl_uint *o = (cl_uint *)out;
    int i;

    for( i = 0; i < count; i++ ) {
    if( gIsEmbedded )
        o[i] = (cl_uint) genrand_int32(d);
    else
        o[i] = (cl_uint) i + start;
    }

    if( 0 == start )
    {
    size_t tableSize = sizeof( specialValuesFloat );
    if( sizeof( float) * count < tableSize )
        tableSize = sizeof( float) * count;
    memcpy( (char*)(o + i) - tableSize, specialValuesFloat, tableSize );
    }

    if( kUnsaturated == sat )
    {
        clampf func = gClampFloat[ destType ][round];
        float *f = (float *)out;

        for( i = 0; i < count; i++ )
            f[i] = func( f[i] );
    }
}

// used to convert a bucket of bits into a search pattern through double
static inline double DoubleFromUInt32( uint32_t bits );
static inline double DoubleFromUInt32( uint32_t bits )
{
    union{ uint64_t u; double d;} u;

    // split 0x89abcdef to 0x89abc00000000def
    u.u = bits & 0xfffU;
    u.u |= (uint64_t) (bits & ~0xfffU) << 32;

    // sign extend the leading bit of def segment as sign bit so that the middle region consists of either all 1s or 0s
    u.u -= (bits & 0x800U) << 1;

    // return result
    return u.d;
}

// A table of more difficult cases to get right
static const double specialValuesDouble[] = {
    -NAN, -INFINITY, -DBL_MAX, MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12), MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11), MAKE_HEX_DOUBLE(-0x1.80000000000001p64, -0x180000000000001LL, 8),
    MAKE_HEX_DOUBLE(-0x1.8p64, -0x18LL, 60), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp64, -0x17ffffffffffffLL, 12),     MAKE_HEX_DOUBLE(-0x1.80000000000001p63, -0x180000000000001LL, 7), MAKE_HEX_DOUBLE(-0x1.8p63, -0x18LL, 59), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp63, -0x17ffffffffffffLL, 11),
     MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11), MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10), MAKE_HEX_DOUBLE(-0x1.80000000000001p32, -0x180000000000001LL, -24), MAKE_HEX_DOUBLE(-0x1.8p32, -0x18LL, 28), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp32, -0x17ffffffffffffLL, -20),
    MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8), MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21), MAKE_HEX_DOUBLE(-0x1.80000000000001p31, -0x180000000000001LL, -25), MAKE_HEX_DOUBLE(-0x1.8p31, -0x18LL, 27), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp31, -0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21), MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22), -1000., -100.,  -4.0, -3.5,
    -3.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51), -2.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51), -2.0, MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52), -1.5, MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), -1.0, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53), -0.5, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54),  MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54), -0.25, MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074), -DBL_MIN, MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074), -0.0,

    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11), MAKE_HEX_DOUBLE(0x1.80000000000001p63, 0x180000000000001LL, 7), MAKE_HEX_DOUBLE(0x1.8p63, 0x18LL, 59), MAKE_HEX_DOUBLE(0x1.7ffffffffffffp63, 0x17ffffffffffffLL, 11),  MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11), MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
     MAKE_HEX_DOUBLE(+0x1.80000000000001p32, +0x180000000000001LL, -24), MAKE_HEX_DOUBLE(+0x1.8p32, +0x18LL, 28), MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp32, +0x17ffffffffffffLL, -20),
    MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8), MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21), MAKE_HEX_DOUBLE(+0x1.80000000000001p31, +0x180000000000001LL, -25), MAKE_HEX_DOUBLE(+0x1.8p31, +0x18LL, 27), MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp31, +0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21), MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22), +1000., +100.,  +4.0, +3.5,
    +3.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51), +2.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51), +2.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52), +1.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), +1.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53), +0.5, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54),  MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54), +0.25, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074), +DBL_MIN, MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074), +0.0,

    MAKE_HEX_DOUBLE(-0x1.ffffffffffffep62, -0x1ffffffffffffeLL, 10), MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp62, -0x1ffffffffffffcLL, 10), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10), MAKE_HEX_DOUBLE(+0x1.ffffffffffffep62, +0x1ffffffffffffeLL, 10), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp62, +0x1ffffffffffffcLL, 10), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffep51, -0x1ffffffffffffeLL, -1), MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp51, -0x1ffffffffffffcLL, -1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp51, -0x1fffffffffffffLL, -1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffep51, +0x1ffffffffffffeLL, -1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp51, +0x1ffffffffffffcLL, -1), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp51, +0x1fffffffffffffLL, -1),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffep52, -0x1ffffffffffffeLL, 0), MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp52, -0x1ffffffffffffcLL, 0), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp52, -0x1fffffffffffffLL, 0), MAKE_HEX_DOUBLE(+0x1.ffffffffffffep52, +0x1ffffffffffffeLL, 0), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp52, +0x1ffffffffffffcLL, 0), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp52, +0x1fffffffffffffLL, 0),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffep53, -0x1ffffffffffffeLL, 1), MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp53, -0x1ffffffffffffcLL, 1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp53, -0x1fffffffffffffLL, 1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffep53, +0x1ffffffffffffeLL, 1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp53, +0x1ffffffffffffcLL, 1), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp53, +0x1fffffffffffffLL, 1),
    MAKE_HEX_DOUBLE(-0x1.0000000000002p52, -0x10000000000002LL, 0), MAKE_HEX_DOUBLE(-0x1.0000000000001p52, -0x10000000000001LL, 0), MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52), MAKE_HEX_DOUBLE(+0x1.0000000000002p52, +0x10000000000002LL, 0), MAKE_HEX_DOUBLE(+0x1.0000000000001p52, +0x10000000000001LL, 0), MAKE_HEX_DOUBLE(+0x1.0p52, +0x1LL, 52),
    MAKE_HEX_DOUBLE(-0x1.0000000000002p53, -0x10000000000002LL, 1), MAKE_HEX_DOUBLE(-0x1.0000000000001p53, -0x10000000000001LL, 1), MAKE_HEX_DOUBLE(-0x1.0p53, -0x1LL, 53), MAKE_HEX_DOUBLE(+0x1.0000000000002p53, +0x10000000000002LL, 1), MAKE_HEX_DOUBLE(+0x1.0000000000001p53, +0x10000000000001LL, 1), MAKE_HEX_DOUBLE(+0x1.0p53, +0x1LL, 53),
    MAKE_HEX_DOUBLE(-0x1.0000000000002p54, -0x10000000000002LL, 2), MAKE_HEX_DOUBLE(-0x1.0000000000001p54, -0x10000000000001LL, 2), MAKE_HEX_DOUBLE(-0x1.0p54, -0x1LL, 54), MAKE_HEX_DOUBLE(+0x1.0000000000002p54, +0x10000000000002LL, 2), MAKE_HEX_DOUBLE(+0x1.0000000000001p54, +0x10000000000001LL, 2), MAKE_HEX_DOUBLE(+0x1.0p54, +0x1LL, 54),
    MAKE_HEX_DOUBLE(-0x1.fffffffefffffp62, -0x1fffffffefffffLL, 10), MAKE_HEX_DOUBLE(-0x1.ffffffffp62, -0x1ffffffffLL, 30), MAKE_HEX_DOUBLE(-0x1.ffffffff00001p62, -0x1ffffffff00001LL, 10), MAKE_HEX_DOUBLE(0x1.fffffffefffffp62, 0x1fffffffefffffLL, 10), MAKE_HEX_DOUBLE(0x1.ffffffffp62, 0x1ffffffffLL, 30), MAKE_HEX_DOUBLE(0x1.ffffffff00001p62, 0x1ffffffff00001LL, 10),
};


void init_double( void *out, SaturationMode sat, RoundingMode round, Type destType, uint64_t start, int count, MTdata UNUSED d )
{
    double *o = (double*)out;
    int i;

    for( i = 0; i < count; i++ )
    {
        uint64_t z = i + start;
        o[i] = DoubleFromUInt32( (uint32_t) z ^ (uint32_t) (z >> 32));
    }

    if( 0 == start )
    {
        size_t tableSize = sizeof( specialValuesDouble );
        if( sizeof( cl_double) * count < tableSize )
            tableSize = sizeof( cl_double) * count;
        memcpy( (char*)(o + i) - tableSize, specialValuesDouble, tableSize );
    }

    if( 0 == sat )
    {
        clampd func = gClampDouble[ destType ][round];

        for( i = 0; i < count; i++ )
            o[i] = func( o[i] );
    }
}

cl_ulong random64( MTdata d )
{
    return (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
}

void init_ulong( void *out, SaturationMode UNUSED sat, RoundingMode UNUSED round, Type UNUSED destType, uint64_t start, int count, MTdata d )
{
    cl_ulong *o = (cl_ulong *)out;
    cl_ulong i, j, k;

    i = 0;
    if( start == 0 )
    {
        //Try various powers of two
        for( j = 0; j < (cl_ulong) count && j < 8 * sizeof(cl_ulong); j++ )
            o[j] = (cl_ulong) 1 << j;
        i = j;

        // try the complement of those
        for( j = 0; i < (cl_ulong) count && j < 8 * sizeof(cl_ulong); j++ )
            o[i++] = ~((cl_ulong) 1 << j);

        //Try various negative powers of two
        for( j = 0; i < (cl_ulong) count && j < 8 * sizeof(cl_ulong); j++ )
            o[i++] = (cl_ulong) 0xFFFFFFFFFFFFFFFEULL << j;

        //try various powers of two plus 1, shifted by various amounts
        for( j = 0; i < (cl_ulong)count && j < 8 * sizeof(cl_ulong); j++ )
            for( k = 0; i < (cl_ulong)count && k < 8 * sizeof(cl_ulong) - j; k++ )
                o[i++] = (((cl_ulong) 1 << j) + 1) << k;

        //try various powers of two minus 1
        for( j = 0; i < (cl_ulong)count && j < 8 * sizeof(cl_ulong); j++ )
            for( k = 0; i < (cl_ulong)count && k < 8 * sizeof(cl_ulong) - j; k++ )
                o[i++] = (((cl_ulong) 1 << j) - 1) << k;

        // Other patterns
        cl_ulong pattern[] = { 0x3333333333333333ULL, 0x5555555555555555ULL, 0x9999999999999999ULL, 0x6666666666666666ULL, 0xccccccccccccccccULL, 0xaaaaaaaaaaaaaaaaULL };
        cl_ulong mask[] = { 0xffffffffffffffffULL, 0xff00ff00ff00ff00ULL, 0xffff0000ffff0000ULL, 0xffffffff00000000ULL };
        for( j = 0; i < (cl_ulong) count && j < sizeof(pattern) / sizeof( pattern[0]); j++ )
            for( k = 0; i + 2 <= (cl_ulong) count && k < sizeof(mask) / sizeof( mask[0]); k++ )
            {
                o[i++] = pattern[j] & mask[k];
                o[i++] = pattern[j] & ~mask[k];
            }
    }

    for( ; i < (cl_ulong) count; i++ )
        o[i] = random64(d);
}

void init_long( void *out, SaturationMode sat, RoundingMode round, Type destType, uint64_t start, int count, MTdata d )
{
    init_ulong( out, sat, round, destType, start, count, d );
}

// ======

void uchar2uchar_many( void *out, void *in, size_t n);
void uchar2uchar_sat_many( void *out, void *in, size_t n);
void char2uchar_many( void *out, void *in, size_t n);
void char2uchar_sat_many( void *out, void *in, size_t n);
void ushort2uchar_many( void *out, void *in, size_t n);
void ushort2uchar_sat_many( void *out, void *in, size_t n);
void short2uchar_many( void *out, void *in, size_t n);
void short2uchar_sat_many( void *out, void *in, size_t n);
void uint2uchar_many( void *out, void *in, size_t n);
void uint2uchar_sat_many( void *out, void *in, size_t n);
void int2uchar_many( void *out, void *in, size_t n);
void int2uchar_sat_many( void *out, void *in, size_t n);
void float2uchar_many( void *out, void *in, size_t n);
void float2uchar_sat_many( void *out, void *in, size_t n);
void double2uchar_many( void *out, void *in, size_t n);
void double2uchar_sat_many( void *out, void *in, size_t n);
void ulong2uchar_many( void *out, void *in, size_t n);
void ulong2uchar_sat_many( void *out, void *in, size_t n);
void long2uchar_many( void *out, void *in, size_t n);
void long2uchar_sat_many( void *out, void *in, size_t n);
void uchar2char_many( void *out, void *in, size_t n);
void uchar2char_sat_many( void *out, void *in, size_t n);
void char2char_many( void *out, void *in, size_t n);
void char2char_sat_many( void *out, void *in, size_t n);
void ushort2char_many( void *out, void *in, size_t n);
void ushort2char_sat_many( void *out, void *in, size_t n);
void short2char_many( void *out, void *in, size_t n);
void short2char_sat_many( void *out, void *in, size_t n);
void uint2char_many( void *out, void *in, size_t n);
void uint2char_sat_many( void *out, void *in, size_t n);
void int2char_many( void *out, void *in, size_t n);
void int2char_sat_many( void *out, void *in, size_t n);
void float2char_many( void *out, void *in, size_t n);
void float2char_sat_many( void *out, void *in, size_t n);
void double2char_many( void *out, void *in, size_t n);
void double2char_sat_many( void *out, void *in, size_t n);
void ulong2char_many( void *out, void *in, size_t n);
void ulong2char_sat_many( void *out, void *in, size_t n);
void long2char_many( void *out, void *in, size_t n);
void long2char_sat_many( void *out, void *in, size_t n);
void uchar2ushort_many( void *out, void *in, size_t n);
void uchar2ushort_sat_many( void *out, void *in, size_t n);
void char2ushort_many( void *out, void *in, size_t n);
void char2ushort_sat_many( void *out, void *in, size_t n);
void ushort2ushort_many( void *out, void *in, size_t n);
void ushort2ushort_sat_many( void *out, void *in, size_t n);
void short2ushort_many( void *out, void *in, size_t n);
void short2ushort_sat_many( void *out, void *in, size_t n);
void uint2ushort_many( void *out, void *in, size_t n);
void uint2ushort_sat_many( void *out, void *in, size_t n);
void int2ushort_many( void *out, void *in, size_t n);
void int2ushort_sat_many( void *out, void *in, size_t n);
void float2ushort_many( void *out, void *in, size_t n);
void float2ushort_sat_many( void *out, void *in, size_t n);
void double2ushort_many( void *out, void *in, size_t n);
void double2ushort_sat_many( void *out, void *in, size_t n);
void ulong2ushort_many( void *out, void *in, size_t n);
void ulong2ushort_sat_many( void *out, void *in, size_t n);
void long2ushort_many( void *out, void *in, size_t n);
void long2ushort_sat_many( void *out, void *in, size_t n);
void uchar2short_many( void *out, void *in, size_t n);
void uchar2short_sat_many( void *out, void *in, size_t n);
void char2short_many( void *out, void *in, size_t n);
void char2short_sat_many( void *out, void *in, size_t n);
void ushort2short_many( void *out, void *in, size_t n);
void ushort2short_sat_many( void *out, void *in, size_t n);
void short2short_many( void *out, void *in, size_t n);
void short2short_sat_many( void *out, void *in, size_t n);
void uint2short_many( void *out, void *in, size_t n);
void uint2short_sat_many( void *out, void *in, size_t n);
void int2short_many( void *out, void *in, size_t n);
void int2short_sat_many( void *out, void *in, size_t n);
void float2short_many( void *out, void *in, size_t n);
void float2short_sat_many( void *out, void *in, size_t n);
void double2short_many( void *out, void *in, size_t n);
void double2short_sat_many( void *out, void *in, size_t n);
void ulong2short_many( void *out, void *in, size_t n);
void ulong2short_sat_many( void *out, void *in, size_t n);
void long2short_many( void *out, void *in, size_t n);
void long2short_sat_many( void *out, void *in, size_t n);
void uchar2uint_many( void *out, void *in, size_t n);
void uchar2uint_sat_many( void *out, void *in, size_t n);
void char2uint_many( void *out, void *in, size_t n);
void char2uint_sat_many( void *out, void *in, size_t n);
void ushort2uint_many( void *out, void *in, size_t n);
void ushort2uint_sat_many( void *out, void *in, size_t n);
void short2uint_many( void *out, void *in, size_t n);
void short2uint_sat_many( void *out, void *in, size_t n);
void uint2uint_many( void *out, void *in, size_t n);
void uint2uint_sat_many( void *out, void *in, size_t n);
void int2uint_many( void *out, void *in, size_t n);
void int2uint_sat_many( void *out, void *in, size_t n);
void float2uint_many( void *out, void *in, size_t n);
void float2uint_sat_many( void *out, void *in, size_t n);
void double2uint_many( void *out, void *in, size_t n);
void double2uint_sat_many( void *out, void *in, size_t n);
void ulong2uint_many( void *out, void *in, size_t n);
void ulong2uint_sat_many( void *out, void *in, size_t n);
void long2uint_many( void *out, void *in, size_t n);
void long2uint_sat_many( void *out, void *in, size_t n);
void uchar2int_many( void *out, void *in, size_t n);
void uchar2int_sat_many( void *out, void *in, size_t n);
void char2int_many( void *out, void *in, size_t n);
void char2int_sat_many( void *out, void *in, size_t n);
void ushort2int_many( void *out, void *in, size_t n);
void ushort2int_sat_many( void *out, void *in, size_t n);
void short2int_many( void *out, void *in, size_t n);
void short2int_sat_many( void *out, void *in, size_t n);
void uint2int_many( void *out, void *in, size_t n);
void uint2int_sat_many( void *out, void *in, size_t n);
void int2int_many( void *out, void *in, size_t n);
void int2int_sat_many( void *out, void *in, size_t n);
void float2int_many( void *out, void *in, size_t n);
void float2int_sat_many( void *out, void *in, size_t n);
void double2int_many( void *out, void *in, size_t n);
void double2int_sat_many( void *out, void *in, size_t n);
void ulong2int_many( void *out, void *in, size_t n);
void ulong2int_sat_many( void *out, void *in, size_t n);
void long2int_many( void *out, void *in, size_t n);
void long2int_sat_many( void *out, void *in, size_t n);
void uchar2float_many( void *out, void *in, size_t n);
void uchar2float_sat_many( void *out, void *in, size_t n);
void char2float_many( void *out, void *in, size_t n);
void char2float_sat_many( void *out, void *in, size_t n);
void ushort2float_many( void *out, void *in, size_t n);
void ushort2float_sat_many( void *out, void *in, size_t n);
void short2float_many( void *out, void *in, size_t n);
void short2float_sat_many( void *out, void *in, size_t n);
void uint2float_many( void *out, void *in, size_t n);
void uint2float_sat_many( void *out, void *in, size_t n);
void int2float_many( void *out, void *in, size_t n);
void int2float_sat_many( void *out, void *in, size_t n);
void float2float_many( void *out, void *in, size_t n);
void float2float_sat_many( void *out, void *in, size_t n);
void double2float_many( void *out, void *in, size_t n);
void double2float_sat_many( void *out, void *in, size_t n);
void ulong2float_many( void *out, void *in, size_t n);
void ulong2float_sat_many( void *out, void *in, size_t n);
void long2float_many( void *out, void *in, size_t n);
void long2float_sat_many( void *out, void *in, size_t n);
void uchar2double_many( void *out, void *in, size_t n);
void uchar2double_sat_many( void *out, void *in, size_t n);
void char2double_many( void *out, void *in, size_t n);
void char2double_sat_many( void *out, void *in, size_t n);
void ushort2double_many( void *out, void *in, size_t n);
void ushort2double_sat_many( void *out, void *in, size_t n);
void short2double_many( void *out, void *in, size_t n);
void short2double_sat_many( void *out, void *in, size_t n);
void uint2double_many( void *out, void *in, size_t n);
void uint2double_sat_many( void *out, void *in, size_t n);
void int2double_many( void *out, void *in, size_t n);
void int2double_sat_many( void *out, void *in, size_t n);
void float2double_many( void *out, void *in, size_t n);
void float2double_sat_many( void *out, void *in, size_t n);
void double2double_many( void *out, void *in, size_t n);
void double2double_sat_many( void *out, void *in, size_t n);
void ulong2double_many( void *out, void *in, size_t n);
void ulong2double_sat_many( void *out, void *in, size_t n);
void long2double_many( void *out, void *in, size_t n);
void long2double_sat_many( void *out, void *in, size_t n);
void uchar2ulong_many( void *out, void *in, size_t n);
void uchar2ulong_sat_many( void *out, void *in, size_t n);
void char2ulong_many( void *out, void *in, size_t n);
void char2ulong_sat_many( void *out, void *in, size_t n);
void ushort2ulong_many( void *out, void *in, size_t n);
void ushort2ulong_sat_many( void *out, void *in, size_t n);
void short2ulong_many( void *out, void *in, size_t n);
void short2ulong_sat_many( void *out, void *in, size_t n);
void uint2ulong_many( void *out, void *in, size_t n);
void uint2ulong_sat_many( void *out, void *in, size_t n);
void int2ulong_many( void *out, void *in, size_t n);
void int2ulong_sat_many( void *out, void *in, size_t n);
void float2ulong_many( void *out, void *in, size_t n);
void float2ulong_sat_many( void *out, void *in, size_t n);
void double2ulong_many( void *out, void *in, size_t n);
void double2ulong_sat_many( void *out, void *in, size_t n);
void ulong2ulong_many( void *out, void *in, size_t n);
void ulong2ulong_sat_many( void *out, void *in, size_t n);
void long2ulong_many( void *out, void *in, size_t n);
void long2ulong_sat_many( void *out, void *in, size_t n);
void uchar2long_many( void *out, void *in, size_t n);
void uchar2long_sat_many( void *out, void *in, size_t n);
void char2long_many( void *out, void *in, size_t n);
void char2long_sat_many( void *out, void *in, size_t n);
void ushort2long_many( void *out, void *in, size_t n);
void ushort2long_sat_many( void *out, void *in, size_t n);
void short2long_many( void *out, void *in, size_t n);
void short2long_sat_many( void *out, void *in, size_t n);
void uint2long_many( void *out, void *in, size_t n);
void uint2long_sat_many( void *out, void *in, size_t n);
void int2long_many( void *out, void *in, size_t n);
void int2long_sat_many( void *out, void *in, size_t n);
void float2long_many( void *out, void *in, size_t n);
void float2long_sat_many( void *out, void *in, size_t n);
void double2long_many( void *out, void *in, size_t n);
void double2long_sat_many( void *out, void *in, size_t n);
void ulong2long_many( void *out, void *in, size_t n);
void ulong2long_sat_many( void *out, void *in, size_t n);
void long2long_many( void *out, void *in, size_t n);
void long2long_sat_many( void *out, void *in, size_t n);

void uchar2uchar_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_uchar )); }
void uchar2uchar_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_uchar )); }
void char2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_char)); }}
void char2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_char)); }}
void ushort2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_ushort)); }}
void short2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_short)); }}
void short2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_short)); }}
void uint2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_uint)); }}
void uint2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_uint)); }}
void int2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_int)); }}
void int2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_int)); }}
void float2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_float)); }}
void float2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_float)); }}
void double2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_double)); }}
void double2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_double)); }}
void ulong2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_ulong)); }}
void long2uchar_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2uchar( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_long)); }}
void long2uchar_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2uchar_sat( (char*) out + i * sizeof(cl_uchar), (char*) in + i * sizeof(cl_long)); }}
void uchar2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_uchar)); }}
void char2char_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_char )); }
void char2char_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_char )); }
void ushort2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_ushort)); }}
void short2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_short)); }}
void short2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_short)); }}
void uint2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_uint)); }}
void uint2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_uint)); }}
void int2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_int)); }}
void int2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_int)); }}
void float2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_float)); }}
void float2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_float)); }}
void double2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_double)); }}
void double2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_double)); }}
void ulong2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_ulong)); }}
void long2char_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2char( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_long)); }}
void long2char_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2char_sat( (char*) out + i * sizeof(cl_char), (char*) in + i * sizeof(cl_long)); }}
void uchar2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_uchar)); }}
void char2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_char)); }}
void char2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_char)); }}
void ushort2ushort_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_ushort )); }
void ushort2ushort_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_ushort )); }
void short2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_short)); }}
void short2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_short)); }}
void uint2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_uint)); }}
void uint2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_uint)); }}
void int2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_int)); }}
void int2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_int)); }}
void float2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_float)); }}
void float2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_float)); }}
void double2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_double)); }}
void double2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_double)); }}
void ulong2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_ulong)); }}
void long2ushort_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2ushort( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_long)); }}
void long2ushort_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2ushort_sat( (char*) out + i * sizeof(cl_ushort), (char*) in + i * sizeof(cl_long)); }}
void uchar2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_uchar)); }}
void char2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_char)); }}
void char2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_char)); }}
void ushort2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_ushort)); }}
void short2short_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_short )); }
void short2short_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_short )); }
void uint2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_uint)); }}
void uint2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_uint)); }}
void int2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_int)); }}
void int2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_int)); }}
void float2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_float)); }}
void float2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_float)); }}
void double2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_double)); }}
void double2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_double)); }}
void ulong2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_ulong)); }}
void long2short_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2short( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_long)); }}
void long2short_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2short_sat( (char*) out + i * sizeof(cl_short), (char*) in + i * sizeof(cl_long)); }}
void uchar2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_uchar)); }}
void char2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_char)); }}
void char2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_char)); }}
void ushort2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_ushort)); }}
void short2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_short)); }}
void short2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_short)); }}
void uint2uint_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_uint )); }
void uint2uint_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_uint )); }
void int2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_int)); }}
void int2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_int)); }}
void float2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_float)); }}
void float2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_float)); }}
void double2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_double)); }}
void double2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_double)); }}
void ulong2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_ulong)); }}
void long2uint_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2uint( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_long)); }}
void long2uint_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2uint_sat( (char*) out + i * sizeof(cl_uint), (char*) in + i * sizeof(cl_long)); }}
void uchar2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_uchar)); }}
void char2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_char)); }}
void char2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_char)); }}
void ushort2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_ushort)); }}
void short2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_short)); }}
void short2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_short)); }}
void uint2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_uint)); }}
void uint2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_uint)); }}
void int2int_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_int )); }
void int2int_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_int )); }
void float2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_float)); }}
void float2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_float)); }}
void double2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_double)); }}
void double2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_double)); }}
void ulong2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_ulong)); }}
void long2int_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2int( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_long)); }}
void long2int_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2int_sat( (char*) out + i * sizeof(cl_int), (char*) in + i * sizeof(cl_long)); }}
void uchar2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_uchar)); }}
void char2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_char)); }}
void char2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_char)); }}
void ushort2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_ushort)); }}
void short2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_short)); }}
void short2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_short)); }}
void uint2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_uint)); }}
void uint2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_uint)); }}
void int2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_int)); }}
void int2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_int)); }}
void float2float_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_float )); }
void float2float_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_float )); }
void double2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_double)); }}
void double2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_double)); }}
void ulong2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_ulong)); }}
void long2float_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2float( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_long)); }}
void long2float_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2float_sat( (char*) out + i * sizeof(cl_float), (char*) in + i * sizeof(cl_long)); }}
void uchar2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_uchar)); }}
void char2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_char)); }}
void char2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_char)); }}
void ushort2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_ushort)); }}
void short2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_short)); }}
void short2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_short)); }}
void uint2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_uint)); }}
void uint2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_uint)); }}
void int2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_int)); }}
void int2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_int)); }}
void float2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_float)); }}
void float2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_float)); }}
void double2double_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_double )); }
void double2double_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_double )); }
void ulong2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_ulong)); }}
void long2double_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2double( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_long)); }}
void long2double_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2double_sat( (char*) out + i * sizeof(cl_double), (char*) in + i * sizeof(cl_long)); }}
void uchar2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_uchar)); }}
void char2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_char)); }}
void char2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_char)); }}
void ushort2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_ushort)); }}
void short2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_short)); }}
void short2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_short)); }}
void uint2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_uint)); }}
void uint2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_uint)); }}
void int2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_int)); }}
void int2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_int)); }}
void float2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_float)); }}
void float2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_float)); }}
void double2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_double)); }}
void double2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_double)); }}
void ulong2ulong_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_ulong )); }
void ulong2ulong_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_ulong )); }
void long2ulong_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2ulong( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_long)); }}
void long2ulong_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ long2ulong_sat( (char*) out + i * sizeof(cl_ulong), (char*) in + i * sizeof(cl_long)); }}
void uchar2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_uchar)); }}
void uchar2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uchar2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_uchar)); }}
void char2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_char)); }}
void char2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ char2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_char)); }}
void ushort2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_ushort)); }}
void ushort2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ushort2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_ushort)); }}
void short2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_short)); }}
void short2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ short2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_short)); }}
void uint2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_uint)); }}
void uint2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ uint2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_uint)); }}
void int2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_int)); }}
void int2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ int2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_int)); }}
void float2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_float)); }}
void float2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ float2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_float)); }}
void double2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_double)); }}
void double2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ double2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_double)); }}
void ulong2long_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2long( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_ulong)); }}
void ulong2long_sat_many( void *out, void *in, size_t n){size_t i; for( i = 0; i < n; i++){ ulong2long_sat( (char*) out + i * sizeof(cl_long), (char*) in + i * sizeof(cl_ulong)); }}
void long2long_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_long )); }
void long2long_sat_many( void *out, void *in, size_t n){ memcpy( out, in, n * sizeof( cl_long )); }

Convert gSaturatedConversions[kTypeCount][kTypeCount] = {
    {    uchar2uchar_sat_many,    char2uchar_sat_many,    ushort2uchar_sat_many,    short2uchar_sat_many,    uint2uchar_sat_many,    int2uchar_sat_many,    float2uchar_sat_many,    double2uchar_sat_many,    ulong2uchar_sat_many,    long2uchar_sat_many,     },
    {    uchar2char_sat_many,    char2char_sat_many,    ushort2char_sat_many,    short2char_sat_many,    uint2char_sat_many,    int2char_sat_many,    float2char_sat_many,    double2char_sat_many,    ulong2char_sat_many, long2char_sat_many,     },
    {    uchar2ushort_sat_many,    char2ushort_sat_many,    ushort2ushort_sat_many,    short2ushort_sat_many,    uint2ushort_sat_many,    int2ushort_sat_many,    float2ushort_sat_many,    double2ushort_sat_many,    ulong2ushort_sat_many,    long2ushort_sat_many,     },
    {    uchar2short_sat_many,    char2short_sat_many,    ushort2short_sat_many,    short2short_sat_many,    uint2short_sat_many,    int2short_sat_many,    float2short_sat_many,    double2short_sat_many,    ulong2short_sat_many,    long2short_sat_many,     },
    {    uchar2uint_sat_many,    char2uint_sat_many,    ushort2uint_sat_many,    short2uint_sat_many,    uint2uint_sat_many,    int2uint_sat_many,    float2uint_sat_many,    double2uint_sat_many,    ulong2uint_sat_many, long2uint_sat_many,     },
    {    uchar2int_sat_many,    char2int_sat_many,    ushort2int_sat_many,    short2int_sat_many,    uint2int_sat_many,    int2int_sat_many,    float2int_sat_many,    double2int_sat_many,    ulong2int_sat_many,long2int_sat_many,     },
    {    uchar2float_sat_many,    char2float_sat_many,    ushort2float_sat_many,    short2float_sat_many,    uint2float_sat_many,    int2float_sat_many,    float2float_sat_many,    double2float_sat_many,    ulong2float_sat_many,    long2float_sat_many,     },
    {    uchar2double_sat_many,    char2double_sat_many,    ushort2double_sat_many,    short2double_sat_many,    uint2double_sat_many,    int2double_sat_many,    float2double_sat_many,    double2double_sat_many,    ulong2double_sat_many,    long2double_sat_many,     },
    {    uchar2ulong_sat_many,    char2ulong_sat_many,    ushort2ulong_sat_many,    short2ulong_sat_many,    uint2ulong_sat_many,    int2ulong_sat_many,    float2ulong_sat_many,    double2ulong_sat_many,    ulong2ulong_sat_many,    long2ulong_sat_many,     },
    {    uchar2long_sat_many,    char2long_sat_many,    ushort2long_sat_many,    short2long_sat_many,    uint2long_sat_many,    int2long_sat_many,    float2long_sat_many,    double2long_sat_many,    ulong2long_sat_many, long2long_sat_many,     },
};

Convert gConversions[kTypeCount][kTypeCount] = {
    {    uchar2uchar_many,    char2uchar_many,    ushort2uchar_many,    short2uchar_many,    uint2uchar_many,    int2uchar_many,    float2uchar_many,    double2uchar_many,    ulong2uchar_many,    long2uchar_many,     },
    {    uchar2char_many,    char2char_many,    ushort2char_many,    short2char_many,    uint2char_many,    int2char_many,    float2char_many,    double2char_many,    ulong2char_many,    long2char_many,     },
    {    uchar2ushort_many,    char2ushort_many,    ushort2ushort_many,    short2ushort_many,    uint2ushort_many,    int2ushort_many,    float2ushort_many,    double2ushort_many,    ulong2ushort_many,    long2ushort_many,     },
    {    uchar2short_many,    char2short_many,    ushort2short_many,    short2short_many,    uint2short_many,    int2short_many,    float2short_many,    double2short_many,    ulong2short_many,    long2short_many,     },
    {    uchar2uint_many,    char2uint_many,    ushort2uint_many,    short2uint_many,    uint2uint_many,    int2uint_many,    float2uint_many,    double2uint_many,    ulong2uint_many,    long2uint_many,     },
    {    uchar2int_many,    char2int_many,    ushort2int_many,    short2int_many,    uint2int_many,    int2int_many,    float2int_many,    double2int_many,    ulong2int_many,    long2int_many,     },
    {    uchar2float_many,    char2float_many,    ushort2float_many,    short2float_many,    uint2float_many,    int2float_many,    float2float_many,    double2float_many,    ulong2float_many,    long2float_many,     },
    {    uchar2double_many,    char2double_many,    ushort2double_many,    short2double_many,    uint2double_many,    int2double_many,    float2double_many,    double2double_many,    ulong2double_many,    long2double_many,     },
    {    uchar2ulong_many,    char2ulong_many,    ushort2ulong_many,    short2ulong_many,    uint2ulong_many,    int2ulong_many,    float2ulong_many,    double2ulong_many,    ulong2ulong_many,    long2ulong_many,     },
    {    uchar2long_many,    char2long_many,    ushort2long_many,    short2long_many,    uint2long_many,    int2long_many,    float2long_many,    double2long_many,    ulong2long_many,    long2long_many,     },
};
