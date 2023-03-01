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
#include "harness/compat.h"

#include "basic_test_conversions.h"
#include <limits.h>
#include <string.h>

#include "harness/mt19937.h"

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
/* Rounding modes and saturation for use with qcom 64 bit to float conversion
 * library */
bool qcom_sat;
roundingMode qcom_rm;
#endif

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

// ======
