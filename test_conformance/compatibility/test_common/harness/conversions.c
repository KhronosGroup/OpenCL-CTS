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
#include "conversions.h"
#include <limits.h>
#include <time.h>
#include <assert.h>
#include "mt19937.h"
#include "compat.h"

#if defined( __SSE__ ) || defined (_MSC_VER)
    #include <xmmintrin.h>
#endif
#if defined( __SSE2__ ) || defined (_MSC_VER)
    #include <emmintrin.h>
#endif

void print_type_to_string(ExplicitType type, void *data, char* string) {
     switch (type) {
       case kBool:
      if (*(char*)data)
        sprintf(string, "true");
      else
        sprintf(string, "false");
            return;
    case kChar:
      sprintf(string, "%d", (int)*((cl_char*)data));
      return;
    case kUChar:
    case kUnsignedChar:
      sprintf(string, "%u", (int)*((cl_uchar*)data));
      return;
    case kShort:
      sprintf(string, "%d", (int)*((cl_short*)data));
      return;
    case kUShort:
    case kUnsignedShort:
      sprintf(string, "%u", (int)*((cl_ushort*)data));
      return;
    case kInt:
      sprintf(string, "%d", *((cl_int*)data));
      return;
    case kUInt:
    case kUnsignedInt:
      sprintf(string, "%u", *((cl_uint*)data));
      return;
    case kLong:
      sprintf(string, "%lld", *((cl_long*)data));
      return;
    case kULong:
    case kUnsignedLong:
      sprintf(string, "%llu", *((cl_ulong*)data));
      return;
    case kFloat:
      sprintf(string, "%f", *((cl_float*)data));
      return;
    case kHalf:
      sprintf(string, "half");
      return;
    case kDouble:
      sprintf(string, "%g", *((cl_double*)data));
      return;
    default:
      sprintf(string, "INVALID");
      return;
  }

}

size_t get_explicit_type_size( ExplicitType type )
{
    /* Quick method to avoid branching: make sure the following array matches the Enum order */
    static size_t    sExplicitTypeSizes[] = {
            sizeof( cl_bool ),
            sizeof( cl_char ),
            sizeof( cl_uchar ),
            sizeof( cl_uchar ),
            sizeof( cl_short ),
            sizeof( cl_ushort ),
            sizeof( cl_ushort ),
            sizeof( cl_int ),
            sizeof( cl_uint ),
            sizeof( cl_uint ),
            sizeof( cl_long ),
            sizeof( cl_ulong ),
            sizeof( cl_ulong ),
            sizeof( cl_float ),
            sizeof( cl_half ),
            sizeof( cl_double )
        };

    return sExplicitTypeSizes[ type ];
}

const char * get_explicit_type_name( ExplicitType type )
{
    /* Quick method to avoid branching: make sure the following array matches the Enum order */
    static const char *sExplicitTypeNames[] = { "bool", "char", "uchar", "unsigned char", "short", "ushort", "unsigned short", "int",
                            "uint", "unsigned int", "long", "ulong", "unsigned long", "float", "half", "double" };

    return sExplicitTypeNames[ type ];
}

static long lrintf_clamped( float f );
static long lrintf_clamped( float f )
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

static long lrint_clamped( double f );
static long lrint_clamped( double f )
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
#if defined( __SSE2__ ) || (defined (_MSC_VER))
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


typedef cl_long Long;
typedef cl_ulong ULong;

static ULong sUpperLimits[ kNumExplicitTypes ] =
    {
        0,
        127, 255, 255,
        32767, 65535, 65535,
        0x7fffffffLL, 0xffffffffLL, 0xffffffffLL,
        0x7fffffffffffffffLL, 0xffffffffffffffffLL, 0xffffffffffffffffLL,
        0, 0 };    // Last two values aren't stored here

static Long sLowerLimits[ kNumExplicitTypes ] =
    {
        -1,
        -128, 0, 0,
        -32768, 0, 0,
        0xffffffff80000000LL, 0, 0,
        0x8000000000000000LL, 0, 0,
        0, 0 };    // Last two values aren't stored here

#define BOOL_CASE(inType) \
        case kBool:    \
            boolPtr = (bool *)outRaw; \
            *boolPtr = ( *inType##Ptr ) != 0 ? true : false; \
            break;

#define SIMPLE_CAST_CASE(inType,outEnum,outType) \
        case outEnum:                                \
            outType##Ptr = (outType *)outRaw;        \
            *outType##Ptr = (outType)(*inType##Ptr);    \
            break;

// Sadly, the ULong downcasting cases need a separate #define to get rid of signed/unsigned comparison warnings
#define DOWN_CAST_CASE(inType,outEnum,outType,sat) \
        case outEnum:                                \
            outType##Ptr = (outType *)outRaw;        \
            if( sat )                                \
            {                                        \
                if( ( sLowerLimits[outEnum] < 0 && *inType##Ptr > (Long)sUpperLimits[outEnum] ) || ( sLowerLimits[outEnum] == 0 && (ULong)*inType##Ptr > sUpperLimits[outEnum] ) )\
                    *outType##Ptr = (outType)sUpperLimits[outEnum];\
                else if( *inType##Ptr < sLowerLimits[outEnum] )\
                    *outType##Ptr = (outType)sLowerLimits[outEnum]; \
                else                                            \
                    *outType##Ptr = (outType)*inType##Ptr;    \
            } else {                                \
                *outType##Ptr = (outType)( *inType##Ptr & ( 0xffffffffffffffffLL >> ( 64 - ( sizeof( outType ) * 8 ) ) ) ); \
            }                                        \
            break;

#define U_DOWN_CAST_CASE(inType,outEnum,outType,sat) \
        case outEnum:                                \
            outType##Ptr = (outType *)outRaw;        \
            if( sat )                                \
            {                                        \
                if( (ULong)*inType##Ptr > sUpperLimits[outEnum] )\
                    *outType##Ptr = (outType)sUpperLimits[outEnum];\
                else                                            \
                    *outType##Ptr = (outType)*inType##Ptr;    \
            } else {                                \
                *outType##Ptr = (outType)( *inType##Ptr & ( 0xffffffffffffffffLL >> ( 64 - ( sizeof( outType ) * 8 ) ) ) ); \
            }                                        \
            break;

#define TO_FLOAT_CASE(inType)                \
        case kFloat:                        \
            floatPtr = (float *)outRaw;        \
            *floatPtr = (float)(*inType##Ptr);    \
            break;
#define TO_DOUBLE_CASE(inType)                \
        case kDouble:                        \
            doublePtr = (double *)outRaw;        \
            *doublePtr = (double)(*inType##Ptr);    \
            break;


/* Note: we use lrintf here to force the rounding instead of whatever the processor's current rounding mode is */
#define FLOAT_ROUND_TO_NEAREST_CASE(outEnum,outType)    \
        case outEnum:                                    \
            outType##Ptr = (outType *)outRaw;            \
            *outType##Ptr = (outType)lrintf_clamped( *floatPtr );    \
            break;

#define FLOAT_ROUND_CASE(outEnum,outType,rounding,sat)    \
        case outEnum:                                    \
        {                                                \
            outType##Ptr = (outType *)outRaw;            \
            /* Get the tens digit */                    \
            Long wholeValue = (Long)*floatPtr;\
            float largeRemainder = ( *floatPtr - (float)wholeValue ) * 10.f; \
            /* What do we do based on that? */                \
            if( rounding == kRoundToEven )                    \
            {                                                \
                if( wholeValue & 1LL )    /*between 1 and 1.99 */    \
                    wholeValue += 1LL;    /* round up to even */  \
            }                                                \
            else if( rounding == kRoundToZero )                \
            {                                                \
                /* Nothing to do, round-to-zero is what C casting does */                            \
            }                                                \
            else if( rounding == kRoundToPosInf )            \
            {                                                \
                /* Only positive numbers are wrong */        \
                if( largeRemainder != 0.f && wholeValue >= 0 )    \
                    wholeValue++;                            \
            }                                                \
            else if( rounding == kRoundToNegInf )            \
            {                                                \
                /* Only negative numbers are off */            \
                if( largeRemainder != 0.f && wholeValue < 0 ) \
                    wholeValue--;                            \
            }                                                \
            else                                            \
            {   /* Default is round-to-nearest */            \
                wholeValue = (Long)lrintf_clamped( *floatPtr );    \
            }                                                \
            /* Now apply saturation rules */                \
            if( sat )                                \
            {                                        \
                if( ( sLowerLimits[outEnum] < 0 && wholeValue > (Long)sUpperLimits[outEnum] ) || ( sLowerLimits[outEnum] == 0 && (ULong)wholeValue > sUpperLimits[outEnum] ) )\
                    *outType##Ptr = (outType)sUpperLimits[outEnum];\
                else if( wholeValue < sLowerLimits[outEnum] )\
                    *outType##Ptr = (outType)sLowerLimits[outEnum]; \
                else                                            \
                    *outType##Ptr = (outType)wholeValue;    \
            } else {                                \
                *outType##Ptr = (outType)( wholeValue & ( 0xffffffffffffffffLL >> ( 64 - ( sizeof( outType ) * 8 ) ) ) ); \
            }                                        \
        }                \
        break;

#define DOUBLE_ROUND_CASE(outEnum,outType,rounding,sat)    \
        case outEnum:                                    \
        {                                                \
            outType##Ptr = (outType *)outRaw;            \
            /* Get the tens digit */                    \
            Long wholeValue = (Long)*doublePtr;\
            double largeRemainder = ( *doublePtr - (double)wholeValue ) * 10.0; \
            /* What do we do based on that? */                \
            if( rounding == kRoundToEven )                    \
            {                                                \
                if( wholeValue & 1LL )    /*between 1 and 1.99 */    \
                    wholeValue += 1LL;    /* round up to even */  \
            }                                                \
            else if( rounding == kRoundToZero )                \
            {                                                \
                /* Nothing to do, round-to-zero is what C casting does */                            \
            }                                                \
            else if( rounding == kRoundToPosInf )            \
            {                                                \
                /* Only positive numbers are wrong */        \
                if( largeRemainder != 0.0 && wholeValue >= 0 )    \
                    wholeValue++;                            \
            }                                                \
            else if( rounding == kRoundToNegInf )            \
            {                                                \
                /* Only negative numbers are off */            \
                if( largeRemainder != 0.0 && wholeValue < 0 ) \
                    wholeValue--;                            \
            }                                                \
            else                                            \
            {   /* Default is round-to-nearest */            \
                wholeValue = (Long)lrint_clamped( *doublePtr );    \
            }                                                \
            /* Now apply saturation rules */                \
            if( sat )                                \
            {                                        \
                if( ( sLowerLimits[outEnum] < 0 && wholeValue > (Long)sUpperLimits[outEnum] ) || ( sLowerLimits[outEnum] == 0 && (ULong)wholeValue > sUpperLimits[outEnum] ) )\
                    *outType##Ptr = (outType)sUpperLimits[outEnum];\
                else if( wholeValue < sLowerLimits[outEnum] )\
                    *outType##Ptr = (outType)sLowerLimits[outEnum]; \
                else                                            \
                    *outType##Ptr = (outType)wholeValue;    \
            } else {                                \
                *outType##Ptr = (outType)( wholeValue & ( 0xffffffffffffffffLL >> ( 64 - ( sizeof( outType ) * 8 ) ) ) ); \
            }                                        \
        }                \
        break;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

void convert_explicit_value( void *inRaw, void *outRaw, ExplicitType inType, bool saturate, RoundingType roundType, ExplicitType outType )
{
    bool *boolPtr;
    char *charPtr;
    uchar *ucharPtr;
    short *shortPtr;
    ushort *ushortPtr;
    int *intPtr;
    uint *uintPtr;
    Long *LongPtr;
    ULong *ULongPtr;
    float *floatPtr;
    double *doublePtr;


    switch( inType )
    {
        case kBool:
            boolPtr = (bool *)inRaw;
            switch( outType )
            {
                case kBool:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                case kChar:
                case kUChar:
                case kUnsignedChar:
                case kShort:
                case kUShort:
                case kUnsignedShort:
                case kInt:
                case kUInt:
                case kUnsignedInt:
                case kLong:
                case kULong:
                case kUnsignedLong:
                    memset( outRaw, *boolPtr ? 0xff : 0, get_explicit_type_size( outType ) );
                    break;

                case kFloat:
                    floatPtr = (float *)outRaw;
                    *floatPtr = ( *boolPtr ) ? -1.f : 0.f;
                    break;
                case kDouble:
                    doublePtr = (double *)outRaw;
                    *doublePtr = ( *boolPtr ) ? -1.0 : 0.0;
                    break;
                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kChar:
            charPtr = (char *)inRaw;
            switch( outType )
            {
                BOOL_CASE(char)

                case kChar:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(char,kUChar,uchar,saturate)
                SIMPLE_CAST_CASE(char,kUnsignedChar,uchar)
                SIMPLE_CAST_CASE(char,kShort,short)
                SIMPLE_CAST_CASE(char,kUShort,ushort)
                SIMPLE_CAST_CASE(char,kUnsignedShort,ushort)
                SIMPLE_CAST_CASE(char,kInt,int)
                SIMPLE_CAST_CASE(char,kUInt,uint)
                SIMPLE_CAST_CASE(char,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(char,kLong,Long)
                SIMPLE_CAST_CASE(char,kULong,ULong)
                SIMPLE_CAST_CASE(char,kUnsignedLong,ULong)

                TO_FLOAT_CASE(char)
                TO_DOUBLE_CASE(char)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUChar:
            ucharPtr = (uchar *)inRaw;
            switch( outType )
            {
                BOOL_CASE(uchar)

                case kUChar:
                case kUnsignedChar:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(uchar,kChar,char,saturate)
                SIMPLE_CAST_CASE(uchar,kShort,short)
                SIMPLE_CAST_CASE(uchar,kUShort,ushort)
                SIMPLE_CAST_CASE(uchar,kUnsignedShort,ushort)
                SIMPLE_CAST_CASE(uchar,kInt,int)
                SIMPLE_CAST_CASE(uchar,kUInt,uint)
                SIMPLE_CAST_CASE(uchar,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(uchar,kLong,Long)
                SIMPLE_CAST_CASE(uchar,kULong,ULong)
                SIMPLE_CAST_CASE(uchar,kUnsignedLong,ULong)

                TO_FLOAT_CASE(uchar)
                TO_DOUBLE_CASE(uchar)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUnsignedChar:
            ucharPtr = (uchar *)inRaw;
            switch( outType )
            {
                BOOL_CASE(uchar)

                case kUChar:
                case kUnsignedChar:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(uchar,kChar,char,saturate)
                SIMPLE_CAST_CASE(uchar,kShort,short)
                SIMPLE_CAST_CASE(uchar,kUShort,ushort)
                SIMPLE_CAST_CASE(uchar,kUnsignedShort,ushort)
                SIMPLE_CAST_CASE(uchar,kInt,int)
                SIMPLE_CAST_CASE(uchar,kUInt,uint)
                SIMPLE_CAST_CASE(uchar,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(uchar,kLong,Long)
                SIMPLE_CAST_CASE(uchar,kULong,ULong)
                SIMPLE_CAST_CASE(uchar,kUnsignedLong,ULong)

                TO_FLOAT_CASE(uchar)
                TO_DOUBLE_CASE(uchar)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kShort:
            shortPtr = (short *)inRaw;
            switch( outType )
            {
                BOOL_CASE(short)

                case kShort:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(short,kChar,char,saturate)
                DOWN_CAST_CASE(short,kUChar,uchar,saturate)
                DOWN_CAST_CASE(short,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(short,kUShort,ushort,saturate)
                DOWN_CAST_CASE(short,kUnsignedShort,ushort,saturate)
                SIMPLE_CAST_CASE(short,kInt,int)
                SIMPLE_CAST_CASE(short,kUInt,uint)
                SIMPLE_CAST_CASE(short,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(short,kLong,Long)
                SIMPLE_CAST_CASE(short,kULong,ULong)
                SIMPLE_CAST_CASE(short,kUnsignedLong,ULong)

                TO_FLOAT_CASE(short)
                TO_DOUBLE_CASE(short)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUShort:
            ushortPtr = (ushort *)inRaw;
            switch( outType )
            {
                BOOL_CASE(ushort)

                case kUShort:
                case kUnsignedShort:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(ushort,kChar,char,saturate)
                DOWN_CAST_CASE(ushort,kUChar,uchar,saturate)
                DOWN_CAST_CASE(ushort,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(ushort,kShort,short,saturate)
                SIMPLE_CAST_CASE(ushort,kInt,int)
                SIMPLE_CAST_CASE(ushort,kUInt,uint)
                SIMPLE_CAST_CASE(ushort,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(ushort,kLong,Long)
                SIMPLE_CAST_CASE(ushort,kULong,ULong)
                SIMPLE_CAST_CASE(ushort,kUnsignedLong,ULong)

                TO_FLOAT_CASE(ushort)
                TO_DOUBLE_CASE(ushort)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUnsignedShort:
            ushortPtr = (ushort *)inRaw;
            switch( outType )
            {
                BOOL_CASE(ushort)

                case kUShort:
                case kUnsignedShort:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(ushort,kChar,char,saturate)
                DOWN_CAST_CASE(ushort,kUChar,uchar,saturate)
                DOWN_CAST_CASE(ushort,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(ushort,kShort,short,saturate)
                SIMPLE_CAST_CASE(ushort,kInt,int)
                SIMPLE_CAST_CASE(ushort,kUInt,uint)
                SIMPLE_CAST_CASE(ushort,kUnsignedInt,uint)
                SIMPLE_CAST_CASE(ushort,kLong,Long)
                SIMPLE_CAST_CASE(ushort,kULong,ULong)
                SIMPLE_CAST_CASE(ushort,kUnsignedLong,ULong)

                TO_FLOAT_CASE(ushort)
                TO_DOUBLE_CASE(ushort)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kInt:
            intPtr = (int *)inRaw;
            switch( outType )
            {
                BOOL_CASE(int)

                case kInt:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(int,kChar,char,saturate)
                DOWN_CAST_CASE(int,kUChar,uchar,saturate)
                DOWN_CAST_CASE(int,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(int,kShort,short,saturate)
                DOWN_CAST_CASE(int,kUShort,ushort,saturate)
                DOWN_CAST_CASE(int,kUnsignedShort,ushort,saturate)
                DOWN_CAST_CASE(int,kUInt,uint,saturate)
                DOWN_CAST_CASE(int,kUnsignedInt,uint,saturate)
                SIMPLE_CAST_CASE(int,kLong,Long)
                SIMPLE_CAST_CASE(int,kULong,ULong)
                SIMPLE_CAST_CASE(int,kUnsignedLong,ULong)

                TO_FLOAT_CASE(int)
                TO_DOUBLE_CASE(int)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUInt:
            uintPtr = (uint *)inRaw;
            switch( outType )
            {
                BOOL_CASE(uint)

                case kUInt:
                case kUnsignedInt:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(uint,kChar,char,saturate)
                DOWN_CAST_CASE(uint,kUChar,uchar,saturate)
                DOWN_CAST_CASE(uint,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(uint,kShort,short,saturate)
                DOWN_CAST_CASE(uint,kUShort,ushort,saturate)
                DOWN_CAST_CASE(uint,kUnsignedShort,ushort,saturate)
                DOWN_CAST_CASE(uint,kInt,int,saturate)
                SIMPLE_CAST_CASE(uint,kLong,Long)
                SIMPLE_CAST_CASE(uint,kULong,ULong)
                SIMPLE_CAST_CASE(uint,kUnsignedLong,ULong)

                TO_FLOAT_CASE(uint)
                TO_DOUBLE_CASE(uint)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUnsignedInt:
            uintPtr = (uint *)inRaw;
            switch( outType )
            {
                BOOL_CASE(uint)

                case kUInt:
                case kUnsignedInt:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(uint,kChar,char,saturate)
                DOWN_CAST_CASE(uint,kUChar,uchar,saturate)
                DOWN_CAST_CASE(uint,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(uint,kShort,short,saturate)
                DOWN_CAST_CASE(uint,kUShort,ushort,saturate)
                DOWN_CAST_CASE(uint,kUnsignedShort,ushort,saturate)
                DOWN_CAST_CASE(uint,kInt,int,saturate)
                SIMPLE_CAST_CASE(uint,kLong,Long)
                SIMPLE_CAST_CASE(uint,kULong,ULong)
                SIMPLE_CAST_CASE(uint,kUnsignedLong,ULong)

                TO_FLOAT_CASE(uint)
                TO_DOUBLE_CASE(uint)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kLong:
            LongPtr = (Long *)inRaw;
            switch( outType )
            {
                BOOL_CASE(Long)

                case kLong:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                DOWN_CAST_CASE(Long,kChar,char,saturate)
                DOWN_CAST_CASE(Long,kUChar,uchar,saturate)
                DOWN_CAST_CASE(Long,kUnsignedChar,uchar,saturate)
                DOWN_CAST_CASE(Long,kShort,short,saturate)
                DOWN_CAST_CASE(Long,kUShort,ushort,saturate)
                DOWN_CAST_CASE(Long,kUnsignedShort,ushort,saturate)
                DOWN_CAST_CASE(Long,kInt,int,saturate)
                DOWN_CAST_CASE(Long,kUInt,uint,saturate)
                DOWN_CAST_CASE(Long,kUnsignedInt,uint,saturate)
                DOWN_CAST_CASE(Long,kULong,ULong,saturate)
                DOWN_CAST_CASE(Long,kUnsignedLong,ULong,saturate)

                TO_FLOAT_CASE(Long)
                TO_DOUBLE_CASE(Long)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kULong:
            ULongPtr = (ULong *)inRaw;
            switch( outType )
            {
                BOOL_CASE(ULong)

                case kUnsignedLong:
                case kULong:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                U_DOWN_CAST_CASE(ULong,kChar,char,saturate)
                U_DOWN_CAST_CASE(ULong,kUChar,uchar,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedChar,uchar,saturate)
                U_DOWN_CAST_CASE(ULong,kShort,short,saturate)
                U_DOWN_CAST_CASE(ULong,kUShort,ushort,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedShort,ushort,saturate)
                U_DOWN_CAST_CASE(ULong,kInt,int,saturate)
                U_DOWN_CAST_CASE(ULong,kUInt,uint,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedInt,uint,saturate)
                U_DOWN_CAST_CASE(ULong,kLong,Long,saturate)

                TO_FLOAT_CASE(ULong)
                TO_DOUBLE_CASE(ULong)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kUnsignedLong:
            ULongPtr = (ULong *)inRaw;
            switch( outType )
            {
                BOOL_CASE(ULong)

                case kULong:
                case kUnsignedLong:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                U_DOWN_CAST_CASE(ULong,kChar,char,saturate)
                U_DOWN_CAST_CASE(ULong,kUChar,uchar,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedChar,uchar,saturate)
                U_DOWN_CAST_CASE(ULong,kShort,short,saturate)
                U_DOWN_CAST_CASE(ULong,kUShort,ushort,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedShort,ushort,saturate)
                U_DOWN_CAST_CASE(ULong,kInt,int,saturate)
                U_DOWN_CAST_CASE(ULong,kUInt,uint,saturate)
                U_DOWN_CAST_CASE(ULong,kUnsignedInt,uint,saturate)
                U_DOWN_CAST_CASE(ULong,kLong,Long,saturate)

                TO_FLOAT_CASE(ULong)
                TO_DOUBLE_CASE(ULong)

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kFloat:
            floatPtr = (float *)inRaw;
            switch( outType )
            {
                BOOL_CASE(float)

                FLOAT_ROUND_CASE(kChar,char,roundType,saturate)
                FLOAT_ROUND_CASE(kUChar,uchar,roundType,saturate)
                FLOAT_ROUND_CASE(kUnsignedChar,uchar,roundType,saturate)
                FLOAT_ROUND_CASE(kShort,short,roundType,saturate)
                FLOAT_ROUND_CASE(kUShort,ushort,roundType,saturate)
                FLOAT_ROUND_CASE(kUnsignedShort,ushort,roundType,saturate)
                FLOAT_ROUND_CASE(kInt,int,roundType,saturate)
                FLOAT_ROUND_CASE(kUInt,uint,roundType,saturate)
                FLOAT_ROUND_CASE(kUnsignedInt,uint,roundType,saturate)
                FLOAT_ROUND_CASE(kLong,Long,roundType,saturate)
                FLOAT_ROUND_CASE(kULong,ULong,roundType,saturate)
                FLOAT_ROUND_CASE(kUnsignedLong,ULong,roundType,saturate)

                case kFloat:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                TO_DOUBLE_CASE(float);

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        case kDouble:
            doublePtr = (double *)inRaw;
            switch( outType )
            {
                BOOL_CASE(double)

                DOUBLE_ROUND_CASE(kChar,char,roundType,saturate)
                DOUBLE_ROUND_CASE(kUChar,uchar,roundType,saturate)
                DOUBLE_ROUND_CASE(kUnsignedChar,uchar,roundType,saturate)
                DOUBLE_ROUND_CASE(kShort,short,roundType,saturate)
                DOUBLE_ROUND_CASE(kUShort,ushort,roundType,saturate)
                DOUBLE_ROUND_CASE(kUnsignedShort,ushort,roundType,saturate)
                DOUBLE_ROUND_CASE(kInt,int,roundType,saturate)
                DOUBLE_ROUND_CASE(kUInt,uint,roundType,saturate)
                DOUBLE_ROUND_CASE(kUnsignedInt,uint,roundType,saturate)
                DOUBLE_ROUND_CASE(kLong,Long,roundType,saturate)
                DOUBLE_ROUND_CASE(kULong,ULong,roundType,saturate)
                DOUBLE_ROUND_CASE(kUnsignedLong,ULong,roundType,saturate)

                TO_FLOAT_CASE(double);

                case kDouble:
                    memcpy( outRaw, inRaw, get_explicit_type_size( inType ) );
                    break;

                default:
                    log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
                    break;
            }
            break;

        default:
            log_error( "ERROR: Invalid type given to convert_explicit_value!!\n" );
            break;
    }
}

void generate_random_data( ExplicitType type, size_t count, MTdata d, void *outData )
{
    bool *boolPtr;
    cl_char *charPtr;
    cl_uchar *ucharPtr;
    cl_short *shortPtr;
    cl_ushort *ushortPtr;
    cl_int *intPtr;
    cl_uint *uintPtr;
    cl_long *longPtr;
    cl_ulong *ulongPtr;
    cl_float *floatPtr;
    cl_double *doublePtr;
    cl_ushort *halfPtr;
    size_t i;
    cl_uint bits = genrand_int32(d);
    cl_uint bitsLeft = 32;

    switch( type )
    {
        case kBool:
            boolPtr = (bool *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                boolPtr[i] = ( bits & 1 ) ? true : false;
                bits >>= 1; bitsLeft -= 1;
            }
            break;

        case kChar:
            charPtr = (cl_char *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                charPtr[i] = (cl_char)( (cl_int)(bits & 255 ) - 127 );
                bits >>= 8; bitsLeft -= 8;
            }
            break;

        case kUChar:
        case kUnsignedChar:
            ucharPtr = (cl_uchar *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                ucharPtr[i] = (cl_uchar)( bits & 255 );
                bits >>= 8; bitsLeft -= 8;
            }
            break;

        case kShort:
            shortPtr = (cl_short *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                shortPtr[i] = (cl_short)( (cl_int)( bits & 65535 ) - 32767 );
                bits >>= 16; bitsLeft -= 16;
            }
            break;

        case kUShort:
        case kUnsignedShort:
            ushortPtr = (cl_ushort *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                ushortPtr[i] = (cl_ushort)( (cl_int)( bits & 65535 ) );
                bits >>= 16; bitsLeft -= 16;
            }
            break;

        case kInt:
            intPtr = (cl_int *)outData;
            for( i = 0; i < count; i++ )
            {
                intPtr[i] = (cl_int)genrand_int32(d);
            }
            break;

        case kUInt:
        case kUnsignedInt:
            uintPtr = (cl_uint *)outData;
            for( i = 0; i < count; i++ )
            {
                uintPtr[i] = (unsigned int)genrand_int32(d);
            }
            break;

        case kLong:
            longPtr = (cl_long *)outData;
            for( i = 0; i < count; i++ )
            {
                longPtr[i] = (cl_long)genrand_int32(d) | ( (cl_long)genrand_int32(d) << 32 );
            }
            break;

        case kULong:
        case kUnsignedLong:
            ulongPtr = (cl_ulong *)outData;
            for( i = 0; i < count; i++ )
            {
                ulongPtr[i] = (cl_ulong)genrand_int32(d) | ( (cl_ulong)genrand_int32(d) << 32 );
            }
            break;

        case kFloat:
            floatPtr = (cl_float *)outData;
            for( i = 0; i < count; i++ )
            {
                // [ -(double) 0x7fffffff, (double) 0x7fffffff ]
                double t = genrand_real1(d);
                floatPtr[i] = (float) ((1.0 - t) * -(double) 0x7fffffff + t * (double) 0x7fffffff);
            }
            break;

        case kDouble:
            doublePtr = (cl_double *)outData;
            for( i = 0; i < count; i++ )
            {
                cl_long u = (cl_long)genrand_int32(d) | ( (cl_long)genrand_int32(d) << 32 );
                double t = (double) u;
                t *= MAKE_HEX_DOUBLE( 0x1.0p-32, 0x1, -32 );        // scale [-2**63, 2**63] to [-2**31, 2**31]
                doublePtr[i] = t;
            }
            break;

        case kHalf:
            halfPtr = (ushort *)outData;
            for( i = 0; i < count; i++ )
            {
                if( 0 == bitsLeft)
                {
                    bits = genrand_int32(d);
                    bitsLeft = 32;
                }
                halfPtr[i] = bits & 65535;     /* Kindly generates random bits for us */
                bits >>= 16; bitsLeft -= 16;
            }
            break;

        default:
            log_error( "ERROR: Invalid type passed in to generate_random_data!\n" );
            break;
    }
}

void * create_random_data( ExplicitType type, MTdata d, size_t count )
{
    void *data = malloc( get_explicit_type_size( type ) * count );
    generate_random_data( type, count, d, data );
    return data;
}

cl_long read_upscale_signed( void *inRaw, ExplicitType inType )
{
    switch( inType )
    {
        case kChar:
            return (cl_long)( *( (cl_char *)inRaw ) );
        case kUChar:
        case kUnsignedChar:
            return (cl_long)( *( (cl_uchar *)inRaw ) );
        case kShort:
            return (cl_long)( *( (cl_short *)inRaw ) );
        case kUShort:
        case kUnsignedShort:
            return (cl_long)( *( (cl_ushort *)inRaw ) );
        case kInt:
            return (cl_long)( *( (cl_int *)inRaw ) );
        case kUInt:
        case kUnsignedInt:
            return (cl_long)( *( (cl_uint *)inRaw ) );
        case kLong:
            return (cl_long)( *( (cl_long *)inRaw ) );
        case kULong:
        case kUnsignedLong:
            return (cl_long)( *( (cl_ulong *)inRaw ) );
        default:
            return 0;
    }
}

cl_ulong read_upscale_unsigned( void *inRaw, ExplicitType inType )
{
    switch( inType )
    {
        case kChar:
            return (cl_ulong)( *( (cl_char *)inRaw ) );
        case kUChar:
        case kUnsignedChar:
            return (cl_ulong)( *( (cl_uchar *)inRaw ) );
        case kShort:
            return (cl_ulong)( *( (cl_short *)inRaw ) );
        case kUShort:
        case kUnsignedShort:
            return (cl_ulong)( *( (cl_ushort *)inRaw ) );
        case kInt:
            return (cl_ulong)( *( (cl_int *)inRaw ) );
        case kUInt:
        case kUnsignedInt:
            return (cl_ulong)( *( (cl_uint *)inRaw ) );
        case kLong:
            return (cl_ulong)( *( (cl_long *)inRaw ) );
        case kULong:
        case kUnsignedLong:
            return (cl_ulong)( *( (cl_ulong *)inRaw ) );
        default:
            return 0;
    }
}

float read_as_float( void *inRaw, ExplicitType inType )
{
    switch( inType )
    {
        case kChar:
            return (float)( *( (cl_char *)inRaw ) );
        case kUChar:
        case kUnsignedChar:
            return (float)( *( (cl_char *)inRaw ) );
        case kShort:
            return (float)( *( (cl_short *)inRaw ) );
        case kUShort:
        case kUnsignedShort:
            return (float)( *( (cl_ushort *)inRaw ) );
        case kInt:
            return (float)( *( (cl_int *)inRaw ) );
        case kUInt:
        case kUnsignedInt:
            return (float)( *( (cl_uint *)inRaw ) );
        case kLong:
            return (float)( *( (cl_long *)inRaw ) );
        case kULong:
        case kUnsignedLong:
            return (float)( *( (cl_ulong *)inRaw ) );
        case kFloat:
            return *( (float *)inRaw );
        case kDouble:
            return (float) *( (double*)inRaw );
        default:
            return 0;
    }
}

float get_random_float(float low, float high, MTdata d)
{
    float t = (float)((double)genrand_int32(d) / (double)0xFFFFFFFF);
    return (1.0f - t) * low + t * high;
}

double get_random_double(double low, double high, MTdata d)
{
    cl_ulong u = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32 );
    double t = (double) u * MAKE_HEX_DOUBLE( 0x1.0p-64, 0x1, -64);
    return (1.0f - t) * low + t * high;
}

float  any_float( MTdata d )
{
    union
    {
        float   f;
        cl_uint u;
    }u;

    u.u = genrand_int32(d);
    return u.f;
}


double  any_double( MTdata d )
{
    union
    {
        double   f;
        cl_ulong u;
    }u;

    u.u = (cl_ulong) genrand_int32(d) | ((cl_ulong) genrand_int32(d) << 32);
    return u.f;
}

int          random_in_range( int minV, int maxV, MTdata d )
{
    cl_ulong r = ((cl_ulong) genrand_int32(d) ) * (maxV - minV + 1);
    return (cl_uint)(r >> 32) + minV;
}

size_t get_random_size_t(size_t low, size_t high, MTdata d)
{
  enum { N = sizeof(size_t)/sizeof(int) };

  union {
    int word[N];
    size_t size;
  } u;

  for (unsigned i=0; i != N; ++i) {
    u.word[i] = genrand_int32(d);
  }

  assert(low <= high && "Invalid random number range specified");
  size_t range = high - low;

  return (range) ? low + ((u.size - low) % range) : low;
}


