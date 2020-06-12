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
#ifndef _conversions_h
#define _conversions_h

#include "compat.h"

#include "errorHelpers.h"
#include "mt19937.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

/* Note: the next three all have to match in size and order!! */

enum ExplicitTypes
{
    kBool        = 0,
    kChar,
    kUChar,
    kUnsignedChar,
    kShort,
    kUShort,
    kUnsignedShort,
    kInt,
    kUInt,
    kUnsignedInt,
    kLong,
    kULong,
    kUnsignedLong,
    kFloat,
    kHalf,
    kDouble,
    kNumExplicitTypes
};

typedef enum ExplicitTypes    ExplicitType;

enum RoundingTypes
{
    kRoundToEven = 0,
    kRoundToZero,
    kRoundToPosInf,
    kRoundToNegInf,
    kRoundToNearest,

    kNumRoundingTypes,

    kDefaultRoundingType = kRoundToNearest
};

typedef enum RoundingTypes    RoundingType;

extern void             print_type_to_string(ExplicitType type, void *data, char* string);
extern size_t           get_explicit_type_size( ExplicitType type );
extern const char *     get_explicit_type_name( ExplicitType type );
extern void             convert_explicit_value( void *inRaw, void *outRaw, ExplicitType inType, bool saturate, RoundingType roundType, ExplicitType outType );

extern void             generate_random_data( ExplicitType type, size_t count, MTdata d, void *outData );
extern void    *         create_random_data( ExplicitType type, MTdata d, size_t count );

extern cl_long          read_upscale_signed( void *inRaw, ExplicitType inType );
extern cl_ulong         read_upscale_unsigned( void *inRaw, ExplicitType inType );
extern float            read_as_float( void *inRaw, ExplicitType inType );

extern float            get_random_float(float low, float high, MTdata d);
extern double           get_random_double(double low, double high, MTdata d);
extern float            any_float( MTdata d );
extern double           any_double( MTdata d );

extern int              random_in_range( int minV, int maxV, MTdata d );

size_t get_random_size_t(size_t low, size_t high, MTdata d);

// Note: though this takes a double, this is for use with single precision tests
static inline int IsFloatSubnormal( float x )
{
#if 2 == FLT_RADIX
    // Do this in integer to avoid problems with FTZ behavior
    union{ float d; uint32_t u;}u;
    u.d = fabsf(x);
    return (u.u-1) < 0x007fffffU;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware -- will fail if you flush subnormals to zero
    return fabs(x) < (double) FLT_MIN && x != 0.0;
#endif
}

static inline int IsDoubleSubnormal( double x )
{
#if 2 == FLT_RADIX
    // Do this in integer to avoid problems with FTZ behavior
    union{ double d; uint64_t u;}u;
    u.d = fabs( x);
    return (u.u-1) < 0x000fffffffffffffULL;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware -- will fail if you flush subnormals to zero
    return fabs(x) < (double) DBL_MIN && x != 0.0;
#endif
}

static inline int IsHalfSubnormal( cl_half x )
{ 
    return ( ( x & 0x7fffU ) - 1U ) < 0x03ffU; 
}

#endif // _conversions_h


