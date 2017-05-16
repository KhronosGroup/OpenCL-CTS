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
#ifndef CL_UTILS_H
#define  CL_UTILS_H

#if !defined (_WIN32)
#include <stdint.h>
#endif

#include <float.h>
#include <math.h>
#include <stdio.h>

#if !defined(_WIN32)
#include <stdbool.h>
#include <sys/param.h> 
#endif

#include "../../test_common/harness/compat.h"
#include "../../test_common/harness/errorHelpers.h"

#if defined (_WIN32)
#define MAX(x,y) ((x>y)?x:y);
#define MIN(x,y) ((x<y)?x:y);
#endif

#include "test_config.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#ifndef __cplusplus
#define true    1
#define false   0
#if defined(_WIN32)
    #define bool    int
#endif
#endif

extern void            *gIn_half;
extern void            *gOut_half;
extern void            *gOut_half_reference;
extern void            *gOut_half_reference_double;
extern void            *gIn_single;
extern void            *gOut_single;
extern void            *gOut_single_reference;
extern void            *gIn_double;
extern void            *gOut_double;
extern void            *gOut_double_reference;
extern cl_mem          gInBuffer_half;
extern cl_mem          gOutBuffer_half;
extern cl_mem          gInBuffer_single;
extern cl_mem          gOutBuffer_single;
extern cl_mem          gInBuffer_double;
extern cl_mem          gOutBuffer_double;

extern uint32_t		   gDeviceIndex;
extern cl_device_type  gDeviceType;
extern cl_device_id    gDevice;
extern cl_context      gContext;
extern cl_command_queue gQueue;
extern uint32_t        gDeviceFrequency;
extern uint32_t        gComputeDevices;
extern size_t          gMaxThreadGroupSize;
extern size_t          gWorkGroupSize;
extern int             gTestCount;
extern int             gFailCount;
extern int             gTestDouble;
extern int             gReportTimes;

// gWimpyMode indicates if we run the test in wimpy mode where we limit the
// size of 32 bit ranges to a much smaller set.  This is meant to be used
// as a smoke test
extern bool            gWimpyMode;

uint64_t ReadTime( void );
double SubtractTime( uint64_t endTime, uint64_t startTime );

cl_uint numVecs(cl_uint count, int vectorSizeIdx, bool aligned);
cl_uint runsOverBy(cl_uint count, int vectorSizeIdx, bool aligned);

void printSource(const char * src[], int len);

extern const char *vector_size_name_extensions[kVectorSizeCount+kStrangeVectorSizeCount];
extern const char *vector_size_strings[kVectorSizeCount+kStrangeVectorSizeCount];
extern const char *align_divisors[kVectorSizeCount+kStrangeVectorSizeCount];
extern const char *align_types[kVectorSizeCount+kStrangeVectorSizeCount];

int InitCL( void );
void ReleaseCL( void );
int RunKernel( cl_kernel kernel, void *inBuf, void *outBuf, uint32_t blockCount , int extraArg);
cl_program   MakeProgram( const char *source[], int count );

#if ! defined( __APPLE__ )
    extern void memset_pattern4(void *dest, const void *src_pattern, size_t bytes );
#endif

#define STRING( _x )    STRINGIFY( _x )
#define STRINGIFY(x)    #x

// used to convert a bucket of bits into a search pattern through double
static inline double DoubleFromUInt32( uint32_t bits );
static inline double DoubleFromUInt32( uint32_t bits )
{
    union{ uint64_t u; double d;} u;

    // split 0x89abcdef to 0x89abc00000000def
    u.u = bits & 0xffU;
    u.u |= (uint64_t) (bits & ~0xffU) << 32;
    
    // sign extend the leading bit of def segment as sign bit so that the middle region consists of either all 1s or 0s 
    u.u -= (bits & 0x80U) << 1;

    // return result
    return u.d;
}

static inline int IsHalfSubnormal( uint16_t x )
{ 
    return ((x&0x7fffU)-1U) < 0x03ffU; 
}

// prevent silent failures due to missing FLT_RADIX
#ifndef FLT_RADIX
    #error FLT_RADIX is not defined by float.h
#endif

static inline int IsFloatSubnormal( double x )
{ 
#if 2 == FLT_RADIX       
    // Do this in integer to avoid problems with FTZ behavior
    union{ float d; uint32_t u;}u;
    u.d = fabsf((float) x); 
    return (u.u-1) < 0x007fffffU;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware -- will fail if you flush subnormals to zero
    return fabs(x) < (double) FLT_MIN && x != 0.0; 
#endif
}

static inline int IsDoubleSubnormal( long double x )
{ 
#if 2 == FLT_RADIX       
    // Do this in integer to avoid problems with FTZ behavior
    union{ double d; uint64_t u;}u;
    u.d = fabs((double)x); 
    return (u.u-1) < 0x000fffffffffffffULL;
#else
    // rely on floating point hardware for non-radix2 non-IEEE-754 hardware -- will fail if you flush subnormals to zero
    return fabs(x) < (double) DBL_MIN && x != 0.0; 
#endif
}

extern int is_extension_available( cl_device_id device, const char *extensionName );

cl_device_fp_config get_default_rounding_mode( cl_device_id device );


#endif /* CL_UTILS_H */



