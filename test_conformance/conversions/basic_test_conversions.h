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
#ifndef BASIC_TEST_CONVERSIONS_H
#define BASIC_TEST_CONVERSIONS_H

#include "harness/compat.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "harness/errorHelpers.h"
#include "harness/rounding_mode.h"

#include <stdio.h>
#if defined( __APPLE__ )
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "harness/mt19937.h"

typedef void (*Convert)( void *dest, void *src, size_t );

#define kVectorSizeCount    6
#define kMaxVectorSize      16

typedef enum
{
    kUnsaturated = 0,
    kSaturated,

    kSaturationModeCount
}SaturationMode;

extern Convert gConversions[kTypeCount][kTypeCount];                // [dest format][source format]
extern Convert gSaturatedConversions[kTypeCount][kTypeCount];       // [dest format][source format]
extern const char *gTypeNames[ kTypeCount ];
extern const char *gRoundingModeNames[ kRoundingModeCount ];        // { "", "_rte", "_rtp", "_rtn", "_rtz" }
extern const char *gSaturationNames[ kSaturationModeCount ];        // { "", "_sat" }
extern const char *gVectorSizeNames[kVectorSizeCount];              // { "", "2", "4", "8", "16" }
extern size_t gTypeSizes[ kTypeCount ];

//Functions for clamping floating point numbers into the representable range for the type
typedef float (*clampf)( float );
typedef double (*clampd)( double );

extern clampf gClampFloat[ kTypeCount ][kRoundingModeCount];
extern clampd gClampDouble[ kTypeCount ][kRoundingModeCount];

typedef void (*InitDataFunc)( void *dest, SaturationMode, RoundingMode, Type destType, uint64_t start, int count, MTdata d );
extern InitDataFunc gInitFunctions[ kTypeCount ];

typedef int (*CheckResults)( void *out1, void *out2, void *allowZ, uint32_t count, int vectorSize );
extern CheckResults gCheckResults[ kTypeCount ];

#endif /* BASIC_TEST_CONVERSIONS_H */

