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
#ifndef __ROUNDING_MODE_H__
#define __ROUNDING_MODE_H__

#include "compat.h"

#if (defined(_WIN32) && defined(_MSC_VER))
#include "errorHelpers.h"
#include "testHarness.h"
#endif

typedef enum
{
    kDefaultRoundingMode = 0,
    kRoundToNearestEven,
    kRoundUp,
    kRoundDown,
    kRoundTowardZero,

    kRoundingModeCount
} RoundingMode;

typedef enum
{
    kuchar = 0,
    kchar = 1,
    kushort = 2,
    kshort = 3,
    kuint = 4,
    kint = 5,
    khalf = 6,
    kfloat = 7,
    kdouble = 8,
    kulong = 9,
    klong = 10,

    // This goes last
    kTypeCount
} Type;

extern RoundingMode set_round(RoundingMode r, Type outType);
extern RoundingMode get_round(void);
extern void *FlushToZero(void);
extern void UnFlushToZero(void *p);


#endif /* __ROUNDING_MODE_H__ */
