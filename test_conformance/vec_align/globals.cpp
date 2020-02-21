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
#include "defines.h"


// 1,2,3,4,8,16 or
// 1,2,4,8,16,3
int g_arrVecSizes[NUM_VECTOR_SIZES] = {1,2,3,4,8,16};
int g_arrVecSteps[NUM_VECTOR_SIZES] = {1,2,4,4,8,16};
const char * g_arrVecSizeNames[NUM_VECTOR_SIZES] = {"", "2","3","4","8","16"};
size_t g_arrVecAlignMasks[NUM_VECTOR_SIZES] = {(size_t)0,
                           (size_t)0x1, // 2
                           (size_t)0x3, // 3
                           (size_t)0x3, // 4
                           (size_t)0x7, // 8
                           (size_t)0xf // 16
};

bool g_wimpyMode = false;

ExplicitType types[] = { kChar, kUChar,
             kShort, kUShort,
             kInt, kUInt,
             kLong, kULong,
             kFloat, kDouble,
             kNumExplicitTypes };


const char *g_arrTypeNames[] =
    {
    "char",  "uchar",
    "short", "ushort",
    "int",   "uint",
    "long",  "ulong",
    "float", "double"
    };

extern const size_t g_arrTypeSizes[] =
    {
    1, 1,
    2, 2,
    4, 4,
    8, 8,
    4, 8
    };

