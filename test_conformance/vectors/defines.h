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
#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/threadTesting.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"


// 1,2,3,4,8,16 or
// 1,2,4,8,16,3
#define NUM_VECTOR_SIZES 6

extern int g_arrVecSizes[NUM_VECTOR_SIZES];
extern int g_arrVecSteps[NUM_VECTOR_SIZES];
extern bool g_wimpyMode;

extern const char * g_arrVecSizeNames[NUM_VECTOR_SIZES];
extern size_t g_arrVecAlignMasks[NUM_VECTOR_SIZES];

// Define the buffer size that we want to block our test with
#define BUFFER_SIZE (1024*1024)
#define KPAGESIZE 4096

extern ExplicitType types[];

extern const char *g_arrTypeNames[];
extern const size_t g_arrTypeSizes[];
