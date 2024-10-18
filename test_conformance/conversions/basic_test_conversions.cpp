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
#include "harness/testHarness.h"
#include "harness/compat.h"
#include "harness/ThreadPool.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach_time.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

#if defined(__MINGW32__)
#include <sys/param.h>
#endif

#include <sstream>
#include <stdarg.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include <algorithm>

#include <vector>
#include <type_traits>
#include <cmath>

#include "basic_test_conversions.h"

#if defined(_WIN32)
#include <mmintrin.h>
#include <emmintrin.h>
#else // !_WIN32
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#endif // _WIN32

cl_context gContext = NULL;
cl_command_queue gQueue = NULL;
int gStartTestNumber = -1;
int gEndTestNumber = 0;
#if defined(__APPLE__)
int gTimeResults = 1;
#else
int gTimeResults = 0;
#endif
int gReportAverageTimes = 0;
void *gIn = NULL;
void *gRef = NULL;
void *gAllowZ = NULL;
void *gOut[kCallStyleCount] = { NULL };
cl_mem gInBuffer;
cl_mem gOutBuffers[kCallStyleCount];
size_t gComputeDevices = 0;
uint32_t gDeviceFrequency = 0;
int gWimpyMode = 0;
int gWimpyReductionFactor = 128;
int gSkipTesting = 0;
int gForceFTZ = 0;
int gIsRTZ = 0;
int gForceHalfFTZ = 0;
int gIsHalfRTZ = 0;
uint32_t gSimdSize = 1;
int gHasDouble = 0;
int gTestDouble = 1;
int gHasHalfs = 0;
int gTestHalfs = 1;
const char *sizeNames[] = { "", "", "2", "3", "4", "8", "16" };
int vectorSizes[] = { 1, 1, 2, 3, 4, 8, 16 };
int gMinVectorSize = 0;
int gMaxVectorSize = sizeof(vectorSizes) / sizeof(vectorSizes[0]);
MTdata gMTdata;
const char **argList = NULL;
int argCount = 0;


double SubtractTime(uint64_t endTime, uint64_t startTime);

cl_half_rounding_mode DataInitInfo::halfRoundingMode = CL_HALF_RTE;
cl_half_rounding_mode ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTE;

// clang-format off
// for readability sake keep this section unformatted

std::vector<unsigned int> DataInitInfo::specialValuesUInt = {
      uint32_t(INT_MIN), uint32_t(INT_MIN + 1), uint32_t(INT_MIN + 2),
      uint32_t(-(1 << 30) - 3), uint32_t(-(1 << 30) - 2), uint32_t(-(1 << 30) - 1), uint32_t(-(1 << 30)),
      uint32_t(-(1 << 30) + 1), uint32_t(-(1 << 30) + 2), uint32_t(-(1 << 30) + 3),
      uint32_t(-(1 << 24) - 3), uint32_t(-(1 << 24) - 2),uint32_t(-(1 << 24) - 1),
      uint32_t(-(1 << 24)), uint32_t(-(1 << 24) + 1), uint32_t(-(1 << 24) + 2), uint32_t(-(1 << 24) + 3),
      uint32_t(-(1 << 23) - 3), uint32_t(-(1 << 23) - 2),uint32_t(-(1 << 23) - 1),
      uint32_t(-(1 << 23)), uint32_t(-(1 << 23) + 1), uint32_t(-(1 << 23) + 2), uint32_t(-(1 << 23) + 3),
      uint32_t(-(1 << 22) - 3), uint32_t(-(1 << 22) - 2),uint32_t(-(1 << 22) - 1),
      uint32_t(-(1 << 22)), uint32_t(-(1 << 22) + 1), uint32_t(-(1 << 22) + 2), uint32_t(-(1 << 22) + 3),
      uint32_t(-(1 << 21) - 3), uint32_t(-(1 << 21) - 2),uint32_t(-(1 << 21) - 1),
      uint32_t(-(1 << 21)), uint32_t(-(1 << 21) + 1), uint32_t(-(1 << 21) + 2), uint32_t(-(1 << 21) + 3),
      uint32_t(-(1 << 16) - 3), uint32_t(-(1 << 16) - 2),uint32_t(-(1 << 16) - 1),
      uint32_t(-(1 << 16)), uint32_t(-(1 << 16) + 1), uint32_t(-(1 << 16) + 2), uint32_t(-(1 << 16) + 3),
      uint32_t(-(1 << 15) - 3), uint32_t(-(1 << 15) - 2),uint32_t(-(1 << 15) - 1),
      uint32_t(-(1 << 15)), uint32_t(-(1 << 15) + 1), uint32_t(-(1 << 15) + 2), uint32_t(-(1 << 15) + 3),
      uint32_t(-(1 << 8) - 3), uint32_t(-(1 << 8) - 2),uint32_t(-(1 << 8) - 1),
      uint32_t(-(1 << 8)), uint32_t(-(1 << 8) + 1), uint32_t(-(1 << 8) + 2), uint32_t(-(1 << 8) + 3),
      uint32_t(-(1 << 7) - 3), uint32_t(-(1 << 7) - 2),uint32_t(-(1 << 7) - 1),
      uint32_t(-(1 << 7)), uint32_t(-(1 << 7) + 1), uint32_t(-(1 << 7) + 2), uint32_t(-(1 << 7) + 3),
      uint32_t(-4), uint32_t(-3), uint32_t(-2), uint32_t(-1), 0, 1, 2, 3, 4,
      (1 << 7) - 3,(1 << 7) - 2,(1 << 7) - 1, (1 << 7), (1 << 7) + 1, (1 << 7) + 2, (1 << 7) + 3,
      (1 << 8) - 3,(1 << 8) - 2,(1 << 8) - 1, (1 << 8), (1 << 8) + 1, (1 << 8) + 2, (1 << 8) + 3,
      (1 << 15) - 3,(1 << 15) - 2,(1 << 15) - 1, (1 << 15), (1 << 15) + 1, (1 << 15) + 2, (1 << 15) + 3,
      (1 << 16) - 3,(1 << 16) - 2,(1 << 16) - 1, (1 << 16), (1 << 16) + 1, (1 << 16) + 2, (1 << 16) + 3,
      (1 << 21) - 3,(1 << 21) - 2,(1 << 21) - 1, (1 << 21), (1 << 21) + 1, (1 << 21) + 2, (1 << 21) + 3,
      (1 << 22) - 3,(1 << 22) - 2,(1 << 22) - 1, (1 << 22), (1 << 22) + 1, (1 << 22) + 2, (1 << 22) + 3,
      (1 << 23) - 3,(1 << 23) - 2,(1 << 23) - 1, (1 << 23), (1 << 23) + 1, (1 << 23) + 2, (1 << 23) + 3,
      (1 << 24) - 3,(1 << 24) - 2,(1 << 24) - 1, (1 << 24), (1 << 24) + 1, (1 << 24) + 2, (1 << 24) + 3,
      (1 << 30) - 3,(1 << 30) - 2,(1 << 30) - 1, (1 << 30), (1 << 30) + 1, (1 << 30) + 2, (1 << 30) + 3,
      INT_MAX - 3, INT_MAX - 2, INT_MAX - 1, INT_MAX, // 0x80000000, 0x80000001 0x80000002 already covered above
      UINT_MAX - 3, UINT_MAX - 2, UINT_MAX - 1, UINT_MAX
};

std::vector<float> DataInitInfo::specialValuesFloat = {
    -NAN, -INFINITY, -FLT_MAX,
    MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40), MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64), MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),
    MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39), MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63), MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8), MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32), MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7),
    MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7), MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31), MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6),
    -1000.f, -100.f, -4.0f, -3.5f, -3.0f,
    MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23), -2.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23), -2.0f,
    MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24), -1.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24), MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24), -1.0f,
    MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25), MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25), -0.5f,
    MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26), MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26), -0.25f,
    MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27), MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150), -FLT_MIN,
    MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150),
    MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150), MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150), MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150), MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150),
    MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150), MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150),
    MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150), -0.0f, +NAN, +INFINITY, +FLT_MAX,
    MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40), MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64), MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39),
    MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39), MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63), MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8), MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32), MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7),
    MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7), MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31), MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6),
    +1000.f, +100.f, +4.0f, +3.5f, +3.0f,
    MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23), 2.5f, MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23), +2.0f,
    MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24), 1.5f, MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24),
    MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24), +1.0f, MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25), +0.5f, MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26),
    MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26), +0.25f, MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150), +FLT_MIN, MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150),
    MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150), MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150), MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150), MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150),
    MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150), MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150),
    MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150), +0.0f
};

// A table of more difficult cases to get right
std::vector<double> DataInitInfo::specialValuesDouble = {
    -NAN, -INFINITY, -DBL_MAX,
    MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12), MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11), MAKE_HEX_DOUBLE(-0x1.80000000000001p64, -0x180000000000001LL, 8),
    MAKE_HEX_DOUBLE(-0x1.8p64, -0x18LL, 60), MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp64, -0x17ffffffffffffLL, 12),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p63, -0x180000000000001LL, 7), MAKE_HEX_DOUBLE(-0x1.8p63, -0x18LL, 59),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp63, -0x17ffffffffffffLL, 11), MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p32, -0x180000000000001LL, -24), MAKE_HEX_DOUBLE(-0x1.8p32, -0x18LL, 28),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp32, -0x17ffffffffffffLL, -20), MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8),
    MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(-0x1.80000000000001p31, -0x180000000000001LL, -25), MAKE_HEX_DOUBLE(-0x1.8p31, -0x18LL, 27),
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp31, -0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22),
    -1000., -100., -4.0, -3.5, -3.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51), -2.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51), -2.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52), -1.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52), MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52), -1.0,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53), MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53), -0.5,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54), MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54), -0.25,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55), MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074),
    -DBL_MIN,
    MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074), MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074), MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074),
    -0.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(0x1.80000000000001p63, 0x180000000000001LL, 7), MAKE_HEX_DOUBLE(0x1.8p63, 0x18LL, 59),
    MAKE_HEX_DOUBLE(0x1.7ffffffffffffp63, 0x17ffffffffffffLL, 11), MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.80000000000001p32, +0x180000000000001LL, -24), MAKE_HEX_DOUBLE(+0x1.8p32, +0x18LL, 28),
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp32, +0x17ffffffffffffLL, -20), MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8),
    MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(+0x1.80000000000001p31, +0x180000000000001LL, -25), MAKE_HEX_DOUBLE(+0x1.8p31, +0x18LL, 27),
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp31, +0x17ffffffffffffLL, -21), MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31), MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22),
    +1000., +100., +4.0, +3.5, +3.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51), +2.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51), +2.0, MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52),
    +1.5, MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52), MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    +1.0, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53), MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53),
    +0.5, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54), MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54),
    +0.25, MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55), MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074),
    +DBL_MIN, MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074), MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074), MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074), MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074), +0.0, MAKE_HEX_DOUBLE(-0x1.ffffffffffffep62, -0x1ffffffffffffeLL, 10),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp62, -0x1ffffffffffffcLL, 10), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep62, +0x1ffffffffffffeLL, 10), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp62, +0x1ffffffffffffcLL, 10),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep51, -0x1ffffffffffffeLL, -1),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp51, -0x1ffffffffffffcLL, -1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp51, -0x1fffffffffffffLL, -1),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep51, +0x1ffffffffffffeLL, -1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp51, +0x1ffffffffffffcLL, -1),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp51, +0x1fffffffffffffLL, -1), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep52, -0x1ffffffffffffeLL, 0),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp52, -0x1ffffffffffffcLL, 0), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp52, -0x1fffffffffffffLL, 0),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep52, +0x1ffffffffffffeLL, 0), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp52, +0x1ffffffffffffcLL, 0),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp52, +0x1fffffffffffffLL, 0), MAKE_HEX_DOUBLE(-0x1.ffffffffffffep53, -0x1ffffffffffffeLL, 1),
    MAKE_HEX_DOUBLE(-0x1.ffffffffffffcp53, -0x1ffffffffffffcLL, 1), MAKE_HEX_DOUBLE(-0x1.fffffffffffffp53, -0x1fffffffffffffLL, 1),
    MAKE_HEX_DOUBLE(+0x1.ffffffffffffep53, +0x1ffffffffffffeLL, 1), MAKE_HEX_DOUBLE(+0x1.ffffffffffffcp53, +0x1ffffffffffffcLL, 1),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp53, +0x1fffffffffffffLL, 1), MAKE_HEX_DOUBLE(-0x1.0000000000002p52, -0x10000000000002LL, 0),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p52, -0x10000000000001LL, 0), MAKE_HEX_DOUBLE(-0x1.0p52, -0x1LL, 52),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p52, +0x10000000000002LL, 0), MAKE_HEX_DOUBLE(+0x1.0000000000001p52, +0x10000000000001LL, 0),
    MAKE_HEX_DOUBLE(+0x1.0p52, +0x1LL, 52), MAKE_HEX_DOUBLE(-0x1.0000000000002p53, -0x10000000000002LL, 1),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p53, -0x10000000000001LL, 1), MAKE_HEX_DOUBLE(-0x1.0p53, -0x1LL, 53),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p53, +0x10000000000002LL, 1), MAKE_HEX_DOUBLE(+0x1.0000000000001p53, +0x10000000000001LL, 1),
    MAKE_HEX_DOUBLE(+0x1.0p53, +0x1LL, 53), MAKE_HEX_DOUBLE(-0x1.0000000000002p54, -0x10000000000002LL, 2),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p54, -0x10000000000001LL, 2), MAKE_HEX_DOUBLE(-0x1.0p54, -0x1LL, 54),
    MAKE_HEX_DOUBLE(+0x1.0000000000002p54, +0x10000000000002LL, 2), MAKE_HEX_DOUBLE(+0x1.0000000000001p54, +0x10000000000001LL, 2),
    MAKE_HEX_DOUBLE(+0x1.0p54, +0x1LL, 54), MAKE_HEX_DOUBLE(-0x1.fffffffefffffp62, -0x1fffffffefffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.ffffffffp62, -0x1ffffffffLL, 30), MAKE_HEX_DOUBLE(-0x1.ffffffff00001p62, -0x1ffffffff00001LL, 10),
    MAKE_HEX_DOUBLE(0x1.fffffffefffffp62, 0x1fffffffefffffLL, 10), MAKE_HEX_DOUBLE(0x1.ffffffffp62, 0x1ffffffffLL, 30),
    MAKE_HEX_DOUBLE(0x1.ffffffff00001p62, 0x1ffffffff00001LL, 10),
};

// A table of more difficult cases to get right
std::vector<cl_half> DataInitInfo::specialValuesHalf = {
    0xffff,
    0x0000,
    0x0001,
    0x7c00, /*INFINITY*/
    0xfc00, /*-INFINITY*/
    0x8000, /*-0*/
    0x7bff, /*HALF_MAX*/
    0x0400, /*HALF_MIN*/
    0x03ff, /* Largest denormal */
    0x3c00, /* 1 */
    0xbc00, /* -1 */
    0x3555, /*nearest value to 1/3*/
    0x3bff, /*largest number less than one*/
    0xc000, /* -2 */
    0xfbff, /* -HALF_MAX */
    0x8400, /* -HALF_MIN */
    0x4248, /* M_PI_H */
    0xc248, /* -M_PI_H */
    0xbbff, /* Largest negative fraction */
};
// clang-format on

// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
static inline void Force64BitFPUPrecision(void)
{
#if __MINGW32__
    // The usual method is to use _controlfp as follows:
    //     #include <float.h>
    //     _controlfp(_PC_64, _MCW_PC);
    //
    // _controlfp is available on MinGW32 but not on MinGW64. Instead of having
    // divergent code just use inline assembly which works for both.
    unsigned short int orig_cw = 0;
    unsigned short int new_cw = 0;
    __asm__ __volatile__("fstcw %0" : "=m"(orig_cw));
    new_cw = orig_cw | 0x0300; // set precision to 64-bit
    __asm__ __volatile__("fldcw  %0" ::"m"(new_cw));
#else
    /* Implement for other platforms if needed */
#endif
}

template <typename InType, typename OutType, bool InFP, bool OutFP>
int CalcRefValsPat<InType, OutType, InFP, OutFP>::check_result(void *test,
                                                               uint32_t count,
                                                               int vectorSize)
{
    const cl_uchar *a = (const cl_uchar *)gAllowZ;

    if (is_half<OutType, OutFP>())
    {
        const cl_half *t = (const cl_half *)test;
        const cl_half *c = (const cl_half *)gRef;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !((t[i] & 0x7fff) > 0x7C00 && (c[i] & 0x7fff) > 0x7C00)
                && !(a[i] != (cl_uchar)0 && t[i] == (c[i] & 0x8000)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, HTF(c[i]), HTF(t[i]));
                return i + 1;
            }
    }
    else if (std::is_integral<OutType>::value)
    { // char/uchar/short/ushort/half/int/uint/long/ulong
        const OutType *t = (const OutType *)test;
        const OutType *c = (const OutType *)gRef;
        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] && !(a[i] != (cl_uchar)0 && t[i] == (OutType)0))
            {
                size_t s = sizeof(OutType) * 2;
                std::stringstream sstr;
                sstr << "\nError for vector size %d found at 0x%8.8x:  *0x%"
                     << s << "." << s << "x vs 0x%" << s << "." << s << "x\n";
                vlog(sstr.str().c_str(), vectorSize, i, c[i], t[i]);
                return i + 1;
            }
    }
    else if (std::is_same<OutType, cl_float>::value)
    {
        // cast to integral - from original test
        const cl_uint *t = (const cl_uint *)test;
        const cl_uint *c = (const cl_uint *)gRef;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !((t[i] & 0x7fffffffU) > 0x7f800000U
                  && (c[i] & 0x7fffffffU) > 0x7f800000U)
                && !(a[i] != (cl_uchar)0 && t[i] == (c[i] & 0x80000000U)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)gRef)[i], ((OutType *)test)[i]);
                return i + 1;
            }
    }
    else
    {
        const cl_ulong *t = (const cl_ulong *)test;
        const cl_ulong *c = (const cl_ulong *)gRef;

        for (uint32_t i = 0; i < count; i++)
            if (t[i] != c[i] &&
                // Allow nan's to be binary different
                !((t[i] & 0x7fffffffffffffffULL) > 0x7ff0000000000000ULL
                  && (c[i] & 0x7fffffffffffffffULL) > 0x7f80000000000000ULL)
                && !(a[i] != (cl_uchar)0
                     && t[i] == (c[i] & 0x8000000000000000ULL)))
            {
                vlog(
                    "\nError for vector size %d found at 0x%8.8x:  *%a vs %a\n",
                    vectorSize, i, ((OutType *)gRef)[i], ((OutType *)test)[i]);
                return i + 1;
            }
    }

    return 0;
}


cl_uint RoundUpToNextPowerOfTwo(cl_uint x)
{
    if (0 == (x & (x - 1))) return x;

    while (x & (x - 1)) x &= x - 1;

    return x + x;
}


cl_int CustomConversionsTest::Run()
{
    int startMinVectorSize = gMinVectorSize;
    Type inType, outType;
    RoundingMode round;
    SaturationMode sat;

    for (int i = 0; i < argCount; i++)
    {
        if (conv_test::GetTestCase(argList[i], &outType, &inType, &sat, &round))
        {
            vlog_error("\n\t\t**** ERROR:  Unable to parse function name "
                       "%s.  Skipping....  *****\n\n",
                       argList[i]);
            continue;
        }

        // skip double if we don't have it
        if (!gTestDouble && (inType == kdouble || outType == kdouble))
        {
            if (gHasDouble)
            {
                vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                           gTypeNames[outType], gSaturationNames[sat],
                           gRoundingModeNames[round], gTypeNames[inType]);
                vlog("\t\tcl_khr_fp64 enabled, but double testing turned "
                     "off.\n");
            }
            continue;
        }

        // skip half if we don't have it
        if (!gTestHalfs && (inType == khalf || outType == khalf))
        {
            if (gHasHalfs)
            {
                vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                           gTypeNames[outType], gSaturationNames[sat],
                           gRoundingModeNames[round], gTypeNames[inType]);
                vlog("\t\tcl_khr_fp16 enabled, but half testing turned "
                     "off.\n");
            }
            continue;
        }

        // skip longs on embedded
        if (!gHasLong
            && (inType == klong || outType == klong || inType == kulong
                || outType == kulong))
        {
            continue;
        }

        // Skip the implicit converts if the rounding mode is not default or
        // test is saturated
        if (0 == startMinVectorSize)
        {
            if (sat || round != kDefaultRoundingMode)
                gMinVectorSize = 1;
            else
                gMinVectorSize = 0;
        }

        IterOverSelectedTypes iter(typeIterator, *this, inType, outType, round,
                                   sat);

        iter.Run();

        if (gFailCount)
        {
            vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                       gTypeNames[outType], gSaturationNames[sat],
                       gRoundingModeNames[round], gTypeNames[inType]);
        }
    }

    return gFailCount;
}


ConversionsTest::ConversionsTest(cl_device_id device, cl_context context,
                                 cl_command_queue queue)
    : context(context), device(device), queue(queue), num_elements(0),
      typeIterator({ cl_uchar(0), cl_char(0), cl_ushort(0), cl_short(0),
                     cl_uint(0), cl_int(0), cl_half(0), cl_float(0),
                     cl_double(0), cl_ulong(0), cl_long(0) })
{}


cl_int ConversionsTest::Run()
{
    IterOverTypes iter(typeIterator, *this);

    iter.Run();

    return gFailCount;
}


cl_int ConversionsTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        const cl_device_fp_config fpConfigHalf =
            get_default_rounding_mode(device, CL_DEVICE_HALF_FP_CONFIG);
        if ((fpConfigHalf & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            DataInitInfo::halfRoundingMode = CL_HALF_RTE;
            ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTE;
        }
        else if ((fpConfigHalf & CL_FP_ROUND_TO_ZERO) != 0)
        {
            DataInitInfo::halfRoundingMode = CL_HALF_RTZ;
            ConversionsTest::defaultHalfRoundingMode = CL_HALF_RTZ;
        }
        else
        {
            log_error("Error while acquiring half rounding mode");
            return TEST_FAIL;
        }
    }

    return CL_SUCCESS;
}

template <typename InType, typename OutType, bool InFP, bool OutFP>
void ConversionsTest::TestTypesConversion(const Type &inType,
                                          const Type &outType, int &testNumber,
                                          int startMinVectorSize)
{
    SaturationMode sat;
    RoundingMode round;
    int error;

    // skip longs on embedded
    if (!gHasLong
        && (inType == klong || outType == klong || inType == kulong
            || outType == kulong))
    {
        return;
    }

    for (sat = (SaturationMode)0; sat < kSaturationModeCount;
         sat = (SaturationMode)(sat + 1))
    {
        // skip illegal saturated conversions to float type
        if (kSaturated == sat
            && (outType == kfloat || outType == kdouble || outType == khalf))
        {
            continue;
        }

        for (round = (RoundingMode)0; round < kRoundingModeCount;
             round = (RoundingMode)(round + 1))
        {
            if (++testNumber < gStartTestNumber)
            {
                continue;
            }
            else
            {
                if (gEndTestNumber > 0 && testNumber >= gEndTestNumber) return;
            }

            vlog("%d) Testing convert_%sn%s%s( %sn ):\n", testNumber,
                 gTypeNames[outType], gSaturationNames[sat],
                 gRoundingModeNames[round], gTypeNames[inType]);

            // skip double if we don't have it
            if (!gTestDouble && (inType == kdouble || outType == kdouble))
            {
                if (gHasDouble)
                {
                    vlog_error("\t *** %d) convert_%sn%s%s( %sn ) "
                               "FAILED ** \n",
                               testNumber, gTypeNames[outType],
                               gSaturationNames[sat], gRoundingModeNames[round],
                               gTypeNames[inType]);
                    vlog("\t\tcl_khr_fp64 enabled, but double "
                         "testing turned off.\n");
                }
                continue;
            }

            // skip half if we don't have it
            if (!gTestHalfs && (inType == khalf || outType == khalf))
            {
                if (gHasHalfs)
                {
                    vlog_error("\t *** convert_%sn%s%s( %sn ) FAILED ** \n",
                               gTypeNames[outType], gSaturationNames[sat],
                               gRoundingModeNames[round], gTypeNames[inType]);
                    vlog("\t\tcl_khr_fp16 enabled, but half testing turned "
                         "off.\n");
                }
                continue;
            }

            // Skip the implicit converts if the rounding mode is
            // not default or test is saturated
            if (0 == startMinVectorSize)
            {
                if (sat || round != kDefaultRoundingMode)
                    gMinVectorSize = 1;
                else
                    gMinVectorSize = 0;
            }

            if ((error = DoTest<InType, OutType, InFP, OutFP>(outType, inType,
                                                              sat, round)))
            {
                vlog_error("\t *** %d) convert_%sn%s%s( %sn ) "
                           "FAILED ** \n",
                           testNumber, gTypeNames[outType],
                           gSaturationNames[sat], gRoundingModeNames[round],
                           gTypeNames[inType]);
            }
        }
    }
}

template <typename InType, typename OutType, bool InFP, bool OutFP>
int ConversionsTest::DoTest(Type outType, Type inType, SaturationMode sat,
                            RoundingMode round)
{
#ifdef __APPLE__
    cl_ulong wall_start = mach_absolute_time();
#endif

    cl_uint threads = GetThreadCount();

    DataInitInfo info = { 0, 0, outType, inType, sat, round, threads };
    DataInfoSpec<InType, OutType, InFP, OutFP> init_info(info);
    WriteInputBufferInfo writeInputBufferInfo;
    int vectorSize;
    int error = 0;
    uint64_t i;

    gTestCount++;
    size_t blockCount =
        BUFFER_SIZE / std::max(gTypeSizes[inType], gTypeSizes[outType]);
    size_t step = blockCount;

    for (i = 0; i < threads; i++)
    {
        init_info.mdv.emplace_back(MTdataHolder(gRandomSeed));
    }

    writeInputBufferInfo.outType = outType;
    writeInputBufferInfo.inType = inType;

    writeInputBufferInfo.calcInfo.resize(gMaxVectorSize);
    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        writeInputBufferInfo.calcInfo[vectorSize].reset(
            new CalcRefValsPat<InType, OutType, InFP, OutFP>());
        writeInputBufferInfo.calcInfo[vectorSize]->program =
            conv_test::MakeProgram(
                outType, inType, sat, round, vectorSize,
                &writeInputBufferInfo.calcInfo[vectorSize]->kernel);
        if (NULL == writeInputBufferInfo.calcInfo[vectorSize]->program)
        {
            gFailCount++;
            return -1;
        }
        if (NULL == writeInputBufferInfo.calcInfo[vectorSize]->kernel)
        {
            gFailCount++;
            vlog_error("\t\tFAILED -- Failed to create kernel.\n");
            return -2;
        }

        writeInputBufferInfo.calcInfo[vectorSize]->parent =
            &writeInputBufferInfo;
        writeInputBufferInfo.calcInfo[vectorSize]->vectorSize = vectorSize;
        writeInputBufferInfo.calcInfo[vectorSize]->result = -1;
    }

    if (gSkipTesting) return error;

    // Patch up rounding mode if default is RTZ
    // We leave the part above in default rounding mode so that the right kernel
    // is compiled.
    if (std::is_same<OutType, cl_float>::value)
    {
        if (round == kDefaultRoundingMode && gIsRTZ)
            init_info.round = round = kRoundTowardZero;
    }
    else if (std::is_same<OutType, cl_half>::value && OutFP)
    {
        if (round == kDefaultRoundingMode && gIsHalfRTZ)
            init_info.round = round = kRoundTowardZero;
    }

    // Figure out how many elements are in a work block
    // we handle 64-bit types a bit differently.
    uint64_t lastCase = (8 * gTypeSizes[inType] > 32)
        ? 0x100000000ULL
        : 1ULL << (8 * gTypeSizes[inType]);

    if (!gWimpyMode && gIsEmbedded)
        step = blockCount * EMBEDDED_REDUCTION_FACTOR;

    if (gWimpyMode) step = (size_t)blockCount * (size_t)gWimpyReductionFactor;
    vlog("Testing... ");
    fflush(stdout);
    for (i = 0; i < (uint64_t)lastCase; i += step)
    {

        if (0 == (i & ((lastCase >> 3) - 1)))
        {
            vlog(".");
            fflush(stdout);
        }

        cl_uint count = (uint32_t)std::min((uint64_t)blockCount, lastCase - i);
        writeInputBufferInfo.count = count;

        // Crate a user event to represent the status of the reference value
        // computation completion
        writeInputBufferInfo.calcReferenceValues =
            clCreateUserEvent(gContext, &error);
        if (error || NULL == writeInputBufferInfo.calcReferenceValues)
        {
            vlog_error("ERROR: Unable to create user event. (%d)\n", error);
            gFailCount++;
            return error;
        }

        // retain for consumption by MapOutputBufferComplete
        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            if ((error =
                     clRetainEvent(writeInputBufferInfo.calcReferenceValues)))
            {
                vlog_error("ERROR: Unable to retain user event. (%d)\n", error);
                gFailCount++;
                return error;
            }
        }

        // Crate a user event to represent when the callbacks are done verifying
        // correctness
        writeInputBufferInfo.doneBarrier = clCreateUserEvent(gContext, &error);
        if (error || NULL == writeInputBufferInfo.doneBarrier)
        {
            vlog_error("ERROR: Unable to create user event for barrier. (%d)\n",
                       error);
            gFailCount++;
            return error;
        }

        // retain for use by the callback that calls this
        if ((error = clRetainEvent(writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("ERROR: Unable to retain user event doneBarrier. (%d)\n",
                       error);
            gFailCount++;
            return error;
        }

        //      Call this in a multithreaded manner
        cl_uint chunks = RoundUpToNextPowerOfTwo(threads) * 2;
        init_info.start = i;
        init_info.size = count / chunks;
        if (init_info.size < 16384)
        {
            chunks = RoundUpToNextPowerOfTwo(threads);
            init_info.size = count / chunks;
            if (init_info.size < 16384)
            {
                init_info.size = count;
                chunks = 1;
            }
        }

        ThreadPool_Do(conv_test::InitData, chunks, &init_info);

        // Copy the results to the device
        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_TRUE, 0,
                                          count * gTypeSizes[inType], gIn, 0,
                                          NULL, NULL)))
        {
            vlog_error("ERROR: clEnqueueWriteBuffer failed. (%d)\n", error);
            gFailCount++;
            return error;
        }

        // Call completion callback for the write, which will enqueue the rest
        // of the work.
        conv_test::WriteInputBufferComplete((void *)&writeInputBufferInfo);

        // Make sure the work is actually running, so we don't deadlock
        if ((error = clFlush(gQueue)))
        {
            vlog_error("clFlush failed with error %d\n", error);
            gFailCount++;
            return error;
        }

        ThreadPool_Do(conv_test::PrepareReference, chunks, &init_info);

        // signal we are done calculating the reference results
        if ((error = clSetUserEventStatus(
                 writeInputBufferInfo.calcReferenceValues, CL_COMPLETE)))
        {
            vlog_error(
                "Error:  Failed to set user event status to CL_COMPLETE:  %d\n",
                error);
            gFailCount++;
            return error;
        }

        // Wait for the event callbacks to finish verifying correctness.
        if ((error = clWaitForEvents(
                 1, (cl_event *)&writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("Error:  Failed to wait for barrier:  %d\n", error);
            gFailCount++;
            return error;
        }

        if ((error = clReleaseEvent(writeInputBufferInfo.calcReferenceValues)))
        {
            vlog_error("Error:  Failed to release calcReferenceValues:  %d\n",
                       error);
            gFailCount++;
            return error;
        }

        if ((error = clReleaseEvent(writeInputBufferInfo.doneBarrier)))
        {
            vlog_error("Error:  Failed to release done barrier:  %d\n", error);
            gFailCount++;
            return error;
        }

        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            if ((error = writeInputBufferInfo.calcInfo[vectorSize]->result))
            {
                switch (inType)
                {
                    case kuchar:
                    case kchar:
                        vlog("Input value: 0x%2.2x ",
                             ((unsigned char *)gIn)[error - 1]);
                        break;
                    case kushort:
                    case kshort:
                        vlog("Input value: 0x%4.4x ",
                             ((unsigned short *)gIn)[error - 1]);
                        break;
                    case kuint:
                    case kint:
                        vlog("Input value: 0x%8.8x ",
                             ((unsigned int *)gIn)[error - 1]);
                        break;
                    case khalf:
                        vlog("Input value: %a ",
                             HTF(((cl_half *)gIn)[error - 1]));
                        break;
                    case kfloat:
                        vlog("Input value: %a ", ((float *)gIn)[error - 1]);
                        break;
                    case kulong:
                    case klong:
                        vlog("Input value: 0x%16.16llx ",
                             ((unsigned long long *)gIn)[error - 1]);
                        break;
                    case kdouble:
                        vlog("Input value: %a ", ((double *)gIn)[error - 1]);
                        break;
                    default:
                        vlog_error("Internal error at %s: %d\n", __FILE__,
                                   __LINE__);
                        abort();
                        break;
                }

                // tell the user which conversion it was.
                if (0 == vectorSize)
                    vlog(" (implicit scalar conversion from %s to %s)\n",
                         gTypeNames[inType], gTypeNames[outType]);
                else
                    vlog(" (convert_%s%s%s%s( %s%s ))\n", gTypeNames[outType],
                         sizeNames[vectorSize], gSaturationNames[sat],
                         gRoundingModeNames[round], gTypeNames[inType],
                         sizeNames[vectorSize]);

                gFailCount++;
                return error;
            }
        }
    }

    log_info("done.\n");

    if (gTimeResults)
    {
        // Kick off tests for the various vector lengths
        for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize;
             vectorSize++)
        {
            size_t workItemCount = blockCount / vectorSizes[vectorSize];
            if (vectorSizes[vectorSize] * gTypeSizes[outType] < 4)
                workItemCount /=
                    4 / (vectorSizes[vectorSize] * gTypeSizes[outType]);

            double sum = 0.0;
            double bestTime = INFINITY;
            cl_uint k;
            for (k = 0; k < PERF_LOOP_COUNT; k++)
            {
                uint64_t startTime = conv_test::GetTime();
                if ((error = conv_test::RunKernel(
                         writeInputBufferInfo.calcInfo[vectorSize]->kernel,
                         gInBuffer, gOutBuffers[vectorSize], workItemCount)))
                {
                    gFailCount++;
                    return error;
                }

                // Make sure OpenCL is done
                if ((error = clFinish(gQueue)))
                {
                    vlog_error("Error %d at clFinish\n", error);
                    return error;
                }

                uint64_t endTime = conv_test::GetTime();
                double time = SubtractTime(endTime, startTime);
                sum += time;
                if (time < bestTime) bestTime = time;
            }

            if (gReportAverageTimes) bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double)gDeviceFrequency
                * gComputeDevices * gSimdSize * 1e6
                / (workItemCount * vectorSizes[vectorSize]);
            if (0 == vectorSize)
                vlog_perf(clocksPerOp, LOWER_IS_BETTER, "clocks / element",
                          "implicit convert %s -> %s", gTypeNames[inType],
                          gTypeNames[outType]);
            else
                vlog_perf(clocksPerOp, LOWER_IS_BETTER, "clocks / element",
                          "convert_%s%s%s%s( %s%s )", gTypeNames[outType],
                          sizeNames[vectorSize], gSaturationNames[sat],
                          gRoundingModeNames[round], gTypeNames[inType],
                          sizeNames[vectorSize]);
        }
    }

    if (gWimpyMode)
        vlog("\tWimp pass");
    else
        vlog("\tpassed");

#ifdef __APPLE__
    // record the run time
    vlog("\t(%f s)", 1e-9 * (mach_absolute_time() - wall_start));
#endif
    vlog("\n\n");
    fflush(stdout);

    return error;
}

#if !defined(__APPLE__)
void memset_pattern4(void *dest, const void *src_pattern, size_t bytes);
#endif

#if defined(_MSC_VER)
/* function is defined in "compat.h" */
#else
double SubtractTime(uint64_t endTime, uint64_t startTime)
{
    uint64_t diff = endTime - startTime;
    static double conversion = 0.0;

    if (0.0 == conversion)
    {
#if defined(__APPLE__)
        mach_timebase_info_data_t info = { 0, 0 };
        kern_return_t err = mach_timebase_info(&info);
        if (0 == err)
            conversion = 1e-9 * (double)info.numer / (double)info.denom;
#else
        // This function consumes output from GetTime() above, and converts the
        // time to secionds.
#warning need accurate ticks to seconds conversion factor here. Times are invalid.
#endif
    }

    // strictly speaking we should also be subtracting out timer latency here
    return conversion * (double)diff;
}
#endif

void MapResultValuesComplete(const std::unique_ptr<CalcRefValsBase> &ptr);

void CL_CALLBACK CalcReferenceValuesComplete(cl_event e, cl_int status,
                                             void *data);

// Note: May be called reentrantly
void MapResultValuesComplete(const std::unique_ptr<CalcRefValsBase> &info)
{
    cl_int status;
    // CalcRefValsBase *info = (CalcRefValsBase *)data;
    cl_event calcReferenceValues = info->parent->calcReferenceValues;

    // we know that the map is done, wait for the main thread to finish
    // calculating the reference values
    if ((status =
             clSetEventCallback(calcReferenceValues, CL_COMPLETE,
                                CalcReferenceValuesComplete, (void *)&info)))
    {
        vlog_error("ERROR: clSetEventCallback failed in "
                   "MapResultValuesComplete with status: %d\n",
                   status);
        gFailCount++; // not thread safe -- being lazy here
    }

    // this thread no longer needs its reference to info->calcReferenceValues,
    // so release it
    if ((status = clReleaseEvent(calcReferenceValues)))
    {
        vlog_error("ERROR: clReleaseEvent(info->calcReferenceValues) failed "
                   "with status: %d\n",
                   status);
        gFailCount++; // not thread safe -- being lazy here
    }

    // no need to flush since we didn't enqueue anything

    // e was already released by WriteInputBufferComplete. It should be
    // destroyed automatically soon after we exit.
}

template <typename T> static bool isnan_fp(const T &v)
{
    if (std::is_same<T, cl_half>::value)
    {
        uint16_t h_exp = (((cl_half)v) >> (CL_HALF_MANT_DIG - 1)) & 0x1F;
        uint16_t h_mant = ((cl_half)v) & 0x3FF;
        return (h_exp == 0x1F && h_mant != 0);
    }
    else
    {
#if !defined(_WIN32)
        return std::isnan(v);
#else
        return _isnan(v);
#endif
    }
}

template <typename InType>
void ZeroNanToIntCases(cl_uint count, void *mapped, Type outType, void *input)
{
    InType *inp = (InType *)input;
    for (auto j = 0; j < count; j++)
    {
        if (isnan_fp<InType>(inp[j]))
            memset((char *)mapped + j * gTypeSizes[outType], 0,
                   gTypeSizes[outType]);
    }
}

template <typename InType, typename OutType>
void FixNanToFltConversions(InType *inp, OutType *outp, cl_uint count)
{
    if (std::is_same<OutType, cl_half>::value)
    {
        for (auto j = 0; j < count; j++)
            if (isnan_fp(inp[j]) && isnan_fp(outp[j]))
                outp[j] = 0x7e00; // HALF_NAN
    }
    else
    {
        for (auto j = 0; j < count; j++)
            if (isnan_fp(inp[j]) && isnan_fp(outp[j])) outp[j] = NAN;
    }
}

void FixNanConversions(Type outType, Type inType, void *d, cl_uint count,
                       void *inp)
{
    if (outType != kfloat && outType != kdouble && outType != khalf)
    {
        if (inType == kfloat)
            ZeroNanToIntCases<float>(count, d, outType, inp);
        else if (inType == kdouble)
            ZeroNanToIntCases<double>(count, d, outType, inp);
        else if (inType == khalf)
            ZeroNanToIntCases<cl_half>(count, d, outType, inp);
    }
    else if (inType == kfloat || inType == kdouble || inType == khalf)
    {
        // outtype and intype is float or double or half.  NaN conversions for
        // float/double/half could be any NaN
        if (inType == kfloat)
        {
            float *inp = (float *)gIn;
            if (outType == kdouble)
            {
                double *outp = (double *)d;
                FixNanToFltConversions(inp, outp, count);
            }
            else if (outType == khalf)
            {
                cl_half *outp = (cl_half *)d;
                FixNanToFltConversions(inp, outp, count);
            }
        }
        else if (inType == kdouble)
        {
            double *inp = (double *)gIn;
            if (outType == kfloat)
            {
                float *outp = (float *)d;
                FixNanToFltConversions(inp, outp, count);
            }
            else if (outType == khalf)
            {
                cl_half *outp = (cl_half *)d;
                FixNanToFltConversions(inp, outp, count);
            }
        }
        else if (inType == khalf)
        {
            cl_half *inp = (cl_half *)gIn;
            if (outType == kfloat)
            {
                float *outp = (float *)d;
                FixNanToFltConversions(inp, outp, count);
            }
            else if (outType == kdouble)
            {
                double *outp = (double *)d;
                FixNanToFltConversions(inp, outp, count);
            }
        }
    }
}


void CL_CALLBACK CalcReferenceValuesComplete(cl_event e, cl_int status,
                                             void *data)
{
    std::unique_ptr<CalcRefValsBase> &info =
        *(std::unique_ptr<CalcRefValsBase> *)data;

    cl_uint vectorSize = info->vectorSize;
    cl_uint count = info->parent->count;
    Type outType =
        info->parent->outType; // the data type of the conversion result
    Type inType = info->parent->inType; // the data type of the conversion input
    cl_int error;
    cl_event doneBarrier = info->parent->doneBarrier;

    // report spurious error condition
    if (CL_SUCCESS != status)
    {
        vlog_error("ERROR: CalcReferenceValuesComplete did not succeed! (%d)\n",
                   status);
        gFailCount++; // lazy about thread safety here
        return;
    }

    // Now we know that both results have been mapped back from the device, and
    // the main thread is done calculating the reference results. It is now time
    // to check the results.

    // verify results
    void *mapped = info->p;

    // Patch up NaNs conversions to integer to zero -- these can be converted to
    // any integer
    FixNanConversions(outType, inType, mapped, count, gIn);

    if (memcmp(mapped, gRef, count * gTypeSizes[outType]))
        info->result =
            info->check_result(mapped, count, vectorSizes[vectorSize]);
    else
        info->result = 0;

    // Fill the output buffer with junk and release it
    {
        cl_uint pattern = 0xffffdead;
        memset_pattern4(mapped, &pattern, count * gTypeSizes[outType]);
        if ((error = clEnqueueUnmapMemObject(gQueue, gOutBuffers[vectorSize],
                                             mapped, 0, NULL, NULL)))
        {
            vlog_error("ERROR: clEnqueueUnmapMemObject failed in "
                       "CalcReferenceValuesComplete  (%d)\n",
                       error);
            gFailCount++;
        }
    }

    if (1 == ThreadPool_AtomicAdd(&info->parent->barrierCount, -1))
    {
        if ((status = clSetUserEventStatus(doneBarrier, CL_COMPLETE)))
        {
            vlog_error("ERROR: clSetUserEventStatus failed in "
                       "CalcReferenceValuesComplete (err: %d). We're probably "
                       "going to deadlock.\n",
                       status);
            gFailCount++;
            return;
        }

        if ((status = clReleaseEvent(doneBarrier)))
        {
            vlog_error("ERROR: clReleaseEvent failed in "
                       "CalcReferenceValuesComplete (err: %d).\n",
                       status);
            gFailCount++;
            return;
        }
    }
    // e was already released by WriteInputBufferComplete. It should be
    // destroyed automatically soon after all the calls to
    // CalcReferenceValuesComplete exit.
}

namespace conv_test {

cl_int InitData(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    info->init(job_id, thread_id);

    return CL_SUCCESS;
}

cl_int PrepareReference(cl_uint job_id, cl_uint thread_id, void *p)
{
    DataInitBase *info = (DataInitBase *)p;

    cl_uint count = info->size;
    Type inType = info->inType;
    Type outType = info->outType;
    RoundingMode round = info->round;

    Force64BitFPUPrecision();

    void *s = (cl_uchar *)gIn + job_id * count * gTypeSizes[info->inType];
    void *a = (cl_uchar *)gAllowZ + job_id * count;
    void *d = (cl_uchar *)gRef + job_id * count * gTypeSizes[info->outType];

    if (outType != inType)
    {
        // create the reference while we wait
#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
        /* ARM VFP doesn't have hardware instruction for converting from 64-bit
         * integer to float types, hence GCC ARM uses the floating-point
         * emulation code despite which -mfloat-abi setting it is. But the
         * emulation code in libgcc.a has only one rounding mode (round to
         * nearest even in this case) and ignores the user rounding mode setting
         * in hardware. As a result setting rounding modes in hardware won't
         * give correct rounding results for type covert from 64-bit integer to
         * float using GCC for ARM compiler so for testing different rounding
         * modes, we need to use alternative reference function. ARM64 does have
         * an instruction, however we cannot guarantee the compiler will use it.
         * On all ARM architechures use emulation to calculate reference.*/
        switch (round)
        {
            /* conversions to floating-point type use the current rounding mode.
             * The only default floating-point rounding mode supported is round
             * to nearest even i.e the current rounding mode will be _rte for
             * floating-point types. */
            case kDefaultRoundingMode: qcom_rm = qcomRTE; break;
            case kRoundToNearestEven: qcom_rm = qcomRTE; break;
            case kRoundUp: qcom_rm = qcomRTP; break;
            case kRoundDown: qcom_rm = qcomRTN; break;
            case kRoundTowardZero: qcom_rm = qcomRTZ; break;
            default:
                vlog_error("ERROR: undefined rounding mode %d\n", round);
                break;
        }
        qcom_sat = info->sat;
#endif

        RoundingMode oldRound;
        if (outType == khalf)
        {
            oldRound = set_round(kRoundToNearestEven, kfloat);
            switch (round)
            {
                default:
                case kDefaultRoundingMode:
                    DataInitInfo::halfRoundingMode =
                        ConversionsTest::defaultHalfRoundingMode;
                    break;
                case kRoundToNearestEven:
                    DataInitInfo::halfRoundingMode = CL_HALF_RTE;
                    break;
                case kRoundUp:
                    DataInitInfo::halfRoundingMode = CL_HALF_RTP;
                    break;
                case kRoundDown:
                    DataInitInfo::halfRoundingMode = CL_HALF_RTN;
                    break;
                case kRoundTowardZero:
                    DataInitInfo::halfRoundingMode = CL_HALF_RTZ;
                    break;
            }
        }
        else
            oldRound = set_round(round, outType);

        if (info->sat)
            info->conv_array_sat(d, s, count);
        else
            info->conv_array(d, s, count);

        set_round(oldRound, outType);

        // Decide if we allow a zero result in addition to the correctly rounded
        // one
        memset(a, 0, count);
        if (gForceFTZ && (inType == kfloat || outType == kfloat))
        {
            info->set_allow_zero_array((uint8_t *)a, d, s, count);
        }
        if (gForceHalfFTZ && (inType == khalf || outType == khalf))
        {
            info->set_allow_zero_array((uint8_t *)a, d, s, count);
        }
    }
    else
    {
        // Copy the input to the reference
        memcpy(d, s, info->size * gTypeSizes[inType]);
    }

    // Patch up NaNs conversions to integer to zero -- these can be converted to
    // any integer
    FixNanConversions(outType, inType, d, count, s);

    return CL_SUCCESS;
}

uint64_t GetTime(void)
{
#if defined(__APPLE__)
    return mach_absolute_time();
#elif defined(_MSC_VER)
    return ReadTime();
#else
    // mach_absolute_time is a high precision timer with precision < 1
    // microsecond.
#warning need accurate clock here.  Times are invalid.
    return 0;
#endif
}

// Note: not called reentrantly
void WriteInputBufferComplete(void *data)
{
    cl_int status;
    WriteInputBufferInfo *info = (WriteInputBufferInfo *)data;
    cl_uint count = info->count;
    int vectorSize;

    info->barrierCount = gMaxVectorSize - gMinVectorSize;

    // now that we know that the write buffer is complete, enqueue callbacks to
    // wait for the main thread to finish calculating the reference results.
    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        size_t workItemCount =
            (count + vectorSizes[vectorSize] - 1) / (vectorSizes[vectorSize]);

        if ((status = conv_test::RunKernel(info->calcInfo[vectorSize]->kernel,
                                           gInBuffer, gOutBuffers[vectorSize],
                                           workItemCount)))
        {
            gFailCount++;
            return;
        }

        info->calcInfo[vectorSize]->p = clEnqueueMapBuffer(
            gQueue, gOutBuffers[vectorSize], CL_TRUE,
            CL_MAP_READ | CL_MAP_WRITE, 0, count * gTypeSizes[info->outType], 0,
            NULL, NULL, &status);
        {
            if (status)
            {
                vlog_error("ERROR: WriteInputBufferComplete calback failed "
                           "with status: %d\n",
                           status);
                gFailCount++;
                return;
            }
        }
    }

    for (vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        MapResultValuesComplete(info->calcInfo[vectorSize]);
    }

    // Make sure the work starts moving -- otherwise we may deadlock
    if ((status = clFlush(gQueue)))
    {
        vlog_error(
            "ERROR: WriteInputBufferComplete calback failed with status: %d\n",
            status);
        gFailCount++;
        return;
    }

    // e was already released by the main thread. It should be destroyed
    // automatically soon after we exit.
}

cl_program MakeProgram(Type outType, Type inType, SaturationMode sat,
                       RoundingMode round, int vectorSize, cl_kernel *outKernel)
{
    cl_program program;
    char testName[256];
    int error = 0;

    std::ostringstream source;
    if (outType == kdouble || inType == kdouble)
        source << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (outType == khalf || inType == khalf)
        source << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

    // Create the program. This is a bit complicated because we are trying to
    // avoid byte and short stores.
    if (0 == vectorSize)
    {
        // Create the type names.
        char inName[32];
        char outName[32];
        strncpy(inName, gTypeNames[inType], sizeof(inName));
        strncpy(outName, gTypeNames[outType], sizeof(outName));
        sprintf(testName, "test_implicit_%s_%s", outName, inName);

        source << "__kernel void " << testName << "( __global " << inName
               << " *src, __global " << outName << " *dest )\n";
        source << "{\n";
        source << "   size_t i = get_global_id(0);\n";
        source << "   dest[i] =  src[i];\n";
        source << "}\n";

        vlog("Building implicit %s -> %s conversion test\n", gTypeNames[inType],
             gTypeNames[outType]);
        fflush(stdout);
    }
    else
    {
        int vectorSizetmp = vectorSizes[vectorSize];

        // Create the type names.
        char convertString[128];
        char inName[32];
        char outName[32];
        switch (vectorSizetmp)
        {
            case 1:
                strncpy(inName, gTypeNames[inType], sizeof(inName));
                strncpy(outName, gTypeNames[outType], sizeof(outName));
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
            case 3:
                strncpy(inName, gTypeNames[inType], sizeof(inName));
                strncpy(outName, gTypeNames[outType], sizeof(outName));
                snprintf(convertString, sizeof(convertString),
                         "convert_%s3%s%s", outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s3", convertString, inName);
                vlog("Building %s( %s3 ) test\n", convertString, inName);
                break;
            default:
                snprintf(inName, sizeof(inName), "%s%d", gTypeNames[inType],
                         vectorSizetmp);
                snprintf(outName, sizeof(outName), "%s%d", gTypeNames[outType],
                         vectorSizetmp);
                snprintf(convertString, sizeof(convertString), "convert_%s%s%s",
                         outName, gSaturationNames[sat],
                         gRoundingModeNames[round]);
                snprintf(testName, 256, "test_%s_%s", convertString, inName);
                vlog("Building %s( %s ) test\n", convertString, inName);
                break;
        }
        fflush(stdout);

        if (vectorSizetmp == 3)
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   if( i + 1 < get_global_size(0))\n";
            source << "       vstore3( " << convertString
                   << "( vload3( i, src)), i, dest );\n";
            source << "   else\n";
            source << "   {\n";
            source << "       " << inName << "3 in;\n";
            source << "       " << outName << "3 out;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           in.y = src[3*i+1];\n";
            source << "       in.x = src[3*i];\n";
            source << "       out = " << convertString << "( in ); \n";
            source << "       dest[3*i] = out.x;\n";
            source << "       if( 0 == (i & 1) )\n";
            source << "           dest[3*i+1] = out.y;\n";
            source << "   }\n";
            source << "}\n";
        }
        else
        {
            source << "__kernel void " << testName << "( __global " << inName
                   << " *src, __global " << outName << " *dest )\n";
            source << "{\n";
            source << "   size_t i = get_global_id(0);\n";
            source << "   dest[i] = " << convertString << "( src[i] );\n";
            source << "}\n";
        }
    }
    *outKernel = NULL;

    const char *flags = NULL;
    if ((gForceFTZ && (inType == kfloat || outType == kfloat))
        || (gForceHalfFTZ && (inType == khalf || outType == khalf)))
    {
        flags = "-cl-denorms-are-zero";
    }

    // build it
    std::string sourceString = source.str();
    const char *programSource = sourceString.c_str();
    error = create_single_kernel_helper(gContext, &program, outKernel, 1,
                                        &programSource, testName, flags);
    if (error)
    {
        vlog_error("Failed to build kernel/program (err = %d).\n", error);
        return NULL;
    }

    return program;
}

//

int RunKernel(cl_kernel kernel, void *inBuf, void *outBuf, size_t blockCount)
{
    // The global dimensions are just the blockCount to execute since we haven't
    // set up multiple queues for multiple devices.
    int error;

    error = clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf);
    error |= clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf);

    if (error)
    {
        vlog_error("FAILED -- could not set kernel args (%d)\n", error);
        return error;
    }

    if ((error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &blockCount,
                                        NULL, 0, NULL, NULL)))
    {
        vlog_error("FAILED -- could not execute kernel (%d)\n", error);
        return error;
    }

    return 0;
}


int GetTestCase(const char *name, Type *outType, Type *inType,
                SaturationMode *sat, RoundingMode *round)
{
    int i;

    // Find the return type
    for (i = 0; i < kTypeCount; i++)
        if (name == strstr(name, gTypeNames[i]))
        {
            *outType = (Type)i;
            name += strlen(gTypeNames[i]);

            break;
        }

    if (i == kTypeCount) return -1;

    // Check to see if _sat appears next
    *sat = (SaturationMode)0;
    for (i = 1; i < kSaturationModeCount; i++)
        if (name == strstr(name, gSaturationNames[i]))
        {
            *sat = (SaturationMode)i;
            name += strlen(gSaturationNames[i]);
            break;
        }

    *round = (RoundingMode)0;
    for (i = 1; i < kRoundingModeCount; i++)
        if (name == strstr(name, gRoundingModeNames[i]))
        {
            *round = (RoundingMode)i;
            name += strlen(gRoundingModeNames[i]);
            break;
        }

    if (*name != '_') return -2;
    name++;

    for (i = 0; i < kTypeCount; i++)
        if (name == strstr(name, gTypeNames[i]))
        {
            *inType = (Type)i;
            name += strlen(gTypeNames[i]);

            break;
        }

    if (i == kTypeCount) return -3;

    if (*name != '\0') return -4;

    return 0;
}

} // namespace conv_test
