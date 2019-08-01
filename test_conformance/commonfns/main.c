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
#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>
#include "procs.h"

#include "harness/compat.h"

int g_arrVecSizes[kVectorSizeCount + kStrangeVectorSizeCount];
int g_arrStrangeVectorSizes[kStrangeVectorSizeCount] = {3};

static void initVecSizes() {
    int i;
    for(i = 0; i < kVectorSizeCount; ++i) {
    g_arrVecSizes[i] = (1<<i);
    }
    for(; i < kVectorSizeCount + kStrangeVectorSizeCount; ++i) {
    g_arrVecSizes[i] = g_arrStrangeVectorSizes[i-kVectorSizeCount];
    }
}


basefn    commonfn_list[] = {
                test_clamp,
                test_degrees,
                test_fmax,
                test_fmaxf,
                test_fmin,
                test_fminf,
                test_max,
                test_maxf,
                test_min,
                test_minf,
                test_mix,
                test_radians,
                test_step,
                test_stepf,
                test_smoothstep,
                test_smoothstepf,
                test_sign,
};

const char *commonfn_names[] = {
    "clamp",
    "degrees",
    "fmax",
    "fmaxf",
    "fmin",
    "fminf",
    "max",
    "maxf",
    "min",
    "minf",
    "mix",
    "radians",
    "step",
    "stepf",
    "smoothstep",
    "smoothstepf",
    "sign",
    "all",
};

ct_assert((sizeof(commonfn_names) / sizeof(commonfn_names[0]) - 1) == (sizeof(commonfn_list) / sizeof(commonfn_list[0])));

int    num_commonfns = sizeof(commonfn_names) / sizeof(char *);

int
main(int argc, const char *argv[])
{
    initVecSizes();
    return runTestHarness( argc, argv, num_commonfns, commonfn_list, commonfn_names, false, false, 0 );
}


