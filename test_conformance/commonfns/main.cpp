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

#include <stdio.h>
#include <string.h>
#include "procs.h"

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


test_definition test_list[] = {
    ADD_TEST( clamp ),
    ADD_TEST( degrees ),
    ADD_TEST( fmax ),
    ADD_TEST( fmaxf ),
    ADD_TEST( fmin ),
    ADD_TEST( fminf ),
    ADD_TEST( max ),
    ADD_TEST( maxf ),
    ADD_TEST( min ),
    ADD_TEST( minf ),
    ADD_TEST( mix ),
    ADD_TEST( radians ),
    ADD_TEST( step ),
    ADD_TEST( stepf ),
    ADD_TEST( smoothstep ),
    ADD_TEST( smoothstepf ),
    ADD_TEST( sign ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    initVecSizes();
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

