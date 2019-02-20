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
#include "../../test_common/harness/testHarness.h"
#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
            test_geom_cross,
            test_geom_dot,
            test_geom_distance,
            test_geom_fast_distance,
            test_geom_length,
            test_geom_fast_length,
            test_geom_normalize,
            test_geom_fast_normalize
};


const char    *basefn_names[] = {
            "geom_cross",
            "geom_dot",
            "geom_distance",
            "geom_fast_distance",
            "geom_length",
            "geom_fast_length",
            "geom_normalize",
            "geom_fast_normalize",

            "all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

const unsigned int g_vecSizeof[] = {0,1,2,4,4,0,0,0,8,
               0,0,0,0,0,0,0,16};

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


