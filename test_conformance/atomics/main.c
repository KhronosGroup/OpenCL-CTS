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
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif


basefn    basefn_list[] = {
            test_atomic_add,
            test_atomic_sub,
            test_atomic_xchg,
            test_atomic_min,
            test_atomic_max,
            test_atomic_inc,
            test_atomic_dec,
            test_atomic_cmpxchg,
            test_atomic_and,
            test_atomic_or,
            test_atomic_xor,

            test_atomic_add_index,
            test_atomic_add_index_bin
};

const char    *basefn_names[] = {
            "atomic_add",
            "atomic_sub",
            "atomic_xchg",
            "atomic_min",
            "atomic_max",
            "atomic_inc",
            "atomic_dec",
            "atomic_cmpxchg",
            "atomic_and",
            "atomic_or",
            "atomic_xor",

            "atomic_add_index",
            "atomic_add_index_bin",

            "all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


