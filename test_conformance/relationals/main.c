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

#if DENSE_PACK_VECS
const int g_vector_aligns[] = {0, 1, 2, 3, 4,
                               5, 6, 7, 8,
                               9, 10, 11, 12,
                               13, 14, 15, 16};

#else
const int g_vector_aligns[] = {0, 1, 2, 4, 4,
                               8, 8, 8, 8,
                               16, 16, 16, 16,
                               16, 16, 16, 16};
#endif


const int g_vector_allocs[] = {0, 1, 2, 4, 4,
                               8, 8, 8, 8,
                               16, 16, 16, 16,
                               16, 16, 16, 16};


basefn    basefn_list[] = {
            test_relational_any,
            test_relational_all,
            test_relational_bitselect,
            test_relational_select_signed,
            test_relational_select_unsigned,

            test_relational_isequal,
            test_relational_isnotequal,
            test_relational_isgreater,
            test_relational_isgreaterequal,
            test_relational_isless,
            test_relational_islessequal,
            test_relational_islessgreater,

            test_shuffle_copy,
            test_shuffle_function_call,
            test_shuffle_array_cast,
            test_shuffle_built_in,
            test_shuffle_built_in_dual_input
};

const char    *basefn_names[] = {
            "relational_any",
            "relational_all",
            "relational_bitselect",
            "relational_select_signed",
            "relational_select_unsigned",

            "relational_isequal",
            "relational_isnotequal",
            "relational_isgreater",
            "relational_isgreaterequal",
            "relational_isless",
            "relational_islessequal",
            "relational_islessgreater",

            "shuffle_copy",
            "shuffle_function_call",
            "shuffle_array_cast",
            "shuffle_built_in",
            "shuffle_built_in_dual_input",

            "all"
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


