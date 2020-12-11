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
#include "harness/testHarness.h"

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


test_definition test_list[] = {
    ADD_TEST( relational_any ),
    ADD_TEST( relational_all ),
    ADD_TEST( relational_bitselect ),
    ADD_TEST( relational_select_signed ),
    ADD_TEST( relational_select_unsigned ),

    ADD_TEST( relational_isequal ),
    ADD_TEST( relational_isnotequal ),
    ADD_TEST( relational_isgreater ),
    ADD_TEST( relational_isgreaterequal ),
    ADD_TEST( relational_isless ),
    ADD_TEST( relational_islessequal ),
    ADD_TEST( relational_islessgreater ),

    ADD_TEST( shuffle_copy ),
    ADD_TEST( shuffle_function_call ),
    ADD_TEST( shuffle_array_cast ),
    ADD_TEST( shuffle_built_in ),
    ADD_TEST( shuffle_built_in_dual_input ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}

