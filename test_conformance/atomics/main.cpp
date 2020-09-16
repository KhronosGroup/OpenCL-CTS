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
#include <stdlib.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST( atomic_add ),
    ADD_TEST( atomic_sub ),
    ADD_TEST( atomic_xchg ),
    ADD_TEST( atomic_min ),
    ADD_TEST( atomic_max ),
    ADD_TEST( atomic_inc ),
    ADD_TEST( atomic_dec ),
    ADD_TEST( atomic_cmpxchg ),
    ADD_TEST( atomic_and ),
    ADD_TEST( atomic_or ),
    ADD_TEST( atomic_xor ),

    ADD_TEST( atomic_add_index ),
    ADD_TEST( atomic_add_index_bin ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, test_num, test_list, false, false, 0 );
}

