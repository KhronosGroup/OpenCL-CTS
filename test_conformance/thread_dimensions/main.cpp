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

test_definition test_list[] = {
    ADD_TEST( quick_1d_explicit_local ),
    ADD_TEST( quick_2d_explicit_local ),
    ADD_TEST( quick_3d_explicit_local ),
    ADD_TEST( quick_1d_implicit_local ),
    ADD_TEST( quick_2d_implicit_local ),
    ADD_TEST( quick_3d_implicit_local ),
    ADD_TEST( full_1d_explicit_local ),
    ADD_TEST( full_2d_explicit_local ),
    ADD_TEST( full_3d_explicit_local ),
    ADD_TEST( full_1d_implicit_local ),
    ADD_TEST( full_2d_implicit_local ),
    ADD_TEST( full_3d_implicit_local ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}

