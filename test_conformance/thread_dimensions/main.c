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

#include "../../test_common/harness/compat.h"

basefn    basefn_list[] = {
test_quick_thread_dimensions_1d_explicit_local,
test_quick_thread_dimensions_2d_explicit_local,
test_quick_thread_dimensions_3d_explicit_local,
test_quick_thread_dimensions_1d_implicit_local,
test_quick_thread_dimensions_2d_implicit_local,
test_quick_thread_dimensions_3d_implicit_local,
test_full_thread_dimensions_1d_explicit_local,
test_full_thread_dimensions_2d_explicit_local,
test_full_thread_dimensions_3d_explicit_local,
test_full_thread_dimensions_1d_implicit_local,
test_full_thread_dimensions_2d_implicit_local,
test_full_thread_dimensions_3d_implicit_local,
};

const char *commonfn_names[] = {
"quick_1d_explicit_local",
"quick_2d_explicit_local",
"quick_3d_explicit_local",
"quick_1d_implicit_local",
"quick_2d_implicit_local",
"quick_3d_implicit_local",
"full_1d_explicit_local",
"full_2d_explicit_local",
"full_3d_explicit_local",
"full_1d_implicit_local",
"full_2d_implicit_local",
"full_3d_implicit_local",

"all",
};

ct_assert((sizeof(commonfn_names) / sizeof(commonfn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_commonfns = sizeof(commonfn_names) / sizeof(char *);

int
main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_commonfns, basefn_list, commonfn_names, false, false, 0 );
}




