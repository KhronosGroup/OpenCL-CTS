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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif





basefn    basefn_list[] = {
    test_step_type,
    test_step_var,
    test_step_typedef_type,
    test_step_typedef_var,
};

const char    *basefn_names[] = {
    "step_type",
    "step_var",
    "step_typedef_type",
    "step_typedef_var",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, false, 0 );
}


