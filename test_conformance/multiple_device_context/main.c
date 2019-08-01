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
#include "harness/mt19937.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn    basefn_list[] = {
          test_multiple_contexts_same_device,
            test_two_contexts_same_device,
            test_three_contexts_same_device,
            test_four_contexts_same_device,

            test_two_devices,
            test_max_devices,

            test_hundred_queues
};


const char    *basefn_names[] = {
          "context_multiple_contexts_same_device",
            "context_two_contexts_same_device",
            "context_three_contexts_same_device",
            "context_four_contexts_same_device",

            "two_devices",
            "max_devices",

            "hundred_queues",

            "all",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    return runTestHarness( argc, argv, num_fns, basefn_list, basefn_names, false, true, 0 );
}


