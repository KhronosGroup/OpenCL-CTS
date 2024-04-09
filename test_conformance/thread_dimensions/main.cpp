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

// Additional parameters to limit test scope (-n,-b,-x)
cl_uint maxThreadDimension = 0;
cl_uint bufferSize = 0;
cl_uint bufferStep = 0;

test_definition test_list[] = {
    ADD_TEST(quick_1d_explicit_local), ADD_TEST(quick_2d_explicit_local),
    ADD_TEST(quick_3d_explicit_local), ADD_TEST(quick_1d_implicit_local),
    ADD_TEST(quick_2d_implicit_local), ADD_TEST(quick_3d_implicit_local),
    ADD_TEST(full_1d_explicit_local),  ADD_TEST(full_2d_explicit_local),
    ADD_TEST(full_3d_explicit_local),  ADD_TEST(full_1d_implicit_local),
    ADD_TEST(full_2d_implicit_local),  ADD_TEST(full_3d_implicit_local),
};

const int test_num = ARRAY_SIZE(test_list);

int main(int argc, const char *argv[])
{
    int delArg = 0;
    for (auto i = 0; i < argc; i++)
    {
        delArg = 0;

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            log_info("Thread dimensions options:\n");
            log_info("\t-n\tMaximum thread dimension value\n");
            log_info("\t-b\tSpecifies a buffer size for calculations\n");
            log_info("\t-x\tSpecifies a step for calculations\n");
        }
        if (strcmp(argv[i], "-n") == 0)
        {
            delArg++;
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -n Maximum thread dimension value must be "
                         "greater than 0");
                return TEST_FAIL;
            }
            maxThreadDimension = atoi(argv[i + 1]);
            delArg++;
        }
        if (strcmp(argv[i], "-b") == 0)
        {
            delArg++;
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -b Buffer size must be greater than 0");
                return TEST_FAIL;
            }
            bufferSize = atoi(argv[i + 1]);
            delArg++;
        }
        if (strcmp(argv[i], "-x") == 0)
        {
            delArg++;
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -x Buffer step must be greater than 0");
                return TEST_FAIL;
            }
            bufferStep = atoi(argv[i + 1]);
            delArg++;
        }
        for (int j = i; j < argc - delArg; j++) argv[j] = argv[j + delArg];
        argc -= delArg;
        i -= delArg;
    }

    return runTestHarness(argc, argv, test_num, test_list, false, 0);
}
