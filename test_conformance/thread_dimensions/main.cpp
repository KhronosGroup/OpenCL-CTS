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
#include "harness/testHarness.h"

#include <stdio.h>
#include <string.h>

// Additional parameters to limit test scope (-n,-b,-x)
cl_uint maxThreadDimension = 0;
cl_uint bufferSize = 0;
cl_uint bufferStep = 0;

static test_status parseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{
    help = R"(        -n    Maximum thread dimension value
        -b    Specifies a buffer size for calculations
        -x    Specifies a step for calculations
)";

    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_info("ERROR: -n Maximum thread dimension value missing\n");
                return TEST_FAIL;
            }
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -n Maximum thread dimension value must be "
                         "greater than 0\n");
                return TEST_FAIL;
            }
            maxThreadDimension = atoi(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-b") == 0)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_info("ERROR: -b Buffer size missing\n");
                return TEST_FAIL;
            }
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -b Buffer size must be greater than 0\n");
                return TEST_FAIL;
            }
            bufferSize = atoi(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-x") == 0)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_info("ERROR: -x Buffer step missing\n");
                return TEST_FAIL;
            }
            if (atoi(argv[i + 1]) < 1)
            {
                log_info("ERROR: -x Buffer step must be greater than 0\n");
                return TEST_FAIL;
            }
            bufferStep = atoi(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            i++;
        }
        else
        {
            argList.push_back(argv[i]);
        }
    }

    update_argc_argv_from_args_list(argList, argc, argv);
    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(argc, argv, false, 0, nullptr,
                                           parseArgs);
}
