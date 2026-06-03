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
#include <string>
#include <filesystem>

#include "harness/testHarness.h"
#include "harness/stringHelpers.h"
#include "harness/os_helpers.h"

std::string spvBinariesPath = "spirv_bin";
std::string spvIncludeTestDirectory = "includeTestDirectory";
std::string spvSecondIncludeTestDirectory = "secondIncludeTestDirectory";

const std::string spvBinariesPathArg = "--spirv-binaries-path";
const std::string spvIncludeTestDirectoryArg = "--include-test-directory";
const std::string spvSecondIncludeTestDirectoryArg =
    "--second-include-test-directory";

void printUsage()
{
    log_info("Reading SPIR-V files from default '%s' path.\n",
             spvBinariesPath.c_str());
    log_info("In case you want to set other directory use '%s' argument.\n",
             spvBinariesPathArg.c_str());
}

int main(int argc, const char *argv[])
{
    char const *sep = get_dir_sep();
    char const *exe_dir = get_exe_dir();

    // Set default include directories
    spvIncludeTestDirectory =
        std::string(exe_dir) + sep + "includeTestDirectory";
    spvSecondIncludeTestDirectory =
        std::string(exe_dir) + sep + "secondIncludeTestDirectory";

    free((void *)sep);
    free((void *)exe_dir);

    bool modifiedSpvBinariesPath = false;
    bool listTests = false;
    for (int i = 0; i < argc; ++i)
    {
        int argsRemoveNum = 0;
        if (argv[i] == spvBinariesPathArg)
        {
            if (i + 1 == argc)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvBinariesPathArg.c_str());
                return TEST_FAIL;
            }
            else
            {
                spvBinariesPath = std::string(argv[i + 1]);
                argsRemoveNum += 2;
                modifiedSpvBinariesPath = true;
            }
        }
        if (argv[i] == spvIncludeTestDirectoryArg)
        {
            if (i + 1 == argc)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvIncludeTestDirectoryArg.c_str());
                return TEST_FAIL;
            }
            else
            {
                spvIncludeTestDirectory = std::string(argv[i + 1]);
                argsRemoveNum += 2;
            }
        }
        if (argv[i] == spvSecondIncludeTestDirectoryArg)
        {
            if (i + 1 == argc)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvSecondIncludeTestDirectoryArg.c_str());
                return TEST_FAIL;
            }
            else
            {
                spvSecondIncludeTestDirectory = std::string(argv[i + 1]);
                argsRemoveNum += 2;
            }
        }

        if (argsRemoveNum > 0)
        {
            for (int j = i; j < (argc - argsRemoveNum); ++j)
                argv[j] = argv[j + argsRemoveNum];

            argc -= argsRemoveNum;
            --i;
        }
        listTests |= (argv[i] == std::string("--list")
                      || argv[i] == std::string("-list"));
    }
    if (modifiedSpvBinariesPath == false && !listTests)
    {
        printUsage();
    }

    return runTestHarness(argc, argv, test_registry::getInstance().num_tests(),
                          test_registry::getInstance().definitions(), false, 0);
}
