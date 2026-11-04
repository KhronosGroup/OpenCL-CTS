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
#include "harness/parseParameters.h"

std::string spvBinariesPath = "spirv_bin";
std::string spvIncludeTestDirectory = "includeTestDirectory";
std::string spvSecondIncludeTestDirectory = "secondIncludeTestDirectory";

const std::string spvBinariesPathArg = "--spirv-binaries-path";
const std::string spvIncludeTestDirectoryArg = "--include-test-directory";
const std::string spvSecondIncludeTestDirectoryArg =
    "--second-include-test-directory";

static test_status parseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{
    help = "        " + spvBinariesPathArg
        + " <path> - Set path to read SPIR-V files from (default: "
        + spvBinariesPath + ")\n" + "        " + spvIncludeTestDirectoryArg
        + " <path> - Set include test directory\n" + "        "
        + spvSecondIncludeTestDirectoryArg
        + " <path> - Set second include test directory\n";

    bool modifiedSpvBinariesPath = false;
    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    for (int i = 1; i < argc; ++i)
    {
        if (argv[i] == spvBinariesPathArg)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvBinariesPathArg.c_str());
                return TEST_FAIL;
            }
            spvBinariesPath = std::string(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            modifiedSpvBinariesPath = true;
            ++i; // skip the value
        }
        else if (argv[i] == spvIncludeTestDirectoryArg)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvIncludeTestDirectoryArg.c_str());
                return TEST_FAIL;
            }
            spvIncludeTestDirectory = std::string(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            ++i;
        }
        else if (argv[i] == spvSecondIncludeTestDirectoryArg)
        {
            if (i + 1 >= argc || argv[i + 1] == NULL)
            {
                log_error("Missing value for '%s' argument.\n",
                          spvSecondIncludeTestDirectoryArg.c_str());
                return TEST_FAIL;
            }
            spvSecondIncludeTestDirectory = std::string(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
            ++i;
        }
        else
        {
            argList.push_back(argv[i]);
        }
    }

    if (!modifiedSpvBinariesPath && !gListTests)
    {
        log_info("Reading SPIR-V files from default '%s' path.\n",
                 spvBinariesPath.c_str());
        log_info("In case you want to set other directory use '%s' argument.\n",
                 spvBinariesPathArg.c_str());
    }

    update_argc_argv_from_args_list(argList, argc, argv);
    return TEST_PASS;
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

    return runTestHarnessWithCheckAndParse(argc, argv, false, 0, nullptr,
                                           parseArgs);
}
