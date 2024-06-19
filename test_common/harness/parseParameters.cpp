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
#include "parseParameters.h"

#include "errorHelpers.h"
#include "testHarness.h"
#include "ThreadPool.h"

#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

using namespace std;

#define DEFAULT_COMPILATION_PROGRAM "cl_offline_compiler"
#define DEFAULT_SPIRV_VALIDATOR "spirv-val"

CompilationMode gCompilationMode = kOnline;
CompilationCacheMode gCompilationCacheMode = kCacheModeCompileIfAbsent;
std::string gCompilationCachePath = ".";
std::string gCompilationProgram = DEFAULT_COMPILATION_PROGRAM;
bool gDisableSPIRVValidation = false;
std::string gSPIRVValidator = DEFAULT_SPIRV_VALIDATOR;
unsigned gNumWorkerThreads;

void helpInfo()
{
    log_info(
        R"(Common options:
    -h, --help
        This help
    --compilation-mode <mode>
        Specify a compilation mode.  Mode can be:
            online     Use online compilation (default)
            binary     Use binary offline compilation
            spir-v     Use SPIR-V offline compilation
    --num-worker-threads <num>
        Select parallel execution with the specified number of worker threads.

For offline compilation (binary and spir-v modes) only:
    --compilation-cache-mode <cache-mode>
        Specify a compilation caching mode:
            compile-if-absent
                Read from cache if already populated, or else perform
                offline compilation (default)
            force-read
                Force reading from the cache
            overwrite
                Disable reading from the cache
            dump-cl-files
                Dumps the .cl and build .options files used by the test suite
    --compilation-cache-path <path>
        Path for offline compiler output and CL source
    --compilation-program <prog>
        Program to use for offline compilation, defaults to:
            )" DEFAULT_COMPILATION_PROGRAM R"(

For spir-v mode only:
    --disable-spirv-validation
        Disable validation of SPIR-V using the SPIR-V validator
    --spirv-validator
        Path for SPIR-V validator, defaults to )" DEFAULT_SPIRV_VALIDATOR "\n"
        "\n");
}

int parseCustomParam(int argc, const char *argv[], const char *ignore)
{
    int delArg = 0;

    for (int i = 1; i < argc; i++)
    {
        if (ignore != 0)
        {
            // skip parameters that require special/different treatment in
            // application (generic interpretation and parameter removal will
            // not be performed)
            const char *ptr = strstr(ignore, argv[i]);
            if (ptr != 0 && (ptr == ignore || ptr[-1] == ' ')
                && // first on list or ' ' before
                (ptr[strlen(argv[i])] == 0
                 || ptr[strlen(argv[i])] == ' ')) // last on list or ' ' after
                continue;
        }

        delArg = 0;

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            // Note: we don't increment delArg to delete this argument,
            // to allow the caller's argument parsing routine to see the
            // option and print its own help.
            helpInfo();
        }
        else if (!strcmp(argv[i], "--compilation-mode"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *mode = argv[i + 1];

                if (!strcmp(mode, "online"))
                {
                    gCompilationMode = kOnline;
                }
                else if (!strcmp(mode, "binary"))
                {
                    gCompilationMode = kBinary;
                }
                else if (!strcmp(mode, "spir-v"))
                {
                    gCompilationMode = kSpir_v;
                }
                else
                {
                    log_error("Compilation mode not recognized: %s\n", mode);
                    return -1;
                }
                log_info("Compilation mode specified: %s\n", mode);
            }
            else
            {
                log_error("Compilation mode parameters are incorrect. Usage:\n"
                          "  --compilation-mode <online|binary|spir-v>\n");
                return -1;
            }
        }
        else if (!strcmp(argv[i], "--num-worker-threads"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *numthstr = argv[i + 1];

                gNumWorkerThreads = atoi(numthstr);
            }
            else
            {
                log_error(
                    "A parameter to --num-worker-threads must be provided!\n");
                return -1;
            }
        }
        else if (!strcmp(argv[i], "--compilation-cache-mode"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *mode = argv[i + 1];

                if (!strcmp(mode, "compile-if-absent"))
                {
                    gCompilationCacheMode = kCacheModeCompileIfAbsent;
                }
                else if (!strcmp(mode, "force-read"))
                {
                    gCompilationCacheMode = kCacheModeForceRead;
                }
                else if (!strcmp(mode, "overwrite"))
                {
                    gCompilationCacheMode = kCacheModeOverwrite;
                }
                else if (!strcmp(mode, "dump-cl-files"))
                {
                    gCompilationCacheMode = kCacheModeDumpCl;
                }
                else
                {
                    log_error("Compilation cache mode not recognized: %s\n",
                              mode);
                    return -1;
                }
                log_info("Compilation cache mode specified: %s\n", mode);
            }
            else
            {
                log_error(
                    "Compilation cache mode parameters are incorrect. Usage:\n"
                    "  --compilation-cache-mode "
                    "<compile-if-absent|force-read|overwrite>\n");
                return -1;
            }
        }
        else if (!strcmp(argv[i], "--compilation-cache-path"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gCompilationCachePath = argv[i + 1];
            }
            else
            {
                log_error("Path argument for --compilation-cache-path was not "
                          "specified.\n");
                return -1;
            }
        }
        else if (!strcmp(argv[i], "--compilation-program"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gCompilationProgram = argv[i + 1];
            }
            else
            {
                log_error("Program argument for --compilation-program was not "
                          "specified.\n");
                return -1;
            }
        }
        else if (!strcmp(argv[i], "--disable-spirv-validation"))
        {
            delArg++;
            gDisableSPIRVValidation = true;
        }
        else if (!strcmp(argv[i], "--spirv-validator"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gSPIRVValidator = argv[i + 1];
            }
            else
            {
                log_error("Program argument for --spirv-validator was not "
                          "specified.\n");
                return -1;
            }
        }

        // cleaning parameters from argv tab
        for (int j = i; j < argc - delArg; j++) argv[j] = argv[j + delArg];
        argc -= delArg;
        i -= delArg;
    }

    if ((gCompilationCacheMode == kCacheModeForceRead
         || gCompilationCacheMode == kCacheModeOverwrite)
        && gCompilationMode == kOnline)
    {
        log_error("Compilation cache mode can only be specified when using an "
                  "offline compilation mode.\n");
        return -1;
    }

    return argc;
}

bool is_power_of_two(int number) { return number && !(number & (number - 1)); }

extern void parseWimpyReductionFactor(const char *&arg,
                                      int &wimpyReductionFactor)
{
    const char *arg_temp = strchr(&arg[1], ']');
    if (arg_temp != 0)
    {
        int new_factor = atoi(&arg[1]);
        arg = arg_temp; // Advance until ']'
        if (is_power_of_two(new_factor))
        {
            log_info("\n Wimpy reduction factor changed from %d to %d \n",
                     wimpyReductionFactor, new_factor);
            wimpyReductionFactor = new_factor;
        }
        else
        {
            log_info("\n WARNING: Incorrect wimpy reduction factor %d, must be "
                     "power of 2. The default value will be used.\n",
                     new_factor);
        }
    }
}
