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

bool             gOfflineCompiler = false;
bool             gForceSpirVCache = false;
bool             gForceSpirVGenerate = false;
std::string      gSpirVPath = ".";
OfflineCompilerOutputType gOfflineCompilerOutputType;

void helpInfo ()
{
  log_info("  '-ILPath path_to_spirv_bin.\n");
  log_info("  '-offlineCompiler <output_type:binary|source|spir_v>': use offline compiler\n");
  log_info("  '                  output_type binary - \"../build_script_binary.py\" is invoked\n");
  log_info("  '                  output_type source - \"../build_script_source.py\"  is invoked\n");
  log_info("  '                  output_type spir_v <mode:generate|cache> - \"../cl_build_script_spir_v.py\" is invoked. optional modes: generate, cache\n");
  log_info("  '                                     mode generate <path> - force binary generation\n");
  log_info("  '                                     mode cache <path> - force reading binary files from cache\n");
  log_info("\n");
}

int parseCustomParam (int argc, const char *argv[], const char *ignore)
{
  int delArg = 0;

  for (int i=1; i<argc; i++)
  {
    if(ignore != 0)
    {
      // skip parameters that require special/different treatment in application
      // (generic interpretation and parameter removal will not be performed)
      const char * ptr = strstr(ignore, argv[i]);
      if(ptr != 0 &&
        (ptr == ignore || ptr[-1] == ' ') && //first on list or ' ' before
        (ptr[strlen(argv[i])] == 0 || ptr[strlen(argv[i])] == ' ')) // last on list or ' ' after
        continue;
    }
    if (i < 0) i = 0;
    delArg = 0;

    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
    {
        helpInfo ();
    }
    else if (!strcmp(argv[i], "-ILPath"))
    {
        gSpirVPath = argv[i + 1];
        delArg = 2;
    }
    else if (!strcmp(argv[i], "-offlineCompiler"))
    {
        log_info(" Offline Compiler enabled\n");
        delArg = 1;
        if ((i + 1) < argc)
        {
            gOfflineCompiler = true;

            if (!strcmp(argv[i + 1], "binary"))
            {
                gOfflineCompilerOutputType = kBinary;
                delArg++;
            }
            else if (!strcmp(argv[i + 1], "source"))
            {
                gOfflineCompilerOutputType = kSource;
                delArg++;
            }
            else if (!strcmp(argv[i + 1], "spir_v"))
            {
                gOfflineCompilerOutputType = kSpir_v;
                delArg++;
                if ((i + 3) < argc)
                {
                    if (!strcmp(argv[i + 2], "cache"))
                    {
                        gForceSpirVCache = true;
                        gSpirVPath = argv[i + 3];
                        log_info(" SpirV reading from cache enabled.\n");
                        delArg += 2;
                    }
                    else if (!strcmp(argv[i + 2], "generate"))
                    {
                        gForceSpirVGenerate = true;
                        gSpirVPath = argv[i + 3];
                        log_info(" SpirV force generate binaries enabled.\n");
                        delArg += 2;
                    }
                }
            }
            else
            {
                log_error(" Offline Compiler output type not supported: %s\n", argv[i + 1]);
                return -1;
            }
        }
        else
        {
            log_error(" Offline Compiler parameters are incorrect. Usage:\n");
            log_error("       -offlineCompiler <input> <output> <output_type:binary | source | spir_v>\n");
            return -1;
        }
    }

    //cleaning parameters from argv tab
	  for (int j=i; j<argc-delArg; j++)
		  argv[j] = argv[j+delArg];
	  argc -= delArg ;
	  i -= delArg;
  }
  return argc;
}

bool is_power_of_two(int number)
{
    return number && !(number & (number - 1));
}

extern void parseWimpyReductionFactor(const char *&arg, int &wimpyReductionFactor)
{
    const char *arg_temp = strchr(&arg[1], ']');
    if (arg_temp != 0)
    {
        int new_factor = atoi(&arg[1]);
        arg = arg_temp; // Advance until ']'
        if (is_power_of_two(new_factor))
        {
            log_info("\n Wimpy reduction factor changed from %d to %d \n", wimpyReductionFactor, new_factor);
            wimpyReductionFactor = new_factor;
        }
        else
        {
            log_info("\n WARNING: Incorrect wimpy reduction factor %d, must be power of 2. The default value will be used.\n", new_factor);
        }
    }
}
