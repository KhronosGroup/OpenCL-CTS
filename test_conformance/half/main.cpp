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
#include <string.h>

#if !defined (_WIN32)
#include <sys/resource.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <libgen.h>
#include <sys/param.h>
#endif

#include "harness/testHarness.h"
#include "harness/mingw_compat.h"
#include "harness/parseParameters.h"
#if defined (__MINGW32__)
#include <sys/param.h>
#endif

#include "cl_utils.h"
#include "tests.h"

const char *addressSpaceNames[AS_NumAddressSpaces] = {"global", "private", "local", "constant"};

#pragma mark -
#pragma mark Declarations


static test_status ParseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help);

int g_arrVecSizes[kVectorSizeCount+kStrangeVectorSizeCount];
int g_arrVecAligns[kLargestVectorSize+1];
static int arrStrangeVecSizes[kStrangeVectorSizeCount] = {3};

int main (int argc, const char **argv )
{
    int error;
    int i;
    int alignbound;

    for(i = 0; i < kVectorSizeCount; ++i) {
      g_arrVecSizes[i] = (1<<i);
    }
    for(i = 0; i < kStrangeVectorSizeCount; ++i) {
      g_arrVecSizes[i+kVectorSizeCount] =
      arrStrangeVecSizes[i];
    }

    for(i = 0, alignbound=1; i <= kLargestVectorSize; ++i) {
        while(alignbound < i) {
            alignbound = alignbound<<1;
        }
        g_arrVecAligns[i] = alignbound;
    }

    fflush( stdout );
    error =
        runTestHarnessWithCheckAndParse(argc, argv, true, 0, InitCL, ParseArgs);

    if(gQueue)
    {
        int flush_error = clFinish(gQueue);
        if(flush_error)
        {
            vlog_error("clFinish failed: %d\n", flush_error);
        }
    }

    ReleaseCL();
    return error;
}

#pragma mark -
#pragma mark setup

static test_status ParseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{

    help =
        R"(        -d     Toggle double precision testing (default: on if double supported)
        -t     Toggle reporting performance data.
        -r     Reset buffers on host instead of on device.
        -[2^n] Set wimpy reduction factor, recommended range of n is 1-12, default factor()"
        + std::to_string(gWimpyReductionFactor) + ")\n";

    std::vector<const char *> argList;
    argList.push_back(argv[0]);

    for (int i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        if( NULL == arg )
            break;

        if( arg[0] == '-' )
        {
            arg++;
            while( *arg != '\0' )
            {
                switch( *arg )
                {
                    case 'd':
                        gTestDouble ^= 1;
                        break;

                    case 't':
                        gReportTimes ^= 1;
                        break;

                    case 'r': gHostReset = true; break;

                    case '[':
                        parseWimpyReductionFactor( arg, gWimpyReductionFactor);
                        break;
                    default:
                        vlog_error( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        return TEST_FAIL;
                }
                arg++;
            }
            removed_args.push_back(argv[i]);
        }
        else
        {
            argList.push_back(argv[i]);
        }
    }
    update_argc_argv_from_args_list(argList, argc, argv);

    if( gWimpyMode )
    {
        vlog( "\n" );
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
        vlog( "*** It gives warm fuzzy feelings and then nevers calls. ***\n\n" );
        vlog( "*** Wimpy Reduction Factor: %-27u ***\n\n", gWimpyReductionFactor);
    }

    return TEST_PASS;
}
