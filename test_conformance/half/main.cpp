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

const char **   argList = NULL;
size_t          argCount = 0;
char            appName[64] = "ctest";
const char *addressSpaceNames[AS_NumAddressSpaces] = {"global", "private", "local", "constant"};

#pragma mark -
#pragma mark Declarations


static int ParseArgs( int argc, const char **argv );
static void PrintUsage( void );


int g_arrVecSizes[kVectorSizeCount+kStrangeVectorSizeCount];
int g_arrVecAligns[kLargestVectorSize+1];
static int arrStrangeVecSizes[kStrangeVectorSizeCount] = {3};

test_definition test_list[] = {
    ADD_TEST( vload_half ),
    ADD_TEST( vloada_half ),
    ADD_TEST( vstore_half ),
    ADD_TEST( vstorea_half ),
    ADD_TEST( vstore_half_rte ),
    ADD_TEST( vstorea_half_rte ),
    ADD_TEST( vstore_half_rtz ),
    ADD_TEST( vstorea_half_rtz ),
    ADD_TEST( vstore_half_rtp ),
    ADD_TEST( vstorea_half_rtp ),
    ADD_TEST( vstore_half_rtn ),
    ADD_TEST( vstorea_half_rtn ),
    ADD_TEST( roundTrip ),
};

const int test_num = ARRAY_SIZE( test_list );

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

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return -1;
    }

    if( (error = ParseArgs( argc, argv )) )
        goto exit;

    if (gIsEmbedded) {
        vlog( "\tProfile: Embedded\n" );
    }else
    {
        vlog( "\tProfile: Full\n" );
    }

    fflush( stdout );
    error = runTestHarnessWithCheck( argCount, argList, test_num, test_list, true, 0, InitCL );

exit:
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

static int ParseArgs( int argc, const char **argv )
{
    int i;
    argList = (const char **)calloc(argc, sizeof(char *));
    if( NULL == argList )
    {
        vlog_error( "Failed to allocate memory for argList.\n" );
        return 1;
    }

    argList[0] = argv[0];
    argCount = 1;

#if (defined( __APPLE__ ) || defined(__linux__) || defined(__MINGW32__))
    { // Extract the app name
        char baseName[ MAXPATHLEN ];
        strncpy( baseName, argv[0], MAXPATHLEN );
        char *base = basename( baseName );
        if( NULL != base )
        {
            strncpy( appName, base, sizeof( appName )  );
            appName[ sizeof( appName ) -1 ] = '\0';
        }
    }
#elif defined (_WIN32)
    {
        char fname[_MAX_FNAME + _MAX_EXT + 1];
        char ext[_MAX_EXT];

        errno_t err = _splitpath_s( argv[0], NULL, 0, NULL, 0,
                                   fname, _MAX_FNAME, ext, _MAX_EXT );
        if (err == 0) { // no error
            strcat (fname, ext); //just cat them, size of frame can keep both
            strncpy (appName, fname, sizeof(appName));
            appName[ sizeof( appName ) -1 ] = '\0';
        }
    }
#endif

    vlog( "\n%s", appName );
    for( i = 1; i < argc; i++ )
    {
        const char *arg = argv[i];
        if( NULL == arg )
            break;

        vlog( "\t%s", arg );
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

                    case 'h':
                        PrintUsage();
                        return -1;

                    case 't':
                        gReportTimes ^= 1;
                        break;

                    case 'w':  // Wimpy mode
                        gWimpyMode = true;
                        break;
                    case '[':
                        parseWimpyReductionFactor( arg, gWimpyReductionFactor);
                        break;
                    default:
                        vlog_error( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        PrintUsage();
                        return -1;
                }
                arg++;
            }
        }
        else
        {
            argList[ argCount ] = arg;
            argCount++;
        }
    }

    if (getenv("CL_WIMPY_MODE")) {
      vlog( "\n" );
      vlog( "*** Detected CL_WIMPY_MODE env                          ***\n" );
      gWimpyMode = 1;
    }

    PrintArch();
    if( gWimpyMode )
    {
        vlog( "\n" );
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
        vlog( "*** It gives warm fuzzy feelings and then nevers calls. ***\n\n" );
        vlog( "*** Wimpy Reduction Factor: %-27u ***\n\n", gWimpyReductionFactor);
    }
    return 0;
}

static void PrintUsage( void )
{
    vlog( "%s [-dthw]: <optional: test names>\n", appName );
    vlog( "\t\t-d\tToggle double precision testing (default: on if double supported)\n" );
    vlog( "\t\t-t\tToggle reporting performance data.\n" );
    vlog( "\t\t-w\tRun in wimpy mode\n" );
    vlog( "\t\t-[2^n]\tSet wimpy reduction factor, recommended range of n is 1-12, default factor(%u)\n", gWimpyReductionFactor);
    vlog( "\t\t-h\tHelp\n" );
    for( int i = 0; i < test_num; i++ )
    {
        vlog("\t\t%s\n", test_list[i].name );
    }
}
