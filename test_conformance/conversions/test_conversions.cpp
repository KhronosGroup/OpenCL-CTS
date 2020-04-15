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
#include "harness/rounding_mode.h"
#include "harness/ThreadPool.h"
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/parseParameters.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include <sys/sysctl.h>
#endif

#if defined( __linux__ )
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

#include "mingw_compat.h"
#if defined(__MINGW32__)
#include <sys/param.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include "Sleep.h"
#include "basic_test_conversions.h"

#pragma STDC FENV_ACCESS ON

#if (defined(_WIN32) && defined (_MSC_VER))
// need for _controlfp_s and rouinding modes in RoundingMode
#include "harness/testHarness.h"
#endif

#pragma mark -
#pragma mark globals

#define BUFFER_SIZE     (1024*1024)
#define kPageSize       4096
#define EMBEDDED_REDUCTION_FACTOR 16
#define PERF_LOOP_COUNT 100

#define      kCallStyleCount (kVectorSizeCount + 1 /* for implicit scalar */)

#if defined( __arm__ ) && defined( __GNUC__ )
#include "fplib.h"
    extern bool            qcom_sat;
    extern roundingMode    qcom_rm;
#endif

const char **   argList = NULL;
int             argCount = 0;
cl_context      gContext = NULL;
cl_command_queue      gQueue = NULL;
char            appName[64] = "ctest";
int             gStartTestNumber = -1;
int             gEndTestNumber = 0;
#if defined( __APPLE__ )
int             gTimeResults = 1;
#else
int             gTimeResults = 0;
#endif
int             gReportAverageTimes = 0;
void            *gIn = NULL;
void            *gRef = NULL;
void        *gAllowZ = NULL;
void            *gOut[ kCallStyleCount ] = { NULL };
cl_mem          gInBuffer;
cl_mem          gOutBuffers[ kCallStyleCount ];
size_t          gComputeDevices = 0;
uint32_t        gDeviceFrequency = 0;
int             gWimpyMode = 0;
int             gWimpyReductionFactor = 128;
int             gSkipTesting = 0;
int             gForceFTZ = 0;
int             gMultithread = 1;
int             gIsRTZ = 0;
uint32_t        gSimdSize = 1;
int             gHasDouble = 0;
int             gTestDouble = 1;
const char *    sizeNames[] = { "", "", "2", "3", "4", "8", "16" };
const int       vectorSizes[] = { 1, 1, 2, 3, 4, 8, 16 };
int             gMinVectorSize = 0;
int             gMaxVectorSize = sizeof(vectorSizes) / sizeof( vectorSizes[0] );
static MTdata   gMTdata;

#pragma mark -
#pragma mark Declarations

static int ParseArgs( int argc, const char **argv );
static void PrintUsage( void );
test_status InitCL( cl_device_id device );
static int GetTestCase( const char *name, Type *outType, Type *inType, SaturationMode *sat, RoundingMode *round );
static int DoTest( cl_device_id device, Type outType, Type inType, SaturationMode sat, RoundingMode round, MTdata d );
static cl_program   MakeProgram( Type outType, Type inType, SaturationMode sat, RoundingMode round, int vectorSize, cl_kernel *outKernel );
static int RunKernel( cl_kernel kernel, void *inBuf, void *outBuf, size_t blockCount );

void *FlushToZero( void );
void UnFlushToZero( void *);

static cl_program CreateImplicitConvertProgram( Type outType, Type inType, SaturationMode sat, RoundingMode round, int vectorSize, char testName[256], cl_int *error );
static cl_program CreateStandardProgram( Type outType, Type inType, SaturationMode sat, RoundingMode round, int vectorSize, char testName[256], cl_int *error );


// Windows (since long double got deprecated) sets the x87 to 53-bit precision
// (that's x87 default state).  This causes problems with the tests that
// convert long and ulong to float and double or otherwise deal with values
// that need more precision than 53-bit. So, set the x87 to 64-bit precision.
static inline void Force64BitFPUPrecision(void)
{
#if __MINGW32__
    // The usual method is to use _controlfp as follows:
    //     #include <float.h>
    //     _controlfp(_PC_64, _MCW_PC);
    //
    // _controlfp is available on MinGW32 but not on MinGW64. Instead of having
    // divergent code just use inline assembly which works for both.
    unsigned short int orig_cw = 0;
    unsigned short int new_cw = 0;
    __asm__ __volatile__ ("fstcw %0":"=m" (orig_cw));
    new_cw = orig_cw | 0x0300;   // set precision to 64-bit
    __asm__ __volatile__ ("fldcw  %0"::"m" (new_cw));
#else
    /* Implement for other platforms if needed */
#endif
}

int test_conversions( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    int error, i, testNumber = -1;
    int startMinVectorSize = gMinVectorSize;
    Type inType, outType;
    RoundingMode round;
    SaturationMode sat;

    if( argCount )
    {
        for( i = 0; i < argCount; i++ )
        {
            if( GetTestCase( argList[i], &outType, &inType, &sat, &round ) )
            {
                vlog_error( "\n\t\t**** ERROR:  Unable to parse function name %s.  Skipping....  *****\n\n", argList[i] );
                continue;
            }

            // skip double if we don't have it
            if( !gTestDouble && (inType == kdouble || outType == kdouble ) )
            {
                if( gHasDouble )
                {
                    vlog_error( "\t *** convert_%sn%s%s( %sn ) FAILED ** \n", gTypeNames[ outType ], gSaturationNames[ sat ], gRoundingModeNames[round], gTypeNames[inType] );
                    vlog( "\t\tcl_khr_fp64 enabled, but double testing turned off.\n" );
                }

                continue;
            }

            // skip longs on embedded
            if( !gHasLong && (inType == klong || outType == klong || inType == kulong || outType == kulong) )
            {
                continue;
            }

            // Skip the implicit converts if the rounding mode is not default or test is saturated
            if( 0 == startMinVectorSize )
            {
                if( sat || round != kDefaultRoundingMode )
                    gMinVectorSize = 1;
                else
                    gMinVectorSize = 0;
            }

            if( ( error = DoTest( device, outType, inType, sat, round, gMTdata ) ) )
            {
                vlog_error( "\t *** convert_%sn%s%s( %sn ) FAILED ** \n", gTypeNames[outType], gSaturationNames[sat], gRoundingModeNames[round], gTypeNames[inType] );
            }
        }
    }
    else
    {
        for( outType = (Type)0; outType < kTypeCount; outType = (Type)(outType+1) )
        {
            for( inType = (Type)0; inType < kTypeCount; inType = (Type)(inType+1) )
            {
                // skip longs on embedded
                if( !gHasLong && (inType == klong || outType == klong || inType == kulong || outType == kulong) )
                {
                    continue;
                }

                for( sat = (SaturationMode)0; sat < kSaturationModeCount; sat = (SaturationMode)(sat+1) )
                {
                    //skip illegal saturated conversions to float type
                    if( kSaturated == sat && ( outType == kfloat || outType == kdouble ) )
                    {
                        continue;
                    }

                    for( round = (RoundingMode)0; round < kRoundingModeCount; round = (RoundingMode)(round+1) )
                    {
                        if( ++testNumber < gStartTestNumber )
                        {
                            //     vlog( "%d) skipping convert_%sn%s%s( %sn )\n", testNumber, gTypeNames[ outType ], gSaturationNames[ sat ], gRoundingModeNames[round], gTypeNames[inType] );
                            continue;
                        }
                        else
                        {
                            if( gEndTestNumber > 0 && testNumber >= gEndTestNumber  )
                            {
                                goto exit;
                            }
                        }

                        vlog( "%d) Testing convert_%sn%s%s( %sn ):\n", testNumber, gTypeNames[ outType ], gSaturationNames[ sat ], gRoundingModeNames[round], gTypeNames[inType] );

                        // skip double if we don't have it
                        if( ! gTestDouble && (inType == kdouble || outType == kdouble ) )
                        {
                            if( gHasDouble )
                            {
                                vlog_error( "\t *** %d) convert_%sn%s%s( %sn ) FAILED ** \n", testNumber, gTypeNames[ outType ], gSaturationNames[ sat ], gRoundingModeNames[round], gTypeNames[inType] );
                                vlog( "\t\tcl_khr_fp64 enabled, but double testing turned off.\n" );
                            }
                            continue;
                        }

                        // Skip the implicit converts if the rounding mode is not default or test is saturated
                        if( 0 == startMinVectorSize )
                        {
                            if( sat || round != kDefaultRoundingMode )
                                gMinVectorSize = 1;
                            else
                                gMinVectorSize = 0;
                        }

                        if( ( error = DoTest( device, outType, inType, sat, round, gMTdata ) ) )
                        {
                            vlog_error( "\t *** %d) convert_%sn%s%s( %sn ) FAILED ** \n", testNumber, gTypeNames[outType], gSaturationNames[sat], gRoundingModeNames[round], gTypeNames[inType] );
                        }
                    }
                }
            }
        }
    }

exit:
    return gFailCount;
}

test_definition test_list[] = {
    ADD_TEST( conversions ),
};

const int test_num = ARRAY_SIZE( test_list );

#pragma mark -

int main (int argc, const char **argv )
{
    int error;
    cl_uint seed = (cl_uint) time( NULL );

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return 1;
    }

    if( (error = ParseArgs( argc, argv )) )
        return error;

    //Turn off sleep so our tests run to completion
    PreventSleep();
    atexit( ResumeSleep );

    if(!gMultithread)
        SetThreadCount(1);

#if defined(_MSC_VER) && defined(_M_IX86)
    // VS2005 (and probably others, since long double got deprecated) sets
    // the x87 to 53-bit precision. This causes problems with the tests
    // that convert long and ulong to float and double, since they deal
    // with values that need more precision than that. So, set the x87
    // to 64-bit precision.
    unsigned int ignored;
    _controlfp_s(&ignored, _PC_64, _MCW_PC);
#endif

    vlog( "===========================================================\n" );
    vlog( "Random seed: %u\n", seed );
    gMTdata = init_genrand( seed );

    const char* arg[] = {argv[0]};
    int ret = runTestHarnessWithCheck( 1, arg, test_num, test_list, true, 0, InitCL );

    free_mtdata( gMTdata );

    error = clFinish(gQueue);
    if (error)
        vlog_error("clFinish failed: %d\n", error);

    clReleaseMemObject(gInBuffer);

    for( int i = 0; i < kCallStyleCount; i++ ) {
        clReleaseMemObject(gOutBuffers[i]);
    }
    clReleaseCommandQueue(gQueue);
    clReleaseContext(gContext);

    return ret;
}

#pragma mark -
#pragma mark setup

static int ParseArgs( int argc, const char **argv )
{
    int i;
    argList = (const char **)calloc( argc - 1, sizeof( char*) );
    argCount = 0;

    if( NULL == argList && argc > 1 )
        return -1;

#if (defined( __APPLE__ ) || defined(__linux__) || defined (__MINGW32__))
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
                    case 'l':
                        gSkipTesting ^= 1;
                        break;
                    case 'm':
                        gMultithread ^= 1;
                        break;
                    case 'w':
                        gWimpyMode ^= 1;
                        break;
                    case '[':
                        parseWimpyReductionFactor(arg, gWimpyReductionFactor);
                        break;
                    case 'z':
                        gForceFTZ ^= 1;
                        break;
                    case 't':
                        gTimeResults ^= 1;
                        break;
                    case 'a':
                        gReportAverageTimes ^= 1;
                        break;
                    case '1':
                        if( arg[1] == '6' )
                        {
                            gMinVectorSize = 6;
                            gMaxVectorSize = 7;
                            arg++;
                        }
                        else
                        {
                            gMinVectorSize = 0;
                            gMaxVectorSize = 2;
                        }
                        break;

                    case '2':
                        gMinVectorSize = 2;
                        gMaxVectorSize = 3;
                        break;

                    case '3':
                        gMinVectorSize = 3;
                        gMaxVectorSize = 4;
                        break;

                    case '4':
                        gMinVectorSize = 4;
                        gMaxVectorSize = 5;
                        break;

                    case '8':
                        gMinVectorSize = 5;
                        gMaxVectorSize = 6;
                        break;

                    default:
                        vlog( " <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg );
                        PrintUsage();
                        return -1;
                }
                arg++;
            }
        }
        else
        {
            char *t = NULL;
            long number = strtol( arg, &t, 0 );
            if( t != arg )
            {
                if( gStartTestNumber != -1 )
                    gEndTestNumber = gStartTestNumber + (int) number;
                else
                    gStartTestNumber = (int) number;
            }
            else
            {
                argList[ argCount ] = arg;
                argCount++;
            }
        }
    }

    // Check for the wimpy mode environment variable
    if (getenv("CL_WIMPY_MODE")) {
      vlog( "\n" );
      vlog( "*** Detected CL_WIMPY_MODE env                          ***\n" );
      gWimpyMode = 1;
    }

    vlog( "\n" );

    vlog( "Test binary built %s %s\n", __DATE__, __TIME__ );

    PrintArch();

    if( gWimpyMode )
    {
        vlog( "\n" );
        vlog( "*** WARNING: Testing in Wimpy mode!                     ***\n" );
        vlog( "*** Wimpy mode is not sufficient to verify correctness. ***\n" );
        vlog( "*** It gives warm fuzzy feelings and then nevers calls. ***\n\n" );
        vlog("*** Wimpy Reduction Factor: %-27u ***\n\n", gWimpyReductionFactor);
    }

    return 0;
}

static void PrintUsage( void )
{
    int i;
    vlog( "%s [-wz#]: <optional: test names>\n", appName );
    vlog( "\ttest names:\n" );
    vlog( "\t\tdestFormat<_sat><_round>_sourceFormat\n" );
    vlog( "\t\t\tPossible format types are:\n\t\t\t\t" );
    for( i = 0; i < kTypeCount; i++ )
        vlog( "%s, ", gTypeNames[i] );
    vlog( "\n\n\t\t\tPossible saturation values are: (empty) and _sat\n" );
    vlog( "\t\t\tPossible rounding values are:\n\t\t\t\t(empty), " );
    for( i = 1; i < kRoundingModeCount; i++ )
        vlog( "%s, ", gRoundingModeNames[i] );
    vlog( "\n\t\t\tExamples:\n" );
    vlog( "\t\t\t\tulong_short   converts short to ulong\n" );
    vlog( "\t\t\t\tchar_sat_rte_float   converts float to char with saturated clipping in round to nearest rounding mode\n\n" );
    vlog( "\toptions:\n" );
    vlog( "\t\t-d\tToggle testing of double precision.  On by default if cl_khr_fp64 is enabled, ignored otherwise.\n" );
    vlog( "\t\t-l\tToggle link check mode. When on, testing is skipped, and we just check to see that the kernels build. (Off by default.)\n" );
    vlog( "\t\t-m\tToggle Multithreading. (On by default.)\n" );
    vlog( "\t\t-w\tToggle wimpy mode. When wimpy mode is on, we run a very small subset of the tests for each fn. NOT A VALID TEST! (Off by default.)\n" );
    vlog(" \t\t-[2^n]\tSet wimpy reduction factor, recommended range of n is 1-12, default factor(%u)\n", gWimpyReductionFactor);
    vlog( "\t\t-z\tToggle flush to zero mode  (Default: per device)\n" );
    vlog( "\t\t-#\tTest just vector size given by #, where # is an element of the set {1,2,3,4,8,16}\n" );
    vlog( "\n" );
    vlog( "You may also pass the number of the test on which to start.\nA second number can be then passed to indicate how many tests to run\n\n" );
}


static int GetTestCase( const char *name, Type *outType, Type *inType, SaturationMode *sat, RoundingMode *round )
{
    int i;

    //Find the return type
    for( i = 0; i < kTypeCount; i++ )
        if( name == strstr( name, gTypeNames[i] ) )
        {
            *outType = (Type)i;
            name += strlen( gTypeNames[i] );

            break;
        }

    if( i == kTypeCount )
        return -1;

    // Check to see if _sat appears next
    *sat = (SaturationMode)0;
    for( i = 1; i < kSaturationModeCount; i++ )
        if( name == strstr( name, gSaturationNames[i] ) )
        {
            *sat = (SaturationMode)i;
            name += strlen( gSaturationNames[i] );
            break;
        }

    *round = (RoundingMode)0;
    for( i = 1; i < kRoundingModeCount; i++ )
        if( name == strstr( name, gRoundingModeNames[i] ) )
        {
            *round = (RoundingMode)i;
            name += strlen( gRoundingModeNames[i] );
            break;
        }

    if( *name != '_' )
        return -2;
    name++;

    for( i = 0; i < kTypeCount; i++ )
        if( name == strstr( name, gTypeNames[i] ) )
        {
            *inType = (Type)i;
            name += strlen( gTypeNames[i] );

            break;
        }

    if( i == kTypeCount )
        return -3;

    if( *name != '\0' )
        return -4;

    return 0;
}

#pragma mark -
#pragma mark OpenCL

test_status InitCL( cl_device_id device )
{
    int error, i;
    size_t configSize = sizeof( gComputeDevices );

    if( (error = clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, configSize, &gComputeDevices, NULL )) )
        gComputeDevices = 1;

    configSize = sizeof( gDeviceFrequency );
    if( (error = clGetDeviceInfo( device, CL_DEVICE_MAX_CLOCK_FREQUENCY, configSize, &gDeviceFrequency, NULL )) )
        gDeviceFrequency = 0;

    cl_device_fp_config floatCapabilities = 0;
    if( (error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(floatCapabilities), &floatCapabilities,  NULL)))
        floatCapabilities = 0;
    if(0 == (CL_FP_DENORM & floatCapabilities) )
        gForceFTZ ^= 1;

    if( 0 == (floatCapabilities & CL_FP_ROUND_TO_NEAREST ) )
    {
        char profileStr[128] = "";
        // Verify that we are an embedded profile device
        if( (error = clGetDeviceInfo( device, CL_DEVICE_PROFILE, sizeof( profileStr ), profileStr, NULL ) ) )
        {
            vlog_error( "FAILURE: Could not get device profile: error %d\n", error );
            return TEST_FAIL;
        }

        if( strcmp( profileStr, "EMBEDDED_PROFILE" ) )
        {
            vlog_error( "FAILURE: non-embedded profile device does not support CL_FP_ROUND_TO_NEAREST\n" );
            return TEST_FAIL;
        }

        if( 0 == (floatCapabilities & CL_FP_ROUND_TO_ZERO ) )
        {
            vlog_error( "FAILURE: embedded profile device supports neither CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n" );
            return TEST_FAIL;
        }

        gIsRTZ = 1;
    }

    else if(is_extension_available(device, "cl_khr_fp64"))
    {
        gHasDouble = 1;
    }
    gTestDouble &= gHasDouble;

    //detect whether profile of the device is embedded
    char profile[1024] = "";
    if( (error = clGetDeviceInfo( device, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL ) ) ){}
    else if( strstr(profile, "EMBEDDED_PROFILE" ) )
    {
        gIsEmbedded = 1;
        if( !is_extension_available(device, "cles_khr_int64" ) )
            gHasLong = 0;
    }


    gContext = clCreateContext( NULL, 1, &device, notify_callback, NULL, &error );
    if( NULL == gContext || error )
    {
        vlog_error( "clCreateContext failed. (%d)\n", error );
        return TEST_FAIL;
    }

    gQueue = clCreateCommandQueue(gContext, device, 0, &error);
    if( NULL == gQueue || error )
    {
        vlog_error( "clCreateCommandQueue failed. (%d)\n", error );
        return TEST_FAIL;
    }

    //Allocate buffers
    //FIXME: use clProtectedArray for guarded allocations?
    gIn   = malloc( BUFFER_SIZE + 2 * kPageSize );
    gAllowZ = malloc( BUFFER_SIZE + 2 * kPageSize );
    gRef  = malloc( BUFFER_SIZE + 2 * kPageSize );
    for( i = 0; i < kCallStyleCount; i++ )
    {
        gOut[i] = malloc( BUFFER_SIZE + 2 * kPageSize );
        if( NULL == gOut[i] )
            return TEST_FAIL;
    }

    // setup input buffers
    gInBuffer = clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, BUFFER_SIZE, NULL, &error);
    if( gInBuffer == NULL || error)
    {
        vlog_error( "clCreateBuffer failed for input (%d)\n", error );
        return TEST_FAIL;
    }

    // setup output buffers
    for( i = 0; i < kCallStyleCount; i++ )
    {
        gOutBuffers[i] = clCreateBuffer(  gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, BUFFER_SIZE, NULL, &error );
        if( gOutBuffers[i] == NULL || error )
        {
            vlog_error( "clCreateArray failed for output (%d)\n", error );
            return TEST_FAIL;
        }
    }


    gMTdata = init_genrand( gRandomSeed );


    char c[1024];
    static const char *no_yes[] = { "NO", "YES" };
    vlog( "\nCompute Device info:\n" );
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(c), c, NULL);
    vlog( "\tDevice Name: %s\n", c );
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(c), c, NULL);
    vlog( "\tVendor: %s\n", c );
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(c), c, NULL);
    vlog( "\tDevice Version: %s\n", c );
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(c), &c, NULL);
    vlog( "\tCL C Version: %s\n", c );
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(c), c, NULL);
    vlog( "\tDriver Version: %s\n", c );
    vlog( "\tProcessing with %ld devices\n", gComputeDevices );
    vlog( "\tDevice Frequency: %d MHz\n", gDeviceFrequency );
    vlog( "\tSubnormal values supported for floats? %s\n", no_yes[0 != (CL_FP_DENORM & floatCapabilities)] );
    vlog( "\tTesting with FTZ mode ON for floats? %s\n", no_yes[0 != gForceFTZ] );
    vlog( "\tTesting with default RTZ mode for floats? %s\n", no_yes[0 != gIsRTZ] );
    vlog( "\tHas Double? %s\n", no_yes[0 != gHasDouble] );
    if( gHasDouble )
        vlog( "\tTest Double? %s\n", no_yes[0 != gTestDouble] );
    vlog( "\tHas Long? %s\n", no_yes[0 != gHasLong] );
    vlog( "\tTesting vector sizes: " );
    for( i = gMinVectorSize; i < gMaxVectorSize; i++ )
        vlog("\t%d", vectorSizes[i]);
    vlog( "\n" );
    return TEST_PASS;
}

static int RunKernel( cl_kernel kernel, void *inBuf, void *outBuf, size_t blockCount )
{
    // The global dimensions are just the blockCount to execute since we haven't set up multiple queues for multiple devices.
    int error;

    error = clSetKernelArg(kernel, 0, sizeof( inBuf ), &inBuf);
    error |= clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf);

    if( error )
    {
        vlog_error( "FAILED -- could not set kernel args (%d)\n", error );
        return error;
    }

    if( (error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &blockCount, NULL, 0, NULL, NULL)))
    {
        vlog_error( "FAILED -- could not execute kernel (%d)\n", error );
        return error;
    }

    return 0;
}

#if ! defined( __APPLE__ )
void memset_pattern4(void *dest, const void *src_pattern, size_t bytes );
#endif

#if defined( __APPLE__ )
#include <mach/mach_time.h>
#endif

uint64_t GetTime( void );
uint64_t GetTime( void )
{
#if defined( __APPLE__ )
    return mach_absolute_time();
#elif defined(_MSC_VER)
    return  ReadTime();
#else
    //mach_absolute_time is a high precision timer with precision < 1 microsecond.
#warning need accurate clock here.  Times are invalid.
    return 0;
#endif
}


#if defined (_MSC_VER)
/* function is defined in "compat.h" */
#else
double SubtractTime( uint64_t endTime, uint64_t startTime );
double SubtractTime( uint64_t endTime, uint64_t startTime )
{
    uint64_t diff = endTime - startTime;
    static double conversion = 0.0;

    if( 0.0 == conversion )
    {
#if defined( __APPLE__ )
        mach_timebase_info_data_t info = {0,0};
        kern_return_t   err = mach_timebase_info( &info );
        if( 0 == err )
            conversion = 1e-9 * (double) info.numer / (double) info.denom;
#else
        // This function consumes output from GetTime() above, and converts the time to secionds.
#warning need accurate ticks to seconds conversion factor here. Times are invalid.
#endif
    }

    // strictly speaking we should also be subtracting out timer latency here
    return conversion * (double) diff;
}
#endif

typedef struct CalcReferenceValuesInfo
{
    struct WriteInputBufferInfo *parent;        // pointer back to the parent WriteInputBufferInfo struct
    cl_kernel                   kernel;         // the kernel for this vector size
    cl_program                  program;        // the program for this vector size
    cl_uint                     vectorSize;     // the vector size for this callback chain
    void                        *p;             // the pointer to mapped result data for this vector size
    cl_int                      result;
}CalcReferenceValuesInfo;

typedef struct WriteInputBufferInfo
{
    volatile cl_event           calcReferenceValues;   // user event which signals when main thread is done calculating reference values
    volatile cl_event           doneBarrier;     // user event which signals when worker threads are done
    cl_uint                     count;           // the number of elements in the array
    Type                        outType;         // the data type of the conversion result
    Type                        inType;          // the data type of the conversion input
    volatile int                barrierCount;
    CalcReferenceValuesInfo     calcInfo[kCallStyleCount];
}WriteInputBufferInfo;

cl_uint RoundUpToNextPowerOfTwo( cl_uint x );
cl_uint RoundUpToNextPowerOfTwo( cl_uint x )
{
    if( 0 == (x & (x-1)))
        return x;

    while( x & (x-1) )
       x &= x-1;

    return x + x;
}

void CL_CALLBACK WriteInputBufferComplete( cl_event, cl_int, void * );

typedef struct DataInitInfo
{
    cl_ulong        start;
    cl_uint         size;
    Type            outType;
    Type            inType;
    SaturationMode  sat;
    RoundingMode    round;
    MTdata          *d;
}DataInitInfo;

cl_int InitData( cl_uint job_id, cl_uint thread_id, void *p );
cl_int InitData( cl_uint job_id, cl_uint thread_id, void *p )
{
    DataInitInfo *info = (DataInitInfo*) p;

    gInitFunctions[ info->inType ]( (char*)gIn + job_id * info->size * gTypeSizes[info->inType], info->sat, info->round,
                                   info->outType, info->start + job_id * info->size, info->size, info->d[thread_id] );
    return CL_SUCCESS;
}

static void setAllowZ(uint8_t *allow, uint32_t *x, cl_uint count)
{
    cl_uint i;
    for (i = 0; i < count; ++i)
    allow[i] |= (uint8_t)((x[i] & 0x7f800000U) == 0);
}

cl_int PrepareReference( cl_uint job_id, cl_uint thread_id, void *p );
cl_int PrepareReference( cl_uint job_id, cl_uint thread_id, void *p )
{
    DataInitInfo *info = (DataInitInfo*) p;
    cl_uint count = info->size;
    Type inType = info->inType;
    Type outType = info->outType;
    RoundingMode round = info->round;
    size_t j;

    Force64BitFPUPrecision();

    void *s = (cl_uchar*) gIn + job_id * count * gTypeSizes[info->inType];
    void *a = (cl_uchar*) gAllowZ + job_id * count;
    void *d = (cl_uchar*) gRef + job_id * count * gTypeSizes[info->outType];

    if (outType != inType)
    {
        //create the reference while we wait
        Convert f = gConversions[ outType ][ inType ];
        if( info->sat )
            f = gSaturatedConversions[ outType ][ inType ];

#if defined( __arm__ ) && defined( __GNUC__ )
       /* ARM VFP doesn't have hardware instruction for converting from 64-bit integer to float types, hence GCC ARM uses the floating-point emulation code
        * despite which -mfloat-abi setting it is. But the emulation code in libgcc.a has only one rounding mode (round to nearest even in this case)
        * and ignores the user rounding mode setting in hardware.
        * As a result setting rounding modes in hardware won't give correct rounding results for type covert from 64-bit integer to float using GCC for ARM compiler
        * so for testing different rounding modes, we need to use alternative reference function */
        switch (round)
        {
            /* conversions to floating-point type use the current rounding mode.
             * The only default floating-point rounding mode supported is round to nearest even
             * i.e the current rounding mode will be _rte for floating-point types. */
            case kDefaultRoundingMode:
                    qcom_rm = qcomRTE;
                    break;
            case kRoundToNearestEven:
                    qcom_rm = qcomRTE;
                    break;
            case kRoundUp:
                    qcom_rm = qcomRTP;
                    break;
            case kRoundDown:
                    qcom_rm = qcomRTN;
                    break;
            case kRoundTowardZero:
                    qcom_rm = qcomRTZ;
                    break;
            default:
                    vlog_error("ERROR: undefined rounding mode %d\n", round);
                    break;
        }
        qcom_sat =  info->sat;
#endif

        RoundingMode oldRound = set_round( round, outType );
        f( d, s, count );
        set_round( oldRound, outType );

    // Decide if we allow a zero result in addition to the correctly rounded one
        memset(a, 0, count);
    if (gForceFTZ) {
        if (inType == kfloat)
        setAllowZ((uint8_t*)a, (uint32_t*)s, count);
        if (outType == kfloat)
        setAllowZ((uint8_t*)a, (uint32_t*)d, count);
    }
    }
    else
    {
        // Copy the input to the reference
        memcpy(d, s, info->size * gTypeSizes[inType]);
    }

    //Patch up NaNs conversions to integer to zero -- these can be converted to any integer
    if( info->outType != kfloat && info->outType != kdouble )
    {
        if( inType == kfloat )
        {
            float *inp = (float*) s;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    memset( (char*) d + j * gTypeSizes[ outType ], 0, gTypeSizes[ outType ] );
            }
        }
        if( inType == kdouble )
        {
            double *inp = (double*) s;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    memset( (char*) d + j * gTypeSizes[ outType ], 0, gTypeSizes[ outType ] );
            }
        }
    }
    else if( inType == kfloat || inType == kdouble )
    {  // outtype and intype is float or double.  NaN conversions for float <-> double can be any NaN
        if( inType == kfloat && outType == kdouble )
        {
            float *inp = (float*) s;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    ((double*) d)[j] = NAN;
            }
        }
        if( inType == kdouble && outType == kfloat )
        {
            double *inp = (double*) s;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    ((float*) d)[j] = NAN;
            }
        }
    }

    return CL_SUCCESS;
}

static int DoTest( cl_device_id device, Type outType, Type inType, SaturationMode sat, RoundingMode round, MTdata d )
{
#ifdef __APPLE__
    cl_ulong wall_start = mach_absolute_time();
#endif

    DataInitInfo  init_info = { 0, 0, outType, inType, sat, round, NULL };
    WriteInputBufferInfo writeInputBufferInfo;
    int vectorSize;
    int error = 0;
    cl_uint threads = GetThreadCount();
    uint64_t i;

    gTestCount++;
    size_t blockCount = BUFFER_SIZE / MAX( gTypeSizes[ inType ], gTypeSizes[ outType ] );
    size_t step = blockCount;
    uint64_t lastCase = 1ULL << (8*gTypeSizes[ inType ]);
    cl_event writeInputBuffer = NULL;

    memset( &writeInputBufferInfo, 0, sizeof( writeInputBufferInfo ) );
    init_info.d = (MTdata*)malloc( threads * sizeof( MTdata ) );
    if( NULL == init_info.d )
    {
        vlog_error( "ERROR: Unable to allocate storage for random number generator!\n" );
        return -1;
    }
    for( i = 0; i < threads; i++ )
    {
        init_info.d[i] = init_genrand( genrand_int32( d ) );
        if( NULL == init_info.d[i] )
        {
            vlog_error( "ERROR: Unable to allocate storage for random number generator!\n" );
            return -1;
        }
    }

    writeInputBufferInfo.outType = outType;
    writeInputBufferInfo.inType = inType;

    for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        writeInputBufferInfo.calcInfo[vectorSize].program = MakeProgram( outType, inType, sat, round, vectorSize,
                                                                        &writeInputBufferInfo.calcInfo[vectorSize].kernel );
        if( NULL == writeInputBufferInfo.calcInfo[vectorSize].program )
        {
            gFailCount++;
            return -1;
        }
        if( NULL == writeInputBufferInfo.calcInfo[vectorSize].kernel )
        {
            gFailCount++;
            vlog_error( "\t\tFAILED -- Failed to create kernel.\n" );
            return -2;
        }

        writeInputBufferInfo.calcInfo[vectorSize].parent = &writeInputBufferInfo;
        writeInputBufferInfo.calcInfo[vectorSize].vectorSize = vectorSize;
        writeInputBufferInfo.calcInfo[vectorSize].result = -1;
    }

    if( gSkipTesting )
        goto exit;

    // Patch up rounding mode if default is RTZ
    // We leave the part above in default rounding mode so that the right kernel is compiled.
    if( round == kDefaultRoundingMode && gIsRTZ && (outType == kfloat) )
        init_info.round = round = kRoundTowardZero;

    // Figure out how many elements are in a work block

    // we handle 64-bit types a bit differently.
    if( 8*gTypeSizes[ inType ] > 32 )
        lastCase = 0x100000000ULL;

    if ( !gWimpyMode && gIsEmbedded )
      step = blockCount * EMBEDDED_REDUCTION_FACTOR;

    if ( gWimpyMode )
        step = (size_t)blockCount * (size_t)gWimpyReductionFactor;
    vlog( "Testing... " );
    fflush(stdout);
    for( i = 0; i < (uint64_t)lastCase; i += step )
    {

        if( 0 == ( i & ((lastCase >> 3) -1))) {
            vlog(".");
            fflush(stdout);
        }

        cl_uint count = (uint32_t) MIN( blockCount, lastCase - i );
        writeInputBufferInfo.count = count;

        // Crate a user event to represent the status of the reference value computation completion
        writeInputBufferInfo.calcReferenceValues = clCreateUserEvent( gContext, &error);
        if( error || NULL == writeInputBufferInfo.calcReferenceValues )
        {
            vlog_error( "ERROR: Unable to create user event. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        // retain for consumption by MapOutputBufferComplete
        for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
        {
            if( (error = clRetainEvent(writeInputBufferInfo.calcReferenceValues) ))
            {
                vlog_error( "ERROR: Unable to retain user event. (%d)\n", error );
                gFailCount++;
                goto exit;
            }
        }

        // Crate a user event to represent when the callbacks are done verifying correctness
        writeInputBufferInfo.doneBarrier = clCreateUserEvent( gContext, &error);
        if( error || NULL == writeInputBufferInfo.calcReferenceValues )
        {
            vlog_error( "ERROR: Unable to create user event for barrier. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        // retain for use by the callback that calls this
        if( (error = clRetainEvent(writeInputBufferInfo.doneBarrier) ))
        {
            vlog_error( "ERROR: Unable to retain user event doneBarrier. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        //      Call this in a multithreaded manner
        //      gInitFunctions[ inType ]( gIn, sat, round, outType, i, count, d );
        cl_uint chunks = RoundUpToNextPowerOfTwo(threads) * 2;
        init_info.start = i;
        init_info.size = count / chunks;
        if( init_info.size < 16384 )
        {
            chunks = RoundUpToNextPowerOfTwo(threads);
            init_info.size = count / chunks;
            if( init_info.size < 16384 )
            {
                init_info.size = count;
                chunks = 1;
            }
        }
        ThreadPool_Do(InitData, chunks, &init_info);

        // Copy the results to the device
        writeInputBuffer = NULL;
        if( (error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0, count * gTypeSizes[inType], gIn, 0, NULL, &writeInputBuffer )))
        {
            vlog_error( "ERROR: clEnqueueWriteBuffer failed. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        // Setup completion callback for the write, which will enqueue the rest of the work
        // This is somewhat gratuitous.  Because this is an in order queue, we didn't really need to
        // do this work in a callback. We could have done it from the main thread.  Here we are
        // verifying that the implementation can enqueue work from a callback, while at the same time
        // also checking to make sure that the conversions work.
        //
        // Because the verification code is also moved to a callback, it is hoped that implementations will
        // achieve a test performance improvement because they can verify the results in parallel.  If the
        // implementation serializes callbacks however, that won't happen.   Consider it some motivation
        // to do the right thing! :-)
        if( (error = clSetEventCallback( writeInputBuffer, CL_COMPLETE, WriteInputBufferComplete, &writeInputBufferInfo)) )
        {
            vlog_error( "ERROR: clSetEventCallback failed. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        // The event can't be destroyed until the callback is called, so we can release it now.
        if( (error = clReleaseEvent(writeInputBuffer) ))
        {
            vlog_error( "ERROR: clReleaseEvent failed. (%d)\n", error );
            gFailCount++;
            goto exit;
        }

        // Make sure the work is actually running, so we don't deadlock
        if( (error = clFlush( gQueue ) ) )
        {
            vlog_error( "clFlush failed with error %d\n", error );
            gFailCount++;
            goto exit;
        }

        ThreadPool_Do(PrepareReference, chunks, &init_info);

        // signal we are done calculating the reference results
        if( (error = clSetUserEventStatus( writeInputBufferInfo.calcReferenceValues, CL_COMPLETE ) ) )
        {
            vlog_error( "Error:  Failed to set user event status to CL_COMPLETE:  %d\n", error );
            gFailCount++;
            goto exit;
        }

        // Wait for the event callbacks to finish verifying correctness.
        if( (error = clWaitForEvents( 1, (cl_event*) &writeInputBufferInfo.doneBarrier ) ))
        {
            vlog_error( "Error:  Failed to wait for barrier:  %d\n", error );
            gFailCount++;
            goto exit;
        }

        if( (error = clReleaseEvent(writeInputBufferInfo.calcReferenceValues ) ))
        {
            vlog_error( "Error:  Failed to release calcReferenceValues:  %d\n", error );
            gFailCount++;
            goto exit;
        }

        if( (error = clReleaseEvent(writeInputBufferInfo.doneBarrier ) ))
        {
            vlog_error( "Error:  Failed to release done barrier:  %d\n", error );
            gFailCount++;
            goto exit;
        }


        for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
        {
            if( ( error = writeInputBufferInfo.calcInfo[ vectorSize ].result ))
            {
                switch( inType )
                {
                    case kuchar:
                    case kchar:
                        vlog( "Input value: 0x%2.2x ", ((unsigned char*)gIn)[error - 1] );
                        break;
                    case kushort:
                    case kshort:
                        vlog( "Input value: 0x%4.4x ", ((unsigned short*)gIn)[error - 1] );
                        break;
                    case kuint:
                    case kint:
                        vlog( "Input value: 0x%8.8x ", ((unsigned int*)gIn)[error - 1] );
                        break;
                    case kfloat:
                        vlog( "Input value: %a ", ((float*)gIn)[error - 1] );
                        break;
                        break;
                    case kulong:
                    case klong:
                        vlog( "Input value: 0x%16.16llx ", ((unsigned long long*)gIn)[error - 1] );
                        break;
                    case kdouble:
                        vlog( "Input value: %a ", ((double*)gIn)[error - 1]);
                        break;
                    default:
                        vlog_error( "Internal error at %s: %d\n", __FILE__, __LINE__ );
                        abort();
                        break;
                }

                // tell the user which conversion it was.
                if( 0 == vectorSize )
                    vlog( " (implicit scalar conversion from %s to %s)\n", gTypeNames[ inType ], gTypeNames[ outType ] );
                else
                    vlog( " (convert_%s%s%s%s( %s%s ))\n", gTypeNames[outType], sizeNames[vectorSize], gSaturationNames[ sat ],
                                                            gRoundingModeNames[ round ], gTypeNames[inType], sizeNames[vectorSize] );

                gFailCount++;
                goto exit;
            }
        }
    }

    log_info( "done.\n" );

    if( gTimeResults )
    {
        //Kick off tests for the various vector lengths
        for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
        {
            size_t workItemCount = blockCount / vectorSizes[vectorSize];
            if( vectorSizes[vectorSize] * gTypeSizes[outType] < 4 )
                workItemCount /= 4 / (vectorSizes[vectorSize] * gTypeSizes[outType]);

            double sum = 0.0;
            double bestTime = INFINITY;
            cl_uint k;
            for( k = 0; k < PERF_LOOP_COUNT; k++ )
            {
                uint64_t startTime = GetTime();
                if( (error = RunKernel( writeInputBufferInfo.calcInfo[vectorSize].kernel, gInBuffer, gOutBuffers[ vectorSize ], workItemCount )) )
                {
                    gFailCount++;
                    goto exit;
                }

                // Make sure OpenCL is done
                if( (error = clFinish(gQueue) ) )
                {
                    vlog_error( "Error %d at clFinish\n", error );
                    goto exit;
                }

                uint64_t endTime = GetTime();
                double time = SubtractTime( endTime, startTime );
                sum += time;
                if( time < bestTime )
                    bestTime = time;

            }

            if( gReportAverageTimes )
                bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double) gDeviceFrequency * gComputeDevices * gSimdSize * 1e6 / (workItemCount * vectorSizes[vectorSize]);
            if( 0 == vectorSize )
                vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "implicit convert %s -> %s", gTypeNames[ inType ], gTypeNames[ outType ] );
            else
                vlog_perf( clocksPerOp, LOWER_IS_BETTER, "clocks / element", "convert_%s%s%s%s( %s%s )", gTypeNames[ outType ], sizeNames[vectorSize], gSaturationNames[ sat ], gRoundingModeNames[round], gTypeNames[inType], sizeNames[vectorSize] );
        }
    }

    if( gWimpyMode )
        vlog( "\tWimp pass" );
    else
        vlog( "\tpassed" );

#ifdef __APPLE__
    // record the run time
    vlog( "\t(%f s)", 1e-9 * ( mach_absolute_time() - wall_start ) );
#endif
    vlog( "\n\n" );
    fflush( stdout );


exit:
    //clean up
    for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        clReleaseProgram( writeInputBufferInfo.calcInfo[vectorSize].program );
        clReleaseKernel( writeInputBufferInfo.calcInfo[vectorSize].kernel );
    }

    if( init_info.d )
    {
        for( i = 0; i < threads; i++ )
            free_mtdata(init_info.d[i]);
        free(init_info.d);
    }

    return error;
}

void CL_CALLBACK MapResultValuesComplete( cl_event e, cl_int status, void *data );

// Note: not called reentrantly
void CL_CALLBACK WriteInputBufferComplete( cl_event e, cl_int status, void *data )
{
    WriteInputBufferInfo *info = (WriteInputBufferInfo*) data;
    cl_uint count = info->count;
    int vectorSize;

    if( CL_SUCCESS != status )
    {
        vlog_error( "ERROR: WriteInputBufferComplete calback failed with status: %d\n", status );
        gFailCount++;
        return;
    }

    info->barrierCount = gMaxVectorSize - gMinVectorSize;

    // now that we know that the write buffer is complete, enqueue callbacks to wait for the main thread to
    // finish calculating the reference results.
    for( vectorSize = gMinVectorSize; vectorSize < gMaxVectorSize; vectorSize++)
    {
        size_t workItemCount = (count + vectorSizes[vectorSize] - 1) / ( vectorSizes[vectorSize]);
        cl_event mapComplete = NULL;

        if( (status = RunKernel( info->calcInfo[ vectorSize ].kernel, gInBuffer, gOutBuffers[ vectorSize ], workItemCount )) )
        {
            gFailCount++;
            return;
        }

        info->calcInfo[vectorSize].p = clEnqueueMapBuffer( gQueue, gOutBuffers[ vectorSize ], CL_FALSE, CL_MAP_READ | CL_MAP_WRITE,
                                                          0, count * gTypeSizes[ info->outType ], 0, NULL, &mapComplete, &status);
        {
            if( status )
            {
                vlog_error( "ERROR: WriteInputBufferComplete calback failed with status: %d\n", status );
                gFailCount++;
                return;
            }
        }

        if( (status = clSetEventCallback( mapComplete, CL_COMPLETE, MapResultValuesComplete, info->calcInfo + vectorSize)))
        {
            vlog_error( "ERROR: WriteInputBufferComplete calback failed with status: %d\n", status );
            gFailCount++;
            return;
        }

        if( (status = clReleaseEvent(mapComplete)))
        {
            vlog_error( "ERROR: clReleaseEvent calback failed in WriteInputBufferComplete for vector size %d with status: %d\n", vectorSize, status );
            gFailCount++;
            return;
        }
    }

    // Make sure the work starts moving -- otherwise we may deadlock
    if( (status = clFlush(gQueue)))
    {
        vlog_error( "ERROR: WriteInputBufferComplete calback failed with status: %d\n", status );
        gFailCount++;
        return;
    }

    // e was already released by the main thread. It should be destroyed automatically soon after we exit.
}

void CL_CALLBACK CalcReferenceValuesComplete( cl_event e, cl_int status, void *data );

// Note: May be called reentrantly
void CL_CALLBACK MapResultValuesComplete( cl_event e, cl_int status, void *data )
{
    CalcReferenceValuesInfo *info = (CalcReferenceValuesInfo*) data;
    cl_event calcReferenceValues = info->parent->calcReferenceValues;

    if( CL_SUCCESS != status )
    {
        vlog_error( "ERROR: MapResultValuesComplete calback failed with status: %d\n", status );
        gFailCount++;       // not thread safe -- being lazy here
        clReleaseEvent(calcReferenceValues);
        return;
    }

    // we know that the map is done, wait for the main thread to finish calculating the reference values
    if( (status = clSetEventCallback( calcReferenceValues, CL_COMPLETE, CalcReferenceValuesComplete, data )))
    {
        vlog_error( "ERROR: clSetEventCallback failed in MapResultValuesComplete with status: %d\n", status );
        gFailCount++;       // not thread safe -- being lazy here
    }

    // this thread no longer needs its reference to info->calcReferenceValues, so release it
    if( (status = clReleaseEvent(calcReferenceValues) ))
    {
        vlog_error( "ERROR: clReleaseEvent(info->calcReferenceValues) failed with status: %d\n", status );
        gFailCount++;       // not thread safe -- being lazy here
    }

    // no need to flush since we didn't enqueue anything

    // e was already released by WriteInputBufferComplete. It should be destroyed automatically soon after we exit.
}


void CL_CALLBACK CalcReferenceValuesComplete( cl_event e, cl_int status, void *data )
{
    CalcReferenceValuesInfo     *info = (CalcReferenceValuesInfo*) data;
    cl_uint                     vectorSize = info->vectorSize;
    cl_uint                     count = info->parent->count;
    Type                        outType = info->parent->outType;        // the data type of the conversion result
    Type                        inType = info->parent->inType;          // the data type of the conversion input
    size_t                      j;
    cl_int                      error;
    cl_event                    doneBarrier = info->parent->doneBarrier;

    // report spurious error condition
    if( CL_SUCCESS != status )
    {
        vlog_error( "ERROR: CalcReferenceValuesComplete did not succeed! (%d)\n", status );
        gFailCount++;       // lazy about thread safety here
        return;
    }

    // Now we know that both results have been mapped back from the device, and the
    // main thread is done calculating the reference results. It is now time to check
    // the results.

    // verify results
    void *mapped = info->p;

    //Patch up NaNs conversions to integer to zero -- these can be converted to any integer
    if( outType != kfloat && outType != kdouble )
    {
        if( inType == kfloat )
        {
            float *inp = (float*) gIn;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    memset( (char*) mapped + j * gTypeSizes[ outType ], 0, gTypeSizes[ outType ] );
            }
        }
        if( inType == kdouble )
        {
            double *inp = (double*) gIn;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) )
                    memset( (char*) mapped + j * gTypeSizes[ outType ], 0, gTypeSizes[ outType ] );
            }
        }
    }
    else if( inType == kfloat || inType == kdouble )
    {  // outtype and intype is float or double.  NaN conversions for float <-> double can be any NaN
        if( inType == kfloat && outType == kdouble )
        {
            float *inp = (float*) gIn;
            double *outp = (double*) mapped;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) && isnan(outp[j]) )
                    outp[j] = NAN;
            }
        }
        if( inType == kdouble && outType == kfloat )
        {
            double *inp = (double*) gIn;
            float *outp = (float*) mapped;
            for( j = 0; j < count; j++ )
            {
                if( isnan( inp[j] ) && isnan(outp[j]) )
                    outp[j] = NAN;
            }
        }
    }

    if( memcmp( mapped, gRef, count * gTypeSizes[ outType ] ) )
        info->result = gCheckResults[outType]( mapped, gRef, gAllowZ, count, vectorSizes[vectorSize] );
    else
        info->result = 0;

    // Fill the output buffer with junk and release it
    {
        cl_uint pattern =  0xffffdead;
        memset_pattern4(mapped, &pattern, count * gTypeSizes[outType]);
        if((error = clEnqueueUnmapMemObject(gQueue, gOutBuffers[ vectorSize ], mapped, 0, NULL, NULL)))
        {
            vlog_error( "ERROR: clEnqueueUnmapMemObject failed in CalcReferenceValuesComplete  (%d)\n", error );
            gFailCount++;
        }
    }

    if( 1 == ThreadPool_AtomicAdd( &info->parent->barrierCount, -1) )
    {
        if( (status = clSetUserEventStatus( doneBarrier, CL_COMPLETE) ))
        {
            vlog_error( "ERROR: clSetUserEventStatus failed in CalcReferenceValuesComplete (err: %d). We're probably going to deadlock.\n", status );
            gFailCount++;
            return;
        }

        if( (status = clReleaseEvent( doneBarrier ) ) )
        {
            vlog_error( "ERROR: clReleaseEvent failed in CalcReferenceValuesComplete (err: %d).\n", status );
            gFailCount++;
            return;
        }
    }


    // e was already released by WriteInputBufferComplete. It should be destroyed automatically soon after
    // all the calls to CalcReferenceValuesComplete exit.
}

static cl_program   MakeProgram( Type outType, Type inType, SaturationMode sat, RoundingMode round, int vectorSize, cl_kernel *outKernel )
{
    cl_program program;
    char testName[256];
    int error = 0;
    const char **strings;
    size_t stringCount = 0;

    // Create the program. This is a bit complicated because we are trying to avoid byte and short stores.
    if (0 == vectorSize)
    {
        char inName[32];
        char outName[32];
        const char *programSource[] =
        {
            "", // optional pragma
            "__kernel void ", testName, "( __global ", inName, " *src, __global ", outName, " *dest )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   dest[i] =  src[i];\n"
            "}\n"
        };
        stringCount = sizeof(programSource) / sizeof(programSource[0]);
        strings = programSource;

        if (outType == kdouble || inType == kdouble)
            programSource[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

        //create the type name
        strncpy(inName, gTypeNames[inType], sizeof(inName));
        strncpy(outName, gTypeNames[outType], sizeof(outName));
        sprintf(testName, "test_implicit_%s_%s", outName, inName);
        vlog("Building implicit %s -> %s conversion test\n", gTypeNames[inType], gTypeNames[outType]);
        fflush(stdout);
    }
    else
    {
        int vectorSizetmp = vectorSizes[vectorSize];

        char convertString[128];
        char inName[32];
        char outName[32];
        const char *programSource[] =
        {
            "", // optional pragma
            "__kernel void ", testName, "( __global ", inName, " *src, __global ", outName, " *dest )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   dest[i] = ", convertString, "( src[i] );\n"
            "}\n"
        };
        const char *programSourceV3[] =
        {
            "", // optional pragma
            "__kernel void ", testName, "( __global ", inName, " *src, __global ", outName, " *dest )\n"
            "{\n"
            "   size_t i = get_global_id(0);\n"
            "   if( i + 1 < get_global_size(0))\n"
            "       vstore3( ", convertString, "( vload3( i, src)), i, dest );\n"
            "   else\n"
            "   {\n"
            "       ", inName, "3 in;\n"
            "       ", outName, "3 out;\n"
            "       if( 0 == (i & 1) )\n"
            "           in.y = src[3*i+1];\n"
            "       in.x = src[3*i];\n"
            "       out = ", convertString, "( in ); \n"
            "       dest[3*i] = out.x;\n"
            "       if( 0 == (i & 1) )\n"
            "           dest[3*i+1] = out.y;\n"
            "   }\n"
            "}\n"
        };
        stringCount = 3 == vectorSizetmp ? sizeof(programSourceV3) / sizeof(programSourceV3[0]) :
            sizeof(programSource) / sizeof(programSource[0]);
        strings = 3 == vectorSizetmp ? programSourceV3 : programSource;

        if (outType == kdouble || inType == kdouble) {
            programSource[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            programSourceV3[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }

        //create the type name
        switch (vectorSizetmp)
        {
        case 1:
            strncpy(inName, gTypeNames[inType], sizeof(inName));
            strncpy(outName, gTypeNames[outType], sizeof(outName));
            snprintf(convertString, sizeof(convertString), "convert_%s%s%s", outName, gSaturationNames[sat], gRoundingModeNames[round]);
            snprintf(testName, 256, "test_%s_%s", convertString, inName);
            vlog("Building %s( %s ) test\n", convertString, inName);
            break;
        case 3:
            strncpy(inName, gTypeNames[inType], sizeof(inName));
            strncpy(outName, gTypeNames[outType], sizeof(outName));
            snprintf(convertString, sizeof(convertString), "convert_%s3%s%s", outName, gSaturationNames[sat], gRoundingModeNames[round]);
            snprintf(testName, 256, "test_%s_%s3", convertString, inName);
            vlog("Building %s( %s3 ) test\n", convertString, inName);
            break;
        default:
            snprintf(inName, sizeof(inName), "%s%d", gTypeNames[inType], vectorSizetmp);
            snprintf(outName, sizeof(outName), "%s%d", gTypeNames[outType], vectorSizetmp);
            snprintf(convertString, sizeof(convertString), "convert_%s%s%s", outName, gSaturationNames[sat], gRoundingModeNames[round]);
            snprintf(testName, 256, "test_%s_%s", convertString, inName);
            vlog("Building %s( %s ) test\n", convertString, inName);
            break;
        }

        fflush(stdout);
    }
    *outKernel = NULL;

    const char *flags = NULL;
    if( gForceFTZ )
        flags = "-cl-denorms-are-zero";

    // build it
    error = create_single_kernel_helper(gContext, &program, outKernel, (cl_uint)stringCount, strings, testName, flags);
    if (error)
    {
        char    buffer[2048] = "";

        vlog_error("Failed to build kernel/program.\n", error);
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}
