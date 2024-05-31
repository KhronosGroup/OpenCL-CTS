//
// Copyright (c) 2017-2024 The Khronos Group Inc.
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
#include "harness/ThreadPool.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"
#include "harness/mt19937.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/param.h>
#include <libgen.h>
#endif

#if defined(__MINGW32__)
#include <sys/param.h>
#endif

#include <sstream>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <libgen.h>
#include <sys/mman.h>
#endif
#include <time.h>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "Sleep.h"

#include "basic_test_conversions.h"
#include <climits>
#include <cstring>

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
#include "fplib.h"
#endif

#if (defined(__arm__) || defined(__aarch64__)) && defined(__GNUC__)
/* Rounding modes and saturation for use with qcom 64 bit to float conversion
 * library */
bool qcom_sat;
roundingMode qcom_rm;
#endif


static int ParseArgs(int argc, const char **argv);
static void PrintUsage(void);
test_status InitCL(cl_device_id device);


const char *gTypeNames[kTypeCount] = { "uchar",  "char",  "ushort", "short",
                                       "uint",   "int",   "half",   "float",
                                       "double", "ulong", "long" };

const char *gRoundingModeNames[kRoundingModeCount] = { "", "_rte", "_rtp",
                                                       "_rtn", "_rtz" };

const char *gSaturationNames[2] = { "", "_sat" };

size_t gTypeSizes[kTypeCount] = {
    sizeof(cl_uchar),  sizeof(cl_char),  sizeof(cl_ushort), sizeof(cl_short),
    sizeof(cl_uint),   sizeof(cl_int),   sizeof(cl_half),   sizeof(cl_float),
    sizeof(cl_double), sizeof(cl_ulong), sizeof(cl_long),
};

char appName[64] = "ctest";
int gMultithread = 1;


int test_conversions(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    if (argCount)
    {
        return MakeAndRunTest<CustomConversionsTest>(device, context, queue,
                                                     num_elements);
    }
    else
    {
        return MakeAndRunTest<ConversionsTest>(device, context, queue,
                                               num_elements);
    }
}


test_definition test_list[] = {
    ADD_TEST(conversions),
};

const int test_num = ARRAY_SIZE(test_list);


int main(int argc, const char **argv)
{
    int error;

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return 1;
    }

    if ((error = ParseArgs(argc, argv))) return error;

    // Turn off sleep so our tests run to completion
    PreventSleep();
    atexit(ResumeSleep);

    if (!gMultithread) SetThreadCount(1);

#if defined(_MSC_VER) && defined(_M_IX86)
    // VS2005 (and probably others, since long double got deprecated) sets
    // the x87 to 53-bit precision. This causes problems with the tests
    // that convert long and ulong to float and double, since they deal
    // with values that need more precision than that. So, set the x87
    // to 64-bit precision.
    unsigned int ignored;
    _controlfp_s(&ignored, _PC_64, _MCW_PC);
#endif

    vlog("===========================================================\n");
    vlog("Random seed: %u\n", gRandomSeed);
    gMTdata = init_genrand(gRandomSeed);

    const char *arg[] = { argv[0] };
    int ret =
        runTestHarnessWithCheck(1, arg, test_num, test_list, true, 0, InitCL);

    free_mtdata(gMTdata);
    if (gQueue)
    {
        error = clFinish(gQueue);
        if (error) vlog_error("clFinish failed: %d\n", error);
    }

    clReleaseMemObject(gInBuffer);

    for (int i = 0; i < kCallStyleCount; i++)
    {
        clReleaseMemObject(gOutBuffers[i]);
    }
    clReleaseCommandQueue(gQueue);
    clReleaseContext(gContext);

    return ret;
}


static int ParseArgs(int argc, const char **argv)
{
    int i;
    argList = (const char **)calloc(argc, sizeof(char *));
    argCount = 0;

    if (NULL == argList && argc > 1) return -1;

#if (defined(__APPLE__) || defined(__linux__) || defined(__MINGW32__))
    { // Extract the app name
        char baseName[MAXPATHLEN];
        strncpy(baseName, argv[0], MAXPATHLEN);
        char *base = basename(baseName);
        if (NULL != base)
        {
            strncpy(appName, base, sizeof(appName));
            appName[sizeof(appName) - 1] = '\0';
        }
    }
#elif defined(_WIN32)
    {
        char fname[_MAX_FNAME + _MAX_EXT + 1];
        char ext[_MAX_EXT];

        errno_t err = _splitpath_s(argv[0], NULL, 0, NULL, 0, fname, _MAX_FNAME,
                                   ext, _MAX_EXT);
        if (err == 0)
        { // no error
            strcat(fname, ext); // just cat them, size of frame can keep both
            strncpy(appName, fname, sizeof(appName));
            appName[sizeof(appName) - 1] = '\0';
        }
    }
#endif

    vlog("\n%s", appName);
    for (i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        if (NULL == arg) break;

        vlog("\t%s", arg);
        if (arg[0] == '-')
        {
            arg++;
            while (*arg != '\0')
            {
                switch (*arg)
                {
                    case 'd': gTestDouble ^= 1; break;
                    case 'h': gTestHalfs ^= 1; break;
                    case 'l': gSkipTesting ^= 1; break;
                    case 'm': gMultithread ^= 1; break;
                    case 'w': gWimpyMode ^= 1; break;
                    case '[':
                        parseWimpyReductionFactor(arg, gWimpyReductionFactor);
                        break;
                    case 'z':
                        gForceFTZ ^= 1;
                        gForceHalfFTZ ^= 1;
                        break;
                    case 't': gTimeResults ^= 1; break;
                    case 'a': gReportAverageTimes ^= 1; break;
                    case '1':
                        if (arg[1] == '6')
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
                        vlog(" <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg);
                        PrintUsage();
                        return -1;
                }
                arg++;
            }
        }
        else
        {
            char *t = NULL;
            long number = strtol(arg, &t, 0);
            if (t != arg)
            {
                if (gStartTestNumber != -1)
                    gEndTestNumber = gStartTestNumber + (int)number;
                else
                    gStartTestNumber = (int)number;
            }
            else
            {
                argList[argCount] = arg;
                argCount++;
            }
        }
    }

    // Check for the wimpy mode environment variable
    if (getenv("CL_WIMPY_MODE"))
    {
        vlog("\n");
        vlog("*** Detected CL_WIMPY_MODE env                          ***\n");
        gWimpyMode = 1;
    }

    vlog("\n");

    PrintArch();

    if (gWimpyMode)
    {
        vlog("\n");
        vlog("*** WARNING: Testing in Wimpy mode!                     ***\n");
        vlog("*** Wimpy mode is not sufficient to verify correctness. ***\n");
        vlog("*** It gives warm fuzzy feelings and then nevers calls. ***\n\n");
        vlog("*** Wimpy Reduction Factor: %-27u ***\n\n",
             gWimpyReductionFactor);
    }

    return 0;
}


static void PrintUsage(void)
{
    int i;
    vlog("%s [-wz#]: <optional: test names>\n", appName);
    vlog("\ttest names:\n");
    vlog("\t\tdestFormat<_sat><_round>_sourceFormat\n");
    vlog("\t\t\tPossible format types are:\n\t\t\t\t");
    for (i = 0; i < kTypeCount; i++) vlog("%s, ", gTypeNames[i]);
    vlog("\n\n\t\t\tPossible saturation values are: (empty) and _sat\n");
    vlog("\t\t\tPossible rounding values are:\n\t\t\t\t(empty), ");
    for (i = 1; i < kRoundingModeCount; i++)
        vlog("%s, ", gRoundingModeNames[i]);
    vlog("\n\t\t\tExamples:\n");
    vlog("\t\t\t\tulong_short   converts short to ulong\n");
    vlog("\t\t\t\tchar_sat_rte_float   converts float to char with saturated "
         "clipping in round to nearest rounding mode\n\n");
    vlog("\toptions:\n");
    vlog("\t\t-d\tToggle testing of double precision.  On by default if "
         "cl_khr_fp64 is enabled, ignored otherwise.\n");
    vlog("\t\t-l\tToggle link check mode. When on, testing is skipped, and we "
         "just check to see that the kernels build. (Off by default.)\n");
    vlog("\t\t-m\tToggle Multithreading. (On by default.)\n");
    vlog("\t\t-w\tToggle wimpy mode. When wimpy mode is on, we run a very "
         "small subset of the tests for each fn. NOT A VALID TEST! (Off by "
         "default.)\n");
    vlog(" \t\t-[2^n]\tSet wimpy reduction factor, recommended range of n is "
         "1-12, default factor(%u)\n",
         gWimpyReductionFactor);
    vlog("\t\t-z\tToggle flush to zero mode  (Default: per device)\n");
    vlog("\t\t-#\tTest just vector size given by #, where # is an element of "
         "the set {1,2,3,4,8,16}\n");
    vlog("\n");
    vlog(
        "You may also pass the number of the test on which to start.\nA second "
        "number can be then passed to indicate how many tests to run\n\n");
}


test_status InitCL(cl_device_id device)
{
    int error, i;
    size_t configSize = sizeof(gComputeDevices);

    if ((error = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 configSize, &gComputeDevices, NULL)))
        gComputeDevices = 1;

    configSize = sizeof(gDeviceFrequency);
    if ((error = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 configSize, &gDeviceFrequency, NULL)))
        gDeviceFrequency = 0;

    cl_device_fp_config floatCapabilities = 0;
    if ((error = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                                 sizeof(floatCapabilities), &floatCapabilities,
                                 NULL)))
        floatCapabilities = 0;
    if (0 == (CL_FP_DENORM & floatCapabilities)) gForceFTZ ^= 1;

    if (0 == (floatCapabilities & CL_FP_ROUND_TO_NEAREST))
    {
        char profileStr[128] = "";
        // Verify that we are an embedded profile device
        if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE,
                                     sizeof(profileStr), profileStr, NULL)))
        {
            vlog_error("FAILURE: Could not get device profile: error %d\n",
                       error);
            return TEST_FAIL;
        }

        if (strcmp(profileStr, "EMBEDDED_PROFILE"))
        {
            vlog_error("FAILURE: non-embedded profile device does not support "
                       "CL_FP_ROUND_TO_NEAREST\n");
            return TEST_FAIL;
        }

        if (0 == (floatCapabilities & CL_FP_ROUND_TO_ZERO))
        {
            vlog_error("FAILURE: embedded profile device supports neither "
                       "CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n");
            return TEST_FAIL;
        }

        gIsRTZ = 1;
    }

    else if (is_extension_available(device, "cl_khr_fp64"))
    {
        gHasDouble = 1;
    }
    gTestDouble &= gHasDouble;

    if (is_extension_available(device, "cl_khr_fp16"))
    {
        gHasHalfs = 1;

        cl_device_fp_config floatCapabilities = 0;
        if ((error = clGetDeviceInfo(device, CL_DEVICE_HALF_FP_CONFIG,
                                     sizeof(floatCapabilities),
                                     &floatCapabilities, NULL)))
            floatCapabilities = 0;

        if (0 == (CL_FP_DENORM & floatCapabilities)) gForceHalfFTZ ^= 1;

        if (0 == (floatCapabilities & CL_FP_ROUND_TO_NEAREST))
        {
            char profileStr[128] = "";
            // Verify that we are an embedded profile device
            if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE,
                                         sizeof(profileStr), profileStr, NULL)))
            {
                vlog_error("FAILURE: Could not get device profile: error %d\n",
                           error);
                return TEST_FAIL;
            }

            if (strcmp(profileStr, "EMBEDDED_PROFILE"))
            {
                vlog_error(
                    "FAILURE: non-embedded profile device does not support "
                    "CL_FP_ROUND_TO_NEAREST\n");
                return TEST_FAIL;
            }

            if (0 == (floatCapabilities & CL_FP_ROUND_TO_ZERO))
            {
                vlog_error("FAILURE: embedded profile device supports neither "
                           "CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n");
                return TEST_FAIL;
            }

            gIsHalfRTZ = 1;
        }
    }
    gTestHalfs &= gHasHalfs;

    // detect whether profile of the device is embedded
    char profile[1024] = "";
    if ((error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile),
                                 profile, NULL)))
    {
        vlog_error("clGetDeviceInfo failed. (%d)\n", error);
        return TEST_FAIL;
    }
    else if (strstr(profile, "EMBEDDED_PROFILE"))
    {
        gIsEmbedded = 1;
        if (!is_extension_available(device, "cles_khr_int64")) gHasLong = 0;
    }

    gContext = clCreateContext(NULL, 1, &device, notify_callback, NULL, &error);
    if (NULL == gContext || error)
    {
        vlog_error("clCreateContext failed. (%d)\n", error);
        return TEST_FAIL;
    }

    gQueue = clCreateCommandQueue(gContext, device, 0, &error);
    if (NULL == gQueue || error)
    {
        vlog_error("clCreateCommandQueue failed. (%d)\n", error);
        return TEST_FAIL;
    }

    // Allocate buffers
    // FIXME: use clProtectedArray for guarded allocations?
    gIn = malloc(BUFFER_SIZE + 2 * kPageSize);
    gAllowZ = malloc(BUFFER_SIZE + 2 * kPageSize);
    gRef = malloc(BUFFER_SIZE + 2 * kPageSize);
    for (i = 0; i < kCallStyleCount; i++)
    {
        gOut[i] = malloc(BUFFER_SIZE + 2 * kPageSize);
        if (NULL == gOut[i]) return TEST_FAIL;
    }

    // setup input buffers
    gInBuffer =
        clCreateBuffer(gContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                       BUFFER_SIZE, NULL, &error);
    if (gInBuffer == NULL || error)
    {
        vlog_error("clCreateBuffer failed for input (%d)\n", error);
        return TEST_FAIL;
    }

    // setup output buffers
    for (i = 0; i < kCallStyleCount; i++)
    {
        gOutBuffers[i] =
            clCreateBuffer(gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                           BUFFER_SIZE, NULL, &error);
        if (gOutBuffers[i] == NULL || error)
        {
            vlog_error("clCreateArray failed for output (%d)\n", error);
            return TEST_FAIL;
        }
    }

    char c[1024];
    static const char *no_yes[] = { "NO", "YES" };
    vlog("\nCompute Device info:\n");
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(c), c, NULL);
    vlog("\tDevice Name: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(c), c, NULL);
    vlog("\tVendor: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(c), c, NULL);
    vlog("\tDevice Version: %s\n", c);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(c), &c, NULL);
    vlog("\tCL C Version: %s\n", c);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(c), c, NULL);
    vlog("\tDriver Version: %s\n", c);
    vlog("\tProcessing with %ld devices\n", gComputeDevices);
    vlog("\tDevice Frequency: %d MHz\n", gDeviceFrequency);
    vlog("\tSubnormal values supported for floats? %s\n",
         no_yes[0 != (CL_FP_DENORM & floatCapabilities)]);
    vlog("\tTesting with FTZ mode ON for floats? %s\n", no_yes[0 != gForceFTZ]);
    vlog("\tTesting with FTZ mode ON for halfs? %s\n",
         no_yes[0 != gForceHalfFTZ]);
    vlog("\tTesting with default RTZ mode for floats? %s\n",
         no_yes[0 != gIsRTZ]);
    vlog("\tTesting with default RTZ mode for halfs? %s\n",
         no_yes[0 != gIsHalfRTZ]);
    vlog("\tHas Double? %s\n", no_yes[0 != gHasDouble]);
    if (gHasDouble) vlog("\tTest Double? %s\n", no_yes[0 != gTestDouble]);
    vlog("\tHas Long? %s\n", no_yes[0 != gHasLong]);
    vlog("\tTesting vector sizes: ");
    for (i = gMinVectorSize; i < gMaxVectorSize; i++)
        vlog("\t%d", vectorSizes[i]);
    vlog("\n");
    return TEST_PASS;
}
