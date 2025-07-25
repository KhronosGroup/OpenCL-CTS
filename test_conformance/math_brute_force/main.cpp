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

#include "function_list.h"
#include "sleep.h"
#include "utility.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/parseParameters.h"
#include "harness/typeWrappers.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/mman.h>
#include <libgen.h>
#include <sys/time.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/sysctl.h>
#include <sys/param.h>
#endif

#if defined(__linux__) || (defined WIN32 && defined __MINGW32__)
#include <sys/param.h>
#endif

#include "harness/testHarness.h"

#define kPageSize 4096
#define HALF_REQUIRED_FEATURES_1 (CL_FP_ROUND_TO_ZERO)
#define HALF_REQUIRED_FEATURES_2 (CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN)
#define DOUBLE_REQUIRED_FEATURES                                               \
    (CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO                  \
     | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM)

static std::vector<const char *> gTestNames;
static char appName[MAXPATHLEN] = "";
cl_device_id gDevice = NULL;
cl_context gContext = NULL;
cl_command_queue gQueue = NULL;
static size_t gStartTestNumber = ~0u;
static size_t gEndTestNumber = ~0u;
int gSkipCorrectnessTesting = 0;
static int gStopOnError = 0;
static bool gSkipRestOfTests;
int gForceFTZ = 0;
int gWimpyMode = 0;
int gHostFill = 0;
static int gHasDouble = 0;
static int gTestFloat = 1;
// This flag should be 'ON' by default and it can be changed through the command
// line arguments.
static int gTestFastRelaxed = 1;
/*This flag corresponds to defining if the implementation has Derived Fast
  Relaxed functions. The spec does not specify ULP for derived function.  The
  derived functions are composed of base functions which are tested for ULP,
  thus when this flag is enabled, Derived functions will not be tested for ULP,
  as per table 7.1 of OpenCL 2.0 spec. Since there is no way of quering the
  device whether it is a derived or non-derived implementation according to
  OpenCL 2.0 spec then it has to be changed through a command line argument.
*/
int gFastRelaxedDerived = 1;
int gHasHalf = 0;
cl_device_fp_config gHalfCapabilities = 0;
int gDeviceILogb0 = 1;
int gDeviceILogbNaN = 1;
int gCheckTininessBeforeRounding = 1;
int gIsInRTZMode = 0;
uint32_t gMaxVectorSizeIndex = VECTOR_SIZE_COUNT;
uint32_t gMinVectorSizeIndex = 0;
void *gIn = NULL;
void *gIn2 = NULL;
void *gIn3 = NULL;
void *gOut_Ref = NULL;
void *gOut[VECTOR_SIZE_COUNT] = { NULL, NULL, NULL, NULL, NULL, NULL };
void *gOut_Ref2 = NULL;
void *gOut2[VECTOR_SIZE_COUNT] = { NULL, NULL, NULL, NULL, NULL, NULL };
cl_mem gInBuffer = NULL;
cl_mem gInBuffer2 = NULL;
cl_mem gInBuffer3 = NULL;
cl_mem gOutBuffer[VECTOR_SIZE_COUNT] = { NULL, NULL, NULL, NULL, NULL, NULL };
cl_mem gOutBuffer2[VECTOR_SIZE_COUNT] = { NULL, NULL, NULL, NULL, NULL, NULL };
static MTdataHolder gMTdata;
cl_device_fp_config gFloatCapabilities = 0;
int gWimpyReductionFactor = 32;
int gVerboseBruteForce = 0;

cl_half_rounding_mode gHalfRoundingMode = CL_HALF_RTE;

static int ParseArgs(int argc, const char **argv);
static void PrintUsage(void);
static void PrintFunctions(void);
static test_status InitCL(cl_device_id device);
static void ReleaseCL(void);
static int InitILogbConstants(void);
static int IsTininessDetectedBeforeRounding(void);
static int
IsInRTZMode(void); // expensive. Please check gIsInRTZMode global instead.

static int doTest(const char *name)
{
    if (gSkipRestOfTests)
    {
        vlog("Skipping function because of an earlier error.\n");
        return 1;
    }

    int error = 0;
    const Func *func_data = NULL;

    for (size_t i = 0; i < functionListCount; i++)
    {
        const Func *const temp_func = functionList + i;
        if (strcmp(temp_func->name, name) == 0)
        {
            if ((gStartTestNumber != ~0u && i < gStartTestNumber)
                || i > gEndTestNumber)
            {
                vlog("Skipping function #%zu\n", i);
                return 0;
            }

            func_data = temp_func;
            break;
        }
    }

    if (func_data == NULL)
    {
        vlog("Function '%s' doesn't exist!\n", name);
        exit(EXIT_FAILURE);
    }

    if (func_data->func.p == NULL && func_data->rfunc.p == NULL)
    {
        vlog("'%s' is missing implementation, skipping function.\n",
             func_data->name);
        return 0;
    }

    // if correctly rounded divide & sqrt are supported by the implementation
    // then test it; otherwise skip the test
    if (strcmp(func_data->name, "sqrt_cr") == 0
        || strcmp(func_data->name, "divide_cr") == 0)
    {
        if ((gFloatCapabilities & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) == 0)
        {
            vlog("Correctly rounded divide and sqrt are not supported, "
                 "skipping function.\n");
            return 0;
        }
    }
    {
        if (0 == strcmp("ilogb", func_data->name))
        {
            InitILogbConstants();
        }

        if (gTestFastRelaxed && func_data->relaxed)
        {
            if (get_device_cl_version(gDevice) > Version(1, 2))
            {
                gTestCount++;
                vlog("%3d: ", gTestCount);
                // Test with relaxed requirements here.
                if (func_data->vtbl_ptr->TestFunc(func_data, gMTdata,
                                                  true /* relaxed mode */))
                {
                    gFailCount++;
                    error++;
                    if (gStopOnError)
                    {
                        gSkipRestOfTests = true;
                        return error;
                    }
                }
            }
            else
            {
                vlog("Skipping reduced precision testing for device with "
                     "version 1.2 or less\n");
            }
        }

        if (gTestFloat)
        {
            gTestCount++;
            vlog("%3d: ", gTestCount);
            // Don't test with relaxed requirements.
            if (func_data->vtbl_ptr->TestFunc(func_data, gMTdata,
                                              false /* relaxed mode */))
            {
                gFailCount++;
                error++;
                if (gStopOnError)
                {
                    gSkipRestOfTests = true;
                    return error;
                }
            }
        }

        if (gHasDouble && NULL != func_data->vtbl_ptr->DoubleTestFunc
            && NULL != func_data->dfunc.p)
        {
            gTestCount++;
            vlog("%3d: ", gTestCount);
            // Don't test with relaxed requirements.
            if (func_data->vtbl_ptr->DoubleTestFunc(func_data, gMTdata,
                                                    false /* relaxed mode*/))
            {
                gFailCount++;
                error++;
                if (gStopOnError)
                {
                    gSkipRestOfTests = true;
                    return error;
                }
            }
        }

        if (gHasHalf && NULL != func_data->vtbl_ptr->HalfTestFunc)
        {
            gTestCount++;
            vlog("%3d: ", gTestCount);
            if (func_data->vtbl_ptr->HalfTestFunc(func_data, gMTdata,
                                                  false /* relaxed mode*/))
            {
                gFailCount++;
                error++;
                if (gStopOnError)
                {
                    gSkipRestOfTests = true;
                    return error;
                }
            }
        }
    }

    return error;
}

#define DO_TEST(name)                                                          \
    REGISTER_TEST_VERSION(name, Version(1, 0)) { return doTest(#name); }

DO_TEST(acos)
DO_TEST(acosh)
DO_TEST(acospi)
DO_TEST(asin)
DO_TEST(asinh)
DO_TEST(asinpi)
DO_TEST(atan)
DO_TEST(atanh)
DO_TEST(atanpi)
DO_TEST(atan2)
DO_TEST(atan2pi)
DO_TEST(cbrt)
DO_TEST(ceil)
DO_TEST(copysign)
DO_TEST(cos)
DO_TEST(cosh)
DO_TEST(cospi)
DO_TEST(exp)
DO_TEST(exp2)
DO_TEST(exp10)
DO_TEST(expm1)
DO_TEST(fabs)
DO_TEST(fdim)
DO_TEST(floor)
DO_TEST(fma)
DO_TEST(fmax)
DO_TEST(fmin)
DO_TEST(fmod)
DO_TEST(fract)
DO_TEST(frexp)
DO_TEST(hypot)
DO_TEST(ilogb)
DO_TEST(isequal)
DO_TEST(isfinite)
DO_TEST(isgreater)
DO_TEST(isgreaterequal)
DO_TEST(isinf)
DO_TEST(isless)
DO_TEST(islessequal)
DO_TEST(islessgreater)
DO_TEST(isnan)
DO_TEST(isnormal)
DO_TEST(isnotequal)
DO_TEST(isordered)
DO_TEST(isunordered)
DO_TEST(ldexp)
DO_TEST(lgamma)
DO_TEST(lgamma_r)
DO_TEST(log)
DO_TEST(log2)
DO_TEST(log10)
DO_TEST(log1p)
DO_TEST(logb)
DO_TEST(mad)
DO_TEST(maxmag)
DO_TEST(minmag)
DO_TEST(modf)
DO_TEST(nan)
DO_TEST(nextafter)
DO_TEST(pow)
DO_TEST(pown)
DO_TEST(powr)
DO_TEST(remainder)
DO_TEST(remquo)
DO_TEST(rint)
DO_TEST(rootn)
DO_TEST(round)
DO_TEST(rsqrt)
DO_TEST(signbit)
DO_TEST(sin)
DO_TEST(sincos)
DO_TEST(sinh)
DO_TEST(sinpi)
DO_TEST(sqrt)
DO_TEST(sqrt_cr)
DO_TEST(tan)
DO_TEST(tanh)
DO_TEST(tanpi)
DO_TEST(trunc)
DO_TEST(half_cos)
DO_TEST(half_divide)
DO_TEST(half_exp)
DO_TEST(half_exp2)
DO_TEST(half_exp10)
DO_TEST(half_log)
DO_TEST(half_log2)
DO_TEST(half_log10)
DO_TEST(half_powr)
DO_TEST(half_recip)
DO_TEST(half_rsqrt)
DO_TEST(half_sin)
DO_TEST(half_sqrt)
DO_TEST(half_tan)
DO_TEST(add)
DO_TEST(subtract)
DO_TEST(reciprocal)
DO_TEST(divide)
DO_TEST(divide_cr)
DO_TEST(multiply)
DO_TEST(assignment)
DO_TEST(not )
DO_TEST(erf)
DO_TEST(erfc)

#pragma mark -

int main(int argc, const char *argv[])
{
    int error;

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return -1;
    }

    error = ParseArgs(argc, argv);
    if (error) return error;

    // This takes a while, so prevent the machine from going to sleep.
    PreventSleep();
    atexit(ResumeSleep);

    if (gSkipCorrectnessTesting)
        vlog("*** Skipping correctness testing! ***\n\n");
    else if (gStopOnError)
        vlog("Stopping at first error.\n");

    vlog("   \t                                        ");
    if (gWimpyMode) vlog("   ");
    if (!gSkipCorrectnessTesting) vlog("\t  max_ulps");

    vlog("\n-------------------------------------------------------------------"
         "----------------------------------------\n");

    gMTdata = MTdataHolder(gRandomSeed);

    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);

    int ret = runTestHarnessWithCheck(
        gTestNames.size(), gTestNames.data(),
        test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), true, 0, InitCL);

    RestoreFPState(&oldMode);

    if (gQueue)
    {
        int error_code = clFinish(gQueue);
        if (error_code) vlog_error("clFinish failed:%d\n", error_code);
    }

    ReleaseCL();

    return ret;
}

static int ParseArgs(int argc, const char **argv)
{
    // We only pass test names to runTestHarnessWithCheck, hence global command
    // line options defined by the harness cannot be used by the user.
    // To respect the implementation details of runTestHarnessWithCheck,
    // gTestNames[0] has to exist although its value is not important.
    gTestNames.push_back("");

    int singleThreaded = 0;
    int forcedWorkerThreads = 0;

    { // Extract the app name
        strncpy(appName, argv[0], MAXPATHLEN - 1);
        appName[MAXPATHLEN - 1] = '\0';

#if defined(__APPLE__)
        char baseName[MAXPATHLEN];
        char *base = NULL;
        strncpy(baseName, argv[0], MAXPATHLEN - 1);
        baseName[MAXPATHLEN - 1] = '\0';
        base = basename(baseName);
        if (NULL != base)
        {
            strncpy(appName, base, sizeof(appName) - 1);
            appName[sizeof(appName) - 1] = '\0';
        }
#endif
    }

    vlog("\n%s\t", appName);
    for (int i = 1; i < argc; i++)
    {
        const char *arg = argv[i];
        if (NULL == arg) break;

        vlog("\t%s", arg);
        int optionFound = 0;
        if (arg[0] == '-')
        {
            while (arg[1] != '\0')
            {
                arg++;
                optionFound = 1;
                switch (*arg)
                {
                    case 'd': gHasDouble ^= 1; break;

                    case 'e': gFastRelaxedDerived ^= 1; break;

                    case 'f': gTestFloat ^= 1; break;

                    case 'h': PrintUsage(); return -1;

                    case 'p': PrintFunctions(); return -1;

                    case 'l': gSkipCorrectnessTesting ^= 1; break;

                    case 'm': singleThreaded ^= 1; break;

                    case 't':
                        forcedWorkerThreads = atoi(argv[++i]);
                        vlog(" %d", forcedWorkerThreads);
                        break;

                    case 'g': gHasHalf ^= 1; break;

                    case 'r': gTestFastRelaxed ^= 1; break;

                    case 's': gStopOnError ^= 1; break;

                    case 'v': gVerboseBruteForce ^= 1; break;

                    case 'w': // wimpy mode
                        gWimpyMode ^= 1;
                        break;

                    case '[':
                        parseWimpyReductionFactor(arg, gWimpyReductionFactor);
                        break;

                    case 'b': gHostFill ^= 1; break;

                    case 'z': gForceFTZ ^= 1; break;

                    case '1':
                        if (arg[1] == '6')
                        {
                            gMinVectorSizeIndex = 5;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                            arg++;
                        }
                        else
                        {
                            gMinVectorSizeIndex = 0;
                            gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        }
                        break;
                    case '2':
                        gMinVectorSizeIndex = 1;
                        gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        break;
                    case '3':
                        gMinVectorSizeIndex = 2;
                        gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        break;
                    case '4':
                        gMinVectorSizeIndex = 3;
                        gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        break;
                    case '8':
                        gMinVectorSizeIndex = 4;
                        gMaxVectorSizeIndex = gMinVectorSizeIndex + 1;
                        break;

                    default:
                        vlog(" <-- unknown flag: %c (0x%2.2x)\n)", *arg, *arg);
                        PrintUsage();
                        return -1;
                }
            }
        }

        if (!optionFound)
        {
            char *t = NULL;
            long number = strtol(arg, &t, 0);
            if (t != arg)
            {
                if (~0u == gStartTestNumber)
                    gStartTestNumber = (int32_t)number;
                else
                    gEndTestNumber = gStartTestNumber + (int32_t)number;
            }
            else
            {
                // Make sure this is a valid name
                unsigned int k;
                for (k = 0; k < functionListCount; k++)
                {
                    const Func *f = functionList + k;
                    if (strcmp(arg, f->name) == 0)
                    {
                        gTestNames.push_back(arg);
                        break;
                    }
                }
                // If we didn't find it in the list of test names
                if (k >= functionListCount)
                {
                    gTestNames.push_back(arg);
                }
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

    PrintArch();

    if (gWimpyMode)
    {
        vlog("\n");
        vlog("*** WARNING: Testing in Wimpy mode!                     ***\n");
        vlog("*** Wimpy mode is not sufficient to verify correctness. ***\n");
        vlog("*** Wimpy Reduction Factor: %-27u ***\n\n",
             gWimpyReductionFactor);
    }

    if (singleThreaded)
    {
        vlog("*** WARNING: Force 1 worker thread                      ***\n");
        SetThreadCount(1);
    }
    else if (forcedWorkerThreads > 0)
    {
        vlog("*** WARNING: Force %d worker threads                    ***\n",
             forcedWorkerThreads);
        SetThreadCount(forcedWorkerThreads);
    }

    return 0;
}


static void PrintFunctions(void)
{
    vlog("\nMath function names:\n");
    for (size_t i = 0; i < functionListCount; i++)
    {
        vlog("\t%s\n", functionList[i].name);
    }
}

static void PrintUsage(void)
{
    vlog("%s [-cglsz]: <optional: math function names>\n", appName);
    vlog("\toptions:\n");
    vlog("\t\t-d\tToggle double precision testing. (Default: on iff khr_fp_64 "
         "on)\n");
    vlog("\t\t-f\tToggle float precision testing. (Default: on)\n");
    vlog("\t\t-g\tToggle half precision testing. (Default: on if khr_fp_16 "
         "on)\n");
    vlog("\t\t-r\tToggle fast relaxed math precision testing. (Default: on)\n");
    vlog("\t\t-e\tToggle test as derived implementations for fast relaxed math "
         "precision. (Default: on)\n");
    vlog("\t\t-h\tPrint this message and quit\n");
    vlog("\t\t-p\tPrint all math function names and quit\n");
    vlog("\t\t-l\tlink check only (make sure functions are present, skip "
         "accuracy checks.)\n");
    vlog("\t\t-m\tToggle run multi-threaded. (Default: on) )\n");
    vlog("\t\t-s\tStop on error\n");
    vlog("\t\t-w\tToggle Wimpy Mode, * Not a valid test * \n");
    vlog("\t\t-[2^n]\tSet wimpy reduction factor, recommended range of n is "
         "1-10, default factor(%u)\n",
         gWimpyReductionFactor);
    vlog("\t\t-b\tFill buffers on host instead of device. (Default: off)\n");
    vlog("\t\t-z\tToggle FTZ mode (Section 6.5.3) for all functions. (Set by "
         "device capabilities by default.)\n");
    vlog("\t\t-v\tToggle Verbosity (Default: off)\n ");
    vlog("\t\t-#\tTest only vector sizes #, e.g. \"-1\" tests scalar only, "
         "\"-16\" tests 16-wide vectors only.\n");
    vlog("\n\tYou may also pass a number instead of a function name.\n");
    vlog("\tThis causes the first N tests to be skipped. The tests are "
         "numbered.\n");
    vlog("\tIf you pass a second number, that is the number tests to run after "
         "the first one.\n");
    vlog("\tA name list may be used in conjunction with a number range. In "
         "that case,\n");
    vlog("\tonly the named cases in the number range will run.\n");
    vlog("\tYou may also choose to pass no arguments, in which case all tests "
         "will be run.\n");
    vlog("\tYou may pass CL_DEVICE_TYPE_CPU/GPU/ACCELERATOR to select the "
         "device.\n");
    vlog("\n");
}

static void CL_CALLBACK bruteforce_notify_callback(const char *errinfo,
                                                   const void *private_info,
                                                   size_t cb, void *user_data)
{
    vlog("%s  (%p, %zd, %p)\n", errinfo, private_info, cb, user_data);
}

test_status InitCL(cl_device_id device)
{
    int error;
    uint32_t i;
    cl_device_type device_type;

    error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type),
                            &device_type, NULL);
    if (error)
    {
        print_error(error, "Unable to get device type");
        return TEST_FAIL;
    }

    gDevice = device;

    // Check extensions
    if (is_extension_available(gDevice, "cl_khr_fp64"))
    {
        gHasDouble ^= 1;
#if defined(CL_DEVICE_DOUBLE_FP_CONFIG)
        cl_device_fp_config doubleCapabilities = 0;
        if ((error = clGetDeviceInfo(gDevice, CL_DEVICE_DOUBLE_FP_CONFIG,
                                     sizeof(doubleCapabilities),
                                     &doubleCapabilities, NULL)))
        {
            vlog_error("ERROR: Unable to get device "
                       "CL_DEVICE_DOUBLE_FP_CONFIG. (%d)\n",
                       error);
            return TEST_FAIL;
        }

        if (DOUBLE_REQUIRED_FEATURES
            != (doubleCapabilities & DOUBLE_REQUIRED_FEATURES))
        {
            std::string list;
            if (0 == (doubleCapabilities & CL_FP_FMA)) list += "CL_FP_FMA, ";
            if (0 == (doubleCapabilities & CL_FP_ROUND_TO_NEAREST))
                list += "CL_FP_ROUND_TO_NEAREST, ";
            if (0 == (doubleCapabilities & CL_FP_ROUND_TO_ZERO))
                list += "CL_FP_ROUND_TO_ZERO, ";
            if (0 == (doubleCapabilities & CL_FP_ROUND_TO_INF))
                list += "CL_FP_ROUND_TO_INF, ";
            if (0 == (doubleCapabilities & CL_FP_INF_NAN))
                list += "CL_FP_INF_NAN, ";
            if (0 == (doubleCapabilities & CL_FP_DENORM))
                list += "CL_FP_DENORM, ";
            vlog_error("ERROR: required double features are missing: %s\n",
                       list.c_str());

            return TEST_FAIL;
        }
#else
        vlog_error("FAIL: device says it supports cl_khr_fp64 but "
                   "CL_DEVICE_DOUBLE_FP_CONFIG is not in the headers!\n");
        return TEST_FAIL;
#endif
    }

    gFloatToHalfRoundingMode = kRoundToNearestEven;
    if (is_extension_available(gDevice, "cl_khr_fp16"))
    {
        gHasHalf ^= 1;
#if defined(CL_DEVICE_HALF_FP_CONFIG)
        if ((error = clGetDeviceInfo(gDevice, CL_DEVICE_HALF_FP_CONFIG,
                                     sizeof(gHalfCapabilities),
                                     &gHalfCapabilities, NULL)))
        {
            vlog_error(
                "ERROR: Unable to get device CL_DEVICE_HALF_FP_CONFIG. (%d)\n",
                error);
            return TEST_FAIL;
        }
        if (HALF_REQUIRED_FEATURES_1
                != (gHalfCapabilities & HALF_REQUIRED_FEATURES_1)
            && HALF_REQUIRED_FEATURES_2
                != (gHalfCapabilities & HALF_REQUIRED_FEATURES_2))
        {
            char list[300] = "";
            if (0 == (gHalfCapabilities & CL_FP_ROUND_TO_NEAREST))
                strncat(list, "CL_FP_ROUND_TO_NEAREST, ", sizeof(list) - 1);
            if (0 == (gHalfCapabilities & CL_FP_ROUND_TO_ZERO))
                strncat(list, "CL_FP_ROUND_TO_ZERO, ", sizeof(list) - 1);
            if (0 == (gHalfCapabilities & CL_FP_INF_NAN))
                strncat(list, "CL_FP_INF_NAN, ", sizeof(list) - 1);
            vlog_error("ERROR: required half features are missing: %s\n", list);

            return TEST_FAIL;
        }

        if ((gHalfCapabilities & CL_FP_ROUND_TO_NEAREST) != 0)
        {
            gHalfRoundingMode = CL_HALF_RTE;
        }
        else // due to above condition it must be RTZ
        {
            gHalfRoundingMode = CL_HALF_RTZ;
        }

#else
        vlog_error("FAIL: device says it supports cl_khr_fp16 but "
                   "CL_DEVICE_HALF_FP_CONFIG is not in the headers!\n");
        return TEST_FAIL;
#endif
    }


    uint32_t deviceFrequency = 0;
    size_t configSize = sizeof(deviceFrequency);
    if ((error = clGetDeviceInfo(gDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 configSize, &deviceFrequency, NULL)))
        deviceFrequency = 0;

    if ((error = clGetDeviceInfo(gDevice, CL_DEVICE_SINGLE_FP_CONFIG,
                                 sizeof(gFloatCapabilities),
                                 &gFloatCapabilities, NULL)))
    {
        vlog_error(
            "ERROR: Unable to get device CL_DEVICE_SINGLE_FP_CONFIG. (%d)\n",
            error);
        return TEST_FAIL;
    }

    gContext = clCreateContext(NULL, 1, &gDevice, bruteforce_notify_callback,
                               NULL, &error);
    if (NULL == gContext || error)
    {
        vlog_error("clCreateContext failed. (%d) \n", error);
        return TEST_FAIL;
    }

    gQueue = clCreateCommandQueue(gContext, gDevice, 0, &error);
    if (NULL == gQueue || error)
    {
        vlog_error("clCreateCommandQueue failed. (%d)\n", error);
        return TEST_FAIL;
    }

    // Allocate buffers
    cl_uint min_alignment = 0;
    error = clGetDeviceInfo(gDevice, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                            sizeof(cl_uint), (void *)&min_alignment, NULL);
    if (CL_SUCCESS != error)
    {
        vlog_error("clGetDeviceInfo failed. (%d)\n", error);
        return TEST_FAIL;
    }
    min_alignment >>= 3; // convert bits to bytes

    gIn = align_malloc(BUFFER_SIZE, min_alignment);
    if (NULL == gIn) return TEST_FAIL;
    gIn2 = align_malloc(BUFFER_SIZE, min_alignment);
    if (NULL == gIn2) return TEST_FAIL;
    gIn3 = align_malloc(BUFFER_SIZE, min_alignment);
    if (NULL == gIn3) return TEST_FAIL;
    gOut_Ref = align_malloc(BUFFER_SIZE, min_alignment);
    if (NULL == gOut_Ref) return TEST_FAIL;
    gOut_Ref2 = align_malloc(BUFFER_SIZE, min_alignment);
    if (NULL == gOut_Ref2) return TEST_FAIL;

    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        gOut[i] = align_malloc(BUFFER_SIZE, min_alignment);
        if (NULL == gOut[i]) return TEST_FAIL;
        gOut2[i] = align_malloc(BUFFER_SIZE, min_alignment);
        if (NULL == gOut2[i]) return TEST_FAIL;
    }

    cl_mem_flags device_flags = CL_MEM_READ_ONLY;
    // save a copy on the host device to make this go faster
    if (CL_DEVICE_TYPE_CPU == device_type)
        device_flags |= CL_MEM_USE_HOST_PTR;
    else
        device_flags |= CL_MEM_COPY_HOST_PTR;

    // setup input buffers
    gInBuffer =
        clCreateBuffer(gContext, device_flags, BUFFER_SIZE, gIn, &error);
    if (gInBuffer == NULL || error)
    {
        vlog_error("clCreateBuffer1 failed for input (%d)\n", error);
        return TEST_FAIL;
    }

    gInBuffer2 =
        clCreateBuffer(gContext, device_flags, BUFFER_SIZE, gIn2, &error);
    if (gInBuffer2 == NULL || error)
    {
        vlog_error("clCreateBuffer2 failed for input (%d)\n", error);
        return TEST_FAIL;
    }

    gInBuffer3 =
        clCreateBuffer(gContext, device_flags, BUFFER_SIZE, gIn3, &error);
    if (gInBuffer3 == NULL || error)
    {
        vlog_error("clCreateBuffer3 failed for input (%d)\n", error);
        return TEST_FAIL;
    }


    // setup output buffers
    device_flags = CL_MEM_READ_WRITE;
    // save a copy on the host device to make this go faster
    if (CL_DEVICE_TYPE_CPU == device_type)
        device_flags |= CL_MEM_USE_HOST_PTR;
    else
        device_flags |= CL_MEM_COPY_HOST_PTR;
    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        gOutBuffer[i] = clCreateBuffer(gContext, device_flags, BUFFER_SIZE,
                                       gOut[i], &error);
        if (gOutBuffer[i] == NULL || error)
        {
            vlog_error("clCreateBuffer failed for output (%d)\n", error);
            return TEST_FAIL;
        }
        gOutBuffer2[i] = clCreateBuffer(gContext, device_flags, BUFFER_SIZE,
                                        gOut2[i], &error);
        if (gOutBuffer2[i] == NULL || error)
        {
            vlog_error("clCreateBuffer2 failed for output (%d)\n", error);
            return TEST_FAIL;
        }
    }

    // we are embedded, check current rounding mode
    if (gIsEmbedded)
    {
        gIsInRTZMode = IsInRTZMode();
    }

    // Check tininess detection
    IsTininessDetectedBeforeRounding();

    cl_platform_id platform;
    int err = clGetDeviceInfo(gDevice, CL_DEVICE_PLATFORM, sizeof(platform),
                              &platform, NULL);
    if (err)
    {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");
        return TEST_FAIL;
    }

    char c[1024];
    static const char *no_yes[] = { "NO", "YES" };
    vlog("\nCompute Device info:\n");
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(c), &c, NULL);
    vlog("\tPlatform Version: %s\n", c);
    clGetDeviceInfo(gDevice, CL_DEVICE_NAME, sizeof(c), &c, NULL);
    vlog("\tDevice Name: %s\n", c);
    clGetDeviceInfo(gDevice, CL_DEVICE_VENDOR, sizeof(c), &c, NULL);
    vlog("\tVendor: %s\n", c);
    clGetDeviceInfo(gDevice, CL_DEVICE_VERSION, sizeof(c), &c, NULL);
    vlog("\tDevice Version: %s\n", c);
    clGetDeviceInfo(gDevice, CL_DEVICE_OPENCL_C_VERSION, sizeof(c), &c, NULL);
    vlog("\tCL C Version: %s\n", c);
    clGetDeviceInfo(gDevice, CL_DRIVER_VERSION, sizeof(c), &c, NULL);
    vlog("\tDriver Version: %s\n", c);
    vlog("\tDevice Frequency: %d MHz\n", deviceFrequency);
    vlog("\tSubnormal values supported for floats? %s\n",
         no_yes[0 != (CL_FP_DENORM & gFloatCapabilities)]);
    vlog("\tCorrectly rounded divide and sqrt supported for floats? %s\n",
         no_yes[0
                != (CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT & gFloatCapabilities)]);
    vlog("\tTesting with FTZ mode ON for floats? %s\n",
         no_yes[0 != gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities)]);
    vlog("\tTesting single precision? %s\n", no_yes[0 != gTestFloat]);
    vlog("\tTesting fast relaxed math? %s\n", no_yes[0 != gTestFastRelaxed]);
    if (gTestFastRelaxed)
    {
        vlog("\tFast relaxed math has derived implementations? %s\n",
             no_yes[0 != gFastRelaxedDerived]);
    }
    vlog("\tTesting double precision? %s\n", no_yes[0 != gHasDouble]);
    if (sizeof(long double) == sizeof(double) && gHasDouble)
    {
        vlog("\n\t\tWARNING: Host system long double does not have better "
             "precision than double!\n");
        vlog("\t\t         All double results that do not match the reference "
             "result have their reported\n");
        vlog("\t\t         error inflated by 0.5 ulps to account for the fact "
             "that this system\n");
        vlog("\t\t         can not accurately represent the right result to an "
             "accuracy closer\n");
        vlog("\t\t         than half an ulp. See comments in "
             "Bruteforce_Ulp_Error_Double() for more details.\n\n");
    }

    vlog("\tTesting half precision? %s\n", no_yes[0 != gHasHalf]);
    vlog("\tIs Embedded? %s\n", no_yes[0 != gIsEmbedded]);
    if (gIsEmbedded)
        vlog("\tRunning in RTZ mode? %s\n", no_yes[0 != gIsInRTZMode]);
    vlog("\tTininess is detected before rounding? %s\n",
         no_yes[0 != gCheckTininessBeforeRounding]);
    vlog("\tWorker threads: %d\n", GetThreadCount());
    vlog("\tTesting vector sizes:");
    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
        vlog("\t%d", sizeValues[i]);

    vlog("\n");
    vlog("\tVerbose? %s\n", no_yes[0 != gVerboseBruteForce]);
    vlog("\n\n");

    // Check to see if we are using single threaded mode on other than a 1.0
    // device
    if (getenv("CL_TEST_SINGLE_THREADED"))
    {

        char device_version[1024] = { 0 };
        clGetDeviceInfo(gDevice, CL_DEVICE_VERSION, sizeof(device_version),
                        device_version, NULL);

        if (strcmp("OpenCL 1.0 ", device_version))
        {
            vlog("ERROR: CL_TEST_SINGLE_THREADED is set in the environment. "
                 "Running single threaded.\n");
        }
    }

    return TEST_PASS;
}

static void ReleaseCL(void)
{
    uint32_t i;
    clReleaseMemObject(gInBuffer);
    clReleaseMemObject(gInBuffer2);
    clReleaseMemObject(gInBuffer3);
    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        clReleaseMemObject(gOutBuffer[i]);
        clReleaseMemObject(gOutBuffer2[i]);
    }
    clReleaseCommandQueue(gQueue);
    clReleaseContext(gContext);

    align_free(gIn);
    align_free(gIn2);
    align_free(gIn3);
    align_free(gOut_Ref);
    align_free(gOut_Ref2);

    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        align_free(gOut[i]);
        align_free(gOut2[i]);
    }
}

int InitILogbConstants(void)
{
    int error;
    const char *kernelSource =
        R"(__kernel void GetILogBConstants( __global int *out )
        {
            out[0] = FP_ILOGB0;
            out[1] = FP_ILOGBNAN;
        })";

    clProgramWrapper query;
    clKernelWrapper kernel;
    error = create_single_kernel_helper(gContext, &query, &kernel, 1,
                                        &kernelSource, "GetILogBConstants");
    if (error != CL_SUCCESS)
    {
        vlog_error("Error: Unable to create kernel to get FP_ILOGB0 and "
                   "FP_ILOGBNAN for the device. (%d)",
                   error);
        return error;
    }

    if ((error =
             clSetKernelArg(kernel, 0, sizeof(gOutBuffer[gMinVectorSizeIndex]),
                            &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error("Error: Unable to set kernel arg to get FP_ILOGB0 and "
                   "FP_ILOGBNAN for the device. Err = %d",
                   error);
        return error;
    }

    size_t dim = 1;
    if ((error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &dim, NULL, 0,
                                        NULL, NULL)))
    {
        vlog_error("Error: Unable to execute kernel to get FP_ILOGB0 and "
                   "FP_ILOGBNAN for the device. Err = %d",
                   error);
        return error;
    }

    struct
    {
        cl_int ilogb0, ilogbnan;
    } data;
    if ((error = clEnqueueReadBuffer(gQueue, gOutBuffer[gMinVectorSizeIndex],
                                     CL_TRUE, 0, sizeof(data), &data, 0, NULL,
                                     NULL)))
    {
        vlog_error("Error: unable to read FP_ILOGB0 and FP_ILOGBNAN from the "
                   "device. Err = %d",
                   error);
        return error;
    }

    gDeviceILogb0 = data.ilogb0;
    gDeviceILogbNaN = data.ilogbnan;

    return 0;
}

int IsTininessDetectedBeforeRounding(void)
{
    int error;
    const char *kernelSource =
        R"(__kernel void IsTininessDetectedBeforeRounding( __global float *out, float a, float b )
        {
           out[0] = a * b; // product is 0x1.fffffffffff8p-127
        })";

    float a = 0x1.000002p-126f;
    float b = 0x1.fffffcp-1f;

    clProgramWrapper query;
    clKernelWrapper kernel;
    error =
        create_single_kernel_helper(gContext, &query, &kernel, 1, &kernelSource,
                                    "IsTininessDetectedBeforeRounding");
    if (error != CL_SUCCESS)
    {
        vlog_error("Error: Unable to create kernel to detect how tininess is "
                   "detected for the device. (%d)",
                   error);
        return error;
    }

    if ((error =
             clSetKernelArg(kernel, 0, sizeof(gOutBuffer[gMinVectorSizeIndex]),
                            &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error("Error: Unable to set kernel arg to detect how tininess is "
                   "detected  for the device. Err = %d",
                   error);
        return error;
    }

    if ((error = clSetKernelArg(kernel, 1, sizeof(a), &a)))
    {
        vlog_error("Error: Unable to set kernel arg to detect how tininess is "
                   "detected  for the device. Err = %d",
                   error);
        return error;
    }

    if ((error = clSetKernelArg(kernel, 2, sizeof(b), &b)))
    {
        vlog_error("Error: Unable to set kernel arg to detect how tininess is "
                   "detected  for the device. Err = %d",
                   error);
        return error;
    }

    size_t dim = 1;
    if ((error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &dim, NULL, 0,
                                        NULL, NULL)))
    {
        vlog_error("Error: Unable to execute kernel to detect how tininess is "
                   "detected  for the device. Err = %d",
                   error);
        return error;
    }

    struct
    {
        cl_uint f;
    } data;
    if ((error = clEnqueueReadBuffer(gQueue, gOutBuffer[gMinVectorSizeIndex],
                                     CL_TRUE, 0, sizeof(data), &data, 0, NULL,
                                     NULL)))
    {
        vlog_error("Error: unable to read result from tininess test from the "
                   "device. Err = %d",
                   error);
        return error;
    }

    gCheckTininessBeforeRounding = 0 == (data.f & 0x7fffffff);

    return 0;
}

static int IsInRTZMode(void)
{
    int error;
    const char *kernelSource =
        R"(__kernel void GetRoundingMode( __global int *out )
        {
            volatile float a = 0x1.0p23f;
            volatile float b = -0x1.0p23f;
            out[0] = (a + 0x1.fffffep-1f == a) && (b - 0x1.fffffep-1f == b);
        })";

    clProgramWrapper query;
    clKernelWrapper kernel;
    error = create_single_kernel_helper(gContext, &query, &kernel, 1,
                                        &kernelSource, "GetRoundingMode");
    if (error != CL_SUCCESS)
    {
        vlog_error("Error: Unable to create kernel to detect RTZ mode for the "
                   "device. (%d)",
                   error);
        return error;
    }

    if ((error =
             clSetKernelArg(kernel, 0, sizeof(gOutBuffer[gMinVectorSizeIndex]),
                            &gOutBuffer[gMinVectorSizeIndex])))
    {
        vlog_error("Error: Unable to set kernel arg to detect RTZ mode for the "
                   "device. Err = %d",
                   error);
        return error;
    }

    size_t dim = 1;
    if ((error = clEnqueueNDRangeKernel(gQueue, kernel, 1, NULL, &dim, NULL, 0,
                                        NULL, NULL)))
    {
        vlog_error("Error: Unable to execute kernel to detect RTZ mode for the "
                   "device. Err = %d",
                   error);
        return error;
    }

    struct
    {
        cl_int isRTZ;
    } data;
    if ((error = clEnqueueReadBuffer(gQueue, gOutBuffer[gMinVectorSizeIndex],
                                     CL_TRUE, 0, sizeof(data), &data, 0, NULL,
                                     NULL)))
    {
        vlog_error(
            "Error: unable to read RTZ mode data from the device. Err = %d",
            error);
        return error;
    }

    return data.isRTZ;
}

#pragma mark -

const char *sizeNames[VECTOR_SIZE_COUNT] = { "", "2", "3", "4", "8", "16" };
const int sizeValues[VECTOR_SIZE_COUNT] = { 1, 2, 3, 4, 8, 16 };

// TODO: There is another version of Ulp_Error_Double defined in
// test_common/harness/errorHelpers.c
float Bruteforce_Ulp_Error_Double(double test, long double reference)
{
    // Check for Non-power-of-two and NaN

    // Note: This function presumes that someone has already tested whether the
    // result is correctly, rounded before calling this function.  That test:
    //
    //    if( (float) reference == test )
    //        return 0.0f;
    //
    // would ensure that cases like fabs(reference) > FLT_MAX are weeded out
    // before we get here. Otherwise, we'll return inf ulp error here, for what
    // are otherwise correctly rounded results.

    // Deal with long double = double
    // On most systems long double is a higher precision type than double. They
    // provide either a 80-bit or greater floating point type, or they provide a
    // head-tail double double format. That is sufficient to represent the
    // accuracy of a floating point result to many more bits than double and we
    // can calculate sub-ulp errors. This is the standard system for which this
    // test suite is designed.
    //
    // On some systems double and long double are the same thing. Then we run
    // into a problem, because our representation of the infinitely precise
    // result (passed in as reference above) can be off by as much as a half
    // double precision ulp itself.  In this case, we inflate the reported error
    // by half an ulp to take this into account.  A more correct and permanent
    // fix would be to undertake refactoring the reference code to return
    // results in this format:
    //
    //    typedef struct DoubleReference
    //    { // true value = correctlyRoundedResult + ulps *
    //    ulp(correctlyRoundedResult)        (infinitely precise)
    //        double  correctlyRoundedResult;     // as best we can
    //        double  ulps;                       // plus a fractional amount to
    //        account for the difference
    //    }DoubleReference;                       //     between infinitely
    //    precise result and correctlyRoundedResult, in units of ulps.
    //
    // This would provide a useful higher-than-double precision format for
    // everyone that we can use, and would solve a few problems with
    // representing absolute errors below DBL_MIN and over DBL_MAX for systems
    // that use a head to tail double double for long double.

    int x;
    long double testVal = test;

    // First, handle special reference values
    if (isinf(reference))
    {
        if (reference == testVal) return 0.0f;

        return INFINITY;
    }

    if (isnan(reference))
    {
        if (isnan(testVal)) return 0.0f;

        return INFINITY;
    }

    if (0.0L != reference && 0.5L != frexpl(reference, &x))
    { // Non-zero and Non-power of two

        // allow correctly rounded results to pass through unmolested. (We might
        // add error to it below.) There is something of a performance
        // optimization here.
        if (testVal == reference) return 0.0f;

        // The unbiased exponent of the ulp unit place
        int ulp_exp =
            DBL_MANT_DIG - 1 - std::max(ilogbl(reference), DBL_MIN_EXP - 1);

        // Scale the exponent of the error
        float result = (float)scalbnl(testVal - reference, ulp_exp);

        // account for rounding error in reference result on systems that do not
        // have a higher precision floating point type (see above)
        if (sizeof(long double) == sizeof(double))
            result += copysignf(0.5f, result);

        return result;
    }

    // reference is a normal power of two or a zero
    // The unbiased exponent of the ulp unit place
    int ulp_exp = DBL_MANT_DIG - std::max(ilogbl(reference), DBL_MIN_EXP);

    // allow correctly rounded results to pass through unmolested. (We might add
    // error to it below.) There is something of a performance optimization here
    // too.
    if (testVal == reference) return 0.0f;

    // Scale the exponent of the error
    float result = (float)scalbnl(testVal - reference, ulp_exp);

    // account for rounding error in reference result on systems that do not
    // have a higher precision floating point type (see above)
    if (sizeof(long double) == sizeof(double))
        result += copysignf(0.5f, result);

    return result;
}

float Abs_Error(float test, double reference)
{
    if (isnan(test) && isnan(reference)) return 0.0f;
    return fabs((float)(reference - (double)test));
}

cl_uint RoundUpToNextPowerOfTwo(cl_uint x)
{
    if (0 == (x & (x - 1))) return x;

    while (x & (x - 1)) x &= x - 1;

    return x + x;
}
