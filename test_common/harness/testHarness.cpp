//
// Copyright (c) 2017-2019 The Khronos Group Inc.
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
#include "testHarness.h"
#include "compat.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include "errorHelpers.h"
#include "kernelHelpers.h"
#include "fpcontrol.h"
#include "typeWrappers.h"
#include "imageHelpers.h"
#include "parseParameters.h"

#if !defined(_WIN32)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <time.h>

#if !defined(__APPLE__)
#include <CL/cl.h>
#endif

int gTestsPassed = 0;
int gTestsFailed = 0;
int gFailCount;
int gTestCount;
cl_uint gRandomSeed = 0;
cl_uint gReSeed = 0;

int gFlushDenormsToZero = 0;
int gInfNanSupport = 1;
int gIsEmbedded = 0;
int gHasLong = 1;
bool gCoreILProgram = true;

#define DEFAULT_NUM_ELEMENTS 0x4000

test_definition *test_registry::definitions() { return &m_definitions[0]; }

size_t test_registry::num_tests() { return m_definitions.size(); }

void test_registry::add_test(test *t, const char *name, Version version)
{

    m_tests.push_back(t);
    test_definition testDef;
    testDef.func = t->getFunction();
    testDef.name = name;
    testDef.min_version = version;
    m_definitions.push_back(testDef);
}

test_registry &test_registry::getInstance()
{
    static test_registry instance;
    return instance;
}

static int saveResultsToJson(const char *suiteName, test_definition testList[],
                             unsigned char selectedTestList[],
                             test_status resultTestList[], int testNum)
{
    char *fileName = getenv("CL_CONFORMANCE_RESULTS_FILENAME");
    if (fileName == nullptr)
    {
        return EXIT_SUCCESS;
    }

    FILE *file = fopen(fileName, "w");
    if (NULL == file)
    {
        log_error("ERROR: Failed to open '%s' for writing results.\n",
                  fileName);
        return EXIT_FAILURE;
    }

    const char *save_map[] = { "success", "failure" };
    const char *result_map[] = { "pass", "fail", "skip" };
    const char *linebreak[] = { "", ",\n" };
    int add_linebreak = 0;

    fprintf(file, "{\n");
    fprintf(file, "\t\"cmd\": \"%s\",\n", suiteName);
    fprintf(file, "\t\"results\": {\n");

    for (int i = 0; i < testNum; ++i)
    {
        if (selectedTestList[i])
        {
            fprintf(file, "%s\t\t\"%s\": \"%s\"", linebreak[add_linebreak],
                    testList[i].name, result_map[(int)resultTestList[i]]);
            add_linebreak = 1;
        }
    }
    fprintf(file, "\n");

    fprintf(file, "\t}\n");
    fprintf(file, "}\n");

    int ret = fclose(file) ? EXIT_FAILURE : EXIT_SUCCESS;

    log_info("Saving results to %s: %s!\n", fileName, save_map[ret]);

    return ret;
}

int runTestHarness(int argc, const char *argv[], int testNum,
                   test_definition testList[], int forceNoContextCreation,
                   cl_command_queue_properties queueProps)
{
    return runTestHarnessWithCheck(argc, argv, testNum, testList,
                                   forceNoContextCreation, queueProps, NULL);
}

int suite_did_not_pass_init(const char *suiteName, test_status status,
                            int testNum, test_definition testList[])
{
    std::vector<unsigned char> selectedTestList(testNum, 1);
    std::vector<test_status> resultTestList(testNum, status);

    int ret = saveResultsToJson(suiteName, testList, selectedTestList.data(),
                                resultTestList.data(), testNum);

    log_info("Test %s while initialization\n",
             status == TEST_SKIP ? "skipped" : "failed");
    log_info("%s %d of %d tests.\n", status == TEST_SKIP ? "SKIPPED" : "FAILED",
             testNum, testNum);

    if (ret != EXIT_SUCCESS)
    {
        return ret;
    }

    return status == TEST_SKIP ? EXIT_SUCCESS : EXIT_FAILURE;
}

void version_expected_info(const char *test_name, const char *api_name,
                           const char *expected_version,
                           const char *device_version)
{
    log_info("%s skipped (requires at least %s version %s, but the device "
             "reports %s version %s)\n",
             test_name, api_name, expected_version, api_name, device_version);
}
int runTestHarnessWithCheck(int argc, const char *argv[], int testNum,
                            test_definition testList[],
                            int forceNoContextCreation,
                            cl_command_queue_properties queueProps,
                            DeviceCheckFn deviceCheckFn)
{
    test_start();

    cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
    cl_uint num_platforms = 0;
    cl_platform_id *platforms;
    cl_device_id device;
    int num_elements = DEFAULT_NUM_ELEMENTS;
    cl_uint num_devices = 0;
    cl_device_id *devices = NULL;
    cl_uint choosen_device_index = 0;
    cl_uint choosen_platform_index = 0;

    int err, ret;
    char *endPtr;
    int based_on_env_var = 0;


    /* Check for environment variable to set device type */
    char *env_mode = getenv("CL_DEVICE_TYPE");
    if (env_mode != NULL)
    {
        based_on_env_var = 1;
        if (strcmp(env_mode, "gpu") == 0
            || strcmp(env_mode, "CL_DEVICE_TYPE_GPU") == 0)
            device_type = CL_DEVICE_TYPE_GPU;
        else if (strcmp(env_mode, "cpu") == 0
                 || strcmp(env_mode, "CL_DEVICE_TYPE_CPU") == 0)
            device_type = CL_DEVICE_TYPE_CPU;
        else if (strcmp(env_mode, "accelerator") == 0
                 || strcmp(env_mode, "CL_DEVICE_TYPE_ACCELERATOR") == 0)
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
        else if (strcmp(env_mode, "custom") == 0
                 || strcmp(env_mode, "CL_DEVICE_TYPE_CUSTOM") == 0)
            device_type = CL_DEVICE_TYPE_CUSTOM;
        else if (strcmp(env_mode, "default") == 0
                 || strcmp(env_mode, "CL_DEVICE_TYPE_DEFAULT") == 0)
            device_type = CL_DEVICE_TYPE_DEFAULT;
        else
        {
            log_error("Unknown CL_DEVICE_TYPE env variable setting: "
                      "%s.\nAborting...\n",
                      env_mode);
            abort();
        }
    }

#if defined(__APPLE__)
    {
        // report on any unusual library search path indirection
        char *libSearchPath = getenv("DYLD_LIBRARY_PATH");
        if (libSearchPath)
            log_info("*** DYLD_LIBRARY_PATH = \"%s\"\n", libSearchPath);

        // report on any unusual framework search path indirection
        char *frameworkSearchPath = getenv("DYLD_FRAMEWORK_PATH");
        if (libSearchPath)
            log_info("*** DYLD_FRAMEWORK_PATH = \"%s\"\n", frameworkSearchPath);
    }
#endif

    env_mode = getenv("CL_DEVICE_INDEX");
    if (env_mode != NULL)
    {
        choosen_device_index = atoi(env_mode);
    }

    env_mode = getenv("CL_PLATFORM_INDEX");
    if (env_mode != NULL)
    {
        choosen_platform_index = atoi(env_mode);
    }

    /* Process the command line arguments */

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return EXIT_FAILURE;
    }

    /* Special case: just list the tests */
    if ((argc > 1)
        && (!strcmp(argv[1], "-list") || !strcmp(argv[1], "-h")
            || !strcmp(argv[1], "--help")))
    {
        char *fileName = getenv("CL_CONFORMANCE_RESULTS_FILENAME");

        log_info(
            "Usage: %s [<test name>*] [pid<num>] [id<num>] [<device type>]\n",
            argv[0]);
        log_info("\t<test name>\tOne or more of: (wildcard character '*') "
                 "(default *)\n");
        log_info("\tpid<num>\tIndicates platform at index <num> should be used "
                 "(default 0).\n");
        log_info("\tid<num>\t\tIndicates device at index <num> should be used "
                 "(default 0).\n");
        log_info("\t<device_type>\tcpu|gpu|accelerator|<CL_DEVICE_TYPE_*> "
                 "(default CL_DEVICE_TYPE_DEFAULT)\n");
        log_info("\n");
        log_info("\tNOTE: You may pass environment variable "
                 "CL_CONFORMANCE_RESULTS_FILENAME (currently '%s')\n",
                 fileName != NULL ? fileName : "<undefined>");
        log_info("\t      to save results to JSON file.\n");

        log_info("\n");
        log_info("Test names:\n");
        for (int i = 0; i < testNum; i++)
        {
            log_info("\t%s\n", testList[i].name);
        }
        return EXIT_SUCCESS;
    }

    /* How are we supposed to seed the random # generators? */
    if (argc > 1 && strcmp(argv[argc - 1], "randomize") == 0)
    {
        gRandomSeed = (cl_uint)time(NULL);
        log_info("Random seed: %u.\n", gRandomSeed);
        gReSeed = 1;
        argc--;
    }
    else
    {
        log_info(" Initializing random seed to 0.\n");
    }

    /* Do we have an integer to specify the number of elements to pass to tests?
     */
    if (argc > 1)
    {
        ret = (int)strtol(argv[argc - 1], &endPtr, 10);
        if (endPtr != argv[argc - 1] && *endPtr == 0)
        {
            /* By spec, this means the entire string was a valid integer, so we
             * treat it as a num_elements spec */
            /* (hence why we stored the result in ret first) */
            num_elements = ret;
            log_info("Testing with num_elements of %d\n", num_elements);
            argc--;
        }
    }

    /* Do we have a CPU/GPU specification? */
    if (argc > 1)
    {
        if (strcmp(argv[argc - 1], "gpu") == 0
            || strcmp(argv[argc - 1], "CL_DEVICE_TYPE_GPU") == 0)
        {
            device_type = CL_DEVICE_TYPE_GPU;
            argc--;
        }
        else if (strcmp(argv[argc - 1], "cpu") == 0
                 || strcmp(argv[argc - 1], "CL_DEVICE_TYPE_CPU") == 0)
        {
            device_type = CL_DEVICE_TYPE_CPU;
            argc--;
        }
        else if (strcmp(argv[argc - 1], "accelerator") == 0
                 || strcmp(argv[argc - 1], "CL_DEVICE_TYPE_ACCELERATOR") == 0)
        {
            device_type = CL_DEVICE_TYPE_ACCELERATOR;
            argc--;
        }
        else if (strcmp(argv[argc - 1], "custom") == 0
                 || strcmp(argv[argc - 1], "CL_DEVICE_TYPE_CUSTOM") == 0)
        {
            device_type = CL_DEVICE_TYPE_CUSTOM;
            argc--;
        }
        else if (strcmp(argv[argc - 1], "CL_DEVICE_TYPE_DEFAULT") == 0)
        {
            device_type = CL_DEVICE_TYPE_DEFAULT;
            argc--;
        }
    }

    /* Did we choose a specific device index? */
    if (argc > 1)
    {
        if (strlen(argv[argc - 1]) >= 3 && argv[argc - 1][0] == 'i'
            && argv[argc - 1][1] == 'd')
        {
            choosen_device_index = atoi(&(argv[argc - 1][2]));
            argc--;
        }
    }

    /* Did we choose a specific platform index? */
    if (argc > 1)
    {
        if (strlen(argv[argc - 1]) >= 3 && argv[argc - 1][0] == 'p'
            && argv[argc - 1][1] == 'i' && argv[argc - 1][2] == 'd')
        {
            choosen_platform_index = atoi(&(argv[argc - 1][3]));
            argc--;
        }
    }


    switch (device_type)
    {
        case CL_DEVICE_TYPE_GPU: log_info("Requesting GPU device "); break;
        case CL_DEVICE_TYPE_CPU: log_info("Requesting CPU device "); break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            log_info("Requesting Accelerator device ");
            break;
        case CL_DEVICE_TYPE_CUSTOM:
            log_info("Requesting Custom device ");
            break;
        case CL_DEVICE_TYPE_DEFAULT:
            log_info("Requesting Default device ");
            break;
        default: log_error("Requesting unknown device "); return EXIT_FAILURE;
    }
    log_info(based_on_env_var ? "based on environment variable "
                              : "based on command line ");
    log_info("for platform index %d and device index %d\n",
             choosen_platform_index, choosen_device_index);

#if defined(__APPLE__)
#if defined(__i386__) || defined(__x86_64__)
#define kHasSSE3 0x00000008
#define kHasSupplementalSSE3 0x00000100
#define kHasSSE4_1 0x00000400
#define kHasSSE4_2 0x00000800
    /* check our environment for a hint to disable SSE variants */
    {
        const char *env = getenv("CL_MAX_SSE");
        if (env)
        {
            extern int _cpu_capabilities;
            int mask = 0;
            if (0 == strcasecmp(env, "SSE4.1"))
                mask = kHasSSE4_2;
            else if (0 == strcasecmp(env, "SSSE3"))
                mask = kHasSSE4_2 | kHasSSE4_1;
            else if (0 == strcasecmp(env, "SSE3"))
                mask = kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3;
            else if (0 == strcasecmp(env, "SSE2"))
                mask =
                    kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3 | kHasSSE3;
            else
            {
                log_error("Error: Unknown CL_MAX_SSE setting: %s\n", env);
                return EXIT_FAILURE;
            }

            log_info("*** Environment: CL_MAX_SSE = %s ***\n", env);
            _cpu_capabilities &= ~mask;
        }
    }
#endif
#endif

    /* Get the platform */
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err)
    {
        print_error(err, "clGetPlatformIDs failed");
        return EXIT_FAILURE;
    }

    platforms =
        (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    if (!platforms || choosen_platform_index >= num_platforms)
    {
        log_error("platform index out of range -- choosen_platform_index (%d) "
                  ">= num_platforms (%d)\n",
                  choosen_platform_index, num_platforms);
        return EXIT_FAILURE;
    }
    BufferOwningPtr<cl_platform_id> platformsBuf(platforms);

    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err)
    {
        print_error(err, "clGetPlatformIDs failed");
        return EXIT_FAILURE;
    }

    /* Get the number of requested devices */
    err = clGetDeviceIDs(platforms[choosen_platform_index], device_type, 0,
                         NULL, &num_devices);
    if (err)
    {
        print_error(err, "clGetDeviceIDs failed");
        return EXIT_FAILURE;
    }

    devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    if (!devices || choosen_device_index >= num_devices)
    {
        log_error("device index out of range -- choosen_device_index (%d) >= "
                  "num_devices (%d)\n",
                  choosen_device_index, num_devices);
        return EXIT_FAILURE;
    }
    BufferOwningPtr<cl_device_id> devicesBuf(devices);


    /* Get the requested device */
    err = clGetDeviceIDs(platforms[choosen_platform_index], device_type,
                         num_devices, devices, NULL);
    if (err)
    {
        print_error(err, "clGetDeviceIDs failed");
        return EXIT_FAILURE;
    }

    device = devices[choosen_device_index];

    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(gDeviceType),
                          &gDeviceType, NULL);
    if (err)
    {
        print_error(err, "Unable to get device type");
        return TEST_FAIL;
    }

    if (printDeviceHeader(device) != CL_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    cl_device_fp_config fpconfig = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fpconfig),
                          &fpconfig, NULL);
    if (err)
    {
        print_error(err,
                    "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed");
        return EXIT_FAILURE;
    }

    gFlushDenormsToZero = (0 == (fpconfig & CL_FP_DENORM));
    log_info("Supports single precision denormals: %s\n",
             gFlushDenormsToZero ? "NO" : "YES");
    log_info("sizeof( void*) = %d  (host)\n", (int)sizeof(void *));

    // detect whether profile of the device is embedded
    char profile[1024] = "";
    err = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(profile), profile,
                          NULL);
    if (err)
    {
        print_error(err, "clGetDeviceInfo for CL_DEVICE_PROFILE failed\n");
        return EXIT_FAILURE;
    }
    gIsEmbedded = NULL != strstr(profile, "EMBEDDED_PROFILE");

    // detect the floating point capabilities
    cl_device_fp_config floatCapabilities = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
                          sizeof(floatCapabilities), &floatCapabilities, NULL);
    if (err)
    {
        print_error(err,
                    "clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed\n");
        return EXIT_FAILURE;
    }

    // Check for problems that only embedded will have
    if (gIsEmbedded)
    {
        // If the device is embedded, we need to detect if the device supports
        // Infinity and NaN
        if ((floatCapabilities & CL_FP_INF_NAN) == 0) gInfNanSupport = 0;

        // check the extensions list to see if ulong and long are supported
        if (!is_extension_available(device, "cles_khr_int64")) gHasLong = 0;
    }

    cl_uint device_address_bits = 0;
    if ((err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
                               sizeof(device_address_bits),
                               &device_address_bits, NULL)))
    {
        print_error(err, "Unable to obtain device address bits");
        return EXIT_FAILURE;
    }
    if (device_address_bits)
        log_info("sizeof( void*) = %d  (device)\n", device_address_bits / 8);
    else
    {
        log_error("Invalid device address bit size returned by device.\n");
        return EXIT_FAILURE;
    }
    const char *suiteName = argv[0];
    if (gCompilationMode == kSpir_v)
    {
        test_status spirv_readiness = check_spirv_compilation_readiness(device);
        if (spirv_readiness != TEST_PASS)
        {
            switch (spirv_readiness)
            {
                case TEST_PASS: break;
                case TEST_FAIL:
                    return suite_did_not_pass_init(suiteName, TEST_FAIL,
                                                   testNum, testList);
                case TEST_SKIP:
                    return suite_did_not_pass_init(suiteName, TEST_SKIP,
                                                   testNum, testList);
                case TEST_SKIPPED_ITSELF:
                    return suite_did_not_pass_init(suiteName, TEST_SKIP,
                                                   testNum, testList);
            }
        }
    }

    /* If we have a device checking function, run it */
    if ((deviceCheckFn != NULL))
    {
        test_status status = deviceCheckFn(device);
        switch (status)
        {
            case TEST_PASS: break;
            case TEST_FAIL:
                return suite_did_not_pass_init(suiteName, TEST_FAIL, testNum,
                                               testList);
            case TEST_SKIP:
                return suite_did_not_pass_init(suiteName, TEST_SKIP, testNum,
                                               testList);
            case TEST_SKIPPED_ITSELF:
                return suite_did_not_pass_init(suiteName, TEST_SKIP, testNum,
                                               testList);
        }
    }

    if (num_elements <= 0) num_elements = DEFAULT_NUM_ELEMENTS;

        // On most platforms which support denorm, default is FTZ off. However,
        // on some hardware where the reference is computed, default might be
        // flush denorms to zero e.g. arm. This creates issues in result
        // verification. Since spec allows the implementation to either flush or
        // not flush denorms to zero, an implementation may choose not be flush
        // i.e. return denorm result whereas reference result may be zero
        // (flushed denorm). Hence we need to disable denorm flushing on host
        // side where reference is being computed to make sure we get
        // non-flushed reference result. If implementation returns flushed
        // result, we correctly take care of that in verification code.
#if defined(__APPLE__) && defined(__arm__)
    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);
#endif
    extern unsigned gNumWorkerThreads;
    test_harness_config config = { forceNoContextCreation, num_elements,
                                   queueProps, gNumWorkerThreads };

    int error = parseAndCallCommandLineTests(argc, argv, device, testNum,
                                             testList, config);

#if defined(__APPLE__) && defined(__arm__)
    // Restore the old FP mode before leaving.
    RestoreFPState(&oldMode);
#endif

    return (error == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int find_matching_tests(test_definition testList[],
                               unsigned char selectedTestList[], int testNum,
                               const char *argument, bool isWildcard)
{
    int found_tests = 0;
    size_t wildcard_length = strlen(argument) - 1; /* -1 for the asterisk */

    for (int i = 0; i < testNum; i++)
    {
        if ((!isWildcard && strcmp(testList[i].name, argument) == 0)
            || (isWildcard
                && strncmp(testList[i].name, argument, wildcard_length) == 0))
        {
            if (selectedTestList[i])
            {
                log_error("ERROR: Test '%s' has already been selected.\n",
                          testList[i].name);
                return EXIT_FAILURE;
            }
            else if (testList[i].func == NULL)
            {
                log_error("ERROR: Test '%s' is missing implementation.\n",
                          testList[i].name);
                return EXIT_FAILURE;
            }
            else
            {
                selectedTestList[i] = 1;
                found_tests = 1;
                if (!isWildcard)
                {
                    break;
                }
            }
        }
    }

    if (!found_tests)
    {
        log_error("ERROR: The argument '%s' did not match any test names.\n",
                  argument);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static void print_results(int failed, int count, const char *name)
{
    if (count < failed)
    {
        count = failed;
    }

    if (failed == 0)
    {
        if (count > 1)
        {
            log_info("PASSED %d of %d %ss.\n", count, count, name);
        }
        else
        {
            log_info("PASSED %s.\n", name);
        }
    }
    else if (failed > 0)
    {
        if (count > 1)
        {
            log_error("FAILED %d of %d %ss.\n", failed, count, name);
        }
        else
        {
            log_error("FAILED %s.\n", name);
        }
    }
    fflush(stdout);
}

int parseAndCallCommandLineTests(int argc, const char *argv[],
                                 cl_device_id device, int testNum,
                                 test_definition testList[],
                                 const test_harness_config &config)
{
    int ret = EXIT_SUCCESS;

    unsigned char *selectedTestList = (unsigned char *)calloc(testNum, 1);

    if (argc == 1)
    {
        /* No actual arguments, all tests will be run. */
        memset(selectedTestList, 1, testNum);
    }
    else
    {
        for (int i = 1; i < argc; i++)
        {
            if (strchr(argv[i], '*') != NULL)
            {
                ret = find_matching_tests(testList, selectedTestList, testNum,
                                          argv[i], true);
            }
            else
            {
                if (strcmp(argv[i], "all") == 0)
                {
                    memset(selectedTestList, 1, testNum);
                    break;
                }
                else
                {
                    ret = find_matching_tests(testList, selectedTestList,
                                              testNum, argv[i], false);
                }
            }

            if (ret == EXIT_FAILURE)
            {
                break;
            }
        }
    }

    if (ret == EXIT_SUCCESS)
    {
        std::vector<test_status> resultTestList(testNum, TEST_PASS);

        callTestFunctions(testList, selectedTestList, resultTestList.data(),
                          testNum, device, config);

        print_results(gFailCount, gTestCount, "sub-test");
        print_results(gTestsFailed, gTestsFailed + gTestsPassed, "test");

        ret = saveResultsToJson(argv[0], testList, selectedTestList,
                                resultTestList.data(), testNum);

        if (std::any_of(resultTestList.begin(), resultTestList.end(),
                        [](test_status result) {
                            switch (result)
                            {
                                case TEST_PASS:
                                case TEST_SKIP: return false;
                                case TEST_FAIL:
                                default: return true;
                            };
                        }))
        {
            ret = EXIT_FAILURE;
        }
    }

    free(selectedTestList);

    return ret;
}

struct test_harness_state
{
    test_definition *tests;
    test_status *results;
    cl_device_id device;
    test_harness_config config;
};

static std::deque<int> gTestQueue;
static std::mutex gTestStateMutex;

void test_function_runner(test_harness_state *state)
{
    int testID;
    test_definition test;
    while (true)
    {
        // Attempt to get a test
        {
            std::lock_guard<std::mutex> lock(gTestStateMutex);

            // The queue is empty, we're done
            if (gTestQueue.size() == 0)
            {
                return;
            }

            // Get the test at the front of the queue
            testID = gTestQueue.front();
            gTestQueue.pop_front();
            test = state->tests[testID];
        }

        // Execute test
        auto status =
            callSingleTestFunction(test, state->device, state->config);

        // Store result
        {
            std::lock_guard<std::mutex> lock(gTestStateMutex);
            state->results[testID] = status;
        }
    }
}

void callTestFunctions(test_definition testList[],
                       unsigned char selectedTestList[],
                       test_status resultTestList[], int testNum,
                       cl_device_id deviceToUse,
                       const test_harness_config &config)
{
    // Execute tests serially
    if (config.numWorkerThreads == 0)
    {
        for (int i = 0; i < testNum; ++i)
        {
            if (selectedTestList[i])
            {
                resultTestList[i] =
                    callSingleTestFunction(testList[i], deviceToUse, config);
            }
        }
        // Execute tests in parallel with the specified number of worker threads
    }
    else
    {
        // Queue all tests that need to run
        for (int i = 0; i < testNum; ++i)
        {
            if (selectedTestList[i])
            {
                gTestQueue.push_back(i);
            }
        }

        // Spawn thread pool
        std::vector<std::thread *> threads;
        test_harness_state state = { testList, resultTestList, deviceToUse,
                                     config };
        for (unsigned i = 0; i < config.numWorkerThreads; i++)
        {
            log_info("Spawning worker thread %u\n", i);
            threads.push_back(new std::thread(test_function_runner, &state));
        }

        // Wait for all threads to complete
        for (auto th : threads)
        {
            th->join();
        }
        assert(gTestQueue.size() == 0);
    }
}

void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info,
                                 size_t cb, void *user_data)
{
    log_info("%s\n", errinfo);
}

// Actual function execution
test_status callSingleTestFunction(test_definition test,
                                   cl_device_id deviceToUse,
                                   const test_harness_config &config)
{
    test_status status;
    cl_int error;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    log_info("%s...\n", test.name);
    fflush(stdout);

    const Version device_version = get_device_cl_version(deviceToUse);
    if (test.min_version > device_version)
    {
        version_expected_info(test.name, "OpenCL",
                              test.min_version.to_string().c_str(),
                              device_version.to_string().c_str());
        return TEST_SKIP;
    }

    if (!check_functions_for_offline_compiler(test.name))
    {
        log_info("Subtest %s tests is not supported in offline compiler "
                 "execution path!\n",
                 test.name);
        return TEST_SKIP;
    }

    /* Create a context to work with, unless we're told not to */
    if (!config.forceNoContextCreation)
    {
        context = clCreateContext(NULL, 1, &deviceToUse, notify_callback, NULL,
                                  &error);
        if (!context)
        {
            print_error(error, "Unable to create testing context");
            gFailCount++;
            gTestsFailed++;
            return TEST_FAIL;
        }

        if (device_version < Version(2, 0))
        {
            queue = clCreateCommandQueue(context, deviceToUse,
                                         config.queueProps, &error);
        }
        else
        {
            const cl_command_queue_properties cmd_queueProps =
                (config.queueProps) ? CL_QUEUE_PROPERTIES : 0;
            cl_command_queue_properties queueCreateProps[] = {
                cmd_queueProps, config.queueProps, 0
            };
            queue = clCreateCommandQueueWithProperties(
                context, deviceToUse, &queueCreateProps[0], &error);
        }

        if (queue == NULL)
        {
            print_error(error, "Unable to create testing command queue");
            clReleaseContext(context);
            gFailCount++;
            gTestsFailed++;
            return TEST_FAIL;
        }
    }

    /* Run the test and print the result */
    if (test.func == NULL)
    {
        // Skip unimplemented test, can happen when all of the tests are
        // selected
        log_info("%s test currently not implemented\n", test.name);
        status = TEST_SKIP;
    }
    else
    {
        int ret =
            test.func(deviceToUse, context, queue, config.numElementsToUse);
        if (ret == TEST_SKIPPED_ITSELF)
        {
            /* Tests can also let us know they're not supported by the
             * implementation */
            log_info("%s test not supported\n", test.name);
            status = TEST_SKIP;
        }
        else
        {
            /* Print result */
            if (ret == 0)
            {
                log_info("%s passed\n", test.name);
                gTestsPassed++;
                status = TEST_PASS;
            }
            else
            {
                log_error("%s FAILED\n", test.name);
                gTestsFailed++;
                status = TEST_FAIL;
            }
        }
    }

    /* Release the context */
    if (!config.forceNoContextCreation)
    {
        int error = clFinish(queue);
        if (error)
        {
            log_error("clFinish failed: %s\n", IGetErrorString(error));
            gFailCount++;
            gTestsFailed++;
            status = TEST_FAIL;
        }
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }

    return status;
}

#if !defined(__APPLE__)
void memset_pattern4(void *dest, const void *src_pattern, size_t bytes)
{
    uint32_t pat = ((uint32_t *)src_pattern)[0];
    size_t count = bytes / 4;
    size_t i;
    uint32_t *d = (uint32_t *)dest;

    for (i = 0; i < count; i++) d[i] = pat;

    d += i;

    bytes &= 3;
    if (bytes) memcpy(d, src_pattern, bytes);
}
#endif

cl_device_type GetDeviceType(cl_device_id d)
{
    cl_device_type result = -1;
    cl_int err =
        clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(result), &result, NULL);
    if (CL_SUCCESS != err)
        log_error("ERROR: Unable to get device type for device %p\n", d);
    return result;
}


cl_device_id GetOpposingDevice(cl_device_id device)
{
    cl_int error;
    cl_device_id *otherDevices;
    cl_uint actualCount;
    cl_platform_id plat;

    // Get the platform of the device to use for getting a list of devices
    error =
        clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(plat), &plat, NULL);
    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to get device's platform");
        return NULL;
    }

    // Get a list of all devices
    error = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &actualCount);
    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to get list of devices size");
        return NULL;
    }
    otherDevices = (cl_device_id *)malloc(actualCount * sizeof(cl_device_id));
    if (NULL == otherDevices)
    {
        print_error(error, "Unable to allocate list of other devices.");
        return NULL;
    }
    BufferOwningPtr<cl_device_id> otherDevicesBuf(otherDevices);

    error = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, actualCount, otherDevices,
                           NULL);
    if (error != CL_SUCCESS)
    {
        print_error(error, "Unable to get list of devices");
        return NULL;
    }

    if (actualCount == 1)
    {
        return device; // NULL means error, returning self means we couldn't
                       // find another one
    }

    // Loop and just find one that isn't the one we were given
    cl_uint i;
    for (i = 0; i < actualCount; i++)
    {
        if (otherDevices[i] != device)
        {
            cl_device_type newType;
            error = clGetDeviceInfo(otherDevices[i], CL_DEVICE_TYPE,
                                    sizeof(newType), &newType, NULL);
            if (error != CL_SUCCESS)
            {
                print_error(error,
                            "Unable to get device type for other device");
                return NULL;
            }
            cl_device_id result = otherDevices[i];
            return result;
        }
    }

    // Should never get here
    return NULL;
}

Version get_device_cl_version(cl_device_id device)
{
    size_t str_size;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &str_size);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    std::vector<char> str(str_size);
    err =
        clGetDeviceInfo(device, CL_DEVICE_VERSION, str_size, str.data(), NULL);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");

    if (strstr(str.data(), "OpenCL 1.0") != NULL)
        return Version(1, 0);
    else if (strstr(str.data(), "OpenCL 1.1") != NULL)
        return Version(1, 1);
    else if (strstr(str.data(), "OpenCL 1.2") != NULL)
        return Version(1, 2);
    else if (strstr(str.data(), "OpenCL 2.0") != NULL)
        return Version(2, 0);
    else if (strstr(str.data(), "OpenCL 2.1") != NULL)
        return Version(2, 1);
    else if (strstr(str.data(), "OpenCL 2.2") != NULL)
        return Version(2, 2);
    else if (strstr(str.data(), "OpenCL 3.0") != NULL)
        return Version(3, 0);

    throw std::runtime_error(std::string("Unknown OpenCL version: ")
                             + str.data());
}

bool check_device_spirv_version_reported(cl_device_id device)
{
    size_t str_size;
    cl_int err;
    std::vector<char> str;
    if (gCoreILProgram)
    {
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, 0, NULL, &str_size);
        if (err != CL_SUCCESS)
        {
            log_error(
                "clGetDeviceInfo: cannot read CL_DEVICE_IL_VERSION size;");
            return false;
        }

        str.resize(str_size);
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, str_size,
                              str.data(), NULL);
        if (err != CL_SUCCESS)
        {
            log_error(
                "clGetDeviceInfo: cannot read CL_DEVICE_IL_VERSION value;");
            return false;
        }
    }
    else
    {
        cl_int err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION_KHR, 0, NULL,
                                     &str_size);
        if (err != CL_SUCCESS)
        {
            log_error(
                "clGetDeviceInfo: cannot read CL_DEVICE_IL_VERSION_KHR size;");
            return false;
        }

        str.resize(str_size);
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION_KHR, str_size,
                              str.data(), NULL);
        if (err != CL_SUCCESS)
        {
            log_error(
                "clGetDeviceInfo: cannot read CL_DEVICE_IL_VERSION_KHR value;");
            return false;
        }
    }

    if (strstr(str.data(), "SPIR-V") == NULL)
    {
        log_info("This device does not support SPIR-V offline compilation.\n");
        return false;
    }
    else
    {
        Version spirv_version = get_device_spirv_il_version(device);
        log_info("This device supports SPIR-V offline compilation. SPIR-V "
                 "version is %s\n",
                 spirv_version.to_string().c_str());
    }
    return true;
}

Version get_device_spirv_il_version(cl_device_id device)
{
    size_t str_size;
    cl_int err;
    std::vector<char> str;
    if (gCoreILProgram)
    {
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, 0, NULL, &str_size);
        ASSERT_SUCCESS(err, "clGetDeviceInfo");

        str.resize(str_size);
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION, str_size,
                              str.data(), NULL);
        ASSERT_SUCCESS(err, "clGetDeviceInfo");
    }
    else
    {
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION_KHR, 0, NULL,
                              &str_size);
        ASSERT_SUCCESS(err, "clGetDeviceInfo");

        str.resize(str_size);
        err = clGetDeviceInfo(device, CL_DEVICE_IL_VERSION_KHR, str_size,
                              str.data(), NULL);
        ASSERT_SUCCESS(err, "clGetDeviceInfo");
    }

    // Because this query returns a space-separated list of IL version strings
    // we should check for SPIR-V versions in reverse order, to return the
    // highest version supported.
    if (strstr(str.data(), "SPIR-V_1.5") != NULL)
        return Version(1, 5);
    else if (strstr(str.data(), "SPIR-V_1.4") != NULL)
        return Version(1, 4);
    else if (strstr(str.data(), "SPIR-V_1.3") != NULL)
        return Version(1, 3);
    else if (strstr(str.data(), "SPIR-V_1.2") != NULL)
        return Version(1, 2);
    else if (strstr(str.data(), "SPIR-V_1.1") != NULL)
        return Version(1, 1);
    else if (strstr(str.data(), "SPIR-V_1.0") != NULL)
        return Version(1, 0);

    throw std::runtime_error(std::string("Unknown SPIR-V version: ")
                             + str.data());
}

test_status check_spirv_compilation_readiness(cl_device_id device)
{
    auto ocl_version = get_device_cl_version(device);
    auto ocl_expected_min_version = Version(2, 1);

    if (ocl_version < ocl_expected_min_version)
    {
        if (is_extension_available(device, "cl_khr_il_program"))
        {
            gCoreILProgram = false;
            bool spirv_supported = check_device_spirv_version_reported(device);
            if (spirv_supported == false)
            {
                log_error("SPIR-V intermediate language not supported !!! "
                          "OpenCL %s requires support.\n",
                          ocl_version.to_string().c_str());
                return TEST_FAIL;
            }
            else
            {
                return TEST_PASS;
            }
        }
        else
        {
            log_error("SPIR-V intermediate language support on OpenCL version "
                      "%s requires cl_khr_il_program extension.\n",
                      ocl_version.to_string().c_str());
            return TEST_SKIP;
        }
    }

    bool spirv_supported = check_device_spirv_version_reported(device);
    if (ocl_version >= ocl_expected_min_version && ocl_version <= Version(2, 2))
    {
        if (spirv_supported == false)
        {
            log_error("SPIR-V intermediate language not supported !!! OpenCL "
                      "%s requires support.\n",
                      ocl_version.to_string().c_str());
            return TEST_FAIL;
        }
    }

    if (ocl_version > Version(2, 2))
    {
        if (spirv_supported == false)
        {
            log_info("SPIR-V intermediate language not supported in OpenCL %s. "
                     "Test skipped.\n",
                     ocl_version.to_string().c_str());
            return TEST_SKIP;
        }
    }
    return TEST_PASS;
}

cl_platform_id getPlatformFromDevice(cl_device_id deviceID)
{
    cl_platform_id platform = nullptr;
    cl_int err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platform),
                                 &platform, nullptr);
    ASSERT_SUCCESS(err, "clGetDeviceInfo");
    return platform;
}

/**
 * Helper to return a string containing platform information
 * for the specified platform info parameter.
 */
std::string get_platform_info_string(cl_platform_id platform,
                                     cl_platform_info param_name)
{
    size_t size = 0;
    int err;

    if ((err = clGetPlatformInfo(platform, param_name, 0, NULL, &size))
            != CL_SUCCESS
        || size == 0)
    {
        throw std::runtime_error("clGetPlatformInfo failed\n");
    }

    std::vector<char> info(size);

    if ((err = clGetPlatformInfo(platform, param_name, size, info.data(), NULL))
        != CL_SUCCESS)
    {
        throw std::runtime_error("clGetPlatformInfo failed\n");
    }

    /* The returned string does not include the null terminator. */
    return std::string(info.data(), size - 1);
}

bool is_platform_extension_available(cl_platform_id platform,
                                     const char *extensionName)
{
    std::string extString =
        get_platform_info_string(platform, CL_PLATFORM_EXTENSIONS);
    return extString.find(extensionName) != std::string::npos;
}

void PrintArch(void)
{
    vlog("sizeof( void*) = %zu\n", sizeof(void *));
#if defined(__ppc__)
    vlog("ARCH:\tppc\n");
#elif defined(__ppc64__)
    vlog("ARCH:\tppc64\n");
#elif defined(__PPC__)
    vlog("ARCH:\tppc\n");
#elif defined(__i386__)
    vlog("ARCH:\ti386\n");
#elif defined(__x86_64__)
    vlog("ARCH:\tx86_64\n");
#elif defined(__arm__)
    vlog("ARCH:\tarm\n");
#elif defined(__aarch64__)
    vlog("ARCH:\taarch64\n");
#elif defined(_WIN32)
    vlog("ARCH:\tWindows\n");
#elif defined(__mips__)
    vlog("ARCH:\tmips\n");
#else
#error unknown arch
#endif

#if defined(__APPLE__)

    int type = 0;
    size_t typeSize = sizeof(type);
    sysctlbyname("hw.cputype", &type, &typeSize, NULL, 0);
    vlog("cpu type:\t%d\n", type);
    typeSize = sizeof(type);
    sysctlbyname("hw.cpusubtype", &type, &typeSize, NULL, 0);
    vlog("cpu subtype:\t%d\n", type);

#elif defined(__linux__)
    struct utsname buffer;

    if (uname(&buffer) != 0)
    {
        vlog("uname error");
    }
    else
    {
        vlog("system name = %s\n", buffer.sysname);
        vlog("node name   = %s\n", buffer.nodename);
        vlog("release     = %s\n", buffer.release);
        vlog("version     = %s\n", buffer.version);
        vlog("machine     = %s\n", buffer.machine);
    }
#endif
}
