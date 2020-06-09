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
#ifndef _testHarness_h
#define _testHarness_h

#include "threadTesting.h"
#include "clImageHelper.h"
#include "feature.h"
#include "version.h"

#define ADD_TEST_FEATURE(fn, feat)                                             \
    test_definition { test_##fn, #fn, Version(1, 0), feat }

#define ADD_TEST(fn)                                                           \
    test_definition { test_##fn, #fn, Version(1, 0), {} }
#define ADD_TEST_VERSION(fn, ver)                                              \
    test_definition { test_##fn, #fn, ver, {} }
#define NOT_IMPLEMENTED_TEST(fn)                                               \
    test_definition { NULL, #fn, Version(0, 0), {} }

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

struct test_definition
{
    test_definition(basefn fn, const char *n, Version v, const feature &feat)
        : func(fn), name(n), min_version(v), required_feature(feat)
    {}
    basefn func;
    const char* name;
    Version min_version;
    const feature &required_feature;
};

typedef enum test_status
{
    TEST_PASS = 0,
    TEST_FAIL = 1,
    TEST_SKIP = 2,
} test_status;

extern int gFailCount;
extern int gTestCount;
extern cl_uint gReSeed;
extern cl_uint gRandomSeed;

// Supply a list of functions to test here. This will allocate a CL device, create a context, all that
// setup work, and then call each function in turn as dictatated by the passed arguments.
// Returns EXIT_SUCCESS iff all tests succeeded or the tests were listed,
// otherwise return EXIT_FAILURE.
extern int runTestHarness( int argc, const char *argv[], int testNum, test_definition testList[],
                           int imageSupportRequired, int forceNoContextCreation, cl_command_queue_properties queueProps );

// Device checking function. See runTestHarnessWithCheck. If this function returns anything other than TEST_PASS, the harness exits.
typedef test_status (*DeviceCheckFn)( cl_device_id device );

// Same as runTestHarness, but also supplies a function that checks the created device for required functionality.
// Returns EXIT_SUCCESS iff all tests succeeded or the tests were listed,
// otherwise return EXIT_FAILURE.
extern int runTestHarnessWithCheck( int argc, const char *argv[], int testNum, test_definition testList[],
                                    int forceNoContextCreation, cl_command_queue_properties queueProps,
                                    DeviceCheckFn deviceCheckFn );

// The command line parser used by runTestHarness to break up parameters into calls to callTestFunctions
extern int parseAndCallCommandLineTests( int argc, const char *argv[], cl_device_id device, int testNum,
                                         test_definition testList[], int forceNoContextCreation,
                                         cl_command_queue_properties queueProps, int num_elements );

// Call this function if you need to do all the setup work yourself, and just need the function list called/
// managed.
//    testList is the data structure that contains test functions and its names
//    selectedTestList is an array of integers (treated as bools) which tell which function is to be called,
//       each element at index i, corresponds to the element in testList at index i
//    resultTestList is an array of statuses which contain the result of each selected test
//    testNum is the number of tests in testList, selectedTestList and resultTestList
//    contextProps are used to create a testing context for each test
//    deviceToUse and numElementsToUse are all just passed to each test function
extern void callTestFunctions( test_definition testList[], unsigned char selectedTestList[], test_status resultTestList[],
                               int testNum, cl_device_id deviceToUse, int forceNoContextCreation, int numElementsToUse,
                               cl_command_queue_properties queueProps );

// This function is called by callTestFunctions, once per function, to do setup, call, logging and cleanup
extern test_status callSingleTestFunction( test_definition test, cl_device_id deviceToUse, int forceNoContextCreation,
                                           int numElementsToUse, cl_command_queue_properties queueProps );

///// Miscellaneous steps

// standard callback function for context pfn_notify
extern void CL_CALLBACK notify_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data);

extern cl_device_type GetDeviceType( cl_device_id );

// Given a device (most likely passed in by the harness, but not required), will attempt to find
// a DIFFERENT device and return it. Useful for finding another device to run multi-device tests against.
// Note that returning NULL means an error was hit, but if no error was hit and the device passed in
// is the only device available, the SAME device is returned, so check!
extern cl_device_id GetOpposingDevice( cl_device_id device );

void version_expected_info(const char * test_name, const char * expected_version, const char * device_version);


extern int      gFlushDenormsToZero;    // This is set to 1 if the device does not support denorms (CL_FP_DENORM)
extern int      gInfNanSupport;         // This is set to 1 if the device supports infinities and NaNs
extern int        gIsEmbedded;            // This is set to 1 if the device is an embedded device
extern int        gHasLong;               // This is set to 1 if the device suppots long and ulong types in OpenCL C.
extern int      gIsOpenCL_C_1_0_Device; // This is set to 1 if the device supports only OpenCL C 1.0.

#if ! defined( __APPLE__ )
    void     memset_pattern4(void *, const void *, size_t);
#endif

extern void PrintArch(void);


#endif // _testHarness_h


