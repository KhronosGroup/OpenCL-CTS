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

#include "clImageHelper.h"
#include <string>
#include <sstream>

#include <string>

class Version {
public:
    Version(): m_major(0), m_minor(0) {}
    Version(int major, int minor): m_major(major), m_minor(minor) {}
    bool operator>(const Version &rhs) const { return to_int() > rhs.to_int(); }
    bool operator<(const Version &rhs) const { return to_int() < rhs.to_int(); }
    bool operator<=(const Version &rhs) const
    {
        return to_int() <= rhs.to_int();
    }
    bool operator>=(const Version &rhs) const
    {
        return to_int() >= rhs.to_int();
    }
    bool operator==(const Version &rhs) const
    {
        return to_int() == rhs.to_int();
    }
    int to_int() const { return m_major * 10 + m_minor; }
    std::string to_string() const
    {
        std::stringstream ss;
        ss << m_major << "." << m_minor;
        return ss.str();
    }

private:
    int m_major;
    int m_minor;
};

Version get_device_cl_version(cl_device_id device);

#define ADD_TEST(fn)                                                           \
    {                                                                          \
        test_##fn, #fn, Version(1, 0)                                          \
    }
#define ADD_TEST_VERSION(fn, ver)                                              \
    {                                                                          \
        test_##fn, #fn, ver                                                    \
    }

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

typedef int (*test_function_pointer)(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements);

typedef struct test_definition
{
    test_function_pointer func;
    const char *name;
    Version min_version;
} test_definition;


typedef enum test_status
{
    TEST_PASS = 0,
    TEST_FAIL = 1,
    TEST_SKIP = 2,
    TEST_SKIPPED_ITSELF = -100,
} test_status;

struct test_harness_config
{
    int forceNoContextCreation;
    int numElementsToUse;
    cl_command_queue_properties queueProps;
    unsigned numWorkerThreads;
};

extern int gFailCount;
extern int gTestCount;
extern cl_uint gReSeed;
extern cl_uint gRandomSeed;

// Supply a list of functions to test here. This will allocate a CL device,
// create a context, all that setup work, and then call each function in turn as
// dictatated by the passed arguments. Returns EXIT_SUCCESS iff all tests
// succeeded or the tests were listed, otherwise return EXIT_FAILURE.
extern int runTestHarness(int argc, const char *argv[], int testNum,
                          test_definition testList[],
                          int forceNoContextCreation,
                          cl_command_queue_properties queueProps);

// Device checking function. See runTestHarnessWithCheck. If this function
// returns anything other than TEST_PASS, the harness exits.
typedef test_status (*DeviceCheckFn)(cl_device_id device);

// Same as runTestHarness, but also supplies a function that checks the created
// device for required functionality. Returns EXIT_SUCCESS iff all tests
// succeeded or the tests were listed, otherwise return EXIT_FAILURE.
extern int runTestHarnessWithCheck(int argc, const char *argv[], int testNum,
                                   test_definition testList[],
                                   int forceNoContextCreation,
                                   cl_command_queue_properties queueProps,
                                   DeviceCheckFn deviceCheckFn);

// The command line parser used by runTestHarness to break up parameters into
// calls to callTestFunctions
extern int parseAndCallCommandLineTests(int argc, const char *argv[],
                                        cl_device_id device, int testNum,
                                        test_definition testList[],
                                        const test_harness_config &config);

// Call this function if you need to do all the setup work yourself, and just
// need the function list called/ managed.
//    testList is the data structure that contains test functions and its names
//    selectedTestList is an array of integers (treated as bools) which tell
//    which function is to be called,
//       each element at index i, corresponds to the element in testList at
//       index i
//    resultTestList is an array of statuses which contain the result of each
//    selected test testNum is the number of tests in testList, selectedTestList
//    and resultTestList contextProps are used to create a testing context for
//    each test deviceToUse and config are all just passed to each
//    test function
extern void callTestFunctions(test_definition testList[],
                              unsigned char selectedTestList[],
                              test_status resultTestList[], int testNum,
                              cl_device_id deviceToUse,
                              const test_harness_config &config);

// This function is called by callTestFunctions, once per function, to do setup,
// call, logging and cleanup
extern test_status callSingleTestFunction(test_definition test,
                                          cl_device_id deviceToUse,
                                          const test_harness_config &config);

///// Miscellaneous steps

// standard callback function for context pfn_notify
extern void CL_CALLBACK notify_callback(const char *errinfo,
                                        const void *private_info, size_t cb,
                                        void *user_data);

extern cl_device_type GetDeviceType(cl_device_id);

// Given a device (most likely passed in by the harness, but not required), will
// attempt to find a DIFFERENT device and return it. Useful for finding another
// device to run multi-device tests against. Note that returning NULL means an
// error was hit, but if no error was hit and the device passed in is the only
// device available, the SAME device is returned, so check!
extern cl_device_id GetOpposingDevice(cl_device_id device);

Version get_device_spirv_il_version(cl_device_id device);
bool check_device_spirv_il_support(cl_device_id device);
void version_expected_info(const char *test_name, const char *api_name,
                           const char *expected_version,
                           const char *device_version);
test_status check_spirv_compilation_readiness(cl_device_id device);


extern int gFlushDenormsToZero; // This is set to 1 if the device does not
                                // support denorms (CL_FP_DENORM)
extern int gInfNanSupport; // This is set to 1 if the device supports infinities
                           // and NaNs
extern int gIsEmbedded; // This is set to 1 if the device is an embedded device
extern int gHasLong; // This is set to 1 if the device suppots long and ulong
                     // types in OpenCL C.
extern bool gCoreILProgram;

extern cl_platform_id getPlatformFromDevice(cl_device_id deviceID);

#if !defined(__APPLE__)
void memset_pattern4(void *, const void *, size_t);
#endif

extern void PrintArch(void);


#endif // _testHarness_h
