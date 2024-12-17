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
#include <vector>

class Version {
public:
    Version(): m_major(0), m_minor(0) {}

    Version(cl_uint major, cl_uint minor): m_major(major), m_minor(minor) {}
    int major() const { return m_major; }
    int minor() const { return m_minor; }
    bool operator>(const Version &rhs) const
    {
        return to_uint() > rhs.to_uint();
    }
    bool operator<(const Version &rhs) const
    {
        return to_uint() < rhs.to_uint();
    }
    bool operator<=(const Version &rhs) const
    {
        return to_uint() <= rhs.to_uint();
    }
    bool operator>=(const Version &rhs) const
    {
        return to_uint() >= rhs.to_uint();
    }
    bool operator==(const Version &rhs) const
    {
        return to_uint() == rhs.to_uint();
    }
    cl_uint to_uint() const { return m_major * 10 + m_minor; }
    std::string to_string() const
    {
        std::stringstream ss;
        ss << m_major << "." << m_minor;
        return ss.str();
    }

private:
    cl_uint m_major;
    cl_uint m_minor;
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


struct test
{
    virtual test_function_pointer getFunction() = 0;
};

class test_registry {
private:
    std::vector<test *> m_tests;
    std::vector<test_definition> m_definitions;

public:
    static test_registry &getInstance();

    test_definition *definitions();

    size_t num_tests();

    void add_test(test *t, const char *name, Version version);
    test_registry() {}
};

template <typename T> T *register_test(const char *name, Version version)
{
    T *t = new T();
    test_registry::getInstance().add_test((test *)t, name, version);
    return t;
}

#define REGISTER_TEST_VERSION(name, version)                                   \
    extern int test_##name(cl_device_id device, cl_context context,            \
                           cl_command_queue queue, int num_elements);          \
    class test_##name##_class : public test {                                  \
    private:                                                                   \
        test_function_pointer fn;                                              \
                                                                               \
    public:                                                                    \
        test_##name##_class(): fn(test_##name) {}                              \
        test_function_pointer getFunction() { return fn; }                     \
    };                                                                         \
    test_##name##_class *var_##name =                                          \
        register_test<test_##name##_class>(#name, version);                    \
    int test_##name(cl_device_id device, cl_context context,                   \
                    cl_command_queue queue, int num_elements)

#define REGISTER_TEST(name) REGISTER_TEST_VERSION(name, Version(1, 2))

#define REQUIRE_EXTENSION(name)                                                \
    do                                                                         \
    {                                                                          \
        if (!is_extension_available(deviceID, name))                           \
        {                                                                      \
            log_info(name                                                      \
                     " is not supported on this device. Skipping test.\n");    \
            return TEST_SKIPPED_ITSELF;                                        \
        }                                                                      \
    } while (0)

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
extern std::string get_platform_info_string(cl_platform_id platform,
                                            cl_platform_info param_name);
extern bool is_platform_extension_available(cl_platform_id platform,
                                            const char *extensionName);

#if !defined(__APPLE__)
void memset_pattern4(void *, const void *, size_t);
#endif

extern void PrintArch(void);


#endif // _testHarness_h
