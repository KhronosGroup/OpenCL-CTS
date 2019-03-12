/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to the Khronos Group, Inc.
This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not be disclosed in whole or in part to
third parties, and may not be reproduced, republished, distributed, transmitted, displayed,
broadcast or otherwise exploited in any manner without the express prior written permission
of Khronos Group. The receipt or possession of this code does not convey any rights to reproduce,
disclose, or distribute its contents, or to manufacture, use, or sell anything that it may describe,
in whole or in part other than under the terms of the Khronos Adopters Agreement
or Khronos Conformance Test Source License Agreement as executed between Khronos and the recipient.
******************************************************************/

#pragma once

#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/typeWrappers.h"
#include "../../test_common/harness/conversions.h"
#include "../../test_common/harness/mt19937.h"
#include "../../test_common/harness/compat.h"
#include "../../test_common/harness/testHarness.h"
#include "../../test_common/harness/parseParameters.h"

#include <vector>

#define SPIRV_CHECK_ERROR(err, fmt, ...) do {               \
        if (err == CL_SUCCESS) break;                       \
        log_error("%s(%d): Error %d\n" fmt "\n",            \
                  __FILE__, __LINE__, err, ##__VA_ARGS__);  \
        return -1;                                          \
    } while(0)


class baseTestClass
{
public:
    baseTestClass() {}
    virtual basefn getFunction() = 0;
};

// parseAndCallCommandLineTests called by runTestHarness expects there to
// be an additinal test at the end of the get list called "all", if the
// user requests "all" on the command line it will not call the specific
// test function but instead run all the tests. This is required to avoid
// off by 1 failures when the user requested the last test in the test list
// by name on the command line.
class test_all_class : public baseTestClass
{
public:
    basefn getFunction() override {
        // Return a function which will never be called because the "all" test
        // name is intercepted and instead runs all the tests.
        return [](cl_device_id, cl_context, cl_command_queue, int) -> int { return 0; };
    }
};

class spirvTestsRegistry {
private:
    std::vector<const char *> testNames;
    std::vector<baseTestClass *> testClasses;
    std::vector<basefn> tests;

public:

    static spirvTestsRegistry& getInstance();

    const char **getTestNames();

    basefn *getTests();

    size_t getNumTests();

    void addTestClass(baseTestClass *test, const char *testName);
    spirvTestsRegistry() {}
};

template<typename T>
T* createAndRegister(const char *name)
{
    T *testClass = new T();
    spirvTestsRegistry::getInstance().addTestClass((baseTestClass *)testClass, name);
    return testClass;
}

#define TEST_SPIRV_FUNC(name)                           \
    extern int test_##name(cl_device_id deviceID,       \
                           cl_context context,          \
                           cl_command_queue queue,      \
                           int num_elements);           \
    class test_##name##_class  : public baseTestClass   \
    {                                                   \
    private:                                            \
        basefn fn;                                      \
                                                        \
    public:                                             \
    test_##name##_class() : fn(test_##name)             \
        {                                               \
        }                                               \
        basefn getFunction()                            \
        {                                               \
            return fn;                                  \
        }                                               \
    };                                                  \
    test_##name##_class *var_##name =                   \
        createAndRegister<test_##name##_class>(#name);  \
    int test_##name(cl_device_id deviceID,              \
                    cl_context context,                 \
                    cl_command_queue queue,             \
                    int num_elements)

std::vector<unsigned char> readSPIRV(const char *file_name);

int get_program_with_il(clProgramWrapper &prog,
                        const cl_device_id deviceID,
                        const cl_context context,
                        const char *prog_name);
