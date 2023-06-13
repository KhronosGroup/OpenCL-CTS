/******************************************************************
Copyright (c) 2016 The Khronos Group Inc. All Rights Reserved.

This code is protected by copyright laws and contains material proprietary to
the Khronos Group, Inc. This is UNPUBLISHED PROPRIETARY SOURCE CODE that may not
be disclosed in whole or in part to third parties, and may not be reproduced,
republished, distributed, transmitted, displayed, broadcast or otherwise
exploited in any manner without the express prior written permission of Khronos
Group. The receipt or possession of this code does not convey any rights to
reproduce, disclose, or distribute its contents, or to manufacture, use, or sell
anything that it may describe, in whole or in part other than under the terms of
the Khronos Adopters Agreement or Khronos Conformance Test Source License
Agreement as executed between Khronos and the recipient.
******************************************************************/

#pragma once

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"
#include "harness/compat.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"

#include <vector>

#define SPIRV_CHECK_ERROR(err, fmt, ...)                                       \
    do                                                                         \
    {                                                                          \
        if (err == CL_SUCCESS) break;                                          \
        log_error("%s(%d): Error %d\n" fmt "\n", __FILE__, __LINE__, err,      \
                  ##__VA_ARGS__);                                              \
        return -1;                                                             \
    } while (0)


class baseTestClass {
public:
    baseTestClass() {}
    virtual test_function_pointer getFunction() = 0;
};

class spirvTestsRegistry {
private:
    std::vector<baseTestClass *> testClasses;
    std::vector<test_definition> testDefinitions;

public:
    static spirvTestsRegistry &getInstance();

    test_definition *getTestDefinitions();

    size_t getNumTests();

    void addTestClass(baseTestClass *test, const char *testName,
                      Version version);
    spirvTestsRegistry() {}
};

template <typename T> T *createAndRegister(const char *name, Version version)
{
    T *testClass = new T();
    spirvTestsRegistry::getInstance().addTestClass((baseTestClass *)testClass,
                                                   name, version);
    return testClass;
}

#define TEST_SPIRV_FUNC_VERSION(name, version)                                 \
    extern int test_##name(cl_device_id deviceID, cl_context context,          \
                           cl_command_queue queue, int num_elements);          \
    class test_##name##_class : public baseTestClass {                         \
    private:                                                                   \
        test_function_pointer fn;                                              \
                                                                               \
    public:                                                                    \
        test_##name##_class(): fn(test_##name) {}                              \
        test_function_pointer getFunction() { return fn; }                     \
    };                                                                         \
    test_##name##_class *var_##name =                                          \
        createAndRegister<test_##name##_class>(#name, version);                \
    int test_##name(cl_device_id deviceID, cl_context context,                 \
                    cl_command_queue queue, int num_elements)

#define TEST_SPIRV_FUNC(name) TEST_SPIRV_FUNC_VERSION(name, Version(1, 2))

struct spec_const
{
    spec_const(cl_int id = 0, size_t sizet = 0, const void *value = NULL)
        : spec_id(id), spec_size(sizet), spec_value(value){};
    cl_int spec_id;
    size_t spec_size;
    const void *spec_value;
};

int get_program_with_il(clProgramWrapper &prog, const cl_device_id deviceID,
                        const cl_context context, const char *prog_name,
                        spec_const spec_const_def = spec_const());
std::vector<unsigned char> readSPIRV(const char *file_name);
