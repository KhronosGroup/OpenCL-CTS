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
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "base.h"

#include <string>
#include <vector>
#include <algorithm>

class CBasicTest : CTest {
public:
    CBasicTest(const std::vector<std::string>& kernel) : CTest(), _kernels(kernel) {

    }

    CBasicTest(const std::string& kernel) : CTest(), _kernels(1, kernel) {

    }

    int ExecuteSubcase(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, const std::string& src) {
        cl_int error;

        clProgramWrapper program;
        clKernelWrapper kernel;

        const char *srcPtr = src.c_str();

        if (create_single_kernel_helper(context, &program, &kernel, 1, &srcPtr,
                                        "testKernel"))
        {
            log_error("create_single_kernel_helper failed");
            return -1;
        }

        size_t bufferSize = num_elements * sizeof(cl_uint);
        clMemWrapper buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
        test_error(error, "clCreateBuffer failed");

        error = clSetKernelArg(kernel, 0, sizeof(buffer), &buffer);
        test_error(error, "clSetKernelArg failed");

        size_t globalWorkGroupSize = num_elements;
        size_t localWorkGroupSize = 0;
        error = get_max_common_work_group_size(context, kernel, globalWorkGroupSize, &localWorkGroupSize);
        test_error(error, "Unable to get common work group size");

        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkGroupSize, &localWorkGroupSize, 0, NULL, NULL);
        test_error(error, "clEnqueueNDRangeKernel failed");

        // verify results
        std::vector<cl_uint> results(num_elements);

        error = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize, &results[0], 0, NULL, NULL);
        test_error(error, "clEnqueueReadBuffer failed");

        size_t passCount = std::count(results.begin(), results.end(), 1);
        if (passCount != results.size()) {
            std::vector<cl_uint>::iterator iter = std::find(results.begin(), results.end(), 0);
            log_error("Verification on device failed at index %ld\n", std::distance(results.begin(), iter));
            log_error("%ld out of %ld failed\n", (results.size()-passCount), results.size());
            return -1;
        }

        return CL_SUCCESS;
    }

    int Execute(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
        cl_int result = CL_SUCCESS;

        for (std::vector<std::string>::const_iterator it = _kernels.begin(); it != _kernels.end(); ++it) {
            log_info("Executing subcase #%ld out of %ld\n", (it - _kernels.begin() + 1), _kernels.size());

            result |= ExecuteSubcase(deviceID, context, queue, num_elements, *it);
        }

        return result;
    }

private:
    const std::vector<std::string> _kernels;
};

int test_function_get_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL "__global uchar guchar = 3;"
        NL
        NL "bool helperFunction(int *intp, float *floatp, uchar *ucharp, ushort *ushortp, long *longp) {"
        NL "    if (!isFenceValid(get_fence(intp)))"
        NL "        return false;"
        NL "    if (!isFenceValid(get_fence(floatp)))"
        NL "        return false;"
        NL "    if (!isFenceValid(get_fence(ucharp)))"
        NL "        return false;"
        NL "    if (!isFenceValid(get_fence(ushortp)))"
        NL "        return false;"
        NL "    if (!isFenceValid(get_fence(longp)))"
        NL "        return false;"
        NL
        NL "    if (*intp != 1 || *floatp != 2.0f || *ucharp != 3 || *ushortp != 4 || *longp != 5)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local float lfloat;"
        NL "    lfloat = 2.0f;"
        NL "    __local ushort lushort;"
        NL "    lushort = 4;"
        NL "    long plong = 5;"
        NL
        NL "    __global int *gintp = &gint;"
        NL "    __local float *lfloatp = &lfloat;"
        NL "    __global uchar *gucharp = &guchar;"
        NL "    __local ushort *lushortp = &lushort;"
        NL "    __private long *plongp = &plong;"
        NL
        NL "    results[tid] = helperFunction(gintp, lfloatp, gucharp, lushortp, plongp);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_function_to_address_space(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION =
        NL
        NL "__global int gint = 1;"
        NL "__global uchar guchar = 3;"
        NL
        NL "bool helperFunction(int *gintp, float *lfloatp, uchar *gucharp, ushort *lushortp, long *plongp) {"
        NL "    if (to_global(gintp) == NULL)"
        NL "        return false;"
        NL "    if (to_local(lfloatp) == NULL)"
        NL "        return false;"
        NL "    if (to_global(gucharp) == NULL)"
        NL "        return false;"
        NL "    if (to_local(lushortp) == NULL)"
        NL "        return false;"
        NL "    if (to_private(plongp) == NULL)"
        NL "        return false;"
        NL
        NL "    if (*gintp != 1 || *lfloatp != 2.0f || *gucharp != 3 || *lushortp != 4 || *plongp != 5)"
        NL "        return false;"
        NL
        NL "    return true;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local float lfloat;"
        NL "    lfloat = 2.0f;"
        NL "    __local ushort lushort;"
        NL "    lushort = 4;"
        NL "    long plong = 5;"
        NL
        NL "    __global int *gintp = &gint;"
        NL "    __local float *lfloatp = &lfloat;"
        NL "    __global uchar *gucharp = &guchar;"
        NL "    __local ushort *lushortp = &lushort;"
        NL "    __private long *plongp = &plong;"
        NL
        NL "    results[tid] = helperFunction(gintp, lfloatp, gucharp, lushortp, plongp);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_variable_get_fence(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local ushort lushort;"
        NL "    lushort = 2;"
        NL "    float pfloat = 3.0f;"
        NL
        NL "    // tested pointers"
        NL "    __global int *gintp = &gint;"
        NL "    __local ushort *lushortp = &lushort;"
        NL "    __private float *pfloatp = &pfloat;"
        NL
        NL "    int failures = 0;"
        NL "    if (!isFenceValid(get_fence(gintp)))"
        NL "        failures++;"
        NL "    if (!isFenceValid(get_fence(lushortp)))"
        NL "        failures++;"
        NL "    if (!isFenceValid(get_fence(pfloatp)))"
        NL "        failures++;"
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_variable_to_address_space(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION =
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local ushort lushort;"
        NL "    lushort = 2;"
        NL "    float pfloat = 3.0f;"
        NL
        NL "    // tested pointers"
        NL "    __global int * gintp = &gint;"
        NL "    __local ushort *lushortp = &lushort;"
        NL "    __private float *pfloatp = &pfloat;"
        NL
        NL "    int failures = 0;"
        NL "    if (to_global(gintp) == NULL)"
        NL "        failures++;"
        NL "    if (to_local(lushortp) == NULL)"
        NL "        failures++;"
        NL "    if (to_private(pfloatp) == NULL)"
        NL "        failures++;"
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    // pointers to global, local or private are implicitly convertible to generic
    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    int pint = 3;"
        NL
        NL "    // count mismatches with expected fence types"
        NL "    int failures = 0;"
        NL
        NL "    // tested pointer"
        NL "    // generic can be reassigned to different named address spaces"
        NL "    int * intp;"
        NL
        NL "    intp = &gint;"
        NL "    failures += !(isFenceValid(get_fence(intp)));"
        NL "    failures += !(to_global(intp));"
        NL "    failures += (*intp != 1);"
        NL
        NL "    intp = &lint;"
        NL "    failures += !(isFenceValid(get_fence(intp)));"
        NL "    failures += !(to_local(intp));"
        NL "    failures += (*intp != 2);"
        NL
        NL "    intp = &pint;"
        NL "    failures += !(isFenceValid(get_fence(intp)));"
        NL "    failures += !(to_private(intp));"
        NL "    failures += (*intp != 3);"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    // converting from a generic pointer to a named address space is legal only with explicit casting
    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    int pint = 3;"
        NL
        NL "    // count mismatches with expected fence types"
        NL "    int failures = 0;"
        NL
        NL "    // tested pointer"
        NL "    // generic can be reassigned to different named address spaces"
        NL "    int * intp;"
        NL
        NL "    intp = &gint;"
        NL "    global int * gintp = (global int *)intp;"
        NL "    failures += !(isFenceValid(get_fence(gintp)));"
        NL "    failures += !(to_global(gintp));"
        NL "    failures += (*gintp != 1);"
        NL
        NL "    intp = &lint;"
        NL "    local int * lintp = (local int *)intp;"
        NL "    failures += !(isFenceValid(get_fence(lintp)));"
        NL "    failures += !(to_local(lintp));"
        NL "    failures += (*lintp != 2);"
        NL
        NL "    intp = &pint;"
        NL "    private int * pintp = (private int *)intp;"
        NL "    failures += !(isFenceValid(get_fence(pintp)));"
        NL "    failures += !(to_private(pintp));"
        NL "    failures += (*pintp != 3);"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    CBasicTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_conditional_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr;"
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL
        NL "    if (tid % 2)"
        NL "        ptr = &gint;"
        NL "    else"
        NL "        ptr = &lint;"
        NL
        NL "    barrier(CLK_GLOBAL_MEM_FENCE);"
        NL
        NL "    if (tid % 2)"
        NL "        results[tid] = (isFenceValid(get_fence(ptr)) && to_global(ptr) && *ptr == 1);"
        NL "    else"
        NL "        results[tid] = (isFenceValid(get_fence(ptr)) && to_local(ptr) && *ptr == 2);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_chain_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "int f4(int val, int *ptr) { return (isFenceValid(get_fence(ptr)) && val == *ptr) ? 0 : 1; }"
        NL "int f3(int val, int *ptr) { return f4(val, ptr); }"
        NL "int f2(int *ptr, int val) { return f3(val, ptr); }"
        NL "int f1(int *ptr, int val) { return f2(ptr, val); }"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr;"
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    __private int pint = 3;"
        NL
        NL "    int failures = 0;"
        NL "    failures += f1(&gint, gint);"
        NL "    failures += f1(&lint, lint);"
        NL "    failures += f1(&pint, pint);"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL;
    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_ternary_operator_casting(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION = common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr;"
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL
        NL "    ptr = (tid % 2) ? &gint : (int *)&lint; // assuming there is an implicit conversion from named address space to generic"
        NL
        NL "    barrier(CLK_GLOBAL_MEM_FENCE);"
        NL
        NL "    if (tid % 2)"
        NL "        results[tid] = (isFenceValid(get_fence(ptr)) && to_global(ptr) && *ptr == gint);"
        NL "    else"
        NL "        results[tid] = (isFenceValid(get_fence(ptr)) && to_local(ptr) && *ptr == lint);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_language_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    // implicit private struct
    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    __private int pint = 3;"
        NL
        NL "    struct {"
        NL "        __global int *gintp;"
        NL "        __local  int *lintp;"
        NL "        __private int *pintp;"
        NL "    } structWithPointers;"
        NL
        NL "    structWithPointers.gintp = &gint;"
        NL "    structWithPointers.lintp = &lint;"
        NL "    structWithPointers.pintp = &pint;"
        NL
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.gintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.lintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.pintp)));"
        NL
        NL "    failures += !(to_global(structWithPointers.gintp));"
        NL "    failures += !(to_local(structWithPointers.lintp));"
        NL "    failures += !(to_private(structWithPointers.pintp));"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    // explicit __private struct
    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    __private int pint = 3;"
        NL
        NL "    typedef struct {"
        NL "        __global int * gintp;"
        NL "        __local  int * lintp;"
        NL "        __private int * pintp;"
        NL "    } S;"
        NL
        NL "    __private S structWithPointers;"
        NL "    structWithPointers.gintp = &gint;"
        NL "    structWithPointers.lintp = &lint;"
        NL "    structWithPointers.pintp = &pint;"
        NL
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.gintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.lintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.pintp)));"
        NL
        NL "    failures += !(to_global(structWithPointers.gintp));"
        NL "    failures += !(to_local(structWithPointers.lintp));"
        NL "    failures += !(to_private(structWithPointers.pintp));"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    __private int pint = 3;"
        NL
        NL "    typedef struct {"
        NL "        __global int * gintp;"
        NL "        __local  int * lintp;"
        NL "        __private int * pintp;"
        NL "    } S;"
        NL
        NL "    __local S structWithPointers;"
        NL "    structWithPointers.gintp = &gint;"
        NL "    structWithPointers.lintp = &lint;"
        NL "    structWithPointers.pintp = &pint;"
        NL
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.gintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.lintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.pintp)));"
        NL
        NL "    failures += !(to_global(structWithPointers.gintp));"
        NL "    failures += !(to_local(structWithPointers.lintp));"
        NL "    failures += !(to_private(structWithPointers.pintp));"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "typedef struct {"
        NL "    __global int *gintp;"
        NL "    __local  int *lintp;"
        NL "    __private int *pintp;"
        NL "} S;"
        NL
        NL "__global S structWithPointers;"
        NL "__global int gint = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int lint;"
        NL "    lint = 2;"
        NL "    __private int pint = 3;"
        NL
        NL "    structWithPointers.gintp = &gint;"
        NL "    structWithPointers.lintp = &lint;"
        NL "    structWithPointers.pintp = &pint;"
        NL
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.gintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.lintp)));"
        NL "    failures += !(isFenceValid(get_fence(structWithPointers.pintp)));"
        NL
        NL "    failures += !(to_global(structWithPointers.gintp));"
        NL "    failures += !(to_local(structWithPointers.lintp));"
        NL "    failures += !(to_private(structWithPointers.pintp));"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    CBasicTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_language_union(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int g = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int l;"
        NL "    l = 2;"
        NL "    int p = 3;"
        NL
        NL "    union {"
        NL "        __global int *gintp;"
        NL "        __local  int *lintp;"
        NL "        __private int *pintp;"
        NL "    } u;"
        NL
        NL "    u.gintp = &g;"
        NL "    failures += !(isFenceValid(get_fence(u.gintp)));"
        NL "    failures += !to_global(u.gintp);"
        NL "    failures += (*(u.gintp) != 1);"
        NL
        NL "    u.lintp = &l;"
        NL "    failures += !(isFenceValid(get_fence(u.lintp)));"
        NL "    failures += !to_local(u.lintp);"
        NL "    failures += (*(u.lintp) != 2);"
        NL
        NL "    u.pintp = &p;"
        NL "    failures += !(isFenceValid(get_fence(u.pintp)));"
        NL "    failures += !to_private(u.pintp);"
        NL "    failures += (*(u.pintp) != 3);"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "__global int g = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int l;"
        NL "    l = 2;"
        NL "    int p = 3;"
        NL
        NL "    typedef union {"
        NL "        __global int * gintp;"
        NL "        __local  int * lintp;"
        NL "        __private int * pintp;"
        NL "    } U;"
        NL
        NL "    __local U u;"
        NL
        NL "    u.gintp = &g;"
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL "    failures += !(isFenceValid(get_fence(u.gintp)));"
        NL "    failures += !to_global(u.gintp);"
        NL "    failures += (*(u.gintp) != 1);"
        NL
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL "    u.lintp = &l;"
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL "    failures += !(isFenceValid(get_fence(u.lintp)));"
        NL "    failures += !to_local(u.lintp);"
        NL "    failures += (*(u.lintp) != 2);"
        NL
        NL "    work_group_barrier(CLK_LOCAL_MEM_FENCE);"
        NL "    if(get_local_id(0) == 0) {"
        NL "      u.pintp = &p;"
        NL "      failures += !(isFenceValid(get_fence(u.pintp)));"
        NL "      failures += !to_private(u.pintp);"
        NL "      failures += (*(u.pintp) != 3);"
        NL "    }"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(common::CONFORMANCE_VERIFY_FENCE +
        NL
        NL "typedef union {"
        NL "    __global int * gintp;"
        NL "    __local  int * lintp;"
        NL "    __private int * pintp;"
        NL "} U;"
        NL
        NL "__global U u;"
        NL "__global int g = 1;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    // for global unions only one thread should modify union's content"
        NL "    if (tid != 0) {"
        NL "        results[tid] = 1;"
        NL "        return;"
        NL "    }"
        NL
        NL "    int failures = 0;"
        NL
        NL "    __local int l;"
        NL "    l = 2;"
        NL "    int p = 3;"
        NL
        NL "    u.gintp = &g;"
        NL "    failures += !(isFenceValid(get_fence(u.gintp)));"
        NL "    failures += !to_global(u.gintp);"
        NL "    failures += (*(u.gintp) != 1);"
        NL
        NL "    u.lintp = &l;"
        NL "    failures += !(isFenceValid(get_fence(u.lintp)));"
        NL "    failures += !to_local(u.lintp);"
        NL "    failures += (*(u.lintp) != 2);"
        NL
        NL "    u.pintp = &p;"
        NL "    failures += !(isFenceValid(get_fence(u.pintp)));"
        NL "    failures += !to_private(u.pintp);"
        NL "    failures += (*(u.pintp) != 3);"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL
    );

    CBasicTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_multiple_calls_same_function(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    const std::string KERNEL_FUNCTION =
        NL
        NL "int shift2(const int *ptr, int arg) {"
        NL "    return *ptr << arg;"
        NL "}"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL "    int failures = 0;"
        NL
        NL "    __local int val;"
        NL "    val = get_group_id(0);"
        NL
        NL "    for (int i = 0; i < 5; i++) {"
        NL "        if (shift2(&val, i) != (val << i))"
        NL "            failures++;"
        NL "    }"
        NL
        NL "    for (int i = 10; i > 5; i--) {"
        NL "        if (shift2(&val, i) != (val << i))"
        NL "            failures++;"
        NL "    }"
        NL
        NL "    results[tid] = (failures == 0);"
        NL "}"
        NL;

    CBasicTest test(KERNEL_FUNCTION);

    return test.Execute(deviceID, context, queue, num_elements);
}

int test_compare_pointers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    std::vector<std::string> KERNEL_FUNCTIONS;

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL
        NL "    results[tid] = (ptr == NULL);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL "    __global int *gptr = NULL;"
        NL
        NL "    results[tid] = (ptr == gptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL "    __local int *lptr = NULL;"
        NL
        NL "    results[tid] = (ptr == lptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL "    __private int *pptr = NULL;"
        NL
        NL "    results[tid] = (ptr == pptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL "    __local int *lptr = NULL;"
        NL "    __global int *gptr = NULL;"
        NL
        NL "    ptr = lptr;"
        NL
        NL "    results[tid] = ((int*)gptr == ptr) && ((int*)lptr == ptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int some_value = 7;"
        NL "    int *ptr = NULL;"
        NL "    __private int *pptr = &some_value;"
        NL
        NL "    results[tid] = (ptr != pptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    __local int some_value;"
        NL "    some_value = 7;"
        NL "    int *ptr = NULL;"
        NL "    __local int *lptr = &some_value;"
        NL
        NL "    results[tid] = (ptr != lptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__global int some_value = 7;"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = NULL;"
        NL "    __global int *gptr = &some_value;"
        NL
        NL "    results[tid] = (ptr != gptr);"
        NL "}"
        NL
    );

    KERNEL_FUNCTIONS.push_back(
        NL "__global int arr[5] = { 0, 1, 2, 3, 4 };"
        NL
        NL "__kernel void testKernel(__global uint *results) {"
        NL "    uint tid = get_global_id(0);"
        NL
        NL "    int *ptr = &arr[1];"
        NL "    __global int *gptr = &arr[3];"
        NL
        NL "    results[tid] = (gptr >= ptr);"
        NL "}"
        NL
    );

    CBasicTest test(KERNEL_FUNCTIONS);

    return test.Execute(deviceID, context, queue, num_elements);
}
