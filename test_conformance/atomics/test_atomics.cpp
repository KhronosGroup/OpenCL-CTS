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
#include "testBase.h"
#include "harness/conversions.h"
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#ifndef _WIN32
#include <unistd.h>
#endif

#include <cinttypes>

#define INT_TEST_VALUE 402258822
#define LONG_TEST_VALUE 515154531254381446LL

// clang-format off
const char *atomic_global_pattern[] = {
    "__kernel void test_atomic_fn(volatile __global %s *destMemory, __global %s *oldValues)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    ,
    "\n"
    "}\n" };

const char *atomic_local_pattern[] = {
    "__kernel void test_atomic_fn(__global %s *finalDest, __global %s *oldValues, volatile __local %s *destMemory, int numDestItems )\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    int  dstItemIdx;\n"
    "\n"
    "    // Everybody does the following line(s), but it all has the same result. We still need to ensure we sync before the atomic op, though\n"
    "    for( dstItemIdx = 0; dstItemIdx < numDestItems; dstItemIdx++ )\n"
    "        destMemory[ dstItemIdx ] = finalDest[ dstItemIdx ];\n"
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "\n"
    ,
    "    barrier( CLK_LOCAL_MEM_FENCE );\n"
    "    // Finally, write out the last value. Again, we're synced, so everyone will be writing the same value\n"
    "    for( dstItemIdx = 0; dstItemIdx < numDestItems; dstItemIdx++ )\n"
    "        finalDest[ dstItemIdx ] = destMemory[ dstItemIdx ];\n"
    "}\n" };
// clang-format on

#include "common.h"
#include "host_atomics.h"

#include <sstream>
#include <vector>

#define TEST_COUNT 128 * 1024


struct TestFns
{
    cl_int mIntStartValue;
    cl_long mLongStartValue;

    size_t (*NumResultsFn)(size_t threadSize, ExplicitType dataType);

    // Integer versions
    cl_int (*ExpectedValueIntFn)(size_t size, cl_int *startRefValues,
                                 size_t whichDestValue);
    void (*GenerateRefsIntFn)(size_t size, cl_int *startRefValues, MTdata d);
    bool (*VerifyRefsIntFn)(size_t size, cl_int *refValues, cl_int finalValue);

    // Long versions
    cl_long (*ExpectedValueLongFn)(size_t size, cl_long *startRefValues,
                                   size_t whichDestValue);
    void (*GenerateRefsLongFn)(size_t size, cl_long *startRefValues, MTdata d);
    bool (*VerifyRefsLongFn)(size_t size, cl_long *refValues,
                             cl_long finalValue);

    // Float versions
    cl_float (*ExpectedValueFloatFn)(size_t size, cl_float *startRefValues,
                                     size_t whichDestValue);
    void (*GenerateRefsFloatFn)(size_t size, cl_float *startRefValues,
                                MTdata d);
    bool (*VerifyRefsFloatFn)(size_t size, cl_float *refValues,
                              cl_float finalValue);
};

bool check_atomic_support(cl_device_id device, bool extended, bool isLocal,
                          ExplicitType dataType)
{
    // clang-format off
    const char *extensionNames[8] = {
        "cl_khr_global_int32_base_atomics", "cl_khr_global_int32_extended_atomics",
        "cl_khr_local_int32_base_atomics",  "cl_khr_local_int32_extended_atomics",
        "cl_khr_int64_base_atomics",        "cl_khr_int64_extended_atomics",
        "cl_khr_int64_base_atomics",        "cl_khr_int64_extended_atomics"       // this line intended to be the same as the last one
    };
    // clang-format on

    size_t index = 0;
    if (extended) index += 1;
    if (isLocal) index += 2;

    Version version = get_device_cl_version(device);

    switch (dataType)
    {
        case kInt:
        case kUInt:
            if (version >= Version(1, 1)) return 1;
            break;
        case kLong:
        case kULong: index += 4; break;
        case kFloat: // this has to stay separate since the float atomics arent
                     // in the 1.0 extensions
            return version >= Version(1, 1);
        default:
            log_error(
                "ERROR:  Unsupported data type (%d) in check_atomic_support\n",
                dataType);
            return 0;
    }

    return is_extension_available(device, extensionNames[index]);
}

int test_atomic_function(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements,
                         const char *programCore, TestFns testFns,
                         bool extended, bool isLocal, ExplicitType dataType,
                         bool matchGroupSize)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    int error;
    size_t threads[1];
    clMemWrapper streams[2];
    void *refValues, *startRefValues;
    size_t threadSize, groupSize;
    const char *programLines[4];
    char pragma[512];
    char programHeader[512];
    MTdata d;
    size_t typeSize = get_explicit_type_size(dataType);


    // Verify we can run first
    bool isUnsigned = (dataType == kULong) || (dataType == kUInt);
    if (!check_atomic_support(deviceID, extended, isLocal, dataType))
    {
        // Only print for the signed (unsigned comes right after, and if signed
        // isn't supported, unsigned isn't either)
        if (dataType == kFloat)
            log_info("\t%s float not supported\n",
                     isLocal ? "Local" : "Global");
        else if (!isUnsigned)
            log_info("\t%s %sint%d not supported\n",
                     isLocal ? "Local" : "Global", isUnsigned ? "u" : "",
                     (int)typeSize * 8);
        // Since we don't support the operation, they implicitly pass
        return 0;
    }
    else
    {
        if (dataType == kFloat)
            log_info("\t%s float%s...", isLocal ? "local" : "global",
                     isLocal ? " " : "");
        else
            log_info("\t%s %sint%d%s%s...", isLocal ? "local" : "global",
                     isUnsigned ? "u" : "", (int)typeSize * 8,
                     isUnsigned ? "" : " ", isLocal ? " " : "");
    }

    //// Set up the kernel code

    // Create the pragma line for this kernel
    bool isLong = (dataType == kLong || dataType == kULong);
    sprintf(pragma,
            "#pragma OPENCL EXTENSION cl_khr%s_int%s_%s_atomics : enable\n",
            isLong ? "" : (isLocal ? "_local" : "_global"),
            isLong ? "64" : "32", extended ? "extended" : "base");

    // Now create the program header
    const char *typeName = get_explicit_type_name(dataType);
    if (isLocal)
        sprintf(programHeader, atomic_local_pattern[0], typeName, typeName,
                typeName);
    else
        sprintf(programHeader, atomic_global_pattern[0], typeName, typeName);

    // Set up our entire program now
    programLines[0] = pragma;
    programLines[1] = programHeader;
    programLines[2] = programCore;
    programLines[3] =
        (isLocal) ? atomic_local_pattern[1] : atomic_global_pattern[1];

    if (create_single_kernel_helper(context, &program, &kernel, 4, programLines,
                                    "test_atomic_fn"))
    {
        return -1;
    }

    //// Set up to actually run
    threadSize = num_elements;

    error =
        get_max_common_work_group_size(context, kernel, threadSize, &groupSize);
    test_error(error, "Unable to get thread group max size");

    if (matchGroupSize)
        // HACK because xchg and cmpxchg apparently are limited by hardware
        threadSize = groupSize;

    if (isLocal)
    {
        size_t maxSizes[3] = { 0, 0, 0 };
        error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                3 * sizeof(size_t), maxSizes, 0);
        test_error(error,
                   "Unable to obtain max work item sizes for the device");

        size_t workSize;
        error = clGetKernelWorkGroupInfo(kernel, deviceID,
                                         CL_KERNEL_WORK_GROUP_SIZE,
                                         sizeof(workSize), &workSize, NULL);
        test_error(
            error,
            "Unable to obtain max work group size for device and kernel combo");

        // Limit workSize to avoid extremely large local buffer size and slow
        // run.
        if (workSize > 65536) workSize = 65536;

        // "workSize" is limited to that of the first dimension as only a
        // 1DRange is executed.
        if (maxSizes[0] < workSize)
        {
            workSize = maxSizes[0];
        }

        threadSize = groupSize = workSize;
    }


    log_info("\t(thread count %d, group size %d)\n", (int)threadSize,
             (int)groupSize);

    refValues = (cl_int *)malloc(typeSize * threadSize);

    if (testFns.GenerateRefsIntFn != NULL)
    {
        // We have a ref generator provided
        d = init_genrand(gRandomSeed);
        startRefValues = malloc(typeSize * threadSize);
        if (typeSize == 4)
            testFns.GenerateRefsIntFn(threadSize, (cl_int *)startRefValues, d);
        else
            testFns.GenerateRefsLongFn(threadSize, (cl_long *)startRefValues,
                                       d);
        free_mtdata(d);
        d = NULL;
    }
    else
        startRefValues = NULL;

    // If we're given a num_results function, we need to determine how many
    // result objects we need. If we don't have it, we assume it's just 1
    size_t numDestItems = (testFns.NumResultsFn != NULL)
        ? testFns.NumResultsFn(threadSize, dataType)
        : 1;

    char *destItems = new char[typeSize * numDestItems];
    if (destItems == NULL)
    {
        log_error("ERROR: Unable to allocate memory!\n");
        return -1;
    }
    void *startValue = (typeSize == 4) ? (void *)&testFns.mIntStartValue
                                       : (void *)&testFns.mLongStartValue;
    for (size_t i = 0; i < numDestItems; i++)
        memcpy(destItems + i * typeSize, startValue, typeSize);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                                typeSize * numDestItems, destItems, NULL);
    if (!streams[0])
    {
        log_error("ERROR: Creating output array failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(
        context,
        ((startRefValues != NULL ? CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE)),
        typeSize * threadSize, startRefValues, NULL);
    if (!streams[1])
    {
        log_error("ERROR: Creating reference array failed!\n");
        return -1;
    }

    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    if (isLocal)
    {
        error = clSetKernelArg(kernel, 2, typeSize * numDestItems, NULL);
        test_error(error, "Unable to set indexed local kernel argument");

        cl_int numDestItemsInt = (cl_int)numDestItems;
        error = clSetKernelArg(kernel, 3, sizeof(cl_int), &numDestItemsInt);
        test_error(error, "Unable to set indexed kernel argument");
    }

    /* Run the kernel */
    threads[0] = threadSize;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, &groupSize,
                                   0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    error =
        clEnqueueReadBuffer(queue, streams[0], true, 0, typeSize * numDestItems,
                            destItems, 0, NULL, NULL);
    test_error(error, "Unable to read result value!");

    error =
        clEnqueueReadBuffer(queue, streams[1], true, 0, typeSize * threadSize,
                            refValues, 0, NULL, NULL);
    test_error(error, "Unable to read reference values!");

    // If we have an expectedFn, then we need to generate a final value to
    // compare against. If we don't have one, it's because we're comparing ref
    // values only
    if (testFns.ExpectedValueIntFn != NULL)
    {
        for (size_t i = 0; i < numDestItems; i++)
        {
            char expected[8];
            cl_int intVal;
            cl_long longVal;
            if (typeSize == 4)
            {
                // Int version
                intVal = testFns.ExpectedValueIntFn(
                    threadSize, (cl_int *)startRefValues, i);
                memcpy(expected, &intVal, sizeof(intVal));
            }
            else
            {
                // Long version
                longVal = testFns.ExpectedValueLongFn(
                    threadSize, (cl_long *)startRefValues, i);
                memcpy(expected, &longVal, sizeof(longVal));
            }

            if (memcmp(expected, destItems + i * typeSize, typeSize) != 0)
            {
                if (typeSize == 4)
                {
                    cl_int *outValue = (cl_int *)(destItems + i * typeSize);
                    log_error("ERROR: Result %zu from kernel does not "
                              "validate! (should be %d, was %d)\n",
                              i, intVal, *outValue);
                    cl_int *startRefs = (cl_int *)startRefValues;
                    cl_int *refs = (cl_int *)refValues;
                    for (i = 0; i < threadSize; i++)
                    {
                        if (startRefs != NULL)
                            log_info(" --- %zu - %d --- %d\n", i, startRefs[i],
                                     refs[i]);
                        else
                            log_info(" --- %zu --- %d\n", i, refs[i]);
                    }
                }
                else
                {
                    cl_long *outValue = (cl_long *)(destItems + i * typeSize);
                    log_error("ERROR: Result %zu from kernel does not "
                              "validate! (should be %" PRId64 ", was %" PRId64
                              ")\n",
                              i, longVal, *outValue);
                    cl_long *startRefs = (cl_long *)startRefValues;
                    cl_long *refs = (cl_long *)refValues;
                    for (i = 0; i < threadSize; i++)
                    {
                        if (startRefs != NULL)
                            log_info(" --- %zu - %" PRId64 " --- %" PRId64 "\n",
                                     i, startRefs[i], refs[i]);
                        else
                            log_info(" --- %zu --- %" PRId64 "\n", i, refs[i]);
                    }
                }
                return -1;
            }
        }
    }

    if (testFns.VerifyRefsIntFn != NULL)
    {
        /* Use the verify function to also check the results */
        if (dataType == kFloat)
        {
            cl_float *outValue = (cl_float *)destItems;
            if (!testFns.VerifyRefsFloatFn(threadSize, (cl_float *)refValues,
                                           *outValue)
                != 0)
            {
                log_error("ERROR: Reference values did not validate!\n");
                return -1;
            }
        }
        else if (typeSize == 4)
        {
            cl_int *outValue = (cl_int *)destItems;
            if (!testFns.VerifyRefsIntFn(threadSize, (cl_int *)refValues,
                                         *outValue)
                != 0)
            {
                log_error("ERROR: Reference values did not validate!\n");
                return -1;
            }
        }
        else
        {
            cl_long *outValue = (cl_long *)destItems;
            if (!testFns.VerifyRefsLongFn(threadSize, (cl_long *)refValues,
                                          *outValue)
                != 0)
            {
                log_error("ERROR: Reference values did not validate!\n");
                return -1;
            }
        }
    }
    else if (testFns.ExpectedValueIntFn == NULL)
    {
        log_error("ERROR: Test doesn't check total or refs; no values are "
                  "verified!\n");
        return -1;
    }


    /* Re-write the starting value */
    for (size_t i = 0; i < numDestItems; i++)
        memcpy(destItems + i * typeSize, startValue, typeSize);
    error =
        clEnqueueWriteBuffer(queue, streams[0], true, 0,
                             typeSize * numDestItems, destItems, 0, NULL, NULL);
    test_error(error, "Unable to write starting values!");

    /* Run the kernel once for a single thread, so we can verify that the
     * returned value is the original one */
    threads[0] = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, threads, 0,
                                   NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    error = clEnqueueReadBuffer(queue, streams[1], true, 0, typeSize, refValues,
                                0, NULL, NULL);
    test_error(error, "Unable to read reference values!");

    if (memcmp(refValues, destItems, typeSize) != 0)
    {
        if (typeSize == 4)
        {
            cl_int *s = (cl_int *)destItems;
            cl_int *r = (cl_int *)refValues;
            log_error("ERROR: atomic function operated correctly but did NOT "
                      "return correct 'old' value "
                      " (should have been %d, returned %d)!\n",
                      *s, *r);
        }
        else
        {
            cl_long *s = (cl_long *)destItems;
            cl_long *r = (cl_long *)refValues;
            log_error("ERROR: atomic function operated correctly but did NOT "
                      "return correct 'old' value "
                      " (should have been %" PRId64 ", returned %" PRId64
                      ")!\n",
                      *s, *r);
        }
        return -1;
    }

    delete[] destItems;
    free(refValues);
    if (startRefValues != NULL) free(startRefValues);

    return 0;
}

int test_atomic_function_set(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements,
                             const char *programCore, TestFns testFns,
                             bool extended, bool matchGroupSize,
                             bool usingAtomicPrefix)
{
    log_info("    Testing %s functions...\n",
             usingAtomicPrefix ? "atomic_" : "atom_");

    int errors = 0;
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   programCore, testFns, extended, false, kInt,
                                   matchGroupSize);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   programCore, testFns, extended, false, kUInt,
                                   matchGroupSize);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   programCore, testFns, extended, true, kInt,
                                   matchGroupSize);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   programCore, testFns, extended, true, kUInt,
                                   matchGroupSize);

    // Only the 32 bit atomic functions use the "atomic" prefix in 1.1, the 64
    // bit functions still use the "atom" prefix. The argument usingAtomicPrefix
    // is set to true if programCore was generated with the "atomic" prefix.
    if (!usingAtomicPrefix)
    {
        errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                       programCore, testFns, extended, false,
                                       kLong, matchGroupSize);
        errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                       programCore, testFns, extended, false,
                                       kULong, matchGroupSize);
        errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                       programCore, testFns, extended, true,
                                       kLong, matchGroupSize);
        errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                       programCore, testFns, extended, true,
                                       kULong, matchGroupSize);
    }

    return errors;
}

#pragma mark ---- add

const char atom_add_core[] =
    "    oldValues[tid] = atom_add( &destMemory[0], tid + 3 );\n"
    "    atom_add( &destMemory[0], tid + 3 );\n"
    "    atom_add( &destMemory[0], tid + 3 );\n"
    "    atom_add( &destMemory[0], tid + 3 );\n";

const char atomic_add_core[] =
    "    oldValues[tid] = atomic_add( &destMemory[0], tid + 3 );\n"
    "    atomic_add( &destMemory[0], tid + 3 );\n"
    "    atomic_add( &destMemory[0], tid + 3 );\n"
    "    atomic_add( &destMemory[0], tid + 3 );\n";

cl_int test_atomic_add_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    cl_int total = 0;
    for (size_t i = 0; i < size; i++) total += ((cl_int)i + 3) * 4;
    return total;
}

cl_long test_atomic_add_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    cl_long total = 0;
    for (size_t i = 0; i < size; i++) total += ((i + 3) * 4);
    return total;
}

int test_atomic_add(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { 0,
                    0LL,
                    NULL,
                    test_atomic_add_result_int,
                    NULL,
                    NULL,
                    test_atomic_add_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_add_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_add_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}

#pragma mark ---- sub

const char atom_sub_core[] =
    "    oldValues[tid] = atom_sub( &destMemory[0], tid + 3 );\n";

const char atomic_sub_core[] =
    "    oldValues[tid] = atomic_sub( &destMemory[0], tid + 3 );\n";

cl_int test_atomic_sub_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    cl_int total = INT_TEST_VALUE;
    for (size_t i = 0; i < size; i++) total -= (cl_int)i + 3;
    return total;
}

cl_long test_atomic_sub_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    cl_long total = LONG_TEST_VALUE;
    for (size_t i = 0; i < size; i++) total -= i + 3;
    return total;
}

int test_atomic_sub(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { INT_TEST_VALUE,
                    LONG_TEST_VALUE,
                    NULL,
                    test_atomic_sub_result_int,
                    NULL,
                    NULL,
                    test_atomic_sub_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_sub_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_sub_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}

#pragma mark ---- xchg

const char atom_xchg_core[] =
    "    oldValues[tid] = atom_xchg( &destMemory[0], tid );\n";

const char atomic_xchg_core[] =
    "    oldValues[tid] = atomic_xchg( &destMemory[0], tid );\n";
const char atomic_xchg_float_core[] =
    "    oldValues[tid] = atomic_xchg( &destMemory[0], tid );\n";

bool test_atomic_xchg_verify_int(size_t size, cl_int *refValues,
                                 cl_int finalValue)
{
    /* For xchg, each value from 0 to size - 1 should have an entry in the ref
     * array, and ONLY one entry */
    char *valids;
    size_t i;
    char originalValidCount = 0;

    valids = (char *)malloc(sizeof(char) * size);
    memset(valids, 0, sizeof(char) * size);

    for (i = 0; i < size; i++)
    {
        if (refValues[i] == INT_TEST_VALUE)
        {
            // Special initial value
            originalValidCount++;
            continue;
        }
        if (refValues[i] < 0 || (size_t)refValues[i] >= size)
        {
            log_error(
                "ERROR: Reference value %zu outside of valid range! (%d)\n", i,
                refValues[i]);
            return false;
        }
        valids[refValues[i]]++;
    }

    /* Note: ONE entry will have zero count. It'll be the last one that
     executed, because that value should be the final value outputted */
    if (valids[finalValue] > 0)
    {
        log_error("ERROR: Final value %d was also in ref list!\n", finalValue);
        return false;
    }
    else
        valids[finalValue] = 1; // So the following loop will be okay

    /* Now check that every entry has one and only one count */
    if (originalValidCount != 1)
    {
        log_error("ERROR: Starting reference value %d did not occur "
                  "once-and-only-once (occurred %d)\n",
                  65191, originalValidCount);
        return false;
    }
    for (i = 0; i < size; i++)
    {
        if (valids[i] != 1)
        {
            log_error("ERROR: Reference value %zu did not occur "
                      "once-and-only-once (occurred %d)\n",
                      i, valids[i]);
            for (size_t j = 0; j < size; j++)
                log_info("%d: %d\n", (int)j, (int)valids[j]);
            return false;
        }
    }

    free(valids);
    return true;
}

bool test_atomic_xchg_verify_long(size_t size, cl_long *refValues,
                                  cl_long finalValue)
{
    /* For xchg, each value from 0 to size - 1 should have an entry in the ref
     * array, and ONLY one entry */
    char *valids;
    size_t i;
    char originalValidCount = 0;

    valids = (char *)malloc(sizeof(char) * size);
    memset(valids, 0, sizeof(char) * size);

    for (i = 0; i < size; i++)
    {
        if (refValues[i] == LONG_TEST_VALUE)
        {
            // Special initial value
            originalValidCount++;
            continue;
        }
        if (refValues[i] < 0 || (size_t)refValues[i] >= size)
        {
            log_error(
                "ERROR: Reference value %zu outside of valid range! (%" PRId64
                ")\n",
                i, refValues[i]);
            return false;
        }
        valids[refValues[i]]++;
    }

    /* Note: ONE entry will have zero count. It'll be the last one that
     executed, because that value should be the final value outputted */
    if (valids[finalValue] > 0)
    {
        log_error("ERROR: Final value %" PRId64 " was also in ref list!\n",
                  finalValue);
        return false;
    }
    else
        valids[finalValue] = 1; // So the following loop will be okay

    /* Now check that every entry has one and only one count */
    if (originalValidCount != 1)
    {
        log_error("ERROR: Starting reference value %d did not occur "
                  "once-and-only-once (occurred %d)\n",
                  65191, originalValidCount);
        return false;
    }
    for (i = 0; i < size; i++)
    {
        if (valids[i] != 1)
        {
            log_error("ERROR: Reference value %zu did not occur "
                      "once-and-only-once (occurred %d)\n",
                      i, valids[i]);
            for (size_t j = 0; j < size; j++)
                log_info("%d: %d\n", (int)j, (int)valids[j]);
            return false;
        }
    }

    free(valids);
    return true;
}

bool test_atomic_xchg_verify_float(size_t size, cl_float *refValues,
                                   cl_float finalValue)
{
    /* For xchg, each value from 0 to size - 1 should have an entry in the ref
     * array, and ONLY one entry */
    char *valids;
    size_t i;
    char originalValidCount = 0;

    valids = (char *)malloc(sizeof(char) * size);
    memset(valids, 0, sizeof(char) * size);

    for (i = 0; i < size; i++)
    {
        cl_int *intRefValue = (cl_int *)(&refValues[i]);
        if (*intRefValue == INT_TEST_VALUE)
        {
            // Special initial value
            originalValidCount++;
            continue;
        }
        if (refValues[i] < 0 || (size_t)refValues[i] >= size)
        {
            log_error(
                "ERROR: Reference value %zu outside of valid range! (%a)\n", i,
                refValues[i]);
            return false;
        }
        valids[(int)refValues[i]]++;
    }

    /* Note: ONE entry will have zero count. It'll be the last one that
     executed, because that value should be the final value outputted */
    if (valids[(int)finalValue] > 0)
    {
        log_error("ERROR: Final value %a was also in ref list!\n", finalValue);
        return false;
    }
    else
        valids[(int)finalValue] = 1; // So the following loop will be okay

    /* Now check that every entry has one and only one count */
    if (originalValidCount != 1)
    {
        log_error("ERROR: Starting reference value %d did not occur "
                  "once-and-only-once (occurred %d)\n",
                  65191, originalValidCount);
        return false;
    }
    for (i = 0; i < size; i++)
    {
        if (valids[i] != 1)
        {
            log_error("ERROR: Reference value %zu did not occur "
                      "once-and-only-once (occurred %d)\n",
                      i, valids[i]);
            for (size_t j = 0; j < size; j++)
                log_info("%d: %d\n", (int)j, (int)valids[j]);
            return false;
        }
    }

    free(valids);
    return true;
}

int test_atomic_xchg(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    TestFns set = { INT_TEST_VALUE,
                    LONG_TEST_VALUE,
                    NULL,
                    NULL,
                    NULL,
                    test_atomic_xchg_verify_int,
                    NULL,
                    NULL,
                    test_atomic_xchg_verify_long,
                    NULL,
                    NULL,
                    test_atomic_xchg_verify_float };

    int errors = test_atomic_function_set(
        deviceID, context, queue, num_elements, atom_xchg_core, set, false,
        true, /*usingAtomicPrefix*/ false);
    errors |= test_atomic_function_set(deviceID, context, queue, num_elements,
                                       atomic_xchg_core, set, false, true,
                                       /*usingAtomicPrefix*/ true);

    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atomic_xchg_float_core, set, false, false,
                                   kFloat, true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atomic_xchg_float_core, set, false, true,
                                   kFloat, true);

    return errors;
}


#pragma mark ---- min

const char atom_min_core[] =
    "    oldValues[tid] = atom_min( &destMemory[0], oldValues[tid] );\n";

const char atomic_min_core[] =
    "    oldValues[tid] = atomic_min( &destMemory[0], oldValues[tid] );\n";

cl_int test_atomic_min_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    cl_int total = 0x7fffffffL;
    for (size_t i = 0; i < size; i++)
    {
        if (startRefValues[i] < total) total = startRefValues[i];
    }
    return total;
}

void test_atomic_min_gen_int(size_t size, cl_int *startRefValues, MTdata d)
{
    for (size_t i = 0; i < size; i++)
        startRefValues[i] =
            (cl_int)(genrand_int32(d) % 0x3fffffff) + 0x3fffffff;
}

cl_long test_atomic_min_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    cl_long total = 0x7fffffffffffffffLL;
    for (size_t i = 0; i < size; i++)
    {
        if (startRefValues[i] < total) total = startRefValues[i];
    }
    return total;
}

void test_atomic_min_gen_long(size_t size, cl_long *startRefValues, MTdata d)
{
    for (size_t i = 0; i < size; i++)
        startRefValues[i] =
            (cl_long)(genrand_int32(d)
                      | (((cl_long)genrand_int32(d) & 0x7fffffffL) << 16));
}

int test_atomic_min(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { 0x7fffffffL,
                    0x7fffffffffffffffLL,
                    NULL,
                    test_atomic_min_result_int,
                    test_atomic_min_gen_int,
                    NULL,
                    test_atomic_min_result_long,
                    test_atomic_min_gen_long,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_min_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_min_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- max

const char atom_max_core[] =
    "    oldValues[tid] = atom_max( &destMemory[0], oldValues[tid] );\n";

const char atomic_max_core[] =
    "    oldValues[tid] = atomic_max( &destMemory[0], oldValues[tid] );\n";

cl_int test_atomic_max_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    cl_int total = 0;
    for (size_t i = 0; i < size; i++)
    {
        if (startRefValues[i] > total) total = startRefValues[i];
    }
    return total;
}

void test_atomic_max_gen_int(size_t size, cl_int *startRefValues, MTdata d)
{
    for (size_t i = 0; i < size; i++)
        startRefValues[i] =
            (cl_int)(genrand_int32(d) % 0x3fffffff) + 0x3fffffff;
}

cl_long test_atomic_max_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    cl_long total = 0;
    for (size_t i = 0; i < size; i++)
    {
        if (startRefValues[i] > total) total = startRefValues[i];
    }
    return total;
}

void test_atomic_max_gen_long(size_t size, cl_long *startRefValues, MTdata d)
{
    for (size_t i = 0; i < size; i++)
        startRefValues[i] =
            (cl_long)(genrand_int32(d)
                      | (((cl_long)genrand_int32(d) & 0x7fffffffL) << 16));
}

int test_atomic_max(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { 0,
                    0,
                    NULL,
                    test_atomic_max_result_int,
                    test_atomic_max_gen_int,
                    NULL,
                    test_atomic_max_result_long,
                    test_atomic_max_gen_long,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_max_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_max_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- inc

const char atom_inc_core[] =
    "    oldValues[tid] = atom_inc( &destMemory[0] );\n";

const char atomic_inc_core[] =
    "    oldValues[tid] = atomic_inc( &destMemory[0] );\n";

cl_int test_atomic_inc_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    return INT_TEST_VALUE + (cl_int)size;
}

cl_long test_atomic_inc_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    return LONG_TEST_VALUE + size;
}

int test_atomic_inc(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { INT_TEST_VALUE,
                    LONG_TEST_VALUE,
                    NULL,
                    test_atomic_inc_result_int,
                    NULL,
                    NULL,
                    test_atomic_inc_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_inc_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_inc_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- dec

const char atom_dec_core[] =
    "    oldValues[tid] = atom_dec( &destMemory[0] );\n";

const char atomic_dec_core[] =
    "    oldValues[tid] = atomic_dec( &destMemory[0] );\n";

cl_int test_atomic_dec_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichDestValue)
{
    return INT_TEST_VALUE - (cl_int)size;
}

cl_long test_atomic_dec_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichDestValue)
{
    return LONG_TEST_VALUE - size;
}

int test_atomic_dec(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { INT_TEST_VALUE,
                    LONG_TEST_VALUE,
                    NULL,
                    test_atomic_dec_result_int,
                    NULL,
                    NULL,
                    test_atomic_dec_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_dec_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_dec_core, set, false,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- cmpxchg

/* We test cmpxchg by implementing (the long way) atom_add */
// clang-format off
const char atom_cmpxchg_core[] =
    "    int oldValue, origValue, newValue;\n"
    "    do { \n"
    "        origValue = destMemory[0];\n"
    "        newValue = origValue + tid + 2;\n"
    "        oldValue = atom_cmpxchg( &destMemory[0], origValue, newValue );\n"
    "    } while( oldValue != origValue );\n"
    "    oldValues[tid] = oldValue;\n";

const char atom_cmpxchg64_core[] =
    "    long oldValue, origValue, newValue;\n"
    "    do { \n"
    "        origValue = destMemory[0];\n"
    "        newValue = origValue + tid + 2;\n"
    "        oldValue = atom_cmpxchg( &destMemory[0], origValue, newValue );\n"
    "    } while( oldValue != origValue );\n"
    "    oldValues[tid] = oldValue;\n";

const char atomic_cmpxchg_core[] =
    "    int oldValue, origValue, newValue;\n"
    "    do { \n"
    "        origValue = destMemory[0];\n"
    "        newValue = origValue + tid + 2;\n"
    "        oldValue = atomic_cmpxchg( &destMemory[0], origValue, newValue );\n"
    "    } while( oldValue != origValue );\n"
    "    oldValues[tid] = oldValue;\n";
// clang-format on

cl_int test_atomic_cmpxchg_result_int(size_t size, cl_int *startRefValues,
                                      size_t whichDestValue)
{
    cl_int total = INT_TEST_VALUE;
    for (size_t i = 0; i < size; i++) total += (cl_int)i + 2;
    return total;
}

cl_long test_atomic_cmpxchg_result_long(size_t size, cl_long *startRefValues,
                                        size_t whichDestValue)
{
    cl_long total = LONG_TEST_VALUE;
    for (size_t i = 0; i < size; i++) total += i + 2;
    return total;
}

int test_atomic_cmpxchg(cl_device_id deviceID, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    TestFns set = { INT_TEST_VALUE,
                    LONG_TEST_VALUE,
                    NULL,
                    test_atomic_cmpxchg_result_int,
                    NULL,
                    NULL,
                    test_atomic_cmpxchg_result_long,
                    NULL,
                    NULL };

    int errors = 0;

    log_info("    Testing atom_ functions...\n");
    errors |=
        test_atomic_function(deviceID, context, queue, num_elements,
                             atom_cmpxchg_core, set, false, false, kInt, true);
    errors |=
        test_atomic_function(deviceID, context, queue, num_elements,
                             atom_cmpxchg_core, set, false, false, kUInt, true);
    errors |=
        test_atomic_function(deviceID, context, queue, num_elements,
                             atom_cmpxchg_core, set, false, true, kInt, true);
    errors |=
        test_atomic_function(deviceID, context, queue, num_elements,
                             atom_cmpxchg_core, set, false, true, kUInt, true);

    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atom_cmpxchg64_core, set, false, false,
                                   kLong, true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atom_cmpxchg64_core, set, false, false,
                                   kULong, true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atom_cmpxchg64_core, set, false, true, kLong,
                                   true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atom_cmpxchg64_core, set, false, true,
                                   kULong, true);

    log_info("    Testing atomic_ functions...\n");
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atomic_cmpxchg_core, set, false, false, kInt,
                                   true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atomic_cmpxchg_core, set, false, false,
                                   kUInt, true);
    errors |=
        test_atomic_function(deviceID, context, queue, num_elements,
                             atomic_cmpxchg_core, set, false, true, kInt, true);
    errors |= test_atomic_function(deviceID, context, queue, num_elements,
                                   atomic_cmpxchg_core, set, false, true, kUInt,
                                   true);

    if (errors) return -1;

    return 0;
}

#pragma mark -------- Bitwise functions

size_t test_bitwise_num_results(size_t threadCount, ExplicitType dataType)
{
    size_t numBits = get_explicit_type_size(dataType) * 8;

    return (threadCount + numBits - 1) / numBits;
}

#pragma mark ---- and

// clang-format off
const char atom_and_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  whichResult = tid / numBits;\n"
    "    int  bitIndex = tid - ( whichResult * numBits );\n"
    "\n"
    "    oldValues[tid] = atom_and( &destMemory[whichResult], ~( 1L << bitIndex ) );\n";

const char atomic_and_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  whichResult = tid / numBits;\n"
    "    int  bitIndex = tid - ( whichResult * numBits );\n"
    "\n"
    "    oldValues[tid] = atomic_and( &destMemory[whichResult], ~( 1L << bitIndex ) );\n";
// clang-format on


cl_int test_atomic_and_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichResult)
{
    size_t numThreads = ((size_t)size + 31) / 32;
    if (whichResult < numThreads - 1) return 0;

    // Last item doesn't get and'ed on every bit, so we have to mask away
    size_t numBits = (size_t)size - whichResult * 32;
    cl_int bits = (cl_int)0xffffffffL;
    for (size_t i = 0; i < numBits; i++) bits &= ~(1 << i);

    return bits;
}

cl_long test_atomic_and_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichResult)
{
    size_t numThreads = ((size_t)size + 63) / 64;
    if (whichResult < numThreads - 1) return 0;

    // Last item doesn't get and'ed on every bit, so we have to mask away
    size_t numBits = (size_t)size - whichResult * 64;
    cl_long bits = (cl_long)0xffffffffffffffffLL;
    for (size_t i = 0; i < numBits; i++) bits &= ~(1LL << i);

    return bits;
}

int test_atomic_and(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { 0xffffffff,
                    0xffffffffffffffffLL,
                    test_bitwise_num_results,
                    test_atomic_and_result_int,
                    NULL,
                    NULL,
                    test_atomic_and_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_and_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_and_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- or

// clang-format off
const char atom_or_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  whichResult = tid / numBits;\n"
    "    int  bitIndex = tid - ( whichResult * numBits );\n"
    "\n"
    "    oldValues[tid] = atom_or( &destMemory[whichResult], ( 1L << bitIndex ) );\n";

const char atomic_or_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  whichResult = tid / numBits;\n"
    "    int  bitIndex = tid - ( whichResult * numBits );\n"
    "\n"
    "    oldValues[tid] = atomic_or( &destMemory[whichResult], ( 1L << bitIndex ) );\n";
// clang-format on

cl_int test_atomic_or_result_int(size_t size, cl_int *startRefValues,
                                 size_t whichResult)
{
    size_t numThreads = ((size_t)size + 31) / 32;
    if (whichResult < numThreads - 1) return 0xffffffff;

    // Last item doesn't get and'ed on every bit, so we have to mask away
    size_t numBits = (size_t)size - whichResult * 32;
    cl_int bits = 0;
    for (size_t i = 0; i < numBits; i++) bits |= (1 << i);

    return bits;
}

cl_long test_atomic_or_result_long(size_t size, cl_long *startRefValues,
                                   size_t whichResult)
{
    size_t numThreads = ((size_t)size + 63) / 64;
    if (whichResult < numThreads - 1) return 0x0ffffffffffffffffLL;

    // Last item doesn't get and'ed on every bit, so we have to mask away
    size_t numBits = (size_t)size - whichResult * 64;
    cl_long bits = 0;
    for (size_t i = 0; i < numBits; i++) bits |= (1LL << i);

    return bits;
}

int test_atomic_or(cl_device_id deviceID, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    TestFns set = {
        0,    0LL,  test_bitwise_num_results,   test_atomic_or_result_int,
        NULL, NULL, test_atomic_or_result_long, NULL,
        NULL
    };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_or_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_or_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}


#pragma mark ---- xor

const char atom_xor_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  bitIndex = tid & ( numBits - 1 );\n"
    "\n"
    "    oldValues[tid] = atom_xor( &destMemory[0], 1L << bitIndex );\n";

const char atomic_xor_core[] =
    "    size_t numBits = sizeof( destMemory[0] ) * 8;\n"
    "    int  bitIndex = tid & ( numBits - 1 );\n"
    "\n"
    "    oldValues[tid] = atomic_xor( &destMemory[0], 1L << bitIndex );\n";

cl_int test_atomic_xor_result_int(size_t size, cl_int *startRefValues,
                                  size_t whichResult)
{
    cl_int total = 0x2f08ab41;
    for (size_t i = 0; i < size; i++) total ^= (1 << (i & 31));
    return total;
}

cl_long test_atomic_xor_result_long(size_t size, cl_long *startRefValues,
                                    size_t whichResult)
{
    cl_long total = 0x2f08ab418ba0541LL;
    for (size_t i = 0; i < size; i++) total ^= (1LL << (i & 63));
    return total;
}

int test_atomic_xor(cl_device_id deviceID, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    TestFns set = { 0x2f08ab41,
                    0x2f08ab418ba0541LL,
                    NULL,
                    test_atomic_xor_result_int,
                    NULL,
                    NULL,
                    test_atomic_xor_result_long,
                    NULL,
                    NULL };

    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atom_xor_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ false)
        != 0)
        return -1;
    if (test_atomic_function_set(
            deviceID, context, queue, num_elements, atomic_xor_core, set, true,
            /*matchGroupSize*/ false, /*usingAtomicPrefix*/ true)
        != 0)
        return -1;
    return 0;
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestStore
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScope;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTest<HostAtomicType, HostDataType>::CheckCapabilities;
    CBasicTestStore(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        OldValueCheck(false);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return threadCount;
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        if (MemoryOrder() == MEMORY_ORDER_ACQUIRE
            || MemoryOrder() == MEMORY_ORDER_ACQ_REL)
            return 0; // skip test - not applicable

        if (CheckCapabilities(MemoryScope(), MemoryOrder())
            == TEST_SKIPPED_ITSELF)
            return 0; // skip test - not applicable

        return CBasicTestMemOrderScope<
            HostAtomicType, HostDataType>::ExecuteSingleTest(deviceID, context,
                                                             queue);
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  atomic_store" + postfix + "(&destMemory[tid], tid"
            + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        host_atomic_store(&destMemory[tid], (HostDataType)tid, MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = (HostDataType)whichDestValue;
        return true;
    }
};

int test_atomic_store_generic(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements,
                              bool useSVM)
{
    int error = 0;
    CBasicTestStore<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                        useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestStore<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestStore<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestStore<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(TYPE_ATOMIC_ULONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    CBasicTestStore<HOST_ATOMIC_FLOAT, HOST_FLOAT> test_float(TYPE_ATOMIC_FLOAT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_float.Execute(deviceID, context, queue, num_elements));
    CBasicTestStore<HOST_ATOMIC_DOUBLE, HOST_DOUBLE> test_double(
        TYPE_ATOMIC_DOUBLE, useSVM);
    EXECUTE_TEST(error,
                 test_double.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestStore<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestStore<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestStore<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_store(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_atomic_store_generic(deviceID, context, queue, num_elements,
                                     false);
}

int test_svm_atomic_store(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_store_generic(deviceID, context, queue, num_elements,
                                     true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestInit : public CBasicTest<HostAtomicType, HostDataType> {
public:
    using CBasicTest<HostAtomicType, HostDataType>::OldValueCheck;
    CBasicTestInit(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTest<HostAtomicType, HostDataType>(dataType, useSVM)
    {
        OldValueCheck(false);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return threadCount;
    }
    virtual std::string ProgramCore()
    {
        return "  atomic_init(&destMemory[tid], tid);\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        host_atomic_init(&destMemory[tid], (HostDataType)tid);
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = (HostDataType)whichDestValue;
        return true;
    }
};

int test_atomic_init_generic(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements,
                             bool useSVM)
{
    int error = 0;
    CBasicTestInit<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT, useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestInit<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestInit<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestInit<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(TYPE_ATOMIC_ULONG,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    CBasicTestInit<HOST_ATOMIC_FLOAT, HOST_FLOAT> test_float(TYPE_ATOMIC_FLOAT,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_float.Execute(deviceID, context, queue, num_elements));
    CBasicTestInit<HOST_ATOMIC_DOUBLE, HOST_DOUBLE> test_double(
        TYPE_ATOMIC_DOUBLE, useSVM);
    EXECUTE_TEST(error,
                 test_double.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestInit<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestInit<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestInit<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_init(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return test_atomic_init_generic(deviceID, context, queue, num_elements,
                                    false);
}

int test_svm_atomic_init(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return test_atomic_init_generic(deviceID, context, queue, num_elements,
                                    true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestLoad
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScope;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScopeStr;
    using CBasicTest<HostAtomicType, HostDataType>::CheckCapabilities;
    CBasicTestLoad(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        OldValueCheck(false);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return threadCount;
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        if (MemoryOrder() == MEMORY_ORDER_RELEASE
            || MemoryOrder() == MEMORY_ORDER_ACQ_REL)
            return 0; // skip test - not applicable

        if (CheckCapabilities(MemoryScope(), MemoryOrder())
            == TEST_SKIPPED_ITSELF)
            return 0; // skip test - not applicable

        return CBasicTestMemOrderScope<
            HostAtomicType, HostDataType>::ExecuteSingleTest(deviceID, context,
                                                             queue);
    }
    virtual std::string ProgramCore()
    {
        // In the case this test is run with MEMORY_ORDER_ACQUIRE, the store
        // should be MEMORY_ORDER_RELEASE
        std::string memoryOrderScopeLoad = MemoryOrderScopeStr();
        std::string memoryOrderScopeStore =
            (MemoryOrder() == MEMORY_ORDER_ACQUIRE)
            ? (", memory_order_release" + MemoryScopeStr())
            : memoryOrderScopeLoad;
        std::string postfix(memoryOrderScopeLoad.empty() ? "" : "_explicit");
        return "  atomic_store" + postfix + "(&destMemory[tid], tid"
            + memoryOrderScopeStore
            + ");\n"
              "  oldValues[tid] = atomic_load"
            + postfix + "(&destMemory[tid]" + memoryOrderScopeLoad + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        host_atomic_store(&destMemory[tid], (HostDataType)tid,
                          MEMORY_ORDER_SEQ_CST);
        oldValues[tid] = host_atomic_load<HostAtomicType, HostDataType>(
            &destMemory[tid], MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = (HostDataType)whichDestValue;
        return true;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        for (cl_uint i = 0; i < threadCount; i++)
        {
            if (refValues[i] != (HostDataType)i)
            {
                log_error("Invalid value for thread %u\n", (cl_uint)i);
                correct = false;
                return true;
            }
        }
        return true;
    }
};

int test_atomic_load_generic(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements,
                             bool useSVM)
{
    int error = 0;
    CBasicTestLoad<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT, useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestLoad<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestLoad<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestLoad<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(TYPE_ATOMIC_ULONG,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    CBasicTestLoad<HOST_ATOMIC_FLOAT, HOST_FLOAT> test_float(TYPE_ATOMIC_FLOAT,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_float.Execute(deviceID, context, queue, num_elements));
    CBasicTestLoad<HOST_ATOMIC_DOUBLE, HOST_DOUBLE> test_double(
        TYPE_ATOMIC_DOUBLE, useSVM);
    EXECUTE_TEST(error,
                 test_double.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestLoad<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestLoad<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestLoad<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_load(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return test_atomic_load_generic(deviceID, context, queue, num_elements,
                                    false);
}

int test_svm_atomic_load(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return test_atomic_load_generic(deviceID, context, queue, num_elements,
                                    true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestExchange
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::Iterations;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::IterationsStr;
    CBasicTestExchange(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(123456);
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  oldValues[tid] = atomic_exchange" + postfix
            + "(&destMemory[0], tid" + memoryOrderScope
            + ");\n"
              "  for(int i = 0; i < "
            + IterationsStr()
            + "; i++)\n"
              "    oldValues[tid] = atomic_exchange"
            + postfix + "(&destMemory[0], oldValues[tid]" + memoryOrderScope
            + ");\n";
    }

    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        oldValues[tid] = host_atomic_exchange(&destMemory[0], (HostDataType)tid,
                                              MemoryOrder());
        for (int i = 0; i < Iterations(); i++)
            oldValues[tid] = host_atomic_exchange(
                &destMemory[0], oldValues[tid], MemoryOrder());
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        OldValueCheck(
            Iterations() % 2
            == 0); // check is valid for even number of iterations only
        correct = true;
        /* We are expecting values from 0 to size-1 and initial value from
         * atomic variable */
        /* These values must be distributed across refValues array and atomic
         * variable finalVaue[0] */
        /* Any repeated value is treated as an error */
        std::vector<bool> tidFound(threadCount);
        bool startValueFound = false;
        cl_uint i;

        for (i = 0; i <= threadCount; i++)
        {
            cl_uint value;
            if (i == threadCount)
                value = (cl_uint)finalValues[0]; // additional value from atomic
                                                 // variable (last written)
            else
                value = (cl_uint)refValues[i];
            if (value == (cl_uint)StartValue())
            {
                // Special initial value
                if (startValueFound)
                {
                    log_error("ERROR: Starting reference value (%u) occurred "
                              "more thane once\n",
                              (cl_uint)StartValue());
                    correct = false;
                    return true;
                }
                startValueFound = true;
                continue;
            }
            if (value >= threadCount)
            {
                log_error(
                    "ERROR: Reference value %u outside of valid range! (%u)\n",
                    i, value);
                correct = false;
                return true;
            }
            if (tidFound[value])
            {
                log_error("ERROR: Value (%u) occurred more thane once\n",
                          value);
                correct = false;
                return true;
            }
            tidFound[value] = true;
        }
        return true;
    }
};

int test_atomic_exchange_generic(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements,
                                 bool useSVM)
{
    int error = 0;
    CBasicTestExchange<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestExchange<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestExchange<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestExchange<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    CBasicTestExchange<HOST_ATOMIC_FLOAT, HOST_FLOAT> test_float(
        TYPE_ATOMIC_FLOAT, useSVM);
    EXECUTE_TEST(error,
                 test_float.Execute(deviceID, context, queue, num_elements));
    CBasicTestExchange<HOST_ATOMIC_DOUBLE, HOST_DOUBLE> test_double(
        TYPE_ATOMIC_DOUBLE, useSVM);
    EXECUTE_TEST(error,
                 test_double.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestExchange<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestExchange<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestExchange<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_exchange(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return test_atomic_exchange_generic(deviceID, context, queue, num_elements,
                                        false);
}

int test_svm_atomic_exchange(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    return test_atomic_exchange_generic(deviceID, context, queue, num_elements,
                                        true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestCompareStrong
    : public CBasicTestMemOrder2Scope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::MemoryOrder2;
    using CBasicTestMemOrder2Scope<HostAtomicType,
                                   HostDataType>::MemoryOrderScope;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::MemoryScope;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::Iterations;
    using CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>::IterationsStr;
    using CBasicTest<HostAtomicType, HostDataType>::CheckCapabilities;
    CBasicTestCompareStrong(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrder2Scope<HostAtomicType, HostDataType>(dataType,
                                                                 useSVM)
    {
        StartValue(123456);
        OldValueCheck(false);
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        if (MemoryOrder2() == MEMORY_ORDER_RELEASE
            || MemoryOrder2() == MEMORY_ORDER_ACQ_REL)
            return 0; // not allowed as 'failure' argument
        if ((MemoryOrder() == MEMORY_ORDER_RELAXED
             && MemoryOrder2() != MEMORY_ORDER_RELAXED)
            || (MemoryOrder() != MEMORY_ORDER_SEQ_CST
                && MemoryOrder2() == MEMORY_ORDER_SEQ_CST))
            return 0; // failure argument shall be no stronger than the success

        if (CheckCapabilities(MemoryScope(), MemoryOrder())
            == TEST_SKIPPED_ITSELF)
            return 0; // skip test - not applicable

        if (CheckCapabilities(MemoryScope(), MemoryOrder2())
            == TEST_SKIPPED_ITSELF)
            return 0; // skip test - not applicable

        return CBasicTestMemOrder2Scope<
            HostAtomicType, HostDataType>::ExecuteSingleTest(deviceID, context,
                                                             queue);
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScope();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  ") + DataType().RegularTypeName()
            + " expected, previous;\n"
              "  int successCount = 0;\n"
              "  oldValues[tid] = tid;\n"
              "  expected = tid;  // force failure at the beginning\n"
              "  if(atomic_compare_exchange_strong"
            + postfix + "(&destMemory[0], &expected, oldValues[tid]"
            + memoryOrderScope
            + ") || expected == tid)\n"
              "    oldValues[tid] = threadCount+1; //mark unexpected success "
              "with invalid value\n"
              "  else\n"
              "  {\n"
              "    for(int i = 0; i < "
            + IterationsStr()
            + " || successCount == 0; i++)\n"
              "    {\n"
              "      previous = expected;\n"
              "      if(atomic_compare_exchange_strong"
            + postfix + "(&destMemory[0], &expected, oldValues[tid]"
            + memoryOrderScope
            + "))\n"
              "      {\n"
              "        oldValues[tid] = expected;\n"
              "        successCount++;\n"
              "      }\n"
              "      else\n"
              "      {\n"
              "        if(previous == expected) // spurious failure - "
              "shouldn't occur for 'strong'\n"
              "        {\n"
              "          oldValues[tid] = threadCount; //mark fail with "
              "invalid value\n"
              "          break;\n"
              "        }\n"
              "      }\n"
              "    }\n"
              "  }\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        HostDataType expected = (HostDataType)StartValue(), previous;
        oldValues[tid] = (HostDataType)tid;
        for (int i = 0; i < Iterations(); i++)
        {
            previous = expected;
            if (host_atomic_compare_exchange(&destMemory[0], &expected,
                                             oldValues[tid], MemoryOrder(),
                                             MemoryOrder2()))
                oldValues[tid] = expected;
            else
            {
                if (previous == expected) // shouldn't occur for 'strong'
                {
                    oldValues[tid] = threadCount; // mark fail with invalid
                                                  // value
                }
            }
        }
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        /* We are expecting values from 0 to size-1 and initial value from
         * atomic variable */
        /* These values must be distributed across refValues array and atomic
         * variable finalVaue[0] */
        /* Any repeated value is treated as an error */
        std::vector<bool> tidFound(threadCount);
        bool startValueFound = false;
        cl_uint i;

        for (i = 0; i <= threadCount; i++)
        {
            cl_uint value;
            if (i == threadCount)
                value = (cl_uint)finalValues[0]; // additional value from atomic
                                                 // variable (last written)
            else
                value = (cl_uint)refValues[i];
            if (value == (cl_uint)StartValue())
            {
                // Special initial value
                if (startValueFound)
                {
                    log_error("ERROR: Starting reference value (%u) occurred "
                              "more thane once\n",
                              (cl_uint)StartValue());
                    correct = false;
                    return true;
                }
                startValueFound = true;
                continue;
            }
            if (value >= threadCount)
            {
                if (value == threadCount)
                    log_error("ERROR: Spurious failure detected for "
                              "atomic_compare_exchange_strong\n");
                log_error(
                    "ERROR: Reference value %u outside of valid range! (%u)\n",
                    i, value);
                correct = false;
                return true;
            }
            if (tidFound[value])
            {
                log_error("ERROR: Value (%u) occurred more thane once\n",
                          value);
                correct = false;
                return true;
            }
            tidFound[value] = true;
        }
        return true;
    }
};

int test_atomic_compare_exchange_strong_generic(cl_device_id deviceID,
                                                cl_context context,
                                                cl_command_queue queue,
                                                int num_elements, bool useSVM)
{
    int error = 0;
    CBasicTestCompareStrong<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                                useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareStrong<HOST_ATOMIC_UINT, HOST_UINT> test_uint(
        TYPE_ATOMIC_UINT, useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareStrong<HOST_ATOMIC_LONG, HOST_LONG> test_long(
        TYPE_ATOMIC_LONG, useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareStrong<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestCompareStrong<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32>
            test_size_t(TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestCompareStrong<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64>
            test_size_t(TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareStrong<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_compare_exchange_strong(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue queue,
                                        int num_elements)
{
    return test_atomic_compare_exchange_strong_generic(deviceID, context, queue,
                                                       num_elements, false);
}

int test_svm_atomic_compare_exchange_strong(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    return test_atomic_compare_exchange_strong_generic(deviceID, context, queue,
                                                       num_elements, true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestCompareWeak
    : public CBasicTestCompareStrong<HostAtomicType, HostDataType> {
public:
    using CBasicTestCompareStrong<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestCompareStrong<HostAtomicType,
                                  HostDataType>::MemoryOrderScope;
    using CBasicTestCompareStrong<HostAtomicType, HostDataType>::DataType;
    using CBasicTestCompareStrong<HostAtomicType, HostDataType>::Iterations;
    using CBasicTestCompareStrong<HostAtomicType, HostDataType>::IterationsStr;
    CBasicTestCompareWeak(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestCompareStrong<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {}
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScope();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  ") + DataType().RegularTypeName()
            + " expected , previous;\n"
              "  int successCount = 0;\n"
              "  oldValues[tid] = tid;\n"
              "  expected = tid;  // force failure at the beginning\n"
              "  if(atomic_compare_exchange_weak"
            + postfix + "(&destMemory[0], &expected, oldValues[tid]"
            + memoryOrderScope
            + ") || expected == tid)\n"
              "    oldValues[tid] = threadCount+1; //mark unexpected success "
              "with invalid value\n"
              "  else\n"
              "  {\n"
              "    for(int i = 0; i < "
            + IterationsStr()
            + " || successCount == 0; i++)\n"
              "    {\n"
              "      previous = expected;\n"
              "      if(atomic_compare_exchange_weak"
            + postfix + "(&destMemory[0], &expected, oldValues[tid]"
            + memoryOrderScope
            + "))\n"
              "      {\n"
              "        oldValues[tid] = expected;\n"
              "        successCount++;\n"
              "      }\n"
              "    }\n"
              "  }\n";
    }
};

int test_atomic_compare_exchange_weak_generic(cl_device_id deviceID,
                                              cl_context context,
                                              cl_command_queue queue,
                                              int num_elements, bool useSVM)
{
    int error = 0;
    CBasicTestCompareWeak<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareWeak<HOST_ATOMIC_UINT, HOST_UINT> test_uint(
        TYPE_ATOMIC_UINT, useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareWeak<HOST_ATOMIC_LONG, HOST_LONG> test_long(
        TYPE_ATOMIC_LONG, useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestCompareWeak<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestCompareWeak<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestCompareWeak<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestCompareWeak<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_compare_exchange_weak(cl_device_id deviceID, cl_context context,
                                      cl_command_queue queue, int num_elements)
{
    return test_atomic_compare_exchange_weak_generic(deviceID, context, queue,
                                                     num_elements, false);
}

int test_svm_atomic_compare_exchange_weak(cl_device_id deviceID,
                                          cl_context context,
                                          cl_command_queue queue,
                                          int num_elements)
{
    return test_atomic_compare_exchange_weak_generic(deviceID, context, queue,
                                                     num_elements, true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchAdd
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    CBasicTestFetchAdd(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {}
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  oldValues[tid] = atomic_fetch_add" + postfix
            + "(&destMemory[0], (" + DataType().AddSubOperandTypeName()
            + ")tid + 3" + memoryOrderScope + ");\n" + "  atomic_fetch_add"
            + postfix + "(&destMemory[0], ("
            + DataType().AddSubOperandTypeName() + ")tid + 3" + memoryOrderScope
            + ");\n"
              "  atomic_fetch_add"
            + postfix + "(&destMemory[0], ("
            + DataType().AddSubOperandTypeName() + ")tid + 3" + memoryOrderScope
            + ");\n"
              "  atomic_fetch_add"
            + postfix + "(&destMemory[0], (("
            + DataType().AddSubOperandTypeName() + ")tid + 3) << (sizeof("
            + DataType().AddSubOperandTypeName() + ")-1)*8" + memoryOrderScope
            + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        oldValues[tid] = host_atomic_fetch_add(
            &destMemory[0], (HostDataType)tid + 3, MemoryOrder());
        host_atomic_fetch_add(&destMemory[0], (HostDataType)tid + 3,
                              MemoryOrder());
        host_atomic_fetch_add(&destMemory[0], (HostDataType)tid + 3,
                              MemoryOrder());
        host_atomic_fetch_add(&destMemory[0],
                              ((HostDataType)tid + 3)
                                  << (sizeof(HostDataType) - 1) * 8,
                              MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = StartValue();
        for (cl_uint i = 0; i < threadCount; i++)
            expected += ((HostDataType)i + 3) * 3
                + (((HostDataType)i + 3) << (sizeof(HostDataType) - 1) * 8);
        return true;
    }
};

int test_atomic_fetch_add_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchAdd<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAdd<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAdd<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAdd<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchAdd<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchAdd<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAdd<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_add(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_add_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_add(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_add_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchSub
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    CBasicTestFetchSub(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {}
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  oldValues[tid] = atomic_fetch_sub" + postfix
            + "(&destMemory[0], tid + 3 +((("
            + DataType().AddSubOperandTypeName() + ")tid + 3) << (sizeof("
            + DataType().AddSubOperandTypeName() + ")-1)*8)" + memoryOrderScope
            + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        oldValues[tid] = host_atomic_fetch_sub(
            &destMemory[0],
            (HostDataType)tid + 3
                + (((HostDataType)tid + 3) << (sizeof(HostDataType) - 1) * 8),
            MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = StartValue();
        for (cl_uint i = 0; i < threadCount; i++)
            expected -= (HostDataType)i + 3
                + (((HostDataType)i + 3) << (sizeof(HostDataType) - 1) * 8);
        return true;
    }
};

int test_atomic_fetch_sub_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchSub<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchSub<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchSub<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchSub<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchSub<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchSub<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchSub<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_sub(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_sub_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_sub(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_sub_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchOr
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    CBasicTestFetchOr(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(0);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        cl_uint numBits = DataType().Size(deviceID) * 8;

        return (threadCount + numBits - 1) / numBits;
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("    size_t numBits = sizeof(")
            + DataType().RegularTypeName()
            + ") * 8;\n"
              "    int whichResult = tid / numBits;\n"
              "    int bitIndex = tid - (whichResult * numBits);\n"
              "\n"
              "    oldValues[tid] = atomic_fetch_or"
            + postfix + "(&destMemory[whichResult], (("
            + DataType().RegularTypeName() + ")1 << bitIndex) "
            + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        size_t numBits = sizeof(HostDataType) * 8;
        size_t whichResult = tid / numBits;
        size_t bitIndex = tid - (whichResult * numBits);

        oldValues[tid] =
            host_atomic_fetch_or(&destMemory[whichResult],
                                 ((HostDataType)1 << bitIndex), MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        cl_uint numValues = (threadCount + (sizeof(HostDataType) * 8 - 1))
            / (sizeof(HostDataType) * 8);
        if (whichDestValue < numValues - 1)
        {
            expected = ~(HostDataType)0;
            return true;
        }
        // Last item doesn't get or'ed on every bit, so we have to mask away
        cl_uint numBits =
            threadCount - whichDestValue * (sizeof(HostDataType) * 8);
        expected = StartValue();
        for (cl_uint i = 0; i < numBits; i++)
            expected |= ((HostDataType)1 << i);
        return true;
    }
};

int test_atomic_fetch_or_generic(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements,
                                 bool useSVM)
{
    int error = 0;
    CBasicTestFetchOr<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOr<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOr<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOr<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchOr<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchOr<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOr<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_or(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_or_generic(deviceID, context, queue, num_elements,
                                        false);
}

int test_svm_atomic_fetch_or(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_or_generic(deviceID, context, queue, num_elements,
                                        true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchXor
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    CBasicTestFetchXor(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue((HostDataType)0x2f08ab418ba0541LL);
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  int numBits = sizeof(")
            + DataType().RegularTypeName()
            + ") * 8;\n"
              "  int bitIndex = (numBits-1)*(tid+1)/threadCount;\n"
              "\n"
              "  oldValues[tid] = atomic_fetch_xor"
            + postfix + "(&destMemory[0], ((" + DataType().RegularTypeName()
            + ")1 << bitIndex) " + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        int numBits = sizeof(HostDataType) * 8;
        int bitIndex = (numBits - 1) * (tid + 1) / threadCount;

        oldValues[tid] = host_atomic_fetch_xor(
            &destMemory[0], ((HostDataType)1 << bitIndex), MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        int numBits = sizeof(HostDataType) * 8;
        expected = StartValue();
        for (cl_uint i = 0; i < threadCount; i++)
        {
            int bitIndex = (numBits - 1) * (i + 1) / threadCount;
            expected ^= ((HostDataType)1 << bitIndex);
        }
        return true;
    }
};

int test_atomic_fetch_xor_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchXor<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchXor<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchXor<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_xor(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_xor_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_xor(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_xor_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchAnd
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    CBasicTestFetchAnd(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(~(HostDataType)0);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        cl_uint numBits = DataType().Size(deviceID) * 8;

        return (threadCount + numBits - 1) / numBits;
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  size_t numBits = sizeof(")
            + DataType().RegularTypeName()
            + ") * 8;\n"
              "  int whichResult = tid / numBits;\n"
              "  int bitIndex = tid - (whichResult * numBits);\n"
              "\n"
              "  oldValues[tid] = atomic_fetch_and"
            + postfix + "(&destMemory[whichResult], ~(("
            + DataType().RegularTypeName() + ")1 << bitIndex) "
            + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        size_t numBits = sizeof(HostDataType) * 8;
        size_t whichResult = tid / numBits;
        size_t bitIndex = tid - (whichResult * numBits);

        oldValues[tid] = host_atomic_fetch_and(&destMemory[whichResult],
                                               ~((HostDataType)1 << bitIndex),
                                               MemoryOrder());
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        cl_uint numValues = (threadCount + (sizeof(HostDataType) * 8 - 1))
            / (sizeof(HostDataType) * 8);
        if (whichDestValue < numValues - 1)
        {
            expected = 0;
            return true;
        }
        // Last item doesn't get and'ed on every bit, so we have to mask away
        size_t numBits =
            threadCount - whichDestValue * (sizeof(HostDataType) * 8);
        expected = StartValue();
        for (size_t i = 0; i < numBits; i++)
            expected &= ~((HostDataType)1 << i);
        return true;
    }
};

int test_atomic_fetch_and_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchAnd<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAnd<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAnd<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchAnd<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchAnd<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchAnd<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchAnd<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_and(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_and_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_and(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_and_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchOrAnd
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::Iterations;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::IterationsStr;
    CBasicTestFetchOrAnd(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(0);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return 1 + (threadCount - 1) / (DataType().Size(deviceID) * 8);
    }
    // each thread modifies (with OR and AND operations) and verifies
    // only one bit in atomic variable
    // other bits are modified by other threads but it must not affect current
    // thread operation
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  int bits = sizeof(")
            + DataType().RegularTypeName() + ")*8;\n"
            + "  size_t valueInd = tid/bits;\n"
              "  "
            + DataType().RegularTypeName() + " value, bitMask = ("
            + DataType().RegularTypeName()
            + ")1 << tid%bits;\n"
              "  oldValues[tid] = 0;\n"
              "  for(int i = 0; i < "
            + IterationsStr()
            + "; i++)\n"
              "  {\n"
              "    value = atomic_fetch_or"
            + postfix + "(destMemory+valueInd, bitMask" + memoryOrderScope
            + ");\n"
              "    if(value & bitMask) // bit should be set to 0\n"
              "      oldValues[tid]++;\n"
              "    value = atomic_fetch_and"
            + postfix + "(destMemory+valueInd, ~bitMask" + memoryOrderScope
            + ");\n"
              "    if(!(value & bitMask)) // bit should be set to 1\n"
              "      oldValues[tid]++;\n"
              "  }\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        int bits = sizeof(HostDataType) * 8;
        size_t valueInd = tid / bits;
        HostDataType value, bitMask = (HostDataType)1 << tid % bits;
        oldValues[tid] = 0;
        for (int i = 0; i < Iterations(); i++)
        {
            value = host_atomic_fetch_or(destMemory + valueInd, bitMask,
                                         MemoryOrder());
            if (value & bitMask) // bit should be set to 0
                oldValues[tid]++;
            value = host_atomic_fetch_and(destMemory + valueInd, ~bitMask,
                                          MemoryOrder());
            if (!(value & bitMask)) // bit should be set to 1
                oldValues[tid]++;
        }
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = 0;
        return true;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        for (cl_uint i = 0; i < threadCount; i++)
        {
            if (refValues[i] > 0)
            {
                log_error("Thread %d found %d mismatch(es)\n", i,
                          (cl_uint)refValues[i]);
                correct = false;
            }
        }
        return true;
    }
};

int test_atomic_fetch_orand_generic(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements,
                                    bool useSVM)
{
    int error = 0;
    CBasicTestFetchOrAnd<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                             useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOrAnd<HOST_ATOMIC_UINT, HOST_UINT> test_uint(
        TYPE_ATOMIC_UINT, useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOrAnd<HOST_ATOMIC_LONG, HOST_LONG> test_long(
        TYPE_ATOMIC_LONG, useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchOrAnd<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchOrAnd<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchOrAnd<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchOrAnd<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_orand(cl_device_id deviceID, cl_context context,
                            cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_orand_generic(deviceID, context, queue,
                                           num_elements, false);
}

int test_svm_atomic_fetch_orand(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_orand_generic(deviceID, context, queue,
                                           num_elements, true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchXor2
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::Iterations;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::IterationsStr;
    CBasicTestFetchXor2(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(0);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return 1 + (threadCount - 1) / (DataType().Size(deviceID) * 8);
    }
    // each thread modifies (with XOR operation) and verifies
    // only one bit in atomic variable
    // other bits are modified by other threads but it must not affect current
    // thread operation
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return std::string("  int bits = sizeof(")
            + DataType().RegularTypeName() + ")*8;\n"
            + "  size_t valueInd = tid/bits;\n"
              "  "
            + DataType().RegularTypeName() + " value, bitMask = ("
            + DataType().RegularTypeName()
            + ")1 << tid%bits;\n"
              "  oldValues[tid] = 0;\n"
              "  for(int i = 0; i < "
            + IterationsStr()
            + "; i++)\n"
              "  {\n"
              "    value = atomic_fetch_xor"
            + postfix + "(destMemory+valueInd, bitMask" + memoryOrderScope
            + ");\n"
              "    if(value & bitMask) // bit should be set to 0\n"
              "      oldValues[tid]++;\n"
              "    value = atomic_fetch_xor"
            + postfix + "(destMemory+valueInd, bitMask" + memoryOrderScope
            + ");\n"
              "    if(!(value & bitMask)) // bit should be set to 1\n"
              "      oldValues[tid]++;\n"
              "  }\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        int bits = sizeof(HostDataType) * 8;
        size_t valueInd = tid / bits;
        HostDataType value, bitMask = (HostDataType)1 << tid % bits;
        oldValues[tid] = 0;
        for (int i = 0; i < Iterations(); i++)
        {
            value = host_atomic_fetch_xor(destMemory + valueInd, bitMask,
                                          MemoryOrder());
            if (value & bitMask) // bit should be set to 0
                oldValues[tid]++;
            value = host_atomic_fetch_xor(destMemory + valueInd, bitMask,
                                          MemoryOrder());
            if (!(value & bitMask)) // bit should be set to 1
                oldValues[tid]++;
        }
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = 0;
        return true;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        for (cl_uint i = 0; i < threadCount; i++)
        {
            if (refValues[i] > 0)
            {
                log_error("Thread %d found %d mismatches\n", i,
                          (cl_uint)refValues[i]);
                correct = false;
            }
        }
        return true;
    }
};

int test_atomic_fetch_xor2_generic(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements,
                                   bool useSVM)
{
    int error = 0;
    CBasicTestFetchXor2<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                            useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor2<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                               useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor2<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                               useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchXor2<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchXor2<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchXor2<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchXor2<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_xor2(cl_device_id deviceID, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_xor2_generic(deviceID, context, queue,
                                          num_elements, false);
}

int test_svm_atomic_fetch_xor2(cl_device_id deviceID, cl_context context,
                               cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_xor2_generic(deviceID, context, queue,
                                          num_elements, true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchMin
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    CBasicTestFetchMin(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(DataType().MaxValue());
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  oldValues[tid] = atomic_fetch_min" + postfix
            + "(&destMemory[0], oldValues[tid] " + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        oldValues[tid] = host_atomic_fetch_min(&destMemory[0], oldValues[tid],
                                               MemoryOrder());
    }
    virtual bool GenerateRefs(cl_uint threadCount, HostDataType *startRefValues,
                              MTdata d)
    {
        for (cl_uint i = 0; i < threadCount; i++)
        {
            startRefValues[i] = genrand_int32(d);
            if (sizeof(HostDataType) >= 8)
                startRefValues[i] |= (HostDataType)genrand_int32(d) << 16;
        }
        return true;
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = StartValue();
        for (cl_uint i = 0; i < threadCount; i++)
        {
            if (startRefValues[i] < expected) expected = startRefValues[i];
        }
        return true;
    }
};

int test_atomic_fetch_min_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchMin<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMin<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMin<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMin<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchMin<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchMin<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMin<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_min(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_min_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_min(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_min_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFetchMax
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    CBasicTestFetchMax(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(DataType().MinValue());
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        return "  oldValues[tid] = atomic_fetch_max" + postfix
            + "(&destMemory[0], oldValues[tid] " + memoryOrderScope + ");\n";
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        oldValues[tid] = host_atomic_fetch_max(&destMemory[0], oldValues[tid],
                                               MemoryOrder());
    }
    virtual bool GenerateRefs(cl_uint threadCount, HostDataType *startRefValues,
                              MTdata d)
    {
        for (cl_uint i = 0; i < threadCount; i++)
        {
            startRefValues[i] = genrand_int32(d);
            if (sizeof(HostDataType) >= 8)
                startRefValues[i] |= (HostDataType)genrand_int32(d) << 16;
        }
        return true;
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = StartValue();
        for (cl_uint i = 0; i < threadCount; i++)
        {
            if (startRefValues[i] > expected) expected = startRefValues[i];
        }
        return true;
    }
};

int test_atomic_fetch_max_generic(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements,
                                  bool useSVM)
{
    int error = 0;
    CBasicTestFetchMax<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMax<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMax<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFetchMax<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(
        TYPE_ATOMIC_ULONG, useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFetchMax<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFetchMax<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64>
            test_intptr_t(TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFetchMax<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fetch_max(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_max_generic(deviceID, context, queue, num_elements,
                                         false);
}

int test_svm_atomic_fetch_max(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements)
{
    return test_atomic_fetch_max_generic(deviceID, context, queue, num_elements,
                                         true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFlag
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
    static const HostDataType CRITICAL_SECTION_NOT_VISITED = 1000000000;

public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::MemoryOrderScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::UseSVM;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::LocalMemory;
    CBasicTestFlag(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(0);
        OldValueCheck(false);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return threadCount;
    }
    TExplicitMemoryOrderType MemoryOrderForClear()
    {
        // Memory ordering for atomic_flag_clear function
        // ("shall not be memory_order_acquire nor memory_order_acq_rel")
        if (MemoryOrder() == MEMORY_ORDER_ACQUIRE) return MEMORY_ORDER_RELAXED;
        if (MemoryOrder() == MEMORY_ORDER_ACQ_REL) return MEMORY_ORDER_RELEASE;
        return MemoryOrder();
    }
    std::string MemoryOrderScopeStrForClear()
    {
        std::string orderStr;
        if (MemoryOrder() != MEMORY_ORDER_EMPTY)
            orderStr = std::string(", ")
                + get_memory_order_type_name(MemoryOrderForClear());
        return orderStr + MemoryScopeStr();
    }

    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        // This test assumes support for the memory_scope_device scope in the
        // case that LocalMemory() == false. Therefore we should skip this test
        // in that configuration on a 3.0 driver since supporting the
        // memory_scope_device scope is optionaly.
        if (get_device_cl_version(deviceID) >= Version{ 3, 0 })
        {
            if (!LocalMemory()
                && !(gAtomicFenceCap & CL_DEVICE_ATOMIC_SCOPE_DEVICE))
            {
                log_info("Skipping atomic_flag test due to use of "
                         "atomic_scope_device "
                         "which is optionally not supported on this device\n");
                return 0; // skip test - not applicable
            }
        }
        return CBasicTestMemOrderScope<
            HostAtomicType, HostDataType>::ExecuteSingleTest(deviceID, context,
                                                             queue);
    }
    virtual std::string ProgramCore()
    {
        std::string memoryOrderScope = MemoryOrderScopeStr();
        std::string postfix(memoryOrderScope.empty() ? "" : "_explicit");
        std::string program =
            "  uint cnt, stop = 0;\n"
            "  for(cnt = 0; !stop && cnt < threadCount; cnt++) // each thread "
            "must find critical section where it is the first visitor\n"
            "  {\n"
            "    bool set = atomic_flag_test_and_set"
            + postfix + "(&destMemory[cnt]" + memoryOrderScope + ");\n";
        if (MemoryOrder() == MEMORY_ORDER_RELAXED
            || MemoryOrder() == MEMORY_ORDER_RELEASE || LocalMemory())
            program += "    atomic_work_item_fence("
                + std::string(
                           LocalMemory()
                               ? "CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE, "
                               : "CLK_GLOBAL_MEM_FENCE, ")
                + "memory_order_acquire,"
                + std::string(LocalMemory()
                                  ? "memory_scope_work_group"
                                  : (UseSVM() ? "memory_scope_all_svm_devices"
                                              : "memory_scope_device"))
                + ");\n";

        program += "    if (!set)\n"
                   "    {\n";

        if (LocalMemory())
            program += "      uint csIndex = "
                       "get_enqueued_local_size(0)*get_group_id(0)+cnt;\n";
        else
            program += "      uint csIndex = cnt;\n";

        std::ostringstream csNotVisited;
        csNotVisited << CRITICAL_SECTION_NOT_VISITED;
        program += "      // verify that thread is the first visitor\n"
                   "      if(oldValues[csIndex] == "
            + csNotVisited.str()
            + ")\n"
              "      {\n"
              "        oldValues[csIndex] = tid; // set the winner id for this "
              "critical section\n"
              "        stop = 1;\n"
              "      }\n";

        if (MemoryOrder() == MEMORY_ORDER_ACQUIRE
            || MemoryOrder() == MEMORY_ORDER_RELAXED || LocalMemory())
            program += "      atomic_work_item_fence("
                + std::string(
                           LocalMemory()
                               ? "CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE, "
                               : "CLK_GLOBAL_MEM_FENCE, ")
                + "memory_order_release,"
                + std::string(LocalMemory()
                                  ? "memory_scope_work_group"
                                  : (UseSVM() ? "memory_scope_all_svm_devices"
                                              : "memory_scope_device"))
                + ");\n";

        program += "      atomic_flag_clear" + postfix + "(&destMemory[cnt]"
            + MemoryOrderScopeStrForClear()
            + ");\n"
              "    }\n"
              "  }\n";
        return program;
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        cl_uint cnt, stop = 0;
        for (cnt = 0; !stop && cnt < threadCount;
             cnt++) // each thread must find critical section where it is the
                    // first visitor\n"
        {
            if (!host_atomic_flag_test_and_set(&destMemory[cnt], MemoryOrder()))
            {
                cl_uint csIndex = cnt;
                // verify that thread is the first visitor\n"
                if (oldValues[csIndex] == CRITICAL_SECTION_NOT_VISITED)
                {
                    oldValues[csIndex] =
                        tid; // set the winner id for this critical section\n"
                    stop = 1;
                }
                host_atomic_flag_clear(&destMemory[cnt], MemoryOrderForClear());
            }
        }
    }
    virtual bool ExpectedValue(HostDataType &expected, cl_uint threadCount,
                               HostDataType *startRefValues,
                               cl_uint whichDestValue)
    {
        expected = StartValue();
        return true;
    }
    virtual bool GenerateRefs(cl_uint threadCount, HostDataType *startRefValues,
                              MTdata d)
    {
        for (cl_uint i = 0; i < threadCount; i++)
            startRefValues[i] = CRITICAL_SECTION_NOT_VISITED;
        return true;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        /* We are expecting unique values from 0 to threadCount-1 (each critical
         * section must be visited) */
        /* These values must be distributed across refValues array */
        std::vector<bool> tidFound(threadCount);
        cl_uint i;

        for (i = 0; i < threadCount; i++)
        {
            cl_uint value = (cl_uint)refValues[i];
            if (value == CRITICAL_SECTION_NOT_VISITED)
            {
                // Special initial value
                log_error("ERROR: Critical section %u not visited\n", i);
                correct = false;
                return true;
            }
            if (value >= threadCount)
            {
                log_error(
                    "ERROR: Reference value %u outside of valid range! (%u)\n",
                    i, value);
                correct = false;
                return true;
            }
            if (tidFound[value])
            {
                log_error("ERROR: Value (%u) occurred more thane once\n",
                          value);
                correct = false;
                return true;
            }
            tidFound[value] = true;
        }
        return true;
    }
};

int test_atomic_flag_generic(cl_device_id deviceID, cl_context context,
                             cl_command_queue queue, int num_elements,
                             bool useSVM)
{
    int error = 0;
    CBasicTestFlag<HOST_ATOMIC_FLAG, HOST_FLAG> test_flag(TYPE_ATOMIC_FLAG,
                                                          useSVM);
    EXECUTE_TEST(error,
                 test_flag.Execute(deviceID, context, queue, num_elements));
    return error;
}

int test_atomic_flag(cl_device_id deviceID, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    return test_atomic_flag_generic(deviceID, context, queue, num_elements,
                                    false);
}

int test_svm_atomic_flag(cl_device_id deviceID, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    return test_atomic_flag_generic(deviceID, context, queue, num_elements,
                                    true);
}

template <typename HostAtomicType, typename HostDataType>
class CBasicTestFence
    : public CBasicTestMemOrderScope<HostAtomicType, HostDataType> {
    struct TestDefinition
    {
        bool op1IsFence;
        TExplicitMemoryOrderType op1MemOrder;
        bool op2IsFence;
        TExplicitMemoryOrderType op2MemOrder;
    };

public:
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::StartValue;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::OldValueCheck;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryOrder;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScope;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::MemoryScopeStr;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::DeclaredInProgram;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::UsedInFunction;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::DataType;
    using CBasicTestMemOrderScope<HostAtomicType,
                                  HostDataType>::CurrentGroupSize;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::UseSVM;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::LocalMemory;
    using CBasicTestMemOrderScope<HostAtomicType, HostDataType>::LocalRefValues;
    CBasicTestFence(TExplicitAtomicType dataType, bool useSVM)
        : CBasicTestMemOrderScope<HostAtomicType, HostDataType>(dataType,
                                                                useSVM)
    {
        StartValue(0);
        OldValueCheck(false);
    }
    virtual cl_uint NumResults(cl_uint threadCount, cl_device_id deviceID)
    {
        return threadCount;
    }
    virtual cl_uint NumNonAtomicVariablesPerThread()
    {
        if (MemoryOrder() == MEMORY_ORDER_SEQ_CST) return 1;
        if (LocalMemory())
        {
            if (gIsEmbedded)
            {
                if (CurrentGroupSize() > 512) CurrentGroupSize(512);
                return 2; // 1KB of local memory required by spec. Clamp group
                          // size to 512 and allow 2 variables per thread
            }
            else
                return 32 * 1024 / 8 / CurrentGroupSize()
                    - 1; // 32KB of local memory required by spec
        }
        return 256;
    }
    virtual std::string SingleTestName()
    {
        std::string testName;
        if (MemoryOrder() == MEMORY_ORDER_SEQ_CST)
            testName += "seq_cst fence, ";
        else
            testName +=
                std::string(get_memory_order_type_name(_subCase.op1MemOrder))
                    .substr(sizeof("memory_order"))
                + (_subCase.op1IsFence ? " fence" : " atomic")
                + " synchronizes-with "
                + std::string(get_memory_order_type_name(_subCase.op2MemOrder))
                      .substr(sizeof("memory_order"))
                + (_subCase.op2IsFence ? " fence" : " atomic") + ", ";
        testName += CBasicTest<HostAtomicType, HostDataType>::SingleTestName();
        testName += std::string(", ")
            + std::string(get_memory_scope_type_name(MemoryScope()))
                  .substr(sizeof("memory"));
        return testName;
    }
    virtual bool SVMDataBufferAllSVMConsistent()
    {
        // Although memory_scope_all_devices doesn't mention SVM it is just an
        // alias for memory_scope_all_svm_devices.  So both scopes interact with
        // SVM allocations, on devices that support those, just the same.
        return MemoryScope() == MEMORY_SCOPE_ALL_DEVICES
            || MemoryScope() == MEMORY_SCOPE_ALL_SVM_DEVICES;
    }
    virtual int ExecuteForEachParameterSet(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue queue)
    {
        int error = 0;
        // execute 3 (maximum) sub cases for each memory order
        for (_subCaseId = 0; _subCaseId < 3; _subCaseId++)
        {
            EXECUTE_TEST(
                error,
                (CBasicTestMemOrderScope<HostAtomicType, HostDataType>::
                     ExecuteForEachParameterSet(deviceID, context, queue)));
        }
        return error;
    }
    virtual int ExecuteSingleTest(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue)
    {
        if (DeclaredInProgram() || UsedInFunction())
            return 0; // skip test - not applicable - no overloaded fence
                      // functions for different address spaces
        if (MemoryOrder() == MEMORY_ORDER_EMPTY
            || MemoryScope()
                == MEMORY_SCOPE_EMPTY) // empty 'scope' not required since
                                       // opencl20-openclc-rev15
            return 0; // skip test - not applicable
        if ((UseSVM() || gHost) && LocalMemory())
            return 0; // skip test - not applicable for SVM and local memory
        struct TestDefinition acqTests[] = {
            // {op1IsFence, op1MemOrder, op2IsFence, op2MemOrder}
            { false, MEMORY_ORDER_RELEASE, true, MEMORY_ORDER_ACQUIRE },
            { true, MEMORY_ORDER_RELEASE, true, MEMORY_ORDER_ACQUIRE },
            { true, MEMORY_ORDER_ACQ_REL, true, MEMORY_ORDER_ACQUIRE }
        };
        struct TestDefinition relTests[] = {
            { true, MEMORY_ORDER_RELEASE, false, MEMORY_ORDER_ACQUIRE },
            { true, MEMORY_ORDER_RELEASE, true, MEMORY_ORDER_ACQ_REL }
        };
        struct TestDefinition arTests[] = {
            { false, MEMORY_ORDER_RELEASE, true, MEMORY_ORDER_ACQ_REL },
            { true, MEMORY_ORDER_ACQ_REL, false, MEMORY_ORDER_ACQUIRE },
            { true, MEMORY_ORDER_ACQ_REL, true, MEMORY_ORDER_ACQ_REL }
        };
        switch (MemoryOrder())
        {
            case MEMORY_ORDER_ACQUIRE:
                if (_subCaseId
                    >= sizeof(acqTests) / sizeof(struct TestDefinition))
                    return 0;
                _subCase = acqTests[_subCaseId];
                break;
            case MEMORY_ORDER_RELEASE:
                if (_subCaseId
                    >= sizeof(relTests) / sizeof(struct TestDefinition))
                    return 0;
                _subCase = relTests[_subCaseId];
                break;
            case MEMORY_ORDER_ACQ_REL:
                if (_subCaseId
                    >= sizeof(arTests) / sizeof(struct TestDefinition))
                    return 0;
                _subCase = arTests[_subCaseId];
                break;
            case MEMORY_ORDER_SEQ_CST:
                if (_subCaseId != 0) // one special case only
                    return 0;
                break;
            default: return 0;
        }
        LocalRefValues(LocalMemory());
        return CBasicTestMemOrderScope<
            HostAtomicType, HostDataType>::ExecuteSingleTest(deviceID, context,
                                                             queue);
    }
    virtual std::string ProgramHeader(cl_uint maxNumDestItems)
    {
        std::string header;
        if (gOldAPI)
        {
            if (MemoryScope() == MEMORY_SCOPE_EMPTY)
            {
                header += "#define atomic_work_item_fence(x,y)                 "
                          "       mem_fence(x)\n";
            }
            else
            {
                header += "#define atomic_work_item_fence(x,y,z)               "
                          "       mem_fence(x)\n";
            }
        }
        return header
            + CBasicTestMemOrderScope<HostAtomicType, HostDataType>::
                ProgramHeader(maxNumDestItems);
    }
    virtual std::string ProgramCore()
    {
        std::ostringstream naValues;
        naValues << NumNonAtomicVariablesPerThread();
        std::string program, fenceType, nonAtomic;
        if (LocalMemory())
        {
            program = "  size_t myId = get_local_id(0), hisId = "
                      "get_local_size(0)-1-myId;\n";
            fenceType = "CLK_LOCAL_MEM_FENCE";
            nonAtomic = "localValues";
        }
        else
        {
            program = "  size_t myId = tid, hisId = threadCount-1-tid;\n";
            fenceType = "CLK_GLOBAL_MEM_FENCE";
            nonAtomic = "oldValues";
        }
        if (MemoryOrder() == MEMORY_ORDER_SEQ_CST)
        {
            // All threads are divided into pairs.
            // Each thread has its own atomic variable and performs the
            // following actions:
            // - increments its own variable
            // - performs fence operation to propagate its value and to see
            // value from other thread
            // - reads value from other thread's variable
            // - repeats the above steps when both values are the same (and less
            // than 1000000)
            // - stores the last value read from other thread (in additional
            // variable) At the end of execution at least one thread should know
            // the last value from other thread
            program += std::string("") + "  " + DataType().RegularTypeName()
                + " myValue = 0, hisValue; \n"
                  "  do {\n"
                  "    myValue++;\n"
                  "    atomic_store_explicit(&destMemory[myId], myValue, "
                  "memory_order_relaxed"
                + MemoryScopeStr()
                + ");\n"
                  "    atomic_work_item_fence("
                + fenceType + ", memory_order_seq_cst" + MemoryScopeStr()
                + "); \n"
                  "    hisValue = atomic_load_explicit(&destMemory[hisId], "
                  "memory_order_relaxed"
                + MemoryScopeStr()
                + ");\n"
                  "  } while(myValue == hisValue && myValue < 1000000);\n"
                  "  "
                + nonAtomic + "[myId] = hisValue; \n";
        }
        else
        {
            // Each thread modifies one of its non-atomic variables, increments
            // value of its atomic variable and reads values from another thread
            // in typical synchronizes-with scenario with:
            // - non-atomic variable (at index A) modification (value change
            // from 0 to A)
            // - release operation (additional fence or within atomic) + atomic
            // variable modification (value A)
            // - atomic variable read (value B) + acquire operation (additional
            // fence or within atomic)
            // - non-atomic variable (at index B) read (value C)
            // Each thread verifies dependency between atomic and non-atomic
            // value read from another thread The following condition must be
            // true: B == C
            program += std::string("") + "  " + DataType().RegularTypeName()
                + " myValue = 0, hisAtomicValue, hisValue; \n"
                  "  do {\n"
                  "    myValue++;\n"
                  "    "
                + nonAtomic + "[myId*" + naValues.str()
                + "+myValue] = myValue;\n";
            if (_subCase.op1IsFence)
                program += std::string("") + "    atomic_work_item_fence("
                    + fenceType + ", "
                    + get_memory_order_type_name(_subCase.op1MemOrder)
                    + MemoryScopeStr()
                    + "); \n"
                      "    atomic_store_explicit(&destMemory[myId], myValue, "
                      "memory_order_relaxed"
                    + MemoryScopeStr() + ");\n";
            else
                program += std::string("")
                    + "    atomic_store_explicit(&destMemory[myId], myValue, "
                    + get_memory_order_type_name(_subCase.op1MemOrder)
                    + MemoryScopeStr() + ");\n";
            if (_subCase.op2IsFence)
                program += std::string("")
                    + "    hisAtomicValue = "
                      "atomic_load_explicit(&destMemory[hisId], "
                      "memory_order_relaxed"
                    + MemoryScopeStr()
                    + ");\n"
                      "    atomic_work_item_fence("
                    + fenceType + ", "
                    + get_memory_order_type_name(_subCase.op2MemOrder)
                    + MemoryScopeStr() + "); \n";
            else
                program += std::string("")
                    + "    hisAtomicValue = "
                      "atomic_load_explicit(&destMemory[hisId], "
                    + get_memory_order_type_name(_subCase.op2MemOrder)
                    + MemoryScopeStr() + ");\n";
            program += "    hisValue = " + nonAtomic + "[hisId*"
                + naValues.str() + "+hisAtomicValue]; \n";
            if (LocalMemory())
                program += "    hisId = (hisId+1)%get_local_size(0);\n";
            else
                program += "    hisId = (hisId+1)%threadCount;\n";
            program += "  } while(hisAtomicValue == hisValue && myValue < "
                + naValues.str()
                + "-1);\n"
                  "  if(hisAtomicValue != hisValue)\n"
                  "  { // fail\n"
                  "    atomic_store_explicit(&destMemory[myId], myValue-1,"
                  " memory_order_relaxed, memory_scope_work_group);\n";
            if (LocalMemory())
                program += "    hisId = "
                           "(hisId+get_local_size(0)-1)%get_local_size(0);\n";
            else
                program += "    hisId = (hisId+threadCount-1)%threadCount;\n";
            program += "    if(myValue+1 < " + naValues.str()
                + ")\n"
                  "      "
                + nonAtomic + "[myId*" + naValues.str()
                + "+myValue+1] = hisId;\n"
                  "    if(myValue+2 < "
                + naValues.str()
                + ")\n"
                  "      "
                + nonAtomic + "[myId*" + naValues.str()
                + "+myValue+2] = hisAtomicValue;\n"
                  "    if(myValue+3 < "
                + naValues.str()
                + ")\n"
                  "      "
                + nonAtomic + "[myId*" + naValues.str()
                + "+myValue+3] = hisValue;\n";
            if (gDebug)
            {
                program += "    printf(\"WI %d: atomic value (%d) at index %d "
                           "is different than non-atomic value (%d)\\n\", tid, "
                           "hisAtomicValue, hisId, hisValue);\n";
            }
            program += "  }\n";
        }
        return program;
    }
    virtual void HostFunction(cl_uint tid, cl_uint threadCount,
                              volatile HostAtomicType *destMemory,
                              HostDataType *oldValues)
    {
        size_t myId = tid, hisId = threadCount - 1 - tid;
        if (MemoryOrder() == MEMORY_ORDER_SEQ_CST)
        {
            HostDataType myValue = 0, hisValue;
            // CPU thread typically starts faster - wait for GPU thread
            myValue++;
            host_atomic_store<HostAtomicType, HostDataType>(
                &destMemory[myId], myValue, MEMORY_ORDER_SEQ_CST);
            while (host_atomic_load<HostAtomicType, HostDataType>(
                       &destMemory[hisId], MEMORY_ORDER_SEQ_CST)
                   == 0)
                ;
            do
            {
                myValue++;
                host_atomic_store<HostAtomicType, HostDataType>(
                    &destMemory[myId], myValue, MEMORY_ORDER_RELAXED);
                host_atomic_thread_fence(MemoryOrder());
                hisValue = host_atomic_load<HostAtomicType, HostDataType>(
                    &destMemory[hisId], MEMORY_ORDER_RELAXED);
            } while (myValue == hisValue && hisValue < 1000000);
            oldValues[tid] = hisValue;
        }
        else
        {
            HostDataType myValue = 0, hisAtomicValue, hisValue;
            do
            {
                myValue++;
                oldValues[myId * NumNonAtomicVariablesPerThread() + myValue] =
                    myValue;
                if (_subCase.op1IsFence)
                {
                    host_atomic_thread_fence(_subCase.op1MemOrder);
                    host_atomic_store<HostAtomicType, HostDataType>(
                        &destMemory[myId], myValue, MEMORY_ORDER_RELAXED);
                }
                else
                    host_atomic_store<HostAtomicType, HostDataType>(
                        &destMemory[myId], myValue, _subCase.op1MemOrder);
                if (_subCase.op2IsFence)
                {
                    hisAtomicValue =
                        host_atomic_load<HostAtomicType, HostDataType>(
                            &destMemory[hisId], MEMORY_ORDER_RELAXED);
                    host_atomic_thread_fence(_subCase.op2MemOrder);
                }
                else
                    hisAtomicValue =
                        host_atomic_load<HostAtomicType, HostDataType>(
                            &destMemory[hisId], _subCase.op2MemOrder);
                hisValue = oldValues[hisId * NumNonAtomicVariablesPerThread()
                                     + hisAtomicValue];
                hisId = (hisId + 1) % threadCount;
            } while (hisAtomicValue == hisValue
                     && myValue
                         < (HostDataType)NumNonAtomicVariablesPerThread() - 1);
            if (hisAtomicValue != hisValue)
            { // fail
                host_atomic_store<HostAtomicType, HostDataType>(
                    &destMemory[myId], myValue - 1, MEMORY_ORDER_SEQ_CST);
                if (gDebug)
                {
                    hisId = (hisId + threadCount - 1) % threadCount;
                    printf("WI %d: atomic value (%d) at index %d is different "
                           "than non-atomic value (%d)\n",
                           tid, hisAtomicValue, hisId, hisValue);
                }
            }
        }
    }
    virtual bool GenerateRefs(cl_uint threadCount, HostDataType *startRefValues,
                              MTdata d)
    {
        for (cl_uint i = 0; i < threadCount * NumNonAtomicVariablesPerThread();
             i++)
            startRefValues[i] = 0;
        return true;
    }
    virtual bool VerifyRefs(bool &correct, cl_uint threadCount,
                            HostDataType *refValues,
                            HostAtomicType *finalValues)
    {
        correct = true;
        cl_uint workSize = LocalMemory() ? CurrentGroupSize() : threadCount;
        for (cl_uint workOffset = 0; workOffset < threadCount;
             workOffset += workSize)
        {
            if (workOffset + workSize > threadCount)
                // last workgroup (host threads)
                workSize = threadCount - workOffset;
            for (cl_uint i = 0; i < workSize && workOffset + i < threadCount;
                 i++)
            {
                HostAtomicType myValue = finalValues[workOffset + i];
                if (MemoryOrder() == MEMORY_ORDER_SEQ_CST)
                {
                    HostDataType hisValue = refValues[workOffset + i];
                    if (myValue == hisValue)
                    {
                        // a draw - both threads should reach final value
                        // 1000000
                        if (myValue != 1000000)
                        {
                            log_error("ERROR: Invalid reference value #%u (%d "
                                      "instead of 1000000)\n",
                                      workOffset + i, myValue);
                            correct = false;
                            return true;
                        }
                    }
                    else
                    {
                        // slower thread (in total order of seq_cst operations)
                        // must know last value written by faster thread
                        HostAtomicType hisRealValue =
                            finalValues[workOffset + workSize - 1 - i];
                        HostDataType myValueReadByHim =
                            refValues[workOffset + workSize - 1 - i];

                        // who is the winner? - thread with lower private
                        // counter value
                        if (myValue == hisRealValue) // forbidden result - fence
                                                     // doesn't work
                        {
                            log_error("ERROR: Atomic counter values #%u and "
                                      "#%u are the same (%u)\n",
                                      workOffset + i,
                                      workOffset + workSize - 1 - i, myValue);
                            log_error(
                                "ERROR: Both threads have outdated values read "
                                "from another thread (%u and %u)\n",
                                hisValue, myValueReadByHim);
                            correct = false;
                            return true;
                        }
                        if (myValue > hisRealValue) // I'm slower
                        {
                            if (hisRealValue != hisValue)
                            {
                                log_error("ERROR: Invalid reference value #%u "
                                          "(%d instead of %d)\n",
                                          workOffset + i, hisValue,
                                          hisRealValue);
                                log_error(
                                    "ERROR: Slower thread #%u should know "
                                    "value written by faster thread #%u\n",
                                    workOffset + i,
                                    workOffset + workSize - 1 - i);
                                correct = false;
                                return true;
                            }
                        }
                        else // I'm faster
                        {
                            if (myValueReadByHim != myValue)
                            {
                                log_error("ERROR: Invalid reference value #%u "
                                          "(%d instead of %d)\n",
                                          workOffset + workSize - 1 - i,
                                          myValueReadByHim, myValue);
                                log_error(
                                    "ERROR: Slower thread #%u should know "
                                    "value written by faster thread #%u\n",
                                    workOffset + workSize - 1 - i,
                                    workOffset + i);
                                correct = false;
                                return true;
                            }
                        }
                    }
                }
                else
                {
                    if (myValue != NumNonAtomicVariablesPerThread() - 1)
                    {
                        log_error("ERROR: Invalid atomic value #%u (%d instead "
                                  "of %d)\n",
                                  workOffset + i, myValue,
                                  NumNonAtomicVariablesPerThread() - 1);
                        log_error("ERROR: Thread #%u observed invalid values "
                                  "in other thread's variables\n",
                                  workOffset + i);
                        correct = false;
                        return true;
                    }
                }
            }
        }
        return true;
    }

private:
    int _subCaseId;
    struct TestDefinition _subCase;
};

int test_atomic_fence_generic(cl_device_id deviceID, cl_context context,
                              cl_command_queue queue, int num_elements,
                              bool useSVM)
{
    int error = 0;
    CBasicTestFence<HOST_ATOMIC_INT, HOST_INT> test_int(TYPE_ATOMIC_INT,
                                                        useSVM);
    EXECUTE_TEST(error,
                 test_int.Execute(deviceID, context, queue, num_elements));
    CBasicTestFence<HOST_ATOMIC_UINT, HOST_UINT> test_uint(TYPE_ATOMIC_UINT,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_uint.Execute(deviceID, context, queue, num_elements));
    CBasicTestFence<HOST_ATOMIC_LONG, HOST_LONG> test_long(TYPE_ATOMIC_LONG,
                                                           useSVM);
    EXECUTE_TEST(error,
                 test_long.Execute(deviceID, context, queue, num_elements));
    CBasicTestFence<HOST_ATOMIC_ULONG, HOST_ULONG> test_ulong(TYPE_ATOMIC_ULONG,
                                                              useSVM);
    EXECUTE_TEST(error,
                 test_ulong.Execute(deviceID, context, queue, num_elements));
    if (AtomicTypeInfo(TYPE_ATOMIC_SIZE_T).Size(deviceID) == 4)
    {
        CBasicTestFence<HOST_ATOMIC_INTPTR_T32, HOST_INTPTR_T32> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_UINTPTR_T32, HOST_UINTPTR_T32>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_SIZE_T32, HOST_SIZE_T32> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_PTRDIFF_T32, HOST_PTRDIFF_T32>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    else
    {
        CBasicTestFence<HOST_ATOMIC_INTPTR_T64, HOST_INTPTR_T64> test_intptr_t(
            TYPE_ATOMIC_INTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_intptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_UINTPTR_T64, HOST_UINTPTR_T64>
            test_uintptr_t(TYPE_ATOMIC_UINTPTR_T, useSVM);
        EXECUTE_TEST(
            error,
            test_uintptr_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_SIZE_T64, HOST_SIZE_T64> test_size_t(
            TYPE_ATOMIC_SIZE_T, useSVM);
        EXECUTE_TEST(
            error, test_size_t.Execute(deviceID, context, queue, num_elements));
        CBasicTestFence<HOST_ATOMIC_PTRDIFF_T64, HOST_PTRDIFF_T64>
            test_ptrdiff_t(TYPE_ATOMIC_PTRDIFF_T, useSVM);
        EXECUTE_TEST(
            error,
            test_ptrdiff_t.Execute(deviceID, context, queue, num_elements));
    }
    return error;
}

int test_atomic_fence(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_atomic_fence_generic(deviceID, context, queue, num_elements,
                                     false);
}

int test_svm_atomic_fence(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_atomic_fence_generic(deviceID, context, queue, num_elements,
                                     true);
}
