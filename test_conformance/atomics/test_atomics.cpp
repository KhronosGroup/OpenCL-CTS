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
    TestFns set = { (cl_int)0xffffffff,
                    (cl_long)0xffffffffffffffffLL,
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
