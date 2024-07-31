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
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"

#include "structs.h"

#include "defines.h"

#include "type_replacer.h"


/*
 test_step_type,
 test_step_var,
 test_step_typedef_type,
 test_step_typedef_var,
 */


int test_step_internal(cl_device_id deviceID, cl_context context,
                       cl_command_queue queue, const char* pattern,
                       const char* testName)
{
    int err;
    int typeIdx, vecSizeIdx;

    char tempBuffer[2048];

    clState* pClState = newClState(deviceID, context, queue);
    bufferStruct* pBuffers =
        newBufferStruct(BUFFER_SIZE, BUFFER_SIZE, pClState);

    if (pBuffers == NULL)
    {
        destroyClState(pClState);
        vlog_error("%s : Could not create buffer\n", testName);
        return -1;
    }

    for (typeIdx = 0; types[typeIdx] != kNumExplicitTypes; ++typeIdx)
    {
        if (types[typeIdx] == kDouble)
        {
            // If we're testing doubles, we need to check for support first
            if (!is_extension_available(deviceID, "cl_khr_fp64"))
            {
                log_info("Not testing doubles (unsupported on this device)\n");
                continue;
            }
        }

        if (types[typeIdx] == kLong || types[typeIdx] == kULong)
        {
            // If we're testing long/ulong, we need to check for embedded
            // support
            if (gIsEmbedded
                && !is_extension_available(deviceID, "cles_khr_int64"))
            {
                log_info("Not testing longs (unsupported on this embedded "
                         "device)\n");
                continue;
            }
        }

        char srcBuffer[2048];

        doSingleReplace(tempBuffer, 2048, pattern, ".EXTENSIONS.",
                        types[typeIdx] == kDouble
                            ? "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                            : "");

        for (vecSizeIdx = 0; vecSizeIdx < NUM_VECTOR_SIZES; ++vecSizeIdx)
        {
            doReplace(srcBuffer, 2048, tempBuffer, ".TYPE.",
                      g_arrTypeNames[typeIdx], ".NUM.",
                      g_arrVecSizeNames[vecSizeIdx]);

            if (srcBuffer[0] == '\0')
            {
                vlog_error("%s: failed to fill source buf for type %s%s\n",
                           testName, g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            err = clStateMakeProgram(pClState, srcBuffer, testName);
            if (err)
            {
                vlog_error("%s: Error compiling \"\n%s\n\"", testName,
                           srcBuffer);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            err = pushArgs(pBuffers, pClState);
            if (err != 0)
            {
                vlog_error("%s: failed to push args %s%s\n", testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            // now we run the kernel
            err = runKernel(pClState, 1024);
            if (err != 0)
            {
                vlog_error("%s: runKernel fail (%ld threads) %s%s\n", testName,
                           pClState->m_numThreads, g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            err = retrieveResults(pBuffers, pClState);
            if (err != 0)
            {
                vlog_error("%s: failed to retrieve results %s%s\n", testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            err = checkCorrectnessStep(pBuffers, pClState,
                                       g_arrTypeSizes[typeIdx],
                                       g_arrVecSizes[vecSizeIdx]);

            if (err != 0)
            {
                vlog_error("%s: incorrect results %s%s\n", testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                vlog_error("%s: Source was \"\n%s\n\"", testName, srcBuffer);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            clStateDestroyProgramAndKernel(pClState);
        }
    }

    destroyBufferStruct(pBuffers, pClState);

    destroyClState(pClState);


    // vlog_error("%s : implementation incomplete : FAIL\n", testName);
    return 0; // -1; // fails on account of not being written.
}

static const char* patterns[] = {
    ".EXTENSIONS.\n"
    "__kernel void test_step_type(__global .TYPE..NUM. *source, __global int "
    "*dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = vec_step(.TYPE..NUM.);\n"
    "\n"
    "}\n",

    ".EXTENSIONS.\n"
    "__kernel void test_step_var(__global .TYPE..NUM. *source, __global int "
    "*dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = vec_step(source[tid]);\n"
    "\n"
    "}\n",

    ".EXTENSIONS.\n"
    " typedef .TYPE..NUM. TypeToTest;\n"
    "__kernel void test_step_typedef_type(__global TypeToTest *source, "
    "__global int *dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = vec_step(TypeToTest);\n"
    "\n"
    "}\n",

    ".EXTENSIONS.\n"
    " typedef .TYPE..NUM. TypeToTest;\n"
    "__kernel void test_step_typedef_var(__global TypeToTest *source, __global "
    "int *dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = vec_step(source[tid]);\n"
    "\n"
    "}\n",
};

/*
 test_step_type,
 test_step_var,
 test_step_typedef_type,
 test_step_typedef_var,
 */

int test_step_type(cl_device_id deviceID, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    return test_step_internal(deviceID, context, queue, patterns[0],
                              "test_step_type");
}

int test_step_var(cl_device_id deviceID, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    return test_step_internal(deviceID, context, queue, patterns[1],
                              "test_step_var");
}

int test_step_typedef_type(cl_device_id deviceID, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    return test_step_internal(deviceID, context, queue, patterns[2],
                              "test_step_typedef_type");
}

int test_step_typedef_var(cl_device_id deviceID, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    return test_step_internal(deviceID, context, queue, patterns[3],
                              "test_step_typedef_var");
}
