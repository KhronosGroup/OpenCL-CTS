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


size_t get_align(size_t vecSize)
{
    if(vecSize == 3)
    {
        return 4;
    }
    return vecSize;
}

/* // Lots of conditionals means this is not gonna be an optimal min on intel. */
/* // That's okay, make sure we only call a few times per test, not for every */
/* // element */
/* size_t min_of_nonzero(size_t a, size_t b) */
/* { */
/*     if(a != 0 && (a<=b || b==0)) */
/*     { */
/*     return a; */
/*     } */
/*     if(b != 0 && (b<a || a==0)) */
/*     { */
/*     return b; */
/*     } */
/*     return 0; */
/* } */


/* size_t get_min_packed_alignment(size_t preSize, size_t typeMultiplePreSize, */
/*                 size_t postSize, size_t typeMultiplePostSize, */
/*                 ExplicitType kType, size_t vecSize) */
/* { */
/*     size_t pre_min = min_of_nonzero(preSize,  */
/*                     typeMultiplePreSize* */
/*                     get_explicit_type_size(kType)); */
/*     size_t post_min = min_of_nonzero(postSize,  */
/*                     typeMultiplePostSize* */
/*                     get_explicit_type_size(kType)); */
/*     size_t struct_min = min_of_nonzero(pre_min, post_min); */
/*     size_t result =  min_of_nonzero(struct_min, get_align(vecSize) */
/*                     *get_explicit_type_size(kType)); */
/*     return result; */

/* } */



int test_vec_internal(cl_device_id deviceID, cl_context context,
                      cl_command_queue queue, const char * pattern,
                      const char * testName, size_t bufSize,
                      size_t preSize, size_t typeMultiplePreSize,
                      size_t postSize, size_t typeMultiplePostSize)
{
    int err;
    int typeIdx, vecSizeIdx;

    char tmpBuffer[2048];
    char srcBuffer[2048];

    size_t preSizeBytes, postSizeBytes, typeSize, totSize;

    clState * pClState = newClState(deviceID, context, queue);
    bufferStruct * pBuffers =
    newBufferStruct(bufSize, bufSize*sizeof(cl_uint)/sizeof(cl_char), pClState);

    if(pBuffers == NULL) {
        destroyClState(pClState);
        vlog_error("%s : Could not create buffer\n", testName);
        return -1;
    }

    for(typeIdx = 0; types[typeIdx] != kNumExplicitTypes; ++typeIdx)
    {

        // Skip doubles if it is not supported otherwise enable pragma
        if (types[typeIdx] == kDouble) {
            if (!is_extension_available(deviceID, "cl_khr_fp64")) {
                continue;
            } else {
                doReplace(tmpBuffer, 2048, pattern,
                          ".PRAGMA.",  "#pragma OPENCL EXTENSION cl_khr_fp64: ",
                          ".STATE.", "enable");
            }
        } else {
            if (types[typeIdx] == kLong || types[typeIdx] == kULong) {
                if (gIsEmbedded)
                    continue;
            }

            doReplace(tmpBuffer, 2048, pattern,
                      ".PRAGMA.",  " ",
                      ".STATE.", " ");
        }

        typeSize = get_explicit_type_size(types[typeIdx]);
        preSizeBytes = preSize + typeSize*typeMultiplePreSize;
        postSizeBytes = postSize + typeSize*typeMultiplePostSize;



        for(vecSizeIdx = 1; vecSizeIdx < NUM_VECTOR_SIZES; ++vecSizeIdx)  {

            totSize = preSizeBytes + postSizeBytes +
            typeSize*get_align(g_arrVecSizes[vecSizeIdx]);

            doReplace(srcBuffer, 2048, tmpBuffer,
                      ".TYPE.",  g_arrTypeNames[typeIdx],
                      ".NUM.", g_arrVecSizeNames[vecSizeIdx]);

            if(srcBuffer[0] == '\0') {
                vlog_error("%s: failed to fill source buf for type %s%s\n",
                           testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            // log_info("Buffer is \"\n%s\n\"\n", srcBuffer);
            // fflush(stdout);

            err = clStateMakeProgram(pClState, srcBuffer, testName );
            if (err) {
                vlog_error("%s: Error compiling \"\n%s\n\"",
                           testName, srcBuffer);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            err = pushArgs(pBuffers, pClState);
            if(err != 0) {
                vlog_error("%s: failed to push args %s%s\n",
                           testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            // log_info("About to Run kernel\n"); fflush(stdout);
            // now we run the kernel
            err = runKernel(pClState,
                            bufSize/(g_arrVecSizes[vecSizeIdx]* g_arrTypeSizes[typeIdx]));
            if(err != 0) {
                vlog_error("%s: runKernel fail (%ld threads) %s%s\n",
                           testName, pClState->m_numThreads,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }

            // log_info("About to retrieve results\n"); fflush(stdout);
            err = retrieveResults(pBuffers, pClState);
            if(err != 0) {
                vlog_error("%s: failed to retrieve results %s%s\n",
                           testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                destroyBufferStruct(pBuffers, pClState);
                destroyClState(pClState);
                return -1;
            }



            if(preSizeBytes+postSizeBytes == 0)
            {
                // log_info("About to Check Correctness\n"); fflush(stdout);
                err = checkCorrectness(pBuffers, pClState,
                                       get_align(g_arrVecSizes[vecSizeIdx])*
                                       typeSize);
            }
            else
            {
                // we're checking for an aligned struct
                err = checkPackedCorrectness(pBuffers, pClState, totSize,
                                             preSizeBytes);
            }

            if(err != 0) {
                vlog_error("%s: incorrect results %s%s\n",
                           testName,
                           g_arrTypeNames[typeIdx],
                           g_arrVecSizeNames[vecSizeIdx]);
                vlog_error("%s: Source was \"\n%s\n\"",
                           testName, srcBuffer);
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



const char * patterns[] = {
    ".PRAGMA..STATE.\n"
    "__kernel void test_vec_align_array(.SRC_SCOPE. .TYPE..NUM. *source, .DST_SCOPE. uint *dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = (uint)((.SRC_SCOPE. uchar *)(source+tid));\n"
    "}\n",
    ".PRAGMA..STATE.\n"
    "typedef struct myUnpackedStruct { \n"
    ".PRE."
    "    .TYPE..NUM. vec;\n"
    ".POST."
    "} testStruct;\n"
    "__kernel void test_vec_align_struct(__constant .TYPE..NUM. *source, .DST_SCOPE. uint *dest)\n"
    "{\n"
    "    .SRC_SCOPE. testStruct test;\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = (uint)((.SRC_SCOPE. uchar *)&(test.vec));\n"
    "}\n",
    ".PRAGMA..STATE.\n"
    "typedef struct __attribute__ ((packed)) myPackedStruct { \n"
    ".PRE."
    "    .TYPE..NUM. vec;\n"
    ".POST."
    "} testStruct;\n"
    "__kernel void test_vec_align_packed_struct(__constant .TYPE..NUM. *source, .DST_SCOPE. uint *dest)\n"
    "{\n"
    "    .SRC_SCOPE. testStruct test;\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = (uint)((.SRC_SCOPE. uchar *)&(test.vec) - (.SRC_SCOPE. uchar *)&test);\n"
    "}\n",
    ".PRAGMA..STATE.\n"
    "typedef struct myStruct { \n"
    ".PRE."
    "    .TYPE..NUM. vec;\n"
    ".POST."
    "} testStruct;\n"
    "__kernel void test_vec_align_struct_arr(.SRC_SCOPE. testStruct *source, .DST_SCOPE. uint *dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = (uint)((.SRC_SCOPE. uchar *)&(source[tid].vec));\n"
    "}\n",
    ".PRAGMA..STATE.\n"
    "typedef struct __attribute__ ((packed)) myPackedStruct { \n"
    ".PRE."
    "    .TYPE..NUM. vec;\n"
    ".POST."
    "} testStruct;\n"
    "__kernel void test_vec_align_packed_struct_arr(.SRC_SCOPE.  testStruct *source, .DST_SCOPE. uint *dest)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dest[tid] = (uint)((.SRC_SCOPE. uchar *)&(source[tid].vec) - (.SRC_SCOPE. uchar *)&(source[0]));\n"
    "}\n",
    // __attribute__ ((packed))
};



const char * pre_substitution_arr[] = {
    "",
    "char c;\n",
    "short3 s;",
    ".TYPE.3 tPre;\n",
    ".TYPE. arrPre[5];\n",
    ".TYPE. arrPre[12];\n",
    NULL
};


// alignments of everything in pre_substitution_arr as raw alignments
// 0 if such a thing is meaningless
size_t pre_align_arr[] = {
    0,
    sizeof(cl_char),
    4*sizeof(cl_short),
    0, // taken care of in type_multiple_pre_align_arr
    0,
    0
};

// alignments of everything in pre_substitution_arr as multiples of
// sizeof(.TYPE.)
// 0 if such a thing is meaningless
size_t type_multiple_pre_align_arr[] = {
    0,
    0,
    0,
    4,
    5,
    12
};

const char * post_substitution_arr[] = {
    "",
    "char cPost;\n",
    ".TYPE. arrPost[3];\n",
    ".TYPE. arrPost[5];\n",
    ".TYPE.3 arrPost;\n",
    ".TYPE. arrPost[12];\n",
    NULL
};


// alignments of everything in post_substitution_arr as raw alignments
// 0 if such a thing is meaningless
size_t post_align_arr[] = {
    0,
    sizeof(cl_char),
    0, // taken care of in type_multiple_post_align_arr
    0,
    0,
    0
};

// alignments of everything in post_substitution_arr as multiples of
// sizeof(.TYPE.)
// 0 if such a thing is meaningless
size_t type_multiple_post_align_arr[] = {
    0,
    0,
    3,
    5,
    4,
    12
};

// there hsould be a packed version of this?
int test_vec_align_array(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char tmp[2048];
    int result;

    log_info("Testing global\n");
    doReplace(tmp, (size_t)2048, patterns[0],
              ".SRC_SCOPE.",  "__global",
              ".DST_SCOPE.", "__global"); //
    result = test_vec_internal(deviceID, context, queue, tmp,
                               "test_vec_align_array",
                               BUFFER_SIZE, 0, 0, 0, 0);
    return result;
}


int test_vec_align_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char tmp1[2048], tmp2[2048];
    int result = 0;
    int preIdx, postIdx;

    log_info("testing __private\n");
    doReplace(tmp2, (size_t)2048, patterns[1],
              ".SRC_SCOPE.",  "__private",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_struct",
                                       512, 0, 0, 0, 0);
            if (result != 0) {
                return result;
            }
        }
    }

    log_info("testing __local\n");
    doReplace(tmp2, (size_t)2048, patterns[1],
              ".SRC_SCOPE.",  "__local",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_struct",
                                       512, 0, 0, 0, 0);
            if(result != 0) {
                return result;
            }
        }
    }
    return 0;
}

int test_vec_align_packed_struct(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char tmp1[2048], tmp2[2048];
    int result = 0;
    int preIdx, postIdx;


    log_info("Testing __private\n");
    doReplace(tmp2, (size_t)2048, patterns[2],
              ".SRC_SCOPE.",  "__private",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_packed_struct",
                                       512, pre_align_arr[preIdx],
                                       type_multiple_pre_align_arr[preIdx],
                                       post_align_arr[postIdx],
                                       type_multiple_post_align_arr[postIdx]);
            if(result != 0) {
                return result;
            }
        }
    }

    log_info("testing __local\n");
    doReplace(tmp2, (size_t)2048, patterns[2],
              ".SRC_SCOPE.",  "__local",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_packed_struct",
                                       512, pre_align_arr[preIdx],
                                       type_multiple_pre_align_arr[preIdx],
                                       post_align_arr[postIdx],
                                       type_multiple_post_align_arr[postIdx]);
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

int test_vec_align_struct_arr(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char tmp1[2048], tmp2[2048];
    int result = 0;
    int preIdx, postIdx;


    log_info("testing __global\n");
    doReplace(tmp2, (size_t)2048, patterns[3],
              ".SRC_SCOPE.",  "__global",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_struct_arr",
                                       BUFFER_SIZE, 0, 0, 0, 0);
            if(result != 0) {
                return result;
            }
        }
    }
    return 0;
}

int test_vec_align_packed_struct_arr(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    char tmp1[2048], tmp2[2048];
    int result = 0;
    int preIdx, postIdx;


    log_info("Testing __global\n");
    doReplace(tmp2, (size_t)2048, patterns[4],
              ".SRC_SCOPE.",  "__global",
              ".DST_SCOPE.", "__global"); //

    for(preIdx = 0; pre_substitution_arr[preIdx] != NULL; ++preIdx) {
        for(postIdx = 0; post_substitution_arr[postIdx] != NULL; ++postIdx) {
            doReplace(tmp1, (size_t)2048, tmp2,
                      ".PRE.",  pre_substitution_arr[preIdx],
                      ".POST.",  post_substitution_arr[postIdx]);

            result = test_vec_internal(deviceID, context, queue, tmp1,
                                       "test_vec_align_packed_struct_arr",
                                       BUFFER_SIZE, pre_align_arr[preIdx],
                                       type_multiple_pre_align_arr[preIdx],
                                       post_align_arr[postIdx],
                                       type_multiple_post_align_arr[postIdx]);
            if(result != 0)
                return result;
        }
    }
    return 0;
}

