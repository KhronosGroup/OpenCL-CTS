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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/ThreadPool.h"

#define NUM_TESTS 23

#define  LONG_MATH_SHIFT_SIZE 26
#define QUICK_MATH_SHIFT_SIZE 16

static const char *kernel_code =
"__kernel void test(__global %s%s *srcA, __global %s%s *srcB, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] %s srcB[tid];\n"
"}\n";

static const char *kernel_code_V3 =
"__kernel void test(__global %s /*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( vload3( tid, srcA ) %s vload3( tid, srcB), tid, dst );\n"
"}\n";

static const char *kernel_code_V3_scalar_vector =
"__kernel void test(__global %s /*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( srcA[tid] %s vload3( tid, srcB), tid, dst );\n"
"}\n";

static const char *kernel_code_V3_vector_scalar =
"__kernel void test(__global %s /*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( vload3( tid, srcA ) %s srcB[tid], tid, dst );\n"
"}\n";


// Separate kernel here because it does not fit the pattern
static const char *not_kernel_code =
"__kernel void test(__global %s%s *srcA, __global %s%s *srcB, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = %ssrcA[tid];\n"
"}\n";

static const char *not_kernel_code_V3 =
"__kernel void test(__global %s /*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( %s vload3( tid, srcA ), tid, dst );\n"
"}\n";

static const char *kernel_code_scalar_shift =
"__kernel void test(__global %s%s *srcA, __global %s%s *srcB, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] %s srcB[tid]%s;\n"
"}\n";

static const char *kernel_code_scalar_shift_V3 =
"__kernel void test(__global %s/*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( vload3( tid, srcA) %s vload3( tid, srcB )%s, tid, dst );\n"
"}\n";

static const char *kernel_code_question_colon =
"__kernel void test(__global %s%s *srcA, __global %s%s *srcB, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (srcA[tid]%s < srcB[tid]%s) ? srcA[tid] : srcB[tid];\n"
"}\n";

static const char *kernel_code_question_colon_V3 =
"__kernel void test(__global %s/*%s*/ *srcA, __global %s/*%s*/ *srcB, __global %s/*%s*/ *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3( (vload3( tid, srcA)%s < vload3(tid, srcB)%s) ? vload3( tid, srcA) : vload3( tid, srcB), tid, dst );\n"
"}\n";




// External verification and data generation functions
extern const char *tests[];
extern const char *test_names[];
extern int verify_long(int test, size_t vector_size, cl_long *inptrA, cl_long *inptrB, cl_long *outptr, size_t n);
extern void init_long_data(uint64_t indx, int num_elements, cl_long *input_ptr[], MTdata d) ;
extern int verify_ulong(int test, size_t vector_size, cl_ulong *inptrA, cl_ulong *inptrB, cl_ulong *outptr, size_t n);
extern void init_ulong_data(uint64_t indx, int num_elements, cl_ulong *input_ptr[], MTdata d) ;
extern int verify_int(int test, size_t vector_size, cl_int *inptrA, cl_int *inptrB, cl_int *outptr, size_t n);
extern void init_int_data(uint64_t indx, int num_elements, cl_int *input_ptr[], MTdata d) ;
extern int verify_uint(int test, size_t vector_size, cl_uint *inptrA, cl_uint *inptrB, cl_uint *outptr, size_t n);
extern void init_uint_data(uint64_t indx, int num_elements, cl_uint *input_ptr[], MTdata d) ;
extern int verify_short(int test, size_t vector_size, cl_short *inptrA, cl_short *inptrB, cl_short *outptr, size_t n);
extern void init_short_data(uint64_t indx, int num_elements, cl_short *input_ptr[], MTdata d) ;
extern int verify_ushort(int test, size_t vector_size, cl_ushort *inptrA, cl_ushort *inptrB, cl_ushort *outptr, size_t n);
extern void init_ushort_data(uint64_t indx, int num_elements, cl_ushort *input_ptr[], MTdata d) ;
extern int verify_char(int test, size_t vector_size, cl_char *inptrA, cl_char *inptrB, cl_char *outptr, size_t n);
extern void init_char_data(uint64_t indx, int num_elements, cl_char *input_ptr[], MTdata d) ;
extern int verify_uchar(int test, size_t vector_size, cl_uchar *inptrA, cl_uchar *inptrB, cl_uchar *outptr, size_t n);
extern void init_uchar_data(uint64_t indx, int num_elements, cl_uchar *input_ptr[], MTdata d) ;

// Supported type list
const ExplicitType types[] = {
    kChar,
    kUChar,
    kShort,
    kUShort,
    kInt,
    kUInt,
    kLong,
    kULong,
};

enum TestStyle
{
    kDontCare=0,
    kBothVectors,
    kInputAScalar,
    kInputBScalar,
    kVectorScalarScalar,    // for the ?: operator only; indicates vector ? scalar : scalar.
    kInputCAlsoScalar = 0x80    // Or'ed flag to indicate that the selector for the ?: operator is also scalar
};

typedef struct _perThreadData
{
    cl_mem            m_streams[3];
    cl_int            *m_input_ptr[2], *m_output_ptr;
    size_t                      m_type_size;
    cl_program                m_program[NUM_TESTS];
    cl_kernel                m_kernel[NUM_TESTS];
} perThreadData;


perThreadData * perThreadDataNew()
{
    perThreadData * pThis = (perThreadData *)malloc(sizeof(perThreadData));


    memset(pThis->m_program, 0, sizeof(cl_program)*NUM_TESTS);
    memset(pThis->m_kernel, 0, sizeof(cl_kernel)*NUM_TESTS);

    pThis->m_input_ptr[0] = pThis->m_input_ptr[1] = NULL;
    pThis->m_output_ptr = NULL;

    return pThis;
}


void perThreadDataDestroy(perThreadData * pThis)
{
    int                i;
    // cleanup
    clReleaseMemObject(pThis->m_streams[0]);
    clReleaseMemObject(pThis->m_streams[1]);
    clReleaseMemObject(pThis->m_streams[2]);
    for (i=0; i<NUM_TESTS; i++)
    {
        if (pThis->m_kernel[i] != NULL) clReleaseKernel(pThis->m_kernel[i]);
        if (pThis->m_program[i] != NULL) clReleaseProgram(pThis->m_program[i]);
    }
    free(pThis->m_input_ptr[0]);
    free(pThis->m_input_ptr[1]);
    free(pThis->m_output_ptr);

    free(pThis);
}


cl_int perThreadDataInit(perThreadData * pThis, ExplicitType type,
                         int num_elements, int vectorSize,
                         int inputAVecSize, int inputBVecSize,
                         cl_context context, int start_test_ID,
                         int end_test_ID, int testID)
{
    int i;
    const char * sizeNames[] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };

    const char *type_name = get_explicit_type_name(type);
    pThis->m_type_size = get_explicit_type_size(type);
    int err;
    // Used for the && and || tests where the vector case returns a signed value
    const char *signed_type_name;
    switch (type) {
        case kChar:
        case kUChar:
            signed_type_name = get_explicit_type_name(kChar);
            break;
        case kShort:
        case kUShort:
            signed_type_name = get_explicit_type_name(kShort);
            break;
        case kInt:
        case kUInt:
            signed_type_name = get_explicit_type_name(kInt);
            break;
        case kLong:
        case kULong:
            signed_type_name = get_explicit_type_name(kLong);
            break;
        default:
            log_error("Invalid type.\n");
            return -1;
            break;
    }

    pThis->m_input_ptr[0] =
    (cl_int*)malloc(pThis->m_type_size * num_elements * vectorSize);
    pThis->m_input_ptr[1] =
    (cl_int*)malloc(pThis->m_type_size * num_elements * vectorSize);
    pThis->m_output_ptr =
    (cl_int*)malloc(pThis->m_type_size * num_elements * vectorSize);
    pThis->m_streams[0] =
    clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), pThis->m_type_size * num_elements * inputAVecSize, NULL, &err);

    test_error(err, "clCreateBuffer failed");

    pThis->m_streams[1] =
    clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), pThis->m_type_size * num_elements * inputBVecSize, NULL, &err );

    test_error(err, "clCreateBuffer failed");

    pThis->m_streams[2] =
    clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), pThis->m_type_size * num_elements * vectorSize, NULL, &err );

    test_error(err, "clCreateBuffer failed");

    const char *vectorString = sizeNames[ vectorSize ];
    const char *inputAVectorString = sizeNames[ inputAVecSize ];
    const char *inputBVectorString = sizeNames[ inputBVecSize ];

    if (testID == -1)
    {
        log_info("\tTesting %s%s (%d bytes)...\n", type_name, vectorString, (int)(pThis->m_type_size*vectorSize));
    }

    char programString[4096];
    const char *ptr;


    const char * kernel_code_base = ( vectorSize != 3 ) ? kernel_code : ( inputAVecSize == 1 ) ? kernel_code_V3_scalar_vector : ( inputBVecSize == 1 ) ? kernel_code_V3_vector_scalar : kernel_code_V3;

    for (i=start_test_ID; i<end_test_ID; i++) {
        switch (i) {
            case 10:
            case 11:
                sprintf(programString, vectorSize == 3 ? kernel_code_scalar_shift_V3 : kernel_code_scalar_shift, type_name, inputAVectorString, type_name, inputBVectorString,
                        type_name, vectorString, tests[i], ((vectorSize == 1) ? "":".s0"));
                break;
            case 12:
                sprintf(programString, vectorSize == 3 ? not_kernel_code_V3 : not_kernel_code, type_name, inputAVectorString, type_name, inputBVectorString,
                        type_name, vectorString, tests[i]);
                break;
            case 13:
                sprintf(programString, vectorSize == 3 ? kernel_code_question_colon_V3 : kernel_code_question_colon,
                        type_name, inputAVectorString, type_name, inputBVectorString,
                        type_name, vectorString, ((vectorSize == 1) ? "":".s0"), ((vectorSize == 1) ? "":".s0")) ;
                break;
            case 14:
            case 15:
            case 16:
            case 17:
            case 18:
            case 19:
            case 20:
            case 21:
                // Need an unsigned result here for vector sizes > 1
                sprintf(programString, kernel_code_base, type_name, inputAVectorString, type_name, inputBVectorString,
                        ((vectorSize == 1) ? type_name : signed_type_name), vectorString, tests[i]);
                break;
            case 22:
                // Need an unsigned result here for vector sizes > 1
                sprintf(programString, vectorSize == 3 ? not_kernel_code_V3 : not_kernel_code, type_name, inputAVectorString, type_name, inputBVectorString,
                        ((vectorSize == 1) ? type_name : signed_type_name), vectorString, tests[i]);
                break;
            default:
                sprintf(programString, kernel_code_base, type_name, inputAVectorString, type_name, inputBVectorString,
                        type_name, vectorString, tests[i]);
                break;
        }

        //printf("kernel: %s\n", programString);
        ptr = programString;
        err = create_single_kernel_helper( context,
                                          &(pThis->m_program[ i ]),
                                          &(pThis->m_kernel[ i ]), 1,
                                          &ptr, "test" );
        test_error( err, "Unable to create test kernel" );
        err = clSetKernelArg(pThis->m_kernel[i], 0,
                             sizeof pThis->m_streams[0],
                             &(pThis->m_streams[0]) );
        err |= clSetKernelArg(pThis->m_kernel[i], 1,
                              sizeof pThis->m_streams[1],
                              &(pThis->m_streams[1]) );
        err |= clSetKernelArg(pThis->m_kernel[i], 2,
                              sizeof pThis->m_streams[2],
                              &(pThis->m_streams[2]) );
        test_error(err, "clSetKernelArgs failed");
    }

    return CL_SUCCESS;
}

typedef struct _globalThreadData
{
    cl_device_id     m_deviceID;
    cl_context       m_context;
    // cl_command_queue m_queue;
    int              m_num_elements;
    int              m_threadcount;
    int              m_vectorSize;
    int              m_num_runs_shift;
    TestStyle        m_style;
    ExplicitType     m_type;
    MTdata *         m_pRandData;
    uint64_t         m_offset;
    int              m_testID;
    perThreadData  **m_arrPerThreadData;
} globalThreadData;



globalThreadData * globalThreadDataNew(cl_device_id deviceID, cl_context context,
                                       cl_command_queue queue, int num_elements,
                                       int vectorSize, TestStyle style, int num_runs_shift,
                                       ExplicitType type, int testID,
                                       int threadcount)
{
    int i;
    globalThreadData * pThis = (globalThreadData *)malloc(sizeof(globalThreadData));
    pThis->m_deviceID = deviceID;
    pThis->m_context = context;
    // pThis->m_queue = queue;
    pThis->m_num_elements = num_elements;
    pThis->m_num_runs_shift = num_runs_shift;
    pThis->m_vectorSize = vectorSize;
    pThis->m_style = style;
    pThis->m_type = type;
    pThis->m_offset = (uint64_t)0;
    pThis->m_testID = testID;
    pThis->m_arrPerThreadData = NULL;
    pThis->m_threadcount = threadcount;

    pThis->m_pRandData = (MTdata *)malloc(threadcount*sizeof(MTdata));
    pThis->m_arrPerThreadData = (perThreadData **)
    malloc(threadcount*sizeof(perThreadData *));
    for(i=0; i < threadcount; ++i)
    {
        pThis->m_pRandData[i] = init_genrand(i+1);
        pThis->m_arrPerThreadData[i] = NULL;
    }

    return pThis;
}

void globalThreadDataDestroy(globalThreadData * pThis)
{
    int i;

    for(i=0; i < pThis->m_threadcount; ++i)
    {
        free_mtdata(pThis->m_pRandData[i]);
        if(pThis->m_arrPerThreadData[i] != NULL)
        {
            perThreadDataDestroy(pThis->m_arrPerThreadData[i]);
        }
    }
    free(pThis->m_arrPerThreadData);
    free(pThis->m_pRandData);
    free(pThis);
}

int
test_integer_ops(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, int vectorSize, TestStyle style, int num_runs_shift, ExplicitType type, int testID, MTdata randIn, uint64_t startIndx, uint64_t endIndx,
                 perThreadData ** ppThreadData);


cl_int test_integer_ops_do_thread( cl_uint job_id, cl_uint thread_id, void *userInfo )
{
    cl_int error; cl_int result;
    globalThreadData * threadInfoGlobal = (globalThreadData *)userInfo;
    cl_command_queue queue;

#if THREAD_DEBUG
    log_error("Thread %x (job %x) about to create command queue\n",
              thread_id, job_id);
#endif

    queue =  clCreateCommandQueue (threadInfoGlobal->m_context,
                                   threadInfoGlobal->m_deviceID,0,
                                   &error);

    if(error != CL_SUCCESS)
    {
        log_error("Thread %x (job %x) could not create command queue\n",
                  thread_id, job_id);
        return error; // should we clean up the queue too?
    }

#if THREAD_DEBUG
    log_error("Thread %x (job %x) created command queue\n",
              thread_id, job_id);
#endif

    result = test_integer_ops(  threadInfoGlobal->m_deviceID,
                              threadInfoGlobal->m_context,
                              queue,
                              threadInfoGlobal->m_num_elements,
                              threadInfoGlobal->m_vectorSize, threadInfoGlobal->m_style,
                              threadInfoGlobal->m_num_runs_shift,
                              threadInfoGlobal->m_type, threadInfoGlobal->m_testID,
                              threadInfoGlobal->m_pRandData[thread_id],
                              threadInfoGlobal->m_offset + threadInfoGlobal->m_num_elements*job_id,
                              threadInfoGlobal->m_offset + threadInfoGlobal->m_num_elements*(job_id+1),
                              &(threadInfoGlobal->m_arrPerThreadData[thread_id])
                              );

    if(result != 0)
    {
        log_error("Thread %x (job %x) failed test_integer_ops with result %x\n",
                  thread_id, job_id, result);
        // return error;
    }


    error = clReleaseCommandQueue(queue);
    if(error != CL_SUCCESS)
    {
        log_error("Thread %x (job %x) could not release command queue\n",
                  thread_id, job_id);
        return error;
    }
    return result;
}

int
test_integer_ops_threaded(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, int vectorSize, TestStyle style, int num_runs_shift, ExplicitType type, int testID)
{
    globalThreadData * pThreadInfo = NULL;
    cl_int result=0;
    cl_uint threadcount = GetThreadCount();

  // Check to see if we are using single threaded mode on other than a 1.0 device
  if (getenv( "CL_TEST_SINGLE_THREADED" )) {

    char device_version[1024] = { 0 };
    result = clGetDeviceInfo( deviceID, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL );
    if(result != CL_SUCCESS)
    {
      log_error("clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed: %d\n", result);
      return result;
    }

    if (strcmp("OpenCL 1.0 ",device_version)) {
      log_error("ERROR: CL_TEST_SINGLE_THREADED is set in the environment. Running single threaded.\n");
    }
  }

    // This test will run threadcount threads concurrently; each thread will execute test_integer_ops()
    // which will allocate 2 OpenCL buffers on the device; each buffer has size num_elements * type_size * vectorSize.
    // We need to make sure that the total device memory allocated by all threads does not exceed the maximum
    // memory on the device. If it does, we decrease num_elements until all threads combined will not
    // over-subscribe device memory.
    cl_ulong maxDeviceGlobalMem;
    result = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxDeviceGlobalMem), &maxDeviceGlobalMem, NULL);
    if(result != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE) failed: %d\n", result);
        return result;
    }

  if (maxDeviceGlobalMem > (cl_ulong)SIZE_MAX) {
    maxDeviceGlobalMem = (cl_ulong)SIZE_MAX;
  }

    // Let's not take all device memory - reduce by 75%
    maxDeviceGlobalMem = (maxDeviceGlobalMem * 3) >> 2;
    // Now reduce num_elements so that the total device memory usage does not exceed 75% of global device memory.
    size_t type_size = get_explicit_type_size(type);
    while ((cl_ulong)threadcount * 4 * num_elements * type_size * vectorSize > maxDeviceGlobalMem)
    {
        num_elements >>= 1;
    }

    uint64_t startIndx = (uint64_t)0;
    uint64_t endIndx = (1ULL<<num_runs_shift);
    uint64_t jobcount = (endIndx-startIndx)/num_elements;

    if(jobcount==0)
    {
        jobcount = 1;
    }

    pThreadInfo = globalThreadDataNew(deviceID, context, queue, num_elements,
                                      vectorSize, style, num_runs_shift,
                                      type, testID, threadcount);


    pThreadInfo->m_offset = startIndx;

#if THREAD_DEBUG
    log_error("Launching %llx jobs\n",
              jobcount);
#endif

    result = ThreadPool_Do(test_integer_ops_do_thread, (cl_uint)jobcount, (void *)pThreadInfo);

    if(result != 0)
    {
        // cleanup ??
        log_error("ThreadPool_Do return non-success value %d\n", result);

    }
    globalThreadDataDestroy(pThreadInfo);
    return result;
}



int
test_integer_ops(cl_device_id deviceID, cl_context context,
                 cl_command_queue queue, int num_elements,
                 int vectorSize, TestStyle style, int num_runs_shift,
                 ExplicitType type, int testID, MTdata randDataIn,
                 uint64_t startIndx, uint64_t endIndx,
                 perThreadData ** ppThreadData)
{
    size_t    threads[1];
    int                err;
    int                i;
    int inputAVecSize, inputBVecSize;



    inputAVecSize = inputBVecSize = vectorSize;
    if( style == kInputAScalar )
        inputAVecSize = 1;
    else if( style == kInputBScalar )
        inputBVecSize = 1;

    /*
     if( inputAVecSize != inputBVecSize )
     log_info("Testing \"%s\" on %s%d (%s-%s inputs) (range %llx - %llx of 0-%llx)\n",
     test_names[testID],
     get_explicit_type_name(type), vectorSize,
     ( inputAVecSize == 1 ) ? "scalar" : "vector",
     ( inputBVecSize == 1 ) ? "scalar" : "vector",
     startIndx, endIndx, (1ULL<<num_runs_shift) );
     else
     log_info("Testing \"%s\" on %s%d (range %llx - %llx of 0-%llx)\n",
     test_names[testID],
     get_explicit_type_name(type), vectorSize,
     startIndx, endIndx, (1ULL<<num_runs_shift));
     */


    // Figure out which sub-test to run, or all of them
    int start_test_ID = 0;
    int end_test_ID = NUM_TESTS;
    if (testID != -1) {
        start_test_ID = testID;
        end_test_ID = testID+1;
    }
    if (testID > NUM_TESTS) {
        log_error("Invalid test ID: %d\n", testID);
        return -1;
    }

    if(*ppThreadData == NULL)
    {
        *ppThreadData = perThreadDataNew();
        err = perThreadDataInit(*ppThreadData,
                                type, num_elements, vectorSize,
                                inputAVecSize, inputBVecSize,
                                context, start_test_ID,
                                end_test_ID, testID);
        test_error(err, "failed to init per thread data\n");
    }

    perThreadData * pThreadData = *ppThreadData;



    threads[0] = (size_t)num_elements;
    int error_count = 0;
    for (i=start_test_ID; i<end_test_ID; i++)
    {
        uint64_t    indx;


        if(startIndx >= endIndx)
        {
            startIndx = (uint64_t)0;
            endIndx = (1ULL<<num_runs_shift);
        }
        for (indx=startIndx; indx < endIndx; indx+=num_elements)
        {

            switch (type) {
                case     kChar:
                    init_char_data(indx, num_elements * vectorSize, (cl_char**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kUChar:
                    init_uchar_data(indx, num_elements * vectorSize, (cl_uchar**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kShort:
                    init_short_data(indx, num_elements * vectorSize, (cl_short**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kUShort:
                    init_ushort_data(indx, num_elements * vectorSize, (cl_ushort**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kInt:
                    init_int_data(indx, num_elements * vectorSize, (cl_int**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kUInt:
                    init_uint_data(indx, num_elements * vectorSize, (cl_uint**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kLong:
                    init_long_data(indx, num_elements * vectorSize, (cl_long**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                case     kULong:
                    init_ulong_data(indx, num_elements * vectorSize, (cl_ulong**)(pThreadData->m_input_ptr), randDataIn);
                    break;
                default:
                    err = 1;
                    log_error("Invalid type.\n");
                    break;
            }


            err = clEnqueueWriteBuffer(queue, pThreadData->m_streams[0], CL_FALSE, 0, pThreadData->m_type_size*num_elements * inputAVecSize, (void *)pThreadData->m_input_ptr[0], 0, NULL, NULL);
            test_error(err, "clEnqueueWriteBuffer failed");
            err = clEnqueueWriteBuffer( queue, pThreadData->m_streams[1], CL_FALSE, 0, pThreadData->m_type_size*num_elements * inputBVecSize, (void *)pThreadData->m_input_ptr[1], 0, NULL, NULL );
            test_error(err, "clEnqueueWriteBuffer failed");

            err = clEnqueueNDRangeKernel( queue, pThreadData->m_kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
            test_error(err, "clEnqueueNDRangeKernel failed");

            err = clEnqueueReadBuffer( queue, pThreadData->m_streams[2], CL_TRUE, 0, pThreadData->m_type_size*num_elements * vectorSize, (void *)pThreadData->m_output_ptr, 0, NULL, NULL );
            test_error(err, "clEnqueueReadBuffer failed");

            // log_info("Performing verification\n");

            // If one of the inputs are scalar, we need to extend the input values to vectors
            // to accommodate the verify functions
            if( vectorSize > 1 )
            {
                char * p = NULL;
                if( style == kInputAScalar )
                    p = (char *)pThreadData->m_input_ptr[ 0 ];
                else if( style == kInputBScalar )
                    p = (char *)pThreadData->m_input_ptr[ 1 ];
                if( p != NULL )
                {
                    for( int element = num_elements - 1; element >= 0; element-- )
                    {
                        for( int vec = ( element == 0 ) ? 1 : 0; vec < vectorSize; vec++ )
                            memcpy( p + ( element * vectorSize + vec ) * pThreadData->m_type_size, p + element * pThreadData->m_type_size, pThreadData->m_type_size );
                    }
                }
            }

            switch (type) {
                case     kChar:
                    err = verify_char(i, vectorSize, (cl_char*)pThreadData->m_input_ptr[0], (cl_char*)pThreadData->m_input_ptr[1], (cl_char*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kUChar:
                    err = verify_uchar(i, vectorSize, (cl_uchar*)pThreadData->m_input_ptr[0], (cl_uchar*)pThreadData->m_input_ptr[1], (cl_uchar*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kShort:
                    err = verify_short(i, vectorSize, (cl_short*)pThreadData->m_input_ptr[0], (cl_short*)pThreadData->m_input_ptr[1], (cl_short*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kUShort:
                    err = verify_ushort(i, vectorSize, (cl_ushort*)pThreadData->m_input_ptr[0], (cl_ushort*)pThreadData->m_input_ptr[1], (cl_ushort*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kInt:
                    err = verify_int(i, vectorSize, (cl_int*)pThreadData->m_input_ptr[0], (cl_int*)pThreadData->m_input_ptr[1], (cl_int*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kUInt:
                    err = verify_uint(i, vectorSize, (cl_uint*)pThreadData->m_input_ptr[0], (cl_uint*)pThreadData->m_input_ptr[1], (cl_uint*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kLong:
                    err = verify_long(i, vectorSize, (cl_long*)pThreadData->m_input_ptr[0], (cl_long*)pThreadData->m_input_ptr[1], (cl_long*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                case     kULong:
                    err = verify_ulong(i, vectorSize, (cl_ulong*)pThreadData->m_input_ptr[0], (cl_ulong*)pThreadData->m_input_ptr[1], (cl_ulong*)pThreadData->m_output_ptr, num_elements * vectorSize);
                    break;
                default:
                    err = 1;
                    log_error("Invalid type.\n");
                    break;
            }

            if (err) {
#if 0
                log_error( "* inASize: %d inBSize: %d numElem: %d\n", inputAVecSize, inputBVecSize, num_elements );
                cl_char *inP = (cl_char *)pThreadData->m_input_ptr[0];
                log_error( "from 18:\n" );
                for( int q = 18; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
                inP = (cl_char *)pThreadData->m_input_ptr[1];
                for( int q = 18; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
                inP = (cl_char *)pThreadData->m_output_ptr;
                for( int q = 18; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
                log_error( "from 36:\n" );
                inP = (cl_char *)pThreadData->m_input_ptr[0];
                for( int q = 36; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
                inP = (cl_char *)pThreadData->m_input_ptr[1];
                for( int q = 36; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
                inP = (cl_char *)pThreadData->m_output_ptr;
                for( int q = 36; q < 64; q++ )
                {
                    log_error( "%02x ", inP[ q ] );
                }
                log_error( "\n" );
#endif
                error_count++;
                break;
            }
        }

        /*

         const char * sizeNames[] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };

         if (err) {
         log_error("\t\t%s%s test %s failed (range %llx - %llx of 0-%llx)\n",
         get_explicit_type_name(type), sizeNames[vectorSize],
         test_names[i],
         startIndx, endIndx,
         (1ULL<<num_runs_shift));
         } else {
         log_info("\t\t%s%s test %s passed (range %llx - %llx of 0-%llx)\n",
         get_explicit_type_name(type), sizeNames[vectorSize],
         test_names[i],
         startIndx, endIndx,
         (1ULL<<num_runs_shift));
         }
         */
    }



    return error_count;
}









// Run all the vector sizes for a given test
int run_specific_test(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num, int testID) {
    int errors = 0;
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/1, 1, kBothVectors, num, type, testID);
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/2, 2, kBothVectors, num, type, testID);
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/3, 3, kBothVectors, num, type, testID);
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/4, 4, kBothVectors, num, type, testID);
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/8, 8, kBothVectors, num, type, testID);
    errors += test_integer_ops_threaded(deviceID, context, queue, (1024*1024*2)/16, 16, kBothVectors, num, type, testID);
    return errors;
}

// Run multiple tests for a given type
int run_multiple_tests(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num, int *tests, int total_tests) {
    int errors = 0;

    if (getenv("CL_WIMPY_MODE") && num == LONG_MATH_SHIFT_SIZE) {
      log_info("Detected CL_WIMPY_MODE env\n");
      log_info("Skipping long test\n");
      return 0;
    }

    int i;
    for (i=0; i<total_tests; i++)
    {
        int localErrors;
        log_info("Testing \"%s\" ", test_names[tests[i]]);  fflush( stdout );
        localErrors = run_specific_test(deviceID, context, queue, num_elements, type, num, tests[i]);
        if( localErrors )
            log_info( "FAILED\n" );
        else
            log_info( "passed\n" );

        errors += localErrors;
    }

    return errors;
}

// Run all the math tests for a given type
int run_test_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num) {
    int tests[] = {0, 1, 2, 3, 4};
    return run_multiple_tests(deviceID, context, queue, num_elements, type, num, tests, (int)(sizeof(tests)/sizeof(int)));
}

// Run all the logic tests for a given type
int run_test_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num) {
    int tests[] = {5, 6, 7, 12, 14, 15, 22};
    return run_multiple_tests(deviceID, context, queue, num_elements, type, num, tests, (int)(sizeof(tests)/sizeof(int)));
}

// Run all the shifting tests for a given type
int run_test_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num) {
    int tests[] = {8, 9, 10, 11};
    return run_multiple_tests(deviceID, context, queue, num_elements, type, num, tests, (int)(sizeof(tests)/sizeof(int)));
}

// Run all the comparison tests for a given type
int run_test_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num) {
    int tests[] = {13, 16, 17, 18, 19, 20, 21};
    return run_multiple_tests(deviceID, context, queue, num_elements, type, num, tests, (int)(sizeof(tests)/sizeof(int)));
}

// Run all tests for a given type
int run_test(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num) {
    int errors = 0;
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 1, kBothVectors, num, type, -1);
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 2, kBothVectors, num, type, -1);
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 3, kBothVectors, num, type, -1);
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 4, kBothVectors, num, type, -1);
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 8, kBothVectors, num, type, -1);
    errors += test_integer_ops_threaded(deviceID, context, queue, 1024*1024*2, 16, kBothVectors, num, type, -1);
    return errors;
}


// -----------------
// Long tests
// -----------------
int test_long_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_math(deviceID, context, queue, num_elements, kLong, LONG_MATH_SHIFT_SIZE);
}
int test_long_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_logic(deviceID, context, queue, num_elements, kLong, LONG_MATH_SHIFT_SIZE);
}
int test_long_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_shift(deviceID, context, queue, num_elements, kLong, LONG_MATH_SHIFT_SIZE);
}
int test_long_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_compare(deviceID, context, queue, num_elements, kLong, LONG_MATH_SHIFT_SIZE);
}
int test_quick_long_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_math(deviceID, context, queue, num_elements, kLong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_long_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_logic(deviceID, context, queue, num_elements, kLong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_long_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_shift(deviceID, context, queue, num_elements, kLong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_long_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_compare(deviceID, context, queue, num_elements, kLong, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// ULong tests
// -----------------
int test_ulong_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_math(deviceID, context, queue, num_elements, kULong, LONG_MATH_SHIFT_SIZE);
}
int test_ulong_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_logic(deviceID, context, queue, num_elements, kULong, LONG_MATH_SHIFT_SIZE);
}
int test_ulong_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_shift(deviceID, context, queue, num_elements, kULong, LONG_MATH_SHIFT_SIZE);
}
int test_ulong_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_compare(deviceID, context, queue, num_elements, kULong, LONG_MATH_SHIFT_SIZE);
}
int test_quick_ulong_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_math(deviceID, context, queue, num_elements, kULong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ulong_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_logic(deviceID, context, queue, num_elements, kULong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ulong_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_shift(deviceID, context, queue, num_elements, kULong, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ulong_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test_compare(deviceID, context, queue, num_elements, kULong, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// Int tests
// -----------------
int test_int_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kInt, LONG_MATH_SHIFT_SIZE);
}
int test_int_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kInt, LONG_MATH_SHIFT_SIZE);
}
int test_int_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kInt, LONG_MATH_SHIFT_SIZE);
}
int test_int_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kInt, LONG_MATH_SHIFT_SIZE);
}
int test_quick_int_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_int_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_int_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_int_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kInt, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// UInt tests
// -----------------
int test_uint_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUInt, LONG_MATH_SHIFT_SIZE);
}
int test_uint_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUInt, LONG_MATH_SHIFT_SIZE);
}
int test_uint_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUInt, LONG_MATH_SHIFT_SIZE);
}
int test_uint_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUInt, LONG_MATH_SHIFT_SIZE);
}
int test_quick_uint_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uint_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uint_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUInt, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uint_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUInt, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// Short tests
// -----------------
int test_short_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kShort, LONG_MATH_SHIFT_SIZE);
}
int test_short_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kShort, LONG_MATH_SHIFT_SIZE);
}
int test_short_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kShort, LONG_MATH_SHIFT_SIZE);
}
int test_short_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kShort, LONG_MATH_SHIFT_SIZE);
}
int test_quick_short_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_short_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_short_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_short_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kShort, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// UShort tests
// -----------------
int test_ushort_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUShort, LONG_MATH_SHIFT_SIZE);
}
int test_ushort_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUShort, LONG_MATH_SHIFT_SIZE);
}
int test_ushort_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUShort, LONG_MATH_SHIFT_SIZE);
}
int test_ushort_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUShort, LONG_MATH_SHIFT_SIZE);
}
int test_quick_ushort_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ushort_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ushort_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUShort, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_ushort_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUShort, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// Char tests
// -----------------
int test_char_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kChar, LONG_MATH_SHIFT_SIZE);
}
int test_char_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kChar, LONG_MATH_SHIFT_SIZE);
}
int test_char_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kChar, LONG_MATH_SHIFT_SIZE);
}
int test_char_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kChar, LONG_MATH_SHIFT_SIZE);
}
int test_quick_char_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_char_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_char_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_char_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kChar, QUICK_MATH_SHIFT_SIZE);
}


// -----------------
// UChar tests
// -----------------
int test_uchar_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUChar, LONG_MATH_SHIFT_SIZE);
}
int test_uchar_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUChar, LONG_MATH_SHIFT_SIZE);
}
int test_uchar_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUChar, LONG_MATH_SHIFT_SIZE);
}
int test_uchar_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUChar, LONG_MATH_SHIFT_SIZE);
}
int test_quick_uchar_math(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_math(deviceID, context, queue, num_elements, kUChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uchar_logic(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_logic(deviceID, context, queue, num_elements, kUChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uchar_shift(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_shift(deviceID, context, queue, num_elements, kUChar, QUICK_MATH_SHIFT_SIZE);
}
int test_quick_uchar_compare(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test_compare(deviceID, context, queue, num_elements, kUChar, QUICK_MATH_SHIFT_SIZE);
}



// These are kept for debugging if you want to run all the tests together.

int test_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test(deviceID, context, queue, num_elements, kLong, LONG_MATH_SHIFT_SIZE);
}

int test_quick_long(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test(deviceID, context, queue, num_elements, kLong, QUICK_MATH_SHIFT_SIZE);
}

int test_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test(deviceID, context, queue, num_elements, kULong, LONG_MATH_SHIFT_SIZE);
}

int test_quick_ulong(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    if (!gHasLong)
    {
        log_info( "WARNING: 64 bit integers are not supported on this device. Skipping\n" );
        return CL_SUCCESS;
    }
    return run_test(deviceID, context, queue, num_elements, kULong, QUICK_MATH_SHIFT_SIZE);
}

int test_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kInt, LONG_MATH_SHIFT_SIZE);
}

int test_quick_int(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kInt, QUICK_MATH_SHIFT_SIZE);
}

int test_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUInt, LONG_MATH_SHIFT_SIZE);
}

int test_quick_uint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUInt, QUICK_MATH_SHIFT_SIZE);
}

int test_short(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kShort, LONG_MATH_SHIFT_SIZE);
}

int test_quick_short(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kShort, QUICK_MATH_SHIFT_SIZE);
}

int test_ushort(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUShort, LONG_MATH_SHIFT_SIZE);
}

int test_quick_ushort(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUShort, QUICK_MATH_SHIFT_SIZE);
}

int test_char(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kChar, LONG_MATH_SHIFT_SIZE);
}

int test_quick_char(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kChar, QUICK_MATH_SHIFT_SIZE);
}

int test_uchar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUChar, LONG_MATH_SHIFT_SIZE);
}

int test_quick_uchar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements) {
    return run_test(deviceID, context, queue, num_elements, kUChar, QUICK_MATH_SHIFT_SIZE);
}

// Prototype for below
int test_question_colon_op(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements,
                           int vectorSize, TestStyle style, ExplicitType type );

// Run all the vector sizes for a given test in scalar-vector and vector-scalar modes
int run_test_sizes(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num, int testID)
{
    int sizes[] = { 2, 3, 4, 8, 16, 0 };
    int errors = 0;

    for( int i = 0; sizes[ i ] != 0; i++ )
    {
        if( testID == 13 )
        {
            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], kInputAScalar, type );
            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], kInputBScalar, type );
            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], kVectorScalarScalar, type );

            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], (TestStyle)(kBothVectors | kInputCAlsoScalar), type );
            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], (TestStyle)(kInputAScalar | kInputCAlsoScalar), type );
            errors += test_question_colon_op( deviceID, context, queue, num_elements / sizes[i], sizes[i], (TestStyle)(kInputBScalar | kInputCAlsoScalar), type );
        }
        else
        {
            errors += test_integer_ops_threaded(deviceID, context, queue, num_elements / sizes[i], sizes[i], kInputAScalar, num, type, testID);
            errors += test_integer_ops_threaded(deviceID, context, queue, num_elements / sizes[i], sizes[i], kInputBScalar, num, type, testID);
        }
    }
    return errors;
}

// Run all the tests for scalar-vector and vector-scalar for a given type
int run_vector_scalar_tests( cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements, ExplicitType type, int num )
{
    int errors = 0;
    size_t i;

    // Shift operators:
    // a) cannot take scalars as first parameter and vectors as second
    // b) have the vector >> scalar case tested by tests 10 and 11
    // so they get skipped entirely

    int testsToRun[] = { 0, 1, 2, 3, 4, 5, 6, 7,
        13, 14, 15, 16, 17, 18, 19, 20, 21 };
    for (i=0; i< sizeof(testsToRun)/sizeof(testsToRun[0]); i++)
    {
        errors += run_test_sizes(deviceID, context, queue, 2048, type, num, testsToRun[i]);
    }
    return errors;
}

int test_vector_scalar(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int errors = 0;
    int numTypes = sizeof( types ) / sizeof( types[ 0 ] );

    for( int t = 0; t < numTypes; t++ )
    {
        if ((types[ t ] == kLong || types[ t ] == kULong) && !gHasLong)
            continue;

        errors += run_vector_scalar_tests( deviceID, context, queue, num_elements, types[ t ], 1 );
        break;
    }

    return errors;
}

void generate_random_bool_data( size_t count, MTdata d, cl_char *outData, size_t outDataSize )
{
    cl_uint bits = genrand_int32(d);
    cl_uint bitsLeft = 32;

    memset( outData, 0, outDataSize * count );

    for( size_t i = 0; i < count; i++ )
    {
        if( 0 == bitsLeft)
        {
            bits = genrand_int32(d);
            bitsLeft = 32;
        }

        // Note: we will be setting just any bit non-zero for the type, so we can easily skip past
        // and just write bytes (assuming the entire output buffer is already zeroed, which we did)
        *outData = ( bits & 1 ) ? 0xff : 0;

        bits >>= 1; bitsLeft -= 1;

        outData += outDataSize;
    }
}

static const char *kernel_question_colon_full =
"__kernel void test(__global %s%s *srcA, __global %s%s *srcB, __global %s%s *srcC, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    %s%s valA = %ssrcA%s"
"    %s%s valB = %ssrcB%s"
"    %s%s valC = %ssrcC%s"
"    %s%s destVal = valC ? valA : valB;\n"
"    %s"
"}\n";

static const char *kernel_qc_load_plain_prefix = "";
static const char *kernel_qc_load_plain_suffix = "[ tid ];\n";

static const char *kernel_qc_load_vec3_prefix = "vload3( tid, ";
static const char *kernel_qc_load_vec3_suffix = ");\n";

static const char *kernel_qc_store_plain = "dst[ tid ] = destVal;\n";
static const char *kernel_qc_store_vec3 = "vstore3( destVal, tid, dst );\n";

int test_question_colon_op(cl_device_id deviceID, cl_context context,
                           cl_command_queue queue, int num_elements,
                           int vectorSize, TestStyle style, ExplicitType type )
{
    cl_mem              streams[4];
    cl_int              *input_ptr[3], *output_ptr;
    cl_program          program;
    cl_kernel           kernel;
    size_t              threads[1];
    int                 err;
    int inputAVecSize, inputBVecSize, inputCVecSize;
    const char * sizeNames[] = { "", "", "2", "3", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    // Identical to sizeNames but with a blank for 3, since we use vload/store there
    const char * paramSizeNames[] = { "", "", "2", "", "4", "", "", "", "8", "", "", "", "", "", "", "", "16" };
    MTdata s_randStates;

    inputAVecSize = inputBVecSize = inputCVecSize = vectorSize;
    if( style & kInputCAlsoScalar )
    {
        style = (TestStyle)( style & ~kInputCAlsoScalar );
        inputCVecSize = 1;
    }
    if( style == kInputAScalar )
        inputAVecSize = 1;
    else if( style == kInputBScalar )
        inputBVecSize = 1;
    else if( style == kVectorScalarScalar )
        inputAVecSize = inputBVecSize = 1;

    log_info("Testing \"?:\" on %s%d (%s?%s:%s inputs)\n",
             get_explicit_type_name(type), vectorSize, ( inputCVecSize == 1 ) ? "scalar" : "vector",
             ( inputAVecSize == 1 ) ? "scalar" : "vector",
             ( inputBVecSize == 1 ) ? "scalar" : "vector" );


    const char *type_name = get_explicit_type_name(type);
    size_t type_size = get_explicit_type_size(type);

    // Create and initialize I/O buffers

    input_ptr[0] = (cl_int*)malloc(type_size * num_elements * vectorSize);
    input_ptr[1] = (cl_int*)malloc(type_size * num_elements * vectorSize);
    input_ptr[2] = (cl_int*)malloc(type_size * num_elements * vectorSize);
    output_ptr = (cl_int*)malloc(type_size * num_elements * vectorSize);

    s_randStates = init_genrand( gRandomSeed );

    generate_random_data( type, num_elements * inputAVecSize, s_randStates, input_ptr[ 0 ] );
    generate_random_data( type, num_elements * inputBVecSize, s_randStates, input_ptr[ 1 ] );
    generate_random_bool_data( num_elements * inputCVecSize, s_randStates, (cl_char *)input_ptr[ 2 ], type_size );

    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), type_size * num_elements * inputAVecSize, input_ptr[0], &err);
    test_error(err, "clCreateBuffer failed");
    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), type_size * num_elements * inputBVecSize, input_ptr[1], &err );
    test_error(err, "clCreateBuffer failed");
    streams[2] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), type_size * num_elements * inputCVecSize, input_ptr[2], &err );
    test_error(err, "clCreateBuffer failed");
    streams[3] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_WRITE_ONLY), type_size * num_elements * vectorSize, NULL, &err );
    test_error(err, "clCreateBuffer failed");

    const char *vectorString = sizeNames[ vectorSize ];
    const char *inputAVectorString = sizeNames[ inputAVecSize ];
    const char *inputBVectorString = sizeNames[ inputBVecSize ];
    const char *inputCVectorString = sizeNames[ inputCVecSize ];

    char programString[4096];
    const char *ptr;

    sprintf( programString, kernel_question_colon_full, type_name, paramSizeNames[ inputAVecSize ],
            type_name, paramSizeNames[ inputBVecSize ],
            type_name, paramSizeNames[ inputCVecSize ],
         type_name, paramSizeNames[ vectorSize ],
            // Loads
            type_name, inputAVectorString, ( inputAVecSize == 3 ) ? kernel_qc_load_vec3_prefix : kernel_qc_load_plain_prefix, ( inputAVecSize == 3 ) ? kernel_qc_load_vec3_suffix : kernel_qc_load_plain_suffix,
            type_name, inputBVectorString, ( inputBVecSize == 3 ) ? kernel_qc_load_vec3_prefix : kernel_qc_load_plain_prefix, ( inputBVecSize == 3 ) ? kernel_qc_load_vec3_suffix : kernel_qc_load_plain_suffix,
            type_name, inputCVectorString, ( inputCVecSize == 3 ) ? kernel_qc_load_vec3_prefix : kernel_qc_load_plain_prefix, ( inputCVecSize == 3 ) ? kernel_qc_load_vec3_suffix : kernel_qc_load_plain_suffix,
            // Dest type
            type_name, vectorString,
            // Store
            ( vectorSize == 3 ) ? kernel_qc_store_vec3 : kernel_qc_store_plain );

    ptr = programString;
    err = create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "test" );
    test_error( err, "Unable to create test kernel" );

    err = clSetKernelArg( kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg( kernel, 1, sizeof streams[1], &streams[1] );
    err |= clSetKernelArg( kernel, 2, sizeof streams[2], &streams[2] );
    err |= clSetKernelArg( kernel, 3, sizeof streams[3], &streams[3] );
    test_error(err, "clSetKernelArgs failed");

    // Run
    threads[0] = (size_t)num_elements;

    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error(err, "clEnqueueNDRangeKernel failed");

    // Read and verify results
    err = clEnqueueReadBuffer( queue, streams[3], CL_TRUE, 0, type_size*num_elements * vectorSize, (void *)output_ptr, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed");

    // log_info("Performing verification\n");
    int error_count = 0;

    char *inputAPtr = (char *)input_ptr[ 0 ];
    char *inputBPtr = (char *)input_ptr[ 1 ];
    cl_char *inputCPtr = (cl_char *)input_ptr[ 2 ];
    char *actualPtr = (char *)output_ptr;

    for( int i = 0; i < num_elements; i++ )
    {
        for( int j = 0; j < vectorSize; j++ )
        {
            char *expectedPtr = ( *inputCPtr ) ? inputAPtr : inputBPtr;
            if( memcmp( expectedPtr, actualPtr, type_size ) != 0 )
            {
#if 0
                char expectedStr[ 128 ], actualStr[ 128 ], inputAStr[ 128 ], inputBStr[ 128 ];
                print_type_to_string( type, inputAPtr, inputAStr );
                print_type_to_string( type, inputBPtr, inputBStr );
                print_type_to_string( type, expectedPtr, expectedStr );
                print_type_to_string( type, actualPtr, actualStr );
                log_error( "cl_%s verification failed at element %d:%d (expected %s, got %s, inputs: %s, %s, %s)\n",
                          type_name, i, j, expectedStr, actualStr, inputAStr, inputBStr, ( *inputCPtr ) ? "true" : "false" );
#endif
                error_count++;
            }
            // Advance for each element member. Note if any of the vec sizes are 1, they don't advance here
            inputAPtr += ( inputAVecSize == 1 ) ? 0 : type_size;
            inputBPtr += ( inputBVecSize == 1 ) ? 0 : type_size;
            inputCPtr += ( inputCVecSize == 1 ) ? 0 : type_size;
            actualPtr += ( vectorSize == 1 ) ? 0 : type_size;
        }
        // Reverse for the member advance. If the vec sizes are 1, we need to advance, but otherwise they're already correct
        inputAPtr += ( inputAVecSize == 1 ) ? type_size : 0;
        inputBPtr += ( inputBVecSize == 1 ) ? type_size : 0;
        inputCPtr += ( inputCVecSize == 1 ) ? type_size : 0;
        actualPtr += ( vectorSize == 1 ) ? type_size : 0;
    }

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseMemObject(streams[3]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(input_ptr[2]);
    free(output_ptr);
    free_mtdata( s_randStates );

    return error_count;
}
