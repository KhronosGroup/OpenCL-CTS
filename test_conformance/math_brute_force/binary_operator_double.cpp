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

#include "function_list.h"
#include "test_functions.h"
#include "utility.h"

#include <cstring>

static int BuildKernel(const char *operator_symbol, int vectorSize,
                       cl_uint kernel_count, cl_kernel *k, cl_program *p,
                       bool relaxedMode)
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global double",
                        sizeNames[vectorSize],
                        "* out, __global double",
                        sizeNames[vectorSize],
                        "* in1, __global double",
                        sizeNames[vectorSize],
                        "* in2 )\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   out[i] = in1[i] ",
                        operator_symbol,
                        " in2[i];\n"
                        "}\n" };

    const char *c3[] = {
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global double* out, __global double* in, __global double* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       double3 d0 = vload3( 0, in + 3 * i );\n"
        "       double3 d1 = vload3( 0, in2 + 3 * i );\n"
        "       d0 = d0 ",
        operator_symbol,
        " d1;\n"
        "       vstore3( d0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       double3 d0;\n"
        "       double3 d1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               d0 = (double3)( in[3*i], NAN, NAN ); \n"
        "               d1 = (double3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               d0 = (double3)( in[3*i], in[3*i+1], NAN ); \n"
        "               d1 = (double3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       d0 = d0 ",
        operator_symbol,
        " d1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = d0.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = d0.x; \n"
        "               break;\n"
        "       }\n"
        "   }\n"
        "}\n"
    };

    const char **kern = c;
    size_t kernSize = sizeof(c) / sizeof(c[0]);

    if (sizeValues[vectorSize] == 3)
    {
        kern = c3;
        kernSize = sizeof(c3) / sizeof(c3[0]);
    }

    char testName[32];
    snprintf(testName, sizeof(testName) - 1, "math_kernel%s",
             sizeNames[vectorSize]);

    return MakeKernels(kern, (cl_uint)kernSize, testName, kernel_count, k, p,
                       relaxedMode);
}

typedef struct BuildKernelInfo
{
    cl_uint offset; // the first vector size to build
    cl_uint kernel_count;
    cl_kernel **kernels;
    cl_program *programs;
    const char *operator_symbol;
    bool relaxedMode; // Whether to build with -cl-fast-relaxed-math.
} BuildKernelInfo;

static cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernel(info->operator_symbol, i, info->kernel_count,
                       info->kernels[i], info->programs + i, info->relaxedMode);
}

// Thread specific data for a worker thread
typedef struct ThreadInfo
{
    cl_mem inBuf; // input buffer for the thread
    cl_mem inBuf2; // input buffer for the thread
    cl_mem outBuf[VECTOR_SIZE_COUNT]; // output buffers for the thread
    float maxError; // max error value. Init to 0.
    double
        maxErrorValue; // position of the max error value (param 1).  Init to 0.
    double maxErrorValue2; // position of the max error value (param 2).  Init
                           // to 0.
    MTdata d;
    cl_command_queue tQueue; // per thread command queue to improve performance
} ThreadInfo;

typedef struct TestInfo
{
    size_t subBufferSize; // Size of the sub-buffer in elements
    const Func *f; // A pointer to the function info
    cl_program programs[VECTOR_SIZE_COUNT]; // programs for various vector sizes
    cl_kernel
        *k[VECTOR_SIZE_COUNT]; // arrays of thread-specific kernels for each
                               // worker thread:  k[vector_size][thread_id]
    ThreadInfo *
        tinfo; // An array of thread specific information for each worker thread
    cl_uint threadCount; // Number of worker threads
    cl_uint jobCount; // Number of jobs
    cl_uint step; // step between each chunk and the next.
    cl_uint scale; // stride between individual test values
    float ulps; // max_allowed ulps
    int ftz; // non-zero if running in flush to zero mode
    bool relaxedMode; // True if the test is being run in relaxed mode, false
                      // otherwise.

    // no special fields
} TestInfo;

// A table of more difficult cases to get right
static const double specialValues[] = {
    -NAN,
    -INFINITY,
    -DBL_MAX,
    MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(-0x1.000002p32, -0x1000002LL, 8),
    MAKE_HEX_DOUBLE(-0x1.0p32, -0x1LL, 32),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp31, -0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p31, -0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(-0x1.0p31, -0x1LL, 31),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp30, -0x1fffffffffffffLL, -22),
    -1000.,
    -100.,
    -4.0,
    -3.5,
    -3.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51),
    -2.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51),
    -2.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52),
    -1.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    -1.0,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1, -0x10000000000001LL, -53),
    -0.5,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-2, -0x1fffffffffffffLL, -54),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-2, -0x10000000000001LL, -54),
    -0.25,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-3, -0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074),
    -DBL_MIN,
    MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000008p-1022, -0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000007p-1022, -0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000006p-1022, -0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000005p-1022, -0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000004p-1022, -0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074),
    -0.0,

    +NAN,
    +INFINITY,
    +DBL_MAX,
    MAKE_HEX_DOUBLE(+0x1.0000000000001p64, +0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(+0x1.0p64, +0x1LL, 64),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    MAKE_HEX_DOUBLE(+0x1.000002p32, +0x1000002LL, 8),
    MAKE_HEX_DOUBLE(+0x1.0p32, +0x1LL, 32),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp31, +0x1fffffffffffffLL, -21),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p31, +0x10000000000001LL, -21),
    MAKE_HEX_DOUBLE(+0x1.0p31, +0x1LL, 31),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp30, +0x1fffffffffffffLL, -22),
    +1000.0,
    +100.0,
    +4.0,
    +3.5,
    +3.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51),
    +2.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51),
    +2.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52),
    +1.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    +1.0,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1, +0x10000000000001LL, -53),
    +0.5,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-2, +0x1fffffffffffffLL, -54),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-2, +0x10000000000001LL, -54),
    +0.25,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-3, +0x1fffffffffffffLL, -55),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074),
    +DBL_MIN,
    MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000008p-1022, +0x00000000000008LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000007p-1022, +0x00000000000007LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000006p-1022, +0x00000000000006LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000005p-1022, +0x00000000000005LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000004p-1022, +0x00000000000004LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074),
    +0.0,
};

static const size_t specialValuesCount =
    sizeof(specialValues) / sizeof(specialValues[0]);

static cl_int Test(cl_uint job_id, cl_uint thread_id, void *data);

int TestFunc_Double_Double_Double_Operator(const Func *f, MTdata d,
                                           bool relaxedMode)
{
    TestInfo test_info;
    cl_int error;
    float maxError = 0.0f;
    double maxErrorVal = 0.0;
    double maxErrorVal2 = 0.0;

    logFunctionInfo(f->name, sizeof(cl_double), relaxedMode);

    // Init test_info
    memset(&test_info, 0, sizeof(test_info));
    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE
        / (sizeof(cl_double) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.scale = getTestScale(sizeof(cl_double));

    test_info.step = (cl_uint)test_info.subBufferSize * test_info.scale;
    if (test_info.step / test_info.subBufferSize != test_info.scale)
    {
        // there was overflow
        test_info.jobCount = 1;
    }
    else
    {
        test_info.jobCount = (cl_uint)((1ULL << 32) / test_info.step);
    }

    test_info.f = f;
    test_info.ulps = f->double_ulps;
    test_info.ftz = f->ftz || gForceFTZ;

    // cl_kernels aren't thread safe, so we make one for each vector size for
    // every thread
    for (auto i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        size_t array_size = test_info.threadCount * sizeof(cl_kernel);
        test_info.k[i] = (cl_kernel *)malloc(array_size);
        if (NULL == test_info.k[i])
        {
            vlog_error("Error: Unable to allocate storage for kernels!\n");
            error = CL_OUT_OF_HOST_MEMORY;
            goto exit;
        }
        memset(test_info.k[i], 0, array_size);
    }
    test_info.tinfo =
        (ThreadInfo *)malloc(test_info.threadCount * sizeof(*test_info.tinfo));
    if (NULL == test_info.tinfo)
    {
        vlog_error(
            "Error: Unable to allocate storage for thread specific data.\n");
        error = CL_OUT_OF_HOST_MEMORY;
        goto exit;
    }
    memset(test_info.tinfo, 0,
           test_info.threadCount * sizeof(*test_info.tinfo));
    for (cl_uint i = 0; i < test_info.threadCount; i++)
    {
        cl_buffer_region region = {
            i * test_info.subBufferSize * sizeof(cl_double),
            test_info.subBufferSize * sizeof(cl_double)
        };
        test_info.tinfo[i].inBuf =
            clCreateSubBuffer(gInBuffer, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            goto exit;
        }
        test_info.tinfo[i].inBuf2 =
            clCreateSubBuffer(gInBuffer2, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf2)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer2 for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            goto exit;
        }

        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer(
                gOutBuffer[j], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &region, &error);
            if (error || NULL == test_info.tinfo[i].outBuf[j])
            {
                vlog_error("Error: Unable to create sub-buffer of "
                           "gOutBuffer[%d] for region {%zd, %zd}\n",
                           (int)j, region.origin, region.size);
                goto exit;
            }
        }
        test_info.tinfo[i].tQueue =
            clCreateCommandQueue(gContext, gDevice, 0, &error);
        if (NULL == test_info.tinfo[i].tQueue || error)
        {
            vlog_error("clCreateCommandQueue failed. (%d)\n", error);
            goto exit;
        }

        test_info.tinfo[i].d = init_genrand(genrand_int32(d));
    }

    // Init the kernels
    {
        BuildKernelInfo build_info = {
            gMinVectorSizeIndex, test_info.threadCount, test_info.k,
            test_info.programs,  f->nameInCode,         relaxedMode
        };
        if ((error = ThreadPool_Do(BuildKernelFn,
                                   gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                   &build_info)))
            goto exit;
    }

    // Run the kernels
    if (!gSkipCorrectnessTesting)
    {
        error = ThreadPool_Do(Test, test_info.jobCount, &test_info);

        // Accumulate the arithmetic errors
        for (cl_uint i = 0; i < test_info.threadCount; i++)
        {
            if (test_info.tinfo[i].maxError > maxError)
            {
                maxError = test_info.tinfo[i].maxError;
                maxErrorVal = test_info.tinfo[i].maxErrorValue;
                maxErrorVal2 = test_info.tinfo[i].maxErrorValue2;
            }
        }

        if (error) goto exit;

        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");

        vlog("\t%8.2f @ {%a, %a}", maxError, maxErrorVal, maxErrorVal2);
    }

    vlog("\n");

exit:
    // Release
    for (auto i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        clReleaseProgram(test_info.programs[i]);
        if (test_info.k[i])
        {
            for (cl_uint j = 0; j < test_info.threadCount; j++)
                clReleaseKernel(test_info.k[i][j]);

            free(test_info.k[i]);
        }
    }
    if (test_info.tinfo)
    {
        for (cl_uint i = 0; i < test_info.threadCount; i++)
        {
            free_mtdata(test_info.tinfo[i].d);
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            clReleaseMemObject(test_info.tinfo[i].inBuf2);
            for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
                clReleaseMemObject(test_info.tinfo[i].outBuf[j]);
            clReleaseCommandQueue(test_info.tinfo[i].tQueue);
        }

        free(test_info.tinfo);
    }

    return error;
}

static cl_int Test(cl_uint job_id, cl_uint thread_id, void *data)
{
    const TestInfo *job = (const TestInfo *)data;
    size_t buffer_elements = job->subBufferSize;
    size_t buffer_size = buffer_elements * sizeof(cl_double);
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = job->tinfo + thread_id;
    float ulps = job->ulps;
    dptr func = job->f->dfunc;
    int ftz = job->ftz;
    MTdata d = tinfo->d;
    cl_int error;
    const char *name = job->f->name;
    cl_ulong *t;
    cl_double *r;
    cl_double *s;
    cl_double *s2;

    Force64BitFPUPrecision();

    // start the map of the output arrays
    cl_event e[VECTOR_SIZE_COUNT];
    cl_ulong *out[VECTOR_SIZE_COUNT];
    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        out[j] = (cl_ulong *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_WRITE, 0,
            buffer_size, 0, NULL, e + j, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            return error;
        }
    }

    // Get that moving
    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush failed\n");

    // Init input array
    cl_ulong *p = (cl_ulong *)gIn + thread_id * buffer_elements;
    cl_ulong *p2 = (cl_ulong *)gIn2 + thread_id * buffer_elements;
    cl_uint idx = 0;
    int totalSpecialValueCount = specialValuesCount * specialValuesCount;
    int lastSpecialJobIndex = (totalSpecialValueCount - 1) / buffer_elements;

    if (job_id <= (cl_uint)lastSpecialJobIndex)
    { // test edge cases
        cl_double *fp = (cl_double *)p;
        cl_double *fp2 = (cl_double *)p2;
        uint32_t x, y;

        x = (job_id * buffer_elements) % specialValuesCount;
        y = (job_id * buffer_elements) / specialValuesCount;

        for (; idx < buffer_elements; idx++)
        {
            fp[idx] = specialValues[x];
            fp2[idx] = specialValues[y];
            if (++x >= specialValuesCount)
            {
                x = 0;
                y++;
                if (y >= specialValuesCount) break;
            }
        }
    }

    // Init any remaining values.
    for (; idx < buffer_elements; idx++)
    {
        p[idx] = genrand_int64(d);
        p2[idx] = genrand_int64(d);
    }

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0,
                                      buffer_size, p, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        goto exit;
    }

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf2, CL_FALSE, 0,
                                      buffer_size, p2, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        goto exit;
    }

    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        // Wait for the map to finish
        if ((error = clWaitForEvents(1, e + j)))
        {
            vlog_error("Error: clWaitForEvents failed! err: %d\n", error);
            goto exit;
        }
        if ((error = clReleaseEvent(e[j])))
        {
            vlog_error("Error: clReleaseEvent failed! err: %d\n", error);
            goto exit;
        }

        // Fill the result buffer with garbage, so that old results don't carry
        // over
        uint32_t pattern = 0xffffdead;
        memset_pattern4(out[j], &pattern, buffer_size);
        if ((error = clEnqueueUnmapMemObject(tinfo->tQueue, tinfo->outBuf[j],
                                             out[j], 0, NULL, NULL)))
        {
            vlog_error("Error: clEnqueueMapBuffer failed! err: %d\n", error);
            goto exit;
        }

        // run the kernel
        size_t vectorCount =
            (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id]; // each worker thread has its
                                                 // own copy of the cl_kernel
        cl_program program = job->programs[j];

        if ((error = clSetKernelArg(kernel, 0, sizeof(tinfo->outBuf[j]),
                                    &tinfo->outBuf[j])))
        {
            LogBuildError(program);
            return error;
        }
        if ((error = clSetKernelArg(kernel, 1, sizeof(tinfo->inBuf),
                                    &tinfo->inBuf)))
        {
            LogBuildError(program);
            return error;
        }
        if ((error = clSetKernelArg(kernel, 2, sizeof(tinfo->inBuf2),
                                    &tinfo->inBuf2)))
        {
            LogBuildError(program);
            return error;
        }

        if ((error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL,
                                            &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error("FAILED -- could not execute kernel\n");
            goto exit;
        }
    }

    // Get that moving
    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush 2 failed\n");

    if (gSkipCorrectnessTesting) return CL_SUCCESS;

    // Calculate the correctly rounded reference result
    r = (cl_double *)gOut_Ref + thread_id * buffer_elements;
    s = (cl_double *)gIn + thread_id * buffer_elements;
    s2 = (cl_double *)gIn2 + thread_id * buffer_elements;
    for (size_t j = 0; j < buffer_elements; j++)
        r[j] = (cl_double)func.f_ff(s[j], s2[j]);

    // Read the data back -- no need to wait for the first N-1 buffers but wait
    // for the last buffer. This is an in order queue.
    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        cl_bool blocking = (j + 1 < gMaxVectorSizeIndex) ? CL_FALSE : CL_TRUE;
        out[j] = (cl_ulong *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], blocking, CL_MAP_READ, 0,
            buffer_size, 0, NULL, NULL, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            goto exit;
        }
    }

    // Verify data
    t = (cl_ulong *)r;
    for (size_t j = 0; j < buffer_elements; j++)
    {
        for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
        {
            cl_ulong *q = out[k];

            // If we aren't getting the correctly rounded result
            if (t[j] != q[j])
            {
                cl_double test = ((cl_double *)q)[j];
                long double correct = func.f_ff(s[j], s2[j]);
                float err = Bruteforce_Ulp_Error_Double(test, correct);
                int fail = !(fabsf(err) <= ulps);

                if (fail && ftz)
                {
                    // retry per section 6.5.3.2
                    if (IsDoubleResultSubnormal(correct, ulps))
                    {
                        fail = fail && (test != 0.0f);
                        if (!fail) err = 0.0f;
                    }


                    // retry per section 6.5.3.3
                    if (IsDoubleSubnormal(s[j]))
                    {
                        long double correct2 = func.f_ff(0.0, s2[j]);
                        long double correct3 = func.f_ff(-0.0, s2[j]);
                        float err2 =
                            Bruteforce_Ulp_Error_Double(test, correct2);
                        float err3 =
                            Bruteforce_Ulp_Error_Double(test, correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsDoubleResultSubnormal(correct2, ulps)
                            || IsDoubleResultSubnormal(correct3, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsDoubleSubnormal(s2[j]))
                        {
                            correct2 = func.f_ff(0.0, 0.0);
                            correct3 = func.f_ff(-0.0, 0.0);
                            long double correct4 = func.f_ff(0.0, -0.0);
                            long double correct5 = func.f_ff(-0.0, -0.0);
                            err2 = Bruteforce_Ulp_Error_Double(test, correct2);
                            err3 = Bruteforce_Ulp_Error_Double(test, correct3);
                            float err4 =
                                Bruteforce_Ulp_Error_Double(test, correct4);
                            float err5 =
                                Bruteforce_Ulp_Error_Double(test, correct5);
                            fail = fail
                                && ((!(fabsf(err2) <= ulps))
                                    && (!(fabsf(err3) <= ulps))
                                    && (!(fabsf(err4) <= ulps))
                                    && (!(fabsf(err5) <= ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;
                            if (fabsf(err4) < fabsf(err)) err = err4;
                            if (fabsf(err5) < fabsf(err)) err = err5;

                            // retry per section 6.5.3.4
                            if (IsDoubleResultSubnormal(correct2, ulps)
                                || IsDoubleResultSubnormal(correct3, ulps)
                                || IsDoubleResultSubnormal(correct4, ulps)
                                || IsDoubleResultSubnormal(correct5, ulps))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsDoubleSubnormal(s2[j]))
                    {
                        long double correct2 = func.f_ff(s[j], 0.0);
                        long double correct3 = func.f_ff(s[j], -0.0);
                        float err2 =
                            Bruteforce_Ulp_Error_Double(test, correct2);
                        float err3 =
                            Bruteforce_Ulp_Error_Double(test, correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsDoubleResultSubnormal(correct2, ulps)
                            || IsDoubleResultSubnormal(correct3, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }
                    }
                }

                if (fabsf(err) > tinfo->maxError)
                {
                    tinfo->maxError = fabsf(err);
                    tinfo->maxErrorValue = s[j];
                    tinfo->maxErrorValue2 = s2[j];
                }
                if (fail)
                {
                    vlog_error(
                        "\nERROR: %s%s: %f ulp error at {%a, %a}: *%a vs. %a\n",
                        name, sizeNames[k], err, s[j], s2[j], r[j], test);
                    error = -1;
                    goto exit;
                }
            }
        }
    }

    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        if ((error = clEnqueueUnmapMemObject(tinfo->tQueue, tinfo->outBuf[j],
                                             out[j], 0, NULL, NULL)))
        {
            vlog_error("Error: clEnqueueUnmapMemObject %d failed 2! err: %d\n",
                       j, error);
            return error;
        }
    }

    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush 3 failed\n");


    if (0 == (base & 0x0fffffff))
    {
        if (gVerboseBruteForce)
        {
            vlog("base:%14u step:%10u scale:%10zu buf_elements:%10u ulps:%5.3f "
                 "ThreadCount:%2u\n",
                 base, job->step, job->scale, buffer_elements, job->ulps,
                 job->threadCount);
        }
        else
        {
            vlog(".");
        }
        fflush(stdout);
    }

exit:
    return error;
}
