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

static int BuildKernel(const char *name, const char *operator_symbol,
                       int vectorSize, cl_uint kernel_count, cl_kernel *k,
                       cl_program *p, bool relaxedMode)
{
    const char *c[] = { "__kernel void ",
                        name,
                        "_kernel",
                        sizeNames[vectorSize],
                        "( __global float",
                        sizeNames[vectorSize],
                        "* out, __global float",
                        sizeNames[vectorSize],
                        "* in1, __global float",
                        sizeNames[vectorSize],
                        "* in2 )\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   out[i] = in1[i] ",
                        operator_symbol,
                        " in2[i];\n"
                        "}\n" };

    const char *c3[] = {
        "__kernel void ",
        name,
        "_kernel",
        sizeNames[vectorSize],
        "( __global float* out, __global float* in, __global float* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       f0 = f0 ",
        operator_symbol,
        " f1;\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       float3 f0;\n"
        "       float3 f1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
        "               f1 = (float3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
        "               f1 = (float3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       f0 = f0 ",
        operator_symbol,
        " f1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = f0.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = f0.x; \n"
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
    snprintf(testName, sizeof(testName) - 1, "%s_kernel%s", name,
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
    const char *name;
    const char *operator_symbol;
    bool relaxedMode; // Whether to build with -cl-fast-relaxed-math.
} BuildKernelInfo;

static cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernel(info->name, info->operator_symbol, i, info->kernel_count,
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
static const float specialValues[] = {
    -NAN,
    -INFINITY,
    -FLT_MAX,
    MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40),
    MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64),
    MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),
    MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39),
    MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63),
    MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    MAKE_HEX_FLOAT(-0x1.000002p32f, -0x1000002L, 8),
    MAKE_HEX_FLOAT(-0x1.0p32f, -0x1L, 32),
    MAKE_HEX_FLOAT(-0x1.fffffep31f, -0x1fffffeL, 7),
    MAKE_HEX_FLOAT(-0x1.000002p31f, -0x1000002L, 7),
    MAKE_HEX_FLOAT(-0x1.0p31f, -0x1L, 31),
    MAKE_HEX_FLOAT(-0x1.fffffep30f, -0x1fffffeL, 6),
    -1000.f,
    -100.f,
    -4.0f,
    -3.5f,
    -3.0f,
    MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23),
    -2.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23),
    -2.0f,
    MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24),
    -1.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24),
    MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24),
    -1.0f,
    MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-1f, -0x1000002L, -25),
    -0.5f,
    MAKE_HEX_FLOAT(-0x1.fffffep-2f, -0x1fffffeL, -26),
    MAKE_HEX_FLOAT(-0x1.000002p-2f, -0x1000002L, -26),
    -0.25f,
    MAKE_HEX_FLOAT(-0x1.fffffep-3f, -0x1fffffeL, -27),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150),
    -FLT_MIN,
    MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150),
    MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150),
    MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150),
    MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150),
    MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150),
    MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150),
    MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150),
    -0.0f,

    +NAN,
    +INFINITY,
    +FLT_MAX,
    MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40),
    MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64),
    MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39),
    MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39),
    MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63),
    MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    MAKE_HEX_FLOAT(+0x1.000002p32f, +0x1000002L, 8),
    MAKE_HEX_FLOAT(+0x1.0p32f, +0x1L, 32),
    MAKE_HEX_FLOAT(+0x1.fffffep31f, +0x1fffffeL, 7),
    MAKE_HEX_FLOAT(+0x1.000002p31f, +0x1000002L, 7),
    MAKE_HEX_FLOAT(+0x1.0p31f, +0x1L, 31),
    MAKE_HEX_FLOAT(+0x1.fffffep30f, +0x1fffffeL, 6),
    +1000.f,
    +100.f,
    +4.0f,
    +3.5f,
    +3.0f,
    MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23),
    2.5f,
    MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),
    +2.0f,
    MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24),
    1.5f,
    MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24),
    MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24),
    +1.0f,
    MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(+0x1.000002p-1f, +0x1000002L, -25),
    +0.5f,
    MAKE_HEX_FLOAT(+0x1.fffffep-2f, +0x1fffffeL, -26),
    MAKE_HEX_FLOAT(+0x1.000002p-2f, +0x1000002L, -26),
    +0.25f,
    MAKE_HEX_FLOAT(+0x1.fffffep-3f, +0x1fffffeL, -27),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150),
    +FLT_MIN,
    MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150),
    MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150),
    MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150),
    MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150),
    MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150),
    MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150),
    MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150),
    +0.0f,
};

static const size_t specialValuesCount =
    sizeof(specialValues) / sizeof(specialValues[0]);

static cl_int Test(cl_uint job_id, cl_uint thread_id, void *data);

int TestFunc_Float_Float_Float_Operator(const Func *f, MTdata d,
                                        bool relaxedMode)
{
    TestInfo test_info;
    cl_int error;
    size_t i, j;
    float maxError = 0.0f;
    double maxErrorVal = 0.0;
    double maxErrorVal2 = 0.0;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    // Init test_info
    memset(&test_info, 0, sizeof(test_info));
    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE
        / (sizeof(cl_float) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.scale = getTestScale(sizeof(cl_float));

    if (gWimpyMode)
    {
        test_info.subBufferSize = gWimpyBufferSize
            / (sizeof(cl_float)
               * RoundUpToNextPowerOfTwo(test_info.threadCount));
    }

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
    test_info.ulps = gIsEmbedded ? f->float_embedded_ulps : f->float_ulps;
    test_info.ftz =
        f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    test_info.relaxedMode = relaxedMode;

    // cl_kernels aren't thread safe, so we make one for each vector size for
    // every thread
    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
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
    for (i = 0; i < test_info.threadCount; i++)
    {
        cl_buffer_region region = {
            i * test_info.subBufferSize * sizeof(cl_float),
            test_info.subBufferSize * sizeof(cl_float)
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

        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer(
                gOutBuffer[j], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
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
        BuildKernelInfo build_info = { gMinVectorSizeIndex,
                                       test_info.threadCount,
                                       test_info.k,
                                       test_info.programs,
                                       f->name,
                                       f->nameInCode,
                                       relaxedMode };
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
        for (i = 0; i < test_info.threadCount; i++)
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
    }

    if (gMeasureTimes)
    {
        // Init input arrays
        uint32_t *p = (uint32_t *)gIn;
        uint32_t *p2 = (uint32_t *)gIn2;
        for (j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            p[j] = (genrand_int32(d) & ~0x40000000) | 0x20000000;
            p2[j] = 0x3fc00000;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn2, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error);
            return error;
        }

        // Run the kernels
        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeof(cl_float) * sizeValues[j];
            size_t localCount = (BUFFER_SIZE + vectorSize - 1)
                / vectorSize; // BUFFER_SIZE / vectorSize  rounded up
            if ((error = clSetKernelArg(test_info.k[j][0], 0,
                                        sizeof(gOutBuffer[j]), &gOutBuffer[j])))
            {
                LogBuildError(test_info.programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(test_info.k[j][0], 1, sizeof(gInBuffer),
                                        &gInBuffer)))
            {
                LogBuildError(test_info.programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(test_info.k[j][0], 2,
                                        sizeof(gInBuffer2), &gInBuffer2)))
            {
                LogBuildError(test_info.programs[j]);
                goto exit;
            }

            double sum = 0.0;
            double bestTime = INFINITY;
            for (i = 0; i < PERF_LOOP_COUNT; i++)
            {
                uint64_t startTime = GetTime();
                if ((error = clEnqueueNDRangeKernel(gQueue, test_info.k[j][0],
                                                    1, NULL, &localCount, NULL,
                                                    0, NULL, NULL)))
                {
                    vlog_error("FAILED -- could not execute kernel\n");
                    goto exit;
                }

                // Make sure OpenCL is done
                if ((error = clFinish(gQueue)))
                {
                    vlog_error("Error %d at clFinish\n", error);
                    goto exit;
                }

                uint64_t endTime = GetTime();
                double time = SubtractTime(endTime, startTime);
                sum += time;
                if (time < bestTime) bestTime = time;
            }

            if (gReportAverageTimes) bestTime = sum / PERF_LOOP_COUNT;
            double clocksPerOp = bestTime * (double)gDeviceFrequency
                * gComputeDevices * gSimdSize * 1e6
                / (BUFFER_SIZE / sizeof(float));
            vlog_perf(clocksPerOp, LOWER_IS_BETTER, "clocks / element", "%sf%s",
                      f->name, sizeNames[j]);
        }
    }

    if (!gSkipCorrectnessTesting)
        vlog("\t%8.2f @ {%a, %a}", maxError, maxErrorVal, maxErrorVal2);
    vlog("\n");

exit:
    // Release
    for (i = gMinVectorSizeIndex; i < gMaxVectorSizeIndex; i++)
    {
        clReleaseProgram(test_info.programs[i]);
        if (test_info.k[i])
        {
            for (j = 0; j < test_info.threadCount; j++)
                clReleaseKernel(test_info.k[i][j]);

            free(test_info.k[i]);
        }
    }
    if (test_info.tinfo)
    {
        for (i = 0; i < test_info.threadCount; i++)
        {
            free_mtdata(test_info.tinfo[i].d);
            clReleaseMemObject(test_info.tinfo[i].inBuf);
            clReleaseMemObject(test_info.tinfo[i].inBuf2);
            for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
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
    size_t buffer_size = buffer_elements * sizeof(cl_float);
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = job->tinfo + thread_id;
    fptr func = job->f->func;
    int ftz = job->ftz;
    bool relaxedMode = job->relaxedMode;
    float ulps = getAllowedUlpError(job->f, relaxedMode);
    MTdata d = tinfo->d;
    cl_uint j, k;
    cl_int error;
    cl_uchar *overflow = (cl_uchar *)malloc(buffer_size);
    const char *name = job->f->name;
    cl_uint *t = 0;
    cl_float *r = 0;
    cl_float *s = 0;
    cl_float *s2 = 0;
    RoundingMode oldRoundMode;

    if (relaxedMode)
    {
        func = job->f->rfunc;
    }

    // start the map of the output arrays
    cl_event e[VECTOR_SIZE_COUNT];
    cl_uint *out[VECTOR_SIZE_COUNT];
    for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        out[j] = (cl_uint *)clEnqueueMapBuffer(
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
    cl_uint *p = (cl_uint *)gIn + thread_id * buffer_elements;
    cl_uint *p2 = (cl_uint *)gIn2 + thread_id * buffer_elements;
    j = 0;

    int totalSpecialValueCount = specialValuesCount * specialValuesCount;
    int indx = (totalSpecialValueCount - 1) / buffer_elements;

    if (job_id <= (cl_uint)indx)
    {
        // Insert special values
        uint32_t x, y;

        x = (job_id * buffer_elements) % specialValuesCount;
        y = (job_id * buffer_elements) / specialValuesCount;

        for (; j < buffer_elements; j++)
        {
            p[j] = ((cl_uint *)specialValues)[x];
            p2[j] = ((cl_uint *)specialValues)[y];
            ++x;
            if (x >= specialValuesCount)
            {
                x = 0;
                y++;
                if (y >= specialValuesCount) break;
            }
            if (relaxedMode && strcmp(name, "divide") == 0)
            {
                cl_uint pj = p[j] & 0x7fffffff;
                cl_uint p2j = p2[j] & 0x7fffffff;
                // Replace values outside [2^-62, 2^62] with QNaN
                if (pj < 0x20800000 || pj > 0x5e800000) p[j] = 0x7fc00000;
                if (p2j < 0x20800000 || p2j > 0x5e800000) p2[j] = 0x7fc00000;
            }
        }
    }

    // Init any remaining values.
    for (; j < buffer_elements; j++)
    {
        p[j] = genrand_int32(d);
        p2[j] = genrand_int32(d);

        if (relaxedMode && strcmp(name, "divide") == 0)
        {
            cl_uint pj = p[j] & 0x7fffffff;
            cl_uint p2j = p2[j] & 0x7fffffff;
            // Replace values outside [2^-62, 2^62] with QNaN
            if (pj < 0x20800000 || pj > 0x5e800000) p[j] = 0x7fc00000;
            if (p2j < 0x20800000 || p2j > 0x5e800000) p2[j] = 0x7fc00000;
        }
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

    for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
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

    if (gSkipCorrectnessTesting)
    {
        free(overflow);
        return CL_SUCCESS;
    }

    // Calculate the correctly rounded reference result
    FPU_mode_type oldMode;
    memset(&oldMode, 0, sizeof(oldMode));
    if (ftz) ForceFTZ(&oldMode);

    // Set the rounding mode to match the device
    oldRoundMode = kRoundToNearestEven;
    if (gIsInRTZMode) oldRoundMode = set_round(kRoundTowardZero, kfloat);

    // Calculate the correctly rounded reference result
    r = (float *)gOut_Ref + thread_id * buffer_elements;
    s = (float *)gIn + thread_id * buffer_elements;
    s2 = (float *)gIn2 + thread_id * buffer_elements;
    if (gInfNanSupport)
    {
        for (j = 0; j < buffer_elements; j++)
            r[j] = (float)func.f_ff(s[j], s2[j]);
    }
    else
    {
        for (j = 0; j < buffer_elements; j++)
        {
            feclearexcept(FE_OVERFLOW);
            r[j] = (float)func.f_ff(s[j], s2[j]);
            overflow[j] =
                FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
        }
    }

    if (gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);

    if (ftz) RestoreFPState(&oldMode);

    // Read the data back -- no need to wait for the first N-1 buffers. This is
    // an in order queue.
    for (j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++)
    {
        out[j] = (cl_uint *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0,
            buffer_size, 0, NULL, NULL, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            goto exit;
        }
    }

    // Wait for the last buffer
    out[j] = (cl_uint *)clEnqueueMapBuffer(tinfo->tQueue, tinfo->outBuf[j],
                                           CL_TRUE, CL_MAP_READ, 0, buffer_size,
                                           0, NULL, NULL, &error);
    if (error || NULL == out[j])
    {
        vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error);
        goto exit;
    }

    // Verify data
    t = (cl_uint *)r;
    for (j = 0; j < buffer_elements; j++)
    {
        for (k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
        {
            cl_uint *q = out[k];

            // If we aren't getting the correctly rounded result
            if (t[j] != q[j])
            {
                float test = ((float *)q)[j];
                double correct = func.f_ff(s[j], s2[j]);

                // Per section 10 paragraph 6, accept any result if an input or
                // output is a infinity or NaN or overflow
                if (!gInfNanSupport)
                {
                    // Note: no double rounding here.  Reference functions
                    // calculate in single precision.
                    if (overflow[j] || IsFloatInfinity(correct)
                        || IsFloatNaN(correct) || IsFloatInfinity(s2[j])
                        || IsFloatNaN(s2[j]) || IsFloatInfinity(s[j])
                        || IsFloatNaN(s[j]))
                        continue;
                }

                // Per section 10 paragraph 6, accept embedded devices always
                // returning positive 0.0.
                if (gIsEmbedded && (t[j] == 0x80000000) && (q[j] == 0x00000000))
                    continue;

                float err = Ulp_Error(test, correct);
                float errB = Ulp_Error(test, (float)correct);

                int fail =
                    ((!(fabsf(err) <= ulps)) && (!(fabsf(errB) <= ulps)));
                if (fabsf(errB) < fabsf(err)) err = errB;

                if (fail && ftz)
                {
                    // retry per section 6.5.3.2
                    if (IsFloatResultSubnormal(correct, ulps))
                    {
                        fail = fail && (test != 0.0f);
                        if (!fail) err = 0.0f;
                    }

                    // retry per section 6.5.3.3
                    if (IsFloatSubnormal(s[j]))
                    {
                        double correct2, correct3;
                        float err2, err3;

                        if (!gInfNanSupport) feclearexcept(FE_OVERFLOW);

                        correct2 = func.f_ff(0.0, s2[j]);
                        correct3 = func.f_ff(-0.0, s2[j]);

                        // Per section 10 paragraph 6, accept any result if an
                        // input or output is a infinity or NaN or overflow
                        if (!gInfNanSupport)
                        {
                            if (fetestexcept(FE_OVERFLOW)) continue;

                            // Note: no double rounding here.  Reference
                            // functions calculate in single precision.
                            if (IsFloatInfinity(correct2)
                                || IsFloatNaN(correct2)
                                || IsFloatInfinity(correct3)
                                || IsFloatNaN(correct3))
                                continue;
                        }

                        err2 = Ulp_Error(test, correct2);
                        err3 = Ulp_Error(test, correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, ulps)
                            || IsFloatResultSubnormal(correct3, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsFloatSubnormal(s2[j]))
                        {
                            double correct4, correct5;
                            float err4, err5;

                            if (!gInfNanSupport) feclearexcept(FE_OVERFLOW);

                            correct2 = func.f_ff(0.0, 0.0);
                            correct3 = func.f_ff(-0.0, 0.0);
                            correct4 = func.f_ff(0.0, -0.0);
                            correct5 = func.f_ff(-0.0, -0.0);

                            // Per section 10 paragraph 6, accept any result if
                            // an input or output is a infinity or NaN or
                            // overflow
                            if (!gInfNanSupport)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correct2)
                                    || IsFloatNaN(correct2)
                                    || IsFloatInfinity(correct3)
                                    || IsFloatNaN(correct3)
                                    || IsFloatInfinity(correct4)
                                    || IsFloatNaN(correct4)
                                    || IsFloatInfinity(correct5)
                                    || IsFloatNaN(correct5))
                                    continue;
                            }

                            err2 = Ulp_Error(test, correct2);
                            err3 = Ulp_Error(test, correct3);
                            err4 = Ulp_Error(test, correct4);
                            err5 = Ulp_Error(test, correct5);
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
                            if (IsFloatResultSubnormal(correct2, ulps)
                                || IsFloatResultSubnormal(correct3, ulps)
                                || IsFloatResultSubnormal(correct4, ulps)
                                || IsFloatResultSubnormal(correct5, ulps))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsFloatSubnormal(s2[j]))
                    {
                        double correct2, correct3;
                        float err2, err3;

                        if (!gInfNanSupport) feclearexcept(FE_OVERFLOW);

                        correct2 = func.f_ff(s[j], 0.0);
                        correct3 = func.f_ff(s[j], -0.0);

                        // Per section 10 paragraph 6, accept any result if an
                        // input or output is a infinity or NaN or overflow
                        if (!gInfNanSupport)
                        {
                            // Note: no double rounding here.  Reference
                            // functions calculate in single precision.
                            if (overflow[j] || IsFloatInfinity(correct)
                                || IsFloatNaN(correct)
                                || IsFloatInfinity(correct2)
                                || IsFloatNaN(correct2))
                                continue;
                        }

                        err2 = Ulp_Error(test, correct2);
                        err3 = Ulp_Error(test, correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, ulps)
                            || IsFloatResultSubnormal(correct3, ulps))
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
                    vlog_error("\nERROR: %s%s: %f ulp error at {%a, %a}: *%a "
                               "vs. %a (0x%8.8x) at index: %d\n",
                               name, sizeNames[k], err, s[j], s2[j], r[j], test,
                               ((cl_uint *)&test)[0], j);
                    error = -1;
                    goto exit;
                }
            }
        }
    }

    for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
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
    if (overflow) free(overflow);
    return error;
}
