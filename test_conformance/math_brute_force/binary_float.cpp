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

const float twoToMinus126 = MAKE_HEX_FLOAT(0x1p-126f, 1, -126);

static int BuildKernel(const char *name, int vectorSize, cl_uint kernel_count,
                       cl_kernel *k, cl_program *p, bool relaxedMode)
{
    const char *c[] = { "__kernel void math_kernel",
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
                        "   out[i] = ",
                        name,
                        "( in1[i], in2[i] );\n"
                        "}\n" };

    const char *c3[] = {
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global float* out, __global float* in, __global float* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       f0 = ",
        name,
        "( f0, f1 );\n"
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
        "       f0 = ",
        name,
        "( f0, f1 );\n"
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
    const char *nameInCode;
    bool relaxedMode; // Whether to build with -cl-fast-relaxed-math.
} BuildKernelInfo;

static cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernel(info->nameInCode, i, info->kernel_count,
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

    int isFDim;
    int skipNanInf;
    int isNextafter;
    bool relaxedMode; // True if test is running in relaxed mode, false
                      // otherwise.
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

int TestFunc_Float_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    TestInfo test_info;
    cl_int error;
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
    test_info.isFDim = 0 == strcmp("fdim", f->nameInCode);
    test_info.skipNanInf = test_info.isFDim && !gInfNanSupport;
    test_info.isNextafter = 0 == strcmp("nextafter", f->nameInCode);

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
    size_t buffer_size = buffer_elements * sizeof(cl_float);
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = job->tinfo + thread_id;
    fptr func = job->f->func;
    int ftz = job->ftz;
    bool relaxedMode = job->relaxedMode;
    float ulps = getAllowedUlpError(job->f, relaxedMode);
    MTdata d = tinfo->d;
    cl_int error;
    cl_uchar *overflow = (cl_uchar *)malloc(buffer_size);
    const char *name = job->f->name;
    int isFDim = job->isFDim;
    int skipNanInf = job->skipNanInf;
    int isNextafter = job->isNextafter;
    cl_uint *t = 0;
    cl_float *r = 0;
    cl_float *s = 0;
    cl_float *s2 = 0;
    cl_int copysign_test = 0;
    RoundingMode oldRoundMode;
    int skipVerification = 0;

    if (relaxedMode)
    {
        func = job->f->rfunc;
        if (strcmp(name, "pow") == 0 && gFastRelaxedDerived)
        {
            ulps = INFINITY;
            skipVerification = 1;
        }
    }

    // start the map of the output arrays
    cl_event e[VECTOR_SIZE_COUNT];
    cl_uint *out[VECTOR_SIZE_COUNT];
    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
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
    cl_uint idx = 0;
    int totalSpecialValueCount = specialValuesCount * specialValuesCount;
    int lastSpecialJobIndex = (totalSpecialValueCount - 1) / buffer_elements;

    if (job_id <= (cl_uint)lastSpecialJobIndex)
    { // test edge cases
        float *fp = (float *)p;
        float *fp2 = (float *)p2;
        uint32_t x, y;

        x = (job_id * buffer_elements) % specialValuesCount;
        y = (job_id * buffer_elements) / specialValuesCount;

        for (; idx < buffer_elements; idx++)
        {
            fp[idx] = specialValues[x];
            fp2[idx] = specialValues[y];
            ++x;
            if (x >= specialValuesCount)
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
        p[idx] = genrand_int32(d);
        p2[idx] = genrand_int32(d);
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

    if (gSkipCorrectnessTesting)
    {
        if ((error = clFinish(tinfo->tQueue)))
        {
            vlog_error("Error: clFinish failed! err: %d\n", error);
            goto exit;
        }
        free(overflow);
        return CL_SUCCESS;
    }

    FPU_mode_type oldMode;
    oldRoundMode = kRoundToNearestEven;
    if (isFDim)
    {
        // Calculate the correctly rounded reference result
        memset(&oldMode, 0, sizeof(oldMode));
        if (ftz) ForceFTZ(&oldMode);

        // Set the rounding mode to match the device
        if (gIsInRTZMode) oldRoundMode = set_round(kRoundTowardZero, kfloat);
    }

    if (!strcmp(name, "copysign")) copysign_test = 1;

#define ref_func(s, s2) (copysign_test ? func.f_ff_f(s, s2) : func.f_ff(s, s2))

    // Calculate the correctly rounded reference result
    r = (float *)gOut_Ref + thread_id * buffer_elements;
    s = (float *)gIn + thread_id * buffer_elements;
    s2 = (float *)gIn2 + thread_id * buffer_elements;
    if (skipNanInf)
    {
        for (size_t j = 0; j < buffer_elements; j++)
        {
            feclearexcept(FE_OVERFLOW);
            r[j] = (float)ref_func(s[j], s2[j]);
            overflow[j] =
                FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
        }
    }
    else
    {
        for (size_t j = 0; j < buffer_elements; j++)
            r[j] = (float)ref_func(s[j], s2[j]);
    }

    if (isFDim && ftz) RestoreFPState(&oldMode);

    // Read the data back -- no need to wait for the first N-1 buffers but wait
    // for the last buffer. This is an in order queue.
    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        cl_bool blocking = (j + 1 < gMaxVectorSizeIndex) ? CL_FALSE : CL_TRUE;
        out[j] = (cl_uint *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], blocking, CL_MAP_READ, 0,
            buffer_size, 0, NULL, NULL, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            goto exit;
        }
    }

    if (!skipVerification)
    {
        // Verify data
        t = (cl_uint *)r;
        for (size_t j = 0; j < buffer_elements; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                cl_uint *q = out[k];

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    float test = ((float *)q)[j];
                    double correct = ref_func(s[j], s2[j]);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow As per
                    // OpenCL 2.0 spec, section 5.8.4.3, enabling
                    // fast-relaxed-math mode also enables -cl-finite-math-only
                    // optimization. This optimization allows to assume that
                    // arguments and results are not NaNs or +/-INFs. Hence,
                    // accept any result if inputs or results are NaNs or INFs.
                    if (relaxedMode || skipNanInf)
                    {
                        if (skipNanInf && overflow[j]) continue;
                        // Note: no double rounding here.  Reference functions
                        // calculate in single precision.
                        if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                            || IsFloatInfinity(s2[j]) || IsFloatNaN(s2[j])
                            || IsFloatInfinity(s[j]) || IsFloatNaN(s[j]))
                            continue;
                    }

                    float err = Ulp_Error(test, correct);
                    int fail = !(fabsf(err) <= ulps);

                    if (fail && ftz)
                    {
                        // retry per section 6.5.3.2
                        if (IsFloatResultSubnormal(correct, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // nextafter on FTZ platforms may return the smallest
                        // normal float (2^-126) given a denormal or a zero
                        // as the first argument. The rationale here is that
                        // nextafter flushes the argument to zero and then
                        // returns the next representable number in the
                        // direction of the second argument, and since
                        // denorms are considered as zero, the smallest
                        // normal number is the next representable number.
                        // In which case, it should have the same sign as the
                        // second argument.
                        if (isNextafter)
                        {
                            if (IsFloatSubnormal(s[j]) || s[j] == 0.0f)
                            {
                                float value = copysignf(twoToMinus126, s2[j]);
                                fail = fail && (test != value);
                                if (!fail) err = 0.0f;
                            }
                        }
                        else
                        {
                            // retry per section 6.5.3.3
                            if (IsFloatSubnormal(s[j]))
                            {
                                double correct2, correct3;
                                float err2, err3;

                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                correct2 = ref_func(0.0, s2[j]);
                                correct3 = ref_func(-0.0, s2[j]);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow As per OpenCL 2.0 spec,
                                // section 5.8.4.3, enabling fast-relaxed-math
                                // mode also enables -cl-finite-math-only
                                // optimization. This optimization allows to
                                // assume that arguments and results are not
                                // NaNs or +/-INFs. Hence, accept any result if
                                // inputs or results are NaNs or INFs.
                                if (relaxedMode || skipNanInf)
                                {
                                    if (fetestexcept(FE_OVERFLOW) && skipNanInf)
                                        continue;

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

                                    if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                    correct2 = ref_func(0.0, 0.0);
                                    correct3 = ref_func(-0.0, 0.0);
                                    correct4 = ref_func(0.0, -0.0);
                                    correct5 = ref_func(-0.0, -0.0);

                                    // Per section 10 paragraph 6, accept any
                                    // result if an input or output is a
                                    // infinity or NaN or overflow As per
                                    // OpenCL 2.0 spec, section 5.8.4.3,
                                    // enabling fast-relaxed-math mode also
                                    // enables -cl-finite-math-only
                                    // optimization. This optimization allows to
                                    // assume that arguments and results are not
                                    // NaNs or +/-INFs. Hence, accept any result
                                    // if inputs or results are NaNs or INFs.
                                    if (relaxedMode || skipNanInf)
                                    {
                                        if (fetestexcept(FE_OVERFLOW)
                                            && skipNanInf)
                                            continue;

                                        // Note: no double rounding here.
                                        // Reference functions calculate in
                                        // single precision.
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
                                        || IsFloatResultSubnormal(correct3,
                                                                  ulps)
                                        || IsFloatResultSubnormal(correct4,
                                                                  ulps)
                                        || IsFloatResultSubnormal(correct5,
                                                                  ulps))
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

                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                correct2 = ref_func(s[j], 0.0);
                                correct3 = ref_func(s[j], -0.0);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow As per OpenCL 2.0 spec,
                                // section 5.8.4.3, enabling fast-relaxed-math
                                // mode also enables -cl-finite-math-only
                                // optimization. This optimization allows to
                                // assume that arguments and results are not
                                // NaNs or +/-INFs. Hence, accept any result if
                                // inputs or results are NaNs or INFs.
                                if (relaxedMode || skipNanInf)
                                {
                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (overflow[j] && skipNanInf) continue;

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
                            "\nERROR: %s%s: %f ulp error at {%a (0x%x), %a "
                            "(0x%x)}: *%a vs. %a (0x%8.8x) at index: %d\n",
                            name, sizeNames[k], err, s[j], ((cl_uint *)s)[j],
                            s2[j], ((cl_uint *)s2)[j], r[j], test,
                            ((cl_uint *)&test)[0], j);
                        error = -1;
                        goto exit;
                    }
                }
            }
        }
    }

    if (isFDim && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);

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
    if (overflow) free(overflow);
    return error;
}
