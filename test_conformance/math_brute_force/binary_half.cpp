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

#include "common.h"
#include "function_list.h"
#include "test_functions.h"
#include "utility.h"
#include "reference_math.h"
#include <cstring>

#include "harness/errorHelpers.h"

#if 0

static int BuildKernelHalf(const char *name, int vectorSize,
                           cl_uint kernel_count, cl_kernel *k, cl_program *p,
                           bool relaxedMode)
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
                        "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global half",
                        sizeNames[vectorSize],
                        "* out, __global half",
                        sizeNames[vectorSize],
                        "* in1, __global half",
                        sizeNames[vectorSize],
                        "* in2 )\n"
                        "{\n"
                        "   int i = get_global_id(0);\n"
                        "   out[i] = ",
                        name,
                        "( in1[i], in2[i] );\n"
                        "}\n" };

    const char *c3[] = {
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global half* out, __global half* in, __global half* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       half3 d0 = vload3( 0, in + 3 * i );\n"
        "       half3 d1 = vload3( 0, in2 + 3 * i );\n"
        "       d0 = ",
        name,
        "( d0, d1 );\n"
        "       vstore3( d0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       half3 d0, d1;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               d0 = (half3)( in[3*i], NAN, NAN ); \n"
        "               d1 = (half3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               d0 = (half3)( in[3*i], in[3*i+1], NAN ); \n"
        "               d1 = (half3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       d0 = ",
        name,
        "( d0, d1 );\n"
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
    const char *nameInCode;
    bool relaxedMode;
} BuildKernelInfo;


static cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED,
                                 void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernelHalf(info->nameInCode, i, info->kernel_count,
                           info->kernels[i], info->programs + i,
                           info->relaxedMode);
}
#else

static cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED,
                                 void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetBinaryKernel(kernel_name, builtin, ParameterType::Half,
                               ParameterType::Half, ParameterType::Half,
                               vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

#endif


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

////////////////////////////////////////////////////////////////////////////////

struct TestInfoBase
{
    size_t subBufferSize; // Size of the sub-buffer in elements
    const Func *f; // A pointer to the function info

    cl_uint threadCount; // Number of worker threads
    cl_uint jobCount; // Number of jobs
    cl_uint step; // step between each chunk and the next.
    cl_uint scale; // stride between individual test values
    float ulps; // max_allowed ulps
    int ftz; // non-zero if running in flush to zero mode

    int isFDim;
    int skipNanInf;
    int isNextafter;
};

////////////////////////////////////////////////////////////////////////////////

struct TestInfo : public TestInfoBase
{
    TestInfo(const TestInfoBase &base): TestInfoBase(base) {}

    // Array of thread specific information
    std::vector<ThreadInfo> tinfo;

    // Programs for various vector sizes.
    Programs programs;

    // Thread-specific kernels for each vector size:
    // k[vector_size][thread_id]
    KernelMatrix k;
};

////////////////////////////////////////////////////////////////////////////////
// A table of more difficult cases to get right
static const cl_half specialValuesHalf[] = {
    0xffff,
    0x0000,
    0x0001,
    0x7c00 /*INFINITY*/,
    0xfc00 /*-INFINITY*/,
    0x8000 /*-0*/,
    0x7bff /*HALF_MAX*/,
    0x0400 /*HALF_MIN*/
};

static size_t specialValuesHalfCount = ARRAY_SIZE(specialValuesHalf);

static cl_int TestHalf(cl_uint job_id, cl_uint thread_id, void *p);

int TestFunc_Half_Half_Half_common(const Func *f, MTdata d, int isNextafter,
                                   bool relaxedMode)
{
    TestInfoBase test_info_base;
    cl_int error;
    size_t i, j;
    float maxError = 0.0f;
    double maxErrorVal = 0.0;
    double maxErrorVal2 = 0.0;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);
    // Init test_info
    memset(&test_info_base, 0, sizeof(test_info_base));
    TestInfo test_info(test_info_base);

    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE
        / (sizeof(cl_half) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.scale = getTestScale(sizeof(cl_half));

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
    test_info.ulps = f->half_ulps;
    test_info.ftz =
        f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);

    test_info.isFDim = 0 == strcmp("fdim", f->nameInCode);
    test_info.skipNanInf = test_info.isFDim && !gInfNanSupport;
    test_info.isNextafter = isNextafter;

#if 0
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

#else

    test_info.tinfo.resize(test_info.threadCount);

#endif

    for (i = 0; i < test_info.threadCount; i++)
    {
        cl_buffer_region region = { i * test_info.subBufferSize
                                        * sizeof(cl_half),
                                    test_info.subBufferSize * sizeof(cl_half) };
        test_info.tinfo[i].inBuf =
            clCreateSubBuffer(gInBuffer, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            return error;
        }
        test_info.tinfo[i].inBuf2 =
            clCreateSubBuffer(gInBuffer2, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer2 for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            return error;
        }

        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer(
                gOutBuffer[j], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &region, &error);
            if (error || NULL == test_info.tinfo[i].outBuf[j])
            {
                vlog_error("Error: Unable to create sub-buffer of gOutBuffer "
                           "for region {%zd, %zd}\n",
                           region.origin, region.size);
                return error;
            }
        }
        test_info.tinfo[i].tQueue =
            clCreateCommandQueue(gContext, gDevice, 0, &error);
        if (NULL == test_info.tinfo[i].tQueue || error)
        {
            vlog_error("clCreateCommandQueue failed. (%d)\n", error);
            return error;
        }
        test_info.tinfo[i].d = init_genrand(genrand_int32(d));
    }

    // Init the kernels
    {
        BuildKernelInfo build_info = { test_info.threadCount, test_info.k,
                                       test_info.programs, f->nameInCode };
        error = ThreadPool_Do(BuildKernel_HalfFn,
                              gMaxVectorSizeIndex - gMinVectorSizeIndex,
                              &build_info);
        test_error(error, "ThreadPool_Do: BuildKernel_HalfFn failed\n");
    }
    if (!gSkipCorrectnessTesting)
    {
        error = ThreadPool_Do(TestHalf, test_info.jobCount, &test_info);

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

        test_error(error, "ThreadPool_Do: TestHalf failed\n");

        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");

        vlog("\t%8.2f @ {%a, %a}", maxError, maxErrorVal, maxErrorVal2);
    }

    vlog("\n");

#if 0
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
#endif

    return error;
}

static cl_int TestHalf(cl_uint job_id, cl_uint thread_id, void *data)
{
    TestInfo *job = (TestInfo *)data;
    size_t buffer_elements = job->subBufferSize;
    size_t buffer_size = buffer_elements * sizeof(cl_half);
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = &(job->tinfo[thread_id]);
    float ulps = job->ulps;
    fptr func = job->f->func;
    int ftz = job->ftz;
    MTdata d = tinfo->d;
    cl_uint j, k;
    cl_int error;
    const char *name = job->f->name;

    int isFDim = job->isFDim;
    int skipNanInf = job->skipNanInf;
    int isNextafter = job->isNextafter;
    cl_ushort *t;
    cl_half *r;
    float *s = 0, *s2 = 0;

    RoundingMode oldRoundMode;
    cl_int copysign_test = 0;

    // start the map of the output arrays
    cl_event e[VECTOR_SIZE_COUNT];
    cl_ushort *out[VECTOR_SIZE_COUNT];
    for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        out[j] = (cl_ushort *)clEnqueueMapBuffer(
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
    cl_ushort *p = (cl_ushort *)gIn + thread_id * buffer_elements;
    cl_ushort *p2 = (cl_ushort *)gIn2 + thread_id * buffer_elements;
    j = 0;
    int totalSpecialValueCount =
        specialValuesHalfCount * specialValuesHalfCount;
    int indx = (totalSpecialValueCount - 1) / buffer_elements;

    if (job_id <= (cl_uint)indx)
    { // test edge cases
        uint32_t x, y;

        x = (job_id * buffer_elements) % specialValuesHalfCount;
        y = (job_id * buffer_elements) / specialValuesHalfCount;

        for (; j < buffer_elements; j++)
        {
            p[j] = specialValuesHalf[x];
            p2[j] = specialValuesHalf[y];
            if (++x >= specialValuesHalfCount)
            {
                x = 0;
                y++;
                if (y >= specialValuesHalfCount) break;
            }
        }
    }

    // Init any remaining values.
    for (; j < buffer_elements; j++)
    {
        p[j] = (cl_ushort)genrand_int32(d);
        p2[j] = (cl_ushort)genrand_int32(d);
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
        uint16_t pattern = 0xdead;
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
    r = (cl_half *)gOut_Ref + thread_id * buffer_elements;
    t = (cl_ushort *)r;
    s = (float *)malloc(buffer_elements * sizeof(float));
    s2 = (float *)malloc(buffer_elements * sizeof(float));
    for (j = 0; j < buffer_elements; j++)
        for (j = 0; j < buffer_elements; j++)
        {
            s[j] = cl_half_to_float(p[j]);
            s2[j] = cl_half_to_float(p2[j]);
            if (isNextafter)
                r[j] = cl_half_from_float(reference_nextafterh(s[j], s2[j]),
                                          CL_HALF_RTE);
            else
                r[j] = cl_half_from_float(ref_func(s[j], s2[j]), CL_HALF_RTE);
        }

    if (isFDim && ftz) RestoreFPState(&oldMode);
    // Read the data back -- no need to wait for the first N-1 buffers. This is
    // an in order queue.
    for (j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++)
    {
        out[j] = (cl_ushort *)clEnqueueMapBuffer(
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
    out[j] = (cl_ushort *)clEnqueueMapBuffer(
        tinfo->tQueue, tinfo->outBuf[j], CL_TRUE, CL_MAP_READ, 0, buffer_size,
        0, NULL, NULL, &error);
    if (error || NULL == out[j])
    {
        vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error);
        goto exit;
    }

    // Verify data

    for (j = 0; j < buffer_elements; j++)
    {
        for (k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
        {
            cl_ushort *q = out[k];

            // If we aren't getting the correctly rounded result
            if (t[j] != q[j])
            {
                double correct;
                if (isNextafter)
                    correct = reference_nextafterh(s[j], s2[j]);
                else
                    correct = ref_func(s[j], s2[j]);

                float test = cl_half_to_float(q[j]);

                // Per section 10 paragraph 6, accept any result if an input or
                // output is a infinity or NaN or overflow
                if (skipNanInf)
                {
                    // Note: no double rounding here.  Reference functions
                    // calculate in single precision.
                    if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                        || IsFloatInfinity(s2[j]) || IsFloatNaN(s2[j])
                        || IsFloatInfinity(s[j]) || IsFloatNaN(s[j]))
                        continue;
                }
                float err = Ulp_Error_Half(q[j], correct);
                int fail = !(fabsf(err) <= ulps);

                if (fail && ftz)
                {
                    // retry per section 6.5.3.2
                    if (IsHalfSubnormal(
                            cl_half_from_float(correct, CL_HALF_RTE)))
                    {
                        fail = fail && (test != 0.0f);
                        if (!fail) err = 0.0f;
                    }

                    if (IsHalfSubnormal(p[j]))
                    {
                        double correct2, correct3;
                        float err2, err3;
                        if (isNextafter)
                            correct2 = reference_nextafterh(0.0, s2[j]);
                        else
                            correct2 = ref_func(0.0, s2[j]);
                        if (isNextafter)
                            correct3 = reference_nextafterh(-0.0, s2[j]);
                        else
                            correct3 = ref_func(-0.0, s2[j]);
                        if (skipNanInf)
                        {
                            // Note: no double rounding here.  Reference
                            // functions calculate in single precision.
                            if (IsFloatInfinity(correct2)
                                || IsFloatNaN(correct2)
                                || IsFloatInfinity(correct3)
                                || IsFloatNaN(correct3))
                                continue;
                        }

                        err2 = Ulp_Error_Half(q[j], correct2);
                        err3 = Ulp_Error_Half(q[j], correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsHalfSubnormal(
                                cl_half_from_float(correct2, CL_HALF_RTE))
                            || IsHalfSubnormal(
                                cl_half_from_float(correct3, CL_HALF_RTE)))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // allow to omit denorm values for platforms with no
                        // denorm support for nextafter
                        if (fail && (isNextafter)
                            && (correct <= cl_half_to_float(0x3FF))
                            && (correct >= cl_half_to_float(0x83FF)))
                        {
                            fail = fail && (q[j] != p[j]);
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsHalfSubnormal(p2[j]))
                        {
                            double correct4, correct5;
                            float err4, err5;

                            if (isNextafter)
                                correct2 = reference_nextafterh(0.0, 0.0);
                            else
                                correct2 = ref_func(0.0, 0.0);
                            if (isNextafter)
                                correct3 = reference_nextafterh(-0.0, 0.0);
                            else
                                correct3 = ref_func(-0.0, 0.0);
                            if (isNextafter)
                                correct4 = reference_nextafterh(0.0, -0.0);
                            else
                                correct4 = ref_func(0.0, -0.0);
                            if (isNextafter)
                                correct5 = reference_nextafterh(-0.0, -0.0);
                            else
                                correct5 = ref_func(-0.0, -0.0);

                            // Per section 10 paragraph 6, accept any result if
                            // an input or output is a infinity or NaN or
                            // overflow
                            if (skipNanInf)
                            {
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
                            err2 = Ulp_Error_Half(q[j], correct2);
                            err3 = Ulp_Error_Half(q[j], correct3);
                            err4 = Ulp_Error_Half(q[j], correct4);
                            err5 = Ulp_Error_Half(q[j], correct5);
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
                            if (IsHalfSubnormal(
                                    cl_half_from_float(correct2, CL_HALF_RTE))
                                || IsHalfSubnormal(
                                    cl_half_from_float(correct3, CL_HALF_RTE))
                                || IsHalfSubnormal(
                                    cl_half_from_float(correct4, CL_HALF_RTE))
                                || IsHalfSubnormal(
                                    cl_half_from_float(correct5, CL_HALF_RTE)))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }

                            // allow to omit denorm values for platforms with no
                            // denorm support for nextafter
                            if (fail && (isNextafter)
                                && (correct <= cl_half_to_float(0x3FF))
                                && (correct >= cl_half_to_float(0x83FF)))
                            {
                                fail = fail && (q[j] != p2[j]);
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsHalfSubnormal(p2[j]))
                    {
                        double correct2, correct3;
                        float err2, err3;

                        if (isNextafter)
                            correct2 = reference_nextafterh(s[j], 0.0);
                        else
                            correct2 = ref_func(s[j], 0.0);
                        if (isNextafter)
                            correct3 = reference_nextafterh(s[j], -0.0);
                        else
                            correct3 = ref_func(s[j], -0.0);
                        if (skipNanInf)
                        {
                            // Note: no double rounding here.  Reference
                            // functions calculate in single precision.
                            if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                                || IsFloatInfinity(correct2)
                                || IsFloatNaN(correct2))
                                continue;
                        }

                        err2 = Ulp_Error_Half(q[j], correct2);
                        err3 = Ulp_Error_Half(q[j], correct3);
                        fail = fail
                            && ((!(fabsf(err2) <= ulps))
                                && (!(fabsf(err3) <= ulps)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;

                        // retry per section 6.5.3.4
                        if (IsHalfSubnormal(
                                cl_half_from_float(correct2, CL_HALF_RTE))
                            || IsHalfSubnormal(
                                cl_half_from_float(correct3, CL_HALF_RTE)))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // allow to omit denorm values for platforms with no
                        // denorm support for nextafter
                        if (fail && (isNextafter)
                            && (correct <= cl_half_to_float(0x3FF))
                            && (correct >= cl_half_to_float(0x83FF)))
                        {
                            fail = fail && (q[j] != p2[j]);
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
                    vlog_error("\nERROR: %s%s: %f ulp error at {%a (0x%0.4x), "
                               "%a (0x%0.4x)}\nExpected: %a  (half 0x%0.4x) "
                               "\nActual: %a (half 0x%0.4x) at index: %d\n",
                               name, sizeNames[k], err, s[j], p[j], s2[j],
                               p2[j], cl_half_to_float(r[j]), r[j], test, q[j],
                               j);
                    error = -1;
                    goto exit;
                }
            }
        }
    }

    if (isFDim && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);

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
    if (s) free(s);
    if (s2) free(s2);
    return error;
}


int TestFunc_Half_Half_Half(const Func *f, MTdata d, bool relaxedMode)
{
    return TestFunc_Half_Half_Half_common(f, d, 0, relaxedMode);
}

int TestFunc_Half_Half_Half_nextafter(const Func *f, MTdata d, bool relaxedMode)
{
    return TestFunc_Half_Half_Half_common(f, d, 1, relaxedMode);
}
