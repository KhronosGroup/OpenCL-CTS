//
// Copyright (c) 2017-2024 The Khronos Group Inc.
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

#include <cstring>

namespace {

cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Half,
                              ParameterType::Half, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

// Thread specific data for a worker thread
typedef struct ThreadInfo
{
    clMemWrapper inBuf; // input buffer for the thread
    clMemWrapper outBuf[VECTOR_SIZE_COUNT]; // output buffers for the thread
    float maxError; // max error value. Init to 0.
    double maxErrorValue; // position of the max error value.  Init to 0.
    clCommandQueueWrapper
        tQueue; // per thread command queue to improve performance
} ThreadInfo;

struct TestInfo : public TestInfoBase
{
    // Array of thread specific information
    std::vector<ThreadInfo> tinfo;

    // Programs for various vector sizes.
    Programs programs;

    // Thread-specific kernels for each vector size:
    // k[vector_size][thread_id]
    KernelMatrix k;
};

cl_int TestHalf(cl_uint job_id, cl_uint thread_id, void *data)
{
    TestInfo *job = (TestInfo *)data;
    size_t buffer_elements = job->subBufferSize;
    size_t buffer_size = buffer_elements * sizeof(cl_half);
    cl_uint scale = job->scale;
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = &(job->tinfo[thread_id]);
    float ulps = job->ulps;
    fptr func = job->f->func;
    cl_uint j, k;
    cl_int error = CL_SUCCESS;

    int isRangeLimited = job->isRangeLimited;
    float half_sin_cos_tan_limit = job->half_sin_cos_tan_limit;
    int ftz = job->ftz;

    std::vector<float> s(0);

    cl_event e[VECTOR_SIZE_COUNT];
    cl_ushort *out[VECTOR_SIZE_COUNT];

    if (gHostFill)
    {
        // start the map of the output arrays
        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            out[j] = (uint16_t *)clEnqueueMapBuffer(
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
    }

    // Write the new values to the input array
    cl_ushort *p = (cl_ushort *)gIn + thread_id * buffer_elements;
    for (j = 0; j < buffer_elements; j++)
    {
        p[j] = base + j * scale;
    }

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0,
                                      buffer_size, p, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        return error;
    }

    for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        if (gHostFill)
        {
            // Wait for the map to finish
            if ((error = clWaitForEvents(1, e + j)))
            {
                vlog_error("Error: clWaitForEvents failed! err: %d\n", error);
                return error;
            }
            if ((error = clReleaseEvent(e[j])))
            {
                vlog_error("Error: clReleaseEvent failed! err: %d\n", error);
                return error;
            }
        }

        // Fill the result buffer with garbage, so that old results don't carry
        // over
        uint32_t pattern = 0xacdcacdc;
        if (gHostFill)
        {
            memset_pattern4(out[j], &pattern, buffer_size);
            error = clEnqueueUnmapMemObject(tinfo->tQueue, tinfo->outBuf[j],
                                            out[j], 0, NULL, NULL);
            test_error(error, "clEnqueueUnmapMemObject failed!\n");
        }
        else
        {
            error = clEnqueueFillBuffer(tinfo->tQueue, tinfo->outBuf[j],
                                        &pattern, sizeof(pattern), 0,
                                        buffer_size, 0, NULL, NULL);
            test_error(error, "clEnqueueFillBuffer failed!\n");
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

        if ((error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL,
                                            &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error("FAILED -- could not execute kernel\n");
            return error;
        }
    }


    // Get that moving
    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush 2 failed\n");

    if (gSkipCorrectnessTesting) return CL_SUCCESS;

    // Calculate the correctly rounded reference result
    cl_half *r = (cl_half *)gOut_Ref + thread_id * buffer_elements;
    s.resize(buffer_elements);
    for (j = 0; j < buffer_elements; j++)
    {
        s[j] = (float)cl_half_to_float(p[j]);
        r[j] = HFF(func.f_f(s[j]));
    }

    // Read the data back -- no need to wait for the first N-1 buffers. This is
    // an in order queue.
    for (j = gMinVectorSizeIndex; j + 1 < gMaxVectorSizeIndex; j++)
    {
        out[j] = (uint16_t *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_READ, 0,
            buffer_size, 0, NULL, NULL, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            return error;
        }
    }
    // Wait for the last buffer
    out[j] = (uint16_t *)clEnqueueMapBuffer(tinfo->tQueue, tinfo->outBuf[j],
                                            CL_TRUE, CL_MAP_READ, 0,
                                            buffer_size, 0, NULL, NULL, &error);
    if (error || NULL == out[j])
    {
        vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j, error);
        return error;
    }

    // Verify data
    for (j = 0; j < buffer_elements; j++)
    {
        for (k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
        {
            cl_ushort *q = out[k];

            // If we aren't getting the correctly rounded result
            if (r[j] != q[j])
            {
                float test = cl_half_to_float(q[j]);
                double correct = func.f_f(s[j]);
                float err = Ulp_Error_Half(q[j], correct);
                int fail = !(fabsf(err) <= ulps);

                // half_sin/cos/tan are only valid between +-2**16, Inf, NaN
                if (isRangeLimited
                    && fabsf(s[j]) > MAKE_HEX_FLOAT(0x1.0p16f, 0x1L, 16)
                    && fabsf(s[j]) < INFINITY)
                {
                    if (fabsf(test) <= half_sin_cos_tan_limit)
                    {
                        err = 0;
                        fail = 0;
                    }
                }

                if (fail)
                {
                    if (ftz)
                    {
                        // retry per section 6.5.3.2
                        if (IsHalfResultSubnormal(correct, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (IsHalfSubnormal(p[j]))
                        {
                            double correct2 = func.f_f(0.0);
                            double correct3 = func.f_f(-0.0);
                            float err2 = Ulp_Error_Half(q[j], correct2);
                            float err3 = Ulp_Error_Half(q[j], correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= ulps))
                                    && (!(fabsf(err3) <= ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (IsHalfResultSubnormal(correct2, ulps)
                                || IsHalfResultSubnormal(correct3, ulps))
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
                }
                if (fail)
                {
                    vlog_error("\nERROR: %s%s: %f ulp error at %a "
                               "(half 0x%04x)\nExpected: %a (half 0x%04x) "
                               "\nActual: %a (half 0x%04x)\n",
                               job->f->name, sizeNames[k], err, s[j], p[j],
                               cl_half_to_float(r[j]), r[j], test, q[j]);
                    error = -1;
                    return error;
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
            vlog("base:%14u step:%10u scale:%10u buf_elements:%10zd ulps:%5.3f "
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

    return error;
}

} // anonymous namespace

int TestFunc_Half_Half(const Func *f, MTdata d, bool relaxedMode)
{
    cl_int error;
    size_t i, j;
    float maxError = 0.0f;
    double maxErrorVal = 0.0;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);

    // Init test_info
    TestInfo test_info;

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
        test_info.jobCount =
            std::max((cl_uint)1,
                     (cl_uint)((1ULL << sizeof(cl_half) * 8) / test_info.step));
    }

    test_info.f = f;
    test_info.ulps = getAllowedUlpError(f, khalf, relaxedMode);
    test_info.ftz =
        f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);

    test_info.tinfo.resize(test_info.threadCount);

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
    }

    // Check for special cases for unary float
    test_info.isRangeLimited = 0;
    test_info.half_sin_cos_tan_limit = 0;
    if (0 == strcmp(f->name, "half_sin") || 0 == strcmp(f->name, "half_cos"))
    {
        test_info.isRangeLimited = 1;
        test_info.half_sin_cos_tan_limit = 1.0f
            + test_info.ulps
                * (FLT_EPSILON / 2.0f); // out of range results from finite
                                        // inputs must be in [-1,1]
    }
    else if (0 == strcmp(f->name, "half_tan"))
    {
        test_info.isRangeLimited = 1;
        test_info.half_sin_cos_tan_limit =
            INFINITY; // out of range resut from finite inputs must be numeric
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
            }
        }

        test_error(error, "ThreadPool_Do: TestHalf failed\n");

        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");
    }

    if (!gSkipCorrectnessTesting) vlog("\t%8.2f @ %a", maxError, maxErrorVal);
    vlog("\n");

    return error;
}
