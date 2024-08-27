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

#include <cstring>

namespace {

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Float,
                              ParameterType::Float, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

// Thread specific data for a worker thread
struct ThreadInfo
{
    // Input and output buffers for the thread
    clMemWrapper inBuf;
    Buffers outBuf;

    float maxError; // max error value. Init to 0.
    double maxErrorValue; // position of the max error value.  Init to 0.

    // Per thread command queue to improve performance
    clCommandQueueWrapper tQueue;
};

struct TestInfo
{
    size_t subBufferSize; // Size of the sub-buffer in elements
    const Func *f; // A pointer to the function info

    // Programs for various vector sizes.
    Programs programs;

    // Thread-specific kernels for each vector size:
    // k[vector_size][thread_id]
    KernelMatrix k;

    // Array of thread specific information
    std::vector<ThreadInfo> tinfo;

    cl_uint threadCount; // Number of worker threads
    cl_uint jobCount; // Number of jobs
    cl_uint step; // step between each chunk and the next.
    cl_uint scale; // stride between individual test values
    float ulps; // max_allowed ulps
    int ftz; // non-zero if running in flush to zero mode

    int isRangeLimited; // 1 if the function is only to be evaluated over a
                        // range
    float half_sin_cos_tan_limit;
    bool relaxedMode; // True if test is running in relaxed mode, false
                      // otherwise.
};

cl_int Test(cl_uint job_id, cl_uint thread_id, void *data)
{
    TestInfo *job = (TestInfo *)data;
    size_t buffer_elements = job->subBufferSize;
    size_t buffer_size = buffer_elements * sizeof(cl_float);
    cl_uint scale = job->scale;
    cl_uint base = job_id * (cl_uint)job->step;
    ThreadInfo *tinfo = &(job->tinfo[thread_id]);
    fptr func = job->f->func;
    const char *fname = job->f->name;
    bool relaxedMode = job->relaxedMode;
    float ulps = getAllowedUlpError(job->f, relaxedMode);
    if (relaxedMode)
    {
        func = job->f->rfunc;
    }

    cl_int error;

    int isRangeLimited = job->isRangeLimited;
    float half_sin_cos_tan_limit = job->half_sin_cos_tan_limit;
    int ftz = job->ftz;

    cl_event e[VECTOR_SIZE_COUNT];
    cl_uint *out[VECTOR_SIZE_COUNT];
    if (gHostFill)
    {
        // start the map of the output arrays
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
    }

    // Write the new values to the input array
    cl_uint *p = (cl_uint *)gIn + thread_id * buffer_elements;
    for (size_t j = 0; j < buffer_elements; j++)
    {
        p[j] = base + j * scale;
        if (relaxedMode)
        {
            float p_j = *(float *)&p[j];
            if (strcmp(fname, "sin") == 0
                || strcmp(fname, "cos")
                    == 0) // the domain of the function is [-pi,pi]
            {
                if (fabs(p_j) > M_PI) ((float *)p)[j] = NAN;
            }

            if (strcmp(fname, "reciprocal") == 0)
            {
                const float l_limit = HEX_FLT(+, 1, 0, -, 126);
                const float u_limit = HEX_FLT(+, 1, 0, +, 126);

                if (fabs(p_j) < l_limit
                    || fabs(p_j) > u_limit) // the domain of the function is
                                            // [2^-126,2^126]
                    ((float *)p)[j] = NAN;
            }
        }
    }

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0,
                                      buffer_size, p, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        return error;
    }

    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
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
        uint32_t pattern = 0xffffdead;
        if (gHostFill)
        {
            memset_pattern4(out[j], &pattern, buffer_size);
            if ((error = clEnqueueUnmapMemObject(
                     tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL)))
            {
                vlog_error("Error: clEnqueueUnmapMemObject failed! err: %d\n",
                           error);
                return error;
            }
        }
        else
        {
            if ((error = clEnqueueFillBuffer(tinfo->tQueue, tinfo->outBuf[j],
                                             &pattern, sizeof(pattern), 0,
                                             buffer_size, 0, NULL, NULL)))
            {
                vlog_error("Error: clEnqueueFillBuffer failed! err: %d\n",
                           error);
                return error;
            }
        }

        // Run the kernel
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
    float *r = (float *)gOut_Ref + thread_id * buffer_elements;
    float *s = (float *)p;
    for (size_t j = 0; j < buffer_elements; j++) r[j] = (float)func.f_f(s[j]);

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
            return error;
        }
    }

    // Verify data
    uint32_t *t = (uint32_t *)r;
    for (size_t j = 0; j < buffer_elements; j++)
    {
        for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
        {
            uint32_t *q = out[k];

            // If we aren't getting the correctly rounded result
            if (t[j] != q[j])
            {
                float test = ((float *)q)[j];
                double correct = func.f_f(s[j]);
                float err = Ulp_Error(test, correct);
                float abs_error = Abs_Error(test, correct);
                int fail = 0;
                int use_abs_error = 0;

                // it is possible for the output to not match the reference
                // result but for Ulp_Error to be zero, for example -1.#QNAN
                // vs. 1.#QNAN. In such cases there is no failure
                if (err == 0.0f)
                {
                    fail = 0;
                }
                else if (relaxedMode)
                {
                    if (strcmp(fname, "sin") == 0 || strcmp(fname, "cos") == 0)
                    {
                        fail = !(fabsf(abs_error) <= ulps);
                        use_abs_error = 1;
                    }
                    if (strcmp(fname, "sinpi") == 0
                        || strcmp(fname, "cospi") == 0)
                    {
                        if (s[j] >= -1.0 && s[j] <= 1.0)
                        {
                            fail = !(fabsf(abs_error) <= ulps);
                            use_abs_error = 1;
                        }
                    }

                    if (strcmp(fname, "reciprocal") == 0)
                    {
                        fail = !(fabsf(err) <= ulps);
                    }

                    if (strcmp(fname, "exp") == 0 || strcmp(fname, "exp2") == 0)
                    {
                        // For full profile, ULP depends on input value.
                        // For embedded profile, ULP comes from functionList.
                        if (!gIsEmbedded)
                        {
                            ulps = 3.0f + floor(fabs(2 * s[j]));
                        }

                        fail = !(fabsf(err) <= ulps);
                    }
                    if (strcmp(fname, "tan") == 0)
                    {

                        if (!gFastRelaxedDerived)
                        {
                            fail = !(fabsf(err) <= ulps);
                        }
                        // Else fast math derived implementation does not
                        // require ULP verification
                    }
                    if (strcmp(fname, "exp10") == 0)
                    {
                        if (!gFastRelaxedDerived)
                        {
                            fail = !(fabsf(err) <= ulps);
                        }
                        // Else fast math derived implementation does not
                        // require ULP verification
                    }
                    if (strcmp(fname, "log") == 0 || strcmp(fname, "log2") == 0
                        || strcmp(fname, "log10") == 0)
                    {
                        if (s[j] >= 0.5 && s[j] <= 2)
                        {
                            fail = !(fabsf(abs_error) <= ulps);
                        }
                        else
                        {
                            ulps = gIsEmbedded ? job->f->float_embedded_ulps
                                               : job->f->float_ulps;
                            fail = !(fabsf(err) <= ulps);
                        }
                    }


                    // fast-relaxed implies finite-only
                    if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                        || IsFloatInfinity(s[j]) || IsFloatNaN(s[j]))
                    {
                        fail = 0;
                        err = 0;
                    }
                }
                else
                {
                    fail = !(fabsf(err) <= ulps);
                }

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
                    if (ftz || relaxedMode)
                    {
                        typedef int (*CheckForSubnormal)(
                            double, float); // If we are in fast relaxed math,
                                            // we have a different calculation
                                            // for the subnormal threshold.
                        CheckForSubnormal isFloatResultSubnormalPtr;

                        if (relaxedMode)
                        {
                            isFloatResultSubnormalPtr =
                                &IsFloatResultSubnormalAbsError;
                        }
                        else
                        {
                            isFloatResultSubnormalPtr = &IsFloatResultSubnormal;
                        }
                        // retry per section 6.5.3.2
                        if ((*isFloatResultSubnormalPtr)(correct, ulps))
                        {
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (IsFloatSubnormal(s[j]))
                        {
                            double correct2 = func.f_f(0.0);
                            double correct3 = func.f_f(-0.0);
                            float err2;
                            float err3;
                            if (use_abs_error)
                            {
                                err2 = Abs_Error(test, correct2);
                                err3 = Abs_Error(test, correct3);
                            }
                            else
                            {
                                err2 = Ulp_Error(test, correct2);
                                err3 = Ulp_Error(test, correct3);
                            }
                            fail = fail
                                && ((!(fabsf(err2) <= ulps))
                                    && (!(fabsf(err3) <= ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if ((*isFloatResultSubnormalPtr)(correct2, ulps)
                                || (*isFloatResultSubnormalPtr)(correct3, ulps))
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
                    vlog_error("\nERROR: %s%s: %f ulp error at %a (0x%8.8x): "
                               "*%a vs. %a\n",
                               job->f->name, sizeNames[k], err, ((float *)s)[j],
                               ((uint32_t *)s)[j], ((float *)t)[j], test);
                    return -1;
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

    return CL_SUCCESS;
}

} // anonymous namespace

int TestFunc_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    TestInfo test_info{};
    cl_int error;
    float maxError = 0.0f;
    double maxErrorVal = 0.0;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    // Init test_info
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
    test_info.tinfo.resize(test_info.threadCount);
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
            return error;
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
    BuildKernelInfo build_info{ test_info.threadCount, test_info.k,
                                test_info.programs, f->nameInCode,
                                relaxedMode };
    if ((error = ThreadPool_Do(BuildKernelFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    // Run the kernels
    if (!gSkipCorrectnessTesting)
    {
        error = ThreadPool_Do(Test, test_info.jobCount, &test_info);
        if (error) return error;

        // Accumulate the arithmetic errors
        for (cl_uint i = 0; i < test_info.threadCount; i++)
        {
            if (test_info.tinfo[i].maxError > maxError)
            {
                maxError = test_info.tinfo[i].maxError;
                maxErrorVal = test_info.tinfo[i].maxErrorValue;
            }
        }

        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");

        vlog("\t%8.2f @ %a", maxError, maxErrorVal);
    }

    vlog("\n");

    return CL_SUCCESS;
}
