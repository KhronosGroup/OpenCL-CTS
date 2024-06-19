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

#include <cinttypes>
#include <climits>
#include <cstring>

namespace {

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetBinaryKernel(kernel_name, builtin, ParameterType::Float,
                               ParameterType::Int, ParameterType::Float,
                               ParameterType::Float, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

struct ComputeReferenceInfoF
{
    const float *x;
    const float *y;
    float *r;
    int *i;
    double (*f_ffpI)(double, double, int *);
    cl_uint lim;
    cl_uint count;
};

cl_int ReferenceF(cl_uint jid, cl_uint tid, void *userInfo)
{
    ComputeReferenceInfoF *cri = (ComputeReferenceInfoF *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    const float *x = cri->x + off;
    const float *y = cri->y + off;
    float *r = cri->r + off;
    int *i = cri->i + off;
    double (*f)(double, double, int *) = cri->f_ffpI;

    if (off + count > lim) count = lim - off;

    for (cl_uint j = 0; j < count; ++j)
        r[j] = (float)f((double)x[j], (double)y[j], i + j);

    return CL_SUCCESS;
}

} // anonymous namespace

int TestFunc_FloatI_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    int64_t maxError2 = 0;
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);

    cl_uint threadCount = GetThreadCount();

    float float_ulps;
    if (gIsEmbedded)
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    int testingRemquo = !strcmp(f->name, "remquo");

    // Init the kernels
    BuildKernelInfo build_info{ 1, kernels, programs, f->nameInCode,
                                relaxedMode };
    if ((error = ThreadPool_Do(BuildKernelFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        cl_uint *p = (cl_uint *)gIn;
        cl_uint *p2 = (cl_uint *)gIn2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            p[j] = genrand_int32(d);
            p2[j] = genrand_int32(d);
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

        // Write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint32_t pattern = 0xffffdead;
            if (gHostFill)
            {
                memset_pattern4(gOut[j], &pattern, BUFFER_SIZE);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j],
                                                  CL_FALSE, 0, BUFFER_SIZE,
                                                  gOut[j], 0, NULL, NULL)))
                {
                    vlog_error(
                        "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                        error, j);
                    return error;
                }

                memset_pattern4(gOut2[j], &pattern, BUFFER_SIZE);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j],
                                                  CL_FALSE, 0, BUFFER_SIZE,
                                                  gOut2[j], 0, NULL, NULL)))
                {
                    vlog_error(
                        "\n*** Error %d in clEnqueueWriteBuffer2b(%d) ***\n",
                        error, j);
                    return error;
                }
            }
            else
            {
                if ((error = clEnqueueFillBuffer(gQueue, gOutBuffer[j],
                                                 &pattern, sizeof(pattern), 0,
                                                 BUFFER_SIZE, 0, NULL, NULL)))
                {
                    vlog_error("Error: clEnqueueFillBuffer 1 failed! err: %d\n",
                               error);
                    return error;
                }

                if ((error = clEnqueueFillBuffer(gQueue, gOutBuffer2[j],
                                                 &pattern, sizeof(pattern), 0,
                                                 BUFFER_SIZE, 0, NULL, NULL)))
                {
                    vlog_error("Error: clEnqueueFillBuffer 2 failed! err: %d\n",
                               error);
                    return error;
                }
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeof(cl_float) * sizeValues[j];
            size_t localCount = (BUFFER_SIZE + vectorSize - 1)
                / vectorSize; // BUFFER_SIZE / vectorSize  rounded up
            if ((error = clSetKernelArg(kernels[j][thread_id], 0,
                                        sizeof(gOutBuffer[j]), &gOutBuffer[j])))
            {
                LogBuildError(programs[j]);
                return error;
            }
            if ((error =
                     clSetKernelArg(kernels[j][thread_id], 1,
                                    sizeof(gOutBuffer2[j]), &gOutBuffer2[j])))
            {
                LogBuildError(programs[j]);
                return error;
            }
            if ((error = clSetKernelArg(kernels[j][thread_id], 2,
                                        sizeof(gInBuffer), &gInBuffer)))
            {
                LogBuildError(programs[j]);
                return error;
            }
            if ((error = clSetKernelArg(kernels[j][thread_id], 3,
                                        sizeof(gInBuffer2), &gInBuffer2)))
            {
                LogBuildError(programs[j]);
                return error;
            }

            if ((error = clEnqueueNDRangeKernel(gQueue, kernels[j][thread_id],
                                                1, NULL, &localCount, NULL, 0,
                                                NULL, NULL)))
            {
                vlog_error("FAILED -- could not execute kernel\n");
                return error;
            }
        }

        // Get that moving
        if ((error = clFlush(gQueue))) vlog("clFlush failed\n");

        // Calculate the correctly rounded reference result
        float *s = (float *)gIn;
        float *s2 = (float *)gIn2;

        if (threadCount > 1)
        {
            ComputeReferenceInfoF cri;
            cri.x = s;
            cri.y = s2;
            cri.r = (float *)gOut_Ref;
            cri.i = (int *)gOut_Ref2;
            cri.f_ffpI = f->func.f_ffpI;
            cri.lim = BUFFER_SIZE / sizeof(float);
            cri.count = (cri.lim + threadCount - 1) / threadCount;
            ThreadPool_Do(ReferenceF, threadCount, &cri);
        }
        else
        {
            float *r = (float *)gOut_Ref;
            int *r2 = (int *)gOut_Ref2;
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
                r[j] = (float)f->func.f_ffpI(s[j], s2[j], r2 + j);
        }

        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0,
                                         BUFFER_SIZE, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                return error;
            }
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer2[j], CL_TRUE, 0,
                                         BUFFER_SIZE, gOut2[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray2 failed %d\n", error);
                return error;
            }
        }

        if (gSkipCorrectnessTesting) break;

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)gOut2[k];

                // Check for exact match to correctly rounded result
                if (t[j] == q[j] && t2[j] == q2[j]) continue;

                // Check for paired NaNs
                if ((t[j] & 0x7fffffff) > 0x7f800000
                    && (q[j] & 0x7fffffff) > 0x7f800000 && t2[j] == q2[j])
                    continue;

                float test = ((float *)q)[j];
                int correct2 = INT_MIN;
                double correct = f->func.f_ffpI(s[j], s2[j], &correct2);
                float err = Ulp_Error(test, correct);
                int64_t iErr;

                // in case of remquo, we only care about the sign and last
                // seven bits of integer as per the spec.
                if (testingRemquo)
                    iErr = (long long)(q2[j] & 0x0000007f)
                        - (long long)(correct2 & 0x0000007f);
                else
                    iErr = (long long)q2[j] - (long long)correct2;

                // For remquo, if y = 0, x is infinite, or either is NaN
                // then the standard either neglects to say what is returned
                // in iptr or leaves it undefined or implementation defined.
                int iptrUndefined = fabs(((float *)gIn)[j]) == INFINITY
                    || ((float *)gIn2)[j] == 0.0f || isnan(((float *)gIn2)[j])
                    || isnan(((float *)gIn)[j]);
                if (iptrUndefined) iErr = 0;

                int fail = !(fabsf(err) <= float_ulps && iErr == 0);
                if ((ftz || relaxedMode) && fail)
                {
                    // retry per section 6.5.3.2
                    if (IsFloatResultSubnormal(correct, float_ulps))
                    {
                        fail = fail && !(test == 0.0f && iErr == 0);
                        if (!fail) err = 0.0f;
                    }

                    // retry per section 6.5.3.3
                    if (IsFloatSubnormal(s[j]))
                    {
                        int correct3i, correct4i;
                        double correct3 =
                            f->func.f_ffpI(0.0, s2[j], &correct3i);
                        double correct4 =
                            f->func.f_ffpI(-0.0, s2[j], &correct4i);
                        float err2 = Ulp_Error(test, correct3);
                        float err3 = Ulp_Error(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= float_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, float_ulps)
                            || IsFloatResultSubnormal(correct3, float_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsFloatSubnormal(s2[j]))
                        {
                            int correct7i, correct8i;
                            correct3 = f->func.f_ffpI(0.0, 0.0, &correct3i);
                            correct4 = f->func.f_ffpI(-0.0, 0.0, &correct4i);
                            double correct7 =
                                f->func.f_ffpI(0.0, -0.0, &correct7i);
                            double correct8 =
                                f->func.f_ffpI(-0.0, -0.0, &correct8i);
                            err2 = Ulp_Error(test, correct3);
                            err3 = Ulp_Error(test, correct4);
                            float err4 = Ulp_Error(test, correct7);
                            float err5 = Ulp_Error(test, correct8);
                            iErr3 = (long long)q2[j] - (long long)correct3i;
                            iErr4 = (long long)q2[j] - (long long)correct4i;
                            int64_t iErr7 =
                                (long long)q2[j] - (long long)correct7i;
                            int64_t iErr8 =
                                (long long)q2[j] - (long long)correct8i;
                            fail = fail
                                && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                    && (!(fabsf(err3) <= float_ulps
                                          && iErr4 == 0))
                                    && (!(fabsf(err4) <= float_ulps
                                          && iErr7 == 0))
                                    && (!(fabsf(err5) <= float_ulps
                                          && iErr8 == 0)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;
                            if (fabsf(err4) < fabsf(err)) err = err4;
                            if (fabsf(err5) < fabsf(err)) err = err5;
                            if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                            if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;
                            if (llabs(iErr7) < llabs(iErr)) iErr = iErr7;
                            if (llabs(iErr8) < llabs(iErr)) iErr = iErr8;

                            // retry per section 6.5.3.4
                            if (IsFloatResultSubnormal(correct3, float_ulps)
                                || IsFloatResultSubnormal(correct4, float_ulps)
                                || IsFloatResultSubnormal(correct7, float_ulps)
                                || IsFloatResultSubnormal(correct8, float_ulps))
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && (iErr3 == 0 || iErr4 == 0
                                             || iErr7 == 0 || iErr8 == 0));
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsFloatSubnormal(s2[j]))
                    {
                        int correct3i, correct4i;
                        double correct3 = f->func.f_ffpI(s[j], 0.0, &correct3i);
                        double correct4 =
                            f->func.f_ffpI(s[j], -0.0, &correct4i);
                        float err2 = Ulp_Error(test, correct3);
                        float err3 = Ulp_Error(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= float_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, float_ulps)
                            || IsFloatResultSubnormal(correct3, float_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
                        }
                    }
                }
                if (fabsf(err) > maxError)
                {
                    maxError = fabsf(err);
                    maxErrorVal = s[j];
                }
                if (llabs(iErr) > maxError2)
                {
                    maxError2 = llabs(iErr);
                    maxErrorVal2 = s[j];
                }

                if (fail)
                {
                    vlog_error("\nERROR: %s%s: {%f, %" PRId64
                               "} ulp error at {%a, %a} "
                               "({0x%8.8x, 0x%8.8x}): *{%a, %d} ({0x%8.8x, "
                               "0x%8.8x}) vs. {%a, %d} ({0x%8.8x, 0x%8.8x})\n",
                               f->name, sizeNames[k], err, iErr,
                               ((float *)gIn)[j], ((float *)gIn2)[j],
                               ((cl_uint *)gIn)[j], ((cl_uint *)gIn2)[j],
                               ((float *)gOut_Ref)[j], ((int *)gOut_Ref2)[j],
                               ((cl_uint *)gOut_Ref)[j],
                               ((cl_uint *)gOut_Ref2)[j], test, q2[j],
                               ((cl_uint *)&test)[0], ((cl_uint *)q2)[j]);
                    return -1;
                }
            }
        }

        if (0 == (i & 0x0fffffff))
        {
            if (gVerboseBruteForce)
            {
                vlog("base:%14" PRIu64 " step:%10" PRIu64
                     "  bufferSize:%10d \n",
                     i, step, BUFFER_SIZE);
            }
            else
            {
                vlog(".");
            }
            fflush(stdout);
        }
    }

    if (!gSkipCorrectnessTesting)
    {
        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");

        vlog("\t{%8.2f, %" PRId64 "} @ {%a, %a}", maxError, maxError2,
             maxErrorVal, maxErrorVal2);
    }

    vlog("\n");

    return CL_SUCCESS;
}
