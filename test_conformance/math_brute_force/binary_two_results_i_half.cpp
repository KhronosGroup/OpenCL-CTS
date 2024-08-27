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

#include <cinttypes>
#include <climits>
#include <cstring>

namespace {

cl_int BuildKernelFn_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetBinaryKernel(kernel_name, builtin, ParameterType::Half,
                               ParameterType::Int, ParameterType::Half,
                               ParameterType::Half, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

struct ComputeReferenceInfoF
{
    const cl_half *x;
    const cl_half *y;
    cl_half *r;
    int32_t *i;
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
    const cl_half *x = cri->x + off;
    const cl_half *y = cri->y + off;
    cl_half *r = cri->r + off;
    int32_t *i = cri->i + off;
    double (*f)(double, double, int *) = cri->f_ffpI;

    if (off + count > lim) count = lim - off;

    for (cl_uint j = 0; j < count; ++j)
        r[j] = HFF((float)f((double)HTF(x[j]), (double)HTF(y[j]), i + j));

    return CL_SUCCESS;
}

} // anonymous namespace

int TestFunc_HalfI_Half_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);

    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);
    int64_t maxError2 = 0;
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);

    // use larger type of output data to prevent overflowing buffer size
    constexpr size_t buffer_size = BUFFER_SIZE / sizeof(int32_t);

    cl_uint threadCount = GetThreadCount();

    float half_ulps = f->half_ulps;

    int testingRemquo = !strcmp(f->name, "remquo");

    // Init the kernels
    BuildKernelInfo build_info{ 1, kernels, programs, f->nameInCode };
    if ((error = ThreadPool_Do(BuildKernelFn_HalfFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        if (gSkipCorrectnessTesting) break;

        // Init input array
        cl_half *p = (cl_half *)gIn;
        cl_half *p2 = (cl_half *)gIn2;
        for (size_t j = 0; j < buffer_size; j++)
        {
            p[j] = (cl_half)genrand_int32(d);
            p2[j] = (cl_half)genrand_int32(d);
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          buffer_size * sizeof(cl_half), gIn, 0,
                                          NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0,
                                          buffer_size * sizeof(cl_half), gIn2,
                                          0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error);
            return error;
        }

        // Write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint32_t pattern = 0xacdcacdc;
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
                error = clEnqueueFillBuffer(gQueue, gOutBuffer[j], &pattern,
                                            sizeof(pattern), 0, BUFFER_SIZE, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 1 failed!\n");

                error = clEnqueueFillBuffer(gQueue, gOutBuffer2[j], &pattern,
                                            sizeof(pattern), 0, BUFFER_SIZE, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 2 failed!\n");
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            // align working group size with the bigger output type
            size_t vectorSize = sizeValues[j] * sizeof(int32_t);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
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

        if (threadCount > 1)
        {
            ComputeReferenceInfoF cri;
            cri.x = p;
            cri.y = p2;
            cri.r = (cl_half *)gOut_Ref;
            cri.i = (int32_t *)gOut_Ref2;
            cri.f_ffpI = f->func.f_ffpI;
            cri.lim = buffer_size;
            cri.count = (cri.lim + threadCount - 1) / threadCount;
            ThreadPool_Do(ReferenceF, threadCount, &cri);
        }
        else
        {
            cl_half *r = (cl_half *)gOut_Ref;
            int32_t *r2 = (int32_t *)gOut_Ref2;
            for (size_t j = 0; j < buffer_size; j++)
                r[j] =
                    HFF((float)f->func.f_ffpI(HTF(p[j]), HTF(p2[j]), r2 + j));
        }

        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            cl_bool blocking =
                (j + 1 < gMaxVectorSizeIndex) ? CL_FALSE : CL_TRUE;
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], blocking, 0,
                                         BUFFER_SIZE, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                return error;
            }
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer2[j], blocking, 0,
                                         BUFFER_SIZE, gOut2[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray2 failed %d\n", error);
                return error;
            }
        }

        // Verify data
        cl_half *t = (cl_half *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < buffer_size; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                cl_half *q = (cl_half *)(gOut[k]);
                int32_t *q2 = (int32_t *)gOut2[k];

                // Check for exact match to correctly rounded result
                if (t[j] == q[j] && t2[j] == q2[j]) continue;

                // Check for paired NaNs
                if (IsHalfNaN(t[j]) && IsHalfNaN(q[j]) && t2[j] == q2[j])
                    continue;

                cl_half test = ((cl_half *)q)[j];
                int correct2 = INT_MIN;
                float correct =
                    (float)f->func.f_ffpI(HTF(p[j]), HTF(p2[j]), &correct2);
                float err = Ulp_Error_Half(test, correct);
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
                int iptrUndefined = IsHalfInfinity(p[j]) || (HTF(p2[j]) == 0.0f)
                    || IsHalfNaN(p2[j]) || IsHalfNaN(p[j]);
                if (iptrUndefined) iErr = 0;

                int fail = !(fabsf(err) <= half_ulps && iErr == 0);
                if (ftz && fail)
                {
                    // retry per section 6.5.3.2
                    if (IsHalfResultSubnormal(correct, half_ulps))
                    {
                        fail = fail && !(test == 0.0f && iErr == 0);
                        if (!fail) err = 0.0f;
                    }

                    // retry per section 6.5.3.3
                    if (IsHalfSubnormal(p[j]))
                    {
                        int correct3i, correct4i;
                        float correct3 =
                            (float)f->func.f_ffpI(0.0, HTF(p2[j]), &correct3i);
                        float correct4 =
                            (float)f->func.f_ffpI(-0.0, HTF(p2[j]), &correct4i);
                        float err2 = Ulp_Error_Half(test, correct3);
                        float err3 = Ulp_Error_Half(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= half_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= half_ulps && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsHalfResultSubnormal(correct2, half_ulps)
                            || IsHalfResultSubnormal(correct3, half_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsHalfSubnormal(p2[j]))
                        {
                            int correct7i, correct8i;
                            correct3 = f->func.f_ffpI(0.0, 0.0, &correct3i);
                            correct4 = f->func.f_ffpI(-0.0, 0.0, &correct4i);
                            double correct7 =
                                f->func.f_ffpI(0.0, -0.0, &correct7i);
                            double correct8 =
                                f->func.f_ffpI(-0.0, -0.0, &correct8i);
                            err2 = Ulp_Error_Half(test, correct3);
                            err3 = Ulp_Error_Half(test, correct4);
                            float err4 = Ulp_Error_Half(test, correct7);
                            float err5 = Ulp_Error_Half(test, correct8);
                            iErr3 = (long long)q2[j] - (long long)correct3i;
                            iErr4 = (long long)q2[j] - (long long)correct4i;
                            int64_t iErr7 =
                                (long long)q2[j] - (long long)correct7i;
                            int64_t iErr8 =
                                (long long)q2[j] - (long long)correct8i;
                            fail = fail
                                && ((!(fabsf(err2) <= half_ulps && iErr3 == 0))
                                    && (!(fabsf(err3) <= half_ulps
                                          && iErr4 == 0))
                                    && (!(fabsf(err4) <= half_ulps
                                          && iErr7 == 0))
                                    && (!(fabsf(err5) <= half_ulps
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
                            if (IsHalfResultSubnormal(correct3, half_ulps)
                                || IsHalfResultSubnormal(correct4, half_ulps)
                                || IsHalfResultSubnormal(correct7, half_ulps)
                                || IsHalfResultSubnormal(correct8, half_ulps))
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && (iErr3 == 0 || iErr4 == 0
                                             || iErr7 == 0 || iErr8 == 0));
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsHalfSubnormal(p2[j]))
                    {
                        int correct3i, correct4i;
                        double correct3 =
                            f->func.f_ffpI(HTF(p[j]), 0.0, &correct3i);
                        double correct4 =
                            f->func.f_ffpI(HTF(p[j]), -0.0, &correct4i);
                        float err2 = Ulp_Error_Half(test, correct3);
                        float err3 = Ulp_Error_Half(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= half_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= half_ulps && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsHalfResultSubnormal(correct2, half_ulps)
                            || IsHalfResultSubnormal(correct3, half_ulps))
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
                    maxErrorVal = HTF(p[j]);
                }
                if (llabs(iErr) > maxError2)
                {
                    maxError2 = llabs(iErr);
                    maxErrorVal2 = HTF(p[j]);
                }

                if (fail)
                {
                    vlog_error("\nERROR: %s%s: {%f, %" PRId64
                               "} ulp error at {%a, %a} "
                               "({0x%04x, 0x%04x}): *{%a, %d} ({0x%04x, "
                               "0x%8.8x}) vs. {%a, %d} ({0x%04x, 0x%8.8x})\n",
                               f->name, sizeNames[k], err, iErr, HTF(p[j]),
                               HTF(p2[j]), p[j], p2[j], HTF(t[j]), t2[j], t[j],
                               t2[j], HTF(test), q2[j], test, q2[j]);
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
