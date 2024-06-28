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
        return GetBinaryKernel(kernel_name, builtin, ParameterType::Double,
                               ParameterType::Int, ParameterType::Double,
                               ParameterType::Double, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}


struct ComputeReferenceInfoD
{
    const double *x;
    const double *y;
    double *r;
    int *i;
    long double (*f_ffpI)(long double, long double, int *);
    cl_uint lim;
    cl_uint count;
};

cl_int ReferenceD(cl_uint jid, cl_uint tid, void *userInfo)
{
    ComputeReferenceInfoD *cri = (ComputeReferenceInfoD *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    const double *x = cri->x + off;
    const double *y = cri->y + off;
    double *r = cri->r + off;
    int *i = cri->i + off;
    long double (*f)(long double, long double, int *) = cri->f_ffpI;

    if (off + count > lim) count = lim - off;

    Force64BitFPUPrecision();

    for (cl_uint j = 0; j < count; ++j)
        r[j] = (double)f((long double)x[j], (long double)y[j], i + j);

    return CL_SUCCESS;
}

} // anonymous namespace

int TestFunc_DoubleI_Double_Double(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int64_t maxError2 = 0;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    double maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(double), BUFFER_SIZE);

    logFunctionInfo(f->name, sizeof(cl_double), relaxedMode);

    cl_uint threadCount = GetThreadCount();

    Force64BitFPUPrecision();

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
        if (gSkipCorrectnessTesting) break;

        // Init input array
        double *p = (double *)gIn;
        double *p2 = (double *)gIn2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
        {
            p[j] = DoubleFromUInt32(genrand_int32(d));
            p2[j] = DoubleFromUInt32(genrand_int32(d));
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
            size_t vectorSize = sizeof(cl_double) * sizeValues[j];
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
        double *s = (double *)gIn;
        double *s2 = (double *)gIn2;

        if (threadCount > 1)
        {
            ComputeReferenceInfoD cri;
            cri.x = s;
            cri.y = s2;
            cri.r = (double *)gOut_Ref;
            cri.i = (int *)gOut_Ref2;
            cri.f_ffpI = f->dfunc.f_ffpI;
            cri.lim = BUFFER_SIZE / sizeof(double);
            cri.count = (cri.lim + threadCount - 1) / threadCount;
            ThreadPool_Do(ReferenceD, threadCount, &cri);
        }
        else
        {
            double *r = (double *)gOut_Ref;
            int *r2 = (int *)gOut_Ref2;
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
                r[j] = (double)f->dfunc.f_ffpI(s[j], s2[j], r2 + j);
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

        // Verify data
        uint64_t *t = (uint64_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint64_t *q = (uint64_t *)gOut[k];
                int32_t *q2 = (int32_t *)gOut2[k];

                // Check for exact match to correctly rounded result
                if (t[j] == q[j] && t2[j] == q2[j]) continue;

                // Check for paired NaNs
                if ((t[j] & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL
                    && (q[j] & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL
                    && t2[j] == q2[j])
                    continue;

                double test = ((double *)q)[j];
                int correct2 = INT_MIN;
                long double correct = f->dfunc.f_ffpI(s[j], s2[j], &correct2);
                float err = Bruteforce_Ulp_Error_Double(test, correct);
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
                int iptrUndefined = fabs(((double *)gIn)[j]) == INFINITY
                    || ((double *)gIn2)[j] == 0.0 || isnan(((double *)gIn2)[j])
                    || isnan(((double *)gIn)[j]);
                if (iptrUndefined) iErr = 0;

                int fail = !(fabsf(err) <= f->double_ulps && iErr == 0);
                if ((ftz || relaxedMode) && fail)
                {
                    // retry per section 6.5.3.2
                    if (IsDoubleResultSubnormal(correct, f->double_ulps))
                    {
                        fail = fail && !(test == 0.0f && iErr == 0);
                        if (!fail) err = 0.0f;
                    }

                    // retry per section 6.5.3.3
                    if (IsDoubleSubnormal(s[j]))
                    {
                        int correct3i, correct4i;
                        long double correct3 =
                            f->dfunc.f_ffpI(0.0, s2[j], &correct3i);
                        long double correct4 =
                            f->dfunc.f_ffpI(-0.0, s2[j], &correct4i);
                        float err2 =
                            Bruteforce_Ulp_Error_Double(test, correct3);
                        float err3 =
                            Bruteforce_Ulp_Error_Double(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= f->double_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= f->double_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsDoubleResultSubnormal(correct2, f->double_ulps)
                            || IsDoubleResultSubnormal(correct3,
                                                       f->double_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsDoubleSubnormal(s2[j]))
                        {
                            int correct7i, correct8i;
                            correct3 = f->dfunc.f_ffpI(0.0, 0.0, &correct3i);
                            correct4 = f->dfunc.f_ffpI(-0.0, 0.0, &correct4i);
                            long double correct7 =
                                f->dfunc.f_ffpI(0.0, -0.0, &correct7i);
                            long double correct8 =
                                f->dfunc.f_ffpI(-0.0, -0.0, &correct8i);
                            err2 = Bruteforce_Ulp_Error_Double(test, correct3);
                            err3 = Bruteforce_Ulp_Error_Double(test, correct4);
                            float err4 =
                                Bruteforce_Ulp_Error_Double(test, correct7);
                            float err5 =
                                Bruteforce_Ulp_Error_Double(test, correct8);
                            iErr3 = (long long)q2[j] - (long long)correct3i;
                            iErr4 = (long long)q2[j] - (long long)correct4i;
                            int64_t iErr7 =
                                (long long)q2[j] - (long long)correct7i;
                            int64_t iErr8 =
                                (long long)q2[j] - (long long)correct8i;
                            fail = fail
                                && ((!(fabsf(err2) <= f->double_ulps
                                       && iErr3 == 0))
                                    && (!(fabsf(err3) <= f->double_ulps
                                          && iErr4 == 0))
                                    && (!(fabsf(err4) <= f->double_ulps
                                          && iErr7 == 0))
                                    && (!(fabsf(err5) <= f->double_ulps
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
                            if (IsDoubleResultSubnormal(correct3,
                                                        f->double_ulps)
                                || IsDoubleResultSubnormal(correct4,
                                                           f->double_ulps)
                                || IsDoubleResultSubnormal(correct7,
                                                           f->double_ulps)
                                || IsDoubleResultSubnormal(correct8,
                                                           f->double_ulps))
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && (iErr3 == 0 || iErr4 == 0
                                             || iErr7 == 0 || iErr8 == 0));
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsDoubleSubnormal(s2[j]))
                    {
                        int correct3i, correct4i;
                        long double correct3 =
                            f->dfunc.f_ffpI(s[j], 0.0, &correct3i);
                        long double correct4 =
                            f->dfunc.f_ffpI(s[j], -0.0, &correct4i);
                        float err2 =
                            Bruteforce_Ulp_Error_Double(test, correct3);
                        float err3 =
                            Bruteforce_Ulp_Error_Double(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= f->double_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= f->double_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsDoubleResultSubnormal(correct2, f->double_ulps)
                            || IsDoubleResultSubnormal(correct3,
                                                       f->double_ulps))
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
                    vlog_error("\nERROR: %sD%s: {%f, %" PRId64
                               "} ulp error at {%.13la, "
                               "%.13la} ({ 0x%16.16" PRIx64 ", 0x%16.16" PRIx64
                               "}): *{%.13la, "
                               "%d} ({ 0x%16.16" PRIx64
                               ", 0x%8.8x}) vs. {%.13la, %d} ({ "
                               "0x%16.16" PRIx64 ", 0x%8.8x})\n",
                               f->name, sizeNames[k], err, iErr,
                               ((double *)gIn)[j], ((double *)gIn2)[j],
                               ((cl_ulong *)gIn)[j], ((cl_ulong *)gIn2)[j],
                               ((double *)gOut_Ref)[j], ((int *)gOut_Ref2)[j],
                               ((cl_ulong *)gOut_Ref)[j],
                               ((cl_uint *)gOut_Ref2)[j], test, q2[j],
                               ((cl_ulong *)q)[j], ((cl_uint *)q2)[j]);
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
