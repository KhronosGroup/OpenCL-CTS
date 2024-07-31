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
#include <cstring>

namespace {

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Float,
                              ParameterType::Float, ParameterType::Float,
                              vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

} // anonymous namespace

int TestFunc_Float2_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError0 = 0.0f;
    float maxError1 = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal0 = 0.0f;
    float maxErrorVal1 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);
    int scale = (int)((1ULL << 32) / (16 * BUFFER_SIZE / sizeof(float)) + 1);
    cl_uchar overflow[BUFFER_SIZE / sizeof(float)];
    int isFract = 0 == strcmp("fract", f->nameInCode);
    int skipNanInf = isFract && !gInfNanSupport;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    float float_ulps = getAllowedUlpError(f, relaxedMode);
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
        uint32_t *p = (uint32_t *)gIn;
        if (gWimpyMode)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                p[j] = (uint32_t)i + j * scale;
                if (relaxedMode && strcmp(f->name, "sincos") == 0)
                {
                    float pj = *(float *)&p[j];
                    if (fabs(pj) > M_PI) ((float *)p)[j] = NAN;
                }
            }
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                p[j] = (uint32_t)i + j;
                if (relaxedMode && strcmp(f->name, "sincos") == 0)
                {
                    float pj = *(float *)&p[j];
                    if (fabs(pj) > M_PI) ((float *)p)[j] = NAN;
                }
            }
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
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

                if ((error = clEnqueueFillBuffer(gQueue, gOutBuffer[j],
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
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
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

        FPU_mode_type oldMode = 0;
        RoundingMode oldRoundMode = kRoundToNearestEven;
        if (isFract)
        {
            // Calculate the correctly rounded reference result
            if (ftz || relaxedMode) ForceFTZ(&oldMode);

            // Set the rounding mode to match the device
            if (gIsInRTZMode)
                oldRoundMode = set_round(kRoundTowardZero, kfloat);
        }

        // Calculate the correctly rounded reference result
        float *r = (float *)gOut_Ref;
        float *r2 = (float *)gOut_Ref2;
        float *s = (float *)gIn;

        if (skipNanInf)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                double dd;
                feclearexcept(FE_OVERFLOW);

                if (relaxedMode)
                    r[j] = (float)f->rfunc.f_fpf(s[j], &dd);
                else
                    r[j] = (float)f->func.f_fpf(s[j], &dd);

                r2[j] = (float)dd;
                overflow[j] =
                    FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                double dd;
                if (relaxedMode)
                    r[j] = (float)f->rfunc.f_fpf(s[j], &dd);
                else
                    r[j] = (float)f->func.f_fpf(s[j], &dd);

                r2[j] = (float)dd;
            }
        }

        if (isFract && ftz) RestoreFPState(&oldMode);

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

        if (gSkipCorrectnessTesting)
        {
            if (isFract && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);
            break;
        }

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        uint32_t *t2 = (uint32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)gOut[k];
                uint32_t *q2 = (uint32_t *)gOut2[k];

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j] || t2[j] != q2[j])
                {
                    double correct, correct2;
                    float err, err2;
                    float test = ((float *)q)[j];
                    float test2 = ((float *)q2)[j];

                    if (relaxedMode)
                        correct = f->rfunc.f_fpf(s[j], &correct2);
                    else
                        correct = f->func.f_fpf(s[j], &correct2);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow
                    if (relaxedMode || skipNanInf)
                    {
                        if (skipNanInf && overflow[j]) continue;
                        // Note: no double rounding here.  Reference functions
                        // calculate in single precision.
                        if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                            || IsFloatInfinity(correct2) || IsFloatNaN(correct2)
                            || IsFloatInfinity(s[j]) || IsFloatNaN(s[j]))
                            continue;
                    }

                    typedef int (*CheckForSubnormal)(
                        double, float); // If we are in fast relaxed math, we
                                        // have a different calculation for the
                                        // subnormal threshold.
                    CheckForSubnormal isFloatResultSubnormalPtr;
                    if (relaxedMode)
                    {
                        err = Abs_Error(test, correct);
                        err2 = Abs_Error(test2, correct2);
                        isFloatResultSubnormalPtr =
                            &IsFloatResultSubnormalAbsError;
                    }
                    else
                    {
                        err = Ulp_Error(test, correct);
                        err2 = Ulp_Error(test2, correct2);
                        isFloatResultSubnormalPtr = &IsFloatResultSubnormal;
                    }
                    int fail = !(fabsf(err) <= float_ulps
                                 && fabsf(err2) <= float_ulps);

                    if (ftz || relaxedMode)
                    {
                        // retry per section 6.5.3.2
                        if ((*isFloatResultSubnormalPtr)(correct, float_ulps))
                        {
                            if ((*isFloatResultSubnormalPtr)(correct2,
                                                             float_ulps))
                            {
                                fail = fail && !(test == 0.0f && test2 == 0.0f);
                                if (!fail)
                                {
                                    err = 0.0f;
                                    err2 = 0.0f;
                                }
                            }
                            else
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && fabsf(err2) <= float_ulps);
                                if (!fail) err = 0.0f;
                            }
                        }
                        else if ((*isFloatResultSubnormalPtr)(correct2,
                                                              float_ulps))
                        {
                            fail = fail
                                && !(test2 == 0.0f && fabsf(err) <= float_ulps);
                            if (!fail) err2 = 0.0f;
                        }


                        // retry per section 6.5.3.3
                        if (IsFloatSubnormal(s[j]))
                        {
                            double correctp, correctn;
                            double correct2p, correct2n;
                            float errp, err2p, errn, err2n;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);
                            if (relaxedMode)
                            {
                                correctp = f->rfunc.f_fpf(0.0, &correct2p);
                                correctn = f->rfunc.f_fpf(-0.0, &correct2n);
                            }
                            else
                            {
                                correctp = f->func.f_fpf(0.0, &correct2p);
                                correctn = f->func.f_fpf(-0.0, &correct2n);
                            }

                            // Per section 10 paragraph 6, accept any result if
                            // an input or output is a infinity or NaN or
                            // overflow
                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correctp)
                                    || IsFloatNaN(correctp)
                                    || IsFloatInfinity(correctn)
                                    || IsFloatNaN(correctn)
                                    || IsFloatInfinity(correct2p)
                                    || IsFloatNaN(correct2p)
                                    || IsFloatInfinity(correct2n)
                                    || IsFloatNaN(correct2n))
                                    continue;
                            }

                            if (relaxedMode)
                            {
                                errp = Abs_Error(test, correctp);
                                err2p = Abs_Error(test, correct2p);
                                errn = Abs_Error(test, correctn);
                                err2n = Abs_Error(test, correct2n);
                            }
                            else
                            {
                                errp = Ulp_Error(test, correctp);
                                err2p = Ulp_Error(test, correct2p);
                                errn = Ulp_Error(test, correctn);
                                err2n = Ulp_Error(test, correct2n);
                            }

                            fail = fail
                                && ((!(fabsf(errp) <= float_ulps))
                                    && (!(fabsf(err2p) <= float_ulps))
                                    && ((!(fabsf(errn) <= float_ulps))
                                        && (!(fabsf(err2n) <= float_ulps))));
                            if (fabsf(errp) < fabsf(err)) err = errp;
                            if (fabsf(errn) < fabsf(err)) err = errn;
                            if (fabsf(err2p) < fabsf(err2)) err2 = err2p;
                            if (fabsf(err2n) < fabsf(err2)) err2 = err2n;

                            // retry per section 6.5.3.4
                            if ((*isFloatResultSubnormalPtr)(correctp,
                                                             float_ulps)
                                || (*isFloatResultSubnormalPtr)(correctn,
                                                                float_ulps))
                            {
                                if ((*isFloatResultSubnormalPtr)(correct2p,
                                                                 float_ulps)
                                    || (*isFloatResultSubnormalPtr)(correct2n,
                                                                    float_ulps))
                                {
                                    fail = fail
                                        && !(test == 0.0f && test2 == 0.0f);
                                    if (!fail) err = err2 = 0.0f;
                                }
                                else
                                {
                                    fail = fail
                                        && !(test == 0.0f
                                             && fabsf(err2) <= float_ulps);
                                    if (!fail) err = 0.0f;
                                }
                            }
                            else if ((*isFloatResultSubnormalPtr)(correct2p,
                                                                  float_ulps)
                                     || (*isFloatResultSubnormalPtr)(
                                         correct2n, float_ulps))
                            {
                                fail = fail
                                    && !(test2 == 0.0f
                                         && (fabsf(err) <= float_ulps));
                                if (!fail) err2 = 0.0f;
                            }
                        }
                    }
                    if (fabsf(err) > maxError0)
                    {
                        maxError0 = fabsf(err);
                        maxErrorVal0 = s[j];
                    }
                    if (fabsf(err2) > maxError1)
                    {
                        maxError1 = fabsf(err2);
                        maxErrorVal1 = s[j];
                    }
                    if (fail)
                    {
                        vlog_error("\nERROR: %s%s: {%f, %f} ulp error at %a: "
                                   "*{%a, %a} vs. {%a, %a}\n",
                                   f->name, sizeNames[k], err, err2,
                                   ((float *)gIn)[j], ((float *)gOut_Ref)[j],
                                   ((float *)gOut_Ref2)[j], test, test2);
                        return -1;
                    }
                }
            }
        }

        if (isFract && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);

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

        vlog("\t{%8.2f, %8.2f} @ {%a, %a}", maxError0, maxError1, maxErrorVal0,
             maxErrorVal1);
    }

    vlog("\n");

    return CL_SUCCESS;
}
