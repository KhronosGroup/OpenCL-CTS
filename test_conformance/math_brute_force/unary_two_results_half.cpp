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
#include <cstring>

namespace {

cl_int BuildKernelFn_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Half,
                              ParameterType::Half, ParameterType::Half,
                              vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

} // anonymous namespace

int TestFunc_Half2_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError0 = 0.0f;
    float maxError1 = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);
    float maxErrorVal0 = 0.0f;
    float maxErrorVal1 = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);

    size_t bufferElements = std::min(BUFFER_SIZE / sizeof(cl_half),
                                     size_t(1ULL << (sizeof(cl_half) * 8)));
    size_t bufferSize = bufferElements * sizeof(cl_half);

    std::vector<cl_uchar> overflow(bufferElements);
    int isFract = 0 == strcmp("fract", f->nameInCode);
    int skipNanInf = isFract;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);

    float half_ulps = getAllowedUlpError(f, khalf, relaxedMode);

    // Init the kernels
    BuildKernelInfo build_info{ 1, kernels, programs, f->nameInCode };
    if ((error = ThreadPool_Do(BuildKernelFn_HalfFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    for (uint64_t i = 0; i < (1ULL << 16); i += step)
    {
        if (gSkipCorrectnessTesting) break;

        // Init input array
        cl_half *pIn = (cl_half *)gIn;
        for (size_t j = 0; j < bufferElements; j++) pIn[j] = (cl_ushort)i + j;

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        // Write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint32_t pattern = 0xacdcacdc;
            if (gHostFill)
            {
                memset_pattern4(gOut[j], &pattern, bufferSize);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j],
                                                  CL_FALSE, 0, bufferSize,
                                                  gOut[j], 0, NULL, NULL)))
                {
                    vlog_error(
                        "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                        error, j);
                    return error;
                }

                memset_pattern4(gOut2[j], &pattern, bufferSize);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j],
                                                  CL_FALSE, 0, bufferSize,
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
                                            sizeof(pattern), 0, bufferSize, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 1 failed!\n");

                error = clEnqueueFillBuffer(gQueue, gOutBuffer[j], &pattern,
                                            sizeof(pattern), 0, bufferSize, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 2 failed!\n");
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_half);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
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
        if ((error = clFlush(gQueue)))
        {
            vlog_error("clFlush failed\n");
            return error;
        }

        FPU_mode_type oldMode;
        RoundingMode oldRoundMode = kRoundToNearestEven;
        if (isFract)
        {
            // Calculate the correctly rounded reference result
            memset(&oldMode, 0, sizeof(oldMode));
            if (ftz) ForceFTZ(&oldMode);

            // Set the rounding mode to match the device
            if (gIsInRTZMode)
                oldRoundMode = set_round(kRoundTowardZero, kfloat);
        }

        // Calculate the correctly rounded reference result
        cl_half *ref1 = (cl_half *)gOut_Ref;
        cl_half *ref2 = (cl_half *)gOut_Ref2;

        if (skipNanInf)
        {
            for (size_t j = 0; j < bufferElements; j++)
            {
                double dd;
                feclearexcept(FE_OVERFLOW);

                ref1[j] = HFF((float)f->func.f_fpf(HTF(pIn[j]), &dd));
                ref2[j] = HFF((float)dd);

                // ensure correct rounding of fract result is not reaching 1
                if (isFract && HTF(ref1[j]) >= 1.f) ref1[j] = 0x3bff;

                overflow[j] =
                    FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for (size_t j = 0; j < bufferElements; j++)
            {
                double dd;
                ref1[j] = HFF((float)f->func.f_fpf(HTF(pIn[j]), &dd));
                ref2[j] = HFF((float)dd);
            }
        }

        if (isFract && ftz) RestoreFPState(&oldMode);

        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0,
                                         bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                return error;
            }
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer2[j], CL_TRUE, 0,
                                         bufferSize, gOut2[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray2 failed %d\n", error);
                return error;
            }
        }

        // Verify data
        for (size_t j = 0; j < bufferElements; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                cl_half *test1 = (cl_half *)gOut[k];
                cl_half *test2 = (cl_half *)gOut2[k];

                // If we aren't getting the correctly rounded result
                if (ref1[j] != test1[j] || ref2[j] != test2[j])
                {
                    double fp_correct1 = 0, fp_correct2 = 0;
                    float err = 0, err2 = 0;

                    fp_correct1 = f->func.f_fpf(HTF(pIn[j]), &fp_correct2);

                    cl_half correct1 = HFF(fp_correct1);
                    cl_half correct2 = HFF(fp_correct2);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow
                    if (skipNanInf)
                    {
                        if (skipNanInf && overflow[j]) continue;
                        // Note: no double rounding here.  Reference functions
                        // calculate in single precision.
                        if (IsHalfInfinity(correct1) || IsHalfNaN(correct1)
                            || IsHalfInfinity(correct2) || IsHalfNaN(correct2)
                            || IsHalfInfinity(pIn[j]) || IsHalfNaN(pIn[j]))
                            continue;
                    }

                    err = Ulp_Error_Half(test1[j], fp_correct1);
                    err2 = Ulp_Error_Half(test2[j], fp_correct2);

                    int fail =
                        !(fabsf(err) <= half_ulps && fabsf(err2) <= half_ulps);

                    if (ftz)
                    {
                        // retry per section 6.5.3.2
                        if (IsHalfResultSubnormal(fp_correct1, half_ulps))
                        {
                            if (IsHalfResultSubnormal(fp_correct2, half_ulps))
                            {
                                fail = fail
                                    && !(HTF(test1[j]) == 0.0f
                                         && HTF(test2[j]) == 0.0f);
                                if (!fail)
                                {
                                    err = 0.0f;
                                    err2 = 0.0f;
                                }
                            }
                            else
                            {
                                fail = fail
                                    && !(HTF(test1[j]) == 0.0f
                                         && fabsf(err2) <= half_ulps);
                                if (!fail) err = 0.0f;
                            }
                        }
                        else if (IsHalfResultSubnormal(fp_correct2, half_ulps))
                        {
                            fail = fail
                                && !(HTF(test2[j]) == 0.0f
                                     && fabsf(err) <= half_ulps);
                            if (!fail) err2 = 0.0f;
                        }


                        // retry per section 6.5.3.3
                        if (IsHalfSubnormal(pIn[j]))
                        {
                            double fp_correctp, fp_correctn;
                            double fp_correct2p, fp_correct2n;
                            float errp, err2p, errn, err2n;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);
                            fp_correctp = f->func.f_fpf(0.0, &fp_correct2p);
                            fp_correctn = f->func.f_fpf(-0.0, &fp_correct2n);

                            cl_half correctp = HFF(fp_correctp);
                            cl_half correctn = HFF(fp_correctn);
                            cl_half correct2p = HFF(fp_correct2p);
                            cl_half correct2n = HFF(fp_correct2n);

                            // Per section 10 paragraph 6, accept any result if
                            // an input or output is a infinity or NaN or
                            // overflow
                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsHalfInfinity(correctp)
                                    || IsHalfNaN(correctp)
                                    || IsHalfInfinity(correctn)
                                    || IsHalfNaN(correctn)
                                    || IsHalfInfinity(correct2p)
                                    || IsHalfNaN(correct2p)
                                    || IsHalfInfinity(correct2n)
                                    || IsHalfNaN(correct2n))
                                    continue;
                            }

                            errp = Ulp_Error_Half(test1[j], fp_correctp);
                            err2p = Ulp_Error_Half(test1[j], fp_correct2p);
                            errn = Ulp_Error_Half(test1[j], fp_correctn);
                            err2n = Ulp_Error_Half(test1[j], fp_correct2n);

                            fail = fail
                                && ((!(fabsf(errp) <= half_ulps))
                                    && (!(fabsf(err2p) <= half_ulps))
                                    && ((!(fabsf(errn) <= half_ulps))
                                        && (!(fabsf(err2n) <= half_ulps))));
                            if (fabsf(errp) < fabsf(err)) err = errp;
                            if (fabsf(errn) < fabsf(err)) err = errn;
                            if (fabsf(err2p) < fabsf(err2)) err2 = err2p;
                            if (fabsf(err2n) < fabsf(err2)) err2 = err2n;

                            // retry per section 6.5.3.4
                            if (IsHalfResultSubnormal(fp_correctp, half_ulps)
                                || IsHalfResultSubnormal(fp_correctn,
                                                         half_ulps))
                            {
                                if (IsHalfResultSubnormal(fp_correct2p,
                                                          half_ulps)
                                    || IsHalfResultSubnormal(fp_correct2n,
                                                             half_ulps))
                                {
                                    fail = fail
                                        && !(HTF(test1[j]) == 0.0f
                                             && HTF(test2[j]) == 0.0f);
                                    if (!fail) err = err2 = 0.0f;
                                }
                                else
                                {
                                    fail = fail
                                        && !(HTF(test1[j]) == 0.0f
                                             && fabsf(err2) <= half_ulps);
                                    if (!fail) err = 0.0f;
                                }
                            }
                            else if (IsHalfResultSubnormal(fp_correct2p,
                                                           half_ulps)
                                     || IsHalfResultSubnormal(fp_correct2n,
                                                              half_ulps))
                            {
                                fail = fail
                                    && !(HTF(test2[j]) == 0.0f
                                         && (fabsf(err) <= half_ulps));
                                if (!fail) err2 = 0.0f;
                            }
                        }
                    }
                    if (fabsf(err) > maxError0)
                    {
                        maxError0 = fabsf(err);
                        maxErrorVal0 = HTF(pIn[j]);
                    }
                    if (fabsf(err2) > maxError1)
                    {
                        maxError1 = fabsf(err2);
                        maxErrorVal1 = HTF(pIn[j]);
                    }
                    if (fail)
                    {
                        vlog_error("\nERROR: %s%s: {%f, %f} ulp error at %a: "
                                   "*{%a, %a} vs. {%a, %a}\n",
                                   f->name, sizeNames[k], err, err2,
                                   HTF(pIn[j]), HTF(ref1[j]), HTF(ref2[j]),
                                   HTF(test1[j]), HTF(test2[j]));
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
                     "  bufferSize:%10zu \n",
                     i, step, bufferSize);
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
