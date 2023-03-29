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
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Double,
                              ParameterType::Double, ParameterType::Double,
                              vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

} // anonymous namespace

int TestFunc_Double2_Double(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError0 = 0.0f;
    float maxError1 = 0.0f;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal0 = 0.0f;
    double maxErrorVal1 = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_double), BUFFER_SIZE);
    int scale =
        (int)((1ULL << 32) / (16 * BUFFER_SIZE / sizeof(cl_double)) + 1);

    logFunctionInfo(f->name, sizeof(cl_double), relaxedMode);

    Force64BitFPUPrecision();

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
        double *p = (double *)gIn;
        if (gWimpyMode)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_double); j++)
                p[j] = DoubleFromUInt32((uint32_t)i + j * scale);
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_double); j++)
                p[j] = DoubleFromUInt32((uint32_t)i + j);
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
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
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

        // Calculate the correctly rounded reference result
        double *r = (double *)gOut_Ref;
        double *r2 = (double *)gOut_Ref2;
        double *s = (double *)gIn;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_double); j++)
        {
            long double dd;
            r[j] = (double)f->dfunc.f_fpf(s[j], &dd);
            r2[j] = (double)dd;
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
        uint64_t *t = (uint64_t *)gOut_Ref;
        uint64_t *t2 = (uint64_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint64_t *q = (uint64_t *)(gOut[k]);
                uint64_t *q2 = (uint64_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j] || t2[j] != q2[j])
                {
                    double test = ((double *)q)[j];
                    double test2 = ((double *)q2)[j];
                    long double correct2;
                    long double correct = f->dfunc.f_fpf(s[j], &correct2);
                    float err = Bruteforce_Ulp_Error_Double(test, correct);
                    float err2 = Bruteforce_Ulp_Error_Double(test2, correct2);
                    int fail = !(fabsf(err) <= f->double_ulps
                                 && fabsf(err2) <= f->double_ulps);
                    if (ftz || relaxedMode)
                    {
                        // retry per section 6.5.3.2
                        if (IsDoubleResultSubnormal(correct, f->double_ulps))
                        {
                            if (IsDoubleResultSubnormal(correct2,
                                                        f->double_ulps))
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
                                         && fabsf(err2) <= f->double_ulps);
                                if (!fail) err = 0.0f;
                            }
                        }
                        else if (IsDoubleResultSubnormal(correct2,
                                                         f->double_ulps))
                        {
                            fail = fail
                                && !(test2 == 0.0f
                                     && fabsf(err) <= f->double_ulps);
                            if (!fail) err2 = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (IsDoubleSubnormal(s[j]))
                        {
                            long double correct2p, correct2n;
                            long double correctp =
                                f->dfunc.f_fpf(0.0, &correct2p);
                            long double correctn =
                                f->dfunc.f_fpf(-0.0, &correct2n);
                            float errp =
                                Bruteforce_Ulp_Error_Double(test, correctp);
                            float err2p =
                                Bruteforce_Ulp_Error_Double(test, correct2p);
                            float errn =
                                Bruteforce_Ulp_Error_Double(test, correctn);
                            float err2n =
                                Bruteforce_Ulp_Error_Double(test, correct2n);
                            fail = fail
                                && ((!(fabsf(errp) <= f->double_ulps))
                                    && (!(fabsf(err2p) <= f->double_ulps))
                                    && ((!(fabsf(errn) <= f->double_ulps))
                                        && (!(fabsf(err2n)
                                              <= f->double_ulps))));
                            if (fabsf(errp) < fabsf(err)) err = errp;
                            if (fabsf(errn) < fabsf(err)) err = errn;
                            if (fabsf(err2p) < fabsf(err2)) err2 = err2p;
                            if (fabsf(err2n) < fabsf(err2)) err2 = err2n;

                            // retry per section 6.5.3.4
                            if (IsDoubleResultSubnormal(correctp,
                                                        f->double_ulps)
                                || IsDoubleResultSubnormal(correctn,
                                                           f->double_ulps))
                            {
                                if (IsDoubleResultSubnormal(correct2p,
                                                            f->double_ulps)
                                    || IsDoubleResultSubnormal(correct2n,
                                                               f->double_ulps))
                                {
                                    fail = fail
                                        && !(test == 0.0f && test2 == 0.0f);
                                    if (!fail) err = err2 = 0.0f;
                                }
                                else
                                {
                                    fail = fail
                                        && !(test == 0.0f
                                             && fabsf(err2) <= f->double_ulps);
                                    if (!fail) err = 0.0f;
                                }
                            }
                            else if (IsDoubleResultSubnormal(correct2p,
                                                             f->double_ulps)
                                     || IsDoubleResultSubnormal(correct2n,
                                                                f->double_ulps))
                            {
                                fail = fail
                                    && !(test2 == 0.0f
                                         && (fabsf(err) <= f->double_ulps));
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
                        vlog_error(
                            "\nERROR: %sD%s: {%f, %f} ulp error at %.13la: "
                            "*{%.13la, %.13la} vs. {%.13la, %.13la}\n",
                            f->name, sizeNames[k], err, err2,
                            ((double *)gIn)[j], ((double *)gOut_Ref)[j],
                            ((double *)gOut_Ref2)[j], test, test2);
                        return -1;
                    }
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

        vlog("\t{%8.2f, %8.2f} @ {%a, %a}", maxError0, maxError1, maxErrorVal0,
             maxErrorVal1);
    }

    vlog("\n");

    return CL_SUCCESS;
}
