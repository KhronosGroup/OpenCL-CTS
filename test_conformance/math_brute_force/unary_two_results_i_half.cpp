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
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Half,
                              ParameterType::Int, ParameterType::Half,
                              vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

cl_ulong abs_cl_long(cl_long i)
{
    cl_long mask = i >> 63;
    return (i ^ mask) - mask;
}

} // anonymous namespace

int TestFunc_HalfI_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int64_t maxError2 = 0;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);

    // sizeof(cl_half) < sizeof (int32_t)
    // to prevent overflowing gOut_Ref2 it is necessary to use
    // bigger type as denominator for buffer size calculation
    size_t bufferElements = std::min(BUFFER_SIZE / sizeof(cl_int),
                                     size_t(1ULL << (sizeof(cl_half) * 8)));

    size_t bufferSizeLo = bufferElements * sizeof(cl_half);
    size_t bufferSizeHi = bufferElements * sizeof(cl_int);

    cl_ulong maxiError = 0;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);

    float half_ulps = getAllowedUlpError(f, khalf, relaxedMode);

    maxiError = half_ulps == INFINITY ? CL_ULONG_MAX : 0;

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
                                          bufferSizeLo, gIn, 0, NULL, NULL)))
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
                memset_pattern4(gOut[j], &pattern, bufferSizeLo);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j],
                                                  CL_FALSE, 0, bufferSizeLo,
                                                  gOut[j], 0, NULL, NULL)))
                {
                    vlog_error(
                        "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                        error, j);
                    return error;
                }

                memset_pattern4(gOut2[j], &pattern, bufferSizeHi);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j],
                                                  CL_FALSE, 0, bufferSizeHi,
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
                                            sizeof(pattern), 0, bufferSizeLo, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 1 failed!\n");

                error = clEnqueueFillBuffer(gQueue, gOutBuffer2[j], &pattern,
                                            sizeof(pattern), 0, bufferSizeHi, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer 2 failed!\n");
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            // align working group size with the bigger output type
            size_t vectorSize = sizeValues[j] * sizeof(cl_int);
            size_t localCount = (bufferSizeHi + vectorSize - 1) / vectorSize;
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

        // Calculate the correctly rounded reference result
        cl_half *ref1 = (cl_half *)gOut_Ref;
        int32_t *ref2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < bufferElements; j++)
            ref1[j] = HFF((float)f->func.f_fpI(HTF(pIn[j]), ref2 + j));

        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            cl_bool blocking =
                (j + 1 < gMaxVectorSizeIndex) ? CL_FALSE : CL_TRUE;
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], blocking, 0,
                                         bufferSizeLo, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                return error;
            }
            if ((error = clEnqueueReadBuffer(gQueue, gOutBuffer2[j], blocking,
                                             0, bufferSizeHi, gOut2[j], 0, NULL,
                                             NULL)))
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
                cl_half *test1 = (cl_half *)(gOut[k]);
                int32_t *test2 = (int32_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if (ref1[j] != test1[j] || ref2[j] != test2[j])
                {
                    cl_half test = ((cl_half *)test1)[j];
                    int correct2 = INT_MIN;
                    float fp_correct =
                        (float)f->func.f_fpI(HTF(pIn[j]), &correct2);
                    cl_half correct = HFF(fp_correct);
                    float err = correct != test
                        ? Ulp_Error_Half(test, fp_correct)
                        : 0.f;
                    cl_long iErr = (int64_t)test2[j] - (int64_t)correct2;
                    int fail = !(fabsf(err) <= half_ulps
                                 && abs_cl_long(iErr) <= maxiError);
                    if (ftz)
                    {
                        // retry per section 6.5.3.2
                        if (IsHalfResultSubnormal(fp_correct, half_ulps))
                        {
                            fail = fail && !(test == 0.0f && iErr == 0);
                            if (!fail) err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (IsHalfSubnormal(pIn[j]))
                        {
                            int correct5, correct6;
                            double fp_correct3 = f->func.f_fpI(0.0, &correct5);
                            double fp_correct4 = f->func.f_fpI(-0.0, &correct6);

                            float err2 = Ulp_Error_Half(test, fp_correct3);
                            float err3 = Ulp_Error_Half(test, fp_correct4);

                            cl_long iErr2 =
                                (long long)test2[j] - (long long)correct5;
                            cl_long iErr3 =
                                (long long)test2[j] - (long long)correct6;

                            // Did +0 work?
                            if (fabsf(err2) <= half_ulps
                                && abs_cl_long(iErr2) <= maxiError)
                            {
                                err = err2;
                                iErr = iErr2;
                                fail = 0;
                            }
                            // Did -0 work?
                            else if (fabsf(err3) <= half_ulps
                                     && abs_cl_long(iErr3) <= maxiError)
                            {
                                err = err3;
                                iErr = iErr3;
                                fail = 0;
                            }

                            // retry per section 6.5.3.4
                            if (fail
                                && (IsHalfResultSubnormal(correct2, half_ulps)
                                    || IsHalfResultSubnormal(fp_correct3,
                                                             half_ulps)))
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && (abs_cl_long(iErr2) <= maxiError
                                             || abs_cl_long(iErr3)
                                                 <= maxiError));
                                if (!fail)
                                {
                                    err = 0.0f;
                                    iErr = 0;
                                }
                            }
                        }
                    }
                    if (fabsf(err) > maxError)
                    {
                        maxError = fabsf(err);
                        maxErrorVal = pIn[j];
                    }
                    if (llabs(iErr) > maxError2)
                    {
                        maxError2 = llabs(iErr);
                        maxErrorVal2 = pIn[j];
                    }

                    if (fail)
                    {
                        vlog_error("\nERROR: %s%s: {%f, %d} ulp error at %a: "
                                   "*{%a, %d} vs. {%a, %d}\n",
                                   f->name, sizeNames[k], err, (int)iErr,
                                   HTF(pIn[j]), HTF(ref1[j]),
                                   ((int *)gOut_Ref2)[j], HTF(test), test2[j]);
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
                     "  bufferSize:%10zu \n",
                     i, step, bufferSizeHi);
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
