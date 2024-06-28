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
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Float,
                              ParameterType::Int, ParameterType::Float,
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

int TestFunc_FloatI_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int64_t maxError2 = 0;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);
    int scale = (int)((1ULL << 32) / (16 * BUFFER_SIZE / sizeof(float)) + 1);
    cl_ulong maxiError;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    float float_ulps;
    if (gIsEmbedded)
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    maxiError = float_ulps == INFINITY ? CL_ULONG_MAX : 0;

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
        uint32_t *p = (uint32_t *)gIn;
        if (gWimpyMode)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
                p[j] = (uint32_t)i + j * scale;
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
                p[j] = (uint32_t)i + j;
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

        // Calculate the correctly rounded reference result
        float *r = (float *)gOut_Ref;
        int *r2 = (int *)gOut_Ref2;
        float *s = (float *)gIn;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            r[j] = (float)f->func.f_fpI(s[j], r2 + j);

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
        uint32_t *t = (uint32_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j] || t2[j] != q2[j])
                {
                    float test = ((float *)q)[j];
                    int correct2 = INT_MIN;
                    double correct = f->func.f_fpI(s[j], &correct2);
                    float err = Ulp_Error(test, correct);
                    cl_long iErr = (int64_t)q2[j] - (int64_t)correct2;
                    int fail = !(fabsf(err) <= float_ulps
                                 && abs_cl_long(iErr) <= maxiError);
                    if (ftz || relaxedMode)
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
                            int correct5, correct6;
                            double correct3 = f->func.f_fpI(0.0, &correct5);
                            double correct4 = f->func.f_fpI(-0.0, &correct6);
                            float err2 = Ulp_Error(test, correct3);
                            float err3 = Ulp_Error(test, correct4);
                            cl_long iErr2 =
                                (long long)q2[j] - (long long)correct5;
                            cl_long iErr3 =
                                (long long)q2[j] - (long long)correct6;

                            // Did +0 work?
                            if (fabsf(err2) <= float_ulps
                                && abs_cl_long(iErr2) <= maxiError)
                            {
                                err = err2;
                                iErr = iErr2;
                                fail = 0;
                            }
                            // Did -0 work?
                            else if (fabsf(err3) <= float_ulps
                                     && abs_cl_long(iErr3) <= maxiError)
                            {
                                err = err3;
                                iErr = iErr3;
                                fail = 0;
                            }

                            // retry per section 6.5.3.4
                            if (fail
                                && (IsFloatResultSubnormal(correct2, float_ulps)
                                    || IsFloatResultSubnormal(correct3,
                                                              float_ulps)))
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
                        maxErrorVal = s[j];
                    }
                    if (llabs(iErr) > maxError2)
                    {
                        maxError2 = llabs(iErr);
                        maxErrorVal2 = s[j];
                    }

                    if (fail)
                    {
                        vlog_error("\nERROR: %s%s: {%f, %d} ulp error at %a: "
                                   "*{%a, %d} vs. {%a, %d}\n",
                                   f->name, sizeNames[k], err, (int)iErr,
                                   ((float *)gIn)[j], ((float *)gOut_Ref)[j],
                                   ((int *)gOut_Ref2)[j], test, q2[j]);
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

        vlog("\t{%8.2f, %" PRId64 "} @ {%a, %a}", maxError, maxError2,
             maxErrorVal, maxErrorVal2);
    }

    vlog("\n");

    return CL_SUCCESS;
}
