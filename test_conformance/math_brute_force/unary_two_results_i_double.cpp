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
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Double,
                              ParameterType::Int, ParameterType::Double,
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

int TestFunc_DoubleI_Double(const Func *f, MTdata d, bool relaxedMode)
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
    cl_ulong maxiError = f->double_ulps == INFINITY ? CL_ULONG_MAX : 0;
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
        if (gSkipCorrectnessTesting) break;

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
        int *r2 = (int *)gOut_Ref2;
        double *s = (double *)gIn;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
            r[j] = (double)f->dfunc.f_fpI(s[j], r2 + j);

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
                uint64_t *q = (uint64_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)(gOut2[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j] || t2[j] != q2[j])
                {
                    double test = ((double *)q)[j];
                    int correct2 = INT_MIN;
                    long double correct = f->dfunc.f_fpI(s[j], &correct2);
                    float err = Bruteforce_Ulp_Error_Double(test, correct);
                    cl_long iErr = (long long)q2[j] - (long long)correct2;
                    int fail = !(fabsf(err) <= f->double_ulps
                                 && abs_cl_long(iErr) <= maxiError);
                    if (ftz || relaxedMode)
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
                            int correct5, correct6;
                            long double correct3 =
                                f->dfunc.f_fpI(0.0, &correct5);
                            long double correct4 =
                                f->dfunc.f_fpI(-0.0, &correct6);
                            float err2 =
                                Bruteforce_Ulp_Error_Double(test, correct3);
                            float err3 =
                                Bruteforce_Ulp_Error_Double(test, correct4);
                            cl_long iErr2 =
                                (long long)q2[j] - (long long)correct5;
                            cl_long iErr3 =
                                (long long)q2[j] - (long long)correct6;

                            // Did +0 work?
                            if (fabsf(err2) <= f->double_ulps
                                && abs_cl_long(iErr2) <= maxiError)
                            {
                                err = err2;
                                iErr = iErr2;
                                fail = 0;
                            }
                            // Did -0 work?
                            else if (fabsf(err3) <= f->double_ulps
                                     && abs_cl_long(iErr3) <= maxiError)
                            {
                                err = err3;
                                iErr = iErr3;
                                fail = 0;
                            }

                            // retry per section 6.5.3.4
                            if (fail
                                && (IsDoubleResultSubnormal(correct2,
                                                            f->double_ulps)
                                    || IsDoubleResultSubnormal(correct3,
                                                               f->double_ulps)))
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
                        vlog_error("\nERROR: %sD%s: {%f, %d} ulp error at "
                                   "%.13la: *{%.13la, %d} vs. {%.13la, %d}\n",
                                   f->name, sizeNames[k], err, (int)iErr,
                                   ((double *)gIn)[j], ((double *)gOut_Ref)[j],
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
