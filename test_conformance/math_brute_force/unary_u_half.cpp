//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "reference_math.h"

#include <cstring>
#include <cinttypes>

////////////////////////////////////////////////////////////////////////////////
static cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED,
                                 void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Half,
                              ParameterType::UShort, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

////////////////////////////////////////////////////////////////////////////////
int TestFunc_Half_UShort(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    KernelMatrix kernels;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);
    float maxErrorVal = 0.0f;
    size_t bufferSize = BUFFER_SIZE;
    size_t bufferElements = bufferSize / sizeof(cl_half);
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);
    int scale = (int)((1ULL << 32) / (16 * bufferElements) + 1);
    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);
    const char *name = f->name;
    float half_ulps = f->half_ulps;
    if (gWimpyMode)
    {
        step = (1ULL << 32) * gWimpyReductionFactor / (512);
    }

    // Init the kernels
    BuildKernelInfo build_info = { 1, kernels, programs, f->nameInCode };
    if ((error = ThreadPool_Do(BuildKernel_HalfFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
    {
        return error;
    }

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        cl_ushort *p = (cl_ushort *)gIn;
        if (gWimpyMode)
        {
            for (size_t j = 0; j < bufferElements; j++) p[j] = i + j * scale;
        }
        else
        {
            for (size_t j = 0; j < bufferElements; j++) p[j] = (uint16_t)i + j;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        // write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint16_t pattern = 0xdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if ((error =
                     clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0,
                                          bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                           error, j);
                return error;
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
            if ((error = clSetKernelArg(kernels[j][thread_id], 1,
                                        sizeof(gInBuffer), &gInBuffer)))
            {
                LogBuildError(programs[j]);
                return error;
            }

            if ((error = clEnqueueNDRangeKernel(gQueue, kernels[j][thread_id],
                                                1, NULL, &localCount, NULL, 0,
                                                NULL, NULL)))
            {
                vlog_error("FAILURE -- could not execute kernel\n");
                return error;
            }
        }

        // Get that moving
        if ((error = clFlush(gQueue))) vlog("clFlush failed\n");

        // Calculate the correctly rounded reference result
        cl_half *r = (cl_half *)gOut_Ref;
        for (size_t j = 0; j < bufferElements; j++)
        {
            if (!strcmp(name, "nan"))
                r[j] = reference_nanh(p[j]);
            else
                r[j] = cl_half_from_float(f->func.f_u(p[j]), CL_HALF_RTE);
        }
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
        }

        if (gSkipCorrectnessTesting) break;

        // Verify data
        cl_ushort *t = (cl_ushort *)gOut_Ref;
        for (size_t j = 0; j < bufferElements; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                cl_ushort *q = (cl_ushort *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    double test = cl_half_to_float(q[j]);
                    double correct;
                    if (!strcmp(name, "nan"))
                        correct = cl_half_to_float(reference_nanh(p[j]));
                    else
                        correct = f->func.f_u(p[j]);

                    float err = Ulp_Error_Half(q[j], correct);
                    int fail = !(fabsf(err) <= half_ulps);

                    if (fail)
                    {
                        if (ftz)
                        {
                            // retry per section 6.5.3.2
                            if (IsHalfSubnormal(
                                    cl_half_from_float(correct, CL_HALF_RTE)))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    if (fabsf(err) > maxError)
                    {
                        maxError = fabsf(err);
                        maxErrorVal = p[j];
                    }
                    if (fail)
                    {
                        vlog_error(
                            "\n%s%s: %f ulp error at 0x%0.4x \nExpected: %a "
                            "(0x%0.4x) \nActual: %a (0x%0.4x)\n",
                            f->name, sizeNames[k], err, p[j],
                            cl_half_to_float(r[j]), r[j], test, q[j]);
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
                     "  bufferSize:%10zd \n",
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
    }

    if (!gSkipCorrectnessTesting) vlog("\t%8.2f @ %a", maxError, maxErrorVal);
    vlog("\n");

    return error;
}
