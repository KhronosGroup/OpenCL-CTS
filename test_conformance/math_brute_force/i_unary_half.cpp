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

#include <algorithm>
#include <cstring>
#include <memory>
#include <cinttypes>

namespace {

static cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED,
                                 void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetUnaryKernel(kernel_name, builtin, ParameterType::Int,
                              ParameterType::Half, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

} // anonymous namespace

int TestFunc_Int_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    KernelMatrix kernels;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    int ftz = f->ftz || 0 == (gHalfCapabilities & CL_FP_DENORM) || gForceFTZ;
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);
    size_t bufferElements = std::min(BUFFER_SIZE / sizeof(cl_int),
                                     size_t(1ULL << (sizeof(cl_half) * 8)));
    size_t bufferSizeIn = bufferElements * sizeof(cl_half);
    size_t bufferSizeOut = bufferElements * sizeof(cl_int);

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);
    // This test is not using ThreadPool so we need to disable FTZ here
    // for reference computations
    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);
    std::shared_ptr<int> at_scope_exit(
        nullptr, [&oldMode](int *) { RestoreFPState(&oldMode); });

    // Init the kernels
    {
        BuildKernelInfo build_info = { 1, kernels, programs, f->nameInCode };
        if ((error = ThreadPool_Do(BuildKernel_HalfFn,
                                   gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                   &build_info)))
            return error;
    }
    std::vector<float> s(bufferElements);

    for (uint64_t i = 0; i < (1ULL << 16); i += step)
    {
        if (gSkipCorrectnessTesting) break;

        // Init input array
        cl_ushort *p = (cl_ushort *)gIn;

        for (size_t j = 0; j < bufferElements; j++) p[j] = (cl_ushort)i + j;

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          bufferSizeIn, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        // write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint32_t pattern = 0xacdcacdc;
            if (gHostFill)
            {
                memset_pattern4(gOut[j], &pattern, bufferSizeOut);
                if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer[j],
                                                  CL_FALSE, 0, bufferSizeOut,
                                                  gOut[j], 0, NULL, NULL)))
                {
                    vlog_error(
                        "\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                        error, j);
                    return error;
                }
            }
            else
            {
                error = clEnqueueFillBuffer(gQueue, gOutBuffer[j], &pattern,
                                            sizeof(pattern), 0, bufferSizeOut,
                                            0, NULL, NULL);
                test_error(error, "clEnqueueFillBuffer failed!\n");
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_int);
            size_t localCount = (bufferSizeOut + vectorSize - 1) / vectorSize;
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
                vlog_error("FAILED -- could not execute kernel\n");
                return error;
            }
        }

        // Get that moving
        if ((error = clFlush(gQueue))) vlog("clFlush failed\n");

        // Calculate the correctly rounded reference result
        int *r = (int *)gOut_Ref;
        for (size_t j = 0; j < bufferElements; j++)
        {
            s[j] = HTF(p[j]);
            r[j] = f->func.i_f(s[j]);
        }
        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            if ((error = clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0,
                                             bufferSizeOut, gOut[j], 0, NULL,
                                             NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                return error;
            }
        }

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        for (size_t j = 0; j < bufferElements; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    if (ftz && IsHalfSubnormal(p[j]))
                    {
                        unsigned int correct0 = f->func.i_f(0.0);
                        unsigned int correct1 = f->func.i_f(-0.0);
                        if (q[j] == correct0 || q[j] == correct1) continue;
                    }

                    uint32_t err = t[j] - q[j];
                    if (q[j] > t[j]) err = q[j] - t[j];
                    vlog_error("\nERROR: %s%s: %d ulp error at %a (0x%04x): "
                               "*%d vs. %d\n",
                               f->name, sizeNames[k], err, s[j], p[j], t[j],
                               q[j]);
                    return -1;
                }
            }
        }

        if (0 == (i & 0x0fffffff))
        {
            if (gVerboseBruteForce)
            {
                vlog("base:%14" PRIu64 " step:%10" PRIu64
                     "  bufferSize:%10zd \n",
                     i, step, bufferSizeOut);
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

    vlog("\n");

    return error;
}
