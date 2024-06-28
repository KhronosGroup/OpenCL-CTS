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

#include <cstring>

namespace {

cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetTernaryKernel(kernel_name, builtin, ParameterType::Half,
                                ParameterType::Half, ParameterType::Half,
                                ParameterType::Half, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

} // anonymous namespace

int TestFunc_mad_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    KernelMatrix kernels;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    float maxError = 0.0f;

    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    float maxErrorVal3 = 0.0f;
    size_t bufferSize = BUFFER_SIZE;

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);
    uint64_t step = getTestStep(sizeof(cl_half), bufferSize);

    // Init the kernels
    {
        BuildKernelInfo build_info = { 1, kernels, programs, f->nameInCode };
        if ((error = ThreadPool_Do(BuildKernel_HalfFn,
                                   gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                   &build_info)))
            return error;
    }
    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        if (gSkipCorrectnessTesting) break;

        // Init input array
        cl_ushort *p = (cl_ushort *)gIn;
        cl_ushort *p2 = (cl_ushort *)gIn2;
        cl_ushort *p3 = (cl_ushort *)gIn3;
        for (size_t j = 0; j < bufferSize / sizeof(cl_ushort); j++)
        {
            p[j] = (cl_ushort)genrand_int32(d);
            p2[j] = (cl_ushort)genrand_int32(d);
            p3[j] = (cl_ushort)genrand_int32(d);
        }
        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }
        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0,
                                          bufferSize, gIn2, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error);
            return error;
        }
        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0,
                                          bufferSize, gIn3, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error);
            return error;
        }

        // write garbage into output arrays
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
            }
            else
            {
                error = clEnqueueFillBuffer(gQueue, gOutBuffer[j], &pattern,
                                            sizeof(pattern), 0, BUFFER_SIZE, 0,
                                            NULL, NULL);
                test_error(error, "clEnqueueFillBuffer failed!\n");
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeof(cl_half) * sizeValues[j];
            size_t localCount = (bufferSize + vectorSize - 1)
                / vectorSize; // bufferSize / vectorSize  rounded up
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
            if ((error = clSetKernelArg(kernels[j][thread_id], 2,
                                        sizeof(gInBuffer2), &gInBuffer2)))
            {
                LogBuildError(programs[j]);
                return error;
            }
            if ((error = clSetKernelArg(kernels[j][thread_id], 3,
                                        sizeof(gInBuffer3), &gInBuffer3)))
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

        // Verify data - no verification possible. MAD is a random number
        // generator.

        if (0 == (i & 0x0fffffff))
        {
            vlog(".");
            fflush(stdout);
        }
    }

    if (!gSkipCorrectnessTesting)
    {
        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("pass");

        vlog("\t%8.2f @ {%a, %a, %a}", maxError, maxErrorVal, maxErrorVal2,
             maxErrorVal3);
    }
    vlog("\n");

    return error;
}
