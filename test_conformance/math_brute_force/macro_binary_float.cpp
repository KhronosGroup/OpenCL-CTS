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

#include <cstring>

namespace {

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetBinaryKernel(kernel_name, builtin, ParameterType::Int,
                               ParameterType::Float, ParameterType::Float,
                               vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

struct TestInfo : public TestInfoBase
{
    // Programs for various vector sizes.
    Programs programs;

    // Thread-specific kernels for each vector size:
    // k[vector_size][thread_id]
    KernelMatrix k;

    // Array of thread specific information
    std::vector<ThreadInfoBinary> tinfo;
};

cl_int Test(cl_uint job_id, cl_uint thread_id, void *data)
{
    TestInfo *job = (TestInfo *)data;
    size_t buffer_elements = job->subBufferSize;
    size_t buffer_size = buffer_elements * sizeof(cl_float);
    cl_uint base = job_id * (cl_uint)buffer_elements;
    ThreadInfoBinary *tinfo = &(job->tinfo[thread_id]);
    fptr func = job->f->func;
    int ftz = job->ftz;
    bool relaxedMode = job->relaxedMode;
    MTdata d = tinfo->d;
    cl_int error;
    const char *name = job->f->name;
    cl_int *t = 0;
    cl_int *r = 0;
    cl_float *s = 0;
    cl_float *s2 = 0;

    cl_event e[VECTOR_SIZE_COUNT];
    cl_int *out[VECTOR_SIZE_COUNT];
    if (gHostFill)
    {
        // start the map of the output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            out[j] = (cl_int *)clEnqueueMapBuffer(
                tinfo->tQueue, tinfo->outBuf[j], CL_FALSE, CL_MAP_WRITE, 0,
                buffer_size, 0, NULL, e + j, &error);
            if (error || NULL == out[j])
            {
                vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                           error);
                return error;
            }
        }

        // Get that moving
        if ((error = clFlush(tinfo->tQueue))) vlog("clFlush failed\n");
    }

    // Init input array
    cl_uint *p = (cl_uint *)gIn + thread_id * buffer_elements;
    cl_uint *p2 = (cl_uint *)gIn2 + thread_id * buffer_elements;
    fillFloatBinaryInput((float *)p, (float *)p2, buffer_elements, base, d);

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf, CL_FALSE, 0,
                                      buffer_size, p, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        return error;
    }

    if ((error = clEnqueueWriteBuffer(tinfo->tQueue, tinfo->inBuf2, CL_FALSE, 0,
                                      buffer_size, p2, 0, NULL, NULL)))
    {
        vlog_error("Error: clEnqueueWriteBuffer failed! err: %d\n", error);
        return error;
    }

    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        if (gHostFill)
        {
            // Wait for the map to finish
            if ((error = clWaitForEvents(1, e + j)))
            {
                vlog_error("Error: clWaitForEvents failed! err: %d\n", error);
                return error;
            }
            if ((error = clReleaseEvent(e[j])))
            {
                vlog_error("Error: clReleaseEvent failed! err: %d\n", error);
                return error;
            }
        }

        // Fill the result buffer with garbage, so that old results don't carry
        // over
        uint32_t pattern = 0xffffdead;
        if (gHostFill)
        {
            memset_pattern4(out[j], &pattern, buffer_size);
            if ((error = clEnqueueUnmapMemObject(
                     tinfo->tQueue, tinfo->outBuf[j], out[j], 0, NULL, NULL)))
            {
                vlog_error("Error: clEnqueueUnmapMemObject failed! err: %d\n",
                           error);
                return error;
            }
        }
        else
        {
            if ((error = clEnqueueFillBuffer(tinfo->tQueue, tinfo->outBuf[j],
                                             &pattern, sizeof(pattern), 0,
                                             buffer_size, 0, NULL, NULL)))
            {
                vlog_error("Error: clEnqueueFillBuffer failed! err: %d\n",
                           error);
                return error;
            }
        }

        // Run the kernel
        size_t vectorCount =
            (buffer_elements + sizeValues[j] - 1) / sizeValues[j];
        cl_kernel kernel = job->k[j][thread_id]; // each worker thread has its
                                                 // own copy of the cl_kernel

        error = clSetKernelArg(kernel, 0, sizeof(tinfo->outBuf[j]),
                               &tinfo->outBuf[j]);
        test_error(error, "Failed to set kernel argument");
        error = clSetKernelArg(kernel, 1, sizeof(tinfo->inBuf), &tinfo->inBuf);
        test_error(error, "Failed to set kernel argument");
        error =
            clSetKernelArg(kernel, 2, sizeof(tinfo->inBuf2), &tinfo->inBuf2);
        test_error(error, "Failed to set kernel argument");

        if ((error = clEnqueueNDRangeKernel(tinfo->tQueue, kernel, 1, NULL,
                                            &vectorCount, NULL, 0, NULL, NULL)))
        {
            vlog_error("FAILED -- could not execute kernel\n");
            return error;
        }
    }

    // Get that moving
    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush 2 failed\n");

    if (gSkipCorrectnessTesting) return CL_SUCCESS;

    // Calculate the correctly rounded reference result
    r = (cl_int *)gOut_Ref + thread_id * buffer_elements;
    s = (float *)gIn + thread_id * buffer_elements;
    s2 = (float *)gIn2 + thread_id * buffer_elements;
    for (size_t j = 0; j < buffer_elements; j++) r[j] = func.i_ff(s[j], s2[j]);

    // Read the data back -- no need to wait for the first N-1 buffers but wait
    // for the last buffer. This is an in order queue.
    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        cl_bool blocking = (j + 1 < gMaxVectorSizeIndex) ? CL_FALSE : CL_TRUE;
        out[j] = (cl_int *)clEnqueueMapBuffer(
            tinfo->tQueue, tinfo->outBuf[j], blocking, CL_MAP_READ, 0,
            buffer_size, 0, NULL, NULL, &error);
        if (error || NULL == out[j])
        {
            vlog_error("Error: clEnqueueMapBuffer %d failed! err: %d\n", j,
                       error);
            return error;
        }
    }

    // Verify data
    t = (cl_int *)r;
    for (size_t j = 0; j < buffer_elements; j++)
    {
        cl_int *q = out[0];

        if (gMinVectorSizeIndex == 0 && t[j] != q[j])
        {
            if (ftz || relaxedMode)
            {
                if (IsFloatSubnormal(s[j]))
                {
                    if (IsFloatSubnormal(s2[j]))
                    {
                        int correct = func.i_ff(0.0f, 0.0f);
                        int correct2 = func.i_ff(0.0f, -0.0f);
                        int correct3 = func.i_ff(-0.0f, 0.0f);
                        int correct4 = func.i_ff(-0.0f, -0.0f);

                        if (correct == q[j] || correct2 == q[j]
                            || correct3 == q[j] || correct4 == q[j])
                            continue;
                    }
                    else
                    {
                        int correct = func.i_ff(0.0f, s2[j]);
                        int correct2 = func.i_ff(-0.0f, s2[j]);
                        if (correct == q[j] || correct2 == q[j]) continue;
                    }
                }
                else if (IsFloatSubnormal(s2[j]))
                {
                    int correct = func.i_ff(s[j], 0.0f);
                    int correct2 = func.i_ff(s[j], -0.0f);
                    if (correct == q[j] || correct2 == q[j]) continue;
                }
            }

            uint32_t err = t[j] - q[j];
            if (q[j] > t[j]) err = q[j] - t[j];
            vlog_error("\nERROR: %s: %d ulp error at {%a, %a}: *0x%8.8x vs. "
                       "0x%8.8x (index: %zu)\n",
                       name, err, ((float *)s)[j], ((float *)s2)[j], t[j], q[j],
                       j);
            return -1;
        }

        for (auto k = std::max(1U, gMinVectorSizeIndex);
             k < gMaxVectorSizeIndex; k++)
        {
            q = out[k];
            // If we aren't getting the correctly rounded result
            if (-t[j] != q[j])
            {
                if (ftz || relaxedMode)
                {
                    if (IsFloatSubnormal(s[j]))
                    {
                        if (IsFloatSubnormal(s2[j]))
                        {
                            int correct = -func.i_ff(0.0f, 0.0f);
                            int correct2 = -func.i_ff(0.0f, -0.0f);
                            int correct3 = -func.i_ff(-0.0f, 0.0f);
                            int correct4 = -func.i_ff(-0.0f, -0.0f);

                            if (correct == q[j] || correct2 == q[j]
                                || correct3 == q[j] || correct4 == q[j])
                                continue;
                        }
                        else
                        {
                            int correct = -func.i_ff(0.0f, s2[j]);
                            int correct2 = -func.i_ff(-0.0f, s2[j]);
                            if (correct == q[j] || correct2 == q[j]) continue;
                        }
                    }
                    else if (IsFloatSubnormal(s2[j]))
                    {
                        int correct = -func.i_ff(s[j], 0.0f);
                        int correct2 = -func.i_ff(s[j], -0.0f);
                        if (correct == q[j] || correct2 == q[j]) continue;
                    }
                }
                cl_uint err = -t[j] - q[j];
                if (q[j] > -t[j]) err = q[j] + t[j];
                vlog_error("\nERROR: %s%s: %d ulp error at {%a, %a}: *0x%8.8x "
                           "vs. 0x%8.8x (index: %zu)\n",
                           name, sizeNames[k], err, ((float *)s)[j],
                           ((float *)s2)[j], -t[j], q[j], j);
                return -1;
            }
        }
    }

    for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
    {
        if ((error = clEnqueueUnmapMemObject(tinfo->tQueue, tinfo->outBuf[j],
                                             out[j], 0, NULL, NULL)))
        {
            vlog_error("Error: clEnqueueUnmapMemObject %d failed 2! err: %d\n",
                       j, error);
            return error;
        }
    }

    if ((error = clFlush(tinfo->tQueue))) vlog("clFlush 3 failed\n");


    if (0 == (base & 0x0fffffff))
    {
        if (gVerboseBruteForce)
        {
            vlog("base:%14u buf_elements:%10zd "
                 "ThreadCount:%2u\n",
                 base, buffer_elements, job->threadCount);
        }
        else
        {
            vlog(".");
        }
        fflush(stdout);
    }

    return CL_SUCCESS;
}

} // anonymous namespace

int TestMacro_Int_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    TestInfo test_info{};
    cl_int error;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    // Init test_info
    test_info.threadCount = GetThreadCount();
    test_info.subBufferSize = BUFFER_SIZE
        / (sizeof(cl_float) * RoundUpToNextPowerOfTwo(test_info.threadCount));
    test_info.jobCount = std::max(
        (cl_uint)1, (cl_uint)(getInputCount() / test_info.subBufferSize));

    test_info.f = f;
    test_info.ftz =
        f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    test_info.relaxedMode = relaxedMode;

    test_info.tinfo.resize(test_info.threadCount);
    for (cl_uint i = 0; i < test_info.threadCount; i++)
    {
        cl_buffer_region region = {
            i * test_info.subBufferSize * sizeof(cl_float),
            test_info.subBufferSize * sizeof(cl_float)
        };
        test_info.tinfo[i].inBuf =
            clCreateSubBuffer(gInBuffer, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            return error;
        }
        test_info.tinfo[i].inBuf2 =
            clCreateSubBuffer(gInBuffer2, CL_MEM_READ_ONLY,
                              CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
        if (error || NULL == test_info.tinfo[i].inBuf2)
        {
            vlog_error("Error: Unable to create sub-buffer of gInBuffer2 for "
                       "region {%zd, %zd}\n",
                       region.origin, region.size);
            return error;
        }

        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            test_info.tinfo[i].outBuf[j] = clCreateSubBuffer(
                gOutBuffer[j], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION,
                &region, &error);
            if (error || NULL == test_info.tinfo[i].outBuf[j])
            {
                vlog_error("Error: Unable to create sub-buffer of "
                           "gOutBuffer[%d] for region {%zd, %zd}\n",
                           (int)j, region.origin, region.size);
                return error;
            }
        }
        test_info.tinfo[i].tQueue =
            clCreateCommandQueue(gContext, gDevice, 0, &error);
        if (NULL == test_info.tinfo[i].tQueue || error)
        {
            vlog_error("clCreateCommandQueue failed. (%d)\n", error);
            return error;
        }

        test_info.tinfo[i].d = MTdataHolder(genrand_int32(d));
    }

    // Init the kernels
    BuildKernelInfo build_info{ test_info.threadCount, test_info.k,
                                test_info.programs, f->nameInCode,
                                relaxedMode };
    if ((error = ThreadPool_Do(BuildKernelFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    // Run the kernels
    if (!gSkipCorrectnessTesting)
    {
        error = ThreadPool_Do(Test, test_info.jobCount, &test_info);
        if (error) return error;

        if (gWimpyMode)
            vlog("Wimp pass");
        else
            vlog("passed");
    }

    vlog("\n");

    return CL_SUCCESS;
}
