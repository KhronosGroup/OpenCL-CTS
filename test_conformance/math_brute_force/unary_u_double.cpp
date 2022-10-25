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

int BuildKernel(const char *name, int vectorSize, cl_kernel *k, cl_program *p,
                bool relaxedMode)
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
                        "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global double",
                        sizeNames[vectorSize],
                        "* out, __global ulong",
                        sizeNames[vectorSize],
                        "* in )\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   out[i] = ",
                        name,
                        "( in[i] );\n"
                        "}\n" };

    const char *c3[] = {
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global double* out, __global ulong* in                 )\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       ulong3 u0 = vload3( 0, in + 3 * i );\n"
        "       double3 f0 = ",
        name,
        "( u0 );\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       ulong3 u0;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               u0 = (ulong3)( in[3*i], 0xdeaddeaddeaddeadUL, "
        "0xdeaddeaddeaddeadUL ); \n"
        "               break;\n"
        "           case 0:\n"
        "               u0 = (ulong3)( in[3*i], in[3*i+1], "
        "0xdeaddeaddeaddeadUL ); \n"
        "               break;\n"
        "       }\n"
        "       double3 f0 = ",
        name,
        "( u0 );\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = f0.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = f0.x; \n"
        "               break;\n"
        "       }\n"
        "   }\n"
        "}\n"
    };

    const char **kern = c;
    size_t kernSize = sizeof(c) / sizeof(c[0]);

    if (sizeValues[vectorSize] == 3)
    {
        kern = c3;
        kernSize = sizeof(c3) / sizeof(c3[0]);
    }

    char testName[32];
    snprintf(testName, sizeof(testName) - 1, "math_kernel%s",
             sizeNames[vectorSize]);

    return MakeKernel(kern, (cl_uint)kernSize, testName, k, p, relaxedMode);
}

struct BuildKernelInfo2
{
    cl_kernel *kernels;
    Programs &programs;
    const char *nameInCode;
    bool relaxedMode; // Whether to build with -cl-fast-relaxed-math.
};

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo2 *info = (BuildKernelInfo2 *)p;
    cl_uint vectorSize = gMinVectorSizeIndex + job_id;
    return BuildKernel(info->nameInCode, vectorSize, info->kernels + vectorSize,
                       &(info->programs[vectorSize]), info->relaxedMode);
}

cl_ulong random64(MTdata d)
{
    return (cl_ulong)genrand_int32(d) | ((cl_ulong)genrand_int32(d) << 32);
}

} // anonymous namespace

int TestFunc_Double_ULong(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    Programs programs;
    cl_kernel kernels[VECTOR_SIZE_COUNT];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_double), BUFFER_SIZE);

    logFunctionInfo(f->name, sizeof(cl_double), relaxedMode);

    Force64BitFPUPrecision();

    // Init the kernels
    {
        BuildKernelInfo2 build_info{ kernels, programs, f->nameInCode,
                                     relaxedMode };
        if ((error = ThreadPool_Do(BuildKernelFn,
                                   gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                   &build_info)))
            return error;
    }

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        cl_ulong *p = (cl_ulong *)gIn;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_ulong); j++)
            p[j] = random64(d);

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        // write garbage into output arrays
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint32_t pattern = 0xffffdead;
            memset_pattern4(gOut[j], &pattern, BUFFER_SIZE);
            if ((error =
                     clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0,
                                          BUFFER_SIZE, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                           error, j);
                goto exit;
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_double);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
            if ((error = clSetKernelArg(kernels[j], 0, sizeof(gOutBuffer[j]),
                                        &gOutBuffer[j])))
            {
                LogBuildError(programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(kernels[j], 1, sizeof(gInBuffer),
                                        &gInBuffer)))
            {
                LogBuildError(programs[j]);
                goto exit;
            }

            if ((error =
                     clEnqueueNDRangeKernel(gQueue, kernels[j], 1, NULL,
                                            &localCount, NULL, 0, NULL, NULL)))
            {
                vlog_error("FAILED -- could not execute kernel\n");
                goto exit;
            }
        }

        // Get that moving
        if ((error = clFlush(gQueue))) vlog("clFlush failed\n");

        // Calculate the correctly rounded reference result
        double *r = (double *)gOut_Ref;
        cl_ulong *s = (cl_ulong *)gIn;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_double); j++)
            r[j] = (double)f->dfunc.f_u(s[j]);

        // Read the data back
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0,
                                         BUFFER_SIZE, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                goto exit;
            }
        }

        if (gSkipCorrectnessTesting) break;

        // Verify data
        uint64_t *t = (uint64_t *)gOut_Ref;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(cl_double); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint64_t *q = (uint64_t *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    double test = ((double *)q)[j];
                    long double correct = f->dfunc.f_u(s[j]);
                    float err = Bruteforce_Ulp_Error_Double(test, correct);
                    int fail = !(fabsf(err) <= f->double_ulps);

                    if (fail)
                    {
                        if (ftz || relaxedMode)
                        {
                            // retry per section 6.5.3.2
                            if (IsDoubleResultSubnormal(correct,
                                                        f->double_ulps))
                            {
                                fail = fail && (test != 0.0);
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    if (fabsf(err) > maxError)
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                    }
                    if (fail)
                    {
                        vlog_error(
                            "\n%s%sD: %f ulp error at 0x%16.16" PRIx64 ": "
                            "*%.13la vs. %.13la\n",
                            f->name, sizeNames[k], err, ((uint64_t *)gIn)[j],
                            ((double *)gOut_Ref)[j], test);
                        error = -1;
                        goto exit;
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

        vlog("\t%8.2f @ %a", maxError, maxErrorVal);
    }

    vlog("\n");

exit:
    // Release
    for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
    {
        clReleaseKernel(kernels[k]);
    }

    return error;
}
