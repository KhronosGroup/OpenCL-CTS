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

#include "function_list.h"
#include "test_functions.h"
#include "utility.h"
#include "reference_math.h"

#include <cstring>


static int BuildKernelHalf(const char *name, int vectorSize, cl_kernel *k,
                           cl_program *p, bool relaxedMode)
{
    const char *c[] = { "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
                        "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global half",
                        sizeNames[vectorSize],
                        "* out, __global ushort",
                        sizeNames[vectorSize],
                        "* in)\n"
                        "{\n"
                        "   int i = get_global_id(0);\n"
                        "   out[i] = ",
                        name,
                        "( in[i] );\n"
                        "}\n" };

    const char *c3[] = {
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n",
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global half* out, __global ushort* in)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       ushort3 u0 = vload3( 0, in + 3 * i );\n"
        "       half3 f0 = ",
        name,
        "( u0 );\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       ushort3 u0;\n"
        "       half3 f0;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               u0 = (ushort3)( in[3*i], 0xdead, 0xdead ); \n"
        "               break;\n"
        "           case 0:\n"
        "               u0 = (ushort3)( in[3*i], in[3*i+1], 0xdead ); \n"
        "               break;\n"
        "       }\n"
        "       f0 = ",
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

typedef struct BuildKernelInfo
{
    cl_uint offset; // the first vector size to build
    cl_kernel *kernels;
    cl_program *programs;
    const char *nameInCode;
    bool relaxedMode;
} BuildKernelInfo;

static cl_int BuildKernel_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED,
                                 void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernelHalf(info->nameInCode, i, info->kernels + i,
                           info->programs + i, info->relaxedMode);
}


int TestFunc_Half_UShort(const Func *f, MTdata d, bool relaxedMode)
{
    uint64_t i;
    uint32_t j, k;
    int error;
    cl_program programs[VECTOR_SIZE_COUNT];
    cl_kernel kernels[VECTOR_SIZE_COUNT];
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
    BuildKernelInfo build_info = { gMinVectorSizeIndex, kernels, programs,
                                   f->nameInCode };
    if ((error = ThreadPool_Do(BuildKernel_HalfFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
    {
        return error;
    }

    for (i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        cl_ushort *p = (cl_ushort *)gIn;
        if (gWimpyMode)
        {
            for (j = 0; j < bufferElements; j++) p[j] = i + j * scale;
        }
        else
        {
            for (j = 0; j < bufferElements; j++) p[j] = (uint16_t)i + j;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          bufferSize, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        // write garbage into output arrays
        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            uint16_t pattern = 0xdead;
            memset_pattern4(gOut[j], &pattern, bufferSize);
            if ((error =
                     clEnqueueWriteBuffer(gQueue, gOutBuffer[j], CL_FALSE, 0,
                                          bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("\n*** Error %d in clEnqueueWriteBuffer2(%d) ***\n",
                           error, j);
                goto exit;
            }
        }

        // Run the kernels
        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeValues[j] * sizeof(cl_half);
            size_t localCount = (bufferSize + vectorSize - 1) / vectorSize;
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
                vlog_error("FAILURE -- could not execute kernel\n");
                goto exit;
            }
        }

        // Get that moving
        if ((error = clFlush(gQueue))) vlog("clFlush failed\n");

        // Calculate the correctly rounded reference result
        cl_half *r = (cl_half *)gOut_Ref;
        for (j = 0; j < bufferElements; j++)
        {
            if (!strcmp(name, "nan"))
                r[j] = reference_nanh(p[j]);
            else
                r[j] = cl_half_from_float(f->func.f_u(p[j]), CL_HALF_RTE);
        }
        // Read the data back
        for (j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer[j], CL_TRUE, 0,
                                         bufferSize, gOut[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray failed %d\n", error);
                goto exit;
            }
        }

        if (gSkipCorrectnessTesting) break;


        // Verify data
        cl_ushort *t = (cl_ushort *)gOut_Ref;
        for (j = 0; j < bufferElements; j++)
        {
            for (k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
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
                vlog("base:%14u step:%10zu  bufferSize:%10zd \n", i, step,
                     bufferSize);
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

exit:
    // Release
    for (k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}