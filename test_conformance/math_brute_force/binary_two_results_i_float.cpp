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

#include <climits>
#include <cstring>

namespace {

int BuildKernel(const char *name, int vectorSize, cl_kernel *k, cl_program *p,
                bool relaxedMode)
{
    const char *c[] = { "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global float",
                        sizeNames[vectorSize],
                        "* out, __global int",
                        sizeNames[vectorSize],
                        "* out2, __global float",
                        sizeNames[vectorSize],
                        "* in1, __global float",
                        sizeNames[vectorSize],
                        "* in2 )\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   out[i] = ",
                        name,
                        "( in1[i], in2[i], out2 + i );\n"
                        "}\n" };

    const char *c3[] = {
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global float* out, __global int* out2, __global float* in, "
        "__global float* in2)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 f1 = vload3( 0, in2 + 3 * i );\n"
        "       int3 i0 = 0xdeaddead;\n"
        "       f0 = ",
        name,
        "( f0, f1, &i0 );\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "       vstore3( i0, 0, out2 + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       float3 f0;\n"
        "       float3 f1;\n"
        "       int3 i0 = 0xdeaddead;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
        "               f1 = (float3)( in2[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
        "               f1 = (float3)( in2[3*i], in2[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       f0 = ",
        name,
        "( f0, f1, &i0 );\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = f0.y; \n"
        "               out2[3*i+1] = i0.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = f0.x; \n"
        "               out2[3*i] = i0.x; \n"
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

struct ComputeReferenceInfoF
{
    const float *x;
    const float *y;
    float *r;
    int *i;
    double (*f_ffpI)(double, double, int *);
    cl_uint lim;
    cl_uint count;
};

cl_int ReferenceF(cl_uint jid, cl_uint tid, void *userInfo)
{
    ComputeReferenceInfoF *cri = (ComputeReferenceInfoF *)userInfo;
    cl_uint lim = cri->lim;
    cl_uint count = cri->count;
    cl_uint off = jid * count;
    const float *x = cri->x + off;
    const float *y = cri->y + off;
    float *r = cri->r + off;
    int *i = cri->i + off;
    double (*f)(double, double, int *) = cri->f_ffpI;

    if (off + count > lim) count = lim - off;

    for (cl_uint j = 0; j < count; ++j)
        r[j] = (float)f((double)x[j], (double)y[j], i + j);

    return CL_SUCCESS;
}

} // anonymous namespace

int TestFunc_FloatI_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    Programs programs;
    cl_kernel kernels[VECTOR_SIZE_COUNT];
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    int64_t maxError2 = 0;
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);

    cl_uint threadCount = GetThreadCount();

    float float_ulps;
    if (gIsEmbedded)
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    int testingRemquo = !strcmp(f->name, "remquo");

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
        cl_uint *p = (cl_uint *)gIn;
        cl_uint *p2 = (cl_uint *)gIn2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            p[j] = genrand_int32(d);
            p2[j] = genrand_int32(d);
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer ***\n", error);
            return error;
        }

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer2, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn2, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer2 ***\n", error);
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

            memset_pattern4(gOut2[j], &pattern, BUFFER_SIZE);
            if ((error = clEnqueueWriteBuffer(gQueue, gOutBuffer2[j], CL_FALSE,
                                              0, BUFFER_SIZE, gOut2[j], 0, NULL,
                                              NULL)))
            {
                vlog_error("\n*** Error %d in clEnqueueWriteBuffer2b(%d) ***\n",
                           error, j);
                goto exit;
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeof(cl_float) * sizeValues[j];
            size_t localCount = (BUFFER_SIZE + vectorSize - 1)
                / vectorSize; // BUFFER_SIZE / vectorSize  rounded up
            if ((error = clSetKernelArg(kernels[j], 0, sizeof(gOutBuffer[j]),
                                        &gOutBuffer[j])))
            {
                LogBuildError(programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(kernels[j], 1, sizeof(gOutBuffer2[j]),
                                        &gOutBuffer2[j])))
            {
                LogBuildError(programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(kernels[j], 2, sizeof(gInBuffer),
                                        &gInBuffer)))
            {
                LogBuildError(programs[j]);
                goto exit;
            }
            if ((error = clSetKernelArg(kernels[j], 3, sizeof(gInBuffer2),
                                        &gInBuffer2)))
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
        float *s = (float *)gIn;
        float *s2 = (float *)gIn2;

        if (threadCount > 1)
        {
            ComputeReferenceInfoF cri;
            cri.x = s;
            cri.y = s2;
            cri.r = (float *)gOut_Ref;
            cri.i = (int *)gOut_Ref2;
            cri.f_ffpI = f->func.f_ffpI;
            cri.lim = BUFFER_SIZE / sizeof(float);
            cri.count = (cri.lim + threadCount - 1) / threadCount;
            ThreadPool_Do(ReferenceF, threadCount, &cri);
        }
        else
        {
            float *r = (float *)gOut_Ref;
            int *r2 = (int *)gOut_Ref2;
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
                r[j] = (float)f->func.f_ffpI(s[j], s2[j], r2 + j);
        }

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
            if ((error =
                     clEnqueueReadBuffer(gQueue, gOutBuffer2[j], CL_TRUE, 0,
                                         BUFFER_SIZE, gOut2[j], 0, NULL, NULL)))
            {
                vlog_error("ReadArray2 failed %d\n", error);
                goto exit;
            }
        }

        if (gSkipCorrectnessTesting) break;

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        int32_t *t2 = (int32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)(gOut[k]);
                int32_t *q2 = (int32_t *)gOut2[k];

                // Check for exact match to correctly rounded result
                if (t[j] == q[j] && t2[j] == q2[j]) continue;

                // Check for paired NaNs
                if ((t[j] & 0x7fffffff) > 0x7f800000
                    && (q[j] & 0x7fffffff) > 0x7f800000 && t2[j] == q2[j])
                    continue;

                float test = ((float *)q)[j];
                int correct2 = INT_MIN;
                double correct = f->func.f_ffpI(s[j], s2[j], &correct2);
                float err = Ulp_Error(test, correct);
                int64_t iErr;

                // in case of remquo, we only care about the sign and last
                // seven bits of integer as per the spec.
                if (testingRemquo)
                    iErr = (long long)(q2[j] & 0x0000007f)
                        - (long long)(correct2 & 0x0000007f);
                else
                    iErr = (long long)q2[j] - (long long)correct2;

                // For remquo, if y = 0, x is infinite, or either is NaN
                // then the standard either neglects to say what is returned
                // in iptr or leaves it undefined or implementation defined.
                int iptrUndefined = fabs(((float *)gIn)[j]) == INFINITY
                    || ((float *)gIn2)[j] == 0.0f || isnan(((float *)gIn2)[j])
                    || isnan(((float *)gIn)[j]);
                if (iptrUndefined) iErr = 0;

                int fail = !(fabsf(err) <= float_ulps && iErr == 0);
                if ((ftz || relaxedMode) && fail)
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
                        int correct3i, correct4i;
                        double correct3 =
                            f->func.f_ffpI(0.0, s2[j], &correct3i);
                        double correct4 =
                            f->func.f_ffpI(-0.0, s2[j], &correct4i);
                        float err2 = Ulp_Error(test, correct3);
                        float err3 = Ulp_Error(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= float_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, float_ulps)
                            || IsFloatResultSubnormal(correct3, float_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
                        }

                        // try with both args as zero
                        if (IsFloatSubnormal(s2[j]))
                        {
                            int correct7i, correct8i;
                            correct3 = f->func.f_ffpI(0.0, 0.0, &correct3i);
                            correct4 = f->func.f_ffpI(-0.0, 0.0, &correct4i);
                            double correct7 =
                                f->func.f_ffpI(0.0, -0.0, &correct7i);
                            double correct8 =
                                f->func.f_ffpI(-0.0, -0.0, &correct8i);
                            err2 = Ulp_Error(test, correct3);
                            err3 = Ulp_Error(test, correct4);
                            float err4 = Ulp_Error(test, correct7);
                            float err5 = Ulp_Error(test, correct8);
                            iErr3 = (long long)q2[j] - (long long)correct3i;
                            iErr4 = (long long)q2[j] - (long long)correct4i;
                            int64_t iErr7 =
                                (long long)q2[j] - (long long)correct7i;
                            int64_t iErr8 =
                                (long long)q2[j] - (long long)correct8i;
                            fail = fail
                                && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                    && (!(fabsf(err3) <= float_ulps
                                          && iErr4 == 0))
                                    && (!(fabsf(err4) <= float_ulps
                                          && iErr7 == 0))
                                    && (!(fabsf(err5) <= float_ulps
                                          && iErr8 == 0)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;
                            if (fabsf(err4) < fabsf(err)) err = err4;
                            if (fabsf(err5) < fabsf(err)) err = err5;
                            if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                            if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;
                            if (llabs(iErr7) < llabs(iErr)) iErr = iErr7;
                            if (llabs(iErr8) < llabs(iErr)) iErr = iErr8;

                            // retry per section 6.5.3.4
                            if (IsFloatResultSubnormal(correct3, float_ulps)
                                || IsFloatResultSubnormal(correct4, float_ulps)
                                || IsFloatResultSubnormal(correct7, float_ulps)
                                || IsFloatResultSubnormal(correct8, float_ulps))
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && (iErr3 == 0 || iErr4 == 0
                                             || iErr7 == 0 || iErr8 == 0));
                                if (!fail) err = 0.0f;
                            }
                        }
                    }
                    else if (IsFloatSubnormal(s2[j]))
                    {
                        int correct3i, correct4i;
                        double correct3 = f->func.f_ffpI(s[j], 0.0, &correct3i);
                        double correct4 =
                            f->func.f_ffpI(s[j], -0.0, &correct4i);
                        float err2 = Ulp_Error(test, correct3);
                        float err3 = Ulp_Error(test, correct4);
                        int64_t iErr3 = (long long)q2[j] - (long long)correct3i;
                        int64_t iErr4 = (long long)q2[j] - (long long)correct4i;
                        fail = fail
                            && ((!(fabsf(err2) <= float_ulps && iErr3 == 0))
                                && (!(fabsf(err3) <= float_ulps
                                      && iErr4 == 0)));
                        if (fabsf(err2) < fabsf(err)) err = err2;
                        if (fabsf(err3) < fabsf(err)) err = err3;
                        if (llabs(iErr3) < llabs(iErr)) iErr = iErr3;
                        if (llabs(iErr4) < llabs(iErr)) iErr = iErr4;

                        // retry per section 6.5.3.4
                        if (IsFloatResultSubnormal(correct2, float_ulps)
                            || IsFloatResultSubnormal(correct3, float_ulps))
                        {
                            fail = fail
                                && !(test == 0.0f
                                     && (iErr3 == 0 || iErr4 == 0));
                            if (!fail) err = 0.0f;
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
                    vlog_error(
                        "\nERROR: %s%s: {%f, %lld} ulp error at {%a, %a} "
                        "({0x%8.8x, 0x%8.8x}): *{%a, %d} ({0x%8.8x, "
                        "0x%8.8x}) vs. {%a, %d} ({0x%8.8x, 0x%8.8x})\n",
                        f->name, sizeNames[k], err, iErr, ((float *)gIn)[j],
                        ((float *)gIn2)[j], ((cl_uint *)gIn)[j],
                        ((cl_uint *)gIn2)[j], ((float *)gOut_Ref)[j],
                        ((int *)gOut_Ref2)[j], ((cl_uint *)gOut_Ref)[j],
                        ((cl_uint *)gOut_Ref2)[j], test, q2[j],
                        ((cl_uint *)&test)[0], ((cl_uint *)q2)[j]);
                    error = -1;
                    goto exit;
                }
            }
        }

        if (0 == (i & 0x0fffffff))
        {
            if (gVerboseBruteForce)
            {
                vlog("base:%14u step:%10zu  bufferSize:%10zd \n", i, step,
                     BUFFER_SIZE);
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

        vlog("\t{%8.2f, %lld} @ {%a, %a}", maxError, maxError2, maxErrorVal,
             maxErrorVal2);
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
