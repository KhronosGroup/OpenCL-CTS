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

#include <cstring>

static int BuildKernel(const char *name, int vectorSize, cl_kernel *k,
                       cl_program *p, bool relaxedMode)
{
    const char *c[] = { "__kernel void math_kernel",
                        sizeNames[vectorSize],
                        "( __global float",
                        sizeNames[vectorSize],
                        "* out, __global float",
                        sizeNames[vectorSize],
                        "* out2, __global float",
                        sizeNames[vectorSize],
                        "* in )\n"
                        "{\n"
                        "   size_t i = get_global_id(0);\n"
                        "   out[i] = ",
                        name,
                        "( in[i], out2 + i );\n"
                        "}\n" };

    const char *c3[] = {
        "__kernel void math_kernel",
        sizeNames[vectorSize],
        "( __global float* out, __global float* out2, __global float* in)\n"
        "{\n"
        "   size_t i = get_global_id(0);\n"
        "   if( i + 1 < get_global_size(0) )\n"
        "   {\n"
        "       float3 f0 = vload3( 0, in + 3 * i );\n"
        "       float3 iout = NAN;\n"
        "       f0 = ",
        name,
        "( f0, &iout );\n"
        "       vstore3( f0, 0, out + 3*i );\n"
        "       vstore3( iout, 0, out2 + 3*i );\n"
        "   }\n"
        "   else\n"
        "   {\n"
        "       size_t parity = i & 1;   // Figure out how many elements are "
        "left over after BUFFER_SIZE % (3*sizeof(float)). Assume power of two "
        "buffer size \n"
        "       float3 iout = NAN;\n"
        "       float3 f0;\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 1:\n"
        "               f0 = (float3)( in[3*i], NAN, NAN ); \n"
        "               break;\n"
        "           case 0:\n"
        "               f0 = (float3)( in[3*i], in[3*i+1], NAN ); \n"
        "               break;\n"
        "       }\n"
        "       f0 = ",
        name,
        "( f0, &iout );\n"
        "       switch( parity )\n"
        "       {\n"
        "           case 0:\n"
        "               out[3*i+1] = f0.y; \n"
        "               out2[3*i+1] = iout.y; \n"
        "               // fall through\n"
        "           case 1:\n"
        "               out[3*i] = f0.x; \n"
        "               out2[3*i] = iout.x; \n"
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
    bool relaxedMode; // Whether to build with -cl-fast-relaxed-math.
} BuildKernelInfo;

static cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo *info = (BuildKernelInfo *)p;
    cl_uint i = info->offset + job_id;
    return BuildKernel(info->nameInCode, i, info->kernels + i,
                       info->programs + i, info->relaxedMode);
}

int TestFunc_Float2_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;
    cl_program programs[VECTOR_SIZE_COUNT];
    cl_kernel kernels[VECTOR_SIZE_COUNT];
    float maxError0 = 0.0f;
    float maxError1 = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal0 = 0.0f;
    float maxErrorVal1 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);
    int scale = (int)((1ULL << 32) / (16 * BUFFER_SIZE / sizeof(float)) + 1);
    cl_uchar overflow[BUFFER_SIZE / sizeof(float)];
    int isFract = 0 == strcmp("fract", f->nameInCode);
    int skipNanInf = isFract && !gInfNanSupport;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    float float_ulps = getAllowedUlpError(f, relaxedMode);
    // Init the kernels
    {
        BuildKernelInfo build_info = { gMinVectorSizeIndex, kernels, programs,
                                       f->nameInCode, relaxedMode };
        if ((error = ThreadPool_Do(BuildKernelFn,
                                   gMaxVectorSizeIndex - gMinVectorSizeIndex,
                                   &build_info)))
            return error;
    }

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        uint32_t *p = (uint32_t *)gIn;
        if (gWimpyMode)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                p[j] = (uint32_t)i + j * scale;
                if (relaxedMode && strcmp(f->name, "sincos") == 0)
                {
                    float pj = *(float *)&p[j];
                    if (fabs(pj) > M_PI) ((float *)p)[j] = NAN;
                }
            }
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                p[j] = (uint32_t)i + j;
                if (relaxedMode && strcmp(f->name, "sincos") == 0)
                {
                    float pj = *(float *)&p[j];
                    if (fabs(pj) > M_PI) ((float *)p)[j] = NAN;
                }
            }
        }

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
            size_t vectorSize = sizeValues[j] * sizeof(cl_float);
            size_t localCount = (BUFFER_SIZE + vectorSize - 1) / vectorSize;
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

        FPU_mode_type oldMode;
        RoundingMode oldRoundMode = kRoundToNearestEven;
        if (isFract)
        {
            // Calculate the correctly rounded reference result
            memset(&oldMode, 0, sizeof(oldMode));
            if (ftz) ForceFTZ(&oldMode);

            // Set the rounding mode to match the device
            if (gIsInRTZMode)
                oldRoundMode = set_round(kRoundTowardZero, kfloat);
        }

        // Calculate the correctly rounded reference result
        float *r = (float *)gOut_Ref;
        float *r2 = (float *)gOut_Ref2;
        float *s = (float *)gIn;

        if (skipNanInf)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                double dd;
                feclearexcept(FE_OVERFLOW);

                if (relaxedMode)
                    r[j] = (float)f->rfunc.f_fpf(s[j], &dd);
                else
                    r[j] = (float)f->func.f_fpf(s[j], &dd);

                r2[j] = (float)dd;
                overflow[j] =
                    FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                double dd;
                if (relaxedMode)
                    r[j] = (float)f->rfunc.f_fpf(s[j], &dd);
                else
                    r[j] = (float)f->func.f_fpf(s[j], &dd);

                r2[j] = (float)dd;
            }
        }

        if (isFract && ftz) RestoreFPState(&oldMode);

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

        if (gSkipCorrectnessTesting)
        {
            if (isFract && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);
            break;
        }

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        uint32_t *t2 = (uint32_t *)gOut_Ref2;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)gOut[k];
                uint32_t *q2 = (uint32_t *)gOut2[k];

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j] || t2[j] != q2[j])
                {
                    double correct, correct2;
                    float err, err2;
                    float test = ((float *)q)[j];
                    float test2 = ((float *)q2)[j];

                    if (relaxedMode)
                        correct = f->rfunc.f_fpf(s[j], &correct2);
                    else
                        correct = f->func.f_fpf(s[j], &correct2);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow
                    if (relaxedMode || skipNanInf)
                    {
                        if (skipNanInf && overflow[j]) continue;
                        // Note: no double rounding here.  Reference functions
                        // calculate in single precision.
                        if (IsFloatInfinity(correct) || IsFloatNaN(correct)
                            || IsFloatInfinity(correct2) || IsFloatNaN(correct2)
                            || IsFloatInfinity(s[j]) || IsFloatNaN(s[j]))
                            continue;
                    }

                    typedef int (*CheckForSubnormal)(
                        double, float); // If we are in fast relaxed math, we
                                        // have a different calculation for the
                                        // subnormal threshold.
                    CheckForSubnormal isFloatResultSubnormalPtr;
                    if (relaxedMode)
                    {
                        err = Abs_Error(test, correct);
                        err2 = Abs_Error(test2, correct2);
                        isFloatResultSubnormalPtr =
                            &IsFloatResultSubnormalAbsError;
                    }
                    else
                    {
                        err = Ulp_Error(test, correct);
                        err2 = Ulp_Error(test2, correct2);
                        isFloatResultSubnormalPtr = &IsFloatResultSubnormal;
                    }
                    int fail = !(fabsf(err) <= float_ulps
                                 && fabsf(err2) <= float_ulps);

                    if (ftz)
                    {
                        // retry per section 6.5.3.2
                        if ((*isFloatResultSubnormalPtr)(correct, float_ulps))
                        {
                            if ((*isFloatResultSubnormalPtr)(correct2,
                                                             float_ulps))
                            {
                                fail = fail && !(test == 0.0f && test2 == 0.0f);
                                if (!fail)
                                {
                                    err = 0.0f;
                                    err2 = 0.0f;
                                }
                            }
                            else
                            {
                                fail = fail
                                    && !(test == 0.0f
                                         && fabsf(err2) <= float_ulps);
                                if (!fail) err = 0.0f;
                            }
                        }
                        else if ((*isFloatResultSubnormalPtr)(correct2,
                                                              float_ulps))
                        {
                            fail = fail
                                && !(test2 == 0.0f && fabsf(err) <= float_ulps);
                            if (!fail) err2 = 0.0f;
                        }


                        // retry per section 6.5.3.3
                        if (IsFloatSubnormal(s[j]))
                        {
                            double correctp, correctn;
                            double correct2p, correct2n;
                            float errp, err2p, errn, err2n;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);
                            if (relaxedMode)
                            {
                                correctp = f->rfunc.f_fpf(0.0, &correct2p);
                                correctn = f->rfunc.f_fpf(-0.0, &correct2n);
                            }
                            else
                            {
                                correctp = f->func.f_fpf(0.0, &correct2p);
                                correctn = f->func.f_fpf(-0.0, &correct2n);
                            }

                            // Per section 10 paragraph 6, accept any result if
                            // an input or output is a infinity or NaN or
                            // overflow
                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correctp)
                                    || IsFloatNaN(correctp)
                                    || IsFloatInfinity(correctn)
                                    || IsFloatNaN(correctn)
                                    || IsFloatInfinity(correct2p)
                                    || IsFloatNaN(correct2p)
                                    || IsFloatInfinity(correct2n)
                                    || IsFloatNaN(correct2n))
                                    continue;
                            }

                            if (relaxedMode)
                            {
                                errp = Abs_Error(test, correctp);
                                err2p = Abs_Error(test, correct2p);
                                errn = Abs_Error(test, correctn);
                                err2n = Abs_Error(test, correct2n);
                            }
                            else
                            {
                                errp = Ulp_Error(test, correctp);
                                err2p = Ulp_Error(test, correct2p);
                                errn = Ulp_Error(test, correctn);
                                err2n = Ulp_Error(test, correct2n);
                            }

                            fail = fail
                                && ((!(fabsf(errp) <= float_ulps))
                                    && (!(fabsf(err2p) <= float_ulps))
                                    && ((!(fabsf(errn) <= float_ulps))
                                        && (!(fabsf(err2n) <= float_ulps))));
                            if (fabsf(errp) < fabsf(err)) err = errp;
                            if (fabsf(errn) < fabsf(err)) err = errn;
                            if (fabsf(err2p) < fabsf(err2)) err2 = err2p;
                            if (fabsf(err2n) < fabsf(err2)) err2 = err2n;

                            // retry per section 6.5.3.4
                            if ((*isFloatResultSubnormalPtr)(correctp,
                                                             float_ulps)
                                || (*isFloatResultSubnormalPtr)(correctn,
                                                                float_ulps))
                            {
                                if ((*isFloatResultSubnormalPtr)(correct2p,
                                                                 float_ulps)
                                    || (*isFloatResultSubnormalPtr)(correct2n,
                                                                    float_ulps))
                                {
                                    fail = fail
                                        && !(test == 0.0f && test2 == 0.0f);
                                    if (!fail) err = err2 = 0.0f;
                                }
                                else
                                {
                                    fail = fail
                                        && !(test == 0.0f
                                             && fabsf(err2) <= float_ulps);
                                    if (!fail) err = 0.0f;
                                }
                            }
                            else if ((*isFloatResultSubnormalPtr)(correct2p,
                                                                  float_ulps)
                                     || (*isFloatResultSubnormalPtr)(
                                         correct2n, float_ulps))
                            {
                                fail = fail
                                    && !(test2 == 0.0f
                                         && (fabsf(err) <= float_ulps));
                                if (!fail) err2 = 0.0f;
                            }
                        }
                    }
                    if (fabsf(err) > maxError0)
                    {
                        maxError0 = fabsf(err);
                        maxErrorVal0 = s[j];
                    }
                    if (fabsf(err2) > maxError1)
                    {
                        maxError1 = fabsf(err2);
                        maxErrorVal1 = s[j];
                    }
                    if (fail)
                    {
                        vlog_error("\nERROR: %s%s: {%f, %f} ulp error at %a: "
                                   "*{%a, %a} vs. {%a, %a}\n",
                                   f->name, sizeNames[k], err, err2,
                                   ((float *)gIn)[j], ((float *)gOut_Ref)[j],
                                   ((float *)gOut_Ref2)[j], test, test2);
                        error = -1;
                        goto exit;
                    }
                }
            }
        }

        if (isFract && gIsInRTZMode) (void)set_round(oldRoundMode, kfloat);

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

        vlog("\t{%8.2f, %8.2f} @ {%a, %a}", maxError0, maxError1, maxErrorVal0,
             maxErrorVal1);
    }

    vlog("\n");

exit:
    // Release
    for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
    {
        clReleaseKernel(kernels[k]);
        clReleaseProgram(programs[k]);
    }

    return error;
}
