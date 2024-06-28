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

#define CORRECTLY_ROUNDED 0
#define FLUSHED 1

namespace {

cl_int BuildKernelFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
{
    BuildKernelInfo &info = *(BuildKernelInfo *)p;
    auto generator = [](const std::string &kernel_name, const char *builtin,
                        cl_uint vector_size_index) {
        return GetTernaryKernel(kernel_name, builtin, ParameterType::Float,
                                ParameterType::Float, ParameterType::Float,
                                ParameterType::Float, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

// A table of more difficult cases to get right
const float specialValues[] = {
    -NAN,
    -INFINITY,
    -FLT_MAX,
    MAKE_HEX_FLOAT(-0x1.000002p64f, -0x1000002L, 40),
    MAKE_HEX_FLOAT(-0x1.0p64f, -0x1L, 64),
    MAKE_HEX_FLOAT(-0x1.fffffep63f, -0x1fffffeL, 39),
    MAKE_HEX_FLOAT(-0x1.000002p63f, -0x1000002L, 39),
    MAKE_HEX_FLOAT(-0x1.0p63f, -0x1L, 63),
    MAKE_HEX_FLOAT(-0x1.fffffep62f, -0x1fffffeL, 38),
    -3.0f,
    MAKE_HEX_FLOAT(-0x1.800002p1f, -0x1800002L, -23),
    -2.5f,
    MAKE_HEX_FLOAT(-0x1.7ffffep1f, -0x17ffffeL, -23),
    -2.0f,
    MAKE_HEX_FLOAT(-0x1.800002p0f, -0x1800002L, -24),
    -1.75f,
    -1.5f,
    -1.25f,
    MAKE_HEX_FLOAT(-0x1.7ffffep0f, -0x17ffffeL, -24),
    MAKE_HEX_FLOAT(-0x1.000002p0f, -0x1000002L, -24),
    MAKE_HEX_FLOAT(-0x1.003p0f, -0x1003000L, -24),
    -MAKE_HEX_FLOAT(0x1.001p0f, 0x1001000L, -24),
    -1.0f,
    MAKE_HEX_FLOAT(-0x1.fffffep-1f, -0x1fffffeL, -25),
    MAKE_HEX_FLOAT(-0x1.000002p-126f, -0x1000002L, -150),
    -FLT_MIN,
    MAKE_HEX_FLOAT(-0x0.fffffep-126f, -0x0fffffeL, -150),
    MAKE_HEX_FLOAT(-0x0.000ffep-126f, -0x0000ffeL, -150),
    MAKE_HEX_FLOAT(-0x0.0000fep-126f, -0x00000feL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ep-126f, -0x000000eL, -150),
    MAKE_HEX_FLOAT(-0x0.00000cp-126f, -0x000000cL, -150),
    MAKE_HEX_FLOAT(-0x0.00000ap-126f, -0x000000aL, -150),
    MAKE_HEX_FLOAT(-0x0.000008p-126f, -0x0000008L, -150),
    MAKE_HEX_FLOAT(-0x0.000006p-126f, -0x0000006L, -150),
    MAKE_HEX_FLOAT(-0x0.000004p-126f, -0x0000004L, -150),
    MAKE_HEX_FLOAT(-0x0.000002p-126f, -0x0000002L, -150),
    -0.0f,

    +NAN,
    +INFINITY,
    +FLT_MAX,
    MAKE_HEX_FLOAT(+0x1.000002p64f, +0x1000002L, 40),
    MAKE_HEX_FLOAT(+0x1.0p64f, +0x1L, 64),
    MAKE_HEX_FLOAT(+0x1.fffffep63f, +0x1fffffeL, 39),
    MAKE_HEX_FLOAT(+0x1.000002p63f, +0x1000002L, 39),
    MAKE_HEX_FLOAT(+0x1.0p63f, +0x1L, 63),
    MAKE_HEX_FLOAT(+0x1.fffffep62f, +0x1fffffeL, 38),
    +3.0f,
    MAKE_HEX_FLOAT(+0x1.800002p1f, +0x1800002L, -23),
    2.5f,
    MAKE_HEX_FLOAT(+0x1.7ffffep1f, +0x17ffffeL, -23),
    +2.0f,
    MAKE_HEX_FLOAT(+0x1.800002p0f, +0x1800002L, -24),
    1.75f,
    1.5f,
    1.25f,
    MAKE_HEX_FLOAT(+0x1.7ffffep0f, +0x17ffffeL, -24),
    MAKE_HEX_FLOAT(+0x1.000002p0f, +0x1000002L, -24),
    MAKE_HEX_FLOAT(0x1.003p0f, 0x1003000L, -24),
    +MAKE_HEX_FLOAT(0x1.001p0f, 0x1001000L, -24),
    +1.0f,
    MAKE_HEX_FLOAT(+0x1.fffffep-1f, +0x1fffffeL, -25),
    MAKE_HEX_FLOAT(0x1.000002p-126f, 0x1000002L, -150),
    +FLT_MIN,
    MAKE_HEX_FLOAT(+0x0.fffffep-126f, +0x0fffffeL, -150),
    MAKE_HEX_FLOAT(+0x0.000ffep-126f, +0x0000ffeL, -150),
    MAKE_HEX_FLOAT(+0x0.0000fep-126f, +0x00000feL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ep-126f, +0x000000eL, -150),
    MAKE_HEX_FLOAT(+0x0.00000cp-126f, +0x000000cL, -150),
    MAKE_HEX_FLOAT(+0x0.00000ap-126f, +0x000000aL, -150),
    MAKE_HEX_FLOAT(+0x0.000008p-126f, +0x0000008L, -150),
    MAKE_HEX_FLOAT(+0x0.000006p-126f, +0x0000006L, -150),
    MAKE_HEX_FLOAT(+0x0.000004p-126f, +0x0000004L, -150),
    MAKE_HEX_FLOAT(+0x0.000002p-126f, +0x0000002L, -150),
    +0.0f,
};

constexpr size_t specialValuesCount =
    sizeof(specialValues) / sizeof(specialValues[0]);

} // anonymous namespace

int TestFunc_Float_Float_Float_Float(const Func *f, MTdata d, bool relaxedMode)
{
    int error;

    logFunctionInfo(f->name, sizeof(cl_float), relaxedMode);

    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gFloatCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    float maxErrorVal3 = 0.0f;
    uint64_t step = getTestStep(sizeof(float), BUFFER_SIZE);

    cl_uchar overflow[BUFFER_SIZE / sizeof(float)];

    float float_ulps;
    if (gIsEmbedded)
        float_ulps = f->float_embedded_ulps;
    else
        float_ulps = f->float_ulps;

    int skipNanInf = (0 == strcmp("fma", f->nameInCode)) && !gInfNanSupport;

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
        cl_uint *p = (cl_uint *)gIn;
        cl_uint *p2 = (cl_uint *)gIn2;
        cl_uint *p3 = (cl_uint *)gIn3;
        size_t idx = 0;

        if (i == 0)
        { // test edge cases
            float *fp = (float *)gIn;
            float *fp2 = (float *)gIn2;
            float *fp3 = (float *)gIn3;
            uint32_t x, y, z;
            x = y = z = 0;
            for (; idx < BUFFER_SIZE / sizeof(float); idx++)
            {
                fp[idx] = specialValues[x];
                fp2[idx] = specialValues[y];
                fp3[idx] = specialValues[z];

                if (++x >= specialValuesCount)
                {
                    x = 0;
                    if (++y >= specialValuesCount)
                    {
                        y = 0;
                        if (++z >= specialValuesCount) break;
                    }
                }
            }
            if (idx == BUFFER_SIZE / sizeof(float))
                vlog_error("Test Error: not all special cases tested!\n");
        }

        for (; idx < BUFFER_SIZE / sizeof(float); idx++)
        {
            p[idx] = genrand_int32(d);
            p2[idx] = genrand_int32(d);
            p3[idx] = genrand_int32(d);
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

        if ((error = clEnqueueWriteBuffer(gQueue, gInBuffer3, CL_FALSE, 0,
                                          BUFFER_SIZE, gIn3, 0, NULL, NULL)))
        {
            vlog_error("\n*** Error %d in clEnqueueWriteBuffer3 ***\n", error);
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
            }
            else
            {
                if ((error = clEnqueueFillBuffer(gQueue, gOutBuffer[j],
                                                 &pattern, sizeof(pattern), 0,
                                                 BUFFER_SIZE, 0, NULL, NULL)))
                {
                    vlog_error("Error: clEnqueueFillBuffer failed! err: %d\n",
                               error);
                    return error;
                }
            }
        }

        // Run the kernels
        for (auto j = gMinVectorSizeIndex; j < gMaxVectorSizeIndex; j++)
        {
            size_t vectorSize = sizeof(cl_float) * sizeValues[j];
            size_t localCount = (BUFFER_SIZE + vectorSize - 1)
                / vectorSize; // BUFFER_SIZE / vectorSize  rounded up
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

        // Calculate the correctly rounded reference result
        float *r = (float *)gOut_Ref;
        float *s = (float *)gIn;
        float *s2 = (float *)gIn2;
        float *s3 = (float *)gIn3;
        if (skipNanInf)
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
            {
                feclearexcept(FE_OVERFLOW);
                r[j] =
                    (float)f->func.f_fma(s[j], s2[j], s3[j], CORRECTLY_ROUNDED);
                overflow[j] =
                    FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
                r[j] =
                    (float)f->func.f_fma(s[j], s2[j], s3[j], CORRECTLY_ROUNDED);
        }

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
        }

        // Verify data
        uint32_t *t = (uint32_t *)gOut_Ref;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(float); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint32_t *q = (uint32_t *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    float err;
                    int fail;
                    float test = ((float *)q)[j];
                    float correct =
                        f->func.f_fma(s[j], s2[j], s3[j], CORRECTLY_ROUNDED);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow
                    if (skipNanInf)
                    {
                        if (overflow[j] || IsFloatInfinity(correct)
                            || IsFloatNaN(correct) || IsFloatInfinity(s[j])
                            || IsFloatNaN(s[j]) || IsFloatInfinity(s2[j])
                            || IsFloatNaN(s2[j]) || IsFloatInfinity(s3[j])
                            || IsFloatNaN(s3[j]))
                            continue;
                    }


                    err = Ulp_Error(test, correct);
                    fail = !(fabsf(err) <= float_ulps);

                    if (fail && (ftz || relaxedMode))
                    {
                        float correct2, err2;

                        // retry per section 6.5.3.2  with flushing on
                        if (0.0f == test
                            && 0.0f
                                == f->func.f_fma(s[j], s2[j], s3[j], FLUSHED))
                        {
                            fail = 0;
                            err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (fail && IsFloatSubnormal(s[j]))
                        { // look at me,
                            float err3, correct3;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            correct2 = f->func.f_fma(0.0f, s2[j], s3[j],
                                                     CORRECTLY_ROUNDED);
                            correct3 = f->func.f_fma(-0.0f, s2[j], s3[j],
                                                     CORRECTLY_ROUNDED);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correct2)
                                    || IsFloatNaN(correct2)
                                    || IsFloatInfinity(correct3)
                                    || IsFloatNaN(correct3))
                                    continue;
                            }

                            err2 = Ulp_Error(test, correct2);
                            err3 = Ulp_Error(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= float_ulps))
                                    && (!(fabsf(err3) <= float_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (0.0f == test
                                && (0.0f
                                        == f->func.f_fma(0.0f, s2[j], s3[j],
                                                         FLUSHED)
                                    || 0.0f
                                        == f->func.f_fma(-0.0f, s2[j], s3[j],
                                                         FLUSHED)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            // try with first two args as zero
                            if (IsFloatSubnormal(s2[j]))
                            { // its fun to have fun,
                                double correct4, correct5;
                                float err4, err5;

                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                correct2 = f->func.f_fma(0.0f, 0.0f, s3[j],
                                                         CORRECTLY_ROUNDED);
                                correct3 = f->func.f_fma(-0.0f, 0.0f, s3[j],
                                                         CORRECTLY_ROUNDED);
                                correct4 = f->func.f_fma(0.0f, -0.0f, s3[j],
                                                         CORRECTLY_ROUNDED);
                                correct5 = f->func.f_fma(-0.0f, -0.0f, s3[j],
                                                         CORRECTLY_ROUNDED);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsFloatInfinity(correct2)
                                        || IsFloatNaN(correct2)
                                        || IsFloatInfinity(correct3)
                                        || IsFloatNaN(correct3)
                                        || IsFloatInfinity(correct4)
                                        || IsFloatNaN(correct4)
                                        || IsFloatInfinity(correct5)
                                        || IsFloatNaN(correct5))
                                        continue;
                                }

                                err2 = Ulp_Error(test, correct2);
                                err3 = Ulp_Error(test, correct3);
                                err4 = Ulp_Error(test, correct4);
                                err5 = Ulp_Error(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= float_ulps))
                                        && (!(fabsf(err3) <= float_ulps))
                                        && (!(fabsf(err4) <= float_ulps))
                                        && (!(fabsf(err5) <= float_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (0.0f == test
                                    && (0.0f
                                            == f->func.f_fma(0.0f, 0.0f, s3[j],
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(-0.0f, 0.0f, s3[j],
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(0.0f, -0.0f, s3[j],
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(-0.0f, -0.0f,
                                                             s3[j], FLUSHED)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }

                                if (IsFloatSubnormal(s3[j]))
                                {
                                    if (test == 0.0f) // 0*0+0 is 0
                                    {
                                        fail = 0;
                                        err = 0.0f;
                                    }
                                }
                            }
                            else if (IsFloatSubnormal(s3[j]))
                            {
                                double correct4, correct5;
                                float err4, err5;

                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                correct2 = f->func.f_fma(0.0f, s2[j], 0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct3 = f->func.f_fma(-0.0f, s2[j], 0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct4 = f->func.f_fma(0.0f, s2[j], -0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct5 = f->func.f_fma(-0.0f, s2[j], -0.0f,
                                                         CORRECTLY_ROUNDED);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsFloatInfinity(correct2)
                                        || IsFloatNaN(correct2)
                                        || IsFloatInfinity(correct3)
                                        || IsFloatNaN(correct3)
                                        || IsFloatInfinity(correct4)
                                        || IsFloatNaN(correct4)
                                        || IsFloatInfinity(correct5)
                                        || IsFloatNaN(correct5))
                                        continue;
                                }

                                err2 = Ulp_Error(test, correct2);
                                err3 = Ulp_Error(test, correct3);
                                err4 = Ulp_Error(test, correct4);
                                err5 = Ulp_Error(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= float_ulps))
                                        && (!(fabsf(err3) <= float_ulps))
                                        && (!(fabsf(err4) <= float_ulps))
                                        && (!(fabsf(err5) <= float_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (0.0f == test
                                    && (0.0f
                                            == f->func.f_fma(0.0f, s2[j], 0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(-0.0f, s2[j], 0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(0.0f, s2[j], -0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(-0.0f, s2[j],
                                                             -0.0f, FLUSHED)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsFloatSubnormal(s2[j]))
                        {
                            double correct2, correct3;
                            float err2, err3;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            correct2 = f->func.f_fma(s[j], 0.0f, s3[j],
                                                     CORRECTLY_ROUNDED);
                            correct3 = f->func.f_fma(s[j], -0.0f, s3[j],
                                                     CORRECTLY_ROUNDED);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correct2)
                                    || IsFloatNaN(correct2)
                                    || IsFloatInfinity(correct3)
                                    || IsFloatNaN(correct3))
                                    continue;
                            }

                            err2 = Ulp_Error(test, correct2);
                            err3 = Ulp_Error(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= float_ulps))
                                    && (!(fabsf(err3) <= float_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (0.0f == test
                                && (0.0f
                                        == f->func.f_fma(s[j], 0.0f, s3[j],
                                                         FLUSHED)
                                    || 0.0f
                                        == f->func.f_fma(s[j], -0.0f, s3[j],
                                                         FLUSHED)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            // try with second two args as zero
                            if (IsFloatSubnormal(s3[j]))
                            {
                                double correct4, correct5;
                                float err4, err5;

                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                correct2 = f->func.f_fma(s[j], 0.0f, 0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct3 = f->func.f_fma(s[j], -0.0f, 0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct4 = f->func.f_fma(s[j], 0.0f, -0.0f,
                                                         CORRECTLY_ROUNDED);
                                correct5 = f->func.f_fma(s[j], -0.0f, -0.0f,
                                                         CORRECTLY_ROUNDED);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsFloatInfinity(correct2)
                                        || IsFloatNaN(correct2)
                                        || IsFloatInfinity(correct3)
                                        || IsFloatNaN(correct3)
                                        || IsFloatInfinity(correct4)
                                        || IsFloatNaN(correct4)
                                        || IsFloatInfinity(correct5)
                                        || IsFloatNaN(correct5))
                                        continue;
                                }

                                err2 = Ulp_Error(test, correct2);
                                err3 = Ulp_Error(test, correct3);
                                err4 = Ulp_Error(test, correct4);
                                err5 = Ulp_Error(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= float_ulps))
                                        && (!(fabsf(err3) <= float_ulps))
                                        && (!(fabsf(err4) <= float_ulps))
                                        && (!(fabsf(err5) <= float_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (0.0f == test
                                    && (0.0f
                                            == f->func.f_fma(s[j], 0.0f, 0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(s[j], -0.0f, 0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(s[j], 0.0f, -0.0f,
                                                             FLUSHED)
                                        || 0.0f
                                            == f->func.f_fma(s[j], -0.0f, -0.0f,
                                                             FLUSHED)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsFloatSubnormal(s3[j]))
                        {
                            double correct2, correct3;
                            float err2, err3;

                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            correct2 = f->func.f_fma(s[j], s2[j], 0.0f,
                                                     CORRECTLY_ROUNDED);
                            correct3 = f->func.f_fma(s[j], s2[j], -0.0f,
                                                     CORRECTLY_ROUNDED);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsFloatInfinity(correct2)
                                    || IsFloatNaN(correct2)
                                    || IsFloatInfinity(correct3)
                                    || IsFloatNaN(correct3))
                                    continue;
                            }

                            err2 = Ulp_Error(test, correct2);
                            err3 = Ulp_Error(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= float_ulps))
                                    && (!(fabsf(err3) <= float_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (0.0f == test
                                && (0.0f
                                        == f->func.f_fma(s[j], s2[j], 0.0f,
                                                         FLUSHED)
                                    || 0.0f
                                        == f->func.f_fma(s[j], s2[j], -0.0f,
                                                         FLUSHED)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }
                        }
                    }

                    if (fabsf(err) > maxError)
                    {
                        maxError = fabsf(err);
                        maxErrorVal = s[j];
                        maxErrorVal2 = s2[j];
                        maxErrorVal3 = s3[j];
                    }

                    if (fail)
                    {
                        vlog_error(
                            "\nERROR: %s%s: %f ulp error at {%a, %a, %a} "
                            "({0x%8.8x, 0x%8.8x, 0x%8.8x}): *%a vs. %a\n",
                            f->name, sizeNames[k], err, s[j], s2[j], s3[j],
                            ((cl_uint *)s)[j], ((cl_uint *)s2)[j],
                            ((cl_uint *)s3)[j], ((float *)gOut_Ref)[j], test);
                        return -1;
                    }
                }
            }
        }

        if (0 == (i & 0x0fffffff))
        {
            if (gVerboseBruteForce)
            {
                vlog("base:%14" PRIu64 " step:%10" PRIu64 " bufferSize:%10d \n",
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

        vlog("\t%8.2f @ {%a, %a, %a}", maxError, maxErrorVal, maxErrorVal2,
             maxErrorVal3);
    }

    vlog("\n");

    return CL_SUCCESS;
}
