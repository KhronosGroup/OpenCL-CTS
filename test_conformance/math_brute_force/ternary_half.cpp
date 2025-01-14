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

#include <cinttypes>
#include <cstring>

#define CORRECTLY_ROUNDED 0
#define FLUSHED 1

namespace {

cl_int BuildKernelFn_HalfFn(cl_uint job_id, cl_uint thread_id UNUSED, void *p)
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

// A table of more difficult cases to get right
static const cl_half specialValuesHalf[] = {
    0xffff, 0x0000, 0x0001, 0x7c00, /*INFINITY*/
    0xfc00, /*-INFINITY*/
    0x8000, /*-0*/
    0x7bff, /*HALF_MAX*/
    0x0400, /*HALF_MIN*/
    0x03ff, /* Largest denormal */
    0x3c00, /* 1 */
    0xbc00, /* -1 */
    0x3555, /*nearest value to 1/3*/
    0x3bff, /*largest number less than one*/
    0xc000, /* -2 */
    0xfbff, /* -HALF_MAX */
    0x8400, /* -HALF_MIN */
    0x4248, /* M_PI_H */
    0xc248, /* -M_PI_H */
    0xbbff, /* Largest negative fraction */
};

constexpr size_t specialValuesHalfCount = ARRAY_SIZE(specialValuesHalf);

} // anonymous namespace

int TestFunc_Half_Half_Half_Half(const Func *f, MTdata d, bool relaxedMode)
{
    int error;

    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ || 0 == (CL_FP_DENORM & gHalfCapabilities);
    float maxErrorVal = 0.0f;
    float maxErrorVal2 = 0.0f;
    float maxErrorVal3 = 0.0f;
    uint64_t step = getTestStep(sizeof(cl_half), BUFFER_SIZE);

    constexpr size_t bufferElements = BUFFER_SIZE / sizeof(cl_half);

    std::vector<cl_uchar> overflow(bufferElements);
    float half_ulps = getAllowedUlpError(f, khalf, relaxedMode);
    int skipNanInf = (0 == strcmp("fma", f->nameInCode));

    logFunctionInfo(f->name, sizeof(cl_half), relaxedMode);

    // Init the kernels
    BuildKernelInfo build_info{ 1, kernels, programs, f->nameInCode };
    if ((error = ThreadPool_Do(BuildKernelFn_HalfFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        if (gSkipCorrectnessTesting) break;

        // Init input array
        cl_half *hp0 = (cl_half *)gIn;
        cl_half *hp1 = (cl_half *)gIn2;
        cl_half *hp2 = (cl_half *)gIn3;
        size_t idx = 0;

        if (i == 0)
        { // test edge cases
            uint32_t x, y, z;
            x = y = z = 0;
            for (; idx < bufferElements; idx++)
            {
                hp0[idx] = specialValuesHalf[x];
                hp1[idx] = specialValuesHalf[y];
                hp2[idx] = specialValuesHalf[z];

                if (++x >= specialValuesHalfCount)
                {
                    x = 0;
                    if (++y >= specialValuesHalfCount)
                    {
                        y = 0;
                        if (++z >= specialValuesHalfCount) break;
                    }
                }
            }
            if (idx == bufferElements)
                vlog_error("Test Error: not all special cases tested!\n");
        }

        auto any_value = [&d]() {
            float t = (float)((double)genrand_int32(d) / (double)0xFFFFFFFF);
            return HFF((1.0f - t) * CL_HALF_MIN + t * CL_HALF_MAX);
        };

        for (; idx < bufferElements; idx++)
        {
            hp0[idx] = any_value();
            hp1[idx] = any_value();
            hp2[idx] = any_value();
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
        if ((error = clFlush(gQueue)))
        {
            vlog("clFlush failed\n");
            return error;
        }

        // Calculate the correctly rounded reference result
        cl_half *res = (cl_half *)gOut_Ref;
        if (skipNanInf)
        {
            for (size_t j = 0; j < bufferElements; j++)
            {
                feclearexcept(FE_OVERFLOW);
                res[j] = HFD((double)f->dfunc.f_fff(HTF(hp0[j]), HTF(hp1[j]),
                                                    HTF(hp2[j])));
                overflow[j] =
                    FE_OVERFLOW == (FE_OVERFLOW & fetestexcept(FE_OVERFLOW));
            }
        }
        else
        {
            for (size_t j = 0; j < bufferElements; j++)
                res[j] = HFD((double)f->dfunc.f_fff(HTF(hp0[j]), HTF(hp1[j]),
                                                    HTF(hp2[j])));
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
        uint16_t *t = (uint16_t *)gOut_Ref;
        for (size_t j = 0; j < bufferElements; j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint16_t *q = (uint16_t *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    int fail;
                    cl_half test = ((cl_half *)q)[j];
                    double ref1 = (double)f->dfunc.f_fff(
                        HTF(hp0[j]), HTF(hp1[j]), HTF(hp2[j]));
                    cl_half correct = HFD(ref1);

                    // Per section 10 paragraph 6, accept any result if an input
                    // or output is a infinity or NaN or overflow
                    if (skipNanInf)
                    {
                        if (overflow[j] || IsHalfInfinity(correct)
                            || IsHalfNaN(correct) || IsHalfInfinity(hp0[j])
                            || IsHalfNaN(hp0[j]) || IsHalfInfinity(hp1[j])
                            || IsHalfNaN(hp1[j]) || IsHalfInfinity(hp2[j])
                            || IsHalfNaN(hp2[j]))
                            continue;
                    }

                    float err =
                        test != correct ? Ulp_Error_Half(test, ref1) : 0.f;
                    fail = !(fabsf(err) <= half_ulps);

                    if (fail && ftz)
                    {
                        // retry per section 6.5.3.2  with flushing on
                        float r = f->func.f_fma(HTF(hp0[j]), HTF(hp1[j]),
                                                HTF(hp2[j]), FLUSHED);
                        cl_half c = HFF(r);
                        if (0.0f == HTF(test) && IsHalfSubnormal(c))
                        {
                            fail = 0;
                            err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (fail && IsHalfSubnormal(hp0[j]))
                        { // look at me,
                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            float ref2 =
                                f->func.f_fma(0.0f, HTF(hp1[j]), HTF(hp2[j]),
                                              CORRECTLY_ROUNDED);
                            cl_half correct2 = HFF(ref2);
                            float ref3 =
                                f->func.f_fma(-0.0f, HTF(hp1[j]), HTF(hp2[j]),
                                              CORRECTLY_ROUNDED);
                            cl_half correct3 = HFF(ref3);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsHalfInfinity(correct2)
                                    || IsHalfNaN(correct2)
                                    || IsHalfInfinity(correct3)
                                    || IsHalfNaN(correct3))
                                    continue;
                            }

                            float err2 = test != correct2
                                ? Ulp_Error_Half(test, ref2)
                                : 0.f;
                            float err3 = test != correct3
                                ? Ulp_Error_Half(test, ref3)
                                : 0.f;
                            fail = fail
                                && ((!(fabsf(err2) <= half_ulps))
                                    && (!(fabsf(err3) <= half_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            float r3 = f->func.f_fma(0.0f, HTF(hp1[j]),
                                                     HTF(hp2[j]), FLUSHED);
                            float r4 = f->func.f_fma(-0.0f, HTF(hp1[j]),
                                                     HTF(hp2[j]), FLUSHED);
                            cl_half c3 = HFF(r3);
                            cl_half c4 = HFF(r4);

                            if (0.0f == HTF(test)
                                && (IsHalfSubnormal(c3) || IsHalfSubnormal(c4)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            // try with first two args as zero
                            if (IsHalfSubnormal(hp1[j]))
                            { // its fun to have fun,
                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                ref2 = f->func.f_fma(0.0f, 0.0f, HTF(hp2[j]),
                                                     CORRECTLY_ROUNDED);
                                correct2 = HFF(ref2);
                                ref3 = f->func.f_fma(-0.0f, 0.0f, HTF(hp2[j]),
                                                     CORRECTLY_ROUNDED);
                                correct3 = HFF(ref3);
                                float ref4 =
                                    f->func.f_fma(0.0f, -0.0f, HTF(hp2[j]),
                                                  CORRECTLY_ROUNDED);
                                cl_half correct4 = HFF(ref4);
                                float ref5 =
                                    f->func.f_fma(-0.0f, -0.0f, HTF(hp2[j]),
                                                  CORRECTLY_ROUNDED);
                                cl_half correct5 = HFF(ref5);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsHalfInfinity(correct2)
                                        || IsHalfNaN(correct2)
                                        || IsHalfInfinity(correct3)
                                        || IsHalfNaN(correct3)
                                        || IsHalfInfinity(correct4)
                                        || IsHalfNaN(correct4)
                                        || IsHalfInfinity(correct5)
                                        || IsHalfNaN(correct5))
                                        continue;
                                }

                                err2 = test != correct2
                                    ? Ulp_Error_Half(test, ref2)
                                    : 0.f;
                                err3 = test != correct3
                                    ? Ulp_Error_Half(test, ref3)
                                    : 0.f;
                                float err4 = test != correct4
                                    ? Ulp_Error_Half(test, ref4)
                                    : 0.f;
                                float err5 = test != correct5
                                    ? Ulp_Error_Half(test, ref5)
                                    : 0.f;
                                fail = fail
                                    && ((!(fabsf(err2) <= half_ulps))
                                        && (!(fabsf(err3) <= half_ulps))
                                        && (!(fabsf(err4) <= half_ulps))
                                        && (!(fabsf(err5) <= half_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                float r5 = f->func.f_fma(0.0f, 0.0f,
                                                         HTF(hp2[j]), FLUSHED);
                                float r6 = f->func.f_fma(-0.0f, 0.0f,
                                                         HTF(hp2[j]), FLUSHED);
                                float r7 = f->func.f_fma(0.0f, -0.0f,
                                                         HTF(hp2[j]), FLUSHED);
                                float r8 = f->func.f_fma(-0.0f, -0.0f,
                                                         HTF(hp2[j]), FLUSHED);

                                cl_half c5 = HFF(r5);
                                cl_half c6 = HFF(r6);
                                cl_half c7 = HFF(r7);
                                cl_half c8 = HFF(r8);
                                if (0.0f == HTF(test)
                                    && (IsHalfSubnormal(c5)
                                        || IsHalfSubnormal(c6)
                                        || IsHalfSubnormal(c7)
                                        || IsHalfSubnormal(c8)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                                if (IsHalfSubnormal(hp2[j]))
                                {
                                    if (test == 0.0f) // 0*0+0 is 0
                                    {
                                        fail = 0;
                                        err = 0.0f;
                                    }
                                }
                            }
                            else if (IsHalfSubnormal(hp2[j]))
                            {
                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                ref2 = f->func.f_fma(0.0f, HTF(hp1[j]), 0.0f,
                                                     CORRECTLY_ROUNDED);
                                correct2 = HFF(ref2);
                                ref3 = f->func.f_fma(-0.0f, HTF(hp1[j]), 0.0f,
                                                     CORRECTLY_ROUNDED);
                                correct3 = HFF(ref3);
                                float ref4 =
                                    f->func.f_fma(0.0f, HTF(hp1[j]), -0.0f,
                                                  CORRECTLY_ROUNDED);
                                cl_half correct4 = HFF(ref4);
                                float ref5 =
                                    f->func.f_fma(-0.0f, HTF(hp1[j]), -0.0f,
                                                  CORRECTLY_ROUNDED);
                                cl_half correct5 = HFF(ref5);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsHalfInfinity(correct2)
                                        || IsHalfNaN(correct2)
                                        || IsHalfInfinity(correct3)
                                        || IsHalfNaN(correct3)
                                        || IsHalfInfinity(correct4)
                                        || IsHalfNaN(correct4)
                                        || IsHalfInfinity(correct5)
                                        || IsHalfNaN(correct5))
                                        continue;
                                }

                                err2 = test != correct2
                                    ? Ulp_Error_Half(test, ref2)
                                    : 0.f;
                                err3 = test != correct3
                                    ? Ulp_Error_Half(test, ref3)
                                    : 0.f;
                                float err4 = test != correct4
                                    ? Ulp_Error_Half(test, ref4)
                                    : 0.f;
                                float err5 = test != correct5
                                    ? Ulp_Error_Half(test, ref5)
                                    : 0.f;
                                fail = fail
                                    && ((!(fabsf(err2) <= half_ulps))
                                        && (!(fabsf(err3) <= half_ulps))
                                        && (!(fabsf(err4) <= half_ulps))
                                        && (!(fabsf(err5) <= half_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                float r9 = f->func.f_fma(0.0f, HTF(hp1[j]),
                                                         0.0f, FLUSHED);
                                float r10 = f->func.f_fma(-0.0f, HTF(hp1[j]),
                                                          0.0f, FLUSHED);
                                float r11 = f->func.f_fma(0.0f, HTF(hp1[j]),
                                                          -0.0f, FLUSHED);
                                float r12 = f->func.f_fma(-0.0f, HTF(hp1[j]),
                                                          -0.0f, FLUSHED);

                                cl_half c9 = HFF(r9);
                                cl_half c10 = HFF(r10);
                                cl_half c11 = HFF(r11);
                                cl_half c12 = HFF(r12);
                                if (0.0f == HTF(test)
                                    && (IsHalfSubnormal(c9)
                                        || IsHalfSubnormal(c10)
                                        || IsHalfSubnormal(c11)
                                        || IsHalfSubnormal(c12)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsHalfSubnormal(hp1[j]))
                        {
                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            float ref2 =
                                f->func.f_fma(HTF(hp0[j]), 0.0f, HTF(hp2[j]),
                                              CORRECTLY_ROUNDED);
                            cl_half correct2 = HFF(ref2);
                            float ref3 =
                                f->func.f_fma(HTF(hp0[j]), -0.0f, HTF(hp2[j]),
                                              CORRECTLY_ROUNDED);
                            cl_half correct3 = HFF(ref3);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsHalfInfinity(correct2)
                                    || IsHalfNaN(correct2)
                                    || IsHalfInfinity(correct3)
                                    || IsHalfNaN(correct3))
                                    continue;
                            }

                            float err2 = test != correct2
                                ? Ulp_Error_Half(test, ref2)
                                : 0.f;
                            float err3 = test != correct3
                                ? Ulp_Error_Half(test, ref3)
                                : 0.f;
                            fail = fail
                                && ((!(fabsf(err2) <= half_ulps))
                                    && (!(fabsf(err3) <= half_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            float r7 = f->func.f_fma(HTF(hp0[j]), 0.0f,
                                                     HTF(hp2[j]), FLUSHED);
                            float r8 = f->func.f_fma(HTF(hp0[j]), -0.0f,
                                                     HTF(hp2[j]), FLUSHED);
                            cl_half c7 = HFF(r7);
                            cl_half c8 = HFF(r8);
                            if (0.0f == HTF(test)
                                && (IsHalfSubnormal(c7) || IsHalfSubnormal(c8)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }

                            // try with second two args as zero
                            if (IsHalfSubnormal(hp2[j]))
                            {
                                if (skipNanInf) feclearexcept(FE_OVERFLOW);

                                ref2 = f->func.f_fma(HTF(hp0[j]), 0.0f, 0.0f,
                                                     CORRECTLY_ROUNDED);
                                correct2 = HFF(ref2);
                                ref3 = f->func.f_fma(HTF(hp0[j]), -0.0f, 0.0f,
                                                     CORRECTLY_ROUNDED);
                                correct3 = HFF(ref3);
                                float ref4 =
                                    f->func.f_fma(HTF(hp0[j]), 0.0f, -0.0f,
                                                  CORRECTLY_ROUNDED);
                                cl_half correct4 = HFF(ref4);
                                float ref5 =
                                    f->func.f_fma(HTF(hp0[j]), -0.0f, -0.0f,
                                                  CORRECTLY_ROUNDED);
                                cl_half correct5 = HFF(ref5);

                                // Per section 10 paragraph 6, accept any result
                                // if an input or output is a infinity or NaN or
                                // overflow
                                if (!gInfNanSupport)
                                {
                                    if (fetestexcept(FE_OVERFLOW)) continue;

                                    // Note: no double rounding here.  Reference
                                    // functions calculate in single precision.
                                    if (IsHalfInfinity(correct2)
                                        || IsHalfNaN(correct2)
                                        || IsHalfInfinity(correct3)
                                        || IsHalfNaN(correct3)
                                        || IsHalfInfinity(correct4)
                                        || IsHalfNaN(correct4)
                                        || IsHalfInfinity(correct5)
                                        || IsHalfNaN(correct5))
                                        continue;
                                }

                                err2 = test != correct2
                                    ? Ulp_Error_Half(test, ref2)
                                    : 0.f;
                                err3 = test != correct3
                                    ? Ulp_Error_Half(test, ref3)
                                    : 0.f;
                                float err4 = test != correct4
                                    ? Ulp_Error_Half(test, ref4)
                                    : 0.f;
                                float err5 = test != correct5
                                    ? Ulp_Error_Half(test, ref5)
                                    : 0.f;
                                fail = fail
                                    && ((!(fabsf(err2) <= half_ulps))
                                        && (!(fabsf(err3) <= half_ulps))
                                        && (!(fabsf(err4) <= half_ulps))
                                        && (!(fabsf(err5) <= half_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                float r13 = f->func.f_fma(HTF(hp0[j]), 0.0f,
                                                          0.0f, FLUSHED);
                                float r14 = f->func.f_fma(HTF(hp0[j]), -0.0f,
                                                          0.0f, FLUSHED);
                                float r15 = f->func.f_fma(HTF(hp0[j]), 0.0f,
                                                          -0.0f, FLUSHED);
                                float r16 = f->func.f_fma(HTF(hp0[j]), -0.0f,
                                                          -0.0f, FLUSHED);

                                cl_half c9 = HFF(r13);
                                cl_half c10 = HFF(r14);
                                cl_half c11 = HFF(r15);
                                cl_half c12 = HFF(r16);
                                if (0.0f == HTF(test)
                                    && (IsHalfSubnormal(c9)
                                        || IsHalfSubnormal(c10)
                                        || IsHalfSubnormal(c11)
                                        || IsHalfSubnormal(c12)))
                                {
                                    fail = 0;
                                    err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsHalfSubnormal(hp2[j]))
                        {
                            if (skipNanInf) feclearexcept(FE_OVERFLOW);

                            float ref2 = f->func.f_fma(HTF(hp0[j]), HTF(hp1[j]),
                                                       0.0f, CORRECTLY_ROUNDED);
                            cl_half correct2 = HFF(ref2);
                            float ref3 =
                                f->func.f_fma(HTF(hp0[j]), HTF(hp1[j]), -0.0f,
                                              CORRECTLY_ROUNDED);
                            cl_half correct3 = HFF(ref3);

                            if (skipNanInf)
                            {
                                if (fetestexcept(FE_OVERFLOW)) continue;

                                // Note: no double rounding here.  Reference
                                // functions calculate in single precision.
                                if (IsHalfInfinity(correct2)
                                    || IsHalfNaN(correct2)
                                    || IsHalfInfinity(correct3)
                                    || IsHalfNaN(correct3))
                                    continue;
                            }

                            float err2 = test != correct2
                                ? Ulp_Error_Half(test, correct2)
                                : 0.f;
                            float err3 = test != correct3
                                ? Ulp_Error_Half(test, correct3)
                                : 0.f;
                            fail = fail
                                && ((!(fabsf(err2) <= half_ulps))
                                    && (!(fabsf(err3) <= half_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            float r17 = f->func.f_fma(HTF(hp0[j]), HTF(hp1[j]),
                                                      0.0f, FLUSHED);
                            float r18 = f->func.f_fma(HTF(hp0[j]), HTF(hp1[j]),
                                                      -0.0f, FLUSHED);
                            cl_half c13 = HFF(r17);
                            cl_half c14 = HFF(r18);
                            if (0.0f == HTF(test)
                                && (IsHalfSubnormal(c13)
                                    || IsHalfSubnormal(c14)))
                            {
                                fail = 0;
                                err = 0.0f;
                            }
                        }
                    }

                    if (fabsf(err) > maxError)
                    {
                        maxError = fabsf(err);
                        maxErrorVal = HTF(hp0[j]);
                        maxErrorVal2 = HTF(hp1[j]);
                        maxErrorVal3 = HTF(hp2[j]);
                    }

                    if (fail)
                    {
                        vlog_error(
                            "\nERROR: %s%s: %f ulp error at {%a, %a, %a} "
                            "({0x%4.4x, 0x%4.4x, 0x%4.4x}): *%a vs. %a\n",
                            f->name, sizeNames[k], err, HTF(hp0[j]),
                            HTF(hp1[j]), HTF(hp2[j]), hp0[j], hp1[j], hp2[j],
                            HTF(res[j]), HTF(test));
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
