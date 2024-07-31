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
        return GetTernaryKernel(kernel_name, builtin, ParameterType::Double,
                                ParameterType::Double, ParameterType::Double,
                                ParameterType::Double, vector_size_index);
    };
    return BuildKernels(info, job_id, generator);
}

// A table of more difficult cases to get right
const double specialValues[] = {
    -NAN,
    -INFINITY,
    -DBL_MAX,
    MAKE_HEX_DOUBLE(-0x1.0000000000001p64, -0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(-0x1.0p64, -0x1LL, 64),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp63, -0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p63, -0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(-0x1.0p63, -0x1LL, 63),
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp62, -0x1fffffffffffffLL, 10),
    -3.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p1, -0x18000000000001LL, -51),
    -2.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp1, -0x17ffffffffffffLL, -51),
    -2.0,
    MAKE_HEX_DOUBLE(-0x1.8000000000001p0, -0x18000000000001LL, -52),
    -1.5,
    MAKE_HEX_DOUBLE(-0x1.7ffffffffffffp0, -0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    -1.0,
    MAKE_HEX_DOUBLE(-0x1.fffffffffffffp-1, -0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p-1022, -0x10000000000001LL, -1074),
    -DBL_MIN,
    MAKE_HEX_DOUBLE(-0x0.fffffffffffffp-1022, -0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000fffp-1022, -0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.00000000000fep-1022, -0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ep-1022, -0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000cp-1022, -0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.000000000000ap-1022, -0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000003p-1022, -0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000002p-1022, -0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(-0x0.0000000000001p-1022, -0x00000000000001LL, -1074),
    -0.0,

    +NAN,
    +INFINITY,
    +DBL_MAX,
    MAKE_HEX_DOUBLE(+0x1.0000000000001p64, +0x10000000000001LL, 12),
    MAKE_HEX_DOUBLE(+0x1.0p64, +0x1LL, 64),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp63, +0x1fffffffffffffLL, 11),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p63, +0x10000000000001LL, 11),
    MAKE_HEX_DOUBLE(+0x1.0p63, +0x1LL, 63),
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp62, +0x1fffffffffffffLL, 10),
    +3.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p1, +0x18000000000001LL, -51),
    +2.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp1, +0x17ffffffffffffLL, -51),
    +2.0,
    MAKE_HEX_DOUBLE(+0x1.8000000000001p0, +0x18000000000001LL, -52),
    +1.5,
    MAKE_HEX_DOUBLE(+0x1.7ffffffffffffp0, +0x17ffffffffffffLL, -52),
    MAKE_HEX_DOUBLE(-0x1.0000000000001p0, -0x10000000000001LL, -52),
    +1.0,
    MAKE_HEX_DOUBLE(+0x1.fffffffffffffp-1, +0x1fffffffffffffLL, -53),
    MAKE_HEX_DOUBLE(+0x1.0000000000001p-1022, +0x10000000000001LL, -1074),
    +DBL_MIN,
    MAKE_HEX_DOUBLE(+0x0.fffffffffffffp-1022, +0x0fffffffffffffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000fffp-1022, +0x00000000000fffLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.00000000000fep-1022, +0x000000000000feLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ep-1022, +0x0000000000000eLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000cp-1022, +0x0000000000000cLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.000000000000ap-1022, +0x0000000000000aLL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000003p-1022, +0x00000000000003LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000002p-1022, +0x00000000000002LL, -1074),
    MAKE_HEX_DOUBLE(+0x0.0000000000001p-1022, +0x00000000000001LL, -1074),
    +0.0,
};

constexpr size_t specialValuesCount =
    sizeof(specialValues) / sizeof(specialValues[0]);

} // anonymous namespace

int TestFunc_Double_Double_Double_Double(const Func *f, MTdata d,
                                         bool relaxedMode)
{
    int error;
    Programs programs;
    const unsigned thread_id = 0; // Test is currently not multithreaded.
    KernelMatrix kernels;
    float maxError = 0.0f;
    int ftz = f->ftz || gForceFTZ;
    double maxErrorVal = 0.0f;
    double maxErrorVal2 = 0.0f;
    double maxErrorVal3 = 0.0f;
    uint64_t step = getTestStep(sizeof(double), BUFFER_SIZE);

    logFunctionInfo(f->name, sizeof(cl_double), relaxedMode);

    Force64BitFPUPrecision();

    // Init the kernels
    BuildKernelInfo build_info{ 1, kernels, programs, f->nameInCode,
                                relaxedMode };
    if ((error = ThreadPool_Do(BuildKernelFn,
                               gMaxVectorSizeIndex - gMinVectorSizeIndex,
                               &build_info)))
        return error;

    for (uint64_t i = 0; i < (1ULL << 32); i += step)
    {
        // Init input array
        double *p = (double *)gIn;
        double *p2 = (double *)gIn2;
        double *p3 = (double *)gIn3;
        size_t idx = 0;

        if (i == 0)
        { // test edge cases
            uint32_t x, y, z;
            x = y = z = 0;
            for (; idx < BUFFER_SIZE / sizeof(double); idx++)
            {
                p[idx] = specialValues[x];
                p2[idx] = specialValues[y];
                p3[idx] = specialValues[z];
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
            if (idx == BUFFER_SIZE / sizeof(double))
                vlog_error("Test Error: not all special cases tested!\n");
        }

        for (; idx < BUFFER_SIZE / sizeof(double); idx++)
        {
            p[idx] = DoubleFromUInt32(genrand_int32(d));
            p2[idx] = DoubleFromUInt32(genrand_int32(d));
            p3[idx] = DoubleFromUInt32(genrand_int32(d));
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
            size_t vectorSize = sizeof(cl_double) * sizeValues[j];
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
        double *r = (double *)gOut_Ref;
        double *s = (double *)gIn;
        double *s2 = (double *)gIn2;
        double *s3 = (double *)gIn3;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
            r[j] = (double)f->dfunc.f_fff(s[j], s2[j], s3[j]);

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

        if (gSkipCorrectnessTesting) break;

        // Verify data
        uint64_t *t = (uint64_t *)gOut_Ref;
        for (size_t j = 0; j < BUFFER_SIZE / sizeof(double); j++)
        {
            for (auto k = gMinVectorSizeIndex; k < gMaxVectorSizeIndex; k++)
            {
                uint64_t *q = (uint64_t *)(gOut[k]);

                // If we aren't getting the correctly rounded result
                if (t[j] != q[j])
                {
                    double test = ((double *)q)[j];
                    long double correct = f->dfunc.f_fff(s[j], s2[j], s3[j]);
                    float err = Bruteforce_Ulp_Error_Double(test, correct);
                    int fail = !(fabsf(err) <= f->double_ulps);

                    if (fail && (ftz || relaxedMode))
                    {
                        // retry per section 6.5.3.2
                        if (IsDoubleSubnormal(correct))
                        { // look at me,
                            fail = fail && (test != 0.0f);
                            if (!fail) err = 0.0f;
                        }

                        // retry per section 6.5.3.3
                        if (fail && IsDoubleSubnormal(s[j]))
                        { // look at me,
                            long double correct2 =
                                f->dfunc.f_fff(0.0, s2[j], s3[j]);
                            long double correct3 =
                                f->dfunc.f_fff(-0.0, s2[j], s3[j]);
                            float err2 =
                                Bruteforce_Ulp_Error_Double(test, correct2);
                            float err3 =
                                Bruteforce_Ulp_Error_Double(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= f->double_ulps))
                                    && (!(fabsf(err3) <= f->double_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (IsDoubleResultSubnormal(correct2,
                                                        f->double_ulps)
                                || IsDoubleResultSubnormal(correct3,
                                                           f->double_ulps))
                            { // look at me now,
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }

                            // try with first two args as zero
                            if (IsDoubleSubnormal(s2[j]))
                            { // its fun to have fun,
                                correct2 = f->dfunc.f_fff(0.0, 0.0, s3[j]);
                                correct3 = f->dfunc.f_fff(-0.0, 0.0, s3[j]);
                                long double correct4 =
                                    f->dfunc.f_fff(0.0, -0.0, s3[j]);
                                long double correct5 =
                                    f->dfunc.f_fff(-0.0, -0.0, s3[j]);
                                err2 =
                                    Bruteforce_Ulp_Error_Double(test, correct2);
                                err3 =
                                    Bruteforce_Ulp_Error_Double(test, correct3);
                                float err4 =
                                    Bruteforce_Ulp_Error_Double(test, correct4);
                                float err5 =
                                    Bruteforce_Ulp_Error_Double(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= f->double_ulps))
                                        && (!(fabsf(err3) <= f->double_ulps))
                                        && (!(fabsf(err4) <= f->double_ulps))
                                        && (!(fabsf(err5) <= f->double_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (IsDoubleResultSubnormal(correct2,
                                                            f->double_ulps)
                                    || IsDoubleResultSubnormal(correct3,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct4,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct5,
                                                               f->double_ulps))
                                {
                                    fail = fail && (test != 0.0f);
                                    if (!fail) err = 0.0f;
                                }

                                if (IsDoubleSubnormal(s3[j]))
                                { // but you have to know how!
                                    correct2 = f->dfunc.f_fff(0.0, 0.0, 0.0f);
                                    correct3 = f->dfunc.f_fff(-0.0, 0.0, 0.0f);
                                    correct4 = f->dfunc.f_fff(0.0, -0.0, 0.0f);
                                    correct5 = f->dfunc.f_fff(-0.0, -0.0, 0.0f);
                                    long double correct6 =
                                        f->dfunc.f_fff(0.0, 0.0, -0.0f);
                                    long double correct7 =
                                        f->dfunc.f_fff(-0.0, 0.0, -0.0f);
                                    long double correct8 =
                                        f->dfunc.f_fff(0.0, -0.0, -0.0f);
                                    long double correct9 =
                                        f->dfunc.f_fff(-0.0, -0.0, -0.0f);
                                    err2 = Bruteforce_Ulp_Error_Double(
                                        test, correct2);
                                    err3 = Bruteforce_Ulp_Error_Double(
                                        test, correct3);
                                    err4 = Bruteforce_Ulp_Error_Double(
                                        test, correct4);
                                    err5 = Bruteforce_Ulp_Error_Double(
                                        test, correct5);
                                    float err6 = Bruteforce_Ulp_Error_Double(
                                        test, correct6);
                                    float err7 = Bruteforce_Ulp_Error_Double(
                                        test, correct7);
                                    float err8 = Bruteforce_Ulp_Error_Double(
                                        test, correct8);
                                    float err9 = Bruteforce_Ulp_Error_Double(
                                        test, correct9);
                                    fail = fail
                                        && ((!(fabsf(err2) <= f->double_ulps))
                                            && (!(fabsf(err3)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err4)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err5)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err5)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err6)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err7)
                                                  <= f->double_ulps))
                                            && (!(fabsf(err8)
                                                  <= f->double_ulps)));
                                    if (fabsf(err2) < fabsf(err)) err = err2;
                                    if (fabsf(err3) < fabsf(err)) err = err3;
                                    if (fabsf(err4) < fabsf(err)) err = err4;
                                    if (fabsf(err5) < fabsf(err)) err = err5;
                                    if (fabsf(err6) < fabsf(err)) err = err6;
                                    if (fabsf(err7) < fabsf(err)) err = err7;
                                    if (fabsf(err8) < fabsf(err)) err = err8;
                                    if (fabsf(err9) < fabsf(err)) err = err9;

                                    // retry per section 6.5.3.4
                                    if (IsDoubleResultSubnormal(correct2,
                                                                f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct3, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct4, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct5, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct6, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct7, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct8, f->double_ulps)
                                        || IsDoubleResultSubnormal(
                                            correct9, f->double_ulps))
                                    {
                                        fail = fail && (test != 0.0f);
                                        if (!fail) err = 0.0f;
                                    }
                                }
                            }
                            else if (IsDoubleSubnormal(s3[j]))
                            {
                                correct2 = f->dfunc.f_fff(0.0, s2[j], 0.0);
                                correct3 = f->dfunc.f_fff(-0.0, s2[j], 0.0);
                                long double correct4 =
                                    f->dfunc.f_fff(0.0, s2[j], -0.0);
                                long double correct5 =
                                    f->dfunc.f_fff(-0.0, s2[j], -0.0);
                                err2 =
                                    Bruteforce_Ulp_Error_Double(test, correct2);
                                err3 =
                                    Bruteforce_Ulp_Error_Double(test, correct3);
                                float err4 =
                                    Bruteforce_Ulp_Error_Double(test, correct4);
                                float err5 =
                                    Bruteforce_Ulp_Error_Double(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= f->double_ulps))
                                        && (!(fabsf(err3) <= f->double_ulps))
                                        && (!(fabsf(err4) <= f->double_ulps))
                                        && (!(fabsf(err5) <= f->double_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (IsDoubleResultSubnormal(correct2,
                                                            f->double_ulps)
                                    || IsDoubleResultSubnormal(correct3,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct4,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct5,
                                                               f->double_ulps))
                                {
                                    fail = fail && (test != 0.0f);
                                    if (!fail) err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsDoubleSubnormal(s2[j]))
                        {
                            long double correct2 =
                                f->dfunc.f_fff(s[j], 0.0, s3[j]);
                            long double correct3 =
                                f->dfunc.f_fff(s[j], -0.0, s3[j]);
                            float err2 =
                                Bruteforce_Ulp_Error_Double(test, correct2);
                            float err3 =
                                Bruteforce_Ulp_Error_Double(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= f->double_ulps))
                                    && (!(fabsf(err3) <= f->double_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (IsDoubleResultSubnormal(correct2,
                                                        f->double_ulps)
                                || IsDoubleResultSubnormal(correct3,
                                                           f->double_ulps))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
                            }

                            // try with second two args as zero
                            if (IsDoubleSubnormal(s3[j]))
                            {
                                correct2 = f->dfunc.f_fff(s[j], 0.0, 0.0);
                                correct3 = f->dfunc.f_fff(s[j], -0.0, 0.0);
                                long double correct4 =
                                    f->dfunc.f_fff(s[j], 0.0, -0.0);
                                long double correct5 =
                                    f->dfunc.f_fff(s[j], -0.0, -0.0);
                                err2 =
                                    Bruteforce_Ulp_Error_Double(test, correct2);
                                err3 =
                                    Bruteforce_Ulp_Error_Double(test, correct3);
                                float err4 =
                                    Bruteforce_Ulp_Error_Double(test, correct4);
                                float err5 =
                                    Bruteforce_Ulp_Error_Double(test, correct5);
                                fail = fail
                                    && ((!(fabsf(err2) <= f->double_ulps))
                                        && (!(fabsf(err3) <= f->double_ulps))
                                        && (!(fabsf(err4) <= f->double_ulps))
                                        && (!(fabsf(err5) <= f->double_ulps)));
                                if (fabsf(err2) < fabsf(err)) err = err2;
                                if (fabsf(err3) < fabsf(err)) err = err3;
                                if (fabsf(err4) < fabsf(err)) err = err4;
                                if (fabsf(err5) < fabsf(err)) err = err5;

                                // retry per section 6.5.3.4
                                if (IsDoubleResultSubnormal(correct2,
                                                            f->double_ulps)
                                    || IsDoubleResultSubnormal(correct3,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct4,
                                                               f->double_ulps)
                                    || IsDoubleResultSubnormal(correct5,
                                                               f->double_ulps))
                                {
                                    fail = fail && (test != 0.0f);
                                    if (!fail) err = 0.0f;
                                }
                            }
                        }
                        else if (fail && IsDoubleSubnormal(s3[j]))
                        {
                            long double correct2 =
                                f->dfunc.f_fff(s[j], s2[j], 0.0);
                            long double correct3 =
                                f->dfunc.f_fff(s[j], s2[j], -0.0);
                            float err2 =
                                Bruteforce_Ulp_Error_Double(test, correct2);
                            float err3 =
                                Bruteforce_Ulp_Error_Double(test, correct3);
                            fail = fail
                                && ((!(fabsf(err2) <= f->double_ulps))
                                    && (!(fabsf(err3) <= f->double_ulps)));
                            if (fabsf(err2) < fabsf(err)) err = err2;
                            if (fabsf(err3) < fabsf(err)) err = err3;

                            // retry per section 6.5.3.4
                            if (IsDoubleResultSubnormal(correct2,
                                                        f->double_ulps)
                                || IsDoubleResultSubnormal(correct3,
                                                           f->double_ulps))
                            {
                                fail = fail && (test != 0.0f);
                                if (!fail) err = 0.0f;
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
                        vlog_error("\nERROR: %sD%s: %f ulp error at {%.13la, "
                                   "%.13la, %.13la}: *%.13la vs. %.13la\n",
                                   f->name, sizeNames[k], err, s[j], s2[j],
                                   s3[j], ((double *)gOut_Ref)[j], test);
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

        vlog("\t%8.2f @ {%a, %a, %a}", maxError, maxErrorVal, maxErrorVal2,
             maxErrorVal3);
    }

    vlog("\n");

    return CL_SUCCESS;
}
