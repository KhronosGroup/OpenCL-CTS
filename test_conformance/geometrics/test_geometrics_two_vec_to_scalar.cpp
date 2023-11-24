//
// Copyright (c) 2022 The Khronos Group Inc.
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
#include "test_geometrics_base.h"
#include "harness/compat.h"

#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/errorHelpers.h"
#include <float.h>
#include <limits>
#include <cmath>

#include <CL/cl_half.h>

//--------------------------------------------------------------------------/
// clang-format off
// for readability sake keep this section unformatted

static const char *TwoVecToScalarToScalarV1 = "dst[tid] = %s( srcA[tid], srcB[tid] );\n";
static const char *TwoVecToScalarToScalarVn = "dst[tid] = %s( vload%d( tid, srcA), vload%d( tid, srcB) );\n";

//--------------------------------------------------------------------------/

static const char *TwoVecToScalarKernelPattern[] = {
    GeomTestBase::extension,
    "__kernel void sample_test(__global ", GeomTestBase::ftype, " *srcA, __global ",
        GeomTestBase::ftype, " *srcB, __global ", GeomTestBase::ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    , GeomTestBase::load_store,
    "}\n"
};

// clang-format on
//--------------------------------------------------------------------------/

template <typename T>
using TwoVecToScalarVerifyFunc = double (*)(const T *, const T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct TwoVecToScalarTestParams : public GeomTestParams<T>
{
    TwoVecToScalarTestParams(const TwoVecToScalarVerifyFunc<T> &fn,
                             const ExplicitTypes &dt, const std::string &name,
                             const float &ulp)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn)
    {}

    TwoVecToScalarVerifyFunc<T> verifyFunc;
};

//--------------------------------------------------------------------------

struct TwoVecToScalarFPTest : public GeometricsFPTest
{
    TwoVecToScalarFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}
    cl_int RunSingleTest(const GeomTestBase *p) override;

    template <typename T>
    cl_int TwoVecToScalar(const TwoVecToScalarTestParams<T> &p);

    template <typename T>
    T GetMaxValue(const T *const, const T *const, const size_t &,
                  const TwoVecToScalarTestParams<T> &);

    template <typename T>
    cl_int TwoVecToScalarKernel(const size_t &, const MTdata &,
                                const TwoVecToScalarTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct DotProdFPTest : public TwoVecToScalarFPTest
{
    DotProdFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoVecToScalarFPTest(d, c, q)
    {}
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

struct FastDistanceFPTest : public TwoVecToScalarFPTest
{
    FastDistanceFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoVecToScalarFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

struct DistanceFPTest : public TwoVecToScalarFPTest
{
    DistanceFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : TwoVecToScalarFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int DistTest(TwoVecToScalarTestParams<T> &p);
};

//--------------------------------------------------------------------------

template <typename T> bool signbit_fp(const T &a)
{
    if (std::is_same<T, half>::value)
        return static_cast<half>(a) & 0x8000 ? 1 : 0;
    else
        return std::signbit(std::is_same<T, float>::value ? (float)a
                                                          : (double)a);
}

//--------------------------------------------------------------------------

template <typename T> double mad_fp(const T &a, const T &b, const T &c)
{
    if (!GeometricsFPTest::isfinite_fp<T>(a))
        return signbit_fp<T>(a) ? -INFINITY : INFINITY;
    if (!GeometricsFPTest::isfinite_fp<T>(b))
        return signbit_fp<T>(b) ? -INFINITY : INFINITY;
    if (!GeometricsFPTest::isfinite_fp<T>(c))
        return signbit_fp<T>(c) ? -INFINITY : INFINITY;
    if (std::is_same<T, half>::value)
        return HTF(a) * HTF(b) + HTF(c);
    else
        return a * b + c;
}

//--------------------------------------------------------------------------

template <typename T>
double verifyDot(const T *srcA, const T *srcB, size_t vecSize)
{
    double total = 0.f;
    if (std::is_same<T, half>::value)
    {
        for (unsigned int i = vecSize; i--;)
            total = mad_fp<T>(srcA[i], srcB[i], HFF(total));
        return (!GeometricsFPTest::isfinite_fp<T>(HFF(total)))
            ? (signbit_fp<T>(total) ? -INFINITY : INFINITY)
            : total;
    }
    else
    {
        for (unsigned int i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcB[i];
        return total;
    }
}

//--------------------------------------------------------------------------

double verifyFastDistance(const float *srcA, const float *srcB, size_t vecSize)
{
    double total = 0, value;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    for (i = 0; i < vecSize; i++)
    {
        value = (double)srcA[i] - (double)srcB[i];
        total += value * value;
    }

    return sqrt(total);
}

//--------------------------------------------------------------------------

template <typename T>
double verifyDistance(const T *srcA, const T *srcB, size_t vecSize)
{
    if (std::is_same<T, double>::value)
    {
        double diff[4];
        for (unsigned i = 0; i < vecSize; i++) diff[i] = srcA[i] - srcB[i];

        return verifyLength<T>((T *)diff, vecSize);
    }
    else
    {
        double total = 0, value;
        if (std::is_same<T, half>::value)
        {
            // We calculate the distance as a double, to try and make up for the
            // fact that the GPU has better precision distance since it's a
            // single op
            for (unsigned i = 0; i < vecSize; i++)
            {
                value = (double)HTF(srcA[i]) - (double)HTF(srcB[i]);
                total += value * value;
            }
        }
        else
        {
            for (unsigned i = 0; i < vecSize; i++)
            {
                value = (double)srcA[i] - (double)srcB[i];
                total += value * value;
            }
        }
        return sqrt(total);
    }
}

//--------------------------------------------------------------------------

cl_int TwoVecToScalarFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error =
                TwoVecToScalar<cl_half>(*((TwoVecToScalarTestParams<half> *)p));
            break;
        case kFloat:
            error = TwoVecToScalar<cl_float>(
                *((TwoVecToScalarTestParams<float> *)p));
            break;
        case kDouble:
            error = TwoVecToScalar<cl_double>(
                *((TwoVecToScalarTestParams<double> *)p));
            break;
        default:
            test_error(
                -1, "TwoVecToScalarFPTest::RunSingleTest: incorrect fp type");
            break;
    }
    test_error(error,
               "TwoVecToScalarFPTest::RunSingleTest: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
cl_int
TwoVecToScalarFPTest::TwoVecToScalar(const TwoVecToScalarTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        cl_int error = TwoVecToScalarKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   TwoVecToScalarFPTest::TwoVecToScalar vector size %zu "
                      "FAILED\n",
                      sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
T TwoVecToScalarFPTest::GetMaxValue(const T *const vecA, const T *const vecB,
                                    const size_t &vecSize,
                                    const TwoVecToScalarTestParams<T> &p)
{
    T a = max_fp<T>(abs_fp<T>(vecA[0]), abs_fp<T>(vecB[0]));
    for (size_t i = 1; i < vecSize; i++)
        a = max_fp<T>(abs_fp<T>(vecA[i]), max_fp<T>(abs_fp<T>(vecB[i]), a));
    return a;
}

//--------------------------------------------------------------------------

template <typename T>
cl_int
TwoVecToScalarFPTest::TwoVecToScalarKernel(const size_t &vecSize,
                                           const MTdata &d,
                                           const TwoVecToScalarTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[3];
    int error;
    size_t i, threads[1], localThreads[1];

    // error in sqrt
    float ulpLimit = p.ulpLimit;

    if ((std::is_same<T, float>::value
         && floatRoundingMode == CL_FP_ROUND_TO_ZERO)
        || (std::is_same<T, half>::value
            && halfRoundingMode == CL_FP_ROUND_TO_ZERO))
    {
        char dev_profile[256];
        error = clGetDeviceInfo(device, CL_DEVICE_PROFILE, sizeof(dev_profile),
                                dev_profile, NULL);
        test_error(error, "Unable to get device profile");
        if (0 == strcmp(dev_profile, "EMBEDDED_PROFILE"))
        {
            // rtz operations average twice the accrued error of rte operations
            ulpLimit *= 2.0f;
        }
    }

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * TEST_SIZE * vecSize;
    size_t totalBufSize = maxAllocSize * 2 + sizeof(T) * TEST_SIZE;

    error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t srcBufSize = sizeof(T) * test_size * vecSize;
    size_t dstBufSize = sizeof(T) * test_size;

    BufferOwningPtr<T> A(malloc(srcBufSize));
    BufferOwningPtr<T> B(malloc(srcBufSize));
    BufferOwningPtr<T> C(malloc(dstBufSize));

    T *inDataA = A, *inDataB = B, *outData = C;

    /* Create the source */
    if (vecSize == 1)
    {
        std::snprintf(GeomTestBase::load_store,
                      sizeof(GeomTestBase::load_store),
                      TwoVecToScalarToScalarV1, p.fnName.c_str());
    }
    else
    {
        std::snprintf(
            GeomTestBase::load_store, sizeof(GeomTestBase::load_store),
            TwoVecToScalarToScalarVn, p.fnName.c_str(), vecSize, vecSize);
    }
    std::string str = concat_kernel(TwoVecToScalarKernelPattern,
                                    sizeof(TwoVecToScalarKernelPattern)
                                        / sizeof(const char *));

    /* Create kernels */
    const char *programPtr = str.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
    {
        return -1;
    }
    /* Generate some streams */
    for (i = 0; i < test_size * vecSize; i++)
    {
        inDataA[i] = get_random<T>(-512, 512, d);
        inDataB[i] = get_random<T>(-512, 512, d);
    }
    bool res = FillWithTrickyNums(inDataA, inDataB, test_size * vecSize,
                                  vecSize, d, p);
    test_assert_error_ret(res, "FillWithTrickyNums failed!", -1);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataB, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "ERROR: Creating input array B failed!\n");
        return -1;
    }
    streams[2] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, dstBufSize, NULL, &error);
    if (streams[2] == NULL)
    {
        print_error(error, "ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    for (i = 0; i < 3; i++)
    {
        error = clSetKernelArg(kernel, (int)i, sizeof(streams[i]), &streams[i]);
        test_error(error, "Unable to set indexed kernel arguments");
    }

    /* Run the kernel */
    threads[0] = test_size;
    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[2], true, 0, dstBufSize, outData,
                                0, NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    int skipCount = 0;
    for (i = 0; i < test_size; i++)
    {
        T *src1 = inDataA + i * vecSize;
        T *src2 = inDataB + i * vecSize;
        double expected = p.verifyFunc(src1, src2, vecSize);
        bool isDif = (std::is_same<T, half>::value)
            ? (HFF(expected) != outData[i])
            : ((T)expected != outData[i]);
        if (isDif)
        {
            if (std::is_same<T, half>::value)
            {
                T expv = HFF(expected);
                if ((isnan_fp<T>(expv) && isnan_fp<T>(outData[i]))
                    || (!isfinite_fp<T>(expv) && !isfinite_fp<T>(outData[i])))
                    continue;
            }
            else if (isnan(expected) && isnan_fp<T>(outData[i]))
                continue;

            if ((std::is_same<T, float>::value && !floatHasInfNan)
                || (std::is_same<T, half>::value && !halfHasInfNan))
            {
                for (size_t ii = 0; ii < vecSize; ii++)
                {
                    if (!isfinite_fp<T>(src1[ii]) || !isfinite_fp<T>(src2[ii]))
                    {
                        skipCount++;
                        continue;
                    }
                }
                if (!isfinite_fp(expected))
                {
                    skipCount++;
                    continue;
                }
            }

            std::stringstream sstr;
            std::string printout;
            if (ulpLimit < 0)
            {
                // Limit below zero means we need to test via a computed error
                // (like cross product does)
                T maxValue = GetMaxValue<T>(inDataA + i * vecSize,
                                            inDataB + i * vecSize, vecSize, p);

                double error = 0.0, errorTolerance = 0.0;
                if (std::is_same<T, half>::value == false)
                {
                    // In this case (dot is the only one that gets here), the
                    // ulp is 2*vecSize - 1 (n + n-1 max # of errors)
                    errorTolerance = maxValue * maxValue
                        * (2.f * (float)vecSize - 1.f) * FLT_EPSILON;
                }
                else
                {
                    float mxv = HTF(maxValue);
                    errorTolerance = mxv * mxv * (2.f * (float)vecSize - 1.f)
                        * CL_HALF_EPSILON;
                }

                // Limit below zero means test via epsilon instead
                error = fabs(expected - ToDouble<T>(outData[i]));
                if (error > errorTolerance)
                {
                    printout = string_format(
                        "ERROR: Data sample %zu at size %zu does not "
                        "validate! Expected (%a), got (%a), sources (%a "
                        "and %a) error of %g against tolerance %g\n",
                        i, vecSize, expected, ToDouble<T>(outData[i]),
                        ToDouble<T>(inDataA[i * vecSize]),
                        ToDouble<T>(inDataB[i * vecSize]), (float)error,
                        (float)errorTolerance);

                    sstr << printout;
                    sstr << "\tvector A: ";
                    vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                    sstr << ", vector B: ";
                    vector2string<T>(sstr, inDataB + i * vecSize, vecSize);
                    log_error(sstr.str().c_str());
                    return -1;
                }
            }
            else
            {
                float error = UlpError<T>(outData[i], expected);
                if (fabsf(error) > ulpLimit)
                {
                    printout = string_format(
                        "ERROR: Data sample %zu at size %zu does not "
                        "validate! Expected (%a), got (%a), sources (%a "
                        "and %a) ulp of %f\n",
                        i, vecSize, expected, ToDouble<T>(outData[i]),
                        ToDouble<T>(inDataA[i * vecSize]),
                        ToDouble<T>(inDataB[i * vecSize]), error);

                    sstr << printout;
                    sstr << "\tvector A: ";
                    vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                    sstr << ", vector B: ";
                    vector2string<T>(sstr, inDataB + i * vecSize, vecSize);
                    log_error(sstr.str().c_str());
                    return -1;
                }
            }
        }
    }

    if (skipCount)
        log_info(
            "Skipped %d tests out of %d because they contained Infs or "
            "NaNs\n\tEMBEDDED_PROFILE Device does not support CL_FP_INF_NAN\n",
            skipCount, test_size);
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int DotProdFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(
            new TwoVecToScalarTestParams<half>(&verifyDot, kHalf, "dot", -1.f));

    params.emplace_back(
        new TwoVecToScalarTestParams<float>(&verifyDot, kFloat, "dot", -1.f));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new TwoVecToScalarTestParams<double>(
            &verifyDot, kDouble, "dot", -1.f));
    return TwoVecToScalarFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastDistanceFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((TwoVecToScalarTestParams<float> *)param);

    float ulpConst = p.ulpLimit;
    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = ulpConst + 2.f * sizes[size];

        cl_int error = TwoVecToScalarKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   FastDistanceFPTest::RunSingleTest vector size %zu FAILED\n",
                sizes[size]);
            return error;
        }
    }

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastDistanceFPTest::SetUp(int elements)
{
    // only float supports fast_distance
    params.emplace_back(new TwoVecToScalarTestParams<float>(
        &verifyFastDistance, kFloat, "fast_distance",
        8191.5f)); // 8191.5 + 2n ulp
    return TwoVecToScalarFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int DistanceFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = DistTest<half>(*((TwoVecToScalarTestParams<half> *)p));
            break;
        case kFloat:
            error = DistTest<float>(*((TwoVecToScalarTestParams<float> *)p));
            break;
        case kDouble:
            error = DistTest<double>(*((TwoVecToScalarTestParams<double> *)p));
            break;
        default:
            test_error(-1, "DistanceFPTest::Run: incorrect fp type");
            break;
    }
    test_error(error, "DistanceFPTest::Run: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
int DistanceFPTest::DistTest(TwoVecToScalarTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = ulpConst + 2.f * sizes[size];

        cl_int error = TwoVecToScalarKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   distance vector size %zu FAILED\n", sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int DistanceFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new TwoVecToScalarTestParams<half>(
            &verifyDistance, kHalf, "distance", 0.f)); // 2n ulp

    params.emplace_back(new TwoVecToScalarTestParams<float>(
        &verifyDistance, kFloat, "distance", 2.5f)); // 2.5 + 2n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new TwoVecToScalarTestParams<double>(
            &verifyDistance, kDouble, "distance", 5.5f)); // 5.5 + 2n ulp
    return TwoVecToScalarFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

int test_geom_dot(cl_device_id device, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<DotProdFPTest>(device, context, queue, num_elements);
}

//--------------------------------------------------------------------------

int test_geom_fast_distance(cl_device_id device, cl_context context,
                            cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastDistanceFPTest>(device, context, queue,
                                              num_elems);
}

//--------------------------------------------------------------------------

int test_geom_distance(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<DistanceFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------
