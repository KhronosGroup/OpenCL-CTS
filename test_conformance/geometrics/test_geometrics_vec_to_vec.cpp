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

static const char *oneArgToVecV1 = "dst[tid] = %s( srcA[tid] );\n";
static const char *oneArgToVecVn = "vstore%d( %s( vload%d( tid, srcA) ), tid, dst );\n";

//--------------------------------------------------------------------------/

static const char *VecToVecKernelPattern[] = {
    GeomTestBase::extension,
    "__kernel void sample_test(__global ", GeomTestBase::ftype, " *srcA, __global ", GeomTestBase::ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    , GeomTestBase::load_store,
    "\n"
    "}\n"
};

// clang-format on
//--------------------------------------------------------------------------

template <typename T>
using VecToVecVerifyFunc = void (*)(const T *, T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct VecToVecTestParams : public GeomTestParams<T>
{
    VecToVecTestParams(const VecToVecVerifyFunc<T> &fn, const ExplicitTypes &dt,
                       const std::string &name, const float &ulp)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn)
    {}

    VecToVecVerifyFunc<T> verifyFunc;
};

//--------------------------------------------------------------------------

struct VecToVecFPTest : public GeometricsFPTest
{
    VecToVecFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}

    template <typename T>
    cl_int VecToVecKernel(const size_t &, const MTdata &,
                          const VecToVecTestParams<T> &p);

    template <typename T>
    cl_int VerifySubnormals(int &fail, const size_t &vecsize,
                            const T *const inA, T *const out, T *const expected,
                            const VecToVecTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct NormalizeFPTest : public VecToVecFPTest
{
    NormalizeFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : VecToVecFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int NormalizeTest(VecToVecTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct FastNormalizeFPTest : public VecToVecFPTest
{
    FastNormalizeFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : VecToVecFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

template <typename T>
void verifyNormalize(const T *srcA, T *dst, size_t vecSize)
{
    double total = 0, value;
    // We calculate everything as a double, to try and make up for the fact that
    // the GPU has better precision distance since it's a single op

    if (std::is_same<T, half>::value)
    {
        for (unsigned i = 0; i < vecSize; i++)
            total += (double)HTF(srcA[i]) * (double)HTF(srcA[i]);

        if (total == 0.f)
        {
            for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
            return;
        }

        if (total == INFINITY)
        {
            total = 0.0f;
            for (unsigned i = 0; i < vecSize; i++)
            {
                if (fabsf(HTF(srcA[i])) == INFINITY)
                    dst[i] = HFF(copysignf(1.0f, HTF(srcA[i])));
                else
                    dst[i] = HFF(copysignf(0.0f, HTF(srcA[i])));
                total += (double)HTF(dst[i]) * (double)HTF(dst[i]);
            }
            srcA = dst;
        }

        value = sqrt(total);
        for (unsigned i = 0; i < vecSize; i++)
            dst[i] = HFF(HTF(srcA[i]) / value);
    }
    else
    {
        for (unsigned i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcA[i];

        if (std::is_same<T, float>::value)
        {
            if (total == 0.f)
            {
                // Special edge case: copy vector over without change
                for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
                return;
            }

            // Deal with infinities
            if (total == INFINITY)
            {
                total = 0.0f;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    if (fabsf((float)srcA[i]) == INFINITY)
                        dst[i] = copysignf(1.0f, srcA[i]);
                    else
                        dst[i] = copysignf(0.0f, srcA[i]);
                    total += (double)dst[i] * (double)dst[i];
                }

                srcA = dst;
            }
        }
        else if (std::is_same<T, double>::value)
        {
            if (total < vecSize * DBL_MIN / DBL_EPSILON)
            { // we may have incurred denormalization loss -- rescale
                total = 0;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    dst[i] = srcA[i]
                        * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700); // exact
                    total += dst[i] * dst[i];
                }

                // If still zero
                if (total == 0.0)
                {
                    // Special edge case: copy vector over without change
                    for (unsigned i = 0; i < vecSize; i++) dst[i] = srcA[i];
                    return;
                }

                srcA = dst;
            }
            else if (total == INFINITY)
            { // we may have incurred spurious overflow
                double scale =
                    MAKE_HEX_DOUBLE(0x1.0p-512, 0x1LL, -512) / vecSize;
                total = 0;
                for (unsigned i = 0; i < vecSize; i++)
                {
                    dst[i] = srcA[i] * scale; // exact
                    total += dst[i] * dst[i];
                }

                // If there are infinities here, handle those
                if (total == INFINITY)
                {
                    total = 0;
                    for (unsigned i = 0; i < vecSize; i++)
                    {
                        if (isinf(dst[i]))
                        {
                            dst[i] = copysign(1.0, srcA[i]);
                            total += 1.0;
                        }
                        else
                            dst[i] = copysign(0.0, srcA[i]);
                    }
                }
                srcA = dst;
            }
        }
        value = sqrt(total);
        for (unsigned i = 0; i < vecSize; i++)
            dst[i] = (T)((double)srcA[i] / value);
    }
}

//--------------------------------------------------------------------------

template <typename T>
cl_int VecToVecFPTest::VerifySubnormals(int &fail, const size_t &vecSize,
                                        const T *const inA, T *const out,
                                        T *const expected,
                                        const VecToVecTestParams<T> &p)
{
    if (std::is_same<T, half>::value)
    {
        if (fail && halfFlushDenormsToZero)
        {
            T temp[4], expected2[4];
            for (size_t j = 0; j < vecSize; j++)
            {
                if (IsHalfSubnormal(inA[j]))
                    temp[j] = HFF(copysignf(0.0f, HTF(inA[j])));
                else
                    temp[j] = inA[j];
            }

            p.verifyFunc(temp, expected2, vecSize);
            fail = 0;

            for (size_t j = 0; j < vecSize; j++)
            {
                // We have to special case NAN
                if (isnan_fp<T>(out[j]) && isnan_fp<T>(expected[j])) continue;

                if (expected2[j] != out[j])
                {
                    float ulp_error =
                        UlpError<T>(out[j], ToDouble<T>(expected[j]));
                    if (fabsf(ulp_error) > p.ulpLimit
                        && IsHalfSubnormal(expected2[j]))
                    {
                        expected2[j] = 0.0f;
                        if (expected2[j] != out[j])
                        {
                            ulp_error =
                                UlpError<T>(out[j], ToDouble<T>(expected[j]));
                            if (fabsf(ulp_error) > p.ulpLimit)
                            {
                                fail = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    else if (std::is_same<T, float>::value)
    {
        if (fail && gFlushDenormsToZero)
        {
            T temp[4], expected2[4];
            for (size_t j = 0; j < vecSize; j++)
            {
                if (IsFloatSubnormal(inA[j]))
                    temp[j] = copysignf(0.0f, inA[j]);
                else
                    temp[j] = inA[j];
            }

            p.verifyFunc(temp, expected2, vecSize);
            fail = 0;

            for (size_t j = 0; j < vecSize; j++)
            {
                // We have to special case NAN
                if (isnan_fp<T>(out[j]) && isnan_fp<T>(expected[j])) continue;

                if (expected2[j] != out[j])
                {
                    float ulp_error =
                        UlpError<T>(out[j], ToDouble<T>(expected[j]));
                    if (fabsf(ulp_error) > p.ulpLimit
                        && IsFloatSubnormal(expected2[j]))
                    {
                        expected2[j] = 0.0f;
                        if (expected2[j] != out[j])
                        {
                            ulp_error =
                                UlpError<T>(out[j], ToDouble<T>(expected[j]));
                            if (fabsf(ulp_error) > p.ulpLimit)
                            {
                                fail = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
cl_int VecToVecFPTest::VecToVecKernel(const size_t &vecSize, const MTdata &d,
                                      const VecToVecTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * test_size * vecSize;
    size_t totalBufSize = maxAllocSize * 2;

    cl_int error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t bufSize = sizeof(T) * test_size * vecSize;

    BufferOwningPtr<T> A(malloc(bufSize));
    BufferOwningPtr<T> B(malloc(bufSize));
    size_t i, j, threads[1], localThreads[1];

    T *inDataA = A, *outData = B;
    float ulp_error = 0;

    /* Create the source */
    if (vecSize == 1)
        std::snprintf(GeomTestBase::load_store,
                      sizeof(GeomTestBase::load_store), oneArgToVecV1,
                      p.fnName.c_str());
    else
        std::snprintf(GeomTestBase::load_store,
                      sizeof(GeomTestBase::load_store), oneArgToVecVn, vecSize,
                      p.fnName.c_str(), vecSize);
    std::string str =
        concat_kernel(VecToVecKernelPattern,
                      sizeof(VecToVecKernelPattern) / sizeof(const char *));

    /* Create kernels */
    const char *programPtr = str.c_str();
    if (create_single_kernel_helper(context, &program, &kernel, 1,
                                    (const char **)&programPtr, "sample_test"))
        return -1;

    /* Initialize data.  First element always 0. */
    memset(inDataA, 0, sizeof(T) * vecSize);
    if (std::is_same<T, float>::value
        && p.fnName.find("fast_") != std::string::npos)
    {
        // keep problematic cases out of the fast function
        for (i = vecSize; i < test_size * vecSize; i++)
        {
            float z = get_random_float(-MAKE_HEX_FLOAT(0x1.0p60f, 1, 60),
                                       MAKE_HEX_FLOAT(0x1.0p60f, 1, 60), d);
            if (fabsf(z) < MAKE_HEX_FLOAT(0x1.0p-60f, 1, -60))
                z = copysignf(0.0f, z);
            inDataA[i] = z;
        }
    }
    else
    {
        for (i = vecSize; i < test_size * vecSize; i++)
            inDataA[i] = any_value<T>(d);
    }

    streams[0] =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize, inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &error);
    if (streams[1] == NULL)
    {
        print_error(error, "ERROR: Creating output array failed!\n");
        return -1;
    }

    /* Assign streams and execute */
    error = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    test_error(error, "Unable to set indexed kernel arguments");
    error = clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    test_error(error, "Unable to set indexed kernel arguments");

    /* Run the kernel */
    threads[0] = test_size;

    error = get_max_common_work_group_size(context, kernel, threads[0],
                                           &localThreads[0]);
    test_error(error, "Unable to get work group size to use");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads,
                                   localThreads, 0, NULL, NULL);
    test_error(error, "Unable to execute test kernel");

    /* Now get the results */
    error = clEnqueueReadBuffer(queue, streams[1], true, 0, bufSize, outData, 0,
                                NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    for (i = 0; i < test_size; i++)
    {
        T expected[4];
        int fail = 0;
        p.verifyFunc(inDataA + i * vecSize, expected, vecSize);
        for (j = 0; j < vecSize; j++)
        {
            // We have to special case NAN
            if (isnan_fp<T>(outData[i * vecSize + j])
                && isnan_fp<T>(expected[j]))
                continue;

            if (expected[j] != outData[i * vecSize + j])
            {
                ulp_error = UlpError<T>(outData[i * vecSize + j],
                                        ToDouble<T>(expected[j]));
                if (fabsf(ulp_error) > p.ulpLimit)
                {
                    fail = 1;
                    break;
                }
            }
        }

        // try again with subnormals flushed to zero if the platform flushes
        error = VerifySubnormals<T>(fail, vecSize, &inDataA[i * vecSize],
                                    &outData[i * vecSize], expected, p);
        test_error(error, "VerifySubnormals failed");

        if (fail)
        {
            std::stringstream sstr;
            std::string printout = string_format(
                "ERROR: Data sample {%zu,%zu} at size %zu does not validate! "
                "Expected %12.24f (%a), got %12.24f (%a), ulp %f\n",
                i, j, vecSize, ToDouble<T>(expected[j]),
                ToDouble<T>(expected[j]), ToDouble<T>(outData[i * vecSize + j]),
                ToDouble<T>(outData[i * vecSize + j]), ulp_error);
            sstr << printout;

            sstr << "       Source: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(inDataA[i * vecSize + q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(inDataA[i * vecSize + q])
                     << " ";
            sstr << "\n";
            sstr << "       Result: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(outData[i * vecSize + q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(outData[i * vecSize + q])
                     << " ";
            sstr << "\n";
            sstr << "       Expected: ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << ToDouble<T>(expected[q]) << " ";
            sstr << "\n             : ";
            for (size_t q = 0; q < vecSize; q++)
                sstr << std::hexfloat << ToDouble<T>(expected[q]) << " ";
            sstr << "\n";

            log_error(sstr.str().c_str());
            return -1;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int NormalizeFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = NormalizeTest<half>(*((VecToVecTestParams<half> *)p));
            break;
        case kFloat:
            error = NormalizeTest<float>(*((VecToVecTestParams<float> *)p));
            break;
        case kDouble:
            error = NormalizeTest<double>(*((VecToVecTestParams<double> *)p));
            break;
        default:
            test_error(-1, "NormalizeFPTest::Run: incorrect fp type");
            break;
    }
    test_error(error, "NormalizeFPTest::Run: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
int NormalizeFPTest::NormalizeTest(VecToVecTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + sizes[size]);

        cl_int error = VecToVecKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   NormalizeFPTest::NormalizeTest vector size %zu FAILED\n",
                sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int NormalizeFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new VecToVecTestParams<half>(
            &verifyNormalize, kHalf, "normalize", 1.f)); // 1 + n ulp

    params.emplace_back(new VecToVecTestParams<float>(
        &verifyNormalize, kFloat, "normalize", 2.f)); // 2 + n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new VecToVecTestParams<double>(
            &verifyNormalize, kDouble, "normalize", 4.5f)); // 4.5 + n ulp
    return VecToVecFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastNormalizeFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((VecToVecTestParams<float> *)param);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = std::ceil(ulpConst + sizes[size]);

        cl_int error = VecToVecKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   FastNormalizeFPTest::RunSingleTest vector size %zu "
                      "FAILED\n",
                      sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastNormalizeFPTest::SetUp(int elements)
{
    // only float supports fast_distance
    params.emplace_back(new VecToVecTestParams<float>(
        &verifyNormalize, kFloat, "fast_normalize", 8192.f)); // 8192 + n ulp
    return VecToVecFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

int test_geom_normalize(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<NormalizeFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_fast_normalize(cl_device_id device, cl_context context,
                             cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastNormalizeFPTest>(device, context, queue,
                                               num_elems);
}

//--------------------------------------------------------------------------
