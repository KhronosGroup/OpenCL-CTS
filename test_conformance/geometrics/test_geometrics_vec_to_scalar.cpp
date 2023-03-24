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

static const char *VecToScalarToScalarV1 = "dst[tid] = %s( srcA[tid] );\n";
static const char *VecToScalarToScalarVn = "dst[tid] = %s( vload%d( tid, srcA) );\n";

//--------------------------------------------------------------------------/

static const char *vecToScalarFloatKernelPattern[] = {
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
using VecToScalarVerifyFunc = double (*)(const T *, size_t);

//--------------------------------------------------------------------------

template <typename T> struct VecToScalarTestParams : public GeomTestParams<T>
{
    VecToScalarTestParams(const VecToScalarVerifyFunc<T> &fn,
                          const ExplicitTypes &dt, const std::string &name,
                          const float &ulp, const float &um)
        : GeomTestParams<T>(dt, name, ulp), verifyFunc(fn), ulpMult(um)
    {}

    VecToScalarVerifyFunc<T> verifyFunc;
    float ulpMult;
};

//--------------------------------------------------------------------------

struct VecToScalarFPTest : public GeometricsFPTest
{
    VecToScalarFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}

    template <typename T>
    cl_int VecToScalarKernel(const size_t &, const MTdata &,
                             const VecToScalarTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct LengthFPTest : public VecToScalarFPTest
{
    LengthFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : VecToScalarFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
    template <typename T> int LengthTest(VecToScalarTestParams<T> &p);
};

//--------------------------------------------------------------------------

struct FastLengthFPTest : public VecToScalarFPTest
{
    FastLengthFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : VecToScalarFPTest(d, c, q)
    {}

    cl_int RunSingleTest(const GeomTestBase *p) override;
    cl_int SetUp(int elements) override;
};

//--------------------------------------------------------------------------

template <typename T> static double verifyLength(const T *srcA, size_t vecSize)
{
    double total = 0;
    unsigned int i;

    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    if (std::is_same<T, half>::value)
    {
        for (i = 0; i < vecSize; i++)
            total += (double)HTF(srcA[i]) * (double)HTF(srcA[i]);
    }
    else
    {
        for (i = 0; i < vecSize; i++)
            total += (double)srcA[i] * (double)srcA[i];
    }


    if (std::is_same<T, double>::value)
    {
        // Deal with spurious overflow
        if (total == INFINITY)
        {
            total = 0.0;
            for (i = 0; i < vecSize; i++)
            {
                double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p-600, 0x1LL, -600);
                total += f * f;
            }

            return sqrt(total) * MAKE_HEX_DOUBLE(0x1.0p600, 0x1LL, 600);
        }

        // Deal with spurious underflow
        if (total < 4 /*max vector length*/ * DBL_MIN / DBL_EPSILON)
        {
            total = 0.0;
            for (i = 0; i < vecSize; i++)
            {
                double f = srcA[i] * MAKE_HEX_DOUBLE(0x1.0p700, 0x1LL, 700);
                total += f * f;
            }

            return sqrt(total) * MAKE_HEX_DOUBLE(0x1.0p-700, 0x1LL, -700);
        }
    }

    return sqrt(total);
}

//--------------------------------------------------------------------------

double verifyFastLength(const float *srcA, size_t vecSize)
{
    double total = 0;
    // We calculate the distance as a double, to try and make up for the fact
    // that the GPU has better precision distance since it's a single op
    for (unsigned i = 0; i < vecSize; i++)
        total += (double)srcA[i] * (double)srcA[i];

    return sqrt(total);
}

//--------------------------------------------------------------------------

template <typename T>
cl_int VecToScalarFPTest::VecToScalarKernel(const size_t &vecSize,
                                            const MTdata &d,
                                            const VecToScalarTestParams<T> &p)
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[2];

    /* Make sure we adhere to the maximum individual allocation size and
     * global memory size limits. */
    size_t test_size = TEST_SIZE;
    size_t maxAllocSize = sizeof(T) * test_size * vecSize;
    size_t totalBufSize = maxAllocSize + sizeof(T) * test_size;

    cl_int error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
    test_error(error, "VerifyTestSize failed");

    size_t srcBufSize = sizeof(T) * test_size * vecSize;
    size_t dstBufSize = sizeof(T) * test_size;

    BufferOwningPtr<T> A(malloc(srcBufSize));
    BufferOwningPtr<T> B(malloc(dstBufSize));

    size_t i, threads[1], localThreads[1];
    T *inDataA = A;
    T *outData = B;

    /* Create the source */
    if (vecSize == 1)
        std::snprintf(GeomTestBase::load_store,
                      sizeof(GeomTestBase::load_store), VecToScalarToScalarV1,
                      p.fnName.c_str());
    else
        std::snprintf(GeomTestBase::load_store,
                      sizeof(GeomTestBase::load_store), VecToScalarToScalarVn,
                      p.fnName.c_str(), vecSize);
    std::string str = concat_kernel(vecToScalarFloatKernelPattern,
                                    sizeof(vecToScalarFloatKernelPattern)
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
    }
    bool res = FillWithTrickyNums(inDataA, (T *)nullptr, test_size * vecSize,
                                  vecSize, d, p);
    test_assert_error_ret(res, "FillWithTrickyNums failed!", -1);

    streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, srcBufSize,
                                inDataA, &error);
    if (streams[0] == NULL)
    {
        print_error(error, "ERROR: Creating input array A failed!\n");
        return -1;
    }
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, dstBufSize, NULL, &error);
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
    error = clEnqueueReadBuffer(queue, streams[1], true, 0, dstBufSize, outData,
                                0, NULL, NULL);
    test_error(error, "Unable to read output array!");

    /* And verify! */
    for (i = 0; i < test_size; i++)
    {
        double expected = p.verifyFunc(inDataA + i * vecSize, vecSize);
        bool isDif = (std::is_same<T, half>::value)
            ? (HFF(expected) != outData[i])
            : ((T)expected != outData[i]);
        if (isDif)
        {
            float ulps = UlpError<T>(outData[i], expected);
            if (fabsf(ulps) <= p.ulpLimit) continue;

            // We have to special case NAN
            if (isnan_fp<T>(outData[i]) && isnan(expected)) continue;

            if (!(fabsf(ulps) < p.ulpLimit))
            {
                std::stringstream sstr;
                std::string printout = string_format(
                    "ERROR: Data sample %zu at size %zu does not validate! "
                    "Expected (%a), got (%a), source (%a), ulp %f\n",
                    i, vecSize, expected, ToDouble<T>(outData[i]),
                    ToDouble<T>(inDataA[i * vecSize]), ulps);

                sstr << printout;
                sstr << "\tvector: ";
                vector2string<T>(sstr, inDataA + i * vecSize, vecSize);
                log_error(sstr.str().c_str());
                return -1;
            }
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int LengthFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = LengthTest<half>(*((VecToScalarTestParams<half> *)p));
            break;
        case kFloat:
            error = LengthTest<float>(*((VecToScalarTestParams<float> *)p));
            break;
        case kDouble:
            error = LengthTest<double>(*((VecToScalarTestParams<double> *)p));
            break;
        default: test_error(-1, "LengthFPTest::Run: incorrect fp type"); break;
    }
    test_error(error, "LengthFPTest::RunSingleTest: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T> int LengthFPTest::LengthTest(VecToScalarTestParams<T> &p)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = ulpConst + p.ulpMult * sizes[size];

        cl_int error = VecToScalarKernel<T>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error("   LengthTest vector size %zu FAILED\n", sizes[size]);
            return error;
        }
    }

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int LengthFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new VecToScalarTestParams<half>(
            &verifyLength, kHalf, "length", 0.25f, 0.5f)); // 0.25 + 0.5n ulp

    params.emplace_back(new VecToScalarTestParams<float>(
        &verifyLength, kFloat, "length", 2.75f, 0.5f)); // 2.75 + 0.5n ulp

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new VecToScalarTestParams<double>(
            &verifyLength, kDouble, "length", 5.5f, 1.f)); // 5.5 + n ulp
    return VecToScalarFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int FastLengthFPTest::RunSingleTest(const GeomTestBase *param)
{
    const size_t sizes[] = { 1, 2, 3, 4, 0 };
    RandomSeed seed(gRandomSeed);
    auto p = *((VecToScalarTestParams<float> *)param);
    float ulpConst = p.ulpLimit;

    for (unsigned size = 0; sizes[size] != 0; size++)
    {
        p.ulpLimit = ulpConst + sizes[size];

        cl_int error = VecToScalarKernel<float>(sizes[size], seed, p);
        if (error != CL_SUCCESS)
        {
            log_error(
                "   FastLengthFPTest::RunSingleTest vector size %zu FAILED\n",
                sizes[size]);
            return error;
        }
    }
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

cl_int FastLengthFPTest::SetUp(int elements)
{
    // only float supports fast_length
    params.emplace_back(new VecToScalarTestParams<float>(
        &verifyFastLength, kFloat, "fast_length", 8191.5f,
        1.f)); // 8191.5 + n ulp
    return VecToScalarFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

int test_geom_length(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<LengthFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------

int test_geom_fast_length(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elems)
{
    return MakeAndRunTest<FastLengthFPTest>(device, context, queue, num_elems);
}

//--------------------------------------------------------------------------
