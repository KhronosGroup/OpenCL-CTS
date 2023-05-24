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

static const char *crossKernelSource[] = {
    GeomTestBase::extension,
    "__kernel void sample_test(__global ", GeomTestBase::ftype, " *srcA, __global ", GeomTestBase::ftype, " *srcB, __global ", GeomTestBase::ftype, " *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    vstore%d( cross( vload%d( tid, srcA), vload%d( tid,  srcB) ), tid, dst );\n"
    "\n"
    "}\n"
};

// clang-format on
//--------------------------------------------------------------------------

struct CrossFPTest : public GeometricsFPTest
{
    CrossFPTest(cl_device_id d, cl_context c, cl_command_queue q)
        : GeometricsFPTest(d, c, q)
    {}
    cl_int SetUp(int elements) override;
    cl_int RunSingleTest(const GeomTestBase *p) override;

    template <typename T> cl_int CrossKernel(const GeomTestParams<T> &p);
};

//--------------------------------------------------------------------------
template <typename T>
static bool verify_cross(int i, const T *const lhs, const T *const rhs,
                         const T *const inA, const T *const inB,
                         const double *const etol)
{
    if (std::is_same<T, half>::value == false)
    {
        double errs[] = { fabs(lhs[0] - rhs[0]), fabs(lhs[1] - rhs[1]),
                          fabs(lhs[2] - rhs[2]) };

        if (errs[0] > etol[0] || errs[1] > etol[1] || errs[2] > etol[2])
        {
            char printout[256];
            std::snprintf(printout, sizeof(printout),
                          "ERROR: Data sample %d does not validate! Expected "
                          "(%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, lhs[0], lhs[1], lhs[2], lhs[3], rhs[0], rhs[1],
                          rhs[2], rhs[3]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Input: (%a %a %a) and (%a %a %a)\n", inA[0],
                          inA[1], inA[2], inB[0], inB[1], inB[2]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Errors: (%a out of %a), (%a out of %a), (%a out "
                          "of %a)\n",
                          errs[0], etol[0], errs[1], etol[1], errs[2], etol[2]);
            log_error(printout);

            return true;
        }
    }
    else
    {
        float errs[] = { fabsf(HTF(lhs[0]) - HTF(rhs[0])),
                         fabsf(HTF(lhs[1]) - HTF(rhs[1])),
                         fabsf(HTF(lhs[2]) - HTF(rhs[2])) };

        if (errs[0] > etol[0] || errs[1] > etol[1] || errs[2] > etol[2])
        {
            char printout[256];
            std::snprintf(printout, sizeof(printout),
                          "ERROR: Data sample %d does not validate! Expected "
                          "(%a,%a,%a,%a), got (%a,%a,%a,%a)\n",
                          i, HTF(lhs[0]), HTF(lhs[1]), HTF(lhs[2]), HTF(lhs[3]),
                          rhs[0], rhs[1], rhs[2], rhs[3]);
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Input: (%a %a %a) and (%a %a %a)\n", HTF(inA[0]),
                          HTF(inA[1]), HTF(inA[2]), HTF(inB[0]), HTF(inB[1]),
                          HTF(inB[2]));
            log_error(printout);

            std::snprintf(printout, sizeof(printout),
                          "    Errors: (%a out of %a), (%a out of %a), (%a out "
                          "of %a)\n",
                          errs[0], etol[0], errs[1], etol[1], errs[2], etol[2]);
            log_error(printout);
            return true;
        }
    }
    return false;
}

//--------------------------------------------------------------------------

template <typename T>
void cross_product(const T *const vecA, const T *const vecB, T *const outVector,
                   double *const errorTolerances, double ulpTolerance)
{
    if (std::is_same<T, half>::value == false)
    {
        outVector[0] = (vecA[1] * vecB[2]) - (vecA[2] * vecB[1]);
        outVector[1] = (vecA[2] * vecB[0]) - (vecA[0] * vecB[2]);
        outVector[2] = (vecA[0] * vecB[1]) - (vecA[1] * vecB[0]);
        outVector[3] = 0.0f;

        errorTolerances[0] =
            fmax(fabs((double)vecA[1]),
                 fmax(fabs((double)vecB[2]),
                      fmax(fabs((double)vecA[2]), fabs((double)vecB[1]))));
        errorTolerances[1] =
            fmax(fabs((double)vecA[2]),
                 fmax(fabs((double)vecB[0]),
                      fmax(fabs((double)vecA[0]), fabs((double)vecB[2]))));
        errorTolerances[2] =
            fmax(fabs((double)vecA[0]),
                 fmax(fabs((double)vecB[1]),
                      fmax(fabs((double)vecA[1]), fabs((double)vecB[0]))));

        // This gives us max squared times ulp tolerance, i.e. the worst-case
        // expected variance we could expect from this result
        errorTolerances[0] = errorTolerances[0] * errorTolerances[0]
            * (ulpTolerance * FLT_EPSILON);
        errorTolerances[1] = errorTolerances[1] * errorTolerances[1]
            * (ulpTolerance * FLT_EPSILON);
        errorTolerances[2] = errorTolerances[2] * errorTolerances[2]
            * (ulpTolerance * FLT_EPSILON);
    }
    else
    {
        const float fvA[] = { HTF(vecA[0]), HTF(vecA[1]), HTF(vecA[2]) };
        const float fvB[] = { HTF(vecB[0]), HTF(vecB[1]), HTF(vecB[2]) };

        outVector[0] = HFF((fvA[1] * fvB[2]) - (fvA[2] * fvB[1]));
        outVector[1] = HFF((fvA[2] * fvB[0]) - (fvA[0] * fvB[2]));
        outVector[2] = HFF((fvA[0] * fvB[1]) - (fvA[1] * fvB[0]));
        outVector[3] = 0.0f;

        errorTolerances[0] =
            fmax(fabs(fvA[1]),
                 fmax(fabs(fvB[2]), fmaxf(fabs(fvA[2]), fabs(fvB[1]))));
        errorTolerances[1] =
            fmax(fabs(fvA[2]),
                 fmax(fabs(fvB[0]), fmaxf(fabs(fvA[0]), fabs(fvB[2]))));
        errorTolerances[2] =
            fmax(fabs(fvA[0]),
                 fmax(fabs(fvB[1]), fmaxf(fabs(fvA[1]), fabs(fvB[0]))));

        errorTolerances[0] = errorTolerances[0] * errorTolerances[0]
            * (ulpTolerance * CL_HALF_EPSILON);
        errorTolerances[1] = errorTolerances[1] * errorTolerances[1]
            * (ulpTolerance * CL_HALF_EPSILON);
        errorTolerances[2] = errorTolerances[2] * errorTolerances[2]
            * (ulpTolerance * CL_HALF_EPSILON);
    }
}

//--------------------------------------------------------------------------

cl_int CrossFPTest::SetUp(int elements)
{
    num_elements = elements;
    if (is_extension_available(device, "cl_khr_fp16"))
        params.emplace_back(new GeomTestParams<half>(kHalf, "", 3.f));

    params.emplace_back(new GeomTestParams<float>(kFloat, "", 3.f));

    if (is_extension_available(device, "cl_khr_fp64"))
        params.emplace_back(new GeomTestParams<double>(kDouble, "", 3.f));
    return GeometricsFPTest::SetUp(elements);
}

//--------------------------------------------------------------------------

cl_int CrossFPTest::RunSingleTest(const GeomTestBase *p)
{
    cl_int error = CL_SUCCESS;
    switch (p->dataType)
    {
        case kHalf:
            error = CrossKernel<cl_half>(*((GeomTestParams<half> *)p));
            break;
        case kFloat:
            error = CrossKernel<cl_float>(*((GeomTestParams<float> *)p));
            break;
        case kDouble:
            error = CrossKernel<cl_double>(*((GeomTestParams<double> *)p));
            break;
        default:
            test_error(-1, "CrossFPTest::RunSingleTest: incorrect fp type");
            break;
    }
    test_error(error, "CrossFPTest::RunSingleTest: test_geometrics failed");
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------

template <typename T>
cl_int CrossFPTest::CrossKernel(const GeomTestParams<T> &p)
{
    RandomSeed seed(gRandomSeed);

    cl_int error;
    /* Check the default rounding mode */
    if (std::is_same<T, float>::value && (floatRoundingMode == 0)
        || std::is_same<T, half>::value && (halfRoundingMode == 0))
        return -1;

    for (int vecsize = 3; vecsize <= 4; ++vecsize)
    {
        /* Make sure we adhere to the maximum individual allocation size and
         * global memory size limits. */
        size_t test_size = TEST_SIZE;
        size_t maxAllocSize = sizeof(T) * TEST_SIZE * vecsize;
        size_t totalBufSize = maxAllocSize * 3;

        error = VerifyTestSize(test_size, maxAllocSize, totalBufSize);
        test_error(error, "VerifyTestSize failed");

        size_t bufSize = sizeof(T) * test_size * vecsize;

        clProgramWrapper program;
        clKernelWrapper kernel;
        clMemWrapper streams[3];
        BufferOwningPtr<T> A(malloc(bufSize));
        BufferOwningPtr<T> B(malloc(bufSize));
        BufferOwningPtr<T> C(malloc(bufSize));
        T testVec[4];
        T *inDataA = A, *inDataB = B, *outData = C;
        size_t threads[1], localThreads[1];

        std::string str =
            concat_kernel(crossKernelSource,
                          sizeof(crossKernelSource) / sizeof(const char *));
        std::string kernelSource =
            string_format(str, vecsize, vecsize, vecsize);

        /* Create kernels */
        const char *programPtr = kernelSource.c_str();
        if (create_single_kernel_helper(context, &program, &kernel, 1,
                                        &programPtr, "sample_test"))
            return -1;

        /* Generate some streams. Note: deliberately do some random data in w to
         * verify that it gets ignored */
        for (int i = 0; i < test_size * vecsize; i++)
        {
            inDataA[i] = get_random<T>(-512, 512, seed);
            inDataB[i] = get_random<T>(-512, 512, seed);
        }
        bool res = FillWithTrickyNums(inDataA, inDataB, test_size * vecsize,
                                      vecsize, seed, p);
        test_assert_error_ret(res, "FillWithTrickyNums failed!", -1);

        streams[0] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataA, &error);
        if (streams[0] == NULL)
        {
            print_error(error, "ERROR: Creating input array A failed!\n");
            return -1;
        }
        streams[1] = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, bufSize,
                                    inDataB, &error);
        if (streams[1] == NULL)
        {
            print_error(error, "ERROR: Creating input array B failed!\n");
            return -1;
        }
        streams[2] =
            clCreateBuffer(context, CL_MEM_READ_WRITE, bufSize, NULL, &error);
        if (streams[2] == NULL)
        {
            print_error(error, "ERROR: Creating output array failed!\n");
            return -1;
        }

        /* Assign streams and execute */
        for (int i = 0; i < 3; i++)
        {
            error = clSetKernelArg(kernel, i, sizeof(streams[i]), &streams[i]);
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
        error = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, bufSize,
                                    outData, 0, NULL, NULL);
        test_error(error, "Unable to read output array!");

        /* And verify! */
        for (int i = 0; i < test_size; i++)
        {
            double errorTolerances[4] = { 0, 0, 0, 0 };
            // On an embedded device w/ round-to-zero, 3 ulps is the worst-case
            // tolerance for cross product
            cross_product<T>(&inDataA[i * vecsize], &inDataB[i * vecsize],
                             testVec, errorTolerances, p.ulpLimit);

            // RTZ devices accrue approximately double the amount of error per
            // operation.  Allow for that.
            if ((std::is_same<T, float>::value
                 && (floatRoundingMode == CL_FP_ROUND_TO_ZERO))
                || (std::is_same<T, half>::value
                    && (halfRoundingMode == CL_FP_ROUND_TO_ZERO)))
            {
                errorTolerances[0] *= 2.0;
                errorTolerances[1] *= 2.0;
                errorTolerances[2] *= 2.0;
                errorTolerances[3] *= 2.0;
            }

            if (verify_cross<T>(i, testVec, &outData[i * vecsize],
                                &inDataA[i * vecsize], &inDataB[i * vecsize],
                                errorTolerances))
            {
                char printout[64];
                std::snprintf(printout, sizeof(printout), "     ulp %f\n",
                              UlpError<T>(outData[i * vecsize + 1],
                                          ToDouble<T>(testVec[1])));
                log_error(printout);
                return -1;
            }
        }
    } // for(vecsize=...
    return CL_SUCCESS;
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

int test_geom_cross(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<CrossFPTest>(device, context, queue, num_elements);
}

//--------------------------------------------------------------------------
