//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <vector>

#include "harness/deviceInfo.h"
#include "harness/typeWrappers.h"

#include "procs.h"
#include "test_base.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif


// clang-format off
const char *unary_fn_code_pattern =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s%s *src, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = %s(src[tid]);\n"
"}\n";

const char *unary_fn_code_pattern_v3 =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s *src, __global %s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(%s(vload3(tid,src)), tid, dst);\n"
"}\n";
// clang-format on


#define MAX_ERR 2.0f

namespace {


template <typename T> float UlpFn(const T &val, const double &r)
{
    if (std::is_same<T, double>::value)
        return Ulp_Error_Double(val, r);
    else if (std::is_same<T, float>::value)
        return Ulp_Error(val, r);
    else if (std::is_same<T, half>::value)
        return Ulp_Error(val, r);
}


template <typename T>
int verify_degrees(const T *const inptr, const T *const outptr, int n)
{
    float error, max_error = 0.0f;
    double r, max_val = NAN;
    int max_index = 0;

    for (int i = 0, j = 0; i < n; i++, j++)
    {
        r = (180.0 / M_PI) * inptr[i];

        error = UlpFn(outptr[i], r);

        if (fabsf(error) > max_error)
        {
            max_error = error;
            max_index = i;
            max_val = r;
            if (fabsf(error) > MAX_ERR)
            {
                log_error("%d) Error @ %a: *%a vs %a  (*%g vs %g) ulps: %f\n",
                          i, inptr[i], r, outptr[i], r, outptr[i], error);
                return 1;
            }
        }
    }

    log_info("degrees: Max error %f ulps at %d: *%a vs %a  (*%g vs %g)\n",
             max_error, max_index, max_val, outptr[max_index], max_val,
             outptr[max_index]);

    return 0;
}


template <typename T>
int verify_radians(const T *const inptr, const T *const outptr, int n)
{
    float error, max_error = 0.0f;
    double r, max_val = NAN;
    int max_index = 0;

    for (int i = 0, j = 0; i < n; i++, j++)
    {
        r = (M_PI / 180.0) * inptr[i];
        error = Ulp_Error(outptr[i], r);
        if (fabsf(error) > max_error)
        {
            max_error = error;
            max_index = i;
            max_val = r;
            if (fabsf(error) > MAX_ERR)
            {
                log_error("%d) Error @ %a: *%a vs %a  (*%g vs %g) ulps: %f\n",
                          i, inptr[i], r, outptr[i], r, outptr[i], error);
                return 1;
            }
        }
    }

    log_info("radians: Max error %f ulps at %d: *%a vs %a  (*%g vs %g)\n",
             max_error, max_index, max_val, outptr[max_index], max_val,
             outptr[max_index]);

    return 0;
}


template <typename T>
int verify_sign(const T *const inptr, const T *const outptr, int n)
{
    T r = 0;
    for (int i = 0; i < n; i++)
    {
        if (inptr[i] > 0.0f)
            r = 1.0;
        else if (inptr[i] < 0.0f)
            r = -1.0;
        else
            r = 0.0;
        if (r != outptr[i]) return -1;
    }
    return 0;
}

}


template <typename T>
int test_unary_fn(cl_device_id device, cl_context context,
                  cl_command_queue queue, int n_elems,
                  const std::string &fnName, VerifyFuncUnary<T> verifyFn)
{
    clMemWrapper streams[2];
    std::vector<T> input_ptr, output_ptr;

    std::vector<clProgramWrapper> programs;
    std::vector<clKernelWrapper> kernels;

    int err, i;
    MTdataHolder d = MTdataHolder(gRandomSeed);

    assert(BaseFunctionTest::type2name.find(sizeof(T))
           != BaseFunctionTest::type2name.end());
    auto tname = BaseFunctionTest::type2name[sizeof(T)];

    programs.resize(kTotalVecCount);
    kernels.resize(kTotalVecCount);

    int num_elements = n_elems * (1 << (kTotalVecCount - 1));

    input_ptr.resize(num_elements);
    output_ptr.resize(num_elements);

    for (i = 0; i < 2; i++)
    {
        streams[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(T) * num_elements, NULL, &err);
        test_error(err, "clCreateBuffer failed");
    }

    std::string pragma_str;
    if (std::is_same<T, float>::value)
    {
        for (int j = 0; j < num_elements; j++)
        {
            input_ptr[j] = get_random_float((float)(-100000.f * M_PI),
                                            (float)(100000.f * M_PI), d);
        }
    }
    else if (std::is_same<T, double>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        for (int j = 0; j < num_elements; j++)
        {
            input_ptr[j] =
                get_random_double(-100000.0 * M_PI, 100000.0 * M_PI, d);
        }
    }

    err = clEnqueueWriteBuffer(queue, streams[0], true, 0,
                               sizeof(T) * num_elements, &input_ptr.front(), 0,
                               NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    for (i = 0; i < kTotalVecCount; i++)
    {
        std::string kernelSource;
        char vecSizeNames[][3] = { "", "2", "4", "8", "16", "3" };

        if (i >= kVectorSizeCount)
        {
            std::string str = unary_fn_code_pattern_v3;
            kernelSource = string_format(str, pragma_str.c_str(), tname.c_str(),
                                         tname.c_str(), fnName.c_str());
        }
        else
        {
            std::string str = unary_fn_code_pattern;
            kernelSource = string_format(str, pragma_str.c_str(), tname.c_str(),
                                         vecSizeNames[i], tname.c_str(),
                                         vecSizeNames[i], fnName.c_str());
        }

        /* Create kernels */
        const char *programPtr = kernelSource.c_str();
        err =
            create_single_kernel_helper(context, &programs[i], &kernels[i], 1,
                                        (const char **)&programPtr, "test_fn");

        err = clSetKernelArg(kernels[i], 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernels[i], 1, sizeof streams[1], &streams[1]);
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            return -1;
        }

        // Line below is troublesome...
        size_t threads = (size_t)num_elements / ((g_arrVecSizes[i]));
        err = clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &threads, NULL,
                                     0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueNDRangeKernel failed\n");
            return -1;
        }

        cl_uint dead = 42;
        memset_pattern4(&output_ptr[0], &dead, sizeof(T) * num_elements);
        err = clEnqueueReadBuffer(queue, streams[1], true, 0,
                                  sizeof(T) * num_elements, &output_ptr[0], 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        if (verifyFn((T *)&input_ptr.front(), (T *)&output_ptr.front(),
                     n_elems * (i + 1)))
        {
            log_error("%s %s%d test failed\n", fnName.c_str(), tname.c_str(),
                      ((g_arrVecSizes[i])));
            err = -1;
        }
        else
        {
            log_info("%s %s%d test passed\n", fnName.c_str(), tname.c_str(),
                     ((g_arrVecSizes[i])));
        }

        if (err) break;
    }

    return err;
}


cl_int DegreesTest::Run()
{
    cl_int error = test_unary_fn<float>(device, context, queue, num_elems,
                                        fnName.c_str(), verify_degrees<float>);
    test_error(error, "DegreesTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_unary_fn<double>(device, context, queue, num_elems,
                                      fnName.c_str(), verify_degrees<double>);
        test_error(error, "DegreesTest::Run<double> failed");
    }

    return error;
}


cl_int RadiansTest::Run()
{
    cl_int error = test_unary_fn<float>(device, context, queue, num_elems,
                                        fnName.c_str(), verify_radians<float>);
    test_error(error, "RadiansTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_unary_fn<double>(device, context, queue, num_elems,
                                      fnName.c_str(), verify_radians<double>);
        test_error(error, "RadiansTest::Run<double> failed");
    }

    return error;
}


cl_int SignTest::Run()
{
    cl_int error = test_unary_fn<float>(device, context, queue, num_elems,
                                        fnName.c_str(), verify_sign<float>);
    test_error(error, "SignTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_unary_fn<double>(device, context, queue, num_elems,
                                      fnName.c_str(), verify_sign<double>);
        test_error(error, "SignTest::Run<double> failed");
    }

    return error;
}


int test_degrees(cl_device_id device, cl_context context,
                 cl_command_queue queue, int n_elems)
{
    return MakeAndRunTest<DegreesTest>(device, context, queue, n_elems,
                                       "degrees");
}


int test_radians(cl_device_id device, cl_context context,
                 cl_command_queue queue, int n_elems)
{
    return MakeAndRunTest<RadiansTest>(device, context, queue, n_elems,
                                       "radians");
}


int test_sign(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<SignTest>(device, context, queue, n_elems, "sign");
}
