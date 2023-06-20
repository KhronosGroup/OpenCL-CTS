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
#include "harness/stringHelpers.h"

#include "procs.h"
#include "test_base.h"

const char *binary_fn_code_pattern =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s%s *x, __global %s%s *y, __global %s%s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = %s(x[tid], y[tid]);\n"
"}\n";

const char *binary_fn_code_pattern_v3 =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s *x, __global %s *y, __global %s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(%s(vload3(tid,x), vload3(tid,y) ), tid, dst);\n"
"}\n";

const char *binary_fn_code_pattern_v3_scalar =
"%s\n" /* optional pragma */
"__kernel void test_fn(__global %s *x, __global %s *y, __global %s *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(%s(vload3(tid,x), y[tid] ), tid, dst);\n"
"}\n";

template <typename T>
int test_binary_fn(cl_device_id device, cl_context context,
                   cl_command_queue queue, int n_elems,
                   const std::string& fnName, bool vecSecParam,
                   VerifyFuncBinary<T> verifyFn)
{
    clMemWrapper streams[3];
    std::vector<T> input_ptr[2], output_ptr;

    std::vector<clProgramWrapper> programs;
    std::vector<clKernelWrapper> kernels;
    int err, i, j;
    MTdataHolder d = MTdataHolder(gRandomSeed);

    assert(BaseFunctionTest::type2name.find(sizeof(T))
           != BaseFunctionTest::type2name.end());
    auto tname = BaseFunctionTest::type2name[sizeof(T)];

    programs.resize(kTotalVecCount);
    kernels.resize(kTotalVecCount);

    int num_elements = n_elems * (1 << (kTotalVecCount - 1));

    for (i = 0; i < 2; i++) input_ptr[i].resize(num_elements);
    output_ptr.resize(num_elements);

    for( i = 0; i < 3; i++ )
    {
        streams[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(T) * num_elements, NULL, &err);
        test_error( err, "clCreateBuffer failed");
    }

    std::string pragma_str;
    if (std::is_same<T, float>::value)
    {
        for (j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = get_random_float(-0x20000000, 0x20000000, d);
            input_ptr[1][j] = get_random_float(-0x20000000, 0x20000000, d);
        }
    }
    else if (std::is_same<T, double>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        for (j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = get_random_double(-0x20000000, 0x20000000, d);
            input_ptr[1][j] = get_random_double(-0x20000000, 0x20000000, d);
        }
    }
    else if (std::is_same<T, half>::value)
    {
        const float fval = CL_HALF_MAX;
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        for (int j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = conv_to_half(get_random_float(-fval, fval, d));
            input_ptr[1][j] = conv_to_half(get_random_float(-fval, fval, d));
        }
    }

    for (i = 0; i < 2; i++)
    {
        err = clEnqueueWriteBuffer(queue, streams[i], CL_TRUE, 0,
                                   sizeof(T) * num_elements,
                                   &input_ptr[i].front(), 0, NULL, NULL);
        test_error(err, "Unable to write input buffer");
    }

    char vecSizeNames[][3] = { "", "2", "4", "8", "16", "3" };

    for (i = 0; i < kTotalVecCount; i++)
    {
        std::string kernelSource;
        if (i >= kVectorSizeCount)
        {
            if (vecSecParam)
            {
                std::string str = binary_fn_code_pattern_v3;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), fnName.c_str());
            }
            else
            {
                std::string str = binary_fn_code_pattern_v3_scalar;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), fnName.c_str());
            }
        }
        else
        {
            // do regular
            std::string str = binary_fn_code_pattern;
            kernelSource = str_sprintf(
                str, pragma_str.c_str(), tname.c_str(), vecSizeNames[i],
                tname.c_str(), vecSecParam ? vecSizeNames[i] : "",
                tname.c_str(), vecSizeNames[i], fnName.c_str());
        }
        const char* programPtr = kernelSource.c_str();
        err = create_single_kernel_helper(context, &programs[i], &kernels[i], 1,
                                          (const char**)&programPtr, "test_fn");
        test_error(err, "Unable to create kernel");

        for( j = 0; j < 3; j++ )
        {
            err =
                clSetKernelArg(kernels[i], j, sizeof(streams[j]), &streams[j]);
            test_error( err, "Unable to set kernel argument" );
        }

        size_t threads = (size_t)n_elems;

        err = clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &threads, NULL,
                                     0, NULL, NULL);
        test_error( err, "Unable to execute kernel" );

        err = clEnqueueReadBuffer(queue, streams[2], true, 0,
                                  sizeof(T) * num_elements, &output_ptr[0], 0,
                                  NULL, NULL);
        test_error( err, "Unable to read results" );

        if (verifyFn((T*)&input_ptr[0].front(), (T*)&input_ptr[1].front(),
                     &output_ptr[0], n_elems, g_arrVecSizes[i],
                     vecSecParam ? 1 : 0))
        {
            log_error("%s %s%d%s test failed\n", fnName.c_str(), tname.c_str(),
                      ((g_arrVecSizes[i])),
                      vecSecParam ? "" : std::string(", " + tname).c_str());
            err = -1;
        }
        else
        {
            log_info("%s %s%d%s test passed\n", fnName.c_str(), tname.c_str(),
                     ((g_arrVecSizes[i])),
                     vecSecParam ? "" : std::string(", " + tname).c_str());
            err = 0;
        }

        if (err)
            break;
    }
    return err;
}

namespace {

template <typename T>
int max_verify(const T* const x, const T* const y, const T* const out,
               int numElements, int vecSize, int vecParam)
{
    for (int i = 0; i < numElements; i++)
    {
        for (int j = 0; j < vecSize; j++)
        {
            int k = i * vecSize + j;
            int l = (k * vecParam + i * (1 - vecParam));
            T v = (conv_to_dbl(x[k]) < conv_to_dbl(y[l])) ? y[l] : x[k];
            if (v != out[k])
            {
                if (std::is_same<T, half>::value)
                    log_error("x[%d]=%g y[%d]=%g out[%d]=%g, expected %g. "
                              "(index %d is "
                              "vector %d, element %d, for vector size %d)\n",
                              k, conv_to_flt(x[k]), l, conv_to_flt(y[l]), k,
                              conv_to_flt(out[k]), v, k, i, j, vecSize);
                else
                    log_error("x[%d]=%g y[%d]=%g out[%d]=%g, expected %g. "
                              "(index %d is "
                              "vector %d, element %d, for vector size %d)\n",
                              k, x[k], l, y[l], k, out[k], v, k, i, j, vecSize);
                return -1;
            }
        }
    }
    return 0;
}

template <typename T>
int min_verify(const T* const x, const T* const y, const T* const out,
               int numElements, int vecSize, int vecParam)
{
    for (int i = 0; i < numElements; i++)
    {
        for (int j = 0; j < vecSize; j++)
        {
            int k = i * vecSize + j;
            int l = (k * vecParam + i * (1 - vecParam));
            T v = (conv_to_dbl(x[k]) > conv_to_dbl(y[l])) ? y[l] : x[k];
            if (v != out[k])
            {
                if (std::is_same<T, half>::value)
                    log_error("x[%d]=%g y[%d]=%g out[%d]=%g, expected %g. "
                              "(index %d is "
                              "vector %d, element %d, for vector size %d)\n",
                              k, conv_to_flt(x[k]), l, conv_to_flt(y[l]), k,
                              conv_to_flt(out[k]), v, k, i, j, vecSize);
                else
                    log_error("x[%d]=%g y[%d]=%g out[%d]=%g, expected %g. "
                              "(index %d is "
                              "vector %d, element %d, for vector size %d)\n",
                              k, x[k], l, y[l], k, out[k], v, k, i, j, vecSize);
                return -1;
            }
        }
    }
    return 0;
}

}

cl_int MaxTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_binary_fn<cl_half>(device, context, queue, num_elems,
                                        fnName.c_str(), vecParam,
                                        max_verify<cl_half>);
        test_error(error, "MaxTest::Run<cl_half> failed");
    }

    error = test_binary_fn<float>(device, context, queue, num_elems,
                                  fnName.c_str(), vecParam, max_verify<float>);
    test_error(error, "MaxTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_binary_fn<double>(device, context, queue, num_elems,
                                       fnName.c_str(), vecParam,
                                       max_verify<double>);
        test_error(error, "MaxTest::Run<double> failed");
    }

    return error;
}

cl_int MinTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_binary_fn<cl_half>(device, context, queue, num_elems,
                                        fnName.c_str(), vecParam,
                                        min_verify<cl_half>);
        test_error(error, "MinTest::Run<cl_half> failed");
    }

    error = test_binary_fn<float>(device, context, queue, num_elems,
                                  fnName.c_str(), vecParam, min_verify<float>);
    test_error(error, "MinTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_binary_fn<double>(device, context, queue, num_elems,
                                       fnName.c_str(), vecParam,
                                       min_verify<double>);
        test_error(error, "MinTest::Run<double> failed");
    }

    return error;
}

int test_min(cl_device_id device, cl_context context, cl_command_queue queue,
             int n_elems)
{
    return MakeAndRunTest<MinTest>(device, context, queue, n_elems, "min",
                                   true);
}

int test_minf(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<MinTest>(device, context, queue, n_elems, "min",
                                   false);
}

int test_fmin(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<MinTest>(device, context, queue, n_elems, "fmin",
                                   true);
}

int test_fminf(cl_device_id device, cl_context context, cl_command_queue queue,
               int n_elems)
{
    return MakeAndRunTest<MinTest>(device, context, queue, n_elems, "fmin",
                                   false);
}

int test_max(cl_device_id device, cl_context context, cl_command_queue queue,
             int n_elems)
{
    return MakeAndRunTest<MaxTest>(device, context, queue, n_elems, "max",
                                   true);
}

int test_maxf(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<MaxTest>(device, context, queue, n_elems, "max",
                                   false);
}

int test_fmax(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<MaxTest>(device, context, queue, n_elems, "fmax",
                                   true);
}

int test_fmaxf(cl_device_id device, cl_context context, cl_command_queue queue,
               int n_elems)
{
    return MakeAndRunTest<MaxTest>(device, context, queue, n_elems, "fmax",
                                   false);
}
