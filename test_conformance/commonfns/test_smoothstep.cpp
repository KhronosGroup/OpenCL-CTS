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

#include "harness/stringHelpers.h"

#include "procs.h"
#include "test_base.h"

const char *smoothstep_fn_code_pattern =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s%s *e0, __global %s%s *e1, __global %s%s "
    "*x, __global %s%s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = smoothstep(e0[tid], e1[tid], x[tid]);\n"
    "}\n";

const char *smoothstep_fn_code_pattern_v3 =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *e0, __global %s *e1, __global %s *x, "
    "__global %s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    vstore3(smoothstep(vload3(tid,e0), vload3(tid,e1), vload3(tid,x)), "
    "tid, dst);\n"
    "}\n";

const char *smoothstep_fn_code_pattern_v3_scalar =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *e0, __global %s *e1, __global %s *x, "
    "__global %s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    vstore3(smoothstep(e0[tid], e1[tid], vload3(tid,x)), tid, dst);\n"
    "}\n";

#define MAX_ERR (1e-5f)

namespace {

template <typename T>
int verify_smoothstep(const T *const edge0, const T *const edge1,
                      const T *const x, const T *const outptr, const int n,
                      const int veclen, const bool vecParam)
{
    double r, t;
    float delta = 0, max_delta = 0;

    if (vecParam)
    {
        for (int i = 0; i < n * veclen; i++)
        {
            t = (conv_to_dbl(x[i]) - conv_to_dbl(edge0[i]))
                / (conv_to_dbl(edge1[i]) - conv_to_dbl(edge0[i]));
            if (t < 0.0)
                t = 0.0;
            else if (t > 1.0)
                t = 1.0;
            r = t * t * (3.0 - 2.0 * t);
            delta = (float)fabs(r - conv_to_dbl(outptr[i]));
            if (!std::is_same<T, half>::value)
            {
                if (delta > MAX_ERR)
                {
                    log_error(
                        "%d) verification error: smoothstep(%a, %a, %a) = "
                        "*%a vs. %a\n",
                        i, x[i], edge0[i], edge1[i], r, outptr[i]);
                    return -1;
                }
            }
            else
                max_delta = std::max(max_delta, delta);
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            int ii = i / veclen;
            int vi = i * veclen;
            for (int j = 0; j < veclen; ++j, ++vi)
            {
                t = (conv_to_dbl(x[vi]) - conv_to_dbl(edge0[i]))
                    / (conv_to_dbl(edge1[i]) - conv_to_dbl(edge0[i]));
                if (t < 0.0)
                    t = 0.0;
                else if (t > 1.0)
                    t = 1.0;
                r = t * t * (3.0 - 2.0 * t);
                delta = (float)fabs(r - conv_to_dbl(outptr[vi]));

                if (!std::is_same<T, half>::value)
                {
                    if (delta > MAX_ERR)
                    {
                        log_error("{%d, element %d}) verification error: "
                                  "smoothstep(%a, %a, %a) = *%a vs. %a\n",
                                  ii, j, x[vi], edge0[i], edge1[i], r,
                                  outptr[vi]);
                        return -1;
                    }
                }
                else
                    max_delta = std::max(max_delta, delta);
            }
        }
    }

    // due to the fact that accuracy of smoothstep for cl_khr_fp16 is
    // implementation defined this test only reports maximum error without
    // testing maximum error threshold
    if (std::is_same<T, half>::value)
        log_error("smoothstep half verification result, max delta: %a\n",
                  max_delta);

    return 0;
}

}

template <typename T>
int test_smoothstep_fn(cl_device_id device, cl_context context,
                       cl_command_queue queue, const int n_elems,
                       const bool vecParam)
{
    clMemWrapper streams[4];
    std::vector<T> input_ptr[3], output_ptr;

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

    for (i = 0; i < 3; i++) input_ptr[i].resize(num_elements);
    output_ptr.resize(num_elements);

    for (i = 0; i < 4; i++)
    {
        streams[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(T) * num_elements, NULL, &err);
        test_error(err, "clCreateBuffer failed");
    }

    std::string pragma_str;
    if (std::is_same<T, float>::value)
    {
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = get_random_float(-0x00200000, 0x00010000, d);
            input_ptr[1][i] = get_random_float(input_ptr[0][i], 0x00200000, d);
            input_ptr[2][i] = get_random_float(-0x20000000, 0x20000000, d);
        }
    }
    else if (std::is_same<T, double>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = get_random_double(-0x00200000, 0x00010000, d);
            input_ptr[1][i] = get_random_double(input_ptr[0][i], 0x00200000, d);
            input_ptr[2][i] = get_random_double(-0x20000000, 0x20000000, d);
        }
    }
    else if (std::is_same<T, half>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = conv_to_half(get_random_float(-65503, 65503, d));
            input_ptr[1][i] = conv_to_half(
                get_random_float(conv_to_flt(input_ptr[0][i]), 65503, d));
            input_ptr[2][i] = conv_to_half(get_random_float(-65503, 65503, d));
        }
    }

    for (i = 0; i < 3; i++)
    {
        err = clEnqueueWriteBuffer(queue, streams[i], CL_TRUE, 0,
                                   sizeof(T) * num_elements,
                                   &input_ptr[i].front(), 0, NULL, NULL);
        test_error(err, "Unable to write input buffer");
    }

    const char vecSizeNames[][3] = { "", "2", "4", "8", "16", "3" };

    for (i = 0; i < kTotalVecCount; i++)
    {
        std::string kernelSource;
        if (i >= kVectorSizeCount)
        {
            if (vecParam)
            {
                std::string str = smoothstep_fn_code_pattern_v3;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), tname.c_str());
            }
            else
            {
                std::string str = smoothstep_fn_code_pattern_v3_scalar;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), tname.c_str());
            }
        }
        else
        {
            // regular path
            std::string str = smoothstep_fn_code_pattern;
            kernelSource =
                str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                            vecParam ? vecSizeNames[i] : "", tname.c_str(),
                            vecParam ? vecSizeNames[i] : "", tname.c_str(),
                            vecSizeNames[i], tname.c_str(), vecSizeNames[i]);
        }

        const char *programPtr = kernelSource.c_str();
        err =
            create_single_kernel_helper(context, &programs[i], &kernels[i], 1,
                                        (const char **)&programPtr, "test_fn");
        test_error(err, "Unable to create kernel");

        for (int j = 0; j < 4; j++)
        {
            err =
                clSetKernelArg(kernels[i], j, sizeof(streams[j]), &streams[j]);
            test_error(err, "Unable to set kernel argument");
        }

        size_t threads = (size_t)n_elems;

        err = clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &threads, NULL,
                                     0, NULL, NULL);
        test_error(err, "Unable to execute kernel");

        err = clEnqueueReadBuffer(queue, streams[3], true, 0,
                                  sizeof(T) * num_elements, &output_ptr[0], 0,
                                  NULL, NULL);
        test_error(err, "Unable to read results");

        if (verify_smoothstep((T *)&input_ptr[0].front(),
                              (T *)&input_ptr[1].front(),
                              (T *)&input_ptr[2].front(), &output_ptr[0],
                              n_elems, g_arrVecSizes[i], vecParam))
        {
            log_error("smoothstep %s%d%s test failed\n", tname.c_str(),
                      ((g_arrVecSizes[i])),
                      vecParam ? "" : std::string(", " + tname).c_str());
            err = -1;
        }
        else
        {
            log_info("smoothstep %s%d%s test passed\n", tname.c_str(),
                     ((g_arrVecSizes[i])),
                     vecParam ? "" : std::string(", " + tname).c_str());
            err = 0;
        }

        if (err) break;
    }

    return err;
}

cl_int SmoothstepTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_smoothstep_fn<half>(device, context, queue, num_elems,
                                         vecParam);
        test_error(error, "SmoothstepTest::Run<cl_half> failed");
    }

    error =
        test_smoothstep_fn<float>(device, context, queue, num_elems, vecParam);
    test_error(error, "SmoothstepTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_smoothstep_fn<double>(device, context, queue, num_elems,
                                           vecParam);
        test_error(error, "SmoothstepTest::Run<double> failed");
    }

    return error;
}

int test_smoothstep(cl_device_id device, cl_context context,
                    cl_command_queue queue, int n_elems)
{
    return MakeAndRunTest<SmoothstepTest>(device, context, queue, n_elems,
                                          "smoothstep", true);
}

int test_smoothstepf(cl_device_id device, cl_context context,
                     cl_command_queue queue, int n_elems)
{
    return MakeAndRunTest<SmoothstepTest>(device, context, queue, n_elems,
                                          "smoothstep", false);
}
