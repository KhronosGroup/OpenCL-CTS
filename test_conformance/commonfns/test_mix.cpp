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

#include <algorithm>
#include <cmath>

#include "harness/stringHelpers.h"

#include "test_base.h"


const char *mix_fn_code_pattern =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s%s *x, __global %s%s *y, __global %s%s "
    "*a, __global %s%s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    dst[tid] = mix(x[tid], y[tid], a[tid]);\n"
    "}\n";

const char *mix_fn_code_pattern_v3 =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *x, __global %s *y, __global %s *a, "
    "__global %s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    vstore3(mix(vload3(tid, x), vload3(tid, y), vload3(tid, a)), tid, "
    "dst);\n"
    "}\n";

const char *mix_fn_code_pattern_v3_scalar =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *x, __global %s *y, __global %s *a, "
    "__global %s *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    vstore3(mix(vload3(tid, x), vload3(tid, y), a[tid]), tid, dst);\n"
    "}\n";

#define MAX_ERR 1e-3

namespace {

double mix_half_error(const double x, const double y, const double r,
                      const double o)
{
    const double absolute_error = std::fabs(r - o);
    // The accuracy of half-precision mix is implementation-defined, so this
    // value is only reported. Normalize it to the scale of the operands and
    // result to make the report meaningful near zero and across the
    // half-precision range.
    const double max_scale =
        std::max(std::fabs(r), std::max(std::fabs(x), std::fabs(y)));
    return max_scale > 0.0 ? absolute_error / max_scale : absolute_error;
}

template <typename T>
bool mix_result_is_valid(const T x, const T y, const T a, const T out)
{
    // mix is specified as x + (y - x) * a. Evaluate that expression in the
    // destination type so the reference includes the rounding that is
    // unavoidable at each operation. Use fma to explicitly round each
    // operation and prevent the host compiler from contracting the
    // uncontracted reference calculation.
    const T difference = std::fma(T(-1), x, y);
    const T product = std::fma(difference, a, T(0));
    const T uncontracted = std::fma(T(1), product, x);

    // OpenCL implementations may contract the final multiply-add, so accept
    // that evaluation as well. Apply the specification's absolute 1e-3
    // tolerance to both permitted evaluations.
    const T contracted = std::fma(difference, a, x);
    const double uncontracted_error =
        std::fabs(static_cast<double>(out) - static_cast<double>(uncontracted));
    const double contracted_error =
        std::fabs(static_cast<double>(out) - static_cast<double>(contracted));
    return uncontracted_error <= MAX_ERR || contracted_error <= MAX_ERR;
}

template <typename T>
int verify_mix(const T *const inptrX, const T *const inptrY,
               const T *const inptrA, const T *const outptr, const int n,
               const int veclen, const bool vecParam)
{
    double r;
    [[maybe_unused]] double max_delta = 0.0;
    int i;

    if (vecParam)
    {
        for (i = 0; i < n * veclen; i++)
        {
            const double x = conv_to_dbl(inptrX[i]);
            const double y = conv_to_dbl(inptrY[i]);
            const double a = conv_to_dbl(inptrA[i]);
            r = x + ((y - x) * a);

            if constexpr (!std::is_same<T, half>::value)
            {
                if (!mix_result_is_valid(inptrX[i], inptrY[i], inptrA[i],
                                         outptr[i]))
                {
                    log_error("%d) verification error: mix(%a, %a, %a) = *%a "
                              "vs. %a\n",
                              i, conv_to_flt(inptrX[i]), conv_to_flt(inptrY[i]),
                              conv_to_flt(inptrA[i]), r,
                              conv_to_flt(outptr[i]));
                    return -1;
                }
            }
            else
            {
                // Half-precision mix accuracy is implementation-defined so
                // is only reported.
                max_delta = std::max(
                    max_delta, mix_half_error(x, y, r, conv_to_dbl(outptr[i])));
            }
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
                const double x = conv_to_dbl(inptrX[vi]);
                const double y = conv_to_dbl(inptrY[vi]);
                const double a = conv_to_dbl(inptrA[i]);
                r = x + ((y - x) * a);
                if constexpr (!std::is_same<T, half>::value)
                {
                    if (!mix_result_is_valid(inptrX[vi], inptrY[vi], inptrA[i],
                                             outptr[vi]))
                    {
                        log_error(
                            "{%d, element %d}) verification error: mix(%a, "
                            "%a, %a) = *%a vs. %a\n",
                            ii, j, conv_to_flt(inptrX[vi]),
                            conv_to_flt(inptrY[vi]), conv_to_flt(inptrA[i]), r,
                            conv_to_flt(outptr[vi]));
                        return -1;
                    }
                }
                else
                {
                    // Half-precision mix accuracy is implementation-defined so
                    // is only reported.
                    max_delta = std::max(
                        max_delta,
                        mix_half_error(x, y, r, conv_to_dbl(outptr[vi])));
                }
            }
        }
    }

    // due to the fact that accuracy of mix for cl_khr_fp16 is implementation
    // defined this test only reports maximum error without testing maximum
    // error threshold
    if constexpr (std::is_same<T, half>::value)
        log_error("mix half verification result, max delta: %a\n", max_delta);

    return 0;
}
} // namespace

template <typename T>
int test_mix_fn(cl_device_id device, cl_context context, cl_command_queue queue,
                int n_elems, bool vecParam)
{
    clMemWrapper streams[4];
    std::vector<T> input_ptr[3], output_ptr;

    std::vector<clProgramWrapper> programs;
    std::vector<clKernelWrapper> kernels;

    int err, i;
    MTdataHolder d(gRandomSeed);

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
    if (std::is_same<T, double>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }

    if (std::is_same<T, half>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        const float half_bound = CL_HALF_MAX / 2.0f;
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] =
                conv_to_half(get_random_float(-half_bound, half_bound, d));
            input_ptr[1][i] =
                conv_to_half(get_random_float(-half_bound, half_bound, d));
            input_ptr[2][i] = conv_to_half((float)genrand_real1(d));
        }
    }
    else if (std::is_same<T, float>::value)
    {
        // Sample a broad signed range while leaving ample headroom for the
        // intermediate results in x + (y - x) * a.
        const float float_bound = 0x1.0p+29f;
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = get_random_float(-float_bound, float_bound, d);
            input_ptr[1][i] = get_random_float(-float_bound, float_bound, d);
            input_ptr[2][i] = (T)genrand_real1(d);
        }
    }
    else
    {
        // Use the same range as float so both paths exercise comparable input
        // values without approaching double-precision overflow.
        const double double_bound = 0x1.0p+29;
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = get_random_double(-double_bound, double_bound, d);
            input_ptr[1][i] = get_random_double(-double_bound, double_bound, d);
            input_ptr[2][i] = (T)genrand_real1(d);
        }
    }

    for (i = 0; i < 3; i++)
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
            if (vecParam)
            {
                std::string str = mix_fn_code_pattern_v3;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), tname.c_str());
            }
            else
            {
                std::string str = mix_fn_code_pattern_v3_scalar;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str(), tname.c_str());
            }
        }
        else
        {
            // regular path
            std::string str = mix_fn_code_pattern;
            kernelSource =
                str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                            vecSizeNames[i], tname.c_str(), vecSizeNames[i],
                            tname.c_str(), vecParam ? vecSizeNames[i] : "",
                            tname.c_str(), vecSizeNames[i]);
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

        if (verify_mix(&input_ptr[0].front(), &input_ptr[1].front(),
                       &input_ptr[2].front(), &output_ptr.front(), n_elems,
                       g_arrVecSizes[i], vecParam))
        {
            log_error("mix %s%d%s test failed\n", tname.c_str(),
                      ((g_arrVecSizes[i])),
                      vecParam ? "" : std::string(", " + tname).c_str());
            err = -1;
        }
        else
        {
            log_info("mix %s%d%s test passed\n", tname.c_str(),
                     ((g_arrVecSizes[i])),
                     vecParam ? "" : std::string(", " + tname).c_str());
            err = 0;
        }

        if (err) break;
    }

    return err;
}

cl_int MixTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_mix_fn<half>(device, context, queue, num_elems, vecParam);
        test_error(error, "MixTest::Run<cl_half> failed");
    }

    error = test_mix_fn<float>(device, context, queue, num_elems, vecParam);
    test_error(error, "MixTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error =
            test_mix_fn<double>(device, context, queue, num_elems, vecParam);
        test_error(error, "MixTest::Run<double> failed");
    }

    return error;
}

REGISTER_TEST(mix)
{
    return MakeAndRunTest<MixTest>(device, context, queue, num_elements, "mix",
                                   true);
}

REGISTER_TEST(mixf)
{
    return MakeAndRunTest<MixTest>(device, context, queue, num_elements, "mix",
                                   false);
}
