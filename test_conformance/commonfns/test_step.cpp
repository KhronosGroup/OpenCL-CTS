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

const char *step_fn_code_pattern = "%s\n" /* optional pragma */
                                   "__kernel void test_fn(__global %s%s *edge, "
                                   "__global %s%s *x, __global %s%s *dst)\n"
                                   "{\n"
                                   "    int  tid = get_global_id(0);\n"
                                   "    dst[tid] = step(edge[tid], x[tid]);\n"
                                   "}\n";

const char *step_fn_code_pattern_v3 =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *edge, __global %s *x, __global %s "
    "*dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    vstore3(step(vload3(tid,edge), vload3(tid,x)), tid, dst);\n"
    "}\n";

const char *step_fn_code_pattern_v3_scalar =
    "%s\n" /* optional pragma */
    "__kernel void test_fn(__global %s *edge, __global %s *x, __global %s "
    "*dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    vstore3(step(edge[tid], vload3(tid,x)), tid, dst);\n"
    "}\n";

namespace {

template <typename T>
int verify_step(const T *const inptrA, const T *const inptrB,
                const T *const outptr, const int n, const int veclen,
                const bool vecParam)
{
    T r;

    if (vecParam)
    {
        for (int i = 0; i < n * veclen; i++)
        {
            r = (conv_to_dbl(inptrB[i]) < conv_to_dbl(inptrA[i])) ? 0.0 : 1.0;
            if (r != conv_to_dbl(outptr[i])) return -1;
        }
    }
    else
    {
        for (int i = 0; i < n;)
        {
            int ii = i / veclen;
            for (int j = 0; j < veclen && i < n; ++j, ++i)
            {
                r = (conv_to_dbl(inptrB[i]) < conv_to_dbl(inptrA[ii])) ? 0.0f
                                                                       : 1.0f;
                if (r != conv_to_dbl(outptr[i]))
                {
                    if (std::is_same<T, half>::value)
                        log_error(
                            "Failure @ {%d, element %d}: step(%a,%a) -> *%a "
                            "vs %a\n",
                            ii, j, conv_to_flt(inptrA[ii]),
                            conv_to_flt(inptrB[i]), r, conv_to_flt(outptr[i]));
                    else
                        log_error(
                            "Failure @ {%d, element %d}: step(%a,%a) -> *%a "
                            "vs %a\n",
                            ii, j, inptrA[ii], inptrB[i], r, outptr[i]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

}

template <typename T>
int test_step_fn(cl_device_id device, cl_context context,
                 cl_command_queue queue, int n_elems, bool vecParam)
{
    clMemWrapper streams[3];
    std::vector<T> input_ptr[2], output_ptr;

    std::vector<clProgramWrapper> programs;
    std::vector<clKernelWrapper> kernels;

    int err, i;
    MTdataHolder d = MTdataHolder(gRandomSeed);

    assert(BaseFunctionTest::type2name.find(sizeof(T))
           != BaseFunctionTest::type2name.end());
    auto tname = BaseFunctionTest::type2name[sizeof(T)];
    int num_elements = n_elems * (1 << (kTotalVecCount - 1));

    programs.resize(kTotalVecCount);
    kernels.resize(kTotalVecCount);

    for (i = 0; i < 2; i++) input_ptr[i].resize(num_elements);
    output_ptr.resize(num_elements);

    for (i = 0; i < 3; i++)
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
            input_ptr[0][i] = get_random_float(-0x40000000, 0x40000000, d);
            input_ptr[1][i] = get_random_float(-0x40000000, 0x40000000, d);
        }
    }
    else if (std::is_same<T, double>::value)
    {
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = get_random_double(-0x40000000, 0x40000000, d);
            input_ptr[1][i] = get_random_double(-0x40000000, 0x40000000, d);
        }
    }
    else if (std::is_same<T, half>::value)
    {
        const float fval = CL_HALF_MAX;
        pragma_str = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        for (i = 0; i < num_elements; i++)
        {
            input_ptr[0][i] = conv_to_half(get_random_float(-fval, fval, d));
            input_ptr[1][i] = conv_to_half(get_random_float(-fval, fval, d));
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
            if (vecParam)
            {
                std::string str = step_fn_code_pattern_v3;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str());
            }
            else
            {
                std::string str = step_fn_code_pattern_v3_scalar;
                kernelSource =
                    str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                                tname.c_str(), tname.c_str());
            }
        }
        else
        {
            // regular path
            std::string str = step_fn_code_pattern;
            kernelSource =
                str_sprintf(str, pragma_str.c_str(), tname.c_str(),
                            vecParam ? vecSizeNames[i] : "", tname.c_str(),
                            vecSizeNames[i], tname.c_str(), vecSizeNames[i]);
        }
        const char *programPtr = kernelSource.c_str();
        err =
            create_single_kernel_helper(context, &programs[i], &kernels[i], 1,
                                        (const char **)&programPtr, "test_fn");
        test_error(err, "Unable to create kernel");

        for (int j = 0; j < 3; j++)
        {
            err =
                clSetKernelArg(kernels[i], j, sizeof(streams[j]), &streams[j]);
            test_error(err, "Unable to set kernel argument");
        }

        size_t threads = (size_t)n_elems;

        err = clEnqueueNDRangeKernel(queue, kernels[i], 1, NULL, &threads, NULL,
                                     0, NULL, NULL);
        test_error(err, "Unable to execute kernel");

        err = clEnqueueReadBuffer(queue, streams[2], true, 0,
                                  sizeof(T) * num_elements, &output_ptr[0], 0,
                                  NULL, NULL);
        test_error(err, "Unable to read results");

        err = verify_step(&input_ptr[0].front(), &input_ptr[1].front(),
                          &output_ptr.front(), n_elems, g_arrVecSizes[i],
                          vecParam);
        if (err)
        {
            log_error("step %s%d%s test failed\n", tname.c_str(),
                      ((g_arrVecSizes[i])),
                      vecParam ? "" : std::string(", " + tname).c_str());
            err = -1;
        }
        else
        {
            log_info("step %s%d%s test passed\n", tname.c_str(),
                     ((g_arrVecSizes[i])),
                     vecParam ? "" : std::string(", " + tname).c_str());
            err = 0;
        }

        if (err)
            break;
    }

    return err;
}

cl_int StepTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_step_fn<half>(device, context, queue, num_elems, vecParam);
        test_error(error, "StepTest::Run<cl_half> failed");
    }

    error = test_step_fn<float>(device, context, queue, num_elems, vecParam);
    test_error(error, "StepTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error =
            test_step_fn<double>(device, context, queue, num_elems, vecParam);
        test_error(error, "StepTest::Run<double> failed");
    }

    return error;
}

int test_step(cl_device_id device, cl_context context, cl_command_queue queue,
              int n_elems)
{
    return MakeAndRunTest<StepTest>(device, context, queue, n_elems, "step",
                                    true);
}

int test_stepf(cl_device_id device, cl_context context, cl_command_queue queue,
               int n_elems)
{
    return MakeAndRunTest<StepTest>(device, context, queue, n_elems, "step",
                                    false);
}
