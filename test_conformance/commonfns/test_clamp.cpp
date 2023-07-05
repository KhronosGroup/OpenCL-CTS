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

#define CLAMP_KERNEL(type)                                                     \
    const char *clamp_##type##_kernel_code = EMIT_PRAGMA_DIRECTIVE             \
        "__kernel void test_clamp(__global " #type " *x, __global " #type      \
        " *minval, __global " #type " *maxval, __global " #type " *dst)\n"     \
        "{\n"                                                                  \
        "    int  tid = get_global_id(0);\n"                                   \
        "\n"                                                                   \
        "    dst[tid] = clamp(x[tid], minval[tid], maxval[tid]);\n"            \
        "}\n";

#define CLAMP_KERNEL_V(type, size)                                             \
    const char *clamp_##type##size##_kernel_code = EMIT_PRAGMA_DIRECTIVE       \
        "__kernel void test_clamp(__global " #type #size                       \
        " *x, __global " #type #size " *minval, __global " #type #size         \
        " *maxval, __global " #type #size " *dst)\n"                           \
        "{\n"                                                                  \
        "    int  tid = get_global_id(0);\n"                                   \
        "\n"                                                                   \
        "    dst[tid] = clamp(x[tid], minval[tid], maxval[tid]);\n"            \
        "}\n";

#define CLAMP_KERNEL_V3(type, size)                                            \
    const char *clamp_##type##size##_kernel_code = EMIT_PRAGMA_DIRECTIVE       \
        "__kernel void test_clamp(__global " #type " *x, __global " #type      \
        " *minval, __global " #type " *maxval, __global " #type " *dst)\n"     \
        "{\n"                                                                  \
        "    int  tid = get_global_id(0);\n"                                   \
        "\n"                                                                   \
        "    vstore3(clamp(vload3(tid, x), vload3(tid,minval), "               \
        "vload3(tid,maxval)), tid, dst);\n"                                    \
        "}\n";

#define EMIT_PRAGMA_DIRECTIVE "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
CLAMP_KERNEL(half)
CLAMP_KERNEL_V(half, 2)
CLAMP_KERNEL_V(half, 4)
CLAMP_KERNEL_V(half, 8)
CLAMP_KERNEL_V(half, 16)
CLAMP_KERNEL_V3(half, 3)
#undef EMIT_PRAGMA_DIRECTIVE

#define EMIT_PRAGMA_DIRECTIVE " "
CLAMP_KERNEL(float)
CLAMP_KERNEL_V(float, 2)
CLAMP_KERNEL_V(float, 4)
CLAMP_KERNEL_V(float, 8)
CLAMP_KERNEL_V(float, 16)
CLAMP_KERNEL_V3(float, 3)
#undef EMIT_PRAGMA_DIRECTIVE

#define EMIT_PRAGMA_DIRECTIVE "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
CLAMP_KERNEL(double)
CLAMP_KERNEL_V(double, 2)
CLAMP_KERNEL_V(double, 4)
CLAMP_KERNEL_V(double, 8)
CLAMP_KERNEL_V(double, 16)
CLAMP_KERNEL_V3(double, 3)
#undef EMIT_PRAGMA_DIRECTIVE

const char *clamp_half_codes[] = {
    clamp_half_kernel_code,  clamp_half2_kernel_code,  clamp_half4_kernel_code,
    clamp_half8_kernel_code, clamp_half16_kernel_code, clamp_half3_kernel_code
};
const char *clamp_float_codes[] = {
    clamp_float_kernel_code,   clamp_float2_kernel_code,
    clamp_float4_kernel_code,  clamp_float8_kernel_code,
    clamp_float16_kernel_code, clamp_float3_kernel_code
};
const char *clamp_double_codes[] = {
    clamp_double_kernel_code,   clamp_double2_kernel_code,
    clamp_double4_kernel_code,  clamp_double8_kernel_code,
    clamp_double16_kernel_code, clamp_double3_kernel_code
};

namespace {

template <typename T>
int verify_clamp(const T *const x, const T *const minval, const T *const maxval,
                 const T *const outptr, int n)
{
    if (std::is_same<T, half>::value)
    {
        float t;
        for (int i = 0; i < n; i++)
        {
            t = std::min(
                std::max(cl_half_to_float(x[i]), cl_half_to_float(minval[i])),
                cl_half_to_float(maxval[i]));
            if (t != cl_half_to_float(outptr[i]))
            {
                log_error(
                    "%d) verification error: clamp( %a, %a, %a) = *%a vs. %a\n",
                    i, cl_half_to_float(x[i]), cl_half_to_float(minval[i]),
                    cl_half_to_float(maxval[i]), t,
                    cl_half_to_float(outptr[i]));
                return -1;
            }
        }
    }
    else
    {
        T t;
        for (int i = 0; i < n; i++)
        {
            t = std::min(std::max(x[i], minval[i]), maxval[i]);
            if (t != outptr[i])
            {
                log_error(
                    "%d) verification error: clamp( %a, %a, %a) = *%a vs. %a\n",
                    i, x[i], minval[i], maxval[i], t, outptr[i]);
                return -1;
            }
        }
    }

    return 0;
}
}

template <typename T>
int test_clamp_fn(cl_device_id device, cl_context context,
                  cl_command_queue queue, int n_elems)
{
    clMemWrapper streams[4];
    std::vector<T> input_ptr[3], output_ptr;

    std::vector<clProgramWrapper> programs;
    std::vector<clKernelWrapper> kernels;

    int err, i, j;
    MTdataHolder d = MTdataHolder(gRandomSeed);

    assert(BaseFunctionTest::type2name.find(sizeof(T))
           != BaseFunctionTest::type2name.end());
    auto tname = BaseFunctionTest::type2name[sizeof(T)];

    programs.resize(kTotalVecCount);
    kernels.resize(kTotalVecCount);

    int num_elements = n_elems * (1 << (kVectorSizeCount - 1));

    for (i = 0; i < 3; i++) input_ptr[i].resize(num_elements);
    output_ptr.resize(num_elements);

    for (i = 0; i < 4; i++)
    {
        streams[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(T) * num_elements, NULL, &err);
        test_error(err, "clCreateBuffer failed");
    }

    if (std::is_same<T, float>::value)
    {
        for (j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = get_random_float(-0x200000, 0x200000, d);
            input_ptr[1][j] = get_random_float(-0x200000, 0x200000, d);
            input_ptr[2][j] = get_random_float(input_ptr[1][j], 0x200000, d);
        }
    }
    else if (std::is_same<T, double>::value)
    {
        for (j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = get_random_double(-0x20000000, 0x20000000, d);
            input_ptr[1][j] = get_random_double(-0x20000000, 0x20000000, d);
            input_ptr[2][j] = get_random_double(input_ptr[1][j], 0x20000000, d);
        }
    }
    else if (std::is_same<T, half>::value)
    {
        const float fval = CL_HALF_MAX;
        for (j = 0; j < num_elements; j++)
        {
            input_ptr[0][j] = conv_to_half(get_random_float(-fval, fval, d));
            input_ptr[1][j] = conv_to_half(get_random_float(-fval, fval, d));
            input_ptr[2][j] = conv_to_half(
                get_random_float(conv_to_flt(input_ptr[1][j]), fval, d));
        }
    }

    for (i = 0; i < 3; i++)
    {
        err = clEnqueueWriteBuffer(queue, streams[i], CL_TRUE, 0,
                                   sizeof(T) * num_elements,
                                   &input_ptr[i].front(), 0, NULL, NULL);
        test_error(err, "Unable to write input buffer");
    }

    for (i = 0; i < kTotalVecCount; i++)
    {
        if (std::is_same<T, float>::value)
        {
            err = create_single_kernel_helper(
                context, &programs[i], &kernels[i], 1, &clamp_float_codes[i],
                "test_clamp");
            test_error(err, "Unable to create kernel");
        }
        else if (std::is_same<T, double>::value)
        {
            err = create_single_kernel_helper(
                context, &programs[i], &kernels[i], 1, &clamp_double_codes[i],
                "test_clamp");
            test_error(err, "Unable to create kernel");
        }
        else if (std::is_same<T, half>::value)
        {
            err = create_single_kernel_helper(
                context, &programs[i], &kernels[i], 1, &clamp_half_codes[i],
                "test_clamp");
            test_error(err, "Unable to create kernel");
        }

        log_info("Just made a program for %s, i=%d, size=%d, in slot %d\n",
                 tname.c_str(), i, g_arrVecSizes[i], i);
        fflush(stdout);

        for (j = 0; j < 4; j++)
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

        if (verify_clamp<T>((T *)&input_ptr[0].front(),
                            (T *)&input_ptr[1].front(),
                            (T *)&input_ptr[2].front(), (T *)&output_ptr[0],
                            n_elems * ((g_arrVecSizes[i]))))
        {
            log_error("CLAMP %s%d test failed\n", tname.c_str(),
                      ((g_arrVecSizes[i])));
            err = -1;
        }
        else
        {
            log_info("CLAMP %s%d test passed\n", tname.c_str(),
                     ((g_arrVecSizes[i])));
            err = 0;
        }

        if (err) break;
    }

    return err;
}

cl_int ClampTest::Run()
{
    cl_int error = CL_SUCCESS;
    if (is_extension_available(device, "cl_khr_fp16"))
    {
        error = test_clamp_fn<cl_half>(device, context, queue, num_elems);
        test_error(error, "ClampTest::Run<cl_half> failed");
    }

    error = test_clamp_fn<float>(device, context, queue, num_elems);
    test_error(error, "ClampTest::Run<float> failed");

    if (is_extension_available(device, "cl_khr_fp64"))
    {
        error = test_clamp_fn<double>(device, context, queue, num_elems);
        test_error(error, "ClampTest::Run<double> failed");
    }

    return error;
}

int test_clamp(cl_device_id device, cl_context context, cl_command_queue queue,
               int n_elems)
{
    return MakeAndRunTest<ClampTest>(device, context, queue, n_elems);
}
