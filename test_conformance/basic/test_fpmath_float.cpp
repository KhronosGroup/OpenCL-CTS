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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "harness/rounding_mode.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "procs.h"

struct TestDef
{
    const char op;
    std::function<float(float, float)> ref;
};

static const char *fp_kernel_code = R"(
__kernel void test_fp(__global TYPE *srcA, __global TYPE *srcB, __global TYPE *dst)
{
    int  tid = get_global_id(0);

    dst[tid] = srcA[tid] OP srcB[tid];
})";

static int verify_fp(std::vector<float> (&input)[2], std::vector<float> &output,
                     const TestDef &test)
{

    auto &inA = input[0];
    auto &inB = input[1];
    for (int i = 0; i < output.size(); i++)
    {
        float r = test.ref(inA[i], inB[i]);
        if (r != output[i])
        {
            log_error("FP '%c' float test failed\n", test.op);
            return -1;
        }
    }

    log_info("FP '%c' float test passed\n", test.op);
    return 0;
}


void generate_random_inputs(std::vector<cl_float> (&input)[2])
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31),
                                MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), seed);
    };

    for (auto &v : input)
    {
        std::generate(v.begin(), v.end(), random_generator);
    }
}

template <size_t N>
int test_fpmath(cl_device_id device, cl_context context, cl_command_queue queue,
                int num_elements, const std::string type_str,
                const TestDef &test)
{
    clMemWrapper streams[3];
    clProgramWrapper program;
    clKernelWrapper kernel;

    int err;

    size_t length = sizeof(cl_float) * num_elements * N;

    int isRTZ = 0;
    RoundingMode oldMode = kDefaultRoundingMode;

    // If we only support rtz mode
    if (CL_FP_ROUND_TO_ZERO == get_default_rounding_mode(device))
    {
        isRTZ = 1;
        oldMode = get_round();
    }


    std::vector<cl_float> inputs[]{ std::vector<cl_float>(N * num_elements),
                                    std::vector<cl_float>(N * num_elements) };
    std::vector<cl_float> output = std::vector<cl_float>(N * num_elements);

    generate_random_inputs(inputs);

    for (int i = 0; i < ARRAY_SIZE(streams); i++)
    {
        streams[i] =
            clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
        test_error(err, "clCreateBuffer failed.");
    }
    for (int i = 0; i < ARRAY_SIZE(inputs); i++)
    {
        err = clEnqueueWriteBuffer(queue, streams[i], CL_TRUE, 0, length,
                                   inputs[i].data(), 0, NULL, NULL);
        test_error(err, "clEnqueueWriteBuffer failed.");
    }

    std::string build_options = "-DTYPE=";
    build_options.append(type_str).append(" -DOP=").append(1, test.op);

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &fp_kernel_code, "test_fp",
                                      build_options.c_str());

    test_error(err, "create_single_kernel_helper failed");

    for (int i = 0; i < ARRAY_SIZE(streams); i++)
    {
        err = clSetKernelArg(kernel, i, sizeof(streams[i]), &streams[i]);
        test_error(err, "clSetKernelArgs failed.");
    }

    size_t threads[] = { static_cast<size_t>(num_elements) };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL,
                                 NULL);
    test_error(err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, length,
                              output.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueReadBuffer failed.");

    if (isRTZ) set_round(kRoundTowardZero, kfloat);

    err = verify_fp(inputs, output, test);

    if (isRTZ) set_round(oldMode, kfloat);

    return err;
}


template <size_t N>
int test_fpmath_common(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements,
                       const std::string type_str)
{
    TestDef tests[] = { { '+', std::plus<float>() },
                        { '-', std::minus<float>() },
                        { '*', std::multiplies<float>() } };
    int err = TEST_PASS;

    for (const auto &test : tests)
    {
        err |= test_fpmath<N>(device, context, queue, num_elements, type_str,
                              test);
    }

    return err;
}

int test_fpmath_float(cl_device_id device, cl_context context,
                      cl_command_queue queue, int num_elements)
{
    return test_fpmath_common<1>(device, context, queue, num_elements, "float");
}

int test_fpmath_float2(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return test_fpmath_common<2>(device, context, queue, num_elements,
                                 "float2");
}

int test_fpmath_float4(cl_device_id device, cl_context context,
                       cl_command_queue queue, int num_elements)
{
    return test_fpmath_common<4>(device, context, queue, num_elements,
                                 "float4");
}
