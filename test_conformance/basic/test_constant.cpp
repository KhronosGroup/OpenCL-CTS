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

#include <algorithm>
#include <vector>

#include "procs.h"

namespace {
const char* constant_kernel_code = R"(
__kernel void constant_kernel(__global float *out, __constant float *tmpF, __constant int *tmpI)
{
    int  tid = get_global_id(0);

    float ftmp = tmpF[tid];
    float Itmp = tmpI[tid];
    out[tid] = ftmp * Itmp;
}
)";

const char* loop_constant_kernel_code = R"(
kernel void loop_constant_kernel(global float *out, constant float *i_pos, int num)
{
    int tid = get_global_id(0);
    float sum = 0;
    for (int i = 0; i < num; i++) {
        float  pos  = i_pos[i*3];
        sum += pos;
    }
    out[tid] = sum;
}
)";


int verify(std::vector<cl_float>& tmpF, std::vector<cl_int>& tmpI,
           std::vector<cl_float>& out)
{
    for (int i = 0; i < out.size(); i++)
    {
        float f = tmpF[i] * tmpI[i];
        if (out[i] != f)
        {
            log_error("CONSTANT test failed\n");
            return -1;
        }
    }

    log_info("CONSTANT test passed\n");
    return 0;
}

int verify_loop_constant(const std::vector<cl_float>& tmp,
                         std::vector<cl_float>& out, cl_int l)
{
    float sum = 0;
    for (int j = 0; j < l; ++j) sum += tmp[j * 3];

    auto predicate = [&sum](cl_float elem) { return sum != elem; };

    if (std::any_of(out.cbegin(), out.cend(), predicate))
    {
        log_error("loop CONSTANT test failed\n");
        return -1;
    }

    log_info("loop CONSTANT test passed\n");
    return 0;
}

template <typename T> void generate_random_inputs(std::vector<T>& v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return static_cast<T>(get_random_float(-0x02000000, 0x02000000, seed));
    };

    std::generate(v.begin(), v.end(), random_generator);
}
}

int test_constant(cl_device_id device, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    clMemWrapper streams[3];
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t global_threads[3];
    int err;
    cl_ulong maxSize, maxGlobalSize, maxAllocSize;
    size_t num_floats, num_ints, constant_values;
    RoundingMode oldRoundMode;
    int isRTZ = 0;

    /* Verify our test buffer won't be bigger than allowed */
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                          sizeof(maxSize), &maxSize, 0);
    test_error(err, "Unable to get max constant buffer size");
    log_info("Device reports CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE %llu bytes.\n",
             maxSize);

    // Limit test buffer size to 1/4 of CL_DEVICE_GLOBAL_MEM_SIZE
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(maxGlobalSize), &maxGlobalSize, 0);
    test_error(err, "Unable to get CL_DEVICE_GLOBAL_MEM_SIZE");

    maxSize = std::min(maxSize, maxGlobalSize / 4);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                          sizeof(maxAllocSize), &maxAllocSize, 0);
    test_error(err, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE");

    maxSize = std::min(maxSize, maxAllocSize);

    maxSize /= 4;
    num_ints = static_cast<size_t>(maxSize / sizeof(cl_int));
    num_floats = static_cast<size_t>(maxSize / sizeof(cl_float));
    constant_values = std::min(num_floats, num_ints);


    log_info(
        "Test will attempt to use %lu bytes with one %lu byte constant int "
        "buffer and one %lu byte constant float buffer.\n",
        constant_values * sizeof(cl_int) + constant_values * sizeof(cl_float),
        constant_values * sizeof(cl_int), constant_values * sizeof(cl_float));

    std::vector<cl_int> tmpI(constant_values);
    std::vector<cl_float> tmpF(constant_values);
    std::vector<cl_float> out(constant_values);


    streams[0] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_float) * constant_values, nullptr, &err);
    test_error(err, "clCreateBuffer failed");

    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_float) * constant_values, nullptr, &err);
    test_error(err, "clCreateBuffer failed");

    streams[2] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_int) * constant_values, nullptr, &err);
    test_error(err, "clCreateBuffer failed");

    generate_random_inputs(tmpI);
    generate_random_inputs(tmpF);

    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0,
                               sizeof(cl_float) * constant_values, tmpF.data(),
                               0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed");
    err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0,
                               sizeof(cl_int) * constant_values, tmpI.data(), 0,
                               nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer faile.");

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &constant_kernel_code, "constant_kernel");
    test_error(err, "Failed to create kernel and program");


    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
    test_error(err, "clSetKernelArgs failed");

    global_threads[0] = constant_values;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_threads,
                                 nullptr, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadBuffer(queue, streams[0], CL_TRUE, 0,
                              sizeof(cl_float) * constant_values, out.data(), 0,
                              nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed");

    // If we only support rtz mode
    if (CL_FP_ROUND_TO_ZERO == get_default_rounding_mode(device) && gIsEmbedded)
    {
        oldRoundMode = set_round(kRoundTowardZero, kfloat);
        isRTZ = 1;
    }

    err = verify(tmpF, tmpI, out);

    if (isRTZ) (void)set_round(oldRoundMode, kfloat);

    // Loop constant buffer test
    clProgramWrapper loop_program;
    clKernelWrapper loop_kernel;
    cl_int limit = 2;

    memset(out.data(), 0, sizeof(cl_float) * constant_values);
    err = create_single_kernel_helper(context, &loop_program, &loop_kernel, 1,
                                      &loop_constant_kernel_code,
                                      "loop_constant_kernel");
    test_error(err, "Failed to create kernel and program");

    err = clSetKernelArg(loop_kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(loop_kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(loop_kernel, 2, sizeof(limit), &limit);
    test_error(err, "clSetKernelArgs failed");

    err = clEnqueueNDRangeKernel(queue, loop_kernel, 1, nullptr, global_threads,
                                 nullptr, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadBuffer(queue, streams[0], CL_TRUE, 0,
                              sizeof(cl_float) * constant_values, out.data(), 0,
                              nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed");

    err = verify_loop_constant(tmpF, out, limit);


    return err;
}
