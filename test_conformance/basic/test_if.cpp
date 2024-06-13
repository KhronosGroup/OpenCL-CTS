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
const char *conditional_kernel_code = R"(
__kernel void test_if(__global int *src, __global int *dst)
{
    int  tid = get_global_id(0);

    if (src[tid] == 0)
        dst[tid] = 0x12345678;
    else if (src[tid] == 1)
        dst[tid] = 0x23456781;
    else if (src[tid] == 2)
        dst[tid] = 0x34567812;
    else if (src[tid] == 3)
        dst[tid] = 0x45678123;
    else if (src[tid] == 4)
        dst[tid] = 0x56781234;
    else if (src[tid] == 5)
        dst[tid] = 0x67812345;
    else if (src[tid] == 6)
        dst[tid] = 0x78123456;
    else if (src[tid] == 7)
        dst[tid] = 0x81234567;
    else
        dst[tid] = 0x7FFFFFFF;
}
)";

int verify_if(std::vector<cl_int> input, std::vector<cl_int> output)
{
    const cl_int results[] = {
        0x12345678, 0x23456781, 0x34567812, 0x45678123,
        0x56781234, 0x67812345, 0x78123456, 0x81234567,
    };

    auto predicate = [&results](cl_int a, cl_int b) {
        if (a <= 7)
            return b == results[a];
        else
            return b == 0x7FFFFFFF;
    };

    if (!std::equal(input.begin(), input.end(), output.begin(), predicate))
    {
        log_error("IF test failed\n");
        return -1;
    }

    log_info("IF test passed\n");
    return 0;
}

void generate_random_inputs(std::vector<cl_int> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return static_cast<cl_int>(get_random_float(0, 32, seed));
    };

    std::generate(v.begin(), v.end(), random_generator);
}
}
int test_if(cl_device_id device, cl_context context, cl_command_queue queue,
            int num_elements)
{
    clMemWrapper streams[2];
    clProgramWrapper program;
    clKernelWrapper kernel;

    int err;

    size_t length = sizeof(cl_int) * num_elements;

    std::vector<cl_int> input(num_elements);
    std::vector<cl_int> output(num_elements);


    streams[0] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, length, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, length, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");

    generate_random_inputs(input);

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length,
                               input.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &conditional_kernel_code, "test_if");
    test_error(err, "create_single_kernel_helper failed.");

    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    test_error(err, "clSetKernelArg failed.");

    size_t threads[] = { (size_t)num_elements };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, threads, nullptr, 0,
                                 nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length,
                              output.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed.");

    err = verify_if(input, output);

    return err;
}
