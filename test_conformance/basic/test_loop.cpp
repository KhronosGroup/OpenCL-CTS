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

#include <vector>

#include "procs.h"

namespace {
const char *loop_kernel_code = R"(
__kernel void test_loop(__global int *src, __global int *loopindx, __global int *loopcnt, __global int *dst)
{
    int  tid = get_global_id(0);
    int  n = get_global_size(0);
    int  i, j;

    dst[tid] = 0;
    for (i=0, j=loopindx[tid]; i<loopcnt[tid]; i++, j++)
    {
        if (j >= n)
            j = 0;
        dst[tid] += src[j];
    }
}
)";


int verify_loop(std::vector<cl_int> inptr, std::vector<cl_int> loopindx,
                std::vector<cl_int> loopcnt, std::vector<cl_int> outptr, int n)
{
    for (int i = 0; i < n; i++)
    {
        int r = 0;
        for (int j = 0, k = loopindx[i]; j < loopcnt[i]; j++, k++)
        {
            if (k >= n) k = 0;
            r += inptr[k];
        }

        if (r != outptr[i])
        {
            log_error("LOOP test failed: %d found, expected %d\n", outptr[i],
                      r);
            return -1;
        }
    }

    log_info("LOOP test passed\n");
    return 0;
}
}
int test_loop(cl_device_id device, cl_context context, cl_command_queue queue,
              int num_elements)
{
    clMemWrapper streams[4];
    clProgramWrapper program;
    clKernelWrapper kernel;
    int err;

    size_t length = sizeof(cl_int) * num_elements;
    std::vector<cl_int> input(length);
    std::vector<cl_int> loop_indx(length);
    std::vector<cl_int> loop_cnt(length);
    std::vector<cl_int> output(length);

    for (auto &stream : streams)
    {
        stream =
            clCreateBuffer(context, CL_MEM_READ_WRITE, length, nullptr, &err);
        test_error(err, "clCreateBuffer failed.");
    }

    RandomSeed seed(gRandomSeed);
    for (int i = 0; i < num_elements; i++)
    {
        input[i] = static_cast<int>(genrand_int32(seed));
        loop_indx[i] =
            static_cast<int>(get_random_float(0, num_elements - 1, seed));
        loop_cnt[i] =
            static_cast<int>(get_random_float(0, num_elements / 32, seed));
    };

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length,
                               input.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");
    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length,
                               loop_indx.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");
    err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length,
                               loop_cnt.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &loop_kernel_code, "test_loop");
    test_error(err, "create_single_kernel_helper failed.");

    for (int i = 0; i < ARRAY_SIZE(streams); i++)
    {
        err = clSetKernelArg(kernel, i, sizeof streams[i], &streams[i]);
        test_error(err, "clSetKernelArgs failed\n");
    }

    size_t threads[] = { (size_t)num_elements };
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, threads, nullptr, 0,
                                 nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed\n");

    err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length,
                              output.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed\n");

    err = verify_loop(input, loop_indx, loop_cnt, output, num_elements);


    return err;
}
