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
#include <vector>

#include "procs.h"

namespace {
const char *global_linear_id_2d_code = R"(
__kernel void test_global_linear_id_2d(global int *dst)
{
    int  tid_x = get_global_id(0);
    int  tid_y = get_global_id(1);

    int linear_id = tid_y * get_global_size(0) + tid_x;
    int result = (linear_id == (int)get_global_linear_id()) ? 0x1 : 0x0;
    dst[linear_id] = result;
}
)";

const char *global_linear_id_1d_code = R"(
__kernel void test_global_linear_id_1d(global int *dst)
{
    int  tid_x = get_global_id(0);

    int result = (tid_x == (int)get_global_linear_id()) ? 0x1 : 0x0;
    dst[tid_x] = result;
}
)";


int verify_global_linear_id(std::vector<cl_int> &result, int n)
{
    if (std::any_of(result.begin(), result.begin() + n,
                    [](cl_int value) { return 0 == value; }))
    {
        log_error("get_global_linear_id failed\n");
        return TEST_FAIL;
    }
    log_info("get_global_linear_id passed\n");
    return TEST_PASS;
}
}

int test_global_linear_id(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    clProgramWrapper program[2];
    clKernelWrapper kernel[2];

    const char *kernel_names[] = { "test_global_linear_id_1d",
                                   "test_global_linear_id_2d" };
    const char *kernel_code[] = { global_linear_id_1d_code,
                                  global_linear_id_2d_code };
    int err = CL_SUCCESS;

    num_elements = static_cast<int>(sqrt(static_cast<float>(num_elements)));
    int length = 1;
    size_t threads[] = { static_cast<size_t>(num_elements),
                         static_cast<size_t>(num_elements) };

    for (int i = 0; i < ARRAY_SIZE(program) && !err; i++)
    {
        length *= num_elements;

        std::vector<cl_int> output(length);

        clMemWrapper streams = clCreateBuffer(
            context, CL_MEM_READ_WRITE, length * sizeof(cl_int), nullptr, &err);
        test_error(err, "clCreateBuffer failed.");

        err = create_single_kernel_helper(context, &program[i], &kernel[i], 1,
                                          &kernel_code[i], kernel_names[i]);
        test_error(err, "create_single_kernel_helper failed");

        err = clSetKernelArg(kernel[i], 0, sizeof streams, &streams);
        test_error(err, "clSetKernelArgs failed.");

        err = clEnqueueNDRangeKernel(queue, kernel[i], i + 1, nullptr, threads,
                                     nullptr, 0, nullptr, nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed.");

        err = clEnqueueReadBuffer(queue, streams, CL_TRUE, 0,
                                  length * sizeof(cl_int), output.data(), 0,
                                  nullptr, nullptr);
        test_error(err, "clEnqueueReadBuffer failed.");

        err = verify_global_linear_id(output, length);
    }

    return err;
}
