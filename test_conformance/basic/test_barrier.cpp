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
#include <numeric>
#include <vector>

#include "procs.h"

namespace {
const char *barrier_kernel_code = R"(
__kernel void compute_sum(__global int *a, int n, __global int *tmp_sum,
                          __global int *sum)
{
    int tid = get_local_id(0);
    int lsize = get_local_size(0);
    int i;

    tmp_sum[tid] = 0;
    for (i = tid; i < n; i += lsize) tmp_sum[tid] += a[i];

    // updated to work for any workgroup size
    for (i = hadd(lsize, 1); lsize > 1; i = hadd(i, 1))
    {
        BARRIER(CLK_GLOBAL_MEM_FENCE);
        if (tid + i < lsize) tmp_sum[tid] += tmp_sum[tid + i];
        lsize = i;
    }

    // no barrier is required here because last person to write to tmp_sum[0]
    // was tid 0
    if (tid == 0) *sum = tmp_sum[0];
}
)";


void generate_random_inputs(std::vector<cl_int> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return static_cast<cl_int>(
            get_random_float(-0x01000000, 0x01000000, seed));
    };

    std::generate(v.begin(), v.end(), random_generator);
}

int test_barrier_common(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements,
                        std::string barrier_str)
{
    clMemWrapper streams[3];
    clProgramWrapper program;
    clKernelWrapper kernel;

    cl_int output;
    int err;

    size_t max_threadgroup_size = 0;
    std::string build_options = std::string("-DBARRIER=") + barrier_str;
    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &barrier_kernel_code, "compute_sum",
                                      build_options.c_str());
    test_error(err, "Failed to build kernel/program.");

    err = get_max_allowed_1d_work_group_size_on_device(device, kernel,
                                                       &max_threadgroup_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed.");

    // work group size must divide evenly into the global size
    while (num_elements % max_threadgroup_size) max_threadgroup_size--;

    std::vector<cl_int> input(num_elements);

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int),
                                nullptr, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[2] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(cl_int) * max_threadgroup_size, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");

    generate_random_inputs(input);

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0,
                               sizeof(cl_int) * num_elements, input.data(), 0,
                               nullptr, nullptr);
    test_error(err, "clEnqueueWriteBuffer failed.");

    err = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(num_elements), &num_elements);
    err |= clSetKernelArg(kernel, 2, sizeof(streams[2]), &streams[2]);
    err |= clSetKernelArg(kernel, 3, sizeof(streams[1]), &streams[1]);
    test_error(err, "clSetKernelArg failed.");

    size_t global_threads[] = { max_threadgroup_size };
    size_t local_threads[] = { max_threadgroup_size };

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_threads,
                                 local_threads, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams[1], true, 0, sizeof(cl_int),
                              &output, 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed.");

    if (std::accumulate(input.begin(), input.end(), 0) != output)
    {
        log_error("%s test failed\n", barrier_str.c_str());
        err = -1;
    }
    else
    {
        log_info("%s test passed\n", barrier_str.c_str());
    }

    return err;
}
}

int test_barrier(cl_device_id device, cl_context context,
                 cl_command_queue queue, int num_elements)
{
    return test_barrier_common(device, context, queue, num_elements, "barrier");
}

int test_wg_barrier(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    return test_barrier_common(device, context, queue, num_elements,
                               "work_group_barrier");
}
