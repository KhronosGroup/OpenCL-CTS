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
#include "harness/rounding_mode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>

#include "procs.h"

static const char *enqueued_local_size_2d_code =
"__kernel void test_enqueued_local_size_2d(global int *dst)\n"
"{\n"
"    if ((get_global_id(0) == 0) && (get_global_id(1) == 0))\n"
"    {\n"
"        dst[0] = (int)get_enqueued_local_size(0)\n;"
"        dst[1] = (int)get_enqueued_local_size(1)\n;"
"    }\n"
"}\n";

static const char *enqueued_local_size_1d_code =
"__kernel void test_enqueued_local_size_1d(global int *dst)\n"
"{\n"
"    int  tid_x = get_global_id(0);\n"
"    if (get_global_id(0) == 0)\n"
"    {\n"
"        dst[tid_x] = (int)get_enqueued_local_size(0)\n;"
"    }\n"
"}\n";


static int
verify_enqueued_local_size(int *result, size_t *expected, int n)
{
    int i;
    for (i=0; i<n; i++)
    {
        if (result[i] != (int)expected[i])
        {
            log_error("get_enqueued_local_size failed\n");
            return -1;
        }
    }
    log_info("get_enqueued_local_size passed\n");
    return 0;
}


int
test_enqueued_local_size(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams;
    cl_program program[2];
    cl_kernel kernel[2];

    int *output_ptr;
    size_t globalsize[2];
    size_t localsize[2];
    int err;

    // For an OpenCL-3.0 device that does not support non-uniform work-groups
    // we cannot enqueue local sizes which do not divide the global dimensions
    // but we can still run the test checking that get_enqueued_local_size ==
    // get_local_size.
    bool use_uniform_work_groups{ false };
    if (get_device_cl_version(device) >= Version(3, 0))
    {
        cl_bool areNonUniformWorkGroupsSupported = false;
        err = clGetDeviceInfo(device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT,
                              sizeof(areNonUniformWorkGroupsSupported),
                              &areNonUniformWorkGroupsSupported, nullptr);
        test_error_ret(err, "clGetDeviceInfo failed.", TEST_FAIL);

        if (CL_FALSE == areNonUniformWorkGroupsSupported)
        {
            log_info("Non-uniform work group sizes are not supported, "
                     "enqueuing with uniform workgroups\n");
            use_uniform_work_groups = true;
        }
    }

    output_ptr   = (int*)malloc(2 * sizeof(int));

    streams =
        clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(int), NULL, &err);
    test_error( err, "clCreateBuffer failed.");

    std::string cl_std = "-cl-std=CL";
    cl_std += (get_device_cl_version(device) == Version(3, 0)) ? "3.0" : "2.0";
    err = create_single_kernel_helper_with_build_options(
        context, &program[0], &kernel[0], 1, &enqueued_local_size_1d_code,
        "test_enqueued_local_size_1d", cl_std.c_str());
    test_error( err, "create_single_kernel_helper failed");
    err = create_single_kernel_helper_with_build_options(
        context, &program[1], &kernel[1], 1, &enqueued_local_size_2d_code,
        "test_enqueued_local_size_2d", cl_std.c_str());
    test_error( err, "create_single_kernel_helper failed");

    err  = clSetKernelArg(kernel[0], 0, sizeof streams, &streams);
    test_error( err, "clSetKernelArgs failed.");
    err  = clSetKernelArg(kernel[1], 0, sizeof streams, &streams);
    test_error( err, "clSetKernelArgs failed.");

    globalsize[0] = (size_t)num_elements;
    globalsize[1] = (size_t)num_elements;

    size_t max_wgs;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wgs), &max_wgs, NULL);
    test_error( err, "clGetDeviceInfo failed.");

    localsize[0] = std::min<size_t>(16, max_wgs);
    localsize[1] = std::min<size_t>(11, max_wgs / localsize[0]);
    // If we need to use uniform workgroups because non-uniform workgroups are
    // not supported, round up to the next global size that is divisible by the
    // local size.
    if (use_uniform_work_groups)
    {
        if (globalsize[0] % localsize[0])
        {
            globalsize[0] += (localsize[0] - (globalsize[0] % localsize[0]));
        }
        if (globalsize[1] % localsize[1])
        {
            globalsize[1] += (localsize[1] - (globalsize[1] % localsize[1]));
        }
    }

    err = clEnqueueNDRangeKernel(queue, kernel[1], 2, NULL, globalsize, localsize, 0, NULL, NULL);
    test_error( err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams, CL_TRUE, 0, 2*sizeof(int), output_ptr, 0, NULL, NULL);
    test_error( err, "clEnqueueReadBuffer failed.");

    err = verify_enqueued_local_size(output_ptr, localsize, 2);

    globalsize[0] = (size_t)num_elements;
    localsize[0] = 9;
    if (use_uniform_work_groups && (globalsize[0] % localsize[0]))
    {
        globalsize[0] += (localsize[0] - (globalsize[0] % localsize[0]));
    }
    err = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL, globalsize, localsize, 0, NULL, NULL);
    test_error( err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams, CL_TRUE, 0, 2*sizeof(int), output_ptr, 0, NULL, NULL);
    test_error( err, "clEnqueueReadBuffer failed.");

    err = verify_enqueued_local_size(output_ptr, localsize, 1);

    // cleanup
    clReleaseMemObject(streams);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseProgram(program[0]);
    clReleaseProgram(program[1]);
    free(output_ptr);

    return err;
}
