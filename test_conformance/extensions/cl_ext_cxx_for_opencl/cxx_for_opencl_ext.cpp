//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include <CL/cl.h>

#include "harness/typeWrappers.h"

int test_cxx_for_opencl(cl_device_id device, cl_context context,
                        cl_command_queue queue)
{
    cl_int error;
    clProgramWrapper program;
    clKernelWrapper kernel1;
    clKernelWrapper kernel2;
    clMemWrapper in_buffer;
    clMemWrapper out_buffer;
    cl_int value = 7;

    const char *kernel_sstr =
        R"(
        __global int x;
        template<typename T>
        void execute(T &a, const T &b) {
            a = b * 2;
        }
        __kernel void k1(__global int *p) {
            execute(x, *p);
        }
        __kernel void k2(__global int *p) {
            execute(*p, x);
        })";

    error = create_single_kernel_helper_with_build_options(
        context, &program, &kernel1, 1, &kernel_sstr, "k1", "-cl-std=CLC++");
    test_error(error, "Failed to create k1 kernel");

    kernel2 = clCreateKernel(program, "k2", &error);
    test_error(error, "Failed to create k2 kernel");

    in_buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(value), &value, &error);
    test_error(error, "clCreateBuffer failed");

    out_buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(value), &value, &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel1, 0, sizeof(in_buffer), &in_buffer);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel2, 0, sizeof(out_buffer), &out_buffer);
    test_error(error, "clSetKernelArg failed");

    size_t global_size = 1;
    error = clEnqueueNDRangeKernel(queue, kernel1, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueNDRangeKernel(queue, kernel2, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, out_buffer, CL_BLOCKING, 0,
                                sizeof(value), &value, 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    if (value != 28)
    {
        log_error("ERROR: Kernel wrote %lu, expected 28\n",
                  static_cast<long unsigned>(value));
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(cxx_for_opencl_ext, Version(2, 0))
{
    if (!is_extension_available(device, "cl_ext_cxx_for_opencl"))
    {
        log_info("Device does not support 'cl_ext_cxx_for_opencl'. Skipping "
                 "the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_cxx_for_opencl(device, context, queue);
}
