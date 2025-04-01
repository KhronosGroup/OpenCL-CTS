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

int test_cxx_for_opencl_version(cl_device_id device, cl_context context,
                                cl_command_queue queue)
{
    cl_int cxx4opencl_version;
    cl_int cxx4opencl_expected_version;
    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int error;
    cl_int value = 0;
    const char *kernel_sstr =
        R"(
        __kernel void k(__global int* buf) {
            buf[0] = __OPENCL_CPP_VERSION__;
        })";
    const size_t lengths[1] = { std::string{ kernel_sstr }.size() };

    clProgramWrapper writer_program =
        clCreateProgramWithSource(context, 1, &kernel_sstr, lengths, &error);
    test_error(error, "Failed to create program with source");

    error = clCompileProgram(writer_program, 1, &device, "-cl-std=CLC++", 0,
                             nullptr, nullptr, nullptr, nullptr);
    test_error(error, "Failed to compile program");

    cl_program progs[1] = { writer_program };
    program = clLinkProgram(context, 1, &device, "", 1, progs, 0, 0, &error);
    test_error(error, "Failed to link program");

    clMemWrapper out =
        clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(cxx4opencl_version), &cxx4opencl_version, &error);
    test_error(error, "clCreateBuffer failed");

    kernel = clCreateKernel(program, "k", &error);
    test_error(error, "Failed to create k kernel");

    error = clSetKernelArg(kernel, 0, sizeof(out), &out);
    test_error(error, "clSetKernelArg failed");

    size_t global_size = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clEnqueueReadBuffer(queue, out, CL_BLOCKING, 0,
                                sizeof(cxx4opencl_version), &cxx4opencl_version,
                                0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    error =
        clGetDeviceInfo(device, CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT,
                        sizeof(value), &value, nullptr);
    test_error(error, "Failed to get device info");

    cxx4opencl_expected_version = CL_VERSION_MAJOR_KHR(value) * 100
        + CL_VERSION_MINOR_KHR(value) * 10 + CL_VERSION_PATCH_KHR(value);

    if (cxx4opencl_version != cxx4opencl_expected_version)
    {
        log_error("ERROR: C++ for OpenCL version mismatch - returned %lu, "
                  "expected %lu\n",
                  static_cast<long unsigned>(value),
                  static_cast<long unsigned>(cxx4opencl_expected_version));
        return TEST_FAIL;
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(cxx_for_opencl_ver, Version(2, 0))
{
    if (!is_extension_available(device, "cl_ext_cxx_for_opencl"))
    {
        log_info("Device does not support 'cl_ext_cxx_for_opencl'. Skipping "
                 "the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_cxx_for_opencl_version(device, context, queue);
}
