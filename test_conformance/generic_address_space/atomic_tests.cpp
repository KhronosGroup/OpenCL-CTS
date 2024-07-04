//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "errorHelpers.h"
#include "typeWrappers.h"

namespace {
// In this source, each workgroup will generate one value.
// Every other workgroup will use either a global or local
// pointer on an atomic operation.
const char* KernelSourceInvariant = R"OpenCLC(
kernel void testKernel(global atomic_int* globalPtr, local atomic_int* localPtr) {
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    int wgid = get_group_id(0);
    int wgsize = get_local_size(0);

    if (tid == 0) atomic_store(localPtr, 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialise the generic pointer to
    // the global.
    generic atomic_int* ptr = globalPtr + wgid;

    // In a workgroup-invariant way, select a localPtr instead.
    if ((wgid % 2) == 0)
        ptr = localPtr;

    int inc = atomic_fetch_add(ptr, 1);

    // In the cases where the local memory ptr was used,
    // save off the final value.
    if ((wgid % 2) == 0 && inc == (wgsize-1))
        atomic_store(&globalPtr[wgid], inc);
}
)OpenCLC";

// In this source, each workgroup will generate two values.
// Every other work item in the workgroup will select either
// a local or global memory pointer and perform an atomic
// operation on that.
const char* KernelSourceVariant = R"OpenCLC(
kernel void testKernel(global atomic_int* globalPtr, local atomic_int* localPtr) {
    int gid = get_global_id(0);
    int tid = get_local_id(0);
    int wgid = get_group_id(0);
    int wgsize = get_local_size(0);

    if (tid == 0) atomic_store(localPtr, 0);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialise the generic pointer to
    // the global.  Two values are written per WG.
    generic atomic_int* ptr = globalPtr + (wgid * 2);

    // In a workgroup-invariant way, select a localPtr instead.
    if ((tid % 2) == 0)
        ptr = localPtr;

    atomic_fetch_add(ptr, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    // In the cases where the local memory ptr was used,
    // save off the final value.
    if (tid == 0)
        atomic_store(&globalPtr[(wgid * 2) + 1], atomic_load(localPtr));
}
)OpenCLC";
}

int test_generic_atomics_invariant(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int)
{
    const auto version = get_device_cl_version(deviceID);

    if (version < Version(2, 0)) return TEST_SKIPPED_ITSELF;

    cl_int err = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper kernel;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &KernelSourceInvariant, "testKernel");
    test_error(err, "Failed to create test kernel");

    size_t wgSize, retSize;
    // Attempt to find the simd unit size for the device.
    err = clGetKernelWorkGroupInfo(kernel, deviceID,
                                   CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(wgSize), &wgSize, &retSize);
    test_error(err, "clGetKernelWorkGroupInfo failed");

    // How many workgroups to run for the test.
    const int numWGs = 2;
    const size_t bufferSize = numWGs * sizeof(cl_uint);
    clMemWrapper buffer =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    test_error(err, "clCreateBuffer failed");
    const cl_int zero = 0;
    err = clEnqueueFillBuffer(queue, buffer, &zero, sizeof(zero), 0, bufferSize,
                              0, nullptr, nullptr);
    test_error(err, "clEnqueueFillBuffer failed");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    test_error(err, "clSetKernelArg failed");
    err = clSetKernelArg(kernel, 1, bufferSize, nullptr);
    test_error(err, "clSetKernelArg failed");

    const size_t globalSize = wgSize * numWGs;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize,
                                 &wgSize, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed");

    std::vector<cl_int> results(numWGs);
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize,
                              results.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed");

    clFinish(queue);

    for (size_t i = 0; i < numWGs; ++i)
    {
        const cl_int expected = ((i % 2) == 0) ? wgSize - 1 : wgSize;
        if (results[i] != expected)
        {
            log_error("Verification on device failed at index %zu\n", i);
            return TEST_FAIL;
        }
    }

    return CL_SUCCESS;
}

int test_generic_atomics_variant(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int)
{
    const auto version = get_device_cl_version(deviceID);

    if (version < Version(2, 0)) return TEST_SKIPPED_ITSELF;

    cl_int err = CL_SUCCESS;
    clProgramWrapper program;
    clKernelWrapper kernel;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &KernelSourceVariant, "testKernel");
    test_error(err, "Failed to create test kernel");

    size_t wgSize, retSize;
    // Attempt to find the simd unit size for the device.
    err = clGetKernelWorkGroupInfo(kernel, deviceID,
                                   CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(wgSize), &wgSize, &retSize);
    test_error(err, "clGetKernelWorkGroupInfo failed");

    // How many workgroups to run for the test.
    const int numWGs = 2;
    const size_t bufferSize = numWGs * sizeof(cl_uint) * 2;
    clMemWrapper buffer =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    test_error(err, "clCreateBuffer failed");
    const cl_int zero = 0;
    err = clEnqueueFillBuffer(queue, buffer, &zero, sizeof(zero), 0, bufferSize,
                              0, nullptr, nullptr);
    test_error(err, "clEnqueueFillBuffer failed");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    test_error(err, "clSetKernelArg failed");
    err = clSetKernelArg(kernel, 1, bufferSize, nullptr);
    test_error(err, "clSetKernelArg failed");

    const size_t globalSize = wgSize * numWGs;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize,
                                 &wgSize, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed");

    std::vector<cl_int> results(numWGs * 2);
    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, bufferSize,
                              results.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed");

    clFinish(queue);

    const cl_int expected = wgSize / 2;
    for (size_t i = 0; i < (numWGs * 2); i += 2)
    {
        if (results[i] != expected)
        {
            log_error("Verification on device failed at index %zu\n", i);
            return TEST_FAIL;
        }
        if (results[i + 1] != expected)
        {
            const size_t index = i + 1;
            log_error("Verification on device failed at index %zu\n", index);
            return TEST_FAIL;
        }
    }

    return CL_SUCCESS;
}
