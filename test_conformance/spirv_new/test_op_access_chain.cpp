//
// Copyright (c) 2026 The Khronos Group Inc.
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

#include "testBase.h"
#include "types.hpp"

// Notes:
//
// The test type for all tests is either a four component array of uints
// (OpTypeArray) or a four component vector of uints (OpTypeVector).
//
//
// The kernels for the access chain tests look like:
//
// kernel void TEST_KERNEL_NAME(global TEST_TYPE* in, global uint* out) {
//     size_t gid_0 = get_global_id(0);
//     global uint* dst = &out[gid_0];
//     global TEST_TYPE* base = &in[gid_0];
//     uint index = (uint)gid_0 % 4;
//     global TEST_TYPE* temp_base = TEST_INSTRUCTION(base);    // no indices
//     global uint* temp_src = TEST_INSTRUCTION(temp_base, index);
//     global uint* src = TEST_INSTRUCTION(temp_src);           // no indices
//     uint data = *src;
//     *dst = data;
// }
//
// The test instruction for these tests is OpAccessChain or
// OpInBoundsAccessChain.
//
//
// The kernels for the pointer access chain tests look like:
//
// kernel void TEST_KERNEL_NAME(global TEST_TYPE* in, global uint* out) {
//     size_t gid_0 = get_global_id(0);
//     global uint* dst = &out[gid_0];
//     uint index = (uint)gid_0 % 4;
//     global TEST_TYPE* temp_base = TEST_INSTRUCTION(in, 0); // no indices
//     global uint* temp_src = TEST_INSTRUCTION(temp_base, gid_0, index);
//     global uint* src = TEST_INSTRUCTION(temp_src, 0);        // no indices
//     uint data = *src;
//     *dst = data;
// }
//
// The test instruction for these tests is OpPtrAccessChain or
// OpInBoundsPtrAccessChain.

static int test_access_chain_helper(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, const char *name)
{
    const int array_size = 4;
    const int num_work_items = 256;

    cl_uint values[array_size] = { 100, 200, 300, 400 };

    std::vector<cl_uint> input_data;
    for (int i = 0; i < num_work_items; i++)
    {
        for (int j = 0; j < array_size; j++)
        {
            input_data.push_back(values[j]);
        }
    }

    std::vector<cl_uint> expected_results;
    for (int i = 0; i < num_work_items; i++)
    {
        int array_index = i % array_size;
        expected_results.push_back(values[array_index]);
    }

    clProgramWrapper prog;
    cl_int err = get_program_with_il(prog, deviceID, context, name);
    SPIRV_CHECK_ERROR(err, "Failed to build program");

    clKernelWrapper kernel = clCreateKernel(prog, name, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create kernel");

    size_t input_bytes = input_data.size() * sizeof(cl_uint);
    size_t output_bytes = expected_results.size() * sizeof(cl_uint);

    clMemWrapper input_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       input_bytes, &input_data[0], &err);
    SPIRV_CHECK_ERROR(err, "Failed to create input buffer");

    clMemWrapper output_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, output_bytes, NULL, &err);
    SPIRV_CHECK_ERROR(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_mem);
    SPIRV_CHECK_ERROR(err, "Failed to set kernel argument 1");

    size_t global = num_work_items;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL,
                                 NULL);
    SPIRV_CHECK_ERROR(err, "Failed to enqueue kernel");

    std::vector<cl_uint> host(num_work_items);
    err = clEnqueueReadBuffer(queue, output_mem, CL_TRUE, 0, output_bytes,
                              &host[0], 0, NULL, NULL);
    SPIRV_CHECK_ERROR(err, "Failed to copy from cl_buffer");

    for (int i = 0; i < num_work_items; i++)
    {
        if (host[i] != expected_results[i])
        {
            log_error(
                "Values do not match at location %d: expected %u, got %u\n", i,
                expected_results[i], host[i]);
            return -1;
        }
    }
    return 0;
}

REGISTER_TEST(access_chain_array)
{
    return test_access_chain_helper(device, context, queue,
                                    "access_chain_array");
}

REGISTER_TEST(access_chain_inbounds_array)
{
    return test_access_chain_helper(device, context, queue,
                                    "access_chain_inbounds_array");
}

REGISTER_TEST(ptr_access_chain_array)
{
    return test_access_chain_helper(device, context, queue,
                                    "ptr_access_chain_array");
}

REGISTER_TEST(ptr_access_chain_inbounds_array)
{
    return test_access_chain_helper(device, context, queue,
                                    "ptr_access_chain_inbounds_array");
}

REGISTER_TEST(access_chain_vector)
{
    return test_access_chain_helper(device, context, queue,
                                    "access_chain_vector");
}

REGISTER_TEST(access_chain_inbounds_vector)
{
    return test_access_chain_helper(device, context, queue,
                                    "access_chain_inbounds_vector");
}

REGISTER_TEST(ptr_access_chain_vector)
{
    return test_access_chain_helper(device, context, queue,
                                    "ptr_access_chain_vector");
}

REGISTER_TEST(ptr_access_chain_inbounds_vector)
{
    return test_access_chain_helper(device, context, queue,
                                    "ptr_access_chain_inbounds_vector");
}
