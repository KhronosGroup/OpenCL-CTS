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
//

#include <numeric>

#include "harness/typeWrappers.h"
#include "harness/testHarness.h"

static const char* kernel_function_inc_buffer = R"(
	kernel void inc_buffer(global uint *src, global uint *imp, global uint *dst)
	{
	    uint global_id = get_global_id(0);

	    imp[global_id] = src[global_id] + 1;
	    dst[global_id] = imp[global_id] + 1;
	}
	)";

/**
 * Demonstrate the functionality of the cl_khr_external_memory_dma_buf extension
 * by creating an imported buffer from a DMA buffer, then writing into, and
 * reading from it.
 */

REGISTER_TEST(external_memory_dma_buf)
{
    if (!is_extension_available(device, "cl_khr_external_memory_dma_buf"))
    {
        log_info("The device does not support the "
                 "cl_khr_external_memory_dma_buf extension.\n");

        return TEST_SKIPPED_ITSELF;
    }

    const size_t buffer_size = static_cast<size_t>(num_elements);
    const size_t buffer_size_bytes = sizeof(uint32_t) * buffer_size;

    clProgramWrapper program;
    clKernelWrapper kernel;
    cl_int error;

    error =
        create_single_kernel_helper(context, &program, &kernel, 1,
                                    &kernel_function_inc_buffer, "inc_buffer");
    test_error(error, "Failed to create program with source.");

    /* Source buffer initialisation */
    std::vector<uint32_t> src_data(buffer_size);
    // Arithmetic progression starting at 0 and incrementing by 1
    std::iota(std::begin(src_data), std::end(src_data), 0);

    clMemWrapper src_buffer =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       buffer_size_bytes, src_data.data(), &error);
    test_error(error, "Failed to create the source buffer.");

    /* Imported buffer creation */
    int dma_buf_fd = allocate_dma_buf(buffer_size_bytes);
    if (dma_buf_fd < 0)
    {
        if (dma_buf_fd == TEST_SKIPPED_ITSELF)
        {
            return TEST_SKIPPED_ITSELF;
        }

        log_error(
            "Failed to obtain a valid DMA buffer file descriptor, got %i.\n",
            dma_buf_fd);

        return TEST_FAIL;
    }

    const cl_mem_properties ext_mem_properties[] = {
        CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        static_cast<cl_mem_properties>(dma_buf_fd), CL_PROPERTIES_LIST_END_EXT
    };

    clMemWrapper imp_buffer = clCreateBufferWithProperties(
        context, ext_mem_properties, CL_MEM_READ_WRITE, buffer_size_bytes,
        nullptr, &error);
    test_error(error, "Failed to create the imported buffer.");

    /* Destination buffer creation */
    clMemWrapper dst_buffer = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY, buffer_size_bytes, nullptr, &error);
    test_error(error, "Failed to create the destination buffer.");

    /* Kernel arguments setup */
    error = clSetKernelArg(kernel, 0, sizeof(src_buffer), &src_buffer);
    test_error(error, "Failed to set kernel argument 0 to src_buffer.");

    error = clSetKernelArg(kernel, 1, sizeof(imp_buffer), &imp_buffer);
    test_error(error, "Failed to set kernel argument 1 to imp_buffer.");

    error = clSetKernelArg(kernel, 2, sizeof(dst_buffer), &dst_buffer);
    test_error(error, "Failed to set kernel argument 2 to dst_buffer.");

    /* Kernel execution */
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &buffer_size,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "Failed to enqueue the kernel.");

    error = clFinish(queue);
    test_error(error, "Failed to finish the queue.");

    /* Verification */
    std::vector<uint32_t> dst_data(buffer_size, 0);

    error = clEnqueueReadBuffer(queue, dst_buffer, CL_BLOCKING, 0,
                                buffer_size_bytes, dst_data.data(), 0, nullptr,
                                nullptr);
    test_error(error, "Failed to read the contents of the destination buffer.");

    std::vector<uint32_t> expected_data(buffer_size);
    std::iota(std::begin(expected_data), std::end(expected_data), 2);

    for (size_t i = 0; i < buffer_size; ++i)
    {
        if (dst_data[i] != expected_data[i])
        {
            log_error(
                "Verification failed at index %zu, expected %u but got %u\n", i,
                expected_data[i], dst_data[i]);

            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}
