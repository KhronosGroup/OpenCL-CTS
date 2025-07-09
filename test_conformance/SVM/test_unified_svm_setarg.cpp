//
// Copyright (c) 2025 The Khronos Group Inc.
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

#include "unified_svm_fixture.h"
#include "harness/conversions.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include <vector>

struct UnifiedSVMSetArg : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clSetKernelArgSVMPointer function for randome ranges
    // of a USM allocation. write a random pattern to the USM memory,
    // and validate that the kernel writes the correct data.
    cl_int test_svm_set_arg(USVMWrapper<cl_uchar> *src)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> src_data(alloc_count, 0);

        test_error(err, "clCreateBuffer failed.");

        for (size_t it = 0; it < test_iterations; it++)
        {
            // Fill src data with a random pattern
            generate_random_inputs(src_data, d);

            err = src->write(src_data);
            test_error(err, "could not write to usvm memory");

            // Select a random range
            size_t offset = get_random_size_t(0, src_data.size() - 1, d);
            size_t length = get_random_size_t(1, src_data.size() - offset, d);

            void *src_ptr = &src->get_ptr()[offset];

            err = clSetKernelArgSVMPointer(test_kernel, 0, src_ptr);
            test_error(err, "clSetKernelArgSVMPointer failed");

            std::vector<cl_uchar> result_data(length, 0);

            clMemWrapper dst_mem = clCreateBuffer(
                context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                result_data.size(), result_data.data(), &err);

            err = clSetKernelArg(test_kernel, 1, sizeof(dst_mem), &dst_mem);
            test_error(err, "clSetKernelArg failed.");

            size_t gws{ length };
            err = clEnqueueNDRangeKernel(queue, test_kernel, 1, nullptr, &gws,
                                         nullptr, 0, nullptr, nullptr);
            test_error(err, "clEnqueueNDRangeKernel failed");

            err = clEnqueueReadBuffer(queue, dst_mem, CL_TRUE, 0,
                                      result_data.size(), result_data.data(), 0,
                                      nullptr, nullptr);
            test_error(err, "clEnqueueReadBuffer failed");

            // Validate result
            const cl_uchar *expected_data = src_data.data() + offset;

            for (size_t i = 0; i < length; i++)
            {
                if (expected_data[i] != result_data[i])
                {
                    log_error("While attempting clSetKernelArgSVMPointer with "
                              "offset:%zu size:%zu \n"
                              "Data verification mismatch at %zu expected: %d "
                              "got: %d\n",
                              offset, length, i, expected_data[i],
                              result_data[i]);
                    return TEST_FAIL;
                }
            }
        }
        return CL_SUCCESS;
    }

    cl_int setup() override
    {
        cl_int err = UnifiedSVMBase::setup();
        if (CL_SUCCESS != err)
        {
            return err;
        }

        const char *programString = R"(
            kernel void test_kernel( const global char* src, global char* dst)
            {
                dst[get_global_id(0)] = src[get_global_id(0)];
            }
        )";

        cl_program program;
        err = create_single_kernel_helper(context, &program, &test_kernel, 1,
                                          &programString, "test_kernel");
        test_error(err, "could not create test_kernel kernel");

        err = clReleaseProgram(program);
        test_error(err, "could not release test_kernel program");

        return err;
    }

    cl_int run() override
    {
        cl_int err;
        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            auto mem = get_usvm_wrapper<cl_uchar>(ti);

            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            log_info("   testing clSetKernelArgSVMPointer() SVM type %u \n",
                     ti);
            err = test_svm_set_arg(mem.get());
            if (CL_SUCCESS != err)
            {
                return err;
            }

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        return CL_SUCCESS;
    }

    clKernelWrapper test_kernel;

    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 100;
};

REGISTER_TEST(unified_svm_set_arg)
{
    if (!is_extension_available(device, "cl_khr_unified_svm"))
    {
        log_info("cl_khr_unified_svm is not supported, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int err;

    clContextWrapper contextWrapper;
    clCommandQueueWrapper queueWrapper;

    // For now: create a new context and queue.
    // If we switch to a new test executable and run the tests without
    // forceNoContextCreation then this can be removed, and we can just use the
    // context and the queue from the harness.
    if (context == nullptr)
    {
        contextWrapper =
            clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        test_error(err, "clCreateContext failed");
        context = contextWrapper;
    }

    if (queue == nullptr)
    {
        queueWrapper = clCreateCommandQueue(context, device, 0, &err);
        test_error(err, "clCreateCommandQueue failed");
        queue = queueWrapper;
    }

    UnifiedSVMSetArg Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
