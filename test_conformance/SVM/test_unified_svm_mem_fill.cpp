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

struct UnifiedSVMMemFill : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clEnqueueSVMMemFill function for random patterns
    // over a random range of a USM allocation.
    cl_int test_SVMMemfill(USVMWrapper<cl_uchar> *mem)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> mem_data(alloc_count, 0);

        for (size_t pattern_size = 1; pattern_size <= 128; pattern_size *= 2)
        {
            std::vector<cl_uchar> fill_data(pattern_size, 0);

            // Fill src data with a random pattern
            generate_random_inputs(fill_data, d);

            err = mem->write(mem_data);
            test_error(err, "could not write to usvm memory");

            // Select a random range
            size_t offset = get_random_size_t(0, mem_data.size() - 1, d);

            // Align offset to pattern size
            offset &= ~(pattern_size - 1);

            // Select a random size.
            size_t fill_size =
                get_random_size_t(pattern_size, mem_data.size() - offset, d);

            // Align length to pattern size
            fill_size &= ~(pattern_size - 1);

            void *ptr = &mem->get_ptr()[offset];

            clEventWrapper event;
            err = clEnqueueSVMMemFill(queue, ptr, fill_data.data(),
                                      fill_data.size(), fill_size, 0, nullptr,
                                      &event);
            test_error(err, "clEnqueueSVMMemFill failed");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            err = check_event_type(event, CL_COMMAND_SVM_MEMFILL);
            test_error(err,
                       "Invalid command type returned for clEnqueueSVMMemFill");

            // Validate result
            std::vector<cl_uchar> result_data(alloc_count, 0);

            err = mem->read(result_data);
            test_error(err, "could not read from usvm memory");

            for (size_t i = 0; i < result_data.size(); i++)
            {
                cl_uchar expected_value;
                if (i >= offset && i < fill_size + offset)
                {
                    expected_value = fill_data[i % pattern_size];
                }
                else
                {
                    expected_value = mem_data[i];
                }

                if (expected_value != result_data[i])
                {
                    log_error("While attempting clEnqueueSVMMemFill with "
                              "offset:%zu size:%zu \n"
                              "Data verification mismatch at %zu expected: %d "
                              "got: %d\n",
                              offset, fill_size, i, expected_value,
                              result_data[i]);
                    return TEST_FAIL;
                }
            }
        }
        return CL_SUCCESS;
    }

    cl_int test_svm_memfill(cl_uint srcTypeIndex)
    {
        cl_int err;

        auto mem = get_usvm_wrapper<cl_uchar>(srcTypeIndex);

        err = mem->allocate(alloc_count);
        test_error(err, "SVM allocation failed");

        err = test_SVMMemfill(mem.get());
        test_error(err, "test_SVMMemfill");

        err = mem->free();
        test_error(err, "SVM free failed");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        // For each supported svm type test clEnqueueSVMMemFill for all
        // possible pattern sizes
        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            log_info("   testing clEnqueueSVMMemFill() SVM type %u \n", ti);
            err = test_svm_memfill(ti);
            if (CL_SUCCESS != err)
            {
                return err;
            }
        }
        return CL_SUCCESS;
    }

    template <typename T>
    std::unique_ptr<USVMWrapper<T>> get_hostptr_usvm_wrapper()
    {
        return std::unique_ptr<USVMWrapper<T>>(
            new USVMWrapper<T>(nullptr, nullptr, nullptr, CL_UINT_MAX,
                               CL_SVM_PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR
                                   | CL_SVM_CAPABILITY_HOST_READ_KHR
                                   | CL_SVM_CAPABILITY_HOST_WRITE_KHR,
                               0, nullptr, nullptr, nullptr, nullptr));
    }

    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 100;
};

REGISTER_TEST(unified_svm_memfill)
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

    UnifiedSVMMemFill Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
