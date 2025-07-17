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
#include <cinttypes>
#include <memory>

struct UnifiedSVMMigrate : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clEnqueueSVMMigrateMem function for random ranges
    // of a USM allocation.
    cl_int test_SVMMigrate(USVMWrapper<cl_uchar> *mem,
                           cl_mem_migration_flags flags, bool random_offset,
                           bool random_length)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> mem_data(alloc_count, 0);

        for (size_t it = 0; it < test_iterations; it++)
        {
            // Fill src data with a random pattern
            generate_random_inputs(mem_data, d);

            err = mem->write(mem_data);
            test_error(err, "could not write to usvm memory");

            // Select a random range
            size_t offset = random_offset
                ? get_random_size_t(0, mem_data.size() - 1, d)
                : 0;

            size_t length = random_length
                ? get_random_size_t(1, mem_data.size() - offset, d)
                : mem_data.size() - offset;

            const void *ptr = &mem->get_ptr()[offset];

            clEventWrapper event;

            err = clEnqueueSVMMigrateMem(queue, 1, &ptr, &length, flags, 0,
                                         nullptr, &event);
            test_error(err, "clEnqueueSVMMigrateMem failed");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            err = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
            test_error(
                err,
                "Invalid command type returned for clEnqueueSVMMigrateMem");
        }
        return CL_SUCCESS;
    }

    cl_int test_svm_migrate(cl_uint typeIndex)
    {
        cl_int err;

        const cl_mem_migration_flags flags[] = {
            0, CL_MIGRATE_MEM_OBJECT_HOST,
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
            CL_MIGRATE_MEM_OBJECT_HOST | CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
        };

        auto mem = get_usvm_wrapper<cl_uchar>(typeIndex);

        // Test migrate whole allocation
        for (auto test_flags : flags)
        {
            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            err = test_SVMMigrate(mem.get(), test_flags, false, false);
            test_error(err, "test_SVMMigrate");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test migrate subset allocation from random offset to end
        for (auto test_flags : flags)
        {
            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            err = test_SVMMigrate(mem.get(), test_flags, true, false);
            test_error(err, "test_SVMMigrate");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test migrate subset allocation from base pointer to random size
        for (auto test_flags : flags)
        {

            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            err = test_SVMMigrate(mem.get(), test_flags, false, true);
            test_error(err, "test_SVMMigrate");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test migrate subset allocation from random offset to random end
        for (auto test_flags : flags)
        {

            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            err = test_SVMMigrate(mem.get(), test_flags, true, true);
            test_error(err, "test_SVMMigrate");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        // For each supported svm type test clEnqueueSVMMigrateMem for all
        // possible pattern sizes
        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            log_info("   testing clEnqueueSVMMigrateMem() SVM type %u \n", ti);
            err = test_svm_migrate(ti);
            test_error(err, "clEnqueueSVMMigrateMem() testing failed");
        }
        return CL_SUCCESS;
    }


    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 10;
};

REGISTER_TEST(unified_svm_migrate)
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

    UnifiedSVMMigrate Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
