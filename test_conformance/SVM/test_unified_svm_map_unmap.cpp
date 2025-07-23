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

struct UnifiedSVMMapUnmap : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clEnqueueSVMMap and clEnqueueSVMUnmap functions for random
    // ranges of a USM allocation and validate the event types.
    cl_int test_SVMMapUnmap(USVMWrapper<cl_uchar> *mem, cl_map_flags flags)
    {
        cl_int err = CL_SUCCESS;

        for (size_t it = 0; it < test_iterations; it++)
        {

            size_t offset = get_random_size_t(0, alloc_count - 1, d);
            size_t length = get_random_size_t(1, alloc_count - offset, d);

            void *ptr = &mem->get_ptr()[offset];

            clEventWrapper map_event;
            err = clEnqueueSVMMap(queue, CL_FALSE, flags, ptr, length, 0,
                                  nullptr, &map_event);
            test_error(err, "clEnqueueSVMMap failed");

            clEventWrapper unmap_event;
            err = clEnqueueSVMUnmap(queue, ptr, 0, nullptr, &unmap_event);
            test_error(err, "clEnqueueSVMUnmap failed");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            err = check_event_type(map_event, CL_COMMAND_SVM_MAP);
            test_error(err,
                       "Invalid command type returned for clEnqueueSVMMap");

            err = check_event_type(unmap_event, CL_COMMAND_SVM_UNMAP);
            test_error(err,
                       "Invalid command type returned for clEnqueueSVMUnmap");
        }

        return err;
    }

    cl_int run() override
    {
        cl_int err;
        cl_map_flags test_flags[] = { CL_MAP_READ, CL_MAP_WRITE,
                                      CL_MAP_WRITE_INVALIDATE_REGION,
                                      CL_MAP_READ | CL_MAP_WRITE };

        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            if (deviceUSVMCaps[ti] & CL_SVM_CAPABILITY_HOST_MAP_KHR)
            {
                for (auto flags : test_flags)
                {
                    auto mem = get_usvm_wrapper<cl_uchar>(ti);

                    err = mem->allocate(alloc_count);
                    test_error(err, "SVM allocation failed");

                    err = test_SVMMapUnmap(mem.get(), flags);
                    test_error(err, "test_SVMMemfill");

                    err = mem->free();
                    test_error(err, "SVM free failed");
                }
            }
        }
        return CL_SUCCESS;
    }

    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 100;
};

REGISTER_TEST(unified_svm_map_unmap)
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

    UnifiedSVMMapUnmap Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
