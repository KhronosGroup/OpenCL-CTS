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

#include "unified_svm_fixture.h"
#include <cinttypes>
#include <memory>
#include <thread>

void worker(USVMWrapper<cl_int>* mem, cl_int* alloc_ret, cl_int* free_ret)
{
    *alloc_ret = mem->allocate(4);
    *free_ret = mem->free();
}

struct UnifiedSVMThreadedAllocFree : UnifiedSVMBase
{
    UnifiedSVMThreadedAllocFree(cl_context context, cl_device_id device,
                                cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int run() override
    {
        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            const unsigned num_threads = 10;
            std::vector<std::thread> threads;
            std::vector<std::unique_ptr<USVMWrapper<cl_int>>> mems(num_threads);
            std::vector<std::pair<cl_int, cl_int>> ret_codes(num_threads);

            for (int i = 0; i < num_threads; ++i)
            {
                mems[i] = get_usvm_wrapper<cl_int>(ti);
                cl_int& alloc = ret_codes[i].first;
                cl_int& free = ret_codes[i].second;
                threads.push_back(
                    std::thread(worker, mems[i].get(), &alloc, &free));
            }

            for (auto& thread : threads)
            {
                thread.join();
            }

            for (const auto& [alloc_ret, free_ret] : ret_codes)
            {
                test_error(alloc_ret, "USVM allocation failed");
                test_error(free_ret, "USVM free failed");
            }
        }

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_threading_alloc_free)
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

    UnifiedSVMThreadedAllocFree Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
