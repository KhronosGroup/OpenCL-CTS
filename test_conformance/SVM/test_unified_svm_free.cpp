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
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>

namespace {

struct CallbackData
{
    CallbackData(cl_context ctx, std::vector<cl_svm_capabilities_khr> &caps)
        : context{ ctx }, status{ 0 }, svm_pointers{}, svm_caps{ caps }
    {}
    cl_context context;
    std::atomic<cl_uint> status;
    std::vector<void *> svm_pointers;
    std::vector<cl_svm_capabilities_khr> &svm_caps;
};

// callback which will be passed to clEnqueueSVMFree command
void CL_CALLBACK callback_svm_free(cl_command_queue queue,
                                   cl_uint num_svm_pointers,
                                   void *svm_pointers[], void *user_data)
{
    auto data = (CallbackData *)user_data;

    data->svm_pointers.resize(num_svm_pointers, 0);

    for (size_t i = 0; i < num_svm_pointers; ++i)
    {
        data->svm_pointers[i] = svm_pointers[i];

        if (data->svm_caps[i] & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR)
        {
            align_free(data);
        }
        else
        {
            clSVMFree(data->context, svm_pointers[i]);
        }
    }

    data->status.store(1, std::memory_order_release);
}

void log_error_usvm_ptrs(const std::vector<void *> &v)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        log_error("\t%zu: %p\n", i, v[i]);
    }
}
}

struct UnifiedSVMFree : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clEnqueueSVMFree function for a vector of USM pointers
    // and validate the callback.
    cl_int
    test_SVMFreeCallback(std::vector<void *> &buffers,
                         std::vector<cl_svm_capabilities_khr> &bufferCaps)
    {
        cl_int err = CL_SUCCESS;

        clEventWrapper event;

        CallbackData data{ context, bufferCaps };

        err = clEnqueueSVMFree(queue, buffers.size(), buffers.data(),
                               callback_svm_free, &data, 0, 0, &event);
        test_error(err, "clEnqueueSVMFree failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        err = check_event_type(event, CL_COMMAND_SVM_FREE);
        test_error(err, "Invalid command type returned for clEnqueueSVMFree");

        // wait for the callback
        while (data.status.load(std::memory_order_acquire) == 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }

        // check if pointers returned in callback are correct
        if (data.svm_pointers != buffers)
        {
            log_error("Invalid SVM pointer returned in the callback \n");
            log_error("Expected:\n");
            log_error_usvm_ptrs(buffers);
            log_error("Got:\n");
            log_error_usvm_ptrs(data.svm_pointers);

            return TEST_FAIL;
        }

        return CL_SUCCESS;
    }

    cl_int test_SVMFree(std::vector<void *> &buffers)
    {
        cl_int err = CL_SUCCESS;

        clEventWrapper event;

        err = clEnqueueSVMFree(queue, buffers.size(), buffers.data(), nullptr,
                               nullptr, 0, 0, &event);
        test_error(err, "clEnqueueSVMFree failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        err = check_event_type(event, CL_COMMAND_SVM_FREE);
        test_error(err, "Invalid command type returned for clEnqueueSVMFree");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        // Test clEnqueueSVMFree function with a callback
        for (int it = 0; it < test_iterations; it++)
        {
            std::vector<void *> buffers;
            std::vector<cl_svm_capabilities_khr> bufferCaps;

            size_t numSVMBuffers = get_random_size_t(1, 20, d);

            for (int i = 0; i < numSVMBuffers; i++)
            {
                size_t typeIndex =
                    get_random_size_t(0, deviceUSVMCaps.size() - 1, d);

                auto mem = get_usvm_wrapper<cl_uchar>(typeIndex);

                err = mem->allocate(alloc_count);
                test_error(err, "SVM allocation failed");

                buffers.push_back(mem->get_ptr());
                bufferCaps.push_back(deviceUSVMCaps[typeIndex]);

                mem->reset();
            }

            err = test_SVMFreeCallback(buffers, bufferCaps);
            test_error(err, "test_SVMFree");
        }

        // We need to filter out the SVM types that are system allocated
        // as we cannot test clEnqueueSVMFree without a callback for them
        std::vector<size_t> test_indexes;
        for (size_t i = 0; i < deviceUSVMCaps.size(); i++)
        {
            auto caps = deviceUSVMCaps[i];
            if (0 == (caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR))
            {
                test_indexes.push_back(i);
            }
        }

        if (!test_indexes.empty())
        {
            // Test clEnqueueSVMFree function with no callback
            for (int it = 0; it < test_iterations; it++)
            {
                std::vector<void *> buffers;

                size_t numSVMBuffers = get_random_size_t(1, 20, d);

                while (buffers.size() != numSVMBuffers)
                {
                    size_t test_index =
                        get_random_size_t(0, test_indexes.size() - 1, d);
                    size_t typeIndex = test_indexes[test_index];

                    auto mem = get_usvm_wrapper<cl_uchar>(typeIndex);

                    err = mem->allocate(alloc_count);
                    test_error(err, "SVM allocation failed");

                    buffers.push_back(mem->get_ptr());

                    mem->reset();
                }

                err = test_SVMFree(buffers);
                test_error(err, "test_SVMFree");
            }
        }
        return CL_SUCCESS;
    }

    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 100;
};

REGISTER_TEST(unified_svm_free)
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

    UnifiedSVMFree Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
