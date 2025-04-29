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

struct UnifiedSVMOPs : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test the clEnqueueSVMMemcpy function for random ranges
    // of a USM allocation and validate the results.
    cl_int test_SVMMemcpy(USVMWrapper<cl_uchar> *src,
                          USVMWrapper<cl_uchar> *dst)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> src_data(alloc_count, 0);
        std::vector<cl_uchar> dst_data(alloc_count, 0);

        for (size_t it = 0; it < test_iterations; it++)
        {
            // Fill src data with a random pattern
            generate_random_inputs(src_data, d);

            err = src->write(src_data);
            test_error(err, "could not write to usvm memory");

            // Fill dst data with zeros
            err = dst->write(dst_data);
            test_error(err, "could not write to usvm memory");

            // Select a random range
            size_t offset = get_random_size_t(0, src_data.size() - 1, d);
            size_t length = get_random_size_t(1, src_data.size() - offset, d);

            void *src_ptr = &src->get_ptr()[offset];
            void *dst_ptr = &dst->get_ptr()[offset];

            clEventWrapper event;
            err = clEnqueueSVMMemcpy(queue, CL_BLOCKING, dst_ptr, src_ptr,
                                     length, 0, nullptr, &event);
            test_error(err, "clEnqueueSVMMemcpy failed");

            err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
            test_error(err,
                       "Invalid command type returned for clEnqueueSVMMemcpy");

            // Validate result
            std::vector<cl_uchar> result_data(alloc_count, 0);

            err = dst->read(result_data);
            test_error(err, "could not read from usvm memory");

            for (size_t i = 0; i < result_data.size(); i++)
            {
                cl_uchar expected_value;
                if (i >= offset && i < length + offset)
                {
                    expected_value = src_data[i];
                }
                else
                {
                    expected_value = 0;
                }

                if (expected_value != result_data[i])
                {
                    log_error("While attempting clEnqueueSVMMemcpy with "
                              "offset:%zu size:%zu \n"
                              "Data verification mismatch at %zu expected: %d "
                              "got: %d\n",
                              offset, length, i, expected_value,
                              result_data[i]);
                    return TEST_FAIL;
                }
            }
        }
        return CL_SUCCESS;
    }

    cl_int test_svm_memcpy(cl_uint srcTypeIndex, cl_uint dstTypeIndex)
    {
        cl_int err;

        auto srcMem = get_usvm_wrapper<cl_uchar>(srcTypeIndex);
        auto dstMem = get_usvm_wrapper<cl_uchar>(dstTypeIndex);

        err = srcMem->allocate(alloc_count);
        test_error(err, "SVM allocation failed");

        err = dstMem->allocate(alloc_count);
        test_error(err, "SVM allocation failed");

        err = test_SVMMemcpy(srcMem.get(), dstMem.get());
        test_error(err, "test_SVMMemcpy");

        err = srcMem->free();
        test_error(err, "SVM free failed");
        err = dstMem->free();
        test_error(err, "SVM free failed");

        return CL_SUCCESS;
    }

    cl_int test_svm_memcpy(cl_uint TypeIndex)
    {
        cl_int err;
        const auto caps = deviceUSVMCaps[TypeIndex];

        auto mem = get_usvm_wrapper<cl_uchar>(TypeIndex);
        auto hostMem = get_hostptr_usvm_wrapper<cl_uchar>();

        err = mem->allocate(alloc_count);
        test_error(err, "SVM allocation failed");

        err = hostMem->allocate(alloc_count);
        test_error(err, "SVM allocation failed");

        // We check if the memory can be read by the host.
        if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR
            || caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
        {
            err = test_SVMMemcpy(mem.get(), hostMem.get());
            test_error(err, "test_SVMMemcpy");
        }

        // We check if the memory can be written by the host.
        if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR
            || caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
        {
            err = test_SVMMemcpy(hostMem.get(), mem.get());
            test_error(err, "test_SVMMemcpy");
        }

        err = mem->free();
        test_error(err, "SVM free failed");
        err = hostMem->free();
        test_error(err, "SVM free failed");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        // Test all possible comabinations between supported types
        for (cl_uint src_ti = 0; src_ti < max_ti; src_ti++)
        {
            for (cl_uint dst_ti = 0; dst_ti < max_ti; dst_ti++)
            {
                if (check_for_common_memory_type(src_ti, dst_ti))
                {
                    log_info(
                        "   testing clEnqueueSVMMemcpy() SVM type %u -> SVM "
                        "type %u\n",
                        src_ti, dst_ti);
                    err = test_svm_memcpy(src_ti, dst_ti);
                    if (CL_SUCCESS != err)
                    {
                        return err;
                    }
                }
            }
        }

        // For each supported svm type test copy from a host ptr and to a host
        // ptr
        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            log_info(
                "   testing clEnqueueSVMMemcpy() SVM type %u <-> host ptr \n",
                ti);
            err = test_svm_memcpy(ti);
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
                               CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR
                                   | CL_SVM_CAPABILITY_HOST_READ_KHR
                                   | CL_SVM_CAPABILITY_HOST_WRITE_KHR,
                               0, nullptr, nullptr, nullptr, nullptr));
    }

    bool check_for_common_memory_type(cl_uint srcTypeIndex,
                                      cl_uint dstTypeIndex)
    {

        const auto srcCaps = deviceUSVMCaps[srcTypeIndex];
        const auto dstCaps = deviceUSVMCaps[dstTypeIndex];

        // Is either allocation a system allocation
        if ((srcCaps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
            || (dstCaps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR))
        {
            return true;
        }

        // Is it possible to use the host
        if ((srcCaps & CL_SVM_CAPABILITY_HOST_READ_KHR)
            && (dstCaps & CL_SVM_CAPABILITY_HOST_WRITE_KHR))
        {
            return true;
        }

        // Is it posible to use the device
        if ((srcCaps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
            && (dstCaps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR))
        {
            return true;
        }

        return false;
    }

    static constexpr size_t alloc_count = 1024;
    static constexpr size_t test_iterations = 100;
};

REGISTER_TEST(unified_svm_memcpy)
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

    UnifiedSVMOPs Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
