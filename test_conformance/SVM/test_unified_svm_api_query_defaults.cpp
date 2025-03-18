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
#include <cinttypes>
#include <memory>

struct UnifiedSVMAPIQueryDefaults : UnifiedSVMBase
{
    UnifiedSVMAPIQueryDefaults(cl_context context, cl_device_id device,
                               cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_query_defaults(cl_device_id queryDevice)
    {
        cl_int err = CL_SUCCESS;
        const void* query_ptr = &err; // a random non-USVM pointer

        cl_uint typeIndexQuery = 0;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_TYPE_INDEX_KHR,
            sizeof(typeIndexQuery), &typeIndexQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_TYPE_INDEX_KHR");
        test_assert_error_ret(typeIndexQuery == CL_UINT_MAX,
                              "type index is not the default",
                              CL_INVALID_VALUE);

        cl_svm_capabilities_khr capabilitiesQuery = ~0;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_CAPABILITIES_KHR,
            sizeof(capabilitiesQuery), &capabilitiesQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_CAPABILITIES_KHR");
        test_assert_error_ret(capabilitiesQuery == 0,
                              "capabilities are not the default",
                              CL_INVALID_VALUE);

        cl_svm_alloc_access_flags_khr accessFlagsQuery = ~0;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_ACCESS_FLAGS_KHR,
            sizeof(accessFlagsQuery), &accessFlagsQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_ACCESS_FLAGS_KHR");
        test_assert_error_ret(accessFlagsQuery == 0,
                              "access flags are not the default",
                              CL_INVALID_VALUE);

        void* basePtrQuery = &err;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_BASE_PTR_KHR,
            sizeof(basePtrQuery), &basePtrQuery, nullptr);
        test_error(err, "clGetSVMPointerInfoKHR for CL_SVM_INFO_BASE_PTR_KHR");
        test_assert_error_ret(basePtrQuery == nullptr,
                              "base pointer is not the default",
                              CL_INVALID_VALUE);

        size_t sizeQuery = ~0;
        err = clGetSVMPointerInfoKHR(context, queryDevice, query_ptr,
                                     CL_SVM_INFO_SIZE_KHR, sizeof(sizeQuery),
                                     &sizeQuery, nullptr);
        test_error(err, "clGetSVMPointerInfoKHR for CL_SVM_INFO_SIZE_KHR");
        test_assert_error_ret(sizeQuery == 0, "size is not the default",
                              CL_INVALID_VALUE);

        cl_device_id associatedDeviceQuery = device;
        err = clGetSVMPointerInfoKHR(context, queryDevice, query_ptr,
                                     CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR,
                                     sizeof(associatedDeviceQuery),
                                     &associatedDeviceQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for "
                   "CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR");
        test_assert_error_ret(associatedDeviceQuery == nullptr,
                              "associated device handle is not the default",
                              CL_INVALID_VALUE);

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing query defaults with no device\n");
        err = test_query_defaults(nullptr);
        test_error(err, "query defaults failed");

        log_info("   testing query defaults with a device\n");
        err = test_query_defaults(device);
        test_error(err, "query defaults failed");

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_api_query_defaults)
{
    REQUIRE_EXTENSION("cl_khr_unified_svm");

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

    UnifiedSVMAPIQueryDefaults Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
