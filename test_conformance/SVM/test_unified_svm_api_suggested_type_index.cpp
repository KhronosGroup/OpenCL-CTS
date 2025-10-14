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

struct UnifiedSVMAPISuggestedTypeIndex : UnifiedSVMBase
{
    UnifiedSVMAPISuggestedTypeIndex(cl_context context, cl_device_id device,
                                    cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int run() override
    {
        constexpr size_t size = 16 * 1024;

        cl_int err = CL_SUCCESS;

        // Get the suggested type index for each set of capabilities supported
        // by the device, and build a set of all capabilities supported by the
        // device.
        cl_svm_capabilities_khr allSupportedDeviceUSVMCaps = 0;
        for (const auto caps : deviceUSVMCaps)
        {
            if (caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR)
            {
                // The system allocator pseudo-capability is not a real
                // capability, so skip it.
                continue;
            }

            err = checkSuggestedTypeIndex(caps, size);
            test_error(err, "suggested type index failed");

            allSupportedDeviceUSVMCaps |= caps;
        }

        // Get the suggested type index for each supported capability
        // individually.
        for (cl_uint bit = 0; bit < sizeof(cl_svm_capabilities_khr) * 8; bit++)
        {
            cl_svm_capabilities_khr testCap =
                static_cast<cl_svm_capabilities_khr>(1 << bit);
            if (allSupportedDeviceUSVMCaps & testCap)
            {
                err = checkSuggestedTypeIndex(testCap, size);
                test_error(err, "suggested type index failed");
            }
        }

        // Build a set of all capabilities supported by the platform.
        cl_svm_capabilities_khr allSupportedPlatformUSVMCaps = 0;
        for (const auto caps : platformUSVMCaps)
        {
            allSupportedPlatformUSVMCaps |= caps;
        }

        // Check that the suggested type index for an unsupported capability is
        // CL_UINT_MAX.
        if (~allSupportedPlatformUSVMCaps != 0)
        {
            cl_uint suggested = ~0;
            err = clGetSVMSuggestedTypeIndexKHR(context,
                                                ~allSupportedPlatformUSVMCaps,
                                                0, nullptr, size, &suggested);
            test_error(err, "suggested type index failed");
            test_assert_error_ret(suggested == CL_UINT_MAX,
                                  "suggested type index for unsupported "
                                  "capability is not CL_UINT_MAX",
                                  CL_INVALID_VALUE);
        }

        return CL_SUCCESS;
    }

    cl_int checkSuggestedTypeIndex(cl_svm_capabilities_khr requiredCaps,
                                   size_t size)
    {
        cl_int err;
        cl_uint suggested = ~0;

        // Test without an associated device handle.
        // This must return platform SVM capabilities with the required
        // capability, but it may return unsupported capabilities for the
        // device.
        err = clGetSVMSuggestedTypeIndexKHR(context, requiredCaps, 0, nullptr,
                                            size, &suggested);
        test_error(err, "clGetSVMSuggestedTypeIndexKHR failed");
        test_assert_error_ret(suggested < deviceUSVMCaps.size(),
                              "suggested type index is out of range",
                              CL_INVALID_VALUE);
        if (deviceUSVMCaps[suggested] != 0)
        {
            test_assert_error_ret(deviceUSVMCaps[suggested] & requiredCaps,
                                  "suggested type index does not have the "
                                  "required device capability",
                                  CL_INVALID_VALUE);
        }
        else
        {
            test_assert_error_ret(platformUSVMCaps[suggested] & requiredCaps,
                                  "suggested type index does not have the "
                                  "required platform capability",
                                  CL_INVALID_VALUE);
        }

        // Test with an associated device handle.
        // This must return device SVM capabilities with the required
        // capability.
        std::vector<cl_svm_alloc_properties_khr> props;
        props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
        props.push_back(reinterpret_cast<cl_svm_alloc_properties_khr>(device));
        props.push_back(0);
        err = clGetSVMSuggestedTypeIndexKHR(context, requiredCaps, 0,
                                            props.data(), size, &suggested);
        test_error(err, "clGetSVMSuggestedTypeIndexKHR failed");
        test_assert_error_ret(suggested < deviceUSVMCaps.size(),
                              "suggested type index is out of range",
                              CL_INVALID_VALUE);
        test_assert_error_ret(deviceUSVMCaps[suggested] & requiredCaps,
                              "suggested type index does not have the "
                              "required device capability",
                              CL_INVALID_VALUE);

        // Test with all properties - an associated device handle, an
        // alignment, and access flags.
        props.clear();
        props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
        props.push_back(reinterpret_cast<cl_svm_alloc_properties_khr>(device));
        props.push_back(CL_SVM_ALLOC_ACCESS_FLAGS_KHR);
        props.push_back(0);
        props.push_back(CL_SVM_ALLOC_ALIGNMENT_KHR);
        props.push_back(0);
        props.push_back(0);
        err = clGetSVMSuggestedTypeIndexKHR(context, requiredCaps, 0,
                                            props.data(), size, &suggested);
        test_error(err, "clGetSVMSuggestedTypeIndexKHR failed");
        test_assert_error_ret(suggested < deviceUSVMCaps.size(),
                              "suggested type index is out of range",
                              CL_INVALID_VALUE);
        test_assert_error_ret(deviceUSVMCaps[suggested] & requiredCaps,
                              "suggested type index does not have the "
                              "required device capability",
                              CL_INVALID_VALUE);

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_api_suggested_type_index)
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

    UnifiedSVMAPISuggestedTypeIndex Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
