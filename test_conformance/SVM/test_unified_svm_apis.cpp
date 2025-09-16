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

struct UnifiedSVMAPIs : UnifiedSVMBase
{
    UnifiedSVMAPIs(cl_context context, cl_device_id device,
                   cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_alloc_query_free(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);

        // Basic allocation with no special properties:
        // Notes:
        //  * The USVM wrapper will add an associated device handle
        //  automatically, unless the CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR
        //  capability is supported.
        //  * The USVM wrapper will also pass nullptr as the property list, if
        //  no properties are needed.
        {
            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, {});
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test with an explicit associated device handle property, for devices
        // that support CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR, as long as
        // this is not a system allocated type.
        if (caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR
            && !(caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR))
        {
            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
            props.push_back(
                reinterpret_cast<cl_svm_alloc_properties_khr>(device));
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test with an explicit nullptr device handle property, for devices
        // that support CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR, as long as
        // this is not a system allocated type.
        // !!! Check: Is this a valid test?
        if (caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR
            && !(caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR))
        {
            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
            props.push_back(
                reinterpret_cast<cl_svm_alloc_properties_khr>(nullptr));
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Also test an empty property list, for devices that support
        // CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR, as long as this is not a
        // system allocated type.
        if (caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR
            && !(caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR))
        {
            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test allocation with an explicit alignment, from 0-128 inclusive
        for (size_t alignment = 0; alignment <= deviceMaxAlignment;
             alignment = alignment ? alignment * 2 : 1)
        {
            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(CL_SVM_ALLOC_ALIGNMENT_KHR);
            props.push_back(alignment);
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), alignment);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test all combinations of SVM allocation access flags
        for (size_t access = 0; access < 16; access++)
        {
            cl_svm_alloc_access_flags_khr flags = 0;
            flags |= (access & 1) ? CL_SVM_ALLOC_ACCESS_HOST_NOREAD_KHR : 0;
            flags |= (access & 2) ? CL_SVM_ALLOC_ACCESS_HOST_NOWRITE_KHR : 0;
            flags |= (access & 4) ? CL_SVM_ALLOC_ACCESS_DEVICE_NOREAD_KHR : 0;
            flags |= (access & 8) ? CL_SVM_ALLOC_ACCESS_DEVICE_NOWRITE_KHR : 0;

            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(CL_SVM_ALLOC_ACCESS_FLAGS_KHR);
            props.push_back(flags);
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        // Test with all properties
        {
            std::vector<cl_svm_alloc_properties_khr> props;
            props.push_back(CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR);
            props.push_back(
                reinterpret_cast<cl_svm_alloc_properties_khr>(device));
            props.push_back(CL_SVM_ALLOC_ACCESS_FLAGS_KHR);
            props.push_back(0);
            props.push_back(CL_SVM_ALLOC_ALIGNMENT_KHR);
            props.push_back(0);
            props.push_back(0);

            err = mem->allocate(alloc_count, props);
            test_error(err, "SVM allocation failed");

            err = checkAlignment(mem->get_ptr(), 0);
            test_error(err, "alignment check failed");

            err = checkQueries(mem->get_ptr(), typeIndex, props);
            test_error(err, "queries failed");

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            log_info("     testing allocation, queries, and frees\n");
            err = test_alloc_query_free(ti);
            test_error(err, "allocation, queries, and frees failed");
        }
        return CL_SUCCESS;
    }

    cl_int checkAlignment(void* ptr, size_t alignment)
    {
        alignment = alignment == 0 ? deviceMaxAlignment : alignment;
        if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0)
        {
            log_error("pointer %p is not aligned to %zu bytes\n", ptr,
                      alignment);
            return CL_INVALID_VALUE;
        }
        return CL_SUCCESS;
    }

    cl_int checkQueries(const void* ptr, cl_uint typeIndex,
                        const std::vector<cl_svm_alloc_properties_khr>& props)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        // We cannot test queries for system allocated memory.
        if (caps & PSEUDO_CAPABILITY_USE_SYSTEM_ALLOCATOR)
        {
            return CL_SUCCESS;
        }

        // Check queries with the base pointer and an explicit device
        err = checkQueriesHelper(ptr, 0, typeIndex, device, props);
        test_error(err, "SVM queries failed for base pointer");

        // Check queries with an offset pointer and an explicit device
        err = checkQueriesHelper(ptr, 4, typeIndex, device, props);
        test_error(err, "SVM queries failed for offset pointer");

        // Check queries with the base pointer and a nullptr device
        err = checkQueriesHelper(ptr, 0, typeIndex, nullptr, props);
        test_error(err,
                   "SVM queries failed for base pointer with nullptr device");

        // Check queries with an offset pointer and a nullptr device
        err = checkQueriesHelper(ptr, 4, typeIndex, nullptr, props);
        test_error(err,
                   "SVM queries failed for offset pointer with nullptr device");

        return CL_SUCCESS;
    }

    cl_int
    checkQueriesHelper(const void* base, size_t offset, cl_uint typeIndex,
                       cl_device_id queryDevice,
                       const std::vector<cl_svm_alloc_properties_khr>& props)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        const void* query_ptr =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset);
        cl_int err;

        // Note: the passed-in properties may not include the associated device
        // handle, since this is added automatically by the USVMWrapper.

        cl_device_id associatedDevice = nullptr;
        cl_svm_alloc_access_flags_khr accessFlags = 0;
        size_t alignment = 0;
        parseSVMAllocProperties(props, associatedDevice, accessFlags,
                                alignment);

        if (!(caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR))
        {
            associatedDevice = device;
        }

        cl_uint typeIndexQuery = ~0;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_TYPE_INDEX_KHR,
            sizeof(typeIndexQuery), &typeIndexQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_TYPE_INDEX_KHR");
        test_assert_error_ret(typeIndexQuery == typeIndex,
                              "type index does not match", CL_INVALID_VALUE);

        cl_svm_capabilities_khr capabilitiesQuery;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_CAPABILITIES_KHR,
            sizeof(capabilitiesQuery), &capabilitiesQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_CAPABILITIES_KHR");
        if (queryDevice)
        {
            test_assert_error_ret(capabilitiesQuery == caps,
                                  "capabilities do not match",
                                  CL_INVALID_VALUE);
        }
        else
        {
            const auto checkCaps = platformUSVMCaps[typeIndex];
            bool isSuperset = (capabilitiesQuery & checkCaps) == checkCaps;
            test_assert_error_ret(isSuperset,
                                  "capabilities are insufficient    ",
                                  CL_INVALID_VALUE);
        }

        cl_svm_alloc_access_flags_khr accessFlagsQuery;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_ACCESS_FLAGS_KHR,
            sizeof(accessFlagsQuery), &accessFlagsQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for CL_SVM_INFO_ACCESS_FLAGS_KHR");
        test_assert_error_ret(accessFlagsQuery == accessFlags,
                              "access flags do not match", CL_INVALID_VALUE);

        void* basePtrQuery;
        err = clGetSVMPointerInfoKHR(
            context, queryDevice, query_ptr, CL_SVM_INFO_BASE_PTR_KHR,
            sizeof(basePtrQuery), &basePtrQuery, nullptr);
        test_error(err, "clGetSVMPointerInfoKHR for CL_SVM_INFO_BASE_PTR_KHR");
        test_assert_error_ret(basePtrQuery == base,
                              "base pointer does not match", CL_INVALID_VALUE);

        size_t sizeQuery;
        err = clGetSVMPointerInfoKHR(context, queryDevice, query_ptr,
                                     CL_SVM_INFO_SIZE_KHR, sizeof(sizeQuery),
                                     &sizeQuery, nullptr);
        test_error(err, "clGetSVMPointerInfoKHR for CL_SVM_INFO_SIZE_KHR");
        test_assert_error_ret(sizeQuery == alloc_count * sizeof(cl_int),
                              "size does not match", CL_INVALID_VALUE);

        cl_device_id associatedDeviceQuery;
        err = clGetSVMPointerInfoKHR(context, queryDevice, query_ptr,
                                     CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR,
                                     sizeof(associatedDeviceQuery),
                                     &associatedDeviceQuery, nullptr);
        test_error(err,
                   "clGetSVMPointerInfoKHR for "
                   "CL_SVM_INFO_ASSOCIATED_DEVICE_HANDLE_KHR");
        test_assert_error_ret(associatedDeviceQuery == associatedDevice,
                              "associated device handle does not match",
                              CL_INVALID_VALUE);

        return CL_SUCCESS;
    }

    static constexpr size_t alloc_count = 16;
};

REGISTER_TEST(unified_svm_apis)
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

    UnifiedSVMAPIs Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
