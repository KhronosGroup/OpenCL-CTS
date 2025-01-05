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

struct UnifiedSVMCapabilities : UnifiedSVMBase
{
    UnifiedSVMCapabilities(cl_context context, cl_device_id device,
                           cl_command_queue queue)
        : UnifiedSVMBase(context, device, queue)
    {}

    cl_int test_CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR(cl_uint typeIndex)
    {
        cl_int err;

        if (!kSINGLE_ADDRESS_SPACE)
        {
            const char* programString = R"(
                kernel void test_SINGLE_ADDRESS_SPACE(const global int* ptr, const global int* global* out)
                {
                    out[0] = ptr;
                }
            )";

            clProgramWrapper program;
            err = create_single_kernel_helper(
                context, &program, &kSINGLE_ADDRESS_SPACE, 1, &programString,
                "test_SINGLE_ADDRESS_SPACE");
            test_error(err, "could not create SINGLE_ADDRESS_SPACE kernel");
        }

        auto src = get_usvm_wrapper<int>(typeIndex);
        err = src->allocate(1);
        test_error(err, "could not allocate source memory");

        clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(int*), nullptr, &err);
        test_error(err, "could not create destination buffer");

        err |=
            clSetKernelArgSVMPointer(kSINGLE_ADDRESS_SPACE, 0, src->get_ptr());
        err |= clSetKernelArg(kSINGLE_ADDRESS_SPACE, 1, sizeof(out), &out);
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kSINGLE_ADDRESS_SPACE, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        int* out_ptr = nullptr;
        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(int*),
                                  &out_ptr, 0, nullptr, nullptr);
        test_error(err, "could not read output buffer");

        test_assert_error(out_ptr == src->get_ptr(),
                          "output pointer does not match input pointer");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        if (caps & CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR)
        {
            return CL_SUCCESS;
        }

        cl_int err;

        void* ptr;

        ptr = clSVMAllocWithPropertiesKHR(context, nullptr, typeIndex, 1, &err);
        test_error(err, "allocating without associated device failed");

        err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, ptr);
        test_error(err, "freeing without associated device failed");

        cl_svm_alloc_properties_khr props[] = {
            CL_SVM_ALLOC_ASSOCIATED_DEVICE_HANDLE_KHR,
            reinterpret_cast<cl_svm_alloc_properties_khr>(device), 0
        };
        ptr = clSVMAllocWithPropertiesKHR(context, props, typeIndex, 1, &err);
        test_error(err, "allocating with associated device failed");

        err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, ptr);
        test_error(err, "freeing with associated device failed");

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_READ_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        int value = 42;
        err = mem->write(value);
        test_error(err, "could not write to usvm memory");

        int check = mem->get_ptr()[0];
        test_assert_error(check == value, "read value does not match");

        if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
        {
            value = 31337;
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value,
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not write to usvm memory on the device");

            check = mem->get_ptr()[0];
            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_WRITE_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        int value = 42;
        mem->get_ptr()[0] = value;

        int check;
        err = mem->read(check);
        test_error(err, "could not read from usvm memory");
        test_assert_error(check == value, "read value does not match");

        if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
        {
            value = 31337;
            mem->get_ptr()[0] = value;

            err = clEnqueueSVMMemcpy(queue, CL_TRUE, &check, mem->get_ptr(),
                                     sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not read from usvm memory on the device");
            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int test_CL_SVM_CAPABILITY_HOST_MAP_KHR(cl_uint typeIndex)
    {
        const auto caps = deviceUSVMCaps[typeIndex];
        cl_int err;

        auto mem = get_usvm_wrapper<int>(typeIndex);
        err = mem->allocate(1);
        test_error(err, "could not allocate usvm memory");

        // map for writing, then map for reading
        int value = 0xCA7;
        err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                              mem->get_ptr(), sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not map usvm memory for writing");

        mem->get_ptr()[0] = value;
        err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
        test_error(err, "could not unmap usvm memory");

        err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                              sizeof(value), 0, nullptr, nullptr);
        test_error(err, "could not map usvm memory for reading");

        int check = mem->get_ptr()[0];
        err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
        test_error(err, "could not unmap usvm memory");

        test_assert_error(check == value, "read value does not match");

        // write directly on the host, map for reading on the host
        if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR)
        {
            value = 42;
            mem->get_ptr()[0] = value;

            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                                  sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for reading");

            check = mem->get_ptr()[0];
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            test_assert_error(check == value, "read value does not match");
        }

        // map for writing on the host, read directly on the host
        if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR)
        {
            value = 777;
            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                  mem->get_ptr(), sizeof(value), 0, nullptr,
                                  nullptr);
            test_error(err, "could not map usvm memory for writing");

            mem->get_ptr()[0] = value;
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            check = mem->get_ptr()[0];
            test_assert_error(check == value, "read value does not match");
        }

        // write on the device, map for reading on the host
        if (caps & CL_SVM_CAPABILITY_DEVICE_WRITE_KHR)
        {
            value = 31337;
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value,
                                    sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not write to usvm memory on the device");

            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, mem->get_ptr(),
                                  sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not map usvm memory for reading");

            check = mem->get_ptr()[0];
            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            test_assert_error(check == value, "read value does not match");
        }

        // map for writing on the host, read on the device
        if (caps & CL_SVM_CAPABILITY_DEVICE_READ_KHR)
        {
            int value = 0xF00D;
            err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                                  mem->get_ptr(), sizeof(value), 0, nullptr,
                                  nullptr);
            test_error(err, "could not map usvm memory for writing");

            mem->get_ptr()[0] = value;

            err = clEnqueueSVMUnmap(queue, mem->get_ptr(), 0, nullptr, nullptr);
            test_error(err, "could not unmap usvm memory");

            int check;
            err = clEnqueueSVMMemcpy(queue, CL_TRUE, &check, mem->get_ptr(),
                                    sizeof(value), 0, nullptr, nullptr);
            test_error(err, "could not read from usvm memory on the device");

            test_assert_error(check == value, "read value does not match");
        }

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            const auto caps = deviceUSVMCaps[ti];
            log_info("   testing SVM type %u, capabilities 0x%08" PRIx64 "\n",
                     ti, caps);

            if (caps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR)
            {
                err = test_CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR(ti);
                test_error(err,
                           "CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR "
                           "failed");
            }
            // CL_SVM_CAPABILITY_SYSTEM_ALLOCATED_KHR
            // CL_SVM_CAPABILITY_DEVICE_OWNED_KHR
            if (caps & CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR)
            {
                err = test_CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR(ti);
                test_error(err,
                           "CL_SVM_CAPABILITY_DEVICE_UNASSOCIATED_KHR failed");
            }
            // CL_SVM_CAPABILITY_CONTEXT_ACCESS_KHR
            // CL_SVM_CAPABILITY_HOST_OWNED_KHR
            if (caps & CL_SVM_CAPABILITY_HOST_READ_KHR)
            {
                err = test_CL_SVM_CAPABILITY_HOST_READ_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_READ_KHR failed");
            }
            if (caps & CL_SVM_CAPABILITY_HOST_WRITE_KHR)
            {
                err = test_CL_SVM_CAPABILITY_HOST_WRITE_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_WRITE_KHR failed");
            }
            if (caps & CL_SVM_CAPABILITY_HOST_MAP_KHR)
            {
                err = test_CL_SVM_CAPABILITY_HOST_MAP_KHR(ti);
                test_error(err, "CL_SVM_CAPABILITY_HOST_MAP_KHR failed");
            }
            // CL_SVM_CAPABILITY_DEVICE_READ_KHR
            // CL_SVM_CAPABILITY_DEVICE_WRITE_KHR
            // CL_SVM_CAPABILITY_DEVICE_ATOMIC_ACCESS_KHR
            // CL_SVM_CAPABILITY_CONCURRENT_ACCESS_KHR
            // CL_SVM_CAPABILITY_CONCURRENT_ATOMIC_ACCESS_KHR
            // CL_SVM_CAPABILITY_INDIRECT_ACCESS_KHR
        }
        return CL_SUCCESS;
    }

    clKernelWrapper kSINGLE_ADDRESS_SPACE;
};

REGISTER_TEST(unified_svm_capabilities)
{
    if (!is_extension_available(deviceID, "cl_khr_unified_svm"))
    {
        log_info("cl_khr_unified_svm is not supported, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int err;

    // For now: create a new context and queue.
    // If we switch to a new test executable and run the tests without
    // forceNoContextCreation then this can be removed, and we can just use the
    // context and the queue from the harness.
    if (context == nullptr)
    {
        context =
            clCreateContext(nullptr, 1, &deviceID, nullptr, nullptr, &err);
        test_error(err, "clCreateContext failed");
    }

    if (queue == nullptr)
    {
        queue = clCreateCommandQueue(context, deviceID, 0, &err);
        test_error(err, "clCreateCommandQueue failed");
    }

    UnifiedSVMCapabilities Test(context, deviceID, queue);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
