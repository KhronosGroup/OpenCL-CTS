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
#include "harness/testHarness.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"

static const char* kernel_string_global_op = R"CLC(
kernel void test_inclusive_scopes(
    global int* scratch, global atomic_int* counter, global int* result)
{
    // every work-item writes its id (non-atomically)
    const int id = get_global_id(0);
    scratch[id] = id;

    // every work-item increments the counter
    // this is an release atomic with the producer scope
    atomic_fetch_add_explicit(
        counter, 1,
        memory_order_release,
        PRODUCER_SCOPE);

    // every work item checks if it is the last one
    // if it is, replace the counter with zero
    const int total = get_global_size(0);
    int count = total;
    atomic_compare_exchange_strong_explicit(
        counter, &count, 0,
        memory_order_acquire,
        memory_order_relaxed,
        CONSUMER_SCOPE);

    // the count will be:
    //  zero: this could have been the last work-item, but another work-item got in first
    //  something between zero and total: this is not the last work-item
    //  total: this is the last work-item

    // if this is the last work-item, sum everything up
    if (count == total)
    {
        int sum = 0;
        for (int i = 0; i < total; i++)
        {
            sum += scratch[i];
        }
        result[0] = sum;
    }
}
)CLC";

static const char* kernel_string_global_fence = R"CLC(
kernel void test_inclusive_scopes(
    global int* scratch, global atomic_int* counter, global int* result)
{
    // every work-item writes its id (non-atomically)
    const int id = get_global_id(0);
    scratch[id] = id;

    // every work-item increments the counter
    // this is an release atomic with the producer scope
    atomic_fetch_add_explicit(
        counter, 1,
        memory_order_relaxed,
        PRODUCER_SCOPE);
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE,
        memory_order_release,
        PRODUCER_SCOPE);

    // every work item checks if it is the last one
    // if it is, replace the counter with zero
    const int total = get_global_size(0);
    int count = total;
    atomic_work_item_fence(
        CLK_GLOBAL_MEM_FENCE,
        memory_order_acquire,
        CONSUMER_SCOPE);
    atomic_compare_exchange_strong_explicit(
        counter, &count, 0,
        memory_order_relaxed,
        memory_order_relaxed,
        CONSUMER_SCOPE);

    // the count will be:
    //  zero: this could have been the last work-item, but another work-item got in first
    //  something between zero and total: this is not the last work-item
    //  total: this is the last work-item

    // if this is the last work-item, sum everything up
    if (count == total)
    {
        int sum = 0;
        for (int i = 0; i < total; i++)
        {
            sum += scratch[i];
        }
        result[0] = sum;
    }
}
)CLC";

static std::string get_scope_name(int scope)
{
    switch (scope)
    {
        case 0: return "memory_scope_sub_group";
        case 1: return "memory_scope_work_group";
        case 2: return "memory_scope_device";
        case 3: return "memory_scope_all_svm_devices";
        default: break;
    }
    return "unknown_scope";
}

static bool supports_scope(cl_device_atomic_capabilities atomicCaps,
                           bool supportsSubGroups, int scope)
{
    switch (scope)
    {
        case 0: return supportsSubGroups;
        case 1: return (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP);
        case 2: return (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_DEVICE);
        case 3: return (atomicCaps & CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES);
        default: break;
    }
    return false;
}

static size_t get_local_work_size_for_scope(int producer_scope,
                                            int consumer_scope,
                                            cl_kernel kernel,
                                            cl_device_id device,
                                            int num_elements)
{
    cl_int error;
    size_t local_work_size = SIZE_MAX;

    const int minScope = std::min(producer_scope, consumer_scope);
    switch (minScope)
    {
        case 0: {
            const size_t one = 1;
            error = clGetKernelSubGroupInfo(
                kernel, device, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
                sizeof(one), &one, sizeof(local_work_size), &local_work_size,
                nullptr);
            test_error_ret(error, "clGetKernelSubGroupInfo failed", SIZE_MAX);
        }
        break;
        case 1: {
            error = get_max_allowed_1d_work_group_size_on_device(
                device, kernel, &local_work_size);
            test_error_ret(error, "Couldn't get max 1D local work size",
                           SIZE_MAX);
        }
        break;
        case 2: // device and all_devices scopes are handled the same
        case 3: {
            local_work_size = 0;
        }
        break;
    }

    local_work_size =
        std::min(local_work_size, static_cast<size_t>(num_elements));
    return local_work_size;
}

static int inclusive_scopes_helper(cl_device_id device, cl_context context,
                                   cl_command_queue queue, int num_elements,
                                   cl_device_atomic_capabilities testScopes,
                                   const char* kernel_string)
{
    cl_int error;

    cl_uint maxNumSubGroups = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                            sizeof(maxNumSubGroups), &maxNumSubGroups, nullptr);
    test_error(error, "clGetDeviceInfo failed to get max number of subgroups");

    const bool supportsSubGroups = maxNumSubGroups > 0;

    clMemWrapper scratchBuf =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       num_elements * sizeof(cl_int), nullptr, &error);
    test_error(error, "clCreateBuffer failed for scratch buffer");

    clMemWrapper counterBuf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             sizeof(cl_int), nullptr, &error);
    test_error(error, "clCreateBuffer failed for counter buffer");

    clMemWrapper resultBuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                            sizeof(cl_int), nullptr, &error);
    test_error(error, "clCreateBuffer failed for result buffer");

    for (int producer_scope = 0; producer_scope < 4; producer_scope++)
    {
        if (!supports_scope(testScopes, supportsSubGroups, producer_scope))
        {
            continue;
        }
        for (int consumer_scope = 0; consumer_scope < 4; consumer_scope++)
        {
            if (!supports_scope(testScopes, supportsSubGroups, consumer_scope))
            {
                continue;
            }

            log_info("    producer: %s, consumer: %s\n",
                     get_scope_name(producer_scope).c_str(),
                     get_scope_name(consumer_scope).c_str());

            std::string buildOptions;
            buildOptions +=
                " -DPRODUCER_SCOPE=" + get_scope_name(producer_scope);
            buildOptions +=
                " -DCONSUMER_SCOPE=" + get_scope_name(consumer_scope);

            clProgramWrapper program;
            clKernelWrapper kernel;
            error = create_single_kernel_helper(
                context, &program, &kernel, 1, &kernel_string,
                "test_inclusive_scopes", buildOptions.c_str());
            test_error(error, "could not create test kernel");

            const cl_int zero = 0;

            clEnqueueFillBuffer(queue, scratchBuf, &zero, sizeof(zero), 0,
                                num_elements * sizeof(cl_int), 0, nullptr,
                                nullptr);
            test_error(
                error,
                "clEnqueueFillBuffer failed to initialize scratch buffer");

            clEnqueueFillBuffer(queue, counterBuf, &zero, sizeof(zero), 0,
                                sizeof(cl_int), 0, nullptr, nullptr);
            test_error(
                error,
                "clEnqueueFillBuffer failed to initialize counter buffer");

            clEnqueueFillBuffer(queue, resultBuf, &zero, sizeof(zero), 0,
                                sizeof(cl_int), 0, nullptr, nullptr);
            test_error(
                error,
                "clEnqueueFillBuffer failed to initialize result buffer");

            error |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &scratchBuf);
            error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &counterBuf);
            error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuf);
            test_error(error, "clSetKernelArg failed");

            const size_t local_work_size = get_local_work_size_for_scope(
                producer_scope, consumer_scope, kernel, device, num_elements);
            const size_t global_work_size =
                local_work_size == 0 ? num_elements : local_work_size;

            error = clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_work_size,
                local_work_size == 0 ? nullptr : &local_work_size, 0, nullptr,
                nullptr);
            test_error(error, "clEnqueueNDRangeKernel failed");

            cl_int result = 0;
            error = clEnqueueReadBuffer(queue, resultBuf, CL_TRUE, 0,
                                        sizeof(result), &result, 0, nullptr,
                                        nullptr);
            test_error(error, "clEnqueueReadBuffer failed");

            const int expected = global_work_size * (global_work_size - 1) / 2;
            if (result != expected)
            {
                test_fail("test failed: expected %d, got %d\n", expected,
                          result);
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(inclusive_scopes_global_fence, Version(3, 1))
{
    cl_int error;

    cl_device_atomic_capabilities atomicFenceCaps = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_ATOMIC_FENCE_CAPABILITIES,
                            sizeof(atomicFenceCaps), &atomicFenceCaps, nullptr);
    test_error(error,
               "clGetDeviceInfo failed to get atomic fence capabilities");

    if ((atomicFenceCaps & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) == 0)
    {
        log_info("This test requires support for acquire-release atomic "
                 "fences, skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_device_atomic_capabilities atomicCaps = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                            sizeof(atomicCaps), &atomicCaps, nullptr);
    test_error(error, "clGetDeviceInfo failed to get atomic capabilities");

    // The scopes we can test are the intersection of the supported atomic fence
    // scopes and the supported atomic operation scopes.
    return inclusive_scopes_helper(device, context, queue, num_elements,
                                   atomicFenceCaps & atomicCaps,
                                   kernel_string_global_fence);
}

REGISTER_TEST_VERSION(inclusive_scopes_global_op, Version(3, 1))
{
    cl_int error;

    cl_device_atomic_capabilities atomicCaps = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES,
                            sizeof(atomicCaps), &atomicCaps, nullptr);
    test_error(error, "clGetDeviceInfo failed to get atomic capabilities");

    if ((atomicCaps & CL_DEVICE_ATOMIC_ORDER_ACQ_REL) == 0)
    {
        log_info("This test requires support for acquire-release atomics, "
                 "skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return inclusive_scopes_helper(device, context, queue, num_elements,
                                   atomicCaps, kernel_string_global_op);
}
