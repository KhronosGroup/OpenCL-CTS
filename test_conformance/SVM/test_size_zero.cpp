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
#include "common.h"

static int svm_size_zero_helper(cl_device_id device, cl_context context,
                                cl_command_queue queue, cl_kernel kernel,
                                cl_svm_mem_flags svmFlags)
{
    cl_int error = CL_SUCCESS;
    cl_int value = 42;

    clEventWrapper event;

    // Try allocating zero bytes of SVM.
    // This should not crash.
    {
        void* svmZeroPtr =
            clSVMAlloc(context, svmFlags, 0, svmFlags | CL_MEM_READ_WRITE);

        // We should be able to free whatever we allocated.
        clSVMFree(context, svmZeroPtr);

        // Try freeing an explicit NULL pointer.
        // This should not crash.
        clSVMFree(context, nullptr);
    }

    // Try to call clEnqueueSVMFree with an empty set
    event = nullptr; // Reset the event
    error = clEnqueueSVMFree(queue, 0, nullptr, nullptr, nullptr, 0, nullptr,
                             &event);
    test_error(error, "clEnqueueSVMFree with an empty set failed");

    error = check_event_type(event, CL_COMMAND_SVM_FREE);
    test_error(error,
               "clEnqueueSVMFree with an empty set did not return a "
               "CL_COMMAND_SVM_FREE event");

    // Try to call clEnqueueSVMFree with an explicit NULL pointer.
    {
        void* svm_pointers[] = { nullptr };
        event = nullptr; // Reset the event
        error = clEnqueueSVMFree(queue, 1, svm_pointers, nullptr, nullptr, 0,
                                 nullptr, &event);
        test_error(error, "clEnqueueSVMFree with NULL pointer failed");

        error = check_event_type(event, CL_COMMAND_SVM_FREE);
        test_error(error,
                   "clEnqueueSVMFree with a NULL pointer did not return a "
                   "CL_COMMAND_SVM_FREE event");
    }

    clSVMWrapper svmPtr(context, sizeof(value), svmFlags | CL_MEM_READ_WRITE);
    test_assert_error(svmPtr() != nullptr, "clSVMAlloc failed");

    // Try filling zero bytes of the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemFill(queue, svmPtr(), &value, sizeof(value), 0, 0,
                                nullptr, &event);
    test_error(error, "clEnqueueSVMMemFill with zero size failed");

    error = check_event_type(event, CL_COMMAND_SVM_MEMFILL);
    test_error(error,
               "clEnqueueSVMMemFill with zero size did not return a "
               "CL_COMMAND_SVM_MEMFILL event");

    // Try copying zero bytes to the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemcpy(queue, CL_TRUE, svmPtr(), &value, 0, 0, nullptr,
                               &event);
    test_error(error, "clEnqueueSVMMemcpy to SVM with zero size failed");

    error = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
    test_error(error,
               "clEnqueueSVMMemcpy to SVM with zero size did not return a "
               "CL_COMMAND_SVM_MEMCPY event");

    // Try copying zero bytes from the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemcpy(queue, CL_TRUE, &value, svmPtr(), 0, 0, nullptr,
                               &event);
    test_error(error, "clEnqueueSVMMemcpy from SVM with zero size failed");

    error = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
    test_error(error,
               "clEnqueueSVMMemcpy from SVM with zero size did not return a "
               "CL_COMMAND_SVM_MEMCPY event");

    // Try migrating zero bytes of the SVM pointer
    {
        const void* svm_pointers[] = { svmPtr() };
        const size_t sizes[] = { 0 };
        event = nullptr; // Reset the event
        error = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                       CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                       0, nullptr, &event);
        test_error(error, "clEnqueueSVMMigrateMem with zero size failed");

        error = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
        test_error(error,
                   "clEnqueueSVMMigrateMem with zero size did not return a "
                   "CL_COMMAND_SVM_MIGRATE_MEM event");
    }

    // Try migrating zero bytes and a NULL pointer
    {
        const void* svm_pointers[] = { nullptr };
        const size_t sizes[] = { 0 };
        event = nullptr; // Reset the event
        error = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                       CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                       0, nullptr, &event);
        test_error(error, "clEnqueueSVMMigrateMem with NULL pointer failed");

        error = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
        test_error(error,
                   "clEnqueueSVMMigrateMem with NULL pointer did not return a "
                   "CL_COMMAND_SVM_MIGRATE_MEM event");
    }

    // Try migrating a NULL pointer with NULL sizes
    {
        const void* svm_pointers[] = { nullptr };
        event = nullptr; // Reset the event
        error = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, nullptr,
                                       CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                       0, nullptr, &event);
        test_error(
            error,
            "clEnqueueSVMMigrateMem with NULL pointer and NULL sizes failed");

        error = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
        test_error(error,
                   "clEnqueueSVMMigrateMem with NULL pointer and NULL sizes "
                   "did not return a CL_COMMAND_SVM_MIGRATE_MEM event");
    }

    // Try to call clSetKernelExecInfo with an empty set
    error =
        clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, nullptr);
    test_error(error, "clSetKernelExecInfo with an empty set failed");

    // Try to call clSetKernelExecInfo with an explicit NULL pointer
    {
        void* svm_pointers[] = { nullptr };
        error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                    sizeof(svm_pointers), svm_pointers);
        test_error(error, "clSetKernelExecInfo with NULL pointer failed");
    }

    return TEST_PASS;
}

REGISTER_TEST(svm_size_zero)
{
    // Note: These are SVM tests, not unified SVM tests, however the scenarios
    // they are testing are ambiguous pre-unified SVM. Therefore, we will only
    // run these tests if the device supports unified SVM.
    REQUIRE_EXTENSION("cl_khr_unified_svm");

    cl_int error;

    // Common test setup - context, queue, shared kernel and output buffer.

    clContextWrapper contextWrapper;
    clCommandQueueWrapper queueWrapper;

    if (context == nullptr)
    {
        contextWrapper =
            clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        test_error(error, "clCreateContext failed");
        context = contextWrapper;
    }

    if (queue == nullptr)
    {
        queueWrapper = clCreateCommandQueue(context, device, 0, &error);
        test_error(error, "clCreateCommandQueue failed");
        queue = queueWrapper;
    }

    cl_device_svm_capabilities svmCaps = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svmCaps),
                            &svmCaps, nullptr);
    test_error(error, "clGetDeviceInfo failed to get SVM capabilities");

    clProgramWrapper program;
    clKernelWrapper kernel;
    const char* programString = R"(
        kernel void test_OneArg(global int* ptr) { ptr[0] = 0; }
    )";
    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &programString, "test_OneArg");
    test_error(error, "could not create test kernel");

    int result = TEST_PASS;

    if (svmCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
    {
        log_info("    testing coarse-grain SVM\n");
        result |= svm_size_zero_helper(device, context, queue, kernel, 0);
    }
    if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
    {
        log_info("    testing fine-grain SVM\n");
        result |= svm_size_zero_helper(device, context, queue, kernel,
                                       CL_MEM_SVM_FINE_GRAIN_BUFFER);
    }
    if (svmCaps & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS))
    {
        log_info("    testing fine-grain SVM with atomics\n");
        result |= svm_size_zero_helper(device, context, queue, kernel,
                                       CL_MEM_SVM_FINE_GRAIN_BUFFER
                                           | CL_MEM_SVM_ATOMICS);
    }

    return result;
}
