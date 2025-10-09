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
    cl_command_type cmdType = 0;

    // Try allocating zero bytes of SVM.
    // This should not crash.
    void* svmZeroPtr =
        clSVMAlloc(context, svmFlags, 0, svmFlags | CL_MEM_READ_WRITE);
    log_info("      allocating zero bytes returned %p\n", svmZeroPtr);

    // We should be able to free whatever we allocated.
    clSVMFree(context, svmZeroPtr);

    // Try freeing an explicit NULL pointer.
    // This should not crash.
    clSVMFree(context, nullptr);

    // Try to call clEnqueueSVMFree with an empty set
    event = nullptr; // Reset the event
    error = clEnqueueSVMFree(queue, 0, nullptr, nullptr, nullptr, 0, nullptr,
                             &event);
    // test_error(error, "clEnqueueSVMFree with an empty set failed");
    log_info("      clEnqueueSVMFree with an empty set returned %s\n",
             IGetErrorString(error));

    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(
        //     cmdType == CL_COMMAND_SVM_FREE,
        //     "Unexpected command type for clEnqueueSVMFree with an empty
        //     set");
        log_info("      clEnqueueSVMFree with an empty set has command type "
                 "%4X (%4X)\n",
                 cmdType, CL_COMMAND_SVM_FREE);
    }

    // Try to call clEnqueueSVMFree with an explicit NULL pointer.
    {
        void* svm_pointers[] = { nullptr };
        event = nullptr; // Reset the event
        error = clEnqueueSVMFree(queue, 1, svm_pointers, nullptr, nullptr, 0,
                                 nullptr, &event);
        // test_error(error, "clEnqueueSVMFree with NULL pointer failed");
        log_info("      clEnqueueSVMFree with NULL pointer returned %s\n",
                 IGetErrorString(error));
    }

    // Check that the event for clEnqueueSVMFree with an explicit NULL pointer
    // has the right command type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(
        //     cmdType == CL_COMMAND_SVM_FREE,
        //     "Unexpected command type for clEnqueueSVMFree with NULL
        //     pointer");
        log_info(
            "      clEnqueueSVMFree with NULL pointer has command type %4X "
            "(%4X)\n",
            cmdType, CL_COMMAND_SVM_FREE);
    }

    clSVMWrapper svmPtr(context, sizeof(value), svmFlags | CL_MEM_READ_WRITE);
    test_assert_error(svmPtr() != nullptr, "clSVMAlloc failed");

    // Try filling zero bytes of the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemFill(queue, svmPtr(), &value, sizeof(value), 0, 0,
                                nullptr, &event);
    // test_error(error, "clEnqueueSVMMemFill with zero size failed");
    log_info("      clEnqueueSVMMemFill with zero size returned %s\n",
             IGetErrorString(error));

    // Check that the event for the zero-sized fill has the right command type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(
        //     cmdType == CL_COMMAND_SVM_MEMFILL,
        //     "Unexpected command type for clEnqueueSVMMemFill with zero
        //     size");
        log_info(
            "      clEnqueueSVMMemFill with zero size has command type %4X "
            "(%4X)\n",
            cmdType, CL_COMMAND_SVM_MEMFILL);
    }

    // Try copying zero bytes to the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemcpy(queue, CL_TRUE, svmPtr(), &value, 0, 0, nullptr,
                               &event);
    // test_error(error, "clEnqueueSVMMemcpy to SVM with zero size failed");
    log_info("      clEnqueueSVMMemcpy to SVM with zero size returned %s\n",
             IGetErrorString(error));

    // Check that the event for the zero-sized copy has the right command type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(
        //     cmdType == CL_COMMAND_SVM_MEMCPY,
        //     "Unexpected command type for clEnqueueSVMMemcpy with zero size");
        log_info("      clEnqueueSVMMemcpy to SVM with zero size has command "
                 "type %4X (%4X)\n",
                 cmdType, CL_COMMAND_SVM_MEMCPY);
    }

    // Try copying zero bytes from the SVM pointer
    event = nullptr; // Reset the event
    error = clEnqueueSVMMemcpy(queue, CL_TRUE, &value, svmPtr(), 0, 0, nullptr,
                               &event);
    // test_error(error, "clEnqueueSVMMemcpy from SVM with zero size failed");
    log_info("      clEnqueueSVMMemcpy from SVM with zero size returned %s\n",
             IGetErrorString(error));

    // Check that the event for the zero-sized copy has the right command type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(
        //     cmdType == CL_COMMAND_SVM_MEMCPY,
        //     "Unexpected command type for clEnqueueSVMMemcpy with zero size");
        log_info("      clEnqueueSVMMemcpy from SVM with zero size has command "
                 "type %4X (%4X)\n",
                 cmdType, CL_COMMAND_SVM_MEMCPY);
    }

    // Try migrating zero bytes of the SVM pointer
    {
        const void* svm_pointers[] = { svmPtr() };
        const size_t sizes[] = { 0 };
        event = nullptr; // Reset the event
        error = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                       CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                       0, nullptr, &event);
        // test_error(error, "clEnqueueSVMMigrateMem with zero size failed");
        log_info("      clEnqueueSVMMigrateMem with zero size returned %s\n",
                 IGetErrorString(error));
    }

    // Check that the event for the zero-sized migration has the right command
    // type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(cmdType == CL_COMMAND_SVM_MIGRATE_MEM,
        //                   "Unexpected command type for clEnqueueSVMMigrateMem
        //                   " "with zero size");
        log_info("      clEnqueueSVMMigrateMem with zero size has command type "
                 "%4X (%4X)\n",
                 cmdType, CL_COMMAND_SVM_MIGRATE_MEM);
    }

    // Try migrating zero bytes and a NULL pointer
    {
        const void* svm_pointers[] = { nullptr };
        const size_t sizes[] = { 0 };
        event = nullptr; // Reset the event
        error = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                       CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                       0, nullptr, &event);
        // test_error(error, "clEnqueueSVMMigrateMem with NULL pointer failed");
        log_info("      clEnqueueSVMMigrateMem with NULL pointer returned %s\n",
                 IGetErrorString(error));
    }

    // Check that the event for the NULL pointer migration has the right command
    // type
    if (error == CL_SUCCESS)
    {
        error = clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cmdType),
                               &cmdType, nullptr);
        test_error(error, "clGetEventInfo failed for CL_EVENT_COMMAND_TYPE");
        // test_assert_error(cmdType == CL_COMMAND_SVM_MIGRATE_MEM,
        //                   "Unexpected command type for clEnqueueSVMMigrateMem
        //                   " "with NULL pointer");
        log_info(
            "      clEnqueueSVMMigrateMem with NULL pointer has command type "
            "%4X (%4X)\n",
            cmdType, CL_COMMAND_SVM_MIGRATE_MEM);
    }


    // Try to call clSetKernelExecInfo with an empty set
    error =
        clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, nullptr);
    // test_error(error, "clSetKernelExecInfo with an empty set failed");
    log_info("      clSetKernelExecInfo with an empty set returned %s\n",
             IGetErrorString(error));

    // Try to call clSetKernelExecInfo with an explicit NULL pointer
    {
        void* svm_pointers[] = { nullptr };
        error = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                    sizeof(svm_pointers), svm_pointers);
        // test_error(error, "clSetKernelExecInfo with NULL pointer failed");
        log_info("      clSetKernelExecInfo with NULL pointer returned %s\n",
                 IGetErrorString(error));
    }

    return TEST_PASS;
}

REGISTER_TEST(svm_size_zero)
{
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
