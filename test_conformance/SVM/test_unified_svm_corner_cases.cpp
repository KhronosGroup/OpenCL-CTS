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

struct UnifiedSVMCornerCaseAllocFree : UnifiedSVMBase
{
    UnifiedSVMCornerCaseAllocFree(cl_context context, cl_device_id device,
                                  cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int run() override
    {
        cl_int err;
        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            auto mem = get_usvm_wrapper<cl_int>(ti);

            log_info("     testing zero-byte allocation\n");
            err = mem->allocate(0);
            test_error(err, "zero-byte SVM allocation failed");
            test_assert_error(
                mem->get_ptr() == nullptr,
                "zero-byte SVM allocation did not return a null pointer");
        }

        log_info("   testing NULL pointer free\n");
        err = clSVMFreeWithPropertiesKHR(context, nullptr, 0, nullptr);
        test_error(err, "clSVMFreeWithPropertiesKHR with NULL pointer failed");

        log_info("   testing asynchronous empty set free\n");
        clEventWrapper event;
        err = clEnqueueSVMFree(queue, 0, nullptr, nullptr, nullptr, 0, nullptr,
                               &event);
        test_error(err, "clEnqueueSVMFree with empty set failed");

        err = clFinish(queue);
        test_error(err,
                   "clFinish after clEnqueueSVMFree with empty set failed");

        err = check_event_type(event, CL_COMMAND_SVM_FREE);
        test_error(err,
                   "clEnqueueSVMFree did not return a "
                   "CL_COMMAND_SVM_FREE event");

        log_info("   testing asynchronous NULL pointer free\n");
        event = nullptr;
        void* svm_pointers[] = { nullptr };
        err = clEnqueueSVMFree(queue, 1, svm_pointers, nullptr, nullptr, 0,
                               nullptr, &event);
        test_error(err, "clEnqueueSVMFree with NULL pointer failed");

        err = clFinish(queue);
        test_error(err,
                   "clFinish after clEnqueueSVMFree with NULL pointer failed");

        err = check_event_type(event, CL_COMMAND_SVM_FREE);
        test_error(err,
                   "clEnqueueSVMFree did not return a "
                   "CL_COMMAND_SVM_FREE event");

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_corner_case_alloc_free)
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

    UnifiedSVMCornerCaseAllocFree Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}


struct UnifiedSVMCornerCaseSetKernelArg : UnifiedSVMBase
{
    UnifiedSVMCornerCaseSetKernelArg(cl_context context, cl_device_id device,
                                     cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_NullPointer()
    {
        cl_int err = clSetKernelArgSVMPointer(kernel_StorePointer, 0, nullptr);
        test_error(
            err,
            "clSetKernelArgSVMPointer with a NULL pointer returned an error");

        return CL_SUCCESS;
    }

    cl_int test_BogusPointer()
    {
        const void* bogus = (const void*)0xDEADBEEF;
        cl_int err = clSetKernelArgSVMPointer(kernel_StorePointer, 0, bogus);
        test_error(
            err,
            "clSetKernelArgSVMPointer with a bogus pointer returned an error");

        clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(cl_int*), nullptr, &err);
        test_error(err, "could not create destination buffer");

        err = clSetKernelArg(kernel_StorePointer, 1, sizeof(out), &out);
        test_error(err, "could not set kernel arguments");

        size_t global_work_size = 1;
        err = clEnqueueNDRangeKernel(queue, kernel_StorePointer, 1, nullptr,
                                     &global_work_size, nullptr, 0, nullptr,
                                     nullptr);
        test_error(err, "clEnqueueNDRangeKernel failed");

        err = clFinish(queue);
        test_error(err, "clFinish failed");

        void* check = nullptr;
        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(cl_int*),
                                  &check, 0, nullptr, nullptr);
        test_error(err, "could not read output buffer");

        test_assert_error(check == bogus,
                          "stored pointer does not match input pointer");

        return CL_SUCCESS;
    }

    cl_int setup() override
    {
        cl_int err = UnifiedSVMBase::setup();
        test_error(err, "UnifiedSVMBase setup failed");

        const char* programString = R"(
            // workaround for error: kernel parameter cannot be declared as a pointer to a pointer
            struct s { const global int* ptr; }; 
            kernel void test_StorePointer(const global int* ptr, global struct s* dst)
            {
                dst[get_global_id(0)].ptr = ptr;
            }
        )";

        clProgramWrapper program;
        err =
            create_single_kernel_helper(context, &program, &kernel_StorePointer,
                                        1, &programString, "test_StorePointer");
        test_error(err, "could not create StorePointer kernel");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing clSetKernelArgSVMPointer with a NULL pointer\n");
        err = test_NullPointer();
        test_error(err, "clSetKernelArgSVMPointer with a NULL pointer failed");

        log_info("   testing clSetKernelArgSVMPointer with a bogus pointer\n");
        err = test_BogusPointer();
        test_error(err, "clSetKernelArgSVMPointer with a bogus pointer failed");

        return CL_SUCCESS;
    }

    clKernelWrapper kernel_StorePointer;
};

REGISTER_TEST(unified_svm_corner_case_set_kernel_arg)
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

    UnifiedSVMCornerCaseSetKernelArg Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}

struct UnifiedSVMCornerCaseSetKernelExecInfo : UnifiedSVMBase
{
    UnifiedSVMCornerCaseSetKernelExecInfo(cl_context context,
                                          cl_device_id device,
                                          cl_command_queue queue,
                                          int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_EmptySet()
    {
        cl_int err = clSetKernelExecInfo(
            kernel_OneArg, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, nullptr);
        test_error(err,
                   "clSetKernelExecInfo with an empty set returned an error");

        return CL_SUCCESS;
    }

    cl_int test_NullPointer()
    {
        const void* svm_ptrs[] = { nullptr };
        cl_int err =
            clSetKernelExecInfo(kernel_OneArg, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                sizeof(svm_ptrs), svm_ptrs);
        test_error(err,
                   "clSetKernelExecInfo with a NULL pointer returned an error");

        return CL_SUCCESS;
    }

    cl_int test_BogusPointer()
    {
        const void* bogus = (const void*)0xDEADBEEF;
        const void* svm_ptrs[] = { bogus };
        cl_int err =
            clSetKernelExecInfo(kernel_OneArg, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                sizeof(svm_ptrs), svm_ptrs);
        test_error(
            err, "clSetKernelExecInfo with a bogus pointer returned an error");

        return CL_SUCCESS;
    }

    cl_int setup() override
    {
        cl_int err = UnifiedSVMBase::setup();
        test_error(err, "UnifiedSVMBase setup failed");

        const char* programString = R"(
            kernel void test_OneArg(global int* dst)
            {
                dst[get_global_id(0)] = -1;
            }
        )";

        clProgramWrapper program;
        err = create_single_kernel_helper(context, &program, &kernel_OneArg, 1,
                                          &programString, "test_OneArg");
        test_error(err, "could not create OneArg kernel");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing clSetKernelExecInfo with an empty set\n");
        err = test_EmptySet();
        test_error(err, "clSetKernelExecInfo with an empty set failed");

        log_info("   testing clSetKernelExecInfo with a NULL pointer\n");
        err = test_NullPointer();
        test_error(err, "clSetKernelExecInfo with a NULL pointer failed");

        log_info("   testing clSetKernelExecInfo with a bogus pointer\n");
        err = test_BogusPointer();
        test_error(err, "clSetKernelExecInfo with a bogus pointer failed");

        return CL_SUCCESS;
    }

    clKernelWrapper kernel_OneArg;
};

REGISTER_TEST(unified_svm_corner_case_set_kernel_exec_info)
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

    UnifiedSVMCornerCaseSetKernelExecInfo Test(context, device, queue,
                                               num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}

struct UnifiedSVMCornerCaseMemcpy : UnifiedSVMBase
{
    UnifiedSVMCornerCaseMemcpy(cl_context context, cl_device_id device,
                               cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_NullPointer()
    {
        cl_int value = 0;

        clEventWrapper event;
        cl_int err = clEnqueueSVMMemcpy(queue, CL_TRUE, nullptr, &value, 0, 0,
                                        nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMemcpy with a NULL destination pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");

        event = nullptr;
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, &value, nullptr, 0, 0, nullptr,
                                 &event);
        test_error(
            err,
            "clEnqueueSVMMemcpy with a NULL source pointer returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");


        return CL_SUCCESS;
    }

    cl_int test_BogusPointer()
    {
        void* bogus = (void*)0xDEADBEEF;
        cl_int value = 0;

        clEventWrapper event;
        cl_int err = clEnqueueSVMMemcpy(queue, CL_TRUE, bogus, &value, 0, 0,
                                        nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMemcpy with a bogus destination pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");

        event = nullptr;
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, &value, bogus, 0, 0, nullptr,
                                 &event);
        test_error(
            err,
            "clEnqueueSVMMemcpy with a bogus source pointer returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");


        return CL_SUCCESS;
    }

    cl_int test_ValidPointer(cl_uint typeIndex)
    {
        cl_int err;
        cl_int value = 0;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);

        clEventWrapper event;
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, mem->get_ptr(), &value, 0, 0,
                                 nullptr, &event);
        test_error(
            err,
            "clEnqueueSVMMemcpy with valid SVM dst pointer returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");

        event = nullptr;
        err = clEnqueueSVMMemcpy(queue, CL_TRUE, &value, mem->get_ptr(), 0, 0,
                                 nullptr, &event);
        test_error(
            err,
            "clEnqueueSVMMemcpy with valid SVM src pointer returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMCPY);
        test_error(err,
                   "clEnqueueSVMMemcpy did not return a "
                   "CL_COMMAND_SVM_MEMCPY event");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing clEnqueueSVMMemcpy with a NULL pointer and a "
                 "size of zero\n");
        err = test_NullPointer();
        test_error(
            err,
            "clEnqueueSVMMemcpy with a NULL pointer and a size of zero failed");

        log_info("   testing clEnqueueSVMMemcpy with a bogus pointer and a "
                 "size of zero\n");
        err = test_BogusPointer();
        test_error(err,
                   "clEnqueueSVMMemcpy with a bogus pointer and a size of zero "
                   "failed");

        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            log_info("     testing clEnqueueSVMMemcpy with a valid pointer and "
                     "a size of zero\n");
            err = test_ValidPointer(ti);
            test_error(err,
                       "clEnqueueSVMMemcpy with a valid pointer and a size of "
                       "zero failed");
        }

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_corner_case_memcpy)
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

    UnifiedSVMCornerCaseMemcpy Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}

struct UnifiedSVMCornerCaseMemFill : UnifiedSVMBase
{
    UnifiedSVMCornerCaseMemFill(cl_context context, cl_device_id device,
                                cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_NullPointer()
    {
        const cl_int pattern = 0;

        clEventWrapper event;
        cl_int err = clEnqueueSVMMemFill(
            queue, nullptr, &pattern, sizeof(pattern), 0, 0, nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMemFill with a NULL destination pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMFILL);
        test_error(err,
                   "clEnqueueSVMMemFill did not return a "
                   "CL_COMMAND_SVM_MEMFILL event");

        return CL_SUCCESS;
    }

    cl_int test_BogusPointer()
    {
        void* bogus = (void*)0xDEADBEEF;
        const cl_int pattern = 0;

        clEventWrapper event;
        cl_int err = clEnqueueSVMMemFill(
            queue, bogus, &pattern, sizeof(pattern), 0, 0, nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMemFill with a bogus destination pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMFILL);
        test_error(err,
                   "clEnqueueSVMMemFill did not return a "
                   "CL_COMMAND_SVM_MEMFILL event");

        return CL_SUCCESS;
    }

    cl_int test_ValidPointer(cl_uint typeIndex)
    {
        cl_int err;
        const cl_int pattern = 0;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);

        clEventWrapper event;
        err = clEnqueueSVMMemFill(queue, mem->get_ptr(), &pattern,
                                  sizeof(pattern), 0, 0, nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMemFill with a valid destination pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MEMFILL);
        test_error(err,
                   "clEnqueueSVMMemFill did not return a "
                   "CL_COMMAND_SVM_MEMFILL event");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing clEnqueueSVMMemFill with a NULL pointer and a "
                 "size of zero\n");
        err = test_NullPointer();
        test_error(err,
                   "clEnqueueSVMMemFill with a NULL pointer and a size of zero "
                   "failed");

        log_info("   testing clEnqueueSVMMemFill with a bogus pointer and a "
                 "size of zero\n");
        err = test_BogusPointer();
        test_error(
            err,
            "clEnqueueSVMMemFill with a bogus pointer and a size of zero "
            "failed");

        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            log_info(
                "     testing clEnqueueSVMMemFill with a valid pointer and "
                "a size of zero\n");
            err = test_ValidPointer(ti);
            test_error(err,
                       "clEnqueueSVMMemFill with a valid pointer and a size of "
                       "zero failed");
        }

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_corner_case_mem_fill)
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

    UnifiedSVMCornerCaseMemFill Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}

struct UnifiedSVMCornerCaseMigrateMem : UnifiedSVMBase
{
    UnifiedSVMCornerCaseMigrateMem(cl_context context, cl_device_id device,
                                   cl_command_queue queue, int num_elements)
        : UnifiedSVMBase(context, device, queue, num_elements)
    {}

    cl_int test_NullPointer()
    {
        cl_int err;

        const void* svm_pointers[] = { nullptr };
        const size_t sizes[] = { 0 };
        clEventWrapper event;
        err = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                     CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                                     nullptr, &event);
        test_error(
            err,
            "clEnqueueSVMMigrateMem with a NULL pointer returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
        test_error(err,
                   "clEnqueueSVMMigrateMem did not return a "
                   "CL_COMMAND_SVM_MIGRATE_MEM event");

        return CL_SUCCESS;
    }

    cl_int test_ValidPointer(cl_uint typeIndex)
    {
        cl_int err;

        auto mem = get_usvm_wrapper<cl_int>(typeIndex);
        err = mem->allocate(1);

        const void* svm_pointers[] = { mem->get_ptr() };
        const size_t sizes[] = { 0 };
        clEventWrapper event;
        err = clEnqueueSVMMigrateMem(queue, 1, svm_pointers, sizes,
                                     CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                                     nullptr, &event);
        test_error(err,
                   "clEnqueueSVMMigrateMem with a valid pointer "
                   "returned an error");

        err = check_event_type(event, CL_COMMAND_SVM_MIGRATE_MEM);
        test_error(err,
                   "clEnqueueSVMMigrateMem did not return a "
                   "CL_COMMAND_SVM_MIGRATE_MEM event");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;

        log_info("   testing clEnqueueSVMMigrateMem with a NULL pointer and a "
                 "size of zero\n");
        err = test_NullPointer();
        test_error(
            err,
            "clEnqueueSVMMigrateMem with a NULL pointer and a size of zero "
            "failed");

        for (cl_uint ti = 0; ti < static_cast<cl_uint>(deviceUSVMCaps.size());
             ti++)
        {
            log_info("   testing SVM type %u\n", ti);

            log_info(
                "     testing clEnqueueSVMMigrateMem with a valid pointer and "
                "a size of zero\n");
            err = test_ValidPointer(ti);
            test_error(
                err,
                "clEnqueueSVMMigrateMem with a valid pointer and a size of "
                "zero failed");
        }

        return CL_SUCCESS;
    }
};

REGISTER_TEST(unified_svm_corner_case_migrate_mem)
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

    UnifiedSVMCornerCaseMigrateMem Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
