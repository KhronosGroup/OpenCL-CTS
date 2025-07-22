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

struct UnifiedSVMExecInfo : UnifiedSVMBase
{
    using UnifiedSVMBase::UnifiedSVMBase;

    // Test reading from USM pointer indirectly using clSetKernelExecInfo.
    // The test will perform a memcpy on the device.
    cl_int test_svm_exec_info_read(USVMWrapper<cl_uchar> *mem)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> src_data(alloc_count, 0);

        auto ptr = mem->get_ptr();
        clMemWrapper indirect =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(ptr), &ptr, &err);
        test_error(err, "could not create indirect buffer");

        clMemWrapper direct = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             src_data.size(), nullptr, &err);
        test_error(err, "could not create direct buffer");

        err = clSetKernelArg(kernel_IndirectAccessRead, 0, sizeof(indirect),
                             &indirect);
        test_error(err, "could not set kernel argument 0");

        err = clSetKernelArg(kernel_IndirectAccessRead, 1, sizeof(direct),
                             &direct);
        test_error(err, "could not set kernel argument 1");

        size_t test_offsets[] = { 0, alloc_count / 2 };

        for (auto offset : test_offsets)
        {
            // Fill src data with a random pattern
            generate_random_inputs(src_data, d);

            err = mem->write(src_data);
            test_error(err, "could not write to usvm memory");

            void *info_ptr = &mem->get_ptr()[offset];

            err = clSetKernelExecInfo(kernel_IndirectAccessRead,
                                      CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                      sizeof(void *), &info_ptr);
            test_error(err, "could not enable indirect access");

            size_t gws{ alloc_count };
            err = clEnqueueNDRangeKernel(queue, kernel_IndirectAccessRead, 1,
                                         nullptr, &gws, nullptr, 0, nullptr,
                                         nullptr);
            test_error(err, "clEnqueueNDRangeKernel failed");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            std::vector<cl_uchar> result_data(alloc_count, 0);
            err = clEnqueueReadBuffer(queue, direct, CL_TRUE, 0,
                                      result_data.size(), result_data.data(), 0,
                                      nullptr, nullptr);
            test_error(err, "clEnqueueReadBuffer failed");

            // Validate result
            if (result_data != src_data)
            {
                for (size_t i = 0; i < alloc_count; i++)
                {
                    if (src_data[i] != result_data[i])
                    {
                        log_error(
                            "While attempting indirect read "
                            "clSetKernelExecInfo with "
                            "offset:%zu size:%zu \n"
                            "Data verification mismatch at %zu expected: %d "
                            "got: %d\n",
                            offset, alloc_count, i, src_data[i],
                            result_data[i]);
                        return TEST_FAIL;
                    }
                }
            }
        }
        return CL_SUCCESS;
    }

    // Test writing to USM pointer indirectly using clSetKernelExecInfo.
    // The test will perform a memcpy on the device.
    cl_int test_svm_exec_info_write(USVMWrapper<cl_uchar> *mem)
    {
        cl_int err = CL_SUCCESS;

        std::vector<cl_uchar> src_data(alloc_count, 0);

        size_t test_offsets[] = { 0, alloc_count / 2 };

        auto ptr = mem->get_ptr();
        clMemWrapper indirect =
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(ptr), &ptr, &err);
        test_error(err, "could not create indirect buffer");

        clMemWrapper direct = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                             alloc_count, nullptr, &err);
        test_error(err, "could not create direct buffer");

        err = clSetKernelArg(kernel_IndirectAccessWrite, 0, sizeof(indirect),
                             &indirect);
        test_error(err, "could not set kernel argument 0");

        err = clSetKernelArg(kernel_IndirectAccessWrite, 1, sizeof(direct),
                             &direct);
        test_error(err, "could not set kernel argument 1");

        for (auto offset : test_offsets)
        {
            // Fill src data with a random pattern
            generate_random_inputs(src_data, d);

            err = clEnqueueWriteBuffer(queue, direct, CL_NON_BLOCKING, 0,
                                       src_data.size(), src_data.data(), 0,
                                       nullptr, nullptr);
            test_error(err, "clEnqueueReadBuffer failed");

            void *info_ptr = &mem->get_ptr()[offset];

            err = clSetKernelExecInfo(kernel_IndirectAccessWrite,
                                      CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                      sizeof(void *), &info_ptr);
            test_error(err, "could not enable indirect access");

            size_t gws{ alloc_count };
            err = clEnqueueNDRangeKernel(queue, kernel_IndirectAccessWrite, 1,
                                         nullptr, &gws, nullptr, 0, nullptr,
                                         nullptr);
            test_error(err, "clEnqueueNDRangeKernel failed");

            err = clFinish(queue);
            test_error(err, "clFinish failed");

            std::vector<cl_uchar> result_data(alloc_count, 0);
            err = mem->read(result_data);
            test_error(err, "could not read from usvm memory");

            // Validate result
            if (result_data != src_data)
            {
                for (size_t i = 0; i < alloc_count; i++)
                {
                    if (src_data[i] != result_data[i])
                    {
                        log_error(
                            "While attempting indirect write "
                            "clSetKernelExecInfo with "
                            "offset:%zu size:%zu \n"
                            "Data verification mismatch at %zu expected: %d "
                            "got: %d\n",
                            offset, alloc_count, i, src_data[i],
                            result_data[i]);
                        return TEST_FAIL;
                    }
                }
            }
        }
        return CL_SUCCESS;
    }

    cl_int setup() override
    {
        cl_int err = UnifiedSVMBase::setup();
        if (CL_SUCCESS != err)
        {
            return err;
        }

        return createIndirectAccessKernel();
    }

    cl_int run() override
    {
        cl_int err;
        cl_uint max_ti = static_cast<cl_uint>(deviceUSVMCaps.size());

        for (cl_uint ti = 0; ti < max_ti; ti++)
        {
            auto mem = get_usvm_wrapper<cl_uchar>(ti);

            err = mem->allocate(alloc_count);
            test_error(err, "SVM allocation failed");

            log_info("   testing clSetKernelArgSVMPointer() SVM type %u \n",
                     ti);
            err = test_svm_exec_info_read(mem.get());
            if (CL_SUCCESS != err)
            {
                return err;
            }

            err = test_svm_exec_info_write(mem.get());
            if (CL_SUCCESS != err)
            {
                return err;
            }

            err = mem->free();
            test_error(err, "SVM free failed");
        }

        return CL_SUCCESS;
    }

    cl_int createIndirectAccessKernel()
    {
        cl_int err;

        const char *programString = R"(
            struct s { const global unsigned char* ptr; };
            kernel void test_IndirectAccessRead(const global struct s* src, global unsigned char* dst)
            {
                dst[get_global_id(0)] = src->ptr[get_global_id(0)];
            }

            struct d { global unsigned char* ptr; };
            kernel void test_IndirectAccessWrite(global struct d* dst, const global unsigned char* src)
            {
                dst->ptr[get_global_id(0)] = src[get_global_id(0)];
            }
        )";

        clProgramWrapper program;
        err = create_single_kernel_helper(
            context, &program, &kernel_IndirectAccessRead, 1, &programString,
            "test_IndirectAccessRead");
        test_error(err, "could not create IndirectAccessRead kernel");

        kernel_IndirectAccessWrite =
            clCreateKernel(program, "test_IndirectAccessWrite", &err);
        test_error(err, "could not create IndirectAccessWrite kernel");

        return CL_SUCCESS;
    }

    clKernelWrapper kernel_IndirectAccessRead;
    clKernelWrapper kernel_IndirectAccessWrite;

    static constexpr size_t alloc_count = 1024;
};

REGISTER_TEST(unified_svm_exec_info)
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

    UnifiedSVMExecInfo Test(context, device, queue, num_elements);
    err = Test.setup();
    test_error(err, "test setup failed");

    err = Test.run();
    test_error(err, "test failed");

    return TEST_PASS;
}
