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

static int svm_buffer_address_equivalence_helper(cl_context context,
                                                 cl_command_queue queue,
                                                 cl_kernel kernel, cl_mem out,
                                                 cl_svm_mem_flags svmFlags)
{
    constexpr size_t sz = 1024 * 1024;
    cl_int error = CL_SUCCESS;

    clSVMWrapper svmPtr(context, sz, svmFlags | CL_MEM_READ_WRITE);
    test_assert_error(svmPtr() != nullptr, "clSVMAlloc failed");

    // printf("SVM pointer is %p\n", svmPtr());

    clMemWrapper svmBuf = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sz, svmPtr(), &error);
    test_error(error, "clCreateBuffer with SVM pointer failed");

    error |= clSetKernelArg(kernel, 0, sizeof(svmBuf), &svmBuf);
    error |= clSetKernelArg(kernel, 1, sizeof(out), &out);
    test_error(error, "clSetKernelArg failed");

    size_t globalWorkSize = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                   nullptr, 0, nullptr, nullptr);
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    void *check = nullptr;
    error = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(check), &check,
                                0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer for output buffer failed");

    // printf("Output pointer is %p\n", check);

    test_assert_error(check == svmPtr(),
                      "stored pointer does not match SVM pointer");

    return TEST_PASS;
}

REGISTER_TEST(svm_buffer_address_equivalence)
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

    const char *programString = R"(
        // workaround for error: kernel parameter cannot be declared as a pointer to a pointer
        struct s { const global int* ptr; }; 
        kernel void test_StorePointer(const global int* ptr, global struct s* dst)
        {
            //printf("Buffer pointer on the device = %p\n", ptr);
            dst[get_global_id(0)].ptr = ptr;
        }
    )";

    clProgramWrapper program;
    clKernelWrapper kernel;
    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &programString, "test_StorePointer");
    test_error(error, "could not create StorePointer kernel");

    clMemWrapper out = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(cl_int *), nullptr, &error);
    test_error(error, "could not create destination buffer");

    cl_device_svm_capabilities svmCaps = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svmCaps),
                            &svmCaps, nullptr);
    test_error(error, "clGetDeviceInfo failed to get SVM capabilities");

    int result = TEST_PASS;

    if (svmCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
    {
        result |= svm_buffer_address_equivalence_helper(context, queue, kernel,
                                                        out, 0);
    }
    if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
    {
        result |= svm_buffer_address_equivalence_helper(
            context, queue, kernel, out, CL_MEM_SVM_FINE_GRAIN_BUFFER);
    }
    if (svmCaps & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS))
    {
        result |= svm_buffer_address_equivalence_helper(
            context, queue, kernel, out,
            CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
    }

    return result;
}
