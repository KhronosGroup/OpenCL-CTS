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
        test_error(err, "could not enqueue kernel");

        err = clFinish(queue);
        test_error(err, "could not finish queue");

        int* out_ptr = nullptr;
        err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(int*),
                                  &out_ptr, 0, nullptr, nullptr);
        test_error(err, "could not read output buffer");

        test_assert_error(out_ptr == src->get_ptr(),
                          "output pointer does not match input pointer");

        return CL_SUCCESS;
    }

    cl_int run() override
    {
        cl_int err;
        for (cl_uint i = 0; i < static_cast<cl_uint>(deviceUSVMCaps.size());
             i++)
        {
            const auto caps = deviceUSVMCaps[i];
            log_info("   testing SVM type %u, capabilities 0x%08" PRIx64 "\n", i,
                     caps);

            if (caps & CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR)
            {
                err = test_CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR(i);
                test_error(err,
                           "CL_SVM_CAPABILITY_SINGLE_ADDRESS_SPACE_KHR "
                           "failed");
            }
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
