//
// Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef CL_KHR_BASIC_COMMAND_BUFFER_H
#define CL_KHR_BASIC_COMMAND_BUFFER_H

#include "command_buffer_test_base.h"
#include "harness/typeWrappers.h"

#define ADD_PROP(prop)                                                         \
    {                                                                          \
        prop, #prop                                                            \
    }

#define CHECK_VERIFICATION_ERROR(reference, result, index)                     \
    {                                                                          \
        if (reference != result)                                               \
        {                                                                      \
            log_error("Expected %d was %d at index %zu\n", reference, result,  \
                      index);                                                  \
            return TEST_FAIL;                                                  \
        }                                                                      \
    }

// If it is supported get the addresses of all the APIs here.
#define GET_EXTENSION_ADDRESS(FUNC)                                            \
    FUNC = reinterpret_cast<FUNC##_fn>(                                        \
        clGetExtensionFunctionAddressForPlatform(platform, #FUNC));            \
    if (FUNC == nullptr)                                                       \
    {                                                                          \
        log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed"     \
                  " with " #FUNC "\n");                                        \
        return TEST_FAIL;                                                      \
    }


// Helper test fixture for constructing OpenCL objects used in testing
// a variety of simple command-buffer enqueue scenarios.
struct BasicCommandBufferTest : CommandBufferTestBase
{

    BasicCommandBufferTest(cl_device_id device, cl_context context,
                           cl_command_queue queue);

    virtual bool Skip();
    virtual cl_int SetUpKernel(void);
    virtual cl_int SetUpKernelArgs(void);
    virtual cl_int SetUp(int elements);

    // Test body returning an OpenCL error code
    virtual cl_int Run() = 0;

protected:
    virtual size_t data_size() const { return num_elements * sizeof(cl_int); }

    cl_context context;
    clCommandQueueWrapper queue;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper in_mem, out_mem, off_mem;
    size_t num_elements;

    // Device support query results
    bool simultaneous_use_support;
    bool out_of_order_support;
    bool queue_out_of_order_support;
    bool device_side_enqueue_support;

    // user request for simultaneous use
    bool simultaneous_use_requested;
    unsigned buffer_size_multiplier;
    clCommandBufferWrapper command_buffer;
};


template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    if (!is_extension_available(device, "cl_khr_command_buffer"))
    {
        log_info("Device does not support 'cl_khr_command_buffer'. Skipping "
                 "the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    Version device_version = get_device_cl_version(device);
    if ((device_version >= Version(3, 0))
        || is_extension_available(device, "cl_khr_extended_versioning"))
    {

        cl_version extension_version =
            get_extension_version(device, "cl_khr_command_buffer");

        if (extension_version != CL_MAKE_VERSION(0, 9, 6))
        {
            log_info("cl_khr_command_buffer version 0.9.6 is required to run "
                     "the test, skipping.\n ");
            return TEST_SKIPPED_ITSELF;
        }
    }
    try
    {
        auto test_fixture = T(device, context, queue);

        if (test_fixture.Skip())
        {
            return TEST_SKIPPED_ITSELF;
        }

        cl_int error = test_fixture.SetUp(num_elements);
        test_error_ret(error, "Error in test initialization", TEST_FAIL);

        error = test_fixture.Run();
        test_error_ret(error, "Test Failed", TEST_FAIL);
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return TEST_PASS;
}

#endif // CL_KHR_BASIC_COMMAND_BUFFER_H
