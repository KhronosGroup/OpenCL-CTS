//
// Copyright (c) 2024 The Khronos Group Inc.
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

#ifndef CL_KHR_SEMAPHORE_BASE_H
#define CL_KHR_SEMAPHORE_BASE_H

#include <CL/cl_ext.h>
#include "harness/deviceInfo.h"
#include "harness/testHarness.h"

#include "harness/typeWrappers.h"

struct SemaphoreBase
{
    SemaphoreBase(cl_device_id device): device(device) {}

    cl_int init_extension_functions()
    {
        cl_platform_id platform;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                            &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        // If it is supported get the addresses of all the APIs here.
        // clang-format off
#define GET_EXTENSION_ADDRESS(FUNC)                                            \
        FUNC = reinterpret_cast<FUNC##_fn>(                                    \
            clGetExtensionFunctionAddressForPlatform(platform, #FUNC));        \
        if (FUNC == nullptr)                                                   \
        {                                                                      \
            log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed" \
                      " with " #FUNC "\n");                                    \
            return TEST_FAIL;                                                  \
        }
        // clang-format on

        GET_EXTENSION_ADDRESS(clCreateSemaphoreWithPropertiesKHR);
        GET_EXTENSION_ADDRESS(clEnqueueSignalSemaphoresKHR);
        GET_EXTENSION_ADDRESS(clEnqueueWaitSemaphoresKHR);
        GET_EXTENSION_ADDRESS(clReleaseSemaphoreKHR);
        GET_EXTENSION_ADDRESS(clGetSemaphoreInfoKHR);
        GET_EXTENSION_ADDRESS(clRetainSemaphoreKHR);
        GET_EXTENSION_ADDRESS(clGetSemaphoreHandleForTypeKHR);

#undef GET_EXTENSION_ADDRESS
        return CL_SUCCESS;
    }

    clCreateSemaphoreWithPropertiesKHR_fn clCreateSemaphoreWithPropertiesKHR =
        nullptr;
    clEnqueueSignalSemaphoresKHR_fn clEnqueueSignalSemaphoresKHR = nullptr;
    clEnqueueWaitSemaphoresKHR_fn clEnqueueWaitSemaphoresKHR = nullptr;
    clReleaseSemaphoreKHR_fn clReleaseSemaphoreKHR = nullptr;
    clGetSemaphoreInfoKHR_fn clGetSemaphoreInfoKHR = nullptr;
    clRetainSemaphoreKHR_fn clRetainSemaphoreKHR = nullptr;
    clGetSemaphoreHandleForTypeKHR_fn clGetSemaphoreHandleForTypeKHR = nullptr;

    cl_device_id device = nullptr;
};

// Wrapper class based off generic typeWrappers.h wrappers. However, because
// the release/retain functions are queried at runtime from the platform,
// rather than known at compile time we cannot link the instantiated template.
// Instead, pass an instance of `SemaphoreTestBase` on wrapper construction
// to access the release/retain functions.
class clSemaphoreWrapper {
    cl_semaphore_khr object = nullptr;

    void retain()
    {
        if (!object) return;

        auto err = base->clRetainSemaphoreKHR(object);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clRetainCommandBufferKHR() failed");
            std::abort();
        }
    }

    void release()
    {
        if (!object) return;

        auto err = base->clReleaseSemaphoreKHR(object);
        if (err != CL_SUCCESS)
        {
            print_error(err, "clReleaseCommandBufferKHR() failed");
            std::abort();
        }
    }

    // Used to access release/retain functions
    SemaphoreBase *base;

public:
    // We always want to have base available to dereference
    clSemaphoreWrapper() = delete;

    clSemaphoreWrapper(SemaphoreBase *base): base(base) {}

    // On assignment, assume the object has a refcount of one.
    clSemaphoreWrapper &operator=(cl_semaphore_khr rhs)
    {
        reset(rhs);
        return *this;
    }

    // Copy semantics, increase retain count.
    clSemaphoreWrapper(clSemaphoreWrapper const &w) { *this = w; }
    clSemaphoreWrapper &operator=(clSemaphoreWrapper const &w)
    {
        reset(w.object);
        retain();
        return *this;
    }

    // Move semantics, directly take ownership.
    clSemaphoreWrapper(clSemaphoreWrapper &&w) { *this = std::move(w); }
    clSemaphoreWrapper &operator=(clSemaphoreWrapper &&w)
    {
        reset(w.object);
        w.object = nullptr;
        return *this;
    }

    ~clSemaphoreWrapper() { reset(); }

    // Release the existing object, if any, and own the new one, if any.
    void reset(cl_semaphore_khr new_object = nullptr)
    {
        release();
        object = new_object;
    }

    operator cl_semaphore_khr() const { return object; }
    operator const cl_semaphore_khr *() { return &object; }
};

struct SemaphoreTestBase : public SemaphoreBase
{
    SemaphoreTestBase(cl_device_id device, cl_context context,
                      cl_command_queue queue)
        : SemaphoreBase(device), context(context), semaphore(this)
    {
        cl_int error = init_extension_functions();
        if (error != CL_SUCCESS)
            throw std::runtime_error("init_extension_functions failed\n");

        error = clRetainCommandQueue(queue);
        if (error != CL_SUCCESS)
            throw std::runtime_error("clRetainCommandQueue failed\n");
        this->queue = queue;
    }

    virtual cl_int Run() = 0;

protected:
    cl_context context = nullptr;
    clCommandQueueWrapper queue = nullptr;
    clSemaphoreWrapper semaphore = nullptr;
};

template <class T>
int MakeAndRunTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
{
    if (!is_extension_available(device, "cl_khr_semaphore"))
    {
        log_info(
            "Device does not support 'cl_khr_semaphore'. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_int status = TEST_PASS;
    try
    {
        auto test_fixture = T(device, context, queue);
        status = test_fixture.Run();
    } catch (const std::runtime_error &e)
    {
        log_error("%s", e.what());
        return TEST_FAIL;
    }

    return status;
}

#endif // CL_KHR_SEMAPHORE_BASE_H
