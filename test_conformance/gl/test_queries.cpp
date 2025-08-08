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
//
#include "common.h"
#include "testBase.h"
#include "gl/setup.h"

namespace {

struct FindDeviceFunctor
{
    FindDeviceFunctor()
    {
        if (init()) throw std::runtime_error("FindDeviceFunctor failed");
    }

    cl_int init()
    {
        cl_uint num_platforms = 0;
        cl_int error = clGetPlatformIDs(0, nullptr, &num_platforms);
        test_error(error, "clGetPlatformIDs failed");

        platforms.resize(num_platforms);

        error =
            clGetPlatformIDs(num_platforms, platforms.data(), &num_platforms);
        test_error(error, "clGetPlatformIDs failed");

        return CL_SUCCESS;
    }

    bool find(const cl_device_id id)
    {
        cl_uint num_devices = 0;
        for (size_t p = 0; p < platforms.size(); p++)
        {
            cl_int error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0,
                                          nullptr, &num_devices);
            test_error(error, "clGetDeviceIDs failed");

            std::vector<cl_device_id> devices(num_devices);
            error = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                                   num_devices, devices.data(), nullptr);
            test_error(error, "clGetDeviceIDs failed");

            for (auto did : devices)
                if (did == id) return false;
        }
        return true;
    }
    std::vector<cl_platform_id> platforms;
};

} // anonymous namespace

int test_queries(cl_device_id device, cl_context context,
                 cl_command_queue queue, int)
{
    // get a platform associated with device id
    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                   sizeof(cl_platform_id), &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

    size_t returned_size = 0;
    error = clGetContextInfo(context, CL_CONTEXT_PROPERTIES, 0, nullptr,
                             &returned_size);
    test_error(error, "clGetContextInfo failed");

    std::vector<cl_context_properties> props(
        returned_size / sizeof(cl_context_properties), 0);

    error = clGetContextInfo(context, CL_CONTEXT_PROPERTIES,
                             sizeof(cl_context_properties) * props.size(),
                             props.data(), nullptr);
    test_error(error, "clGetContextInfo failed");

    // get GL context info function pointer
    size_t dev_size = 0;
    clGetGLContextInfoKHR_fn clGetGLContextInfoKHR =
        (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(
            platform, "clGetGLContextInfoKHR");

    test_assert_error(clGetGLContextInfoKHR != NULL,
                      "unable to get the function pointer for "
                      "clGetGLContextInfoKHR\n");

    // get the size of all GL interop capable devices
    error = clGetGLContextInfoKHR(props.data(), CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                  0, nullptr, &dev_size);
    test_error(error,
               "clGetGLContextInfoKHR(CL_DEVICES_FOR_GL_CONTEXT_KHR) failed");

    dev_size /= sizeof(cl_device_id);
    log_info("GL _context supports %zu compute devices\n", dev_size);


    // get all GL interop capable devices
    std::vector<cl_device_id> devices(dev_size, 0);
    error = clGetGLContextInfoKHR(props.data(), CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                  devices.size() * sizeof(cl_device_id),
                                  devices.data(), &dev_size);
    test_error(error,
               "clGetGLContextInfoKHR(CL_DEVICES_FOR_GL_CONTEXT_KHR) failed");
    if (devices.size() != dev_size / sizeof(cl_device_id))
    {
        log_error("unexpected clGetGLContextInfoKHR result");
        return TEST_FAIL;
    }

    // comparability test for CL_DEVICES_FOR_GL_CONTEXT_KHR
    FindDeviceFunctor fdf;
    for (auto &did : devices)
        if (fdf.find(did) != 0) return TEST_FAIL;

    // get current device associated with GL environment
    error = clGetGLContextInfoKHR(
        props.data(), CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
        devices.size() * sizeof(cl_device_id), devices.data(), &dev_size);
    test_error(
        error,
        "clGetGLContextInfoKHR(CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR) failed");

    // verify if CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR query result with only one
    // device
    if (dev_size != sizeof(cl_device_id))
    {
        log_info("GL _context current device is not a CL device.\n");
        return TEST_FAIL;
    }

    // comparability test for CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR
    test_assert_error(device == devices[0],
                      "Unexpected result returned by clGetGLContextInfo for "
                      "CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR query");

    return TEST_PASS;
}
