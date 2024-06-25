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

#include "testBase.h"
#include "harness/typeWrappers.h"

int test_negative_create_command_queue(cl_device_id deviceID,
                                       cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    cl_command_queue_properties device_props = 0;
    cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_PROPERTIES,
                                   sizeof(device_props), &device_props, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

    // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE is the only optional property to
    // clCreateCommandQueue, CL_QUEUE_PROFILING_ENABLE is mandatory.
    const bool out_of_order_device_support =
        device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (out_of_order_device_support)
    {
        // Early return as we can't check correct error is returned for
        // unsupported property.
        return TEST_PASS;
    }

    // Try create a command queue with out-of-order property and check return
    // code
    cl_int test_error = CL_SUCCESS;
    clCommandQueueWrapper test_queue = clCreateCommandQueue(
        context, deviceID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &test_error);

    test_failure_error_ret(
        test_error, CL_INVALID_QUEUE_PROPERTIES,
        "clCreateCommandQueue should return CL_INVALID_QUEUE_PROPERTIES if "
        "values specified in properties are valid but are not supported by "
        "the "
        "device.",
        TEST_FAIL);
    return TEST_PASS;
}

int test_negative_create_command_queue_with_properties(cl_device_id deviceID,
                                                       cl_context context,
                                                       cl_command_queue queue,
                                                       int num_elements)
{
    cl_command_queue_properties device_props = 0;
    cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_PROPERTIES,
                                   sizeof(device_props), &device_props, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

    cl_command_queue_properties device_on_host_props = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                            sizeof(device_on_host_props), &device_on_host_props,
                            NULL);
    test_error(error,
               "clGetDeviceInfo for CL_DEVICE_QUEUE_ON_HOST_PROPERTIES failed");

    if (device_on_host_props != device_props)
    {
        log_error(
            "ERROR: CL_DEVICE_QUEUE_PROPERTIES and "
            "CL_DEVICE_QUEUE_ON_HOST_PROPERTIES properties should match\n");
        return TEST_FAIL;
    }

    // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE is the only optional host-queue
    // property to clCreateCommandQueueWithProperties,
    // CL_QUEUE_PROFILING_ENABLE is mandatory.
    const bool out_of_order_device_support =
        device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (out_of_order_device_support)
    {
        // Early return as we can't check correct error is returned for
        // unsupported property.
        return TEST_PASS;
    }

    // Try create a command queue with out-of-order property and check return
    // code
    cl_command_queue_properties queue_prop_def[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
    };

    cl_int test_error = CL_SUCCESS;
    clCommandQueueWrapper test_queue = clCreateCommandQueueWithProperties(
        context, deviceID, queue_prop_def, &test_error);

    test_failure_error_ret(test_error, CL_INVALID_QUEUE_PROPERTIES,
                           "clCreateCommandQueueWithProperties should "
                           "return CL_INVALID_QUEUE_PROPERTIES if "
                           "values specified in properties are valid but "
                           "are not supported by the "
                           "device.",
                           TEST_FAIL);

    return TEST_PASS;
}

int test_negative_create_command_queue_with_properties_khr(
    cl_device_id deviceID, cl_context context, cl_command_queue queue,
    int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_create_command_queue"))
    {
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM,
                                   sizeof(cl_platform_id), &platform, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

    clCreateCommandQueueWithPropertiesKHR_fn
        clCreateCommandQueueWithPropertiesKHR =
            (clCreateCommandQueueWithPropertiesKHR_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clCreateCommandQueueWithPropertiesKHR");
    if (clCreateCommandQueueWithPropertiesKHR == NULL)
    {
        log_error("ERROR: clGetExtensionFunctionAddressForPlatform failed\n");
        return -1;
    }

    cl_command_queue_properties device_props = 0;
    error = clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(device_props), &device_props, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

    // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE is the only optional host-queue
    // property to clCreateCommandQueueWithPropertiesKHR,
    // CL_QUEUE_PROFILING_ENABLE is mandatory.
    const bool out_of_order_device_support =
        device_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if (out_of_order_device_support)
    {
        // Early return as we can't check correct error is returned for
        // unsupported property.
        return TEST_PASS;
    }

    // Try create a command queue with out-of-order property and check return
    // code
    cl_queue_properties_khr queue_prop_def[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
    };

    cl_int test_error = CL_SUCCESS;
    clCommandQueueWrapper test_khr_queue =
        clCreateCommandQueueWithPropertiesKHR(context, deviceID, queue_prop_def,
                                              &test_error);

    test_failure_error_ret(test_error, CL_INVALID_QUEUE_PROPERTIES,
                           "clCreateCommandQueueWithPropertiesKHR should "
                           "return CL_INVALID_QUEUE_PROPERTIES if "
                           "values specified in properties are valid but "
                           "are not supported by the "
                           "device.",
                           TEST_FAIL);
    return TEST_PASS;
}
