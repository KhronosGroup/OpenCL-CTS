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

#include <vector>

REGISTER_TEST(negative_create_command_queue)
{
    cl_int err = 0;
    clCreateCommandQueue(nullptr, device, 0, &err);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clCreateCommandQueue should return CL_INVALID_CONTEXT when: \"context "
        "is not a valid context\" using a nullptr",
        TEST_FAIL);

    clCreateCommandQueue(context, nullptr, 0, &err);
    test_failure_error_ret(
        err, CL_INVALID_DEVICE,
        "clCreateCommandQueue should return CL_INVALID_DEVICE when: \"device "
        "is not a valid device\" using a nullptr",
        TEST_FAIL);

    cl_device_id different_device = GetOpposingDevice(device);
    if (different_device && device != different_device)
    {
        clCreateCommandQueue(context, different_device, 0, &err);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE,
            "clCreateCommandQueue should return CL_INVALID_DEVICE when: "
            "\"device is not associated with context\"",
            TEST_FAIL);
    }

    cl_queue_properties invalid_property{ static_cast<cl_queue_properties>(
        ~CL_QUEUE_ALL_PROPERTIES) };
    clCreateCommandQueue(context, device, invalid_property, &err);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clCreateCommandQueue should return CL_INVALID_VALUE when: \"values "
        "specified in properties are not valid\"",
        TEST_FAIL);

    cl_command_queue_properties device_queue_properties = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                          sizeof(device_queue_properties),
                          &device_queue_properties, nullptr);
    test_error(err, "clGetDeviceInfo");
    cl_command_queue_properties valid_properties[] = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE
    };
    cl_queue_properties property{ 0 };
    bool missing_property = false;
    // Iterate through all possible properties to find one which isn't supported
    for (auto prop : valid_properties)
    {
        if ((device_queue_properties & prop) == 0)
        {
            missing_property = true;
            property = prop;
            break;
        }
    }
    // This test can only run when a device does not support a property
    if (missing_property)
    {
        clCreateCommandQueue(context, device, property, &err);
        test_failure_error_ret(
            err, CL_INVALID_QUEUE_PROPERTIES,
            "clCreateCommandQueue should return CL_INVALID_QUEUE_PROPERTIES "
            "when: \"values specified in properties are valid but are not "
            "supported by the device\"",
            TEST_FAIL);
    }

    return TEST_PASS;
}

REGISTER_TEST_VERSION(negative_create_command_queue_with_properties,
                      Version(2, 0))
{
    cl_int err = 0;
    clCreateCommandQueueWithProperties(nullptr, device, nullptr, &err);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clCreateCommandQueueWithProperties should return CL_INVALID_CONTEXT "
        "when: \"context is not a valid context\" using a nullptr",
        TEST_FAIL);

    clCreateCommandQueueWithProperties(context, nullptr, nullptr, &err);
    test_failure_error_ret(
        err, CL_INVALID_DEVICE,
        "clCreateCommandQueueWithProperties should return CL_INVALID_DEVICE "
        "when: \"device is not a valid device\" using a nullptr",
        TEST_FAIL);

    cl_device_id different_device = GetOpposingDevice(device);
    if (different_device && device != different_device)
    {
        clCreateCommandQueueWithProperties(context, different_device, nullptr,
                                           &err);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE,
            "clCreateCommandQueueWithProperties should return "
            "CL_INVALID_DEVICE when: \"device is not associated with context\"",
            TEST_FAIL);
    }

    cl_queue_properties invalid_property{ static_cast<cl_queue_properties>(
        -1) };

    // Depending on the OpenCL Version, there can be up to 2 properties which
    // each take values, and the list should be terminated with a 0
    cl_queue_properties properties[] = { invalid_property, invalid_property, 0,
                                         0, 0 };
    clCreateCommandQueueWithProperties(context, device, properties, &err);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clCreateCommandQueueWithProperties should return CL_INVALID_VALUE "
        "when: \"values specified in properties are not valid\"",
        TEST_FAIL);

    cl_command_queue_properties device_queue_properties = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                          sizeof(device_queue_properties),
                          &device_queue_properties, nullptr);
    test_error(err, "clGetDeviceInfo");
    cl_command_queue_properties valid_properties[] = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE
    };
    properties[0] = CL_QUEUE_PROPERTIES;
    bool missing_property = false;
    // Iterate through all possible properties to find one which isn't supported
    for (auto property : valid_properties)
    {
        if ((device_queue_properties & property) == 0)
        {
            missing_property = true;
            properties[1] = property;
            break;
        }
    }
    if (missing_property)
    {
        clCreateCommandQueueWithProperties(context, device, properties, &err);
        test_failure_error_ret(
            err, CL_INVALID_QUEUE_PROPERTIES,
            "clCreateCommandQueueWithProperties should return "
            "CL_INVALID_QUEUE_PROPERTIES when: \"values specified in "
            "properties are valid but are not supported by the device\"",
            TEST_FAIL);
    }
    else if (get_device_cl_version(device) >= Version(2, 0))
    {
        cl_uint max_size = -1;
        err = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                              sizeof(max_size), &max_size, nullptr);
        test_error(err, "clGetDeviceInfo");
        if (max_size > 0 && max_size < CL_UINT_MAX)
        {
            properties[0] = CL_QUEUE_PROPERTIES;
            properties[1] =
                CL_QUEUE_ON_DEVICE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            properties[2] = CL_QUEUE_SIZE;
            properties[3] = max_size + 1;
            clCreateCommandQueueWithProperties(context, device, properties,
                                               &err);
            if (err != CL_INVALID_VALUE && err != CL_INVALID_QUEUE_PROPERTIES)
            {
                log_error("ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
                          "clCreateCommandQueueWithProperties should return "
                          "CL_INVALID_VALUE or CL_INVALID_QUEUE_PROPERTIES "
                          "when: \"values specified in properties are not "
                          "valid\" using a queue size greather than "
                          "CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE",
                          IGetErrorString(err),
                          "CL_INVALID_VALUE or CL_INVALID_QUEUE_PROPERTIES",
                          __FILE__, __LINE__);
                return TEST_FAIL;
            }
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(negative_create_command_queue_with_properties_khr)
{
    if (!is_extension_available(device, "cl_khr_create_command_queue"))
    {
        return TEST_SKIPPED_ITSELF;
    }

    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
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
    error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
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
        clCreateCommandQueueWithPropertiesKHR(context, device, queue_prop_def,
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

REGISTER_TEST_VERSION(negative_set_default_device_command_queue, Version(2, 1))
{
    cl_int err = 0;
    if (get_device_cl_version(device) >= Version(3, 0))
    {
        cl_device_device_enqueue_capabilities device_capabilities = 0;
        cl_int err = clGetDeviceInfo(
            device, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
            sizeof(device_capabilities), &device_capabilities, nullptr);
        test_error(err, "clGetDeviceInfo");
        if (((device_capabilities & CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT) == 0)
            && ((device_capabilities & CL_DEVICE_QUEUE_SUPPORTED) == 1))
        {
            const cl_queue_properties properties[] = {
                CL_QUEUE_PROPERTIES,
                CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_ON_DEVICE
                    | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                0
            };
            clCommandQueueWrapper cmd_queue =
                clCreateCommandQueueWithProperties(context, device, properties,
                                                   &err);
            test_error(err, "clCreateCommandQueueWithProperties");
            err = clSetDefaultDeviceCommandQueue(context, device, cmd_queue);
            test_failure_error_ret(
                err, CL_INVALID_OPERATION,
                "clSetDefaultDeviceCommandQueue should return "
                "CL_INVALID_OPERATION when \"device does not support a "
                "replaceable default on-device queue\"",
                TEST_FAIL);
        }
    }

    err = clSetDefaultDeviceCommandQueue(nullptr, device, queue);
    if (err != CL_INVALID_OPERATION && err != CL_INVALID_CONTEXT)
    {
        log_error("ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
                  "clSetDefaultDeviceCommandQueue should return "
                  "CL_INVALID_OPERATION or CL_INVALID_CONTEXT when: \"context "
                  "is not a valid context\" using a nullptr",
                  IGetErrorString(err),
                  "CL_INVALID_OPERATION or CL_INVALID_CONTEXT", __FILE__,
                  __LINE__);
        return TEST_FAIL;
    }

    err = clSetDefaultDeviceCommandQueue(context, nullptr, queue);
    if (err != CL_INVALID_OPERATION && err != CL_INVALID_DEVICE)
    {
        log_error("ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
                  "clSetDefaultDeviceCommandQueue should return "
                  "CL_INVALID_OPERATION or CL_INVALID_DEVICE when: \"device "
                  "is not a valid device\" using a nullptr",
                  IGetErrorString(err),
                  "CL_INVALID_OPERATION or CL_INVALID_DEVICE", __FILE__,
                  __LINE__);
        return TEST_FAIL;
    }

    cl_device_id different_device = GetOpposingDevice(device);
    if (different_device && device != different_device)
    {
        err = clSetDefaultDeviceCommandQueue(context, different_device, queue);
        if (err != CL_INVALID_OPERATION && err != CL_INVALID_DEVICE)
        {
            log_error("ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
                      "clSetDefaultDeviceCommandQueue should return "
                      "CL_INVALID_OPERATION or CL_INVALID_DEVICE when: "
                      "\"device is not associated with context\"",
                      IGetErrorString(err),
                      "CL_INVALID_OPERATION or CL_INVALID_DEVICE", __FILE__,
                      __LINE__);
            return TEST_FAIL;
        }
    }
    err = clSetDefaultDeviceCommandQueue(context, device, nullptr);
    if (err != CL_INVALID_OPERATION && err != CL_INVALID_COMMAND_QUEUE)
    {
        log_error(
            "ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
            "clSetDefaultDeviceCommandQueue should return CL_INVALID_OPERATION "
            "or CL_INVALID_COMMAND_QUEUE when: \"command_queue is not a valid "
            "command-queue for device\" using a nullptr",
            IGetErrorString(err),
            "CL_INVALID_OPERATION or CL_INVALID_COMMAND_QUEUE", __FILE__,
            __LINE__);
        return TEST_FAIL;
    }

    {
        constexpr cl_queue_properties props[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
        };
        clCommandQueueWrapper not_on_device_queue =
            clCreateCommandQueueWithProperties(context, device, props, &err);
        test_error_fail(err, "clCreateCommandQueueWithProperties failed");
        err = clSetDefaultDeviceCommandQueue(context, device,
                                             not_on_device_queue);
        if (err != CL_INVALID_OPERATION && err != CL_INVALID_COMMAND_QUEUE)
        {
            log_error("ERROR: %s! (Got %s, expected (%s) from %s:%d)\n",
                      "clSetDefaultDeviceCommandQueue should return "
                      "CL_INVALID_OPERATION or CL_INVALID_COMMAND_QUEUE when: "
                      "\"command_queue is not a valid command-queue for "
                      "device\" using a command queue that is not on device",
                      IGetErrorString(err),
                      "CL_INVALID_OPERATION or CL_INVALID_COMMAND_QUEUE",
                      __FILE__, __LINE__);
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(negative_retain_command_queue)
{
    cl_int err = clRetainCommandQueue(nullptr);
    test_failure_error_ret(
        err, CL_INVALID_COMMAND_QUEUE,
        "clRetainCommandQueue should return CL_INVALID_COMMAND_QUEUE when: "
        "\"command_queue is not a valid command-queue\" using a nullptr",
        TEST_FAIL);

    return TEST_PASS;
}

REGISTER_TEST(negative_release_command_queue)
{
    cl_int err = clReleaseCommandQueue(nullptr);
    test_failure_error_ret(
        err, CL_INVALID_COMMAND_QUEUE,
        "clReleaseCommandQueue should return CL_INVALID_COMMAND_QUEUE when: "
        "\"command_queue is not a valid command-queue\" using a nullptr",
        TEST_FAIL);

    return TEST_PASS;
}

static bool device_supports_on_device_queue(cl_device_id deviceID)
{
    cl_command_queue_properties device_queue_properties = 0;
    if (get_device_cl_version(deviceID) >= Version(2, 0))
    {
        cl_int err = clGetDeviceInfo(
            deviceID, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
            sizeof(device_queue_properties), &device_queue_properties, nullptr);
        test_error(err, "clGetDeviceInfo");
        return (device_queue_properties
                & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    }
    return false;
}

REGISTER_TEST(negative_get_command_queue_info)
{
    cl_int err =
        clGetCommandQueueInfo(nullptr, CL_QUEUE_CONTEXT, 0, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_COMMAND_QUEUE,
        "clGetCommandQueueInfo should return CL_INVALID_COMMAND_QUEUE when: "
        "\"command_queue is not a valid command-queue\" using a nullptr",
        TEST_FAIL);

    if (device_supports_on_device_queue(device))
    {
        const cl_queue_properties properties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0
        };
        cl_int err = CL_INVALID_VALUE;
        clCommandQueueWrapper cmd_queue = clCreateCommandQueueWithProperties(
            context, device, properties, &err);
        test_error(err, "clCreateCommandQueueWithProperties");
        cl_uint queue_size = -1;
        err = clGetCommandQueueInfo(cmd_queue, CL_QUEUE_SIZE,
                                    sizeof(queue_size), &queue_size, nullptr);
        test_failure_error_ret(err, CL_INVALID_COMMAND_QUEUE,
                               "clGetCommandQueueInfo should return "
                               "CL_INVALID_COMMAND_QUEUE when: \"command_queue "
                               "is not a valid command-queue for param_name\"",
                               TEST_FAIL);
    }

    constexpr cl_command_queue_info invalid_param = -1;
    err = clGetCommandQueueInfo(queue, invalid_param, 0, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetCommandQueueInfo should return CL_INVALID_VALUE when: "
        "\"param_name is not one of the supported values\"",
        TEST_FAIL);


    cl_uint ref_count = -1;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_REFERENCE_COUNT, 0, &ref_count,
                                nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetCommandQueueInfo should return CL_INVALID_VALUE when: \"size in "
        "bytes specified by param_value_size is < size of return type and "
        "param_value is not a NULL value\"",
        TEST_FAIL);

    return TEST_PASS;
}

REGISTER_TEST_VERSION(negative_set_command_queue_property, Version(1, 0))
{
    auto version = get_device_cl_version(device);
    if (version >= Version(1, 1))
    {
        // Implementations are allowed to return an error for
        // non-OpenCL 1.0 devices. In which case, skip the test.
        return TEST_SKIPPED_ITSELF;
    }

    cl_queue_properties property{ CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE };
    cl_int err = clSetCommandQueueProperty(nullptr, property, CL_TRUE, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_COMMAND_QUEUE,
        "clSetCommandQueueProperty should return CL_INVALID_COMMAND_QUEUE "
        "when: \"command_queue is not a valid command-queue\" using a nullptr",
        TEST_FAIL);

    property = -1;
    err = clSetCommandQueueProperty(queue, property, CL_TRUE, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clSetCommandQueueProperty should return CL_INVALID_VALUE when: "
        "\"values specified in properties are not valid\"",
        TEST_FAIL);

    return TEST_PASS;
}
