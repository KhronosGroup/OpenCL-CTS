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
#include "testBase.h"
#include "harness/propertyHelpers.h"
#include "harness/typeWrappers.h"
#include <vector>
#include <algorithm>

struct test_queue_array_properties_data
{
    std::vector<cl_queue_properties> properties;
    std::string description;
};

int verify_if_properties_supported(
    cl_device_id device, cl_command_queue_properties requested_bitfield,
    cl_uint requested_size)
{
    int error = CL_SUCCESS;
    bool on_host_queue = true;

    if (requested_bitfield & CL_QUEUE_ON_DEVICE)
    {
        on_host_queue = false;

        if (requested_size > 0)
        {
            cl_uint max_queue_size = 0;
            error =
                clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                                sizeof(max_queue_size), &max_queue_size, NULL);
            test_error(error,
                       "clGetDeviceInfo for "
                       "CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE failed");
            if (requested_size > max_queue_size)
            {
                log_info(
                    "The value of CL_QUEUE_SIZE = %d cannot be bigger than "
                    "CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = %d, skipped\n",
                    requested_size, max_queue_size);
                return TEST_SKIPPED_ITSELF;
            }
        }
    }

    cl_command_queue_properties supported_properties = 0;
    cl_command_queue_properties all_properties = 0;

    std::vector<cl_command_queue_properties> all_properties_vector{
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE
    };
    for (auto each_property : all_properties_vector)
    {
        all_properties |= each_property;
    }
    cl_command_queue_properties requested_properties =
        all_properties & requested_bitfield;

    if (on_host_queue)
    {
        error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                                sizeof(supported_properties),
                                &supported_properties, NULL);
        test_error(error,
                   "clGetDeviceInfo asking for "
                   "CL_DEVICE_QUEUE_ON_HOST_PROPERTIES failed");
    }
    else
    {
        error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                                sizeof(supported_properties),
                                &supported_properties, NULL);
        test_error(error,
                   "clGetDeviceInfo asking for "
                   "CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES failed");
    }

    for (auto each_property : all_properties_vector)
    {
        if ((each_property & requested_properties)
            && !(each_property & supported_properties))
        {
            log_info("\t%s not supported, skipped\n",
                     GetQueuePropertyName(each_property));
            return TEST_SKIPPED_ITSELF;
        }
        else if ((each_property & requested_properties)
                 && each_property & supported_properties)
        {
            log_info("\t%s supported\n", GetQueuePropertyName(each_property));
        }
    }
    return error;
}

static int create_queue_and_check_array_properties(
    cl_context context, cl_device_id device,
    test_queue_array_properties_data test_case)
{
    cl_int error = CL_SUCCESS;

    clCommandQueueWrapper test_queue;

    if (test_case.properties.size() > 0)
    {
        test_queue = clCreateCommandQueueWithProperties(
            context, device, test_case.properties.data(), &error);
        test_error(error, "clCreateCommandQueueWithProperties failed");
    }
    else
    {
        test_queue =
            clCreateCommandQueueWithProperties(context, device, NULL, &error);
        test_error(error, "clCreateCommandQueueWithProperties failed");
    }

    std::vector<cl_queue_properties> check_properties;
    size_t set_size = 0;

    error = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES_ARRAY, 0,
                                  NULL, &set_size);
    test_error(error,
               "clGetCommandQueueInfo failed asking for "
               "CL_QUEUE_PROPERTIES_ARRAY size.");

    if (set_size == 0 && test_case.properties.size() == 0)
    {
        return TEST_PASS;
    }
    if (set_size != test_case.properties.size() * sizeof(cl_queue_properties))
    {
        log_error(
            "ERROR: CL_QUEUE_PROPERTIES_ARRAY size is %zu, expected %zu.\n",
            set_size,
            test_case.properties.size() * sizeof(cl_queue_properties));
        return TEST_FAIL;
    }

    cl_uint number_of_props = set_size / sizeof(cl_queue_properties);
    check_properties.resize(number_of_props);
    error = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES_ARRAY,
                                  set_size, check_properties.data(), NULL);
    test_error(
        error,
        "clGetCommandQueueInfo failed asking for CL_QUEUE_PROPERTIES_ARRAY.");

    error = compareProperties(check_properties, test_case.properties);
    return error;
}

static int
run_test_queue_array_properties(cl_context context, cl_device_id device,
                                test_queue_array_properties_data test_case)
{
    int error = TEST_PASS;

    std::vector<cl_queue_properties> requested_properties =
        test_case.properties;
    log_info("\nTC description: %s\n", test_case.description.c_str());

    // first verify if user properties are supported
    if (requested_properties.size() != 0)
    {
        requested_properties.pop_back();
        cl_command_queue_properties requested_bitfield = 0;
        cl_uint requested_size = 0;
        for (cl_uint i = 0; i < requested_properties.size(); i = i + 2)
        {
            if (requested_properties[i] == CL_QUEUE_PROPERTIES)
            {
                requested_bitfield = requested_properties[i + 1];
            }
            if (requested_properties[i] == CL_QUEUE_SIZE)
            {
                requested_size = requested_properties[i + 1];
            }
        }

        error = verify_if_properties_supported(device, requested_bitfield,
                                               requested_size);
        if (error == TEST_SKIPPED_ITSELF)
        {
            log_info("TC result: skipped\n");
            return TEST_PASS;
        }
        test_error(error,
                   "Checking which queue properties supported failed.\n");
    }

    // continue testing if supported user properties
    error = create_queue_and_check_array_properties(context, device, test_case);
    test_error(error, "create_queue_and_check_array_properties failed.\n");

    log_info("TC result: passed\n");
    return TEST_PASS;
}

REGISTER_TEST_VERSION(queue_properties_queries, Version(3, 0))
{
    int error = TEST_PASS;
    std::vector<test_queue_array_properties_data> test_cases;

    test_cases.push_back({ {}, "host queue, NULL properties" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES, 0, 0 }, "host queue, zero properties" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 },
          "host queue, CL_QUEUE_PROFILING_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 },
          "host queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
            0 },
          "host queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | "
          "CL_QUEUE_PROFILING_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE, 0 },
          "device queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | "
          "CL_QUEUE_ON_DEVICE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE, 124, 0 },
          "device queue, all possible properties" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE, 124, 0 },
          "device queue, all without CL_QUEUE_ON_DEVICE_DEFAULT" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE,
            0 },
          "device queue, all without CL_QUEUE_SIZE" });

    for (auto test_case : test_cases)
    {
        error |= run_test_queue_array_properties(context, device, test_case);
    }
    return error;
}

REGISTER_TEST(set_command_queue_property)
{
    int err;

    // Required minimum command queue properties
    std::vector<cl_command_queue_properties> queue_property_options = {
        0, CL_QUEUE_PROFILING_ENABLE
    };

    // Add other supported properties combinations
    cl_command_queue_properties supported_queue_props;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                    sizeof(supported_queue_props), &supported_queue_props,
                    NULL);
    if (supported_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
        queue_property_options.push_back(
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        queue_property_options.push_back(
            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    }

    // Initialise each queue with a different set of properties.
    for (cl_command_queue_properties initial_properties :
         queue_property_options)
    {
        clCommandQueueWrapper test_queue =
            clCreateCommandQueue(context, device, initial_properties, &err);
        test_error(err, "clCreateCommandQueue failed");

        cl_command_queue_properties old_properties, set_properties,
            current_properties = initial_properties;

        // Test clSetCommandQueueProperty with each set of properties, ignoring
        // 0 as a property.
        for (size_t i = 1; i < queue_property_options.size(); i++)
        {
            set_properties = queue_property_options[i];
            err = clSetCommandQueueProperty(test_queue, set_properties,
                                            CL_FALSE, &old_properties);
            if (err == CL_INVALID_OPERATION)
            {
                // Implementations are allowed to return an error for
                // non-OpenCL 1.0 devices. In which case, skip the test.
                return TEST_SKIPPED_ITSELF;
            }

            test_error(err, "clSetCommandQueueProperty failed");
            test_assert_error(old_properties == current_properties,
                              "The old properties for this command queue were "
                              "not as expected");

            err = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES,
                                        sizeof(cl_queue_properties),
                                        &current_properties, NULL);
            test_error(err, "clGetCommandQueueInfo failed");
            test_assert_error(current_properties
                                  == (old_properties & ~set_properties),
                              "The current properties for this command queue "
                              "were not as expected");

            err = clSetCommandQueueProperty(test_queue, set_properties, CL_TRUE,
                                            &old_properties);
            test_error(err, "clSetCommandQueueProperty failed");
            test_assert_error(old_properties == current_properties,
                              "The old properties for this command queue were "
                              "not as expected");

            err = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES,
                                        sizeof(cl_queue_properties),
                                        &current_properties, NULL);
            test_error(err, "clGetCommandQueueInfo failed");
            test_assert_error(current_properties
                                  == (set_properties | old_properties),
                              "The current properties for this command queue "
                              "were not as expected");
        }
    }

    return TEST_PASS;
}
