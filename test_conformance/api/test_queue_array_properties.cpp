//
// Copyright (c) 2020 The Khronos Group Inc.
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
#include "harness/typeWrappers.h"
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct test_queue_array_properties_data
{
    std::vector<cl_queue_properties> properties;
    std::string description;
};

int verify_if_properties_supported(
    cl_device_id deviceID, cl_command_queue_properties requested_bitfield,
    cl_uint requested_size)
{
    int error = CL_SUCCESS;
    cl_command_queue_properties on_host_properties = 0;
    cl_device_device_enqueue_capabilities device_enqueue_caps;
    std::vector<cl_command_queue_properties> all_properties = {
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE,
        CL_QUEUE_ON_DEVICE, CL_QUEUE_ON_DEVICE_DEFAULT
    };

    error =
        clGetDeviceInfo(deviceID, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                        sizeof(on_host_properties), &on_host_properties, NULL);
    test_error(error,
               "clGetDeviceInfo for CL_DEVICE_QUEUE_ON_HOST_PROPERTIES failed");

    error = clGetDeviceInfo(deviceID, CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                            sizeof(device_enqueue_caps), &device_enqueue_caps,
                            NULL);
    test_error(
        error,
        "clGetDeviceInfo for CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES failed");
    if (!(device_enqueue_caps & CL_DEVICE_QUEUE_SUPPORTED)
        && requested_size > 0)
    {
        log_info("\tCL_DEVICE_QUEUE_SUPPORTED false, skipped\n");
        return TEST_SKIPPED_ITSELF;
    }
    for (auto check_property : all_properties)
    {
        if (check_property & requested_bitfield)
        {
            if (check_property & on_host_properties)
            {
                log_info("\t%s supported\n",
                         GetQueuePropertyName(check_property));
            }
            else if (device_enqueue_caps & CL_DEVICE_QUEUE_SUPPORTED)
            {
                if ((check_property & CL_QUEUE_ON_DEVICE)
                    || (check_property & CL_QUEUE_ON_DEVICE_DEFAULT))
                {
                    log_info("\t%s supported\n",
                             GetQueuePropertyName(check_property));
                }
            }
            else
            {
                log_info("\t%s not supported, skipped\n",
                         GetQueuePropertyName(check_property));
                return TEST_SKIPPED_ITSELF;
            }
        }
    }
    return error;
}

int create_queue_and_check_array_properties(
    cl_context context, cl_device_id deviceID,
    clCommandQueueWrapper& test_queue,
    test_queue_array_properties_data test_case)
{
    int error = CL_SUCCESS;
    if (test_case.properties.size() > 0)
    {
        test_queue = clCreateCommandQueueWithProperties(
            context, deviceID, test_case.properties.data(), &error);
        test_error(error, "clCreateCommandQueueWithProperties failed");
    }
    else
    {
        test_queue =
            clCreateCommandQueueWithProperties(context, deviceID, NULL, &error);
        test_error(error, "clCreateCommandQueueWithProperties failed");
    }
    std::vector<cl_queue_properties> get_properties;
    size_t set_size = 0;

    error = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES_ARRAY, 0,
                                  NULL, &set_size);
    test_error(
        error,
        "clGetCommandQueueInfo failed asking for CL_QUEUE_PROPERTIES_ARRAY");

    if (test_case.properties.size() == 0 && set_size == 0)
    {
        return TEST_PASS;
    }

    cl_uint number_of_props = set_size / sizeof(cl_queue_properties);
    get_properties.resize(number_of_props);
    error = clGetCommandQueueInfo(test_queue, CL_QUEUE_PROPERTIES_ARRAY,
                                  set_size, get_properties.data(), NULL);
    test_error(
        error,
        "clGetCommandQueueInfo failed asking for CL_QUEUE_PROPERTIES_ARRAY");

    if (get_properties.size() == 1 && get_properties[0] == 0
        && test_case.properties.size() == 0)
    {
        return TEST_PASS;
    }

    if (get_properties.back() != 0)
    {
        log_error("ERROR: Incorrect last property value - should be 0!\n");
        return TEST_FAIL;
    }
    get_properties.pop_back();
    test_case.properties.pop_back();

    if (get_properties != test_case.properties)
    {
        for (cl_uint i = 0; i < test_case.properties.size(); i = i + 2)
        {
            cl_queue_properties set_property = test_case.properties[i];
            cl_queue_properties set_property_value =
                test_case.properties[i + 1];

            std::vector<cl_mem_properties>::iterator it = std::find(
                get_properties.begin(), get_properties.end(), set_property);

            if (it == get_properties.end())
            {
                log_error("ERROR: Property not found ... 0x%x\n", set_property);
                return TEST_FAIL;
            }
            else
            {
                if (set_property_value != *std::next(it))
                {
                    log_error("ERROR: Incorrect preperty value expected %x, "
                              "obtained %x\n",
                              set_property_value, *std::next(it));
                    return TEST_FAIL;
                }
            }
        }
        log_error("ERROR: ALL properties and values matched but order "
                  "incorrect!\n");
        return TEST_FAIL;
    }
    return error;
}

int run_test_queue_array_properties(cl_context context, cl_device_id deviceID,
                                    test_queue_array_properties_data test_case)
{
    int error = CL_SUCCESS;
    clCommandQueueWrapper queue;
    std::vector<cl_queue_properties> requested_properties =
        test_case.properties;

    // first verify if user properties are supported
    if (requested_properties.size() != 0)
    {
        requested_properties.pop_back();
        cl_command_queue_properties requested_bitfield;
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

        error = verify_if_properties_supported(deviceID, requested_bitfield,
                                               requested_size);
        if (error == TEST_SKIPPED_ITSELF)
        {
            log_info("TC result: skipped\n");
            return CL_SUCCESS;
        }
        test_error(error,
                   "Checking which queue properites supported failed.\n");
    }

    // continue testing if supported user properties
    error = create_queue_and_check_array_properties(context, deviceID, queue,
                                                    test_case);
    test_error(error, "Queue properties array verification result failed.\n");

    log_info("TC result: passed\n");
    return TEST_PASS;
}

int test_queue_array_properties(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    int error = CL_SUCCESS;
    std::vector<test_queue_array_properties_data> test_cases;

    test_cases.push_back({ {}, "queue, NULL properties" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
            0 },
          "queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | "
          "CL_QUEUE_PROFILING_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 },
          "queue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 },
          "queue, CL_QUEUE_PROFILING_ENABLE" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE, 124, 0 },
          "queue, all possible properties" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_PROFILING_ENABLE,
            CL_QUEUE_SIZE, 124, 0 },
          "queue, all without CL_QUEUE_ON_DEVICE_DEFAULT" });

    test_cases.push_back(
        { { CL_QUEUE_PROPERTIES,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE
                | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_PROFILING_ENABLE,
            0 },
          "queue, all without CL_QUEUE_SIZE" });

    for (auto test_case : test_cases)
    {
        error |= run_test_queue_array_properties(context, deviceID, test_case);
    }
    return error;
}