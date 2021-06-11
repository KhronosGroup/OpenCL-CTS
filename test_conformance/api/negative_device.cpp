//
// Copyright (c) 2021 The Khronos Group Inc.
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
#include "harness/testHarness.h"
#include <vector>

/* Negative Tests for clGetDeviceInfo */
int test_negative_get_device_info(cl_device_id deviceID, cl_context context,
                                  cl_command_queue queue, int num_elements)
{

    cl_device_type device_type = 0;
    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clGetDeviceInfo(invalid_device, CL_DEVICE_TYPE,
                              sizeof(device_type), &device_type, nullptr);
        test_failure_error_ret(err, CL_INVALID_DEVICE,
                               "clGetDeviceInfo should return "
                               "CL_INVALID_DEVICE when: \"device is not "
                               "a valid device\"",
                               TEST_FAIL);
    }

    constexpr cl_device_info INVALID_PARAM_VALUE = 0;
    err = clGetDeviceInfo(deviceID, INVALID_PARAM_VALUE, 0, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetDeviceInfo should return CL_INVALID_VALUE when: \"param_name is "
        "not one of the supported values\"",
        TEST_FAIL);

    err = clGetDeviceInfo(deviceID, CL_DEVICE_TYPE, 0, &device_type, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetDeviceInfo should return CL_INVALID_VALUE when: \"size in bytes "
        "specified by param_value_size is < size of return type and "
        "param_value is not a NULL value\"",
        TEST_FAIL);

    return TEST_PASS;
}

/* Negative Tests for clGetDeviceIDs */
int test_negative_get_device_ids(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    cl_platform_id platform = getPlatformFromDevice(deviceID);

    cl_device_id devices = nullptr;

    cl_int err(CL_SUCCESS);
    for (auto invalid_platform : get_invalid_objects<cl_platform_id>(deviceID))
    {
        err = clGetDeviceIDs(invalid_platform, CL_DEVICE_TYPE_DEFAULT, 1,
                             &devices, nullptr);
        test_failure_error_ret(err, CL_INVALID_PLATFORM,
                               "clGetDeviceIDs should return "
                               "CL_INVALID_PLATFORM when: \"platform is "
                               "not a valid platform\"",
                               TEST_FAIL);
    }

    cl_device_type INVALID_DEVICE_TYPE = 0;
    err = clGetDeviceIDs(platform, INVALID_DEVICE_TYPE, 1, &devices, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_DEVICE_TYPE,
        "clGetDeviceIDs should return CL_INVALID_DEVICE_TYPE when: "
        "\"device_type is not a valid value\"",
        TEST_FAIL);

    err =
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, &devices, nullptr);
    test_failure_error_ret(err, CL_INVALID_VALUE,
                           "clGetDeviceIDs should return when: \"num_entries "
                           "is equal to zero and devices is not NULL\"",
                           TEST_FAIL);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, nullptr, nullptr);
    test_failure_error_ret(err, CL_INVALID_VALUE,
                           "clGetDeviceIDs should return CL_INVALID_VALUE "
                           "when: \"both num_devices and devices are NULL\"",
                           TEST_FAIL);

    devices = nullptr;
    std::vector<cl_device_type> device_types{ CL_DEVICE_TYPE_CPU,
                                              CL_DEVICE_TYPE_GPU,
                                              CL_DEVICE_TYPE_ACCELERATOR };
    if (get_device_cl_version(deviceID) >= Version(1, 2))
    {
        device_types.push_back(CL_DEVICE_TYPE_CUSTOM);
    }

    bool platform_supports_all_device_types = true;
    for (auto device_type : device_types)
    {
        err = clGetDeviceIDs(platform, device_type, 1, &devices, nullptr);
        if (err == CL_SUCCESS)
        {
            continue;
        }
        platform_supports_all_device_types = false;
        break;
    }
    if (platform_supports_all_device_types)
    {
        log_info("Platform has every Device Type... Skipping Test\n");
    }
    else
    {
        test_failure_error_ret(
            err, CL_DEVICE_NOT_FOUND,
            "clGetDeviceIDs should return CL_DEVICE_NOT_FOUND when: \"no "
            "OpenCL devices that matched device_type were found\"",
            TEST_FAIL);
    }

    return TEST_PASS;
}

/* Negative Tests for clGetDeviceAndHostTimer */
int test_negative_get_device_and_host_timer(cl_device_id deviceID,
                                            cl_context context,
                                            cl_command_queue queue,
                                            int num_elements)
{
    cl_ulong *device_timestamp = nullptr, *host_timestamp = nullptr;
    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clGetDeviceAndHostTimer(invalid_device, device_timestamp,
                                      host_timestamp);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE,
            "clGetDeviceAndHostTimer should return CL_INVALID_DEVICE when: "
            "\"device is not a valid device\"",
            TEST_FAIL);
    }

    cl_platform_id platform = getPlatformFromDevice(deviceID);

    // Initialise timer_resolution to a Non-0 value as CL2.1/2 devices must
    // support timer synchronisation
    cl_ulong timer_resolution = 1;
    if (get_device_cl_version(deviceID) >= Version(3, 0))
    {
        cl_int err = clGetPlatformInfo(
            platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
            sizeof(timer_resolution), &timer_resolution, nullptr);
        test_error(err, "clGetPlatformInfo");
    }

    if (timer_resolution != 0)
    {
        log_info("Platform Supports Timers\n");
        log_info("Skipping CL_INVALID_OPERATION tests\n");

        err = clGetDeviceAndHostTimer(deviceID, nullptr, host_timestamp);
        test_failure_error_ret(
            err, CL_INVALID_VALUE,
            "clGetDeviceAndHostTimer should return CL_INVALID_VALUE when: "
            "\"host_timestamp or device_timestamp is NULL\" using nullptr for "
            "device_timestamp ",
            TEST_FAIL);

        err = clGetDeviceAndHostTimer(deviceID, device_timestamp, nullptr);
        test_failure_error_ret(
            err, CL_INVALID_VALUE,
            "clGetDeviceAndHostTimer should return CL_INVALID_VALUE when: "
            "\"host_timestamp or device_timestamp is NULL\" using nullptr for "
            "host_timestamp ",
            TEST_FAIL);
    }
    else
    {
        log_info("Platform does not Support Timers\n");
        log_info("Skipping CL_INVALID_VALUE tests\n");

        err =
            clGetDeviceAndHostTimer(deviceID, device_timestamp, host_timestamp);
        test_failure_error_ret(
            err, CL_INVALID_OPERATION,
            "clGetDeviceAndHostTimer should return CL_INVALID_OPERATION when: "
            "\"the platform associated with device does not support device and "
            "host timer synchronization\"",
            TEST_FAIL);
    }

    return TEST_PASS;
}

/* Negative Tests for clGetHostTimer */
int test_negative_get_host_timer(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    cl_ulong *host_timestamp = nullptr;
    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clGetHostTimer(invalid_device, host_timestamp);
        test_failure_error_ret(err, CL_INVALID_DEVICE,
                               "clGetHostTimer should return CL_INVALID_DEVICE "
                               "when: \"device is not "
                               "a valid device\"",
                               TEST_FAIL);
    }

    cl_platform_id platform = getPlatformFromDevice(deviceID);
    // Initialise timer_resolution to a Non-0 value as CL2.1/2 devices must
    // support timer synchronisation
    cl_ulong timer_resolution = 1;
    if (get_device_cl_version(deviceID) >= Version(3, 0))
    {
        err = clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
                                sizeof(timer_resolution), &timer_resolution,
                                nullptr);
        test_error(err, "clGetPlatformInfo");
    }

    if (timer_resolution != 0)
    {
        log_info("Platform Supports Timers\n");
        log_info("Skipping CL_INVALID_OPERATION tests\n");

        err = clGetHostTimer(deviceID, nullptr);
        test_failure_error_ret(err, CL_INVALID_VALUE,
                               "clGetHostTimer should return CL_INVALID_VALUE "
                               "when: \"host_timestamp is NULL\"",
                               TEST_FAIL);
    }
    else
    {
        log_info("Platform does not Support Timers\n");
        log_info("Skipping CL_INVALID_VALUE tests\n");

        err = clGetHostTimer(deviceID, host_timestamp);
        test_failure_error_ret(
            err, CL_INVALID_OPERATION,
            "clGetHostTimer should return CL_INVALID_OPERATION when: \"the "
            "platform associated with device does not support device and host "
            "timer synchronization\"",
            TEST_FAIL);
    }

    return TEST_PASS;
}

/* Negative Tests for clCreateSubDevices */
enum SupportedPartitionSchemes
{
    None = 0,
    Equally = 1 << 0,
    Counts = 1 << 1,
    Affinity = 1 << 2,
    All_Schemes = Affinity | Counts | Equally,
};

static int get_supported_properties(cl_device_id deviceID)
{
    size_t number_of_properties = 0;
    int err = clGetDeviceInfo(deviceID, CL_DEVICE_PARTITION_PROPERTIES, 0,
                              nullptr, &number_of_properties);
    test_error(err, "clGetDeviceInfo");
    std::vector<cl_device_partition_property> supported_properties(
        number_of_properties / sizeof(cl_device_partition_property));
    err = clGetDeviceInfo(deviceID, CL_DEVICE_PARTITION_PROPERTIES,
                          number_of_properties, &supported_properties.front(),
                          nullptr);
    test_error(err, "clGetDeviceInfo");
    int ret = SupportedPartitionSchemes::None;
    for (auto property : supported_properties)
    {
        switch (property)
        {
            case CL_DEVICE_PARTITION_EQUALLY:
                ret |= SupportedPartitionSchemes::Equally;
                break;
            case CL_DEVICE_PARTITION_BY_COUNTS:
                ret |= SupportedPartitionSchemes::Counts;
                break;
            case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
                ret |= SupportedPartitionSchemes::Affinity;
                break;
            default: break;
        }
    }
    return ret;
}

static std::vector<cl_device_partition_property>
get_invalid_properties(int unsupported_properties)
{
    if (unsupported_properties & SupportedPartitionSchemes::Equally)
    {
        return { CL_DEVICE_PARTITION_EQUALLY, 1, 0 };
    }
    else if (unsupported_properties & SupportedPartitionSchemes::Counts)
    {
        return { CL_DEVICE_PARTITION_BY_COUNTS, 1,
                 CL_DEVICE_PARTITION_BY_COUNTS_LIST_END };
    }
    else if (unsupported_properties & SupportedPartitionSchemes::Affinity)
    {
        return { CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                 CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE, 0 };
    }
    else
    {
        return {};
    }
}

static cl_uint get_uint_device_info(const cl_device_id deviceID,
                                    const cl_device_info param_name)
{
    cl_uint ret = 0;
    cl_int err =
        clGetDeviceInfo(deviceID, param_name, sizeof(ret), &ret, nullptr);
    test_error(err, "clGetDeviceInfo");
    return ret;
}

int test_negative_create_sub_devices(cl_device_id deviceID, cl_context context,
                                     cl_command_queue queue, int num_elements)
{
    int supported_properties = get_supported_properties(deviceID);
    if (supported_properties == SupportedPartitionSchemes::None)
    {
        printf("Device does not support creating subdevices... Skipping\n");
        return TEST_SKIPPED_ITSELF;
    }

    cl_device_partition_property properties[4] = {};
    cl_uint max_compute_units =
        get_uint_device_info(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS);
    cl_uint max_sub_devices =
        get_uint_device_info(deviceID, CL_DEVICE_PARTITION_MAX_SUB_DEVICES);
    if (supported_properties & SupportedPartitionSchemes::Equally)
    {
        properties[0] = CL_DEVICE_PARTITION_EQUALLY;
        properties[1] = max_compute_units;
        properties[2] = 0;
    }
    else if (supported_properties & SupportedPartitionSchemes::Counts)
    {
        properties[0] = CL_DEVICE_PARTITION_BY_COUNTS;
        properties[1] = max_sub_devices;
        properties[2] = CL_DEVICE_PARTITION_BY_COUNTS_LIST_END;
    }
    else
    {
        properties[0] = CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
        properties[1] = CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE;
        properties[2] = 0;
    }

    properties[3] = 0;
    cl_device_id *out_devices = nullptr;

    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clCreateSubDevices(invalid_device, properties, 1, out_devices,
                                 nullptr);
        test_failure_error_ret(err, CL_INVALID_DEVICE,
                               "clCreateSubDevices should return "
                               "CL_INVALID_DEVICE when: \"in_device "
                               "is not a valid device\"",
                               TEST_FAIL);
    }

    err = clCreateSubDevices(deviceID, nullptr, 1, out_devices, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clCreateSubDevices should return CL_INVALID_VALUE when: \"values "
        "specified in properties are not valid\" using a nullptr",
        TEST_FAIL);

    err = clCreateSubDevices(deviceID, properties, 0, out_devices, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clCreateSubDevices should return CL_INVALID_VALUE when: \"out_devices "
        "is not NULL and num_devices is less than the number of sub-devices "
        "created by the partition scheme\"",
        TEST_FAIL);

    if (supported_properties != SupportedPartitionSchemes::All_Schemes)
    {
        std::vector<cl_device_partition_property> invalid_properties =
            get_invalid_properties(supported_properties
                                   ^ SupportedPartitionSchemes::All_Schemes);
        err = clCreateSubDevices(deviceID, invalid_properties.data(), 1,
                                 out_devices, nullptr);
        test_failure_error_ret(
            err, CL_INVALID_VALUE,
            "clCreateSubDevices should return CL_INVALID_VALUE when: \"values "
            "specified in properties are valid but not supported by the "
            "device\"",
            TEST_FAIL);
    }

    err = clCreateSubDevices(deviceID, properties, properties[1], out_devices,
                             nullptr);
    test_failure_error_ret(
        err, CL_DEVICE_PARTITION_FAILED,
        "clCreateSubDevices should return CL_DEVICE_PARTITION_FAILED when: "
        "\"if the partition name is supported by the implementation but "
        "in_device could not be further partitioned\"",
        TEST_FAIL);

    constexpr cl_device_partition_property INVALID_PARTITION_PROPERTY =
        -1; // Aribitrary Invalid number
    properties[0] = INVALID_PARTITION_PROPERTY;
    err = clCreateSubDevices(deviceID, properties, 1, out_devices, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clCreateSubDevices should return CL_INVALID_VALUE when: \"values "
        "specified in properties are not valid\" using an invalid property",
        TEST_FAIL);

    if (supported_properties & SupportedPartitionSchemes::Counts)
    {
        properties[0] = CL_DEVICE_PARTITION_BY_COUNTS;
        properties[1] = max_sub_devices + 1;
        err = clCreateSubDevices(deviceID, properties, max_sub_devices + 1,
                                 out_devices, nullptr);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE_PARTITION_COUNT,
            "clCreateSubDevices should return "
            "CL_INVALID_DEVICE_PARTITION_COUNT when: \"the partition name "
            "specified in properties is CL_DEVICE_ PARTITION_BY_COUNTS and the "
            "number of sub-devices requested exceeds "
            "CL_DEVICE_PARTITION_MAX_SUB_DEVICES\"",
            TEST_FAIL);

        properties[1] = -1;
        err =
            clCreateSubDevices(deviceID, properties, -1, out_devices, nullptr);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE_PARTITION_COUNT,
            "clCreateSubDevices should return "
            "CL_INVALID_DEVICE_PARTITION_COUNT when: \"the number of compute "
            "units requested for one or more sub-devices is less than zero\"",
            TEST_FAIL);
    }

    if (supported_properties & SupportedPartitionSchemes::Equally)
    {
        properties[0] = CL_DEVICE_PARTITION_EQUALLY;
        properties[1] = max_compute_units + 1;
        err = clCreateSubDevices(deviceID, properties, max_compute_units + 1,
                                 out_devices, nullptr);
        test_failure_error_ret(
            err, CL_INVALID_DEVICE_PARTITION_COUNT,
            "clCreateSubDevices should return "
            "CL_INVALID_DEVICE_PARTITION_COUNT when: \"the total number of "
            "compute units requested exceeds CL_DEVICE_MAX_COMPUTE_UNITS for "
            "in_device\"",
            TEST_FAIL);
    }

    return TEST_PASS;
}

/* Negative Tests for clRetainDevice */
int test_negative_retain_device(cl_device_id deviceID, cl_context context,
                                cl_command_queue queue, int num_elements)
{
    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clRetainDevice(invalid_device);
        test_failure_error_ret(err, CL_INVALID_DEVICE,
                               "clRetainDevice should return CL_INVALID_DEVICE "
                               "when: \"device is not "
                               "a valid device\"",
                               TEST_FAIL);
    }

    return TEST_PASS;
}

/* Negative Tests for clReleaseDevice */
int test_negative_release_device(cl_device_id deviceID, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    cl_int err(CL_SUCCESS);
    for (auto invalid_device : get_invalid_objects<cl_device_id>(deviceID))
    {
        err = clReleaseDevice(invalid_device);
        test_failure_error_ret(err, CL_INVALID_DEVICE,
                               "clReleaseDevice should return "
                               "CL_INVALID_DEVICE when: \"device is not "
                               "a valid device\"",
                               TEST_FAIL);
    }

    return TEST_PASS;
}
