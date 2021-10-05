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
#include "harness/compat.h"

#include <array>
#include <bitset>

#include "harness/testHarness.h"
#include "harness/deviceInfo.h"

using uuid = std::array<cl_uchar, CL_UUID_SIZE_KHR>;
using luid = std::array<cl_uchar, CL_LUID_SIZE_KHR>;

template <typename T> static void log_info_uuid(const T &id)
{
    for (const cl_uchar c : id)
    {
        log_info("%02x", static_cast<unsigned>(c));
    }
}

template <typename T> static void log_error_uuid(const T &id)
{
    for (const cl_uchar c : id)
    {
        log_error("%02x", static_cast<unsigned>(c));
    }
}

static bool check_device_info_returns(const cl_int err, const size_t size,
                                      const size_t expected_size)
{
    if (err != CL_SUCCESS)
    {
        print_error(err, "clGetDeviceInfo failed");
        return false;
    }
    else if (size != expected_size)
    {
        log_error("Invalid size written by clGetDeviceInfo (%zu != %zu)\n",
                  size, expected_size);
        return false;
    }

    return true;
}

template <typename T>
static bool get_uuid(const cl_device_id device, const cl_device_info info,
                     T &id, const bool twice = true)
{
    const size_t id_size = id.size() * sizeof(id[0]);

    size_t size_ret;
    cl_int err = clGetDeviceInfo(device, info, id_size, id.data(), &size_ret);
    if (!check_device_info_returns(err, size_ret, id_size))
    {
        return false;
    }

    /* Check that IDs are (at the very least) stable across two successive
     * clGetDeviceInfo calls. Check conditionally, as it is undefined what the
     * query for CL_DEVICE_LUID_KHR returns if CL_DEVICE_LUID_VALID_KHR returns
     * false. */
    if (twice)
    {
        T id_2;
        size_t size_ret_2;
        err = clGetDeviceInfo(device, info, id_size, id_2.data(), &size_ret_2);
        if (!check_device_info_returns(err, size_ret_2, id_size))
        {
            return false;
        }

        if (id != id_2)
        {
            log_error("Got different IDs from the same ID device info (");
            log_error_uuid(id);
            log_error(" != ");
            log_error_uuid(id_2);
            log_error(")\n");
            return false;
        }
    }

    return true;
}

int test_device_uuid(cl_device_id deviceID, cl_context context,
                     cl_command_queue ignoreQueue, int num_elements)
{
    if (!is_extension_available(deviceID, "cl_khr_device_uuid"))
    {
        log_info("cl_khr_device_uuid not supported. Skipping test...\n");
        return TEST_SKIPPED_ITSELF;
    }

    int total_errors = 0;

    /* CL_DEVICE_UUID_KHR */
    uuid device_uuid;
    bool success = get_uuid(deviceID, CL_DEVICE_UUID_KHR, device_uuid);
    if (!success)
    {
        log_error("Error getting device UUID\n");
        ++total_errors;
    }
    else
    {
        log_info("\tDevice UUID: ");
        log_info_uuid(device_uuid);
        log_info("\n");
    }

    /* CL_DRIVER_UUID_KHR */
    uuid driver_uuid;
    success = get_uuid(deviceID, CL_DRIVER_UUID_KHR, driver_uuid);
    if (!success)
    {
        log_error("Error getting driver UUID\n");
        ++total_errors;
    }
    else
    {
        log_info("\tDriver UUID: ");
        log_info_uuid(driver_uuid);
        log_info("\n");
    }

    size_t size_ret{};

    /* CL_DEVICE_LUID_VALID_KHR */
    cl_bool device_luid_valid{};
    cl_int err = clGetDeviceInfo(deviceID, CL_DEVICE_LUID_VALID_KHR,
                                 sizeof(device_luid_valid), &device_luid_valid,
                                 &size_ret);
    if (!check_device_info_returns(err, size_ret, sizeof(device_luid_valid)))
    {
        log_error("Error getting device LUID validity\n");
        ++total_errors;
        device_luid_valid = false;
    }
    else
    {
        log_info("\tDevice LUID validity is %s\n",
                 device_luid_valid ? "true" : "false");
    }

    /* CL_DEVICE_LUID_KHR */
    luid device_luid;
    success =
        get_uuid(deviceID, CL_DEVICE_LUID_KHR, device_luid, device_luid_valid);
    if (!success)
    {
        log_error("Error getting device LUID\n");
        ++total_errors;
    }
    else
    {
        log_info("\tDevice LUID: ");
        log_info_uuid(device_luid);
        log_info("\n");
    }

    /* CL_DEVICE_NODE_MASK_KHR */
    cl_uint device_node_mask{};
    err =
        clGetDeviceInfo(deviceID, CL_DEVICE_NODE_MASK_KHR,
                        sizeof(device_node_mask), &device_node_mask, &size_ret);
    if (!check_device_info_returns(err, size_ret, sizeof(device_node_mask)))
    {
        log_error("Error getting device node mask\n");
        ++total_errors;
    }
    else
    {
        log_info("\tNode mask  : %08lx\n",
                 static_cast<unsigned long>(device_node_mask));

        /* If the LUID is valid, there must be one and only one bit set in the
         * node mask */
        if (device_luid_valid)
        {
            static constexpr size_t cl_uint_size_in_bits = 32;
            const size_t bit_count =
                std::bitset<cl_uint_size_in_bits>(device_node_mask).count();
            if (1 != bit_count)
            {
                log_error("Wrong amount of bits set in node mask (%zu != 1) "
                          "with valid LUID\n",
                          bit_count);
                ++total_errors;
            }
        }
    }

    return total_errors;
}
