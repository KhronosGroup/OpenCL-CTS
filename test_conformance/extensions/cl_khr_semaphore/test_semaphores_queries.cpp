//
// Copyright (c) 2023 The Khronos Group Inc.
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

#include <cinttypes>
#include "semaphore_base.h"

#define FLUSH_DELAY_S 5

#define SEMAPHORE_PARAM_TEST(param_name, param_type, format, expected)         \
    do                                                                         \
    {                                                                          \
        param_type value;                                                      \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(semaphore, param_name,            \
                                             sizeof(value), &value, &size);    \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (value != expected)                                                 \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Parameter %s did not validate! (expected " format ", " \
                "got " format ")\n",                                           \
                #param_name, expected, value);                                 \
        }                                                                      \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
    } while (false)

#define SEMAPHORE_PARAM_TEST_ARRAY(param_name, param_type, num_params,         \
                                   expected)                                   \
    do                                                                         \
    {                                                                          \
        param_type value[num_params];                                          \
        size_t size;                                                           \
        cl_int error = clGetSemaphoreInfoKHR(semaphore, param_name,            \
                                             sizeof(value), &value, &size);    \
        test_error(error, "Unable to get " #param_name " from semaphore");     \
        if (size != sizeof(value))                                             \
        {                                                                      \
            test_fail(                                                         \
                "ERROR: Returned size of parameter %s does not validate! "     \
                "(expected %d, got %d)\n",                                     \
                #param_name, (int)sizeof(value), (int)size);                   \
        }                                                                      \
        if (memcmp(value, expected, size) != 0)                                \
        {                                                                      \
            test_fail("ERROR: Parameter %s did not validate!\n", #param_name); \
        }                                                                      \
    } while (false)

namespace {

struct SemaphoreWithDeviceListQueries : public SemaphoreTestBase
{
    SemaphoreWithDeviceListQueries(cl_device_id device, cl_context context,
                                   cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;

        // Query binary semaphore created with
        // CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR

        // Create binary semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR),
            (cl_semaphore_properties_khr)device,
            CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Confirm that querying CL_SEMAPHORE_TYPE_KHR returns
        // CL_SEMAPHORE_TYPE_BINARY_KHR
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_TYPE_KHR, cl_semaphore_type_khr, "%d",
                             CL_SEMAPHORE_TYPE_BINARY_KHR);

        // Confirm that querying CL_SEMAPHORE_CONTEXT_KHR returns the right
        // context
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_CONTEXT_KHR, cl_context, "%p",
                             context);

        // Confirm that querying CL_SEMAPHORE_REFERENCE_COUNT_KHR returns the
        // right value
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, "%u",
                             1);

        err = clRetainSemaphoreKHR(semaphore);
        test_error(err, "Could not retain semaphore");
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, "%u",
                             2);

        err = clReleaseSemaphoreKHR(semaphore);
        test_error(err, "Could not release semaphore");
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_REFERENCE_COUNT_KHR, cl_uint, "%u",
                             1);

        // Confirm that querying CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR returns the
        // same device id the semaphore was created with
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR, cl_device_id,
                             "%p", device);

        // Confirm that querying CL_SEMAPHORE_PROPERTIES_KHR returns the same
        // properties the semaphore was created with
        SEMAPHORE_PARAM_TEST_ARRAY(CL_SEMAPHORE_PROPERTIES_KHR,
                                   cl_semaphore_properties_khr, 6, sema_props);

        // Confirm that querying CL_SEMAPHORE_PAYLOAD_KHR returns the unsignaled
        // state
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_PAYLOAD_KHR, cl_semaphore_payload_khr,
                             "%" PRIu64,
                             static_cast<cl_semaphore_payload_khr>(0));

        return TEST_PASS;
    }
};

struct SemaphoreNoDeviceListQueries : public SemaphoreTestBase
{
    SemaphoreNoDeviceListQueries(cl_device_id device, cl_context context,
                                 cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        cl_int err = CL_SUCCESS;

        // Query binary semaphore created without
        // CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR

        // Create binary semaphore
        cl_semaphore_properties_khr sema_props[] = {
            static_cast<cl_semaphore_properties_khr>(CL_SEMAPHORE_TYPE_KHR),
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_TYPE_BINARY_KHR),
            0
        };
        semaphore =
            clCreateSemaphoreWithPropertiesKHR(context, sema_props, &err);
        test_error(err, "Could not create semaphore");

        // Confirm that querying CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR returns
        // device id the semaphore was created with
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR, cl_device_id,
                             "%p", device);

        return TEST_PASS;
    }
};

struct SemaphoreMultiDeviceContextQueries : public SemaphoreTestBase
{
    SemaphoreMultiDeviceContextQueries(cl_device_id device, cl_context context,
                                       cl_command_queue queue, cl_int nelems)
        : SemaphoreTestBase(device, context, queue, nelems)
    {}

    cl_int Run() override
    {
        // partition device and create new context if possible
        cl_uint maxComputeUnits = 0;
        cl_int err =
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
        test_error(err, "Unable to get maximal number of compute units");

        cl_device_partition_property partitionProp[] = {
            CL_DEVICE_PARTITION_EQUALLY,
            static_cast<cl_device_partition_property>(maxComputeUnits / 2), 0
        };

        cl_uint deviceCount = 0;
        // how many sub-devices can we create?
        err =
            clCreateSubDevices(device, partitionProp, 0, nullptr, &deviceCount);
        if (err != CL_SUCCESS)
        {
            log_info("Can't partition device, test not supported\n");
            return TEST_SKIPPED_ITSELF;
        }

        if (deviceCount < 2)
            test_error_ret(
                CL_INVALID_VALUE,
                "Multi context test for CL_INVALID_PROPERTY not supported",
                TEST_SKIPPED_ITSELF);

        // get the list of subDevices
        SubDevicesScopeGuarded scope_guard(deviceCount);
        err = clCreateSubDevices(device, partitionProp, deviceCount,
                                 scope_guard.sub_devices.data(), &deviceCount);
        if (err != CL_SUCCESS)
        {
            log_info("Can't partition device, test not supported\n");
            return TEST_SKIPPED_ITSELF;
        }

        /* Create a multi device context */
        clContextWrapper multi_device_context = clCreateContext(
            NULL, (cl_uint)deviceCount, scope_guard.sub_devices.data(), nullptr,
            nullptr, &err);
        test_error_ret(err, "Unable to create testing context", CL_SUCCESS);

        cl_semaphore_properties_khr sema_props[] = {
            (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
            (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
            static_cast<cl_semaphore_properties_khr>(
                CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR),
            (cl_semaphore_properties_khr)scope_guard.sub_devices[0],
            CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
            0
        };

        // Try to create semaphore with multi device context
        semaphore = clCreateSemaphoreWithPropertiesKHR(multi_device_context,
                                                       sema_props, &err);
        test_error(err, "Unable to create semaphore with properties");

        // Confirm that querying CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR returns
        // the same device id the semaphore was created with
        SEMAPHORE_PARAM_TEST(CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR, cl_device_id,
                             "%p", scope_guard.sub_devices[0]);

        return TEST_PASS;
    }
};

} // anonymous namespace

// Confirm the semaphore with device list can be successfully queried
int test_semaphores_device_list_queries(cl_device_id deviceID,
                                        cl_context context,
                                        cl_command_queue defaultQueue,
                                        int num_elements)
{
    return MakeAndRunTest<SemaphoreWithDeviceListQueries>(
        deviceID, context, defaultQueue, num_elements);
}

// Confirm the semaphore without device list can be successfully queried
int test_semaphores_no_device_list_queries(cl_device_id deviceID,
                                           cl_context context,
                                           cl_command_queue defaultQueue,
                                           int num_elements)
{
    return MakeAndRunTest<SemaphoreNoDeviceListQueries>(
        deviceID, context, defaultQueue, num_elements);
}

// Confirm the semaphore created with multi-device context can be successfully
// queried
int test_semaphores_multi_device_context_queries(cl_device_id deviceID,
                                                 cl_context context,
                                                 cl_command_queue defaultQueue,
                                                 int num_elements)
{
    return MakeAndRunTest<SemaphoreMultiDeviceContextQueries>(
        deviceID, context, defaultQueue, num_elements);
}
