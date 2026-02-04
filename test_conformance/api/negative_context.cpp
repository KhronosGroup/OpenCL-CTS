//
// Copyright (c) 2025 The Khronos Group Inc.
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

/* Negative Tests for clCreateContext */
REGISTER_TEST(negative_create_context)
{
    cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(nullptr), 0
    };
    cl_int err = 0;
    cl_context ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_PLATFORM,
        "clCreateContext should return CL_INVALID_PLATFORM when:\"an invalid "
        "platform object is used with the CL_CONTEXT_PLATFORM property\" using "
        "a nullptr",
        TEST_FAIL);

    props[0] = reinterpret_cast<cl_context_properties>("INVALID_PROPERTY");

    props[1] = reinterpret_cast<cl_context_properties>(nullptr);
    ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_PROPERTY,
        "clCreateContext should return CL_INVALID_PROPERTY when: \"context "
        "property name in properties is not a supported property name\"",
        TEST_FAIL);

    if (get_device_cl_version(device) >= Version(1, 2))
    {
        cl_context_properties invalid_value{ -1 };
        props[0] = CL_CONTEXT_INTEROP_USER_SYNC;
        props[1] = invalid_value;
        ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
        test_object_failure_ret(
            ctx, err, CL_INVALID_PROPERTY,
            "clCreateContext should return CL_INVALID_PROPERTY when: \"the "
            "value specified for a supported property name is not valid\"",
            TEST_FAIL);

        cl_bool property_value = CL_FALSE;
        cl_context_properties duplicated_property[5] = {
            CL_CONTEXT_INTEROP_USER_SYNC,
            static_cast<cl_context_properties>(property_value),
            CL_CONTEXT_INTEROP_USER_SYNC,
            static_cast<cl_context_properties>(property_value), 0
        };
        ctx = clCreateContext(duplicated_property, 1, &device, nullptr, nullptr,
                              &err);
        test_object_failure_ret(
            ctx, err, CL_INVALID_PROPERTY,
            "clCreateContext should return CL_INVALID_PROPERTY when: \"the "
            "same property name is specified more than once\"",
            TEST_FAIL);
    }

    ctx = clCreateContext(nullptr, 1, nullptr, nullptr, nullptr, &err);
    test_object_failure_ret(ctx, err, CL_INVALID_VALUE,
                            "clCreateContext should return CL_INVALID_VALUE "
                            "when: \"devices is NULL\"",
                            TEST_FAIL);

    ctx = clCreateContext(nullptr, 0, &device, nullptr, nullptr, &err);
    test_object_failure_ret(ctx, err, CL_INVALID_VALUE,
                            "clCreateContext should return CL_INVALID_VALUE "
                            "when: \"num_devices is equal to zero\"",
                            TEST_FAIL);

    int user_data = 1; // Arbitrary non-NULL value
    ctx = clCreateContext(nullptr, 1, &device, nullptr, &user_data, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_VALUE,
        "clCreateContext should return CL_INVALID_VALUE when: \"pfn_notify is "
        "NULL but user_data is not NULL\"",
        TEST_FAIL);

    cl_device_id invalid_device = nullptr;
    ctx = clCreateContext(nullptr, 1, &invalid_device, nullptr, nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_DEVICE,
        "clCreateContext should return CL_INVALID_DEVICE when: \"any device in "
        "devices is not a valid device\" using a device set to nullptr",
        TEST_FAIL);

    return TEST_PASS;
}

/* Negative Tests for clCreateContextFromType */
REGISTER_TEST(negative_create_context_from_type)
{
    cl_platform_id platform = getPlatformFromDevice(device);

    cl_context_properties props[5] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(nullptr),
        0, 0, 0
    };
    cl_int err = 0;
    cl_context ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT,
                                             nullptr, nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_PLATFORM,
        "clCreateContextFromType should return CL_INVALID_PLATFORM when: \"an "
        "invalid platform object is used with the CL_CONTEXT_PLATFORM "
        "property\" using a nullptr",
        TEST_FAIL);

    ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT, nullptr,
                                  nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_PLATFORM,
        "clCreateContextFromType should return CL_INVALID_PLATFORM when: \"an "
        "invalid platform object is used with the CL_CONTEXT_PLATFORM "
        "property\" using a valid object that is NOT a platform",
        TEST_FAIL);

    props[1] = reinterpret_cast<cl_context_properties>(platform);
    props[2] = reinterpret_cast<cl_context_properties>("INVALID_PROPERTY");
    props[3] = reinterpret_cast<cl_context_properties>(nullptr);

    ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT, nullptr,
                                  nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_PROPERTY,
        "clCreateContextFromType should return CL_INVALID_PROPERTY when: "
        "\"context property name in properties is not a supported property "
        "name\"",
        TEST_FAIL);

    if (get_device_cl_version(device) >= Version(1, 2))
    {
        cl_context_properties invalid_value{ -1 };
        props[2] = CL_CONTEXT_INTEROP_USER_SYNC;
        props[3] = invalid_value;
        ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT, nullptr,
                                      nullptr, &err);
        test_object_failure_ret(
            ctx, err, CL_INVALID_PROPERTY,
            "clCreateContextFromType should return CL_INVALID_PROPERTY when: "
            "\"the value specified for a supported property name is not "
            "valid\"",
            TEST_FAIL);

        props[2] = CL_CONTEXT_PLATFORM;
        props[3] = reinterpret_cast<cl_context_properties>(platform);
        ctx = clCreateContextFromType(props, CL_DEVICE_TYPE_DEFAULT, nullptr,
                                      nullptr, &err);
        test_object_failure_ret(
            ctx, err, CL_INVALID_PROPERTY,
            "clCreateContextFromType should return CL_INVALID_PROPERTY when: "
            "\"the same property name is specified more than once\"",
            TEST_FAIL);
    }

    int user_data = 1; // Arbitrary non-NULL value
    ctx = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_DEFAULT, nullptr,
                                  &user_data, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_VALUE,
        "clCreateContextFromType should return CL_INVALID_VALUE when: "
        "\"pfn_notify is NULL but user_data is not NULL\"",
        TEST_FAIL);

    cl_device_type INVALID_DEVICE_TYPE = 0;
    ctx = clCreateContextFromType(nullptr, INVALID_DEVICE_TYPE, nullptr,
                                  nullptr, &err);
    test_object_failure_ret(
        ctx, err, CL_INVALID_DEVICE_TYPE,
        "clCreateContextFromType should return CL_INVALID_DEVICE_TYPE when: "
        "\"device_type is not a valid value\"",
        TEST_FAIL);

    std::vector<cl_device_type> device_types = { CL_DEVICE_TYPE_CPU,
                                                 CL_DEVICE_TYPE_GPU,
                                                 CL_DEVICE_TYPE_ACCELERATOR };
    if (get_device_cl_version(device) >= Version(1, 2))
    {
        device_types.push_back(CL_DEVICE_TYPE_CUSTOM);
    }
    for (auto type : device_types)
    {
        clContextWrapper tmp_context =
            clCreateContextFromType(nullptr, type, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            test_object_failure_ret(
                tmp_context, err, CL_DEVICE_NOT_FOUND,
                "clCreateContextFromType should return CL_DEVICE_NOT_FOUND "
                "when: \"no devices that match device_type and property values "
                "specified in properties are currently available\"",
                TEST_FAIL);
            break;
        }
    }

    return TEST_PASS;
}

/* Negative Tests for clRetainContext */
REGISTER_TEST(negative_retain_context)
{
    cl_int err = clRetainContext(nullptr);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clRetainContext should return CL_INVALID_CONTEXT when: \"context is "
        "not a valid OpenCL context\" using a nullptr",
        TEST_FAIL);

    return TEST_PASS;
}

/* Negative Tests for clReleaseContext */
REGISTER_TEST(negative_release_context)
{
    cl_int err = clReleaseContext(nullptr);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clReleaseContext should return CL_INVALID_CONTEXT when: \"context is "
        "not a valid OpenCL context\" using a nullptr",
        TEST_FAIL);

    return TEST_PASS;
}

/* Negative Tests for clGetContextInfo */
REGISTER_TEST(negative_get_context_info)
{

    cl_uint param_value = 0;
    cl_int err = clGetContextInfo(nullptr, CL_CONTEXT_REFERENCE_COUNT,
                                  sizeof(param_value), &param_value, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clGetContextInfo should return CL_INVALID_CONTEXT when: \"context is "
        "not a valid context\" using a nullptr",
        TEST_FAIL);

    cl_context_info INVALID_PARAM_VALUE = 0;
    err = clGetContextInfo(context, INVALID_PARAM_VALUE, 0, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetContextInfo should return CL_INVALID_VALUE when: \"param_name is "
        "not one of the supported values\"",
        TEST_FAIL);

    err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT,
                           sizeof(param_value) - 1, &param_value, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetContextInfo should return CL_INVALID_VALUE when: \"size in bytes "
        "specified by param_value_size is < size of return type and "
        "param_value is not a NULL value\"",
        TEST_FAIL);

    return TEST_PASS;
}

/* Negative Tests for clSetContextDestructorCallback */
static void CL_CALLBACK callback(cl_context context, void* user_data) {}

REGISTER_TEST_VERSION(negative_set_context_destructor_callback, Version(3, 0))
{
    cl_int err = clSetContextDestructorCallback(nullptr, callback, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_CONTEXT,
        "clSetContextDestructorCallback should return CL_INVALID_CONTEXT when: "
        "\"context is not a valid context\" using a nullptr",
        TEST_FAIL);

    err = clSetContextDestructorCallback(context, nullptr, nullptr);
    test_failure_error_ret(err, CL_INVALID_VALUE,
                           "clSetContextDestructorCallback should return "
                           "CL_INVALID_VALUE when: \"pfn_notify is NULL\"",
                           TEST_FAIL);

    return TEST_PASS;
}
