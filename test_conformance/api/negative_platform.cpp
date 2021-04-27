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

int test_negative_get_platform_ids(cl_device_id deviceID, cl_context context,
                                   cl_command_queue queue, int num_elements)
{
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(0, &platform, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetPlatformIDs should return CL_INVALID_VALUE when: \"num_entries "
        "is equal to zero and platforms is not NULL\"",
        TEST_FAIL);

    err = clGetPlatformIDs(1, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetPlatformIDs should return CL_INVALID_VALUE when: \"both "
        "num_platforms and platforms are NULL\"",
        TEST_FAIL);

    return TEST_PASS;
}

int test_negative_get_platform_info(cl_device_id deviceID, cl_context context,
                                    cl_command_queue queue, int num_elements)
{
    cl_platform_id platform = getPlatformFromDevice(deviceID);

    cl_int err =
        clGetPlatformInfo(reinterpret_cast<cl_platform_id>(deviceID),
                          CL_PLATFORM_VERSION, sizeof(char*), nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_PLATFORM,
        "clGetPlatformInfo should return CL_INVALID_PLATFORM  when: \"platform "
        "is not a valid platform\" using a valid object which is NOT a "
        "platform",
        TEST_FAIL);

    constexpr cl_platform_info INVALID_PARAM_VALUE = 0;
    err = clGetPlatformInfo(platform, INVALID_PARAM_VALUE, 0, nullptr, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetPlatformInfo should return CL_INVALID_VALUE when: \"param_name "
        "is not one of the supported values\"",
        TEST_FAIL);

    char* version;
    err =
        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, &version, nullptr);
    test_failure_error_ret(
        err, CL_INVALID_VALUE,
        "clGetPlatformInfo should return CL_INVALID_VALUE when: \"size in "
        "bytes specified by param_value_size is < size of return type and "
        "param_value is not a NULL value\"",
        TEST_FAIL);

    return TEST_PASS;
}
