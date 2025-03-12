//
// Copyright (c) 2017 The Khronos Group Inc.
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
#include "harness/testHarness.h"

#include <iostream>

// basic tests
test_status InitCL(cl_device_id device)
{
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);

    if (version < expected_min_version)
    {
        version_expected_info("Test", "OpenCL",
                              expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    if (version >= Version(3, 0))
    {
        cl_int error;
        cl_bool support_generic = CL_FALSE;
        size_t max_gvar_size = 0;

        error = clGetDeviceInfo(device, CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT,
                                sizeof(support_generic), &support_generic, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get generic address space support");
            return TEST_FAIL;
        }

        if (!support_generic)
        {
            return TEST_SKIP;
        }

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
                                sizeof(max_gvar_size), &max_gvar_size, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error,
                        "Unable to query CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE.");
            return TEST_FAIL;
        }

        if (!max_gvar_size)
        {
            return TEST_SKIP;
        }
    }

    return TEST_PASS;
}

/*
    Generic Address Space
    Tests for unnamed generic address space. This feature allows developers to create single generic functions
    that are able to operate on pointers from various address spaces instead of writing separate instances for every combination.
*/

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, false, InitCL);
}
