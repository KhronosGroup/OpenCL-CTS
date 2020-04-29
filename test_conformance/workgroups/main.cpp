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
#include "harness/compat.h"

#include "harness/testHarness.h"
#include "procs.h"
#include <stdio.h>
#include <string.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

test_definition test_list[] = {
    ADD_TEST(work_group_all),
    ADD_TEST(work_group_any),
    ADD_TEST(work_group_reduce_add),
    ADD_TEST(work_group_reduce_min),
    ADD_TEST(work_group_reduce_max),
    ADD_TEST(work_group_scan_inclusive_add),
    ADD_TEST(work_group_scan_inclusive_min),
    ADD_TEST(work_group_scan_inclusive_max),
    ADD_TEST(work_group_scan_exclusive_add),
    ADD_TEST(work_group_scan_exclusive_min),
    ADD_TEST(work_group_scan_exclusive_max),
    ADD_TEST(work_group_broadcast_1D),
    ADD_TEST(work_group_broadcast_2D),
    ADD_TEST(work_group_broadcast_3D),
};

const int test_num = ARRAY_SIZE(test_list);

test_status InitCL(cl_device_id device) {
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);
    if (version < expected_min_version)
    {
        version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
        return TEST_SKIP;
    }

    if (version >= Version(3, 0))
    {
        int error;
        cl_bool isSupported;
        error = clGetDeviceInfo(
            device, CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT,
            sizeof(isSupported), &isSupported, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error,
                        "Unable to query support for collective functions");
            return TEST_FAIL;
        }

        if (isSupported == CL_FALSE)
        {
            return TEST_SKIP;
        }
    }

  return TEST_PASS;
}

int main(int argc, const char *argv[]) {
  return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, 0, InitCL);
}

