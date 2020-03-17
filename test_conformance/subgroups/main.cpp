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

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "harness/testHarness.h"

MTdata gMTdata;

test_definition test_list[] = {
    ADD_TEST(sub_group_info),
    ADD_TEST(work_item_functions),
    ADD_TEST(work_group_functions),
    ADD_TEST(barrier_functions),
};

const int test_num = ARRAY_SIZE(test_list);
bool use_core_subgroups = false;
bool test_ifp = true;

static test_status checkSubGroupsExtension(cl_device_id device)
{
    // The extension is optional in OpenCL 2.0 (minimum required version) and
    // required in later versions.
    auto version = get_device_cl_version(device);
    auto expected_min_version = Version(2, 0);

    if (version < expected_min_version)
    {
        version_expected_info("Test", expected_min_version.to_string().c_str(),
                              version.to_string().c_str());
        return TEST_SKIP;
    }

    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if ((version == expected_min_version) && !hasExtension)
    {
        log_info(
            "Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIP;
    }

    if ((version > expected_min_version) && !hasExtension)
    {
        log_info("'cl_khr_subgroups' not found. Using OpenCL 2.1 core subgroups.\n");
        use_core_subgroups = true;
        cl_uint ifp_supported;
        cl_uint error;
        error = clGetDeviceInfo(device, CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
            sizeof(ifp_supported), &ifp_supported, NULL);
        if (error != CL_SUCCESS) {
            print_error(error, "Unable to get CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS capability");
            return TEST_FAIL;
        }
        //skip testing ifp
        if (ifp_supported != 1) {
            log_info("INDEPENDENT FORWARD PROGRESS not supported...\n");
            test_ifp = false;
        }
        return TEST_PASS;
    }

    return TEST_PASS;
}

static test_status InitCL(cl_device_id device)
{

    auto version = get_device_cl_version(device);
    test_status ret = TEST_PASS;
    if (version >= Version(3, 0))
    {
        cl_uint max_sub_groups;
        int error;

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                                sizeof(max_sub_groups), &max_sub_groups, NULL);
        if (error != CL_SUCCESS)
        {
            print_error(error, "Unable to get max number of subgroups");
            return TEST_FAIL;
        }

        if (max_sub_groups == 0)
        {
            ret = TEST_SKIP;
        }
    }
    else
    {
        ret = checkSubGroupsExtension(device);
    }
    return ret;
}

int main(int argc, const char *argv[])
{
    gMTdata = init_genrand(0);
    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, 0,
                                   InitCL);
}
