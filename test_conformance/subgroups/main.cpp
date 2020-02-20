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
    ADD_TEST( sub_group_info ),
    ADD_TEST( work_item_functions ),
    ADD_TEST( work_group_functions ),
    ADD_TEST( barrier_functions ),
};

const int test_num = ARRAY_SIZE( test_list );

static test_status checkSubGroupsExtension(cl_device_id device)
{
    // The extension is optional in OpenCL 2.0 (minimum required version) and
    // required in later versions.
    auto version = get_device_cl_version(device);

    if (version < Version(2, 0)) {
        return TEST_SKIP;
    }

    cl_uint max_sub_groups;
    int error;

    error = clGetDeviceInfo(device, CL_DEVICE_MAX_NUM_SUB_GROUPS,
                          sizeof(max_sub_groups), &max_sub_groups, NULL);
    if (error != CL_SUCCESS) {
        print_error(error, "Unable to get max number of subgroups");
        return TEST_FAIL;
    }

    if ((max_sub_groups == 0) && (version > Version(2,2))) {
        return TEST_SKIP;
    }

    bool hasExtension = is_extension_available(device, "cl_khr_subgroups");

    if ((version == Version(2, 0)) && !hasExtension) {
        log_info("Device does not support 'cl_khr_subgroups'. Skipping the test.\n");
        return TEST_SKIP;
    }

    if ((version > Version(2, 0)) && !hasExtension) {
        log_error("'cl_khr_subgroups' is a required extension, failing.\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    gMTdata = init_genrand(0);
    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, false, 0, checkSubGroupsExtension);
}

