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
#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include "procs.h"
#include "../../test_common/harness/testHarness.h"

MTdata gMTdata;

basefn basefn_list[] = {
    test_sub_group_info,
    test_work_item_functions,
    test_work_group_functions,
    test_barrier_functions,
};

const char *basefn_names[] = {
    "sub_group_info",
    "work_item_functions",
    "work_group_functions",
    "barrier_functions",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

static const int num_fns = sizeof(basefn_names) / sizeof(char *);

static test_status
checkSubGroupsExtension(cl_device_id device)
{
    if (!is_extension_available(device, "cl_khr_subgroups")) {
        log_error("'cl_khr_subgroups' is a required extension, failing.\n");
        return TEST_FAIL;
    }

    return TEST_PASS;
}

int
main(int argc, const char *argv[])
{
    gMTdata = init_genrand(0);
    return runTestHarnessWithCheck(argc, argv, num_fns, basefn_list, basefn_names, false, false, NULL, checkSubGroupsExtension);
}

