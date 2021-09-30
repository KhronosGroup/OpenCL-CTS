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
#include "CL/cl_half.h"

MTdata gMTdata;
cl_half_rounding_mode g_rounding_mode;

test_definition test_list[] = {
    ADD_TEST_VERSION(sub_group_info_ext, Version(2, 0)),
    ADD_TEST_VERSION(sub_group_info_core, Version(2, 1)),
    ADD_TEST_VERSION(work_item_functions_ext, Version(2, 0)),
    ADD_TEST_VERSION(work_item_functions_core, Version(2, 1)),
    ADD_TEST_VERSION(subgroup_functions_ext, Version(2, 0)),
    ADD_TEST_VERSION(subgroup_functions_core, Version(2, 1)),
    ADD_TEST_VERSION(barrier_functions_ext, Version(2, 0)),
    ADD_TEST_VERSION(barrier_functions_core, Version(2, 1)),
    ADD_TEST_VERSION(ifp_ext, Version(2, 0)),
    ADD_TEST_VERSION(ifp_core, Version(2, 1)),
    ADD_TEST(subgroup_functions_extended_types),
    ADD_TEST(subgroup_functions_non_uniform_vote),
    ADD_TEST(subgroup_functions_non_uniform_arithmetic),
    ADD_TEST(subgroup_functions_ballot),
    ADD_TEST(subgroup_functions_clustered_reduce),
    ADD_TEST(subgroup_functions_shuffle),
    ADD_TEST(subgroup_functions_shuffle_relative)
};

const int test_num = ARRAY_SIZE(test_list);

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
    // Determine the rounding mode to be used in float to half conversions in
    // init and reference code
    const cl_device_fp_config fpConfig = get_default_rounding_mode(device);

    if (fpConfig == CL_FP_ROUND_TO_NEAREST)
    {
        g_rounding_mode = CL_HALF_RTE;
    }
    else if (fpConfig == CL_FP_ROUND_TO_ZERO && gIsEmbedded)
    {
        g_rounding_mode = CL_HALF_RTZ;
    }
    else
    {
        assert(false && "Unreachable");
    }
    return ret;
}

int main(int argc, const char *argv[])
{
    gMTdata = init_genrand(0);
    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, 0,
                                   InitCL);
}
