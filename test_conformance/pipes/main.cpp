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

test_status InitCL(cl_device_id device) {
  auto version = get_device_cl_version(device);
  auto expected_min_version = Version(2, 0);
  if (version < expected_min_version) {
    version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
    return TEST_SKIP;
  }

  int error;
  cl_uint max_packet_size;
  error = clGetDeviceInfo(device, CL_DEVICE_PIPE_MAX_PACKET_SIZE,
                          sizeof(max_packet_size), &max_packet_size, NULL);
  if (error != CL_SUCCESS) {
    print_error(error, "Unable to get pipe max packet size");
    return TEST_FAIL;
  }

  if ((max_packet_size == 0) && (version >= Version(3, 0)))
  {
      return TEST_SKIP;
  }

  return TEST_PASS;
}

test_definition test_list[] = {
    ADD_TEST(pipe_readwrite_int),
    ADD_TEST(pipe_readwrite_uint),
    ADD_TEST(pipe_readwrite_long),
    ADD_TEST(pipe_readwrite_ulong),
    ADD_TEST(pipe_readwrite_short),
    ADD_TEST(pipe_readwrite_ushort),
    ADD_TEST(pipe_readwrite_float),
    ADD_TEST(pipe_readwrite_half),
    ADD_TEST(pipe_readwrite_char),
    ADD_TEST(pipe_readwrite_uchar),
    ADD_TEST(pipe_readwrite_double),
    ADD_TEST(pipe_readwrite_struct),
    ADD_TEST(pipe_workgroup_readwrite_int),
    ADD_TEST(pipe_workgroup_readwrite_uint),
    ADD_TEST(pipe_workgroup_readwrite_long),
    ADD_TEST(pipe_workgroup_readwrite_ulong),
    ADD_TEST(pipe_workgroup_readwrite_short),
    ADD_TEST(pipe_workgroup_readwrite_ushort),
    ADD_TEST(pipe_workgroup_readwrite_float),
    ADD_TEST(pipe_workgroup_readwrite_half),
    ADD_TEST(pipe_workgroup_readwrite_char),
    ADD_TEST(pipe_workgroup_readwrite_uchar),
    ADD_TEST(pipe_workgroup_readwrite_double),
    ADD_TEST(pipe_workgroup_readwrite_struct),
    ADD_TEST(pipe_subgroup_readwrite_int),
    ADD_TEST(pipe_subgroup_readwrite_uint),
    ADD_TEST(pipe_subgroup_readwrite_long),
    ADD_TEST(pipe_subgroup_readwrite_ulong),
    ADD_TEST(pipe_subgroup_readwrite_short),
    ADD_TEST(pipe_subgroup_readwrite_ushort),
    ADD_TEST(pipe_subgroup_readwrite_float),
    ADD_TEST(pipe_subgroup_readwrite_half),
    ADD_TEST(pipe_subgroup_readwrite_char),
    ADD_TEST(pipe_subgroup_readwrite_uchar),
    ADD_TEST(pipe_subgroup_readwrite_double),
    ADD_TEST(pipe_subgroup_readwrite_struct),
    ADD_TEST(pipe_convenience_readwrite_int),
    ADD_TEST(pipe_convenience_readwrite_uint),
    ADD_TEST(pipe_convenience_readwrite_long),
    ADD_TEST(pipe_convenience_readwrite_ulong),
    ADD_TEST(pipe_convenience_readwrite_short),
    ADD_TEST(pipe_convenience_readwrite_ushort),
    ADD_TEST(pipe_convenience_readwrite_float),
    ADD_TEST(pipe_convenience_readwrite_half),
    ADD_TEST(pipe_convenience_readwrite_char),
    ADD_TEST(pipe_convenience_readwrite_uchar),
    ADD_TEST(pipe_convenience_readwrite_double),
    ADD_TEST(pipe_convenience_readwrite_struct),
    ADD_TEST(pipe_info),
    ADD_TEST(pipe_max_args),
    ADD_TEST(pipe_max_packet_size),
    ADD_TEST(pipe_max_active_reservations),
    ADD_TEST(pipe_query_functions),
    ADD_TEST(pipe_readwrite_errors),
    ADD_TEST(pipe_subgroups_divergence),
};

const int test_num = ARRAY_SIZE(test_list);

int main(int argc, const char *argv[]) {
  return runTestHarnessWithCheck(argc, argv, test_num, test_list, false,
                                 0, InitCL);
}
