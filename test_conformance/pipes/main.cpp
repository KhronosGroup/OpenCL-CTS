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
#include <stdio.h>
#include <string.h>

test_status InitCL(cl_device_id device) {
  auto version = get_device_cl_version(device);
  auto expected_min_version = Version(2, 0);
  if (version < expected_min_version)
  {
      version_expected_info("Test", "OpenCL",
                            expected_min_version.to_string().c_str(),
                            version.to_string().c_str());
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

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheck(
        argc, argv, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0, InitCL);
}
