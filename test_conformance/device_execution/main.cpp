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
#include <stdio.h>
#include <string.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include "harness/testHarness.h"
#include "utils.h"

std::string gKernelName;

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
  cl_uint max_queues_size;
  error = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES,
                          sizeof(max_queues_size), &max_queues_size, NULL);
  if (error != CL_SUCCESS)
  {
      print_error(error, "Unable to get max queues on device");
      return TEST_FAIL;
  }

  if ((max_queues_size == 0) && (version >= Version(3, 0)))
  {
      return TEST_SKIP;
  }

  return TEST_PASS;
}

static test_status ParseArgs(int &argc, const char *argv[],
                             std::vector<std::string> &removed_args,
                             std::string &help)
{
    std::vector<const char *> argList;
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "-kernelName") == 0)
        {
            if ((i + 1) > argc || argv[i + 1] == NULL)
            {
                vlog("Missing value for -kernelName argument\n");
                return TEST_FAIL;
            }

            gKernelName = std::string(argv[i + 1]);
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else
        {
            argList.push_back(argv[i]);
        }
    }
    update_argc_argv_from_args_list(argList, argc, argv);

    return TEST_PASS;
}

int main(int argc, const char *argv[])
{
    return runTestHarnessWithCheckAndParse(argc, argv, false, 0, InitCL,
                                           ParseArgs);
}
