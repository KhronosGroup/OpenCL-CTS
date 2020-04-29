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
#include "harness/parseParameters.h"
#include "utils.h"
#include "procs.h"

std::string gKernelName;
int gWimpyMode = 0;

test_status InitCL(cl_device_id device) {
  auto version = get_device_cl_version(device);
  auto expected_min_version = Version(2, 0);
  if (version < expected_min_version) {
    version_expected_info("Test", expected_min_version.to_string().c_str(), version.to_string().c_str());
    return TEST_SKIP;
  }

  int error;
  cl_uint max_queues_size;
  error = clGetDeviceInfo(device, CL_DEVICE_MAX_ON_DEVICE_QUEUES,
                          sizeof(max_queues_size), &max_queues_size, NULL);
  if (error != CL_SUCCESS) {
    print_error(error, "Unable to get max queues on device");
    return TEST_FAIL;
  }

  if ((max_queues_size == 0) && (version >= Version(3, 0)))
  {
      return TEST_SKIP;
  }

  return TEST_PASS;
}

test_definition test_list[] = {
    ADD_TEST( device_info ),
    ADD_TEST( device_queue ),
    ADD_TEST( execute_block ),
    ADD_TEST( enqueue_block ),
    ADD_TEST( enqueue_nested_blocks ),
    ADD_TEST( enqueue_wg_size ),
    ADD_TEST( enqueue_flags ),
    ADD_TEST( enqueue_multi_queue ),
    ADD_TEST( host_multi_queue ),
    ADD_TEST( enqueue_ndrange ),
    ADD_TEST( host_queue_order ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    argc = parseCustomParam(argc, argv);

    for (int i = 0; i < argc; ++i) {
      int argsRemoveNum = 0;
      if ( strcmp(argv[i], "-kernelName") == 0 ) {
        if((i + 1) > argc && argv[i + 1] == NULL) {
          vlog( "Missing value for -kernelName argument\n");
          return -1;
        }

        gKernelName = std::string(argv[i + 1]);
        argsRemoveNum += 2;
      }
     if (strcmp(argv[i], "-w") == 0 ){
        gWimpyMode = 1;
        argsRemoveNum += 1;
     }


      if (argsRemoveNum > 0) {
        for (int j = i; j < (argc - argsRemoveNum); ++j)
          argv[j] = argv[j + argsRemoveNum];

        argc -= argsRemoveNum;
        --i;
      }
    }

    return runTestHarnessWithCheck(argc, argv, test_num, test_list, false, 0, InitCL);
}
