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

#include "../../test_common/harness/testHarness.h"
#include "utils.h"
#include "procs.h"

std::string gKernelName;
int gWimpyMode = 0;

basefn basefn_list[] =
{
#ifdef CL_VERSION_2_0
    test_device_info,
    test_device_queue,
    test_execute_block,
    test_enqueue_block,
    test_enqueue_nested_blocks,
    test_enqueue_wg_size,
    test_enqueue_flags,
    test_enqueue_multi_queue,
    test_host_multi_queue,
    test_enqueue_ndrange,
    test_host_queue_order,
#endif
};

const char *commonfn_names[] =
{
#ifdef CL_VERSION_2_0
    "test_device_info",
    "test_device_queue",
    "test_execute_block",
    "test_enqueue_block",
    "test_enqueue_nested_blocks",
    "test_enqueue_wg_size",
    "test_enqueue_flags",
    "test_enqueue_multi_queue",
    "test_host_multi_queue",
    "test_enqueue_ndrange",
    "test_host_queue_order",
#endif
};

ct_assert(arr_size(commonfn_names) == arr_size(basefn_list))

static const int num_commonfns = arr_size(commonfn_names);

int deviceCheck(cl_device_id device)
{
    static const char expected_cl[] =  "OpenCL 2.0";
    static const char expected_clc[] = "OpenCL C 2.0";
    char version[128] = {0};
    size_t ret_len = 0;
    cl_int res;

    res = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, &ret_len);
    if(res != CL_SUCCESS || ret_len < strlen(expected_cl) || strncmp(version, expected_cl, strlen(expected_cl)))
    {
        log_info("Device does not support '%s'. Skipping the test.\n", expected_cl);
        return CL_INVALID_DEVICE;
    }

    version[0] = 0;
    ret_len = 0;

    res = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(version), version, &ret_len);
    if(res != CL_SUCCESS || ret_len < strlen(expected_clc) || strncmp(version, expected_clc, strlen(expected_clc)))
    {
        log_info("Device does not support '%s'. Skipping the test.\n", expected_clc);
        return CL_INVALID_DEVICE;
    }

    return CL_SUCCESS;
}

int
main(int argc, const char *argv[])
{
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

    return runTestHarnessWithCheck(argc, argv, num_commonfns, basefn_list, commonfn_names, false, false, 0, deviceCheck);
}
