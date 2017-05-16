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
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

#include "procs.h"
#include "../../test_common/harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

basefn clfn_list[] = {test_mem_host_read_only_buffer,
  test_mem_host_read_only_subbuffer, 
  test_mem_host_write_only_buffer,
  test_mem_host_write_only_subbuffer,   
  test_mem_host_no_access_buffer,
  test_mem_host_no_access_subbuffer,
  test_mem_host_read_only_image,
  test_mem_host_write_only_image,
  test_mem_host_no_access_image};

const char *clfn_names[] = {"test_mem_host_read_only_buffer",
  "test_mem_host_read_only_subbuffer", 
  "test_mem_host_write_only_buffer",
  "test_mem_host_write_only_subbuffer",   
  "test_mem_host_no_access_buffer",
  "test_mem_host_no_access_subbuffer",
  "test_mem_host_read_only_image",
  "test_mem_host_write_only_image",
  "test_mem_host_no_access_image",
  "all"};

cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
bool gTestRounding = true;

int main(int argc, const char *argv[])
{
  int error = 0;    
  test_start();// in fact no code    
  log_info("1st part, non gl-sharing objects...\n");
  error = runTestHarness(argc, argv, 10, clfn_list, clfn_names, false, false, 0);
  
  return error;
}
