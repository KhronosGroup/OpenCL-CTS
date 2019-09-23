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
#include "testBase.h"
#include "harness/testHarness.h"
#ifndef _WIN32
#include <unistd.h>
#endif

int IsAPowerOfTwo( unsigned long x )
{
  return 0 == (x & (x-1));
}


int test_min_data_type_align_size_alignment(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems )
{
  cl_uint min_alignment;

  if (gHasLong)
    min_alignment = sizeof(cl_long)*16;
  else
    min_alignment = sizeof(cl_int)*16;

  int error = 0;
  cl_uint alignment;

  error = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(alignment), &alignment, NULL);
  test_error(error, "clGetDeviceInfo for CL_DEVICE_MEM_BASE_ADDR_ALIGN failed");
  log_info("Device reported CL_DEVICE_MEM_BASE_ADDR_ALIGN = %lu bits.\n", (unsigned long)alignment);

  // Verify the size is large enough
  if (alignment < min_alignment*8) {
    log_error("ERROR: alignment too small. Minimum alignment for %s16 is %lu bits, device reported %lu bits.",
              (gHasLong) ? "long" : "int",
              (unsigned long)(min_alignment*8), (unsigned long)alignment);
    return -1;
  }

  // Verify the size is a power of two
  if (!IsAPowerOfTwo((unsigned long)alignment)) {
    log_error("ERROR: alignment is not a power of two.\n");
    return -1;
  }

  return 0;

}
