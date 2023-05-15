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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>

#include "procs.h"
#include "harness/conversions.h"
#include "harness/typeWrappers.h"

static const char *sample_kernel = {
    "%s\n"
    "__kernel __attribute__((vec_type_hint(%s%s))) void sample_test(__global "
    "int *src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     dst[tid] = src[tid];\n"
    "\n"
    "}\n"
};

int test_vec_type_hint(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
  int error;
  int vec_type_index, vec_size_index;

  ExplicitType vecType[] = { kChar, kUChar, kShort, kUShort, kInt,   kUInt,
                             kLong, kULong, kFloat, kHalf,   kDouble };
  const char *size_names[] = { "", "2", "4", "8", "16" };
  std::vector<char> program_source(4096);

  for (vec_type_index = 0;
       vec_type_index < sizeof(vecType) / sizeof(vecType[0]); vec_type_index++)
  {

      if (vecType[vec_type_index] == kHalf
          && !is_extension_available(deviceID, "cl_khr_fp16"))
      {
          log_info(
              "Extension cl_khr_fp16 not supported; skipping half tests.\n");
          continue;
      }
      else if (vecType[vec_type_index] == kDouble
               && !is_extension_available(deviceID, "cl_khr_fp64"))
      {
          log_info(
              "Extension cl_khr_fp64 not supported; skipping double tests.\n");
          continue;
      }
      else if ((vecType[vec_type_index] == kLong
                || vecType[vec_type_index] == kULong)
               && !gHasLong)
      {
          log_info(
              "Extension cl_khr_int64 not supported; skipping long tests.\n");
          continue;
      }

      for (vec_size_index = 0; vec_size_index < 5; vec_size_index++)
      {
          clProgramWrapper program;
          clKernelWrapper kernel;
          clMemWrapper in, out;
          size_t global[] = { 1, 1, 1 };

          log_info("Testing __attribute__((vec_type_hint(%s%s))...\n",
                   get_explicit_type_name(vecType[vec_type_index]),
                   size_names[vec_size_index]);
          char extension[128] = { 0 };
          if (vecType[vec_type_index] == kDouble)
              std::snprintf(extension, sizeof(extension),
                            "#pragma OPENCL EXTENSION cl_khr_fp64 : enable");
          else if (vecType[vec_type_index] == kHalf)
              std::snprintf(extension, sizeof(extension),
                            "#pragma OPENCL EXTENSION cl_khr_fp16 : enable");

          sprintf(program_source.data(), sample_kernel, extension,
                  get_explicit_type_name(vecType[vec_type_index]),
                  size_names[vec_size_index]);

          const char *src = &program_source.front();
          error = create_single_kernel_helper(context, &program, &kernel, 1,
                                              &src, "sample_test");
          test_error(error, "create_single_kernel_helper failed");

          in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * 10,
                              NULL, &error);
          test_error(error, "clCreateBuffer failed");
          out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 10,
                               NULL, &error);
          test_error(error, "clCreateBuffer failed");

          error = clSetKernelArg(kernel, 0, sizeof(in), &in);
          test_error(error, "clSetKernelArg failed");
          error = clSetKernelArg(kernel, 1, sizeof(out), &out);
          test_error(error, "clSetKernelArg failed");

          error = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global, NULL,
                                         0, NULL, NULL);
          test_error(error, "clEnqueueNDRangeKernel failed");

          error = clFinish(queue);
          test_error(error, "clFinish failed");
      }
  }

  return 0;
}
