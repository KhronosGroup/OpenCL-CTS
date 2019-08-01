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
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"

const char *kernel_code =
"__kernel void test_kernel(\n"
"char%s c, uchar%s uc, short%s s, ushort%s us, int%s i, uint%s ui, float%s f,\n"
"__global float%s *result)\n"
"{\n"
"  result[0] = %s(c);\n"
"  result[1] = %s(uc);\n"
"  result[2] = %s(s);\n"
"  result[3] = %s(us);\n"
"  result[4] = %s(i);\n"
"  result[5] = %s(ui);\n"
"  result[6] = f;\n"
"}\n";

const char *kernel_code_long =
"__kernel void test_kernel_long(\n"
"long%s l, ulong%s ul,\n"
"__global float%s *result)\n"
"{\n"
"  result[0] = %s(l);\n"
"  result[1] = %s(ul);\n"
"}\n";

int test_parameter_types_long(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  clMemWrapper results;
  int error;
  size_t global[3] = {1, 1, 1};
  float results_back[2*16];
  int count, index;
  const char* types[] = { "long", "ulong" };
  char kernel_string[8192];
  int sizes[] = {1, 2, 4, 8, 16};
  const char* size_strings[] = {"", "2", "4", "8", "16"};
  float expected;
  int total_errors = 0;
  int size_to_test;
  char *ptr;
  char convert_string[1024];
  size_t max_parameter_size;

  // We don't really care about the contents since we're just testing that the types work.
  cl_long l[16]={-21,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};
  cl_ulong ul[16]={22,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

  // Calculate how large our paramter size is to the kernel
  size_t parameter_size = sizeof(cl_long) + sizeof(cl_ulong);

  // Init our strings.
  kernel_string[0] = '\0';
  convert_string[0] = '\0';

  // Get the maximum parameter size allowed
  error = clGetDeviceInfo( device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( max_parameter_size ), &max_parameter_size, NULL );
    test_error( error, "Unable to get max parameter size from device" );

  // Create the results buffer
  results = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*2*16, NULL, &error);
  test_error(error, "clCreateBuffer failed");

  // Go over all the vector sizes
  for (size_to_test = 0; size_to_test < 5; size_to_test++) {
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t total_parameter_size = parameter_size*sizes[size_to_test] + sizeof(cl_mem);
    if (total_parameter_size > max_parameter_size) {
      log_info("Can not test with vector size %d because it would exceed the maximum allowed parameter size to the kernel. (%d > %d)\n",
               (int)sizes[size_to_test], (int)total_parameter_size, (int)max_parameter_size);
      continue;
    }

    log_info("Testing vector size %d\n", sizes[size_to_test]);

    // If size is > 1, then we need a explicit convert call.
    if (sizes[size_to_test] > 1) {
      sprintf(convert_string, "convert_float%s",  size_strings[size_to_test]);
    } else {
      sprintf(convert_string, " ");
    }

    // Build the kernel
    sprintf(kernel_string, kernel_code_long,
            size_strings[size_to_test], size_strings[size_to_test], size_strings[size_to_test],
            convert_string, convert_string
    );

    ptr = kernel_string;
    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&ptr, "test_kernel_long");
    test_error(error, "create single kernel failed");

    // Set the arguments
    for (count = 0; count < 2; count++) {
      switch (count) {
        case 0: error = clSetKernelArg(kernel, count, sizeof(cl_long)*sizes[size_to_test], &l); break;
        case 1: error = clSetKernelArg(kernel, count, sizeof(cl_ulong)*sizes[size_to_test], &ul); break;
        default: log_error("Test error"); break;
      }
      if (error)
        log_error("Setting kernel arg %d %s%s: ", count, types[count], size_strings[size_to_test]);
      test_error(error, "clSetKernelArgs failed");
    }
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &results);
    test_error(error, "clSetKernelArgs failed");

    // Execute
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    // Read back the results
    error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0, sizeof(cl_float)*2*16, results_back, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    // Verify the results
    for (count = 0; count < 2; count++) {
      for (index=0; index < sizes[size_to_test]; index++) {
        switch (count) {
          case 0: expected = (float)l[index]; break;
          case 1: expected = (float)ul[index]; break;
          default: log_error("Test error"); break;
        }

        if (results_back[count*sizes[size_to_test]+index] != expected) {
          total_errors++;
          log_error("Conversion from %s%s failed: index %d got %g, expected %g.\n", types[count], size_strings[size_to_test],
                    index, results_back[count*sizes[size_to_test]+index], expected);
        }
      }
    }
  }

  return total_errors;
}

int test_parameter_types(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  clMemWrapper results;
  int error;
  size_t global[3] = {1, 1, 1};
  float results_back[7*16];
  int count, index;
  const char* types[] = {"char", "uchar", "short", "ushort", "int", "uint", "float"};
  char kernel_string[8192];
  int sizes[] = {1, 2, 4, 8, 16};
  const char* size_strings[] = {"", "2", "4", "8", "16"};
  float expected;
  int total_errors = 0;
  int size_to_test;
  char *ptr;
  char convert_string[1024];
  size_t max_parameter_size;

  // We don't really care about the contents since we're just testing that the types work.
  cl_char c[16]={0,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};
  cl_uchar uc[16]={16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  cl_short s[16]={-17,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};
  cl_ushort us[16]={18,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  cl_int i[16]={-19,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};
  cl_uint ui[16]={20,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  cl_float f[16]={-23,-1,2,-3,4,-5,6,-7,8,-9,10,-11,12,-13,14,-15};

  // Calculate how large our paramter size is to the kernel
  size_t parameter_size = sizeof(cl_char) + sizeof(cl_uchar) +
  sizeof(cl_short) +sizeof(cl_ushort) +
  sizeof(cl_int) +sizeof(cl_uint) +
  sizeof(cl_float);

  // Init our strings.
  kernel_string[0] = '\0';
  convert_string[0] = '\0';

  // Get the maximum parameter size allowed
  error = clGetDeviceInfo( device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( max_parameter_size ), &max_parameter_size, NULL );
    test_error( error, "Unable to get max parameter size from device" );

  // Create the results buffer
  results = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*7*16, NULL, &error);
  test_error(error, "clCreateBuffer failed");

  // Go over all the vector sizes
  for (size_to_test = 0; size_to_test < 5; size_to_test++) {
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t total_parameter_size = parameter_size*sizes[size_to_test] + sizeof(cl_mem);
    if (total_parameter_size > max_parameter_size) {
      log_info("Can not test with vector size %d because it would exceed the maximum allowed parameter size to the kernel. (%d > %d)\n",
               (int)sizes[size_to_test], (int)total_parameter_size, (int)max_parameter_size);
      continue;
    }

    log_info("Testing vector size %d\n", sizes[size_to_test]);

    // If size is > 1, then we need a explicit convert call.
    if (sizes[size_to_test] > 1) {
      sprintf(convert_string, "convert_float%s",  size_strings[size_to_test]);
    } else {
      sprintf(convert_string, " ");
    }

    // Build the kernel
    sprintf(kernel_string, kernel_code,
            size_strings[size_to_test], size_strings[size_to_test], size_strings[size_to_test],
            size_strings[size_to_test], size_strings[size_to_test], size_strings[size_to_test],
            size_strings[size_to_test], size_strings[size_to_test],
            convert_string, convert_string, convert_string,
            convert_string, convert_string, convert_string
    );

    ptr = kernel_string;
    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&ptr, "test_kernel");
    test_error(error, "create single kernel failed");

    // Set the arguments
    for (count = 0; count < 7; count++) {
      switch (count) {
        case 0: error = clSetKernelArg(kernel, count, sizeof(cl_char)*sizes[size_to_test], &c); break;
        case 1: error = clSetKernelArg(kernel, count, sizeof(cl_uchar)*sizes[size_to_test], &uc); break;
        case 2: error = clSetKernelArg(kernel, count, sizeof(cl_short)*sizes[size_to_test], &s); break;
        case 3: error = clSetKernelArg(kernel, count, sizeof(cl_ushort)*sizes[size_to_test], &us); break;
        case 4: error = clSetKernelArg(kernel, count, sizeof(cl_int)*sizes[size_to_test], &i); break;
        case 5: error = clSetKernelArg(kernel, count, sizeof(cl_uint)*sizes[size_to_test], &ui); break;
        case 6: error = clSetKernelArg(kernel, count, sizeof(cl_float)*sizes[size_to_test], &f); break;
        default: log_error("Test error"); break;
      }
      if (error)
        log_error("Setting kernel arg %d %s%s: ", count, types[count], size_strings[size_to_test]);
      test_error(error, "clSetKernelArgs failed");
    }
    error = clSetKernelArg(kernel, 7, sizeof(cl_mem), &results);
    test_error(error, "clSetKernelArgs failed");

    // Execute
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, NULL);
    test_error(error, "clEnqueueNDRangeKernel failed");

    // Read back the results
    error = clEnqueueReadBuffer(queue, results, CL_TRUE, 0, sizeof(cl_float)*7*16, results_back, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    // Verify the results
    for (count = 0; count < 7; count++) {
      for (index=0; index < sizes[size_to_test]; index++) {
        switch (count) {
          case 0: expected = (float)c[index]; break;
          case 1: expected = (float)uc[index]; break;
          case 2: expected = (float)s[index]; break;
          case 3: expected = (float)us[index]; break;
          case 4: expected = (float)i[index]; break;
          case 5: expected = (float)ui[index]; break;
          case 6: expected = (float)f[index]; break;
          default: log_error("Test error"); break;
        }

        if (results_back[count*sizes[size_to_test]+index] != expected) {
          total_errors++;
          log_error("Conversion from %s%s failed: index %d got %g, expected %g.\n", types[count], size_strings[size_to_test],
                    index, results_back[count*sizes[size_to_test]+index], expected);
        }
      }
    }
  }

  if (gHasLong) {
    log_info("Testing long types...\n");
    total_errors += test_parameter_types_long( device, context, queue, num_elements );
  }
  else {
    log_info("Longs unsupported, skipping.");
  }

  return total_errors;
}



