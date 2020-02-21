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
#include "harness/rounding_mode.h"

#include "procs.h"

static const char *local_linear_id_1d_code =
"__kernel void test_local_linear_id_1d(global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    int linear_id = get_local_id(0);\n"
"    int result = (linear_id == (int)get_local_linear_id()) ? 0x1 : 0x0;\n"
"    dst[tid] = result;\n"
"}\n";

static const char *local_linear_id_2d_code =
"__kernel void test_local_linear_id_2d(global int *dst)\n"
"{\n"
"    int  tid_x = get_global_id(0);\n"
"    int  tid_y = get_global_id(1);\n"
"\n"
"    int linear_id = get_local_id(1) * get_local_size(0) + get_local_id(0);\n"
"    int result = (linear_id == (int)get_local_linear_id()) ? 0x1 : 0x0;\n"
"    dst[tid_y * get_global_size(0) + tid_x] = result;\n"
"}\n";


static int
verify_local_linear_id(int *result, int n)
{
    int i;
    for (i=0; i<n; i++)
    {
        if (result[i] == 0)
        {
            log_error("get_local_linear_id failed\n");
            return -1;
        }
    }
    log_info("get_local_linear_id passed\n");
    return 0;
}


int
test_local_linear_id(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
      cl_mem streams;
      cl_program program[2];
      cl_kernel kernel[2];

    int *output_ptr;
      size_t threads[2];
      int err;
      num_elements = (int)sqrt((float)num_elements);
      int length = num_elements * num_elements;

      output_ptr   = (cl_int*)malloc(sizeof(int) * length);

    streams = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), length*sizeof(int), NULL, &err);
    test_error( err, "clCreateBuffer failed.");

    err = create_single_kernel_helper_with_build_options(context, &program[0], &kernel[0], 1, &local_linear_id_1d_code, "test_local_linear_id_1d", "-cl-std=CL2.0");
    test_error( err, "create_single_kernel_helper failed");
    err = create_single_kernel_helper_with_build_options(context, &program[1], &kernel[1], 1, &local_linear_id_2d_code, "test_local_linear_id_2d", "-cl-std=CL2.0");
    test_error( err, "create_single_kernel_helper failed");

    err  = clSetKernelArg(kernel[0], 0, sizeof streams, &streams);
    test_error( err, "clSetKernelArgs failed.");
    err  = clSetKernelArg(kernel[1], 0, sizeof streams, &streams);
    test_error( err, "clSetKernelArgs failed.");

    threads[0] = (size_t)num_elements;
    threads[1] = (size_t)num_elements;
    err = clEnqueueNDRangeKernel(queue, kernel[1], 2, NULL, threads, NULL, 0, NULL, NULL);
    test_error( err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams, CL_TRUE, 0, length*sizeof(int), output_ptr, 0, NULL, NULL);
    test_error( err, "clEnqueueReadBuffer failed.");

    err = verify_local_linear_id(output_ptr, length);

    threads[0] = (size_t)num_elements;
    err = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL, threads, NULL, 0, NULL, NULL);
    test_error( err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams, CL_TRUE, 0, num_elements*sizeof(int), output_ptr, 0, NULL, NULL);
    test_error( err, "clEnqueueReadBuffer failed.");

    err = verify_local_linear_id(output_ptr, num_elements);

    // cleanup
    clReleaseMemObject(streams);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseProgram(program[0]);
    clReleaseProgram(program[1]);
    free(output_ptr);

    return err;
}
