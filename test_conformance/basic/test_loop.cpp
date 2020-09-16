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

const char *loop_kernel_code =
"__kernel void test_loop(__global int *src, __global int *loopindx, __global int *loopcnt, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    int  n = get_global_size(0);\n"
"    int  i, j;\n"
"\n"
"    dst[tid] = 0;\n"
"    for (i=0,j=loopindx[tid]; i<loopcnt[tid]; i++,j++)\n"
"    {\n"
"        if (j >= n)\n"
"            j = 0;\n"
"        dst[tid] += src[j];\n"
"    }\n"
"\n"
"}\n";


int
verify_loop(int *inptr, int *loopindx, int *loopcnt, int *outptr, int n)
{
    int     r, i, j, k;

    for (i=0; i<n; i++)
    {
        r = 0;
        for (j=0,k=loopindx[i]; j<loopcnt[i]; j++,k++)
        {
            if (k >= n)
                k = 0;
            r += inptr[k];
        }

        if (r != outptr[i])
        {
            log_error("LOOP test failed: %d found, expected %d\n", outptr[i], r);
            return -1;
        }
    }

    log_info("LOOP test passed\n");
    return 0;
}

int test_loop(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[4];
    cl_int *input_ptr, *loop_indx, *loop_cnt, *output_ptr;
    cl_program program;
    cl_kernel kernel;
    size_t threads[1];
    int err, i;

    size_t length = sizeof(cl_int) * num_elements;
    input_ptr  = (cl_int*)malloc(length);
    loop_indx  = (cl_int*)malloc(length);
    loop_cnt   = (cl_int*)malloc(length);
    output_ptr = (cl_int*)malloc(length);

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[3] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[3])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    MTdata d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        input_ptr[i] = (int)genrand_int32(d);
        loop_indx[i] = (int)get_random_float(0, num_elements-1, d);
        loop_cnt[i] = (int)get_random_float(0, num_elements/32, d);
    }
    free_mtdata(d); d = NULL;

  err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueWriteBuffer failed\n");
    return -1;
  }
  err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, loop_indx, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueWriteBuffer failed\n");
    return -1;
  }
  err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, loop_cnt, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueWriteBuffer failed\n");
    return -1;
  }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &loop_kernel_code, "test_loop" );
  if (err)
    return -1;

  err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel, 3, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    threads[0] = (unsigned int)num_elements;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueNDRangeKernel failed\n");
    return -1;
  }

  err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clReadArray failed\n");
    return -1;
  }

  err = verify_loop(input_ptr, loop_indx, loop_cnt, output_ptr, num_elements);

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseMemObject(streams[3]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(loop_indx);
    free(loop_cnt);
    free(output_ptr);

    return err;
}


