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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"

const char *long_add2_kernel_code =
"__kernel void test_long_add2(__global long2 *srcA, __global long2 *srcB, __global long2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] + srcB[tid];\n"
"}\n";

const char *long_sub2_kernel_code =
"__kernel void test_long_sub2(__global long2 *srcA, __global long2 *srcB, __global long2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] - srcB[tid];\n"
"}\n";

const char *long_mul2_kernel_code =
"__kernel void test_long_mul2(__global long2 *srcA, __global long2 *srcB, __global long2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] * srcB[tid];\n"
"}\n";

const char *long_mad2_kernel_code =
"__kernel void test_long_mad2(__global long2 *srcA, __global long2 *srcB, __global long2 *srcC, __global long2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] * srcB[tid] + srcC[tid];\n"
"}\n";

int
verify_long_add2(cl_long *inptrA, cl_long *inptrB, cl_long *outptr, int n)
{
    cl_long    r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] + inptrB[i];
        if (r != outptr[i])
        {
            log_error("LONG_ADD long2 test failed\n");
            return -1;
        }
    }

    log_info("LONG_ADD long2 test passed\n");
    return 0;
}

int
verify_long_sub2(cl_long *inptrA, cl_long *inptrB, cl_long *outptr, int n)
{
    cl_long    r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] - inptrB[i];
        if (r != outptr[i])
        {
            log_error("LONG_SUB long2 test failed\n");
            return -1;
        }
    }

    log_info("LONG_SUB long2 test passed\n");
    return 0;
}

int
verify_long_mul2(cl_long *inptrA, cl_long *inptrB, cl_long *outptr, int n)
{
    cl_long    r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] * inptrB[i];
        if (r != outptr[i])
        {
            log_error("LONG_MUL long2 test failed\n");
            return -1;
        }
    }

    log_info("LONG_MUL long2 test passed\n");
    return 0;
}

int
verify_long_mad2(cl_long *inptrA, cl_long *inptrB, cl_long *inptrC, cl_long *outptr, int n)
{
    cl_long    r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] * inptrB[i] + inptrC[i];
        if (r != outptr[i])
        {
            log_error("LONG_MAD long2 test failed\n");
            return -1;
        }
    }

    log_info("LONG_MAD long2 test passed\n");
    return 0;
}

int
test_intmath_long2(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[4];
    cl_program program[4];
    cl_kernel kernel[4];

    cl_long *input_ptr[3], *output_ptr, *p;
    size_t threads[1];
    int err, i;

    if(! gHasLong)
    {
        log_info("64-bit integers are not supported in this device. Skipping test.\n");
        return 0;
    }

    MTdata d = init_genrand( gRandomSeed );

    size_t length = sizeof(cl_long) * 2* num_elements;

    input_ptr[0] = (cl_long*)malloc(length);
    input_ptr[1] = (cl_long*)malloc(length);
    input_ptr[2] = (cl_long*)malloc(length);
    output_ptr   = (cl_long*)malloc(length);

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

    p = input_ptr[0];
    for (i=0; i<num_elements * 2; i++)
        p[i] = (cl_long)genrand_int32(d) | ((cl_long) genrand_int32(d) << 32);
    p = input_ptr[1];
    for (i=0; i<num_elements * 2; i++)
        p[i] = (cl_long)genrand_int32(d) | ((cl_long) genrand_int32(d) << 32);
    p = input_ptr[2];
    for (i=0; i<num_elements * 2; i++)
        p[i] = (cl_long)genrand_int32(d) | ((cl_long) genrand_int32(d) << 32);

    free_mtdata(d);
    d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }
    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }
    err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, input_ptr[2], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    program[0] = clCreateProgramWithSource(context, 1, &long_add2_kernel_code, NULL, NULL);
    if (!program[0])
    {
        log_error("clCreateProgramWithSource failed\n");
        return -1;
    }

    err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clBuildProgram failed\n");
        return -1;
    }

    kernel[0] = clCreateKernel(program[0], "test_long_add2", NULL);
    if (!kernel[0])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

  program[1] = clCreateProgramWithSource(context, 1, &long_sub2_kernel_code, NULL, NULL);
    if (!program[1])
    {
        log_error("clCreateProgramWithSource failed\n");
        return -1;
    }

    err = clBuildProgram(program[1], 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clBuildProgram failed\n");
        return -1;
    }

    kernel[1] = clCreateKernel(program[1], "test_long_sub2", NULL);
    if (!kernel[1])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

  program[2] = clCreateProgramWithSource(context, 1, &long_mul2_kernel_code, NULL, NULL);
    if (!program[2])
    {
        log_error("clCreateProgramWithSource failed\n");
        return -1;
    }

    err = clBuildProgram(program[2], 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clBuildProgram failed\n");
        return -1;
    }

    kernel[2] = clCreateKernel(program[2], "test_long_mul2", NULL);
    if (!kernel[2])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

  program[3] = clCreateProgramWithSource(context, 1, &long_mad2_kernel_code, NULL, NULL);
    if (!program[3])
    {
        log_error("clCreateProgramWithSource failed\n");
        return -1;
    }

    err = clBuildProgram(program[3], 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clBuildProgram failed\n");
        return -1;
    }

    kernel[3] = clCreateKernel(program[3], "test_long_mad2", NULL);
    if (!kernel[3])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[0], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[0], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[0], 2, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[1], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[1], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[1], 2, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[2], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[2], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[2], 2, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[3], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[3], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[3], 2, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel[3], 3, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  threads[0] = (unsigned int)num_elements;
  for (i=0; i<4; i++)
  {
    err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueNDRangeKernel failed\n");
      return -1;
    }

    err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueReadBuffer failed\n");
      return -1;
    }

    switch (i)
    {
      case 0:
        err = verify_long_add2(input_ptr[0], input_ptr[1], output_ptr, num_elements);
        break;
      case 1:
        err = verify_long_sub2(input_ptr[0], input_ptr[1], output_ptr, num_elements);
        break;
      case 2:
        err = verify_long_mul2(input_ptr[0], input_ptr[1], output_ptr, num_elements);
        break;
      case 3:
        err = verify_long_mad2(input_ptr[0], input_ptr[1], input_ptr[2], output_ptr, num_elements);
        break;
    }
    if (err)
      break;
    }

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseMemObject(streams[3]);
    for (i=0; i<4; i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(input_ptr[2]);
    free(output_ptr);
    return err;
}


