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

static const char *smoothstep_kernel_code =
"__kernel void test_smoothstep(__global float *edge0, __global float *edge1, __global float *x, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = smoothstep(edge0[tid], edge1[tid], x[tid]);\n"
"}\n";

static const char *smoothstep2_kernel_code =
"__kernel void test_smoothstep2(__global float2 *edge0, __global float2 *edge1, __global float2 *x, __global float2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = smoothstep(edge0[tid], edge1[tid], x[tid]);\n"
"}\n";

static const char *smoothstep4_kernel_code =
"__kernel void test_smoothstep4(__global float4 *edge0, __global float4 *edge1, __global float4 *x, __global float4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = smoothstep(edge0[tid], edge1[tid], x[tid]);\n"
"}\n";

static const char *smoothstep8_kernel_code =
"__kernel void test_smoothstep8(__global float8 *edge0, __global float8 *edge1, __global float8 *x, __global float8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = smoothstep(edge0[tid], edge1[tid], x[tid]);\n"
"}\n";

static const char *smoothstep16_kernel_code =
"__kernel void test_smoothstep16(__global float16 *edge0, __global float16 *edge1, __global float16 *x, __global float16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = smoothstep(edge0[tid], edge1[tid], x[tid]);\n"
"}\n";

static const char *smoothstep3_kernel_code =
"__kernel void test_smoothstep3(__global float *edge0, __global float *edge1, __global float *x, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(smoothstep(vload3(tid,edge0),vload3(tid,edge1),vload3(tid,x)), tid, dst);\n"
"}\n";

#define MAX_ERR (1e-5f)

static float
verify_smoothstep(float *edge0, float *edge1, float *x, float *outptr, int n)
{
  float       r, t, delta, max_err = 0.0f;
  int         i;

  for (i=0; i<n; i++)
  {
    t = (x[i] - edge0[i]) / (edge1[i] - edge0[i]);
    if (t < 0.0f)
      t = 0.0f;
    else if (t > 1.0f)
      t = 1.0f;
    r = t * t * (3.0f - 2.0f * t);
    delta = (float)fabs(r - outptr[i]);
    if (delta > max_err)
      max_err = delta;
  }

  return max_err;
}

const static char *fn_names[] = { "SMOOTHSTEP float", "SMOOTHSTEP float2", "SMOOTHSTEP float4", "SMOOTHSTEP float8", "SMOOTHSTEP float16", "SMOOTHSTEP float3" };

int
test_smoothstep(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
  cl_mem      streams[4];
  cl_float    *input_ptr[3], *output_ptr, *p, *p_edge0;
  cl_program  program[kTotalVecCount];
  cl_kernel   kernel[kTotalVecCount];
  size_t  threads[1];
  float max_err;
  int num_elements;
  int err;
  int i;
  MTdata d;

  num_elements = n_elems * 16;

  input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  input_ptr[1] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  input_ptr[2] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, NULL );
  if (!streams[0])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }
  streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, NULL );
  if (!streams[1])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }
  streams[2] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, NULL );
  if (!streams[2])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  streams[3] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * num_elements, NULL, NULL );
  if (!streams[3])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  p = input_ptr[0];
  d = init_genrand( gRandomSeed );
  for (i=0; i<num_elements; i++)
  {
    p[i] = get_random_float(-0x00400000, 0x00400000, d);
  }

  p = input_ptr[1];
  p_edge0 = input_ptr[0];
  for (i=0; i<num_elements; i++)
  {
    float edge0 = p_edge0[i];
    float edge1;
    do {
      edge1 = get_random_float(-0x00400000, 0x00400000, d);
      if (edge0 < edge1)
        break;
    } while (1);
    p[i] = edge1;
  }

  p = input_ptr[2];
  for (i=0; i<num_elements; i++)
  {
    p[i] = get_random_float(-0x00400000, 0x00400000, d);
  }
  free_mtdata(d);
  d = NULL;

  err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    log_error("clWriteArray failed\n");
    return -1;
  }
  err = clEnqueueWriteBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[1], 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    log_error("clWriteArray failed\n");
    return -1;
  }
  err = clEnqueueWriteBuffer( queue, streams[2], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[2], 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    log_error("clWriteArray failed\n");
    return -1;
  }

  err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &smoothstep_kernel_code, "test_smoothstep" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &smoothstep2_kernel_code, "test_smoothstep2" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &smoothstep4_kernel_code, "test_smoothstep4" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &smoothstep8_kernel_code, "test_smoothstep8" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &smoothstep16_kernel_code, "test_smoothstep16" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &smoothstep3_kernel_code, "test_smoothstep3" );
  if (err)
    return -1;

  for (i=0; i<kTotalVecCount; i++)
  {
      err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
      err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
      err |= clSetKernelArg(kernel[i], 2, sizeof streams[2], &streams[2] );
      err |= clSetKernelArg(kernel[i], 3, sizeof streams[3], &streams[3] );
      if (err != CL_SUCCESS)
    {
      log_error("clSetKernelArgs failed\n");
      return -1;
    }
  }


  threads[0] = (size_t)n_elems;
  for (i=0; i<kTotalVecCount; i++)
  {
    err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueNDRangeKernel failed\n");
      return -1;
    }


    err = clEnqueueReadBuffer( queue, streams[3], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueReadBuffer failed\n");
      return -1;
    }

    max_err = verify_smoothstep(input_ptr[0], input_ptr[1], input_ptr[2], output_ptr, n_elems * g_arrVecSizes[i]);

    if (max_err > MAX_ERR)
    {
      log_error("%s test failed %g max err\n", fn_names[i], max_err);
      err = -1;
    }
    else
    {
      log_info("%s test passed %g max err\n", fn_names[i], max_err);
      err = 0;
    }

    if (err)
      break;
  }

  clReleaseMemObject(streams[0]);
  clReleaseMemObject(streams[1]);
  clReleaseMemObject(streams[2]);
  clReleaseMemObject(streams[3]);
  for (i=0; i<kTotalVecCount; i++)
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


