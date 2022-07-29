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

static int
test_sign_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems);


const char *sign_kernel_code =
"__kernel void test_sign(__global float *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign2_kernel_code =
"__kernel void test_sign2(__global float2 *src, __global float2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign4_kernel_code =
"__kernel void test_sign4(__global float4 *src, __global float4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign8_kernel_code =
"__kernel void test_sign8(__global float8 *src, __global float8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign16_kernel_code =
"__kernel void test_sign16(__global float16 *src, __global float16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign3_kernel_code =
"__kernel void test_sign3(__global float *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(sign(vload3(tid,src)), tid, dst);\n"
"}\n";



static int
verify_sign(float *inptr, float *outptr, int n)
{
  float       r;
  int         i;

  for (i=0; i<n; i++)
  {
    if (inptr[i] > 0.0f)
      r = 1.0f;
    else if (inptr[i] < 0.0f)
      r = -1.0f;
    else
      r = 0.0f;
    if (r != outptr[i])
      return -1;
  }

  return 0;
}

static const char *fn_names[] = { "SIGN float", "SIGN float2", "SIGN float4", "SIGN float8", "SIGN float16", "SIGN float3" };

int
test_sign(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
  cl_mem      streams[2];
  cl_float    *input_ptr[1], *output_ptr, *p;
  cl_program  program[kTotalVecCount];
  cl_kernel   kernel[kTotalVecCount];
  void        *values[2];
  size_t  threads[1];
  int num_elements;
  int err;
  int i;
  MTdata    d;

  num_elements = n_elems * 16;

  input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
  streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(cl_float) * num_elements, NULL, NULL);
  if (!streams[0])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(cl_float) * num_elements, NULL, NULL);
  if (!streams[1])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  d = init_genrand( gRandomSeed );
  p = input_ptr[0];
  for (i=0; i<num_elements; i++)
  {
    p[i] = get_random_float(-0x20000000, 0x20000000, d);
  }
  free_mtdata(d);   d = NULL;


  err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    log_error("clWriteArray failed\n");
    return -1;
  }

  err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &sign_kernel_code, "test_sign" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &sign2_kernel_code, "test_sign2" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &sign4_kernel_code, "test_sign4" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &sign8_kernel_code, "test_sign8" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &sign16_kernel_code, "test_sign16" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &sign3_kernel_code, "test_sign3" );
  if (err)
    return -1;

  values[0] = streams[0];
  values[1] = streams[1];
  for (i=0; i<kTotalVecCount; i++)
  {
      err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
      err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
      if (err != CL_SUCCESS)
    {
      log_error("clSetKernelArgs failed\n");
      return -1;
    }
  }

  threads[0] = (size_t)n_elems;
  for (i=0; i<kTotalVecCount; i++) // change this so we test all
  {
    err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueNDRangeKernel failed\n");
      return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueReadBuffer failed\n");
      return -1;
    }

    if (verify_sign(input_ptr[0], output_ptr, n_elems*(i+1)))
    {
      log_error("%s test failed\n", fn_names[i]);
      err = -1;
    }
    else
    {
      log_info("%s test passed\n", fn_names[i]);
      err = 0;
    }

    if (err)
      break;
  }

  clReleaseMemObject(streams[0]);
  clReleaseMemObject(streams[1]);
  for (i=0; i<kTotalVecCount; i++)
  {
    clReleaseKernel(kernel[i]);
    clReleaseProgram(program[i]);
  }
  free(input_ptr[0]);
  free(output_ptr);

  if (err) return err;

  if (!is_extension_available(device, "cl_khr_fp64"))
  {
      log_info("skipping double test -- cl_khr_fp64 not supported.\n");
      return 0;
  }

    return test_sign_double( device, context, queue, n_elems);
}

#pragma mark -

const char *sign_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign_double(__global double *src, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign2_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign2_double(__global double2 *src, __global double2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign4_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign4_double(__global double4 *src, __global double4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign8_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign8_double(__global double8 *src, __global double8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign16_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign16_double(__global double16 *src, __global double16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = sign(src[tid]);\n"
"}\n";

const char *sign3_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_sign3_double(__global double *src, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(sign(vload3(tid,src)), tid, dst);\n"
"}\n";


static int
verify_sign_double(double *inptr, double *outptr, int n)
{
  double       r;
  int         i;

  for (i=0; i<n; i++)
  {
    if (inptr[i] > 0.0)
      r = 1.0;
    else if (inptr[i] < 0.0)
      r = -1.0;
    else
      r = 0.0f;
    if (r != outptr[i])
      return -1;
  }

  return 0;
}

static const char *fn_names_double[] = { "SIGN double", "SIGN double2", "SIGN double4", "SIGN double8", "SIGN double16", "SIGN double3" };

int
test_sign_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
  cl_mem      streams[2];
  cl_double    *input_ptr[1], *output_ptr, *p;
  cl_program  program[kTotalVecCount];
  cl_kernel   kernel[kTotalVecCount];
  void        *values[2];
  size_t  threads[1];
  int num_elements;
  int err;
  int i;
  MTdata    d;

  num_elements = n_elems * 16;

  input_ptr[0] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
  output_ptr = (cl_double*)malloc(sizeof(cl_double) * num_elements);
  streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(cl_double) * num_elements, NULL, NULL);
  if (!streams[0])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(cl_double) * num_elements, NULL, NULL);
  if (!streams[1])
  {
    log_error("clCreateBuffer failed\n");
    return -1;
  }

  d = init_genrand( gRandomSeed );
  p = input_ptr[0];
  for (i=0; i<num_elements; i++)
    p[i] = get_random_double(-0x20000000, 0x20000000, d);

  free_mtdata(d);   d = NULL;


  err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_double)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
  if (err != CL_SUCCESS)
  {
    log_error("clWriteArray failed\n");
    return -1;
  }

  err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &sign_kernel_code_double, "test_sign_double" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &sign2_kernel_code_double, "test_sign2_double" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &sign4_kernel_code_double, "test_sign4_double" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &sign8_kernel_code_double, "test_sign8_double" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &sign16_kernel_code_double, "test_sign16_double" );
  if (err)
    return -1;
  err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &sign3_kernel_code_double, "test_sign3_double" );
  if (err)
    return -1;

  values[0] = streams[0];
  values[1] = streams[1];
  for (i=0; i<kTotalVecCount; i++)
  {
      err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
      err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
      if (err != CL_SUCCESS)
    {
      log_error("clSetKernelArgs failed\n");
      return -1;
    }
  }

  threads[0] = (size_t)n_elems;
  for (i=0; i<kTotalVecCount; i++) // this hsould be changed
  {
    err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueNDRangeKernel failed\n");
      return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_double)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
      log_error("clEnqueueReadBuffer failed\n");
      return -1;
    }

    if (verify_sign_double(input_ptr[0], output_ptr, n_elems*(i+1)))
    {
      log_error("%s test failed\n", fn_names_double[i]);
      err = -1;
    }
    else
    {
      log_info("%s test passed\n", fn_names_double[i]);
      err = 0;
    }

    if (err)
      break;
  }

  clReleaseMemObject(streams[0]);
  clReleaseMemObject(streams[1]);
  for (i=0; i<kTotalVecCount; i++)
  {
    clReleaseKernel(kernel[i]);
    clReleaseProgram(program[i]);
  }
  free(input_ptr[0]);
  free(output_ptr);

  return err;
}


