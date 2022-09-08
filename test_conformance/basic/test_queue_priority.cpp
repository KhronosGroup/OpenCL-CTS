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

static const char *fpadd_kernel_code =
"__kernel void test_fpadd(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] + srcB[tid];\n"
"}\n";

static const char *fpsub_kernel_code =
"__kernel void test_fpsub(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] - srcB[tid];\n"
"}\n";

static const char *fpmul_kernel_code =
"__kernel void test_fpmul(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] * srcB[tid];\n"
"}\n";

static int
verify_fpadd(float *inptrA, float *inptrB, float *outptr, int n, int fileNum)
{
    int         i;

    float * reference_ptr = (float *)malloc(n * sizeof(float));

    for (i=0; i<n; i++)
    {
        reference_ptr[i] = inptrA[i] + inptrB[i];
    }

    for (i=0; i<n; i++)
    {
        if (reference_ptr[i] != outptr[i])
        {
            log_error("FP_ADD float test failed\n");
            return -1;
        }
    }

    free(reference_ptr);

    log_info("FP_ADD float test passed\n");
    return 0;
}

static int
verify_fpsub(float *inptrA, float *inptrB, float *outptr, int n, int fileNum)
{
    int         i;

    float * reference_ptr = (float *)malloc(n * sizeof(float));

    for (i=0; i<n; i++)
    {
        reference_ptr[i] = inptrA[i] - inptrB[i];
    }

    for (i=0; i<n; i++)
    {
        if (reference_ptr[i] != outptr[i])
        {
            log_error("FP_SUB float test failed\n");
            return -1;
        }
    }

    free(reference_ptr);

    log_info("FP_SUB float test passed\n");
    return 0;
}

static int
verify_fpmul(float *inptrA, float *inptrB, float *outptr, int n, int fileNum)
{
    int         i;

    float * reference_ptr = (float *)malloc(n * sizeof(float));

    for (i=0; i<n; i++)
    {
        reference_ptr[i] = inptrA[i] * inptrB[i];
    }

    for (i=0; i<n; i++)
    {
        if (reference_ptr[i] != outptr[i])
        {
            log_error("FP_MUL float test failed\n");
            return -1;
        }
    }

    free(reference_ptr);

    log_info("FP_MUL float test passed\n");
    return 0;
}

#if defined( __APPLE__ )

int test_queue_priority(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
  int err;
  int command_queue_priority = 0;
  int command_queue_select_compute_units = 0;

  cl_queue_properties queue_properties[] = { CL_QUEUE_PROPERTIES, 0, 0, 0, 0, 0, 0 };
  int idx = 2;

  // Check to see if queue priority is supported
  if (((command_queue_priority = is_extension_available(device, "cl_APPLE_command_queue_priority"))) == 0)
  {
    log_info("cl_APPLE_command_queue_priority extension is not supported - skipping test\n");
  }

  // Check to see if selecting the number of compute units is supported
  if (((command_queue_select_compute_units = is_extension_available(device, "cl_APPLE_command_queue_select_compute_units"))) == 0)
  {
    log_info("cl_APPLE_command_queue_select_compute_units extension is not supported - skipping test\n");
  }

  // If neither extension is supported, skip the test
  if (!command_queue_priority && !command_queue_select_compute_units)
    return 0;

  // Setup the queue properties
#ifdef cl_APPLE_command_queue_priority
  if (command_queue_priority) {
    queue_properties[idx++] = CL_QUEUE_PRIORITY_APPLE;
    queue_properties[idx++] = CL_QUEUE_PRIORITY_BACKGROUND_APPLE;
  }
#endif

#ifdef cl_APPLE_command_queue_select_compute_units
  // Check the number of compute units on the device
  cl_uint num_compute_units = 0;
  err = clGetDeviceInfo( device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( num_compute_units ), &num_compute_units, NULL );
  if (err) {
    log_error("clGetDeviceInfo for CL_DEVICE_MAX_COMPUTE_UNITS failed: %d", err);
    return -1;
  }

  if (command_queue_select_compute_units) {
    queue_properties[idx++] = CL_QUEUE_NUM_COMPUTE_UNITS_APPLE;
    queue_properties[idx++] = num_compute_units/2;
  }
#endif
  queue_properties[idx++] = 0;

  // Create the command queue
  cl_command_queue background_queue = clCreateCommandQueueWithProperties(context, device, queue_properties, &err);
  if (err) {
    log_error("clCreateCommandQueueWithPropertiesAPPLE failed: %d", err);
    return -1;
  }

  // Test the command queue
  cl_mem streams[4];
    cl_program program[3];
    cl_kernel kernel[3];
  cl_event marker_event;

  float *input_ptr[3], *output_ptr, *p;
    size_t threads[1];
    int i;
  MTdata d = init_genrand( gRandomSeed );
  size_t length = sizeof(cl_float) * num_elements;
  int isRTZ = 0;
  RoundingMode oldMode = kDefaultRoundingMode;

  // check for floating point capabilities
  cl_device_fp_config single_config = 0;
  err = clGetDeviceInfo( device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( single_config ), &single_config, NULL );
  if (err) {
    log_error("clGetDeviceInfo for CL_DEVICE_SINGLE_FP_CONFIG failed: %d", err);
    return -1;
  }
  //If we only support rtz mode
  if( CL_FP_ROUND_TO_ZERO == ( single_config & (CL_FP_ROUND_TO_ZERO|CL_FP_ROUND_TO_NEAREST) ) )
  {
    //Check to make sure we are an embedded device
    char profile[32];
    err = clGetDeviceInfo( device, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL);
    if( err )
    {
      log_error("clGetDeviceInfo for CL_DEVICE_PROFILE failed: %d", err);
      return -1;
    }
    if( 0 != strcmp( profile, "EMBEDDED_PROFILE"))
    {
      log_error( "FAILURE:  Device doesn't support CL_FP_ROUND_TO_NEAREST and isn't EMBEDDED_PROFILE\n" );
      return -1;
    }

    isRTZ = 1;
    oldMode = get_round();
  }

  input_ptr[0] = (cl_float *)malloc(length);
  input_ptr[1] = (cl_float *)malloc(length);
  input_ptr[2] = (cl_float *)malloc(length);
  output_ptr = (cl_float *)malloc(length);

  streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error( err, "clCreateBuffer failed.");
  streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error( err, "clCreateBuffer failed.");
  streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error( err, "clCreateBuffer failed.");
  streams[3] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error( err, "clCreateBuffer failed.");

  p = input_ptr[0];
  for (i=0; i<num_elements; i++)
    p[i] = get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), d);
  p = input_ptr[1];
  for (i=0; i<num_elements; i++)
    p[i] = get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), d);
  p = input_ptr[2];
  for (i=0; i<num_elements; i++)
    p[i] = get_random_float(-MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), MAKE_HEX_FLOAT(0x1.0p31f, 0x1, 31), d);

  err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
  test_error( err, "clEnqueueWriteBuffer failed.");

  err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
  test_error( err, "clEnqueueWriteBuffer failed.");

  err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, input_ptr[2], 0, NULL, NULL);
  test_error( err, "clEnqueueWriteBuffer failed.");

  err = clEnqueueMarkerWithWaitList(queue, 0, NULL, &marker_event);
  test_error( err, "clEnqueueMarkerWithWaitList failed.");
  clFlush(queue);

  err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &fpadd_kernel_code, "test_fpadd");
  test_error( err, "create_single_kernel_helper failed");

  err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &fpsub_kernel_code, "test_fpsub");
  test_error( err, "create_single_kernel_helper failed");

  err = create_single_kernel_helper(context, &program[2], &kernel[2], 1, &fpmul_kernel_code, "test_fpmul");
  test_error( err, "create_single_kernel_helper failed");


  err  = clSetKernelArg(kernel[0], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[0], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[0], 2, sizeof streams[3], &streams[3]);
  test_error( err, "clSetKernelArgs failed.");

  err  = clSetKernelArg(kernel[1], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[1], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[1], 2, sizeof streams[3], &streams[3]);
  test_error( err, "clSetKernelArgs failed.");

  err  = clSetKernelArg(kernel[2], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[2], 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[2], 2, sizeof streams[3], &streams[3]);
  test_error( err, "clSetKernelArgs failed.");

  threads[0] = (unsigned int)num_elements;
  for (i=0; i<3; i++)
  {
    err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL, 1, &marker_event, NULL);
    test_error( err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
    test_error( err, "clEnqueueReadBuffer failed.");

    if( isRTZ )
      set_round( kRoundTowardZero, kfloat );

    switch (i)
    {
      case 0:
        err = verify_fpadd(input_ptr[0], input_ptr[1], output_ptr, num_elements, i);
        break;
      case 1:
        err = verify_fpsub(input_ptr[0], input_ptr[1], output_ptr, num_elements, i);
        break;
      case 2:
        err = verify_fpmul(input_ptr[0], input_ptr[1], output_ptr, num_elements, i);
        break;
    }

    if( isRTZ )
      set_round( oldMode, kfloat );
  }

  // cleanup
  clReleaseCommandQueue(background_queue);
  clReleaseEvent(marker_event);
  clReleaseMemObject(streams[0]);
  clReleaseMemObject(streams[1]);
  clReleaseMemObject(streams[2]);
  clReleaseMemObject(streams[3]);
  for (i=0; i<3; i++)
  {
    clReleaseKernel(kernel[i]);
    clReleaseProgram(program[i]);
  }
  free(input_ptr[0]);
  free(input_ptr[1]);
  free(input_ptr[2]);
  free(output_ptr);
  free_mtdata( d );

  return err;
}



#endif

