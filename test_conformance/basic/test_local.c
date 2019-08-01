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

const char *barrier_with_localmem_kernel_code[] = {
"__kernel void compute_sum_with_localmem(__global int *a, int n, __local int *tmp_sum, __global int *sum)\n"
"{\n"
"    int  tid = get_local_id(0);\n"
"    int  lsize = get_local_size(0);\n"
"    int  i;\n"
"\n"
"    tmp_sum[tid] = 0;\n"
"    for (i=tid; i<n; i+=lsize)\n"
"        tmp_sum[tid] += a[i];\n"
"\n"
"    if( lsize == 1 )\n"
"    {\n"
"       if( tid == 0 )\n"
"           *sum = tmp_sum[0];\n"
"       return;\n"
"    }\n"
"\n"
"    do\n"
"    {\n"
"       barrier(CLK_LOCAL_MEM_FENCE);\n"
"       if (tid < lsize/2)\n"
"       {\n"
"           int sum = tmp_sum[tid];\n"
"           if( (lsize & 1) && tid == 0 )\n"
"               sum += tmp_sum[tid + lsize - 1];\n"
"           tmp_sum[tid] = sum + tmp_sum[tid + lsize/2];\n"
"       }\n"
"       lsize = lsize/2; \n"
"    }while( lsize );\n"
"\n"
"    if( tid == 0 )\n"
"       *sum = tmp_sum[0];\n"
"}\n",
"__kernel void compute_sum_with_localmem(__global int *a, int n, __global int *sum)\n"
"{\n"
"     __local int tmp_sum[%d];\n"
"    int  tid = get_local_id(0);\n"
"    int  lsize = get_local_size(0);\n"
"    int  i;\n"
"\n"
"    tmp_sum[tid] = 0;\n"
"    for (i=tid; i<n; i+=lsize)\n"
"        tmp_sum[tid] += a[i];\n"
"\n"
"    if( lsize == 1 )\n"
"    {\n"
"       if( tid == 0 )\n"
"           *sum = tmp_sum[0];\n"
"       return;\n"
"    }\n"
"\n"
"    do\n"
"    {\n"
"       barrier(CLK_LOCAL_MEM_FENCE);\n"
"       if (tid < lsize/2)\n"
"       {\n"
"           int sum = tmp_sum[tid];\n"
"           if( (lsize & 1) && tid == 0 )\n"
"               sum += tmp_sum[tid + lsize - 1];\n"
"           tmp_sum[tid] = sum + tmp_sum[tid + lsize/2];\n"
"       }\n"
"       lsize = lsize/2; \n"
"    }while( lsize );\n"
"\n"
"    if( tid == 0 )\n"
"       *sum = tmp_sum[0];\n"
"}\n"
};

static int
verify_sum(int *inptr, int *outptr, int n)
{
    int            r = 0;
    int         i;

    for (i=0; i<n; i++)
    {
        r += inptr[i];
    }

    if (r != outptr[0])
    {
        log_error("LOCAL test failed: *%d vs %d\n", r, outptr[0] );
        return -1;
    }

    log_info("LOCAL test passed\n");
    return 0;
}

int test_local_arg_def(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_program program;
    cl_kernel kernel;

    cl_int *input_ptr, *output_ptr;
    size_t global_threads[1], local_threads[1];
    size_t wgsize, kwgsize;
    size_t max_local_workgroup_size[3];
    int err, i;
    MTdata d = init_genrand( gRandomSeed );

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wgsize, &wgsize, NULL);
    if (err) {
        log_error("clGetDeviceInfo failed, %d\n\n", err);
        return -1;
    }
    wgsize/=2;
    if (wgsize < 1)
        wgsize = 1;

    size_t in_length = sizeof(cl_int) * num_elements;
    size_t out_length = sizeof(cl_int) * wgsize;

    input_ptr = (cl_int *)malloc(in_length);
    output_ptr = (cl_int *)malloc(out_length);

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, in_length, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, out_length, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    for (i=0; i<num_elements; i++)
        input_ptr[i] = (int)genrand_int32(d);

    free_mtdata(d); d = NULL;

  err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, in_length, input_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueWriteBuffer failed\n");
    return -1;
  }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &barrier_with_localmem_kernel_code[0], "compute_sum_with_localmem" );
  if (err)
    return -1;

  err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof kwgsize, &kwgsize, NULL);
  test_error(err, "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE");

  err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
  test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

  // Pick the minimum of the device and the kernel
  if (kwgsize > max_local_workgroup_size[0])
    kwgsize = max_local_workgroup_size[0];

  //    err = clSetKernelArgs(context, kernel, 4, NULL, values, sizes);
  err  = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
  err |= clSetKernelArg(kernel, 1, sizeof num_elements, &num_elements);
  err |= clSetKernelArg(kernel, 2, wgsize * sizeof(cl_int), NULL);
  err |= clSetKernelArg(kernel, 3, sizeof streams[1], &streams[1]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    global_threads[0] = wgsize;
    local_threads[0] = wgsize;

  // Adjust the local thread size to fit and be a nice multiple.
  if (kwgsize < wgsize) {
    log_info("Adjusting wgsize down from %lu to %lu.\n", wgsize, kwgsize);
        local_threads[0] = kwgsize;
  }
  while (global_threads[0] % local_threads[0] != 0)
    local_threads[0]--;

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_threads, local_threads, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueNDRangeKernel failed\n");
    return -1;
  }

  err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, out_length, output_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueReadBuffer failed\n");
    return -1;
  }

  err = verify_sum(input_ptr, output_ptr, num_elements);

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return err;
}

int test_local_kernel_def(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_program program;
    cl_kernel kernel;

    cl_int *input_ptr, *output_ptr;
    size_t global_threads[1], local_threads[1];
    size_t wgsize, kwgsize;
    int err, i;
    char *program_source = (char*)malloc(sizeof(char)*2048);
    MTdata d = init_genrand( gRandomSeed );
    size_t max_local_workgroup_size[3];
    memset(program_source, 0, 2048);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wgsize, &wgsize, NULL);
    if (err) {
        log_error("clGetDeviceInfo failed, %d\n\n", err);
        return -1;
    }
    wgsize/=2;
    if (wgsize < 1)
        wgsize = 1;

    size_t in_length = sizeof(cl_int) * num_elements;
    size_t out_length = sizeof(cl_int) * wgsize;

    input_ptr = (cl_int *)malloc(in_length);
    output_ptr = (cl_int *)malloc(out_length);

    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, in_length, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, out_length, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    for (i=0; i<num_elements; i++)
        input_ptr[i] = (cl_int) genrand_int32(d);

    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, in_length, input_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    // Validate that created kernel doesn't violate local memory size allowed by the device
    cl_ulong localMemSize = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clGetDeviceInfo failed\n");
        return -1;
    }
    if ( wgsize > (localMemSize / (sizeof(cl_int)*sizeof(cl_int))) )
    {
        wgsize = localMemSize / (sizeof(cl_int)*sizeof(cl_int));
    }

    sprintf(program_source, barrier_with_localmem_kernel_code[1], (int)(wgsize * sizeof(cl_int)));

    err = create_single_kernel_helper(context, &program, &kernel, 1, (const char**)&program_source, "compute_sum_with_localmem" );
    free(program_source);
    if (err)
        return -1;

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof kwgsize, &kwgsize, NULL);
    test_error(err, "clGetKernelWorkGroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Pick the minimum of the device and the kernel
    if (kwgsize > max_local_workgroup_size[0])
        kwgsize = max_local_workgroup_size[0];

    //    err = clSetKernelArgs(context, kernel, 4, NULL, values, sizes);
    err  = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof num_elements, &num_elements);
    err |= clSetKernelArg(kernel, 2, sizeof streams[1], &streams[1]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    global_threads[0] = wgsize;
    local_threads[0] = wgsize;

  // Adjust the local thread size to fit and be a nice multiple.
  if (kwgsize < wgsize) {
    log_info("Adjusting wgsize down from %lu to %lu.\n", wgsize, kwgsize);
        local_threads[0] = kwgsize;
  }
  while (global_threads[0] % local_threads[0] != 0)
    local_threads[0]--;

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_threads, local_threads, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueNDRangeKernel failed\n");
    return -1;
  }

  err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, out_length, output_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueReadBuffer failed\n");
    return -1;
  }

  err = verify_sum(input_ptr, output_ptr, num_elements);

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return err;
}



