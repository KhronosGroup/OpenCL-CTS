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

const char *barrier_kernel_code =
"__kernel void compute_sum(__global int *a, int n, __global int *tmp_sum, __global int *sum)\n"
"{\n"
"    int  tid = get_local_id(0);\n"
"    int  lsize = get_local_size(0);\n"
"    int  i;\n"
"\n"
"    tmp_sum[tid] = 0;\n"
"    for (i=tid; i<n; i+=lsize)\n"
"        tmp_sum[tid] += a[i];\n"
"     \n"
"     // updated to work for any workgroup size \n"
"    for (i=hadd(lsize,1); lsize>1; i = hadd(i,1))\n"
"    {\n"
"        barrier(CLK_GLOBAL_MEM_FENCE);\n"
"        if (tid + i < lsize)\n"
"            tmp_sum[tid] += tmp_sum[tid + i];\n"
"         lsize = i; \n"
"    }\n"
"\n"
"     //no barrier is required here because last person to write to tmp_sum[0] was tid 0 \n"
"    if (tid == 0)\n"
"        *sum = tmp_sum[0];\n"
"}\n";


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
        log_error("BARRIER test failed\n");
        return -1;
    }

  log_info("BARRIER test passed\n");
  return 0;
}


int
test_barrier(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[3];
    cl_int            *input_ptr = NULL, *output_ptr = NULL;
    cl_program        program;
    cl_kernel        kernel;
    size_t    global_threads[3];
    size_t    local_threads[3];
    int                err;
    int                i;
    size_t max_local_workgroup_size[3];
    size_t max_threadgroup_size = 0;
    MTdata d;

    err = create_single_kernel_helper(context, &program, &kernel, 1, &barrier_kernel_code, "compute_sum" );
    test_error(err, "Failed to build kernel/program.");

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                 sizeof(max_threadgroup_size), &max_threadgroup_size, NULL);
    test_error(err, "clGetKernelWorkgroupInfo failed.");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Pick the minimum of the device and the kernel
    if (max_threadgroup_size > max_local_workgroup_size[0])
        max_threadgroup_size = max_local_workgroup_size[0];

    // work group size must divide evenly into the global size
    while( num_elements % max_threadgroup_size )
        max_threadgroup_size--;

    input_ptr = (int*)malloc(sizeof(int) * num_elements);
    output_ptr = (int*)malloc(sizeof(int));

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int), NULL, &err);
    test_error(err, "clCreateBuffer failed.");
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * max_threadgroup_size, NULL, &err);
    test_error(err, "clCreateBuffer failed.");

    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        input_ptr[i] = (int)get_random_float(-0x01000000, 0x01000000, d);
    free_mtdata(d);  d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, sizeof(cl_int)*num_elements, (void *)input_ptr, 0, NULL, NULL);
    test_error(err, "clEnqueueWriteBuffer failed.");

    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof num_elements, &num_elements);
    err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
    err |= clSetKernelArg(kernel, 3, sizeof streams[1], &streams[1]);
    test_error(err, "clSetKernelArg failed.");

    global_threads[0] = max_threadgroup_size;
    local_threads[0] = max_threadgroup_size;

    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, global_threads, local_threads, 0, NULL, NULL );
    test_error(err, "clEnqueueNDRangeKernel failed.");

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int), (void *)output_ptr, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed.");

        err = verify_sum(input_ptr, output_ptr, num_elements);


    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return err;
}





