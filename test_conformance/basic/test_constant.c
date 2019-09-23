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

const char *constant_kernel_code =
"__kernel void constant_kernel(__global float *out, __constant float *tmpF, __constant int *tmpI)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    float ftmp = tmpF[tid]; \n"
"    float Itmp = tmpI[tid]; \n"
"    out[tid] = ftmp * Itmp; \n"
"}\n";

const char *loop_constant_kernel_code =
"kernel void loop_constant_kernel(global float *out, constant float *i_pos, int num)\n"
"{\n"
"    int tid = get_global_id(0);\n"
"    float sum = 0;\n"
"    for (int i = 0; i < num; i++) {\n"
"        float  pos  = i_pos[i*3];\n"
"        sum += pos;\n"
"    }\n"
"    out[tid] = sum;\n"
"}\n";


static int
verify(cl_float *tmpF, cl_int *tmpI, cl_float *out, int n)
{
    int         i;

    for (i=0; i < n; i++)
    {
        float f = tmpF[i] * tmpI[i];
        if( out[i] != f )
        {
            log_error("CONSTANT test failed\n");
            return -1;
        }
    }

    log_info("CONSTANT test passed\n");
    return 0;
}


static int
verify_loop_constant(const cl_float *tmp, cl_float *out, cl_int l, int n)
{
    int i;
    cl_int j;
    for (i=0; i < n; i++)
    {
        float sum = 0;
        for (j=0; j < l; ++j)
            sum += tmp[j*3];

        if( out[i] != sum )
        {
            log_error("loop CONSTANT test failed\n");
            return -1;
        }
    }

    log_info("loop CONSTANT test passed\n");
    return 0;
}

int
test_constant(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[3];
    cl_int            *tmpI;
    cl_float        *tmpF, *out;
    cl_program        program;
    cl_kernel        kernel;
    size_t    global_threads[3];
    int                err;
    unsigned int                i;
    cl_ulong maxSize, maxGlobalSize, maxAllocSize;
    size_t num_floats, num_ints, constant_values;
    MTdata          d;
    RoundingMode     oldRoundMode;
    int isRTZ = 0;

  /* Verify our test buffer won't be bigger than allowed */
    err = clGetDeviceInfo( device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof( maxSize ), &maxSize, 0 );
    test_error( err, "Unable to get max constant buffer size" );

  log_info("Device reports CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE %llu bytes.\n", maxSize);
  
  // Limit test buffer size to 1/4 of CL_DEVICE_GLOBAL_MEM_SIZE
  err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalSize), &maxGlobalSize, 0);
  test_error(err, "Unable to get CL_DEVICE_GLOBAL_MEM_SIZE");

  if (maxSize > maxGlobalSize / 4)
    maxSize = maxGlobalSize / 4;

  err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(maxAllocSize), &maxAllocSize, 0);
  test_error(err, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE ");

  if (maxSize > maxAllocSize)
    maxSize = maxAllocSize;
  
  maxSize/=4;
  num_ints = (size_t)maxSize/sizeof(cl_int);
  num_floats = (size_t)maxSize/sizeof(cl_float);
  if (num_ints >= num_floats) {
    constant_values = num_floats;
  } else {
    constant_values = num_ints;
  }

  log_info("Test will attempt to use %lu bytes with one %lu byte constant int buffer and one %lu byte constant float buffer.\n",
           constant_values*sizeof(cl_int) + constant_values*sizeof(cl_float), constant_values*sizeof(cl_int), constant_values*sizeof(cl_float));

    tmpI = (cl_int*)malloc(sizeof(cl_int) * constant_values);
    tmpF = (cl_float*)malloc(sizeof(cl_float) * constant_values);
    out  = (cl_float*)malloc(sizeof(cl_float) * constant_values);
    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * constant_values, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * constant_values, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * constant_values, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    d = init_genrand( gRandomSeed );
    for (i=0; i<constant_values; i++) {
        tmpI[i] = (int)get_random_float(-0x02000000, 0x02000000, d);
        tmpF[i] = get_random_float(-0x02000000, 0x02000000, d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, sizeof(cl_float)*constant_values, (void *)tmpF, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }
  err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, sizeof(cl_int)*constant_values, (void *)tmpI, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &constant_kernel_code, "constant_kernel" );
    if (err) {
    log_error("Failed to create kernel and program: %d\n", err);
    return -1;
  }


    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    global_threads[0] = constant_values;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, global_threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed: %d\n", err);
        return -1;
    }
    err = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(cl_float)*constant_values, (void *)out, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    //If we only support rtz mode
    if( CL_FP_ROUND_TO_ZERO == get_default_rounding_mode(device) && gIsEmbedded)
    {
        oldRoundMode = set_round(kRoundTowardZero, kfloat);
        isRTZ = 1;
    }

    err = verify(tmpF, tmpI, out, (int)constant_values);

    if (isRTZ)
        (void)set_round(oldRoundMode, kfloat);

    // Loop constant buffer test
    cl_program loop_program;
    cl_kernel  loop_kernel;
    cl_int limit = 2;

    memset(out, 0, sizeof(cl_float) * constant_values);
    err = create_single_kernel_helper(context, &loop_program, &loop_kernel, 1,
                                      &loop_constant_kernel_code, "loop_constant_kernel" );
    if (err) {
        log_error("Failed to create loop kernel and program: %d\n", err);
        return -1;
    }

    err = clSetKernelArg(loop_kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(loop_kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(loop_kernel, 2, sizeof(limit), &limit);
    if (err != CL_SUCCESS) {
        log_error("clSetKernelArgs for loop kernel failed\n");
        return -1;
    }

    err = clEnqueueNDRangeKernel( queue, loop_kernel, 1, NULL, global_threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
        log_error("clEnqueueNDRangeKernel failed: %d\n", err);
        return -1;
    }
    err = clEnqueueReadBuffer( queue, streams[0], CL_TRUE, 0, sizeof(cl_float)*constant_values, (void *)out, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    err = verify_loop_constant(tmpF, out, limit, (int)constant_values);

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseKernel(loop_kernel);
    clReleaseProgram(loop_program);
    free(tmpI);
    free(tmpF);
    free(out);

    return err;
}





