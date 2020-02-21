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

const char *mix_kernel_code =
"__kernel void test_mix(__global float *srcA, __global float *srcB, __global float *srcC, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mix(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

#define MAX_ERR 1e-3

float
verify_mix(float *inptrA, float *inptrB, float *inptrC, float *outptr, int n)
{
    float       r, delta, max_err = 0.0f;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] + ((inptrB[i] - inptrA[i]) * inptrC[i]);
        delta = fabsf(r - outptr[i]) / r;
        if(delta > max_err) max_err = delta;
    }
    return max_err;
}

int
test_mix(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[4];
    cl_float        *input_ptr[3], *output_ptr, *p;
    cl_program        program;
    cl_kernel        kernel;
    void            *values[4];
    size_t            lengths[1];
    size_t    threads[1];
    float            max_err;
    int                err;
    int                i;
    MTdata          d;

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
        p[i] =  (float) genrand_real1(d);
    }
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
    {
        p[i] = (float) genrand_real1(d);
    }
    p = input_ptr[2];
    for (i=0; i<num_elements; i++)
    {
        p[i] = (float) genrand_real1(d);
    }
    free_mtdata(d); d = NULL;

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

    lengths[0] = strlen(mix_kernel_code);
    err = create_single_kernel_helper( context, &program, &kernel, 1, &mix_kernel_code, "test_mix" );
    test_error( err, "Unable to create test kernel" );


    values[0] = streams[0];
    values[1] = streams[1];
    values[2] = streams[2];
    values[3] = streams[3];
  err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
  err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
  err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2] );
  err |= clSetKernelArg(kernel, 3, sizeof streams[3], &streams[3] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    threads[0] = (size_t)num_elements;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
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

    max_err = verify_mix(input_ptr[0], input_ptr[1], input_ptr[2], output_ptr, num_elements);
    if (max_err > MAX_ERR)
    {
        log_error("MIX test failed %g max err\n", max_err);
        err = -1;
    }
    else
    {
        log_info("MIX test passed %g max err\n", max_err);
        err = 0;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseMemObject(streams[3]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(input_ptr[2]);
    free(output_ptr);

    return err;
}





