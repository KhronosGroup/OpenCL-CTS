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

static const char *fmin_kernel_code =
    "__kernel void test_fmin(__global float *srcA, __global float *srcB, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = fmin(srcA[tid], srcB[tid]);\n"
    "}\n";

static const char *fmin2_kernel_code =
    "__kernel void test_fmin2(__global float2 *srcA, __global float2 *srcB, __global float2 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = fmin(srcA[tid], srcB[tid]);\n"
    "}\n";

static const char *fmin4_kernel_code =
    "__kernel void test_fmin4(__global float4 *srcA, __global float4 *srcB, __global float4 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = fmin(srcA[tid], srcB[tid]);\n"
    "}\n";

static const char *fmin8_kernel_code =
    "__kernel void test_fmin8(__global float8 *srcA, __global float8 *srcB, __global float8 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = fmin(srcA[tid], srcB[tid]);\n"
    "}\n";

static const char *fmin16_kernel_code =
    "__kernel void test_fmin16(__global float16 *srcA, __global float16 *srcB, __global float16 *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = fmin(srcA[tid], srcB[tid]);\n"
    "}\n";


static const char *fmin3_kernel_code =
    "__kernel void test_fmin3(__global float *srcA, __global float *srcB, __global float *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "    vstore3(fmin(vload3(tid,srcA), vload3(tid,srcB)),tid,dst);\n"
    "}\n";

int
verify_fmin(float *inptrA, float *inptrB, float *outptr, int n)
{
    float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = (inptrA[i] > inptrB[i]) ? inptrB[i] : inptrA[i];
        if (r != outptr[i])
        return -1;
    }

    return 0;
}

int
test_fmin(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[3];
    cl_float    *input_ptr[2], *output_ptr, *p;
    cl_program   *program;
    cl_kernel    *kernel;
    void        *values[3];
    size_t threads[1];
    int num_elements;
    int err;
    int i;
    MTdata d;

    program = (cl_program*)malloc(sizeof(cl_program)*kTotalVecCount);
    kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*kTotalVecCount);

    num_elements = n_elems * (1 << (kTotalVecCount-1));;

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    input_ptr[1] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
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

    d = init_genrand( gRandomSeed );
    p = input_ptr[0];
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float(-0x20000000, 0x20000000, d);
    }
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float(-0x20000000, 0x20000000, d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements,
                (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }
    err = clEnqueueWriteBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements,
                (void *)input_ptr[1], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &fmin_kernel_code, "test_fmin" );
    if (err)
    return -1;
    err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &fmin2_kernel_code, "test_fmin2" );
    if (err)
    return -1;
    err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &fmin4_kernel_code, "test_fmin4" );
    if (err)
    return -1;
    err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &fmin8_kernel_code, "test_fmin8" );
    if (err)
    return -1;
    err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &fmin16_kernel_code, "test_fmin16" );
    if (err)
    return -1;
    err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &fmin3_kernel_code, "test_fmin3" );
    if (err)
    return -1;

    values[0] = streams[0];
    values[1] = streams[1];
    values[2] = streams[2];
    for (i=0; i<kTotalVecCount; i++)
    {
        err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
        err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
        err |= clSetKernelArg(kernel[i], 2, sizeof streams[2], &streams[2] );
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

        err = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        if (verify_fmin(input_ptr[0], input_ptr[1], output_ptr, n_elems*((g_arrVecSizes[i]))))
        {
            log_error("FMIN float%d test failed\n", (g_arrVecSizes[i]));
            err = -1;
        }
        else
        {
            log_info("FMIN float%d test passed\n", (g_arrVecSizes[i]));
            err = 0;
        }
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    for (i=0; i<kTotalVecCount; i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(program);
    free(kernel);
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    return err;
}


