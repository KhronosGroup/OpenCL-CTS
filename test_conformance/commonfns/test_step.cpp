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
test_step_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems);


const char *step_kernel_code =
"__kernel void test_step(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step2_kernel_code =
"__kernel void test_step2(__global float2 *srcA, __global float2 *srcB, __global float2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step4_kernel_code =
"__kernel void test_step4(__global float4 *srcA, __global float4 *srcB, __global float4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step8_kernel_code =
"__kernel void test_step8(__global float8 *srcA, __global float8 *srcB, __global float8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step16_kernel_code =
"__kernel void test_step16(__global float16 *srcA, __global float16 *srcB, __global float16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step3_kernel_code =
"__kernel void test_step3(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(step(vload3(tid,srcA), vload3(tid,srcB)),tid,dst);\n"
"}\n";


int
verify_step(float *inptrA, float *inptrB, float *outptr, int n)
{
    float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = (inptrB[i] < inptrA[i]) ? 0.0f : 1.0f;
        if (r != outptr[i])
            return -1;
    }

    return 0;
}

int
test_step(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem      streams[3];
    cl_float    *input_ptr[2], *output_ptr, *p;
  cl_program  program[kTotalVecCount];
  cl_kernel   kernel[kTotalVecCount];
    void        *values[3];
    size_t  threads[1];
    int num_elements;
    int err;
    int i;
    MTdata d;
  num_elements = n_elems * 16;

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    input_ptr[1] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
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
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float(-0x40000000, 0x40000000, d);
    }
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float(-0x40000000, 0x40000000, d);
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

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &step_kernel_code, "test_step" );
    if (err) return -1;
    err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &step2_kernel_code, "test_step2" );
    if (err) return -1;
    err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &step4_kernel_code, "test_step4" );
    if (err) return -1;
    err = create_single_kernel_helper(context, &program[3], &kernel[3], 1,
                                      &step8_kernel_code, "test_step8");
    if (err) return -1;
    err = create_single_kernel_helper(context, &program[4], &kernel[4], 1,
                                      &step16_kernel_code, "test_step16");
    if (err) return -1;
    err = create_single_kernel_helper(context, &program[5], &kernel[5], 1,
                                      &step3_kernel_code, "test_step3");
    if (err) return -1;

    values[0] = streams[0];
    values[1] = streams[1];
    values[2] = streams[2];
  for (i=0; i <kTotalVecCount; i++)
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

        switch (i)
        {
            case 0:
                err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems);
                if (err)
                    log_error("STEP float test failed\n");
                else
                    log_info("STEP float test passed\n");
                break;

            case 1:
                err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems*2);
                if (err)
                    log_error("STEP float2 test failed\n");
                else
                    log_info("STEP float2 test passed\n");
                break;

            case 2:
                err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems*4);
                if (err)
                    log_error("STEP float4 test failed\n");
                else
                    log_info("STEP float4 test passed\n");
                break;

        case 3:
        err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems*8);
        if (err)
          log_error("STEP float8 test failed\n");
        else
          log_info("STEP float8 test passed\n");
        break;

        case 4:
        err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems*16);
        if (err)
          log_error("STEP float16 test failed\n");
        else
          log_info("STEP float16 test passed\n");
        break;

        case 5:
        err = verify_step(input_ptr[0], input_ptr[1], output_ptr, n_elems*3);
        if (err)
          log_error("STEP float3 test failed\n");
        else
          log_info("STEP float3 test passed\n");
        break;
        }

        if (err)
            break;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
  for (i=0; i<kTotalVecCount; i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    if( err )
        return err;

    if( ! is_extension_available( device, "cl_khr_fp64" ))
        return 0;

    return test_step_double( device, context, queue, n_elems);
}


#pragma mark -

const char *step_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step_double(__global double *srcA, __global double *srcB, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step2_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step2_double(__global double2 *srcA, __global double2 *srcB, __global double2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step4_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step4_double(__global double4 *srcA, __global double4 *srcB, __global double4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step8_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step8_double(__global double8 *srcA, __global double8 *srcB, __global double8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step16_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step16_double(__global double16 *srcA, __global double16 *srcB, __global double16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = step(srcA[tid], srcB[tid]);\n"
"}\n";

const char *step3_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_step3_double(__global double *srcA, __global double *srcB, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(step(vload3(tid,srcA), vload3(tid,srcB)),tid,dst);\n"
"}\n";


int
verify_step_double(double *inptrA, double *inptrB, double *outptr, int n)
{
    double       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = (inptrB[i] < inptrA[i]) ? 0.0 : 1.0;
        if (r != outptr[i])
            return -1;
    }

    return 0;
}

static int
test_step_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem      streams[3];
    cl_double    *input_ptr[2], *output_ptr, *p;
    cl_program  program[kTotalVecCount];
    cl_kernel   kernel[kTotalVecCount];
    void        *values[3];
    size_t  threads[1];
    int num_elements;
    int err;
    int i;
    MTdata d;
    num_elements = n_elems * 16;

    input_ptr[0] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
    input_ptr[1] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
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
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_double) * num_elements, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_double(-0x40000000, 0x40000000, d);
    }
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_double(-0x40000000, 0x40000000, d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_double)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }
    err = clEnqueueWriteBuffer( queue, streams[1], true, 0, sizeof(cl_double)*num_elements, (void *)input_ptr[1], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &step_kernel_code_double, "test_step_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &step2_kernel_code_double, "test_step2_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &step4_kernel_code_double, "test_step4_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &step8_kernel_code_double, "test_step8_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &step16_kernel_code_double, "test_step16_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &step3_kernel_code_double, "test_step3_double" );
    if (err)
        return -1;

    values[0] = streams[0];
    values[1] = streams[1];
    values[2] = streams[2];
    for (i=0; i < kTotalVecCount; i++)
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

        err = clEnqueueReadBuffer( queue, streams[2], true, 0, sizeof(cl_double)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        switch (i)
        {
            case 0:
                err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems);
                if (err)
                    log_error("STEP double test failed\n");
                else
                    log_info("STEP double test passed\n");
                break;

            case 1:
                err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems*2);
                if (err)
                    log_error("STEP double2 test failed\n");
                else
                    log_info("STEP double2 test passed\n");
                break;

            case 2:
                err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems*4);
                if (err)
                    log_error("STEP double4 test failed\n");
                else
                    log_info("STEP double4 test passed\n");
                break;

        case 3:
        err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems*8);
        if (err)
          log_error("STEP double8 test failed\n");
        else
          log_info("STEP double8 test passed\n");
        break;

        case 4:
        err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems*16);
        if (err)
          log_error("STEP double16 test failed\n");
        else
          log_info("STEP double16 test passed\n");
        break;

        case 5:
        err = verify_step_double(input_ptr[0], input_ptr[1], output_ptr, n_elems*3);
        if (err)
          log_error("STEP double3 test failed\n");
        else
          log_info("STEP double3 test passed\n");
        break;
        }

        if (err)
            break;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    for (i=0; i<kTotalVecCount; i++)
    {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    return err;
}

