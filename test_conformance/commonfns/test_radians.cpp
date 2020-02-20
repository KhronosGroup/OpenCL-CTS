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

#ifndef M_PI
#define M_PI    3.14159265358979323846264338327950288
#endif

static int test_radians_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems);


const char *radians_kernel_code =
"__kernel void test_radians(__global float *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians2_kernel_code =
"__kernel void test_radians2(__global float2 *src, __global float2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians4_kernel_code =
"__kernel void test_radians4(__global float4 *src, __global float4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians8_kernel_code =
"__kernel void test_radians8(__global float8 *src, __global float8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians16_kernel_code =
"__kernel void test_radians16(__global float16 *src, __global float16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians3_kernel_code =
"__kernel void test_radians3(__global float *src, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(radians(vload3(tid,src)),tid,dst);\n"
"}\n";


#define MAX_ERR  2.0f

static float
verify_radians(float *inptr, float *outptr, int n)
{
    float error, max_error = 0.0f;
    double   r, max_val = NAN;
    int     i, j, max_index = 0;

    for (i=0,j=0; i<n; i++,j++)
    {
        r = (M_PI / 180.0) * inptr[i];
        error = Ulp_Error( outptr[i], r );
        if( fabsf(error) > max_error)
        {
            max_error = error;
            max_index = i;
            max_val = r;
            if( fabsf(error) > MAX_ERR)
            {
                log_error( "%d) Error @ %a: *%a vs %a  (*%g vs %g) ulps: %f\n", i, inptr[i], r, outptr[i], r, outptr[i], error );
                return 1;
            }
        }
    }

    log_info( "radians: Max error %f ulps at %d: *%a vs %a  (*%g vs %g)\n", max_error, max_index, max_val, outptr[max_index], max_val, outptr[max_index] );

    return 0;
}


int
test_radians(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_float     *input_ptr[1], *output_ptr, *p;
    cl_program   *program;
    cl_kernel    *kernel;
    void         *values[2];
    size_t       threads[1];
    int          num_elements;
    int          err;
    int          i;
    MTdata       d;

    program = (cl_program*)malloc(sizeof(cl_program)*kTotalVecCount);
    kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*kTotalVecCount);

    num_elements = n_elems * (1 << (kTotalVecCount-1));

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
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

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
    {
        p[i] = get_random_float((float)(-100000.f * M_PI), (float)(100000.f * M_PI) ,d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &radians_kernel_code, "test_radians" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &radians2_kernel_code, "test_radians2" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &radians4_kernel_code, "test_radians4" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &radians8_kernel_code, "test_radians8" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &radians16_kernel_code, "test_radians16" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &radians3_kernel_code, "test_radians3" );
    if (err)
        return -1;

    values[0] = streams[0];
    values[1] = streams[1];
    for (i=0; i < kTotalVecCount; i++)
    {
        err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
        err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            return -1;
        }
    }

    for (i=0; i < kTotalVecCount; i++)
    {
        threads[0] = (size_t)num_elements / ((g_arrVecSizes[i]));
        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueNDRangeKernel failed\n");
            return -1;
        }

        cl_uint dead = 0xdeaddead;
        memset_pattern4(output_ptr, &dead, sizeof(cl_float)*num_elements);
        err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_float)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        if (verify_radians(input_ptr[0], output_ptr, n_elems*(i+1)))
        {
            log_error("RADIANS float%d test failed\n",((g_arrVecSizes[i])));
            err = -1;
        }
        else
        {
            log_info("RADIANS float%d test passed\n", ((g_arrVecSizes[i])));
        }

        if (err)
            break;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    for (i=0; i < kTotalVecCount; i++) {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(program);
    free(kernel);
    free(input_ptr[0]);
    free(output_ptr);
    if( err )
        return err;

    if( ! is_extension_available( device, "cl_khr_fp64" ) )
    {
        log_info( "Skipping double -- cl_khr_fp64 is not supported by this device.\n" );
        return 0;
    }

    return test_radians_double( device,  context,  queue,  n_elems);
}



#pragma mark -

const char *radians_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians_double(__global double *src, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians2_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians2_double(__global double2 *src, __global double2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians4_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians4_double(__global double4 *src, __global double4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians8_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians8_double(__global double8 *src, __global double8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians16_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians16_double(__global double16 *src, __global double16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = radians(src[tid]);\n"
"}\n";

const char *radians3_kernel_code_double =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void test_radians3_double(__global double *src, __global double *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    vstore3(radians(vload3(tid,src)),tid,dst);\n"
"}\n";


#define MAX_ERR  2.0f

static double
verify_radians_double(double *inptr, double *outptr, int n)
{
    float error, max_error = 0.0f;
    double   r, max_val = NAN;
    int     i, j, max_index = 0;

    for (i=0,j=0; i<n; i++,j++)
    {
        r = (3.14159265358979323846264338327950288L / 180.0L) * inptr[i];
        error = Ulp_Error_Double( outptr[i], r );
        if( fabsf(error) > max_error)
        {
            max_error = error;
            max_index = i;
            max_val = r;
            if( fabsf(error) > MAX_ERR)
            {
                log_error( "%d) Error @ %a: *%a vs %a  (*%g vs %g) ulps: %f\n", i, inptr[i], r, outptr[i], r, outptr[i], error );
                return 1;
            }
        }
    }

    log_info( "radiansd: Max error %f ulps at %d: *%a vs %a  (*%g vs %g)\n", max_error, max_index, max_val, outptr[max_index], max_val, outptr[max_index] );

    return 0;
}


int
test_radians_double(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_double     *input_ptr[1], *output_ptr, *p;
    cl_program   *program;
    cl_kernel    *kernel;
    void         *values[2];
    size_t       threads[1];
    int          num_elements;
    int          err;
    int          i;
    MTdata       d;


    program = (cl_program*)malloc(sizeof(cl_program)*kTotalVecCount);
    kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*kTotalVecCount);

    //TODO: line below is clearly wrong
    num_elements = n_elems * (1 << (kTotalVecCount-1));

    input_ptr[0] = (cl_double*)malloc(sizeof(cl_double) * num_elements);
    output_ptr = (cl_double*)malloc(sizeof(cl_double) * num_elements);
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_double) * num_elements, NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_double) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        p[i] = get_random_double((float)(-100000.0 * M_PI), (float)(100000.0 * M_PI) ,d);

    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*num_elements, (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &radians_kernel_code_double, "test_radians_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[1], &kernel[1], 1, &radians2_kernel_code_double, "test_radians2_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[2], &kernel[2], 1, &radians4_kernel_code_double, "test_radians4_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[3], &kernel[3], 1, &radians8_kernel_code_double, "test_radians8_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[4], &kernel[4], 1, &radians16_kernel_code_double, "test_radians16_double" );
    if (err)
        return -1;
    err = create_single_kernel_helper( context, &program[5], &kernel[5], 1, &radians3_kernel_code_double, "test_radians3_double" );
    if (err)
        return -1;

    values[0] = streams[0];
    values[1] = streams[1];
    for (i=0; i < kTotalVecCount; i++)
    {
        err = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0] );
        err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1] );
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            return -1;
        }
    }

    for (i=0; i < kTotalVecCount; i++)
    {
        threads[0] = (size_t)num_elements / ((g_arrVecSizes[i]));
        err = clEnqueueNDRangeKernel( queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueNDRangeKernel failed\n");
            return -1;
        }

        cl_uint dead = 0xdeaddead;
        memset_pattern4(output_ptr, &dead, sizeof(cl_double)*num_elements);
        err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_double)*num_elements, (void *)output_ptr, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        if (verify_radians_double(input_ptr[0], output_ptr, n_elems*(i+1)))
        {
            log_error("RADIANS double%d test failed\n",((g_arrVecSizes[i])));
            err = -1;
        }
        else
        {
            log_info("RADIANS double%d test passed\n", ((g_arrVecSizes[i])));
        }

        if (err)
            break;
    }

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    for (i=0; i < kTotalVecCount; i++) {
        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(program);
    free(kernel);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}

