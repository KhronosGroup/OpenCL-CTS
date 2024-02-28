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

#define NUM_PROGRAMS 6

static const int vector_sizes[] = {1, 2, 3, 4, 8, 16};


const char *int_mad24_kernel_code =
"__kernel void test_int_mad24(__global int *srcA, __global int *srcB, __global int *srcC, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *int2_mad24_kernel_code =
"__kernel void test_int2_mad24(__global int2 *srcA, __global int2 *srcB, __global int2 *srcC, __global int2 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *int3_mad24_kernel_code =
"__kernel void test_int3_mad24(__global int *srcA, __global int *srcB, __global int *srcC, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    int3 tmp = mad24(vload3(tid, srcA), vload3(tid, srcB), vload3(tid, srcC));\n"
"    vstore3(tmp, tid, dst);\n"
"}\n";

const char *int4_mad24_kernel_code =
"__kernel void test_int4_mad24(__global int4 *srcA, __global int4 *srcB, __global int4 *srcC, __global int4 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *int8_mad24_kernel_code =
"__kernel void test_int8_mad24(__global int8 *srcA, __global int8 *srcB, __global int8 *srcC, __global int8 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *int16_mad24_kernel_code =
"__kernel void test_int16_mad24(__global int16 *srcA, __global int16 *srcB, __global int16 *srcC, __global int16 *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";


const char *uint_mad24_kernel_code =
"__kernel void test_uint_mad24(__global uint *srcA, __global uint *srcB, __global uint *srcC, __global uint *dst)\n"
"{\n"
"    uint  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *uint2_mad24_kernel_code =
"__kernel void test_uint2_mad24(__global uint2 *srcA, __global uint2 *srcB, __global uint2 *srcC, __global uint2 *dst)\n"
"{\n"
"    uint  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *uint3_mad24_kernel_code =
"__kernel void test_uint3_mad24(__global uint *srcA, __global uint *srcB, __global uint *srcC, __global uint *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"    uint3 tmp = mad24(vload3(tid, srcA), vload3(tid, srcB), vload3(tid, srcC));\n"
"    vstore3(tmp, tid, dst);\n"
"}\n";


const char *uint4_mad24_kernel_code =
"__kernel void test_uint4_mad24(__global uint4 *srcA, __global uint4 *srcB, __global uint4 *srcC, __global uint4 *dst)\n"
"{\n"
"    uint  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *uint8_mad24_kernel_code =
"__kernel void test_uint8_mad24(__global uint8 *srcA, __global uint8 *srcB, __global uint8 *srcC, __global uint8 *dst)\n"
"{\n"
"    uint  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";

const char *uint16_mad24_kernel_code =
"__kernel void test_uint16_mad24(__global uint16 *srcA, __global uint16 *srcB, __global uint16 *srcC, __global uint16 *dst)\n"
"{\n"
"    uint  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = mad24(srcA[tid], srcB[tid], srcC[tid]);\n"
"}\n";


int
verify_int_mad24(int *inptrA, int *inptrB, int *inptrC, int *outptr, size_t n, size_t vecSize)
{
    int            r;
    size_t         i;

    for (i=0; i<n; i++)
    {
        int a = inptrA[i];
        int b = inptrB[i];
        r = a * b + inptrC[i];
        if (r != outptr[i])
        {
            log_error("Failed at %zu)  0x%8.8x * 0x%8.8x + 0x%8.8x = *0x%8.8x "
                      "vs 0x%8.8x\n",
                      i, a, b, inptrC[i], r, outptr[i]);
            return -1;
        }
    }

    return 0;
}

int
verify_uint_mad24(cl_uint *inptrA, cl_uint *inptrB, cl_uint *inptrC, cl_uint *outptr, size_t n, size_t vecSize)
{
    cl_uint         r;
    size_t         i;

    for (i=0; i<n; i++)
    {
        cl_uint a = inptrA[i] & 0xFFFFFFU;
        cl_uint b = inptrB[i] & 0xFFFFFFU;
        r = a * b + inptrC[i];
        if (r != outptr[i])
        {
            log_error("Failed at %zu)  0x%8.8x * 0x%8.8x + 0x%8.8x = *0x%8.8x "
                      "vs 0x%8.8x\n",
                      i, a, b, inptrC[i], r, outptr[i]);
            return -1;
        }
    }

    return 0;
}

static const char *test_str_names[] = { "int", "int2", "int3", "int4", "int8", "int16", "uint", "uint2", "uint3", "uint4", "uint8", "uint16" };

static inline int random_int24( MTdata d )
{
    int result = genrand_int32(d);

    return (result << 8) >> 8;
}

static inline int random_int32( MTdata d )
{
    return genrand_int32(d);
}


int test_integer_mad24(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem streams[4];
    cl_int *input_ptr[3], *output_ptr, *p;

    cl_program program[2*NUM_PROGRAMS];
    cl_kernel kernel[2*NUM_PROGRAMS];
    size_t threads[1];

    int                num_elements;
    int                err;
    int                i;
    MTdata              d;

    size_t length = sizeof(cl_int) * 16 * n_elems;
    num_elements = n_elems * 16;

    input_ptr[0] = (cl_int*)malloc(length);
    input_ptr[1] = (cl_int*)malloc(length);
    input_ptr[2] = (cl_int*)malloc(length);
    output_ptr   = (cl_int*)malloc(length);

    streams[0] = clCreateBuffer(context, 0, length, NULL, &err);
      test_error(err, "clCreateBuffer failed");
    streams[1] = clCreateBuffer(context, 0, length, NULL, &err);
      test_error(err, "clCreateBuffer failed");
    streams[2] = clCreateBuffer(context, 0, length, NULL, &err);
      test_error(err, "clCreateBuffer failed");
    streams[3] = clCreateBuffer(context, 0, length, NULL, &err);
      test_error(err, "clCreateBuffer failed");

    d = init_genrand( gRandomSeed );
    p = input_ptr[0];
    for (i=0; i<num_elements; i++)
        p[i] = random_int24(d);
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
        p[i] = random_int24(d);
    p = input_ptr[2];
    for (i=0; i<num_elements; i++)
        p[i] = random_int32(d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
         test_error(err, "clEnqueueWriteBuffer failed");
    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
      test_error(err, "clEnqueueWriteBuffer failed");
    err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, input_ptr[2], 0, NULL, NULL);
      test_error(err, "clEnqueueWriteBuffer failed");

    err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &int_mad24_kernel_code, "test_int_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &int2_mad24_kernel_code, "test_int2_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[2], &kernel[2], 1, &int3_mad24_kernel_code, "test_int3_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[3], &kernel[3], 1, &int4_mad24_kernel_code, "test_int4_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[4], &kernel[4], 1, &int8_mad24_kernel_code, "test_int8_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[5], &kernel[5], 1, &int16_mad24_kernel_code, "test_int16_mad24");
    if (err)
        return -1;

    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS], &kernel[NUM_PROGRAMS], 1, &uint_mad24_kernel_code, "test_uint_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS+1], &kernel[NUM_PROGRAMS+1], 1, &uint2_mad24_kernel_code, "test_uint2_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS+2], &kernel[NUM_PROGRAMS+2], 1, &uint3_mad24_kernel_code, "test_uint3_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS+3], &kernel[NUM_PROGRAMS+3], 1, &uint4_mad24_kernel_code, "test_uint4_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS+4], &kernel[NUM_PROGRAMS+4], 1, &uint8_mad24_kernel_code, "test_uint8_mad24");
    if (err)
        return -1;
    err = create_single_kernel_helper(context, &program[NUM_PROGRAMS+5], &kernel[NUM_PROGRAMS+5], 1, &uint16_mad24_kernel_code, "test_uint16_mad24");
    if (err)
        return -1;

    for (i=0; i< 2*NUM_PROGRAMS; i++)
    {
        err  = clSetKernelArg(kernel[i], 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernel[i], 1, sizeof streams[1], &streams[1]);
        err |= clSetKernelArg(kernel[i], 2, sizeof streams[2], &streams[2]);
        err |= clSetKernelArg(kernel[i], 3, sizeof streams[3], &streams[3]);
          test_error(err, "clSetKernelArg failed");
    }


    threads[0] = (unsigned int)n_elems;
    // test signed
    for (i=0; i<NUM_PROGRAMS; i++)
    {
        err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL);
          test_error(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
          test_error(err, "clEnqueueNDRangeKernel failed");

        if (verify_int_mad24(input_ptr[0], input_ptr[1], input_ptr[2], output_ptr, n_elems * vector_sizes[i], vector_sizes[i]))
        {
            log_error("INT_MAD24 %s test failed\n", test_str_names[i]);
            err = -1;
        }
        else
        {
            log_info("INT_MAD24 %s test passed\n", test_str_names[i]);
            err = 0;
        }

        if (err)
            break;
    }

    p = input_ptr[0];
    for (i=0; i<num_elements; i++)
        p[i] &= 0xffffffU;
    p = input_ptr[1];
    for (i=0; i<num_elements; i++)
        p[i] &= 0xffffffU;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
      test_error(err, "clEnqueueWriteBuffer failed");
    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
      test_error(err, "clEnqueueWriteBuffer failed");


    // test unsigned
    for (i=NUM_PROGRAMS; i<2*NUM_PROGRAMS; i++)
    {
        err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL);
          test_error(err, "clEnqueueNDRangeKernel failed");

        err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
          test_error(err, "clEnqueueNDRangeKernel failed");

        if (verify_uint_mad24( (cl_uint*) input_ptr[0], (cl_uint*) input_ptr[1], (cl_uint*) input_ptr[2], (cl_uint*)output_ptr, n_elems * vector_sizes[i-NUM_PROGRAMS], vector_sizes[i-NUM_PROGRAMS]))
        {
            log_error("UINT_MAD24 %s test failed\n", test_str_names[i]);
            err = -1;
        }
        else
        {
            log_info("UINT_MAD24 %s test passed\n", test_str_names[i]);
            err = 0;
        }

        if (err)
        break;
    }

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseMemObject(streams[3]);
    for (i=0; i<2*NUM_PROGRAMS; i++)
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


