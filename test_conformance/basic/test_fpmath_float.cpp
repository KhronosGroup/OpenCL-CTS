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
verify_fpadd(float *inptrA, float *inptrB, float *outptr, int n)
{
    float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] + inptrB[i];
        if (r != outptr[i])
        {
            log_error("FP_ADD float test failed\n");
            return -1;
        }
    }

    log_info("FP_ADD float test passed\n");
    return 0;
}

static int
verify_fpsub(float *inptrA, float *inptrB, float *outptr, int n)
{
    float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] - inptrB[i];
        if (r != outptr[i])
        {
            log_error("FP_SUB float test failed\n");
            return -1;
        }
    }

    log_info("FP_SUB float test passed\n");
    return 0;
}

static int
verify_fpmul(float *inptrA, float *inptrB, float *outptr, int n)
{
    float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] * inptrB[i];
        if (r != outptr[i])
        {
            log_error("FP_MUL float test failed\n");
            return -1;
        }
    }

    log_info("FP_MUL float test passed\n");
    return 0;
}


int
test_fpmath_float(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[4];
    cl_program program[3];
    cl_kernel kernel[3];

    float *input_ptr[3], *output_ptr, *p;
    size_t threads[1];
    int err, i;
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


    input_ptr[0] = (cl_float*)malloc(length);
    input_ptr[1] = (cl_float*)malloc(length);
    input_ptr[2] = (cl_float*)malloc(length);
    output_ptr   = (cl_float*)malloc(length);

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
        err = clEnqueueNDRangeKernel(queue, kernel[i], 1, NULL, threads, NULL, 0, NULL, NULL);
        test_error( err, "clEnqueueNDRangeKernel failed.");

        err = clEnqueueReadBuffer(queue, streams[3], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
        test_error( err, "clEnqueueReadBuffer failed.");

        if( isRTZ )
            set_round( kRoundTowardZero, kfloat );

        switch (i)
        {
            case 0:
                err = verify_fpadd(input_ptr[0], input_ptr[1], output_ptr, num_elements);
                break;
            case 1:
                err = verify_fpsub(input_ptr[0], input_ptr[1], output_ptr, num_elements);
                break;
            case 2:
                err = verify_fpmul(input_ptr[0], input_ptr[1], output_ptr, num_elements);
                break;
        }

        if( isRTZ )
            set_round( oldMode, kfloat );

        if (err)
            break;
    }

    // cleanup
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


