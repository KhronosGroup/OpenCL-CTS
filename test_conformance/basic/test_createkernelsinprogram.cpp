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

const char *sample_single_kernel = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n"};

const char *sample_double_kernel = {
"__kernel void sample_test(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n"
"__kernel void sample_test2(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n"};


int
test_createkernelsinprogram(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_program        program;
    cl_kernel        kernel[2];
    unsigned int    num_kernels;
    int                err;

    err = create_single_kernel_helper(context, &program, NULL, 1, &sample_single_kernel, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("create_single_kernel_helper failed\n");
        return -1;
    }

    err = clCreateKernelsInProgram(program, 1, kernel, &num_kernels);
    if ( (err != CL_SUCCESS) || (num_kernels != 1) )
    {
        log_error("clCreateKernelsInProgram test failed for a single kernel\n");
        return -1;
    }

    clReleaseKernel(kernel[0]);
    clReleaseProgram(program);

    err = create_single_kernel_helper(context, &program, NULL, 1, &sample_double_kernel, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("create_single_kernel_helper failed\n");
        return -1;
    }

    err = clCreateKernelsInProgram(program, 2, kernel, &num_kernels);
    if ( (err != CL_SUCCESS) || (num_kernels != 2) )
    {
        log_error("clCreateKernelsInProgram test failed for two kernels\n");
        return -1;
    }

  log_info("clCreateKernelsInProgram test passed\n");

    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseProgram(program);


    return err;
}





