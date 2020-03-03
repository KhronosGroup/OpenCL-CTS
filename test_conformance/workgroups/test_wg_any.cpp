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


const char *wg_any_kernel_code =
"__kernel void test_wg_any(global float *input, global int *output)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    int result = work_group_any((input[tid] > input[tid+1]));\n"
"    output[tid] = result;\n"
"}\n";


static int
verify_wg_any(float *inptr, int *outptr, size_t n, size_t wg_size)
{
    size_t     i, j;

    for (i=0; i<n; i+=wg_size)
    {
        int predicate_any = 0x0;
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if (inptr[i+j] > inptr[i+j+1])
            {
                predicate_any = 0xFFFFFFFF;
                break;
            }
        }
        for (j=0; j<((n-i) > wg_size ? wg_size : (n-i)); j++)
        {
            if ( (predicate_any && (outptr[i+j] == 0)) ||
                 ((predicate_any == 0) && outptr[i+j]) )
            {
                log_info("work_group_any: Error at %lu: expected = %d, got = %d\n", i+j, predicate_any, outptr[i+j]);
                return -1;
            }
        }
    }

    return 0;
}

int
test_work_group_any(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem       streams[2];
    cl_float     *input_ptr[1], *p;
    cl_int       *output_ptr;
    cl_program   program;
    cl_kernel    kernel;
    void         *values[2];
    size_t       threads[1];
    size_t       wg_size[1];
    size_t       num_elements;
    int          err;
    int          i;
    MTdata       d;

    err = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &wg_any_kernel_code, "test_wg_any", "-cl-std=CL2.0" );
    if (err)
        return -1;

    // "wg_size" is limited to that of the first dimension as only a 1DRange is executed.
    err = get_max_allowed_1d_work_group_size_on_device(device, kernel, wg_size);
    test_error(err, "get_max_allowed_1d_work_group_size_on_device failed");

    num_elements = n_elems;

    input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * (num_elements+1));
    output_ptr = (cl_int*)malloc(sizeof(cl_int) * (num_elements+1));
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_float) * (num_elements+1), NULL, NULL );
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_int) * num_elements, NULL, NULL );
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    p = input_ptr[0];
    d = init_genrand( gRandomSeed );
    for (i=0; i<(num_elements+1); i++)
    {
        p[i] = get_random_float((float)(-100000.f * M_PI), (float)(100000.f * M_PI) ,d);
    }
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer( queue, streams[0], true, 0, sizeof(cl_float)*(num_elements+1), (void *)input_ptr[0], 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    values[0] = streams[0];
    values[1] = streams[1];
    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0] );
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1] );
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    // Line below is troublesome...
    threads[0] = (size_t)n_elems;
    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, wg_size, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    cl_uint dead = 0xdeaddead;
    memset_pattern4(output_ptr, &dead, sizeof(cl_float)*num_elements);
    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    if (verify_wg_any(input_ptr[0], output_ptr, num_elements, wg_size[0]))
    {
        log_error("work_group_any test failed\n");
        return -1;
    }
    log_info("work_group_any test passed\n");

    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(output_ptr);

    return err;
}

