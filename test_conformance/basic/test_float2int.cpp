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

const char *float2int_kernel_code =
"__kernel void test_float2int(__global float *src, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = (int)src[tid];\n"
"\n"
"}\n";


int
verify_float2int(cl_float *inptr, cl_int *outptr, int n)
{
  int     i;

  for (i=0; i<n; i++)
  {
    if (outptr[i] != (int)inptr[i])
    {
      log_error("FLOAT2INT test failed\n");
      return -1;
    }
  }

  log_info("FLOAT2INT test passed\n");
  return 0;
}


int
test_float2int(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[2];
    cl_float        *input_ptr;
    cl_int          *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    threads[1];
    int                err;
    int                i;
    MTdata          d;

    input_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);
    output_ptr = (cl_int*)malloc(sizeof(cl_int) * num_elements);
    streams[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_float) * num_elements, NULL, NULL);
    if (!streams[0])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(cl_int) * num_elements, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        input_ptr[i] = get_random_float(-MAKE_HEX_FLOAT( 0x1.0p31f, 0x1, 31), MAKE_HEX_FLOAT( 0x1.0p31f, 0x1, 31), d);
    free_mtdata(d); d = NULL;

    err = clEnqueueWriteBuffer(queue, streams[0], CL_TRUE, 0, sizeof(cl_float)*num_elements, (void *)input_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper(context, &program, &kernel, 1, &float2int_kernel_code, "test_float2int");
    if (err != CL_SUCCESS)
    {
        log_error("create_single_kernel_helper failed\n");
        return -1;
    }

  err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
  err = clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
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

    err = clEnqueueReadBuffer( queue, streams[1], true, 0, sizeof(cl_int)*num_elements, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    err = verify_float2int(input_ptr, output_ptr, num_elements);

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return err;
}





