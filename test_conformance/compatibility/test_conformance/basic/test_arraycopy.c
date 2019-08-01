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

const char *copy_kernel_code =
"__kernel void test_copy(__global unsigned int *src, __global unsigned int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src[tid];\n"
"}\n";

int
test_arraycopy(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_uint    *input_ptr, *output_ptr;
    cl_mem                streams[4], results;
    cl_program          program;
    cl_kernel            kernel;
    unsigned            num_elements = 128 * 1024;
    cl_uint             num_copies = 1;
    size_t                delta_offset;
    unsigned            i;
    cl_int err;
    MTdata              d;

    int error_count = 0;

    input_ptr = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);
    output_ptr = (cl_uint*)malloc(sizeof(cl_uint) * num_elements);

    // results
    results = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_uint) * num_elements, NULL, &err);
    test_error(err, "clCreateBuffer failed");

/*****************************************************************************************************************************************/
#pragma mark client backing

    log_info("Testing CL_MEM_USE_HOST_PTR buffer with clEnqueueCopyBuffer\n");
    // randomize data
    d = init_genrand( gRandomSeed );
    for (i=0; i<num_elements; i++)
        input_ptr[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);

    // client backing
    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_USE_HOST_PTR), sizeof(cl_uint) * num_elements, input_ptr, &err);
    test_error(err, "clCreateBuffer failed");

    delta_offset = num_elements * sizeof(cl_uint) / num_copies;
    for (i=0; i<num_copies; i++)
    {
        size_t    offset = i * delta_offset;
        err = clEnqueueCopyBuffer(queue, streams[0], results, offset, offset, delta_offset, 0, NULL, NULL);
        test_error(err, "clEnqueueCopyBuffer failed");
    }

    // Try upload from client backing
    err = clEnqueueReadBuffer( queue, results, CL_TRUE, 0, num_elements*sizeof(cl_uint), output_ptr, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed");

    for (i=0; i<num_elements; i++)
    {
        if (input_ptr[i] != output_ptr[i])
        {
            err = -1;
            error_count++;
        }
    }

    if (err)
        log_error("\tCL_MEM_USE_HOST_PTR buffer with clEnqueueCopyBuffer FAILED\n");
    else
        log_info("\tCL_MEM_USE_HOST_PTR buffer with clEnqueueCopyBuffer passed\n");



#pragma mark framework backing (no client data)

    log_info("Testing with clEnqueueWriteBuffer and clEnqueueCopyBuffer\n");
    // randomize data
    for (i=0; i<num_elements; i++)
        input_ptr[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);

    // no backing
    streams[2] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE) , sizeof(cl_uint) * num_elements, NULL, &err);
    test_error(err, "clCreateBuffer failed");

    for (i=0; i<num_copies; i++)
    {
        size_t    offset = i * delta_offset;

        // Copy the array up from host ptr
        err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, sizeof(cl_uint)*num_elements, input_ptr, 0, NULL, NULL);
        test_error(err, "clEnqueueWriteBuffer failed");

        err = clEnqueueCopyBuffer(queue, streams[2], results, offset, offset, delta_offset, 0, NULL, NULL);
        test_error(err, "clEnqueueCopyBuffer failed");
    }

    err = clEnqueueReadBuffer( queue, results, true, 0, num_elements*sizeof(cl_uint), output_ptr, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed");

    for (i=0; i<num_elements; i++)
    {
        if (input_ptr[i] != output_ptr[i])
        {
            err = -1;
            error_count++;
            break;
        }
    }

    if (err)
        log_error("\tclEnqueueWriteBuffer and clEnqueueCopyBuffer FAILED\n");
    else
        log_info("\tclEnqueueWriteBuffer and clEnqueueCopyBuffer passed\n");

/*****************************************************************************************************************************************/
#pragma mark kernel copy test

    log_info("Testing CL_MEM_USE_HOST_PTR buffer with kernel copy\n");
    // randomize data
    for (i=0; i<num_elements; i++)
        input_ptr[i] = (cl_uint)(genrand_int32(d) & 0x7FFFFFFF);
    free_mtdata(d); d= NULL;

    // client backing
  streams[3] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_USE_HOST_PTR), sizeof(cl_uint) * num_elements, input_ptr, &err);
  test_error(err, "clCreateBuffer failed");

  err = create_single_kernel_helper(context, &program, &kernel, 1, &copy_kernel_code, "test_copy" );
  test_error(err, "create_single_kernel_helper failed");

  err = clSetKernelArg(kernel, 0, sizeof streams[3], &streams[3]);
  err |= clSetKernelArg(kernel, 1, sizeof results, &results);
  test_error(err, "clSetKernelArg failed");

  size_t threads[3] = {num_elements, 0, 0};

    err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
  test_error(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadBuffer( queue, results, CL_TRUE, 0, num_elements*sizeof(cl_uint), output_ptr, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed");

    for (i=0; i<num_elements; i++)
    {
        if (input_ptr[i] != output_ptr[i])
        {
            err = -1;
      error_count++;
            break;
        }
    }

  // Keep track of multiple errors.
  if (error_count != 0)
    err = error_count;

    if (err)
        log_error("\tCL_MEM_USE_HOST_PTR buffer with kernel copy FAILED\n");
    else
        log_info("\tCL_MEM_USE_HOST_PTR buffer with kernel copy passed\n");


  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(results);
  clReleaseMemObject(streams[0]);
  clReleaseMemObject(streams[2]);
  clReleaseMemObject(streams[3]);

  free(input_ptr);
  free(output_ptr);

    return err;
}



