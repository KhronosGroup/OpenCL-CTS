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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/clImageHelper.h"

static const char* rw_kernel_code =
"kernel void test_rw_images(read_write image2d_t src_image) {\n"
"  int tid_x = get_global_id(0);\n"
"  int tid_y = get_global_id(1);\n"
"\n"
"  int2 coords = (int2)(tid_x, tid_y);\n"
"\n"
"  uint4 src_val = read_imageui(src_image, coords);\n"
"  src_val += 3;\n"
"\n"
"  // required to ensure that following read from image at\n"
"  // location coord returns the latest color value.\n"
"  atomic_work_item_fence(CLK_IMAGE_MEM_FENCE,\n"
"                         memory_order_acq_rel,\n"
"                         memory_scope_work_item);\n"
"\n"
"  write_imageui(src_image, coords, src_val);\n"
"}\n";


int test_rw_image_access_qualifier(cl_device_id device_id, cl_context context, cl_command_queue commands, int num_elements)
{

    unsigned int i;

    unsigned int size_x;
    unsigned int size_y;
    unsigned int size;

    cl_int err;

    cl_program program;
    cl_kernel kernel;

    cl_mem_flags flags;
    cl_image_format format;
    cl_mem src_image;

    unsigned int *input;
    unsigned int *output;

    /* Create test input */
    size_x = 4;
    size_y = 4;
    size = size_x * size_y * 4;

    input = (unsigned int *)malloc(size*sizeof(unsigned int));
    output = (unsigned int *)malloc(size*sizeof(unsigned int));

    if (!input && !output) {
        log_error("Error: memory allocation failed\n");
    return -1;
    }

    /* Fill input array with random values */
    for (i = 0; i < size; i++) {
        input[i] = (unsigned int)(rand()/((double)RAND_MAX + 1)*255);
    }

    /* Zero out output array */
    for (i = 0; i < size; i++) {
        output[i] = 0.0f;
    }

    /* Build the program executable */
  err = create_single_kernel_helper_with_build_options(context,&program,&kernel,1,&rw_kernel_code,"test_rw_images", "-cl-std=CL2.0");
    if (err != CL_SUCCESS || !program) {
        log_error("Error: clCreateProgramWithSource failed\n");
    return err;
    }

    /* Create arrays for input and output data */
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT32;

    /* Create input image */
    flags = (cl_mem_flags) (CL_MEM_READ_WRITE
                            | CL_MEM_COPY_HOST_PTR);
    src_image = create_image_2d(context, flags, &format,
                                size_x, size_y, 0,
                                (void *)input, &err);
    if (err != CL_SUCCESS || !src_image) {
        log_error("Error: clCreateImage2D failed\n");
        return err;
    }

    /* Set kernel arguments */
  err = clSetKernelArg(kernel, 0, sizeof(src_image), &src_image);
  if (err != CL_SUCCESS) {
    log_error("Error: clSetKernelArg failed\n");
    return err;
  }

    /* Set kernel execution parameters */
    int dim_count = 2;
    size_t global_dim[2];
    size_t local_dim[2];

    global_dim[0] = size_x;
    global_dim[1] = size_y;

    local_dim[0] = 1;
    local_dim[1] = 1;

    /* Execute kernel */
    err = CL_SUCCESS;
    unsigned int num_iter = 1;
    for(i = 0; i < num_iter; i++) {
        err |= clEnqueueNDRangeKernel(commands, kernel, dim_count,
                                      NULL, global_dim, local_dim,
                                      0, NULL, NULL);
    }

    /* Read back the results from the device to verify the output */
    const size_t origin[3] = {0, 0, 0};
    const size_t region[3] = {size_x, size_y, 1};
    err |= clEnqueueReadImage(commands, src_image, CL_TRUE, origin, region, 0, 0,
                              output, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        log_error("Error: clEnqueueReadBuffer failed\n");
    return err;
    }

    /* Verify the correctness of kernel result */
  err = 0;
    for (i = 0; i < size; i++) {
        if (output[i] != (input[i] + 3)) {
      log_error("Error: mismatch at index %d\n", i);
            err++;
            break;
        }
    }

  /* Release programs, kernel, contect, and memory objects */
    clReleaseMemObject(src_image);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

  /* Deallocate arrays */
    free(input);
    free(output);

    return err;
}
