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

static const char *r_uint8_kernel_code =
"__kernel void test_r_uint8(read_only image2d_t srcimg, __global unsigned char *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    uint4    color;\n"
"\n"
"    color = read_imageui(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"    dst[indx] = (unsigned char)(color.x);\n"
"\n"
"}\n";


static unsigned char *
generate_8bit_image(int w, int h, MTdata d)
{
    unsigned char    *ptr = (unsigned char*)malloc(w * h * sizeof(unsigned char));
    int             i;

    for (i=0; i<w*h; i++)
      ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_8bit_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("READ_IMAGE_R_UNSIGNED_INT8 test failed\n");
            return -1;
        }
    }

    log_info("READ_IMAGE_R_UNSIGNED_INT8 test passed\n");
    return 0;
}

int
test_image_r8(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem            streams[2];
    cl_image_format    img_format;
    cl_uchar    *input_ptr, *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    threads[3];
    int                img_width = 512;
    int                img_height = 512;
    int                err;
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    img_format.image_channel_order = CL_R;
    img_format.image_channel_data_type = CL_UNSIGNED_INT8;

    // early out if this image type is not supported
    if( ! is_image_format_supported( context, (cl_mem_flags)(CL_MEM_READ_ONLY), CL_MEM_OBJECT_IMAGE2D, &img_format ) ) {
        log_info("WARNING: Image type not supported; skipping test.\n");
        return 0;
    }

    d = init_genrand( gRandomSeed );
    input_ptr = generate_8bit_image(img_width, img_height, d);
    free_mtdata(d); d = NULL;

    output_ptr = (cl_uchar*)malloc(sizeof(cl_uchar) * img_width * img_height);
    streams[0] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_ONLY), &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[0])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }

    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeof(cl_uchar) * img_width*img_height, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    size_t origin[3] = {0,0,0}, region[3]={img_width, img_height, 1};
    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE,
                            origin, region, 0, 0,
                            input_ptr,
                            0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteImage failed: %d\n", err);
        return -1;
    }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &r_uint8_kernel_code, "test_r_uint8" );
    if (err) {
    log_error("Failed to create kernel and program: %d\n", err);
    return -1;
  }

  cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
  test_error(err, "clCreateSampler failed");

  err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed: %d\n", err);
        return -1;
    }

    threads[0] = (size_t)img_width;
    threads[1] = (size_t)img_height;
    err = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueNDRangeKernel failed\n");
        return -1;
    }

    err = clEnqueueReadBuffer( queue, streams[1], CL_TRUE, 0, sizeof(cl_uchar)*img_width*img_height, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    err = verify_8bit_image(input_ptr, output_ptr, img_width, img_height);


    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseSampler(sampler);
    free(input_ptr);
    free(output_ptr);

    return err;
}





