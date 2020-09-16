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


static const char *rgbaFFFF_kernel_code =
"__kernel void test_rgbaFFFF(read_only image3d_t srcimg, __global float *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    indx *= 4;\n"
"    dst[indx+0] = color.x;\n"
"    dst[indx+1] = color.y;\n"
"    dst[indx+2] = color.z;\n"
"    dst[indx+3] = color.w;\n"
"\n"
"}\n";


static float *
generate_float_image(int w, int h, int d, MTdata data)
{
    float   *ptr = (float*)malloc(w * h * d * 4 * sizeof(float));
    int     i;

    for (i=0; i<w*h*d*4; i++)
        ptr[i] = get_random_float(-0x40000000, 0x40000000, data);

    return ptr;
}

static int
verify_float_image(float *image, float *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("READ_IMAGE3D_RGBA_FLOAT test failed\n");
            return -1;
        }
    }

    log_info("READ_IMAGE3D_RGBA_FLOAT test passed\n");
    return 0;
}


int test_readimage3d_fp32(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_program program;
    cl_kernel kernel;
    cl_image_format    img_format;
    float *input_ptr, *output_ptr;
    size_t threads[3];
    int img_width = 64;
    int img_height = 64;
    int img_depth = 64;
    int err;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, img_depth};
    size_t length = img_width * img_height * img_depth * 4 * sizeof(float);

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

    MTdata d = init_genrand( gRandomSeed );
    input_ptr = generate_float_image(img_width, img_height, img_depth, d);
    free_mtdata(d); d = NULL;

    output_ptr = (float*)malloc(length);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[0] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
  test_error(err, "create_image_3d failed");

  streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error(err, "clCreateBuffer failed");

    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0, input_ptr, 0, NULL, NULL);
  test_error(err, "clEnqueueWriteImage failed");

  err = create_single_kernel_helper(context, &program, &kernel, 1, &rgbaFFFF_kernel_code, "test_rgbaFFFF" );
  if (err)
    return -1;

  cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
  test_error(err, "clCreateSampler failed");

  err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
  test_error(err, "clSetKernelArg failed");

    threads[0] = (unsigned int)img_width;
    threads[1] = (unsigned int)img_height;
    threads[2] = (unsigned int)img_depth;
  err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, threads, NULL, 0, NULL, NULL);
  test_error(err, "clEnqueueNDRangeKernel failed");

  err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
  test_error(err, "clEnqueueReadBuffer failed");

  err = verify_float_image(input_ptr, output_ptr, img_width, img_height, img_depth);

    // cleanup
  clReleaseSampler(sampler);
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return err;
}


