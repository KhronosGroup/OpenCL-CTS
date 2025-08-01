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

#include "testBase.h"

static const char *multireadimage_kernel_code =
"__kernel void test_multireadimage(read_only image2d_t img0, read_only image2d_t img1, \n"
"                                  read_only image2d_t img2, __global float4 *dst, sampler_t sampler)\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    int2           tid = (int2)(tid_x, tid_y);\n"
"    int            indx = tid_y * get_image_width(img1) + tid_x;\n"
"    float4         sum;\n"
"\n"
"    sum = read_imagef(img0, sampler, tid);\n"
"    sum += read_imagef(img1, sampler, tid);\n"
"    sum += read_imagef(img2, sampler, tid);\n"
"\n"
"    dst[indx] = sum;\n"
"}\n";

#define MAX_ERR    1e-7f

static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static unsigned short *
generate_16bit_image(int w, int h, MTdata d)
{
    unsigned short    *ptr = (unsigned short*)malloc(w * h * 4 * sizeof(unsigned short));
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned short)genrand_int32(d);

    return ptr;
}

static float *
generate_float_image(int w, int h, MTdata d)
{
    float   *ptr = (float*)malloc(w * h * 4 * (int)sizeof(float));
    int     i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = get_random_float(-0x40000000, 0x40000000, d);

    return ptr;
}


static int
verify_multireadimage(void *image[], float *outptr, int w, int h)
{
  int     i;
  float   sum;
  float ulp, max_ulp = 0.0f;

  // ULP error of 1.5 for each read_imagef plus 0.5 for each addition.
  float max_ulp_allowed = (float)(3*1.5+2*0.5);

  for (i=0; i<w*h*4; i++)
  {
    sum = (float)((unsigned char *)image[0])[i] / 255.0f;
    sum += (float)((unsigned short *)image[1])[i] / 65535.0f;
    sum += (float)((float *)image[2])[i];
    ulp = Ulp_Error(outptr[i], sum);
    if (ulp > max_ulp)
      max_ulp = ulp;
  }

  if (max_ulp > max_ulp_allowed) {
    log_error("READ_MULTIREADIMAGE_MULTIFORMAT test failed.  Max ulp error = %g\n", max_ulp);
        return -1;
  }

  log_info("READ_MULTIREADIMAGE_MULTIFORMAT test passed.  Max ulp error = %g\n", max_ulp);
  return 0;
}


REGISTER_TEST(mri_multiple)
{
    cl_mem            streams[4];
    cl_image_format    img_format;
    void            *input_ptr[3], *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    threads[2];
    size_t img_width = 512;
    size_t img_height = 512;
    int                i, err;
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    input_ptr[0] = (void *)generate_8888_image(img_width, img_height, d);
    input_ptr[1] = (void *)generate_16bit_image(img_width, img_height, d);
    input_ptr[2] = (void *)generate_float_image(img_width, img_height, d);
    free_mtdata(d); d = NULL;

    output_ptr = (void *)malloc(sizeof(float) * 4 * img_width * img_height);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[0] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format,
                                 img_width, img_height, 0, NULL, NULL);
    if (!streams[0])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }
    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[1] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format,
                                 img_width, img_height, 0, NULL, NULL);
    if (!streams[1])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }
    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[2] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format,
                                 img_width, img_height, 0, NULL, NULL);
    if (!streams[2])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }

    streams[3] =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       sizeof(float) * 4 * img_width * img_height, NULL, NULL);
    if (!streams[3])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    for (i=0; i<3; i++)
    {
      size_t origin[3] = {0,0,0}, region[3]={img_width, img_height,1};
      err = clEnqueueWriteImage(queue, streams[i], CL_TRUE, origin, region, 0, 0, input_ptr[i], 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clWriteImage failed\n");
            return -1;
        }
    }

    err = create_single_kernel_helper( context, &program, &kernel, 1, &multireadimage_kernel_code, "test_multireadimage");
    if (err)
        return -1;

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    for (i=0; i<4; i++)
      err |= clSetKernelArg(kernel, i,sizeof streams[i], &streams[i]);
    err |= clSetKernelArg(kernel, 4, sizeof sampler, &sampler);

    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
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
    err = clEnqueueReadBuffer( queue, streams[3], CL_TRUE, 0, sizeof(float)*4*img_width*img_height, (void *)output_ptr, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueReadBuffer failed\n");
        return -1;
    }

    err = verify_multireadimage(input_ptr, (float*)output_ptr, img_width, img_height);

    // cleanup
    clReleaseSampler(sampler);
    for (i=0; i<4; i++)
        clReleaseMemObject(streams[i]);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    for (i=0; i<3; i++)
        free(input_ptr[i]);
    free(output_ptr);

    return err;
}
