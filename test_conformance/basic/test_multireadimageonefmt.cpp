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

static const char *multireadimage_kernel_code =
"__kernel void test_multireadimage(int n, int m, sampler_t sampler, \n"
"                                  read_only image2d_t img0, read_only image2d_t img1, \n"
"                                  read_only image2d_t img2, read_only image2d_t img3, \n"
"                                  read_only image2d_t img4, read_only image2d_t img5, \n"
"                                  read_only image2d_t img6, __global float4 *dst)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int2   tid = (int2)(tid_x, tid_y);\n"
"    int    indx = tid_y * get_image_width(img5) + tid_x;\n"
"    float4 sum;\n"
"\n"
"    sum = read_imagef(img0, sampler, tid);\n"
"    sum += read_imagef(img1, sampler, tid);\n"
"    sum += read_imagef(img2, sampler, tid);\n"
"    sum += read_imagef(img3, sampler, tid);\n"
"    sum += read_imagef(img4, sampler, tid);\n"
"    sum += read_imagef(img5, sampler, tid);\n"
"    sum += read_imagef(img6, sampler, tid);\n"
"\n"
"    dst[indx] = sum;\n"
"}\n";


static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_multireadimage(void *image[], int num_images, float *outptr, int w, int h)
{
  int     i, j;
  float   sum;
  float ulp, max_ulp = 0.0f;

  // ULP error of 1.5 for each read_imagef plus 0.5 for each addition.
  float max_ulp_allowed = (float)(num_images*1.5+0.5*(num_images-1));

  for (i=0; i<w*h*4; i++)
  {
    sum = 0.0f;
    for (j=0; j<num_images; j++)
    {
      sum += ((float)((unsigned char *)image[j])[i] / 255.0f);
    }
    ulp = Ulp_Error(outptr[i], sum);
    if (ulp > max_ulp)
      max_ulp = ulp;
  }

    if (max_ulp > max_ulp_allowed)
    {
        log_error("READ_MULTIREADIMAGE_RGBA8888 test failed.  Max ULP err = %g\n", max_ulp);
        return -1;
    }
  log_info("READ_MULTIREADIMAGE_RGBA8888 test passed.  Max ULP err = %g\n", max_ulp);
  return 0;
}


int test_mri_one(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[8];
    cl_image_format    img_format;
    void *input_ptr[7], *output_ptr;
    cl_program program;
    cl_kernel kernel;
    size_t threads[2];
    int img_width = 512;
    int img_height = 512;
    int i, err;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, 1};
    size_t length = img_width * img_height * 4 * sizeof(float);
    MTdata d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    output_ptr = malloc(length);

    d = init_genrand( gRandomSeed );
    for (i=0; i<7; i++) {
        input_ptr[i] = (void *)generate_8888_image(img_width, img_height, d);

        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[i] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!streams[i])
        {
          log_error("create_image_2d failed\n");
          return -1;
        }

        err = clEnqueueWriteImage(queue, streams[i], CL_TRUE, origin, region, 0, 0, input_ptr[i], 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
          log_error("clWriteImage failed\n");
          return -1;
        }
    }
    free_mtdata(d); d = NULL;


  streams[7] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[7])
    {
        log_error("clCreateArray failed\n");
        return -1;
    }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &multireadimage_kernel_code, "test_multireadimage");
    if (err)
        return -1;

  cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
  test_error(err, "clCreateSampler failed");

  err  = clSetKernelArg(kernel, 0, sizeof i, &i);
  err |= clSetKernelArg(kernel, 1, sizeof err, &err);
  err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
  for (i = 0; i < 8; i++)
      err |= clSetKernelArg(kernel, 3 + i, sizeof streams[i], &streams[i]);

  if (err != CL_SUCCESS)
  {
      log_error("clSetKernelArgs failed\n");
      return -1;
  }

    threads[0] = (unsigned int)img_width;
    threads[1] = (unsigned int)img_height;

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, threads, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clExecuteKernel failed\n");
    return -1;
  }
  err = clEnqueueReadBuffer(queue, streams[7], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clReadArray failed\n");
    return -1;
  }

  err = verify_multireadimage(input_ptr, 7, (float *)output_ptr, img_width, img_height);

    // cleanup
  clReleaseSampler(sampler);
  for (i = 0; i < 8; i++) clReleaseMemObject(streams[i]);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  for (i = 0; i < 7; i++) free(input_ptr[i]);
  free(output_ptr);

  return err;
}





