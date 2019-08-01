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

static const char *bgra8888_kernel_code =
"\n"
"__kernel void test_bgra8888(read_only image3d_t srcimg, __global float4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    dst[indx].x = color.z;\n"
"    dst[indx].y = color.y;\n"
"    dst[indx].z = color.x;\n"
"    dst[indx].w = color.w;\n"
"\n"
"}\n";


static const char *rgba8888_kernel_code =
"\n"
"__kernel void test_rgba8888(read_only image3d_t srcimg, __global float4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    //indx *= 4;\n"
"    dst[indx].x = color.x;\n"
"    dst[indx].y = color.y;\n"
"    dst[indx].z = color.z;\n"
"    dst[indx].w = color.w;\n"
"\n"
"}\n";


static unsigned char *
generate_3d_image8(int w, int h, int d, MTdata data)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * d * 4);
    int             i;

    for (i=0; i<w*h*d*4; i++)
        ptr[i] = (unsigned char)genrand_int32(data);

    return ptr;
}

static int
verify_3d_image8(double *image, float *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != (float)image[i])
        {
            float ulps = Ulp_Error( outptr[i], image[i]);

            if(! (fabsf(ulps) < 1.5f) )
            {
                log_error( "ERROR: Data sample %d does not validate! Expected (%a), got (%a), ulp %f\n",
                    (int)i, image[i], outptr[ i ],  ulps );
                return -1;
            }
        }
    }

    return 0;
}

static double *
prepare_reference(unsigned char * input_ptr, int w, int h, int d)
{
    double   *ptr = (double*)malloc(w * h * d * 4 * sizeof(double));
    int         i;
    for (i=0; i<w*h*d*4; i++)
        ptr[i] = ((double)input_ptr[i]/255);

    return ptr;
}


int test_readimage3d(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[3];
    cl_program program[2];
    cl_kernel kernel[2];
    cl_image_format    img_format;
    unsigned char    *input_ptr[2];
    float *output_ptr;
    double *ref_ptr[2];
    size_t threads[3];
    int img_width = 64;
    int img_height = 64;
    int img_depth = 64;
    int i, err;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, img_depth};
    size_t length = img_width * img_height * img_depth * 4 * sizeof(float);


    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

    MTdata d = init_genrand( gRandomSeed );
    input_ptr[0] = generate_3d_image8(img_width, img_height, img_depth, d);
    input_ptr[1] = generate_3d_image8(img_width, img_height, img_depth, d);
    ref_ptr[0] = prepare_reference(input_ptr[0], img_width, img_height, img_depth);
    ref_ptr[1] = prepare_reference(input_ptr[1], img_width, img_height, img_depth);
    free_mtdata(d); d = NULL;
    output_ptr = (float*)malloc(length);

    img_format.image_channel_order = CL_BGRA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[0] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[1] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
  test_error(err, "create_image_3d failed");

  streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
  test_error(err, "clCreateBuffer failed");

    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0, input_ptr[0], 0, NULL, NULL);
  test_error(err, "clEnqueueWriteImage failed");

    err = clEnqueueWriteImage(queue, streams[1], CL_TRUE, origin, region, 0, 0, input_ptr[1], 0, NULL, NULL);
  test_error(err, "clEnqueueWriteImage failed");

  err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &bgra8888_kernel_code, "test_bgra8888" );
  if (err)
    return -1;

  err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &rgba8888_kernel_code, "test_rgba8888" );
  if (err)
    return -1;

  cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
  test_error(err, "clCreateSampler failed");

  err  = clSetKernelArg(kernel[0], 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel[0], 1, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel[0], 2, sizeof sampler, &sampler);
  test_error(err, "clSetKernelArg failed");

  err  = clSetKernelArg(kernel[1], 0, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel[1], 1, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel[1], 2, sizeof sampler, &sampler);
  test_error(err, "clSetKernelArg failed");

    threads[0] = (unsigned int)img_width;
    threads[1] = (unsigned int)img_height;
     threads[2] = (unsigned int)img_depth;

  for (i=0; i<2; i++)
  {
    err = clEnqueueNDRangeKernel(queue, kernel[i], 3, NULL, threads, NULL, 0, NULL, NULL);
    test_error(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
    test_error(err, "clEnqueueReadBuffer failed");

    switch (i)
    {
      case 0:
        err = verify_3d_image8(ref_ptr[i], output_ptr, img_width, img_height, img_depth);
        if ( err != 0 )
            log_info("READ_IMAGE3D_BGRA_UNORM_INT8 test passed\n");
        break;
      case 1:
        err = verify_3d_image8(ref_ptr[i], output_ptr, img_width, img_height, img_depth);
        if ( err != 0 )
            log_info("READ_IMAGE3D_RGBA_UNORM_INT8 test passed\n");
        break;
    }

    if (err)
      break;
  }

    // cleanup
  clReleaseSampler(sampler);
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
  for (i=0; i<2; i++)
  {
    clReleaseKernel(kernel[i]);
    clReleaseProgram(program[i]);
  }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);
  free(ref_ptr[0]);
  free(ref_ptr[1]);

    return err;
}


