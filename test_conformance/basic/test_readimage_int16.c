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

static const char *rgba16_kernel_code =
"__kernel void test_rgba16(read_only image2d_t srcimg, __global ushort4 *dst, sampler_t smp)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, smp, (int2)(tid_x, tid_y));\n"
"    ushort4 dst_write;\n"
"    dst_write.x = convert_ushort_rte(color.x * 65535.0f);\n"
"    dst_write.y = convert_ushort_rte(color.y * 65535.0f);\n"
"    dst_write.z = convert_ushort_rte(color.z * 65535.0f);\n"
"    dst_write.w = convert_ushort_rte(color.w * 65535.0f);\n"
"    dst[indx] = dst_write;\n"
"\n"
"}\n";


static unsigned short *
generate_16bit_image(int w, int h, MTdata d)
{
    cl_ushort    *ptr = (cl_ushort*)malloc(w * h * 4 * sizeof(cl_ushort));
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (cl_ushort)genrand_int32(d);

    return ptr;
}

static int
verify_16bit_image(cl_ushort *image, cl_ushort *outptr, int w, int h)
{
    int     i;
    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("READ_IMAGE_RGBA_UNORM_INT16 test failed\n");
            return -1;
        }
    }

    log_info("READ_IMAGE_RGBA_UNORM_INT16 test passed\n");
    return 0;
}

int test_readimage_int16(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[2];
    cl_program program;
    cl_kernel kernel;
    cl_image_format    img_format;
    cl_ushort *input_ptr, *output_ptr;
    size_t threads[2];
    int img_width = 512;
    int img_height = 512;
    int err;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, 1};
    size_t length = img_width * img_height * 4 * sizeof(cl_ushort);
    MTdata d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    input_ptr = generate_16bit_image(img_width, img_height, d);
    free_mtdata(d); d = NULL;

    output_ptr = (cl_ushort*)malloc(length);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[0] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[0])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }
    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateArray failed\n");
        return -1;
    }

    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0, input_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clWriteImage failed\n");
        return -1;
    }

  err = create_single_kernel_helper(context, &program, &kernel, 1, &rgba16_kernel_code, "test_rgba16" );
  if (err)
    return -1;

  cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
  test_error(err, "clCreateSampler failed");

  err  = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
  err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
  err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
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
    log_error("%s clEnqueueNDRangeKernel failed\n", __FUNCTION__);
    return -1;
  }

  err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    log_error("clEnqueueReadBuffer failed\n");
    return -1;
  }

  err = verify_16bit_image(input_ptr, output_ptr, img_width, img_height);

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


