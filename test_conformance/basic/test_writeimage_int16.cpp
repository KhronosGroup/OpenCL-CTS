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

static const char *rgba16_write_kernel_code =
"__kernel void test_rgba16_write(__global unsigned short *src, write_only image2d_t dstimg)\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    int            indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    float4         color;\n"
"\n"
"    indx *= 4;\n"
"    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);\n"
"    color /= 65535.0f;\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static unsigned short *
generate_16bit_image(int w, int h, MTdata d)
{
    cl_ushort  *ptr = (cl_ushort*)malloc(w * h * 4 * sizeof(cl_ushort));
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (cl_ushort)genrand_int32(d);

    return ptr;
}

// normalized 16bit ints ... get dived by 64k then muled by 64k...
// give the poor things some tolerance
#define MAX_ERR 1

static int
verify_16bit_image(const char *string, cl_ushort *image, cl_ushort *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (abs(outptr[i] - image[i]) > MAX_ERR)
        {
            log_error("%s failed\n", string);
            return -1;
        }
    }

    log_info("%s passed\n", string);
    return 0;
}

int test_writeimage_int16(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[3];
    cl_program program;
    cl_kernel kernel[2];
    cl_image_format    img_format;
    cl_ushort *input_ptr, *output_ptr;
    size_t threads[2];
    int img_width = 512;
    int img_height = 512;
    int i, err, any_err = 0;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, 1};
    size_t length = img_width * img_height * 4 * sizeof(cl_ushort);

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    MTdata d = init_genrand( gRandomSeed );
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

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[1] = create_image_2d(context, CL_MEM_WRITE_ONLY, &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[1])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }
  streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateArray failed\n");
        return -1;
    }

  err = clEnqueueWriteBuffer(queue, streams[2], CL_TRUE, 0, length, input_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    err = create_single_kernel_helper(context, &program, &kernel[0], 1,
                                      &rgba16_write_kernel_code,
                                      "test_rgba16_write");
    if (err) return -1;
    kernel[1] = clCreateKernel(program, "test_rgba16_write", NULL);
    if (!kernel[1])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[0], 0, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel[0], 1, sizeof streams[0], &streams[0]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  err  = clSetKernelArg(kernel[1], 0, sizeof streams[2], &streams[2]);
  err |= clSetKernelArg(kernel[1], 1, sizeof streams[1], &streams[1]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

  threads[0] = (unsigned int)img_width;
  threads[1] = (unsigned int)img_height;

    for (i=0; i<2; i++)
    {
    err = clEnqueueNDRangeKernel(queue, kernel[i], 2, NULL, threads, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clExecuteKernel failed\n");
            return -1;
        }

    err = clEnqueueReadImage(queue, streams[i], CL_TRUE, origin, region, 0, 0, output_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clReadImage failed\n");
            return -1;
        }

        err = verify_16bit_image((i == 0) ? "WRITE_IMAGE_RGBA_UNORM_INT16 test with memflags = CL_MEM_READ_WRITE" :
                             "WRITE_IMAGE_RGBA_UNORM_INT16 test with memflags = CL_MEM_WRITE_ONLY",
                             input_ptr, output_ptr, img_width, img_height);
        any_err |= err;
    }

    // cleanup
    clReleaseMemObject(streams[0]);
    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseProgram(program);
    free(input_ptr);
    free(output_ptr);

    return any_err;
}


