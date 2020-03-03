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

static const char *bgra8888_kernel_code =
"\n"
"__kernel void test_bgra8888(read_only image2d_t srcimg, __global uchar4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y)) * 255.0f;\n"
"    dst[indx] = convert_uchar4_rte(color.zyxw);\n"
"\n"
"}\n";


static const char *rgba8888_kernel_code =
"\n"
"__kernel void test_rgba8888(read_only image2d_t srcimg, __global uchar4 *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y)) * 255.0f;\n"
"    dst[indx] = convert_uchar4_rte(color);\n"
"\n"
"}\n";


static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32( d);

    return ptr;
}

static int
verify_bgra8888_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("READ_IMAGE_BGRA_UNORM_INT8 test failed\n");
            return -1;
        }
    }

    log_info("READ_IMAGE_BGRA_UNORM_INT8 test passed\n");
    return 0;
}

static int
verify_rgba8888_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("READ_IMAGE_RGBA_UNORM_INT8 test failed\n");
            return -1;
        }
    }

    log_info("READ_IMAGE_RGBA_UNORM_INT8 test passed\n");
    return 0;
}


int test_readimage(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[3];
    cl_program program[2];
    cl_kernel kernel[2];
    cl_image_format    img_format;
    cl_image_format *supported_formats;
    unsigned char    *input_ptr[2], *output_ptr;
    size_t threads[2];
    int img_width = 512;
    int img_height = 512;
    int i, err;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, 1};
    size_t length = img_width * img_height * 4 * sizeof(unsigned char);
    MTdata d = init_genrand( gRandomSeed );
    int supportsBGRA = 0;
    cl_uint numFormats = 0;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    input_ptr[0] = generate_8888_image(img_width, img_height, d);
    input_ptr[1] = generate_8888_image(img_width, img_height, d);
    free_mtdata(d); d = NULL;

    output_ptr = (unsigned char*)malloc(length);

    if(gIsEmbedded)
    {
        /* Get the supported image formats to see if BGRA is supported */
        clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &numFormats);
        supported_formats = (cl_image_format *) malloc(sizeof(cl_image_format) * numFormats);
        clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, numFormats, supported_formats, NULL);

        for(i = 0; i < numFormats; i++)
        {
            if(supported_formats[i].image_channel_order == CL_BGRA)
            {
                supportsBGRA = 1;
                break;
            }
        }
    }
    else
    {
        supportsBGRA = 1;
    }

    if(supportsBGRA)
    {
        img_format.image_channel_order = CL_BGRA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[0] = clCreateImage2D(context, CL_MEM_READ_WRITE, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!streams[0])
        {
            log_error("clCreateImage2D failed\n");
            return -1;
        }
    }

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[1] = clCreateImage2D(context, CL_MEM_READ_WRITE, &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[1])
    {
        log_error("clCreateImage2D failed\n");
        return -1;
    }
    streams[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[2])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0, input_ptr[0], 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueWriteImage failed\n");
            return -1;
        }
    }

    err = clEnqueueWriteImage(queue, streams[1], CL_TRUE, origin, region, 0, 0, input_ptr[1], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteImage failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &bgra8888_kernel_code, "test_bgra8888" );
        if (err)
            return -1;
    }

    err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &rgba8888_kernel_code, "test_rgba8888" );
    if (err)
        return -1;

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    if(supportsBGRA)
    {
        err  = clSetKernelArg(kernel[0], 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernel[0], 1, sizeof streams[2], &streams[2]);
        err |= clSetKernelArg(kernel[0], 2, sizeof sampler, &sampler);
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArg failed\n");
            return -1;
        }
    }

    err  = clSetKernelArg(kernel[1], 0, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(kernel[1], 1, sizeof streams[2], &streams[2]);
    err |= clSetKernelArg(kernel[1], 2, sizeof sampler, &sampler);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArg failed\n");
        return -1;
    }

    threads[0] = (unsigned int)img_width;
    threads[1] = (unsigned int)img_height;

    for (i=0; i<2; i++)
    {
        if(i == 0 && !supportsBGRA)
            continue;

        err = clEnqueueNDRangeKernel(queue, kernel[i], 2, NULL, threads, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("%s clEnqueueNDRangeKernel failed\n", __FUNCTION__);
            return -1;
        }
        err = clEnqueueReadBuffer(queue, streams[2], CL_TRUE, 0, length, output_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueReadBuffer failed\n");
            return -1;
        }

        switch (i)
        {
            case 0:
                err = verify_bgra8888_image(input_ptr[i], output_ptr, img_width, img_height);
                break;
            case 1:
                err = verify_rgba8888_image(input_ptr[i], output_ptr, img_width, img_height);
                break;
        }

        if (err)
            break;
    }

    // cleanup
    clReleaseSampler(sampler);

    if(supportsBGRA)
            clReleaseMemObject(streams[0]);

    clReleaseMemObject(streams[1]);
    clReleaseMemObject(streams[2]);
    for (i=0; i<2; i++)
    {
        if(i == 0 && !supportsBGRA)
            continue;

        clReleaseKernel(kernel[i]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    return err;
}
