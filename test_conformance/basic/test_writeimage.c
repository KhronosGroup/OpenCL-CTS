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

static const char *bgra8888_write_kernel_code =
"\n"
"__kernel void test_bgra8888_write(__global unsigned char *src, write_only image2d_t dstimg)\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    int            indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    float4         color;\n"
"\n"
"    indx *= 4;\n"
"    color = (float4)((float)src[indx+2], (float)src[indx+1], (float)src[indx+0], (float)src[indx+3]);\n"
"    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static const char *rgba8888_write_kernel_code =
"\n"
"__kernel void test_rgba8888_write(__global unsigned char *src, write_only image2d_t dstimg)\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    int            indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    float4         color;\n"
"\n"
"    indx *= 4;\n"
"    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);\n"
"    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    cl_uchar   *ptr = (cl_uchar *)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (cl_uchar)genrand_int32(d);

    return ptr;
}

static int
verify_bgra8888_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("WRITE_IMAGE_BGRA_UNORM_INT8 test failed\n");
            return -1;
        }
    }

    log_info("WRITE_IMAGE_BGRA_UNORM_INT8 test passed\n");
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
            log_error("WRITE_IMAGE_RGBA_UNORM_INT8 test failed\n");
            return -1;
        }
    }

    log_info("WRITE_IMAGE_RGBA_UNORM_INT8 test passed\n");
    return 0;
}


int test_writeimage(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_mem streams[6];
    cl_program program[2];
    cl_kernel kernel[4];

    unsigned char    *input_ptr[2], *output_ptr;
    cl_image_format    img_format;
    cl_image_format *supported_formats;
    size_t threads[2];
    int img_width = 512;
    int img_height = 512;
    int i, err, any_err = 0;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {img_width, img_height, 1};
    size_t length = img_width * img_height * 4 * sizeof(unsigned char);
    int supportsBGRA = 0;
    cl_uint numFormats = 0;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    MTdata d = init_genrand( gRandomSeed );
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
    streams[1] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[1])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        img_format.image_channel_order = CL_BGRA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[2] = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!streams[2])
        {
            log_error("clCreateImage2D failed\n");
            return -1;
        }
    }

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[3] = create_image_2d(context, CL_MEM_WRITE_ONLY, &img_format, img_width, img_height, 0, NULL, NULL);
    if (!streams[3])
    {
        log_error("create_image_2d failed\n");
        return -1;
    }

    streams[4] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[4])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }
    streams[5] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, NULL);
    if (!streams[5])
    {
        log_error("clCreateBuffer failed\n");
        return -1;
    }

    err = clEnqueueWriteBuffer(queue, streams[4], CL_TRUE, 0, length, input_ptr[0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }
    err = clEnqueueWriteBuffer(queue, streams[5], CL_TRUE, 0, length, input_ptr[1], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        log_error("clEnqueueWriteBuffer failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &bgra8888_write_kernel_code, "test_bgra8888_write" );
        if (err)
                return -1;

        kernel[2] = clCreateKernel(program[0], "test_bgra8888_write", NULL);
        if (!kernel[2])
        {
                log_error("clCreateKernel failed\n");
                return -1;
        }
    }

    err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &rgba8888_write_kernel_code, "test_rgba8888_write" );
    if (err)
    return -1;
    kernel[3] = clCreateKernel(program[1], "test_rgba8888_write", NULL);
    if (!kernel[3])
    {
        log_error("clCreateKernel failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        err  = clSetKernelArg(kernel[0], 0, sizeof streams[4], &streams[4]);
        err |= clSetKernelArg(kernel[0], 1, sizeof streams[0], &streams[0]);
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            return -1;
        }
    }

    err  = clSetKernelArg(kernel[1], 0, sizeof streams[5], &streams[5]);
    err |= clSetKernelArg(kernel[1], 1, sizeof streams[1], &streams[1]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    if(supportsBGRA)
    {
        err  = clSetKernelArg(kernel[2], 0, sizeof streams[4], &streams[4]);
        err |= clSetKernelArg(kernel[2], 1, sizeof streams[2], &streams[2]);
        if (err != CL_SUCCESS)
        {
            log_error("clSetKernelArgs failed\n");
            return -1;
        }
    }

    err  = clSetKernelArg(kernel[3], 0, sizeof streams[5], &streams[5]);
    err |= clSetKernelArg(kernel[3], 1, sizeof streams[3], &streams[3]);
    if (err != CL_SUCCESS)
    {
        log_error("clSetKernelArgs failed\n");
        return -1;
    }

    threads[0] = (unsigned int)img_width;
    threads[1] = (unsigned int)img_height;

    for (i=0; i<4; i++)
    {
         if(!supportsBGRA && (i == 0 || i == 2))
            continue;

        err = clEnqueueNDRangeKernel(queue, kernel[i], 2, NULL, threads, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clEnqueueNDRangeKernel failed\n");
            return -1;
        }

        err = clEnqueueReadImage(queue, streams[i], CL_TRUE, origin, region, 0, 0, output_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clReadImage failed\n");
            return -1;
        }

        switch (i)
        {
            case 0:
            case 2:
                err = verify_bgra8888_image(input_ptr[i&0x01], output_ptr, img_width, img_height);
                break;
            case 1:
            case 3:
                err = verify_rgba8888_image(input_ptr[i&0x01], output_ptr, img_width, img_height);
                break;
        }

        //if (err)
        //break;

        any_err |= err;
    }

    // cleanup
    if(supportsBGRA)
        clReleaseMemObject(streams[0]);

    clReleaseMemObject(streams[1]);

    if(supportsBGRA)
        clReleaseMemObject(streams[2]);

    clReleaseMemObject(streams[3]);
    clReleaseMemObject(streams[4]);
    clReleaseMemObject(streams[5]);
    for (i=0; i<2; i++)
    {
        if(i == 0 && !supportsBGRA)
            continue;

        clReleaseKernel(kernel[i]);
        clReleaseKernel(kernel[i+2]);
        clReleaseProgram(program[i]);
    }
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    return any_err;
}
