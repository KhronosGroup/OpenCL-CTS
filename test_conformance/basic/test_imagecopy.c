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

static unsigned char *
generate_rgba8_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_rgba8_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


static unsigned short *
generate_rgba16_image(int w, int h, MTdata d)
{
    unsigned short    *ptr = (unsigned short *)malloc(w * h * 4 * sizeof(unsigned short));
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned short)genrand_int32(d);

    return ptr;
}

static int
verify_rgba16_image(unsigned short *image, unsigned short *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


static float *
generate_rgbafp_image(int w, int h, MTdata d)
{
    float   *ptr = (float*)malloc(w * h * 4 * sizeof(float));
    int     i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = get_random_float(-0x40000000, 0x40000000, d);

    return ptr;
}

static int
verify_rgbafp_image(float *image, float *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


int
test_imagecopy(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_image_format    img_format;
    unsigned char    *rgba8_inptr, *rgba8_outptr;
    unsigned short    *rgba16_inptr, *rgba16_outptr;
    float            *rgbafp_inptr, *rgbafp_outptr;
    clMemWrapper            streams[6];
    int                img_width = 512;
    int                img_height = 512;
    int                i, err;
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    rgba8_inptr = (unsigned char *)generate_rgba8_image(img_width, img_height, d);
    rgba16_inptr = (unsigned short *)generate_rgba16_image(img_width, img_height, d);
    rgbafp_inptr = (float *)generate_rgbafp_image(img_width, img_height, d);
    free_mtdata(d); d = NULL;

    rgba8_outptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height);
    rgba16_outptr = (unsigned short*)malloc(sizeof(unsigned short) * 4 * img_width * img_height);
    rgbafp_outptr = (float*)malloc(sizeof(float) * 4 * img_width * img_height);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[0] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");
    streams[1] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[2] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");
    streams[3] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[4] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");
    streams[5] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");

    for (i=0; i<3; i++)
    {
        void    *p, *outp;
        int        x, y, delta_w = img_width/8, delta_h = img_height/16;

        switch (i)
        {
            case 0:
                p = (void *)rgba8_inptr;
                outp = (void *)rgba8_outptr;
        log_info("Testing CL_RGBA CL_UNORM_INT8\n");
                break;
            case 1:
                p = (void *)rgba16_inptr;
                outp = (void *)rgba16_outptr;
        log_info("Testing CL_RGBA CL_UNORM_INT16\n");
                break;
            case 2:
                p = (void *)rgbafp_inptr;
                outp = (void *)rgbafp_outptr;
        log_info("Testing CL_RGBA CL_FLOAT\n");
                break;
        }

        size_t origin[3] = {0,0,0}, region[3] = {img_width, img_height, 1};
        err = clEnqueueWriteImage(queue, streams[i*2], CL_TRUE, origin, region, 0, 0, p, 0, NULL, NULL);
        test_error(err, "create_image_2d failed");

        int copy_number = 0;
        for (y=0; y<img_height; y+=delta_h)
        {
            for (x=0; x<img_width; x+=delta_w)
            {
        copy_number++;
        size_t copy_origin[3] = {x,y,0}, copy_region[3]={delta_w, delta_h, 1};
        err = clEnqueueCopyImage(queue, streams[i*2], streams[i*2+1],
                                 copy_origin, copy_origin, copy_region,
                                 0, NULL, NULL);
        if (err) {
          log_error("Copy %d (origin [%d, %d], size [%d, %d], image size [%d x %d]) Failed\n", copy_number, x, y, delta_w, delta_h, img_width, img_height);
        }
        test_error(err, "clEnqueueCopyImage failed");
            }
        }

        err = clEnqueueReadImage(queue, streams[i*2+1], CL_TRUE, origin, region, 0, 0, outp, 0, NULL, NULL);
        test_error(err, "clEnqueueReadImage failed");

        switch (i)
        {
            case 0:
                err = verify_rgba8_image(rgba8_inptr, rgba8_outptr, img_width, img_height);
                break;
            case 1:
                err = verify_rgba16_image(rgba16_inptr, rgba16_outptr, img_width, img_height);
                break;
            case 2:
                err = verify_rgbafp_image(rgbafp_inptr, rgbafp_outptr, img_width, img_height);
                break;
        }

        if (err)
            break;
    }

  free(rgba8_inptr);
  free(rgba16_inptr);
  free(rgbafp_inptr);
  free(rgba8_outptr);
  free(rgba16_outptr);
  free(rgbafp_outptr);

    if (err)
        log_error("IMAGE copy test failed\n");
    else
        log_info("IMAGE copy test passed\n");

    return err;
}



