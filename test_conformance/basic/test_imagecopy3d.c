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
generate_uint8_image(unsigned num_elements, MTdata d)
{
    unsigned char *ptr = (unsigned char*)malloc(num_elements);
    unsigned i;

    for (i=0; i<num_elements; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_uint8_image(unsigned char *image, unsigned char *outptr, unsigned num_elements)
{
    unsigned i;

    for (i=0; i<num_elements; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


static unsigned short *
generate_uint16_image(unsigned num_elements, MTdata d)
{
    unsigned short    *ptr = (unsigned short *)malloc(num_elements * sizeof(unsigned short));
    unsigned i;

    for (i=0; i<num_elements; i++)
        ptr[i] = (unsigned short)genrand_int32(d);

    return ptr;
}

static int
verify_uint16_image(unsigned short *image, unsigned short *outptr, unsigned num_elements)
{
    unsigned i;

    for (i=0; i<num_elements; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


static float *
generate_float_image(unsigned num_elements, MTdata d)
{
    float   *ptr = (float*)malloc(num_elements * sizeof(float));
    unsigned i;

    for (i=0; i<num_elements; i++)
        ptr[i] = get_random_float(-0x40000000, 0x40000000, d);

    return ptr;
}

static int
verify_float_image(float *image, float *outptr, unsigned num_elements)
{
    unsigned i;

    for (i=0; i<num_elements; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


int
test_imagecopy3d(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements_ignored)
{
    cl_image_format    img_format;
    unsigned char    *rgba8_inptr, *rgba8_outptr;
    unsigned short *rgba16_inptr, *rgba16_outptr;
    float *rgbafp_inptr, *rgbafp_outptr;
    clMemWrapper streams[6];
    int img_width = 128;
    int img_height = 128;
    int img_depth = 64;
    int i;
    cl_int        err;
    unsigned    num_elements = img_width * img_height * img_depth * 4;
    MTdata      d;

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    rgba8_inptr = (unsigned char *)generate_uint8_image(num_elements, d);
    rgba16_inptr = (unsigned short *)generate_uint16_image(num_elements, d);
    rgbafp_inptr = (float *)generate_float_image(num_elements, d);
    free_mtdata(d); d = NULL;

    rgba8_outptr = (unsigned char*)malloc(sizeof(unsigned char) * num_elements);
    rgba16_outptr = (unsigned short*)malloc(sizeof(unsigned short) * num_elements);
    rgbafp_outptr = (float*)malloc(sizeof(float) * num_elements);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[0] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");
    streams[1] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[2] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");
    streams[3] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[4] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");
    streams[5] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    for (i=0; i<3; i++)
    {
        void    *p, *outp;
        int        x, y, z, delta_w = img_width/8, delta_h = img_height/16, delta_d = img_depth/4;

        switch (i)
        {
            case 0:
                p = (void *)rgba8_inptr;
                outp = (void *)rgba8_outptr;
                break;
            case 1:
                p = (void *)rgba16_inptr;
                outp = (void *)rgba16_outptr;
                break;
            case 2:
                p = (void *)rgbafp_inptr;
                outp = (void *)rgbafp_outptr;
                break;
        }

        size_t origin[3]={0,0,0}, region[3]={img_width, img_height, img_depth};
        err = clEnqueueWriteImage(queue, streams[i*2], CL_TRUE, origin, region, 0, 0, p, 0, NULL, NULL);
        test_error(err, "clEnqueueWriteImage failed");

        for (z=0; z<img_depth; z+=delta_d)
        {
            for (y=0; y<img_height; y+=delta_h)
            {
                for (x=0; x<img_width; x+=delta_w)
                {
                  origin[0] = x; origin[1] = y; origin[2] = z;
                  region[0] = delta_w; region[1] = delta_h; region[2] = delta_d;

                  err = clEnqueueCopyImage(queue, streams[i*2], streams[i*2+1], origin, origin, region, 0, NULL, NULL);
                  test_error(err, "clEnqueueCopyImage failed");
                }
            }
        }

        origin[0] = 0; origin[1] = 0; origin[2] = 0;
        region[0] = img_width; region[1] = img_height; region[2] = img_depth;
        err = clEnqueueReadImage(queue, streams[i*2+1], CL_TRUE, origin, region, 0, 0, outp, 0, NULL, NULL);
        test_error(err, "clEnqueueReadImage failed");

        switch (i)
        {
            case 0:
                err = verify_uint8_image(rgba8_inptr, rgba8_outptr, num_elements);
        if (err) log_error("Failed uint8\n");
                break;
            case 1:
                err = verify_uint16_image(rgba16_inptr, rgba16_outptr, num_elements);
        if (err) log_error("Failed uint16\n");
                break;
            case 2:
                err = verify_float_image(rgbafp_inptr, rgbafp_outptr, num_elements);
        if (err) log_error("Failed float\n");
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
        log_error("IMAGE3D copy test failed\n");
    else
        log_info("IMAGE3D copy test passed\n");

    return err;
}



