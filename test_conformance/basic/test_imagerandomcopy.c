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
verify_rgba8_image(unsigned char *image, unsigned char *outptr, int x, int y, int w, int h, int img_width)
{
    int     i, j, indx;

    for (j=y; j<(y+h); j++)
    {
        indx = j*img_width*4;
        for (i=x*4; i<(x+w)*4; i++)
        {
            if (outptr[indx+i] != image[indx+i])
                return -1;
        }
    }
    return 0;
}


static unsigned short *
generate_rgba16_image(int w, int h, MTdata d)
{
    unsigned short    *ptr = (unsigned short*)malloc(w * h * 4 * sizeof(unsigned short));
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned short)genrand_int32(d);

    return ptr;
}

static int
verify_rgba16_image(unsigned short *image, unsigned short *outptr, int x, int y, int w, int h, int img_width)
{
    int     i, j, indx;

    for (j=y; j<(y+h); j++)
    {
        indx = j*img_width*4;
        for (i=x*4; i<(x+w)*4; i++)
        {
            if (outptr[indx+i] != image[indx+i])
                return -1;
        }
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
verify_rgbafp_image(float *image, float *outptr, int x, int y, int w, int h, int img_width)
{
    int     i, j, indx;

    for (j=y; j<(y+h); j++)
    {
        indx = j*img_width*4;
        for (i=x*4; i<(x+w)*4; i++)
        {
            if (outptr[indx+i] != image[indx+i])
                return -1;
        }
    }
    return 0;
}


#define NUM_COPIES    10
static const char *test_str_names[] = { "CL_RGBA CL_UNORM_INT8", "CL_RGBA CL_UNORM_INT16", "CL_RGBA CL_FLOAT" };

int
test_imagerandomcopy(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_image_format    img_format;
    unsigned char    *rgba8_inptr, *rgba8_outptr;
    unsigned short    *rgba16_inptr, *rgba16_outptr;
    float            *rgbafp_inptr, *rgbafp_outptr;
    clMemWrapper            streams[6];
    int                img_width = 512;
    int                img_height = 512;
    int                i, j;
    cl_int          err;
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    log_info("Testing with image %d x %d.\n", img_width, img_height);

    d = init_genrand( gRandomSeed );
    rgba8_inptr = (unsigned char *)generate_rgba8_image(img_width, img_height, d);
    rgba16_inptr = (unsigned short *)generate_rgba16_image(img_width, img_height, d);
    rgbafp_inptr = (float *)generate_rgbafp_image(img_width, img_height, d);

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
        void            *p, *outp;
        unsigned int    x[2], y[2], delta_w, delta_h ;

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

        size_t origin[3]={0,0,0}, region[3]={img_width, img_height,1};
        err = clEnqueueWriteImage(queue, streams[i*2], CL_TRUE, origin, region, 0, 0, p, 0, NULL, NULL);
//        err = clWriteImage(context, streams[i*2], false, 0, 0, 0, img_width, img_height, 0, NULL, 0, 0, p, NULL);
        test_error(err, "clEnqueueWriteImage failed");

        for (j=0; j<NUM_COPIES; j++)
        {
            x[0] = (int)get_random_float(0, img_width, d);
            do
            {
                x[1] = (int)get_random_float(0, img_width, d);
            } while (x[1] <= x[0]);

            y[0] = (int)get_random_float(0, img_height, d);
            do
            {
                y[1] = (int)get_random_float(0, img_height, d);
            } while (y[1] <= y[0]);

            delta_w = x[1] - x[0];
            delta_h = y[1] - y[0];
            log_info("Testing clCopyImage for %s: x = %d, y = %d, w = %d, h = %d\n", test_str_names[i], x[0], y[0], delta_w, delta_h);
            origin[0] = x[0];
            origin[1] = y[0];
            origin[2] = 0;
            region[0] = delta_w;
            region[1] = delta_h;
            region[2] = 1;
            err = clEnqueueCopyImage(queue, streams[i*2], streams[i*2+1], origin, origin, region, 0, NULL, NULL);
//          err = clCopyImage(context, streams[i*2], streams[i*2+1],
//                              x[0], y[0], 0, x[0], y[0], 0, delta_w, delta_h, 0, NULL);
            test_error(err, "clEnqueueCopyImage failed");

            origin[0] = 0;
            origin[1] = 0;
            origin[2] = 0;
            region[0] = img_width;
            region[1] = img_height;
            region[2] = 1;
            err = clEnqueueReadImage(queue, streams[i*2+1], CL_TRUE, origin, region, 0, 0, outp, 0, NULL, NULL);
//            err = clReadImage(context, streams[i*2+1], false, 0, 0, 0, img_width, img_height, 0, 0, 0, outp, NULL);
            test_error(err, "clEnqueueReadImage failed");

            switch (i)
            {
                case 0:
                    err = verify_rgba8_image(rgba8_inptr, rgba8_outptr, x[0], y[0], delta_w, delta_h, img_width);
                    break;
                case 1:
                    err = verify_rgba16_image(rgba16_inptr, rgba16_outptr, x[0], y[0], delta_w, delta_h, img_width);
                    break;
                case 2:
                    err = verify_rgbafp_image(rgbafp_inptr, rgbafp_outptr, x[0], y[0], delta_w, delta_h, img_width);
                    break;
            }

            if (err)
                break;
        }

        if (err)
            break;
    }

    free_mtdata(d); d = NULL;
    free(rgba8_inptr);
    free(rgba16_inptr);
    free(rgbafp_inptr);
    free(rgba8_outptr);
    free(rgba16_outptr);
    free(rgbafp_outptr);

    if (err)
        log_error("IMAGE random copy test failed\n");
    else
        log_info("IMAGE random copy test passed\n");

    return err;
}



