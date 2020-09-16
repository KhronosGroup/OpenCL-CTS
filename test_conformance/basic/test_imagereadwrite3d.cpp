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
generate_rgba8_image(int w, int h, int d, MTdata mtData)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * d *4);
    int             i;

    for (i=0; i<w*h*d*4; i++)
            ptr[i] = (unsigned char)genrand_int32(mtData);

    return ptr;
}

static void
update_rgba8_image(unsigned char *p, int x, int y, int z, int w, int h, int d, int img_width, int img_height, int img_depth, MTdata mtData)
{
    int        i, j, k, indx;
    int        img_slice = img_width * img_height;

    for (k=z; k<z+d; k++)
        for (j=y; j<y+h; j++)
        {
            indx = (k * img_slice + j * img_width + x) * 4;
            for (i=x; i<x+w; i++,indx+=4)
            {
                p[indx+0] = (unsigned char)genrand_int32(mtData);
                p[indx+1] = (unsigned char)genrand_int32(mtData);
                p[indx+2] = (unsigned char)genrand_int32(mtData);
                p[indx+3] = (unsigned char)genrand_int32(mtData);
            }
        }
}

static void
update_image_from_image(void *out, void *in, int x, int y, int z, int w, int h, int d, int img_width, int img_height, int img_depth, int elem_size)
{
    int        i, j, k, elem, out_indx, in_indx;
    int        img_slice = img_width * img_height;
    in_indx = 0;

    for (k=z; k<z+d; k++)
        for (j=y; j<y+h; j++)
        {
            out_indx = (k * img_slice + j * img_width + x) * elem_size;
            for (i=x; i<x+w; i++,out_indx+=elem_size)
            {
                for (elem=0; elem<elem_size; elem++)
                {
                    ((char*)out)[out_indx + elem] = ((char*)in)[in_indx];
                    in_indx++;
                }
            }
        }
}

static int
verify_rgba8_image(unsigned char *image, unsigned char *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("i = %d. Expected (%d %d %d %d), got (%d %d %d %d)\n", i, image[i], image[i+1], image[i+2], image[i+3], outptr[i], outptr[i+1], outptr[i+2], outptr[i+3]);
            return -1;
        }
    }

    return 0;
}


static unsigned short *
generate_rgba16_image(int w, int h, int d, MTdata mtData)
{
    unsigned short    *ptr = (unsigned short*)malloc(w * h * d * 4 * sizeof(unsigned short));
    int             i;

    for (i=0; i<w*h*d*4; i++)
            ptr[i] = (unsigned short)genrand_int32(mtData);

    return ptr;
}

static void
update_rgba16_image(unsigned short *p, int x, int y, int z, int w, int h, int d, int img_width, int img_height, int img_depth, MTdata mtData)
{
    int        i, j, k, indx;
    int        img_slice = img_width * img_height;

    for (k=z; k<z+d; k++)
        for (j=y; j<y+h; j++)
    {
        indx = (k * img_slice + j * img_width + x) * 4;
        for (i=x; i<x+w; i++,indx+=4)
        {
            p[indx+0] = (unsigned short)genrand_int32(mtData);
            p[indx+1] = (unsigned short)genrand_int32(mtData);
            p[indx+2] = (unsigned short)genrand_int32(mtData);
            p[indx+3] = (unsigned short)genrand_int32(mtData);
        }
    }
}

static int
verify_rgba16_image(unsigned short *image, unsigned short *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("i = %d. Expected (%d %d %d %d), got (%d %d %d %d)\n", i, image[i], image[i+1], image[i+2], image[i+3], outptr[i], outptr[i+1], outptr[i+2], outptr[i+3]);
            return -1;
        }
    }

    return 0;
}


static float *
generate_rgbafp_image(int w, int h, int d, MTdata mtData)
{
    float   *ptr = (float*)malloc(w * h * d *4 * sizeof(float));
    int     i;

    for (i=0; i<w*h*d*4; i++)
            ptr[i] = get_random_float(-0x40000000, 0x40000000, mtData);

    return ptr;
}

static void
update_rgbafp_image(float *p, int x, int y, int z, int w, int h, int d, int img_width, int img_height, int img_depth, MTdata mtData)
{
    int        i, j, k, indx;
    int        img_slice = img_width * img_height;

    for (k=z; k<z+d; k++)
        for (j=y; j<y+h; j++)
        {
            indx = (k * img_slice + j * img_width + x) * 4;
            for (i=x; i<x+w; i++,indx+=4)
            {
                p[indx+0] = get_random_float(-0x40000000, 0x40000000, mtData);
                p[indx+1] = get_random_float(-0x40000000, 0x40000000, mtData);
                p[indx+2] = get_random_float(-0x40000000, 0x40000000, mtData);
                p[indx+3] = get_random_float(-0x40000000, 0x40000000, mtData);
            }
        }
}

static int
verify_rgbafp_image(float *image, float *outptr, int w, int h, int d)
{
    int     i;

    for (i=0; i<w*h*d*4; i++)
    {
        if (outptr[i] != image[i])
        {
            log_error("i = %d. Expected (%f %f %f %f), got (%f %f %f %f)\n", i, image[i], image[i+1], image[i+2], image[i+3], outptr[i], outptr[i+1], outptr[i+2], outptr[i+3]);
            return -1;
        }
    }

    return 0;
}


int
test_imagereadwrite3d(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_image_format    img_format;
    unsigned char    *rgba8_inptr, *rgba8_outptr;
    unsigned short *rgba16_inptr, *rgba16_outptr;
    float            *rgbafp_inptr, *rgbafp_outptr;
    clMemWrapper    streams[3];
    int       img_width = 64;
    int       img_height = 64;
    int       img_depth = 32;
    int       img_slice = img_width * img_height;
    int       num_tries = 30;
    int       i, j, err;
    MTdata      mtData;

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

    mtData = init_genrand( gRandomSeed );
    rgba8_inptr = (unsigned char *)generate_rgba8_image(img_width, img_height, img_depth, mtData);
    rgba16_inptr = (unsigned short *)generate_rgba16_image(img_width, img_height, img_depth, mtData);
    rgbafp_inptr = (float *)generate_rgbafp_image(img_width, img_height, img_depth, mtData);

    rgba8_outptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height * img_depth);
    rgba16_outptr = (unsigned short*)malloc(sizeof(unsigned short) * 4 * img_width * img_height * img_depth);
    rgbafp_outptr = (float*)malloc(sizeof(float) * 4 * img_width * img_height * img_depth);

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;
    streams[0] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[1] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[2] = create_image_3d(context, CL_MEM_READ_ONLY, &img_format, img_width, img_height, img_depth, 0, 0, NULL, &err);
    test_error(err, "create_image_3d failed");

    for (i=0; i<3; i++)
    {
        void    *p;

        if (i == 0)
            p = (void *)rgba8_inptr;
        else if (i == 1)
            p = (void *)rgba16_inptr;
        else
            p = (void *)rgbafp_inptr;

        size_t origin[3] = {0,0,0}, region[3] = {img_width, img_height, img_depth};
        err = clEnqueueWriteImage(queue, streams[i], CL_TRUE,
                                  origin, region, 0, 0,
                                  p,
                                  0, NULL, NULL);
        test_error(err, "clEnqueueWriteImage failed");
    }

    for (i=0,j=0; i<num_tries*3; i++,j++)
    {
        int        x = (int)get_random_float(0, (float)img_width - 1, mtData);
        int        y = (int)get_random_float(0, (float)img_height - 1, mtData);
        int        z = (int)get_random_float(0, (float)img_depth - 1, mtData);
        int        w = (int)get_random_float(1, (float)(img_width - x), mtData);
        int        h = (int)get_random_float(1, (float)(img_height - y), mtData);
        int        d = (int)get_random_float(1, (float)(img_depth - z), mtData);
        size_t    input_pitch, input_slice_pitch;
        int     set_input_pitch = (int)(genrand_int32(mtData) & 0x01);
        int     packed_update = (int)(genrand_int32(mtData) & 0x01);
        void    *p, *outp;
        int        elem_size;

        if (j == 3)
            j = 0;

        // packed: the source image for the write is a whole image                                                                                                                                                                                                                                                      .
        // unpacked: the source image for the write is a subset within a larger image
        switch (j)
        {
            case 0:
                elem_size = 4;
                if(packed_update)
                {
                    p = generate_rgba8_image(w, h, d, mtData);
                    update_image_from_image(rgba8_inptr, p, x, y, z, w, h, d, img_width, img_height, img_depth, elem_size);
                }
                else
                {
                    update_rgba8_image(rgba8_inptr, x, y, z, w, h, d, img_width, img_height, img_depth, mtData);
                    p = (void *)(rgba8_inptr + ((z * img_slice + y * img_width + x) * 4));
                }
                outp = (void *)rgba8_outptr;
                break;
            case 1:
                elem_size = 2*4;
                if(packed_update)
                {
                    p = generate_rgba16_image(w, h, d, mtData);
                    update_image_from_image(rgba16_inptr, p, x, y, z, w, h, d, img_width, img_height, img_depth, elem_size);
                }
                else
                {
                    update_rgba16_image(rgba16_inptr, x, y, z, w, h, d, img_width, img_height, img_depth, mtData);
                    p = (void *)(rgba16_inptr + ((z * img_slice + y * img_width + x) * 4));
                }
                outp = (void *)rgba16_outptr;
                break;
            case 2:
                elem_size = 4*4;
                if(packed_update)
                {
                    p = generate_rgbafp_image(w, h, d, mtData);
                    update_image_from_image(rgbafp_inptr, p, x, y, z, w, h, d, img_width, img_height, img_depth, elem_size);
                }
                else
                {
                    update_rgbafp_image(rgbafp_inptr, x, y, z, w, h, d, img_width, img_height, img_depth, mtData);
                    p = (void *)(rgbafp_inptr + ((z * img_slice + y * img_width + x) * 4));
                }
                outp = (void *)rgbafp_outptr;
                break;
        }

        const char* update_packed_pitch_name = "";
        if(packed_update)
        {
            if(set_input_pitch)
            {
                // for packed updates the pitch does not need to be calculated here (but can be)
                update_packed_pitch_name = "'packed with pitch'";
                input_pitch = w*elem_size;
                input_slice_pitch = w*h*elem_size;
            }
            else
            {
                // for packed updates the pitch does not need to be calculated here
                update_packed_pitch_name = "'packed without pitch'";
                input_pitch = 0;
                input_slice_pitch = 0;
            }
        }
        else
        {
            // for unpacked updates the pitch is required
            update_packed_pitch_name = "'unpacked with pitch'";
            input_pitch = img_width*elem_size;
            input_slice_pitch = input_pitch*img_height;
        }

    size_t origin[3] = {x,y,z}, region[3] = {w, h, d};
        err = clEnqueueWriteImage(queue, streams[j], CL_TRUE,
                              origin, region, input_pitch, input_slice_pitch,
                              p, 0, NULL, NULL);
    test_error(err, "clEnqueueWriteImage failed");

        if(packed_update)
        {
            free(p);
            p = NULL;
        }

        memset(outp, 0x7, img_width*img_height*img_depth*elem_size);

    origin[0]=0; origin[1]=0; origin[2]=0; region[0]=img_width; region[1]=img_height; region[2]=img_depth;
        err = clEnqueueReadImage(queue, streams[j], CL_TRUE,
                             origin, region, 0, 0,
                             outp, 0, NULL, NULL);
    test_error(err, "clEnqueueReadImage failed");

        switch (j)
        {
            case 0:
                err = verify_rgba8_image(rgba8_inptr, rgba8_outptr, img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%d y=%d z=%d w=%d h=%d d=%d pitch=%d, slice_pitch=%d, try=%d\n", x, y, z, w, h, d, (int)input_pitch, (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA8 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 1:
                err = verify_rgba16_image(rgba16_inptr, rgba16_outptr, img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%d y=%d z=%d w=%d h=%d d=%d pitch=%d, slice_pitch=%d, try=%d\n", x, y, z, w, h, d, (int)input_pitch, (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA16 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 2:
                err = verify_rgbafp_image(rgbafp_inptr, rgbafp_outptr, img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%d y=%d z=%d w=%d h=%d d=%d pitch=%d, slice_pitch=%d, try=%d\n", x, y, z, w, h, d, (int)input_pitch, (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA FP read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
        }

        if (err)
            break;
    }

    free_mtdata(mtData);
    free(rgba8_inptr);
    free(rgba16_inptr);
    free(rgbafp_inptr);
    free(rgba8_outptr);
    free(rgba16_outptr);
    free(rgbafp_outptr);

    if (!err)
        log_info("IMAGE read, write test passed\n");

    return err;
}



