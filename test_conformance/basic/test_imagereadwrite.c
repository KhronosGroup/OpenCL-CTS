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

static void
update_rgba8_image(unsigned char *p, int x, int y, int w, int h, int img_width, MTdata d)
{
    int        i, j, indx;

    for (i=y; i<y+h; i++)
    {
        indx = (i * img_width + x) * 4;
        for (j=x; j<x+w; j++,indx+=4)
        {
            p[indx+0] = (unsigned char)genrand_int32(d);
            p[indx+1] = (unsigned char)genrand_int32(d);
            p[indx+2] = (unsigned char)genrand_int32(d);
            p[indx+3] = (unsigned char)genrand_int32(d);
        }
    }
}

static void
update_image_from_image(void *out, void *in, int x, int y, int w, int h, int img_width, int elem_size)
{
    int        i, j, k, out_indx, in_indx;
    in_indx = 0;

    for (i=y; i<y+h; i++)
    {
        out_indx = (i * img_width + x) * elem_size;
        for (j=x; j<x+w; j++,out_indx+=elem_size)
        {
            for (k=0; k<elem_size; k++)
            {
                ((char*)out)[out_indx + k] = ((char*)in)[in_indx];
                in_indx++;
            }
        }
    }
}

static int
verify_rgba8_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
  int     i;

  for (i=0; i<w*h*4; i++)
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
generate_rgba16_image(int w, int h, MTdata d)
{
  unsigned short    *ptr = (unsigned short*)malloc(w * h * 4 * sizeof(unsigned short));
  int             i;

  for (i=0; i<w*h*4; i++)
    ptr[i] = (unsigned short)genrand_int32(d);

  return ptr;
}

static void
update_rgba16_image(unsigned short *p, int x, int y, int w, int h, int img_width, MTdata d)
{
    int        i, j, indx;

    for (i=y; i<y+h; i++)
    {
        indx = (i * img_width + x) * 4;
        for (j=x; j<x+w; j++,indx+=4)
        {
            p[indx+0] = (unsigned short)genrand_int32(d);
            p[indx+1] = (unsigned short)genrand_int32(d);
            p[indx+2] = (unsigned short)genrand_int32(d);
            p[indx+3] = (unsigned short)genrand_int32(d);
        }
    }
}

static int
verify_rgba16_image(unsigned short *image, unsigned short *outptr, int w, int h)
{
  int     i;

  for (i=0; i<w*h*4; i++)
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
generate_rgbafp_image(int w, int h, MTdata d)
{
  float   *ptr = (float*)malloc(w * h * 4 * sizeof(float));
  int     i;

  for (i=0; i<w*h*4; i++)
    ptr[i] = get_random_float(-0x40000000, 0x40000000, d);

  return ptr;
}

static void
update_rgbafp_image(float *p, int x, int y, int w, int h, int img_width, MTdata d)
{
    int        i, j, indx;

    for (i=y; i<y+h; i++)
    {
        indx = (i * img_width + x) * 4;
        for (j=x; j<x+w; j++,indx+=4)
        {
            p[indx+0] = get_random_float(-0x40000000, 0x40000000, d);
            p[indx+1] = get_random_float(-0x40000000, 0x40000000, d);
            p[indx+2] = get_random_float(-0x40000000, 0x40000000, d);
            p[indx+3] = get_random_float(-0x40000000, 0x40000000, d);
        }
    }
}

static int
verify_rgbafp_image(float *image, float *outptr, int w, int h)
{
  int     i;

  for (i=0; i<w*h*4; i++)
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
test_imagereadwrite(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_image_format    img_format;
    unsigned char    *rgba8_inptr, *rgba8_outptr;
    unsigned short    *rgba16_inptr, *rgba16_outptr;
    float            *rgbafp_inptr, *rgbafp_outptr;
    clMemWrapper            streams[3];
    int                img_width = 512;
    int                img_height = 512;
    int                num_tries = 200;
    int                i, j, err;
    MTdata          d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

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

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT16;
    streams[1] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_FLOAT;
    streams[2] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, &err);
    test_error(err, "create_image_2d failed");

    for (i=0; i<3; i++)
    {
        void    *p;

        if (i == 0)
            p = (void *)rgba8_inptr;
        else if (i == 1)
            p = (void *)rgba16_inptr;
        else
            p = (void *)rgbafp_inptr;
        size_t origin[3] = {0,0,0}, region[3] = {img_width, img_height, 1};
        err = clEnqueueWriteImage(queue, streams[i], CL_TRUE,
                              origin, region, 0, 0,
                              p, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clWriteImage2D failed\n");
            return -1;
        }
    }

    for (i=0,j=0; i<num_tries*3; i++,j++)
    {
        int        x = (int)get_random_float(0, img_width, d);
        int        y = (int)get_random_float(0, img_height, d);
        int        w = (int)get_random_float(1, (img_width - x), d);
        int        h = (int)get_random_float(1, (img_height - y), d);
        size_t    input_pitch;
        int     set_input_pitch = (int)(genrand_int32(d) & 0x01);
        int     packed_update = (int)(genrand_int32(d) & 0x01);
        void    *p, *outp;
        int        elem_size;

        if (j == 3)
            j = 0;

        switch (j)
        {
            case 0:
                //if ((w<=10) || (h<=10)) continue;
                elem_size = 4;
                if(packed_update)
                {
                    p = generate_rgba8_image(w, h, d);
                    update_image_from_image(rgba8_inptr, p, x, y, w, h, img_width, elem_size);
                }
                else
                {
                    update_rgba8_image(rgba8_inptr, x, y, w, h, img_width, d);
                    p = (void *)(rgba8_inptr + ((y * img_width + x) * 4));
                }
                outp = (void *)rgba8_outptr;
                break;
            case 1:
                //if ((w<=8) || (h<=8)) continue;
                elem_size = 2*4;
                if(packed_update)
                {
                    p = generate_rgba16_image(w, h, d);
                    update_image_from_image(rgba16_inptr, p, x, y, w, h, img_width, elem_size);
                }
                else
                {
                    update_rgba16_image(rgba16_inptr, x, y, w, h, img_width, d);
                    p = (void *)(rgba16_inptr + ((y * img_width + x) * 4));
                }
                outp = (void *)rgba16_outptr;
                break;
            case 2:
                //if ((w<=8) || (h<=8)) continue;
                elem_size = 4*4;
                if(packed_update)
                {
                    p = generate_rgbafp_image(w, h, d);
                    update_image_from_image(rgbafp_inptr, p, x, y, w, h, img_width, elem_size);
                }
                else
                {
                    update_rgbafp_image(rgbafp_inptr, x, y, w, h, img_width, d);
                    p = (void *)(rgbafp_inptr + ((y * img_width + x) * 4));
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
            }
            else
            {
                // for packed updates the pitch does not need to be calculated here
                update_packed_pitch_name = "'packed without pitch'";
                input_pitch = 0;
            }
        }
        else
        {
            // for unpacked updates the pitch is required
            update_packed_pitch_name = "'unpacked with pitch'";
            input_pitch = img_width*elem_size;
        }

        size_t origin[3] = {x,y,0}, region[3] = {w, h, 1};
        err = clEnqueueWriteImage(queue, streams[j], CL_TRUE,
                              origin, region, input_pitch, 0, p,
                              0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clWriteImage update failed for %s %s: %d\n",
                (packed_update) ? "packed" : "unpacked",
                (set_input_pitch) ? "set pitch" : "unset pitch", err);
            free_mtdata(d);
            return -1;
        }

        if(packed_update)
        {
            free(p);
            p = NULL;
        }

        memset(outp, 0x7, img_width*img_height*elem_size);

        origin[0]=0; origin[1]=0; origin[2]=0;
        region[0]=img_width; region[1]=img_height; region[2]=1;
        err = clEnqueueReadImage(queue, streams[j], CL_TRUE,
                             origin, region, 0,0,
                             outp, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            log_error("clReadImage failed\n");
            free_mtdata(d);
            return -1;
        }

        switch (j)
        {
            case 0:
                err = verify_rgba8_image(rgba8_inptr, rgba8_outptr, img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA8 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 1:
                err = verify_rgba16_image(rgba16_inptr, rgba16_outptr, img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA16 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 2:
                err = verify_rgbafp_image(rgbafp_inptr, rgbafp_outptr, img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA FP read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
        }

        if (err) break;
    }

    free_mtdata(d);
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



