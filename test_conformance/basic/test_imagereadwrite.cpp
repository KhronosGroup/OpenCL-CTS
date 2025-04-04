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

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "testBase.h"

static std::unique_ptr<unsigned char[]> generate_rgba8_image(int w, int h,
                                                             MTdata d)
{
    std::unique_ptr<unsigned char[]> ptr{ new unsigned char[w * h * 4] };

    for (int i = 0; i < w * h * 4; i++)
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

static int verify_rgba8_image(const unsigned char *image,
                              const unsigned char *outptr, int w, int h)
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


static std::unique_ptr<unsigned short[]> generate_rgba16_image(int w, int h,
                                                               MTdata d)
{
    std::unique_ptr<unsigned short[]> ptr{ new unsigned short[w * h * 4] };

    for (int i = 0; i < w * h * 4; i++)
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

static int verify_rgba16_image(const unsigned short *image,
                               const unsigned short *outptr, int w, int h)
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


static std::unique_ptr<float[]> generate_rgbafp_image(int w, int h, MTdata d)
{
    std::unique_ptr<float[]> ptr{ new float[w * h * 4] };

    for (int i = 0; i < w * h * 4; i++)
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

static int verify_rgbafp_image(const float *image, const float *outptr, int w,
                               int h)
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

static constexpr cl_image_format image_formats[] = { { CL_RGBA, CL_UNORM_INT8 },
                                                     { CL_RGBA,
                                                       CL_UNORM_INT16 },
                                                     { CL_RGBA, CL_FLOAT } };

REGISTER_TEST(imagereadwrite)
{
    constexpr size_t image_formats_count = ARRAY_SIZE(image_formats);
    std::unique_ptr<unsigned char[]> rgba8_inptr, rgba8_outptr;
    std::unique_ptr<unsigned short[]> rgba16_inptr, rgba16_outptr;
    std::unique_ptr<float[]> rgbafp_inptr, rgbafp_outptr;
    clMemWrapper            streams[3];
    int                img_width = 512;
    int                img_height = 512;
    int                num_tries = 200;
    int                i, j, err;
    MTdataHolder d(gRandomSeed);

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    rgba8_inptr = generate_rgba8_image(img_width, img_height, d);
    rgba16_inptr = generate_rgba16_image(img_width, img_height, d);
    rgbafp_inptr = generate_rgbafp_image(img_width, img_height, d);

    rgba8_outptr.reset(new unsigned char[4 * img_width * img_height]);
    rgba16_outptr.reset(new unsigned short[4 * img_width * img_height]);
    rgbafp_outptr.reset(new float[4 * img_width * img_height]);

    for (size_t index = 0; index < image_formats_count; ++index)
    {
        streams[index] =
            create_image_2d(context, CL_MEM_READ_WRITE, &image_formats[index],
                            img_width, img_height, 0, NULL, &err);
        test_error(err, "create_image_2d failed");
    }

    for (i=0; i<3; i++)
    {
        void    *p;

        if (i == 0)
            p = rgba8_inptr.get();
        else if (i == 1)
            p = rgba16_inptr.get();
        else
            p = rgbafp_inptr.get();
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

    for (i = 0, j = 0; i < num_tries * image_formats_count; i++, j++)
    {
        int        x = (int)get_random_float(0, img_width, d);
        int        y = (int)get_random_float(0, img_height, d);
        int        w = (int)get_random_float(1, (img_width - x), d);
        int        h = (int)get_random_float(1, (img_height - y), d);
        size_t    input_pitch;
        int     set_input_pitch = (int)(genrand_int32(d) & 0x01);
        int     packed_update = (int)(genrand_int32(d) & 0x01);
        void    *p, *outp;
        std::unique_ptr<unsigned char[]> p_rgba8;
        std::unique_ptr<unsigned short[]> p_rgba16;
        std::unique_ptr<float[]> p_rgbaf;
        int        elem_size;

        if (j == image_formats_count) j = 0;

        switch (j)
        {
            case 0:
                //if ((w<=10) || (h<=10)) continue;
                elem_size = 4;
                if(packed_update)
                {
                    p_rgba8 = generate_rgba8_image(w, h, d);
                    p = p_rgba8.get();
                    update_image_from_image(rgba8_inptr.get(), p, x, y, w, h,
                                            img_width, elem_size);
                }
                else
                {
                    update_rgba8_image(rgba8_inptr.get(), x, y, w, h, img_width,
                                       d);
                    p = static_cast<void *>(rgba8_inptr.get()
                                            + ((y * img_width + x) * 4));
                }
                outp = static_cast<void *>(rgba8_outptr.get());
                break;
            case 1:
                //if ((w<=8) || (h<=8)) continue;
                elem_size = 2*4;
                if(packed_update)
                {
                    p_rgba16 = generate_rgba16_image(w, h, d);
                    p = p_rgba16.get();
                    update_image_from_image(rgba16_inptr.get(), p, x, y, w, h,
                                            img_width, elem_size);
                }
                else
                {
                    update_rgba16_image(rgba16_inptr.get(), x, y, w, h,
                                        img_width, d);
                    p = static_cast<void *>(rgba16_inptr.get()
                                            + ((y * img_width + x) * 4));
                }
                outp = static_cast<void *>(rgba16_outptr.get());
                break;
            case 2:
                //if ((w<=8) || (h<=8)) continue;
                elem_size = 4*4;
                if(packed_update)
                {
                    p_rgbaf = generate_rgbafp_image(w, h, d);
                    p = p_rgbaf.get();
                    update_image_from_image(rgbafp_inptr.get(), p, x, y, w, h,
                                            img_width, elem_size);
                }
                else
                {
                    update_rgbafp_image(rgbafp_inptr.get(), x, y, w, h,
                                        img_width, d);
                    p = static_cast<void *>(rgbafp_inptr.get()
                                            + ((y * img_width + x) * 4));
                }
                outp = static_cast<void *>(rgbafp_outptr.get());
                break;
            default:
                log_error("ERROR Invalid j = %d\n", j);
                elem_size = 0;
                p = nullptr;
                outp = nullptr;
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
            p = nullptr;
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
                err = verify_rgba8_image(rgba8_inptr.get(), rgba8_outptr.get(),
                                         img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA8 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 1:
                err =
                    verify_rgba16_image(rgba16_inptr.get(), rgba16_outptr.get(),
                                        img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA16 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 2:
                err =
                    verify_rgbafp_image(rgbafp_inptr.get(), rgbafp_outptr.get(),
                                        img_width, img_height);
                if (err)
                {
                    log_error("x=%d y=%d w=%d h=%d, pitch=%d, try=%d\n", x, y, w, h, (int)input_pitch, (int)i);
                    log_error("IMAGE RGBA FP read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
        }

        if (err) break;
    }

    if (!err)
        log_info("IMAGE read, write test passed\n");

    return err;
}
