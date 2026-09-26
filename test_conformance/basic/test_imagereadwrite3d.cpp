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

static std::unique_ptr<unsigned char[]>
generate_rgba8_image(int w, int h, int d, MTdata mtData)
{
    std::unique_ptr<unsigned char[]> ptr{ new unsigned char[w * h * d * 4] };

    for (int i = 0; i < w * h * d * 4; i++)
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

static int verify_rgba8_image(const unsigned char *image,
                              const unsigned char *outptr, int w, int h, int d)
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


static std::unique_ptr<unsigned short[]>
generate_rgba16_image(int w, int h, int d, MTdata mtData)
{
    std::unique_ptr<unsigned short[]> ptr{ new unsigned short[w * h * d * 4] };

    for (int i = 0; i < w * h * d * 4; i++)
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

static int verify_rgba16_image(const unsigned short *image,
                               const unsigned short *outptr, int w, int h,
                               int d)
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


static std::unique_ptr<float[]> generate_rgbafp_image(int w, int h, int d,
                                                      MTdata mtData)
{
    std::unique_ptr<float[]> ptr{ new float[w * h * d * 4] };

    for (int i = 0; i < w * h * d * 4; i++)
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

static int verify_rgbafp_image(const float *image, const float *outptr, int w,
                               int h, int d)
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

static constexpr cl_image_format image_formats[] = { { CL_RGBA, CL_UNORM_INT8 },
                                                     { CL_RGBA,
                                                       CL_UNORM_INT16 },
                                                     { CL_RGBA, CL_FLOAT } };

REGISTER_TEST(imagereadwrite3d)
{
    constexpr size_t image_formats_count = ARRAY_SIZE(image_formats);
    std::unique_ptr<unsigned char[]> rgba8_inptr, rgba8_outptr;
    std::unique_ptr<unsigned short[]> rgba16_inptr, rgba16_outptr;
    std::unique_ptr<float[]> rgbafp_inptr, rgbafp_outptr;
    clMemWrapper    streams[3];
    size_t img_width = 64;
    size_t img_height = 64;
    size_t img_depth = 32;
    size_t img_slice = img_width * img_height;
    int       num_tries = 30;
    int       i, j, err;
    MTdataHolder mtData(gRandomSeed);

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( device )

    rgba8_inptr =
        generate_rgba8_image(img_width, img_height, img_depth, mtData);
    rgba16_inptr =
        generate_rgba16_image(img_width, img_height, img_depth, mtData);
    rgbafp_inptr =
        generate_rgbafp_image(img_width, img_height, img_depth, mtData);

    rgba8_outptr.reset(
        new unsigned char[4 * img_width * img_height * img_depth]);
    rgba16_outptr.reset(
        new unsigned short[4 * img_width * img_height * img_depth]);
    rgbafp_outptr.reset(new float[4 * img_width * img_height * img_depth]);

    for (size_t index = 0; index < image_formats_count; ++index)
    {
        streams[index] = create_image_3d(
            context, CL_MEM_READ_ONLY, &image_formats[index], img_width,
            img_height, img_depth, 0, 0, nullptr, &err);
        test_error(err, "create_image_3d failed");
    }

    for (i = 0; i < image_formats_count; i++)
    {
        void    *p;

        if (i == 0)
            p = rgba8_inptr.get();
        else if (i == 1)
            p = rgba16_inptr.get();
        else
            p = rgbafp_inptr.get();

        size_t origin[3] = {0,0,0}, region[3] = {img_width, img_height, img_depth};
        err = clEnqueueWriteImage(queue, streams[i], CL_TRUE,
                                  origin, region, 0, 0,
                                  p,
                                  0, NULL, NULL);
        test_error(err, "clEnqueueWriteImage failed");
    }

    for (i = 0, j = 0; i < num_tries * image_formats_count; i++, j++)
    {
        size_t x = (size_t)get_random_float(0, (float)img_width - 1, mtData);
        size_t y = (size_t)get_random_float(0, (float)img_height - 1, mtData);
        size_t z = (size_t)get_random_float(0, (float)img_depth - 1, mtData);
        size_t w = (size_t)get_random_float(1, (float)(img_width - x), mtData);
        size_t h = (size_t)get_random_float(1, (float)(img_height - y), mtData);
        size_t d = (size_t)get_random_float(1, (float)(img_depth - z), mtData);
        size_t    input_pitch, input_slice_pitch;
        int     set_input_pitch = (int)(genrand_int32(mtData) & 0x01);
        int     packed_update = (int)(genrand_int32(mtData) & 0x01);
        void    *p, *outp;
        std::unique_ptr<unsigned char[]> p_rgba8;
        std::unique_ptr<unsigned short[]> p_rgba16;
        std::unique_ptr<float[]> p_rgbaf;
        int        elem_size;

        if (j == image_formats_count) j = 0;

        // packed: the source image for the write is a whole image                                                                                                                                                                                                                                                      .
        // unpacked: the source image for the write is a subset within a larger image
        switch (j)
        {
            case 0:
                elem_size = 4;
                if(packed_update)
                {
                    p_rgba8 = generate_rgba8_image(w, h, d, mtData);
                    p = p_rgba8.get();
                    update_image_from_image(rgba8_inptr.get(), p, x, y, z, w, h,
                                            d, img_width, img_height, img_depth,
                                            elem_size);
                }
                else
                {
                    update_rgba8_image(rgba8_inptr.get(), x, y, z, w, h, d,
                                       img_width, img_height, img_depth,
                                       mtData);
                    p = static_cast<void *>(
                        rgba8_inptr.get()
                        + ((z * img_slice + y * img_width + x) * 4));
                }
                outp = static_cast<void *>(rgba8_outptr.get());
                break;
            case 1:
                elem_size = 2*4;
                if(packed_update)
                {
                    p_rgba16 = generate_rgba16_image(w, h, d, mtData);
                    p = p_rgba16.get();
                    update_image_from_image(rgba16_inptr.get(), p, x, y, z, w,
                                            h, d, img_width, img_height,
                                            img_depth, elem_size);
                }
                else
                {
                    update_rgba16_image(rgba16_inptr.get(), x, y, z, w, h, d,
                                        img_width, img_height, img_depth,
                                        mtData);
                    p = static_cast<void *>(
                        rgba16_inptr.get()
                        + ((z * img_slice + y * img_width + x) * 4));
                }
                outp = static_cast<void *>(rgba16_outptr.get());
                break;
            case 2:
                elem_size = 4*4;
                if(packed_update)
                {
                    p_rgbaf = generate_rgbafp_image(w, h, d, mtData);
                    p = p_rgbaf.get();
                    update_image_from_image(rgbafp_inptr.get(), p, x, y, z, w,
                                            h, d, img_width, img_height,
                                            img_depth, elem_size);
                }
                else
                {
                    update_rgbafp_image(rgbafp_inptr.get(), x, y, z, w, h, d,
                                        img_width, img_height, img_depth,
                                        mtData);
                    p = static_cast<void *>(
                        rgbafp_inptr.get()
                        + ((z * img_slice + y * img_width + x) * 4));
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
            p = nullptr;
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
                err = verify_rgba8_image(rgba8_inptr.get(), rgba8_outptr.get(),
                                         img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%zu y=%zu z=%zu w=%zu h=%zu d=%zu pitch=%d, "
                              "slice_pitch=%d, try=%d\n",
                              x, y, z, w, h, d, (int)input_pitch,
                              (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA8 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 1:
                err =
                    verify_rgba16_image(rgba16_inptr.get(), rgba16_outptr.get(),
                                        img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%zu y=%zu z=%zu w=%zu h=%zu d=%zu pitch=%d, "
                              "slice_pitch=%d, try=%d\n",
                              x, y, z, w, h, d, (int)input_pitch,
                              (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA16 read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
            case 2:
                err =
                    verify_rgbafp_image(rgbafp_inptr.get(), rgbafp_outptr.get(),
                                        img_width, img_height, img_depth);
                if (err)
                {
                    log_error("x=%zu y=%zu z=%zu w=%zu h=%zu d=%zu pitch=%d, "
                              "slice_pitch=%d, try=%d\n",
                              x, y, z, w, h, d, (int)input_pitch,
                              (int)input_slice_pitch, (int)i);
                    log_error("IMAGE RGBA FP read, write %s test failed\n", update_packed_pitch_name);
                }
                break;
        }

        if (err)
            break;
    }

    if (!err)
        log_info("IMAGE read, write test passed\n");

    return err;
}
