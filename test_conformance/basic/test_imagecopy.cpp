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
#include <memory>

#include "testBase.h"

static std::unique_ptr<unsigned char[]> generate_rgba8_image(int w, int h,
                                                             MTdata d)
{
    std::unique_ptr<unsigned char[]> ptr{ new unsigned char[w * h * 4] };

    for (int i = 0; i < w * h * 4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int verify_rgba8_image(const unsigned char *image,
                              const unsigned char *outptr, int w, int h)
{
    int i;

    for (i = 0; i < w * h * 4; i++)
    {
        if (outptr[i] != image[i]) return -1;
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

static int verify_rgba16_image(const unsigned short *image,
                               const unsigned short *outptr, int w, int h)
{
    int i;

    for (i = 0; i < w * h * 4; i++)
    {
        if (outptr[i] != image[i]) return -1;
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

static int verify_rgbafp_image(const float *image, const float *outptr, int w,
                               int h)
{
    int i;

    for (i = 0; i < w * h * 4; i++)
    {
        if (outptr[i] != image[i]) return -1;
    }

    return 0;
}

static constexpr cl_image_format image_formats[] = { { CL_RGBA, CL_UNORM_INT8 },
                                                     { CL_RGBA,
                                                       CL_UNORM_INT16 },
                                                     { CL_RGBA, CL_FLOAT } };

static int test_imagecopy_impl(cl_device_id device, cl_context context,
                               cl_command_queue queue, int num_elements,
                               cl_mem_flags src_image_flags)
{
    constexpr size_t image_formats_count = ARRAY_SIZE(image_formats);
    std::unique_ptr<unsigned char[]> rgba8_inptr, rgba8_outptr;
    std::unique_ptr<unsigned short[]> rgba16_inptr, rgba16_outptr;
    std::unique_ptr<float[]> rgbafp_inptr, rgbafp_outptr;
    clMemWrapper streams[6];
    int img_width = 512;
    int img_height = 512;
    int i, err;
    MTdataHolder d(gRandomSeed);

    rgba8_inptr = generate_rgba8_image(img_width, img_height, d);
    rgba16_inptr = generate_rgba16_image(img_width, img_height, d);
    rgbafp_inptr = generate_rgbafp_image(img_width, img_height, d);

    rgba8_outptr.reset(new unsigned char[4 * img_width * img_height]);
    rgba16_outptr.reset(new unsigned short[4 * img_width * img_height]);
    rgbafp_outptr.reset(new float[4 * img_width * img_height]);

    for (size_t index = 0; index < image_formats_count; ++index)
    {
        void *ptr = nullptr;
        if (src_image_flags & CL_MEM_USE_HOST_PTR
            || src_image_flags & CL_MEM_COPY_HOST_PTR)

        {
            switch (index)
            {
                case 0: ptr = rgba8_inptr.get(); break;
                case 1: ptr = rgba16_inptr.get(); break;
                case 2: ptr = rgbafp_inptr.get(); break;
                default: break;
            }
        }
        streams[index * 2] =
            create_image_2d(context, src_image_flags, &image_formats[index],
                            img_width, img_height, 0, ptr, &err);
        test_error(err, "create_image_2d failed");

        streams[index * 2 + 1] =
            create_image_2d(context, CL_MEM_READ_WRITE, &image_formats[index],
                            img_width, img_height, 0, nullptr, &err);
        test_error(err, "create_image_2d failed");
    }

    for (i = 0; i < 3; i++)
    {
        void *p, *outp;
        int x, y, delta_w = img_width / 8, delta_h = img_height / 16;

        switch (i)
        {
            case 0:
                p = rgba8_inptr.get();
                outp = rgba8_outptr.get();
                log_info("Testing CL_RGBA CL_UNORM_INT8\n");
                break;
            case 1:
                p = rgba16_inptr.get();
                outp = rgba16_outptr.get();
                log_info("Testing CL_RGBA CL_UNORM_INT16\n");
                break;
            case 2:
                p = rgbafp_inptr.get();
                outp = rgbafp_outptr.get();
                log_info("Testing CL_RGBA CL_FLOAT\n");
                break;
        }

        size_t origin[3] = { 0, 0, 0 },
               region[3] = { img_width, img_height, 1 };
        if (!(src_image_flags & CL_MEM_USE_HOST_PTR
              || src_image_flags & CL_MEM_COPY_HOST_PTR))
        {
            err = clEnqueueWriteImage(queue, streams[i * 2], CL_TRUE, origin,
                                      region, 0, 0, p, 0, nullptr, nullptr);
            test_error(err, "create_image_2d failed");
        }

        int copy_number = 0;
        for (y = 0; y < img_height; y += delta_h)
        {
            for (x = 0; x < img_width; x += delta_w)
            {
                copy_number++;
                size_t copy_origin[3] = { x, y, 0 },
                       copy_region[3] = { delta_w, delta_h, 1 };
                err = clEnqueueCopyImage(
                    queue, streams[i * 2], streams[i * 2 + 1], copy_origin,
                    copy_origin, copy_region, 0, NULL, NULL);
                if (err)
                {
                    log_error("Copy %d (origin [%d, %d], size [%d, %d], image "
                              "size [%d x %d]) Failed\n",
                              copy_number, x, y, delta_w, delta_h, img_width,
                              img_height);
                }
                test_error(err, "clEnqueueCopyImage failed");
            }
        }

        err = clEnqueueReadImage(queue, streams[i * 2 + 1], CL_TRUE, origin,
                                 region, 0, 0, outp, 0, NULL, NULL);
        test_error(err, "clEnqueueReadImage failed");

        switch (i)
        {
            case 0:
                err = verify_rgba8_image(rgba8_inptr.get(), rgba8_outptr.get(),
                                         img_width, img_height);
                break;
            case 1:
                err =
                    verify_rgba16_image(rgba16_inptr.get(), rgba16_outptr.get(),
                                        img_width, img_height);
                break;
            case 2:
                err =
                    verify_rgbafp_image(rgbafp_inptr.get(), rgbafp_outptr.get(),
                                        img_width, img_height);
                break;
        }

        if (err) break;
    }

    if (err)
        log_error("IMAGE copy test failed\n");
    else
        log_info("IMAGE copy test passed\n");

    return err;
}

REGISTER_TEST(imagecopy)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device);

    return test_imagecopy_impl(device, context, queue, num_elements,
                               CL_MEM_READ_WRITE);
}
