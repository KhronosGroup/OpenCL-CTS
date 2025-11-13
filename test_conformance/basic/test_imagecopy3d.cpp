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

static std::unique_ptr<unsigned char[]>
generate_uint8_image(unsigned num_elements, MTdata d)
{
    std::unique_ptr<unsigned char[]> ptr{ new unsigned char[num_elements] };

    for (unsigned i = 0; i < num_elements; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int verify_uint8_image(const unsigned char *image,
                              const unsigned char *outptr,
                              unsigned num_elements)
{
    unsigned i;

    for (i = 0; i < num_elements; i++)
    {
        if (outptr[i] != image[i]) return -1;
    }

    return 0;
}


static std::unique_ptr<unsigned short[]>
generate_uint16_image(unsigned num_elements, MTdata d)
{
    std::unique_ptr<unsigned short[]> ptr{ new unsigned short[num_elements] };

    for (unsigned i = 0; i < num_elements; i++)
        ptr[i] = (unsigned short)genrand_int32(d);

    return ptr;
}

static int verify_uint16_image(const unsigned short *image,
                               const unsigned short *outptr,
                               unsigned num_elements)
{
    unsigned i;

    for (i = 0; i < num_elements; i++)
    {
        if (outptr[i] != image[i]) return -1;
    }

    return 0;
}


static std::unique_ptr<float[]> generate_float_image(unsigned num_elements,
                                                     MTdata d)
{
    std::unique_ptr<float[]> ptr{ new float[num_elements] };

    for (unsigned i = 0; i < num_elements; i++)
        ptr[i] = get_random_float(-0x40000000, 0x40000000, d);

    return ptr;
}

static int verify_float_image(const float *image, const float *outptr,
                              unsigned num_elements)
{
    unsigned i;

    for (i = 0; i < num_elements; i++)
    {
        if (outptr[i] != image[i]) return -1;
    }

    return 0;
}

static constexpr cl_image_format image_formats[] = { { CL_RGBA, CL_UNORM_INT8 },
                                                     { CL_RGBA,
                                                       CL_UNORM_INT16 },
                                                     { CL_RGBA, CL_FLOAT } };

static int test_imagecopy3d_impl(cl_device_id device, cl_context context,
                                 cl_command_queue queue,
                                 int num_elements_ignored,
                                 cl_mem_flags src_image_flags)
{
    constexpr size_t image_formats_count = ARRAY_SIZE(image_formats);
    std::unique_ptr<unsigned char[]> rgba8_inptr, rgba8_outptr;
    std::unique_ptr<unsigned short[]> rgba16_inptr, rgba16_outptr;
    std::unique_ptr<float[]> rgbafp_inptr, rgbafp_outptr;
    clMemWrapper streams[6];
    size_t img_width = 128;
    size_t img_height = 128;
    size_t img_depth = 64;
    int i;
    cl_int err;
    unsigned num_elements = img_width * img_height * img_depth * 4;
    MTdataHolder d(gRandomSeed);

    rgba8_inptr = generate_uint8_image(num_elements, d);
    rgba16_inptr = generate_uint16_image(num_elements, d);
    rgbafp_inptr = generate_float_image(num_elements, d);

    rgba8_outptr.reset(new unsigned char[num_elements]);
    rgba16_outptr.reset(new unsigned short[num_elements]);
    rgbafp_outptr.reset(new float[num_elements]);

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
            create_image_3d(context, src_image_flags, &image_formats[index],
                            img_width, img_height, img_depth, 0, 0, ptr, &err);
        test_error(err, "create_image_3d failed");

        streams[index * 2 + 1] = create_image_3d(
            context, CL_MEM_READ_ONLY, &image_formats[index], img_width,
            img_height, img_depth, 0, 0, nullptr, &err);
        test_error(err, "create_image_3d failed");
    }

    for (i = 0; i < image_formats_count; i++)
    {
        void *p, *outp;
        int x, y, z, delta_w = img_width / 8, delta_h = img_height / 16,
                     delta_d = img_depth / 4;

        switch (i)
        {
            case 0:
                p = rgba8_inptr.get();
                outp = rgba8_outptr.get();
                break;
            case 1:
                p = rgba16_inptr.get();
                outp = rgba16_outptr.get();
                break;
            case 2:
                p = rgbafp_inptr.get();
                outp = rgbafp_outptr.get();
                break;
        }

        size_t origin[3] = { 0, 0, 0 },
               region[3] = { img_width, img_height, img_depth };
        if (!(src_image_flags & CL_MEM_USE_HOST_PTR
              || src_image_flags & CL_MEM_COPY_HOST_PTR))
        {
            err = clEnqueueWriteImage(queue, streams[i * 2], CL_TRUE, origin,
                                      region, 0, 0, p, 0, nullptr, nullptr);
            test_error(err, "clEnqueueWriteImage failed");
        }

        for (z = 0; z < img_depth; z += delta_d)
        {
            for (y = 0; y < img_height; y += delta_h)
            {
                for (x = 0; x < img_width; x += delta_w)
                {
                    origin[0] = x;
                    origin[1] = y;
                    origin[2] = z;
                    region[0] = delta_w;
                    region[1] = delta_h;
                    region[2] = delta_d;

                    err = clEnqueueCopyImage(queue, streams[i * 2],
                                             streams[i * 2 + 1], origin, origin,
                                             region, 0, NULL, NULL);
                    test_error(err, "clEnqueueCopyImage failed");
                }
            }
        }

        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = img_width;
        region[1] = img_height;
        region[2] = img_depth;
        err = clEnqueueReadImage(queue, streams[i * 2 + 1], CL_TRUE, origin,
                                 region, 0, 0, outp, 0, NULL, NULL);
        test_error(err, "clEnqueueReadImage failed");

        switch (i)
        {
            case 0:
                err = verify_uint8_image(rgba8_inptr.get(), rgba8_outptr.get(),
                                         num_elements);
                if (err) log_error("Failed uint8\n");
                break;
            case 1:
                err = verify_uint16_image(rgba16_inptr.get(),
                                          rgba16_outptr.get(), num_elements);
                if (err) log_error("Failed uint16\n");
                break;
            case 2:
                err = verify_float_image(rgbafp_inptr.get(),
                                         rgbafp_outptr.get(), num_elements);
                if (err) log_error("Failed float\n");
                break;
        }

        if (err) break;
    }

    if (err)
        log_error("IMAGE3D copy test failed\n");
    else
        log_info("IMAGE3D copy test passed\n");

    return err;
}

REGISTER_TEST(imagecopy3d)
{
    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device);

    return test_imagecopy3d_impl(device, context, queue, num_elements,
                                 CL_MEM_READ_WRITE);
}
