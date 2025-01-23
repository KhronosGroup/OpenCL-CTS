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
#include "harness/imageHelpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <memory>

#include "procs.h"

using test_function_t = int (*)(cl_device_id, cl_context, cl_command_queue,
                                cl_mem_flags, cl_mem_flags, cl_mem_object_type,
                                const cl_image_format *);

int test_imagearraycopy_single_format(cl_device_id device, cl_context context,
                                      cl_command_queue queue,
                                      cl_mem_flags image_flags,
                                      cl_mem_flags buffer_flags,
                                      cl_mem_object_type image_type,
                                      const cl_image_format *format)
{
    std::unique_ptr<cl_uchar, decltype(&free)> bufptr{ nullptr, free },
        imgptr{ nullptr, free };
    clMemWrapper buffer, image;
    const int img_width = 512;
    const int img_height = 512;
    const int img_depth = (image_type == CL_MEM_OBJECT_IMAGE3D) ? 32 : 1;
    size_t elem_size;
    size_t buffer_size;
    cl_int err;
    cl_event copyevent;
    RandomSeed seed(gRandomSeed);

    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, img_depth };

    log_info("Testing %s %s\n",
             GetChannelOrderName(format->image_channel_order),
             GetChannelTypeName(format->image_channel_data_type));

    elem_size = get_pixel_size(format);
    buffer_size =
        sizeof(cl_uchar) * elem_size * img_width * img_height * img_depth;

    if (image_flags & CL_MEM_USE_HOST_PTR || image_flags & CL_MEM_COPY_HOST_PTR)
    {
        imgptr.reset(static_cast<cl_uchar *>(
            create_random_data(kUChar, seed, buffer_size)));
    }

    bufptr.reset(
        static_cast<cl_uchar *>(create_random_data(kUChar, seed, buffer_size)));

    if (CL_MEM_OBJECT_IMAGE2D == image_type)
    {
        image = create_image_2d(context, image_flags, format, img_width,
                                img_height, 0, imgptr.get(), &err);
    }
    else
    {
        image =
            create_image_3d(context, image_flags, format, img_width, img_height,
                            img_depth, 0, 0, imgptr.get(), &err);
    }
    test_error(err, "create_image_xd failed");

    if (!(image_flags & CL_MEM_USE_HOST_PTR
          || image_flags & CL_MEM_COPY_HOST_PTR))
    {
        imgptr.reset(static_cast<cl_uchar *>(
            create_random_data(kUChar, seed, buffer_size)));

        err = clEnqueueWriteImage(queue, image, CL_TRUE, origin, region, 0, 0,
                                  imgptr.get(), 0, nullptr, nullptr);
        test_error(err, "clEnqueueWriteImage failed");
    }

    buffer = clCreateBuffer(context, buffer_flags, buffer_size, nullptr, &err);
    test_error(err, "clCreateBuffer failed");

    err = clEnqueueCopyImageToBuffer(queue, image, buffer, origin, region, 0, 0,
                                     nullptr, &copyevent);
    test_error(err, "clEnqueueCopyImageToBuffer failed");

    bufptr.reset(static_cast<cl_uchar *>(malloc(buffer_size)));

    err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, buffer_size,
                              bufptr.get(), 1, &copyevent, nullptr);
    test_error(err, "clEnqueueReadBuffer failed");

    err = clReleaseEvent(copyevent);
    test_error(err, "clReleaseEvent failed");

    image_descriptor compareImageInfo = { 0 };
    compareImageInfo.format = format;
    compareImageInfo.width = buffer_size / get_pixel_size(format);
    size_t where = compare_scanlines(
        &compareImageInfo, reinterpret_cast<const char *>(imgptr.get()),
        reinterpret_cast<const char *>(bufptr.get()));
    if (where < compareImageInfo.width)
    {
        log_error("ERROR: Results did not validate!\n");
        auto inchar = static_cast<unsigned char *>(imgptr.get());
        auto outchar = static_cast<unsigned char *>(bufptr.get());
        int failuresPrinted = 0;
        for (size_t i = 0; i < buffer_size; i += elem_size)
        {
            if (memcmp(&inchar[i], &outchar[i], elem_size) != 0)
            {
                log_error("%d(0x%x) -> expected [", i, i);
                for (size_t j = 0; j < elem_size; j++)
                    log_error("0x%02x ", inchar[i + j]);
                log_error("] != actual [");
                for (size_t j = 0; j < elem_size; j++)
                    log_error("0x%02x ", outchar[i + j]);
                log_error("]\n");
                failuresPrinted++;
            }
            if (failuresPrinted > 5)
            {
                log_error("Not printing further failures...\n");
                break;
            }
        }
        err = -1;
    }

    if (err)
        log_error(
            "IMAGE to ARRAY copy test failed for image_channel_order=0x%lx and "
            "image_channel_data_type=0x%lx\n",
            static_cast<unsigned long>(format->image_channel_order),
            static_cast<unsigned long>(format->image_channel_data_type));

    return err;
}

int test_imagearraycommon(cl_device_id device, cl_context context,
                          cl_command_queue queue, cl_mem_flags image_flags,
                          cl_mem_flags buffer_flags,
                          cl_mem_object_type image_type,
                          test_function_t test_function)
{
    cl_int err;
    cl_uint num_formats;

    err = clGetSupportedImageFormats(context, image_flags, image_type, 0,
                                     nullptr, &num_formats);
    test_error(err, "clGetSupportedImageFormats failed");

    std::vector<cl_image_format> formats(num_formats);

    err = clGetSupportedImageFormats(context, image_flags, image_type,
                                     num_formats, formats.data(), nullptr);
    test_error(err, "clGetSupportedImageFormats failed");

    for (const auto &format : formats)
    {
        err |= test_function(device, context, queue, image_flags, buffer_flags,
                             image_type, &format);
    }

    if (err)
        log_error("IMAGE%s to ARRAY copy test failed\n",
                  convert_image_type_to_string(image_type));
    else
        log_info("IMAGE%s to ARRAY copy test passed\n",
                 convert_image_type_to_string(image_type));

    return err;
}

int test_imagearraycopy(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    return test_imagearraycommon(device, context, queue, CL_MEM_READ_WRITE,
                                 CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D,
                                 test_imagearraycopy_single_format);
}


int test_imagearraycopy3d(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)

    return test_imagearraycommon(device, context, queue, CL_MEM_READ_ONLY,
                                 CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE3D,
                                 test_imagearraycopy_single_format);
}
