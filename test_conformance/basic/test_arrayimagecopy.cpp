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

#include "procs.h"

int test_arrayimagecopy_single_format(cl_device_id device, cl_context context,
                                      cl_command_queue queue,
                                      cl_mem_flags flags,
                                      cl_mem_object_type image_type,
                                      const cl_image_format *format)
{
    cl_uchar *bufptr, *imgptr;
    clMemWrapper buffer, image;
    int img_width = 512;
    int img_height = 512;
    int img_depth = (image_type == CL_MEM_OBJECT_IMAGE3D) ? 32 : 1;
    size_t elem_size;
    size_t buffer_size;
    cl_int err;
    cl_event copyevent;

    log_info("Testing %s %s\n",
             GetChannelOrderName(format->image_channel_order),
             GetChannelTypeName(format->image_channel_data_type));

    if (CL_MEM_OBJECT_IMAGE2D == image_type)
    {
        image = create_image_2d(context, flags, format, img_width, img_height,
                                0, nullptr, &err);
    }
    else
    {
        image = create_image_3d(context, flags, format, img_width, img_height,
                                img_depth, 0, 0, nullptr, &err);
    }
    test_error(err, "create_image_xd failed");

    err = clGetImageInfo(image, CL_IMAGE_ELEMENT_SIZE, sizeof(size_t),
                         &elem_size, NULL);
    test_error(err, "clGetImageInfo failed");

    buffer_size =
        sizeof(cl_uchar) * elem_size * img_width * img_height * img_depth;

    buffer =
        clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    test_error(err, "clCreateBuffer failed");

    RandomSeed seed(gRandomSeed);
    bufptr =
        static_cast<cl_uchar *>(create_random_data(kUChar, seed, buffer_size));

    size_t origin[3] = { 0, 0, 0 },
           region[3] = { img_width, img_height, img_depth };
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, buffer_size, bufptr,
                               0, NULL, NULL);
    test_error(err, "clEnqueueWriteBuffer failed");

    err = clEnqueueCopyBufferToImage(queue, buffer, image, 0, origin, region, 0,
                                     NULL, &copyevent);
    test_error(err, "clEnqueueCopyImageToBuffer failed");

    imgptr = static_cast<cl_uchar *>(malloc(buffer_size));

    err = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                             imgptr, 1, &copyevent, NULL);
    test_error(err, "clEnqueueReadImage failed");

    err = clReleaseEvent(copyevent);
    test_error(err, "clReleaseEvent failed");

    if (memcmp(bufptr, imgptr, buffer_size) != 0)
    {
        log_error("ERROR: Results did not validate!\n");
        auto inchar = static_cast<unsigned char *>(bufptr);
        auto outchar = static_cast<unsigned char *>(imgptr);
        int failuresPrinted = 0;
        for (int i = 0; i < (int)buffer_size; i += (int)elem_size)
        {
            if (memcmp(&inchar[i], &outchar[i], elem_size) != 0)
            {
                log_error("%d(0x%x) -> actual [", i, i);
                for (int j = 0; j < (int)elem_size; j++)
                    log_error("0x%02x ", inchar[i + j]);
                log_error("] != expected [");
                for (int j = 0; j < (int)elem_size; j++)
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

    free(bufptr);
    free(imgptr);

    if (err)
        log_error(
            "ARRAY to IMAGE copy test failed for image_channel_order=0x%lx and "
            "image_channel_data_type=0x%lx\n",
            (unsigned long)format->image_channel_order,
            (unsigned long)format->image_channel_data_type);

    return err;
}


int test_arrayimagecommon(cl_device_id device, cl_context context,
                          cl_command_queue queue, cl_mem_flags flags,
                          cl_mem_object_type image_type)
{
    cl_int err;
    cl_uint num_formats;

    err = clGetSupportedImageFormats(context, flags, image_type, 0, NULL,
                                     &num_formats);
    test_error(err, "clGetSupportedImageFormats failed");

    std::vector<cl_image_format> formats(num_formats);

    err = clGetSupportedImageFormats(context, flags, image_type, num_formats,
                                     formats.data(), NULL);
    test_error(err, "clGetSupportedImageFormats failed");

    for (const auto &format : formats)
    {
        err |= test_arrayimagecopy_single_format(device, context, queue, flags,
                                                 image_type, &format);
    }

    if (err)
        log_error("ARRAY to IMAGE%s copy test failed\n",
                  convert_image_type_to_string(image_type));
    else
        log_info("ARRAY to IMAGE%s copy test passed\n",
                 convert_image_type_to_string(image_type));

    return err;
}

int test_arrayimagecopy(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    return test_arrayimagecommon(device, context, queue, CL_MEM_READ_WRITE,
                                 CL_MEM_OBJECT_IMAGE2D);
}


int test_arrayimagecopy3d(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)

    return test_arrayimagecommon(device, context, queue, CL_MEM_READ_ONLY,
                                 CL_MEM_OBJECT_IMAGE3D);
}
