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
#include "testBase.h"

REGISTER_TEST(min_image_formats)
{
    int missingFormats = 0;

    cl_int error = CL_SUCCESS;

    Version version = get_device_cl_version(device);

    cl_bool supports_images = CL_FALSE;
    error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT,
                            sizeof(supports_images), &supports_images, NULL);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_IMAGE_SUPPORT failed");

    if (supports_images == CL_FALSE)
    {
        log_info("No image support on current device - skipped\n");
        return TEST_SKIPPED_ITSELF;
    }

    const cl_mem_object_type image_types[] = {
        CL_MEM_OBJECT_IMAGE1D,       CL_MEM_OBJECT_IMAGE1D_BUFFER,
        CL_MEM_OBJECT_IMAGE2D,       CL_MEM_OBJECT_IMAGE3D,
        CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D_ARRAY,
    };
    const cl_mem_flags mem_flags[] = {
        CL_MEM_READ_ONLY,
        CL_MEM_WRITE_ONLY,
        CL_MEM_KERNEL_READ_AND_WRITE,
    };

    cl_bool supports_read_write_images = CL_FALSE;
    if (version >= Version(3, 0))
    {
        cl_uint maxReadWriteImageArgs = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
                                sizeof(maxReadWriteImageArgs),
                                &maxReadWriteImageArgs, NULL);
        test_error(error,
                   "Unable to query "
                   "CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS");

        // read-write images are supported if MAX_READ_WRITE_IMAGE_ARGS is
        // nonzero
        supports_read_write_images =
            maxReadWriteImageArgs != 0 ? CL_TRUE : CL_FALSE;
    }
    else if (version >= Version(2, 0))
    {
        // read-write images are required for OpenCL 2.x
        supports_read_write_images = CL_TRUE;
    }

    int supports_3D_image_writes =
        is_extension_available(device, "cl_khr_3d_image_writes");

    for (size_t t = 0; t < ARRAY_SIZE(image_types); t++)
    {
        const cl_mem_object_type type = image_types[t];
        log_info("    testing %s...\n", convert_image_type_to_string(type));
        for (size_t f = 0; f < ARRAY_SIZE(mem_flags); f++)
        {
            const cl_mem_flags flags = mem_flags[f];
            const char* testTypeString = flags == CL_MEM_READ_ONLY
                ? "read-only"
                : flags == CL_MEM_WRITE_ONLY
                    ? "write only"
                    : flags == CL_MEM_KERNEL_READ_AND_WRITE ? "read and write"
                                                            : "unknown???";

            if (flags == CL_MEM_KERNEL_READ_AND_WRITE
                && !supports_read_write_images)
            {
                continue;
            }

            if (type == CL_MEM_OBJECT_IMAGE3D && flags != CL_MEM_READ_ONLY
                && !supports_3D_image_writes)
            {
                continue;
            }

            cl_uint numImageFormats = 0;
            error = clGetSupportedImageFormats(context, flags, type, 0, NULL,
                                               &numImageFormats);
            test_error(error, "Unable to query number of image formats");

            std::vector<cl_image_format> supportedFormats(numImageFormats);
            if (numImageFormats != 0)
            {
                error = clGetSupportedImageFormats(
                    context, flags, type, supportedFormats.size(),
                    supportedFormats.data(), NULL);
                test_error(error, "Unable to query image formats");
            }

            std::vector<cl_image_format> requiredFormats;
            build_required_image_formats(flags, type, device, requiredFormats);

            for (auto& format : requiredFormats)
            {
                if (!find_format(supportedFormats.data(),
                                 supportedFormats.size(), &format))
                {
                    log_error(
                        "Missing required %s format %s + %s.\n", testTypeString,
                        GetChannelOrderName(format.image_channel_order),
                        GetChannelTypeName(format.image_channel_data_type));
                    ++missingFormats;
                }
            }
        }
    }

    return missingFormats == 0 ? TEST_PASS : TEST_FAIL;
}
