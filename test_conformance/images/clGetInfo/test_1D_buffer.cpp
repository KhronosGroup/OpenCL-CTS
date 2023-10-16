//
// Copyright (c) 2023 The Khronos Group Inc.
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
#include "../testBase.h"
#include <CL/cl.h>

extern int test_get_image_info_single(cl_context context,
                                      image_descriptor *imageInfo, MTdata d,
                                      cl_mem_flags flags, size_t row_pitch,
                                      size_t slice_pitch);


int test_get_image_info_1D_buffer(cl_device_id device, cl_context context,
                                  cl_image_format *format, cl_mem_flags flags)
{
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    size_t pixelSize;

    memset(&imageInfo, 0x0, sizeof(image_descriptor));
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    imageInfo.format = format;
    pixelSize = get_pixel_size(imageInfo.format);

    int error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                             &memSize, NULL);
    test_error(error, "Unable to get max image 1D size from device");

    if (memSize > (cl_ulong)SIZE_MAX)
    {
        memSize = (cl_ulong)SIZE_MAX;
        maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if (gTestSmallImages)
    {
        for (imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++)
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            if (gDebugTrace)
                log_info("   at size %d (flags 0x%x pitch %d)\n",
                         (int)imageInfo.width, (unsigned int)flags,
                         (int)imageInfo.rowPitch);
            if (test_get_image_info_single(context, &imageInfo, seed, flags, 0,
                                           0))
                return -1;
        }
    }
    else if (gTestMaxImages)
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, 1,
                      maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE1D_BUFFER,
                      imageInfo.format);

        for (size_t idx = 0; idx < numbeOfSizes; idx++)
        {
            imageInfo.width = sizes[idx][0];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            log_info("Testing %d x 1\n", (int)sizes[idx][0]);
            if (gDebugTrace)
                log_info("   at max size %d (flags 0x%x pitch %d)\n",
                         (int)imageInfo.width, (unsigned int)flags,
                         (int)imageInfo.rowPitch);
            if (test_get_image_info_single(context, &imageInfo, seed, flags, 0,
                                           0))
                return -1;
        }
    }
    else
    {
        for (int i = 0; i < NUM_IMAGE_ITERATIONS; i++)
        {
            cl_ulong size;
            // Loop until we get a size that a) will fit in the max alloc size
            // and b) that an allocation of that image, the result array, plus
            // offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width =
                    (size_t)random_log_in_range(16, (int)maxWidth / 32, seed);

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                size_t extraWidth = (int)random_log_in_range(0, 64, seed);
                imageInfo.rowPitch += extraWidth;

                do
                {
                    extraWidth++;
                    imageInfo.rowPitch += extraWidth;
                } while ((imageInfo.rowPitch % pixelSize) != 0);

                size = (cl_ulong)imageInfo.rowPitch * 4;
            } while (size > maxAllocSize || (size * 3) > memSize);

            if (gDebugTrace)
                log_info("   at size %d (flags 0x%x pitch %d) out of %d\n",
                         (int)imageInfo.width, (unsigned int)flags,
                         (int)imageInfo.rowPitch, (int)maxWidth);
            if (test_get_image_info_single(context, &imageInfo, seed, flags, 0,
                                           0))
                return -1;
        }
    }

    return 0;
}
