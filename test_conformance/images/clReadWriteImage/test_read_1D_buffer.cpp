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

int test_read_image_1D_buffer(cl_context context, cl_command_queue queue,
                              image_descriptor *imageInfo, MTdata d,
                              cl_mem_flags flags)
{
    int error;

    clMemWrapper image;
    clMemWrapper buffer;

    // Generate some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data(imageInfo, imageValues, d);

    if (gDebugTrace)
    {
        log_info(" - Creating 1D image %d...\n", (int)imageInfo->width);
        log_info(" with %llu mip levels\n",
                 (unsigned long long)imageInfo->num_mip_levels);
    }

    buffer = clCreateBuffer(context, flags, imageInfo->rowPitch, NULL, &error);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to create buffer for 1D image buffer of size "
                  "%d (%s)",
                  (int)imageInfo->rowPitch, IGetErrorString(error));
    }

    image = create_image_1d(context, flags, imageInfo->format, imageInfo->width,
                            0, NULL, buffer, &error);
    if (image == NULL)
    {
        log_error("ERROR: Unable to create 1D image buffer of size %d (%s)",
                  (int)imageInfo->width, IGetErrorString(error));
        return -1;
    }

    if (gDebugTrace) log_info(" - Writing image...\n");

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { imageInfo->width, 1, 1 };
    size_t fullImageSize = imageInfo->rowPitch;

    BufferOwningPtr<char> resultValues(malloc(fullImageSize));
    size_t imgValMipLevelOffset = 0;

    error = clEnqueueWriteImage(queue, image, CL_FALSE, origin, region,
                                (gEnablePitch ? imageInfo->rowPitch : 0), 0,
                                (char *)imageValues + imgValMipLevelOffset, 0,
                                NULL, NULL);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: Unable to write to 1D image of size %d \n",
                  (int)imageInfo->width);
        return -1;
    }

    // To verify, we just read the results right back and see whether they
    // match the input
    if (gDebugTrace)
    {
        log_info(" - Initing result array...\n");
    }

    // Note: we read back without any pitch, to verify pitch actually WORKED
    size_t scanlineSize = imageInfo->width * get_pixel_size(imageInfo->format);
    size_t imageSize = scanlineSize;
    memset(resultValues, 0xff, imageSize);

    if (gDebugTrace) log_info(" - Reading results...\n");

    error = clEnqueueReadImage(queue, image, CL_TRUE, origin, region, 0, 0,
                               resultValues, 0, NULL, NULL);
    test_error(error, "Unable to read image values");

    // Verify scanline by scanline, since the pitches are different
    char *sourcePtr = (char *)imageValues + imgValMipLevelOffset;
    char *destPtr = resultValues;

    if (memcmp(sourcePtr, destPtr, scanlineSize) != 0)
    {
        log_error("ERROR: Scanline did not verify for image size %d pitch "
                  "%d (extra %d bytes)\n",
                  (int)imageInfo->width, (int)imageInfo->rowPitch,
                  (int)imageInfo->rowPitch
                      - (int)imageInfo->width
                          * (int)get_pixel_size(imageInfo->format));

        log_error("First few values: \n");
        log_error(" Input: ");
        uint32_t *s = (uint32_t *)sourcePtr;
        uint32_t *d = (uint32_t *)destPtr;
        for (int q = 0; q < 12; q++) log_error("%08x ", s[q]);
        log_error("\nOutput: ");
        for (int q = 0; q < 12; q++) log_error("%08x ", d[q]);
        log_error("\n");

        int outX;
        int offset = (int)get_pixel_size(imageInfo->format)
            * (int)(imageInfo->width - 16);
        if (offset < 0) offset = 0;
        int foundCount = debug_find_vector_in_image(
            (char *)imageValues + imgValMipLevelOffset, imageInfo,
            destPtr + offset, get_pixel_size(imageInfo->format), &outX, NULL,
            NULL);
        if (foundCount > 0)
        {
            int returnedOffset =
                (offset / (int)get_pixel_size(imageInfo->format)) - outX;

            if (memcmp(sourcePtr
                           + returnedOffset * get_pixel_size(imageInfo->format),
                       destPtr, get_pixel_size(imageInfo->format) * 8)
                == 0)
                log_error("       Values appear to be offsetted by %d\n",
                          returnedOffset);
            else
                log_error(
                    "       Calculated offset is %d but unable to verify\n",
                    returnedOffset);
        }
        else
        {
            log_error("      Unable to determine offset\n");
        }
        return -1;
    }
    imgValMipLevelOffset +=
        imageInfo->width * get_pixel_size(imageInfo->format);
    return 0;
}

int test_read_image_set_1D_buffer(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  cl_image_format *format, cl_mem_flags flags)
{
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    size_t pixelSize;

    if (gTestMipmaps)
    {
        // 1D image buffers don't support mipmaps
        // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_mipmap_image
        return 0;
    }

    imageInfo.type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    imageInfo.format = format;
    imageInfo.height = imageInfo.depth = imageInfo.slicePitch = 0;
    pixelSize = get_pixel_size(imageInfo.format);

    int error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                             &memSize, NULL);
    test_error(error, "Unable to get max image 2D size from device");

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

            if (gDebugTrace) log_info("   at size %d\n", (int)imageInfo.width);

            int ret = test_read_image_1D_buffer(context, queue, &imageInfo,
                                                seed, flags);
            if (ret) return -1;
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

            log_info("Testing %d\n", (int)imageInfo.width);
            if (gDebugTrace) log_info("   at max size %d\n", (int)maxWidth);
            if (test_read_image_1D_buffer(context, queue, &imageInfo, seed,
                                          flags))
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
                    (size_t)random_log_in_range(16, (int)(maxWidth / 32), seed);

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                if (gEnablePitch)
                {
                    size_t extraWidth = (int)random_log_in_range(0, 64, seed);
                    imageInfo.rowPitch += extraWidth * pixelSize;
                }

                size = (size_t)imageInfo.rowPitch * 4;
            } while (size > maxAllocSize || (size / 3) > memSize);

            if (gDebugTrace)
                log_info("   at size %d (row pitch %d) out of %d\n",
                         (int)imageInfo.width, (int)imageInfo.rowPitch,
                         (int)maxWidth);
            int ret = test_read_image_1D_buffer(context, queue, &imageInfo,
                                                seed, flags);
            if (ret) return -1;
        }
    }

    return 0;
}
