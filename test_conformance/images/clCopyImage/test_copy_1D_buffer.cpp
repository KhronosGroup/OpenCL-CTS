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

extern int test_copy_image_generic(cl_context context, cl_command_queue queue,
                                   image_descriptor *srcImageInfo,
                                   image_descriptor *dstImageInfo,
                                   const size_t sourcePos[],
                                   const size_t destPos[],
                                   const size_t regionSize[], MTdata d);

int test_copy_image_size_1D_buffer(cl_context context, cl_command_queue queue,
                                   image_descriptor *srcImageInfo,
                                   image_descriptor *dstImageInfo, MTdata d)
{
    size_t sourcePos[3], destPos[3], regionSize[3];
    int ret = 0, retCode;
    size_t width_lod = srcImageInfo->width;

    // First, try just a full covering region
    sourcePos[0] = sourcePos[1] = sourcePos[2] = 0;
    destPos[0] = destPos[1] = destPos[2] = 0;
    regionSize[0] = srcImageInfo->width;
    regionSize[1] = 1;
    regionSize[2] = 1;

    retCode =
        test_copy_image_generic(context, queue, srcImageInfo, dstImageInfo,
                                sourcePos, destPos, regionSize, d);
    if (retCode < 0)
        return retCode;
    else
        ret += retCode;

    // Now try a sampling of different random regions
    for (int i = 0; i < 8; i++)
    {
        // Pick a random size
        regionSize[0] = (width_lod > 8)
            ? (size_t)random_in_range(8, (int)width_lod - 1, d)
            : width_lod;

        // Now pick positions within valid ranges
        sourcePos[0] = (width_lod > regionSize[0]) ? (size_t)random_in_range(
                           0, (int)(width_lod - regionSize[0] - 1), d)
                                                   : 0;
        destPos[0] = (width_lod > regionSize[0]) ? (size_t)random_in_range(
                         0, (int)(width_lod - regionSize[0] - 1), d)
                                                 : 0;


        // Go for it!
        retCode =
            test_copy_image_generic(context, queue, srcImageInfo, srcImageInfo,
                                    sourcePos, destPos, regionSize, d);
        if (retCode < 0)
            return retCode;
        else
            ret += retCode;
    }

    return ret;
}

int test_copy_image_set_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format)
{
    assert(
        dst_type
        == src_type); // This test expects to copy 1D buffer -> 1D buffer images
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor srcImageInfo = { 0 };
    image_descriptor dstImageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    size_t pixelSize;

    if (gTestMipmaps)
    {
        // 1D image buffers don't support mipmaps
        // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_mipmap_image
        return 0;
    }

    srcImageInfo.format = format;
    srcImageInfo.height = srcImageInfo.depth = srcImageInfo.arraySize =
        srcImageInfo.slicePitch = 0;
    srcImageInfo.type = src_type;
    srcImageInfo.mem_flags = src_flags;
    pixelSize = get_pixel_size(srcImageInfo.format);

    int error = clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                                sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                             &memSize, NULL);
    test_error(error, "Unable to get max image 1D buffer size from device");

    if (memSize > (cl_ulong)SIZE_MAX)
    {
        memSize = (cl_ulong)SIZE_MAX;
        maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if (gTestSmallImages)
    {
        for (srcImageInfo.width = 1; srcImageInfo.width < 13;
             srcImageInfo.width++)
        {
            size_t rowPadding = gEnablePitch ? 48 : 0;
            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
                do
                {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            if (gDebugTrace)
                log_info("   at size %d\n", (int)srcImageInfo.width);

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            int ret = test_copy_image_size_1D_buffer(
                context, queue, &srcImageInfo, &dstImageInfo, seed);
            if (ret) return -1;
        }
    }
    else if (gTestMaxImages)
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, 1,
                      maxAllocSize, memSize, src_type, srcImageInfo.format);

        for (size_t idx = 0; idx < numbeOfSizes; idx++)
        {
            size_t rowPadding = gEnablePitch ? 48 : 0;
            srcImageInfo.width = sizes[idx][0];
            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
                do
                {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            log_info("Testing %d\n", (int)sizes[idx][0]);
            if (gDebugTrace)
                log_info("   at max size %d\n", (int)sizes[idx][0]);

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            if (test_copy_image_size_1D_buffer(context, queue, &srcImageInfo,
                                               &dstImageInfo, seed))
                return -1;
        }
    }
    else
    {
        for (int i = 0; i < NUM_IMAGE_ITERATIONS; i++)
        {
            cl_ulong size;
            size_t rowPadding = gEnablePitch ? 48 : 0;
            // Loop until we get a size that a) will fit in the max alloc size
            // and b) that an allocation of that image, the result array, plus
            // offset arrays, will fit in the global ram space
            do
            {
                srcImageInfo.width =
                    (size_t)random_log_in_range(16, (int)maxWidth / 32, seed);

                srcImageInfo.rowPitch =
                    srcImageInfo.width * pixelSize + rowPadding;

                if (gEnablePitch)
                {
                    do
                    {
                        rowPadding++;
                        srcImageInfo.rowPitch =
                            srcImageInfo.width * pixelSize + rowPadding;
                    } while ((srcImageInfo.rowPitch % pixelSize) != 0);
                }

                size = (size_t)srcImageInfo.rowPitch * 4;
            } while (size > maxAllocSize || (size * 3) > memSize);

            if (gDebugTrace)
            {
                log_info("   at size %d (row pitch %d) out of %d\n",
                         (int)srcImageInfo.width, (int)srcImageInfo.rowPitch,
                         (int)maxWidth);
            }

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            int ret = test_copy_image_size_1D_buffer(
                context, queue, &srcImageInfo, &dstImageInfo, seed);
            if (ret) return -1;
        }
    }

    return 0;
}

int test_copy_image_set_1D_1D_buffer(
    cl_device_id device, cl_context context, cl_command_queue queue,
    cl_mem_flags src_flags, cl_mem_object_type src_type, cl_mem_flags dst_flags,
    cl_mem_object_type dst_type, cl_image_format *format)
{
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor srcImageInfo = { 0 };
    image_descriptor dstImageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    size_t pixelSize;

    if (gTestMipmaps)
    {
        // 1D image buffers don't support mipmaps
        // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_mipmap_image
        return 0;
    }

    srcImageInfo.format = format;
    srcImageInfo.height = srcImageInfo.depth = srcImageInfo.arraySize =
        srcImageInfo.slicePitch = 0;
    srcImageInfo.type = src_type;
    srcImageInfo.mem_flags = src_flags;
    pixelSize = get_pixel_size(srcImageInfo.format);

    int error = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                sizeof(maxWidth), &maxWidth, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                             sizeof(maxAllocSize), &maxAllocSize, NULL);
    error |= clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                             &memSize, NULL);
    test_error(error, "Unable to get max image 1D buffer size from device");

    if (memSize > (cl_ulong)SIZE_MAX)
    {
        memSize = (cl_ulong)SIZE_MAX;
        maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if (gTestSmallImages)
    {
        for (srcImageInfo.width = 1; srcImageInfo.width < 13;
             srcImageInfo.width++)
        {
            size_t rowPadding = gEnablePitch ? 48 : 0;
            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
                do
                {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            if (gDebugTrace)
                log_info("   at size %d\n", (int)srcImageInfo.width);

            dstImageInfo = srcImageInfo;
            dstImageInfo.type = dst_type;
            dstImageInfo.mem_flags = dst_flags;

            int ret = test_copy_image_size_1D_buffer(
                context, queue, &srcImageInfo, &dstImageInfo, seed);
            if (ret) return -1;
        }
    }
    else if (gTestMaxImages)
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, 1,
                      maxAllocSize, memSize, src_type, srcImageInfo.format);

        for (size_t idx = 0; idx < numbeOfSizes; idx++)
        {
            size_t rowPadding = gEnablePitch ? 48 : 0;
            srcImageInfo.width = sizes[idx][0];
            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
                do
                {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            log_info("Testing %d\n", (int)sizes[idx][0]);
            if (gDebugTrace)
                log_info("   at max size %d\n", (int)sizes[idx][0]);

            dstImageInfo = srcImageInfo;
            dstImageInfo.type = dst_type;
            dstImageInfo.mem_flags = dst_flags;

            if (test_copy_image_size_1D_buffer(context, queue, &srcImageInfo,
                                               &dstImageInfo, seed))
                return -1;
        }
    }
    else
    {
        for (int i = 0; i < NUM_IMAGE_ITERATIONS; i++)
        {
            cl_ulong size;
            size_t rowPadding = gEnablePitch ? 48 : 0;
            // Loop until we get a size that a) will fit in the max alloc size
            // and b) that an allocation of that image, the result array, plus
            // offset arrays, will fit in the global ram space
            do
            {
                srcImageInfo.width =
                    (size_t)random_log_in_range(16, (int)maxWidth / 32, seed);

                srcImageInfo.rowPitch =
                    srcImageInfo.width * pixelSize + rowPadding;

                if (gEnablePitch)
                {
                    do
                    {
                        rowPadding++;
                        srcImageInfo.rowPitch =
                            srcImageInfo.width * pixelSize + rowPadding;
                    } while ((srcImageInfo.rowPitch % pixelSize) != 0);
                }

                size = (size_t)srcImageInfo.rowPitch * 4;
            } while (size > maxAllocSize || (size * 3) > memSize);

            if (gDebugTrace)
            {
                log_info("   at size %d (row pitch %d) out of %d\n",
                         (int)srcImageInfo.width, (int)srcImageInfo.rowPitch,
                         (int)maxWidth);
            }

            dstImageInfo = srcImageInfo;
            dstImageInfo.type = dst_type;
            dstImageInfo.mem_flags = dst_flags;

            int ret = test_copy_image_size_1D_buffer(
                context, queue, &srcImageInfo, &dstImageInfo, seed);
            if (ret) return -1;
        }
    }

    return 0;
}
