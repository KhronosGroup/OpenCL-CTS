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
#include "../testBase.h"

// Defined in test_copy_generic.cpp
extern int test_copy_image_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo,
                                   const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d );

int test_copy_image_3D(cl_context context, cl_command_queue queue,
                       image_descriptor *srcImageInfo,
                       image_descriptor *dstImageInfo, MTdata d)
{
    size_t origin[] = { 0, 0, 0, 0};
    size_t region[] = { srcImageInfo->width, srcImageInfo->height,
                        srcImageInfo->depth };

    if( gTestMipmaps )
    {
        size_t lod = (srcImageInfo->num_mip_levels > 1)
            ? (size_t)random_in_range(0, srcImageInfo->num_mip_levels - 1, d)
            : 0;
        origin[ 3 ] = lod;
        region[0] =
            (srcImageInfo->width >> lod) ? (srcImageInfo->width >> lod) : 1;
        region[1] =
            (srcImageInfo->height >> lod) ? (srcImageInfo->height >> lod) : 1;
        region[2] =
            (srcImageInfo->depth >> lod) ? (srcImageInfo->depth >> lod) : 1;
    }

    return test_copy_image_generic(context, queue, srcImageInfo, dstImageInfo,
                                   origin, origin, region, d);
}

int test_copy_image_set_3D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_mem_flags src_flags,
                           cl_mem_object_type src_type, cl_mem_flags dst_flags,
                           cl_mem_object_type dst_type, cl_image_format *format)
{
    assert(dst_type == src_type); // This test expects to copy 3D -> 3D images
    size_t maxWidth, maxHeight, maxDepth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor srcImageInfo = { 0 };
    image_descriptor dstImageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    srcImageInfo.format = format;
    srcImageInfo.type = src_type;
    srcImageInfo.mem_flags = src_flags;
    pixelSize = get_pixel_size(srcImageInfo.format);

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof( maxDepth ), &maxDepth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 3D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
      maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for (srcImageInfo.width = 1; srcImageInfo.width < 13;
             srcImageInfo.width++)
        {
            size_t rowPadding = gEnablePitch ? 80 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gTestMipmaps)
                srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
                    2,
                    (int)compute_max_mip_levels(srcImageInfo.width,
                                                srcImageInfo.height,
                                                srcImageInfo.depth),
                    seed);

            if (gEnablePitch)
            {
                do {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            for (srcImageInfo.height = 1; srcImageInfo.height < 9;
                 srcImageInfo.height++)
            {
                srcImageInfo.slicePitch = srcImageInfo.rowPitch
                    * (srcImageInfo.height + slicePadding);
                for (srcImageInfo.depth = 2; srcImageInfo.depth < 9;
                     srcImageInfo.depth++)
                {
                    if( gDebugTrace )
                        log_info(
                            "   at size %d,%d,%d\n", (int)srcImageInfo.width,
                            (int)srcImageInfo.height, (int)srcImageInfo.depth);

                    dstImageInfo = srcImageInfo;
                    dstImageInfo.mem_flags = dst_flags;
                    int ret = test_copy_image_3D(context, queue, &srcImageInfo,
                                                 &dstImageInfo, seed);
                    if( ret )
                        return -1;
                }
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];
        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, maxDepth,
                      1, maxAllocSize, memSize, srcImageInfo.type,
                      srcImageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            size_t rowPadding = gEnablePitch ? 80 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            srcImageInfo.width = sizes[idx][0];
            srcImageInfo.height = sizes[idx][1];
            srcImageInfo.depth = sizes[idx][2];
            srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

            if (gTestMipmaps)
                srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
                    2,
                    (int)compute_max_mip_levels(srcImageInfo.width,
                                                srcImageInfo.height,
                                                srcImageInfo.depth),
                    seed);

            if (gEnablePitch)
            {
                do {
                    rowPadding++;
                    srcImageInfo.rowPitch =
                        srcImageInfo.width * pixelSize + rowPadding;
                } while ((srcImageInfo.rowPitch % pixelSize) != 0);
            }

            srcImageInfo.slicePitch =
                srcImageInfo.rowPitch * (srcImageInfo.height + slicePadding);
            log_info( "Testing %d x %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            if( gDebugTrace )
                log_info( "   at max size %d,%d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            if (test_copy_image_3D(context, queue, &srcImageInfo, &dstImageInfo,
                                   seed))
                return -1;
        }
    }
    else
    {
        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            size_t rowPadding = gEnablePitch ? 80 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                srcImageInfo.width =
                    (size_t)random_log_in_range(16, (int)maxWidth / 32, seed);
                srcImageInfo.height =
                    (size_t)random_log_in_range(16, (int)maxHeight / 32, seed);
                srcImageInfo.depth =
                    (size_t)random_log_in_range(16, (int)maxDepth / 32, seed);

                if (gTestMipmaps)
                {
                    srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
                        2,
                        (int)compute_max_mip_levels(srcImageInfo.width,
                                                    srcImageInfo.height,
                                                    srcImageInfo.depth),
                        seed);
                    srcImageInfo.rowPitch = srcImageInfo.width
                        * get_pixel_size(srcImageInfo.format);
                    srcImageInfo.slicePitch =
                        srcImageInfo.height * srcImageInfo.rowPitch;
                    size = compute_mipmapped_image_size(srcImageInfo);
                    size = size*4;
                }
                else
                {
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

                    srcImageInfo.slicePitch = srcImageInfo.rowPitch
                        * (srcImageInfo.height + slicePadding);

                    size = (cl_ulong)srcImageInfo.slicePitch
                        * (cl_ulong)srcImageInfo.depth * 4 * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info("   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n",
                         (int)srcImageInfo.width, (int)srcImageInfo.height,
                         (int)srcImageInfo.depth, (int)srcImageInfo.rowPitch,
                         (int)srcImageInfo.slicePitch, (int)maxWidth,
                         (int)maxHeight, (int)maxDepth);

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            int ret = test_copy_image_3D(context, queue, &srcImageInfo,
                                         &dstImageInfo, seed);
            if( ret )
                return -1;
        }
    }

    return 0;
}
