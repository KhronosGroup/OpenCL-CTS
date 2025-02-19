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

extern int test_copy_image_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo,
                                   const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d );

int test_copy_image_size_2D(cl_context context, cl_command_queue queue,
                            image_descriptor *srcImageInfo,
                            image_descriptor *dstImageInfo, MTdata d)
{
    size_t sourcePos[ 3 ], destPos[ 3 ], regionSize[ 3 ];
    int ret = 0, retCode;
    size_t src_lod = 0, src_width_lod = srcImageInfo->width, src_row_pitch_lod;
    size_t src_height_lod = srcImageInfo->height;
    size_t dst_lod = 0, dst_width_lod = dstImageInfo->width, dst_row_pitch_lod;
    size_t dst_height_lod = dstImageInfo->height;
    size_t width_lod = srcImageInfo->width, height_lod = srcImageInfo->height;
    size_t max_mip_level = 0;

    if( gTestMipmaps )
    {
        max_mip_level = srcImageInfo->num_mip_levels;
        // Work at a random mip level
        src_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
        dst_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
        src_width_lod = (srcImageInfo->width >> src_lod)
            ? (srcImageInfo->width >> src_lod)
            : 1;
        dst_width_lod = (dstImageInfo->width >> dst_lod)
            ? (dstImageInfo->width >> dst_lod)
            : 1;
        src_height_lod = (srcImageInfo->height >> src_lod)
            ? (srcImageInfo->height >> src_lod)
            : 1;
        dst_height_lod = (dstImageInfo->height >> dst_lod)
            ? (dstImageInfo->height >> dst_lod)
            : 1;
        width_lod  = ( src_width_lod > dst_width_lod ) ? dst_width_lod : src_width_lod;
        height_lod  = ( src_height_lod > dst_height_lod ) ? dst_height_lod : src_height_lod;
        src_row_pitch_lod =
            src_width_lod * get_pixel_size(srcImageInfo->format);
        dst_row_pitch_lod =
            dst_width_lod * get_pixel_size(srcImageInfo->format);
    }

    // First, try just a full covering region
    sourcePos[ 0 ] = sourcePos[ 1 ] = sourcePos[ 2 ] = 0;
    destPos[ 0 ] = destPos[ 1 ] = destPos[ 2 ] = 0;
    regionSize[0] = srcImageInfo->width;
    regionSize[1] = srcImageInfo->height;
    regionSize[ 2 ] = 1;

    if(gTestMipmaps)
    {
        sourcePos[ 2 ] = src_lod;
        destPos[ 2 ] = dst_lod;
        regionSize[ 0 ] = width_lod;
        regionSize[ 1 ] = height_lod;
    }

    retCode =
        test_copy_image_generic(context, queue, srcImageInfo, dstImageInfo,
                                sourcePos, destPos, regionSize, d);
    if( retCode < 0 )
        return retCode;
    else
        ret += retCode;

    // Now try a sampling of different random regions
    for( int i = 0; i < 8; i++ )
    {
        if( gTestMipmaps )
        {
            // Work at a random mip level
            src_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
            dst_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
            src_width_lod = (srcImageInfo->width >> src_lod)
                ? (srcImageInfo->width >> src_lod)
                : 1;
            dst_width_lod = (dstImageInfo->width >> dst_lod)
                ? (dstImageInfo->width >> dst_lod)
                : 1;
            src_height_lod = (srcImageInfo->height >> src_lod)
                ? (srcImageInfo->height >> src_lod)
                : 1;
            dst_height_lod = (dstImageInfo->height >> dst_lod)
                ? (dstImageInfo->height >> dst_lod)
                : 1;
            width_lod  = ( src_width_lod > dst_width_lod ) ? dst_width_lod : src_width_lod;
            height_lod  = ( src_height_lod > dst_height_lod ) ? dst_height_lod : src_height_lod;
            sourcePos[ 2 ] = src_lod;
            destPos[ 2 ] = dst_lod;
        }
        // Pick a random size
        regionSize[ 0 ] = ( width_lod > 8 ) ? (size_t)random_in_range( 8, (int)width_lod - 1, d ) : width_lod;
        regionSize[ 1 ] = ( height_lod > 8 ) ? (size_t)random_in_range( 8, (int)height_lod - 1, d ) : height_lod;

        // Now pick positions within valid ranges
        sourcePos[ 0 ] = ( width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
        sourcePos[ 1 ] = ( height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( height_lod - regionSize[ 1 ] - 1 ), d ) : 0;

        destPos[ 0 ] = ( width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
        destPos[ 1 ] = ( height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( height_lod - regionSize[ 1 ] - 1 ), d ) : 0;

        // Go for it!
        retCode =
            test_copy_image_generic(context, queue, srcImageInfo, dstImageInfo,
                                    sourcePos, destPos, regionSize, d);
        if( retCode < 0 )
            return retCode;
        else
            ret += retCode;
    }

    return ret;
}

int test_copy_image_set_2D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_mem_flags src_flags,
                           cl_mem_object_type src_type, cl_mem_flags dst_flags,
                           cl_mem_object_type dst_type, cl_image_format *format)
{
    assert(dst_type == src_type); // This test expects to copy 2D -> 2D images
    size_t maxWidth, maxHeight;
    cl_ulong maxAllocSize, memSize;
    image_descriptor srcImageInfo = { 0 };
    image_descriptor dstImageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    size_t pixelSize;

    srcImageInfo.format = format;
    srcImageInfo.type = src_type;
    srcImageInfo.mem_flags = src_flags;
    pixelSize = get_pixel_size(srcImageInfo.format);

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
        memSize = (cl_ulong)SIZE_MAX;
        maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for (srcImageInfo.width = 1; srcImageInfo.width < 13;
             srcImageInfo.width++)
        {
      size_t rowPadding = gEnablePitch ? 48 : 0;

      srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

      if (gTestMipmaps)
          srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
              2,
              (int)compute_max_mip_levels(srcImageInfo.width,
                                          srcImageInfo.height, 0),
              seed);

      if (gEnablePitch)
      {
          do
          {
              rowPadding++;
              srcImageInfo.rowPitch =
                  srcImageInfo.width * pixelSize + rowPadding;
          } while ((srcImageInfo.rowPitch % pixelSize) != 0);
      }

      for (srcImageInfo.height = 1; srcImageInfo.height < 9;
           srcImageInfo.height++)
      {
          if (gDebugTrace)
              log_info("   at size %d,%d\n", (int)srcImageInfo.width,
                       (int)srcImageInfo.height);

          dstImageInfo = srcImageInfo;
          dstImageInfo.mem_flags = dst_flags;
          int ret = test_copy_image_size_2D(context, queue, &srcImageInfo,
                                            &dstImageInfo, seed);
          if (ret) return -1;
      }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, 1,
                      maxAllocSize, memSize, src_type, srcImageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
      size_t rowPadding = gEnablePitch ? 48 : 0;

      srcImageInfo.width = sizes[idx][0];
      srcImageInfo.height = sizes[idx][1];
      srcImageInfo.rowPitch = srcImageInfo.width * pixelSize + rowPadding;

      if (gTestMipmaps)
          srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
              2,
              (int)compute_max_mip_levels(srcImageInfo.width,
                                          srcImageInfo.height, 0),
              seed);

      if (gEnablePitch)
      {
          do
          {
              rowPadding++;
              srcImageInfo.rowPitch =
                  srcImageInfo.width * pixelSize + rowPadding;
          } while ((srcImageInfo.rowPitch % pixelSize) != 0);
      }

            log_info( "Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
            if( gDebugTrace )
                log_info( "   at max size %d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            if (test_copy_image_size_2D(context, queue, &srcImageInfo,
                                        &dstImageInfo, seed))
                return -1;
        }
    }
    else
    {
        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
      size_t rowPadding = gEnablePitch ? 48 : 0;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                srcImageInfo.width =
                    (size_t)random_log_in_range(16, (int)maxWidth / 32, seed);
                srcImageInfo.height =
                    (size_t)random_log_in_range(16, (int)maxHeight / 32, seed);

                if (gTestMipmaps)
                {
                    srcImageInfo.num_mip_levels = (cl_uint)random_log_in_range(
                        2,
                        (int)compute_max_mip_levels(srcImageInfo.width,
                                                    srcImageInfo.height, 0),
                        seed);
                    srcImageInfo.rowPitch = srcImageInfo.width
                        * get_pixel_size(srcImageInfo.format);
                    size = compute_mipmapped_image_size(srcImageInfo);
                    size = size * 4;
                }
        else
        {
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

            size =
                (size_t)srcImageInfo.rowPitch * (size_t)srcImageInfo.height * 4;
        }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info("   at size %d,%d (row pitch %d) out of %d,%d\n",
                         (int)srcImageInfo.width, (int)srcImageInfo.height,
                         (int)srcImageInfo.rowPitch, (int)maxWidth,
                         (int)maxHeight);

            dstImageInfo = srcImageInfo;
            dstImageInfo.mem_flags = dst_flags;
            int ret = test_copy_image_size_2D(context, queue, &srcImageInfo,
                                              &dstImageInfo, seed);
            if( ret )
                return -1;
        }
    }

    return 0;
}
