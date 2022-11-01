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
#include "../common.h"

extern int test_copy_image_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo,
                                   const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d );

static void set_image_dimensions( image_descriptor *imageInfo, size_t width, size_t height, size_t arraySize, size_t rowPadding, size_t slicePadding )
{
    size_t pixelSize = get_pixel_size( imageInfo->format );

    imageInfo->width = width;
    imageInfo->height = height;
    imageInfo->arraySize = arraySize;
    imageInfo->rowPitch = imageInfo->width * pixelSize + rowPadding;

    if (gEnablePitch)
    {
        do {
            rowPadding++;
            imageInfo->rowPitch = imageInfo->width * pixelSize + rowPadding;
        } while ((imageInfo->rowPitch % pixelSize) != 0);
    }

    if (arraySize == 0)
    {
        imageInfo->type = CL_MEM_OBJECT_IMAGE2D;
        imageInfo->slicePitch = 0;
    }
    else
    {
        imageInfo->type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        imageInfo->slicePitch = imageInfo->rowPitch * (imageInfo->height + slicePadding);
    }
}


int test_copy_image_size_2D_2D_array( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo, MTdata d )
{
    size_t sourcePos[ 4 ] = { 0 }, destPos[ 4 ] = { 0 }, regionSize[ 3 ];
    int ret = 0, retCode;

    image_descriptor *threeImage, *twoImage;

    if( srcImageInfo->arraySize > 0 )
    {
        threeImage = srcImageInfo;
        twoImage = dstImageInfo;
    }
    else
    {
        threeImage = dstImageInfo;
        twoImage = srcImageInfo;
    }

    size_t twoImage_lod = 0, twoImage_width_lod = twoImage->width, twoImage_row_pitch_lod;
    size_t twoImage_height_lod = twoImage->height;
    size_t threeImage_lod = 0, threeImage_width_lod = threeImage->width, threeImage_row_pitch_lod, threeImage_slice_pitch_lod;
    size_t threeImage_height_lod = threeImage->height;
    size_t width_lod, height_lod;
    size_t twoImage_max_mip_level,threeImage_max_mip_level;

    if( gTestMipmaps )
    {
        twoImage_max_mip_level = twoImage->num_mip_levels;
        threeImage_max_mip_level = threeImage->num_mip_levels;
        // Work at random mip levels
        twoImage_lod = (size_t)random_in_range( 0, twoImage_max_mip_level ? twoImage_max_mip_level - 1 : 0, d );
        threeImage_lod = (size_t)random_in_range( 0, threeImage_max_mip_level ? threeImage_max_mip_level - 1 : 0, d );
        twoImage_width_lod = ( twoImage->width >> twoImage_lod )? ( twoImage->width >> twoImage_lod ) : 1;
        threeImage_width_lod = ( threeImage->width >> threeImage_lod )? ( threeImage->width >> threeImage_lod ) : 1;
        twoImage_height_lod = ( twoImage->height >> twoImage_lod )? ( twoImage->height >> twoImage_lod ) : 1;
        threeImage_height_lod = ( threeImage->height >> threeImage_lod )? ( threeImage->height >> threeImage_lod ) : 1;
        twoImage_row_pitch_lod = twoImage_width_lod * get_pixel_size( twoImage->format );
        threeImage_row_pitch_lod = threeImage_width_lod * get_pixel_size( threeImage->format );
        threeImage_slice_pitch_lod = threeImage_height_lod * threeImage_row_pitch_lod;
    }
    width_lod  = ( twoImage_width_lod > threeImage_width_lod ) ? threeImage_width_lod : twoImage_width_lod;
    height_lod  = ( twoImage_height_lod > threeImage_height_lod ) ? threeImage_height_lod : twoImage_height_lod;

    // First, try just a full covering region
    sourcePos[ 0 ] = sourcePos[ 1 ] = sourcePos[ 2 ] = sourcePos[ 3 ] = 0;
    destPos[ 0 ] = destPos[ 1 ] = destPos[ 2 ] = destPos[ 3 ] = 0;
    regionSize[ 0 ] = width_lod;
    regionSize[ 1 ] = height_lod;
    regionSize[ 2 ] = 1;

    if( srcImageInfo->arraySize == 0 )
    {
        // 2D to 2D array
        destPos[ 2 ] = (size_t)random_in_range( 0, (int)dstImageInfo->arraySize - 1, d );
        if(gTestMipmaps)
        {
            sourcePos[ 2 ] = twoImage_lod;
            destPos[ 3 ] = threeImage_lod;
            regionSize[ 0 ] = width_lod;
            regionSize[ 1 ] = height_lod;
        }
    }
    else
    {
        // 2D array to 2D
        sourcePos[ 2 ] = (size_t)random_in_range( 0, (int)srcImageInfo->arraySize - 1, d );
        if(gTestMipmaps)
        {
            sourcePos[ 3 ] = threeImage_lod;
            destPos[ 2 ] = twoImage_lod;
            regionSize[ 0 ] = width_lod;
            regionSize[ 1 ] = height_lod;
        }
    }

    retCode = test_copy_image_generic( context, queue, srcImageInfo, dstImageInfo, sourcePos, destPos, regionSize, d );
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
            twoImage_lod = (size_t)random_in_range( 0, twoImage_max_mip_level ? twoImage_max_mip_level - 1 : 0, d );
            threeImage_lod = (size_t)random_in_range( 0, threeImage_max_mip_level ? threeImage_max_mip_level - 1 : 0, d );
            twoImage_width_lod = ( twoImage->width >> twoImage_lod )? ( twoImage->width >> twoImage_lod ) : 1;
            threeImage_width_lod = ( threeImage->width >> threeImage_lod )? ( threeImage->width >> threeImage_lod ) : 1;
            twoImage_height_lod = ( twoImage->height >> twoImage_lod )? ( twoImage->height >> twoImage_lod ) : 1;
            threeImage_height_lod = ( threeImage->height >> threeImage_lod )? ( threeImage->height >> threeImage_lod ) : 1;
            width_lod  = ( twoImage_width_lod > threeImage_width_lod ) ? threeImage_width_lod : twoImage_width_lod;
            height_lod  = ( twoImage_height_lod > threeImage_height_lod ) ? threeImage_height_lod : twoImage_height_lod;
        }
        // Pick a random size
        regionSize[ 0 ] = random_in_ranges( 8, srcImageInfo->width, dstImageInfo->width, d );
        regionSize[ 1 ] = random_in_ranges( 8, srcImageInfo->height, dstImageInfo->height, d );
        if( gTestMipmaps )
        {
            regionSize[ 0 ] = ( width_lod > 8 ) ? random_in_range( 8, width_lod, d ) : width_lod;
            regionSize[ 1 ] = ( height_lod > 8) ? random_in_range( 8, height_lod, d ): height_lod;
        }

        // Now pick positions within valid ranges
        sourcePos[ 0 ] = ( srcImageInfo->width > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( srcImageInfo->width - regionSize[ 0 ] - 1 ), d ) : 0;
        sourcePos[ 1 ] = ( srcImageInfo->height > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( srcImageInfo->height - regionSize[ 1 ] - 1 ), d ) : 0;
        sourcePos[ 2 ] = ( srcImageInfo->arraySize > 0 ) ? (size_t)random_in_range( 0, (int)( srcImageInfo->arraySize - 1 ), d ) : gTestMipmaps ? twoImage_lod : 0;
        if (gTestMipmaps)
        {
            if( srcImageInfo->arraySize > 0 )
            {
                sourcePos[ 0 ] = ( threeImage_width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( threeImage_width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
                sourcePos[ 1 ] = ( threeImage_height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( threeImage_height_lod - regionSize[ 1 ] - 1 ), d ) : 0;
                sourcePos[ 3 ] = threeImage_lod;
            }
            else
            {
                sourcePos[ 0 ] = ( twoImage_width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( twoImage_width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
                sourcePos[ 1 ] = ( twoImage_height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( twoImage_height_lod - regionSize[ 1 ] - 1 ), d ) : 0;

            }
        }

        destPos[ 0 ] = ( dstImageInfo->width > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( dstImageInfo->width - regionSize[ 0 ] - 1 ), d ) : 0;
        destPos[ 1 ] = ( dstImageInfo->height > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( dstImageInfo->height - regionSize[ 1 ] - 1 ), d ) : 0;
        destPos[ 2 ] = ( dstImageInfo->arraySize > 0 ) ? (size_t)random_in_range( 0, (int)( dstImageInfo->arraySize - 1 ), d ) : gTestMipmaps ? twoImage_lod : 0;
        if (gTestMipmaps)
        {
            if( dstImageInfo->arraySize > 0 )
            {
                destPos[ 0 ] = ( threeImage_width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( threeImage_width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
                destPos[ 1 ] = ( threeImage_height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( threeImage_height_lod - regionSize[ 1 ] - 1 ), d ) : 0;
                destPos[ 3 ] = threeImage_lod;
            }
            else
            {
                destPos[ 0 ] = ( twoImage_width_lod > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( twoImage_width_lod - regionSize[ 0 ] - 1 ), d ) : 0;
                destPos[ 1 ] = ( twoImage_height_lod > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( twoImage_height_lod - regionSize[ 1 ] - 1 ), d ) : 0;

            }
        }

        // Go for it!
        retCode = test_copy_image_generic( context, queue, srcImageInfo, dstImageInfo, sourcePos, destPos, regionSize, d );
        if( retCode < 0 )
            return retCode;
        else
            ret += retCode;
    }

    return ret;
}


int test_copy_image_set_2D_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse = false )
{
    size_t maxWidth, maxHeight, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor srcImageInfo = { 0 };
    image_descriptor dstImageInfo = { 0 };
    RandomSeed  seed( gRandomSeed );

    srcImageInfo.format = dstImageInfo.format = format;

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
      maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for( dstImageInfo.width = 4; dstImageInfo.width < 17; dstImageInfo.width++ )
        {
            for( dstImageInfo.height = 4; dstImageInfo.height < 13; dstImageInfo.height++ )
            {
                for( dstImageInfo.arraySize = 4; dstImageInfo.arraySize < 9; dstImageInfo.arraySize++ )
                {
                    size_t rowPadding = gEnablePitch ? 256 : 0;
                    size_t slicePadding = gEnablePitch ? 3 : 0;

                    set_image_dimensions( &dstImageInfo, dstImageInfo.width, dstImageInfo.height, dstImageInfo.arraySize, rowPadding, slicePadding );
                    set_image_dimensions( &srcImageInfo, dstImageInfo.width, dstImageInfo.height, 0, rowPadding, slicePadding );

                    if (gTestMipmaps)
                    {
                      srcImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(srcImageInfo.width, srcImageInfo.height, 0), seed);
                      srcImageInfo.type = CL_MEM_OBJECT_IMAGE2D;
                      dstImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(dstImageInfo.width, dstImageInfo.height, 0), seed);
                      dstImageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
                      srcImageInfo.rowPitch = srcImageInfo.width * get_pixel_size( srcImageInfo.format );
                      srcImageInfo.slicePitch = 0;
                      dstImageInfo.rowPitch = dstImageInfo.width * get_pixel_size( dstImageInfo.format );
                      dstImageInfo.slicePitch = dstImageInfo.rowPitch * dstImageInfo.height;
                    }

                    if( gDebugTrace )
                    {
                        if (reverse)
                            log_info( "   at size %d,%d,%d to %d,%d\n", (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize, (int)srcImageInfo.width, (int)srcImageInfo.height );
                        else
                            log_info( "   at size %d,%d to %d,%d,%d\n", (int)srcImageInfo.width, (int)srcImageInfo.height, (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize );
                    }
                    int ret;
                    if( reverse )
                        ret = test_copy_image_size_2D_2D_array( context, queue, &dstImageInfo, &srcImageInfo, seed );
                    else
                        ret = test_copy_image_size_2D_2D_array( context, queue, &srcImageInfo, &dstImageInfo, seed );
                    if( ret )
                        return -1;
                }
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numberOfSizes2DArray, numberOfSizes2D;
        size_t sizes2DArray[100][3], sizes2D[100][3];

        // Try to allocate a bit smaller images because we need the 2D ones as well for the copy.
        get_max_sizes(&numberOfSizes2DArray, 100, sizes2DArray, maxWidth, maxHeight, 1, maxArraySize, maxAllocSize/2, memSize/2, CL_MEM_OBJECT_IMAGE2D_ARRAY, dstImageInfo.format);
        get_max_sizes(&numberOfSizes2D, 100, sizes2D, maxWidth, maxHeight, 1, 1, maxAllocSize/2, memSize/2, CL_MEM_OBJECT_IMAGE2D, dstImageInfo.format);

        for( size_t i = 0; i < numberOfSizes2D; i++ )
        {
          for( size_t j = 0; j < numberOfSizes2DArray; j++ )
          {
            size_t rowPadding = gEnablePitch ? 256 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            set_image_dimensions( &dstImageInfo, sizes2DArray[ j ][ 0 ], sizes2DArray[ j ][ 1 ], sizes2DArray[ j ][ 2 ], rowPadding, slicePadding );
            set_image_dimensions( &srcImageInfo, sizes2D[ i ][ 0 ], sizes2D[ i ][ 1 ], 0, rowPadding, slicePadding );

            cl_ulong dstSize = get_image_size(&dstImageInfo);
            cl_ulong srcSize = get_image_size(&srcImageInfo);

            if (gTestMipmaps)
            {
              srcImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(srcImageInfo.width, srcImageInfo.height, 0), seed);
              srcImageInfo.type = CL_MEM_OBJECT_IMAGE2D;
              dstImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(dstImageInfo.width, dstImageInfo.height, 0), seed);
              dstImageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
              srcImageInfo.rowPitch = srcImageInfo.width * get_pixel_size( srcImageInfo.format );
              srcImageInfo.slicePitch = 0;
              dstImageInfo.rowPitch = dstImageInfo.width * get_pixel_size( dstImageInfo.format );
              dstImageInfo.slicePitch = dstImageInfo.rowPitch * dstImageInfo.height;
              dstSize = 4 * compute_mipmapped_image_size( dstImageInfo );
              srcSize = 4 * compute_mipmapped_image_size( srcImageInfo );
            }

            if( dstSize < maxAllocSize && dstSize < ( memSize / 3 ) && srcSize < maxAllocSize && srcSize < ( memSize / 3 ) )
            {
              if (reverse)
                log_info( "Testing %d x %d x %d to %d x %d\n", (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize, (int)srcImageInfo.width, (int)srcImageInfo.height );
              else
                log_info( "Testing %d x %d to %d x %d x %d\n", (int)srcImageInfo.width, (int)srcImageInfo.height, (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize );

              if( gDebugTrace )
              {
                if (reverse)
                  log_info( "   at max size %d,%d,%d to %d,%d\n", (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize, (int)srcImageInfo.width, (int)srcImageInfo.height );
                else
                  log_info( "   at max size %d,%d to %d,%d,%d\n", (int)srcImageInfo.width, (int)srcImageInfo.height, (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize );
              }
              int ret;
              if( reverse )
                ret = test_copy_image_size_2D_2D_array( context, queue, &dstImageInfo, &srcImageInfo, seed );
              else
                ret = test_copy_image_size_2D_2D_array( context, queue, &srcImageInfo, &dstImageInfo, seed );
              if( ret )
                return -1;
            }
            else
            {
              if (reverse)
                log_info("Not testing max size %d x %d x %d to %d x %d due to memory constraints.\n",
                (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize, (int)srcImageInfo.width, (int)srcImageInfo.height);
              else
                log_info("Not testing max size %d x %d to %d x %d x %d due to memory constraints.\n",
                (int)srcImageInfo.width, (int)srcImageInfo.height, (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize);
            }

          }
        }
    }
    else
    {
        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong srcSize, dstSize;
            size_t rowPadding = gEnablePitch ? 256 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                dstImageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                dstImageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );
                dstImageInfo.arraySize = (size_t)random_log_in_range( 16, (int)maxArraySize / 32, seed );
                srcImageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                srcImageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );

                if (gTestMipmaps)
                {
                    srcImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(srcImageInfo.width, srcImageInfo.height, 0), seed);
                    srcImageInfo.type = CL_MEM_OBJECT_IMAGE2D;
                    dstImageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(dstImageInfo.width, dstImageInfo.height, 0), seed);
                    dstImageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
                    srcImageInfo.rowPitch = srcImageInfo.width * get_pixel_size( srcImageInfo.format );
                    srcImageInfo.slicePitch = 0;
                    dstImageInfo.rowPitch = dstImageInfo.width * get_pixel_size( dstImageInfo.format );
                    dstImageInfo.slicePitch = dstImageInfo.rowPitch * dstImageInfo.height;
                    srcSize = 4 * compute_mipmapped_image_size( srcImageInfo );
                    dstSize = 4 * compute_mipmapped_image_size( dstImageInfo );
                }
                else
                {
                    set_image_dimensions( &srcImageInfo, srcImageInfo.width, srcImageInfo.height, 0, rowPadding, slicePadding );
                    set_image_dimensions( &dstImageInfo, dstImageInfo.width, dstImageInfo.height, dstImageInfo.arraySize, rowPadding, slicePadding );

                    srcSize = (cl_ulong)srcImageInfo.rowPitch * (cl_ulong)srcImageInfo.height * 4;
                    dstSize = (cl_ulong)dstImageInfo.slicePitch * (cl_ulong)dstImageInfo.arraySize * 4;
                }
            } while( srcSize > maxAllocSize || ( srcSize * 3 ) > memSize || dstSize > maxAllocSize || ( dstSize * 3 ) > memSize);

            if( gDebugTrace )
            {
                if (reverse)
                    log_info( "   at size %d,%d,%d to %d,%d\n", (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize, (int)srcImageInfo.width, (int)srcImageInfo.height );
                else
                    log_info( "   at size %d,%d to %d,%d,%d\n", (int)srcImageInfo.width, (int)srcImageInfo.height, (int)dstImageInfo.width, (int)dstImageInfo.height, (int)dstImageInfo.arraySize );
            }
            int ret;
            if( reverse )
                ret = test_copy_image_size_2D_2D_array( context, queue, &dstImageInfo, &srcImageInfo, seed );
            else
                ret = test_copy_image_size_2D_2D_array( context, queue, &srcImageInfo, &dstImageInfo, seed );
            if( ret )
                return -1;
        }
    }

    return 0;
}
