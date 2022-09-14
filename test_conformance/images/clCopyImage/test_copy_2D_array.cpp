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

int test_copy_image_2D_array( cl_context context, cl_command_queue queue, image_descriptor *imageInfo, MTdata d )
{
    size_t srcPos[] = { 0, 0, 0, 0}, dstPos[] = {0, 0, 0, 0};
    size_t region[] = { imageInfo->width, imageInfo->height, imageInfo->arraySize };

    size_t src_lod = 0, src_width_lod = imageInfo->width, src_height_lod = imageInfo->height;
    size_t dst_lod = 0, dst_width_lod = imageInfo->width, dst_height_lod = imageInfo->height;
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height;
    size_t max_mip_level;

    if( gTestMipmaps )
    {
        max_mip_level = imageInfo->num_mip_levels;
        // Work at a random mip level
        src_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
        dst_lod = (size_t)random_in_range( 0, max_mip_level ? max_mip_level - 1 : 0, d );
        src_width_lod = ( imageInfo->width >> src_lod )? ( imageInfo->width >> src_lod ) : 1;
        dst_width_lod = ( imageInfo->width >> dst_lod )? ( imageInfo->width >> dst_lod ) : 1;
        src_height_lod = ( imageInfo->height >> src_lod )? ( imageInfo->height >> src_lod ) : 1;
        dst_height_lod = ( imageInfo->height >> dst_lod )? ( imageInfo->height >> dst_lod ) : 1;
        width_lod  = ( src_width_lod > dst_width_lod ) ? dst_width_lod : src_width_lod;
        height_lod  = ( src_height_lod > dst_height_lod ) ? dst_height_lod : src_height_lod;

        region[ 0 ] = width_lod;
        region[ 1 ] = height_lod;
        srcPos[ 3 ] = src_lod;
        dstPos[ 3 ] = dst_lod;
}
    return test_copy_image_generic( context, queue, imageInfo, imageInfo, srcPos, dstPos, region, d );
}

int test_copy_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format )
{
    size_t maxWidth, maxHeight, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    imageInfo.format = format;
    imageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D array size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
      maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            size_t rowPadding = gEnablePitch ? 80 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;

            if (gTestMipmaps)
              imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

            if (gEnablePitch)
            {
                do {
                    rowPadding++;
                    imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
                } while ((imageInfo.rowPitch % pixelSize) != 0);
            }

            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + slicePadding);

                for( imageInfo.arraySize = 2; imageInfo.arraySize < 9; imageInfo.arraySize++ )
                {
                    if( gDebugTrace )
                        log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize );
                    int ret = test_copy_image_2D_array( context, queue, &imageInfo, seed );
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
        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, maxArraySize, maxAllocSize, memSize, imageInfo.type, imageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            size_t rowPadding = gEnablePitch ? 80 : 0;
            size_t slicePadding = gEnablePitch ? 3 : 0;

            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.arraySize = sizes[ idx ][ 2 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;

            if (gTestMipmaps)
              imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

            if (gEnablePitch)
            {
                do {
                    rowPadding++;
                    imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
                } while ((imageInfo.rowPitch % pixelSize) != 0);
            }

            imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + slicePadding);
            log_info( "Testing %d x %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            if( gDebugTrace )
                log_info( "   at max size %d,%d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            if( test_copy_image_2D_array( context, queue, &imageInfo, seed ) )
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
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                imageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );
                imageInfo.arraySize = (size_t)random_log_in_range( 16, (int)maxArraySize / 32, seed );

                if (gTestMipmaps)
                {
                    imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);
                    imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
                    imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                    size = compute_mipmapped_image_size( imageInfo );
                    size = size*4;
                }
                else
                {
                  imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
                  if (gEnablePitch)
                  {
                    do {
                      rowPadding++;
                      imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
                    } while ((imageInfo.rowPitch % pixelSize) != 0);
                  }

                  imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + slicePadding);

                  size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxArraySize );
            int ret = test_copy_image_2D_array( context, queue, &imageInfo,seed );
            if( ret )
                return -1;
        }
    }

    return 0;
}
