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

#define MAX_ERR 0.005f
#define MAX_HALF_LINEAR_ERR 0.3f

extern bool               gDebugTrace, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages;
extern cl_filter_mode     gFilterModeToUse;
extern cl_addressing_mode gAddressModeToUse;
extern uint64_t           gRoundingStartValue;

// Defined in test_fill_2D_3D.cpp
extern int test_fill_image_generic( cl_context context, cl_command_queue queue, image_descriptor *imageInfo,
                                    const size_t origin[], const size_t region[], ExplicitType outputType, MTdata d );


int test_fill_image_size_1D_array( cl_context context, cl_command_queue queue, image_descriptor *imageInfo, ExplicitType outputType, MTdata d )
{
    size_t origin[ 3 ], region[ 3 ];
    int ret = 0, retCode;

    // First, try just a full covering region fill
    origin[ 0 ] = origin[ 1 ] = origin[ 2 ] = 0;
    region[ 0 ] = imageInfo->width;
    region[ 1 ] = imageInfo->arraySize;
    region[ 2 ] = 1;

    retCode = test_fill_image_generic( context, queue, imageInfo, origin, region, outputType, d );
    if ( retCode < 0 )
        return retCode;
    else
        ret += retCode;

    // Now try a sampling of different random regions
    for ( int i = 0; i < 8; i++ )
    {
        // Pick a random size
        region[ 0 ] = ( imageInfo->width > 8 ) ? (size_t)random_in_range( 8, (int)imageInfo->width - 1, d ) : imageInfo->width;
        region[ 1 ] = ( imageInfo->arraySize > 8 ) ? (size_t)random_in_range( 8, (int)imageInfo->arraySize - 1, d ) : imageInfo->arraySize;

        // Now pick positions within valid ranges
        origin[ 0 ] = ( imageInfo->width > region[ 0 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->width - region[ 0 ] - 1 ), d ) : 0;
        origin[ 1 ] = ( imageInfo->arraySize > region[ 1 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->arraySize - region[ 1 ] - 1 ), d ) : 0;

        // Go for it!
        retCode = test_fill_image_generic( context, queue, imageInfo, origin, region, outputType, d );
        if ( retCode < 0 )
            return retCode;
        else
            ret += retCode;
    }

    return ret;
}


int test_fill_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType )
{
    size_t maxWidth, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed(gRandomSeed);
    const size_t rowPadding_default = 48;
    size_t rowPadding = gEnablePitch ? rowPadding_default : 0;
    size_t pixelSize;

    memset(&imageInfo, 0x0, sizeof(image_descriptor));
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    imageInfo.format = format;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 1D array size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
    }

    if ( gTestSmallImages )
    {
        for ( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
              rowPadding = rowPadding_default;
              do {
                rowPadding++;
                imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
              } while ((imageInfo.rowPitch % pixelSize) != 0);
            }

            imageInfo.slicePitch = imageInfo.rowPitch;
            for ( imageInfo.arraySize = 2; imageInfo.arraySize < 9; imageInfo.arraySize++ )
            {
                if ( gDebugTrace )
                    log_info( "   at size %d,%d\n", (int)imageInfo.width, (int)imageInfo.arraySize );

                int ret = test_fill_image_size_1D_array( context, queue, &imageInfo, outputType, seed );
                if ( ret )
                    return -1;
            }
        }
    }
    else if ( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, maxArraySize, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE1D_ARRAY, imageInfo.format);

        for ( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.arraySize = sizes[ idx ][ 2 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;

            if (gEnablePitch)
            {
              rowPadding = rowPadding_default;
              do {
                rowPadding++;
                imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
              } while ((imageInfo.rowPitch % pixelSize) != 0);
            }

            imageInfo.slicePitch = imageInfo.rowPitch;
            log_info( "Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 2 ] );
            if ( gDebugTrace )
                log_info( "   at max size %d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 2 ] );
            if ( test_fill_image_size_1D_array( context, queue, &imageInfo, outputType, seed ) )
                return -1;
        }
    }
    else
    {
        for ( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                imageInfo.arraySize = (size_t)random_log_in_range( 16, (int)maxArraySize / 32, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;

                if (gEnablePitch)
                {
                  rowPadding = rowPadding_default;
                  do {
                    rowPadding++;
                    imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
                  } while ((imageInfo.rowPitch % pixelSize) != 0);
                }

                imageInfo.slicePitch = imageInfo.rowPitch;

                size = (size_t)imageInfo.rowPitch * (size_t)imageInfo.arraySize * 4;
            } while (  size > maxAllocSize || ( size * 3 ) > memSize );

            if ( gDebugTrace )
                log_info( "   at size %d,%d (row pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)maxWidth, (int)maxArraySize );
            int ret = test_fill_image_size_1D_array( context, queue, &imageInfo, outputType, seed );
            if ( ret )
                return -1;
        }
    }

    return 0;
}
