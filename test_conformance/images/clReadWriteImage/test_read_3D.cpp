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

extern bool            gDebugTrace, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages, gTestRounding;
extern cl_filter_mode    gFilterModeToUse;
extern cl_addressing_mode    gAddressModeToUse;
extern cl_command_queue queue;
extern cl_context context;

int test_read_image_3D( cl_device_id device, image_descriptor *imageInfo, MTdata d )
{
    int error;

    clMemWrapper image;

    // Create some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if( gDebugTrace )
        log_info( " - Creating image %d by %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth );

    // Construct testing sources
    image = create_image_3d( context, (cl_mem_flags)(CL_MEM_READ_ONLY), imageInfo->format, imageInfo->width, imageInfo->height, imageInfo->depth, 0, 0, NULL, &error );
    if( image == NULL )
    {
        log_error( "ERROR: Unable to create 2D image of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, IGetErrorString( error ) );
        return -1;
    }

    if( gDebugTrace )
        log_info( " - Writing image...\n" );

    size_t origin[ 3 ] = { 0, 0, 0 };
    size_t region[ 3 ] = { imageInfo->width, imageInfo->height, imageInfo->depth };

    error = clEnqueueWriteImage(queue, image, CL_TRUE,
                                origin, region, ( gEnablePitch ? imageInfo->rowPitch : 0 ), ( gEnablePitch ? imageInfo->slicePitch : 0 ),
                                imageValues, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
        log_error( "ERROR: Unable to write to 3D image of size %d x %d x %d\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth );
        return -1;
    }

    // To verify, we just read the results right back and see whether they match the input
    if( gDebugTrace )
        log_info( " - Initing result array...\n" );

    // Note: we read back without any pitch, to verify pitch actually WORKED
    size_t scanlineSize = imageInfo->width * get_pixel_size( imageInfo->format );
    size_t pageSize = scanlineSize * imageInfo->height;
    size_t imageSize = pageSize * imageInfo->depth;
    BufferOwningPtr<char> resultValues(malloc(imageSize));
    memset( resultValues, 0xff, imageSize );

    if( gDebugTrace )
        log_info( " - Reading results...\n" );

    error = clEnqueueReadImage( queue, image, CL_TRUE, origin, region, 0, 0, resultValues, 0, NULL, NULL );
    test_error( error, "Unable to read image values" );

    // Verify scanline by scanline, since the pitches are different
    char *sourcePtr = (char *)(void *)imageValues;
    char *destPtr = resultValues;

    for( size_t z = 0; z < imageInfo->depth; z++ )
    {
        for( size_t y = 0; y < imageInfo->height; y++ )
        {
            if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
            {
                log_error( "ERROR: Scanline %d,%d did not verify for image size %d,%d,%d pitch %d,%d\n", (int)y, (int)z, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch );
                return -1;
            }
            sourcePtr += imageInfo->rowPitch;
            destPtr += scanlineSize;
        }
        sourcePtr += imageInfo->slicePitch - ( imageInfo->rowPitch * imageInfo->height );
        destPtr += pageSize - scanlineSize * imageInfo->height;
    }

    return 0;
}

int test_read_image_set_3D( cl_device_id device, cl_image_format *format )
{
    size_t maxWidth, maxHeight, maxDepth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo;
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    imageInfo.type = CL_MEM_OBJECT_IMAGE3D;
    imageInfo.format = format;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof( maxDepth ), &maxDepth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 3D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
        memSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                for( imageInfo.depth = 2; imageInfo.depth < 9; imageInfo.depth++ )
                {
                    if( gDebugTrace )
                        log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth );
                    int ret = test_read_image_3D( device, &imageInfo, seed );
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

    get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, maxDepth, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE3D, imageInfo.format);

    for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
      // Try a specific set of maximum sizes
      imageInfo.width = sizes[idx][0];
      imageInfo.height = sizes[idx][1];
      imageInfo.depth = sizes[idx][2];
      imageInfo.rowPitch = imageInfo.width * pixelSize;
      imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
      log_info("Testing %d x %d x %d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth);
      if( test_read_image_3D( device, &imageInfo, seed ) )
        return -1;
    }
  }
    else
    {
        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                imageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );
                imageInfo.depth = (size_t)random_log_in_range( 16, (int)maxDepth / 32, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;

                if( gEnablePitch )
                {
                    size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                    imageInfo.rowPitch += extraWidth * pixelSize;

                    size_t extraHeight = (int)random_log_in_range( 0, 8, seed );
                    imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + extraHeight);
                }

                size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.depth * 4 * 4;
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxDepth );
            int ret = test_read_image_3D( device, &imageInfo, seed );
            if( ret )
                return -1;
        }
    }

    return 0;
}
