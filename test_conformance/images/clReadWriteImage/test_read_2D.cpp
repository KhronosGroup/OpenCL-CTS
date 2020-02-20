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

extern bool            gDebugTrace, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages, gTestMipmaps;
extern cl_filter_mode    gFilterModeToUse;
extern cl_addressing_mode    gAddressModeToUse;
extern uint64_t gRoundingStartValue;


int test_read_image_2D( cl_context context, cl_command_queue queue, image_descriptor *imageInfo, MTdata d )
{
    int error;

    clMemWrapper image;

    // Generate some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if( gDebugTrace )
    {
        log_info( " - Creating %s image %d by %d...\n", gTestMipmaps?"mipmapped":"", (int)imageInfo->width, (int)imageInfo->height );
        if( gTestMipmaps )
            log_info( " with %llu mip levels\n", (unsigned long long) imageInfo->num_mip_levels );
    }

    // Construct testing sources
    if(!gTestMipmaps)
    {
        image = create_image_2d( context, (cl_mem_flags)(CL_MEM_READ_ONLY), imageInfo->format, imageInfo->width, imageInfo->height, 0, NULL, &error );
        if( image == NULL )
        {
            log_error( "ERROR: Unable to create 2D image of size %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, IGetErrorString( error ) );
            return -1;
        }
    }
    else
    {
        cl_image_desc image_desc = {0};
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = imageInfo->width;
        image_desc.image_height = imageInfo->height;
        image_desc.num_mip_levels = imageInfo->num_mip_levels;

        image = clCreateImage( context, CL_MEM_READ_ONLY, imageInfo->format, &image_desc, NULL, &error);
        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create %d level mipmapped 2D image of size %d x %d (pitch %d ) (%s)",(int)imageInfo->num_mip_levels, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->rowPitch, IGetErrorString( error ) );
            return error;
        }
    }
    if( gDebugTrace )
        log_info( " - Writing image...\n" );

    size_t origin[ 3 ] = { 0, 0, 0 };
    size_t region[ 3 ] = { 0, 0, 1 };
    size_t fullImageSize;
    if( gTestMipmaps )
    {
        fullImageSize = (size_t)compute_mipmapped_image_size( *imageInfo );
    }
    else
    {
        fullImageSize = imageInfo->height * imageInfo->rowPitch;
    }
    BufferOwningPtr<char> resultValues(malloc(fullImageSize));
    size_t imgValMipLevelOffset = 0;

    for( size_t lod = 0; (gTestMipmaps && lod < imageInfo->num_mip_levels) || (!gTestMipmaps && lod < 1); lod++)
    {
        float lod_float = (float) lod;
        origin[2] = lod;
        size_t width_lod, height_lod, row_pitch_lod;

        width_lod = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        height_lod = (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
        row_pitch_lod = gTestMipmaps ? (width_lod * get_pixel_size( imageInfo->format )): imageInfo->rowPitch;

        region[0] = width_lod;
        region[1] = height_lod;

        if ( gDebugTrace && gTestMipmaps) {
            log_info(" - Working at mipLevel :%llu\n", (unsigned long long)lod);
        }
        error = clEnqueueWriteImage(queue, image, CL_FALSE,
                                    origin, region, ( gEnablePitch ? row_pitch_lod : 0 ), 0,
                                   (char*)imageValues + imgValMipLevelOffset, 0, NULL, NULL);
        if (error != CL_SUCCESS) {
            log_error( "ERROR: Unable to write to 2D image of size %d x %d \n", (int)width_lod, (int)height_lod );
            return -1;
        }

        // To verify, we just read the results right back and see whether they match the input
        if( gDebugTrace ) {
            log_info( " - Initing result array...\n" );
        }

        // Note: we read back without any pitch, to verify pitch actually WORKED
        size_t scanlineSize = width_lod * get_pixel_size( imageInfo->format );
        size_t imageSize = scanlineSize * height_lod;
        memset( resultValues, 0xff, imageSize );

        if( gDebugTrace )
            log_info( " - Reading results...\n" );

        error = clEnqueueReadImage( queue, image, CL_TRUE, origin, region, 0, 0, resultValues, 0, NULL, NULL );
        test_error( error, "Unable to read image values" );

        // Verify scanline by scanline, since the pitches are different
        char *sourcePtr = (char *)imageValues + imgValMipLevelOffset;
        char *destPtr = resultValues;

        for( size_t y = 0; y < height_lod; y++ )
        {
            if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
            {
                if(gTestMipmaps)
                {
                    log_error("At mip level %llu\n",(unsigned long long) lod);
                }
                log_error( "ERROR: Scanline %d did not verify for image size %d,%d pitch %d (extra %d bytes)\n", (int)y, (int)width_lod, (int)height_lod, (int)row_pitch_lod, (int)row_pitch_lod - (int)width_lod * (int)get_pixel_size( imageInfo->format ) );

                log_error( "First few values: \n" );
                log_error( " Input: " );
                uint32_t *s = (uint32_t *)sourcePtr;
                uint32_t *d = (uint32_t *)destPtr;
                for( int q = 0; q < 12; q++ )
                    log_error( "%08x ", s[ q ] );
                log_error( "\nOutput: " );
                for( int q = 0; q < 12; q++ )
                    log_error( "%08x ", d[ q ] );
                log_error( "\n" );

                int outX, outY;
                int offset = (int)get_pixel_size( imageInfo->format ) * (int)( width_lod - 16 );
                if( offset < 0 )
                    offset = 0;
                int foundCount = debug_find_vector_in_image( (char*)imageValues + imgValMipLevelOffset, imageInfo, destPtr + offset, get_pixel_size( imageInfo->format ), &outX, &outY, NULL );
                if( foundCount > 0 )
                {
                    int returnedOffset = ( (int)y * (int)width_lod + offset / (int)get_pixel_size( imageInfo->format ) ) - ( outY * (int)width_lod + outX );

                    if( memcmp( sourcePtr + returnedOffset * get_pixel_size( imageInfo->format ), destPtr, get_pixel_size( imageInfo->format ) * 8 ) == 0 )
                        log_error( "       Values appear to be offsetted by %d\n", returnedOffset );
                    else
                        log_error( "       Calculated offset is %d but unable to verify\n", returnedOffset );
                }
                else
                {
                    log_error( "      Unable to determine offset\n" );
                }
                return -1;
            }
            sourcePtr += row_pitch_lod;
            destPtr += scanlineSize;
        }
        imgValMipLevelOffset += width_lod * height_lod * get_pixel_size( imageInfo->format );
    }
    return 0;
}

int test_read_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format )
{
    size_t maxWidth, maxHeight;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed  seed( gRandomSeed );
    size_t pixelSize;

    imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
    imageInfo.format = format;
    imageInfo.depth = imageInfo.slicePitch = 0;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D size from device" );

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
                if (gTestMipmaps)
                    imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

                if( gDebugTrace )
                    log_info( "   at size %d,%d\n", (int)imageInfo.width, (int)imageInfo.height );

                int ret = test_read_image_2D( context, queue, &imageInfo, seed );
                if( ret )
                    return -1;
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE2D, imageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[idx][0];
            imageInfo.height = sizes[idx][1];
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            if (gTestMipmaps)
                imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

            log_info("Testing %d x %d\n", (int)imageInfo.width, (int)imageInfo.height);
            if( gDebugTrace )
                log_info( "   at max size %d,%d\n", (int)maxWidth, (int)maxHeight );
            if( test_read_image_2D( context, queue, &imageInfo, seed ) )
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
                if (gTestMipmaps)
                {
                    imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);
                    imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
                    size = compute_mipmapped_image_size( imageInfo );
                }
                else
                {
                    imageInfo.rowPitch = imageInfo.width * pixelSize;
                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.rowPitch += extraWidth * pixelSize;
                    }

                    size = (size_t)imageInfo.rowPitch * (size_t)imageInfo.height * 4;
                }
            } while(  size > maxAllocSize || ( size / 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d (row pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.rowPitch, (int)maxWidth, (int)maxHeight );
            int ret = test_read_image_2D( context, queue, &imageInfo, seed );
            if( ret )
                return -1;
        }
    }

    return 0;
}
