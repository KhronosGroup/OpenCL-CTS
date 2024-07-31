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

int test_read_image_2D_array(cl_context context, cl_command_queue queue,
                             image_descriptor *imageInfo, MTdata d,
                             cl_mem_flags flags)
{
    int error;

    clMemWrapper image;

    // Create some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if( gDebugTrace )
    {
        log_info( " - Creating %s image %d by %d by %d...\n", gTestMipmaps?"mipmapped":"", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );
        if( gTestMipmaps )
            log_info( " with %llu mip levels\n", (unsigned long long) imageInfo->num_mip_levels );
    }

    // Construct testing sources
    if(!gTestMipmaps)
    {
        image = create_image_2d_array(context, flags, imageInfo->format,
                                      imageInfo->width, imageInfo->height,
                                      imageInfo->arraySize, 0, 0, NULL, &error);
        if( image == NULL )
        {
            log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, IGetErrorString( error ) );
            return -1;
        }
    }
    else
    {
        cl_image_desc image_desc = {0};
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
        image_desc.image_width = imageInfo->width;
        image_desc.image_height = imageInfo->height;
        image_desc.image_array_size = imageInfo->arraySize;
        image_desc.num_mip_levels = imageInfo->num_mip_levels;

        image = clCreateImage(context, flags, imageInfo->format, &image_desc,
                              NULL, &error);
        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create %d level mipmapped 3D image of size %d x %d x %d (pitch %d, %d ) (%s)",(int)imageInfo->num_mip_levels, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
            return error;
        }
    }

    if( gDebugTrace )
        log_info( " - Writing image...\n" );

    size_t origin[ 4 ] = { 0, 0, 0, 0 };
    size_t region[ 3 ] = { 0, 0, 0 };
    size_t fullImageSize;
    if( gTestMipmaps )
    {
        fullImageSize = (size_t)compute_mipmapped_image_size( *imageInfo );
    }
    else
    {
        fullImageSize = imageInfo->arraySize * imageInfo->slicePitch;
    }
    BufferOwningPtr<char> resultValues(malloc(fullImageSize));
    size_t imgValMipLevelOffset = 0;

    for(size_t lod = 0; (gTestMipmaps && lod < imageInfo->num_mip_levels) || (!gTestMipmaps && lod < 1); lod++)
    {
        origin[3] = lod;
        size_t width_lod, height_lod, row_pitch_lod, slice_pitch_lod;

        width_lod = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;
        height_lod = (imageInfo->height >> lod) ? (imageInfo->height >> lod) : 1;
        row_pitch_lod = gTestMipmaps ? (width_lod * get_pixel_size( imageInfo->format )): imageInfo->rowPitch;
        slice_pitch_lod = gTestMipmaps ? (row_pitch_lod * height_lod): imageInfo->slicePitch;
        region[0] = width_lod;
        region[1] = height_lod;
        region[2] = imageInfo->arraySize;

        if ( gDebugTrace && gTestMipmaps) {
            log_info(" - Working at mipLevel :%llu\n", (unsigned long long)lod);
        }

        error = clEnqueueWriteImage(queue, image, CL_FALSE,
                                    origin, region, ( gEnablePitch ? row_pitch_lod : 0 ), ( gEnablePitch ? slice_pitch_lod : 0 ),
                                    (char*)imageValues + imgValMipLevelOffset, 0, NULL, NULL);
        if (error != CL_SUCCESS) {
            log_error( "ERROR: Unable to write to 2D image array of size %d x %d x %d\n", (int)width_lod, (int)height_lod, (int)imageInfo->arraySize );
            return -1;
        }

        // To verify, we just read the results right back and see whether they match the input
        if( gDebugTrace )
            log_info( " - Initing result array...\n" );

        // Note: we read back without any pitch, to verify pitch actually WORKED
        size_t scanlineSize = width_lod * get_pixel_size( imageInfo->format );
        size_t pageSize = scanlineSize * height_lod;
        size_t imageSize = pageSize * imageInfo->arraySize;
        memset( resultValues, 0xff, imageSize );

        if( gDebugTrace )
            log_info( " - Reading results...\n" );

        error = clEnqueueReadImage( queue, image, CL_TRUE, origin, region, 0, 0, resultValues, 0, NULL, NULL );
        test_error( error, "Unable to read image values" );

        // Verify scanline by scanline, since the pitches are different
        char *sourcePtr = (char *)imageValues + imgValMipLevelOffset;
        char *destPtr = resultValues;

        for( size_t z = 0; z < imageInfo->arraySize; z++ )
        {
            for( size_t y = 0; y < height_lod; y++ )
            {
                if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
                {
                    log_error( "ERROR: Scanline %d,%d did not verify for image size %d,%d,%d pitch %d,%d\n", (int)y, (int)z, (int)width_lod, (int)height_lod, (int)imageInfo->arraySize, (int)row_pitch_lod, (int)slice_pitch_lod );
                    return -1;
                }
                sourcePtr += row_pitch_lod;
                destPtr += scanlineSize;
            }
            sourcePtr += slice_pitch_lod - ( row_pitch_lod * height_lod );
            destPtr += pageSize - scanlineSize * height_lod;
        }
        imgValMipLevelOffset += width_lod * height_lod * imageInfo->arraySize * get_pixel_size( imageInfo->format );
    }
    return 0;
}

int test_read_image_set_2D_array(cl_device_id device, cl_context context,
                                 cl_command_queue queue,
                                 cl_image_format *format, cl_mem_flags flags)
{
    size_t maxWidth, maxHeight, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    imageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    imageInfo.format = format;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 3D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
        memSize = (cl_ulong)SIZE_MAX;
        maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                for( imageInfo.arraySize = 2; imageInfo.arraySize < 9; imageInfo.arraySize++ )
                {
                    if (gTestMipmaps)
                        imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

                    if( gDebugTrace )
                        log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize );
                    int ret = test_read_image_2D_array(context, queue,
                                                       &imageInfo, seed, flags);
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

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, maxArraySize, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE2D_ARRAY, imageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            // Try a specific set of maximum sizes
            imageInfo.width = sizes[idx][0];
            imageInfo.height = sizes[idx][1];
            imageInfo.arraySize = sizes[idx][2];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;

            if (gTestMipmaps)
                imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);

            log_info("Testing %d x %d x %d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize);
            if (test_read_image_2D_array(context, queue, &imageInfo, seed,
                                         flags))
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
                imageInfo.arraySize = (size_t)random_log_in_range( 16, (int)maxArraySize / 32, seed );

                if (gTestMipmaps)
                {
                    imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.height, 0), seed);
                    imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
                    imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                    size = compute_mipmapped_image_size( imageInfo ) * 4;
                }
                else
                {
                    imageInfo.rowPitch = imageInfo.width * pixelSize;
                    imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;

                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.rowPitch += extraWidth * pixelSize;

                        size_t extraHeight = (int)random_log_in_range( 0, 8, seed );
                        imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + extraHeight);
                    }

                    size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxArraySize );
            int ret = test_read_image_2D_array(context, queue, &imageInfo, seed,
                                               flags);
            if( ret )
                return -1;
        }
    }

    return 0;
}
