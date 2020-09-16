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

extern bool            gDebugTrace, gDisableOffsets, gTestSmallImages, gTestMaxImages, gEnablePitch, gTestMipmaps;
extern cl_filter_mode    gFilterModeToUse;
extern cl_addressing_mode    gAddressModeToUse;
extern uint64_t gRoundingStartValue;

static void CL_CALLBACK free_pitch_buffer( cl_mem image, void *buf )
{
    free( buf );
}

cl_mem create_image( cl_context context, cl_command_queue queue, BufferOwningPtr<char>& data, image_descriptor *imageInfo, int *error )
{
    cl_mem img;
    cl_image_desc imageDesc;
    cl_mem_flags mem_flags = CL_MEM_READ_ONLY;
    void *host_ptr = NULL;

    memset(&imageDesc, 0x0, sizeof(cl_image_desc));
    imageDesc.image_type = imageInfo->type;
    imageDesc.image_width = imageInfo->width;
    imageDesc.image_height = imageInfo->height;
    imageDesc.image_depth = imageInfo->depth;
    imageDesc.image_array_size = imageInfo->arraySize;
    imageDesc.image_row_pitch = gEnablePitch ? imageInfo->rowPitch : 0;
    imageDesc.image_slice_pitch = gEnablePitch ? imageInfo->slicePitch : 0;
    imageDesc.num_mip_levels = gTestMipmaps ? imageInfo->num_mip_levels : 0;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            if ( gDebugTrace )
                log_info( " - Creating 1D image %d ...\n", (int)imageInfo->width );
            if ( gEnablePitch )
                host_ptr = malloc( imageInfo->rowPitch );
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            if ( gDebugTrace )
                log_info( " - Creating 2D image %d by %d ...\n", (int)imageInfo->width, (int)imageInfo->height );
            if ( gEnablePitch )
                host_ptr = malloc( imageInfo->height * imageInfo->rowPitch );
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            if ( gDebugTrace )
                log_info( " - Creating 3D image %d by %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth );
            if ( gEnablePitch )
                host_ptr = malloc( imageInfo->depth * imageInfo->slicePitch );
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            if ( gDebugTrace )
                log_info( " - Creating 1D image array %d by %d...\n", (int)imageInfo->width, (int)imageInfo->arraySize );
            if ( gEnablePitch )
                host_ptr = malloc( imageInfo->arraySize * imageInfo->slicePitch );
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            if ( gDebugTrace )
                log_info( " - Creating 2D image array %d by %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );
            if ( gEnablePitch )
                host_ptr = malloc( imageInfo->arraySize * imageInfo->slicePitch );
            break;
    }

    if ( gDebugTrace && gTestMipmaps )
        log_info(" - with %llu mip levels\n", (unsigned long long) imageInfo->num_mip_levels);

    if (gEnablePitch)
    {
        if ( NULL == host_ptr )
        {
            log_error( "ERROR: Unable to create backing store for pitched 3D image. %ld bytes\n",  imageInfo->depth * imageInfo->slicePitch );
            return NULL;
        }
        mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
    }

    img = clCreateImage(context, mem_flags, imageInfo->format, &imageDesc, host_ptr, error);

    if (gEnablePitch)
    {
        if ( *error == CL_SUCCESS )
        {
            int callbackError = clSetMemObjectDestructorCallback( img, free_pitch_buffer, host_ptr );
            if ( CL_SUCCESS != callbackError )
            {
                free( host_ptr );
                log_error( "ERROR: Unable to attach destructor callback to pitched 3D image. Err: %d\n", callbackError );
                clReleaseMemObject( img );
                return NULL;
            }
        }
        else
            free(host_ptr);
    }

    if ( *error != CL_SUCCESS )
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                log_error( "ERROR: Unable to create 1D image of size %d (%s)", (int)imageInfo->width, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_error( "ERROR: Unable to create 2D image of size %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                log_error( "ERROR: Unable to create 3D image of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_error( "ERROR: Unable to create 1D image array of size %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->arraySize, IGetErrorString( *error ) );
                break;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, IGetErrorString( *error ) );
                break;
        }
        log_error("ERROR: and %llu mip levels\n", (unsigned long long) imageInfo->num_mip_levels);
        return NULL;
    }

    // Copy the specified data to the image via a Map operation.
    size_t mappedRow, mappedSlice;
    size_t width = imageInfo->width;
    size_t height = 1;
    size_t depth = 1;
    size_t row_pitch_lod, slice_pitch_lod;
    row_pitch_lod = imageInfo->rowPitch;
    slice_pitch_lod = imageInfo->slicePitch;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            height = imageInfo->arraySize;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE1D:
            height = depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            height = imageInfo->height;
            depth = 1;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            height = imageInfo->height;
            depth = imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            height = imageInfo->height;
            depth = imageInfo->depth;
            break;
    }

    size_t origin[ 4 ] = { 0, 0, 0, 0 };
    size_t region[ 3 ] = { imageInfo->width, height, depth };

    for ( size_t lod = 0; (gTestMipmaps && (lod < imageInfo->num_mip_levels)) || (!gTestMipmaps && (lod < 1)); lod++)
    {
        // Map the appropriate miplevel to copy the specified data.
        if(gTestMipmaps)
        {
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                    origin[ 3 ] = lod;
                    break;
                case CL_MEM_OBJECT_IMAGE2D:
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                    origin[ 2 ] =  lod;
                    break;
                case CL_MEM_OBJECT_IMAGE1D:
                    origin[ 1 ] = lod;
                    break;
            }

            //Adjust image dimensions as per miplevel
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                    depth = ( imageInfo->depth >> lod ) ? (imageInfo->depth >> lod) : 1;
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D:
                    height = ( imageInfo->height >> lod ) ? (imageInfo->height >> lod) : 1;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE1D:
                    width = ( imageInfo->width >> lod ) ? (imageInfo->width >> lod) : 1;
            }
            row_pitch_lod = width * get_pixel_size(imageInfo->format);
            slice_pitch_lod = row_pitch_lod * height;
            region[0] = width;
            region[1] = height;
            region[2] = depth;
        }

        void* mapped = (char*)clEnqueueMapImage(queue, img, CL_TRUE, CL_MAP_WRITE, origin, region, &mappedRow, &mappedSlice, 0, NULL, NULL, error);
        if (*error != CL_SUCCESS)
        {
            log_error( "ERROR: Unable to map image for writing: %s\n", IGetErrorString( *error ) );
            return NULL;
        }
        size_t mappedSlicePad = mappedSlice - (mappedRow * height);

        // Copy the image.
        size_t scanlineSize = row_pitch_lod;
        size_t sliceSize = slice_pitch_lod - scanlineSize * height;
        size_t imageSize = scanlineSize * height * depth;
        size_t data_lod_offset = 0;
        if( gTestMipmaps )
            data_lod_offset = compute_mip_level_offset(imageInfo, lod);

        char* src = (char*)data + data_lod_offset;
        char* dst = (char*)mapped;

        if ((mappedRow == scanlineSize) && (mappedSlicePad==0 || (imageInfo->depth==0 && imageInfo->arraySize==0))) {
            // Copy the whole image.
            memcpy( dst, src, imageSize );
        }
        else {
            // Else copy one scan line at a time.
            size_t dstPitch2D = 0;
            switch (imageInfo->type)
            {
                case CL_MEM_OBJECT_IMAGE3D:
                case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                case CL_MEM_OBJECT_IMAGE2D:
                    dstPitch2D = mappedRow;
                    break;
                case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                case CL_MEM_OBJECT_IMAGE1D:
                    dstPitch2D = mappedSlice;
                    break;
            }
            for ( size_t z = 0; z < depth; z++ )
            {
                for ( size_t y = 0; y < height; y++ )
                {
                    memcpy( dst, src, scanlineSize );
                    dst += dstPitch2D;
                    src += scanlineSize;
                }

                // mappedSlicePad is incorrect for 2D images here, but we will exit the z loop before this is a problem.
                dst += mappedSlicePad;
                src += sliceSize;
            }
        }

        // Unmap the image.
        *error = clEnqueueUnmapMemObject(queue, img, mapped, 0, NULL, NULL);
        if (*error != CL_SUCCESS)
        {
            log_error( "ERROR: Unable to unmap image after writing: %s\n", IGetErrorString( *error ) );
            return NULL;
        }
    }
    return img;
}


// WARNING -- not thread safe
BufferOwningPtr<char> srcData;
BufferOwningPtr<char> dstData;
BufferOwningPtr<char> srcHost;
BufferOwningPtr<char> dstHost;

int test_copy_image_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo,
                            const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d )
{
    int error;

    clMemWrapper srcImage, dstImage;

    if( gDebugTrace )
        log_info( " ++ Entering inner test loop...\n" );

    // Generate some data to test against
    size_t srcBytes = 0;
    if( gTestMipmaps )
    {
        srcBytes = (size_t)compute_mipmapped_image_size( *srcImageInfo );
    }
    else
    {
        switch (srcImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                srcBytes = srcImageInfo->rowPitch;
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                srcBytes = srcImageInfo->height * srcImageInfo->rowPitch;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                srcBytes = srcImageInfo->depth * srcImageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                srcBytes = srcImageInfo->arraySize * srcImageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                srcBytes = srcImageInfo->arraySize * srcImageInfo->slicePitch;
                break;
        }
    }

    if (srcBytes > srcData.getSize())
    {
        if( gDebugTrace )
            log_info( " - Resizing random image data...\n" );

        generate_random_image_data( srcImageInfo, srcData, d  );

        // Update the host verification copy of the data.
        srcHost.reset(malloc(srcBytes),NULL,0,srcBytes);
        if (srcHost == NULL) {
            log_error( "ERROR: Unable to malloc %lu bytes for srcHost\n", srcBytes );
            return -1;
        }
        memcpy(srcHost,srcData,srcBytes);
    }

    // Construct testing sources
    if( gDebugTrace )
        log_info( " - Writing source image...\n" );

    srcImage = create_image( context, queue, srcData, srcImageInfo, &error );
    if( srcImage == NULL )
        return error;


    // Initialize the destination to empty
    size_t destImageSize = 0;
    if( gTestMipmaps )
    {
        destImageSize = (size_t)compute_mipmapped_image_size( *dstImageInfo );
    }
    else
    {
        switch (dstImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                destImageSize = dstImageInfo->rowPitch;
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                destImageSize = dstImageInfo->height * dstImageInfo->rowPitch;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                destImageSize = dstImageInfo->depth * dstImageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                destImageSize = dstImageInfo->arraySize * dstImageInfo->slicePitch;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                destImageSize = dstImageInfo->arraySize * dstImageInfo->slicePitch;
                break;
        }
    }

    if (destImageSize > dstData.getSize())
    {
        if( gDebugTrace )
            log_info( " - Resizing destination buffer...\n" );
        dstData.reset(malloc(destImageSize),NULL,0,destImageSize);
        if (dstData == NULL) {
            log_error( "ERROR: Unable to malloc %lu bytes for dstData\n", destImageSize );
            return -1;
        }

        dstHost.reset(malloc(destImageSize),NULL,0,destImageSize);
        if (dstHost == NULL) {
            dstData.reset(NULL);
            log_error( "ERROR: Unable to malloc %lu bytes for dstHost\n", destImageSize );
            return -1;
        }
    }
    memset( dstData, 0xff, destImageSize );
    memset( dstHost, 0xff, destImageSize );

    if( gDebugTrace )
        log_info( " - Writing destination image...\n" );

    dstImage = create_image( context, queue, dstData, dstImageInfo, &error );
    if( dstImage == NULL )
        return error;

    size_t dstRegion[ 3 ] = { dstImageInfo->width, 1, 1};
    size_t dst_lod = 0;
    size_t origin[ 4 ] = { 0, 0, 0, 0 };

    if(gTestMipmaps)
    {
        switch(dstImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                dst_lod = destPos[1];
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            case CL_MEM_OBJECT_IMAGE2D:
                dst_lod = destPos[2];
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            case CL_MEM_OBJECT_IMAGE3D:
                dst_lod = destPos[3];
                break;
        }

        dstRegion[ 0 ] = (dstImageInfo->width >> dst_lod)?(dstImageInfo->width >> dst_lod) : 1;
    }
    switch (dstImageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            if( gTestMipmaps )
                origin[ 1 ] = dst_lod;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            dstRegion[ 1 ] = dstImageInfo->height;
            if( gTestMipmaps )
            {
                dstRegion[ 1 ] = (dstImageInfo->height >> dst_lod) ?(dstImageInfo->height >> dst_lod): 1;
                origin[ 2 ] = dst_lod;
            }
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            dstRegion[ 1 ] = dstImageInfo->height;
            dstRegion[ 2 ] = dstImageInfo->depth;
            if( gTestMipmaps )
            {
                dstRegion[ 1 ] = (dstImageInfo->height >> dst_lod) ?(dstImageInfo->height >> dst_lod): 1;
                dstRegion[ 2 ] = (dstImageInfo->depth >> dst_lod) ?(dstImageInfo->depth >> dst_lod): 1;
                origin[ 3 ] = dst_lod;
            }
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            dstRegion[ 1 ] = dstImageInfo->arraySize;
            if( gTestMipmaps )
                origin[ 2 ] = dst_lod;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            dstRegion[ 1 ] = dstImageInfo->height;
            dstRegion[ 2 ] = dstImageInfo->arraySize;
            if( gTestMipmaps )
            {
                dstRegion[ 1 ] = (dstImageInfo->height >> dst_lod) ?(dstImageInfo->height >> dst_lod): 1;
                origin[ 3 ] = dst_lod;
            }
            break;
    }

    size_t region[ 3 ] = { dstRegion[ 0 ], dstRegion[ 1 ], dstRegion[ 2 ] };

    // Now copy a subset to the destination image. This is the meat of what we're testing
    if( gDebugTrace )
    {
        if( gTestMipmaps )
        {
            log_info( " - Copying from %d,%d,%d,%d to %d,%d,%d,%d size %d,%d,%d\n", (int)sourcePos[ 0 ], (int)sourcePos[ 1 ], (int)sourcePos[ 2 ],(int)sourcePos[ 3 ],
                     (int)destPos[ 0 ], (int)destPos[ 1 ], (int)destPos[ 2 ],(int)destPos[ 3 ],
                     (int)regionSize[ 0 ], (int)regionSize[ 1 ], (int)regionSize[ 2 ] );
        }
        else
        {
            log_info( " - Copying from %d,%d,%d to %d,%d,%d size %d,%d,%d\n", (int)sourcePos[ 0 ], (int)sourcePos[ 1 ], (int)sourcePos[ 2 ],
                     (int)destPos[ 0 ], (int)destPos[ 1 ], (int)destPos[ 2 ],
                     (int)regionSize[ 0 ], (int)regionSize[ 1 ], (int)regionSize[ 2 ] );
        }
    }

    error = clEnqueueCopyImage( queue, srcImage, dstImage, sourcePos, destPos, regionSize, 0, NULL, NULL );
    if( error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to copy image from pos %d,%d,%d to %d,%d,%d size %d,%d,%d! (%s)\n",
                  (int)sourcePos[ 0 ], (int)sourcePos[ 1 ], (int)sourcePos[ 2 ], (int)destPos[ 0 ], (int)destPos[ 1 ], (int)destPos[ 2 ],
                  (int)regionSize[ 0 ], (int)regionSize[ 1 ], (int)regionSize[ 2 ], IGetErrorString( error ) );
        return error;
    }

    // Construct the final dest image values to test against
    if( gDebugTrace )
        log_info( " - Host verification copy...\n" );

    copy_image_data( srcImageInfo, dstImageInfo, srcHost, dstHost, sourcePos, destPos, regionSize );

    // Map the destination image to verify the results with the host
    // copy. The contents of the entire buffer are compared.
    if( gDebugTrace )
        log_info( " - Mapping results...\n" );

    size_t mappedRow, mappedSlice;
    void* mapped = (char*)clEnqueueMapImage(queue, dstImage, CL_TRUE, CL_MAP_READ, origin, region, &mappedRow, &mappedSlice, 0, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
        log_error( "ERROR: Unable to map image for verification: %s\n", IGetErrorString( error ) );
        return error;
    }

    // Verify scanline by scanline, since the pitches are different
    char *sourcePtr = dstHost;
    size_t cur_lod_offset = 0;
    char *destPtr = (char*)mapped;

    if( gTestMipmaps )
    {
        cur_lod_offset = compute_mip_level_offset(dstImageInfo, dst_lod);
        sourcePtr += cur_lod_offset;
    }

    size_t scanlineSize = dstImageInfo->width * get_pixel_size( dstImageInfo->format );
    size_t rowPitch = dstImageInfo->rowPitch;
    size_t slicePitch = dstImageInfo->slicePitch;
    size_t dst_height_lod = dstImageInfo->height;
    if(gTestMipmaps)
    {
        size_t dst_width_lod = (dstImageInfo->width >> dst_lod)?(dstImageInfo->width >> dst_lod) : 1;
        dst_height_lod = (dstImageInfo->height >> dst_lod)?(dstImageInfo->height >> dst_lod) : 1;
        scanlineSize = dst_width_lod * get_pixel_size(dstImageInfo->format);
        rowPitch = scanlineSize;
        slicePitch = rowPitch * dst_height_lod;
    }

    if( gDebugTrace )
        log_info( " - Scanline verification...\n" );

    size_t thirdDim;
    size_t secondDim;
    if (dstImageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    {
        secondDim = dstImageInfo->arraySize;
        thirdDim = 1;
    }
    else if (dstImageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
    {
        secondDim = dstImageInfo->height;
        if( gTestMipmaps )
            secondDim = (dstImageInfo->height >> dst_lod) ? (dstImageInfo->height >> dst_lod):1;
        thirdDim = dstImageInfo->arraySize;
    }
    else
    {
        secondDim = dstImageInfo->height;
        thirdDim = dstImageInfo->depth;
        if( gTestMipmaps )
        {
            secondDim = (dstImageInfo->height >> dst_lod) ? (dstImageInfo->height >> dst_lod):1;
            if(dstImageInfo->type == CL_MEM_OBJECT_IMAGE3D)
                thirdDim = (dstImageInfo->depth >> dst_lod) ? (dstImageInfo->depth >> dst_lod):1;
        }
    }

    for( size_t z = 0; z < thirdDim; z++ )
    {
        for( size_t y = 0; y < secondDim; y++ )
        {
            if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
            {
                log_error( "ERROR: Scanline %d did not verify for image size %d,%d,%d pitch %d (extra %d bytes)\n", (int)y, (int)dstImageInfo->width, (int)dstImageInfo->height, (int)dstImageInfo->depth, (int)dstImageInfo->rowPitch, (int)dstImageInfo->rowPitch - (int)dstImageInfo->width * (int)get_pixel_size( dstImageInfo->format ) );

                // Find the first missing pixel
                size_t pixel_size = get_pixel_size( dstImageInfo->format );
                size_t where = 0;
                for( where = 0; where < dstImageInfo->width; where++ )
                    if( memcmp( sourcePtr + pixel_size * where, destPtr + pixel_size * where, pixel_size) )
                        break;
                log_error( "Failed at column: %ld   ", where );
                switch( pixel_size )
                {
                    case 1:
                        log_error( "*0x%2.2x vs. 0x%2.2x\n", ((cl_uchar*)(sourcePtr + pixel_size * where))[0], ((cl_uchar*)(destPtr + pixel_size * where))[0] );
                        break;
                    case 2:
                        log_error( "*0x%4.4x vs. 0x%4.4x\n", ((cl_ushort*)(sourcePtr + pixel_size * where))[0], ((cl_ushort*)(destPtr + pixel_size * where))[0] );
                        break;
                    case 3:
                        log_error( "*{0x%2.2x, 0x%2.2x, 0x%2.2x} vs. {0x%2.2x, 0x%2.2x, 0x%2.2x}\n",
                                  ((cl_uchar*)(sourcePtr + pixel_size * where))[0], ((cl_uchar*)(sourcePtr + pixel_size * where))[1], ((cl_uchar*)(sourcePtr + pixel_size * where))[2],
                                  ((cl_uchar*)(destPtr + pixel_size * where))[0], ((cl_uchar*)(destPtr + pixel_size * where))[1], ((cl_uchar*)(destPtr + pixel_size * where))[2]
                                  );
                        break;
                    case 4:
                        log_error( "*0x%8.8x vs. 0x%8.8x\n", ((cl_uint*)(sourcePtr + pixel_size * where))[0], ((cl_uint*)(destPtr + pixel_size * where))[0] );
                        break;
                    case 6:
                        log_error( "*{0x%4.4x, 0x%4.4x, 0x%4.4x} vs. {0x%4.4x, 0x%4.4x, 0x%4.4x}\n",
                                  ((cl_ushort*)(sourcePtr + pixel_size * where))[0], ((cl_ushort*)(sourcePtr + pixel_size * where))[1], ((cl_ushort*)(sourcePtr + pixel_size * where))[2],
                                  ((cl_ushort*)(destPtr + pixel_size * where))[0], ((cl_ushort*)(destPtr + pixel_size * where))[1], ((cl_ushort*)(destPtr + pixel_size * where))[2]
                                  );
                        break;
                    case 8:
                        log_error( "*0x%16.16llx vs. 0x%16.16llx\n", ((cl_ulong*)(sourcePtr + pixel_size * where))[0], ((cl_ulong*)(destPtr + pixel_size * where))[0] );
                        break;
                    case 12:
                        log_error( "*{0x%8.8x, 0x%8.8x, 0x%8.8x} vs. {0x%8.8x, 0x%8.8x, 0x%8.8x}\n",
                                  ((cl_uint*)(sourcePtr + pixel_size * where))[0], ((cl_uint*)(sourcePtr + pixel_size * where))[1], ((cl_uint*)(sourcePtr + pixel_size * where))[2],
                                  ((cl_uint*)(destPtr + pixel_size * where))[0], ((cl_uint*)(destPtr + pixel_size * where))[1], ((cl_uint*)(destPtr + pixel_size * where))[2]
                                  );
                        break;
                    case 16:
                        log_error( "*{0x%8.8x, 0x%8.8x, 0x%8.8x, 0x%8.8x} vs. {0x%8.8x, 0x%8.8x, 0x%8.8x, 0x%8.8x}\n",
                                  ((cl_uint*)(sourcePtr + pixel_size * where))[0], ((cl_uint*)(sourcePtr + pixel_size * where))[1], ((cl_uint*)(sourcePtr + pixel_size * where))[2], ((cl_uint*)(sourcePtr + pixel_size * where))[3],
                                  ((cl_uint*)(destPtr + pixel_size * where))[0], ((cl_uint*)(destPtr + pixel_size * where))[1], ((cl_uint*)(destPtr + pixel_size * where))[2], ((cl_uint*)(destPtr + pixel_size * where))[3]
                                  );
                        break;
                    default:
                        log_error( "Don't know how to print pixel size of %ld\n", pixel_size );
                        break;
                }

                return -1;
            }
            sourcePtr += rowPitch;
            if((dstImageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY || dstImageInfo->type == CL_MEM_OBJECT_IMAGE1D))
            destPtr += mappedSlice;
            else
            destPtr += mappedRow;
        }
        sourcePtr += slicePitch - rowPitch * dst_height_lod;
        destPtr += mappedSlice - mappedRow * dst_height_lod;
    }

    // Unmap the image.
    error = clEnqueueUnmapMemObject(queue, dstImage, mapped, 0, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        log_error( "ERROR: Unable to unmap image after verify: %s\n", IGetErrorString( error ) );
        return error;
    }

    return 0;
}

int test_copy_image_size_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo, MTdata d )
{
    size_t sourcePos[ 3 ], destPos[ 3 ], regionSize[ 3 ];
    int ret = 0, retCode;

    for (int i = 0; i < 8; i++)
    {
        switch (srcImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                sourcePos[ 0 ] = random_in_range( 0, (int)(srcImageInfo->width - 4), d );
                sourcePos[ 1 ] = 1;
                sourcePos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                sourcePos[ 0 ] = random_in_range( 0, (int)(srcImageInfo->width - 4), d );
                sourcePos[ 1 ] = random_in_range( 0, (int)(srcImageInfo->height - 4), d );
                sourcePos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                sourcePos[ 0 ] = random_in_range( 0, (int)(srcImageInfo->width - 4), d );
                sourcePos[ 1 ] = random_in_range( 0, (int)(srcImageInfo->height - 4), d );
                sourcePos[ 2 ] = random_in_range( 0, (int)(srcImageInfo->depth - 4), d );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                sourcePos[ 0 ] = random_in_range( 0, (int)(srcImageInfo->width - 4), d );
                sourcePos[ 1 ] = random_in_range( 0, (int)(srcImageInfo->arraySize - 4), d );
                sourcePos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                sourcePos[ 0 ] = random_in_range( 0, (int)(srcImageInfo->width - 4), d );
                sourcePos[ 1 ] = random_in_range( 0, (int)(srcImageInfo->height - 4), d );
                sourcePos[ 2 ] = random_in_range( 0, (int)(srcImageInfo->arraySize - 4), d );
                break;
        }

        switch (dstImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                destPos[ 0 ] = random_in_range( 0, (int)(dstImageInfo->width - 4), d );
                destPos[ 1 ] = 1;
                destPos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                destPos[ 0 ] = random_in_range( 0, (int)(dstImageInfo->width - 4), d );
                destPos[ 1 ] = random_in_range( 0, (int)(dstImageInfo->height - 4), d );
                destPos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                destPos[ 0 ] = random_in_range( 0, (int)(dstImageInfo->width - 4), d );
                destPos[ 1 ] = random_in_range( 0, (int)(dstImageInfo->height - 4), d );
                destPos[ 2 ] = random_in_range( 0, (int)(dstImageInfo->depth - 4), d );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                destPos[ 0 ] = random_in_range( 0, (int)(dstImageInfo->width - 4), d );
                destPos[ 1 ] = random_in_range( 0, (int)(dstImageInfo->arraySize - 4), d );
                destPos[ 2 ] = 1;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                destPos[ 0 ] = random_in_range( 0, (int)(dstImageInfo->width - 4), d );
                destPos[ 1 ] = random_in_range( 0, (int)(dstImageInfo->height - 4), d );
                destPos[ 2 ] = random_in_range( 0, (int)(dstImageInfo->arraySize - 4), d );
                break;
        }

        if ( (dstImageInfo->width - destPos[0]) < (srcImageInfo->width - sourcePos[0]) )
            regionSize[0] = random_in_range(1, (dstImageInfo->width - destPos[0]), d);
        else
            regionSize[0] = random_in_range(1, (srcImageInfo->width - sourcePos[0]), d);

        if (srcImageInfo->type == CL_MEM_OBJECT_IMAGE1D || dstImageInfo->type == CL_MEM_OBJECT_IMAGE1D)
            regionSize[1] = 0;
        else
        {
            if ( (dstImageInfo->height - destPos[1]) < (srcImageInfo->height - sourcePos[1]) )
                regionSize[1] = random_in_range(1, (dstImageInfo->height - destPos[1]), d);
            else
                regionSize[1] = random_in_range(1, (srcImageInfo->height - sourcePos[1]), d);
        }

        regionSize[2] = 0;
        if (dstImageInfo->type == CL_MEM_OBJECT_IMAGE3D && srcImageInfo->type == CL_MEM_OBJECT_IMAGE3D)
        {
            if ( (dstImageInfo->depth - destPos[2]) < (srcImageInfo->depth - sourcePos[2]) )
                regionSize[2] = random_in_range(1, (dstImageInfo->depth - destPos[2]), d);
            else
                regionSize[2] = random_in_range(1, (srcImageInfo->depth - sourcePos[2]), d);
        }
        else if ( (dstImageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY && srcImageInfo->type == CL_MEM_OBJECT_IMAGE2D_ARRAY) )
        {
            if ( (dstImageInfo->arraySize - destPos[2]) < (srcImageInfo->arraySize - sourcePos[2]) )
                regionSize[2] = random_in_range(1, (dstImageInfo->arraySize - destPos[2]), d);
            else
                regionSize[2] = random_in_range(1, (srcImageInfo->arraySize - sourcePos[2]), d);
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

