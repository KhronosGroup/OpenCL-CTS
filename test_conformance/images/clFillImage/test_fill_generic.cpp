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

extern void read_image_pixel_float( void *imageData, image_descriptor *imageInfo, int x, int y, int z, float *outData );

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
        long long unsigned imageSize = get_image_size_mb( imageInfo );
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                log_error( "ERROR: Unable to create 1D image of size %d (%llu MB): %s\n", (int)imageInfo->width, imageSize, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_error( "ERROR: Unable to create 2D image of size %d x %d (%llu MB): %s\n", (int)imageInfo->width, (int)imageInfo->height, imageSize, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                log_error( "ERROR: Unable to create 3D image of size %d x %d x %d (%llu MB): %s\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, imageSize, IGetErrorString( *error ) );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_error( "ERROR: Unable to create 1D image array of size %d x %d (%llu MB): %s\n", (int)imageInfo->width, (int)imageInfo->arraySize, imageSize, IGetErrorString( *error ) );
                break;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (%llu MB): %s\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, imageSize, IGetErrorString( *error ) );
                break;
        }
        return NULL;
    }

    // Copy the specified data to the image via a Map operation.
    size_t mappedRow, mappedSlice;
    size_t height;
    size_t depth;
    size_t imageSize = 0;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            height = imageInfo->arraySize;
            depth = 1;
            imageSize = imageInfo->rowPitch * imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE1D:
            height = depth = 1;
            imageSize = imageInfo->rowPitch;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            height = imageInfo->height;
            depth = 1;
            imageSize = imageInfo->rowPitch * imageInfo->height;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            height = imageInfo->height;
            depth = imageInfo->arraySize;
            imageSize = imageInfo->slicePitch * imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            height = imageInfo->height;
            depth = imageInfo->depth;
            imageSize = imageInfo->slicePitch * imageInfo->depth;
            break;
    }

    size_t origin[ 3 ] = { 0, 0, 0 };
    size_t region[ 3 ] = { imageInfo->width, height, depth };

    void* mapped = (char*)clEnqueueMapImage(queue, img, CL_TRUE, CL_MAP_WRITE, origin, region, &mappedRow, &mappedSlice, 0, NULL, NULL, error);
    if (*error != CL_SUCCESS || !mapped)
    {
        log_error( "ERROR: Unable to map image for writing: %s\n", IGetErrorString( *error ) );
        return NULL;
    }
    size_t mappedSlicePad = mappedSlice - (mappedRow * height);

    // Copy the image.
    size_t scanlineSize = imageInfo->rowPitch;
    size_t sliceSize = imageInfo->slicePitch - scanlineSize * height;

    char* src = (char*)data;
    char* dst = (char*)mapped;

    if ((mappedRow == scanlineSize) && ((mappedSlice == imageInfo->slicePitch) || (imageInfo->depth==0 && imageInfo->arraySize==0))) {
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
                memcpy( dst, src, imageInfo->width * get_pixel_size(imageInfo->format) );
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

    return img;
}

static void fill_region_with_value( image_descriptor *imageInfo, void *imageValues,
    void *value, const size_t origin[], const size_t region[] )
{
    size_t pixelSize = get_pixel_size( imageInfo->format );

    // Get initial pointer
    char *destPtr   = (char *)imageValues + origin[ 2 ] * imageInfo->slicePitch
        + origin[ 1 ] * imageInfo->rowPitch + pixelSize * origin[ 0 ];

    char *fillColor = (char *)malloc(pixelSize);
    memcpy(fillColor, value, pixelSize);

    // Use pixel at origin to fill region.
    for( size_t z = 0; z < ( region[ 2 ] > 0 ? region[ 2 ] : 1 ); z++ ) {
        char *rowDestPtr = destPtr;
        for( size_t y = 0; y < region[ 1 ]; y++ ) {
            char *pixelDestPtr = rowDestPtr;

            for( size_t x = 0; x < region[ 0 ]; x++ ) {
                memcpy( pixelDestPtr, fillColor, pixelSize );
                pixelDestPtr += pixelSize;
            }
            rowDestPtr += imageInfo->rowPitch;
        }
        destPtr += imageInfo->slicePitch;
    }

    free(fillColor);
}

int test_fill_image_generic( cl_context context, cl_command_queue queue, image_descriptor *imageInfo,
                             const size_t origin[], const size_t region[], ExplicitType outputType, MTdata d )
{
    BufferOwningPtr<char> imgData;
    BufferOwningPtr<char> imgHost;

    int error;
    clMemWrapper image;

    if ( gDebugTrace )
        log_info( " ++ Entering inner test loop...\n" );

    // Generate some data to test against
    size_t dataBytes = 0;

    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            dataBytes = imageInfo->rowPitch;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            dataBytes = imageInfo->height * imageInfo->rowPitch;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            dataBytes = imageInfo->depth * imageInfo->slicePitch;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            dataBytes = imageInfo->arraySize * imageInfo->slicePitch;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            dataBytes = imageInfo->arraySize * imageInfo->slicePitch;
            break;
    }

    if (dataBytes > imgData.getSize())
    {
        if ( gDebugTrace )
            log_info( " - Resizing random image data...\n" );

        generate_random_image_data( imageInfo, imgData, d  );

        imgHost.reset( NULL ); // Free previously allocated memory first.
        imgHost.reset(malloc(dataBytes),NULL,0,dataBytes);
        if (imgHost == NULL)
        {
            log_error( "ERROR: Unable to malloc %lu bytes for imgHost\n", dataBytes );
            return -1;
        }
    }

    // Reset the host verification copy of the data.
    memcpy(imgHost, imgData, dataBytes);

    // Construct testing sources
    if ( gDebugTrace )
        log_info( " - Creating image...\n" );

    image = create_image( context, queue, imgData, imageInfo, &error );
    if ( image == NULL )
        return error;

    // Now fill the region defined by origin, region with the pixel value found at origin.
    if ( gDebugTrace )
        log_info( " - Filling at %d,%d,%d size %d,%d,%d\n", (int)origin[ 0 ], (int)origin[ 1 ], (int)origin[ 2 ],
                 (int)region[ 0 ], (int)region[ 1 ], (int)region[ 2 ] );

    // We need to know the rounding mode, in the case of half to allow the
    // pixel pack that generates the verification value to succeed.
    if (imageInfo->format->image_channel_data_type == CL_HALF_FLOAT)
        DetectFloatToHalfRoundingMode(queue);

    if( outputType == kFloat )
    {
        cl_float fillColor[ 4 ];
        read_image_pixel_float( imgHost, imageInfo, origin[ 0 ], origin[ 1 ], origin[ 2 ], fillColor );
        if ( gDebugTrace )
            log_info( " - with value %g, %g, %g, %g\n", fillColor[ 0 ], fillColor[ 1 ], fillColor[ 2 ], fillColor[ 3 ] );
        error = clEnqueueFillImage ( queue, image, fillColor, origin, region, 0, NULL, NULL );
        if ( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to fill image at %d,%d,%d size %d,%d,%d! (%s)\n",
                      (int)origin[ 0 ], (int)origin[ 1 ], (int)origin[ 2 ],
                      (int)region[ 0 ], (int)region[ 1 ], (int)region[ 2 ], IGetErrorString( error ) );
            return error;
        }

        // Write the approriate verification value to the correct region.
        void* verificationValue = malloc(get_pixel_size(imageInfo->format));
        pack_image_pixel(fillColor, imageInfo->format, verificationValue);
        fill_region_with_value( imageInfo, imgHost, verificationValue, origin, region );
        free(verificationValue);
    }
    else if( outputType == kInt )
    {
        cl_int fillColor[ 4 ];
        read_image_pixel<cl_int>( imgHost, imageInfo, origin[ 0 ], origin[ 1 ], origin[ 2 ], fillColor );
        if ( gDebugTrace )
            log_info( " - with value %d, %d, %d, %d\n", fillColor[ 0 ], fillColor[ 1 ], fillColor[ 2 ], fillColor[ 3 ] );
        error = clEnqueueFillImage ( queue, image, fillColor, origin, region, 0, NULL, NULL );
        if ( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to fill image at %d,%d,%d size %d,%d,%d! (%s)\n",
                      (int)origin[ 0 ], (int)origin[ 1 ], (int)origin[ 2 ],
                      (int)region[ 0 ], (int)region[ 1 ], (int)region[ 2 ], IGetErrorString( error ) );
            return error;
        }

        // Write the approriate verification value to the correct region.
        void* verificationValue = malloc(get_pixel_size(imageInfo->format));
        pack_image_pixel(fillColor, imageInfo->format, verificationValue);
        fill_region_with_value( imageInfo, imgHost, verificationValue, origin, region );
        free(verificationValue);
    }
    else // if( outputType == kUInt )
    {
        cl_uint fillColor[ 4 ];
        read_image_pixel<cl_uint>( imgHost, imageInfo, origin[ 0 ], origin[ 1 ], origin[ 2 ], fillColor );
        if ( gDebugTrace )
            log_info( " - with value %u, %u, %u, %u\n", fillColor[ 0 ], fillColor[ 1 ], fillColor[ 2 ], fillColor[ 3 ] );
        error = clEnqueueFillImage ( queue, image, fillColor, origin, region, 0, NULL, NULL );
        if ( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to fill image at %d,%d,%d size %d,%d,%d! (%s)\n",
                      (int)origin[ 0 ], (int)origin[ 1 ], (int)origin[ 2 ],
                      (int)region[ 0 ], (int)region[ 1 ], (int)region[ 2 ], IGetErrorString( error ) );
            return error;
        }

        // Write the approriate verification value to the correct region.
        void* verificationValue = malloc(get_pixel_size(imageInfo->format));
        pack_image_pixel(fillColor, imageInfo->format, verificationValue);
        fill_region_with_value( imageInfo, imgHost, verificationValue, origin, region );
        free(verificationValue);
    }

    // Map the destination image to verify the results with the host
    // copy. The contents of the entire buffer are compared.
    if ( gDebugTrace )
        log_info( " - Mapping results...\n" );

    size_t imageOrigin[ 3 ] = { 0, 0, 0 };
    size_t imageRegion[ 3 ] = { imageInfo->width, 1, 1 };
    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            imageRegion[ 1 ] = imageInfo->height;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            imageRegion[ 1 ] = imageInfo->height;
            imageRegion[ 2 ] = imageInfo->depth;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            imageRegion[ 1 ] = imageInfo->arraySize;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            imageRegion[ 1 ] = imageInfo->height;
            imageRegion[ 2 ] = imageInfo->arraySize;
            break;
    }

    size_t mappedRow, mappedSlice;
    void* mapped = (char*)clEnqueueMapImage(queue, image, CL_TRUE, CL_MAP_READ, imageOrigin, imageRegion, &mappedRow, &mappedSlice, 0, NULL, NULL, &error);
    if (error != CL_SUCCESS)
    {
        log_error( "ERROR: Unable to map image for verification: %s\n", IGetErrorString( error ) );
        return -1;
    }

    // Verify scanline by scanline, since the pitches are different
    char *sourcePtr = imgHost;
    char *destPtr = (char*)mapped;

    size_t scanlineSize = imageInfo->width * get_pixel_size( imageInfo->format );

    if ( gDebugTrace )
        log_info( " - Scanline verification...\n" );

    size_t thirdDim = 1;
    size_t secondDim = 1;

    switch (imageInfo->type) {
      case CL_MEM_OBJECT_IMAGE1D:
        secondDim = 1;
        thirdDim = 1;
        break;
      case CL_MEM_OBJECT_IMAGE2D:
        secondDim = imageInfo->height;
        thirdDim = 1;
        break;
      case CL_MEM_OBJECT_IMAGE3D:
        secondDim = imageInfo->height;
        thirdDim = imageInfo->depth;
        break;
      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        secondDim = imageInfo->arraySize;
        thirdDim = 1;
        break;
      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        secondDim = imageInfo->height;
        thirdDim = imageInfo->arraySize;
        break;
      default:
        log_error("Test error: unhandled image type at %s:%d\n",__FILE__,__LINE__);
    };

    // Count the number of bytes successfully matched
    size_t total_matched = 0;

    for ( size_t z = 0; z < thirdDim; z++ )
    {
        for ( size_t y = 0; y < secondDim; y++ )
        {
            // If the data type is 101010 ignore bits 31 and 32 when comparing the row
            if (imageInfo->format->image_channel_data_type == CL_UNORM_INT_101010) {
              for (size_t w=0;w!=scanlineSize/4;++w) {
                ((cl_uint*)sourcePtr)[w] &= 0x3FFFFFFF;
                ((cl_uint*)destPtr)[w] &= 0x3FFFFFFF;
              }
            }

            if (memcmp( sourcePtr, destPtr, scanlineSize ) != 0)
            {
                // Find the first missing pixel
                size_t pixel_size = get_pixel_size( imageInfo->format );
                size_t where = 0;
                for ( where = 0; where < imageInfo->width; where++ )
                    if ( memcmp( sourcePtr + pixel_size * where, destPtr + pixel_size * where, pixel_size) )
                        break;

                print_first_pixel_difference_error(
                    where, sourcePtr + pixel_size * where,
                    destPtr + pixel_size * where, imageInfo, y, thirdDim);
                return -1;
            }

            total_matched += scanlineSize;
            sourcePtr += imageInfo->rowPitch;
            if((imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY || imageInfo->type == CL_MEM_OBJECT_IMAGE1D))
            destPtr += mappedSlice;
            else
            destPtr += mappedRow;
        }

        sourcePtr += imageInfo->slicePitch - ( imageInfo->rowPitch * (imageInfo->height > 0 ? imageInfo->height : 1) );
        destPtr += mappedSlice - ( mappedRow * (imageInfo->height > 0 ? imageInfo->height : 1) );
    }

    // Unmap the image.
    error = clEnqueueUnmapMemObject(queue, image, mapped, 0, NULL, NULL);
    if (error != CL_SUCCESS)
    {
        log_error( "ERROR: Unable to unmap image after verify: %s\n", IGetErrorString( error ) );
        return -1;
    }

    imgHost.reset(0x0);
    imgData.reset(0x0);

    size_t expected_bytes = scanlineSize * imageRegion[1] * imageRegion[2];
    return (total_matched == expected_bytes) ? 0 : -1;
}
