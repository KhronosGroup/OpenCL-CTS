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

extern void read_image_pixel_float(void *imageData, image_descriptor *imageInfo,
                                   int x, int y, int z, float *outData);

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
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            dataBytes = imageInfo->rowPitch;
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
            log_error("ERROR: Unable to malloc %zu bytes for imgHost\n",
                      dataBytes);
            return -1;
        }
    }

    // Reset the host verification copy of the data.
    memcpy(imgHost, imgData, dataBytes);

    // Construct testing sources
    if ( gDebugTrace )
        log_info( " - Creating image...\n" );

    image = create_image(context, queue, imgData, imageInfo, gEnablePitch,
                         false, &error);
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
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
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
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
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
            log_error("Test error: unhandled image type at %s:%d\n", __FILE__,
                      __LINE__);
    };

    // Count the number of bytes successfully matched
    size_t total_matched = 0;

    for ( size_t z = 0; z < thirdDim; z++ )
    {
        for ( size_t y = 0; y < secondDim; y++ )
        {
            if (memcmp( sourcePtr, destPtr, scanlineSize ) != 0)
            {
                // Find the first differing pixel
                size_t pixel_size = get_pixel_size( imageInfo->format );
                size_t where = compare_scanlines(imageInfo, sourcePtr, destPtr);

                if (where < imageInfo->width)
                {
                    print_first_pixel_difference_error(
                        where, sourcePtr + pixel_size * where,
                        destPtr + pixel_size * where, imageInfo, y, thirdDim);
                    return -1;
                }
            }

            total_matched += scanlineSize;
            sourcePtr += imageInfo->rowPitch;
            if ((imageInfo->type == CL_MEM_OBJECT_IMAGE1D_ARRAY
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE1D
                 || imageInfo->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))
                destPtr += mappedSlice;
            else
            destPtr += mappedRow;
        }

        sourcePtr += imageInfo->slicePitch - ( imageInfo->rowPitch * (imageInfo->height > 0 ? imageInfo->height : 1) );
        destPtr += mappedSlice - ( mappedRow * (imageInfo->height > 0 ? imageInfo->height : 1) );
    }

    // Unmap the image.
    error = clEnqueueUnmapMemObject(queue, image, mapped, 0, NULL, NULL);
    error = error | clFinish(queue);
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
