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
#include <CL/cl.h>
#include "../common.h"

int test_copy_image_generic( cl_context context, cl_command_queue queue, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo,
                            const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d )
{
    int error;

    clMemWrapper srcImage, dstImage;

    BufferOwningPtr<char> srcData;
    BufferOwningPtr<char> dstData;
    BufferOwningPtr<char> srcHost;
    BufferOwningPtr<char> dstHost;

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
        srcBytes = get_image_size(srcImageInfo);
    }

    if (srcBytes > srcData.getSize())
    {
        if( gDebugTrace )
            log_info( " - Resizing random image data...\n" );

        generate_random_image_data( srcImageInfo, srcData, d  );

        // Update the host verification copy of the data.
        srcHost.reset(malloc(srcBytes),NULL,0,srcBytes);
        if (srcHost == NULL) {
            log_error("ERROR: Unable to malloc %zu bytes for srcHost\n",
                      srcBytes);
            return -1;
        }
        memcpy(srcHost,srcData,srcBytes);
    }

    // Construct testing sources
    if( gDebugTrace )
        log_info( " - Writing source image...\n" );

    srcImage = create_image(context, queue, srcData, srcImageInfo, gEnablePitch,
                            gTestMipmaps, &error);
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
        destImageSize = get_image_size(dstImageInfo);
    }

    if (destImageSize > dstData.getSize())
    {
        if( gDebugTrace )
            log_info( " - Resizing destination buffer...\n" );
        dstData.reset(malloc(destImageSize),NULL,0,destImageSize);
        if (dstData == NULL) {
            log_error("ERROR: Unable to malloc %zu bytes for dstData\n",
                      destImageSize);
            return -1;
        }
    }

    if (destImageSize > dstHost.getSize())
    {
        dstHost.reset(NULL);
        dstHost.reset(malloc(destImageSize),NULL,0,destImageSize);
        if (dstHost == NULL) {
            dstData.reset(NULL);
            log_error("ERROR: Unable to malloc %zu bytes for dstHost\n",
                      destImageSize);
            return -1;
        }
    }
    memset( dstData, 0xff, destImageSize );
    memset( dstHost, 0xff, destImageSize );

    if( gDebugTrace )
        log_info( " - Writing destination image...\n" );

    dstImage = create_image(context, queue, dstData, dstImageInfo, gEnablePitch,
                            gTestMipmaps, &error);
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
            case CL_MEM_OBJECT_IMAGE1D_BUFFER: dst_lod = destPos[1]; break;
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
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
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

    size_t thirdDim = 1;
    size_t secondDim = 1;

    switch (dstImageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY: {
            secondDim = dstImageInfo->arraySize;
            break;
        }
        case CL_MEM_OBJECT_IMAGE2D_ARRAY: {
            secondDim = dstImageInfo->height;
            thirdDim = dstImageInfo->arraySize;
            break;
        }
        case CL_MEM_OBJECT_IMAGE3D: {
            secondDim = dstImageInfo->height;
            thirdDim = dstImageInfo->depth;
            break;
        }
        case CL_MEM_OBJECT_IMAGE2D: {
            secondDim = dstImageInfo->height;
            break;
        }
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
        case CL_MEM_OBJECT_IMAGE1D: {
            break;
        }
        default: {
            log_error("ERROR: Unsupported Image type. \n");
            return error;
            break;
        }
    }
    if (gTestMipmaps)
    {
        switch (dstImageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE3D:
                thirdDim = (dstImageInfo->depth >> dst_lod) ? (dstImageInfo->depth >> dst_lod):1;
                /* Fallthrough */
            case CL_MEM_OBJECT_IMAGE2D:
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                secondDim = (dstImageInfo->height >> dst_lod)
                    ? (dstImageInfo->height >> dst_lod)
                    : 1;
                break;
        }
    }
    for( size_t z = 0; z < thirdDim; z++ )
    {
        for( size_t y = 0; y < secondDim; y++ )
        {
            if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
            {
                // Find the first differing pixel
                size_t pixel_size = get_pixel_size( dstImageInfo->format );
                size_t where =
                    compare_scanlines(dstImageInfo, sourcePtr, destPtr);

                if (where < dstImageInfo->width)
                {
                    print_first_pixel_difference_error(
                        where, sourcePtr + pixel_size * where,
                        destPtr + pixel_size * where, dstImageInfo, y,
                        dstImageInfo->depth);
                    return -1;
                }
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

    // Ensure the unmap call completes.
    error = clFinish(queue);
    if (error != CL_SUCCESS)
    {
        log_error("ERROR: clFinish() failed to return CL_SUCCESS: %s\n",
                  IGetErrorString(error));
        return error;
    }

    return 0;
}
