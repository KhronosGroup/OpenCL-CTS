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

extern int test_get_image_info_single(cl_context context,
                                      cl_command_queue queue,
                                      image_descriptor *imageInfo, MTdata d,
                                      cl_mem_flags flags);

int test_get_image_info_3D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_image_format *format,
                           cl_mem_flags flags)
{
    size_t maxWidth, maxHeight, maxDepth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    if ((flags != CL_MEM_READ_ONLY)
        && !is_extension_available(device, "cl_khr_3d_image_writes"))
    {
        log_info("-----------------------------------------------------\n");
        log_info("This device does not support cl_khr_3d_image_writes.\n"
                 "Skipping 3d image write test.\n");
        log_info("-----------------------------------------------------\n\n");
        return 0;
    }

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
                    int ret = test_get_image_info_single(
                        context, queue, &imageInfo, seed, flags);
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
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.depth = sizes[ idx ][ 2 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;

            log_info( "Testing %d x %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            if( gDebugTrace )
                log_info( "   at max size %d,%d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            if (test_get_image_info_single(context, queue, &imageInfo, seed,
                                           flags))
                return -1;
        }
    }
    else
    {
        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            cl_ulong slicePitch;
            cl_ulong rowPitch;

            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );
                imageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );
                imageInfo.depth = (size_t)random_log_in_range( 16, (int)maxDepth / 32, seed );

                rowPitch = imageInfo.width * pixelSize;
                slicePitch = imageInfo.rowPitch * imageInfo.height;

                size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                rowPitch += extraWidth;

                do {
                    extraWidth++;
                    rowPitch += extraWidth;
                } while ((rowPitch % pixelSize) != 0);

                size_t extraHeight = (int)random_log_in_range( 0, 8, seed );
                slicePitch = rowPitch * (imageInfo.height + extraHeight);

                size = slicePitch * imageInfo.depth * 4 * 4;
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            imageInfo.slicePitch = slicePitch;
            imageInfo.rowPitch = rowPitch;

            if( gDebugTrace )
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxDepth );
            int ret = test_get_image_info_single(context, queue, &imageInfo,
                                                 seed, flags);
            if( ret )
                return -1;
        }
    }

    return 0;
}

