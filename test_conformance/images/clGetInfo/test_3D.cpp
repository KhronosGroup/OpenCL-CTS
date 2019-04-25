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

extern bool            gDebugTrace, gTestSmallImages, gTestMaxImages;

extern int test_get_image_info_single( cl_context context, image_descriptor *imageInfo, MTdata d, cl_mem_flags flags, size_t row_pitch, size_t slice_pitch );

int test_get_image_info_3D( cl_device_id device, cl_context context, cl_image_format *format, cl_mem_flags flags )
{
    size_t maxWidth, maxHeight, maxDepth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    cl_mem_flags all_host_ptr_flags[] = {
        flags,
        CL_MEM_ALLOC_HOST_PTR | flags,
        CL_MEM_COPY_HOST_PTR | flags,
        CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | flags,
        CL_MEM_USE_HOST_PTR | flags
    };

    memset(&imageInfo, 0x0, sizeof(image_descriptor));
    imageInfo.format = format;
    imageInfo.type = CL_MEM_OBJECT_IMAGE3D;
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
                    for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
                    {
                        if( gDebugTrace )
                            log_info( "   at size %d,%d,%d (flags[%u] 0x%lx pitch %d,%d)\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth, j, (unsigned long)all_host_ptr_flags[j], (int)imageInfo.rowPitch, (int)imageInfo.slicePitch );
                        if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                            return -1;
                        if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                            if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, imageInfo.slicePitch ) )
                                return -1;
                        }
                    }
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
            for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
            {
                if( gDebugTrace )
                    log_info( "   at max size %d,%d,%d (flags[%u] 0x%lx pitch %d,%d)\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ], j, (unsigned long)all_host_ptr_flags[j], (int)imageInfo.rowPitch, (int)imageInfo.slicePitch );
                if( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                    return -1;
                if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                    if( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, imageInfo.slicePitch ) )
                        return -1;
                }
            }
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

                size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                imageInfo.rowPitch += extraWidth;

                do {
                    extraWidth++;
                    imageInfo.rowPitch += extraWidth;
                } while ((imageInfo.rowPitch % pixelSize) != 0);

                size_t extraHeight = (int)random_log_in_range( 0, 8, seed );
                imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + extraHeight);

                size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.depth * 4 * 4;
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
            {
                if( gDebugTrace )
                    log_info( "   at size %d,%d,%d (flags[%u] 0x%lx pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth, j, (unsigned long) all_host_ptr_flags[i], (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxDepth );
                if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                    return -1;
                if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                    if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, imageInfo.slicePitch ) )
                        return -1;
                }
            }
        }
    }

    return 0;
}
