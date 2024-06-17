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

int test_get_image_info_single( cl_context context, image_descriptor *imageInfo, MTdata d, cl_mem_flags flags, size_t row_pitch, size_t slice_pitch )
{
    int error;
    clMemWrapper image;
    clMemWrapper buffer;
    cl_image_desc imageDesc;
    void *host_ptr = NULL;

    // Generate some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if (flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) {
        host_ptr = (char *)imageValues;
    }

    memset(&imageDesc, 0x0, sizeof(cl_image_desc));
    imageDesc.image_type = imageInfo->type;
    imageDesc.image_width = imageInfo->width;
    imageDesc.image_height = imageInfo->height;
    imageDesc.image_depth = imageInfo->depth;
    imageDesc.image_array_size = imageInfo->arraySize;
    imageDesc.image_row_pitch = row_pitch;
    imageDesc.image_slice_pitch = slice_pitch;

    // Construct testing source
    // Note: for now, just reset the pitches, since they only can actually be different
    // if we use CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR
    imageInfo->rowPitch = imageInfo->width * get_pixel_size( imageInfo->format );
    imageInfo->slicePitch = 0;
    switch (imageInfo->type)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            if ( gDebugTrace )
                log_info( " - Creating 1D image %d with flags=0x%lx row_pitch=%d slice_pitch=%d host_ptr=%p...\n", (int)imageInfo->width, (unsigned long)flags, (int)row_pitch, (int)slice_pitch, host_ptr );
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            if ( gDebugTrace )
                log_info( " - Creating 2D image %d by %d with flags=0x%lx row_pitch=%d slice_pitch=%d host_ptr=%p...\n", (int)imageInfo->width, (int)imageInfo->height, (unsigned long)flags, (int)row_pitch, (int)slice_pitch, host_ptr );
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            imageInfo->slicePitch = imageInfo->rowPitch * imageInfo->height;
            if ( gDebugTrace )
                log_info( " - Creating 3D image %d by %d by %d with flags=0x%lx row_pitch=%d slice_pitch=%d host_ptr=%p...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, (unsigned long)flags, (int)row_pitch, (int)slice_pitch, host_ptr );
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            imageInfo->slicePitch = imageInfo->rowPitch;
            if ( gDebugTrace )
                log_info( " - Creating 1D image array %d by %d with flags=0x%lx row_pitch=%d slice_pitch=%d host_ptr=%p...\n", (int)imageInfo->width, (int)imageInfo->arraySize, (unsigned long)flags, (int)row_pitch, (int)slice_pitch, host_ptr );
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            imageInfo->slicePitch = imageInfo->rowPitch * imageInfo->height;
            if ( gDebugTrace )
                log_info( " - Creating 2D image array %d by %d by %d with flags=0x%lx row_pitch=%d slice_pitch=%d host_ptr=%p...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (unsigned long)flags, (int)row_pitch, (int)slice_pitch, host_ptr );
            break;
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            if (gDebugTrace)
                log_info(" - Creating 1D buffer image %d with flags=0x%lx "
                         "row_pitch=%d slice_pitch=%d host_ptr=%p...\n",
                         (int)imageInfo->width, (unsigned long)flags,
                         (int)row_pitch, (int)slice_pitch, host_ptr);
            int err;
            buffer = clCreateBuffer(context, flags, imageInfo->rowPitch,
                                    host_ptr, &err);
            if (err != CL_SUCCESS)
            {
                log_error("ERROR: Unable to create buffer for 1D image buffer "
                          "of size %d (%s)",
                          (int)imageInfo->rowPitch, IGetErrorString(err));
                return -1;
            }
            imageDesc.buffer = imageInfo->buffer = buffer;
            break;
    }

    image = clCreateImage(context, flags, imageInfo->format, &imageDesc, host_ptr, &error);
    if( image == NULL )
    {
        switch (imageInfo->type)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                log_error( "ERROR: Unable to create 1D image of size %d (%s)", (int)imageInfo->width, IGetErrorString( error ) );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                log_error( "ERROR: Unable to create 2D image of size %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, IGetErrorString( error ) );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                log_error( "ERROR: Unable to create 3D image of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, IGetErrorString( error ) );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                log_error( "ERROR: Unable to create 1D image array of size %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->arraySize, IGetErrorString( error ) );
                break;
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, IGetErrorString( error ) );
                break;
            case CL_MEM_OBJECT_IMAGE1D_BUFFER:
                log_error(
                    "ERROR: Unable to create 1D image buffer of size %d (%s)",
                    (int)imageInfo->width, IGetErrorString(error));
                break;
        }
        return -1;
    }

    // Get info of the image and verify each item is correct
    cl_image_format outFormat;
    error = clGetImageInfo( image, CL_IMAGE_FORMAT, sizeof( outFormat ), &outFormat, NULL );
    test_error( error, "Unable to get image info (format)" );
    if( outFormat.image_channel_order != imageInfo->format->image_channel_order ||
        outFormat.image_channel_data_type != imageInfo->format->image_channel_data_type )
    {
        log_error( "ERROR: image format returned is invalid! (expected %s:%s, got %s:%s (%d:%d))\n",
                    GetChannelOrderName( imageInfo->format->image_channel_order ), GetChannelTypeName( imageInfo->format->image_channel_data_type ),
                      GetChannelOrderName( outFormat.image_channel_order ), GetChannelTypeName( outFormat.image_channel_data_type ),
                       (int)outFormat.image_channel_order, (int)outFormat.image_channel_data_type );
        return 1;
    }

    size_t outElementSize;
    error = clGetImageInfo( image, CL_IMAGE_ELEMENT_SIZE, sizeof( outElementSize ), &outElementSize, NULL );
    test_error( error, "Unable to get image info (element size)" );
    if( outElementSize != get_pixel_size( imageInfo->format ) )
    {
        log_error( "ERROR: image element size returned is invalid! (expected %d, got %d)\n",
                  (int)get_pixel_size( imageInfo->format ), (int)outElementSize );
        return 1;
    }

    size_t outRowPitch;
    error = clGetImageInfo( image, CL_IMAGE_ROW_PITCH, sizeof( outRowPitch ), &outRowPitch, NULL );
    test_error( error, "Unable to get image info (row pitch)" );

  size_t outSlicePitch;
  error = clGetImageInfo( image, CL_IMAGE_SLICE_PITCH, sizeof( outSlicePitch ), &outSlicePitch, NULL );
  test_error( error, "Unable to get image info (slice pitch)" );
    if( imageInfo->type == CL_MEM_OBJECT_IMAGE1D && outSlicePitch != 0 )
    {
        log_error( "ERROR: slice pitch returned is invalid! (expected %d, got %d)\n",
              (int)0, (int)outSlicePitch );
        return 1;
    }

    size_t outWidth;
    error = clGetImageInfo( image, CL_IMAGE_WIDTH, sizeof( outWidth ), &outWidth, NULL );
    test_error( error, "Unable to get image info (width)" );
    if( outWidth != imageInfo->width )
    {
        log_error( "ERROR: image width returned is invalid! (expected %d, got %d)\n",
                  (int)imageInfo->width, (int)outWidth );
        return 1;
    }

  size_t required_height;
  switch (imageInfo->type)
  {
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      required_height = 0;
      break;
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    case CL_MEM_OBJECT_IMAGE3D:
      required_height = imageInfo->height;
      break;
  }

    size_t outHeight;
  error = clGetImageInfo( image, CL_IMAGE_HEIGHT, sizeof( outHeight ), &outHeight, NULL );
  test_error( error, "Unable to get image info (height)" );
  if( outHeight != required_height )
  {
    log_error( "ERROR: image height returned is invalid! (expected %d, got %d)\n",
              (int)required_height, (int)outHeight );
    return 1;
  }

  size_t required_depth;
  switch (imageInfo->type)
  {
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: required_depth = 0; break;
    case CL_MEM_OBJECT_IMAGE3D:
      required_depth = imageInfo->depth;
      break;
  }

  size_t outDepth;
  error = clGetImageInfo( image, CL_IMAGE_DEPTH, sizeof( outDepth ), &outDepth, NULL );
  test_error( error, "Unable to get image info (depth)" );
  if( outDepth != required_depth )
  {
    log_error( "ERROR: image depth returned is invalid! (expected %d, got %d)\n",
        (int)required_depth, (int)outDepth );
    return 1;
  }

  size_t required_array_size;
  switch (imageInfo->type)
  {
    case CL_MEM_OBJECT_IMAGE1D:
    case CL_MEM_OBJECT_IMAGE2D:
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE1D_BUFFER: required_array_size = 0; break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      required_array_size = imageInfo->arraySize;
      break;
  }

  size_t outArraySize;
  error = clGetImageInfo( image, CL_IMAGE_ARRAY_SIZE, sizeof( outArraySize ), &outArraySize, NULL );
  test_error( error, "Unable to get image info (array size)" );
  if( outArraySize != required_array_size )
  {
      log_error( "ERROR: image array size returned is invalid! (expected %d, got %d)\n",
                (int)required_array_size, (int)outArraySize );
      return 1;
  }

  cl_mem outBuffer;
  error = clGetImageInfo( image, CL_IMAGE_BUFFER, sizeof( outBuffer ), &outBuffer, NULL );
  test_error( error, "Unable to get image info (buffer)" );
  if (imageInfo->type == CL_MEM_OBJECT_IMAGE1D_BUFFER) {
      if (outBuffer != imageInfo->buffer) {
          log_error( "ERROR: cl_mem returned is invalid! (expected %p, got %p)\n",
                    imageInfo->buffer, outBuffer );
          return 1;
      }
  } else {
      if (outBuffer != (cl_mem)NULL) {
          log_error( "ERROR: cl_mem returned is invalid! (expected %p, got %p)\n",
                    (cl_mem)NULL, outBuffer );
          return 1;
      }
  }

    cl_uint numMipLevels;
    error = clGetImageInfo( image, CL_IMAGE_NUM_MIP_LEVELS, sizeof( numMipLevels ), &numMipLevels, NULL );
    test_error( error, "Unable to get image info (num mip levels)" );
    if( numMipLevels != 0 )
    {
        log_error( "ERROR: image num_mip_levels returned is invalid! (expected %d, got %d)\n",
                  (int)0, (int)numMipLevels );
        return 1;
    }

    cl_uint numSamples;
    error = clGetImageInfo( image, CL_IMAGE_NUM_SAMPLES, sizeof( numSamples ), &numSamples, NULL );
    test_error( error, "Unable to get image info (num samples)" );
    if( numSamples != 0 )
    {
        log_error( "ERROR: image num_samples returned is invalid! (expected %d, got %d)\n",
                  (int)0, (int)numSamples );
        return 1;
    }

    return 0;
}

int test_get_image_info_2D( cl_device_id device, cl_context context, cl_image_format *format, cl_mem_flags flags )
{
    size_t maxWidth, maxHeight;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    cl_mem_flags all_host_ptr_flags[5] = {
        flags,
        CL_MEM_ALLOC_HOST_PTR | flags,
        CL_MEM_COPY_HOST_PTR | flags,
        CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | flags,
        CL_MEM_USE_HOST_PTR | flags
    };

    memset(&imageInfo, 0x0, sizeof(image_descriptor));
    imageInfo.format = format;
    imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                             sizeof(maxHeight), &maxHeight, NULL);
    test_error( error, "Unable to get max image 2D width or max image 3D height or max memory allocation size or global memory size from device" );

    /* Reduce the size used by the test by half */
    maxAllocSize = get_device_info_max_mem_alloc_size(
        device, MAX_DEVICE_MEMORY_SIZE_DIVISOR);
    memSize =
        get_device_info_global_mem_size(device, MAX_DEVICE_MEMORY_SIZE_DIVISOR);

    if (memSize > (cl_ulong)SIZE_MAX)
    {
        memSize = (cl_ulong)SIZE_MAX;
    }

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
                {
                    if( gDebugTrace )
                        log_info( "   at size %d,%d (flags[%u] 0x%x pitch %d)\n", (int)imageInfo.width, (int)imageInfo.height, j, (unsigned int) all_host_ptr_flags[j], (int)imageInfo.rowPitch );
                    if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                        return -1;
                    if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                        if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, 0 ) )
                            return -1;
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

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE2D, imageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            log_info( "Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
            for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
            {
                if( gDebugTrace )
                    log_info( "   at max size %d,%d (flags[%u] 0x%x pitch %d)\n", (int)imageInfo.width, (int)imageInfo.height, j, (unsigned int) all_host_ptr_flags[j], (int)imageInfo.rowPitch );
                if( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                    return -1;
                if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                    if( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, 0 ) )
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

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                imageInfo.rowPitch += extraWidth;

                do {
                    extraWidth++;
                    imageInfo.rowPitch += extraWidth;
                } while ((imageInfo.rowPitch % pixelSize) != 0);

                size = (cl_ulong)imageInfo.rowPitch * (cl_ulong)imageInfo.height * 4;
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            for (unsigned int j=0; j < sizeof(all_host_ptr_flags)/sizeof(cl_mem_flags); j++)
            {
                if( gDebugTrace )
                    log_info( "   at size %d,%d (flags[%u] 0x%x pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.height, j, (unsigned int) all_host_ptr_flags[j], (int)imageInfo.rowPitch, (int)maxWidth, (int)maxHeight );
                if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], 0, 0 ) )
                    return -1;
                if (all_host_ptr_flags[j] & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) { // skip test when host_ptr is NULL
                    if ( test_get_image_info_single( context, &imageInfo, seed, all_host_ptr_flags[j], imageInfo.rowPitch, 0 ) )
                        return -1;
                }
            }
        }
    }

    return 0;
}
