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

extern bool			gDebugTrace, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages, gTestRounding;
extern cl_filter_mode	gFilterModeToUse;
extern cl_addressing_mode	gAddressModeToUse;
extern cl_command_queue queue;
extern cl_context context;

int test_read_image_2D_array( cl_device_id device, image_descriptor *imageInfo, MTdata d )
{
	int error;
	
	clMemWrapper image;
	
	// Create some data to test against
    BufferOwningPtr<char> imageValues;
	generate_random_image_data( imageInfo, imageValues, d );
	
	if( gDebugTrace )
		log_info( " - Creating image %d by %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );

	// Construct testing sources
	image = create_image_2d_array( context, (cl_mem_flags)(CL_MEM_READ_ONLY), imageInfo->format, imageInfo->width, imageInfo->height, imageInfo->arraySize, 0, 0, NULL, &error );
	if( image == NULL )
	{
		log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, IGetErrorString( error ) );
		return -1;
	}
	
	if( gDebugTrace )
		log_info( " - Writing image...\n" );

	size_t origin[ 3 ] = { 0, 0, 0 };
	size_t region[ 3 ] = { imageInfo->width, imageInfo->height, imageInfo->arraySize };
	
	error = clEnqueueWriteImage(queue, image, CL_TRUE,
								origin, region, ( gEnablePitch ? imageInfo->rowPitch : 0 ), ( gEnablePitch ? imageInfo->slicePitch : 0 ),
								imageValues, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		log_error( "ERROR: Unable to write to 2D image array of size %d x %d x %d\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );
		return -1;
	}
	
	// To verify, we just read the results right back and see whether they match the input
	if( gDebugTrace )
		log_info( " - Initing result array...\n" );
	
	// Note: we read back without any pitch, to verify pitch actually WORKED
	size_t scanlineSize = imageInfo->width * get_pixel_size( imageInfo->format );
	size_t pageSize = scanlineSize * imageInfo->height;
	size_t imageSize = pageSize * imageInfo->arraySize;
	BufferOwningPtr<char> resultValues(malloc(imageSize));
	memset( resultValues, 0xff, imageSize );
	
	if( gDebugTrace )
		log_info( " - Reading results...\n" );
	
	error = clEnqueueReadImage( queue, image, CL_TRUE, origin, region, 0, 0, resultValues, 0, NULL, NULL );
	test_error( error, "Unable to read image values" );
	
	// Verify scanline by scanline, since the pitches are different
	char *sourcePtr = (char *)(void *)imageValues;
	char *destPtr = resultValues;
	
	for( size_t z = 0; z < imageInfo->arraySize; z++ )
	{
		for( size_t y = 0; y < imageInfo->height; y++ )
		{
			if( memcmp( sourcePtr, destPtr, scanlineSize ) != 0 )
			{
				log_error( "ERROR: Scanline %d,%d did not verify for image size %d,%d,%d pitch %d,%d\n", (int)y, (int)z, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch );
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

int test_read_image_set_2D_array( cl_device_id device, cl_image_format *format )
{
	size_t maxWidth, maxHeight, maxArraySize;
	cl_ulong maxAllocSize, memSize;
	image_descriptor imageInfo;
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
					if( gDebugTrace )
						log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize );
					int ret = test_read_image_2D_array( device, &imageInfo, seed );	
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
            log_info("Testing %d x %d x %d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize);
            if( test_read_image_2D_array( device, &imageInfo, seed ) )
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
			} while(  size > maxAllocSize || ( size * 3 ) > memSize );
			
			if( gDebugTrace )
				log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxArraySize );
			int ret = test_read_image_2D_array( device, &imageInfo, seed );	
			if( ret )
				return -1;
		}
	}
	
	return 0;
}
