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
extern uint64_t gRoundingStartValue;
extern cl_command_queue queue;
extern cl_context context;

extern int test_copy_image_generic( cl_device_id device, image_descriptor *srcImageInfo, image_descriptor *dstImageInfo, 
								   const size_t sourcePos[], const size_t destPos[], const size_t regionSize[], MTdata d );
								   
int test_copy_image_size_2D( cl_device_id device, image_descriptor *imageInfo, MTdata d )
{
	size_t sourcePos[ 3 ], destPos[ 3 ], regionSize[ 3 ];
	int ret = 0, retCode;
	
	// First, try just a full covering region
	sourcePos[ 0 ] = sourcePos[ 1 ] = sourcePos[ 2 ] = 0;
	destPos[ 0 ] = destPos[ 1 ] = destPos[ 2 ] = 0;
	regionSize[ 0 ] = imageInfo->width;
	regionSize[ 1 ] = imageInfo->height;
	regionSize[ 2 ] = 1;
	
	retCode = test_copy_image_generic( device, imageInfo, imageInfo, sourcePos, destPos, regionSize, d ); 
	if( retCode < 0 )
		return retCode;
	else
		ret += retCode;
		
	// Now try a sampling of different random regions
	for( int i = 0; i < 8; i++ )
	{
		// Pick a random size
		regionSize[ 0 ] = ( imageInfo->width > 8 ) ? (size_t)random_in_range( 8, (int)imageInfo->width - 1, d ) : imageInfo->width;
		regionSize[ 1 ] = ( imageInfo->height > 8 ) ? (size_t)random_in_range( 8, (int)imageInfo->height - 1, d ) : imageInfo->height;
		
		// Now pick positions within valid ranges
		sourcePos[ 0 ] = ( imageInfo->width > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->width - regionSize[ 0 ] - 1 ), d ) : 0;
		sourcePos[ 1 ] = ( imageInfo->height > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->height - regionSize[ 1 ] - 1 ), d ) : 0;

		destPos[ 0 ] = ( imageInfo->width > regionSize[ 0 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->width - regionSize[ 0 ] - 1 ), d ) : 0;
		destPos[ 1 ] = ( imageInfo->height > regionSize[ 1 ] ) ? (size_t)random_in_range( 0, (int)( imageInfo->height - regionSize[ 1 ] - 1 ), d ) : 0;
		
		// Go for it!
		retCode = test_copy_image_generic( device, imageInfo, imageInfo, sourcePos, destPos, regionSize, d ); 
		if( retCode < 0 )
			return retCode;
		else
			ret += retCode;
	}
	
	return ret;
}

int test_copy_image_set_2D( cl_device_id device, cl_image_format *format )
{
	size_t maxWidth, maxHeight;
	cl_ulong maxAllocSize, memSize;
	image_descriptor imageInfo = { 0 };
	RandomSeed seed(gRandomSeed); 
	size_t rowPadding = gEnablePitch ? 48 : 0;
	size_t pixelSize;
		
	imageInfo.format = format;
	imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
	pixelSize = get_pixel_size( imageInfo.format );
	
	int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
	error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
	error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
	error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
	test_error( error, "Unable to get max image 2D size from device" );

	if( gTestSmallImages )
	{
		for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
		{
			imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
			
			if (gEnablePitch)
			{
				do {
					rowPadding++;
					imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
				} while ((imageInfo.rowPitch % pixelSize) != 0);
			}
			
			for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
			{
				if( gDebugTrace )
					log_info( "   at size %d,%d\n", (int)imageInfo.width, (int)imageInfo.height );

				int ret = test_copy_image_size_2D( device, &imageInfo, seed );	
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
			imageInfo.width = sizes[ idx ][ 0 ];
			imageInfo.height = sizes[ idx ][ 1 ];
			imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
			
			if (gEnablePitch)
			{
				do {
					rowPadding++;
					imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
				} while ((imageInfo.rowPitch % pixelSize) != 0);
			}
			
			log_info( "Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
			if( gDebugTrace )
				log_info( "   at max size %d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
			if( test_copy_image_size_2D( device, &imageInfo, seed ) )
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
				
				imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
				
				if (gEnablePitch)
				{
					do {
						rowPadding++;
						imageInfo.rowPitch = imageInfo.width * pixelSize + rowPadding;
					} while ((imageInfo.rowPitch % pixelSize) != 0);
				}

				size = (size_t)imageInfo.rowPitch * (size_t)imageInfo.height * 4;
			} while(  size > maxAllocSize || ( size * 3 ) > memSize );
			
			if( gDebugTrace )
				log_info( "   at size %d,%d (row pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.rowPitch, (int)maxWidth, (int)maxHeight );
			int ret = test_copy_image_size_2D( device, &imageInfo, seed );	
			if( ret )
				return -1;
		}
	}
	
	return 0;
}
