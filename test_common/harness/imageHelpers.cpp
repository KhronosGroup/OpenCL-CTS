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
#include "imageHelpers.h"

size_t get_format_type_size( const cl_image_format *format )
{
    return get_channel_data_type_size( format->image_channel_data_type );
}

size_t get_channel_data_type_size( cl_channel_type channelType )
{
    switch( channelType )
    {
        case CL_SNORM_INT8:
        case CL_UNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_UNSIGNED_INT8:
            return 1;
            
        case CL_SNORM_INT16:
        case CL_UNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_UNSIGNED_INT16:
        case CL_HALF_FLOAT:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return sizeof( cl_short );
            
        case CL_SIGNED_INT32:
        case CL_UNSIGNED_INT32:
            return sizeof( cl_int );
            
        case CL_UNORM_SHORT_565:
        case CL_UNORM_SHORT_555:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_SHORT_565_REV:
        case CL_UNORM_SHORT_555_REV:
#endif
            return 2;
            
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_8888:
        case CL_UNORM_INT_8888_REV:
            return 4;
#endif
            
        case CL_UNORM_INT_101010:
#ifdef OBSOLETE_FORAMT
        case CL_UNORM_INT_101010_REV:
#endif
            return 4;
            
        case CL_FLOAT:
            return sizeof( cl_float );
            
        default:
            return 0;
    }
}

size_t get_format_channel_count( const cl_image_format *format )
{
    return get_channel_order_channel_count( format->image_channel_order );
}

size_t get_channel_order_channel_count( cl_channel_order order )
{
    switch( order )
    {
        case CL_R:
        case CL_A:
        case CL_Rx:
        case CL_INTENSITY:
        case CL_LUMINANCE:
            return 1;
            
        case CL_RG:
        case CL_RA:
        case CL_RGx:
            return 2;
            
        case CL_RGB:
        case CL_RGBx:
            return 3;
            
        case CL_RGBA:
        case CL_ARGB:
        case CL_BGRA:
#ifdef CL_1RGB_APPLE
        case CL_1RGB_APPLE:
#endif
#ifdef CL_BGR1_APPLE
        case CL_BGR1_APPLE:
#endif
            return 4;
            
        default:
            return 0;
    }
}

int is_format_signed( const cl_image_format *format )
{
    switch( format->image_channel_data_type )
    {
        case CL_SNORM_INT8:
        case CL_SIGNED_INT8:
        case CL_SNORM_INT16:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
        case CL_HALF_FLOAT:
        case CL_FLOAT:
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
            return 1;
            
        default:
            return 0;
    }
}

size_t get_pixel_size( cl_image_format *format )
{
  switch( format->image_channel_data_type )
  {
    case CL_SNORM_INT8:
    case CL_UNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
      return get_format_channel_count( format );
            
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
#ifdef  CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
#endif
      return get_format_channel_count( format ) * sizeof( cl_ushort );
            
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
      return get_format_channel_count( format ) * sizeof( cl_int );
            
    case CL_UNORM_SHORT_565:
    case CL_UNORM_SHORT_555:
#ifdef OBSOLETE_FORAMT
    case CL_UNORM_SHORT_565_REV:
    case CL_UNORM_SHORT_555_REV:
#endif
      return 2;
            
#ifdef OBSOLETE_FORAMT
    case CL_UNORM_INT_8888:
    case CL_UNORM_INT_8888_REV:
      return 4;
#endif
            
    case CL_UNORM_INT_101010:
#ifdef OBSOLETE_FORAMT
    case CL_UNORM_INT_101010_REV:
#endif
      return 4;
            
    case CL_FLOAT:
      return get_format_channel_count( format ) * sizeof( cl_float );
            
    default:
      return 0;
  }
}

int get_8_bit_image_format( cl_context context, cl_mem_object_type objType, cl_mem_flags flags, size_t channelCount, cl_image_format *outFormat )
{
	cl_image_format formatList[ 128 ];
	unsigned int outFormatCount, i;
	int error;
	
	
	/* Make sure each image format is supported */
	if ((error = clGetSupportedImageFormats( context, flags, objType, 128, formatList, &outFormatCount )))
    return error;
	
  
	/* Look for one that is an 8-bit format */
	for( i = 0; i < outFormatCount; i++ )
	{
		if( formatList[ i ].image_channel_data_type == CL_SNORM_INT8 ||
       formatList[ i ].image_channel_data_type == CL_UNORM_INT8 ||
		   formatList[ i ].image_channel_data_type == CL_SIGNED_INT8 ||
		   formatList[ i ].image_channel_data_type == CL_UNSIGNED_INT8 )
		{
      if ( !channelCount || ( channelCount && ( get_format_channel_count( &formatList[ i ] ) == channelCount ) ) )
      {
        *outFormat = formatList[ i ];
        return 0;
      }
		}
	}
	
	return -1;
}

int get_32_bit_image_format( cl_context context, cl_mem_object_type objType, cl_mem_flags flags, size_t channelCount, cl_image_format *outFormat )
{
	cl_image_format formatList[ 128 ];
	unsigned int outFormatCount, i;
	int error;
	
	
  /* Make sure each image format is supported */
  if ((error = clGetSupportedImageFormats( context, flags, objType, 128, formatList, &outFormatCount )))
    return error;
    
  /* Look for one that is an 8-bit format */
  for( i = 0; i < outFormatCount; i++ )
  {
		if( formatList[ i ].image_channel_data_type == CL_UNORM_INT_101010 ||
		   formatList[ i ].image_channel_data_type == CL_FLOAT ||
		   formatList[ i ].image_channel_data_type == CL_SIGNED_INT32 ||
		   formatList[ i ].image_channel_data_type == CL_UNSIGNED_INT32 )
    {
      if ( !channelCount || ( channelCount && ( get_format_channel_count( &formatList[ i ] ) == channelCount ) ) )
      {
        *outFormat = formatList[ i ];
        return 0;
      }
    }
	}
	
	return -1;
}

