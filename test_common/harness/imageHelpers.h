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
#ifndef _imageHelpers_h
#define _imageHelpers_h

#include "errorHelpers.h"


extern size_t get_format_type_size( const cl_image_format *format );
extern size_t get_channel_data_type_size( cl_channel_type channelType );
extern size_t get_format_channel_count( const cl_image_format *format );
extern size_t get_channel_order_channel_count( cl_channel_order order );
extern int    is_format_signed( const cl_image_format *format );
extern size_t get_pixel_size( cl_image_format *format );

/* Helper to get any ol image format as long as it is 8-bits-per-channel */
extern int get_8_bit_image_format( cl_context context, cl_mem_object_type objType, cl_mem_flags flags, size_t channelCount, cl_image_format *outFormat );

/* Helper to get any ol image format as long as it is 32-bits-per-channel */
extern int get_32_bit_image_format( cl_context context, cl_mem_object_type objType, cl_mem_flags flags, size_t channelCount, cl_image_format *outFormat );


#endif // _imageHelpers_h

