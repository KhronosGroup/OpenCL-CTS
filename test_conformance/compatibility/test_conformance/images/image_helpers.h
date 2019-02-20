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
#ifndef _image_helpers_h
#define _image_helpers_h

#include "../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#include "../../test_common/harness/conversions.h"
#include "../../test_common/harness/typeWrappers.h"
#include "../../test_common/harness/kernelHelpers.h"
#include "../../test_common/harness/imageHelpers.h"
#include "../../test_common/harness/errorHelpers.h"
#include "../../test_common/harness/mt19937.h"
#include "../../test_common/harness/rounding_mode.h"
#include "../../test_common/harness/clImageHelper.h"

extern int gTestCount;
extern int gTestFailure;
extern cl_device_type gDeviceType;

// Number of iterations per image format to test if not testing max images, rounding, or small images
#define NUM_IMAGE_ITERATIONS 3

// Definition for our own sampler type, to mirror the cl_sampler internals
typedef struct {
    cl_addressing_mode addressing_mode;
    cl_filter_mode     filter_mode;
    bool               normalized_coords;
} image_sampler_data;

extern void print_read_header( cl_image_format *format, image_sampler_data *sampler, bool err = false, int t = 0 );
extern void print_write_header( cl_image_format *format, bool err);
extern void print_header( cl_image_format *format, bool err );
extern bool find_format( cl_image_format *formatList, unsigned int numFormats, cl_image_format *formatToFind );
extern bool check_minimum_supported( cl_image_format *formatList, unsigned int numFormats, cl_mem_flags flags );

cl_channel_type  get_channel_type_from_name( const char *name );
cl_channel_order  get_channel_order_from_name( const char *name );
int random_in_range( int minV, int maxV, MTdata d );
int random_log_in_range( int minV, int maxV, MTdata d );

typedef struct
{
    size_t width;
    size_t height;
    size_t depth;
    size_t rowPitch;
    size_t slicePitch;
    size_t arraySize;
    cl_image_format *format;
    cl_mem buffer;
    cl_mem_object_type type;
} image_descriptor;

typedef struct
{
    float p[4];
}FloatPixel;

void get_max_sizes(size_t *numberOfSizes, const int maxNumberOfSizes,
                   size_t sizes[][3], size_t maxWidth, size_t maxHeight, size_t maxDepth, size_t maxArraySize,
                   const cl_ulong maxIndividualAllocSize, const cl_ulong maxTotalAllocSize, cl_mem_object_type image_type, cl_image_format *format);
extern size_t get_format_max_int( cl_image_format *format );

extern char * generate_random_image_data( image_descriptor *imageInfo, BufferOwningPtr<char> &Owner, MTdata d );

extern int debug_find_vector_in_image( void *imagePtr, image_descriptor *imageInfo,
                                      void *vectorToFind, size_t vectorSize, int *outX, int *outY, int *outZ );

extern int debug_find_pixel_in_image( void *imagePtr, image_descriptor *imageInfo,
                                     unsigned int *valuesToFind, int *outX, int *outY, int *outZ );
extern int debug_find_pixel_in_image( void *imagePtr, image_descriptor *imageInfo,
                                     int *valuesToFind, int *outX, int *outY, int *outZ );
extern int debug_find_pixel_in_image( void *imagePtr, image_descriptor *imageInfo,
                                     float *valuesToFind, int *outX, int *outY, int *outZ );

extern void copy_image_data( image_descriptor *srcImageInfo, image_descriptor *dstImageInfo, void *imageValues, void *destImageValues,
                            const size_t sourcePos[], const size_t destPos[], const size_t regionSize[] );

int has_alpha(cl_image_format *format);

inline float calculate_array_index( float coord, float extent );

template <class T> void read_image_pixel( void *imageData, image_descriptor *imageInfo,
                                         int x, int y, int z, T *outData )
{
    float convert_half_to_float( unsigned short halfValue );

    if ( x < 0 || x >= (int)imageInfo->width
               || ( imageInfo->height != 0 && ( y < 0 || y >= (int)imageInfo->height ) )
               || ( imageInfo->depth != 0 && ( z < 0 || z >= (int)imageInfo->depth ) )
               || ( imageInfo->arraySize != 0 && ( z < 0 || z >= (int)imageInfo->arraySize ) ) )
    {
        // Border color
        outData[ 0 ] = outData[ 1 ] = outData[ 2 ] = outData[ 3 ] = 0;
        if (!has_alpha(imageInfo->format))
            outData[3] = 1;
        return;
    }

    cl_image_format *format = imageInfo->format;

    unsigned int i;
    T tempData[ 4 ];

    // Advance to the right spot
    char *ptr = (char *)imageData;
    size_t pixelSize = get_pixel_size( format );

    ptr += z * imageInfo->slicePitch + y * imageInfo->rowPitch + x * pixelSize;

    // OpenCL only supports reading floats from certain formats
    switch( format->image_channel_data_type )
    {
        case CL_SNORM_INT8:
        {
            cl_char *dPtr = (cl_char *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNORM_INT8:
        {
            cl_uchar *dPtr = (cl_uchar *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_SIGNED_INT8:
        {
            cl_char *dPtr = (cl_char *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNSIGNED_INT8:
        {
            cl_uchar *dPtr = (cl_uchar*)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_SNORM_INT16:
        {
            cl_short *dPtr = (cl_short *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNORM_INT16:
        {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_SIGNED_INT16:
        {
            cl_short *dPtr = (cl_short *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNSIGNED_INT16:
        {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_HALF_FLOAT:
        {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)convert_half_to_float( dPtr[ i ] );
            break;
        }

        case CL_SIGNED_INT32:
        {
            cl_int *dPtr = (cl_int *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNSIGNED_INT32:
        {
            cl_uint *dPtr = (cl_uint *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }

        case CL_UNORM_SHORT_565:
        {
            cl_ushort *dPtr = (cl_ushort*)ptr;
            tempData[ 0 ] = (T)( dPtr[ 0 ] >> 11 );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 5 ) & 63 );
            tempData[ 2 ] = (T)( dPtr[ 0 ] & 31 );
            break;
        }

#ifdef OBSOLETE_FORMAT
        case CL_UNORM_SHORT_565_REV:
        {
            unsigned short *dPtr = (unsigned short *)ptr;
            tempData[ 2 ] = (T)( dPtr[ 0 ] >> 11 );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 5 ) & 63 );
            tempData[ 0 ] = (T)( dPtr[ 0 ] & 31 );
            break;
        }

        case CL_UNORM_SHORT_555_REV:
        {
            unsigned short *dPtr = (unsigned short *)ptr;
            tempData[ 2 ] = (T)( ( dPtr[ 0 ] >> 10 ) & 31 );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 5 ) & 31 );
            tempData[ 0 ] = (T)( dPtr[ 0 ] & 31 );
            break;
        }

        case CL_UNORM_INT_8888:
        {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[ 3 ] = (T)( dPtr[ 0 ] >> 24 );
            tempData[ 2 ] = (T)( ( dPtr[ 0 ] >> 16 ) & 0xff );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 8 ) & 0xff );
            tempData[ 0 ] = (T)( dPtr[ 0 ] & 0xff );
            break;
        }
        case CL_UNORM_INT_8888_REV:
        {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[ 0 ] = (T)( dPtr[ 0 ] >> 24 );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 16 ) & 0xff );
            tempData[ 2 ] = (T)( ( dPtr[ 0 ] >> 8 ) & 0xff );
            tempData[ 3 ] = (T)( dPtr[ 0 ] & 0xff );
            break;
        }

        case CL_UNORM_INT_101010_REV:
        {
            unsigned int *dPtr = (unsigned int *)ptr;
            tempData[ 2 ] = (T)( ( dPtr[ 0 ] >> 20 ) & 0x3ff );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 10 ) & 0x3ff );
            tempData[ 0 ] = (T)( dPtr[ 0 ] & 0x3ff );
            break;
        }
#endif
        case CL_UNORM_SHORT_555:
        {
            cl_ushort *dPtr = (cl_ushort *)ptr;
            tempData[ 0 ] = (T)( ( dPtr[ 0 ] >> 10 ) & 31 );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 5 ) & 31 );
            tempData[ 2 ] = (T)( dPtr[ 0 ] & 31 );
            break;
        }

        case CL_UNORM_INT_101010:
        {
            cl_uint *dPtr = (cl_uint *)ptr;
            tempData[ 0 ] = (T)( ( dPtr[ 0 ] >> 20 ) & 0x3ff );
            tempData[ 1 ] = (T)( ( dPtr[ 0 ] >> 10 ) & 0x3ff );
            tempData[ 2 ] = (T)( dPtr[ 0 ] & 0x3ff );
            break;
        }

        case CL_FLOAT:
        {
            cl_float *dPtr = (cl_float *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ];
            break;
        }
#ifdef CL_SFIXED14_APPLE
        case CL_SFIXED14_APPLE:
        {
            cl_float *dPtr = (cl_float *)ptr;
            for( i = 0; i < get_format_channel_count( format ); i++ )
                tempData[ i ] = (T)dPtr[ i ] + 0x4000;
            break;
        }
#endif
    }


    outData[ 0 ] = outData[ 1 ] = outData[ 2 ] = 0;
    outData[ 3 ] = 1;

    if( format->image_channel_order == CL_A )
    {
        outData[ 3 ] = tempData[ 0 ];
    }
    else if( format->image_channel_order == CL_R   )
    {
        outData[ 0 ] = tempData[ 0 ];
    }
    else if( format->image_channel_order == CL_Rx   )
    {
        outData[ 0 ] = tempData[ 0 ];
    }
    else if( format->image_channel_order == CL_RA )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 3 ] = tempData[ 1 ];
    }
    else if( format->image_channel_order == CL_RG  )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 1 ] = tempData[ 1 ];
    }
    else if( format->image_channel_order == CL_RGx  )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 1 ] = tempData[ 1 ];
    }
    else if( format->image_channel_order == CL_RGB  )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 1 ] = tempData[ 1 ];
        outData[ 2 ] = tempData[ 2 ];
    }
    else if( format->image_channel_order == CL_RGBx  )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 1 ] = tempData[ 1 ];
        outData[ 2 ] = tempData[ 2 ];
    }
    else if( format->image_channel_order == CL_RGBA )
    {
        outData[ 0 ] = tempData[ 0 ];
        outData[ 1 ] = tempData[ 1 ];
        outData[ 2 ] = tempData[ 2 ];
        outData[ 3 ] = tempData[ 3 ];
    }
    else if( format->image_channel_order == CL_ARGB )
    {
        outData[ 0 ] = tempData[ 1 ];
        outData[ 1 ] = tempData[ 2 ];
        outData[ 2 ] = tempData[ 3 ];
        outData[ 3 ] = tempData[ 0 ];
    }
    else if( format->image_channel_order == CL_BGRA )
    {
        outData[ 0 ] = tempData[ 2 ];
        outData[ 1 ] = tempData[ 1 ];
        outData[ 2 ] = tempData[ 0 ];
        outData[ 3 ] = tempData[ 3 ];
    }
    else if( format->image_channel_order == CL_INTENSITY )
    {
        outData[ 1 ] = tempData[ 0 ];
        outData[ 2 ] = tempData[ 0 ];
        outData[ 3 ] = tempData[ 0 ];
    }
    else if( format->image_channel_order == CL_LUMINANCE )
    {
        outData[ 1 ] = tempData[ 0 ];
        outData[ 2 ] = tempData[ 0 ];
    }
#ifdef CL_1RGB_APPLE
    else if( format->image_channel_order == CL_1RGB_APPLE )
    {
        outData[ 0 ] = tempData[ 1 ];
        outData[ 1 ] = tempData[ 2 ];
        outData[ 2 ] = tempData[ 3 ];
        outData[ 3 ] = 0xff;
    }
#endif
#ifdef CL_BGR1_APPLE
    else if( format->image_channel_order == CL_BGR1_APPLE )
    {
        outData[ 0 ] = tempData[ 2 ];
        outData[ 1 ] = tempData[ 1 ];
        outData[ 2 ] = tempData[ 0 ];
        outData[ 3 ] = 0xff;
    }
#endif
    else
    {
        log_error("Invalid format:");
        print_header(format, true);
    }
}

// Stupid template rules
bool get_integer_coords( float x, float y, float z,
                        size_t width, size_t height, size_t depth,
                        image_sampler_data *imageSampler, image_descriptor *imageInfo,
                        int &outX, int &outY, int &outZ );
bool get_integer_coords_offset( float x, float y, float z,
                               float xAddressOffset, float yAddressOffset, float zAddressOffset,
                               size_t width, size_t height, size_t depth,
                               image_sampler_data *imageSampler, image_descriptor *imageInfo,
                               int &outX, int &outY, int &outZ );


template <class T> void sample_image_pixel_offset( void *imageData, image_descriptor *imageInfo,
                                                  float x, float y, float z, float xAddressOffset, float yAddressOffset, float zAddressOffset,
                                                  image_sampler_data *imageSampler, T *outData )
{
    int iX, iY, iZ;

    float max_w = imageInfo->width;
    float max_h;
    float max_d;

    switch (imageInfo->type) {
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            max_h = imageInfo->arraySize;
            max_d = 0;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            max_h = imageInfo->height;
            max_d = imageInfo->arraySize;
            break;
        default:
            max_h = imageInfo->height;
            max_d = imageInfo->depth;
            break;
    }

    get_integer_coords_offset( x, y, z, xAddressOffset, yAddressOffset, zAddressOffset, max_w, max_h, max_d, imageSampler, imageInfo, iX, iY, iZ );

    read_image_pixel<T>( imageData, imageInfo, iX, iY, iZ, outData );
}


template <class T> void sample_image_pixel( void *imageData, image_descriptor *imageInfo,
                                           float x, float y, float z, image_sampler_data *imageSampler, T *outData )
{
    return sample_image_pixel_offset<T>(imageData, imageInfo, x, y, z, 0.0f, 0.0f, 0.0f, imageSampler, outData);
}

FloatPixel sample_image_pixel_float( void *imageData, image_descriptor *imageInfo,
                                    float x, float y, float z, image_sampler_data *imageSampler, float *outData, int verbose, int *containsDenorms );

FloatPixel sample_image_pixel_float_offset( void *imageData, image_descriptor *imageInfo,
                                           float x, float y, float z, float xAddressOffset, float yAddressOffset, float zAddressOffset,
                                           image_sampler_data *imageSampler, float *outData, int verbose, int *containsDenorms );


extern void pack_image_pixel( unsigned int *srcVector, const cl_image_format *imageFormat, void *outData );
extern void pack_image_pixel( int *srcVector, const cl_image_format *imageFormat, void *outData );
extern void pack_image_pixel( float *srcVector, const cl_image_format *imageFormat, void *outData );
extern void pack_image_pixel_error( const float *srcVector, const cl_image_format *imageFormat, const void *results,  float *errors );

extern char *create_random_image_data( ExplicitType dataType, image_descriptor *imageInfo, BufferOwningPtr<char> &P, MTdata d );

// deprecated
// extern bool clamp_image_coord( image_sampler_data *imageSampler, float value, size_t max, int &outValue );

extern void get_sampler_kernel_code( image_sampler_data *imageSampler, char *outLine );
extern float get_max_absolute_error( cl_image_format *format, image_sampler_data *sampler);
extern float get_max_relative_error( cl_image_format *format, image_sampler_data *sampler, int is3D, int isLinearFilter );


#define errMax( _x , _y )       ( (_x) != (_x) ? (_x) : (_x) > (_y) ? (_x) : (_y) )

static inline cl_uint abs_diff_uint( cl_uint x, cl_uint y )
{
    return y > x ? y - x : x - y;
}

static inline cl_uint abs_diff_int( cl_int x, cl_int y )
{
    return (cl_uint) (y > x ? y - x : x - y);
}

static inline cl_float relative_error( float test, float expected )
{
    // 0-0/0 is 0 in this case, not NaN
    if( test == 0.0f && expected == 0.0f )
        return 0.0f;

    return (test - expected) / expected;
}

extern float random_float(float low, float high);

class CoordWalker
{
public:
    CoordWalker( void * coords, bool useFloats, size_t vecSize );
    ~CoordWalker();

    cl_float    Get( size_t idx, size_t el );

protected:
    cl_float * mFloatCoords;
    cl_int * mIntCoords;
    size_t    mVecSize;
};

extern int  DetectFloatToHalfRoundingMode( cl_command_queue );  // Returns CL_SUCCESS on success

int inline is_half_nan( cl_ushort half ){ return (half & 0x7fff) > 0x7c00; }

cl_ushort convert_float_to_half( cl_float f );
cl_float  convert_half_to_float( cl_ushort h );


#endif // _image_helpers_h


