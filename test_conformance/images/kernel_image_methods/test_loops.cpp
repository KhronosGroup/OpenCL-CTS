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

extern cl_filter_mode     gFilterModeToUse;
extern cl_addressing_mode gAddressModeToUse;
extern int                gTypesToTest;
extern int                gNormalizedModeToUse;
extern cl_channel_type      gChannelTypeToUse;


extern bool gDebugTrace;

extern int test_get_image_info_1D( cl_device_id device, cl_image_format *format );
extern int test_get_image_info_2D( cl_device_id device, cl_image_format *format );
extern int test_get_image_info_3D( cl_device_id device, cl_image_format *format );
extern int test_get_image_info_1D_array( cl_device_id device, cl_image_format *format );
extern int test_get_image_info_2D_array( cl_device_id device, cl_image_format *format );

static const char *str_1d_image = "1D";
static const char *str_2d_image = "2D";
static const char *str_3d_image = "3D";
static const char *str_1d_image_array = "1D array";
static const char *str_2d_image_array = "2D array";

static const char *convert_image_type_to_string(cl_mem_object_type imageType)
{
    const char *p;
    switch (imageType)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            p = str_1d_image;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            p = str_2d_image;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            p = str_3d_image;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            p = str_1d_image_array;
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            p = str_2d_image_array;
            break;
    }
    return p;
}


int filter_formats( cl_image_format *formatList, bool *filterFlags, unsigned int formatCount, cl_channel_type *channelDataTypesToFilter )
{
    int numSupported = 0;
    for( unsigned int j = 0; j < formatCount; j++ )
    {
        // If this format has been previously filtered, remove the filter
        if( filterFlags[ j ] )
            filterFlags[ j ] = false;

        // Have we already discarded this via the command line?
        if( gChannelTypeToUse != (cl_channel_type)-1 && gChannelTypeToUse != formatList[ j ].image_channel_data_type )
        {
            filterFlags[ j ] = true;
            continue;
        }

        // Is given format standard channel order and type given by spec. We don't want to test it if this is vendor extension
        if( !IsChannelOrderSupported( formatList[ j ].image_channel_order ) || !IsChannelTypeSupported( formatList[ j ].image_channel_data_type ) )
        {
            filterFlags[ j ] = true;
            continue;
        }

        // We don't filter by channel type
        if( !channelDataTypesToFilter )
        {
            numSupported++;
            continue;
        }

        // Is the format supported?
        int i;
        for( i = 0; channelDataTypesToFilter[ i ] != (cl_channel_type)-1; i++ )
        {
            if( formatList[ j ].image_channel_data_type == channelDataTypesToFilter[ i ] )
            {
                numSupported++;
                break;
            }
        }
        if( channelDataTypesToFilter[ i ] == (cl_channel_type)-1 )
        {
            // Format is NOT supported, so mark it as such
            filterFlags[ j ] = true;
        }
    }
    return numSupported;
}

int get_format_list( cl_device_id device, cl_mem_object_type imageType, cl_image_format * &outFormatList, unsigned int &outFormatCount, cl_mem_flags flags )
{
    extern clContextWrapper context;
    int error = clGetSupportedImageFormats( context, (cl_mem_flags)flags,
                                       imageType, 0, NULL, &outFormatCount );
    test_error( error, "Unable to get count of supported image formats" );

    outFormatList = new cl_image_format[ outFormatCount ];
    error = clGetSupportedImageFormats( context, (cl_mem_flags)flags,
                                        imageType, outFormatCount, outFormatList, NULL );
    test_error( error, "Unable to get list of supported image formats" );
    return 0;
}


int test_image_type( cl_device_id device, cl_mem_object_type imageType, cl_mem_flags flags )
{
    log_info( "Running %s %s-only tests...\n", convert_image_type_to_string(imageType), flags == CL_MEM_READ_ONLY ? "read" : "write" );

    int ret = 0;

    // Grab the list of supported image formats for integer reads
    cl_image_format *formatList;
    bool *filterFlags;
    unsigned int numFormats;

    if( get_format_list( device, imageType, formatList, numFormats, flags ) )
        return -1;

    filterFlags = new bool[ numFormats ];
    if( filterFlags == NULL )
    {
        log_error( "ERROR: Out of memory allocating filter flags list!\n" );
        return -1;
    }
    memset( filterFlags, 0, sizeof( bool ) * numFormats );
    filter_formats( formatList, filterFlags, numFormats, 0 );

    // Run the format list
    for( unsigned int i = 0; i < numFormats; i++ )
    {
        int test_return = 0;
        if( filterFlags[i] )
        {
            log_info( "NOT RUNNING: " );
            print_header( &formatList[ i ], false );
            continue;
        }

        print_header( &formatList[ i ], false );

        gTestCount++;

        switch (imageType) {
            case CL_MEM_OBJECT_IMAGE1D:
                test_return = test_get_image_info_1D( device, &formatList[ i ] );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                test_return = test_get_image_info_2D( device, &formatList[ i ] );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                test_return = test_get_image_info_3D( device, &formatList[ i ] );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                test_return = test_get_image_info_1D_array( device, &formatList[ i ] );
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                test_return = test_get_image_info_2D_array( device, &formatList[ i ] );
                break;
        }

        if (test_return) {
            gTestFailure++;
            log_error( "FAILED: " );
            print_header( &formatList[ i ], true );
            log_info( "\n" );
        }

        ret += test_return;
    }

    delete filterFlags;
    delete formatList;

    return ret;
}

int test_image_set( cl_device_id device, cl_mem_object_type imageType )
{
    int version_check;
  if ((version_check = check_opencl_version(device,1,2))) {
    switch (imageType) {
      case CL_MEM_OBJECT_IMAGE1D:
        test_missing_feature(version_check, "image_1D");
      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        test_missing_feature(version_check, "image_1D_array");
      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
        test_missing_feature(version_check, "image_2D_array");
    }
  }

  int ret = 0;
    ret += test_image_type( device, imageType, CL_MEM_READ_ONLY );
    ret += test_image_type( device, imageType, CL_MEM_WRITE_ONLY );

    return ret;
}
