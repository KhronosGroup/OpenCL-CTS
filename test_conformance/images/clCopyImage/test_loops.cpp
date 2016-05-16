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
#include "../common.h"

extern cl_filter_mode     gFilterModeToUse;
extern cl_addressing_mode gAddressModeToUse;
extern int                gTypesToTest;
extern int                gNormalizedModeToUse;
extern bool               gTestMipmaps;
extern cl_channel_type      gChannelTypeToUse;
extern cl_channel_type      gChannelTypeToUse;
extern cl_channel_order      gChannelOrderToUse;


extern bool gDebugTrace;

extern int test_copy_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format );
extern int test_copy_image_set_2D_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );
extern int test_copy_image_set_2D_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );
extern int test_copy_image_set_3D_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool reverse );

int test_image_type( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod, cl_mem_flags flags )
{
    const char *name;
    cl_mem_object_type imageType;

    if ( gTestMipmaps )
    {
        if ( 0 == is_extension_available( device, "cl_khr_mipmap_image" ))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image.\nSkipping mipmapped image test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return 0;
        }
    }

    if( testMethod == k1D )
    {
        name = "1D -> 1D";
        imageType = CL_MEM_OBJECT_IMAGE1D;
    }
    else if( testMethod == k2D )
    {
        name = "2D -> 2D";
        imageType = CL_MEM_OBJECT_IMAGE2D;
    }
    else if( testMethod == k3D )
    {
        name = "3D -> 3D";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }
    else if( testMethod == k1DArray )
    {
        name = "1D array -> 1D array";
        imageType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    }
    else if( testMethod == k2DArray )
    {
        name = "2D array -> 2D array";
        imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
    else if( testMethod == k2DTo3D )
    {
        name = "2D -> 3D";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }
    else if( testMethod == k3DTo2D )
    {
        name = "3D -> 2D";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }
    else if( testMethod == k2DArrayTo2D )
    {
        name = "2D array -> 2D";
        imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
    else if( testMethod == k2DTo2DArray )
    {
        name = "2D -> 2D array";
        imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
    else if( testMethod == k2DArrayTo3D )
    {
        name = "2D array -> 3D";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }
    else if( testMethod == k3DTo2DArray )
    {
        name = "3D -> 2D array";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }

    if(gTestMipmaps)
        log_info( "Running mipmapped %s tests...\n", name );
    else
        log_info( "Running %s tests...\n", name );

    int ret = 0;

    // Grab the list of supported image formats for integer reads
    cl_image_format *formatList;
    bool *filterFlags;
    unsigned int numFormats;

    if( get_format_list( context, imageType, formatList, numFormats, flags ) )
        return -1;

    filterFlags = new bool[ numFormats ];
    if( filterFlags == NULL )
    {
        log_error( "ERROR: Out of memory allocating filter flags list!\n" );
        return -1;
    }
    memset( filterFlags, 0, sizeof( bool ) * numFormats );

    filter_formats(formatList, filterFlags, numFormats, NULL);

    // Run the format list
    for( unsigned int i = 0; i < numFormats; i++ )
    {
        int test_return = 0;
        if( filterFlags[i] )
        {
            continue;
        }

        print_header( &formatList[ i ], false );

        gTestCount++;

        if( testMethod == k1D )
            test_return = test_copy_image_set_1D( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2D )
            test_return = test_copy_image_set_2D( device, context, queue, &formatList[ i ] );
        else if( testMethod == k3D )
            test_return = test_copy_image_set_3D( device, context, queue,&formatList[ i ] );
        else if( testMethod == k1DArray )
            test_return = test_copy_image_set_1D_array( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2DArray )
            test_return = test_copy_image_set_2D_array( device, context, queue, &formatList[ i ] );
        else if( testMethod == k2DTo3D )
            test_return = test_copy_image_set_2D_3D( device, context, queue, &formatList[ i ], false );
        else if( testMethod == k3DTo2D )
            test_return = test_copy_image_set_2D_3D( device, context, queue, &formatList[ i ], true );
        else if( testMethod == k2DArrayTo2D)
            test_return = test_copy_image_set_2D_2D_array( device, context, queue, &formatList[ i ], true);
        else if( testMethod == k2DTo2DArray)
            test_return = test_copy_image_set_2D_2D_array( device, context, queue, &formatList[ i ], false);
        else if( testMethod == k2DArrayTo3D)
            test_return = test_copy_image_set_3D_2D_array( device, context, queue, &formatList[ i ], true);
        else if( testMethod == k3DTo2DArray)
            test_return = test_copy_image_set_3D_2D_array( device, context, queue, &formatList[ i ], false);

        if (test_return) {
            gFailCount++;
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

int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod )
{
    int ret = 0;

    ret += test_image_type( device, context, queue, testMethod, CL_MEM_READ_ONLY );

    return ret;
}




