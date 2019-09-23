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

extern bool               gDebugTrace;
extern cl_filter_mode     gFilterModeToUse;
extern cl_addressing_mode gAddressModeToUse;
extern int                gTypesToTest;
extern int                gNormalizedModeToUse;
extern cl_channel_type    gChannelTypeToUse;
extern cl_channel_order   gChannelOrderToUse;


extern int test_fill_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );
extern int test_fill_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType outputType );


int filter_formats( cl_image_format *formatList, bool *filterFlags, unsigned int formatCount, cl_channel_type *channelDataTypesToFilter )
{
    int numSupported = 0;
    for ( unsigned int j = 0; j < formatCount; j++ )
    {
        // If this format has been previously filtered, remove the filter
        if ( filterFlags[ j ] )
            filterFlags[ j ] = false;

        // Have we already discarded this via the command line?
        if ( gChannelTypeToUse != (cl_channel_type)-1 && gChannelTypeToUse != formatList[ j ].image_channel_data_type )
        {
            filterFlags[ j ] = true;
            continue;
        }

    // Have we already discarded the channel order via the command line?
        if ( gChannelOrderToUse != (cl_channel_order)-1 && gChannelOrderToUse != formatList[ j ].image_channel_order )
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
        for ( i = 0; channelDataTypesToFilter[ i ] != (cl_channel_type)-1; i++ )
        {
            if ( formatList[ j ].image_channel_data_type == channelDataTypesToFilter[ i ] )
            {
                numSupported++;
                break;
            }
        }
        if ( channelDataTypesToFilter[ i ] == (cl_channel_type)-1 )
        {
            // Format is NOT supported, so mark it as such
            filterFlags[ j ] = true;
        }


    }
    return numSupported;
}


int get_format_list( cl_context context, cl_mem_object_type image_type, cl_image_format * &outFormatList,
                    unsigned int &outFormatCount, cl_mem_flags flags )
{
    int error;

    cl_image_format tempList[ 128 ];
    error = clGetSupportedImageFormats( context, (cl_mem_flags)flags,
                                       image_type, 128, tempList, &outFormatCount );
    test_error( error, "Unable to get count of supported image formats" );

    outFormatList = new cl_image_format[ outFormatCount ];
    error = clGetSupportedImageFormats( context, (cl_mem_flags)flags,
                                       image_type, outFormatCount, outFormatList, NULL );
    test_error( error, "Unable to get list of supported image formats" );
    return 0;
}


int test_image_type( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod, cl_mem_flags flags )
{
    const char *name;
    cl_mem_object_type imageType;

    if ( testMethod == k1D )
    {
        name = "1D Image Fill";
        imageType = CL_MEM_OBJECT_IMAGE1D;
    }
    else if ( testMethod == k2D )
    {
        name = "2D Image Fill";
        imageType = CL_MEM_OBJECT_IMAGE2D;
    }
    else if ( testMethod == k1DArray )
    {
        name = "1D Image Array Fill";
        imageType = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    }
    else if ( testMethod == k2DArray )
    {
        name = "2D Image Array Fill";
        imageType = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    }
    else if ( testMethod == k3D )
    {
        name = "3D Image Fill";
        imageType = CL_MEM_OBJECT_IMAGE3D;
    }

    log_info( "Running %s tests...\n", name );

    int ret = 0;

    // Grab the list of supported image formats
    cl_image_format *formatList;
    bool *filterFlags;
    unsigned int numFormats;

    if ( get_format_list( context, imageType, formatList, numFormats, flags ) )
        return -1;

    filterFlags = new bool[ numFormats ];
    if ( filterFlags == NULL )
    {
        log_error( "ERROR: Out of memory allocating filter flags list!\n" );
        return -1;
    }
    memset( filterFlags, 0, sizeof( bool ) * numFormats );

    /////// float tests ///////

    if( gTypesToTest & kTestFloat )
    {
        cl_channel_type floatFormats[] = { CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
#ifdef OBSOLETE_FORAMT
            CL_UNORM_SHORT_565_REV, CL_UNORM_SHORT_555_REV, CL_UNORM_INT_8888, CL_UNORM_INT_8888_REV, CL_UNORM_INT_101010_REV,
#endif
#ifdef CL_SFIXED14_APPLE
            CL_SFIXED14_APPLE,
#endif
            CL_UNORM_INT8, CL_SNORM_INT8,
            CL_UNORM_INT16, CL_SNORM_INT16, CL_FLOAT, CL_HALF_FLOAT, (cl_channel_type)-1 };
        if( filter_formats( formatList, filterFlags, numFormats, floatFormats ) == 0 )
        {
            log_info( "No formats supported for float type\n" );
        }
        else
        {
            // Run the format list
            for ( unsigned int i = 0; i < numFormats; i++ )
            {
                int test_return = 0;
                if ( filterFlags[i] )
                {
                    continue;
                }

                print_header( &formatList[ i ], false );

                gTestCount++;

                if ( testMethod == k1D )
                    test_return = test_fill_image_set_1D( device, context, queue, &formatList[ i ], kFloat );
                else if ( testMethod == k2D )
                    test_return = test_fill_image_set_2D( device, context, queue, &formatList[ i ], kFloat );
                else if ( testMethod == k1DArray )
                    test_return = test_fill_image_set_1D_array( device, context, queue, &formatList[ i ], kFloat );
                else if ( testMethod == k2DArray )
                    test_return = test_fill_image_set_2D_array( device, context, queue, &formatList[ i ], kFloat );
                else if ( testMethod == k3D )
                    test_return = test_fill_image_set_3D( device, context, queue, &formatList[ i ], kFloat );

                if (test_return)
                {
                    gFailCount++;
                    log_error( "FAILED: " );
                    print_header( &formatList[ i ], true );
                    log_info( "\n" );
                }

                ret += test_return;
            }
        }
    }

    /////// int tests ///////
    if( gTypesToTest & kTestInt )
    {
        cl_channel_type intFormats[] = { CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, (cl_channel_type)-1 };
        if( filter_formats( formatList, filterFlags, numFormats, intFormats ) == 0 )
        {
            log_info( "No formats supported for integer type\n" );
        }
        else
        {
            // Run the format list
            for ( unsigned int i = 0; i < numFormats; i++ )
            {
                int test_return = 0;
                if ( filterFlags[i] )
                {
                    continue;
                }

                print_header( &formatList[ i ], false );

                gTestCount++;

                if ( testMethod == k1D )
                    test_return = test_fill_image_set_1D( device, context, queue, &formatList[ i ], kInt );
                else if ( testMethod == k2D )
                    test_return = test_fill_image_set_2D( device, context, queue, &formatList[ i ], kInt );
                else if ( testMethod == k1DArray )
                    test_return = test_fill_image_set_1D_array( device, context, queue, &formatList[ i ], kInt );
                else if ( testMethod == k2DArray )
                    test_return = test_fill_image_set_2D_array( device, context, queue, &formatList[ i ], kInt );
                else if ( testMethod == k3D )
                    test_return = test_fill_image_set_3D( device, context, queue, &formatList[ i ], kInt );

                if (test_return) {
                    gFailCount++;
                    log_error( "FAILED: " );
                    print_header( &formatList[ i ], true );
                    log_info( "\n" );
                }

                ret += test_return;
            }
        }
    }

    /////// uint tests ///////

    if( gTypesToTest & kTestUInt )
    {
        cl_channel_type uintFormats[] = { CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, (cl_channel_type)-1 };
        if( filter_formats( formatList, filterFlags, numFormats, uintFormats ) == 0 )
        {
            log_info( "No formats supported for unsigned int type\n" );
        }
        else
        {
            // Run the format list
            for ( unsigned int i = 0; i < numFormats; i++ )
            {
                int test_return = 0;
                if ( filterFlags[i] )
                {
                    continue;
                }

                print_header( &formatList[ i ], false );

                gTestCount++;

                if ( testMethod == k1D )
                    test_return = test_fill_image_set_1D( device, context, queue, &formatList[ i ], kUInt );
                else if ( testMethod == k2D )
                    test_return = test_fill_image_set_2D( device, context, queue, &formatList[ i ], kUInt );
                else if ( testMethod == k1DArray )
                    test_return = test_fill_image_set_1D_array( device, context, queue, &formatList[ i ], kUInt );
                else if ( testMethod == k2DArray )
                    test_return = test_fill_image_set_2D_array( device, context, queue, &formatList[ i ], kUInt );
                else if ( testMethod == k3D )
                    test_return = test_fill_image_set_3D( device, context, queue, &formatList[ i ], kUInt );

                if (test_return) {
                    gFailCount++;
                    log_error( "FAILED: " );
                    print_header( &formatList[ i ], true );
                    log_info( "\n" );
                }

                ret += test_return;
            }
        }
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
