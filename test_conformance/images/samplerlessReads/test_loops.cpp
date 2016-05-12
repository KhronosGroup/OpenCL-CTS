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

extern int                  gTypesToTest;
extern cl_channel_type      gChannelTypeToUse;
extern cl_channel_order     gChannelOrderToUse;

extern bool                 gDebugTrace;

extern bool                 gTestReadWrite;

extern int test_read_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );
extern int test_read_image_set_1D_buffer( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );
extern int test_read_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );
extern int test_read_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );
extern int test_read_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );
extern int test_read_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler, ExplicitType outputType );

static const char *str_1d_image = "1D";
static const char *str_2d_image = "2D";
static const char *str_3d_image = "3D";
static const char *str_1d_image_array = "1D array";
static const char *str_2d_image_array = "2D array";
static const char *str_1d_image_buffer = "1D image buffer";

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
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            p = str_1d_image_buffer;
    }
    return p;
}


int filter_formats( cl_image_format *formatList, bool *filterFlags, unsigned int formatCount, cl_channel_type *channelDataTypesToFilter )
{
    int numSupported = 0;
    for ( unsigned int j = 0; j < formatCount; j++ )
    {
        // If this format has been previously filtered, remove the filter
        if ( filterFlags[ j ] )
            filterFlags[ j ] = false;

        // Have we already discarded the channel type via the command line?
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

int get_format_list( cl_context context, cl_mem_object_type imageType, cl_image_format * &outFormatList, unsigned int &outFormatCount, cl_mem_flags flags )
{
    int error;

    cl_image_format tempList[ 128 ];
    error = clGetSupportedImageFormats( context, flags,
                                        imageType, 128, tempList, &outFormatCount );
    test_error( error, "Unable to get count of supported image formats" );

    outFormatList = new cl_image_format[ outFormatCount ];
    error = clGetSupportedImageFormats( context, flags,
                                        imageType, outFormatCount, outFormatList, NULL );
    test_error( error, "Unable to get list of supported image formats" );

    return 0;
}

int test_read_image_type( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format,
                          image_sampler_data *imageSampler, ExplicitType outputType, cl_mem_object_type imageType )
{
    int ret = 0;
    imageSampler->addressing_mode = CL_ADDRESS_NONE;

    print_read_header( format, imageSampler, false );

    gTestCount++;

    switch (imageType)
    {
        case CL_MEM_OBJECT_IMAGE1D:
            ret = test_read_image_set_1D( device, context, queue, format, imageSampler, outputType );
            break;
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            ret += test_read_image_set_1D_buffer( device, context, queue, format, imageSampler, outputType );
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            ret = test_read_image_set_2D( device, context, queue, format, imageSampler, outputType );
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            ret = test_read_image_set_3D( device, context, queue, format, imageSampler, outputType );
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
            ret = test_read_image_set_1D_array( device, context, queue, format, imageSampler, outputType );
            break;
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            ret = test_read_image_set_2D_array( device, context, queue, format, imageSampler, outputType );
            break;
    }

    if ( ret != 0 )
    {
        gTestFailure++;
        log_error( "FAILED: " );
        print_read_header( format, imageSampler, true );
        log_info( "\n" );
    }
    return ret;
}

int test_read_image_formats( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *formatList, bool *filterFlags, unsigned int numFormats,
                             image_sampler_data *imageSampler, ExplicitType outputType, cl_mem_object_type imageType )
{
    int ret = 0;
    imageSampler->normalized_coords = false;
    log_info( "read_image (%s coords, %s results) *****************************\n",
              "integer", get_explicit_type_name( outputType ) );

    for ( unsigned int i = 0; i < numFormats; i++ )
    {
        if ( filterFlags[i] )
            continue;

        cl_image_format &imageFormat = formatList[ i ];

        ret |= test_read_image_type( device, context, queue, &imageFormat, imageSampler, outputType, imageType );
    }
    return ret;
}


int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type imageType )
{
    int ret = 0;
    static int printedFormatList = -1;

    // Grab the list of supported image formats
    cl_image_format *formatList;
    bool *filterFlags;
    unsigned int numFormats;

    // This flag is only for querying the list of supported formats
    // The flag for creating image will be set explicitly in test functions
    cl_mem_flags flags = (gTestReadWrite)? CL_MEM_KERNEL_READ_AND_WRITE : CL_MEM_READ_ONLY;

    if ( get_format_list( context, imageType, formatList, numFormats, flags ) )
        return -1;

    filterFlags = new bool[ numFormats ];
    if ( filterFlags == NULL )
    {
        log_error( "ERROR: Out of memory allocating filter flags list!\n" );
        return -1;
    }
    memset( filterFlags, 0, sizeof( bool ) * numFormats );

    // First time through, we'll go ahead and print the formats supported, regardless of type
    if ( printedFormatList != (int)imageType )
    {
        log_info( "---- Supported %s read formats for this device ---- \n", convert_image_type_to_string(imageType) );
        for ( unsigned int f = 0; f < numFormats; f++ )
            log_info( "  %-7s %-24s %d\n", GetChannelOrderName( formatList[ f ].image_channel_order ),
                      GetChannelTypeName( formatList[ f ].image_channel_data_type ),
                      (int)get_format_channel_count( &formatList[ f ] ) );
        log_info( "------------------------------------------- \n" );
        printedFormatList = imageType;
    }

    image_sampler_data imageSampler;

    /////// float tests ///////

    if ( gTypesToTest & kTestFloat )
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
        if ( filter_formats( formatList, filterFlags, numFormats, floatFormats ) == 0 )
        {
            log_info( "No formats supported for float type\n" );
        }
        else
        {
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += test_read_image_formats( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kFloat, imageType );
        }
    }

    /////// int tests ///////
    if ( gTypesToTest & kTestInt )
    {
        cl_channel_type intFormats[] = { CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, (cl_channel_type)-1 };
        if ( filter_formats( formatList, filterFlags, numFormats, intFormats ) == 0 )
        {
            log_info( "No formats supported for integer type\n" );
        }
        else
        {
            // Only filter mode we support on int is nearest
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += test_read_image_formats( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kInt, imageType );
        }
    }

    /////// uint tests ///////

    if ( gTypesToTest & kTestUInt )
    {
        cl_channel_type uintFormats[] = { CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, (cl_channel_type)-1 };
        if ( filter_formats( formatList, filterFlags, numFormats, uintFormats ) == 0 )
        {
            log_info( "No formats supported for unsigned int type\n" );
        }
        else
        {
            // Only filter mode we support on uint is nearest
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += test_read_image_formats( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kUInt, imageType );
        }
    }


    delete[] filterFlags;
    delete[] formatList;

    return ret;
}
