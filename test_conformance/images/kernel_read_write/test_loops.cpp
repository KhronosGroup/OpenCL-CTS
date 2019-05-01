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
extern cl_channel_order      gChannelOrderToUse;

extern bool gDebugTrace;
extern bool gTestMipmaps;

extern int  gtestTypesToRun;

extern int test_read_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue,  cl_image_format *format, image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType );
extern int test_read_image_set_2D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType );
extern int test_read_image_set_3D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                                  bool floatCoords, ExplicitType outputType );
extern int test_read_image_set_1D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                                        bool floatCoords, ExplicitType outputType );
extern int test_read_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                                        bool floatCoords, ExplicitType outputType );

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

        // skip mipmap tests for CL_DEPTH formats (re# Khronos Bug 13762)
        if(gTestMipmaps && (formatList[ j ].image_channel_order == CL_DEPTH))
        {
            log_info("Skip mipmap tests for CL_DEPTH format\n");
            filterFlags[ j ] = true;
            continue;
        }

        // Have we already discarded the channel type via the command line?
        if( gChannelTypeToUse != (cl_channel_type)-1 && gChannelTypeToUse != formatList[ j ].image_channel_data_type )
        {
            filterFlags[ j ] = true;
            continue;
        }

        // Have we already discarded the channel order via the command line?
        if( gChannelOrderToUse != (cl_channel_order)-1 && gChannelOrderToUse != formatList[ j ].image_channel_order )
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

        if ( !channelDataTypesToFilter )
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

int test_read_image_type( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, bool floatCoords,
                         image_sampler_data *imageSampler, ExplicitType outputType, cl_mem_object_type imageType )
{
    int ret = 0;
    cl_addressing_mode *addressModes = NULL;

    // The sampler-less read image functions behave exactly as the corresponding read image functions
    // described in section 6.13.14.2 that take integer coordinates and a sampler with filter mode set to
    // CLK_FILTER_NEAREST, normalized coordinates set to CLK_NORMALIZED_COORDS_FALSE and addressing mode to CLK_ADDRESS_NONE
    cl_addressing_mode addressModes_rw[] = { CL_ADDRESS_NONE, (cl_addressing_mode)-1 };
    cl_addressing_mode addressModes_ro[] = { /* CL_ADDRESS_CLAMP_NONE,*/ CL_ADDRESS_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP, CL_ADDRESS_REPEAT, CL_ADDRESS_MIRRORED_REPEAT, (cl_addressing_mode)-1 };

    if(gtestTypesToRun & kReadWriteTests)
    {
        addressModes = addressModes_rw;
    }
    else
    {
        addressModes = addressModes_ro;
    }

#if defined( __APPLE__ )
    // According to the OpenCL specification, we do not guarantee the precision
    // of operations for linear filtering on the GPU.  We do not test linear
    // filtering for the CL_RGB CL_UNORM_INT_101010 image format; however, we
    // test it internally for a set of other image formats.
    if ((gDeviceType == CL_DEVICE_TYPE_GPU) &&
        (imageSampler->filter_mode == CL_FILTER_LINEAR) &&
        (format->image_channel_order == CL_RGB) &&
        (format->image_channel_data_type == CL_UNORM_INT_101010))
    {
        log_info("--- Skipping CL_RGB CL_UNORM_INT_101010 format with CL_FILTER_LINEAR on GPU.\n");
        return 0;
    }
#endif

    for( int adMode = 0; addressModes[ adMode ] != (cl_addressing_mode)-1; adMode++ )
    {
        imageSampler->addressing_mode = addressModes[ adMode ];

        if( (addressModes[ adMode ] == CL_ADDRESS_REPEAT || addressModes[ adMode ] == CL_ADDRESS_MIRRORED_REPEAT) && !( imageSampler->normalized_coords ) )
            continue; // Repeat doesn't make sense for non-normalized coords

        // Use this run if we were told to only run a certain filter mode
        if( gAddressModeToUse != (cl_addressing_mode)-1 && imageSampler->addressing_mode != gAddressModeToUse )
            continue;

        /*
         Remove redundant check to see if workaround still necessary
         // Check added in because this case was leaking through causing a crash on CPU
         if( ! imageSampler->normalized_coords && imageSampler->addressing_mode == CL_ADDRESS_REPEAT )
         continue;       //repeat mode requires normalized coordinates
         */
        print_read_header( format, imageSampler, false );

        gTestCount++;

        int retCode = 0;
        switch (imageType)
        {
            case CL_MEM_OBJECT_IMAGE1D:
                retCode = test_read_image_set_1D( device, context, queue, format, imageSampler, floatCoords, outputType );
                break;
            case CL_MEM_OBJECT_IMAGE1D_ARRAY:
                retCode = test_read_image_set_1D_array( device, context, queue, format, imageSampler, floatCoords, outputType );
                break;
            case CL_MEM_OBJECT_IMAGE2D:
                retCode = test_read_image_set_2D( device, context, queue, format, imageSampler, floatCoords, outputType );
                break;
            case CL_MEM_OBJECT_IMAGE2D_ARRAY:
                retCode = test_read_image_set_2D_array( device, context, queue, format, imageSampler, floatCoords, outputType );
                break;
            case CL_MEM_OBJECT_IMAGE3D:
                retCode = test_read_image_set_3D( device, context, queue, format, imageSampler, floatCoords, outputType );
                break;
        }
        if( retCode != 0 )
        {
            gTestFailure++;
            log_error( "FAILED: " );
            print_read_header( format, imageSampler, true );
            log_info( "\n" );
        }
        ret |= retCode;
    }

    return ret;
}

int test_read_image_formats( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *formatList, bool *filterFlags, unsigned int numFormats,
                            image_sampler_data *imageSampler, ExplicitType outputType, cl_mem_object_type imageType )
{
    int ret = 0;
    bool flipFlop[2] = { false, true };
    int normalizedIdx, floatCoordIdx;


    // Use this run if we were told to only run a certain filter mode
    if( gFilterModeToUse != (cl_filter_mode)-1 && imageSampler->filter_mode != gFilterModeToUse )
        return 0;

    // Test normalized/non-normalized
    for( normalizedIdx = 0; normalizedIdx < 2; normalizedIdx++ )
    {
        imageSampler->normalized_coords = flipFlop[ normalizedIdx ];
        if( gNormalizedModeToUse != 7 && gNormalizedModeToUse != (int)imageSampler->normalized_coords )
            continue;

        for( floatCoordIdx = 0; floatCoordIdx < 2; floatCoordIdx++ )
        {
            // Checks added in because this case was leaking through causing a crash on CPU
            if( !flipFlop[ floatCoordIdx ] )
                if( imageSampler->filter_mode != CL_FILTER_NEAREST      ||  // integer coords can only be used with nearest
                   flipFlop[ normalizedIdx ])                               // Normalized integer coords makes no sense (they'd all be zero)
                    continue;

            if( flipFlop[ floatCoordIdx ] && (gtestTypesToRun & kReadWriteTests))
                // sampler-less read in read_write tests run only integer coord
                continue;


            log_info( "read_image (%s coords, %s results) *****************************\n",
                     flipFlop[ floatCoordIdx ] ? ( imageSampler->normalized_coords ? "normalized float" : "unnormalized float" ) : "integer",
                     get_explicit_type_name( outputType ) );

            for( unsigned int i = 0; i < numFormats; i++ )
            {
                if( filterFlags[i] )
                    continue;

                cl_image_format &imageFormat = formatList[ i ];

                ret |= test_read_image_type( device, context, queue, &imageFormat, flipFlop[ floatCoordIdx ], imageSampler, outputType, imageType );
            }
        }
    }
    return ret;
}


int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, test_format_set_fn formatTestFn, cl_mem_object_type imageType )
{
    int ret = 0;
    static int printedFormatList = -1;


    if ( ( 0 == is_extension_available( device, "cl_khr_3d_image_writes" )) && (imageType == CL_MEM_OBJECT_IMAGE3D) && (formatTestFn == test_write_image_formats) )
    {
        gTestFailure++;
        log_error( "-----------------------------------------------------\n" );
        log_error( "FAILED: test writing CL_MEM_OBJECT_IMAGE3D images\n" );
        log_error( "This device does not support the mandated extension cl_khr_3d_image_writes.\n");
        log_error( "-----------------------------------------------------\n\n" );
        return -1;
    }

    if ( gTestMipmaps )
    {
        if ( 0 == is_extension_available( device, "cl_khr_mipmap_image" ))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image.\nSkipping mipmapped image test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return 0;
        }
        if ( ( 0 == is_extension_available( device, "cl_khr_mipmap_image_writes" )) && (formatTestFn == test_write_image_formats))
        {
            log_info( "-----------------------------------------------------\n" );
            log_info( "This device does not support cl_khr_mipmap_image_writes.\nSkipping mipmapped image write test. \n" );
            log_info( "-----------------------------------------------------\n\n" );
            return 0;
        }
    }

    int version_check = check_opencl_version(device,1,2);
    if (version_check != 0) {
      switch (imageType) {
        case CL_MEM_OBJECT_IMAGE1D:
          test_missing_feature(version_check, "image_1D");
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
          test_missing_feature(version_check, "image_1D_array");
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
          test_missing_feature(version_check, "image_2D_array");
      }
    }

    // Grab the list of supported image formats for integer reads
    cl_image_format *formatList;
    bool *filterFlags;
    unsigned int numFormats;

    // This flag is only for querying the list of supported formats
    // The flag for creating image will be set explicitly in test functions
    cl_mem_flags flags;
    const char *flagNames;
    if( formatTestFn == test_read_image_formats )
    {
        if(gtestTypesToRun & kReadTests)
        {
            flags = CL_MEM_READ_ONLY;
            flagNames = "read";
        }
        else
        {
            flags = CL_MEM_KERNEL_READ_AND_WRITE;
            flagNames = "read_write";
        }
    }
    else
    {
        if(gtestTypesToRun & kWriteTests)
        {
            flags = CL_MEM_WRITE_ONLY;
            flagNames = "write";
        }
        else
        {
            flags = CL_MEM_KERNEL_READ_AND_WRITE;
            flagNames = "read_write";
        }
    }

    if( get_format_list( context, imageType, formatList, numFormats, flags ) )
        return -1;
    BufferOwningPtr<cl_image_format> formatListBuf(formatList);


    filterFlags = new bool[ numFormats ];
    if( filterFlags == NULL )
    {
        log_error( "ERROR: Out of memory allocating filter flags list!\n" );
        return -1;
    }
    BufferOwningPtr<bool> filterFlagsBuf(filterFlags);
    memset( filterFlags, 0, sizeof( bool ) * numFormats );

    // First time through, we'll go ahead and print the formats supported, regardless of type
    int test = imageType | (formatTestFn == test_read_image_formats ? (1 << 16) : (1 << 17));
    if( printedFormatList != test )
    {
        log_info( "---- Supported %s %s formats for this device ---- \n", convert_image_type_to_string(imageType), flagNames );
        for( unsigned int f = 0; f < numFormats; f++ )
        {
            if ( IsChannelOrderSupported( formatList[ f ].image_channel_order ) && IsChannelTypeSupported( formatList[ f ].image_channel_data_type ) )
                log_info( "  %-7s %-24s %d\n", GetChannelOrderName( formatList[ f ].image_channel_order ),
                        GetChannelTypeName( formatList[ f ].image_channel_data_type ),
                        (int)get_format_channel_count( &formatList[ f ] ) );
        }
        log_info( "------------------------------------------- \n" );
        printedFormatList = test;
    }

    image_sampler_data imageSampler;

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
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += formatTestFn( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kFloat, imageType );

            imageSampler.filter_mode = CL_FILTER_LINEAR;
            ret += formatTestFn( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kFloat, imageType );
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
            // Only filter mode we support on int is nearest
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += formatTestFn( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kInt, imageType );
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
            // Only filter mode we support on uint is nearest
            imageSampler.filter_mode = CL_FILTER_NEAREST;
            ret += formatTestFn( device, context, queue, formatList, filterFlags, numFormats, &imageSampler, kUInt, imageType );
        }
    }
    return ret;
}
