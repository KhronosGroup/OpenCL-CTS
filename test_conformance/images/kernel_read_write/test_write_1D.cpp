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

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

extern bool            gDebugTrace, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages, gTestMipmaps;
extern cl_filter_mode    gFilterModeToSkip;
extern cl_mem_flags gMemFlagsToUse;

extern int gtestTypesToRun;
extern bool gDeviceLt20;

extern bool validate_float_write_results( float *expected, float *actual, image_descriptor *imageInfo );
extern bool validate_half_write_results( cl_half *expected, cl_half *actual, image_descriptor* imageInfo );

const char *readwrite1DKernelSourcePattern =
"__kernel void sample_kernel( __global %s4 *input, read_write image1d_t output %s)\n"
"{\n"
"   int tidX = get_global_id(0);\n"
"   int offset = tidX;\n"
"   write_image%s( output, tidX %s, input[ offset ]);\n"
"}";

const char *write1DKernelSourcePattern =
"__kernel void sample_kernel( __global %s4 *input, write_only image1d_t output %s)\n"
"{\n"
"   int tidX = get_global_id(0);\n"
"   int offset = tidX;\n"
"   write_image%s( output, tidX %s, input[ offset ]);\n"
"}";

int test_write_image_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_kernel kernel,
                     image_descriptor *imageInfo, ExplicitType inputType, MTdata d )
{
    int                 totalErrors = 0;
    size_t              num_flags   = 0;
    const cl_mem_flags  *mem_flag_types = NULL;
    const char *        *mem_flag_names = NULL;
    const cl_mem_flags  write_only_mem_flag_types[2] = {  CL_MEM_WRITE_ONLY,   CL_MEM_READ_WRITE };
    const char *        write_only_mem_flag_names[2] = { "CL_MEM_WRITE_ONLY", "CL_MEM_READ_WRITE" };
    const cl_mem_flags  read_write_mem_flag_types[1] = {  CL_MEM_READ_WRITE};
    const char *        read_write_mem_flag_names[1] = { "CL_MEM_READ_WRITE"};

    if(gtestTypesToRun & kWriteTests)
    {
        mem_flag_types = write_only_mem_flag_types;
        mem_flag_names = write_only_mem_flag_names;
        num_flags      = sizeof( write_only_mem_flag_types ) / sizeof( write_only_mem_flag_types[0] );
    }
    else
    {
        mem_flag_types = read_write_mem_flag_types;
        mem_flag_names = read_write_mem_flag_names;
        num_flags      = sizeof( read_write_mem_flag_types ) / sizeof( read_write_mem_flag_types[0] );
    }
    for( size_t mem_flag_index = 0; mem_flag_index < num_flags; mem_flag_index++ )
    {
        int error;
        size_t threads[2];
        bool verifyRounding = false;
        int forceCorrectlyRoundedWrites = 0;

#if defined( __APPLE__ )
        // Require Apple's CPU implementation to be correctly rounded, not just within 0.6
        if( GetDeviceType(device) == CL_DEVICE_TYPE_CPU )
            forceCorrectlyRoundedWrites = 1;
#endif

        if( imageInfo->format->image_channel_data_type == CL_HALF_FLOAT )
            if( DetectFloatToHalfRoundingMode(queue) )
                return 1;

        BufferOwningPtr<char> maxImageUseHostPtrBackingStore, imageValues;

        create_random_image_data( inputType, imageInfo, imageValues, d );

        if(!gTestMipmaps)
        {
            if( inputType == kFloat && imageInfo->format->image_channel_data_type != CL_FLOAT && imageInfo->format->image_channel_data_type != CL_HALF_FLOAT )
            {
                /* Pilot data for sRGB images */
                if(is_sRGBA_order(imageInfo->format->image_channel_order))
                {
                    // We want to generate ints (mostly) in range of the target format which should be [0,255]
                    // However the range chosen here is [-test_range_ext, 255 + test_range_ext] so that
                    // it can test some out-of-range data points
                    const unsigned int test_range_ext = 16;
                    int formatMin = 0 - test_range_ext;
                    int formatMax = 255 + test_range_ext;
                    int pixel_value = 0;
                    float *inputValues = NULL;

                    // First, fill with arbitrary floats
                    {
                         inputValues = (float *)(char*)imageValues;
                        for( size_t i = 0; i < imageInfo->width * 4; i++ )
                        {
                            pixel_value = random_in_range( formatMin, (int)formatMax, d );
                            inputValues[ i ] = (float)(pixel_value/255.0f);
                        }
                    }

                    // Throw a few extra test values in there
                    inputValues = (float *)(char*)imageValues;
                    size_t i = 0;

                    // Piloting some debug inputs.
                    inputValues[ i++ ] = -0.5f;
                    inputValues[ i++ ] = 0.5f;
                    inputValues[ i++ ] = 2.f;
                    inputValues[ i++ ] = 0.5f;

                    // Also fill in the first few vectors with some deliberate tests to determine the rounding mode
                    // is correct
                    if( imageInfo->width > 12 )
                    {
                        float formatMax = (float)get_format_max_int( imageInfo->format );
                        inputValues[ i++ ] = 4.0f / formatMax;
                        inputValues[ i++ ] = 4.3f / formatMax;
                        inputValues[ i++ ] = 4.5f / formatMax;
                        inputValues[ i++ ] = 4.7f / formatMax;
                        inputValues[ i++ ] = 5.0f / formatMax;
                        inputValues[ i++ ] = 5.3f / formatMax;
                        inputValues[ i++ ] = 5.5f / formatMax;
                        inputValues[ i++ ] = 5.7f / formatMax;
                    }
                }
                else
                {
                    // First, fill with arbitrary floats
                    {
                        float *inputValues = (float *)(char*)imageValues;
                        for( size_t i = 0; i < imageInfo->width * 4; i++ )
                            inputValues[ i ] = get_random_float( -0.1f, 1.1f, d );
                    }

                    // Throw a few extra test values in there
                    float *inputValues = (float *)(char*)imageValues;
                    size_t i = 0;
                    inputValues[ i++ ] = -0.0000000000009f;
                    inputValues[ i++ ] = 1.f;
                    inputValues[ i++ ] = -1.f;
                    inputValues[ i++ ] = 2.f;

                    // Also fill in the first few vectors with some deliberate tests to determine the rounding mode
                    // is correct
                    if( imageInfo->width > 12 )
                    {
                        float formatMax = (float)get_format_max_int( imageInfo->format );
                        inputValues[ i++ ] = 4.0f / formatMax;
                        inputValues[ i++ ] = 4.3f / formatMax;
                        inputValues[ i++ ] = 4.5f / formatMax;
                        inputValues[ i++ ] = 4.7f / formatMax;
                        inputValues[ i++ ] = 5.0f / formatMax;
                        inputValues[ i++ ] = 5.3f / formatMax;
                        inputValues[ i++ ] = 5.5f / formatMax;
                        inputValues[ i++ ] = 5.7f / formatMax;
                        verifyRounding = true;
                    }
                }
            }
            else if( inputType == kUInt )
            {
                unsigned int *inputValues = (unsigned int*)(char*)imageValues;
                size_t i = 0;
                inputValues[ i++ ] = 0;
                inputValues[ i++ ] = 65535;
                inputValues[ i++ ] = 7271820;
                inputValues[ i++ ] = 0;
            }
        }

        // Construct testing sources
        clProtectedImage protImage;
        clMemWrapper unprotImage;
        cl_mem image;

        if( gMemFlagsToUse == CL_MEM_USE_HOST_PTR )
        {
            // clProtectedImage uses USE_HOST_PTR, so just rely on that for the testing (via Ian)
            // Do not use protected images for max image size test since it rounds the row size to a page size
            if (gTestMaxImages) {
                create_random_image_data( inputType, imageInfo, maxImageUseHostPtrBackingStore, d );

                unprotImage = create_image_1d( context, mem_flag_types[mem_flag_index] | CL_MEM_USE_HOST_PTR, imageInfo->format,
                                              imageInfo->width, 0,
                                              maxImageUseHostPtrBackingStore, NULL, &error );
            } else {
                error = protImage.Create( context, mem_flag_types[mem_flag_index], imageInfo->format, imageInfo->width );
            }
            if( error != CL_SUCCESS )
            {
                log_error( "ERROR: Unable to create 1D image of size %ld pitch %ld (%s, %s)\n", imageInfo->width,
                          imageInfo->rowPitch, IGetErrorString( error ), mem_flag_names[mem_flag_index] );
                return error;
            }

            if (gTestMaxImages)
                image = (cl_mem)unprotImage;
            else
                image = (cl_mem)protImage;
        }
        else // Either CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR or none
        {
            // Note: if ALLOC_HOST_PTR is used, the driver allocates memory that can be accessed by the host, but otherwise
            // it works just as if no flag is specified, so we just do the same thing either way
            // Note: if the flags is really CL_MEM_COPY_HOST_PTR, we want to remove it, because we don't want to copy any incoming data
            if( gTestMipmaps )
            {
                cl_image_desc image_desc = {0};
                image_desc.image_type = imageInfo->type;
                image_desc.num_mip_levels = imageInfo->num_mip_levels;
                image_desc.image_width = imageInfo->width;
                image_desc.image_array_size = imageInfo->arraySize;

                unprotImage = clCreateImage( context, mem_flag_types[mem_flag_index] | ( gMemFlagsToUse & ~(CL_MEM_COPY_HOST_PTR) ),
                                             imageInfo->format, &image_desc, NULL, &error);
                if( error != CL_SUCCESS )
                {
                    log_error( "ERROR: Unable to create %d level 1D image of size %ld (%s, %s)\n", imageInfo->num_mip_levels, imageInfo->width,
                               IGetErrorString( error ), mem_flag_names[mem_flag_index] );
                    return error;
                }
            }
            else
            {
                unprotImage = create_image_1d( context, mem_flag_types[mem_flag_index] | ( gMemFlagsToUse & ~(CL_MEM_COPY_HOST_PTR) ), imageInfo->format,
                                              imageInfo->width, 0,
                                              imageValues, NULL, &error );
                if( error != CL_SUCCESS )
                {
                    log_error( "ERROR: Unable to create 1D image of size %ld pitch %ld (%s, %s)\n", imageInfo->width,
                              imageInfo->rowPitch, IGetErrorString( error ), mem_flag_names[mem_flag_index] );
                    return error;
                }
            }
            image = unprotImage;
        }

        error = clSetKernelArg( kernel, 1, sizeof( cl_mem ), &image );
        test_error( error, "Unable to set kernel arguments" );

        size_t width_lod = imageInfo->width, nextLevelOffset = 0;
        size_t origin[ 3 ] = { 0, 0, 0 };
        size_t region[ 3 ] = { imageInfo->width, 1, 1 };
        size_t resultSize;

        for( int lod = 0; (gTestMipmaps && lod < imageInfo->num_mip_levels) || (!gTestMipmaps && lod < 1); lod++)
        {
            if(gTestMipmaps)
            {
                error = clSetKernelArg( kernel, 2, sizeof( int ), &lod );
            }

            clMemWrapper inputStream;

            char *imagePtrOffset = imageValues + nextLevelOffset;
            inputStream = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ),
                                     get_explicit_type_size( inputType ) * 4 * width_lod, imagePtrOffset, &error );
            test_error( error, "Unable to create input buffer" );

            // Set arguments
            error = clSetKernelArg( kernel, 0, sizeof( cl_mem ), &inputStream );
            test_error( error, "Unable to set kernel arguments" );

            // Run the kernel
            threads[0] = (size_t)width_lod;
            error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
            test_error( error, "Unable to run kernel" );

            // Get results
            if( gTestMipmaps )
                resultSize = width_lod * get_pixel_size( imageInfo->format );
            else
                resultSize = imageInfo->rowPitch;
            clProtectedArray PA(resultSize);
            char *resultValues = (char *)((void *)PA);

            if( gDebugTrace )
                log_info( "    reading results, %ld kbytes\n", (unsigned long)( resultSize / 1024 ) );

            origin[ 1 ] = lod;
            region[ 0 ] = width_lod;

            error = clEnqueueReadImage( queue, image, CL_TRUE, origin, region, gEnablePitch ? imageInfo->rowPitch : 0, 0, resultValues, 0, NULL, NULL );
            test_error( error, "Unable to read results from kernel" );
            if( gDebugTrace )
                log_info( "    results read\n" );

            // Validate results element by element
            char *imagePtr = imageValues + nextLevelOffset;
            int numTries = 5;
            {
                char *resultPtr = (char *)resultValues;
                for( size_t x = 0, i = 0; i < width_lod; x++, i++ )
                {
                    char resultBuffer[ 16 ]; // Largest format would be 4 channels * 4 bytes (32 bits) each

                    // Convert this pixel
                    if( inputType == kFloat )
                        pack_image_pixel( (float *)imagePtr, imageInfo->format, resultBuffer );
                    else if( inputType == kInt )
                        pack_image_pixel( (int *)imagePtr, imageInfo->format, resultBuffer );
                    else // if( inputType == kUInt )
                        pack_image_pixel( (unsigned int *)imagePtr, imageInfo->format, resultBuffer );

                    // Compare against the results
                    if(is_sRGBA_order(imageInfo->format->image_channel_order))
                    {
                        // Compare sRGB-mapped values
                        cl_float expected[4]    = {0};
                        cl_float* input_values  = (float*)imagePtr;
                        cl_uchar *actual        = (cl_uchar*)resultPtr;
                        float max_err           = MAX_lRGB_TO_sRGB_CONVERSION_ERROR;
                        float err[4]            = {0.0f};

                        for( unsigned int j = 0; j < get_format_channel_count( imageInfo->format ); j++ )
                        {
                            if(j < 3)
                            {
                                expected[j] = sRGBmap(input_values[j]);
                            }
                            else // there is no sRGB conversion for alpha component if it exists
                            {
                                expected[j] = NORMALIZE(input_values[j], 255.0f);
                            }

                            err[j] = fabsf( expected[ j ] - actual[ j ] );
                        }

                        if ((err[0] > max_err) ||
                            (err[1] > max_err) ||
                            (err[2] > max_err) ||
                            (err[3] > 0)) // there is no conversion for alpha so the error should be zero
                        {
                            log_error( "       Error:     %g %g %g %g\n", err[0], err[1], err[2], err[3]);
                            log_error( "       Input:     %g %g %g %g\n", *((float *)imagePtr), *((float *)imagePtr + 1), *((float *)imagePtr + 2), *((float *)imagePtr + 3));
                            log_error( "       Expected: %g %g %g %g\n", expected[ 0 ], expected[ 1 ], expected[ 2 ], expected[ 3 ] );
                            log_error( "       Actual:   %d %d %d %d\n", actual[ 0 ], actual[ 1 ], actual[ 2 ], actual[ 3 ] );
                            return 1;
                        }
                    }
                    else if( imageInfo->format->image_channel_data_type == CL_FLOAT )
                    {
                        float *expected = (float *)resultBuffer;
                        float *actual = (float *)resultPtr;

                        if( !validate_float_write_results( expected, actual, imageInfo ) )
                        {
                            unsigned int *e = (unsigned int *)resultBuffer;
                            unsigned int *a = (unsigned int *)resultPtr;
                            log_error( "ERROR: Sample %ld did not validate! (%s)\n", i, mem_flag_names[ mem_flag_index ] );
                            log_error( "       Expected: %a %a %a %a\n", expected[ 0 ], expected[ 1 ], expected[ 2 ], expected[ 3 ] );
                            log_error( "       Expected: %08x %08x %08x %08x\n", e[ 0 ], e[ 1 ], e[ 2 ], e[ 3 ] );
                            log_error( "       Actual:   %a %a %a %a\n", actual[ 0 ], actual[ 1 ], actual[ 2 ], actual[ 3 ] );
                            log_error( "       Actual:   %08x %08x %08x %08x\n", a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ] );
                            totalErrors++;
                            if( ( --numTries ) == 0 )
                                return 1;
                        }
                    }
                    else if( imageInfo->format->image_channel_data_type == CL_HALF_FLOAT )
                    {
                        cl_half *e = (cl_half *)resultBuffer;
                        cl_half *a = (cl_half *)resultPtr;
                        if( !validate_half_write_results( e, a, imageInfo ) )
                        {
                            totalErrors++;
                            log_error( "ERROR: Sample %ld did not validate! (%s)\n", i, mem_flag_names[ mem_flag_index ] );
                            log_error( "    Expected: 0x%04x 0x%04x 0x%04x 0x%04x\n", e[ 0 ], e[ 1 ], e[ 2 ], e[ 3 ] );
                            log_error( "    Actual:   0x%04x 0x%04x 0x%04x 0x%04x\n", a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ] );
                            if( inputType == kFloat )
                            {
                                float *p = (float *)imagePtr;
                                log_error( "    Source: %a %a %a %a\n", p[ 0 ], p[ 1 ], p[ 2] , p[ 3] );
                                log_error( "          : %12.24f %12.24f %12.24f %12.24f\n", p[ 0 ], p[ 1 ], p[ 2 ], p[ 3 ] );
                            }
                            if( ( --numTries ) == 0 )
                                return 1;
                        }
                    }
                    else
                    {
                        // Exact result passes every time
                        if( memcmp( resultBuffer, resultPtr, get_pixel_size( imageInfo->format ) ) != 0 )
                        {
                            // result is inexact.  Calculate error
                            int failure = 1;
                            float errors[4] = {NAN, NAN, NAN, NAN};
                            pack_image_pixel_error( (float *)imagePtr, imageInfo->format, resultBuffer, errors );

                            // We are allowed 0.6 absolute error vs. infinitely precise for some normalized formats
                            if( 0 == forceCorrectlyRoundedWrites    &&
                               (
                                imageInfo->format->image_channel_data_type == CL_UNORM_INT8 ||
                                imageInfo->format->image_channel_data_type == CL_UNORM_INT_101010 ||
                                imageInfo->format->image_channel_data_type == CL_UNORM_INT16 ||
                                imageInfo->format->image_channel_data_type == CL_SNORM_INT8 ||
                                imageInfo->format->image_channel_data_type == CL_SNORM_INT16
                                ))
                            {
                                if( ! (fabsf( errors[0] ) > 0.6f) && ! (fabsf( errors[1] ) > 0.6f) &&
                                   ! (fabsf( errors[2] ) > 0.6f) && ! (fabsf( errors[3] ) > 0.6f)  )
                                    failure = 0;
                            }


                            if( failure )
                            {
                                totalErrors++;
                                // Is it our special rounding test?
                                if( verifyRounding && i >= 1 && i <= 2 )
                                {
                                    // Try to guess what the rounding mode of the device really is based on what it returned
                                    const char *deviceRounding = "unknown";
                                    unsigned int deviceResults[8];
                                    read_image_pixel<unsigned int>( resultPtr, imageInfo, 0, 0, 0, deviceResults, lod);
                                    read_image_pixel<unsigned int>( resultPtr, imageInfo, 1, 0, 0, &deviceResults[ 4 ], lod );

                                    if( deviceResults[ 0 ] == 4 && deviceResults[ 1 ] == 4 && deviceResults[ 2 ] == 4 && deviceResults[ 3 ] == 4 &&
                                       deviceResults[ 4 ] == 5 && deviceResults[ 5 ] == 5 && deviceResults[ 6 ] == 5 && deviceResults[ 7 ] == 5 )
                                        deviceRounding = "truncate";
                                    else if( deviceResults[ 0 ] == 4 && deviceResults[ 1 ] == 4 && deviceResults[ 2 ] == 5 && deviceResults[ 3 ] == 5 &&
                                            deviceResults[ 4 ] == 5 && deviceResults[ 5 ] == 5 && deviceResults[ 6 ] == 6 && deviceResults[ 7 ] == 6 )
                                        deviceRounding = "round to nearest";
                                    else if( deviceResults[ 0 ] == 4 && deviceResults[ 1 ] == 4 && deviceResults[ 2 ] == 4 && deviceResults[ 3 ] == 5 &&
                                            deviceResults[ 4 ] == 5 && deviceResults[ 5 ] == 5 && deviceResults[ 6 ] == 6 && deviceResults[ 7 ] == 6 )
                                        deviceRounding = "round to even";

                                    log_error( "ERROR: Rounding mode sample (%ld) did not validate, probably due to the device's rounding mode being wrong (%s)\n", i, mem_flag_names[mem_flag_index] );
                                    log_error( "       Actual values rounded by device: %x %x %x %x %x %x %x %x\n", deviceResults[ 0 ], deviceResults[ 1 ], deviceResults[ 2 ], deviceResults[ 3 ],
                                              deviceResults[ 4 ], deviceResults[ 5 ], deviceResults[ 6 ], deviceResults[ 7 ] );
                                    log_error( "       Rounding mode of device appears to be %s\n", deviceRounding );
                                    return 1;
                                }
                                log_error( "ERROR: Sample %d (%d) did not validate!\n", (int)i, (int)x );
                                switch(imageInfo->format->image_channel_data_type)
                                {
                                    case CL_UNORM_INT8:
                                    case CL_SNORM_INT8:
                                    case CL_UNSIGNED_INT8:
                                    case CL_SIGNED_INT8:
                                        log_error( "    Expected: 0x%2.2x 0x%2.2x 0x%2.2x 0x%2.2x\n", ((cl_uchar*)resultBuffer)[0], ((cl_uchar*)resultBuffer)[1], ((cl_uchar*)resultBuffer)[2], ((cl_uchar*)resultBuffer)[3] );
                                        log_error( "    Actual:   0x%2.2x 0x%2.2x 0x%2.2x 0x%2.2x\n", ((cl_uchar*)resultPtr)[0], ((cl_uchar*)resultPtr)[1], ((cl_uchar*)resultPtr)[2], ((cl_uchar*)resultPtr)[3] );
                                        log_error( "    Error:    %f %f %f %f\n", errors[0], errors[1], errors[2], errors[3] );
                                        break;
                                    case CL_UNORM_INT16:
                                    case CL_SNORM_INT16:
                                    case CL_UNSIGNED_INT16:
                                    case CL_SIGNED_INT16:
#ifdef CL_SFIXED14_APPLE
                                    case CL_SFIXED14_APPLE:
#endif
                                        log_error( "    Expected: 0x%4.4x 0x%4.4x 0x%4.4x 0x%4.4x\n", ((cl_ushort*)resultBuffer)[0], ((cl_ushort*)resultBuffer)[1], ((cl_ushort*)resultBuffer)[2], ((cl_ushort*)resultBuffer)[3] );
                                        log_error( "    Actual:   0x%4.4x 0x%4.4x 0x%4.4x 0x%4.4x\n", ((cl_ushort*)resultPtr)[0], ((cl_ushort*)resultPtr)[1], ((cl_ushort*)resultPtr)[2], ((cl_ushort*)resultPtr)[3] );
                                        log_error( "    Error:    %f %f %f %f\n", errors[0], errors[1], errors[2], errors[3] );
                                        break;
                                    case CL_HALF_FLOAT:
                                        log_error( "    Expected: 0x%4.4x 0x%4.4x 0x%4.4x 0x%4.4x\n", ((cl_ushort*)resultBuffer)[0], ((cl_ushort*)resultBuffer)[1], ((cl_ushort*)resultBuffer)[2], ((cl_ushort*)resultBuffer)[3] );
                                        log_error( "    Actual:   0x%4.4x 0x%4.4x 0x%4.4x 0x%4.4x\n", ((cl_ushort*)resultPtr)[0], ((cl_ushort*)resultPtr)[1], ((cl_ushort*)resultPtr)[2], ((cl_ushort*)resultPtr)[3] );
                                        log_error( "    Ulps:     %f %f %f %f\n", errors[0], errors[1], errors[2], errors[3] );
                                        break;
                                    case CL_UNSIGNED_INT32:
                                    case CL_SIGNED_INT32:
                                        log_error( "    Expected: 0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x\n", ((cl_uint*)resultBuffer)[0], ((cl_uint*)resultBuffer)[1], ((cl_uint*)resultBuffer)[2], ((cl_uint*)resultBuffer)[3] );
                                        log_error( "    Actual:   0x%8.8x 0x%8.8x 0x%8.8x 0x%8.8x\n", ((cl_uint*)resultPtr)[0], ((cl_uint*)resultPtr)[1], ((cl_uint*)resultPtr)[2], ((cl_uint*)resultPtr)[3] );
                                        break;
                                    case CL_FLOAT:
                                        log_error( "    Expected: %a %a %a %a\n", ((cl_float*)resultBuffer)[0], ((cl_float*)resultBuffer)[1], ((cl_float*)resultBuffer)[2], ((cl_float*)resultBuffer)[3] );
                                        log_error( "    Actual:   %a %a %a %a\n", ((cl_float*)resultPtr)[0], ((cl_float*)resultPtr)[1], ((cl_float*)resultPtr)[2], ((cl_float*)resultPtr)[3] );
                                        log_error( "    Ulps:     %f %f %f %f\n", errors[0], errors[1], errors[2], errors[3] );
                                        break;
                                }

                                float *v = (float *)(char *)imagePtr;
                                log_error( "   src: %g %g %g %g\n", v[ 0 ], v[ 1], v[ 2 ], v[ 3 ] );
                                log_error( "      : %a %a %a %a\n", v[ 0 ], v[ 1], v[ 2 ], v[ 3 ] );
                                log_error( "   src: %12.24f %12.24f %12.24f %12.24f\n", v[0 ], v[  1], v[ 2 ], v[ 3 ] );

                                if( ( --numTries ) == 0 )
                                    return 1;
                            }
                        }
                    }
                    imagePtr += get_explicit_type_size( inputType ) * 4;
                    resultPtr += get_pixel_size( imageInfo->format );
                }
            }
            {
                nextLevelOffset += width_lod * get_pixel_size( imageInfo->format );
                width_lod = (width_lod >> 1) ? (width_lod >> 1) : 1;
            }
        }
    }

    // All done!
    return totalErrors;
}

int test_write_image_1D_set( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, ExplicitType inputType, MTdata d )
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    clProgramWrapper program;
    clKernelWrapper kernel;
    const char *KernelSourcePattern = NULL;
    int error;

    // Get our operating parameters
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    size_t pixelSize;

    image_descriptor imageInfo = { 0x0 };

    imageInfo.format = format;
    imageInfo.slicePitch = imageInfo.arraySize = 0;
    imageInfo.height = imageInfo.depth = 1;
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
    pixelSize = get_pixel_size( imageInfo.format );

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
    }

    // Determine types
    if( inputType == kInt )
        readFormat = "i";
    else if( inputType == kUInt )
        readFormat = "ui";
    else // kFloat
        readFormat = "f";

    // Construct the source
    if(gtestTypesToRun & kWriteTests)
    {
        KernelSourcePattern = write1DKernelSourcePattern;
    }
    else
    {
        KernelSourcePattern = readwrite1DKernelSourcePattern;
    }

    sprintf( programSrc,
             KernelSourcePattern,
             get_explicit_type_name( inputType ),
             gTestMipmaps ? ", int lod" : "",
             readFormat,
             gTestMipmaps ? ", lod" :"" );

    ptr = programSrc;
    error = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &ptr, "sample_kernel", gDeviceLt20 ? "" : "-cl-std=CL2.0");
    test_error( error, "Unable to create testing kernel" );

    // Run tests
    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            if(gTestMipmaps)
                imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), d);

            if( gDebugTrace )
                log_info( "   at size %d\n", (int)imageInfo.width );
            int retCode = test_write_image_1D( device, context, queue, kernel, &imageInfo, inputType, d );
            if( retCode )
                return retCode;
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE1D, imageInfo.format, CL_TRUE);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            if(gTestMipmaps)
                imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), d);
            log_info("Testing %d\n", (int)imageInfo.width);
            int retCode = test_write_image_1D( device, context, queue, kernel, &imageInfo, inputType, d );
            if( retCode )
                return retCode;
        }
    }
    else if( gTestRounding )
    {
        size_t typeRange = 1 << ( get_format_type_size( imageInfo.format ) * 8 );
        imageInfo.width = typeRange / 256;

        imageInfo.rowPitch = imageInfo.width * pixelSize;
        int retCode = test_write_image_1D( device, context, queue, kernel, &imageInfo, inputType, d );
        if( retCode )
            return retCode;
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
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, d );

                if( gTestMipmaps)
                {
                    imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), d);
                    size = (cl_ulong) compute_mipmapped_image_size(imageInfo) * 4;
                }
                else
                {
                    imageInfo.rowPitch = imageInfo.width * pixelSize;
                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, d );
                        imageInfo.rowPitch += extraWidth * pixelSize;
                    }

                    size = (size_t)imageInfo.rowPitch * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
            {
                log_info( "   at size %d (pitch %d) out of %d\n", (int)imageInfo.width, (int)imageInfo.rowPitch, (int)maxWidth );
                if( gTestMipmaps )
                    log_info( " and %d mip levels\n", (int)imageInfo.num_mip_levels );
            }

            int retCode = test_write_image_1D( device, context, queue, kernel, &imageInfo, inputType, d );
            if( retCode )
                return retCode;
        }
    }

    return 0;
}
