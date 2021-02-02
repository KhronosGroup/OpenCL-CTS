//
// Copyright (c) 2017, 2021 The Khronos Group Inc.
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
#include "test_common.h"
#include <float.h>

#if defined( __APPLE__ )
#include <signal.h>
#include <sys/signal.h>
#include <setjmp.h>
#endif


const char *read1DArrayKernelSourcePattern =
"__kernel void sample_kernel( read_only image1d_array_t input,%s __global float *xOffsets, __global float *yOffsets, __global %s4 *results %s)\n"
"{\n"
"%s"
"   int tidX = get_global_id(0), tidY = get_global_id(1);\n"
"%s"
"%s"
"   results[offset] = read_image%s( input, imageSampler, coords %s);\n"
"}";

const char *read_write1DArrayKernelSourcePattern =
"__kernel void sample_kernel( read_write image1d_array_t input,%s __global float *xOffsets, __global float *yOffsets, __global %s4 *results %s )\n"
"{\n"
"%s"
"   int tidX = get_global_id(0), tidY = get_global_id(1);\n"
"%s"
"%s"
"   results[offset] = read_image%s( input, coords %s);\n"
"}";

const char *offset1DArrayKernelSource =
"   int offset = tidY*get_image_width(input) + tidX;\n";

const char *offset1DArrayLodKernelSource =
"   int lod_int = (int)lod;\n"
"   int width_lod = (get_image_width(input) >> lod_int) ? (get_image_width(input) >> lod_int): 1;\n"
"   int offset = tidY*width_lod + tidX;\n";

const char *intCoordKernelSource1DArray =
"   int2 coords = (int2)( xOffsets[offset], yOffsets[offset]);\n";

const char *floatKernelSource1DArray =
"   float2 coords = (float2)( (float)( xOffsets[offset] ), (float)( yOffsets[offset] ) );\n";

static const char *samplerKernelArg = " sampler_t imageSampler,";

template <class T> int determine_validation_error_1D_arr( void *imagePtr, image_descriptor *imageInfo, image_sampler_data *imageSampler,
                                                  T *resultPtr, T * expected, float error,
                                                  float x, float y, float xAddressOffset, float yAddressOffset, size_t j, int &numTries, int &numClamped, bool printAsFloat, int lod )
{
    int actualX, actualY;
    int found = debug_find_pixel_in_image( imagePtr, imageInfo, resultPtr, &actualX, &actualY, NULL, lod );
    bool clampingErr = false, clamped = false, otherClampingBug = false;
    int clampedX, clampedY, ignoreMe;

    // FIXME: I do not believe this is correct for 1D or 2D image arrays;
    //        it will report spurious validation failure reasons since
    //        the clamping for such image objects is different than 1D-3D
    //        image objects.
    clamped = get_integer_coords_offset( x, y, 0.0f, xAddressOffset, yAddressOffset, 0.0f, imageInfo->width, imageInfo->arraySize, 0, imageSampler, imageInfo, clampedX, clampedY, ignoreMe );

    if( found )
    {
        // Is it a clamping bug?
        if( clamped && clampedX == actualX && clampedY == actualY )
        {
            if( (--numClamped) == 0 )
            {
                log_error( "ERROR: TEST FAILED: Read is erroneously clamping coordinates for image size %ld x %ld!\n", imageInfo->width, imageInfo->arraySize );
                if( printAsFloat )
                {
                    log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%g,%g,%g,%g),\n\tgot      (%g,%g,%g,%g),\n\terror of %g\n",
                              (int)j, x, x, y, y, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                              (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
                }
                else
                {
                    log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%x,%x,%x,%x),\n\tgot      (%x,%x,%x,%x)\n",
                              (int)j, x, x, y, y, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                              (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
                }
                return 1;
            }
            clampingErr = true;
            otherClampingBug = true;
        }
    }
    if( clamped && !otherClampingBug )
    {
        // If we are in clamp-to-edge mode and we're getting zeroes, it's possible we're getting border erroneously
        if( resultPtr[ 0 ] == 0 && resultPtr[ 1 ] == 0 && resultPtr[ 2 ] == 0 && resultPtr[ 3 ] == 0 )
        {
            if( (--numClamped) == 0 )
            {
                log_error( "ERROR: TEST FAILED: Clamping is erroneously returning border color for image size %ld x %ld!\n", imageInfo->width, imageInfo->arraySize );
                if( printAsFloat )
                {
                    log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%g,%g,%g,%g),\n\tgot      (%g,%g,%g,%g),\n\terror of %g\n",
                              (int)j, x, x, y, y, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                              (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
                }
                else
                {
                    log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%x,%x,%x,%x),\n\tgot      (%x,%x,%x,%x)\n",
                              (int)j, x, x, y, y, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                              (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
                }
                return 1;
            }
            clampingErr = true;
        }
    }
    if( !clampingErr )
    {
        if( printAsFloat )
        {
            log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%g,%g,%g,%g),\n\tgot      (%g,%g,%g,%g), error of %g\n",
                      (int)j, x, x, y, y, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                      (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
        }
        else
        {
            log_error( "Sample %d: coord {%f(%a), %f(%a)} did not validate!\n\tExpected (%x,%x,%x,%x),\n\tgot      (%x,%x,%x,%x)\n",
                      (int)j, x, x, y, y, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                      (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
        }
        log_error( "img size %ld,%ld (pitch %ld)", imageInfo->width, imageInfo->arraySize, imageInfo->rowPitch );
        if( clamped )
        {
            log_error( " which would clamp to %d,%d\n", clampedX, clampedY );
        }
        if( printAsFloat && gExtraValidateInfo)
        {
            log_error( "Nearby values:\n" );
            log_error( "\t%d\t%d\t%d\t%d\n", clampedX - 2, clampedX - 1, clampedX, clampedX + 1 );
            for( int yOff = -2; yOff <= 1; yOff++ )
            {
                float top[ 4 ], real[ 4 ], bot[ 4 ], bot2[ 4 ];
                read_image_pixel_float( imagePtr, imageInfo, clampedX - 2 , clampedY + yOff, 0, top );
                read_image_pixel_float( imagePtr, imageInfo, clampedX - 1 ,clampedY + yOff, 0, real );
                read_image_pixel_float( imagePtr, imageInfo, clampedX, clampedY + yOff, 0, bot );
                read_image_pixel_float( imagePtr, imageInfo, clampedX + 1, clampedY + yOff, 0, bot2 );
                log_error( "%d\t(%g,%g,%g,%g)",clampedY + yOff, top[0], top[1], top[2], top[3] );
                log_error( " (%g,%g,%g,%g)", real[0], real[1], real[2], real[3] );
                log_error( " (%g,%g,%g,%g)",bot[0], bot[1], bot[2], bot[3] );
                log_error( " (%g,%g,%g,%g)\n",bot2[0], bot2[1], bot2[2], bot2[3] );
            }

            if( clampedY < 1 )
            {
                log_error( "Nearby values:\n" );
                log_error( "\t%d\t%d\t%d\t%d\n", clampedX - 2, clampedX - 1, clampedX, clampedX + 1 );
                for( int yOff = (int)imageInfo->arraySize - 2; yOff <= (int)imageInfo->arraySize + 1; yOff++ )
                {
                    float top[ 4 ], real[ 4 ], bot[ 4 ], bot2[ 4 ];
                    read_image_pixel_float( imagePtr, imageInfo, clampedX - 2 , clampedY + yOff, 0, top );
                    read_image_pixel_float( imagePtr, imageInfo, clampedX - 1 ,clampedY + yOff, 0, real );
                    read_image_pixel_float( imagePtr, imageInfo, clampedX, clampedY + yOff, 0, bot );
                    read_image_pixel_float( imagePtr, imageInfo, clampedX + 1, clampedY + yOff, 0, bot2 );
                    log_error( "%d\t(%g,%g,%g,%g)",clampedY + yOff, top[0], top[1], top[2], top[3] );
                    log_error( " (%g,%g,%g,%g)", real[0], real[1], real[2], real[3] );
                    log_error( " (%g,%g,%g,%g)",bot[0], bot[1], bot[2], bot[3] );
                    log_error( " (%g,%g,%g,%g)\n",bot2[0], bot2[1], bot2[2], bot2[3] );
                }
            }
        }

        if( imageSampler->filter_mode != CL_FILTER_LINEAR )
        {
            if( found )
                log_error( "\tValue really found in image at %d,%d (%s)\n", actualX, actualY, ( found > 1 ) ? "NOT unique!!" : "unique" );
            else
                log_error( "\tValue not actually found in image\n" );
        }
        log_error( "\n" );

        numClamped = -1; // We force the clamped counter to never work
        if( ( --numTries ) == 0 )
        {
            return 1;
        }
    }
    return 0;
}

static void InitFloatCoords( image_descriptor *imageInfo, image_sampler_data *imageSampler, float *xOffsets, float *yOffsets, float xfract, float yfract, int normalized_coords, MTdata d , int lod)
{
    size_t i = 0;
    size_t width_lod = imageInfo->width;

    if(gTestMipmaps)
        width_lod = (imageInfo->width >> lod) ? (imageInfo->width >> lod) : 1;

    if( gDisableOffsets )
    {
        for( size_t y = 0; y < imageInfo->arraySize; y++ )
        {
            for( size_t x = 0; x < width_lod; x++, i++ )
            {
                xOffsets[ i ] = (float) (xfract + (double) x);
                yOffsets[ i ] = (float) (yfract + (double) y);
            }
        }
    }
    else
    {
        for( size_t y = 0; y < imageInfo->arraySize; y++ )
        {
            for( size_t x = 0; x < width_lod; x++, i++ )
            {
                xOffsets[ i ] = (float) (xfract + (double) ((int) x + random_in_range( -10, 10, d )));
                yOffsets[ i ] = (float) (yfract + (double) ((int) y + random_in_range( -10, 10, d )));
            }
        }
    }

    if( imageSampler->addressing_mode == CL_ADDRESS_NONE )
    {
        i = 0;
        for( size_t y = 0; y < imageInfo->arraySize; y++ )
        {
            for( size_t x = 0; x < width_lod; x++, i++ )
            {
                xOffsets[ i ] = (float) CLAMP( (double) xOffsets[ i ], 0.0, (double)width_lod - 1.0);
                yOffsets[ i ] = (float) CLAMP( (double) yOffsets[ i ], 0.0, (double)imageInfo->arraySize - 1.0);
            }
        }
    }

    if( normalized_coords )
    {
        i = 0;
        for( size_t y = 0; y < imageInfo->arraySize; y++ )
        {
            for( size_t x = 0; x < width_lod; x++, i++ )
            {
                xOffsets[ i ] = (float) ((double) xOffsets[ i ] / (double) width_lod);
            }
        }
    }
}

int test_read_image_1D_array( cl_context context, cl_command_queue queue, cl_kernel kernel,
                             image_descriptor *imageInfo, image_sampler_data *imageSampler,
                             bool useFloatCoords, ExplicitType outputType, MTdata d )
{
    int error;
    static int initHalf = 0;

    size_t threads[2];
    cl_mem_flags    image_read_write_flags = CL_MEM_READ_ONLY;
    clMemWrapper xOffsets, yOffsets, results;
    clSamplerWrapper actualSampler;
    BufferOwningPtr<char> maxImageUseHostPtrBackingStore;

    // The DataBuffer template class really does use delete[], not free -- IRO
    BufferOwningPtr<cl_float> xOffsetValues(malloc(sizeof(cl_float) * imageInfo->width * imageInfo->arraySize));
    BufferOwningPtr<cl_float> yOffsetValues(malloc(sizeof(cl_float) * imageInfo->width * imageInfo->arraySize));

    if( imageInfo->format->image_channel_data_type == CL_HALF_FLOAT )
        if( DetectFloatToHalfRoundingMode(queue) )
            return 1;

    // generate_random_image_data allocates with malloc, so we use a MallocDataBuffer here
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if( gDebugTrace )
    {
        log_info( " - Creating 1D image array %d by %d...\n", (int)imageInfo->width, (int)imageInfo->arraySize );
        if(gTestMipmaps)
            log_info("  - and %d mip levels\n", (int)imageInfo->num_mip_levels);
    }

    // Construct testing sources
    clProtectedImage protImage;
    clMemWrapper unprotImage;
    cl_mem image;

    if(gtestTypesToRun & kReadTests)
    {
        image_read_write_flags = CL_MEM_READ_ONLY;
    }
    else
    {
        image_read_write_flags = CL_MEM_READ_WRITE;
    }

    if( gMemFlagsToUse == CL_MEM_USE_HOST_PTR )
    {
        // clProtectedImage uses USE_HOST_PTR, so just rely on that for the testing (via Ian)
        // Do not use protected images for max image size test since it rounds the row size to a page size
        if (gTestMaxImages) {
            generate_random_image_data( imageInfo, maxImageUseHostPtrBackingStore, d );

            unprotImage = create_image_1d_array(context,
                                                image_read_write_flags | CL_MEM_USE_HOST_PTR,
                                                imageInfo->format,
                                                imageInfo->width, imageInfo->arraySize,
                                                ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                                ( gEnablePitch ? imageInfo->slicePitch : 0),
                                                maxImageUseHostPtrBackingStore, &error);
        } else {
            error = protImage.Create( context, CL_MEM_OBJECT_IMAGE1D_ARRAY,
                                    image_read_write_flags,
                                    imageInfo->format,
                imageInfo->width, 1, 1, imageInfo->arraySize );
        }
        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create 1D image array of size %d x %d pitch %d (%s)\n",
                      (int)imageInfo->width, (int)imageInfo->arraySize,
                      (int)imageInfo->rowPitch, IGetErrorString( error ) );
            return error;
        }

        if (gTestMaxImages)
            image = (cl_mem)unprotImage;
        else
            image = (cl_mem)protImage;
    }
    else if( gMemFlagsToUse == CL_MEM_COPY_HOST_PTR )
    {
        // Don't use clEnqueueWriteImage; just use copy host ptr to get the data in
        unprotImage = create_image_1d_array(context,
                                            image_read_write_flags | CL_MEM_COPY_HOST_PTR,
                                            imageInfo->format,
                                            imageInfo->width, imageInfo->arraySize,
                                            ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                            ( gEnablePitch ? imageInfo->slicePitch : 0),
                                            imageValues, &error);

        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create 1D image array of size %d x %d pitch %d (%s)\n",
                      (int)imageInfo->width, (int)imageInfo->arraySize,
                      (int)imageInfo->rowPitch, IGetErrorString( error ) );
            return error;
        }
        image = unprotImage;
    }
    else // Either CL_MEM_ALLOC_HOST_PTR or none
    {
        // Note: if ALLOC_HOST_PTR is used, the driver allocates memory that can be accessed by the host, but otherwise
        // it works just as if no flag is specified, so we just do the same thing either way
        if(gTestMipmaps)
        {
            cl_image_desc image_desc = {0};
            image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
            image_desc.image_width = imageInfo->width;
            image_desc.image_array_size = imageInfo->arraySize;
            image_desc.num_mip_levels = imageInfo->num_mip_levels;

            unprotImage = clCreateImage( context,
                                        image_read_write_flags,
                                        imageInfo->format, &image_desc, NULL, &error);
            if( error != CL_SUCCESS )
            {
                log_error( "ERROR: Unable to create %d level mipmapped 1D image array of size %d x %d (pitch %d, %d ) (%s)",(int)imageInfo->num_mip_levels, (int)imageInfo->width, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
                return error;
            }
        }
        else
        {
            unprotImage = create_image_1d_array(context,
                                                image_read_write_flags | gMemFlagsToUse,
                                                imageInfo->format,
                                                imageInfo->width, imageInfo->arraySize,
                                                ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                                ( gEnablePitch ? imageInfo->slicePitch : 0),
                                                imageValues, &error);

            if( error != CL_SUCCESS )
            {
                log_error( "ERROR: Unable to create 1D image array of size %d x %d pitch %d (%s)\n",
                          (int)imageInfo->width, (int)imageInfo->arraySize,
                          (int)imageInfo->rowPitch, IGetErrorString( error ) );
                return error;
            }
        }
        image = unprotImage;
    }

    if( gMemFlagsToUse != CL_MEM_COPY_HOST_PTR )
    {
        if( gDebugTrace )
            log_info( " - Writing image...\n" );

        size_t origin[ 3 ] = { 0, 0, 0 };
        size_t region[ 3 ] = { imageInfo->width, imageInfo->arraySize, 1 };

        if(gTestMipmaps)
        {
            int nextLevelOffset = 0;

            for (int i =0; i < imageInfo->num_mip_levels; i++)
            {   origin[2] = i;
                error = clEnqueueWriteImage(queue, image, CL_TRUE,
                                            origin, region, /*gEnablePitch ? imageInfo->rowPitch :*/ 0, /*gEnablePitch ? imageInfo->slicePitch :*/ 0,
                                            ((char*)imageValues + nextLevelOffset), 0, NULL, NULL);
                if (error != CL_SUCCESS)
                {
                    log_error( "ERROR: Unable to write to %d level mipmapped 3D image of size %d x %d x %d\n", (int)imageInfo->num_mip_levels,(int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth );
                    return error;
                }
                nextLevelOffset += region[0]*region[1]*get_pixel_size(imageInfo->format);
                //Subsequent mip level dimensions keep halving
                region[0] = region[0] >> 1 ? region[0] >> 1 : 1;
            }
        }
        else
        {
            error = clEnqueueWriteImage(queue, image, CL_TRUE,
                                        origin, region, ( gEnablePitch ? imageInfo->rowPitch : 0 ), 0,
                                        imageValues, 0, NULL, NULL);
            if (error != CL_SUCCESS)
            {
                log_error( "ERROR: Unable to write to %d level 1D image array of size %d x %d\n",
                          (int)imageInfo->num_mip_levels,
                          (int)imageInfo->width, (int)imageInfo->arraySize );
                return error;
            }
        }
    }

    if( gDebugTrace )
        log_info( " - Creating kernel arguments...\n" );

    xOffsets = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * imageInfo->width
                                  * imageInfo->arraySize,
                              xOffsetValues, &error);
    test_error( error, "Unable to create x offset buffer" );

    yOffsets = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float) * imageInfo->width
                                  * imageInfo->arraySize,
                              yOffsetValues, &error);
    test_error( error, "Unable to create y offset buffer" );

    results = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             get_explicit_type_size(outputType) * 4
                                 * imageInfo->width * imageInfo->arraySize,
                             NULL, &error);
    test_error( error, "Unable to create result buffer" );

    // Create sampler to use
    actualSampler = create_sampler(context, imageSampler, gTestMipmaps, &error);
    test_error(error, "Unable to create image sampler");

    // Set arguments
    int idx = 0;
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &image );
    test_error( error, "Unable to set kernel arguments" );
    if( !gUseKernelSamplers )
    {
        error = clSetKernelArg( kernel, idx++, sizeof( cl_sampler ), &actualSampler );
        test_error( error, "Unable to set kernel arguments" );
    }
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &xOffsets );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &yOffsets );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &results );
    test_error( error, "Unable to set kernel arguments" );

    // A cast of troublesome offsets. The first one has to be zero.
    const float float_offsets[] = { 0.0f, MAKE_HEX_FLOAT(0x1.0p-30f, 0x1L, -30), 0.25f, 0.3f, 0.5f - FLT_EPSILON/4.0f, 0.5f, 0.9f, 1.0f - FLT_EPSILON/2 };
    int float_offset_count = sizeof( float_offsets) / sizeof( float_offsets[0] );
    int numTries = MAX_TRIES, numClamped = MAX_CLAMPED;
    int loopCount = 2 * float_offset_count;
    if( ! useFloatCoords )
        loopCount = 1;
    if (gTestMaxImages) {
        loopCount = 1;
        log_info("Testing each size only once with pixel offsets of %g for max sized images.\n", float_offsets[0]);
    }

    // Get the maximum absolute error for this format
    if(gtestTypesToRun & kReadWriteTests)
    {
        loopCount = 1;
    }

    // Get the maximum absolute error for this format
    double formatAbsoluteError = get_max_absolute_error(imageInfo->format, imageSampler);
    if (gDebugTrace) log_info("\tformatAbsoluteError is %e\n", formatAbsoluteError);

    if (0 == initHalf && imageInfo->format->image_channel_data_type == CL_HALF_FLOAT ) {
        initHalf = CL_SUCCESS == DetectFloatToHalfRoundingMode( queue );
        if (initHalf) {
            log_info("Half rounding mode successfully detected.\n");
        }
    }

    size_t width_lod = imageInfo->width;
    size_t nextLevelOffset = 0;
    char * imagePtr;
    for(int lod = 0; (gTestMipmaps && lod < imageInfo->num_mip_levels) || (!gTestMipmaps && lod < 1); lod++)
    {
        size_t resultValuesSize = width_lod * imageInfo->arraySize * get_explicit_type_size( outputType ) * 4;
        BufferOwningPtr<char> resultValues(malloc(resultValuesSize));
        float lod_float = (float)lod;
        if (gTestMipmaps) {
            //Set the lod kernel arg
            if(gDebugTrace)
                log_info(" - Working at mip level %d\n", lod);
            error = clSetKernelArg( kernel, idx, sizeof( float ), &lod_float);
            test_error( error, "Unable to set kernel arguments" );
        }

    for( int q = 0; q < loopCount; q++ )
    {
        float offset = float_offsets[ q % float_offset_count ];

        // Init the coordinates
        InitFloatCoords(imageInfo, imageSampler, xOffsetValues, yOffsetValues,
                        q>=float_offset_count ? -offset: offset,
                        q>=float_offset_count ? offset: -offset, imageSampler->normalized_coords, d, lod );

        error = clEnqueueWriteBuffer( queue, xOffsets, CL_TRUE, 0, sizeof(cl_float) * imageInfo->arraySize * imageInfo->width, xOffsetValues, 0, NULL, NULL );
        test_error( error, "Unable to write x offsets" );
        error = clEnqueueWriteBuffer( queue, yOffsets, CL_TRUE, 0, sizeof(cl_float) * imageInfo->arraySize * imageInfo->width, yOffsetValues, 0, NULL, NULL );
        test_error( error, "Unable to write y offsets" );

        // Get results
        memset( resultValues, 0xff, resultValuesSize );
        clEnqueueWriteBuffer( queue, results, CL_TRUE, 0, resultValuesSize, resultValues, 0, NULL, NULL );

        // Run the kernel
        threads[0] = (size_t)width_lod;
        threads[1] = (size_t)imageInfo->arraySize;
        error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Unable to run kernel" );

        if( gDebugTrace )
            log_info( "    reading results, %ld kbytes\n", (unsigned long)( width_lod * imageInfo->arraySize * get_explicit_type_size( outputType ) * 4 / 1024 ) );

        error = clEnqueueReadBuffer( queue, results, CL_TRUE, 0, width_lod * imageInfo->arraySize * get_explicit_type_size( outputType ) * 4, resultValues, 0, NULL, NULL );
        test_error( error, "Unable to read results from kernel" );
        if( gDebugTrace )
            log_info( "    results read\n" );

        // Validate results element by element
        imagePtr = (char*)imageValues + nextLevelOffset;
        /*
         * FLOAT output type, order= sRGB
         */
        if(is_sRGBA_order(imageInfo->format->image_channel_order) && ( outputType == kFloat ))
        {
            // Validate float results
            float *resultPtr = (float *)(char *)resultValues;
            float expected[4], error=0.0f;
            float maxErr = get_max_relative_error( imageInfo->format, imageSampler, 0 /*not 3D*/, CL_FILTER_LINEAR == imageSampler->filter_mode );
            for( size_t y = 0, j = 0; y < imageInfo->arraySize; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, j++ )
                {
                    // Step 1: go through and see if the results verify for the pixel
                    // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                    // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                    int checkOnlyOnePixel = 0;
                    int found_pixel = 0;
                    float offset = NORM_OFFSET;
                    if (!imageSampler->normalized_coords ||  imageSampler->filter_mode != CL_FILTER_NEAREST || NORM_OFFSET == 0
#if defined( __APPLE__ )
                        // Apple requires its CPU implementation to do correctly rounded address arithmetic in all modes
                        || gDeviceType != CL_DEVICE_TYPE_GPU
#endif
                        )
                        offset = 0.0f;          // Loop only once

                    for (float norm_offset_x = -offset; norm_offset_x <= offset && !found_pixel; norm_offset_x += NORM_OFFSET) {
                        for (float norm_offset_y = -offset; norm_offset_y <= offset && !found_pixel; norm_offset_y += NORM_OFFSET) {


                            // Try sampling the pixel, without flushing denormals.
                            int containsDenormals = 0;
                            FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                imageSampler, expected, 0, &containsDenormals, lod );

                            float err1 = ABS_ERROR(sRGBmap(resultPtr[0]),
                                                   sRGBmap(expected[0]));
                            float err2 = ABS_ERROR(sRGBmap(resultPtr[1]),
                                                   sRGBmap(expected[1]));
                            float err3 = ABS_ERROR(sRGBmap(resultPtr[2]),
                                                   sRGBmap(expected[2]));
                            float err4 = ABS_ERROR(resultPtr[3], expected[3]);
                            float maxErr = 0.5;

                            // Check if the result matches.
                            if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    ||
                               ! (err3 <= maxErr) || ! (err4 <= maxErr)    )
                            {
                                //try flushing the denormals, if there is a failure.
                                if( containsDenormals )
                                {
                                    // If implementation decide to flush subnormals to zero,
                                    // max error needs to be adjusted
                                    maxErr += 4 * FLT_MIN;

                                    maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                               xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                               imageSampler, expected, 0, NULL, lod );

                                    err1 = ABS_ERROR(sRGBmap(resultPtr[0]),
                                                     sRGBmap(expected[0]));
                                    err2 = ABS_ERROR(sRGBmap(resultPtr[1]),
                                                     sRGBmap(expected[1]));
                                    err3 = ABS_ERROR(sRGBmap(resultPtr[2]),
                                                     sRGBmap(expected[2]));
                                    err4 = ABS_ERROR(resultPtr[3], expected[3]);
                                }
                            }

                            // If the final result DOES match, then we've found a valid result and we're done with this pixel.
                            found_pixel = (err1 <= maxErr) && (err2 <= maxErr)  && (err3 <= maxErr) && (err4 <= maxErr);
                        }//norm_offset_x
                    }//norm_offset_y


                    // Step 2: If we did not find a match, then print out debugging info.
                    if (!found_pixel) {
                        // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        checkOnlyOnePixel = 0;
                        int shouldReturn = 0;
                        for (float norm_offset_x = -offset; norm_offset_x <= offset && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -offset; norm_offset_y <= offset && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                                // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                // E.g., test one pixel.
                                if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                    norm_offset_x = 0.0f;
                                    norm_offset_y = 0.0f;
                                    checkOnlyOnePixel = 1;
                                }

                                int containsDenormals = 0;
                                FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                      xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                      imageSampler, expected, 0, &containsDenormals, lod );

                                float err1 = ABS_ERROR(sRGBmap(resultPtr[0]),
                                                       sRGBmap(expected[0]));
                                float err2 = ABS_ERROR(sRGBmap(resultPtr[1]),
                                                       sRGBmap(expected[1]));
                                float err3 = ABS_ERROR(sRGBmap(resultPtr[2]),
                                                       sRGBmap(expected[2]));
                                float err4 =
                                    ABS_ERROR(resultPtr[3], expected[3]);

                                float maxErr = 0.6;

                                if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    ||
                                   ! (err3 <= maxErr) || ! (err4 <= maxErr)    )
                                {
                                    //try flushing the denormals, if there is a failure.
                                    if( containsDenormals )
                                    {
                                        // If implementation decide to flush subnormals to zero,
                                        // max error needs to be adjusted
                                        maxErr += 4 * FLT_MIN;

                                        maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                   xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                   imageSampler, expected, 0, NULL, lod );

                                        err1 = ABS_ERROR(sRGBmap(resultPtr[0]),
                                                         sRGBmap(expected[0]));
                                        err2 = ABS_ERROR(sRGBmap(resultPtr[1]),
                                                         sRGBmap(expected[1]));
                                        err3 = ABS_ERROR(sRGBmap(resultPtr[2]),
                                                         sRGBmap(expected[2]));
                                        err4 = ABS_ERROR(resultPtr[3],
                                                         expected[3]);
                                    }
                                }
                                if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    ||
                                   ! (err3 <= maxErr) || ! (err4 <= maxErr)    )
                                {
                                    log_error("FAILED norm_offsets: %g , %g:\n", norm_offset_x, norm_offset_y);

                                    float tempOut[4];
                                    shouldReturn |= determine_validation_error_1D_arr<float>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                      expected, error, xOffsetValues[ j ], yOffsetValues[ j ], norm_offset_x, norm_offset_y, j, numTries, numClamped, true, lod );

                                    log_error( "Step by step:\n" );
                                    FloatPixel temp = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                      xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                      imageSampler, tempOut, 1 /* verbose */, &containsDenormals /*dont flush while error reporting*/, lod );
                                    log_error( "\tulps: %2.2f, %2.2f, %2.2f, %2.2f  (max allowed: %2.2f)\n\n",
                                              Ulp_Error( resultPtr[0], expected[0] ),
                                              Ulp_Error( resultPtr[1], expected[1] ),
                                              Ulp_Error( resultPtr[2], expected[2] ),
                                              Ulp_Error( resultPtr[3], expected[3] ),
                                              Ulp_Error( MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) + maxErr, MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) ) );

                                } else {
                                    log_error("Test error: we should have detected this passing above.\n");
                                }

                            }//norm_offset_x
                        }//norm_offset_y
                        if( shouldReturn )
                            return 1;
                    } // if (!found_pixel)

                    resultPtr += 4;
                }
            }
        }
        /*
         * FLOAT output type
         */
        else if( outputType == kFloat )
        {
            // Validate float results
            float *resultPtr = (float *)(char *)resultValues;
            float expected[4], error=0.0f;
            float maxErr = get_max_relative_error( imageInfo->format, imageSampler, 0 /*not 3D*/, CL_FILTER_LINEAR == imageSampler->filter_mode );
            for( size_t y = 0, j = 0; y < imageInfo->arraySize; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, j++ )
                {
                    // Step 1: go through and see if the results verify for the pixel
                    // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                    // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                    int checkOnlyOnePixel = 0;
                    int found_pixel = 0;
                    float offset = NORM_OFFSET;
                    if (!imageSampler->normalized_coords ||  imageSampler->filter_mode != CL_FILTER_NEAREST || NORM_OFFSET == 0
#if defined( __APPLE__ )
                        // Apple requires its CPU implementation to do correctly rounded address arithmetic in all modes
                        || gDeviceType != CL_DEVICE_TYPE_GPU
#endif
                        )
                        offset = 0.0f;          // Loop only once

                    for (float norm_offset_x = -offset; norm_offset_x <= offset && !found_pixel; norm_offset_x += NORM_OFFSET) {
                        for (float norm_offset_y = -offset; norm_offset_y <= offset && !found_pixel; norm_offset_y += NORM_OFFSET) {


                            // Try sampling the pixel, without flushing denormals.
                            int containsDenormals = 0;
                            FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                imageSampler, expected, 0, &containsDenormals, lod );

                            float err1 = ABS_ERROR(resultPtr[0], expected[0]);
                            float err2 = ABS_ERROR(resultPtr[1], expected[1]);
                            float err3 = ABS_ERROR(resultPtr[2], expected[2]);
                            float err4 = ABS_ERROR(resultPtr[3], expected[3]);
                            // Clamp to the minimum absolute error for the format
                            if (err1 > 0 && err1 < formatAbsoluteError) { err1 = 0.0f; }
                            if (err2 > 0 && err2 < formatAbsoluteError) { err2 = 0.0f; }
                            if (err3 > 0 && err3 < formatAbsoluteError) { err3 = 0.0f; }
                            if (err4 > 0 && err4 < formatAbsoluteError) { err4 = 0.0f; }
                            float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );
                            float maxErr2 = MAX( maxErr * maxPixel.p[1], FLT_MIN );
                            float maxErr3 = MAX( maxErr * maxPixel.p[2], FLT_MIN );
                            float maxErr4 = MAX( maxErr * maxPixel.p[3], FLT_MIN );

                            // Check if the result matches.
                            if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    ||
                               ! (err3 <= maxErr3) || ! (err4 <= maxErr4)    )
                            {
                                //try flushing the denormals, if there is a failure.
                                if( containsDenormals )
                                {
                                    // If implementation decide to flush subnormals to zero,
                                    // max error needs to be adjusted
                                    maxErr1 += 4 * FLT_MIN;
                                    maxErr2 += 4 * FLT_MIN;
                                    maxErr3 += 4 * FLT_MIN;
                                    maxErr4 += 4 * FLT_MIN;

                                    maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                               xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                               imageSampler, expected, 0, NULL, lod );

                                    err1 = ABS_ERROR(resultPtr[0], expected[0]);
                                    err2 = ABS_ERROR(resultPtr[1], expected[1]);
                                    err3 = ABS_ERROR(resultPtr[2], expected[2]);
                                    err4 = ABS_ERROR(resultPtr[3], expected[3]);
                                }
                            }

                            // If the final result DOES match, then we've found a valid result and we're done with this pixel.
                            found_pixel = (err1 <= maxErr1) && (err2 <= maxErr2)  && (err3 <= maxErr3) && (err4 <= maxErr4);
                        }//norm_offset_x
                    }//norm_offset_y


                    // Step 2: If we did not find a match, then print out debugging info.
                    if (!found_pixel) {
                        // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        checkOnlyOnePixel = 0;
                        int shouldReturn = 0;
                        for (float norm_offset_x = -offset; norm_offset_x <= offset && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -offset; norm_offset_y <= offset && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                                // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                // E.g., test one pixel.
                                if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                    norm_offset_x = 0.0f;
                                    norm_offset_y = 0.0f;
                                    checkOnlyOnePixel = 1;
                                }

                                int containsDenormals = 0;
                                FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                      xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                      imageSampler, expected, 0, &containsDenormals, lod );

                                float err1 =
                                    ABS_ERROR(resultPtr[0], expected[0]);
                                float err2 =
                                    ABS_ERROR(resultPtr[1], expected[1]);
                                float err3 =
                                    ABS_ERROR(resultPtr[2], expected[2]);
                                float err4 =
                                    ABS_ERROR(resultPtr[3], expected[3]);
                                float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );
                                float maxErr2 = MAX( maxErr * maxPixel.p[1], FLT_MIN );
                                float maxErr3 = MAX( maxErr * maxPixel.p[2], FLT_MIN );
                                float maxErr4 = MAX( maxErr * maxPixel.p[3], FLT_MIN );


                                if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    ||
                                   ! (err3 <= maxErr3) || ! (err4 <= maxErr4)    )
                                {
                                    //try flushing the denormals, if there is a failure.
                                    if( containsDenormals )
                                    {
                                        maxErr1 += 4 * FLT_MIN;
                                        maxErr2 += 4 * FLT_MIN;
                                        maxErr3 += 4 * FLT_MIN;
                                        maxErr4 += 4 * FLT_MIN;

                                        maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                   xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                   imageSampler, expected, 0, NULL, lod );

                                        err1 = ABS_ERROR(resultPtr[0],
                                                         expected[0]);
                                        err2 = ABS_ERROR(resultPtr[1],
                                                         expected[1]);
                                        err3 = ABS_ERROR(resultPtr[2],
                                                         expected[2]);
                                        err4 = ABS_ERROR(resultPtr[3],
                                                         expected[3]);
                                    }
                                }
                                if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    ||
                                   ! (err3 <= maxErr3) || ! (err4 <= maxErr4)    )
                                {
                                    log_error("FAILED norm_offsets: %g , %g:\n", norm_offset_x, norm_offset_y);

                                    float tempOut[4];
                                    shouldReturn |= determine_validation_error_1D_arr<float>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                      expected, error, xOffsetValues[ j ], yOffsetValues[ j ], norm_offset_x, norm_offset_y, j, numTries, numClamped, true, lod );

                                    log_error( "Step by step:\n" );
                                    FloatPixel temp = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                      xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                                      imageSampler, tempOut, 1 /* verbose */, &containsDenormals /*dont flush while error reporting*/, lod );
                                    log_error( "\tulps: %2.2f, %2.2f, %2.2f, %2.2f  (max allowed: %2.2f)\n\n",
                                              Ulp_Error( resultPtr[0], expected[0] ),
                                              Ulp_Error( resultPtr[1], expected[1] ),
                                              Ulp_Error( resultPtr[2], expected[2] ),
                                              Ulp_Error( resultPtr[3], expected[3] ),
                                              Ulp_Error( MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) + maxErr, MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) ) );

                                } else {
                                    log_error("Test error: we should have detected this passing above.\n");
                                }

                            }//norm_offset_x
                        }//norm_offset_y
                        if( shouldReturn )
                            return 1;
                    } // if (!found_pixel)

                    resultPtr += 4;
                }
            }
        }
        /*
         * UINT output type
         */
        else if( outputType == kUInt )
        {
            // Validate unsigned integer results
            unsigned int *resultPtr = (unsigned int *)(char *)resultValues;
            unsigned int expected[4];
            float error;
            for( size_t y = 0, j = 0; y < imageInfo->arraySize; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, j++ )
                {
                    // Step 1: go through and see if the results verify for the pixel
                    // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                    // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                    int checkOnlyOnePixel = 0;
                    int found_pixel = 0;
                    for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                        for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                            // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                            // E.g., test one pixel.
                            if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                norm_offset_x = 0.0f;
                                norm_offset_y = 0.0f;
                                checkOnlyOnePixel = 1;
                            }

                            sample_image_pixel_offset<unsigned int>( imagePtr, imageInfo,
                                                                    xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                    imageSampler, expected, lod );


                            error = errMax( errMax( abs_diff_uint(expected[ 0 ], resultPtr[ 0 ]), abs_diff_uint(expected[ 1 ], resultPtr[ 1 ]) ),
                                           errMax( abs_diff_uint(expected[ 2 ], resultPtr[ 2 ]), abs_diff_uint(expected[ 3 ], resultPtr[ 3 ]) ) );

                            if (error <= MAX_ERR)
                                found_pixel = 1;
                        }//norm_offset_x
                    }//norm_offset_y

                    // Step 2: If we did not find a match, then print out debugging info.
                    if (!found_pixel) {
                        // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        checkOnlyOnePixel = 0;
                        int shouldReturn = 0;
                        for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                                // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                // E.g., test one pixel.
                                if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                    norm_offset_x = 0.0f;
                                    norm_offset_y = 0.0f;
                                    checkOnlyOnePixel = 1;
                                }

                                sample_image_pixel_offset<unsigned int>( imagePtr, imageInfo,
                                                                        xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                                        imageSampler, expected, lod );


                                error = errMax( errMax( abs_diff_uint(expected[ 0 ], resultPtr[ 0 ]), abs_diff_uint(expected[ 1 ], resultPtr[ 1 ]) ),
                                               errMax( abs_diff_uint(expected[ 2 ], resultPtr[ 2 ]), abs_diff_uint(expected[ 3 ], resultPtr[ 3 ]) ) );

                                if( error > MAX_ERR )
                                {
                                    log_error("FAILED norm_offsets: %g , %g:\n", norm_offset_x, norm_offset_y);

                                    shouldReturn |= determine_validation_error_1D_arr<unsigned int>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                             expected, error, xOffsetValues[j], yOffsetValues[j], norm_offset_x, norm_offset_y, j, numTries, numClamped, false, lod );
                                } else {
                                    log_error("Test error: we should have detected this passing above.\n");
                                }
                            }//norm_offset_x
                        }//norm_offset_y
                        if( shouldReturn )
                            return 1;
                    } // if (!found_pixel)

                    resultPtr += 4;
                }
            }
        }
        /*
         * INT output type
         */
        else
        {
            // Validate integer results
            int *resultPtr = (int *)(char *)resultValues;
            int expected[4];
            float error;
            for( size_t y = 0, j = 0; y < imageInfo->arraySize; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, j++ )
                {
                    // Step 1: go through and see if the results verify for the pixel
                    // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                    // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                    int checkOnlyOnePixel = 0;
                    int found_pixel = 0;
                    for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                        for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                            // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                            // E.g., test one pixel.
                            if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                norm_offset_x = 0.0f;
                                norm_offset_y = 0.0f;
                                checkOnlyOnePixel = 1;
                            }

                            sample_image_pixel_offset<int>( imagePtr, imageInfo,
                                                           xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                           imageSampler, expected, lod );


                            error = errMax( errMax( abs_diff_int(expected[ 0 ], resultPtr[ 0 ]), abs_diff_int(expected[ 1 ], resultPtr[ 1 ]) ),
                                           errMax( abs_diff_int(expected[ 2 ], resultPtr[ 2 ]), abs_diff_int(expected[ 3 ], resultPtr[ 3 ]) ) );

                            if (error <= MAX_ERR)
                                found_pixel = 1;
                        }//norm_offset_x
                    }//norm_offset_y

                    // Step 2: If we did not find a match, then print out debugging info.
                    if (!found_pixel) {
                        // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        checkOnlyOnePixel = 0;
                        int shouldReturn = 0;
                        for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {

                                // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                // E.g., test one pixel.
                                if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                    norm_offset_x = 0.0f;
                                    norm_offset_y = 0.0f;
                                    checkOnlyOnePixel = 1;
                                }

                                sample_image_pixel_offset<int>( imagePtr, imageInfo,
                                                               xOffsetValues[ j ], yOffsetValues[ j ], 0.f, norm_offset_x, norm_offset_y, 0.0f,
                                                               imageSampler, expected, lod );


                                error = errMax( errMax( abs_diff_int(expected[ 0 ], resultPtr[ 0 ]), abs_diff_int(expected[ 1 ], resultPtr[ 1 ]) ),
                                               errMax( abs_diff_int(expected[ 2 ], resultPtr[ 2 ]), abs_diff_int(expected[ 3 ], resultPtr[ 3 ]) ) );

                                if( error > MAX_ERR )
                                {
                                    log_error("FAILED norm_offsets: %g , %g:\n", norm_offset_x, norm_offset_y);

                                    shouldReturn |= determine_validation_error_1D_arr<int>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                    expected, error, xOffsetValues[j], yOffsetValues[j], norm_offset_x, norm_offset_y, j, numTries, numClamped, false, lod );
                                } else {
                                    log_error("Test error: we should have detected this passing above.\n");
                                }
                            }//norm_offset_x
                        }//norm_offset_y
                        if( shouldReturn )
                            return 1;
                    } // if (!found_pixel)

                    resultPtr += 4;
                }
                }
            }
        }
        {
            nextLevelOffset += width_lod * imageInfo->arraySize * get_pixel_size(imageInfo->format);
            width_lod = (width_lod >> 1) ? (width_lod >> 1) : 1;
        }
    }

    return numTries != MAX_TRIES || numClamped != MAX_CLAMPED;
}

int test_read_image_set_1D_array(cl_device_id device, cl_context context,
                                 cl_command_queue queue,
                                 const cl_image_format *format,
                                 image_sampler_data *imageSampler,
                                 bool floatCoords, ExplicitType outputType)
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    clProgramWrapper program;
    clKernelWrapper kernel;
    RandomSeed seed( gRandomSeed );
    int error;
    const char *KernelSourcePattern = NULL;

    // Get our operating params
    size_t maxWidth, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0x0 };
    size_t pixelSize;

    imageInfo.format = format;
    imageInfo.depth = imageInfo.height = 0;
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    pixelSize = get_pixel_size( imageInfo.format );

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D array size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
    }

    // Determine types
    if( outputType == kInt )
        readFormat = "i";
    else if( outputType == kUInt )
        readFormat = "ui";
    else // kFloat
        readFormat = "f";

    // Construct the source
    const char *samplerArg = samplerKernelArg;
    char samplerVar[ 1024 ] = "";
    if( gUseKernelSamplers )
    {
        get_sampler_kernel_code( imageSampler, samplerVar );
        samplerArg = "";
    }

    if(gtestTypesToRun & kReadTests)
    {
        KernelSourcePattern = read1DArrayKernelSourcePattern;
    }
    else
    {
        KernelSourcePattern = read_write1DArrayKernelSourcePattern;
    }

    sprintf( programSrc,
            KernelSourcePattern,
            samplerArg, get_explicit_type_name( outputType ),
            gTestMipmaps ? ", float lod" : "",
            samplerVar,
            gTestMipmaps ? offset1DArrayLodKernelSource : offset1DArrayKernelSource,
            floatCoords ? floatKernelSource1DArray : intCoordKernelSource1DArray,
            readFormat,
            gTestMipmaps ? ", lod" : "" );

    ptr = programSrc;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    test_error( error, "Unable to create testing kernel" );

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.slicePitch = imageInfo.width * pixelSize;
            for( imageInfo.arraySize = 2; imageInfo.arraySize < 9; imageInfo.arraySize++ )
            {
                if(gTestMipmaps)
                    imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), seed);

                if( gDebugTrace )
                    log_info( "   at size %d,%d\n", (int)imageInfo.width, (int)imageInfo.arraySize );

                int retCode = test_read_image_1D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
                if( retCode )
                    return retCode;
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, maxArraySize, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE1D_ARRAY, imageInfo.format, CL_TRUE);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.arraySize = sizes[ idx ][ 2 ]; // 3rd dimension in get_max_sizes
            imageInfo.rowPitch = imageInfo.slicePitch = imageInfo.width * pixelSize;
            log_info("Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ]);
            if(gTestMipmaps)
                imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), seed);
            if( gDebugTrace )
                log_info( "   at max size %d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
            int retCode = test_read_image_1D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
            if( retCode )
                return retCode;
        }
    }
    else if( gTestRounding )
    {
        uint64_t typeRange = 1LL << ( get_format_type_size( imageInfo.format ) * 8 );
        typeRange /= pixelSize / get_format_type_size( imageInfo.format );
        imageInfo.arraySize = (size_t)( ( typeRange + 255LL ) / 256LL );
        imageInfo.width = (size_t)( typeRange / (cl_ulong)imageInfo.arraySize );
        while( imageInfo.arraySize >= maxArraySize / 2 )
        {
            imageInfo.width <<= 1;
            imageInfo.arraySize >>= 1;
        }

        while( imageInfo.width >= maxWidth / 2 )
            imageInfo.width >>= 1;
        imageInfo.rowPitch = imageInfo.slicePitch = imageInfo.width * pixelSize;

        gRoundingStartValue = 0;
        do
        {
            if( gDebugTrace )
                log_info( "   at size %d,%d, starting round ramp at %llu for range %llu\n", (int)imageInfo.width, (int)imageInfo.arraySize, gRoundingStartValue, typeRange );
            int retCode = test_read_image_1D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
            if( retCode )
                return retCode;

            gRoundingStartValue += imageInfo.width * imageInfo.arraySize * pixelSize / get_format_type_size( imageInfo.format );

        } while( gRoundingStartValue < typeRange );
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
                imageInfo.arraySize = (size_t)random_log_in_range( 16, (int)maxArraySize / 32, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                if(gTestMipmaps)
                {
                    imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), seed);
                    size = (cl_ulong) compute_mipmapped_image_size(imageInfo) * 4;
                }
                else
                {
                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.rowPitch += extraWidth * pixelSize;
                    }
                    imageInfo.slicePitch = imageInfo.rowPitch;

                    size = (size_t)imageInfo.rowPitch * (size_t)imageInfo.arraySize * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d (row pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)maxWidth, (int)maxArraySize );
            int retCode = test_read_image_1D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
            if( retCode )
                return retCode;
        }
    }

    return 0;
}
