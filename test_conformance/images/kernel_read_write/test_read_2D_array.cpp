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
#include "test_common.h"
#include <float.h>

#define MAX_ERR 0.005f
#define MAX_HALF_LINEAR_ERR 0.3f

extern bool         gDebugTrace, gExtraValidateInfo, gDisableOffsets, gTestSmallImages, gEnablePitch, gTestMaxImages, gTestMipmaps;
extern bool         gUseKernelSamplers;
extern cl_filter_mode   gFilterModeToUse;
extern cl_addressing_mode   gAddressModeToUse;
extern cl_mem_flags gMemFlagsToUse;
extern int gtestTypesToRun;
extern bool gDeviceLt20;
#define MAX_TRIES               1
#define MAX_CLAMPED             1

// Utility function to clamp down image sizes for certain tests to avoid
// using too much memory.
static size_t reduceImageSizeRange(size_t maxDimSize) {
  size_t DimSize = maxDimSize/128;
  if (DimSize < (size_t) 16)
    return 16;
  else if (DimSize > (size_t) 64)
    return 64;
  else
    return DimSize;
}

static size_t reduceImageDepth(size_t maxDepth) {
  size_t Depth = maxDepth/32;
  if (Depth < (size_t) 8)
    return 8;
  else if (Depth > (size_t) 32)
    return 32;
  else
    return Depth;
}

const char *read2DArrayKernelSourcePattern =
"__kernel void sample_kernel( read_only %s input,%s __global float *xOffsets, __global float *yOffsets, __global float *zOffsets,  __global %s%s *results %s )\n"
"{\n"
"%s"
"   int tidX = get_global_id(0), tidY = get_global_id(1), tidZ = get_global_id(2);\n"
"%s"
"%s"
"   results[offset] = read_image%s( input, imageSampler, coords %s);\n"
"}";

const char *read_write2DArrayKernelSourcePattern =
"__kernel void sample_kernel( read_write %s input,%s __global float *xOffsets, __global float *yOffsets, __global float *zOffsets,  __global %s%s *results %s)\n"
"{\n"
"%s"
"   int tidX = get_global_id(0), tidY = get_global_id(1), tidZ = get_global_id(2);\n"
"%s"
"%s"
"   results[offset] = read_image%s( input, coords %s);\n"
"}";

const char* offset2DarraySource ="   int offset = tidZ*get_image_width(input)*get_image_height(input) + tidY*get_image_width(input) + tidX;\n";
const char* offset2DarraySourceLod =
    "   int lod_int = (int)lod;\n"
    "   int width_lod, height_lod;\n"
    "   width_lod = (get_image_width(input) >> lod_int ) ? (get_image_width(input) >> lod_int ) : 1;\n"
    "   height_lod = (get_image_height(input) >> lod_int ) ? (get_image_height(input) >> lod_int ) : 1;\n"
    "   int offset = tidZ*width_lod*height_lod + tidY*width_lod + tidX;\n";

const char *int2DArrayCoordKernelSource =
"   int4 coords = (int4)( (int) xOffsets[offset], (int) yOffsets[offset], (int) zOffsets[offset], 0 );\n";

const char *float2DArrayUnnormalizedCoordKernelSource =
"   float4 coords = (float4)( xOffsets[offset], yOffsets[offset], zOffsets[offset], 0.0f );\n";


static const char *samplerKernelArg = " sampler_t imageSampler,";

extern void read_image_pixel_float( void *imageData, image_descriptor *imageInfo, int x, int y, int z, float *outData );
template <class T> int determine_validation_error_offset_2D_array( void *imagePtr, image_descriptor *imageInfo, image_sampler_data *imageSampler,
                                                         T *resultPtr, T * expected, float error,
                                                         float x, float y, float z, float xAddressOffset, float yAddressOffset, float zAddressOffset, size_t j, int &numTries, int &numClamped, bool printAsFloat, int lod )
{
    int actualX, actualY, actualZ;
    int found = debug_find_pixel_in_image( imagePtr, imageInfo, resultPtr, &actualX, &actualY, &actualZ, lod );
    bool clampingErr = false, clamped = false, otherClampingBug = false;
    int clampedX, clampedY, clampedZ;

    size_t imageWidth = imageInfo->width, imageHeight = imageInfo->height, imageDepth = imageInfo->arraySize;

    clamped = get_integer_coords_offset( x, y, z, xAddressOffset, yAddressOffset, zAddressOffset, imageWidth, imageHeight, imageDepth, imageSampler, imageInfo, clampedX, clampedY, clampedZ );

    if( found )
    {
        // Is it a clamping bug?
        if( clamped && clampedX == actualX && clampedY == actualY && clampedZ == actualZ )
        {
            if( (--numClamped) == 0 )
            {
                log_error( "\nERROR: TEST FAILED! Read is erroneously clamping coordinates!\n" );
                if( printAsFloat )
                {
                    log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                               "   Expected (%g,%g,%g,%g)\n"
                               "   Observed (%g,%g,%g,%g)\n"
                               "   error of %g\n",
                               j, x, x, y, y, z, z, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                               (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
                }
                else
                {
                    log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                               "   Expected (%x,%x,%x,%x)\n"
                               "   Observed (%x,%x,%x,%x)\n",
                               j, x, x, y, y, z, z, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                               (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
                }

                if( imageSampler->filter_mode != CL_FILTER_LINEAR )
                {
                  if( found )
                    log_error( "\tValue really found in image at %d,%d,%d (%s)\n", actualX, actualY, actualZ, ( found > 1 ) ? "NOT unique!!" : "unique" );
                  else
                    log_error( "\tValue not actually found in image\n" );
                }
                log_error( "\n" );

                return -1;
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
                log_error( "\nERROR: TEST FAILED: Clamping is erroneously returning border color!\n" );
                if( printAsFloat )
                {
                    log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                               "   Expected (%g,%g,%g,%g)\n"
                               "   Observed (%g,%g,%g,%g)\n"
                               "   error of %g\n",
                               j, x, x, y, y, z, z, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                               (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
                }
                else
                {
                    log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                               "   Expected (%x,%x,%x,%x)\n"
                               "   Observed (%x,%x,%x,%x)\n",
                               j, x, x, y, y, z, z, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                               (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
                }
                return -1;
            }
            clampingErr = true;
        }
    }
    if( !clampingErr )
    {
        if( true ) // gExtraValidateInfo )
        {
            if( printAsFloat )
            {
                log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                    "   Expected (%g,%g,%g,%g)\n"
                    "   Observed (%g,%g,%g,%g)\n"
                    "   error of %g\n",
                    j, x, x, y, y, z, z, (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ],
                    (float)resultPtr[ 0 ], (float)resultPtr[ 1 ], (float)resultPtr[ 2 ], (float)resultPtr[ 3 ], error );
            }
            else
            {
                log_error( "Sample %ld: coord {%f(%a),%f(%a),%f(%a)} did not validate!\n"
                    "   Expected (%x,%x,%x,%x)\n"
                    "   Observed (%x,%x,%x,%x)\n",
                    j, x, x, y, y, z, z, (int)expected[ 0 ], (int)expected[ 1 ], (int)expected[ 2 ], (int)expected[ 3 ],
                    (int)resultPtr[ 0 ], (int)resultPtr[ 1 ], (int)resultPtr[ 2 ], (int)resultPtr[ 3 ] );
            }
            log_error( "Integer coords resolve to %d,%d,%d, image size = %d,%d,%d\n", clampedX, clampedY, clampedZ, (int)imageWidth, (int)imageHeight, (int)imageDepth );

            if( printAsFloat && gExtraValidateInfo )
            {
                log_error( "\nNearby values:\n" );
                for( int zOff = -1; zOff <= 1; zOff++ )
                {
                    for( int yOff = -1; yOff <= 1; yOff++ )
                    {
                        float top[ 4 ], real[ 4 ], bot[ 4 ];
                        read_image_pixel_float( imagePtr, imageInfo, clampedX - 1 , clampedY + yOff, clampedZ + zOff, top );
                        read_image_pixel_float( imagePtr, imageInfo, clampedX ,clampedY + yOff, clampedZ + zOff, real );
                        read_image_pixel_float( imagePtr, imageInfo, clampedX + 1, clampedY + yOff, clampedZ + zOff, bot );
                        log_error( "\t(%g,%g,%g,%g)",top[0], top[1], top[2], top[3] );
                        log_error( " (%g,%g,%g,%g)", real[0], real[1], real[2], real[3] );
                        log_error( " (%g,%g,%g,%g)\n",bot[0], bot[1], bot[2], bot[3] );
                    }
                }
            }
            if( imageSampler->filter_mode != CL_FILTER_LINEAR )
            {
                if( found )
                    log_error( "Value really found in image at %d,%d,%d (%s)\n", actualX, actualY, actualZ, ( found > 1 ) ? "NOT unique!!" : "unique" );
                else
                    log_error( "Value not actually found in image\n" );
            }
            log_error( "\n" );
        }

        numClamped = -1; // We force the clamped counter to never work
        if( ( --numTries ) == 0 )
            return -1;
    }
    return 0;
}

#define CLAMP( _val, _min, _max )           ((_val) < (_min) ? (_min) : (_val) > (_max) ? (_max) : (_val))

static void InitFloatCoords( image_descriptor *imageInfo, image_sampler_data *imageSampler, float *xOffsets, float *yOffsets, float *zOffsets, float xfract, float yfract, float zfract, int normalized_coords, MTdata d , int lod)
{
    size_t i = 0;
    size_t width_lod = imageInfo->width;
    size_t height_lod = imageInfo->height;
    if(gTestMipmaps)
    {
        width_lod = ( imageInfo->width >> lod) ?( imageInfo->width >> lod)  : 1;
        height_lod = ( imageInfo->height >> lod) ?( imageInfo->height >> lod)  : 1;

    }
    if( gDisableOffsets )
    {
        for( size_t z = 0; z < imageInfo->arraySize; z++ )
        {
            for( size_t y = 0; y < height_lod; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, i++ )
                {
                    xOffsets[ i ] = (float) (xfract + (double) x);
                    yOffsets[ i ] = (float) (yfract + (double) y);
                    zOffsets[ i ] = (float) (zfract + (double) z);
                }
            }
        }
    }
    else
    {
        for( size_t z = 0; z < imageInfo->arraySize; z++ )
        {
            for( size_t y = 0; y < height_lod; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, i++ )
                {
                    xOffsets[ i ] = (float) (xfract + (double) ((int) x + random_in_range( -10, 10, d )));
                    yOffsets[ i ] = (float) (yfract + (double) ((int) y + random_in_range( -10, 10, d )));
                    zOffsets[ i ] = (float) (zfract + (double) ((int) z + random_in_range( -10, 10, d )));
                }
            }
        }
    }

    if( imageSampler->addressing_mode == CL_ADDRESS_NONE )
    {
        i = 0;
        for( size_t z = 0; z < imageInfo->arraySize; z++ )
        {
            for( size_t y = 0; y < height_lod; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, i++ )
                {
                    xOffsets[ i ] = (float) CLAMP( (double) xOffsets[ i ], 0.0, (double) width_lod - 1.0);
                    yOffsets[ i ] = (float) CLAMP( (double) yOffsets[ i ], 0.0, (double) height_lod - 1.0);
                    zOffsets[ i ] = (float) CLAMP( (double) zOffsets[ i ], 0.0, (double) imageInfo->arraySize - 1.0);
                }
            }
        }
    }

    if( normalized_coords )
    {
        i = 0;
        for( size_t z = 0; z < imageInfo->arraySize; z++ )
        {
            for( size_t y = 0; y < height_lod; y++ )
            {
                for( size_t x = 0; x < width_lod; x++, i++ )
                {
                    xOffsets[ i ] = (float) ((double) xOffsets[ i ] / (double) width_lod);
                    yOffsets[ i ] = (float) ((double) yOffsets[ i ] / (double) height_lod);
                }
            }
        }
    }
}

#ifndef MAX
#define MAX(_a, _b)             ((_a) > (_b) ? (_a) : (_b))
#endif

int test_read_image_2D_array( cl_context context, cl_command_queue queue, cl_kernel kernel,
                       image_descriptor *imageInfo, image_sampler_data *imageSampler,
                       bool useFloatCoords, ExplicitType outputType, MTdata d )
{
    int error;
    size_t threads[3];
    static int initHalf = 0;
    cl_mem_flags    image_read_write_flags = CL_MEM_READ_ONLY;

    clMemWrapper xOffsets, yOffsets, zOffsets, results;
    clSamplerWrapper actualSampler;
    BufferOwningPtr<char> maxImageUseHostPtrBackingStore;

    // Create offset data
    BufferOwningPtr<cl_float> xOffsetValues(malloc(sizeof(cl_float) *imageInfo->width * imageInfo->height * imageInfo->arraySize));
    BufferOwningPtr<cl_float> yOffsetValues(malloc(sizeof(cl_float) *imageInfo->width * imageInfo->height * imageInfo->arraySize));
    BufferOwningPtr<cl_float> zOffsetValues(malloc(sizeof(cl_float) *imageInfo->width * imageInfo->height * imageInfo->arraySize));

    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

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
            unprotImage = create_image_2d_array( context,
                                                image_read_write_flags | CL_MEM_USE_HOST_PTR,
                                                imageInfo->format,
                                                imageInfo->width, imageInfo->height,
                                                imageInfo->arraySize,
                                                ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                                ( gEnablePitch ? imageInfo->slicePitch : 0 ),
                                          maxImageUseHostPtrBackingStore, &error );
        } else {
            error = protImage.Create( context, CL_MEM_OBJECT_IMAGE2D_ARRAY,
                                    image_read_write_flags,
                                    imageInfo->format, imageInfo->width, imageInfo->height, 1, imageInfo->arraySize );
        }
        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (pitch %d, %d ) (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
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
        unprotImage = create_image_2d_array( context,
                                            image_read_write_flags | CL_MEM_COPY_HOST_PTR,
                                            imageInfo->format,
                                            imageInfo->width,
                                            imageInfo->height,
                                            imageInfo->arraySize,
                                            ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                            ( gEnablePitch ? imageInfo->slicePitch : 0 ),
                                      imageValues, &error );
        if( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (pitch %d, %d ) (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
            return error;
        }
        image = unprotImage;
    }
    else // Either CL_MEM_ALLOC_HOST_PTR or none
    {
        if ( gTestMipmaps )
        {
            cl_image_desc image_desc = {0};
            image_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
            image_desc.image_width = imageInfo->width;
            image_desc.image_height = imageInfo->height;
            image_desc.image_array_size = imageInfo->arraySize;
            //image_desc.image_rowPitch = imageInfo->rowPitch;
            //image_desc.image_slicePitch = imageInfo->slicePitch;
            image_desc.num_mip_levels = imageInfo->num_mip_levels;

            unprotImage = clCreateImage( context,
                                        image_read_write_flags,
                                        imageInfo->format, &image_desc, NULL, &error);
            if( error != CL_SUCCESS )
            {
                log_error( "ERROR: Unable to create %d level mipmapped 2D image array of size %d x %d x %d (pitch %d, %d ) (%s)", (int)imageInfo->num_mip_levels, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
                return error;
            }
        }
        else
        {
            // Note: if ALLOC_HOST_PTR is used, the driver allocates memory that can be accessed by the host, but otherwise
            // it works just as if no flag is specified, so we just do the same thing either way
            unprotImage = create_image_2d_array( context,
                                          image_read_write_flags | gMemFlagsToUse,
                                          imageInfo->format,
                                          imageInfo->width, imageInfo->height, imageInfo->arraySize,
                                          ( gEnablePitch ? imageInfo->rowPitch : 0 ),
                                          ( gEnablePitch ? imageInfo->slicePitch : 0 ),
                                          imageValues, &error );
            if( error != CL_SUCCESS )
            {
                log_error( "ERROR: Unable to create 2D image array of size %d x %d x %d (pitch %d, %d ) (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize, (int)imageInfo->rowPitch, (int)imageInfo->slicePitch, IGetErrorString( error ) );
                return error;
            }
        }
        image = unprotImage;
    }

    if( gMemFlagsToUse != CL_MEM_COPY_HOST_PTR )
    {
        if( gDebugTrace )
            log_info( " - Writing image...\n" );

        size_t origin[ 4 ] = { 0, 0, 0, 0 };
        size_t region[ 3 ] = { imageInfo->width, imageInfo->height, imageInfo->arraySize };
        size_t tmpNextLevelOffset = 0;

        if( gTestMipmaps )
        {
            for(int level = 0; level < imageInfo->num_mip_levels; level++)
            {
                origin[3] = level;
                error = clEnqueueWriteImage(queue, image, CL_TRUE,
                                            origin, region, /*gEnablePitch ? imageInfo->rowPitch :*/ 0, /*gEnablePitch ? imageInfo->slicePitch :*/ 0,
                                            imageValues + tmpNextLevelOffset, 0, NULL, NULL);
                if (error != CL_SUCCESS)
                {
                    log_error( "ERROR: Unable to write to level %d of 2D image array of size %d x %d x %d\n", (int)imageInfo->num_mip_levels, (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );
                    return error;
                }

                tmpNextLevelOffset += region[0]*region[1]*region[2]*get_pixel_size(imageInfo->format);
                region[0] = ( region[0] >> 1 ) ? ( region[0] >> 1 ) : 1;
                region[1] = ( region[1] >> 1 ) ? ( region[1] >> 1 ) : 1;
            }
       }
       else
       {
            error = clEnqueueWriteImage(queue, image, CL_TRUE,
                                        origin, region, gEnablePitch ? imageInfo->rowPitch : 0, gEnablePitch ? imageInfo->slicePitch : 0,
                                        imageValues, 0, NULL, NULL);
            if (error != CL_SUCCESS)
            {
                log_error( "ERROR: Unable to write to 2D image array of size %d x %d x %d\n", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->arraySize );
                return error;
            }
        }
    }

    xOffsets = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), sizeof( cl_float ) * imageInfo->width * imageInfo->height * imageInfo->arraySize, xOffsetValues, &error );
    test_error( error, "Unable to create x offset buffer" );
    yOffsets = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), sizeof( cl_float ) * imageInfo->width * imageInfo->height * imageInfo->arraySize, yOffsetValues, &error );
    test_error( error, "Unable to create y offset buffer" );
    zOffsets = clCreateBuffer( context, (cl_mem_flags)( CL_MEM_COPY_HOST_PTR ), sizeof( cl_float ) * imageInfo->width * imageInfo->height * imageInfo->arraySize, zOffsetValues, &error );
    test_error( error, "Unable to create y offset buffer" );
    results = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE),  get_explicit_type_size( outputType ) * 4 * imageInfo->width * imageInfo->height * imageInfo->arraySize, NULL, &error );
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
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &zOffsets );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &results );
    test_error( error, "Unable to set kernel arguments" );

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
    double formatAbsoluteError = get_max_absolute_error(imageInfo->format, imageSampler);
    if (gDebugTrace) log_info("\tformatAbsoluteError is %e\n", formatAbsoluteError);

    if (0 == initHalf && imageInfo->format->image_channel_data_type == CL_HALF_FLOAT ) {
        initHalf = CL_SUCCESS == DetectFloatToHalfRoundingMode( queue );
        if (initHalf) {
            log_info("Half rounding mode successfully detected.\n");
        }
    }
    size_t nextLevelOffset = 0;
    size_t width_lod = imageInfo->width, height_lod = imageInfo->height;
    for( size_t lod = 0; (gTestMipmaps && (lod < imageInfo->num_mip_levels))|| (!gTestMipmaps && lod < 1); lod ++)
    {
        size_t resultValuesSize = width_lod * height_lod * imageInfo->arraySize * get_explicit_type_size( outputType ) * 4;
        BufferOwningPtr<char> resultValues(malloc( resultValuesSize ));
        float lod_float = (float)lod;
        if( gTestMipmaps )
        {
            if(gDebugTrace)
                log_info(" - Working at mip level %d\n", lod);
            error = clSetKernelArg( kernel, idx, sizeof(float), &lod_float);
        }
    for( int q = 0; q < loopCount; q++ )
    {
        float offset = float_offsets[ q % float_offset_count ];

        // Init the coordinates
        InitFloatCoords( imageInfo, imageSampler, xOffsetValues, yOffsetValues, zOffsetValues,
                        q>=float_offset_count ? -offset: offset,
                        q>=float_offset_count ? offset: -offset,
                        q>=float_offset_count ? -offset: offset,
                        imageSampler->normalized_coords, d, lod );

        error = clEnqueueWriteBuffer( queue, xOffsets, CL_TRUE, 0, sizeof(cl_float) * imageInfo->height * imageInfo->width * imageInfo->arraySize, xOffsetValues, 0, NULL, NULL );
        test_error( error, "Unable to write x offsets" );
        error = clEnqueueWriteBuffer( queue, yOffsets, CL_TRUE, 0, sizeof(cl_float) * imageInfo->height * imageInfo->width * imageInfo->arraySize, yOffsetValues, 0, NULL, NULL );
        test_error( error, "Unable to write y offsets" );
        error = clEnqueueWriteBuffer( queue, zOffsets, CL_TRUE, 0, sizeof(cl_float) * imageInfo->height * imageInfo->width * imageInfo->arraySize, zOffsetValues, 0, NULL, NULL );
        test_error( error, "Unable to write z offsets" );


        memset( resultValues, 0xff, resultValuesSize );
        clEnqueueWriteBuffer( queue, results, CL_TRUE, 0, resultValuesSize, resultValues, 0, NULL, NULL );

        // Figure out thread dimensions
        threads[0] = (size_t)width_lod;
        threads[1] = (size_t)height_lod;
        threads[2] = (size_t)imageInfo->arraySize;

        // Run the kernel
        error = clEnqueueNDRangeKernel( queue, kernel, 3, NULL, threads, NULL, 0, NULL, NULL );
        test_error( error, "Unable to run kernel" );

        // Get results
        error = clEnqueueReadBuffer( queue, results, CL_TRUE, 0, width_lod * height_lod * imageInfo->arraySize * get_explicit_type_size( outputType ) * 4, resultValues, 0, NULL, NULL );
        test_error( error, "Unable to read results from kernel" );
        if( gDebugTrace )
            log_info( "    results read\n" );

        // Validate results element by element
        char *imagePtr = (char *)imageValues + nextLevelOffset;

        if((imageInfo->format->image_channel_order == CL_DEPTH) && (outputType == kFloat) )
        {
            // Validate float results
            float *resultPtr = (float *)(char *)resultValues;
            float expected[4], error=0.0f;
            float maxErr = get_max_relative_error( imageInfo->format, imageSampler, 1 /*3D*/, CL_FILTER_LINEAR == imageSampler->filter_mode );

            for( size_t z = 0, j = 0; z < imageInfo->arraySize; z++ )
            {
                for( size_t y = 0; y < height_lod; y++ )
                {
                    for( size_t x = 0; x < width_lod; x++, j++ )
                    {
                        // Step 1: go through and see if the results verify for the pixel
                        // For the normalized case on a GPU we put in offsets to the X, Y and Z to see if we land on the
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

                        for (float norm_offset_x = -offset; norm_offset_x <= offset && !found_pixel ; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -offset; norm_offset_y <= offset && !found_pixel ; norm_offset_y += NORM_OFFSET) {
                                for (float norm_offset_z = -offset; norm_offset_z <= NORM_OFFSET && !found_pixel; norm_offset_z += NORM_OFFSET) {

                                    int hasDenormals = 0;
                                    FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                          xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                          norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                          imageSampler, expected, 0, &hasDenormals, lod );

                                    float err1 =
                                        ABS_ERROR(resultPtr[0], expected[0]);
                                    // Clamp to the minimum absolute error for the format
                                    if (err1 > 0 && err1 < formatAbsoluteError) { err1 = 0.0f; }
                                    float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );

                                    if( ! (err1 <= maxErr1) )
                                    {
                                        // Try flushing the denormals
                                        if( hasDenormals )
                                        {
                                            // If implementation decide to flush subnormals to zero,
                                            // max error needs to be adjusted
                                            maxErr1 += 4 * FLT_MIN;

                                            maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                       xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                       norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                       imageSampler, expected, 0, NULL, lod );

                                            err1 = ABS_ERROR(resultPtr[0],
                                                             expected[0]);
                                        }
                                    }

                                    found_pixel = (err1 <= maxErr1);
                                }//norm_offset_z
                            }//norm_offset_y
                        }//norm_offset_x

                        // Step 2: If we did not find a match, then print out debugging info.
                        if (!found_pixel) {
                            // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                            // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                            checkOnlyOnePixel = 0;
                            int shouldReturn = 0;
                            for (float norm_offset_x = -offset; norm_offset_x <= offset && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                                for (float norm_offset_y = -offset; norm_offset_y <= offset && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                    for (float norm_offset_z = -offset; norm_offset_z <= offset && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                        int hasDenormals = 0;
                                        FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, expected, 0, &hasDenormals, lod );

                                        float err1 = ABS_ERROR(resultPtr[0],
                                                               expected[0]);
                                        float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );


                                        if( ! (err1 <= maxErr1) )
                                        {
                                            // Try flushing the denormals
                                            if( hasDenormals )
                                            {
                                                maxErr1 += 4 * FLT_MIN;

                                                maxPixel = sample_image_pixel_float( imagePtr, imageInfo,
                                                                                    xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                    imageSampler, expected, 0, NULL, lod );

                                                err1 = ABS_ERROR(resultPtr[0],
                                                                 expected[0]);
                                            }
                                        }

                                        if( ! (err1 <= maxErr1) )
                                        {
                                            log_error("FAILED norm_offsets: %g , %g , %g:\n", norm_offset_x, norm_offset_y, norm_offset_z);

                                            float tempOut[4];
                                            shouldReturn |= determine_validation_error_offset_2D_array<float>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                                     expected, error, xOffsetValues[j], yOffsetValues[j], zOffsetValues[j],
                                                                                                     norm_offset_x, norm_offset_y, norm_offset_z, j,
                                                                                                     numTries, numClamped, true, lod );
                                            log_error( "Step by step:\n" );
                                            FloatPixel temp = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, tempOut, 1 /*verbose*/, &hasDenormals, lod);
                                            log_error( "\tulps: %2.2f  (max allowed: %2.2f)\n\n",
                                                      Ulp_Error( resultPtr[0], expected[0] ),
                                                      Ulp_Error( MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) + maxErr, MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) ) );
                                        } else {
                                            log_error("Test error: we should have detected this passing above.\n");
                                        }
                                    }//norm_offset_z
                                }//norm_offset_y
                            }//norm_offset_x
                            if( shouldReturn )
                                return 1;
                        } // if (!found_pixel)

                        resultPtr += 1;
                    }
                }
            }
        }
        /*
         * FLOAT output type, order=CL_sRGBA, CL_sRGB, CL_sRGBx, CL_BGRA
         */
        else if(is_sRGBA_order(imageInfo->format->image_channel_order) && (outputType == kFloat) )
        {
            // Validate float results
            float *resultPtr = (float *)(char *)resultValues;
            float expected[4], error=0.0f;
            float maxErr = get_max_relative_error( imageInfo->format, imageSampler, 1 /*3D*/, CL_FILTER_LINEAR == imageSampler->filter_mode );

            for( size_t z = 0, j = 0; z < imageInfo->arraySize; z++ )
            {
                for( size_t y = 0; y < height_lod; y++ )
                {
                    for( size_t x = 0; x < width_lod; x++, j++ )
                    {
                        // Step 1: go through and see if the results verify for the pixel
                        // For the normalized case on a GPU we put in offsets to the X, Y and Z to see if we land on the
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

                        for (float norm_offset_x = -offset; norm_offset_x <= offset && !found_pixel ; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -offset; norm_offset_y <= offset && !found_pixel ; norm_offset_y += NORM_OFFSET) {
                                for (float norm_offset_z = -offset; norm_offset_z <= NORM_OFFSET && !found_pixel; norm_offset_z += NORM_OFFSET) {

                                    int hasDenormals = 0;
                                    FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                          xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                          norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                          imageSampler, expected, 0, &hasDenormals, lod );

                                    float err1 =
                                        ABS_ERROR(sRGBmap(resultPtr[0]),
                                                  sRGBmap(expected[0]));
                                    float err2 =
                                        ABS_ERROR(sRGBmap(resultPtr[1]),
                                                  sRGBmap(expected[1]));
                                    float err3 =
                                        ABS_ERROR(sRGBmap(resultPtr[2]),
                                                  sRGBmap(expected[2]));
                                    float err4 =
                                        ABS_ERROR(resultPtr[3], expected[3]);
                                    float maxErr = 0.5;

                                    if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    || ! (err3 <= maxErr) || ! (err4 <= maxErr) )
                                    {
                                        // Try flushing the denormals
                                        if( hasDenormals )
                                        {
                                            // If implementation decide to flush subnormals to zero,
                                            // max error needs to be adjusted
                                              maxErr += 4 * FLT_MIN;

                                            maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                       xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                       norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                       imageSampler, expected, 0, NULL, lod );

                                            err1 =
                                                ABS_ERROR(sRGBmap(resultPtr[0]),
                                                          sRGBmap(expected[0]));
                                            err2 =
                                                ABS_ERROR(sRGBmap(resultPtr[1]),
                                                          sRGBmap(expected[1]));
                                            err3 =
                                                ABS_ERROR(sRGBmap(resultPtr[2]),
                                                          sRGBmap(expected[2]));
                                            err4 = ABS_ERROR(resultPtr[3],
                                                             expected[3]);
                                        }
                                    }

                                    found_pixel = (err1 <= maxErr) && (err2 <= maxErr)  && (err3 <= maxErr) && (err4 <= maxErr);
                                }//norm_offset_z
                            }//norm_offset_y
                        }//norm_offset_x

                        // Step 2: If we did not find a match, then print out debugging info.
                        if (!found_pixel) {
                            // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                            // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                            checkOnlyOnePixel = 0;
                            int shouldReturn = 0;
                            for (float norm_offset_x = -offset; norm_offset_x <= offset && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                                for (float norm_offset_y = -offset; norm_offset_y <= offset && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                    for (float norm_offset_z = -offset; norm_offset_z <= offset && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                        int hasDenormals = 0;
                                        FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, expected, 0, &hasDenormals, lod );

                                        float err1 =
                                            ABS_ERROR(sRGBmap(resultPtr[0]),
                                                      sRGBmap(expected[0]));
                                        float err2 =
                                            ABS_ERROR(sRGBmap(resultPtr[1]),
                                                      sRGBmap(expected[1]));
                                        float err3 =
                                            ABS_ERROR(sRGBmap(resultPtr[2]),
                                                      sRGBmap(expected[2]));
                                        float err4 = ABS_ERROR(resultPtr[3],
                                                               expected[3]);
                                        float maxErr = 0.6;

                                        if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    || ! (err3 <= maxErr) || ! (err4 <= maxErr) )
                                        {
                                            // Try flushing the denormals
                                            if( hasDenormals )
                                            {
                                                // If implementation decide to flush subnormals to zero,
                                                // max error needs to be adjusted
                                                maxErr += 4 * FLT_MIN;

                                                maxPixel = sample_image_pixel_float( imagePtr, imageInfo,
                                                                                    xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                    imageSampler, expected, 0, NULL, lod );

                                                err1 = ABS_ERROR(
                                                    sRGBmap(resultPtr[0]),
                                                    sRGBmap(expected[0]));
                                                err2 = ABS_ERROR(
                                                    sRGBmap(resultPtr[1]),
                                                    sRGBmap(expected[1]));
                                                err3 = ABS_ERROR(
                                                    sRGBmap(resultPtr[2]),
                                                    sRGBmap(expected[2]));
                                                err4 = ABS_ERROR(resultPtr[3],
                                                                 expected[3]);
                                            }
                                        }

                                        if( ! (err1 <= maxErr) || ! (err2 <= maxErr)    || ! (err3 <= maxErr) || ! (err4 <= maxErr) )
                                        {
                                            log_error("FAILED norm_offsets: %g , %g , %g:\n", norm_offset_x, norm_offset_y, norm_offset_z);

                                            float tempOut[4];
                                            shouldReturn |= determine_validation_error_offset_2D_array<float>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                                     expected, error, xOffsetValues[j], yOffsetValues[j], zOffsetValues[j],
                                                                                                     norm_offset_x, norm_offset_y, norm_offset_z, j,
                                                                                                     numTries, numClamped, true, lod );
                                            log_error( "Step by step:\n" );
                                            FloatPixel temp = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, tempOut, 1 /*verbose*/, &hasDenormals, lod);
                                            log_error( "\tulps: %2.2f, %2.2f, %2.2f, %2.2f  (max allowed: %2.2f)\n\n",
                                                      Ulp_Error( resultPtr[0], expected[0] ),
                                                      Ulp_Error( resultPtr[1], expected[1] ),
                                                      Ulp_Error( resultPtr[2], expected[2] ),
                                                      Ulp_Error( resultPtr[3], expected[3] ),
                                                      Ulp_Error( MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) + maxErr, MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) ) );
                                        } else {
                                            log_error("Test error: we should have detected this passing above.\n");
                                        }
                                    }//norm_offset_z
                                }//norm_offset_y
                            }//norm_offset_x
                            if( shouldReturn )
                                return 1;
                        } // if (!found_pixel)

                        resultPtr += 4;
                    }
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
            float maxErr = get_max_relative_error( imageInfo->format, imageSampler, 1 /*3D*/, CL_FILTER_LINEAR == imageSampler->filter_mode );

            for( size_t z = 0, j = 0; z < imageInfo->arraySize; z++ )
            {
                for( size_t y = 0; y < height_lod; y++ )
                {
                    for( size_t x = 0; x < width_lod; x++, j++ )
                    {
                        // Step 1: go through and see if the results verify for the pixel
                        // For the normalized case on a GPU we put in offsets to the X, Y and Z to see if we land on the
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

                        for (float norm_offset_x = -offset; norm_offset_x <= offset && !found_pixel ; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -offset; norm_offset_y <= offset && !found_pixel ; norm_offset_y += NORM_OFFSET) {
                                for (float norm_offset_z = -offset; norm_offset_z <= NORM_OFFSET && !found_pixel; norm_offset_z += NORM_OFFSET) {

                                    int hasDenormals = 0;
                                    FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                          xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                          norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                          imageSampler, expected, 0, &hasDenormals, lod );

                                    float err1 =
                                        ABS_ERROR(resultPtr[0], expected[0]);
                                    float err2 =
                                        ABS_ERROR(resultPtr[1], expected[1]);
                                    float err3 =
                                        ABS_ERROR(resultPtr[2], expected[2]);
                                    float err4 =
                                        ABS_ERROR(resultPtr[3], expected[3]);
                                    // Clamp to the minimum absolute error for the format
                                    if (err1 > 0 && err1 < formatAbsoluteError) { err1 = 0.0f; }
                                    if (err2 > 0 && err2 < formatAbsoluteError) { err2 = 0.0f; }
                                    if (err3 > 0 && err3 < formatAbsoluteError) { err3 = 0.0f; }
                                    if (err4 > 0 && err4 < formatAbsoluteError) { err4 = 0.0f; }
                                    float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );
                                    float maxErr2 = MAX( maxErr * maxPixel.p[1], FLT_MIN );
                                    float maxErr3 = MAX( maxErr * maxPixel.p[2], FLT_MIN );
                                    float maxErr4 = MAX( maxErr * maxPixel.p[3], FLT_MIN );

                                    if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    || ! (err3 <= maxErr3) || ! (err4 <= maxErr4) )
                                    {
                                        // Try flushing the denormals
                                        if( hasDenormals )
                                        {
                                            // If implementation decide to flush subnormals to zero,
                                            // max error needs to be adjusted
                                            maxErr1 += 4 * FLT_MIN;
                                            maxErr2 += 4 * FLT_MIN;
                                            maxErr3 += 4 * FLT_MIN;
                                            maxErr4 += 4 * FLT_MIN;

                                            maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                       xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                       norm_offset_x, norm_offset_y, norm_offset_z,
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

                                    found_pixel = (err1 <= maxErr1) && (err2 <= maxErr2)  && (err3 <= maxErr3) && (err4 <= maxErr4);
                                }//norm_offset_z
                            }//norm_offset_y
                        }//norm_offset_x

                        // Step 2: If we did not find a match, then print out debugging info.
                        if (!found_pixel) {
                            // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                            // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                            checkOnlyOnePixel = 0;
                            int shouldReturn = 0;
                            for (float norm_offset_x = -offset; norm_offset_x <= offset && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                                for (float norm_offset_y = -offset; norm_offset_y <= offset && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                    for (float norm_offset_z = -offset; norm_offset_z <= offset && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                        int hasDenormals = 0;
                                        FloatPixel maxPixel = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, expected, 0, &hasDenormals, lod );

                                        float err1 = ABS_ERROR(resultPtr[0],
                                                               expected[0]);
                                        float err2 = ABS_ERROR(resultPtr[1],
                                                               expected[1]);
                                        float err3 = ABS_ERROR(resultPtr[2],
                                                               expected[2]);
                                        float err4 = ABS_ERROR(resultPtr[3],
                                                               expected[3]);
                                        float maxErr1 = MAX( maxErr * maxPixel.p[0], FLT_MIN );
                                        float maxErr2 = MAX( maxErr * maxPixel.p[1], FLT_MIN );
                                        float maxErr3 = MAX( maxErr * maxPixel.p[2], FLT_MIN );
                                        float maxErr4 = MAX( maxErr * maxPixel.p[3], FLT_MIN );


                                        if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    || ! (err3 <= maxErr3) || ! (err4 <= maxErr4) )
                                        {
                                            // Try flushing the denormals
                                            if( hasDenormals )
                                            {
                                                maxErr1 += 4 * FLT_MIN;
                                                maxErr2 += 4 * FLT_MIN;
                                                maxErr3 += 4 * FLT_MIN;
                                                maxErr4 += 4 * FLT_MIN;

                                                maxPixel = sample_image_pixel_float( imagePtr, imageInfo,
                                                                                    xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
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

                                        if( ! (err1 <= maxErr1) || ! (err2 <= maxErr2)    || ! (err3 <= maxErr3) || ! (err4 <= maxErr4) )
                                        {
                                            log_error("FAILED norm_offsets: %g , %g , %g:\n", norm_offset_x, norm_offset_y, norm_offset_z);

                                            float tempOut[4];
                                            shouldReturn |= determine_validation_error_offset_2D_array<float>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                                     expected, error, xOffsetValues[j], yOffsetValues[j], zOffsetValues[j],
                                                                                                     norm_offset_x, norm_offset_y, norm_offset_z, j,
                                                                                                     numTries, numClamped, true, lod );
                                            log_error( "Step by step:\n" );
                                            FloatPixel temp = sample_image_pixel_float_offset( imagePtr, imageInfo,
                                                                                              xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                              norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                              imageSampler, tempOut, 1 /*verbose*/, &hasDenormals, lod);
                                            log_error( "\tulps: %2.2f, %2.2f, %2.2f, %2.2f  (max allowed: %2.2f)\n\n",
                                                      Ulp_Error( resultPtr[0], expected[0] ),
                                                      Ulp_Error( resultPtr[1], expected[1] ),
                                                      Ulp_Error( resultPtr[2], expected[2] ),
                                                      Ulp_Error( resultPtr[3], expected[3] ),
                                                      Ulp_Error( MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) + maxErr, MAKE_HEX_FLOAT(0x1.000002p0f, 0x1000002L, -24) ) );
                                        } else {
                                            log_error("Test error: we should have detected this passing above.\n");
                                        }
                                    }//norm_offset_z
                                }//norm_offset_y
                            }//norm_offset_x
                            if( shouldReturn )
                                return 1;
                        } // if (!found_pixel)

                        resultPtr += 4;
                    }
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
            for( size_t z = 0, j = 0; z < imageInfo->arraySize; z++ )
            {
                for( size_t y = 0; y < height_lod; y++ )
                {
                    for( size_t x = 0; x < width_lod; x++, j++ )
                    {
                        // Step 1: go through and see if the results verify for the pixel
                        // For the normalized case on a GPU we put in offsets to the X, Y and Z to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        int checkOnlyOnePixel = 0;
                        int found_pixel = 0;
                        for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                for (float norm_offset_z = -NORM_OFFSET; norm_offset_z <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                    // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                    // E.g., test one pixel.
                                    if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                        norm_offset_x = 0.0f;
                                        norm_offset_y = 0.0f;
                                        norm_offset_z = 0.0f;
                                        checkOnlyOnePixel = 1;
                                    }

                                        if(gTestMipmaps)
                                            sample_image_pixel_offset<unsigned int>( imagePtr, imageInfo,
                                                                                    xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                    norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                    imageSampler, expected, lod );
                                        else
                                            sample_image_pixel_offset<unsigned int>( imageValues, imageInfo,
                                                                                    xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                    norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                    imageSampler, expected );

                                    error = errMax( errMax( abs_diff_uint(expected[ 0 ], resultPtr[ 0 ]), abs_diff_uint(expected[ 1 ], resultPtr[ 1 ]) ),
                                                   errMax( abs_diff_uint(expected[ 2 ], resultPtr[ 2 ]), abs_diff_uint(expected[ 3 ], resultPtr[ 3 ]) ) );

                                    if (error < MAX_ERR)
                                        found_pixel = 1;
                                }//norm_offset_z
                            }//norm_offset_y
                        }//norm_offset_x

                        // Step 2: If we did not find a match, then print out debugging info.
                        if (!found_pixel) {
                            // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                            // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                            checkOnlyOnePixel = 0;
                            int shouldReturn = 0;
                            for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                                for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                    for (float norm_offset_z = -NORM_OFFSET; norm_offset_z <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                        // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                        // E.g., test one pixel.
                                        if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                            norm_offset_x = 0.0f;
                                            norm_offset_y = 0.0f;
                                            norm_offset_z = 0.0f;
                                            checkOnlyOnePixel = 1;
                                        }

                                            if(gTestMipmaps)
                                                sample_image_pixel_offset<unsigned int>( imagePtr, imageInfo,
                                                                                        xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                        norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                        imageSampler, expected, lod );
                                            else
                                                sample_image_pixel_offset<unsigned int>( imageValues, imageInfo,
                                                                                        xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                                        norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                        imageSampler, expected );

                                        error = errMax( errMax( abs_diff_uint(expected[ 0 ], resultPtr[ 0 ]), abs_diff_uint(expected[ 1 ], resultPtr[ 1 ]) ),
                                                       errMax( abs_diff_uint(expected[ 2 ], resultPtr[ 2 ]), abs_diff_uint(expected[ 3 ], resultPtr[ 3 ]) ) );

                                        if( error > MAX_ERR )
                                        {
                                            log_error("FAILED norm_offsets: %g , %g , %g:\n", norm_offset_x, norm_offset_y, norm_offset_z);
                                            shouldReturn |=  determine_validation_error_offset_2D_array<unsigned int>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                                             expected, error, xOffsetValues[j], yOffsetValues[j], zOffsetValues[j],
                                                                                                             norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                                             j, numTries, numClamped, false, lod );
                                        } else {
                                            log_error("Test error: we should have detected this passing above.\n");
                                        }
                                    }//norm_offset_z
                                }//norm_offset_y
                            }//norm_offset_x
                            if( shouldReturn )
                                return 1;
                        } // if (!found_pixel)

                        resultPtr += 4;
                    }
                }
            }
        }
        else
        /*
         * INT output type
         */
        {
            // Validate integer results
            int *resultPtr = (int *)(char *)resultValues;
            int expected[4];
            float error;
            for( size_t z = 0, j = 0; z < imageInfo->arraySize; z++ )
            {
                for( size_t y = 0; y < height_lod; y++ )
                {
                    for( size_t x = 0; x < width_lod; x++, j++ )
                    {
                        // Step 1: go through and see if the results verify for the pixel
                        // For the normalized case on a GPU we put in offsets to the X, Y and Z to see if we land on the
                        // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                        int checkOnlyOnePixel = 0;
                        int found_pixel = 0;
                        for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                            for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                for (float norm_offset_z = -NORM_OFFSET; norm_offset_z <= NORM_OFFSET && !found_pixel && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                    // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                    // E.g., test one pixel.
                                    if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0) {
                                        norm_offset_x = 0.0f;
                                        norm_offset_y = 0.0f;
                                        norm_offset_z = 0.0f;
                                        checkOnlyOnePixel = 1;
                                    }

                                        if(gTestMipmaps)
                                            sample_image_pixel_offset<int>( imagePtr, imageInfo,
                                                                           xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                           norm_offset_x, norm_offset_y, norm_offset_z,
                                                                           imageSampler, expected, lod );
                                        else
                                            sample_image_pixel_offset<int>( imageValues, imageInfo,
                                                                           xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                           norm_offset_x, norm_offset_y, norm_offset_z,
                                                                           imageSampler, expected );

                                    error = errMax( errMax( abs_diff_int(expected[ 0 ], resultPtr[ 0 ]), abs_diff_int(expected[ 1 ], resultPtr[ 1 ]) ),
                                                   errMax( abs_diff_int(expected[ 2 ], resultPtr[ 2 ]), abs_diff_int(expected[ 3 ], resultPtr[ 3 ]) ) );

                                    if (error < MAX_ERR)
                                        found_pixel = 1;
                                }//norm_offset_z
                            }//norm_offset_y
                        }//norm_offset_x

                        // Step 2: If we did not find a match, then print out debugging info.
                        if (!found_pixel) {
                            // For the normalized case on a GPU we put in offsets to the X and Y to see if we land on the
                            // right pixel. This addresses the significant inaccuracy in GPU normalization in OpenCL 1.0.
                            checkOnlyOnePixel = 0;
                            int shouldReturn = 0;
                            for (float norm_offset_x = -NORM_OFFSET; norm_offset_x <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_x += NORM_OFFSET) {
                                for (float norm_offset_y = -NORM_OFFSET; norm_offset_y <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_y += NORM_OFFSET) {
                                    for (float norm_offset_z = -NORM_OFFSET; norm_offset_z <= NORM_OFFSET && !checkOnlyOnePixel; norm_offset_z += NORM_OFFSET) {

                                        // If we are not on a GPU, or we are not normalized, then only test with offsets (0.0, 0.0)
                                        // E.g., test one pixel.
                                        if (!imageSampler->normalized_coords || gDeviceType != CL_DEVICE_TYPE_GPU || NORM_OFFSET == 0 || NORM_OFFSET == 0 || NORM_OFFSET == 0) {
                                            norm_offset_x = 0.0f;
                                            norm_offset_y = 0.0f;
                                            norm_offset_z = 0.0f;
                                            checkOnlyOnePixel = 1;
                                        }

                                            if(gTestMipmaps)
                                                sample_image_pixel_offset<int>( imagePtr, imageInfo,
                                                                               xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                               norm_offset_x, norm_offset_y, norm_offset_z,
                                                                               imageSampler, expected, lod );
                                            else
                                                sample_image_pixel_offset<int>( imageValues, imageInfo,
                                                                               xOffsetValues[ j ], yOffsetValues[ j ], zOffsetValues[ j ],
                                                                               norm_offset_x, norm_offset_y, norm_offset_z,
                                                                               imageSampler, expected, 0 );

                                        error = errMax( errMax( abs_diff_int(expected[ 0 ], resultPtr[ 0 ]), abs_diff_int(expected[ 1 ], resultPtr[ 1 ]) ),
                                                       errMax( abs_diff_int(expected[ 2 ], resultPtr[ 2 ]), abs_diff_int(expected[ 3 ], resultPtr[ 3 ]) ) );

                                        if( error > MAX_ERR )
                                        {
                                            log_error("FAILED norm_offsets: %g , %g , %g:\n", norm_offset_x, norm_offset_y, norm_offset_z);
                                            shouldReturn |=  determine_validation_error_offset_2D_array<int>( imagePtr, imageInfo, imageSampler, resultPtr,
                                                                                                    expected, error, xOffsetValues[j], yOffsetValues[j], zOffsetValues[j],
                                                                                                    norm_offset_x, norm_offset_y, norm_offset_z,
                                                                                                    j, numTries, numClamped, false, lod );
                                        } else {
                                            log_error("Test error: we should have detected this passing above.\n");
                                        }
                                    }//norm_offset_z
                                }//norm_offset_y
                            }//norm_offset_x
                            if( shouldReturn )
                                return 1;
                        } // if (!found_pixel)

                        resultPtr += 4;
                    }
                }
            }
        }
        }
        {
            nextLevelOffset += width_lod * height_lod * imageInfo->arraySize * get_pixel_size(imageInfo->format);
            width_lod = ( width_lod >> 1 ) ? ( width_lod >> 1 ) : 1;
            height_lod = ( height_lod >> 1 ) ? ( height_lod >> 1 ) : 1;
        }
    }

    return numTries != MAX_TRIES || numClamped != MAX_CLAMPED;
}

int test_read_image_set_2D_array( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                           bool floatCoords, ExplicitType outputType )
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    RandomSeed seed( gRandomSeed );

    const char *KernelSourcePattern = NULL;

    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;

    // Get operating parameters
    size_t maxWidth, maxHeight, maxArraySize;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0x0 };
    size_t pixelSize;

    imageInfo.format = format;
    imageInfo.type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    pixelSize = get_pixel_size( imageInfo.format );

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxArraySize ), &maxArraySize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 3D size from device" );

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

    const char *imageType;
    const char *imageElement;
    if (format->image_channel_order == CL_DEPTH)
    {
        imageType = "image2d_array_depth_t";
        imageElement = "";
    }
    else
    {
        imageType = "image2d_array_t";
        imageElement = "4";
    }

    // Construct the source
    if(gtestTypesToRun & kReadTests)
    {
        KernelSourcePattern = read2DArrayKernelSourcePattern;
    }
    else
    {
        KernelSourcePattern = read_write2DArrayKernelSourcePattern;
    }

    // Construct the source
    sprintf( programSrc,
            KernelSourcePattern,
            imageType,
            samplerArg, get_explicit_type_name( outputType ),
            imageElement,
            gTestMipmaps ? ", float lod" : " ",
            samplerVar,
            gTestMipmaps ? offset2DarraySourceLod : offset2DarraySource,
            floatCoords ? float2DArrayUnnormalizedCoordKernelSource : int2DArrayCoordKernelSource,
            readFormat,
            gTestMipmaps ? ", lod" : " " );

    ptr = programSrc;
    error = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &ptr, "sample_kernel", gDeviceLt20 ? "" : "-cl-std=CL2.0");
    test_error( error, "Unable to create testing kernel" );

    // Run tests

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                for( imageInfo.arraySize = 2; imageInfo.arraySize < 9; imageInfo.arraySize++ )
                {
                    if( gTestMipmaps )
                        imageInfo.num_mip_levels = (size_t) random_in_range(2, compute_max_mip_levels(imageInfo.width, imageInfo.height, 0)-1, seed);

                    if( gDebugTrace )
                        log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize );
                    int retCode = test_read_image_2D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
                    if( retCode )
                        return retCode;
                }
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, maxArraySize, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE2D_ARRAY, imageInfo.format, CL_TRUE);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.arraySize = sizes[ idx ][ 2 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
            if( gTestMipmaps )
                imageInfo.num_mip_levels = (size_t) random_in_range(2, compute_max_mip_levels(imageInfo.width, imageInfo.height, 0)-1, seed);
            cl_ulong size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            while(  size > maxAllocSize || ( size * 3 ) > memSize )
            {
                if(imageInfo.arraySize == 1)
                {
                    // ArraySize cannot be 0.
                    break;
                }
                imageInfo.arraySize--;
                size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
            }

            while(  size > maxAllocSize || ( size * 3 ) > memSize )
            {
                imageInfo.height--;
                imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
                size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
            }
            log_info("Testing %d x %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ]);
            if( gDebugTrace )
                log_info( "   at max size %d,%d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            int retCode = test_read_image_2D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
            if( retCode )
                return retCode;
        }
    }
    else if( gTestRounding )
    {
        size_t typeRange = 1 << ( get_format_type_size( imageInfo.format ) * 8 );
        imageInfo.height = typeRange / 256;
        imageInfo.width = (size_t)( typeRange / (cl_ulong)imageInfo.height );
        imageInfo.arraySize = 2;

        imageInfo.rowPitch = imageInfo.width * pixelSize;
        imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
        int retCode = test_read_image_2D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
        if( retCode )
            return retCode;
    }
    else
    {
        int maxWidthRange = (int) reduceImageSizeRange(maxWidth);
        int maxHeighthRange = (int) reduceImageSizeRange(maxHeight);
        int maxArraySizeRange = (int) reduceImageDepth(maxArraySize);

        for( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width = (size_t)random_log_in_range( 16, maxWidthRange, seed );
                imageInfo.height = (size_t)random_log_in_range( 16, maxHeighthRange, seed );
                imageInfo.arraySize = (size_t)random_log_in_range( 8, maxArraySizeRange, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;

                if( gTestMipmaps )
                {
                    imageInfo.num_mip_levels = random_in_range(2,compute_max_mip_levels(imageInfo.width, imageInfo.height, 0) - 1, seed);
                    //Need to take into account the output buffer size, otherwise we will end up with input buffer that is exceeding MaxAlloc
                    size = (cl_ulong) 4*compute_mipmapped_image_size( imageInfo ) * get_explicit_type_size( outputType );
                }
                else
                {
                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.rowPitch += extraWidth * pixelSize;

                        size_t extraHeight = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + extraHeight);
                    }

                    size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.arraySize * 4 * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
            {
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.arraySize, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxArraySize );
                if ( gTestMipmaps )
                    log_info("  and %d mip levels\n", (int) imageInfo.num_mip_levels);
            }
            int retCode = test_read_image_2D_array( context, queue, kernel, &imageInfo, imageSampler, floatCoords, outputType, seed );
            if( retCode )
                return retCode;
        }
    }

    return 0;
}
