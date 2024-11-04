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

// Utility function to clamp down image sizes for certain tests to avoid
// using too much memory.
static size_t reduceImageSizeRange(size_t maxDimSize, RandomSeed& seed) {
  size_t DimSize = random_log_in_range(16, (int) maxDimSize/32, seed);
  if (DimSize > (size_t) 128)
    return 128;
  else
    return DimSize;
}

static size_t reduceImageDepth(size_t maxDimSize, RandomSeed& seed) {
  size_t DimSize = random_log_in_range(8, (int) maxDimSize/32, seed);
  if (DimSize > (size_t) 32)
    return 32;
  else
    return DimSize;
}


const char *read3DKernelSourcePattern =
    "%s\n"
    "__kernel void sample_kernel( read_only image3d_t input,%s __global float "
    "*xOffsets, __global float *yOffsets, __global float *zOffsets,  __global "
    "%s4 *results %s)\n"
    "{\n"
    "%s"
    "   int tidX = get_global_id(0), tidY = get_global_id(1), tidZ = "
    "get_global_id(2);\n"
    "%s"
    "%s"
    "   results[offset] = read_image%s( input, imageSampler, coords %s);\n"
    "}";

const char *read_write3DKernelSourcePattern =
    "%s\n"
    "__kernel void sample_kernel( read_write image3d_t input,%s __global float "
    "*xOffsets, __global float *yOffsets, __global float *zOffsets,  __global "
    "%s4 *results %s)\n"
    "{\n"
    "%s"
    "   int tidX = get_global_id(0), tidY = get_global_id(1), tidZ = "
    "get_global_id(2);\n"
    "%s"
    "%s"
    "   results[offset] = read_image%s( input, coords %s);\n"
    "}";

const char *offset3DKernelSource =
"   int offset = tidZ*get_image_width(input)*get_image_height(input) + tidY*get_image_width(input) + tidX;\n";

const char *offset3DLodKernelSource =
"   int lod_int = (int)lod;\n"
"   int width_lod = (get_image_width(input) >> lod_int) ?(get_image_width(input) >> lod_int): 1;\n"
"   int height_lod = (get_image_height(input) >> lod_int) ?(get_image_height(input) >> lod_int): 1;\n"
"   int offset = tidZ*width_lod*height_lod + tidY*width_lod + tidX;\n";

const char *int3DCoordKernelSource =
"   int4 coords = (int4)( (int) xOffsets[offset], (int) yOffsets[offset], (int) zOffsets[offset], 0 );\n";

const char *float3DUnnormalizedCoordKernelSource =
"   float4 coords = (float4)( xOffsets[offset], yOffsets[offset], zOffsets[offset], 0.0f );\n";


static const char *samplerKernelArg = " sampler_t imageSampler,";


int test_read_image_set_3D(cl_device_id device, cl_context context,
                           cl_command_queue queue,
                           const cl_image_format *format,
                           image_sampler_data *imageSampler, bool floatCoords,
                           ExplicitType outputType)
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    RandomSeed seed( gRandomSeed );

    int error;

    clProgramWrapper program;
    clKernelWrapper kernel;
    const char *KernelSourcePattern = NULL;

    // Get operating parameters
    size_t maxWidth, maxHeight, maxDepth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0x0 };

    imageInfo.format = format;
    imageInfo.type = CL_MEM_OBJECT_IMAGE3D;

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof( maxDepth ), &maxDepth, NULL );
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

    // Construct the source
    if(gtestTypesToRun & kReadTests)
    {
        KernelSourcePattern = read3DKernelSourcePattern;
    }
    else
    {
        KernelSourcePattern = read_write3DKernelSourcePattern;
    }

    sprintf(programSrc, KernelSourcePattern,
            gTestMipmaps
                ? "#pragma OPENCL EXTENSION cl_khr_mipmap_image: enable"
                : "",
            samplerArg, get_explicit_type_name(outputType),
            gTestMipmaps ? ", float lod" : " ", samplerVar,
            gTestMipmaps ? offset3DLodKernelSource : offset3DKernelSource,
            floatCoords ? float3DUnnormalizedCoordKernelSource
                        : int3DCoordKernelSource,
            readFormat, gTestMipmaps ? ",lod" : " ");

    ptr = programSrc;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    test_error( error, "Unable to create testing kernel" );

    // Run tests
    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );

            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                for( imageInfo.depth = 2; imageInfo.depth < 9; imageInfo.depth++ )
                {
                    if (gTestMipmaps)
                        imageInfo.num_mip_levels = (cl_uint) (2+rand()%(compute_max_mip_levels(imageInfo.width,imageInfo.height,imageInfo.depth) - 1));

                    if( gDebugTrace )
                        log_info( "   at size %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth );
                    int retCode = test_read_image(
                        context, queue, kernel, &imageInfo, imageSampler,
                        floatCoords, outputType, seed);
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

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, maxDepth, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE3D, imageInfo.format, CL_TRUE);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.depth = sizes[ idx ][ 2 ];
            imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
            imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
            if (gTestMipmaps)
                imageInfo.num_mip_levels = (cl_uint) (2+rand()%(compute_max_mip_levels(imageInfo.width,imageInfo.height,imageInfo.depth) - 1));
            log_info("Testing %d x %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ]);
            if( gDebugTrace )
                log_info( "   at max size %d,%d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ], (int)sizes[ idx ][ 2 ] );
            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
            if( retCode )
                return retCode;
        }
    }
    else if( gTestRounding )
    {
        size_t typeRange = 1 << ( get_format_type_size( imageInfo.format ) * 8 );
        imageInfo.height = typeRange / 256;
        imageInfo.width = (size_t)( typeRange / (cl_ulong)imageInfo.height );
        imageInfo.depth = 2;

        imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
        imageInfo.slicePitch = imageInfo.height * imageInfo.rowPitch;
        int retCode =
            test_read_image(context, queue, kernel, &imageInfo, imageSampler,
                            floatCoords, outputType, seed);
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
                imageInfo.width = reduceImageSizeRange(maxWidth, seed );
                imageInfo.height = reduceImageSizeRange(maxHeight, seed );
                imageInfo.depth = reduceImageDepth(maxDepth, seed );

                if (gTestMipmaps)
                {
                    //imageInfo.num_mip_levels = (cl_uint) random_log_in_range(2, (int)compute_max_mip_levels(imageInfo.width, imageInfo.depth, imageInfo.depth), seed);
                    imageInfo.num_mip_levels = (cl_uint) (2+rand()%(compute_max_mip_levels(imageInfo.width,imageInfo.height,imageInfo.depth) - 1));
                    //Need to take into account the output buffer size, otherwise we will end up with input buffer that is exceeding MaxAlloc
                    size = compute_mipmapped_image_size( imageInfo )*4 * get_explicit_type_size( outputType );
                    imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
                    imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;
                }
                else
                {
                    imageInfo.rowPitch = imageInfo.width * get_pixel_size( imageInfo.format );
                    imageInfo.slicePitch = imageInfo.rowPitch * imageInfo.height;

                    if( gEnablePitch )
                    {
                        size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.rowPitch += extraWidth * get_pixel_size( imageInfo.format );

                        size_t extraHeight = (int)random_log_in_range( 0, 64, seed );
                        imageInfo.slicePitch = imageInfo.rowPitch * (imageInfo.height + extraHeight);
                    }

                    size = (cl_ulong)imageInfo.slicePitch * (cl_ulong)imageInfo.depth * 4 * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
            {
                log_info( "   at size %d,%d,%d (pitch %d,%d) out of %d,%d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.depth, (int)imageInfo.rowPitch, (int)imageInfo.slicePitch, (int)maxWidth, (int)maxHeight, (int)maxDepth );
                if ( gTestMipmaps )
                    log_info( "   and number of mip levels :%d\n", (int)imageInfo.num_mip_levels );
            }
            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
            if( retCode )
                return retCode;
        }
    }

    return 0;
}
