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

#include <algorithm>
#include <cinttypes>

#if defined( __APPLE__ )
    #include <signal.h>
    #include <sys/signal.h>
    #include <setjmp.h>
#endif

const char *read1DKernelSourcePattern =
    "%s\n"
    "__kernel void sample_kernel( read_only image1d_t input,%s __global float "
    "*xOffsets, __global %s4 *results %s)\n"
    "{\n"
    "%s"
    "   int tidX = get_global_id(0);\n"
    "   int offset = tidX;\n"
    "%s"
    "   results[offset] = read_image%s( input, imageSampler, coord %s);\n"
    "}";

const char *read_write1DKernelSourcePattern =
    "%s\n"
    "__kernel void sample_kernel( read_write image1d_t input,%s __global float "
    "*xOffsets, __global %s4 *results %s)\n"
    "{\n"
    "%s"
    "   int tidX = get_global_id(0);\n"
    "   int offset = tidX;\n"
    "%s"
    "   results[offset] = read_image%s( input, coord %s);\n"
    "}";

const char *int1DCoordKernelSource =
"   int coord = xOffsets[offset];\n";

const char *float1DKernelSource =
"   float coord = (float)xOffsets[offset];\n";

static const char *samplerKernelArg = " sampler_t imageSampler,";

int test_read_image_set_1D(cl_device_id device, cl_context context,
                           cl_command_queue queue,
                           const cl_image_format *format,
                           image_sampler_data *imageSampler, bool floatCoords,
                           ExplicitType outputType)
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    clProgramWrapper program;
    clKernelWrapper kernel;
    RandomSeed seed( gRandomSeed );
    int error;

    // Get our operating params
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0x0 };
    size_t pixelSize;

    const char *KernelSourcePattern = NULL;

    imageInfo.format = format;
    imageInfo.height = 1;
    imageInfo.depth = imageInfo.arraySize = imageInfo.slicePitch = 0;
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

    if(gtestTypesToRun & kReadWriteTests)
    {
        KernelSourcePattern = read_write1DKernelSourcePattern;
    }
    else
    {
        KernelSourcePattern = read1DKernelSourcePattern;
    }
    sprintf(programSrc, KernelSourcePattern,
            gTestMipmaps
                ? "#pragma OPENCL EXTENSION cl_khr_mipmap_image: enable"
                : "",
            samplerArg, get_explicit_type_name(outputType),
            gTestMipmaps ? ", float lod" : "", samplerVar,
            floatCoords ? float1DKernelSource : int1DCoordKernelSource,
            readFormat, gTestMipmaps ? ", lod" : "");

    ptr = programSrc;

    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    if(error)
    {
        exit(1);
    }
    test_error( error, "Unable to create testing kernel" );


    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            if(gTestMipmaps)
                imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), seed);

            if( gDebugTrace )
                log_info( "   at size %d\n", (int)imageInfo.width );

            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
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
            log_info("Testing %d\n", (int)sizes[ idx ][ 0 ]);
            if(gTestMipmaps)
                imageInfo.num_mip_levels = (size_t)random_in_range(2, (compute_max_mip_levels(imageInfo.width, 0, 0)-1), seed);
            if( gDebugTrace )
                log_info( "   at max size %d\n", (int)sizes[ idx ][ 0 ] );
            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
            if( retCode )
                return retCode;
        }
    }
    else if( gTestRounding )
    {
        uint64_t typeRange = 1LL << ( get_format_type_size( imageInfo.format ) * 8 );
        typeRange /= get_pixel_size( imageInfo.format ) / get_format_type_size( imageInfo.format );
        imageInfo.width = (size_t)( ( typeRange + 255LL ) / 256LL );

        while( imageInfo.width >= maxWidth / 2 )
            imageInfo.width >>= 1;
        imageInfo.rowPitch = imageInfo.width * pixelSize;

        gRoundingStartValue = 0;
        do
        {
            if( gDebugTrace )
                log_info("   at size %d, starting round ramp at %" PRIu64
                         " for range %" PRIu64 "\n",
                         (int)imageInfo.width, gRoundingStartValue, typeRange);
            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
            if( retCode )
                return retCode;

            gRoundingStartValue += imageInfo.width * pixelSize / get_format_type_size( imageInfo.format );

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

                    size = (size_t)imageInfo.rowPitch * 4;
                }
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d (row pitch %d) out of %d\n", (int)imageInfo.width, (int)imageInfo.rowPitch, (int)maxWidth );
            int retCode =
                test_read_image(context, queue, kernel, &imageInfo,
                                imageSampler, floatCoords, outputType, seed);
            if( retCode )
                return retCode;
        }
    }

    return 0;
}
