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
#include <float.h>

#if defined( __APPLE__ )
    #include <signal.h>
    #include <sys/signal.h>
    #include <setjmp.h>
#endif

#define MAX_ERR 0.005f
#define MAX_HALF_LINEAR_ERR 0.3f

extern bool                 gDebugTrace, gTestSmallImages, gEnablePitch, gTestMaxImages, gDeviceLt20;
extern bool                 gTestReadWrite;

#define MAX_TRIES   1
#define MAX_CLAMPED 1

const char *read1DKernelSourcePattern =
"__kernel void sample_kernel( read_only image1d_t input, sampler_t sampler, __global int *results )\n"
"{\n"
"   int tidX = get_global_id(0);\n"
"   int offset = tidX;\n"
"   %s clr = read_image%s( input, tidX );\n"
"   int4 test = (clr != read_image%s( input, sampler, tidX ));\n"
"   if ( test.x || test.y || test.z || test.w )\n"
"      results[offset] = -1;\n"
"   else\n"
"      results[offset] = 0;\n"
"}";

const char *read_write1DKernelSourcePattern =
"__kernel void sample_kernel( read_only image1d_t read_only_image, read_write image1d_t read_write_image, sampler_t sampler, __global int *results )\n"
"{\n"
"   int tidX = get_global_id(0);\n"
"   int offset = tidX;\n"
"   %s clr = read_image%s( read_only_image, sampler, tidX );\n"
"   write_image%s(read_write_image, tidX, clr);\n"
"   atomic_work_item_fence(CLK_IMAGE_MEM_FENCE, memory_order_acq_rel, memory_scope_work_item);\n"
"   int4 test = (clr != read_image%s(read_write_image, tidX));\n"
"   if ( test.x || test.y || test.z || test.w )\n"
"      results[offset] = -1;\n"
"   else\n"
"      results[offset] = 0;\n"
"}";

int test_read_image_1D( cl_context context, cl_command_queue queue, cl_kernel kernel,
                        image_descriptor *imageInfo, image_sampler_data *imageSampler,
                        ExplicitType outputType, MTdata d )
{
    int error;
    size_t threads[2];
    cl_sampler actualSampler;

    // generate_random_image_data allocates with malloc, so we use a MallocDataBuffer here
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if ( gDebugTrace )
        log_info( " - Creating image %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height );

    // Construct testing sources
    cl_mem read_only_image, read_write_image;
    cl_image_desc image_desc;

    memset(&image_desc, 0x0, sizeof(cl_image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
    image_desc.image_width = imageInfo->width;
    image_desc.image_row_pitch = ( gEnablePitch ? imageInfo->rowPitch : 0 );
    image_desc.num_mip_levels = 0;

    read_only_image = clCreateImage(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    imageInfo->format,
                                    &image_desc,
                                    imageValues,
                                    &error );
    if ( error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create read_only 1D image of size %d pitch %d (%s)\n", (int)imageInfo->width, (int)imageInfo->rowPitch, IGetErrorString( error ) );
        return error;
    }

    if(gTestReadWrite)
    {
        read_write_image = clCreateImage(context,
                                        CL_MEM_READ_WRITE,
                                        imageInfo->format,
                                        &image_desc,
                                        NULL,
                                        &error );
        if ( error != CL_SUCCESS )
        {
            log_error( "ERROR: Unable to create read_write 1D image of size %d pitch %d (%s)\n",
                        (int)imageInfo->width,
                        (int)imageInfo->rowPitch,
                        IGetErrorString( error ) );
            return error;
        }
    }


    if ( gDebugTrace )
        log_info( " - Creating kernel arguments...\n" );

    // Create sampler to use
    actualSampler = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
    test_error( error, "Unable to create image sampler" );

    // Create results buffer
    cl_mem results = clCreateBuffer( context, 0, imageInfo->width * sizeof(cl_int), NULL, &error);
    test_error( error, "Unable to create results buffer" );

    size_t resultValuesSize = imageInfo->width * sizeof(cl_int);
    BufferOwningPtr<int> resultValues(malloc( resultValuesSize ));
    memset( resultValues, 0xff, resultValuesSize );
    clEnqueueWriteBuffer( queue, results, CL_TRUE, 0, resultValuesSize, resultValues, 0, NULL, NULL );

    // Set arguments
    int idx = 0;
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &read_only_image );
    test_error( error, "Unable to set kernel arguments" );
    if(gTestReadWrite)
    {
        error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &read_write_image );
        test_error( error, "Unable to set kernel arguments" );
    }
    error = clSetKernelArg( kernel, idx++, sizeof( cl_sampler ), &actualSampler );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &results );
    test_error( error, "Unable to set kernel arguments" );

    // Run the kernel
    threads[0] = (size_t)imageInfo->width;
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
    test_error( error, "Unable to run kernel" );

    if ( gDebugTrace )
        log_info( "    reading results, %ld kbytes\n", (unsigned long)( imageInfo->width * sizeof(cl_int) / 1024 ) );

    error = clEnqueueReadBuffer( queue, results, CL_TRUE, 0, resultValuesSize, resultValues, 0, NULL, NULL );
    test_error( error, "Unable to read results from kernel" );
    if ( gDebugTrace )
        log_info( "    results read\n" );

    // Check for non-zero comps
    bool allZeroes = true;
    for ( size_t ic = 0; ic < imageInfo->width; ++ic )
    {
        if ( resultValues[ic] ) {
            allZeroes = false;
            break;
        }
    }
    if ( !allZeroes )
    {
        log_error( " Sampler-less reads differ from reads with sampler.\n" );
        return -1;
    }

    clReleaseSampler(actualSampler);
    clReleaseMemObject(results);
    clReleaseMemObject(read_only_image);
    if(gTestReadWrite)
    {
        clReleaseMemObject(read_write_image);
    }

    return 0;
}

int test_read_image_set_1D( cl_device_id device, cl_context context, cl_command_queue queue, cl_image_format *format, image_sampler_data *imageSampler,
                            ExplicitType outputType )
{
    char programSrc[10240];
    const char *ptr;
    const char *readFormat;
    const char *dataType;
    clProgramWrapper program;
    clKernelWrapper kernel;
    RandomSeed seed( gRandomSeed );
    int error;

    // Get our operating params
    size_t maxWidth;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    size_t pixelSize;

    imageInfo.format = format;
    imageInfo.height = imageInfo.depth = imageInfo.arraySize = imageInfo.slicePitch = 0;
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
    pixelSize = get_pixel_size( imageInfo.format );

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 1D size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
    }

    // Determine types
    if ( outputType == kInt )
    {
        readFormat = "i";
        dataType = "int4";
    }
    else if ( outputType == kUInt )
    {
        readFormat = "ui";
        dataType = "uint4";
    }
    else // kFloat
    {
        readFormat = "f";
        dataType = "float4";
    }

    if(gTestReadWrite)
    {
        sprintf( programSrc,
                 read_write1DKernelSourcePattern,
                 dataType,
                 readFormat,
                 readFormat,
                 readFormat);
    }
    else
    {
        sprintf( programSrc,
                 read1DKernelSourcePattern,
                 dataType,
                 readFormat,
                 readFormat );
    }


    ptr = programSrc;
    error = create_single_kernel_helper_with_build_options( context, &program, &kernel, 1, &ptr, "sample_kernel", gDeviceLt20 ? "" : "-cl-std=CL2.0" );
    test_error( error, "Unable to create testing kernel" );

    if ( gTestSmallImages )
    {
        for ( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            {
                if ( gDebugTrace )
                    log_info( "   at size %d\n", (int)imageInfo.width );

                int retCode = test_read_image_1D( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
                if ( retCode )
                    return retCode;
            }
        }
    }
    else if ( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, 1, 1, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE1D, imageInfo.format);

        for ( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            log_info("Testing %d\n", (int)sizes[ idx ][ 0 ]);
            if ( gDebugTrace )
                log_info( "   at max size %d\n", (int)sizes[ idx ][ 0 ] );
            int retCode = test_read_image_1D( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
            if ( retCode )
                return retCode;
        }
    }
    else
    {
        for ( int i = 0; i < NUM_IMAGE_ITERATIONS; i++ )
        {
            cl_ulong size;
            // Loop until we get a size that a) will fit in the max alloc size and b) that an allocation of that
            // image, the result array, plus offset arrays, will fit in the global ram space
            do
            {
                imageInfo.width = (size_t)random_log_in_range( 16, (int)maxWidth / 32, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                if ( gEnablePitch )
                {
                    size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                    imageInfo.rowPitch += extraWidth * pixelSize;
                }

                size = (size_t)imageInfo.rowPitch * 4;
            } while (  size > maxAllocSize || ( size * 3 ) > memSize );

            if ( gDebugTrace )
                log_info( "   at size %d (row pitch %d) out of %d\n", (int)imageInfo.width, (int)imageInfo.rowPitch, (int)maxWidth );
            int retCode = test_read_image_1D( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
            if ( retCode )
                return retCode;
        }
    }

    return 0;
}
