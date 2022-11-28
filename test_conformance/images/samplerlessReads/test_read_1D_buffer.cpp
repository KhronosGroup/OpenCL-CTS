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


const char *read1DBufferKernelSourcePattern =
"__kernel void sample_kernel( read_only image1d_buffer_t inputA, read_only image1d_t inputB, sampler_t sampler, __global int *results )\n"
"{\n"
"   int tidX = get_global_id(0);\n"
"   int offset = tidX;\n"
"   %s clr = read_image%s( inputA, tidX );\n"
"   int4 test = (clr != read_image%s( inputB, sampler, tidX ));\n"
"   if ( test.x || test.y || test.z || test.w )\n"
"      results[offset] = -1;\n"
"   else\n"
"      results[offset] = 0;\n"
"}";


int test_read_image_1D_buffer( cl_context context, cl_command_queue queue, cl_kernel kernel,
                        image_descriptor *imageInfo, image_sampler_data *imageSampler,
                        ExplicitType outputType, MTdata d )
{
    int error;
    size_t threads[2];
    cl_sampler actualSampler;

    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    if ( gDebugTrace )
        log_info( " - Creating 1D image from buffer %d ...\n", (int)imageInfo->width );

    // Construct testing sources
    cl_mem image[2];
    cl_image_desc image_desc;

    cl_mem imageBuffer = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageInfo->rowPitch, imageValues, &error);
    if ( error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create buffer of size %d bytes (%s)\n", (int)imageInfo->rowPitch, IGetErrorString( error ) );
        return error;
    }

    memset(&image_desc, 0x0, sizeof(cl_image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    image_desc.image_width = imageInfo->width;
    image_desc.mem_object = imageBuffer;
    image[0] = clCreateImage( context, CL_MEM_READ_ONLY, imageInfo->format,
        &image_desc, NULL, &error );
    if ( error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create IMAGE1D_BUFFER of size %d pitch %d (%s)\n", (int)imageInfo->width, (int)imageInfo->rowPitch, IGetErrorString( error ) );
        return error;
    }

    cl_mem ret = NULL;
    error = clGetMemObjectInfo(image[0], CL_MEM_ASSOCIATED_MEMOBJECT, sizeof(ret), &ret, NULL);
    if ( error != CL_SUCCESS )
    {
        log_error("ERROR: Unable to query CL_MEM_ASSOCIATED_MEMOBJECT (%s)\n",
                  IGetErrorString(error));
        return error;
    }

    if (ret != imageBuffer) {
      log_error("ERROR: clGetImageInfo for CL_IMAGE_BUFFER returned wrong value\n");
      return -1;
    }

    memset(&image_desc, 0x0, sizeof(cl_image_desc));
    image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
    image_desc.image_width = imageInfo->width;
    image[1] = clCreateImage( context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, imageInfo->format, &image_desc, imageValues, &error );
    if ( error != CL_SUCCESS )
    {
        log_error( "ERROR: Unable to create IMAGE1D of size %d pitch %d (%s)\n", (int)imageInfo->width, (int)imageInfo->rowPitch, IGetErrorString( error ) );
        return error;
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
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &image[0] );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, idx++, sizeof( cl_mem ), &image[1] );
    test_error( error, "Unable to set kernel arguments" );
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
    clReleaseMemObject(image[0]);
    clReleaseMemObject(image[1]);
    clReleaseMemObject(imageBuffer);
    return 0;
}

int test_read_image_set_1D_buffer(cl_device_id device, cl_context context,
                                  cl_command_queue queue,
                                  const cl_image_format *format,
                                  image_sampler_data *imageSampler,
                                  ExplicitType outputType)
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
    size_t maxWidth, maxWidth1D;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    size_t pixelSize;

    if (format->image_channel_order == CL_RGB || format->image_channel_order == CL_RGBx)
    {
        switch (format->image_channel_data_type)
        {
            case CL_UNORM_INT8:
            case CL_UNORM_INT16:
            case CL_SNORM_INT8:
            case CL_SNORM_INT16:
            case CL_HALF_FLOAT:
            case CL_FLOAT:
            case CL_SIGNED_INT8:
            case CL_SIGNED_INT16:
            case CL_SIGNED_INT32:
            case CL_UNSIGNED_INT8:
            case CL_UNSIGNED_INT16:
            case CL_UNSIGNED_INT32:
            case CL_UNORM_INT_101010:
                log_info( "Skipping image format: %s %s\n", GetChannelOrderName( format->image_channel_order ),
                         GetChannelTypeName( format->image_channel_data_type ));
                return 0;
            default:
                break;
        }
    }

    imageInfo.format = format;
    imageInfo.height = imageInfo.depth = imageInfo.arraySize = imageInfo.slicePitch = 0;
    imageInfo.type = CL_MEM_OBJECT_IMAGE1D;
    pixelSize = get_pixel_size( imageInfo.format );

    error = clGetDeviceInfo( device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth1D, NULL );
    test_error( error, "Unable to get max image 1D buffer size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
      maxAllocSize = (cl_ulong)SIZE_MAX;
    }

    // note: image_buffer test uses image1D for results validation.
    // So the test can't use the biggest possible size for image_buffer if it's bigger than the max image1D size
    maxWidth = (maxWidth > maxWidth1D) ? maxWidth1D : maxWidth;
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

    sprintf( programSrc, read1DBufferKernelSourcePattern, dataType,
             readFormat,
             readFormat );

    ptr = programSrc;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    test_error( error, "Unable to create testing kernel" );

    if ( gTestSmallImages )
    {
        for ( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            {
                if ( gDebugTrace )
                    log_info( "   at size %d\n", (int)imageInfo.width );

                int retCode = test_read_image_1D_buffer( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
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
            int retCode = test_read_image_1D_buffer( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
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
                size = (size_t)imageInfo.rowPitch * 4;
            } while (  size > maxAllocSize || ( size * 3 ) > memSize );

            if ( gDebugTrace )
                log_info( "   at size %d (row pitch %d) out of %d\n", (int)imageInfo.width, (int)imageInfo.rowPitch, (int)maxWidth );
            int retCode = test_read_image_1D_buffer( context, queue, kernel, &imageInfo, imageSampler, outputType, seed );
            if ( retCode )
                return retCode;
        }
    }

    return 0;
}


