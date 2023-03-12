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


struct image_kernel_data
{
    cl_int width;
    cl_int height;
    cl_int depth;
    cl_int widthDim;
    cl_int heightDim;
    cl_int depthDim;
    cl_int channelType;
    cl_int channelOrder;
    cl_int expectedChannelType;
    cl_int expectedChannelOrder;
};

static const char *methodTestKernelPattern =
    "typedef struct {\n"
    "    int width;\n"
    "    int height;\n"
    "    int depth;\n"
    "    int widthDim;\n"
    "    int heightDim;\n"
    "    int depthDim;\n"
    "    int channelType;\n"
    "    int channelOrder;\n"
    "    int expectedChannelType;\n"
    "    int expectedChannelOrder;\n"
    " } image_kernel_data;\n"
    " %s\n"
    "__kernel void sample_kernel( %s image%dd%s_t input, __global "
    "image_kernel_data *outData )\n"
    "{\n"
    "   outData->width = get_image_width( input );\n"
    "   outData->height = get_image_height( input );\n"
    "%s\n"
    "   int%d dim = get_image_dim( input );\n"
    "   outData->widthDim = dim.x;\n"
    "   outData->heightDim = dim.y;\n"
    "%s\n"
    "   outData->channelType = get_image_channel_data_type( input );\n"
    "   outData->channelOrder = get_image_channel_order( input );\n"
    "\n"
    "   outData->expectedChannelType = %s;\n"
    "   outData->expectedChannelOrder = %s;\n"
    "}";

static const char *depthKernelLine = "   outData->depth = get_image_depth( input );\n";
static const char *depthDimKernelLine = "   outData->depthDim = dim.z;\n";

int test_get_image_info_single(cl_context context, cl_command_queue queue,
                               image_descriptor *imageInfo, MTdata d,
                               cl_mem_flags flags)
{
    int error = 0;

    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper image, outDataBuffer;
    char programSrc[ 10240 ];

    image_kernel_data    outKernelData;

    // Generate some data to test against
    BufferOwningPtr<char> imageValues;
    generate_random_image_data( imageInfo, imageValues, d );

    // Construct testing source
    if( gDebugTrace )
        log_info( " - Creating image %d by %d...\n", (int)imageInfo->width, (int)imageInfo->height );

    if( imageInfo->depth != 0 )
        image = create_image_3d(context, flags, imageInfo->format,
                                imageInfo->width, imageInfo->height,
                                imageInfo->depth, 0, 0, NULL, &error);
    else
        image =
            create_image_2d(context, flags, imageInfo->format, imageInfo->width,
                            imageInfo->height, 0, NULL, &error);
    if( image == NULL )
    {
        log_error( "ERROR: Unable to create image of size %d x %d x %d (%s)", (int)imageInfo->width, (int)imageInfo->height, (int)imageInfo->depth, IGetErrorString( error ) );
        return -1;
    }

    char channelTypeConstantString[256] = {0};
    char channelOrderConstantString[256] = {0};

    const char* channelTypeName = GetChannelTypeName( imageInfo->format->image_channel_data_type );
    const char* channelOrderName = GetChannelOrderName( imageInfo->format->image_channel_order );
    const char *image_access_qualifier =
        (flags == CL_MEM_READ_ONLY) ? "read_only" : "write_only";
    const char *cl_khr_3d_image_writes_enabler = "";
    if ((flags != CL_MEM_READ_ONLY) && (imageInfo->depth != 0))
        cl_khr_3d_image_writes_enabler =
            "#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable";

    if(channelTypeName && strlen(channelTypeName))
        sprintf(channelTypeConstantString, "CLK_%s", &channelTypeName[3]);  // replace CL_* with CLK_*

    if(channelOrderName && strlen(channelOrderName))
        sprintf(channelOrderConstantString, "CLK_%s", &channelOrderName[3]); // replace CL_* with CLK_*

    // Create a program to run against
    sprintf(programSrc, methodTestKernelPattern, cl_khr_3d_image_writes_enabler,
            image_access_qualifier, (imageInfo->depth != 0) ? 3 : 2,
            (imageInfo->format->image_channel_order == CL_DEPTH) ? "_depth"
                                                                 : "",
            (imageInfo->depth != 0) ? depthKernelLine : "",
            (imageInfo->depth != 0) ? 4 : 2,
            (imageInfo->depth != 0) ? depthDimKernelLine : "",
            channelTypeConstantString, channelOrderConstantString);

    //log_info("-----------------------------------\n%s\n", programSrc);
    error = clFinish(queue);
    if (error)
        print_error(error, "clFinish failed.\n");
    const char *ptr = programSrc;
    error = create_single_kernel_helper(context, &program, &kernel, 1, &ptr,
                                        "sample_kernel");
    test_error( error, "Unable to create kernel to test against" );

    // Create an output buffer
    outDataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(outKernelData), NULL, &error);
    test_error( error, "Unable to create output buffer" );

    // Set up arguments and run
    error = clSetKernelArg( kernel, 0, sizeof( image ), &image );
    test_error( error, "Unable to set kernel argument" );
    error = clSetKernelArg( kernel, 1, sizeof( outDataBuffer ), &outDataBuffer );
    test_error( error, "Unable to set kernel argument" );

    size_t threads[1] = { 1 }, localThreads[1] = { 1 };

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to run kernel" );

    error = clEnqueueReadBuffer( queue, outDataBuffer, CL_TRUE, 0, sizeof( outKernelData ), &outKernelData, 0, NULL, NULL );
    test_error( error, "Unable to read data buffer" );


    // Verify the results now
    if( outKernelData.width != (cl_int)imageInfo->width )
    {
        log_error( "ERROR: Returned width did not validate (expected %d, got %d)\n", (int)imageInfo->width, (int)outKernelData.width );
        error = -1;
    }
    if( outKernelData.height != (cl_int)imageInfo->height )
    {
        log_error( "ERROR: Returned height did not validate (expected %d, got %d)\n", (int)imageInfo->height, (int)outKernelData.height );
        error = -1;
    }
    if( ( imageInfo->depth != 0 ) && ( outKernelData.depth != (cl_int)imageInfo->depth ) )
    {
        log_error( "ERROR: Returned depth did not validate (expected %d, got %d)\n", (int)imageInfo->depth, (int)outKernelData.depth );
        error = -1;
    }
    if( outKernelData.widthDim != (cl_int)imageInfo->width )
    {
        log_error( "ERROR: Returned width from get_image_dim did not validate (expected %d, got %d)\n", (int)imageInfo->width, (int)outKernelData.widthDim );
        error = -1;
    }
    if( outKernelData.heightDim != (cl_int)imageInfo->height )
    {
        log_error( "ERROR: Returned height from get_image_dim did not validate (expected %d, got %d)\n", (int)imageInfo->height, (int)outKernelData.heightDim );
        error = -1;
    }
    if( ( imageInfo->depth != 0 ) && ( outKernelData.depthDim != (cl_int)imageInfo->depth ) )
    {
        log_error( "ERROR: Returned depth from get_image_dim did not validate (expected %d, got %d)\n", (int)imageInfo->depth, (int)outKernelData.depthDim );
        error = -1;
    }
    if( outKernelData.channelType != (cl_int)outKernelData.expectedChannelType )
    {
        log_error( "ERROR: Returned channel type did not validate (expected %s (%d), got %d)\n", GetChannelTypeName( imageInfo->format->image_channel_data_type ),
                                                                                              (int)outKernelData.expectedChannelType, (int)outKernelData.channelType );
        error = -1;
    }
    if( outKernelData.channelOrder != (cl_int)outKernelData.expectedChannelOrder )
    {
        log_error( "ERROR: Returned channel order did not validate (expected %s (%d), got %d)\n", GetChannelOrderName( imageInfo->format->image_channel_order ),
                                                                                              (int)outKernelData.expectedChannelOrder, (int)outKernelData.channelOrder );
        error = -1;
    }

     if( clFinish(queue) != CL_SUCCESS )
     {
         log_error( "ERROR: CL Finished failed in %s \n", __FUNCTION__);
         error = -1;
     }

    return error;
}

int test_get_image_info_2D(cl_device_id device, cl_context context,
                           cl_command_queue queue, cl_image_format *format,
                           cl_mem_flags flags)
{
    size_t maxWidth, maxHeight;
    cl_ulong maxAllocSize, memSize;
    image_descriptor imageInfo = { 0 };
    RandomSeed seed( gRandomSeed );
    size_t pixelSize;

    imageInfo.type = CL_MEM_OBJECT_IMAGE2D;
    imageInfo.format = format;
    imageInfo.depth = imageInfo.slicePitch = 0;
    pixelSize = get_pixel_size( imageInfo.format );

    int error = clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxWidth ), &maxWidth, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxHeight ), &maxHeight, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get max image 2D size from device" );

  if (memSize > (cl_ulong)SIZE_MAX) {
    memSize = (cl_ulong)SIZE_MAX;
    maxAllocSize = (cl_ulong)SIZE_MAX;
  }

    if( gTestSmallImages )
    {
        for( imageInfo.width = 1; imageInfo.width < 13; imageInfo.width++ )
        {
            imageInfo.rowPitch = imageInfo.width * pixelSize;
            for( imageInfo.height = 1; imageInfo.height < 9; imageInfo.height++ )
            {
                if( gDebugTrace )
                    log_info( "   at size %d,%d\n", (int)imageInfo.width, (int)imageInfo.height );

                int ret = test_get_image_info_single(context, queue, &imageInfo,
                                                     seed, flags);
                if( ret )
                    return -1;
            }
        }
    }
    else if( gTestMaxImages )
    {
        // Try a specific set of maximum sizes
        size_t numbeOfSizes;
        size_t sizes[100][3];

        get_max_sizes(&numbeOfSizes, 100, sizes, maxWidth, maxHeight, 1, 1, maxAllocSize, memSize, CL_MEM_OBJECT_IMAGE2D, imageInfo.format);

        for( size_t idx = 0; idx < numbeOfSizes; idx++ )
        {
            imageInfo.width = sizes[ idx ][ 0 ];
            imageInfo.height = sizes[ idx ][ 1 ];
            imageInfo.rowPitch = imageInfo.width * pixelSize;

            log_info( "Testing %d x %d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ]);
            if( gDebugTrace )
                log_info( "   at max size %d,%d\n", (int)sizes[ idx ][ 0 ], (int)sizes[ idx ][ 1 ] );
            if (test_get_image_info_single(context, queue, &imageInfo, seed,
                                           flags))
                return -1;
        }
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
                imageInfo.height = (size_t)random_log_in_range( 16, (int)maxHeight / 32, seed );

                imageInfo.rowPitch = imageInfo.width * pixelSize;
                size_t extraWidth = (int)random_log_in_range( 0, 64, seed );
                imageInfo.rowPitch += extraWidth;

                do {
                    extraWidth++;
                    imageInfo.rowPitch += extraWidth;
                } while ((imageInfo.rowPitch % pixelSize) != 0);

                size = (cl_ulong)imageInfo.rowPitch * (cl_ulong)imageInfo.height * 4;
            } while(  size > maxAllocSize || ( size * 3 ) > memSize );

            if( gDebugTrace )
                log_info( "   at size %d,%d (row pitch %d) out of %d,%d\n", (int)imageInfo.width, (int)imageInfo.height, (int)imageInfo.rowPitch, (int)maxWidth, (int)maxHeight );
            int ret = test_get_image_info_single(context, queue, &imageInfo,
                                                 seed, flags);
            if( ret )
                return -1;
        }
    }

    return 0;
}
