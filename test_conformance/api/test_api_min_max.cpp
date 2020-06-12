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
#include "testBase.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"
#include <ctype.h>
#include <string.h>

const char *sample_single_param_kernel[] = {
    "__kernel void sample_test(__global int *src)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "}\n" };

const char *sample_single_param_write_kernel[] = {
    "__kernel void sample_test(__global int *src)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "     src[tid] = tid;\n"
    "\n"
    "}\n" };

const char *sample_read_image_kernel_pattern[] = {
    "__kernel void sample_test( __global float *result, ",  " )\n"
    "{\n"
    "  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    "    int  tid = get_global_id(0);\n"
    "    result[0] = 0.0f;\n",
    "\n"
    "}\n" };

const char *sample_write_image_kernel_pattern[] = {
    "__kernel void sample_test( ",  " )\n"
    "{\n"
    "    int  tid = get_global_id(0);\n",
    "\n"
    "}\n" };


const char *sample_large_parmam_kernel_pattern[] = {
    "__kernel void sample_test(%s, __global long *result)\n"
    "{\n"
    "result[0] = 0;\n"
    "%s"
    "\n"
    "}\n" };

const char *sample_large_int_parmam_kernel_pattern[] = {
    "__kernel void sample_test(%s, __global int *result)\n"
    "{\n"
    "result[0] = 0;\n"
    "%s"
    "\n"
    "}\n" };

const char *sample_sampler_kernel_pattern[] = {
    "__kernel void sample_test( read_only image2d_t src, __global int4 *dst", ", sampler_t sampler%d", ")\n"
    "{\n"
    "    int  tid = get_global_id(0);\n",
    "     dst[ 0 ] = read_imagei( src, sampler%d, (int2)( 0, 0 ) );\n",
    "\n"
    "}\n" };

const char *sample_const_arg_kernel[] = {
    "__kernel void sample_test(__constant int *src1, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    dst[tid] = src1[tid];\n"
    "\n"
    "}\n" };

const char *sample_local_arg_kernel[] = {
    "__kernel void sample_test(__local int *src1, __global int *global_src, __global int *dst)\n"
    "{\n"
    "    int  tid = get_global_id(0);\n"
    "\n"
    "    src1[tid] = global_src[tid];\n"
    "    barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "    dst[tid] = src1[tid];\n"
    "\n"
    "}\n" };

const char *sample_const_max_arg_kernel_pattern =
"__kernel void sample_test(__constant int *src1 %s, __global int *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = src1[tid];\n"
"%s"
"\n"
"}\n";

int test_min_max_thread_dimensions(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error, retVal;
    unsigned int maxThreadDim, threadDim, i;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[1];
    size_t *threads, *localThreads;
    cl_event event;
    cl_int event_status;


    /* Get the max thread dimensions */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof( maxThreadDim ), &maxThreadDim, NULL );
    test_error( error, "Unable to get max work item dimensions from device" );

    if( maxThreadDim < 3 )
    {
        log_error( "ERROR: Reported max work item dimensions is less than required! (%d)\n", maxThreadDim );
        return -1;
    }

    log_info("Reported max thread dimensions of %d.\n", maxThreadDim);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_single_param_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams[0] = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_int) * 100, NULL, &error );
    if( streams[0] == NULL )
    {
        log_error("ERROR: Creating test array failed!\n");
        return -1;
    }

    /* Set the arguments */
    error = clSetKernelArg( kernel, 0, sizeof( streams[0] ), &streams[0] );
    test_error( error, "Unable to set kernel arguments" );

    retVal = 0;

    /* Now try running the kernel with up to that many threads */
    for (threadDim=1; threadDim <= maxThreadDim; threadDim++)
    {
        threads = (size_t *)malloc( sizeof( size_t ) * maxThreadDim );
        localThreads = (size_t *)malloc( sizeof( size_t ) * maxThreadDim );
        for( i = 0; i < maxThreadDim; i++ )
        {
            threads[ i ] = 1;
            localThreads[i] = 1;
        }

        error = clEnqueueNDRangeKernel( queue, kernel, maxThreadDim, NULL, threads, localThreads, 0, NULL, &event );
        test_error( error, "Failed clEnqueueNDRangeKernel");

        // Verify that the event does not return an error from the execution
        error = clWaitForEvents(1, &event);
        test_error( error, "clWaitForEvent failed");
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
        test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
        clReleaseEvent(event);
        if (event_status < 0)
            test_error(error, "Kernel execution event returned error");

        /* All done */
        free( threads );
        free( localThreads );
    }

    return retVal;
}


int test_min_max_work_items_sizes(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t *deviceMaxWorkItemSize;
    unsigned int maxWorkItemDim;

    /* Get the max work item dimensions */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof( maxWorkItemDim ), &maxWorkItemDim, NULL );
    test_error( error, "Unable to get max work item dimensions from device" );

    log_info("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS returned %d\n", maxWorkItemDim);
    deviceMaxWorkItemSize = (size_t*)malloc(sizeof(size_t)*maxWorkItemDim);
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxWorkItemDim, deviceMaxWorkItemSize, NULL );
    test_error( error, "clDeviceInfo for CL_DEVICE_MAX_WORK_ITEM_SIZES failed" );

    unsigned int i;
    int errors = 0;
    for(i=0; i<maxWorkItemDim; i++) {
        if (deviceMaxWorkItemSize[i]<1) {
            log_error("MAX_WORK_ITEM_SIZE in dimension %d is invalid: %lu\n", i, deviceMaxWorkItemSize[i]);
            errors++;
        } else {
            log_info("Dimension %d has max work item size %lu\n", i, deviceMaxWorkItemSize[i]);
        }
    }

    free(deviceMaxWorkItemSize);

    if (errors)
        return -1;
    return 0;
}



int test_min_max_work_group_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t deviceMaxThreadSize;

    /* Get the max thread dimensions */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof( deviceMaxThreadSize ), &deviceMaxThreadSize, NULL );
    test_error( error, "Unable to get max work group size from device" );

    log_info("Reported %ld max device work group size.\n", deviceMaxThreadSize);

    if( deviceMaxThreadSize == 0 )
    {
        log_error( "ERROR: Max work group size is reported as zero!\n" );
        return -1;
    }
    return 0;
}

int test_min_max_read_image_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    unsigned int maxReadImages, i;
    unsigned int deviceAddressSize;
    clProgramWrapper program;
    char readArgLine[128], *programSrc;
    const char *readArgPattern = ", read_only image2d_t srcimg%d";
    clKernelWrapper kernel;
    clMemWrapper    *streams, result;
    size_t threads[2];
    cl_image_format    image_format_desc;
    size_t maxParameterSize;
    cl_event event;
    cl_int event_status;
    cl_float image_data[4*4];
    float image_result = 0.0f;
    float actual_image_result;
    cl_uint minRequiredReadImages = gIsEmbedded ? 8 : 128;
    cl_device_type deviceType;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )
    image_format_desc.image_channel_order = CL_RGBA;
    image_format_desc.image_channel_data_type = CL_FLOAT;

    /* Get the max read image arg count */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof( maxReadImages ), &maxReadImages, NULL );
    test_error( error, "Unable to get max read image arg count from device" );

    if( maxReadImages < minRequiredReadImages )
    {
        log_error( "ERROR: Reported max read image arg count is less than required! (%d)\n", maxReadImages );
        return -1;
    }

    log_info("Reported %d max read image args.\n", maxReadImages);

    error = clGetDeviceInfo( deviceID, CL_DEVICE_ADDRESS_BITS, sizeof( deviceAddressSize ), &deviceAddressSize, NULL );
    test_error( error, "Unable to query CL_DEVICE_ADDRESS_BITS for device" );
    deviceAddressSize /= 8; // convert from bits to bytes


    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( maxParameterSize ), &maxParameterSize, NULL );
    test_error( error, "Unable to get max parameter size from device" );

    if (!gIsEmbedded && maxReadImages >= 128 && maxParameterSize == 1024)
    {
        error = clGetDeviceInfo( deviceID, CL_DEVICE_TYPE, sizeof( deviceType ), &deviceType, NULL );
        test_error( error, "Unable to get device type from device" );

        if(deviceType != CL_DEVICE_TYPE_CUSTOM)
        {
            maxReadImages = 127;
        }
    }
    // Subtract the size of the result
    maxParameterSize -= deviceAddressSize;

    // Calculate the number we can use
    if (maxParameterSize/deviceAddressSize < maxReadImages) {
        log_info("WARNING: Max parameter size of %d bytes limits test to %d max image arguments.\n", (int)maxParameterSize, (int)(maxParameterSize/deviceAddressSize));
        maxReadImages = (unsigned int)(maxParameterSize/deviceAddressSize);
    }

    /* Create a program with that many read args */
    programSrc = (char *)malloc( strlen( sample_read_image_kernel_pattern[ 0 ] ) + ( strlen( readArgPattern ) + 6 ) * ( maxReadImages ) +
                                strlen( sample_read_image_kernel_pattern[ 1 ] ) + 1 + 40240);

    strcpy( programSrc, sample_read_image_kernel_pattern[ 0 ] );
    strcat( programSrc, "read_only image2d_t srcimg0" );
    for( i = 0; i < maxReadImages-1; i++ )
    {
        sprintf( readArgLine, readArgPattern, i+1 );
        strcat( programSrc, readArgLine );
    }
    strcat( programSrc, sample_read_image_kernel_pattern[ 1 ] );
    for ( i = 0; i < maxReadImages; i++) {
        sprintf( readArgLine, "\tresult[0] += read_imagef( srcimg%d, sampler, (int2)(0,0)).x;\n", i);
        strcat( programSrc, readArgLine );
    }
    strcat( programSrc, sample_read_image_kernel_pattern[ 2 ] );

    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&programSrc, "sample_test");
    test_error( error, "Failed to create the program and kernel.");
    free( programSrc );

    result = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_float), NULL, &error);
    test_error( error, "clCreateBufer failed");

    /* Create some I/O streams */
    streams = new clMemWrapper[maxReadImages + 1];
    for( i = 0; i < maxReadImages; i++ )
    {
        image_data[0]=i;
        image_result+= image_data[0];
        streams[i] = create_image_2d( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &image_format_desc, 4, 4, 0, image_data, &error );
        test_error( error, "Unable to allocate test image" );
    }

    error = clSetKernelArg( kernel, 0, sizeof( result ), &result );
    test_error( error, "Unable to set kernel arguments" );

    /* Set the arguments */
    for( i = 1; i < maxReadImages+1; i++ )
    {
        error = clSetKernelArg( kernel, i, sizeof( streams[i-1] ), &streams[i-1] );
        test_error( error, "Unable to set kernel arguments" );
    }

    /* Now try running the kernel */
    threads[0] = threads[1] = 1;
    error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, &event );
    test_error( error, "clEnqueueNDRangeKernel failed");

    // Verify that the event does not return an error from the execution
    error = clWaitForEvents(1, &event);
    test_error( error, "clWaitForEvent failed");
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
    test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
    clReleaseEvent(event);
    if (event_status < 0)
        test_error(error, "Kernel execution event returned error");

    error = clEnqueueReadBuffer(queue, result, CL_TRUE, 0, sizeof(cl_float), &actual_image_result, 0, NULL, NULL);
    test_error(error, "clEnqueueReadBuffer failed");

    delete[] streams;

    if (actual_image_result != image_result) {
        log_error("Result failed to verify. Got %g, expected %g.\n", actual_image_result, image_result);
        return 1;
    }

    return 0;
}

int test_min_max_write_image_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    unsigned int maxWriteImages, i;
    clProgramWrapper program;
    char writeArgLine[128], *programSrc;
    const char *writeArgPattern = ", write_only image2d_t dstimg%d";
    clKernelWrapper kernel;
    clMemWrapper    *streams;
    size_t threads[2];
    cl_image_format    image_format_desc;
    size_t maxParameterSize;
    cl_event event;
    cl_int event_status;
    cl_uint minRequiredWriteImages = gIsEmbedded ? 1 : 8;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )
    image_format_desc.image_channel_order = CL_RGBA;
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;

    /* Get the max read image arg count */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof( maxWriteImages ), &maxWriteImages, NULL );
    test_error( error, "Unable to get max write image arg count from device" );

    if( maxWriteImages == 0 )
    {
        log_info( "WARNING: Device reports 0 for a max write image arg count (write image arguments unsupported). Skipping test (implicitly passes). This is only valid if the number of image formats is also 0.\n" );
        return 0;
    }

    if( maxWriteImages < minRequiredWriteImages )
    {
        log_error( "ERROR: Reported max write image arg count is less than required! (%d)\n", maxWriteImages );
        return -1;
    }

    log_info("Reported %d max write image args.\n", maxWriteImages);

    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( maxParameterSize ), &maxParameterSize, NULL );
    test_error( error, "Unable to get max parameter size from device" );

    // Calculate the number we can use
    if (maxParameterSize/sizeof(cl_mem) < maxWriteImages) {
        log_info("WARNING: Max parameter size of %d bytes limits test to %d max image arguments.\n", (int)maxParameterSize, (int)(maxParameterSize/sizeof(cl_mem)));
        maxWriteImages = (unsigned int)(maxParameterSize/sizeof(cl_mem));
    }

    /* Create a program with that many write args + 1 */
    programSrc = (char *)malloc( strlen( sample_write_image_kernel_pattern[ 0 ] ) + ( strlen( writeArgPattern ) + 6 ) * ( maxWriteImages + 1 ) +
                                strlen( sample_write_image_kernel_pattern[ 1 ] ) + 1 + 40240 );

    strcpy( programSrc, sample_write_image_kernel_pattern[ 0 ] );
    strcat( programSrc, "write_only image2d_t dstimg0" );
    for( i = 1; i < maxWriteImages; i++ )
    {
        sprintf( writeArgLine, writeArgPattern, i );
        strcat( programSrc, writeArgLine );
    }
    strcat( programSrc, sample_write_image_kernel_pattern[ 1 ] );
    for ( i = 0; i < maxWriteImages; i++) {
        sprintf( writeArgLine, "\twrite_imagef( dstimg%d, (int2)(0,0), (float4)(0,0,0,0));\n", i);
        strcat( programSrc, writeArgLine );
    }
    strcat( programSrc, sample_write_image_kernel_pattern[ 2 ] );

    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&programSrc, "sample_test");
    test_error( error, "Failed to create the program and kernel.");
    free( programSrc );


    /* Create some I/O streams */
    streams = new clMemWrapper[maxWriteImages + 1];
    for( i = 0; i < maxWriteImages; i++ )
    {
        streams[i] = create_image_2d( context, CL_MEM_READ_WRITE, &image_format_desc, 16, 16, 0, NULL, &error );
        test_error( error, "Unable to allocate test image" );
    }

    /* Set the arguments */
    for( i = 0; i < maxWriteImages; i++ )
    {
        error = clSetKernelArg( kernel, i, sizeof( streams[i] ), &streams[i] );
        test_error( error, "Unable to set kernel arguments" );
    }

    /* Now try running the kernel */
    threads[0] = threads[1] = 16;
    error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, &event );
    test_error( error, "clEnqueueNDRangeKernel failed.");

    // Verify that the event does not return an error from the execution
    error = clWaitForEvents(1, &event);
    test_error( error, "clWaitForEvent failed");
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
    test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
    clReleaseEvent(event);
    if (event_status < 0)
        test_error(error, "Kernel execution event returned error");

    /* All done */
    delete[] streams;
    return 0;
}

int test_min_max_mem_alloc_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_ulong maxAllocSize, memSize, minSizeToTry;
    clMemWrapper memHdl;

    cl_ulong requiredAllocSize;

    if (gIsEmbedded)
        requiredAllocSize = 1 * 1024 * 1024;
    else
        requiredAllocSize = 128 * 1024 * 1024;

    /* Get the max mem alloc size */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get max mem alloc size from device" );

    error = clGetDeviceInfo( deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof( memSize ), &memSize, NULL );
    test_error( error, "Unable to get global memory size from device" );

    if (memSize > (cl_ulong)SIZE_MAX) {
      memSize = (cl_ulong)SIZE_MAX;
    }

    if( maxAllocSize < requiredAllocSize)
    {
        log_error( "ERROR: Reported max allocation size is less than required %lldMB! (%llu or %lluMB, from a total mem size of %lldMB)\n", (requiredAllocSize / 1024) / 1024, maxAllocSize, (maxAllocSize / 1024)/1024, (memSize / 1024)/1024 );
        return -1;
    }

    requiredAllocSize = ((memSize / 4) > (1024 * 1024 * 1024)) ? 1024 * 1024 * 1024 : memSize / 4;

    if (gIsEmbedded)
        requiredAllocSize = (requiredAllocSize < 1 * 1024 * 1024) ? 1 * 1024 * 1024 : requiredAllocSize;
    else
    requiredAllocSize = (requiredAllocSize < 128 * 1024 * 1024) ? 128 * 1024 * 1024 : requiredAllocSize;

    if( maxAllocSize < requiredAllocSize )
    {
        log_error( "ERROR: Reported max allocation size is less than required of total memory! (%llu or %lluMB, from a total mem size of %lluMB)\n", maxAllocSize, (maxAllocSize / 1024)/1024, (requiredAllocSize / 1024)/1024 );
        return -1;
    }

    log_info("Reported max allocation size of %lld bytes (%gMB) and global mem size of %lld bytes (%gMB).\n",
             maxAllocSize, maxAllocSize/(1024.0*1024.0), requiredAllocSize, requiredAllocSize/(1024.0*1024.0));

    if ( memSize < maxAllocSize ) {
        log_info("Global memory size is less than max allocation size, using that.\n");
        maxAllocSize = memSize;
    }

    minSizeToTry = maxAllocSize/16;
    while (maxAllocSize > (maxAllocSize/4)) {

        log_info("Trying to create a buffer of size of %lld bytes (%gMB).\n", maxAllocSize, (double)maxAllocSize/(1024.0*1024.0));
        memHdl = clCreateBuffer( context, CL_MEM_READ_ONLY, (size_t)maxAllocSize, NULL, &error );
        if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE || error == CL_OUT_OF_RESOURCES || error == CL_OUT_OF_HOST_MEMORY) {
            log_info("\tAllocation failed at size of %lld bytes (%gMB).\n", maxAllocSize, (double)maxAllocSize/(1024.0*1024.0));
            maxAllocSize -= minSizeToTry;
            continue;
        }
        test_error( error, "clCreateBuffer failed for maximum sized buffer.");
        return 0;
    }
    log_error("Failed to allocate even %lld bytes (%gMB).\n", maxAllocSize, (double)maxAllocSize/(1024.0*1024.0));
    return -1;
}

int test_min_max_image_2d_width(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format image_format_desc;
    cl_ulong maxAllocSize;
    cl_uint minRequiredDimension;
    size_t length;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    auto version = get_device_cl_version(deviceID);
    if (version == Version(1, 0))
    {
        minRequiredDimension = gIsEmbedded ? 2048 : 4096;
    }
    else
    {
        minRequiredDimension = gIsEmbedded ? 2048 : 8192;
    }


    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max 2d image width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image 2d width from device" );

    if( maxDimension < minRequiredDimension )
    {
        log_error( "ERROR: Reported max image 2d width is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported width is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*1*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*1*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size %d x 1 = %gMB.\n", (int)maxDimension, ((float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_2d( context, CL_MEM_READ_ONLY, &image_format_desc, maxDimension, 1, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Image 2D creation failed for maximum width" );
        return -1;
    }

    return 0;
}

int test_min_max_image_2d_height(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format image_format_desc;
    cl_ulong maxAllocSize;
    cl_uint minRequiredDimension;
    size_t length;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    auto version = get_device_cl_version(deviceID);
    if (version == Version(1, 0))
    {
        minRequiredDimension = gIsEmbedded ? 2048 : 4096;
    }
    else
    {
        minRequiredDimension = gIsEmbedded ? 2048 : 8192;
    }

    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max 2d image width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image 2d height from device" );

    if( maxDimension < minRequiredDimension )
    {
        log_error( "ERROR: Reported max image 2d height is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported height is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*1*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*1*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size 1 x %d = %gMB.\n", (int)maxDimension, ((float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_2d( context, CL_MEM_READ_ONLY, &image_format_desc, 1, maxDimension, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Image 2D creation failed for maximum height" );
        return -1;
    }

    return 0;
}

int test_min_max_image_3d_width(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format    image_format_desc;
    cl_ulong maxAllocSize;


    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( deviceID )

    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE3D, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max 2d image width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image 3d width from device" );

    if( maxDimension < 2048 )
    {
        log_error( "ERROR: Reported max image 3d width is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported width is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*2*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*2*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size %d x 1 x 2 = %gMB.\n", (int)maxDimension, (2*(float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_3d( context, CL_MEM_READ_ONLY, &image_format_desc, maxDimension, 1, 2, 0, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Image 3D creation failed for maximum width" );
        return -1;
    }

    return 0;
}

int test_min_max_image_3d_height(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format    image_format_desc;
    cl_ulong maxAllocSize;


    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( deviceID )

    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE3D, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max 2d image width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image 3d height from device" );

    if( maxDimension < 2048 )
    {
        log_error( "ERROR: Reported max image 3d height is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported height is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*2*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*2*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size 1 x %d x 2 = %gMB.\n", (int)maxDimension, (2*(float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_3d( context, CL_MEM_READ_ONLY, &image_format_desc, 1, maxDimension, 2, 0, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Image 3D creation failed for maximum height" );
        return -1;
    }

    return 0;
}


int test_min_max_image_3d_depth(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format    image_format_desc;
    cl_ulong maxAllocSize;


    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT( deviceID )

    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE3D, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max 2d image width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image 3d depth from device" );

    if( maxDimension < 2048 )
    {
        log_error( "ERROR: Reported max image 3d depth is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported depth is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE3D, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*1*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*1*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size 1 x 1 x %d = %gMB.\n", (int)maxDimension, ((float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_3d( context, CL_MEM_READ_ONLY, &image_format_desc, 1, 1, maxDimension, 0, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Image 3D creation failed for maximum depth" );
        return -1;
    }

    return 0;
}

int test_min_max_image_array_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimension;
    clMemWrapper streams[1];
    cl_image_format    image_format_desc;
    cl_ulong maxAllocSize;
    size_t minRequiredDimension = gIsEmbedded ? 256 : 2048;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID );

    /* Just get any ol format to test with */
    error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE2D_ARRAY, CL_MEM_READ_WRITE, 0, &image_format_desc );
    test_error( error, "Unable to obtain suitable image format to test with!" );

    /* Get the max image array width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, sizeof( maxDimension ), &maxDimension, NULL );
    test_error( error, "Unable to get max image array size from device" );

    if( maxDimension < minRequiredDimension )
    {
        log_error( "ERROR: Reported max image array size is less than required! (%d)\n", (int)maxDimension );
        return -1;
    }
    log_info("Max reported image array size is %ld.\n", maxDimension);

    /* Verify we can use the format */
    image_format_desc.image_channel_data_type = CL_UNORM_INT8;
    image_format_desc.image_channel_order = CL_RGBA;
    if (!is_image_format_supported( context, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE2D_ARRAY, &image_format_desc)) {
        log_error("CL_UNORM_INT8 CL_RGBA not supported. Can not test.");
        return -1;
    }

    /* Verify that we can actually allocate an image that large */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );
    if ( (cl_ulong)maxDimension*1*4 > maxAllocSize ) {
        log_error("Can not allocate a large enough image (min size: %lld bytes, max allowed: %lld bytes) to test.\n",
                  (cl_ulong)maxDimension*1*4, maxAllocSize);
        return -1;
    }

    log_info("Attempting to create an image of size 1 x 1 x %d = %gMB.\n", (int)maxDimension, ((float)maxDimension*4/1024.0/1024.0));

    /* Try to allocate a very big image */
    streams[0] = create_image_2d_array( context, CL_MEM_READ_ONLY, &image_format_desc, 1, 1, maxDimension, 0, 0, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "2D Image Array creation failed for maximum array size" );
        return -1;
    }

    return 0;
}

int test_min_max_image_buffer_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    size_t maxDimensionPixels;
    clMemWrapper streams[2];
    cl_image_format image_format_desc = {0};
    cl_ulong maxAllocSize;
    size_t minRequiredDimension = gIsEmbedded ? 2048 : 65536;
    unsigned int i = 0;
    size_t pixelBytes = 0;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID );

    /* Get the max memory allocation size */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ( maxAllocSize ), &maxAllocSize, NULL );
    test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE." );

    /* Get the max image array width */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof( maxDimensionPixels ), &maxDimensionPixels, NULL );
    test_error( error, "Unable to get max image buffer size from device" );

    if( maxDimensionPixels < minRequiredDimension )
    {
        log_error( "ERROR: Reported max image buffer size is less than required! (%d)\n", (int)maxDimensionPixels );
        return -1;
    }
    log_info("Max reported image buffer size is %ld pixels.\n", maxDimensionPixels);

    pixelBytes = maxAllocSize / maxDimensionPixels;
    if ( pixelBytes == 0 )
    {
        log_error( "Value of CL_DEVICE_IMAGE_MAX_BUFFER_SIZE is greater than CL_MAX_MEM_ALLOC_SIZE so there is no way to allocate image of maximum size!\n" );
        return -1;
    }

    error = -1;
    for ( i = pixelBytes; i > 0; --i )
    {
        error = get_8_bit_image_format( context, CL_MEM_OBJECT_IMAGE1D, CL_MEM_READ_ONLY, i, &image_format_desc );
        if ( error == CL_SUCCESS )
        {
            pixelBytes = i;
            break;
        }
    }
    test_error( error, "Device does not support format to be used to allocate image of CL_DEVICE_IMAGE_MAX_BUFFER_SIZE\n" );

    log_info("Attempting to create an 1D image with channel order %s from buffer of size %d = %gMB.\n",
        GetChannelOrderName( image_format_desc.image_channel_order ), (int)maxDimensionPixels, ((float)maxDimensionPixels*pixelBytes/1024.0/1024.0));

    /* Try to allocate a buffer */
    streams[0] = clCreateBuffer( context, CL_MEM_READ_ONLY, maxDimensionPixels*pixelBytes, NULL, &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "Buffer creation failed for maximum image buffer size" );
        return -1;
    }

    /* Try to allocate a 1D image array from buffer */
    streams[1] = create_image_1d( context, CL_MEM_READ_ONLY, &image_format_desc, maxDimensionPixels, 0, NULL, streams[0], &error );
    if( ( streams[0] == NULL ) || ( error != CL_SUCCESS ))
    {
        print_error( error, "1D Image from buffer creation failed for maximum image buffer size" );
        return -1;
    }

    return 0;
}



int test_min_max_parameter_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error, retVal, i;
    size_t maxSize;
    char *programSrc;
    char *ptr;
    size_t numberExpected;
    long numberOfIntParametersToTry;
    char *argumentLine, *codeLines;
    void *data;
    cl_long long_result, expectedResult;
    cl_int int_result;
    size_t decrement;
    cl_event event;
    cl_int event_status;


    /* Get the max param size */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( maxSize ), &maxSize, NULL );
    test_error( error, "Unable to get max parameter size from device" );


    if( ((!gIsEmbedded) && (maxSize < 1024)) || ((gIsEmbedded) && (maxSize < 256)) )
    {
        log_error( "ERROR: Reported max parameter size is less than required! (%d)\n", (int)maxSize );
        return -1;
    }

    /* The embedded profile does not require longs, so use ints */
    if(gIsEmbedded)
        numberOfIntParametersToTry = numberExpected = (maxSize-sizeof(cl_mem))/sizeof(cl_int);
    else
        numberOfIntParametersToTry = numberExpected = (maxSize-sizeof(cl_mem))/sizeof(cl_long);

    decrement = (size_t)(numberOfIntParametersToTry/8);
    if (decrement < 1)
        decrement = 1;
    log_info("Reported max parameter size of %d bytes.\n", (int)maxSize);

    while (numberOfIntParametersToTry > 0) {
        // These need to be inside to be deallocated automatically on each loop iteration.
        clProgramWrapper program;
        clMemWrapper mem;
        clKernelWrapper kernel;

        if(gIsEmbedded)
        {
            log_info("Trying a kernel with %ld int arguments (%ld bytes) and one cl_mem (%ld bytes) for %ld bytes total.\n",
                     numberOfIntParametersToTry, sizeof(cl_int)*numberOfIntParametersToTry, sizeof(cl_mem),
                     sizeof(cl_mem)+numberOfIntParametersToTry*sizeof(cl_int));
        }
        else
        {
            log_info("Trying a kernel with %ld long arguments (%ld bytes) and one cl_mem (%ld bytes) for %ld bytes total.\n",
                     numberOfIntParametersToTry, sizeof(cl_long)*numberOfIntParametersToTry, sizeof(cl_mem),
                     sizeof(cl_mem)+numberOfIntParametersToTry*sizeof(cl_long));
        }

        // Allocate memory for the program storage
        data = malloc(sizeof(cl_long)*numberOfIntParametersToTry);

        argumentLine = (char*)malloc(sizeof(char)*numberOfIntParametersToTry*32);
        codeLines = (char*)malloc(sizeof(char)*numberOfIntParametersToTry*32);
        programSrc = (char*)malloc(sizeof(char)*(numberOfIntParametersToTry*64+1024));
        argumentLine[0] = '\0';
        codeLines[0] = '\0';
        programSrc[0] = '\0';

        // Generate our results
        expectedResult = 0;
        for (i=0; i<(int)numberOfIntParametersToTry; i++)
            {
            if( gHasLong )
            {
                ((cl_long *)data)[i] = i;
                expectedResult += i;
            }
            else
            {
                ((cl_int *)data)[i] = i;
                expectedResult += i;
            }
        }

        // Build the program
        if( gHasLong)
            sprintf(argumentLine, "%s", "long arg0");
        else
            sprintf(argumentLine, "%s", "int arg0");

        sprintf(codeLines, "%s", "result[0] += arg0;");
        for (i=1; i<(int)numberOfIntParametersToTry; i++)
        {
            if( gHasLong)
                sprintf(argumentLine + strlen( argumentLine), ", long arg%d", i);
            else
                sprintf(argumentLine + strlen( argumentLine), ", int arg%d", i);

            sprintf(codeLines + strlen( codeLines), "\nresult[0] += arg%d;", i);
        }

        /* Create a kernel to test with */
        sprintf( programSrc, gHasLong ?  sample_large_parmam_kernel_pattern[0]:
                                        sample_large_int_parmam_kernel_pattern[0], argumentLine, codeLines);

        ptr = programSrc;
        if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&ptr, "sample_test" ) != 0 )
        {
            log_info("Create program failed, decrementing number of parameters to try.\n");
            numberOfIntParametersToTry -= decrement;
            continue;
        }

        /* Try to set a large argument to the kernel */
        retVal = 0;

        mem = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), sizeof(cl_long), NULL, &error);
        test_error(error, "clCreateBuffer failed");

        for (i=0; i<(int)numberOfIntParametersToTry; i++) {
            if(gHasLong)
                error = clSetKernelArg(kernel, i, sizeof(cl_long), &(((cl_long*)data)[i]));
            else
                error = clSetKernelArg(kernel, i, sizeof(cl_int), &(((cl_int*)data)[i]));

            if (error != CL_SUCCESS) {
                log_info( "clSetKernelArg failed (%s), decrementing number of parameters to try.\n", IGetErrorString(error));
                numberOfIntParametersToTry -= decrement;
                break;
            }
        }
        if (error != CL_SUCCESS)
            continue;


        error = clSetKernelArg(kernel, i, sizeof(cl_mem), &mem);
        if (error != CL_SUCCESS) {
            log_info( "clSetKernelArg failed (%s), decrementing number of parameters to try.\n", IGetErrorString(error));
            numberOfIntParametersToTry -= decrement;
            continue;
        }

        size_t globalDim[3]={1,1,1}, localDim[3]={1,1,1};
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalDim, localDim, 0, NULL, &event);
        if (error != CL_SUCCESS) {
            log_info( "clEnqueueNDRangeKernel failed (%s), decrementing number of parameters to try.\n", IGetErrorString(error));
            numberOfIntParametersToTry -= decrement;
            continue;
        }

        // Verify that the event does not return an error from the execution
        error = clWaitForEvents(1, &event);
        test_error( error, "clWaitForEvent failed");
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
        test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
        clReleaseEvent(event);
        if (event_status < 0)
            test_error(error, "Kernel execution event returned error");

        if(gHasLong)
            error = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(cl_long), &long_result, 0, NULL, NULL);
        else
            error = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, sizeof(cl_int), &int_result, 0, NULL, NULL);

        test_error(error, "clEnqueueReadBuffer failed")

        free(data);
        free(argumentLine);
        free(codeLines);
        free(programSrc);

        if(gHasLong)
        {
            if (long_result != expectedResult) {
                log_error("Expected result (%lld) does not equal actual result (%lld).\n", expectedResult, long_result);
                numberOfIntParametersToTry -= decrement;
                continue;
            } else {
                log_info("Results verified at %ld bytes of arguments.\n", sizeof(cl_mem)+numberOfIntParametersToTry*sizeof(cl_long));
                break;
            }
        }
        else
        {
            if (int_result != expectedResult) {
                log_error("Expected result (%lld) does not equal actual result (%d).\n", expectedResult, int_result);
                numberOfIntParametersToTry -= decrement;
                continue;
            } else {
                log_info("Results verified at %ld bytes of arguments.\n", sizeof(cl_mem)+numberOfIntParametersToTry*sizeof(cl_int));
                break;
            }
        }
    }

    if (numberOfIntParametersToTry == (long)numberExpected)
        return 0;
    return -1;
}

int test_min_max_samplers(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_uint maxSamplers, i;
    clProgramWrapper program;
    clKernelWrapper kernel;
    char *programSrc, samplerLine[1024];
    size_t maxParameterSize;
    cl_event event;
    cl_int event_status;
    cl_uint minRequiredSamplers = gIsEmbedded ? 8 : 16;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( deviceID )

    /* Get the max value */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_SAMPLERS, sizeof( maxSamplers ), &maxSamplers, NULL );
    test_error( error, "Unable to get max sampler count from device" );

    if( maxSamplers < minRequiredSamplers )
    {
        log_error( "ERROR: Reported max sampler count is less than required! (%d)\n", (int)maxSamplers );
        return -1;
    }

    log_info("Reported max %d samplers.\n", maxSamplers);

    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( maxParameterSize ), &maxParameterSize, NULL );
    test_error( error, "Unable to get max parameter size from device" );

    // Subtract the size of the result
    maxParameterSize -= 2*sizeof(cl_mem);

    // Calculate the number we can use
    if (maxParameterSize/sizeof(cl_sampler) < maxSamplers) {
        log_info("WARNING: Max parameter size of %d bytes limits test to %d max sampler arguments.\n", (int)maxParameterSize, (int)(maxParameterSize/sizeof(cl_sampler)));
        maxSamplers = (unsigned int)(maxParameterSize/sizeof(cl_sampler));
    }

    /* Create a kernel to test with */
    programSrc = (char *)malloc( ( strlen( sample_sampler_kernel_pattern[ 1 ] ) + 8 ) * ( maxSamplers ) +
                                strlen( sample_sampler_kernel_pattern[ 0 ] ) + strlen( sample_sampler_kernel_pattern[ 2 ] ) +
                                ( strlen( sample_sampler_kernel_pattern[ 3 ] ) + 8 ) * maxSamplers +
                                strlen( sample_sampler_kernel_pattern[ 4 ] ) );
    strcpy( programSrc, sample_sampler_kernel_pattern[ 0 ] );
    for( i = 0; i < maxSamplers; i++ )
    {
        sprintf( samplerLine, sample_sampler_kernel_pattern[ 1 ], i );
        strcat( programSrc, samplerLine );
    }
    strcat( programSrc, sample_sampler_kernel_pattern[ 2 ] );
    for( i = 0; i < maxSamplers; i++ )
    {
        sprintf( samplerLine, sample_sampler_kernel_pattern[ 3 ], i );
        strcat( programSrc, samplerLine );
    }
    strcat( programSrc, sample_sampler_kernel_pattern[ 4 ] );


    error = create_single_kernel_helper(context, &program, &kernel, 1, (const char **)&programSrc, "sample_test");
    test_error( error, "Failed to create the program and kernel.");

    // We have to set up some fake parameters so it'll work
    clSamplerWrapper *samplers = new clSamplerWrapper[maxSamplers];

    cl_image_format format = { CL_RGBA, CL_SIGNED_INT8 };

    clMemWrapper image = create_image_2d( context, CL_MEM_READ_WRITE, &format, 16, 16, 0, NULL, &error );
    test_error( error, "Unable to create a test image" );

    clMemWrapper stream = clCreateBuffer( context, (cl_mem_flags)(CL_MEM_READ_WRITE), 16, NULL, &error );
    test_error( error, "Unable to create test buffer" );

    error = clSetKernelArg( kernel, 0, sizeof( cl_mem ), &image );
    error |= clSetKernelArg( kernel, 1, sizeof( cl_mem ), &stream );
    test_error( error, "Unable to set kernel arguments" );
    for( i = 0; i < maxSamplers; i++ )
    {
        samplers[ i ] = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
        test_error( error, "Unable to create sampler" );

        error = clSetKernelArg( kernel, 2 + i, sizeof( cl_sampler ), &samplers[ i ] );
        test_error( error, "Unable to set sampler argument" );
    }

    size_t globalDim[3]={1,1,1}, localDim[3]={1,1,1};
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalDim, localDim, 0, NULL, &event);
    test_error(error, "clEnqueueNDRangeKernel failed with maximum number of samplers.");

    // Verify that the event does not return an error from the execution
    error = clWaitForEvents(1, &event);
    test_error( error, "clWaitForEvent failed");
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
    test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
    clReleaseEvent(event);
    if (event_status < 0)
        test_error(error, "Kernel execution event returned error");

    free( programSrc );
    delete[] samplers;
    return 0;
}

#define PASSING_FRACTION 4
int test_min_max_constant_buffer_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    size_t    threads[1], localThreads[1];
    cl_int *constantData, *resultData;
    cl_ulong maxSize, stepSize, currentSize, maxGlobalSize, maxAllocSize;
    int i;
    cl_event event;
    cl_int event_status;
    MTdata d;

    /* Verify our test buffer won't be bigger than allowed */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof( maxSize ), &maxSize, 0 );
    test_error( error, "Unable to get max constant buffer size" );

    if( ( 0 == gIsEmbedded && maxSize < 64L * 1024L ) || maxSize <  1L * 1024L )
    {
        log_error( "ERROR: Reported max constant buffer size less than required by OpenCL 1.0 (reported %d KB)\n", (int)( maxSize / 1024L ) );
        return -1;
    }

    log_info("Reported max constant buffer size of %lld bytes.\n", maxSize);

    // Limit test buffer size to 1/8 of CL_DEVICE_GLOBAL_MEM_SIZE
    error = clGetDeviceInfo(deviceID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(maxGlobalSize), &maxGlobalSize, 0);
    test_error(error, "Unable to get CL_DEVICE_GLOBAL_MEM_SIZE");

    if (maxSize > maxGlobalSize / 8)
        maxSize = maxGlobalSize / 8;

    error = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(maxAllocSize), &maxAllocSize, 0);
    test_error(error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE ");
    
    if (maxSize > maxAllocSize)
        maxSize = maxAllocSize;
    
    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_const_arg_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Try the returned max size and decrease it until we get one that works. */
    stepSize = maxSize/16;
    currentSize = maxSize;
    int allocPassed = 0;
    d = init_genrand( gRandomSeed );
    while (!allocPassed && currentSize >= maxSize/PASSING_FRACTION) {
        log_info("Attempting to allocate constant buffer of size %lld bytes\n", maxSize);

        /* Create some I/O streams */
        size_t sizeToAllocate = ((size_t)currentSize/sizeof( cl_int ))*sizeof(cl_int);
        size_t numberOfInts = sizeToAllocate/sizeof(cl_int);
        constantData = (cl_int *)malloc( sizeToAllocate);
        if (constantData == NULL)
        {
            log_error("Failed to allocate memory for constantData!\n");
            free_mtdata(d);
            return EXIT_FAILURE;
        }

        for(i=0; i<(int)(numberOfInts); i++)
            constantData[i] = (int)genrand_int32(d);

        clMemWrapper streams[3];
        streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeToAllocate, constantData, &error);
        test_error( error, "Creating test array failed" );
        streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeToAllocate, NULL, &error);
        test_error( error, "Creating test array failed" );


        /* Set the arguments */
        error = clSetKernelArg(kernel, 0, sizeof( streams[0] ), &streams[0]);
        test_error( error, "Unable to set indexed kernel arguments" );
        error = clSetKernelArg(kernel, 1, sizeof( streams[1] ), &streams[1]);
        test_error( error, "Unable to set indexed kernel arguments" );


        /* Test running the kernel and verifying it */
        threads[0] = numberOfInts;
        localThreads[0] = 1;
        log_info("Filling constant buffer with %d cl_ints (%d bytes).\n", (int)threads[0], (int)(threads[0]*sizeof(cl_int)));

        error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, &event );
        /* If we failed due to a resource issue, reduce the size and try again. */
        if ((error == CL_OUT_OF_RESOURCES) || (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) || (error == CL_OUT_OF_HOST_MEMORY)) {
            log_info("Kernel enqueue failed at size %lld, trying at a reduced size.\n", currentSize);
            currentSize -= stepSize;
            free(constantData);
            continue;
        }
        test_error( error, "clEnqueueNDRangeKernel with maximum constant buffer size failed.");

        // Verify that the event does not return an error from the execution
        error = clWaitForEvents(1, &event);
        test_error( error, "clWaitForEvent failed");
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
        test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
        clReleaseEvent(event);
        if (event_status < 0) {
            if ((event_status == CL_OUT_OF_RESOURCES) || (event_status == CL_MEM_OBJECT_ALLOCATION_FAILURE) || (event_status == CL_OUT_OF_HOST_MEMORY)) {
                log_info("Kernel event indicates failure at size %lld, trying at a reduced size.\n", currentSize);
                currentSize -= stepSize;
                free(constantData);
                continue;
            } else {
                test_error(error, "Kernel execution event returned error");
            }
        }

        /* Otherwise we did not fail due to resource issues. */
        allocPassed = 1;

        resultData = (cl_int *)malloc(sizeToAllocate);
        if (resultData == NULL)
        {
            log_error("Failed to allocate memory for resultData!\n");
            free(constantData);
            free_mtdata(d);
            return EXIT_FAILURE;
        }

        error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, sizeToAllocate, resultData, 0, NULL, NULL);
        test_error( error, "clEnqueueReadBuffer failed");

        for(i=0; i<(int)(numberOfInts); i++)
            if (constantData[i] != resultData[i]) {
                log_error("Data failed to verify: constantData[%d]=%d != resultData[%d]=%d\n",
                          i, constantData[i], i, resultData[i]);
                free( constantData );
                free(resultData);
                free_mtdata(d);   d = NULL;
                return -1;
            }

        free( constantData );
        free(resultData);
    }
    free_mtdata(d);   d = NULL;

    if (allocPassed) {
        if (currentSize < maxSize/PASSING_FRACTION) {
            log_error("Failed to allocate at least 1/8 of the reported constant size.\n");
            return -1;
        } else if (currentSize != maxSize) {
            log_info("Passed at reduced size. (%lld of %lld bytes)\n", currentSize, maxSize);
            return 0;
        }
        return 0;
    }
    return -1;
}

int test_min_max_constant_args(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper    *streams;
    size_t    threads[1], localThreads[1];
    cl_uint i, maxArgs;
    cl_ulong maxSize;
    cl_ulong maxParameterSize;
    size_t individualBufferSize;
    char *programSrc, *constArgs, *str2;
    char str[512];
    const char *ptr;
    cl_event event;
    cl_int event_status;


    /* Verify our test buffer won't be bigger than allowed */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof( maxArgs ), &maxArgs, 0 );
    test_error( error, "Unable to get max constant arg count" );

    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof( maxParameterSize ), &maxParameterSize, NULL );
    test_error( error, "Unable to get max parameter size from device" );

    // Subtract the size of the result
    maxParameterSize -= sizeof(cl_mem);

    // Calculate the number we can use
    if (maxParameterSize/sizeof(cl_mem) < maxArgs) {
        log_info("WARNING: Max parameter size of %d bytes limits test to %d max image arguments.\n", (int)maxParameterSize, (int)(maxParameterSize/sizeof(cl_mem)));
        maxArgs = (unsigned int)(maxParameterSize/sizeof(cl_mem));
    }


    if( maxArgs < (gIsEmbedded ? 4 : 8) )
    {
        log_error( "ERROR: Reported max constant arg count less than required by OpenCL 1.0 (reported %d)\n", (int)maxArgs );
        return -1;
    }

    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof( maxSize ), &maxSize, 0 );
    test_error( error, "Unable to get max constant buffer size" );
    individualBufferSize = ((int)maxSize/2)/maxArgs;

    log_info("Reported max constant arg count of %d and max constant buffer size of %d. Test will attempt to allocate half of that, or %d buffers of size %d.\n",
             (int)maxArgs, (int)maxSize, (int)maxArgs, (int)individualBufferSize);

    str2 = (char*)malloc(sizeof(char)*32*(maxArgs+2));
    constArgs = (char*)malloc(sizeof(char)*32*(maxArgs+2));
    programSrc = (char*)malloc(sizeof(char)*32*2*(maxArgs+2)+1024);

    /* Create a test program */
    constArgs[0] = 0;
    str2[0] = 0;
    for( i = 0; i < maxArgs-1; i++ )
    {
        sprintf( str, ", __constant int *src%d", (int)( i + 2 ) );
        strcat( constArgs, str );
        sprintf( str2 + strlen( str2), "\tdst[tid] += src%d[tid];\n", (int)(i+2));
        if (strlen(str2) > (sizeof(char)*32*(maxArgs+2)-32) || strlen(constArgs) > (sizeof(char)*32*(maxArgs+2)-32)) {
            log_info("Limiting number of arguments tested to %d due to test program allocation size.\n", i);
            break;
        }
    }
    sprintf( programSrc, sample_const_max_arg_kernel_pattern, constArgs, str2 );

    /* Create a kernel to test with */
    ptr = programSrc;
    if( create_single_kernel_helper( context, &program, &kernel, 1, &ptr, "sample_test" ) != 0 )
    {
        return -1;
    }

    /* Create some I/O streams */
    streams = new clMemWrapper[ maxArgs + 1 ];
    for( i = 0; i < maxArgs + 1; i++ )
    {
        streams[i] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE), individualBufferSize, NULL, &error);
        test_error( error, "Creating test array failed" );
    }

    /* Set the arguments */
    for( i = 0; i < maxArgs + 1; i++ )
    {
        error = clSetKernelArg(kernel, i, sizeof( streams[i] ), &streams[i]);
        test_error( error, "Unable to set kernel argument" );
    }

    /* Test running the kernel and verifying it */
    threads[0] = (size_t)10;
    while (threads[0]*sizeof(cl_int) > individualBufferSize)
        threads[0]--;

    error = get_max_common_work_group_size( context, kernel, threads[0], &localThreads[0] );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, &event );
    test_error( error, "clEnqueueNDRangeKernel failed");

    // Verify that the event does not return an error from the execution
    error = clWaitForEvents(1, &event);
    test_error( error, "clWaitForEvent failed");
    error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);
    test_error( error, "clGetEventInfo for CL_EVENT_COMMAND_EXECUTION_STATUS failed");
    clReleaseEvent(event);
    if (event_status < 0)
        test_error(error, "Kernel execution event returned error");

    error = clFinish(queue);
    test_error( error, "clFinish failed.");

    delete [] streams;
    free(str2);
    free(constArgs);
    free(programSrc);
    return 0;
}

int test_min_max_compute_units(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_uint value;


    error = clGetDeviceInfo( deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get compute unit count" );

    if( value < 1 )
    {
        log_error( "ERROR: Reported compute unit count less than required by OpenCL 1.0 (reported %d)\n", (int)value );
        return -1;
    }

    log_info("Reported %d max compute units.\n", value);

    return 0;
}

int test_min_max_address_bits(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_uint value;


    error = clGetDeviceInfo( deviceID, CL_DEVICE_ADDRESS_BITS, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get address bit count" );

    if( value != 32 && value != 64 )
    {
        log_error( "ERROR: Reported address bit count not valid by OpenCL 1.0 (reported %d)\n", (int)value );
        return -1;
    }

    log_info("Reported %d device address bits.\n", value);

    return 0;
}

int test_min_max_single_fp_config(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_device_fp_config value;
    char profile[128] = "";

    error = clGetDeviceInfo( deviceID, CL_DEVICE_SINGLE_FP_CONFIG, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get device single fp config" );

    //Check to see if we are an embedded profile device
    if((error = clGetDeviceInfo( deviceID, CL_DEVICE_PROFILE, sizeof(profile), profile, NULL )))
    {
        log_error( "FAILURE: Unable to get CL_DEVICE_PROFILE: error %d\n", error );
        return error;
    }

    if( 0 == strcmp( profile, "EMBEDDED_PROFILE" ))
    { // embedded device

        if( 0 == (value & (CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO)))
        {
            log_error( "FAILURE: embedded device supports neither CL_FP_ROUND_TO_NEAREST or CL_FP_ROUND_TO_ZERO\n" );
            return -1;
        }
    }
    else
    { // Full profile
        if( ( value & ( CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN )) != ( CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN ) )
        {
            log_error( "ERROR: Reported single fp config doesn't meet minimum set by OpenCL 1.0 (reported 0x%08x)\n", (int)value );
            return -1;
        }
    }
    return 0;
}

int test_min_max_double_fp_config(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_device_fp_config value;

    error = clGetDeviceInfo( deviceID, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get device double fp config" );

    if (value == 0)
        return 0;

    if( ( value & (CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM)) != ( CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM) )
    {
        log_error( "ERROR: Reported double fp config doesn't meet minimum set by OpenCL 1.0 (reported 0x%08x)\n", (int)value );
        return -1;
    }
    return 0;
}

int test_min_max_local_mem_size(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper            streams[3];
    size_t    threads[1], localThreads[1];
    cl_int *localData, *resultData;
    cl_ulong maxSize, kernelLocalUsage, min_max_local_mem_size;
    cl_char buffer[ 4098 ];
    size_t length;
    int i;
    int err = 0;
    MTdata d;

    /* Verify our test buffer won't be bigger than allowed */
    error = clGetDeviceInfo( deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof( maxSize ), &maxSize, 0 );
    test_error( error, "Unable to get max local buffer size" );

    // Device version should fit the regex "OpenCL [0-9]+\.[0-9]+ *.*"
    error = clGetDeviceInfo( deviceID, CL_DEVICE_VERSION, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get device version string" );
    if (!gIsEmbedded)
    {
        if( memcmp( buffer, "OpenCL 2.0", strlen( "OpenCL 2.0" ) ) == 0 )
            min_max_local_mem_size = 16L * 1024L;
        else if( memcmp( buffer, "OpenCL 2.1", strlen( "OpenCL 2.1" ) ) != 0 )
            min_max_local_mem_size = 16L * 1024L;
        else if( memcmp( buffer, "OpenCL 1.2", strlen( "OpenCL 1.2" ) ) != 0 )
            min_max_local_mem_size = 16L * 1024L;
        else if( memcmp( buffer, "OpenCL 1.1", strlen( "OpenCL 1.1" ) ) != 0 )
            min_max_local_mem_size = 16L * 1024L;
        else if ( memcmp( buffer, "OpenCL 1.0", strlen( "OpenCL 1.0" ) ) != 0 )
            min_max_local_mem_size = 32L * 1024L;
        else
        {
            log_error( "ERROR: device version string does not match required format! (returned: %s)\n", (char *)buffer );
            return -1;
        }
    }

    if( maxSize < (gIsEmbedded ? 1L * 1024L : min_max_local_mem_size) )
    {
        log_error( "ERROR: Reported local mem size less than required by OpenCL 1.1 (reported %dKb)\n", (int)( maxSize / 1024L ) );
        return -1;
    }

    log_info("Reported max local buffer size for device: %lld bytes.\n", maxSize);

    /* Create a kernel to test with */
    if( create_single_kernel_helper( context, &program, &kernel, 1, sample_local_arg_kernel, "sample_test" ) != 0 )
    {
        return -1;
    }

    error = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(kernelLocalUsage), &kernelLocalUsage, NULL);
    test_error(error, "clGetKernelWorkGroupInfo for CL_KERNEL_LOCAL_MEM_SIZE failed");

    log_info("Reported local buffer usage for kernel (CL_KERNEL_LOCAL_MEM_SIZE): %lld bytes.\n", kernelLocalUsage);

    /* Create some I/O streams */
    size_t sizeToAllocate = ((size_t)(maxSize-kernelLocalUsage)/sizeof( cl_int ))*sizeof(cl_int);
    size_t numberOfInts = sizeToAllocate/sizeof(cl_int);

    log_info("Attempting to use %lld bytes of local memory.\n", (cl_ulong)sizeToAllocate);

    localData = (cl_int *)malloc( sizeToAllocate );
    d = init_genrand( gRandomSeed );
    for(i=0; i<(int)(numberOfInts); i++)
        localData[i] = (int)genrand_int32(d);
    free_mtdata(d); d = NULL;

    streams[0] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_COPY_HOST_PTR), sizeToAllocate, localData, &error);
    test_error( error, "Creating test array failed" );
    streams[1] = clCreateBuffer(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  sizeToAllocate, NULL, &error);
    test_error( error, "Creating test array failed" );


    /* Set the arguments */
    error = clSetKernelArg(kernel, 0, sizeToAllocate, NULL);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 1, sizeof( streams[0] ), &streams[0]);
    test_error( error, "Unable to set indexed kernel arguments" );
    error = clSetKernelArg(kernel, 2, sizeof( streams[1] ), &streams[1]);
    test_error( error, "Unable to set indexed kernel arguments" );


    /* Test running the kernel and verifying it */
    threads[0] = numberOfInts;
    localThreads[0] = 1;
    log_info("Creating local buffer with %d cl_ints (%d bytes).\n", (int)numberOfInts, (int)sizeToAllocate);

    cl_event evt;
    cl_int   evt_err;
    error = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, localThreads, 0, NULL, &evt );
    test_error(error, "clEnqueueNDRangeKernel failed");

    error = clFinish(queue);
    test_error( error, "clFinish failed");

    error = clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof evt_err, &evt_err, NULL);
    test_error( error, "clGetEventInfo with maximum local buffer size failed.");

    if (evt_err != CL_COMPLETE) {
        print_error(evt_err, "Kernel event returned error");
        clReleaseEvent(evt);
        return -1;
    }

    resultData = (cl_int *)malloc(sizeToAllocate);

    error = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, sizeToAllocate, resultData, 0, NULL, NULL);
    test_error( error, "clEnqueueReadBuffer failed");

    for(i=0; i<(int)(numberOfInts); i++)
        if (localData[i] != resultData[i]) {
            clReleaseEvent(evt);
            free( localData );
            free(resultData);
            log_error("Results failed to verify.\n");
            return -1;
        }
    clReleaseEvent(evt);
    free( localData );
    free(resultData);

    return err;
}

int test_min_max_kernel_preferred_work_group_size_multiple(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int                err;
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t max_local_workgroup_size[3];
    size_t max_workgroup_size = 0, preferred_workgroup_size = 0;

    err = create_single_kernel_helper(context, &program, &kernel, 1, sample_local_arg_kernel, "sample_test" );
    test_error(err, "Failed to build kernel/program.");

    err = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(max_workgroup_size), &max_workgroup_size, NULL);
    test_error(err, "clGetKernelWorkgroupInfo failed.");

    err = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                   sizeof(preferred_workgroup_size), &preferred_workgroup_size, NULL);
    test_error(err, "clGetKernelWorkgroupInfo failed.");

    err = clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // Since the preferred size is only a performance hint, we can only really check that we get a sane value
    // back
    log_info( "size: %ld     preferred: %ld      max: %ld\n", max_workgroup_size, preferred_workgroup_size, max_local_workgroup_size[0] );

    if( preferred_workgroup_size > max_workgroup_size )
    {
        log_error( "ERROR: Reported preferred workgroup multiple larger than max workgroup size (preferred %ld, max %ld)\n", preferred_workgroup_size, max_workgroup_size );
        return -1;
    }

    return 0;
}

int test_min_max_execution_capabilities(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_device_exec_capabilities value;


    error = clGetDeviceInfo( deviceID, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get execution capabilities" );

    if( ( value & CL_EXEC_KERNEL ) != CL_EXEC_KERNEL )
    {
        log_error( "ERROR: Reported execution capabilities less than required by OpenCL 1.0 (reported 0x%08x)\n", (int)value );
        return -1;
    }
    return 0;
}

int test_min_max_queue_properties(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    int error;
    cl_command_queue_properties value;


    error = clGetDeviceInfo( deviceID, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES, sizeof( value ), &value, 0 );
    test_error( error, "Unable to get queue properties" );

    if( ( value & CL_QUEUE_PROFILING_ENABLE ) != CL_QUEUE_PROFILING_ENABLE )
    {
        log_error( "ERROR: Reported queue properties less than required by OpenCL 1.0 (reported 0x%08x)\n", (int)value );
        return -1;
    }
    return 0;
}

int test_min_max_device_version(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    // Query for the device version.
    Version device_cl_version = get_device_cl_version(deviceID);
    log_info("Returned version %s.\n", device_cl_version.to_string().c_str());

    // Make sure 2.x devices support required extensions for 2.x
    // note: these extensions are **not** required for devices
    // supporting OpenCL-3.0
    const char *requiredExtensions2x[] = {
        "cl_khr_3d_image_writes",
        "cl_khr_image2d_from_buffer",
        "cl_khr_depth_images",
    };

    // Make sure 1.1 devices support required extensions for 1.1
    const char *requiredExtensions11[] = {
        "cl_khr_global_int32_base_atomics",
        "cl_khr_global_int32_extended_atomics",
        "cl_khr_local_int32_base_atomics",
        "cl_khr_local_int32_extended_atomics",
        "cl_khr_byte_addressable_store",
    };


    if (device_cl_version >= Version(1, 1))
    {
        log_info("Checking for required extensions for OpenCL 1.1 and later "
                 "devices...\n");
        for (int i = 0; i < ARRAY_SIZE(requiredExtensions11); i++)
        {
            if (!is_extension_available(deviceID, requiredExtensions11[i]))
            {
                log_error("ERROR: Required extension for 1.1 and greater "
                          "devices is not in extension string: %s\n",
                          requiredExtensions11[i]);
                return -1;
            }
            else
                log_info("\t%s\n", requiredExtensions11[i]);
        }

        if (device_cl_version >= Version(1, 2))
        {
            log_info("Checking for required extensions for OpenCL 1.2 and "
                     "later devices...\n");
            // The only required extension for an OpenCL-1.2 device is
            // cl_khr_fp64 and it is only required if double precision is
            // supported.
            cl_device_fp_config doubles_supported;
            cl_int error = clGetDeviceInfo(deviceID, CL_DEVICE_DOUBLE_FP_CONFIG,
                                           sizeof(doubles_supported),
                                           &doubles_supported, 0);
            test_error(error, "Unable to get device double fp config");
            if (doubles_supported)
            {
                if (!is_extension_available(deviceID, "cl_khr_fp64"))
                {
                    log_error(
                        "ERROR: Required extension for 1.2 and greater devices "
                        "is not in extension string: cl_khr_fp64\n");
                }
                else
                {
                    log_info("\t%s\n", "cl_khr_fp64");
                }
            }
        }

        if (device_cl_version >= Version(2, 0)
            && device_cl_version < Version(3, 0))
        {
            log_info("Checking for required extensions for OpenCL 2.0, 2.1 and "
                     "2.2 devices...\n");
            for (int i = 0; i < ARRAY_SIZE(requiredExtensions2x); i++)
            {
                if (!is_extension_available(deviceID, requiredExtensions2x[i]))
                {
                    log_error("ERROR: Required extension for 2.0, 2.1 and 2.2 "
                              "devices is not in extension string: %s\n",
                              requiredExtensions2x[i]);
                    return -1;
                }
                else
                {
                    log_info("\t%s\n", requiredExtensions2x[i]);
                }
            }
        }
    }
    else
        log_info("WARNING: skipping required extension test -- OpenCL 1.0 "
                 "device.\n");
    return 0;
}

int test_min_max_language_version(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_int error;
    cl_char buffer[ 4098 ];
    size_t length;

    // Device version should fit the regex "OpenCL [0-9]+\.[0-9]+ *.*"
    error = clGetDeviceInfo( deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof( buffer ), buffer, &length );
    test_error( error, "Unable to get device opencl c version string" );
    if( memcmp( buffer, "OpenCL C ", strlen( "OpenCL C " ) ) != 0 )
    {
        log_error( "ERROR: Initial part of device language version string does not match required format! (returned: \"%s\")\n", (char *)buffer );
        return -1;
    }

    log_info("Returned version \"%s\".\n", buffer);

    char *p1 = (char *)buffer + strlen( "OpenCL C " );
    while( *p1 == ' ' )
        p1++;
    char *p2 = p1;
    if( ! isdigit(*p2) )
    {
        log_error( "ERROR: Major revision number must follow space behind OpenCL C! (returned %s)\n", (char*) buffer );
        return -1;
    }
    while( isdigit( *p2 ) )
        p2++;
    if( *p2 != '.' )
    {
        log_error( "ERROR: Version number must contain a decimal point! (returned: %s)\n", (char *)buffer );
        return -1;
    }
    char *p3 = p2 + 1;
    if( ! isdigit(*p3) )
    {
        log_error( "ERROR: Minor revision number is missing or does not abut the decimal point! (returned %s)\n", (char*) buffer );
        return -1;
    }
    while( isdigit( *p3 ) )
        p3++;
    if( *p3 != ' ' )
    {
        log_error( "ERROR: A space must appear after the minor version! (returned: %s)\n", (char *)buffer );
        return -1;
    }
    *p2 = ' '; // Put in a space for atoi below.
    p2++;

    int major = atoi( p1 );
    int minor = atoi( p2 );
    int minor_revision = 2;

    if( major * 10 + minor < 10 + minor_revision )
    {
        // If the language version did not match, check to see if OPENCL_1_0_DEVICE is set.
        if( getenv("OPENCL_1_0_DEVICE"))
        {
          log_info( "WARNING: This test was run with OPENCL_1_0_DEVICE defined!  This is not a OpenCL 1.1 or OpenCL 1.2 compatible device!!!\n" );
        }
        else if( getenv("OPENCL_1_1_DEVICE"))
        {
          log_info( "WARNING: This test was run with OPENCL_1_1_DEVICE defined!  This is not a OpenCL 1.2 compatible device!!!\n" );
        }
        else
        {
          log_error( "ERROR: OpenCL device language version returned is less than 1.%d! (Returned: %s)\n", minor_revision, (char *)buffer );
          return -1;
        }
    }

    // Sanity checks on the returned values
    if( length != (strlen( (char *)buffer ) + 1 ))
    {
        log_error( "ERROR: Returned length of version string does not match actual length (actual: %d, returned: %d)\n", (int)strlen( (char *)buffer ), (int)length );
        return -1;
    }

    return 0;
}

