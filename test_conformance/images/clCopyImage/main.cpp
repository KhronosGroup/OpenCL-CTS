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
#include <stdio.h>
#include <stdlib.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <math.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#include <sys/time.h>
#endif

#include "../testBase.h"

bool            gDebugTrace = false, gTestSmallImages = false, gTestMaxImages = false, gUseRamp = false, gTestRounding = false, gEnablePitch = false;
int                gTypesToTest = 0;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order gChannelOrderToUse = (cl_channel_order)-1;
cl_device_type    gDeviceType = CL_DEVICE_TYPE_DEFAULT;
cl_context context;
cl_command_queue queue;

extern int test_image_set( cl_device_id device, MethodsToTest testMethod );

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [debug_trace] [small_images]\n", execName );
    log_info( "Where:\n" );
    log_info( "\t1D - Only test 1D images\n" );
    log_info( "\t2D - Only test 2D images\n" );
    log_info( "\t3D - Only test 3D images\n" );
    log_info( "\t1Darray - Only test 1D image arrays\n" );
    log_info( "\t2Darray - Only test 2D image arrays\n" );
    log_info( "\t2Dto3D - Only test 2D -> 3D images\n" );
    log_info( "\t3Dto2D - Only test 3D -> 2D images\n" );
    log_info( "\t2Darrayto2D - Only test 2D image arrays -> 2D images\n" );
    log_info( "\t2Dto2Darray - Only test 2D images -> 2D image arrays\n" );
    log_info( "\t2Darrayto3D - Only test 2D image arrays -> 3D images\n" );
    log_info( "\t3Dto2Darray - Only test 3D images -> 2D image arrays\n" );
    log_info( "\n" );
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\trounding - Runs every format through a single image filled with every possible value for that image format, to verify rounding works properly\n" );
    //log_info( "\tuse_pitches - Enables row and slice pitches\n" );
    log_info( "\tuse_ramp - Instead of random data, uses images filled with ramps (and 0xff on any padding pixels) to ease debugging\n" );
}


int main(int argc, const char *argv[])
{
    cl_platform_id  platform;
    cl_device_id       device;
    cl_channel_type chanType;
    cl_channel_order chanOrder;
    char            str[ 128 ];
    int                testMethods = 0;
    bool            randomize = false;

    test_start();

    checkDeviceTypeOverride( &gDeviceType );

    // Parse arguments
    for( int i = 1; i < argc; i++ )
    {
        strncpy( str, argv[ i ], sizeof( str ) - 1 );

        if( strcmp( str, "cpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_CPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_CPU;
        else if( strcmp( str, "gpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_GPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_GPU;
        else if( strcmp( str, "accelerator" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( str, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_DEFAULT;

        else if( strcmp( str, "debug_trace" ) == 0 )
            gDebugTrace = true;

        else if( strcmp( str, "small_images" ) == 0 )
            gTestSmallImages = true;
        else if( strcmp( str, "max_images" ) == 0 )
            gTestMaxImages = true;
        else if( strcmp( str, "use_ramps" ) == 0 )
            gUseRamp = true;

        else if( strcmp( str, "use_pitches" ) == 0 )
            gEnablePitch = true;

        else if( strcmp( str, "randomize" ) == 0 )
            randomize = true;

        else if( strcmp( str, "1D" ) == 0 )
            testMethods |= k1D;
        else if( strcmp( str, "2D" ) == 0 )
            testMethods |= k2D;
        else if( strcmp( str, "3D" ) == 0 )
            testMethods |= k3D;
        else if( strcmp( str, "1Darray" ) == 0 )
            testMethods |= k1DArray;
        else if( strcmp( str, "2Darray" ) == 0 )
            testMethods |= k2DArray;
        else if( strcmp( str, "2Dto3D" ) == 0 )
            testMethods |= k2DTo3D;
        else if( strcmp( str, "3Dto2D" ) == 0 )
            testMethods |= k3DTo2D;
        else if( strcmp( str, "2Darrayto2D" ) == 0 )
            testMethods |= k2DArrayTo2D;
        else if( strcmp( str, "2Dto2Darray" ) == 0 )
            testMethods |= k2DTo2DArray;
        else if( strcmp( str, "2Darrayto3D" ) == 0 )
            testMethods |= k2DArrayTo3D;
        else if( strcmp( str, "3Dto2Darray" ) == 0 )
            testMethods |= k3DTo2DArray;

        else if( strcmp( str, "help" ) == 0 || strcmp( str, "?" ) == 0 )
        {
            printUsage( argv[ 0 ] );
            return -1;
        }

        else if( ( chanType = get_channel_type_from_name( str ) ) != (cl_channel_type)-1 )
            gChannelTypeToUse = chanType;

        else if( ( chanOrder = get_channel_order_from_name( str ) ) != (cl_channel_order)-1 )
            gChannelOrderToUse = chanOrder;
        else
        {
            log_error( "ERROR: Unknown argument %d: %s.  Exiting....\n", i, str );
            return -1;
        }

    }

    if( testMethods == 0 )
        testMethods = k1D | k2D | k3D | k1DArray | k2DArray | k2DTo3D | k3DTo2D | k2DArrayTo2D | k2DTo2DArray | k2DArrayTo3D | k3DTo2DArray;

    // Seed the random # generators
    if( randomize )
    {
        gRandomSeed = (cl_uint) clock();
        gReSeed = 1;
    }

    int error;
    // Get our platform
    error = clGetPlatformIDs(1, &platform, NULL);
    if( error )
    {
        print_error( error, "Unable to get platform" );
        test_finish();
        return -1;
    }

    // Get our device
    error = clGetDeviceIDs(platform,  gDeviceType, 1, &device, NULL );
    if( error )
    {
        print_error( error, "Unable to get specified device" );
        test_finish();
        return -1;
    }

    char deviceName[ 128 ], deviceVendor[ 128 ], deviceVersion[ 128 ];
    error = clGetDeviceInfo( device, CL_DEVICE_NAME, sizeof( deviceName ), deviceName, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_VENDOR, sizeof( deviceVendor ), deviceVendor, NULL );
    error |= clGetDeviceInfo( device, CL_DEVICE_VERSION, sizeof( deviceVersion ), deviceVersion, NULL );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to get device information" );
        test_finish();
        return -1;
    }
    log_info("Using compute device: Name = %s, Vendor = %s, Version = %s\n", deviceName, deviceVendor, deviceVersion );

    // Check for image support
    if(checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED) {
        log_info("Device does not support images. Skipping test.\n");
        test_finish();
        return 0;
    }

    // Create a context to test with
    context = clCreateContext( NULL, 1, &device, notify_callback, NULL, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create testing context" );
        test_finish();
        return -1;
    }

    // Create a queue against the context
    queue = clCreateCommandQueue( context, device, 0, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create testing command queue" );
        test_finish();
        return -1;
    }

    if( gTestSmallImages )
        log_info( "Note: Using small test images\n" );

    // Run the test now
    int ret = 0;
    for( int test = k1D; test <= k3DTo2DArray; test <<= 1 )
    {
        if( testMethods & test )
            ret += test_image_set( device, (MethodsToTest)test );
    }

    error = clFinish(queue);
    if (error)
        print_error(error, "clFinish failed.");

    if (gTestFailure == 0) {
        if (gTestCount > 1)
            log_info("PASSED %d of %d tests.\n", gTestCount, gTestCount);
        else
            log_info("PASSED test.\n");
    } else if (gTestFailure > 0) {
        if (gTestCount > 1)
            log_error("FAILED %d of %d tests.\n", gTestFailure, gTestCount);
        else
            log_error("FAILED test.\n");
    }

    // Clean up
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    test_finish();

    if (gTestFailure > 0)
        return gTestFailure;

    return ret;
}
