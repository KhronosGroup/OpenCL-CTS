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
#include "../../../test_common/harness/fpcontrol.h"

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif


bool                gDebugTrace = false;
bool                gTestMaxImages = false, gTestSmallImages = false, gTestRounding = false;
int                 gTypesToTest = 0;
cl_channel_type     gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order    gChannelOrderToUse = (cl_channel_order)-1;
bool                gEnablePitch = false;
cl_device_type      gDeviceType = CL_DEVICE_TYPE_DEFAULT;

cl_command_queue    queue;
cl_context          context;

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if ( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [options]\n", execName );
    log_info( "Where:\n" );
    log_info( "\n" );
    log_info( "\tThe following flags specify the types to test. They can be combined; if none are specified, all are tested:\n" );
    log_info( "\t\tint - Test integer I/O (read_imagei)\n" );
    log_info( "\t\tuint - Test unsigned integer I/O (read_imageui)\n" );
    log_info( "\t\tfloat - Test float I/O (read_imagef)\n" );
    log_info( "\n" );
    log_info( "You may also use appropriate CL_ channel type and ordering constants.\n" );
    log_info( "\n" );
    log_info( "\t1D - Only test 1D images\n" );
    log_info( "\t2D - Only test 2D images\n" );
    log_info( "\t3D - Only test 3D images\n" );
    log_info( "\t1Darray - Only test 1D image arrays\n" );
    log_info( "\t2Darray - Only test 2D image arrays\n" );
    log_info( "\n" );
    log_info( "\tThe following modify the types of images tested:\n" );
    log_info( "\t\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\t\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\n" );
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\tuse_pitches - Enables row and slice pitches\n" );
}


extern int test_image_set( cl_device_id device, cl_mem_object_type imageType );

int main(int argc, const char *argv[])
{
    cl_platform_id  platform;
    cl_device_id    device;
    cl_channel_type chanType;
    cl_channel_order chanOrder;
    char            str[ 128 ];
    int             testMethods = 0;
    bool            randomize = false;

    test_start();

    //Check CL_DEVICE_TYPE environment variable
    checkDeviceTypeOverride( &gDeviceType );

    // Parse arguments
    for ( int i = 1; i < argc; i++ )
    {
        strncpy( str, argv[ i ], sizeof( str ) - 1 );

        if ( strcmp( str, "cpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_CPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_CPU;
        else if ( strcmp( str, "gpu" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_GPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_GPU;
        else if ( strcmp( str, "accelerator" ) == 0 || strcmp( str, "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_ACCELERATOR;
        else if ( strcmp( str, "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_DEFAULT;

        else if ( strcmp( str, "debug_trace" ) == 0 )
            gDebugTrace = true;

        else if ( strcmp( str, "small_images" ) == 0 )
            gTestSmallImages = true;
        else if ( strcmp( str, "max_images" ) == 0 )
            gTestMaxImages = true;
        else if ( strcmp( str, "use_pitches" ) == 0 )
            gEnablePitch = true;

        else if ( strcmp( str, "int" ) == 0 )
            gTypesToTest |= kTestInt;
        else if ( strcmp( str, "uint" ) == 0 )
            gTypesToTest |= kTestUInt;
        else if ( strcmp( str, "float" ) == 0 )
            gTypesToTest |= kTestFloat;

        else if ( strcmp( str, "randomize" ) == 0 )
            randomize = true;

        else if ( strcmp( str, "1D" ) == 0 )
            testMethods |= k1D;
        else if( strcmp( str, "2D" ) == 0 )
            testMethods |= k2D;
        else if( strcmp( str, "3D" ) == 0 )
            testMethods |= k3D;
        else if( strcmp( str, "1Darray" ) == 0 )
            testMethods |= k1DArray;
        else if( strcmp( str, "2Darray" ) == 0 )
            testMethods |= k2DArray;

        else if ( strcmp( str, "help" ) == 0 || strcmp( str, "?" ) == 0 )
        {
            printUsage( argv[ 0 ] );
            return -1;
        }

        else if ( ( chanType = get_channel_type_from_name( str ) ) != (cl_channel_type)-1 )
            gChannelTypeToUse = chanType;

        else if ( ( chanOrder = get_channel_order_from_name( str ) ) != (cl_channel_order)-1 )
            gChannelOrderToUse = chanOrder;
        else
        {
            log_error( "ERROR: Unknown argument %d: %s.  Exiting....\n", i, str );
            return -1;
        }
    }

    if (testMethods == 0)
        testMethods = k1D | k2D | k3D | k1DArray | k2DArray;
    if ( gTypesToTest == 0 )
        gTypesToTest = kTestAllTypes;

    // Seed the random # generators
    if ( randomize )
    {
        gRandomSeed = (unsigned) (((int64_t) clock() * 1103515245 + 12345) >> 8);
        gReSeed = 1;
        log_info( "Random seed: %d\n", gRandomSeed );
    }

    int error;
    // Get our platform
    error = clGetPlatformIDs(1, &platform, NULL);
    if ( error )
    {
        print_error( error, "Unable to get platform" );
        test_finish();
        return -1;
    }

    // Get our device
    error = clGetDeviceIDs(platform,  gDeviceType, 1, &device, NULL );
    if ( error )
    {
        print_error( error, "Unable to get specified device" );
        test_finish();
        return -1;
    }

    // Get the device type so we know if it is a GPU even if default is passed in.
    error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(gDeviceType), &gDeviceType, NULL);
    if ( error )
    {
        print_error( error, "Unable to get device type" );
        test_finish();
        return -1;
    }


    if ( printDeviceHeader( device ) != CL_SUCCESS )
    {
        test_finish();
        return -1;
    }

    // Check for image support
    if (checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED) {
        log_info("Device does not support images. Skipping test.\n");
        test_finish();
        return 0;
    }

    // Create a context to test with
    context = clCreateContext( NULL, 1, &device, notify_callback, NULL, &error );
    if ( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create testing context" );
        test_finish();
        return -1;
    }

    // Create a queue against the context
    queue = clCreateCommandQueue( context, device, 0, &error );
    if ( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create testing command queue" );
        test_finish();
        return -1;
    }

    if ( gTestSmallImages )
        log_info( "Note: Using small test images\n" );

    // On most platforms which support denorm, default is FTZ off. However,
    // on some hardware where the reference is computed, default might be flush denorms to zero e.g. arm.
    // This creates issues in result verification. Since spec allows the implementation to either flush or
    // not flush denorms to zero, an implementation may choose not to flush i.e. return denorm result whereas
    // reference result may be zero (flushed denorm). Hence we need to disable denorm flushing on host side
    // where reference is being computed to make sure we get non-flushed reference result. If implementation
    // returns flushed result, we correctly take care of that in verification code.

    FPU_mode_type oldMode;
    DisableFTZ(&oldMode);

    // Run the test now
    int ret = 0;
    if (testMethods & k1D)
        ret += test_image_set( device, CL_MEM_OBJECT_IMAGE1D );
    if (testMethods & k2D)
        ret += test_image_set( device, CL_MEM_OBJECT_IMAGE2D );
    if (testMethods & k3D)
        ret += test_image_set( device, CL_MEM_OBJECT_IMAGE3D );
    if (testMethods & k1DArray)
        ret += test_image_set( device, CL_MEM_OBJECT_IMAGE1D_ARRAY );
    if (testMethods & k2DArray)
        ret += test_image_set( device, CL_MEM_OBJECT_IMAGE2D_ARRAY );

    // Restore FP state before leaving
    RestoreFPState(&oldMode);

    error = clFinish(queue);
    if (error)
        print_error(error, "clFinish failed.");

    clReleaseContext(context);
    clReleaseCommandQueue(queue);

    if (gTestFailure == 0) {
        if (gTestCount > 1)
            log_info("PASSED %d of %d tests.\n", gTestCount, gTestCount);
        else
            log_info("PASSED test.\n");
    }
    else if (gTestFailure > 0) {
        if (gTestCount > 1)
            log_error("FAILED %d of %d tests.\n", gTestFailure, gTestCount);
        else
            log_error("FAILED test.\n");
    }

    // Clean up
    test_finish();

    if (gTestFailure > 0)
        return gTestFailure;

    return ret;
}
