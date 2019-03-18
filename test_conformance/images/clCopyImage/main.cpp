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
#include "../../../test_common/harness/compat.h"

#include <stdio.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#include <sys/time.h>
#endif

#include "../testBase.h"
#include "../../../test_common/harness/testHarness.h"

bool            gDebugTrace = false, gTestSmallImages = false, gTestMaxImages = false, gUseRamp = false, gTestRounding = false, gEnablePitch = false, gTestMipmaps = false;
int             gTypesToTest = 0;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order gChannelOrderToUse = (cl_channel_order)-1;
cl_device_type    gDeviceType = CL_DEVICE_TYPE_DEFAULT;
cl_context context;
cl_command_queue queue;
static cl_device_id device;

extern int test_image_set( cl_device_id device, MethodsToTest testMethod );

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

static void printUsage( const char *execName );

int test_1D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k1D );
}
int test_2D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2D );
}
int test_3D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k3D );
}
int test_1Darray(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k1DArray );
}
int test_2Darray(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2DArray );
}
int test_2Dto3D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2DTo3D );
}
int test_3Dto2D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k3DTo2D );
}
int test_2Darrayto2D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2DArrayTo2D );
}
int test_2Dto2Darray(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2DTo2DArray );
}
int test_2Darrayto3D(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k2DArrayTo3D );
}
int test_3Dto2Darray(cl_device_id deviceID, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, k3DTo2DArray );
}

basefn basefn_list[] = {
    test_1D,
    test_2D,
    test_3D,
    test_1Darray,
    test_2Darray,
    test_2Dto3D,
    test_3Dto2D,
    test_2Darrayto2D,
    test_2Dto2Darray,
    test_2Darrayto3D,
    test_3Dto2Darray,
};

const char *basefn_names[] = {
    "1D",
    "2D",
    "3D",
    "1Darray",
    "2Darray",
    "2Dto3D",
    "3Dto2D",
    "2Darrayto2D",
    "2Dto2Darray",
    "2Darrayto3D",
    "3Dto2Darray",
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0])) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int num_fns = sizeof(basefn_names) / sizeof(char *);

int main(int argc, const char *argv[])
{
    cl_platform_id  platform;
    cl_channel_type chanType;
    cl_channel_order chanOrder;
    bool            randomize = false;

    test_start();

    checkDeviceTypeOverride( &gDeviceType );

    const char ** argList = (const char **)calloc( argc, sizeof( char*) );

    if( NULL == argList )
    {
        log_error( "Failed to allocate memory for argList array.\n" );
        return 1;
    }

    argList[0] = argv[0];
    size_t argCount = 1;

    // Parse arguments
    for( int i = 1; i < argc; i++ )
    {
        if( strcmp( argv[i], "cpu" ) == 0 || strcmp( argv[i], "CL_DEVICE_TYPE_CPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_CPU;
        else if( strcmp( argv[i], "gpu" ) == 0 || strcmp( argv[i], "CL_DEVICE_TYPE_GPU" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_GPU;
        else if( strcmp( argv[i], "accelerator" ) == 0 || strcmp( argv[i], "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_ACCELERATOR;
        else if( strcmp( argv[i], "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
            gDeviceType = CL_DEVICE_TYPE_DEFAULT;

        else if( strcmp( argv[i], "test_mipmaps" ) == 0 )
        {
            gTestMipmaps = true;
            // Don't test pitches with mipmaps, at least currently.
            gEnablePitch = false;
        }
        else if( strcmp( argv[i], "debug_trace" ) == 0 )
            gDebugTrace = true;

        else if( strcmp( argv[i], "small_images" ) == 0 )
            gTestSmallImages = true;
        else if( strcmp( argv[i], "max_images" ) == 0 )
            gTestMaxImages = true;
        else if( strcmp( argv[i], "use_ramps" ) == 0 )
            gUseRamp = true;

        else if( strcmp( argv[i], "use_pitches" ) == 0 )
            gEnablePitch = true;

        else if( strcmp( argv[i], "randomize" ) == 0 )
            randomize = true;

        else if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 )
        {
            printUsage( argv[ 0 ] );
            return -1;
        }

        else if( ( chanType = get_channel_type_from_name( argv[i] ) ) != (cl_channel_type)-1 )
            gChannelTypeToUse = chanType;

        else if( ( chanOrder = get_channel_order_from_name( argv[i] ) ) != (cl_channel_order)-1 )
            gChannelOrderToUse = chanOrder;
        else
        {
            argList[argCount] = argv[i];
            argCount++;
        }
    }

    // Seed the random # generators
    if( randomize )
    {
        gRandomSeed = (cl_uint) time( NULL );
        log_info( "Random seed: %u.\n", gRandomSeed );
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
    queue = clCreateCommandQueueWithProperties( context, device, 0, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create testing command queue" );
        test_finish();
        return -1;
    }

    if( gTestSmallImages )
        log_info( "Note: Using small test images\n" );

    int ret = parseAndCallCommandLineTests( argCount, argList, NULL, num_fns, basefn_list, basefn_names, true, 0, 0 );

    error = clFinish(queue);
    if (error)
        print_error(error, "clFinish failed.");

    if (gTestFailure == 0) {
        if (gTestCount > 1)
            log_info("PASSED %d of %d sub-tests.\n", gTestCount, gTestCount);
        else
            log_info("PASSED sub-test.\n");
    } else if (gTestFailure > 0) {
        if (gTestCount > 1)
            log_error("FAILED %d of %d sub-tests.\n", gTestFailure, gTestCount);
        else
            log_error("FAILED sub-test.\n");
    }

    // Clean up
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(argList);
    test_finish();

    return ret;
}

static void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [option] [test_names]\n", execName );
    log_info( "Options:\n" );
    log_info( "\ttest_mipmaps - Test with mipmapped images\n" );
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\trandomize - Use random seed\n" );
    log_info( "\tuse_pitches - Enables row and slice pitches\n" );
    log_info( "\tuse_ramp - Instead of random data, uses images filled with ramps (and 0xff on any padding pixels) to ease debugging\n" );
    log_info( "\n" );
    log_info( "Test names:\n" );
    for( int i = 0; i < num_fns; i++ )
    {
        log_info( "\t%s\n", basefn_names[i] );
    }
}
