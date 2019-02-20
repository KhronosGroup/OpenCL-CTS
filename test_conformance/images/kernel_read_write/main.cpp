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

bool            gDebugTrace = false, gExtraValidateInfo = false, gDisableOffsets = false, gTestSmallImages = false, gTestMaxImages = false, gTestRounding = false;
cl_filter_mode    gFilterModeToUse = (cl_filter_mode)-1;
// Default is CL_MEM_USE_HOST_PTR for the test
cl_mem_flags    gMemFlagsToUse = CL_MEM_USE_HOST_PTR;
bool            gUseKernelSamplers = false;
int                gTypesToTest = 0;
cl_addressing_mode gAddressModeToUse = (cl_addressing_mode)-1;
int             gNormalizedModeToUse = 7;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order gChannelOrderToUse = (cl_channel_order)-1;
bool            gEnablePitch = false;
cl_device_type    gDeviceType = CL_DEVICE_TYPE_DEFAULT;

cl_command_queue queue;
cl_context context;

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [read] [write] [CL_FILTER_LINEAR|CL_FILTER_NEAREST] [no_offsets] [debug_trace] [small_images]\n", execName );
    log_info( "Where:\n" );
    log_info( "\n" );
    log_info( "\tThe following flags specify what kinds of operations to test. They can be combined; if none are specified, all are tested:\n" );
    log_info( "\t\tread - Tests reading from an image\n" );
    log_info( "\t\twrite - Tests writing to an image (can be specified with read to run both; default is both)\n" );
    log_info( "\n" );
    log_info( "\tThe following flags specify the types to test. They can be combined; if none are specified, all are tested:\n" );
    log_info( "\t\tint - Test integer I/O (read_imagei, write_imagei)\n" );
    log_info( "\t\tuint - Test unsigned integer I/O (read_imageui, write_imageui)\n" );
    log_info( "\t\tfloat - Test float I/O (read_imagef, write_imagef)\n" );
    log_info( "\n" );
    log_info( "\tCL_FILTER_LINEAR - Only tests formats with CL_FILTER_LINEAR filtering\n" );
    log_info( "\tCL_FILTER_NEAREST - Only tests formats with CL_FILTER_NEAREST filtering\n" );
    log_info( "\n" );
    log_info( "\tNORMALIZED - Only tests formats with NORMALIZED coordinates\n" );
    log_info( "\tUNNORMALIZED - Only tests formats with UNNORMALIZED coordinates\n" );
    log_info( "\n" );
    log_info( "\tCL_ADDRESS_CLAMP - Only tests formats with CL_ADDRESS_CLAMP addressing\n" );
    log_info( "\tCL_ADDRESS_CLAMP_TO_EDGE - Only tests formats with CL_ADDRESS_CLAMP_TO_EDGE addressing\n" );
    log_info( "\tCL_ADDRESS_REPEAT - Only tests formats with CL_ADDRESS_REPEAT addressing\n" );
    log_info( "\tCL_ADDRESS_MIRRORED_REPEAT - Only tests formats with CL_ADDRESS_MIRRORED_REPEAT addressing\n" );
    log_info( "\n" );
    log_info( "You may also use appropriate CL_ channel type and ordering constants.\n" );
    log_info( "\n" );
    log_info( "\t1D - Only test 1D images\n" );
    log_info( "\t2D - Only test 2D images\n" );
    log_info( "\t3D - Only test 3D images\n" );
    log_info( "\t1Darray - Only test 1D image arrays\n" );
    log_info( "\t2Darray - Only test 2D image arrays\n" );
    log_info( "\n" );
    log_info( "\tlocal_samplers - Use samplers declared in the kernel functions instead of passed in as arguments\n" );
    log_info( "\n" );
    log_info( "\tThe following specify to use the specific flag to allocate images to use in the tests:\n" );
    log_info( "\t\tCL_MEM_COPY_HOST_PTR\n" );
    log_info( "\t\tCL_MEM_USE_HOST_PTR (default)\n" );
    log_info( "\t\tCL_MEM_ALLOC_HOST_PTR\n" );
    log_info( "\t\tNO_HOST_PTR - Specifies to use none of the above flags\n" );
    log_info( "\n" );
    log_info( "\tThe following modify the types of images tested:\n" );
    log_info( "\t\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\t\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\t\trounding - Runs every format through a single image filled with every possible value for that image format, to verify rounding works properly\n" );
    log_info( "\n" );
    log_info( "\tno_offsets - Disables offsets when testing reads (can be good for diagnosing address repeating/clamping problems)\n" );
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\textra_validate - Enables additional validation failure debug information\n" );
    log_info( "\tuse_pitches - Enables row and slice pitches\n" );
}



enum TestTypes
{
    kReadTests = 1 << 0 ,
    kWriteTests = 1 << 1,
    kAllTests = ( kReadTests | kWriteTests )
};

extern int test_image_set( cl_device_id device, test_format_set_fn formatTestFn, cl_mem_object_type imageType );

int main(int argc, const char *argv[])
{
    cl_platform_id  platform;
    cl_device_id       device;
    cl_channel_type chanType;
    cl_channel_order chanOrder;
    char            str[ 128 ];
    int                testTypesToRun = 0;
    int             testMethods = 0;
    bool            randomize = false;

    test_start();

    //Check CL_DEVICE_TYPE environment variable
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

        else if( strcmp( str, "CL_FILTER_NEAREST" ) == 0 || strcmp( str, "NEAREST" ) == 0 )
            gFilterModeToUse = CL_FILTER_NEAREST;
        else if( strcmp( str, "CL_FILTER_LINEAR" ) == 0 || strcmp( str, "LINEAR" ) == 0 )
            gFilterModeToUse = CL_FILTER_LINEAR;

        else if( strcmp( str, "CL_ADDRESS_NONE" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_NONE;
        else if( strcmp( str, "CL_ADDRESS_CLAMP" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_CLAMP;
        else if( strcmp( str, "CL_ADDRESS_CLAMP_TO_EDGE" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_CLAMP_TO_EDGE;
        else if( strcmp( str, "CL_ADDRESS_REPEAT" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_REPEAT;
        else if( strcmp( str, "CL_ADDRESS_MIRRORED_REPEAT" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_MIRRORED_REPEAT;

        else if( strcmp( str, "NORMALIZED" ) == 0 )
            gNormalizedModeToUse = true;
        else if( strcmp( str, "UNNORMALIZED" ) == 0 )
            gNormalizedModeToUse = false;


        else if( strcmp( str, "no_offsets" ) == 0 )
            gDisableOffsets = true;
        else if( strcmp( str, "small_images" ) == 0 )
            gTestSmallImages = true;
        else if( strcmp( str, "max_images" ) == 0 )
            gTestMaxImages = true;
        else if( strcmp( str, "use_pitches" ) == 0 )
            gEnablePitch = true;
        else if( strcmp( str, "rounding" ) == 0 )
            gTestRounding = true;
        else if( strcmp( str, "extra_validate" ) == 0 )
            gExtraValidateInfo = true;

        else if( strcmp( str, "read" ) == 0 )
            testTypesToRun |= kReadTests;
        else if( strcmp( str, "write" ) == 0 )
            testTypesToRun |= kWriteTests;

        else if( strcmp( str, "local_samplers" ) == 0 )
            gUseKernelSamplers = true;

        else if( strcmp( str, "int" ) == 0 )
            gTypesToTest |= kTestInt;
        else if( strcmp( str, "uint" ) == 0 )
            gTypesToTest |= kTestUInt;
        else if( strcmp( str, "float" ) == 0 )
            gTypesToTest |= kTestFloat;

        else if( strcmp( str, "randomize" ) == 0 )
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

        else if( strcmp( str, "CL_MEM_COPY_HOST_PTR" ) == 0 || strcmp( str, "COPY_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;
        else if( strcmp( str, "CL_MEM_USE_HOST_PTR" ) == 0 || strcmp( str, "USE_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_USE_HOST_PTR;
        else if( strcmp( str, "CL_MEM_ALLOC_HOST_PTR" ) == 0 || strcmp( str, "ALLOC_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_ALLOC_HOST_PTR;
        else if( strcmp( str, "NO_HOST_PTR" ) == 0 )
            gMemFlagsToUse = 0;

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

    if (testMethods == 0)
        testMethods = k1D | k2D | k3D | k1DArray | k2DArray;
    if( testTypesToRun == 0 )
        testTypesToRun = kAllTests;
    if( gTypesToTest == 0 )
        gTypesToTest = kTestAllTypes;

#if defined( __APPLE__ )
#if defined( __i386__ ) || defined( __x86_64__ )
#define    kHasSSE3                0x00000008
#define kHasSupplementalSSE3    0x00000100
#define    kHasSSE4_1              0x00000400
#define    kHasSSE4_2              0x00000800
    /* check our environment for a hint to disable SSE variants */
    {
        const char *env = getenv( "CL_MAX_SSE" );
        if( env )
        {
            extern int _cpu_capabilities;
            int mask = 0;
            if( 0 == strcmp( env, "SSE4.1" ) )
                mask = kHasSSE4_2;
            else if( 0 == strcmp( env, "SSSE3" ) )
                mask = kHasSSE4_2 | kHasSSE4_1;
            else if( 0 == strcmp( env, "SSE3" ) )
                mask = kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3;
            else if( 0 == strcmp( env, "SSE2" ) )
                mask = kHasSSE4_2 | kHasSSE4_1 | kHasSupplementalSSE3 | kHasSSE3;

            log_info( "*** Environment: CL_MAX_SSE = %s ***\n", env );
            _cpu_capabilities &= ~mask;
        }
    }
#endif
#endif

    // Seed the random # generators
    if( randomize )
    {
        gRandomSeed = (unsigned) (((int64_t) clock() * 1103515245 + 12345) >> 8);
        gReSeed = 1;
        log_info( "Random seed: %d\n", gRandomSeed );
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

    // Get the device type so we know if it is a GPU even if default is passed in.
    error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(gDeviceType), &gDeviceType, NULL);
    if( error )
    {
        print_error( error, "Unable to get device type" );
        test_finish();
        return -1;
    }


    if( printDeviceHeader( device ) != CL_SUCCESS )
    {
        test_finish();
        return -1;
    }

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
    {
        if (testTypesToRun & kReadTests)
            ret += test_image_set( device, test_read_image_formats, CL_MEM_OBJECT_IMAGE1D );
        if (testTypesToRun & kWriteTests)
            ret += test_image_set( device, test_write_image_formats, CL_MEM_OBJECT_IMAGE1D );
    }
    if (testMethods & k2D)
    {
        if (testTypesToRun & kReadTests)
            ret += test_image_set( device, test_read_image_formats, CL_MEM_OBJECT_IMAGE2D );
        if (testTypesToRun & kWriteTests)
            ret += test_image_set( device, test_write_image_formats, CL_MEM_OBJECT_IMAGE2D );
    }
    if (testMethods & k3D)
    {
        if (testTypesToRun & kReadTests)
            ret += test_image_set( device, test_read_image_formats, CL_MEM_OBJECT_IMAGE3D );
        if (testTypesToRun & kWriteTests)
            ret += test_image_set( device, test_write_image_formats, CL_MEM_OBJECT_IMAGE3D );
    }
    if (testMethods & k1DArray)
    {
        if (testTypesToRun & kReadTests)
            ret += test_image_set( device, test_read_image_formats, CL_MEM_OBJECT_IMAGE1D_ARRAY );
        if (testTypesToRun & kWriteTests)
            ret += test_image_set( device, test_write_image_formats, CL_MEM_OBJECT_IMAGE1D_ARRAY );
    }
    if (testMethods & k2DArray)
    {
        if (testTypesToRun & kReadTests)
            ret += test_image_set( device, test_read_image_formats, CL_MEM_OBJECT_IMAGE2D_ARRAY );
        if (testTypesToRun & kWriteTests)
            ret += test_image_set( device, test_write_image_formats, CL_MEM_OBJECT_IMAGE2D_ARRAY );
    }

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
    } else if (gTestFailure > 0) {
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


