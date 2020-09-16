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
#include "../harness/compat.h"

#include <stdio.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#include <sys/time.h>
#endif

#include "../testBase.h"
#include "../harness/fpcontrol.h"
#include "../harness/parseParameters.h"

#include <vector>

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif

bool gDebugTrace;
bool gExtraValidateInfo;
bool gDisableOffsets;
bool gTestSmallImages;
bool gTestMaxImages;
bool gTestImage2DFromBuffer;
bool gTestMipmaps;
bool gDeviceLt20 = false;
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

int             gtestTypesToRun = 0;
static int testTypesToRun;

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

static void printUsage( const char *execName );

extern int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, test_format_set_fn formatTestFn, cl_mem_object_type imageType );

/** read_write images only support sampler-less read buildt-ins which require special settings
  * for some global parameters. This pair of functions temporarily overwrite those global parameters
  * and then recover them after completing a read_write test.
  */
static void overwrite_global_params_for_read_write_test(  bool            *tTestMipmaps,
                                                            bool            *tDisableOffsets,
                                                            bool            *tNormalizedModeToUse,
                                                            cl_filter_mode  *tFilterModeToUse)
{
    log_info("Overwrite global settings for read_write image tests. The overwritten values:\n");
    log_info("gTestMipmaps = false, gDisableOffsets = true, gNormalizedModeToUse = false, gFilterModeToUse = CL_FILTER_NEAREST\n" );
    // mipmap images only support sampler read built-in while read_write images only support
    // sampler-less read built-in. Hence we cannot test mipmap for read_write image.
    *tTestMipmaps = gTestMipmaps;
    gTestMipmaps = false;

    // Read_write images are read by sampler-less read which does not handle out-of-bound read
    // It's application responsibility to make sure that the read happens in-bound
    // Therefore we should not enable offset in testing read_write images because it will cause out-of-bound
    *tDisableOffsets    = gDisableOffsets;
    gDisableOffsets     = true;

    // The sampler-less read image functions behave exactly as the corresponding read image functions


    *tNormalizedModeToUse   = gNormalizedModeToUse;
    gNormalizedModeToUse    = false;
    *tFilterModeToUse       = gFilterModeToUse;
    gFilterModeToUse        = CL_FILTER_NEAREST;
}

/** Recover the global settings overwritten for read_write tests. This is necessary because
  * there may be other tests (i.e. read or write) are called together with read_write test.
  */
static void recover_global_params_from_read_write_test(bool            tTestMipmaps,
                                                         bool            tDisableOffsets,
                                                         bool            tNormalizedModeToUse,
                                                         cl_filter_mode  tFilterModeToUse)
{
    gTestMipmaps            = tTestMipmaps;
    gDisableOffsets         = tDisableOffsets;
    gNormalizedModeToUse    = tNormalizedModeToUse;
    gFilterModeToUse        = tFilterModeToUse;
}

static int doTest( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type imageType )
{
    int ret = 0;
    bool is_2d_image = imageType == CL_MEM_OBJECT_IMAGE2D;
    bool            tTestMipMaps = false;
    bool            tDisableOffsets = false;
    bool            tNormalizedModeToUse = false;
    cl_filter_mode  tFilterModeToUse = (cl_filter_mode)-1;
    auto version = get_device_cl_version(device);
    if (version < Version(2, 0)) {
        gDeviceLt20 = true;
    }

    if( testTypesToRun & kReadTests )
    {
        gtestTypesToRun = kReadTests;
        ret += test_image_set( device, context, queue, test_read_image_formats, imageType );

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info( "Testing read_image{f | i | ui} for 2D image from buffer\n" );

            // NOTE: for 2D image from buffer test, gTestSmallImages, gTestMaxImages, gTestRounding and gTestMipmaps must be false
            if( gTestSmallImages == false && gTestMaxImages == false && gTestRounding == false && gTestMipmaps == false )
            {
                cl_mem_flags saved_gMemFlagsToUse = gMemFlagsToUse;
                gTestImage2DFromBuffer = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this for 2.0
                gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;

                ret += test_image_set( device, context, queue, test_read_image_formats, imageType );

                gTestImage2DFromBuffer = false;
                gMemFlagsToUse = saved_gMemFlagsToUse;
            }
        }
    }

    if( testTypesToRun & kWriteTests )
    {
        gtestTypesToRun = kWriteTests;
        ret += test_image_set( device, context, queue, test_write_image_formats, imageType );

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info( "Testing write_image{f | i | ui} for 2D image from buffer\n" );

            // NOTE: for 2D image from buffer test, gTestSmallImages, gTestMaxImages,gTestRounding and gTestMipmaps must be false
            if( gTestSmallImages == false && gTestMaxImages == false && gTestRounding == false && gTestMipmaps == false )
            {
                bool saved_gEnablePitch = gEnablePitch;
                cl_mem_flags saved_gMemFlagsToUse = gMemFlagsToUse;
                gEnablePitch = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this for 2.0
                gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;
                gTestImage2DFromBuffer = true;

                ret += test_image_set( device, context, queue, test_write_image_formats, imageType );

                gTestImage2DFromBuffer = false;
                gMemFlagsToUse = saved_gMemFlagsToUse;
                gEnablePitch = saved_gEnablePitch;
            }
        }
    }

    if (testTypesToRun & kReadWriteTests) {
        if (gDeviceLt20)  {
            log_info("TEST skipped, Opencl 2.0 + requried for this test");
            return ret;
        }
    }

    if( ( testTypesToRun & kReadWriteTests ) && !gTestMipmaps )
    {
        gtestTypesToRun = kReadWriteTests;
        overwrite_global_params_for_read_write_test(&tTestMipMaps, &tDisableOffsets, &tNormalizedModeToUse, &tFilterModeToUse);
        ret += test_image_set( device, context, queue, test_read_image_formats, imageType );

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info("Testing read_image{f | i | ui} for 2D image from buffer\n");

            // NOTE: for 2D image from buffer test, gTestSmallImages, gTestMaxImages, gTestRounding and gTestMipmaps must be false
            if( gTestSmallImages == false && gTestMaxImages == false && gTestRounding == false && gTestMipmaps == false )
            {
                cl_mem_flags saved_gMemFlagsToUse = gMemFlagsToUse;
                gTestImage2DFromBuffer = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this for 2.0
                gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;

                ret += test_image_set( device, context, queue, test_read_image_formats, imageType );

                gTestImage2DFromBuffer = false;
                gMemFlagsToUse = saved_gMemFlagsToUse;
            }
        }

        ret += test_image_set( device, context, queue, test_write_image_formats, imageType );

        if( is_2d_image && is_extension_available( device, "cl_khr_image2d_from_buffer" ) )
        {
            log_info("Testing write_image{f | i | ui} for 2D image from buffer\n");

            // NOTE: for 2D image from buffer test, gTestSmallImages, gTestMaxImages,gTestRounding and gTestMipmaps must be false
            if( gTestSmallImages == false && gTestMaxImages == false && gTestRounding == false && gTestMipmaps == false )
            {
                bool saved_gEnablePitch = gEnablePitch;
                cl_mem_flags saved_gMemFlagsToUse = gMemFlagsToUse;
                gEnablePitch = true;

                // disable CL_MEM_USE_HOST_PTR for 1.2 extension but enable this for 2.0
                gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;
                gTestImage2DFromBuffer = true;

                ret += test_image_set( device, context, queue, test_write_image_formats, imageType );

                gTestImage2DFromBuffer = false;
                gMemFlagsToUse = saved_gMemFlagsToUse;
                gEnablePitch = saved_gEnablePitch;
            }
        }

        recover_global_params_from_read_write_test( tTestMipMaps, tDisableOffsets, tNormalizedModeToUse, tFilterModeToUse );
    }

    return ret;
}

int test_1D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, CL_MEM_OBJECT_IMAGE1D );
}
int test_2D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, CL_MEM_OBJECT_IMAGE2D );
}
int test_3D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, CL_MEM_OBJECT_IMAGE3D );
}
int test_1Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, CL_MEM_OBJECT_IMAGE1D_ARRAY );
}
int test_2Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return doTest( device, context, queue, CL_MEM_OBJECT_IMAGE2D_ARRAY );
}

test_definition test_list[] = {
    ADD_TEST( 1D ),
    ADD_TEST( 2D ),
    ADD_TEST( 3D ),
    ADD_TEST( 1Darray ),
    ADD_TEST( 2Darray ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    cl_channel_type chanType;
    cl_channel_order chanOrder;

    argc = parseCustomParam(argc, argv);
    if (argc == -1)
    {
        return -1;
    }

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
        if( strcmp( argv[i], "debug_trace" ) == 0 )
            gDebugTrace = true;

        else if( strcmp( argv[i], "CL_FILTER_NEAREST" ) == 0 || strcmp( argv[i], "NEAREST" ) == 0 )
            gFilterModeToUse = CL_FILTER_NEAREST;
        else if( strcmp( argv[i], "CL_FILTER_LINEAR" ) == 0 || strcmp( argv[i], "LINEAR" ) == 0 )
            gFilterModeToUse = CL_FILTER_LINEAR;

        else if( strcmp( argv[i], "CL_ADDRESS_NONE" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_NONE;
        else if( strcmp( argv[i], "CL_ADDRESS_CLAMP" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_CLAMP;
        else if( strcmp( argv[i], "CL_ADDRESS_CLAMP_TO_EDGE" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_CLAMP_TO_EDGE;
        else if( strcmp( argv[i], "CL_ADDRESS_REPEAT" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_REPEAT;
        else if( strcmp( argv[i], "CL_ADDRESS_MIRRORED_REPEAT" ) == 0 )
            gAddressModeToUse = CL_ADDRESS_MIRRORED_REPEAT;

        else if( strcmp( argv[i], "NORMALIZED" ) == 0 )
            gNormalizedModeToUse = true;
        else if( strcmp( argv[i], "UNNORMALIZED" ) == 0 )
            gNormalizedModeToUse = false;


        else if( strcmp( argv[i], "no_offsets" ) == 0 )
            gDisableOffsets = true;
        else if( strcmp( argv[i], "small_images" ) == 0 )
            gTestSmallImages = true;
        else if( strcmp( argv[i], "max_images" ) == 0 )
            gTestMaxImages = true;
        else if( strcmp( argv[i], "use_pitches" ) == 0 )
            gEnablePitch = true;
        else if( strcmp( argv[i], "rounding" ) == 0 )
            gTestRounding = true;
        else if( strcmp( argv[i], "extra_validate" ) == 0 )
            gExtraValidateInfo = true;
        else if( strcmp( argv[i], "test_mipmaps" ) == 0 ) {
            // 2.0 Spec does not allow using mem flags, unnormalized coordinates with mipmapped images
            gTestMipmaps = true;
            gMemFlagsToUse = 0;
            gNormalizedModeToUse = true;
        }

        else if( strcmp( argv[i], "read" ) == 0 )
            testTypesToRun |= kReadTests;
        else if( strcmp( argv[i], "write" ) == 0 )
            testTypesToRun |= kWriteTests;
        else if( strcmp( argv[i], "read_write" ) == 0 )
        {
            testTypesToRun |= kReadWriteTests;
        }

        else if( strcmp( argv[i], "local_samplers" ) == 0 )
            gUseKernelSamplers = true;

        else if( strcmp( argv[i], "int" ) == 0 )
            gTypesToTest |= kTestInt;
        else if( strcmp( argv[i], "uint" ) == 0 )
            gTypesToTest |= kTestUInt;
        else if( strcmp( argv[i], "float" ) == 0 )
            gTypesToTest |= kTestFloat;

        else if( strcmp( argv[i], "CL_MEM_COPY_HOST_PTR" ) == 0 || strcmp( argv[i], "COPY_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_COPY_HOST_PTR;
        else if( strcmp( argv[i], "CL_MEM_USE_HOST_PTR" ) == 0 || strcmp( argv[i], "USE_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_USE_HOST_PTR;
        else if( strcmp( argv[i], "CL_MEM_ALLOC_HOST_PTR" ) == 0 || strcmp( argv[i], "ALLOC_HOST_PTR" ) == 0 )
            gMemFlagsToUse = CL_MEM_ALLOC_HOST_PTR;
        else if( strcmp( argv[i], "NO_HOST_PTR" ) == 0 )
            gMemFlagsToUse = 0;

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

    if( testTypesToRun == 0 )
        testTypesToRun = kAllTests;
    if( gTypesToTest == 0 )
        gTypesToTest = kTestAllTypes;

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

    int ret = runTestHarness( argCount, argList, test_num, test_list, true, false, 0 );

    // Restore FP state before leaving
    RestoreFPState(&oldMode);

    free(argList);
    return ret;
}

static void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [options] [test_names]\n", execName );
    log_info( "Options:\n" );
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
    log_info( "\ttest_mipmaps - Enables mipmapped images\n");
    log_info( "\n" );
    log_info( "Test names:\n" );
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}
