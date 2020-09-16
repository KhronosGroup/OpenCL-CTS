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

#if defined(__PPC__)
// Global varaiable used to hold the FPU control register state. The FPSCR register can not
// be used because not all Power implementations retain or observed the NI (non-IEEE
// mode) bit.
__thread fpu_control_t fpu_control = 0;
#endif

bool                gTestReadWrite;
bool                gDebugTrace;
bool                gTestMaxImages;
bool                gTestSmallImages;
int                 gTypesToTest;
cl_channel_type     gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order    gChannelOrderToUse = (cl_channel_order)-1;
bool                gEnablePitch = false;
bool                gDeviceLt20 = false;

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

static void printUsage( const char *execName );

extern int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type imageType );

int test_1D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE1D ) +
           test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE1D_BUFFER );
}
int test_2D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE2D );
}
int test_3D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE3D );
}
int test_1Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE1D_ARRAY );
}
int test_2Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE2D_ARRAY );
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
    for ( int i = 1; i < argc; i++ )
    {
        if ( strcmp( argv[i], "debug_trace" ) == 0 )
            gDebugTrace = true;
        else if ( strcmp( argv[i], "read_write" ) == 0 )
            gTestReadWrite = true;
        else if ( strcmp( argv[i], "small_images" ) == 0 )
            gTestSmallImages = true;
        else if ( strcmp( argv[i], "max_images" ) == 0 )
            gTestMaxImages = true;
        else if ( strcmp( argv[i], "use_pitches" ) == 0 )
            gEnablePitch = true;

        else if ( strcmp( argv[i], "int" ) == 0 )
            gTypesToTest |= kTestInt;
        else if ( strcmp( argv[i], "uint" ) == 0 )
            gTypesToTest |= kTestUInt;
        else if ( strcmp( argv[i], "float" ) == 0 )
            gTypesToTest |= kTestFloat;

        else if ( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 )
        {
            printUsage( argv[ 0 ] );
            return -1;
        }

        else if ( ( chanType = get_channel_type_from_name( argv[i] ) ) != (cl_channel_type)-1 )
            gChannelTypeToUse = chanType;

        else if ( ( chanOrder = get_channel_order_from_name( argv[i] ) ) != (cl_channel_order)-1 )
            gChannelOrderToUse = chanOrder;
        else
        {
            argList[argCount] = argv[i];
            argCount++;
        }
    }

    if ( gTypesToTest == 0 )
        gTypesToTest = kTestAllTypes;

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

    int ret = runTestHarness( argCount, argList, test_num, test_list, true, false, 0 );

    // Restore FP state before leaving
    RestoreFPState(&oldMode);

    free(argList);
    return ret;
}

static void printUsage( const char *execName )
{
    const char *p = strrchr( execName, '/' );
    if ( p != NULL )
        execName = p + 1;

    log_info( "Usage: %s [options] [test_names]\n", execName );
    log_info( "Options:\n" );
    log_info( "\n" );
    log_info( "\tThe following flags specify the types to test. They can be combined; if none are specified, all are tested:\n" );
    log_info( "\t\tint - Test integer I/O (read_imagei)\n" );
    log_info( "\t\tuint - Test unsigned integer I/O (read_imageui)\n" );
    log_info( "\t\tfloat - Test float I/O (read_imagef)\n" );
    log_info( "\n" );
    log_info( "You may also use appropriate CL_ channel type and ordering constants.\n" );
    log_info( "\n" );
    log_info( "\tThe following modify the types of images tested:\n" );
    log_info( "\t\read_write - Runs the tests with read_write images which allow a kernel do both read and write to the same image \n" );
    log_info( "\t\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\t\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\n" );
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\tuse_pitches - Enables row and slice pitches\n" );
    log_info( "\n" );
    log_info( "Test names:\n" );
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}
