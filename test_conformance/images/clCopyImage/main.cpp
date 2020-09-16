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
#include "../harness/testHarness.h"

bool gDebugTrace;
bool gTestSmallImages;
bool gTestMaxImages;
bool gUseRamp;
bool gEnablePitch;
bool gTestMipmaps;
int gTypesToTest;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order gChannelOrderToUse = (cl_channel_order)-1;

extern int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, MethodsToTest testMethod );

#define MAX_ALLOWED_STD_DEVIATION_IN_MB        8.0

static void printUsage( const char *execName );

int test_1D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k1D );
}
int test_2D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2D );
}
int test_3D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k3D );
}
int test_1Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k1DArray );
}
int test_2Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2DArray );
}
int test_2Dto3D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2DTo3D );
}
int test_3Dto2D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k3DTo2D );
}
int test_2Darrayto2D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2DArrayTo2D );
}
int test_2Dto2Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2DTo2DArray );
}
int test_2Darrayto3D(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k2DArrayTo3D );
}
int test_3Dto2Darray(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    return test_image_set( device, context, queue, k3DTo2DArray );
}

test_definition test_list[] = {
    ADD_TEST( 1D ),
    ADD_TEST( 2D ),
    ADD_TEST( 3D ),
    ADD_TEST( 1Darray ),
    ADD_TEST( 2Darray ),
    ADD_TEST( 2Dto3D ),
    ADD_TEST( 3Dto2D ),
    ADD_TEST( 2Darrayto2D ),
    ADD_TEST( 2Dto2Darray ),
    ADD_TEST( 2Darrayto3D ),
    ADD_TEST( 3Dto2Darray ),
};

const int test_num = ARRAY_SIZE( test_list );

int main(int argc, const char *argv[])
{
    cl_channel_type chanType;
    cl_channel_order chanOrder;

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
        if( strcmp( argv[i], "test_mipmaps" ) == 0 )
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

    if( gTestSmallImages )
        log_info( "Note: Using small test images\n" );

    int ret = runTestHarness( argCount, argList, test_num, test_list, true, false, 0 );

    free(argList);
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
    for( int i = 0; i < test_num; i++ )
    {
        log_info( "\t%s\n", test_list[i].name );
    }
}
