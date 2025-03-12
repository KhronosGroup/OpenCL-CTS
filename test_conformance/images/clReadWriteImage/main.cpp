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
#include <string.h>
#include "../testBase.h"
#include "../harness/compat.h"

bool gDebugTrace;
bool gTestSmallImages;
bool gTestMaxImages;
bool gTestMipmaps;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
cl_channel_order gChannelOrderToUse = (cl_channel_order)-1;
bool            gEnablePitch = false;

static void printUsage( const char *execName );

extern int test_image_set( cl_device_id device, cl_context context, cl_command_queue queue, cl_mem_object_type image_type );

REGISTER_TEST(1D)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE1D );
}
REGISTER_TEST(2D)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE2D );
}
REGISTER_TEST(3D)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE3D );
}
REGISTER_TEST(1Darray)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE1D_ARRAY );
}
REGISTER_TEST(2Darray)
{
    return test_image_set( device, context, queue, CL_MEM_OBJECT_IMAGE2D_ARRAY );
}
REGISTER_TEST(1Dbuffer)
{
    return test_image_set(device, context, queue, CL_MEM_OBJECT_IMAGE1D_BUFFER);
}

int main(int argc, const char *argv[])
{
    cl_channel_type chanType;

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

        else if( strcmp( argv[i], "small_images" ) == 0 )
            gTestSmallImages = true;
        else if( strcmp( argv[i], "max_images" ) == 0 )
            gTestMaxImages = true;
        else if( strcmp( argv[i], "use_pitches" ) == 0 )
            gEnablePitch = true;
        else if( strcmp( argv[i], "test_mipmaps") == 0 ) {
            gTestMipmaps = true;
            // Don't test pitches with mipmaps right now.
            gEnablePitch = false;
        }

        else if( strcmp( argv[i], "--help" ) == 0 || strcmp( argv[i], "-h" ) == 0 )
        {
            printUsage( argv[ 0 ] );
            return -1;
        }
        else if( ( chanType = get_channel_type_from_name( argv[i] ) ) != (cl_channel_type)-1 )
            gChannelTypeToUse = chanType;
        else
        {
            argList[argCount] = argv[i];
            argCount++;
        }
    }

    if( gTestSmallImages )
        log_info( "Note: Using small test images\n" );

    int ret = runTestHarnessWithCheck(
        argCount, argList, test_registry::getInstance().num_tests(),
        test_registry::getInstance().definitions(), false, 0,
        verifyImageSupport);

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
    log_info( "\tdebug_trace - Enables additional debug info logging\n" );
    log_info( "\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
    log_info( "\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
    log_info( "\tuse_pitches - Enables row and slice pitches\n" );
    log_info( "\ttest_mipmaps - Test mipmapped images\n" );
    log_info( "\trandomize - Uses random seed\n" );
    log_info( "\n" );
    log_info( "Test names:\n" );
    for (size_t i = 0; i < test_registry::getInstance().num_tests(); i++)
    {
        log_info("\t%s\n", test_registry::getInstance().definitions()[i].name);
    }
}
