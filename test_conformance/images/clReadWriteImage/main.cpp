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

bool			gDebugTrace = false, gTestSmallImages = false, gTestMaxImages = false, gUseRamp = false, gTestRounding = false;
int				gTypesToTest = 0;
cl_channel_type gChannelTypeToUse = (cl_channel_type)-1;
bool			gEnablePitch = false;
cl_device_type	gDeviceType = CL_DEVICE_TYPE_DEFAULT;
cl_command_queue queue;
cl_context context;

#define MAX_ALLOWED_STD_DEVIATION_IN_MB		8.0

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
	log_info( "\n" );
	log_info( "\tdebug_trace - Enables additional debug info logging\n" );
	log_info( "\tsmall_images - Runs every format through a loop of widths 1-13 and heights 1-9, instead of random sizes\n" );
	log_info( "\tmax_images - Runs every format through a set of size combinations with the max values, max values - 1, and max values / 128\n" );
	log_info( "\trounding - Runs every format through a single image filled with every possible value for that image format, to verify rounding works properly\n" );
	log_info( "\tuse_pitches - Enables row and slice pitches\n" );
	log_info( "\tuse_ramp - Instead of random data, uses images filled with ramps (and 0xff on any padding pixels) to ease debugging\n" );
}


extern int test_image_set( cl_device_id device, cl_mem_object_type image_type );

int main(int argc, const char *argv[])
{
	cl_platform_id platform;
	cl_device_id device;
	cl_channel_type chanType;
	char str[ 128 ];
	bool randomize = false;
  int testMethods = 0;
	
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
		else if( strcmp( str, "use_pitches" ) == 0 )
			gEnablePitch = true;
		else if( strcmp( str, "use_ramps" ) == 0 )
			gUseRamp = true;
    
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
    
		else if( strcmp( str, "help" ) == 0 || strcmp( str, "?" ) == 0 )
		{
			printUsage( argv[ 0 ] );
			return -1;
		}
		else if( ( chanType = get_channel_type_from_name( str ) ) != (cl_channel_type)-1 )
			gChannelTypeToUse = chanType;
    else
    {
      log_error( "ERROR: Unknown argument %d: %s.  Exiting....\n", i, str );
      return -1;
    }
	}
  
  if (testMethods == 0)
    testMethods = k1D | k2D | k3D | k1DArray | k2DArray;
  
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
  unsigned int num_devices;
  error = clGetDeviceIDs(platform, gDeviceType, 0, NULL, &num_devices);
  if( error )
  {
    print_error( error, "Unable to get number of devices" );
    test_finish();
    return -1;
  }
	
  uint32_t gDeviceIndex = 0;
  const char* device_index_env = getenv("CL_DEVICE_INDEX");
  if (device_index_env) {
    if (device_index_env) {
      gDeviceIndex = atoi(device_index_env);
    }
		
    if (gDeviceIndex >= num_devices) {
      vlog("Specified CL_DEVICE_INDEX=%d out of range, using index 0.\n", gDeviceIndex);
      gDeviceIndex = 0;
    }
  }
  
  cl_device_id *gDeviceList = (cl_device_id *)malloc( num_devices * sizeof( cl_device_id ) );
  error = clGetDeviceIDs(platform, gDeviceType, num_devices, gDeviceList, NULL);
  if( error )
  {
    print_error( error, "Unable to get devices" );
    free( gDeviceList );
    test_finish();
    return -1;
  }
  
  device = gDeviceList[gDeviceIndex];
  free( gDeviceList );
  
  log_info( "Using " );
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
