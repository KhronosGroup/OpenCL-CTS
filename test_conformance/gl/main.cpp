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
#include "harness/compat.h"

#include <stdio.h>
#include <string.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

#include "procs.h"
#include "gl/setup.h"
#include "harness/testHarness.h"
#include "harness/parseParameters.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

static cl_context        sCurrentContext = NULL;


#define TEST_FN_REDIRECT( fn ) ADD_TEST( redirect_##fn )
#define TEST_FN_REDIRECTOR( fn ) \
int test_redirect_##fn(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )    \
{ \
    int error; \
    clCommandQueueWrapper realQueue = clCreateCommandQueueWithProperties( sCurrentContext, device, 0, &error ); \
    test_error( error, "Unable to create command queue" );    \
    return test_##fn( device, sCurrentContext, realQueue, numElements ); \
}

// buffers:
TEST_FN_REDIRECTOR( buffers )
TEST_FN_REDIRECTOR( buffers_getinfo )

// 1D images:
TEST_FN_REDIRECTOR( images_read_1D )
TEST_FN_REDIRECTOR( images_write_1D )
TEST_FN_REDIRECTOR( images_1D_getinfo )

// 1D image arrays:
TEST_FN_REDIRECTOR( images_read_1Darray )
TEST_FN_REDIRECTOR( images_write_1Darray )
TEST_FN_REDIRECTOR( images_1Darray_getinfo )

// 2D images:
TEST_FN_REDIRECTOR( images_read_2D )
TEST_FN_REDIRECTOR( images_read_cube )
TEST_FN_REDIRECTOR( images_write )
TEST_FN_REDIRECTOR( images_write_cube )
TEST_FN_REDIRECTOR( images_2D_getinfo )
TEST_FN_REDIRECTOR( images_cube_getinfo )

// 2D image arrays:
TEST_FN_REDIRECTOR( images_read_2Darray )
TEST_FN_REDIRECTOR( images_write_2Darray )
TEST_FN_REDIRECTOR( images_2Darray_getinfo )

// 3D images:
TEST_FN_REDIRECTOR( images_read_3D )
TEST_FN_REDIRECTOR( images_write_3D )
TEST_FN_REDIRECTOR( images_3D_getinfo )

#ifdef GL_VERSION_3_2

TEST_FN_REDIRECTOR( images_read_texturebuffer )
TEST_FN_REDIRECTOR( images_write_texturebuffer )
TEST_FN_REDIRECTOR( images_texturebuffer_getinfo )

// depth textures
TEST_FN_REDIRECTOR( images_read_2D_depth )
TEST_FN_REDIRECTOR( images_write_2D_depth )
TEST_FN_REDIRECTOR( images_read_2Darray_depth )
TEST_FN_REDIRECTOR( images_write_2Darray_depth )

TEST_FN_REDIRECTOR( images_read_2D_multisample )
TEST_FN_REDIRECTOR( images_read_2Darray_multisample )
TEST_FN_REDIRECTOR( image_methods_depth )
TEST_FN_REDIRECTOR( image_methods_multisample )
#endif

// Renderbuffer-backed images:
TEST_FN_REDIRECTOR( renderbuffer_read )
TEST_FN_REDIRECTOR( renderbuffer_write )
TEST_FN_REDIRECTOR( renderbuffer_getinfo )

TEST_FN_REDIRECTOR( fence_sync )

test_definition test_list[] = {
    TEST_FN_REDIRECT( buffers ),
    TEST_FN_REDIRECT( buffers_getinfo ),

    TEST_FN_REDIRECT( images_read_1D ),
    TEST_FN_REDIRECT( images_write_1D ),
    TEST_FN_REDIRECT( images_1D_getinfo ),

    TEST_FN_REDIRECT( images_read_1Darray ),
    TEST_FN_REDIRECT( images_write_1Darray ),
    TEST_FN_REDIRECT( images_1Darray_getinfo ),

    TEST_FN_REDIRECT( images_read_2D ),
    TEST_FN_REDIRECT( images_write ),
    TEST_FN_REDIRECT( images_2D_getinfo ),

    TEST_FN_REDIRECT( images_read_cube ),
    TEST_FN_REDIRECT( images_write_cube ),
    TEST_FN_REDIRECT( images_cube_getinfo ),

    TEST_FN_REDIRECT( images_read_2Darray ),
    TEST_FN_REDIRECT( images_write_2Darray),
    TEST_FN_REDIRECT( images_2Darray_getinfo ),

    TEST_FN_REDIRECT( images_read_3D ),
    TEST_FN_REDIRECT( images_write_3D ),
    TEST_FN_REDIRECT( images_3D_getinfo ),

    TEST_FN_REDIRECT( renderbuffer_read ),
    TEST_FN_REDIRECT( renderbuffer_write ),
    TEST_FN_REDIRECT( renderbuffer_getinfo )
};

test_definition test_list32[] = {
    TEST_FN_REDIRECT( images_read_texturebuffer ),
    TEST_FN_REDIRECT( images_write_texturebuffer ),
    TEST_FN_REDIRECT( images_texturebuffer_getinfo ),

    TEST_FN_REDIRECT( fence_sync ),
    TEST_FN_REDIRECT( images_read_2D_depth ),
    TEST_FN_REDIRECT( images_write_2D_depth ),
    TEST_FN_REDIRECT( images_read_2Darray_depth ),
    TEST_FN_REDIRECT( images_write_2Darray_depth ),
    TEST_FN_REDIRECT( images_read_2D_multisample ),
    TEST_FN_REDIRECT( images_read_2Darray_multisample ),
    TEST_FN_REDIRECT( image_methods_depth ),
    TEST_FN_REDIRECT( image_methods_multisample )
};

const int test_num = ARRAY_SIZE( test_list );
const int test_num32 = ARRAY_SIZE( test_list32 );

int main(int argc, const char *argv[])
{
  gTestRounding = true;
  int error = 0;
  int numErrors = 0;

  test_start();
  argc = parseCustomParam(argc, argv);
  if (argc == -1)
  {
    return -1;
  }	

  cl_device_type requestedDeviceType = CL_DEVICE_TYPE_DEFAULT;

  /* Do we have a CPU/GPU specification? */
  if( argc > 1 )
  {
    if( strcmp( argv[ argc - 1 ], "gpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_GPU" ) == 0 )
    {
      requestedDeviceType = CL_DEVICE_TYPE_GPU;
      argc--;
  }
    else if( strcmp( argv[ argc - 1 ], "cpu" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_CPU" ) == 0 )
    {
      requestedDeviceType = CL_DEVICE_TYPE_CPU;
      argc--;
    }
    else if( strcmp( argv[ argc - 1 ], "accelerator" ) == 0 || strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_ACCELERATOR" ) == 0 )
    {
      requestedDeviceType = CL_DEVICE_TYPE_ACCELERATOR;
      argc--;
    }
    else if( strcmp( argv[ argc - 1 ], "CL_DEVICE_TYPE_DEFAULT" ) == 0 )
    {
      requestedDeviceType = CL_DEVICE_TYPE_DEFAULT;
      argc--;
    }
  }

    if( argc > 1 && strcmp( argv[ 1 ], "-list" ) == 0 )
    {
        log_info( "Available 2.x tests:\n" );
        for( int i = 0; i < test_num; i++ )
            log_info( "\t%s\n", test_list[i].name );

        log_info( "Available 3.2 tests:\n" );
        for( int i = 0; i < test_num32; i++ )
            log_info( "\t%s\n", test_list32[i].name );

    log_info( "Note: Any 3.2 test names must follow 2.1 test names on the command line.\n" );
    log_info( "Use environment variables to specify desired device.\n" );

        return 0;
    }

  // Check to see if any 2.x or 3.2 test names were specified on the command line.
  unsigned first_32_testname = 0;

  for (int j=1; (j<argc) && (!first_32_testname); ++j)
    for (int i = 0; i < test_num32; ++i)
      if (strcmp(test_list32[i].name, argv[j]) == 0) {
        first_32_testname = j;
        break;
      }

  // Create the environment for the test.
    GLEnvironment *glEnv = GLEnvironment::Instance();

  // Check if any devices of the requested type support CL/GL interop.
  int supported = glEnv->SupportsCLGLInterop( requestedDeviceType );
  if( supported == 0 ) {
    log_info("Test not run because GL-CL interop is not supported for any devices of the requested type.\n");
    return 0;
  } else if ( supported == -1 ) {
    log_error("Unable to setup the test or failed to determine if CL-GL interop is supported.\n");
    return -1;
  }

  // Initialize function pointers.
  error = init_clgl_ext();
  if (error < 0) {
    return error;
  }

  // OpenGL tests for non-3.2 ////////////////////////////////////////////////////////
  if ((argc == 1) || (first_32_testname != 1)) {

    // At least one device supports CL-GL interop, so init the test.
    if( glEnv->Init( &argc, (char **)argv, CL_FALSE ) ) {
      log_error("Failed to initialize the GL environment for this test.\n");
      return -1;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL )
      {
        log_error( "ERROR: Unable to obtain CL context from GL\n" );
        return -1;
      }

    size_t numDevices = 0;
    cl_device_id *deviceIDs;

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS )
      {
        print_error( error, "Unable to get device count from context" );
        return -1;
      }
    deviceIDs = (cl_device_id *)malloc(numDevices);
    if (deviceIDs == NULL) {
        print_error( error, "malloc failed" );
        return -1;
    }
    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, numDevices, deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      return -1;
    }

    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      return -1;
    }

    // Execute tests.
    int argc_ = (first_32_testname) ? first_32_testname : argc;

      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 2.x\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          return -1;
        }

        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_, argv, deviceIDs[i], test_num, test_list, true, 0, 1024 );
        if( error != 0 )
          break;
    }

    numErrors += error;

    // Clean-up.
      free(deviceIDs);
      clReleaseContext( sCurrentContext );
      //delete glEnv;
  }

  // OpenGL 3.2 tests. ////////////////////////////////////////////////////////
  if ((argc==1) || first_32_testname) {

    // At least one device supports CL-GL interop, so init the test.
    if( glEnv->Init( &argc, (char **)argv, CL_TRUE ) ) {
      log_error("Failed to initialize the GL environment for this test.\n");
      return -1;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL ) {
      log_error( "ERROR: Unable to obtain CL context from GL\n" );
      return -1;
    }

    size_t numDevices = 0;
    cl_device_id *deviceIDs;

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device count from context" );
      return -1;
    }
    deviceIDs = (cl_device_id *)malloc(numDevices);
    if (deviceIDs == NULL) {
        print_error( error, "malloc failed" );
        return -1;
    }
    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, numDevices, deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      return -1;
    }

    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      return -1;
    }

    int argc_ = (first_32_testname) ? 1 + (argc - first_32_testname) : argc;
    const char** argv_ = (first_32_testname) ? &argv[first_32_testname-1] : argv;

    // Execute the tests.
      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 3.2\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          return -1;
        }

        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_, argv_, deviceIDs[i], test_num32, test_list32, true, 0, 1024 );
        if( error != 0 )
          break;
    }

    numErrors += error;

    // Clean-up.
      free(deviceIDs);
      clReleaseContext( sCurrentContext );
      delete glEnv;

  }

  //All done.
  return numErrors;
}

