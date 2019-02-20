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

#if !defined (__APPLE__)
#include <CL/cl.h>
#endif

#include "procs.h"
#include "../../test_common/gl/setup.h"
#include "../../test_common/harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

static cl_context        sCurrentContext = NULL;


#define TEST_FN_REDIRECT( fn )    redirect_##fn
#define TEST_FN_REDIRECTOR( fn ) \
int redirect_##fn(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )    \
{ \
    int error; \
    clCommandQueueWrapper realQueue = clCreateCommandQueue( sCurrentContext, device, 0, &error ); \
    test_error( error, "Unable to create command queue" );    \
    return fn( device, sCurrentContext, realQueue, numElements ); \
}

// buffers:
TEST_FN_REDIRECTOR( test_buffers )
TEST_FN_REDIRECTOR( test_buffers_getinfo )

// 1D images:
TEST_FN_REDIRECTOR( test_images_read_1D )
TEST_FN_REDIRECTOR( test_images_write_1D )
TEST_FN_REDIRECTOR( test_images_1D_getinfo )

// 1D image arrays:
TEST_FN_REDIRECTOR( test_images_read_1Darray )
TEST_FN_REDIRECTOR( test_images_write_1Darray )
TEST_FN_REDIRECTOR( test_images_1Darray_getinfo )

// 2D images:
TEST_FN_REDIRECTOR( test_images_read_2D )
TEST_FN_REDIRECTOR( test_images_read_cube )
TEST_FN_REDIRECTOR( test_images_write )
TEST_FN_REDIRECTOR( test_images_write_cube )
TEST_FN_REDIRECTOR( test_images_2D_getinfo )
TEST_FN_REDIRECTOR( test_images_cube_getinfo )

// 2D image arrays:
TEST_FN_REDIRECTOR( test_images_read_2Darray )
TEST_FN_REDIRECTOR( test_images_write_2Darray )
TEST_FN_REDIRECTOR( test_images_2Darray_getinfo )

// 3D images:
TEST_FN_REDIRECTOR( test_images_read_3D )
TEST_FN_REDIRECTOR( test_images_write_3D )
TEST_FN_REDIRECTOR( test_images_3D_getinfo )

// Renderbuffer-backed images:
TEST_FN_REDIRECTOR( test_renderbuffer_read )
TEST_FN_REDIRECTOR( test_renderbuffer_write )
TEST_FN_REDIRECTOR( test_renderbuffer_getinfo )

TEST_FN_REDIRECTOR( test_fence_sync )

basefn    basefn_list[] = {
    TEST_FN_REDIRECT( test_buffers ),
  TEST_FN_REDIRECT( test_buffers_getinfo ),

  TEST_FN_REDIRECT( test_images_read_1D ),
  TEST_FN_REDIRECT( test_images_write_1D ),
  TEST_FN_REDIRECT( test_images_1D_getinfo ),

  TEST_FN_REDIRECT( test_images_read_1Darray ),
  TEST_FN_REDIRECT( test_images_write_1Darray ),
  TEST_FN_REDIRECT( test_images_1Darray_getinfo ),

    TEST_FN_REDIRECT( test_images_read_2D ),
  TEST_FN_REDIRECT( test_images_write ),
  TEST_FN_REDIRECT( test_images_2D_getinfo ),

    TEST_FN_REDIRECT( test_images_read_cube ),
  TEST_FN_REDIRECT( test_images_write_cube ),
  TEST_FN_REDIRECT( test_images_cube_getinfo ),

  TEST_FN_REDIRECT( test_images_read_2Darray ),
  TEST_FN_REDIRECT( test_images_write_2Darray),
  TEST_FN_REDIRECT( test_images_2Darray_getinfo ),

    TEST_FN_REDIRECT( test_images_read_3D ),
  TEST_FN_REDIRECT( test_images_write_3D ),
  TEST_FN_REDIRECT( test_images_3D_getinfo ),

    TEST_FN_REDIRECT( test_renderbuffer_read ),
     TEST_FN_REDIRECT( test_renderbuffer_write ),
  TEST_FN_REDIRECT( test_renderbuffer_getinfo )
};

basefn    basefn_list32[] = {
    TEST_FN_REDIRECT( test_fence_sync )
};

const char    *basefn_names[] = {
    "buffers",
  "buffers_getinfo",

  "images_read_1D",
  "images_write_1D",
  "images_1D_getinfo",

  "images_read_1Darray",
  "images_write_1Darray",
  "images_1Darray_getinfo",

    "images_read", /* 2D */
  "images_write",
  "images_2D_getinfo",

     "images_read_cube",
  "images_write_cube",
  "images_cube_getinfo",

  "images_read_2Darray",
  "images_write_2Darray",
  "images_2Darray_getinfo",

    "images_read_3D",
  "images_write_3D",
  "images_3D_getinfo",

    "renderbuffer_read",
    "renderbuffer_write",
  "renderbuffer_getinfo",

    "all"
};

const char    *basefn_names32[] = {
    "fence_sync",
  "all"
};

ct_assert((sizeof(basefn_names) / sizeof(basefn_names[0]) - 1) == (sizeof(basefn_list) / sizeof(basefn_list[0])));

int    num_fns = sizeof(basefn_names) / sizeof(char *);
int num_fns32 = sizeof(basefn_names32) / sizeof(char *);

cl_device_type gDeviceType = CL_DEVICE_TYPE_DEFAULT;
bool gTestRounding = true;

int main(int argc, const char *argv[])
{
  int error = 0;

    test_start();

  cl_device_type requestedDeviceType = CL_DEVICE_TYPE_GPU;
  checkDeviceTypeOverride(&requestedDeviceType);
  if (requestedDeviceType != CL_DEVICE_TYPE_GPU) {
    log_info("GL tests can only run on a GPU device.\n");
    test_finish();
    return 0;
  }
  gDeviceType = CL_DEVICE_TYPE_GPU;

    if( argc > 1 && strcmp( argv[ 1 ], "-list" ) == 0 )
    {
        log_info( "Available 2.x tests:\n" );
        for( int i = 0; i < num_fns - 1; i++ )
            log_info( "\t%s\n", basefn_names[ i ] );

        log_info( "Available 3.2 tests:\n" );
        for( int i = 0; i < num_fns32 - 1; i++ )
            log_info( "\t%s\n", basefn_names32[ i ] );

    log_info( "Note: Any 3.2 test names must follow 2.1 test names on the command line.\n" );
    log_info( "Use environment variables to specify desired device.\n" );

        test_finish();
        return 0;
    }

  // Check to see if any 2.x or 3.2 test names were specified on the command line.
  unsigned first_32_testname = 0;

  for (int j=1; (j<argc) && (!first_32_testname); ++j)
    for (int i=0;i<num_fns32-1;++i)
      if (strcmp(basefn_names32[i],argv[j])==0) {
        first_32_testname = j;
        break;
      }

  // Create the environment for the test.
    GLEnvironment *glEnv = GLEnvironment::Instance();

  // Check if any devices of the requested type support CL/GL interop.
  int supported = glEnv->SupportsCLGLInterop( requestedDeviceType );
  if( supported == 0 ) {
    log_info("Test not run because GL-CL interop is not supported for any devices of the requested type.\n");
    test_finish();
    return 0;
  } else if ( supported == -1 ) {
    log_error("Unable to setup the test or failed to determine if CL-GL interop is supported.\n");
    test_finish();
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
      test_finish();
      return -1;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL )
      {
        log_error( "ERROR: Unable to obtain CL context from GL\n" );
        test_finish();
        return -1;
      }

    size_t numDevices = 0;
    cl_device_id *deviceIDs;

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS )
      {
        print_error( error, "Unable to get device count from context" );
        test_finish();
        return -1;
      }
    deviceIDs = (cl_device_id *)malloc(numDevices);
    if (deviceIDs == NULL) {
        print_error( error, "malloc failed" );
        test_finish();
        return -1;
    }
    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, numDevices, deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      test_finish();
      return -1;
    }

    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      test_finish();
      return -1;
    }

    // Execute tests.
    int argc_ = (first_32_testname) ? first_32_testname : argc;

      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 2.x\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          test_finish();
          return -1;
        }

        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_, argv, deviceIDs[ i ], num_fns, basefn_list, basefn_names, true, 0, 1024 );
        if( error != 0 )
          break;
    }

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
      test_finish();
      return -1;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL ) {
      log_error( "ERROR: Unable to obtain CL context from GL\n" );
      test_finish();
      return -1;
    }

    size_t numDevices = 0;
    cl_device_id *deviceIDs;

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device count from context" );
      test_finish();
      return -1;
    }
    deviceIDs = (cl_device_id *)malloc(numDevices);
    if (deviceIDs == NULL) {
        print_error( error, "malloc failed" );
        test_finish();
        return -1;
    }
    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, numDevices, deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      test_finish();
      return -1;
    }

    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      test_finish();
      return -1;
    }

    int argc_ = (first_32_testname) ? 1 + (argc - first_32_testname) : argc;
    const char** argv_ = (first_32_testname) ? &argv[first_32_testname-1] : argv;

    // Execute the tests.
      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 3.2\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          test_finish();
          return -1;
        }

        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_, argv_, deviceIDs[ i ], num_fns32, basefn_list32, basefn_names32, true, 0, 1024 );
        if( error != 0 )
          break;
    }

    // Clean-up.
      free(deviceIDs);
      clReleaseContext( sCurrentContext );
      delete glEnv;

  }

  //All done.
  return error;
}


