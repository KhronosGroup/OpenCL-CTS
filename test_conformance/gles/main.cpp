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
#include "gles/setup.h"
#include "harness/testHarness.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

static cl_context        sCurrentContext = NULL;


#define TEST_FN_REDIRECT( fn ) ADD_TEST( redirect_##fn )
#define TEST_FN_REDIRECTOR( fn ) \
int test_redirect_##fn(cl_device_id device, cl_context context, cl_command_queue queue, int numElements )    \
{ \
    int error; \
    clCommandQueueWrapper realQueue = clCreateCommandQueue( sCurrentContext, device, 0, &error ); \
    test_error( error, "Unable to create command queue" );    \
    return test_##fn( device, sCurrentContext, realQueue, numElements ); \
}

TEST_FN_REDIRECTOR( buffers )
TEST_FN_REDIRECTOR( buffers_getinfo )
TEST_FN_REDIRECTOR( images_read )
TEST_FN_REDIRECTOR( images_2D_getinfo )
TEST_FN_REDIRECTOR( images_read_cube )
TEST_FN_REDIRECTOR( images_cube_getinfo )
TEST_FN_REDIRECTOR( images_read_3D )
TEST_FN_REDIRECTOR( images_3D_getinfo )
TEST_FN_REDIRECTOR( images_write )
TEST_FN_REDIRECTOR( images_write_cube )
TEST_FN_REDIRECTOR( renderbuffer_read )
TEST_FN_REDIRECTOR( renderbuffer_write )
TEST_FN_REDIRECTOR( renderbuffer_getinfo )

#ifdef GL_ES_VERSION_3_0
TEST_FN_REDIRECTOR(fence_sync)
#endif

test_definition test_list[] = {
    TEST_FN_REDIRECT( buffers ),
    TEST_FN_REDIRECT( buffers_getinfo ),
    TEST_FN_REDIRECT( images_read ),
    TEST_FN_REDIRECT( images_2D_getinfo ),
    TEST_FN_REDIRECT( images_read_cube ),
    TEST_FN_REDIRECT( images_cube_getinfo ),
    TEST_FN_REDIRECT( images_read_3D ),
    TEST_FN_REDIRECT( images_3D_getinfo ),
    TEST_FN_REDIRECT( images_write ),
    TEST_FN_REDIRECT( images_write_cube ),
    TEST_FN_REDIRECT( renderbuffer_read ),
    TEST_FN_REDIRECT( renderbuffer_write ),
    TEST_FN_REDIRECT( renderbuffer_getinfo )
};

#ifdef GL_ES_VERSION_3_0
test_definition test_list32[] = {
    TEST_FN_REDIRECT( fence_sync )
};
#endif

const int test_num = ARRAY_SIZE( test_list );

#ifdef GL_ES_VERSION_3_0
const int test_num32 = ARRAY_SIZE( test_list32 );
#endif


int main(int argc, const char *argv[])
{
  int error = 0;
  cl_platform_id platform_id = NULL;
  /* To keep it simple, use a static allocation of 32 argv pointers.
     argc is not expected to go beyond 32 */
  const char* argv_tmp[32] = {0};
  int argc_tmp = 0;

    test_start();

  cl_device_type requestedDeviceType = CL_DEVICE_TYPE_DEFAULT;

    for(int z = 1; z < argc; ++z)
    {//for
    if(strcmp( argv[ z ], "-list" ) == 0 )
    {
        log_info( "Available 2.x tests:\n" );
        for( int i = 0; i < test_num; i++ )
            log_info( "\t%s\n", test_list[i].name );

#ifdef GL_ES_VERSION_3_0
        log_info( "Available 3.2 tests:\n" );
        for( int i = 0; i < test_num32; i++ )
            log_info( "\t%s\n", test_list32[i].name );
#endif

        log_info("Note: Any 3.2 test names must follow 2.1 test names on the "
                 "command line.");
        log_info("Use environment variables to specify desired device.");

        return 0;
    }

    /* support requested device type */
        if(!strcmp(argv[z], "CL_DEVICE_TYPE_GPU"))
        {
           printf("Requested device type is CL_DEVICE_TYPE_GPU\n");
           requestedDeviceType = CL_DEVICE_TYPE_GPU;
        }
        else
        if(!strcmp(argv[z], "CL_DEVICE_TYPE_CPU"))
        {
           printf("Requested device type is CL_DEVICE_TYPE_CPU\n");
           log_info("Invalid CL device type. GL tests can only run on a GPU device.\n");
           return 0;
        }
    }//for

  // Check to see if any 2.x or 3.2 test names were specified on the command line.
  unsigned first_32_testname = 0;

#ifdef GL_ES_VERSION_3_0
  for (int j=1; (j<argc) && (!first_32_testname); ++j)
    for (int i = 0; i < test_num32; ++i)
      if (strcmp(test_list32[i].name, argv[j]) == 0 ) {
        first_32_testname = j;
        break;
      }
#endif

  // Create the environment for the test.
    GLEnvironment *glEnv = GLEnvironment::Instance();

  // Check if any devices of the requested type support CL/GL interop.
  int supported = glEnv->SupportsCLGLInterop( requestedDeviceType );
  if( supported == 0 ) {
    log_info("Test not run because GL-CL interop is not supported for any devices of the requested type.\n");
    error = 0;
    goto cleanup;
  } else if ( supported == -1 ) {
    log_error("Failed to determine if CL-GL interop is supported.\n");
    error = -1;
    goto cleanup;
  }

  // OpenGL tests for non-3.2 ////////////////////////////////////////////////////////
  if ((argc == 1) || (first_32_testname != 1)) {

    // At least one device supports CL-GL interop, so init the test.
    if( glEnv->Init( &argc, (char **)argv, CL_FALSE ) ) {
      log_error("Failed to initialize the GL environment for this test.\n");
      error = -1;
      goto cleanup;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL )
      {
        log_error( "ERROR: Unable to obtain CL context from GL\n" );
        error = -1;
        goto cleanup;
      }

    size_t numDevices = 0;
    cl_device_id deviceIDs[ 16 ];

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS )
      {
        print_error( error, "Unable to get device count from context" );
        error = -1;
        goto cleanup;
      }
    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      error = -1;
      goto cleanup;
    }

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, sizeof( deviceIDs ), deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      error = -1;
      goto cleanup;
    }

    // Execute tests.
    int argc_ = (first_32_testname) ? first_32_testname : argc;

      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 2.x\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          error = -1;
          goto cleanup;
        }

        error = clGetDeviceInfo(deviceIDs[ i ],
                                CL_DEVICE_PLATFORM,
                                sizeof(platform_id),
                                &platform_id,
                                NULL);
        if(error)
        {
          goto cleanup;
        }

        error = init_clgl_ext(platform_id);
        if (error < 0)
        {
          goto cleanup;
        }

        /* parseAndCallCommandLineTests considers every command line argument
           as a test name. This results in the test failing because of considering
           args such as 'CL_DEVICE_TYPE_GPU' as test names unless
           the actual test name happens to be the first argument.
           Instead of changing the behaviour of parseAndCallCommandLineTests
           modify the arguments passed to it so as to not affect other tests.
        */
    int w = 1;
    argc_tmp= argc_;
    for(int k = 1; k < argc; k++)
    {
        if( (strcmp(argv[k], "full") == 0) ||
            (strcmp(argv[k], "CL_DEVICE_TYPE_CPU") == 0) ||
            (strcmp(argv[k], "CL_DEVICE_TYPE_GPU") == 0))
        {
            argc_tmp--;
            continue;
        }
        else
        {
            argv_tmp[w++] = argv[k];
        }
    }

        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_tmp, argv_tmp, deviceIDs[i], test_num, test_list, true, 0, 1024 );
        if( error != 0 )
          break;
    }

    // Clean-up.
      // We move this to a common cleanup step to make sure that things will be released properly before the test exit
      goto cleanup;
      // clReleaseContext( sCurrentContext );
      // delete glEnv;
  }

  // OpenGL 3.2 tests. ////////////////////////////////////////////////////////
  if ((argc==1) || first_32_testname) {

    // At least one device supports CL-GL interop, so init the test.
    if( glEnv->Init( &argc, (char **)argv, CL_TRUE ) ) {
      log_error("Failed to initialize the GL environment for this test.\n");
      error = -1;
      goto cleanup;
    }

    // Create a context to use and then grab a device (or devices) from it
    sCurrentContext = glEnv->CreateCLContext();
    if( sCurrentContext == NULL ) {
      log_error( "ERROR: Unable to obtain CL context from GL\n" );
      error = -1;
      goto cleanup;
    }

    size_t numDevices = 0;
    cl_device_id deviceIDs[ 16 ];

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, 0, NULL, &numDevices);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device count from context" );
      error = -1;
      goto cleanup;
    }
    numDevices /= sizeof(cl_device_id);

    if (numDevices < 1) {
      log_error("No devices found.\n");
      error = -1;
      goto cleanup;
    }

    error = clGetContextInfo( sCurrentContext, CL_CONTEXT_DEVICES, sizeof( deviceIDs ), deviceIDs, NULL);
    if( error != CL_SUCCESS ) {
      print_error( error, "Unable to get device list from context" );
      error = -1;
      goto cleanup;
    }

#ifdef GLES3
    int argc_ = (first_32_testname) ? 1 + (argc - first_32_testname) : argc;
    const char** argv_ = (first_32_testname) ? &argv[first_32_testname-1] : argv;
#endif

    // Execute the tests.
      for( size_t i = 0; i < numDevices; i++ ) {
        log_info( "\nTesting OpenGL 3.2\n" );
        if( printDeviceHeader( deviceIDs[ i ] ) != CL_SUCCESS ) {
          error = -1;
          goto cleanup;
        }
#ifndef GLES3
        log_info("Cannot test OpenGL 3.2! This test was built for OpenGL ES 2.0\n");
        error = -1;
        goto cleanup;
#else
        // Note: don't use the entire harness, because we have a different way of obtaining the device (via the context)
        error = parseAndCallCommandLineTests( argc_, argv_, deviceIDs[ i ], test_num32, test_list32, true, 0, 1024 );
        if( error != 0 )
          break;
#endif
    }

    // Converge on a common cleanup to make sure that things will be released properly before the test exit
    goto cleanup;
  }


// cleanup CL/GL/EGL environment properly when the test exit.
// This change does not affect any functionality of the test

// Intentional falling through
cleanup:

    // Always make sure that OpenCL context is released properly when the test exit
    if(sCurrentContext)
    {
        clReleaseContext( sCurrentContext );
        sCurrentContext = NULL;
    }

    // Cleanup EGL
    glEnv->terminate_egl_display();

    delete glEnv;

    return error;
}
