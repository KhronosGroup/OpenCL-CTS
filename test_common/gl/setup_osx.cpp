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
#include "setup.h"
#include "harness/errorHelpers.h"
#include <OpenGL/CGLDevice.h>

class OSXGLEnvironment : public GLEnvironment
{
    public:
        OSXGLEnvironment()
        {
      mCGLContext = NULL;
        }

  virtual int Init( int *argc, char **argv, int use_opengl_32 )
        {
      if (!use_opengl_32) {

        // Create a GLUT window to render into
        glutInit( argc, argv );
        glutInitWindowSize( 512, 512 );
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow( "OpenCL <-> OpenGL Test" );
      }

      else {

        CGLPixelFormatAttribute attribs[] = {
          kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core,
          kCGLPFAAllowOfflineRenderers,
          kCGLPFANoRecovery,
          kCGLPFAAccelerated,
          kCGLPFADoubleBuffer,
          (CGLPixelFormatAttribute)0
        };

        CGLError err;
        CGLPixelFormatObj pix;
        GLint npix;
        err = CGLChoosePixelFormat (attribs, &pix, &npix);
        if(err != kCGLNoError)
          {
            log_error("Failed to choose pixel format\n");
            return -1;
          }
        err = CGLCreateContext(pix, NULL, &mCGLContext);
        if(err != kCGLNoError)
          {
            log_error("Failed to create GL context\n");
            return -1;
          }
        CGLSetCurrentContext(mCGLContext);
      }

            return 0;
        }

        virtual cl_context CreateCLContext( void )
    {
      int error;

      if( mCGLContext == NULL )
        mCGLContext = CGLGetCurrentContext();

      CGLShareGroupObj share_group = CGLGetShareGroup(mCGLContext);
      cl_context_properties properties[] = { CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)share_group, 0 };
      cl_context context = clCreateContext(properties, 0, 0, 0, 0, &error);
      if (error) {
        print_error(error, "clCreateContext failed");
        return NULL;
      }

      // Verify that all devices in the context support the required extension
      cl_device_id devices[64];
      size_t size_out;
      error = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &size_out);
      if (error) {
        print_error(error, "clGetContextInfo failed");
        return NULL;
      }

      for (int i=0; i<(int)(size_out/sizeof(cl_device_id)); i++) {
        if (!is_extension_available(devices[i], "cl_APPLE_gl_sharing")) {
          log_error("Device %d does not support required extension cl_APPLE_gl_sharing.\n", i);
          return NULL;
        }
      }
      return context;
    }

    virtual int SupportsCLGLInterop( cl_device_type device_type )
    {
      int found_valid_device = 0;
      cl_device_id devices[64];
      cl_uint num_of_devices;
      int error;
      error = clGetDeviceIDs(NULL, device_type, 64, devices, &num_of_devices);
      if (error) {
        print_error(error, "clGetDeviceIDs failed");
        return -1;
      }

      for (int i=0; i<(int)num_of_devices; i++) {
        if (!is_extension_available(devices[i], "cl_APPLE_gl_sharing")) {
          log_info("Device %d of %d does not support required extension cl_APPLE_gl_sharing.\n", i, num_of_devices);
        } else {
          log_info("Device %d of %d does support required extension cl_APPLE_gl_sharing.\n", i, num_of_devices);
          found_valid_device = 1;
        }
      }
            return found_valid_device;
    }

        virtual ~OSXGLEnvironment()
        {
            CGLDestroyContext( mCGLContext );
        }

        CGLContextObj mCGLContext;

};

GLEnvironment * GLEnvironment::Instance( void )
{
    static OSXGLEnvironment * env = NULL;
    if( env == NULL )
        env = new OSXGLEnvironment();
    return env;
}
