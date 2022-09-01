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
#include "testBase.h"
#include "harness/errorHelpers.h"

#include <GL/glx.h>
#include <CL/cl_ext.h>

class X11GLEnvironment : public GLEnvironment
{
private:
    cl_device_id m_devices[64];
    cl_uint m_device_count;

public:
    X11GLEnvironment()
    {
        m_device_count = 0;
    }
    virtual int Init( int *argc, char **argv, int use_opencl_32 )
    {
         // Create a GLUT window to render into
        glutInit( argc, argv );
        glutInitWindowSize( 512, 512 );
        glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
        glutCreateWindow( "OpenCL <-> OpenGL Test" );
        glewInit();
        return 0;
    }

    virtual cl_context CreateCLContext( void )
    {
        GLXContext context = glXGetCurrentContext();
        Display *dpy = glXGetCurrentDisplay();

        cl_context_properties properties[] = {
            CL_GL_CONTEXT_KHR,  (cl_context_properties) context,
            CL_GLX_DISPLAY_KHR, (cl_context_properties) dpy,
            0
        };
        cl_int status;

        if (!context || !dpy) {
            print_error(CL_INVALID_CONTEXT, "No GL context bound");
            return 0;
        }

        return clCreateContext(properties, 1, m_devices, NULL, NULL, &status);
    }

    virtual int SupportsCLGLInterop( cl_device_type device_type )
    {
        int found_valid_device = 0;
        cl_platform_id platform;
        cl_device_id devices[64];
        cl_uint num_of_devices;
        int error;
        error = clGetPlatformIDs(1, &platform, NULL);
        if (error) {
            print_error(error, "clGetPlatformIDs failed");
            return -1;
        }
        error = clGetDeviceIDs(platform, device_type, 64, devices, &num_of_devices);
        // If this platform doesn't have any of the requested device_type (namely GPUs) then return 0
        if (error == CL_DEVICE_NOT_FOUND)
          return 0;
        if (error) {
            print_error(error, "clGetDeviceIDs failed");
            return -1;
        }

        for (int i=0; i<(int)num_of_devices; i++) {
            if (!is_extension_available(devices[i], "cl_khr_gl_sharing"))
            {
                log_info("Device %d of %d does not support required extension "
                         "cl_khr_gl_sharing.\n",
                         i + 1, num_of_devices);
            }
            else
            {
                log_info("Device %d of %d supports required extension "
                         "cl_khr_gl_sharing.\n",
                         i + 1, num_of_devices);
                found_valid_device = 1;
                m_devices[m_device_count++] = devices[i];
            }
        }
        return found_valid_device;
    }

    virtual ~X11GLEnvironment()
    {
    }
};

GLEnvironment * GLEnvironment::Instance( void )
{
    static X11GLEnvironment * env = NULL;
    if( env == NULL )
        env = new X11GLEnvironment();
    return env;
}
