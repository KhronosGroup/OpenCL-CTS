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
    bool m_glut_init;

    cl_platform_id m_platform;
    GLXContext m_context;
    Display *m_dpy;

public:
    X11GLEnvironment()
    {
        m_device_count = 0;
        m_glut_init = false;
        m_platform = 0;
        m_context = 0;
        m_dpy = nullptr;
    }

    int Init(int *argc, char **argv, int use_opencl_32) override
    {
         // Create a GLUT window to render into
         if (!m_glut_init)
         {
             glutInit(argc, argv);
             glutInitWindowSize(512, 512);
             glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
             glutCreateWindow("OpenCL <-> OpenGL Test");
             glewInit();
             m_glut_init = true;
         }
        return 0;
    }

    cl_context CreateCLContext(void) override
    {
        m_context = glXGetCurrentContext();
        m_dpy = glXGetCurrentDisplay();

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)m_platform,
            CL_GL_CONTEXT_KHR,
            (cl_context_properties)m_context,
            CL_GLX_DISPLAY_KHR,
            (cl_context_properties)m_dpy,
            0
        };
        cl_int status;

        if (!m_context || !m_dpy)
        {
            print_error(CL_INVALID_CONTEXT, "No GL context bound");
            return 0;
        }

        return clCreateContext(properties, 1, m_devices, NULL, NULL, &status);
    }

    int SupportsCLGLInterop(cl_device_type device_type) override
    {
        int found_valid_device = 0;
        cl_device_id devices[64];
        cl_uint num_of_devices;
        int error;
        error = clGetPlatformIDs(1, &m_platform, NULL);
        if (error) {
            print_error(error, "clGetPlatformIDs failed");
            return -1;
        }
        error = clGetDeviceIDs(m_platform, device_type, 64, devices,
                               &num_of_devices);
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
