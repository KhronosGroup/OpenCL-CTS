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
#define GL_GLEXT_PROTOTYPES

#include "setup.h"
#include "testBase.h"
#include "harness/errorHelpers.h"

#include <GL/gl.h>
#include <GL/glut.h>
#include <CL/cl_ext.h>

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    const cl_context_properties *properties,
    cl_gl_context_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret);

// Rename references to this dynamically linked function to avoid
// collision with static link version
#define clGetGLContextInfoKHR clGetGLContextInfoKHR_proc
static clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;

#define MAX_DEVICES 32

class WGLEnvironment : public GLEnvironment
{
private:
    cl_device_id m_devices[MAX_DEVICES];
    int m_device_count;
    cl_platform_id m_platform;
    bool m_is_glut_init;

public:
    WGLEnvironment()
    {
        m_device_count = 0;
        m_platform = 0;
        m_is_glut_init = false;
    }
    virtual int Init( int *argc, char **argv, int use_opengl_32 )
    {
        if (!m_is_glut_init)
        {
            // Create a GLUT window to render into
            glutInit( argc, argv );
            glutInitWindowSize( 512, 512 );
            glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE );
            glutCreateWindow( "OpenCL <-> OpenGL Test" );
            glewInit();
            m_is_glut_init = true;
        }
        return 0;
    }

    virtual cl_context CreateCLContext( void )
    {
        HGLRC hGLRC = wglGetCurrentContext();
        HDC hDC = wglGetCurrentDC();
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) m_platform,
            CL_GL_CONTEXT_KHR,   (cl_context_properties) hGLRC,
            CL_WGL_HDC_KHR,      (cl_context_properties) hDC,
            0
        };
        cl_device_id devices[MAX_DEVICES];
        size_t dev_size;
        cl_int status;

        if (!hGLRC || !hDC) {
            print_error(CL_INVALID_CONTEXT, "No GL context bound");
            return 0;
        }

        if (!clGetGLContextInfoKHR) {
            // As OpenCL for the platforms.  Warn if more than one platform found,
            // since this might not be the platform we want.  By default, we simply
            // use the first returned platform.

            cl_uint nplatforms;
            cl_platform_id platform;
            clGetPlatformIDs(0, NULL, &nplatforms);
            clGetPlatformIDs(1, &platform, NULL);

            if (nplatforms > 1) {
                log_info("clGetPlatformIDs returned multiple values.  This is not "
                    "an error, but might result in obtaining incorrect function "
                    "pointers if you do not want the first returned platform.\n");

                // Show them the platform name, in case it is a problem.

                size_t size;
                char *name;

                clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &size);
                name = (char*)malloc(size);
                clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, name, NULL);

                log_info("Using platform with name: %s \n", name);
                free(name);
            }

            clGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn) clGetExtensionFunctionAddressForPlatform(platform, "clGetGLContextInfoKHR");
            if (!clGetGLContextInfoKHR) {
                print_error(CL_INVALID_PLATFORM, "Failed to query proc address for clGetGLContextInfoKHR");
            }
        }

        status = clGetGLContextInfoKHR(properties,
                                       CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                       sizeof(devices),
                                       devices,
                                       &dev_size);
        if (status != CL_SUCCESS) {
            print_error(status, "clGetGLContextInfoKHR failed");
            return 0;
        }
        dev_size /= sizeof(cl_device_id);
        log_info("GL context supports %d compute devices\n", dev_size);

        status = clGetGLContextInfoKHR(properties,
                                       CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                       sizeof(devices),
                                       devices,
                                       &dev_size);
        if (status != CL_SUCCESS) {
            print_error(status, "clGetGLContextInfoKHR failed");
            return 0;
        }

        cl_device_id ctxDevice = m_devices[0];
        if (dev_size > 0) {
            log_info("GL context current device: 0x%x\n", devices[0]);
            for (int i = 0; i < m_device_count; i++) {
                if (m_devices[i] == devices[0]) {
                    ctxDevice = devices[0];
                    break;
                }
            }
        } else {
            log_info("GL context current device is not a CL device, using device %d.\n", ctxDevice);
        }

        return clCreateContext(properties, 1, &ctxDevice, NULL, NULL, &status);
    }

    virtual int SupportsCLGLInterop( cl_device_type device_type )
    {
        cl_device_id devices[MAX_DEVICES];
        cl_uint num_of_devices;
        int error;
        error = clGetPlatformIDs(1, &m_platform, NULL);
        if (error) {
            print_error(error, "clGetPlatformIDs failed");
            return -1;
        }
        error = clGetDeviceIDs(m_platform, device_type, MAX_DEVICES, devices, &num_of_devices);
        if (error) {
            print_error(error, "clGetDeviceIDs failed");
            return -1;
        }

        // Check all devices, search for one that supports cl_khr_gl_sharing
        for (int i=0; i<(int)num_of_devices; i++) {
            if (!is_extension_available(devices[i], "cl_khr_gl_sharing")) {
                log_info("Device %d of %d does not support required extension cl_khr_gl_sharing.\n", i+1, num_of_devices);
            } else {
                log_info("Device %d of %d supports required extension cl_khr_gl_sharing.\n", i+1, num_of_devices);
                m_devices[m_device_count++] = devices[i];
            }
        }
        return m_device_count > 0;
    }

    virtual ~WGLEnvironment()
    {
    }
};

GLEnvironment * GLEnvironment::Instance( void )
{
    static WGLEnvironment * env = NULL;
    if( env == NULL )
        env = new WGLEnvironment();
    return env;
}
