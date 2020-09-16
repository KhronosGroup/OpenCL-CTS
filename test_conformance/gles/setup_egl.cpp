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
#include <assert.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

#define EGLERR() \
    assert(eglGetError() == EGL_SUCCESS); \

#define MAX_DEVICES 10

class EGLGLEnvironment : public GLEnvironment
{
private:
    cl_platform_id _platform;
    EGLDisplay     _display;
    EGLContext     _context;
    EGLSurface     _surface;

public:
    EGLGLEnvironment()
        :_platform(NULL)
        ,_display(EGL_NO_DISPLAY)
        ,_context(NULL)
        ,_surface(EGL_NO_SURFACE)
    {
    }

    virtual int Init( int *argc, char **argv, int use_opengl_32 )
    {
        EGLint ConfigAttribs[] =
        {
            EGL_RED_SIZE,        8,
            EGL_GREEN_SIZE,     8,
            EGL_BLUE_SIZE,        8,
            EGL_ALPHA_SIZE,     8,
            EGL_DEPTH_SIZE,     16,
            EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
//            EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE,
            EGL_NONE
        };

        static const EGLint ContextAttribs[] =
        {
            EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL_NONE
        };

        EGLint conf_list[] = {
                     EGL_WIDTH,  512,
                     EGL_HEIGHT, 512,
                     EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA,
                     EGL_TEXTURE_TARGET, EGL_TEXTURE_2D,
                     EGL_NONE};

        EGLint        majorVersion;
        EGLint        minorVersion;
        EGLConfig    config;
        EGLint      numConfigs;

        EGLERR();
        _display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLERR();

        eglInitialize(_display, &majorVersion, &minorVersion);
        EGLERR();

        eglBindAPI(EGL_OPENGL_ES_API);
        EGLERR();

        eglChooseConfig(_display, ConfigAttribs, &config, 1, &numConfigs);
        EGLERR();

        _context = eglCreateContext(_display, config, NULL, ContextAttribs);
        EGLERR();

        _surface = eglCreatePbufferSurface(_display, config, conf_list);
        EGLERR();

        eglMakeCurrent(_display, _surface, _surface, _context);
        EGLERR();

        return 0;
    }

    virtual cl_context CreateCLContext( void )
    {
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) _platform,
            CL_GL_CONTEXT_KHR,   (cl_context_properties) _context,
            CL_EGL_DISPLAY_KHR,  (cl_context_properties) _display,
            0
        };
        cl_device_id devices[MAX_DEVICES];
        size_t dev_size;
        cl_int status;

        status = clGetGLContextInfoKHR(properties,
                                       CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                       sizeof(devices),
                                       devices,
                                       &dev_size);
        if (status != CL_SUCCESS) {
            print_error(status, "clGetGLContextInfoKHR failed");
            return NULL;
        }
        dev_size /= sizeof(cl_device_id);
        log_info("GL _context supports %d compute devices\n", dev_size);

        status = clGetGLContextInfoKHR(properties,
                                       CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                                       sizeof(devices),
                                       devices,
                                       &dev_size);
        if (status != CL_SUCCESS) {
            print_error(status, "clGetGLContextInfoKHR failed");
            return NULL;
        }

        if (!dev_size)
        {
            log_info("GL _context current device is not a CL device.\n");
            return NULL;
        }

        return clCreateContext(properties, 1, &devices[0], NULL, NULL, &status);
    }

    virtual int SupportsCLGLInterop( cl_device_type device_type )
    {
        cl_device_id devices[MAX_DEVICES];
        cl_uint num_of_devices;
        int interop_devices = 0;
        int error;

        error = clGetPlatformIDs(1, &_platform, NULL);
        if (error) {
            print_error(error, "clGetPlatformIDs failed");
            return -1;
        }

        error = clGetDeviceIDs(_platform, device_type, MAX_DEVICES, devices, &num_of_devices);
        if (error) {
            print_error(error, "clGetDeviceIDs failed");
            return -1;
        }

        // Check all devices, search for one that supports cl_khr_gl_sharing
        for (int i=0; i<(int)num_of_devices; i++) {
            if (!is_extension_available(devices[i], "cl_khr_gl_sharing"){
                log_info("Device %d of %d does not support required extension cl_khr_gl_sharing.\n", i+1, num_of_devices);
            } else {
                log_info("Device %d of %d supports required extension cl_khr_gl_sharing.\n", i+1, num_of_devices);
                interop_devices++;
            }
        }
        return interop_devices > 0;
    }

    // Change to cleanup egl environment properly when the test exit.
    // This change does not affect any functionality of the test it self
    virtual void terminate_egl_display()
    {
        if(_display != EGL_NO_DISPLAY)
        {
            eglMakeCurrent(_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            EGLERR();

            eglDestroyContext(_display, _context);
            EGLERR();
            _context = EGL_NO_CONTEXT;

            eglDestroySurface(_display, _surface);
            EGLERR();
            _surface = EGL_NO_SURFACE;

            eglTerminate(_display);
            EGLERR();
            _display = EGL_NO_DISPLAY;
        }
    }

    virtual ~EGLGLEnvironment()
    {
    }
};

GLEnvironment * GLEnvironment::Instance( void )
{
    return new EGLGLEnvironment();
}
