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
#ifndef _gl_headers_h
#define _gl_headers_h

#if defined( __APPLE__ )
    #include <OpenGL/OpenGL.h>
#if defined(CGL_VERSION_1_3)
    #include <OpenGL/gl3.h>
    #include <OpenGL/gl3ext.h>
#else
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
#endif
    #include <GLUT/glut.h>
#else
#ifdef _WIN32
    #include <windows.h>
#endif
#if defined( __ANDROID__ )
    #ifndef GL_GLEXT_PROTOTYPES
        #define GL_GLEXT_PROTOTYPES
    #endif
    #include <GLES/gl.h>
    #include <GLES/glext.h>
#else
    #include <GL/glew.h>
    #include <GL/gl.h>
#endif
#ifdef _WIN32
    #include <GL/glut.h>
#elif !defined(__ANDROID__)
    #include <GL/freeglut.h>
#endif

#endif

#ifdef _WIN32
    GLboolean gluCheckExtension(const GLubyte *extName, const GLubyte *extString);
    // No glutGetProcAddress in the standard glut v3.7.
    #define glutGetProcAddress(procName) wglGetProcAddress(procName)
#endif


#endif    // __gl_headers_h

