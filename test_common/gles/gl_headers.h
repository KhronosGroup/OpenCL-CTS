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

#define GL_GLEXT_PROTOTYPES 1

#include <EGL/egl.h>

#ifdef GLES3
#include <GLES3/gl3.h>
#else
#include <GLES2/gl2.h>
#define glTexImage3DOES glTexImage3D
#define glUnmapBufferOES glUnmapBuffer
#define glMapBufferRangeEXT glMapBufferRange
#endif

#include <GLES2/gl2ext.h>

// Some macros to minimize the changes in the tests from GL to GLES2
#define glGenRenderbuffersEXT        glGenRenderbuffers
#define glDeleteRenderbuffersEXT     glDeleteRenderbuffers
#define glBindRenderbufferEXT        glBindRenderbuffer
#define glRenderbufferStorageEXT     glRenderbufferStorage
#define glGetRenderbufferParameterivEXT glGetRenderbufferParameteriv
#define glCheckFramebufferStatusEXT  glCheckFramebufferStatus
#define glGenFramebuffersEXT         glGenFramebuffers
#define glDeleteFramebuffersEXT      glDeleteFramebuffers
#define glBindFramebufferEXT         glBindFramebuffer
#define glFramebufferRenderbufferEXT glFramebufferRenderbuffer

#ifndef GL_ES_VERSION_3_0
#define GL_RGBA32F GL_RGBA32F_EXT
#define GL_READ_ONLY GL_BUFFER_ACCESS_OES
#define GL_HALF_FLOAT_ARB GL_HALF_FLOAT_OES
#define GL_BGRA GL_BGRA_EXT
#else
#define GL_HALF_FLOAT_ARB GL_HALF_FLOAT
#endif

#define glutGetProcAddress           eglGetProcAddress

#define GL_FRAMEBUFFER_EXT           GL_FRAMEBUFFER
#define GL_FRAMEBUFFER_COMPLETE_EXT  GL_FRAMEBUFFER_COMPLETE
#define GL_RENDERBUFFER_INTERNAL_FORMAT_EXT GL_RENDERBUFFER_INTERNAL_FORMAT
#define GL_RENDERBUFFER_EXT          GL_RENDERBUFFER
#define GL_DEPTH_ATTACHMENT_EXT      GL_DEPTH_ATTACHMENT

#define GL_RGBA32F_ARB               GL_RGBA
#define GL_BGRA GL_BGRA_EXT

typedef unsigned short GLhalf;

GLboolean gluCheckExtension(const GLubyte *extName, const GLubyte *extString);

#endif    // __gl_headers_h

