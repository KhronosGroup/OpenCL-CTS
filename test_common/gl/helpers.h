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
#ifndef _helpers_h
#define _helpers_h

#include "../harness/compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#include "gl_headers.h"
#include <CL/cl_gl.h>
#else
#include "gl_headers.h"
#endif

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

typedef cl_mem
(CL_API_CALL *clCreateFromGLBuffer_fn)(cl_context     context,
                          cl_mem_flags   flags,
                          GLuint         bufobj,
                          int *          errcode_ret);

typedef cl_mem
(CL_API_CALL *clCreateFromGLTexture_fn)(cl_context       context ,
                        cl_mem_flags     flags ,
                        GLenum           target ,
                        GLint            miplevel ,
                        GLuint           texture ,
                        cl_int *         errcode_ret) ;

typedef cl_mem
(CL_API_CALL *clCreateFromGLTexture2D_fn)(cl_context       context ,
                        cl_mem_flags     flags ,
                        GLenum           target ,
                        GLint            miplevel ,
                        GLuint           texture ,
                        cl_int *         errcode_ret) ;

typedef cl_mem
(CL_API_CALL *clCreateFromGLTexture3D_fn)(cl_context       context ,
                        cl_mem_flags     flags ,
                        GLenum           target ,
                        GLint            miplevel ,
                        GLuint           texture ,
                        cl_int *         errcode_ret) ;

typedef cl_mem
(CL_API_CALL *clCreateFromGLRenderbuffer_fn)(cl_context    context ,
                           cl_mem_flags  flags ,
                           GLuint        renderbuffer ,
                           cl_int *      errcode_ret) ;

typedef cl_int
(CL_API_CALL *clGetGLObjectInfo_fn)(cl_mem                 memobj ,
                  cl_gl_object_type *    gl_object_type ,
                  GLuint *               gl_object_name) ;

typedef cl_int
(CL_API_CALL *clGetGLTextureInfo_fn)(cl_mem                memobj ,
                   cl_gl_texture_info    param_name ,
                   size_t                param_value_size ,
                   void *                param_value ,
                   size_t *              param_value_size_ret) ;

typedef cl_int
(CL_API_CALL *clEnqueueAcquireGLObjects_fn)(cl_command_queue       command_queue ,
                          cl_uint                num_objects ,
                          const cl_mem *         mem_objects ,
                          cl_uint                num_events_in_wait_list ,
                          const cl_event *       event_wait_list ,
                                cl_event *             event) ;

typedef cl_int
(CL_API_CALL *clEnqueueReleaseGLObjects_fn)(cl_command_queue       command_queue ,
                          cl_uint                num_objects ,
                          const cl_mem *         mem_objects ,
                          cl_uint                num_events_in_wait_list ,
                          const cl_event *       event_wait_list ,
                                cl_event *             event) ;


extern clCreateFromGLBuffer_fn clCreateFromGLBuffer_ptr;
extern clCreateFromGLTexture_fn clCreateFromGLTexture_ptr;
extern clCreateFromGLTexture2D_fn clCreateFromGLTexture2D_ptr;
extern clCreateFromGLTexture3D_fn clCreateFromGLTexture3D_ptr;
extern clCreateFromGLRenderbuffer_fn clCreateFromGLRenderbuffer_ptr;
extern clGetGLObjectInfo_fn clGetGLObjectInfo_ptr;
extern clGetGLTextureInfo_fn clGetGLTextureInfo_ptr;
extern clEnqueueAcquireGLObjects_fn clEnqueueAcquireGLObjects_ptr;
extern clEnqueueReleaseGLObjects_fn clEnqueueReleaseGLObjects_ptr;


class glBufferWrapper
{
    public:
        glBufferWrapper() { mBuffer = 0; }
        glBufferWrapper( GLuint b ) { mBuffer = b; }
        ~glBufferWrapper() { if( mBuffer != 0 ) glDeleteBuffers( 1, &mBuffer ); }

        glBufferWrapper & operator=( const GLuint &rhs ) { mBuffer = rhs; return *this; }
        operator GLuint() { return mBuffer; }
        operator GLuint *() { return &mBuffer; }

        GLuint * operator&() { return &mBuffer; }

        bool operator==( GLuint rhs ) { return mBuffer == rhs; }

    protected:

        GLuint mBuffer;
};

class glTextureWrapper
{
    public:
        glTextureWrapper() { mHandle = 0; }
        glTextureWrapper( GLuint b ) { mHandle = b; }
        ~glTextureWrapper() {
         if( mHandle != 0 ) glDeleteTextures( 1, &mHandle );
        }

        glTextureWrapper & operator=( const GLuint &rhs ) { mHandle = rhs; return *this; }
        operator GLuint() { return mHandle; }
        operator GLuint *() { return &mHandle; }

        GLuint * operator&() { return &mHandle; }

        bool operator==( GLuint rhs ) { return mHandle == rhs; }

    protected:

    // The texture handle.
        GLuint mHandle;
};

class glRenderbufferWrapper
{
    public:
        glRenderbufferWrapper() { mBuffer = 0; }
        glRenderbufferWrapper( GLuint b ) { mBuffer = b; }
        ~glRenderbufferWrapper() { if( mBuffer != 0 ) glDeleteRenderbuffersEXT( 1, &mBuffer ); }

        glRenderbufferWrapper & operator=( const GLuint &rhs ) { mBuffer = rhs; return *this; }
        operator GLuint() { return mBuffer; }
        operator GLuint *() { return &mBuffer; }

        GLuint * operator&() { return &mBuffer; }

        bool operator==( GLuint rhs ) { return mBuffer == rhs; }

    protected:

        GLuint mBuffer;
};

class glFramebufferWrapper
{
    public:
        glFramebufferWrapper() { mBuffer = 0; }
        glFramebufferWrapper( GLuint b ) { mBuffer = b; }
        ~glFramebufferWrapper() { if( mBuffer != 0 ) glDeleteFramebuffersEXT( 1, &mBuffer ); }

        glFramebufferWrapper & operator=( const GLuint &rhs ) { mBuffer = rhs; return *this; }
        operator GLuint() { return mBuffer; }
        operator GLuint *() { return &mBuffer; }

        GLuint * operator&() { return &mBuffer; }

        bool operator==( GLuint rhs ) { return mBuffer == rhs; }

    protected:

        GLuint mBuffer;
};

class glVertexArraysWrapper
{
public:
  glVertexArraysWrapper() { mBuffer = 0; }
  glVertexArraysWrapper( GLuint b ) { mBuffer = b; }
  ~glVertexArraysWrapper() { if( mBuffer != 0 ) glDeleteVertexArrays( 1, &mBuffer ); }

  glVertexArraysWrapper & operator=( const GLuint &rhs ) { mBuffer = rhs; return *this; }
  operator GLuint() { return mBuffer; }
  operator GLuint *() { return &mBuffer; }

  GLuint * operator&() { return &mBuffer; }

  bool operator==( GLuint rhs ) { return mBuffer == rhs; }

protected:

  GLuint mBuffer;
};

class glProgramWrapper
{
public:
  glProgramWrapper() { mProgram = 0; }
  glProgramWrapper( GLuint b ) { mProgram = b; }
  ~glProgramWrapper() { if( mProgram != 0 ) glDeleteProgram( mProgram ); }

  glProgramWrapper & operator=( const GLuint &rhs ) { mProgram = rhs; return *this; }
  operator GLuint() { return mProgram; }
  operator GLuint *() { return &mProgram; }

  GLuint * operator&() { return &mProgram; }

  bool operator==( GLuint rhs ) { return mProgram == rhs; }

protected:

  GLuint mProgram;
};

class glShaderWrapper
{
public:
  glShaderWrapper() { mShader = 0; }
  glShaderWrapper( GLuint b ) { mShader = b; }
  ~glShaderWrapper() { if( mShader != 0 ) glDeleteShader( mShader ); }

  glShaderWrapper & operator=( const GLuint &rhs ) { mShader = rhs; return *this; }
  operator GLuint() { return mShader; }
  operator GLuint *() { return &mShader; }

  GLuint * operator&() { return &mShader; }

  bool operator==( GLuint rhs ) { return mShader == rhs; }

protected:

  GLuint mShader;
};

// Helper functions (defined in helpers.cpp)

extern void * CreateGLTexture1DArray( size_t width, size_t length,
  GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
  ExplicitType type, GLuint *outTextureID, int *outError,
  bool allocateMem, MTdata d);

extern void * CreateGLTexture2DArray( size_t width, size_t height, size_t length,
  GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
  ExplicitType type, GLuint *outTextureID, int *outError,
  bool allocateMem, MTdata d);

extern void * CreateGLTextureBuffer( size_t width,
  GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
  ExplicitType type, GLuint *outTex, GLuint *outBuf, int *outError,
  bool allocateMem, MTdata d);

extern void * CreateGLTexture1D(size_t width,
                                GLenum target, GLenum glFormat,
                                GLenum internalFormat, GLenum glType,
                                ExplicitType type, GLuint *outTextureID,
                                int *outError, bool allocateMem, MTdata d );

extern void * CreateGLTexture2D( size_t width, size_t height,
                               GLenum target, GLenum glFormat,
                               GLenum internalFormat, GLenum glType,
                               ExplicitType type, GLuint *outTextureID,
                               int *outError, bool allocateMem, MTdata d );


extern void * CreateGLTexture3D( size_t width, size_t height, size_t depth,
                                 GLenum target, GLenum glFormat,
                                 GLenum internalFormat, GLenum glType,
                                 ExplicitType type, GLuint *outTextureID,
                                 int *outError, MTdata d, bool allocateMem = true );

#ifdef GL_VERSION_3_2
extern void * CreateGLTexture2DMultisample( size_t width, size_t height, size_t samples,
                                           GLenum target, GLenum glFormat,
                                           GLenum internalFormat, GLenum glType,
                                           ExplicitType type, GLuint *outTextureID,
                                           int *outError, bool allocateMem, MTdata d,
                                           bool fixedSampleLocations );

extern void * CreateGLTexture2DArrayMultisample( size_t width, size_t height,
                                                size_t length, size_t samples,
                                                GLenum target, GLenum glFormat,
                                                GLenum internalFormat, GLenum glType,
                                                ExplicitType type, GLuint *outTextureID,
                                                int *outError, bool allocateMem, MTdata d,
                                                bool fixedSampleLocations );
#endif

extern void * ReadGLTexture( GLenum glTarget, GLuint glTexture, GLuint glBuf, GLint width,
                             GLenum glFormat, GLenum glInternalFormat,
                             GLenum glType, ExplicitType typeToReadAs,
                             size_t outWidth, size_t outHeight );

extern int CreateGLRenderbufferRaw( GLsizei width, GLsizei height,
                                   GLenum target, GLenum glFormat,
                                   GLenum internalFormat, GLenum glType,
                                   GLuint *outFramebuffer,
                                   GLuint *outRenderbuffer );

extern void * CreateGLRenderbuffer( GLsizei width, GLsizei height,
                                    GLenum target, GLenum glFormat,
                                    GLenum internalFormat, GLenum glType,
                                    ExplicitType type,
                                    GLuint *outFramebuffer,
                                    GLuint *outRenderbuffer,
                                    int *outError, MTdata d, bool allocateMem );

extern void * ReadGLRenderbuffer( GLuint glFramebuffer, GLuint glRenderbuffer,
                                  GLenum attachment, GLenum glFormat,
                                  GLenum glInternalFormat, GLenum glType,
                                  ExplicitType typeToReadAs,
                                  size_t outWidth, size_t outHeight );

extern void DumpGLBuffer(GLenum type, size_t width, size_t height, void* buffer);
extern const char *GetGLTypeName( GLenum type );
extern const char *GetGLAttachmentName( GLenum att );
extern const char *GetGLTargetName( GLenum tgt );
extern const char *GetGLBaseFormatName( GLenum baseformat );
extern const char *GetGLFormatName( GLenum format );

extern void* CreateRandomData( ExplicitType type, size_t count, MTdata d );

extern GLenum GetGLFormat(GLenum internalFormat);
extern GLenum GetGLTypeForExplicitType(ExplicitType type);
extern size_t GetGLTypeSize(GLenum type);
extern ExplicitType GetExplicitTypeForGLType(GLenum type);

extern GLenum get_base_gl_target( GLenum target );

extern int init_clgl_ext( void );

extern GLint get_gl_max_samples( GLenum target, GLenum internalformat );

#endif // _helpers_h



