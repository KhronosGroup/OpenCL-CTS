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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

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
#include "harness/threadTesting.h"
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
        glTextureWrapper() { mBuffer = 0; }
        glTextureWrapper( GLuint b ) { mBuffer = b; }
        ~glTextureWrapper() { if( mBuffer != 0 ) glDeleteTextures( 1, &mBuffer ); }

        glTextureWrapper & operator=( const GLuint &rhs ) { mBuffer = rhs; return *this; }
        operator GLuint() { return mBuffer; }
        operator GLuint *() { return &mBuffer; }

        GLuint * operator&() { return &mBuffer; }

        bool operator==( GLuint rhs ) { return mBuffer == rhs; }

    protected:

        GLuint mBuffer;
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


// Helper functions (defined in helpers.cpp)
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

extern void * ReadGLTexture( GLenum glTarget, GLuint glTexture,
                             GLenum glFormat, GLenum glInternalFormat,
                             GLenum glType, ExplicitType typeToReadAs,
                             size_t outWidth, size_t outHeight );

void * CreateGLRenderbuffer( GLsizei width, GLsizei height,
                             GLenum attachment,
                             GLenum rbFormat, GLenum rbType,
                             GLenum texFormat, GLenum texType,
                             ExplicitType type,
                             GLuint *outFramebuffer,
                             GLuint *outRenderbuffer,
                             int *outError, MTdata d, bool allocateMem );

int CreateGLRenderbufferRaw( GLsizei width, GLsizei height,
                            GLenum attachment,
                            GLenum rbFormat, GLenum rbType,
                            GLuint *outFramebuffer,
                            GLuint *outRenderbuffer );

void * ReadGLRenderbuffer( GLuint glFramebuffer, GLuint glRenderbuffer,
                           GLenum attachment,
                           GLenum rbFormat, GLenum rbType,
                           GLenum texFormat, GLenum texType,
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

extern int init_clgl_ext( cl_platform_id platform_id );

#endif // _helpers_h



