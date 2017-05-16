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
#include "helpers.h"

#include "gl_headers.h"

#define CHECK_ERROR()\
    {GLint __error = glGetError(); if(__error) {log_error( "GL ERROR: %s!\n", gluErrorString( err ));}}

#if defined(__linux__) || defined(GL_ES_VERSION_2_0)
// On linux we dont link to GLU library to avoid comaptibility issues with
// libstdc++
// FIXME: Implement this
const GLubyte* gluErrorString (GLenum error)
{
    const char* gl_Error = "OpenGL Error";
    return (const GLubyte*)gl_Error;
}
#endif

static void DrawQuad(void);

void * CreateGLTexture2D( size_t width, size_t height,
                        GLenum target, GLenum glFormat,
                        GLenum internalFormat, GLenum glType,
                        ExplicitType type, GLuint *outTextureID,
                        int *outError, bool allocateMem, MTdata d )
{
    *outError = 0;
    GLenum err = 0;

    char * buffer = (char *)CreateRandomData(type, width * height * 4, d);

    glGenTextures( 1, outTextureID );
    glBindTexture( get_base_gl_target( target ), *outTextureID );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Failed to create GL texture object: %s!\n", gluErrorString( err ));
        *outError = -1;
        free( buffer );
        return NULL;
    }

#ifndef GL_ES_VERSION_2_0
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
#endif
    glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    if( get_base_gl_target( target ) == GL_TEXTURE_CUBE_MAP )
    {
        char * temp = (char *)malloc(width * height * 4 * get_explicit_type_size( type ) * sizeof(cl_char));
        if(allocateMem)
            memcpy( temp, buffer, width * height * 4 * get_explicit_type_size( type ) );
        else
            memset( temp, 0, width * height * 4 * get_explicit_type_size( type ) );

        glTexImage2D( GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        glTexImage2D( GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        glTexImage2D( GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        glTexImage2D( GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        glTexImage2D( GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        glTexImage2D( GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, temp );
        free(temp);
    }
    else
    {
#ifdef GLES_DEBUG
        log_info("- glTexImage2D : %s : %s : %d : %d : %s : %s\n",
            GetGLTargetName(target),
            GetGLFormatName(internalFormat),
            width, height,
            GetGLFormatName(glFormat),
            GetGLTypeName(glType));

        DumpGLBuffer(glType, width, height, buffer);

#endif
        glTexImage2D( get_base_gl_target(target), 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, buffer );
    }

    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        /**  In section 9.8.3.1. of the CL 1.1. spec it says that:
          *
          *     If a GL texture object with an internal format from table 9.4 is successfully created by
          *     OpenGL, then there is guaranteed to be a mapping to one of the corresponding CL image
          *     format(s) in that table.
          *
          *  Notice that some of the formats in table 9.4 are not supported in OpenGL ES 2.0.
          */
        log_info( "Warning: Skipping %s : %s : %d : %d : %s : %s : because glTexImage2D returned %s\n",
            GetGLTargetName(target),
            GetGLFormatName(internalFormat),
            (int)(width), (int)(height),
            GetGLFormatName(glFormat),
            GetGLTypeName(glType),
            gluErrorString( err ));

        glDeleteTextures( 1, outTextureID );
        *outTextureID = 0;
        *outError = 0;
        free( buffer );
        err = glGetError();
        return NULL;
    }

#ifdef GLES_DEBUG
    memset(buffer, 0, width * height * 4 * get_explicit_type_size( type ));

    log_info("- glGetTexImage : %s : %s : %s\n",
        GetGLTargetName(target),
        GetGLFormatName(glFormat),
        GetGLTypeName(glType));

    glGetTexImage(target, 0, glFormat, glType, buffer);

    DumpGLBuffer(type, width, height, buffer);

    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Unable to read data from glGetTexImage : %s : %s : %s : Error %s\n",
        GetGLTargetName(target),
        GetGLFormatName(glFormat),
        GetGLTypeName(glType),
        gluErrorString( err ));
        return NULL;
    }
#endif

    if( !allocateMem )
    {
        free( buffer );
        return NULL;
    }

#ifndef GL_ES_VERSION_2_0
    if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
    {
        // Reverse and reorder to validate since in the
        // kernel the read_imagef() call always returns RGBA
        cl_uchar *p = (cl_uchar *)buffer;
        for( size_t i = 0; i < width * height; i++ )
        {
            cl_uchar uc0 = p[i * 4 + 0];
            cl_uchar uc1 = p[i * 4 + 1];
            cl_uchar uc2 = p[i * 4 + 2];
            cl_uchar uc3 = p[i * 4 + 3];

            p[ i * 4 + 0 ] = uc2;
            p[ i * 4 + 1 ] = uc1;
            p[ i * 4 + 2 ] = uc0;
            p[ i * 4 + 3 ] = uc3;
        }
    }
#endif

    return buffer;
}

void * CreateGLTexture3D( size_t width, size_t height, size_t depth,
                          GLenum target, GLenum glFormat,
                          GLenum internalFormat, GLenum glType,
                          ExplicitType type, GLuint *outTextureID,
                          int *outError, MTdata d, bool allocateMem)
{
    *outError = 0;

    char * buffer = (char *)create_random_data( type, d, width * height * depth * 4 );

    if( type == kFloat && allocateMem )
    {
        // Re-fill the created buffer to just have [0-1] floats, since that's what it'd expect
        cl_float *p = (cl_float *)buffer;
        for( size_t i = 0; i < width * height * depth * 4; i++ )
        {
            p[ i ] = (float) genrand_real1( d );
        }
    }
    else if( !allocateMem )
        memset( buffer, 0, width * height * depth * 4 * get_explicit_type_size( type ) );

    glGenTextures( 1, outTextureID );

    glBindTexture( target, *outTextureID );
#ifndef GL_ES_VERSION_2_0
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
#endif
    glTexParameteri( target, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( target, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glGetError();
    glTexImage3D( target, 0, internalFormat, (GLsizei)width, (GLsizei)height, (GLsizei)depth, 0, glFormat, glType, buffer );
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        /**  In section 9.8.3.1. of the CL 1.1. spec it says that:
          *
          *     If a GL texture object with an internal format from table 9.4 is successfully created by
          *     OpenGL, then there is guaranteed to be a mapping to one of the corresponding CL image
          *     format(s) in that table.
          *
          *  Notice that some of the formats in table 9.4 are not supported in OpenGL ES 2.0.
          */
        log_info( "Warning: Skipping %s : %s : %d : %d : %s : %s : because glTexImage3D returned %s\n",
            GetGLTargetName(target),
            GetGLFormatName(internalFormat),
            (int)(width), (int)(height),
            GetGLFormatName(glFormat),
            GetGLTypeName(glType),
            gluErrorString( err ));

        *outError = 0;
        delete[] buffer;
        return NULL;
    }

    if( !allocateMem )
    {
        delete [] buffer;
        return NULL;
    }

#ifndef GL_ES_VERSION_2_0
    if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
    {
        // Reverse and reorder to validate since in the
        // kernel the read_imagef() call always returns RGBA

        cl_uchar *p = (cl_uchar *)buffer;
        for( size_t i = 0; i < width * height * depth; i++ )
        {
            cl_uchar uc0 = p[i * 4 + 0];
            cl_uchar uc1 = p[i * 4 + 1];
            cl_uchar uc2 = p[i * 4 + 2];
            cl_uchar uc3 = p[i * 4 + 3];

            p[ i * 4 + 0 ] = uc2;
            p[ i * 4 + 1 ] = uc1;
            p[ i * 4 + 2 ] = uc0;
            p[ i * 4 + 3 ] = uc3;
        }
    }
#endif

    return buffer;
}

void * ReadGLTexture( GLenum glTarget, GLuint glTexture,
                        GLenum glFormat, GLenum glInternalFormat,
                        GLenum glType, ExplicitType typeToReadAs,
                        size_t outWidth, size_t outHeight )
{
    // Read results from the GL texture
    glBindTexture(get_base_gl_target(glTarget), glTexture);

    GLint realWidth, realHeight;
    GLint realInternalFormat;
    GLenum readBackFormat = GL_RGBA;
    GLenum readBackType = glType;
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    size_t outBytes = outWidth * outHeight * 4 * GetGLTypeSize(readBackType);
    cl_char *outBuffer = (cl_char *)malloc( outBytes );
    GLenum err = 0;

    memset(outBuffer, 0, outBytes);
    glGenFramebuffersEXT( 1, &glFramebuffer );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, glFramebuffer );
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glTarget, glTexture, 0);
    err = glGetError();
    if (err != GL_NO_ERROR)
    {
        log_error("Failed to attach texture to FBO!\n");
        return NULL;
    }

    glReadPixels( 0, 0, (GLsizei)outWidth, (GLsizei)outHeight, readBackFormat, readBackType, outBuffer );

#ifdef GLES_DEBUG

    log_info( "- glGetTexImage: %s : %s : %s \n",
        GetGLTargetName( glTarget),
        GetGLFormatName(readBackFormat),
        GetGLTypeName(readBackType));

    DumpGLBuffer(readBackType, realWidth, realHeight, (void*)outBuffer);

#endif

    return (void *)outBuffer;
}

int CreateGLRenderbufferRaw( GLsizei width, GLsizei height,
                            GLenum attachment,
                            GLenum rbFormat, GLenum rbType,
                            GLuint *outFramebuffer,
                            GLuint *outRenderbuffer )
{
    GLenum err = 0;

    // Generate a renderbuffer and bind
    glGenRenderbuffersEXT( 1, outRenderbuffer );
    glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, *outRenderbuffer );

    // Allocate storage to the renderbuffer
    glGetError();
    glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, rbFormat, (GLsizei)width,  (GLsizei)height );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error("Failed to allocate render buffer storage!\n");
        return 1701;
    }

    GLint realInternalFormat;
    glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_INTERNAL_FORMAT_EXT, &realInternalFormat );
    rbFormat = realInternalFormat;

#ifdef GLES_DEBUG
    GLint rsize, gsize, bsize, asize;
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_RED_SIZE_EXT,&rsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_GREEN_SIZE_EXT,&gsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_BLUE_SIZE_EXT,&bsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_ALPHA_SIZE_EXT,&asize);

    log_info("Renderbuffer internal format requested: %s actual: %s sizes: r=%d g=%d b=%d a=%d\n",
             GetGLFormatName( internalFormat ), GetGLFormatName( realInternalFormat ),
             rsize, gsize, bsize, asize );
#endif

    // Create and bind a framebuffer to render with
    glGenFramebuffersEXT( 1, outFramebuffer );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, *outFramebuffer );
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Unable to bind framebuffer : Error %s\n",
                  gluErrorString( err ));

        return -1;
    }

    // Attach to the framebuffer
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, *outRenderbuffer );
    err = glGetError();
    GLint status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
    if( status != GL_FRAMEBUFFER_COMPLETE_EXT )
    {
        log_error( "ERROR: Unable to attach renderbuffer to framebuffer (%s, status %x)\n", gluErrorString( err ), (int)status );
        return -1;
    }

    return 0;
}

static void DrawQuad(void)
{
    const char *vssrc =
        "varying   mediump vec2 texCoord;\n"
        "attribute vec2 inPosition;\n"
        "void main() {\n"
        "    texCoord    = vec2((inPosition.x+1.0)/2.0, (inPosition.y+1.0)/2.0);\n"
        "    gl_Position = vec4(inPosition.x, inPosition.y, 0.0, 1.0);\n"
        "}\n";
    const char *fssrc =
        "uniform sampler2D tex;\n"
        "varying mediump vec2      texCoord;\n"
        "void main() {\n"
        "    gl_FragColor =  texture2D(tex, texCoord);\n"
        "}\n";
    GLuint vs, fs, program;
    GLuint positionIdx = 0;
    GLfloat x1 = -1.0f, x2 = 1.0f, y1 = -1.0f, y2 = 1.0f;
    GLfloat vertices[4][2];
    vertices[0][0] = x1; vertices[0][1] = y1;
    vertices[1][0] = x2; vertices[1][1] = y1;
    vertices[2][0] = x1; vertices[2][1] = y2;
    vertices[3][0] = x2; vertices[3][1] = y2;

    vs = glCreateShader(GL_VERTEX_SHADER);
    fs = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vs, 1, &vssrc, NULL);
    glShaderSource(fs, 1, &fssrc, NULL);

    glCompileShader(vs);
    glCompileShader(fs);

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glUseProgram(program);

    positionIdx = glGetAttribLocation(program, "inPosition");
    glEnableVertexAttribArray(positionIdx);
    glVertexAttribPointer(positionIdx, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glUseProgram(0);
    glDeleteProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void * CreateGLRenderbuffer( GLsizei width, GLsizei height,
                             GLenum attachment,
                             GLenum rbFormat, GLenum rbType,
                             GLenum texFormat, GLenum texType,
                             ExplicitType type,
                             GLuint *outFramebuffer,
                             GLuint *outRenderbuffer,
                             int *outError, MTdata d, bool allocateMem )
{
    *outError = CreateGLRenderbufferRaw( width, height, attachment, rbFormat, rbType, outFramebuffer, outRenderbuffer );

    if( *outError != 0 )
        return NULL;

    GLenum err = 0;

    // Generate a renderbuffer and bind
    glGenRenderbuffersEXT( 1, outRenderbuffer );
    glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, *outRenderbuffer );

    // Allocate storage to the renderbuffer
    glGetError();
    glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, rbFormat, (GLsizei)width,  (GLsizei)height );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        *outError = 1701;
        log_error("Failed to allocate render buffer storage!\n");
        return NULL;
    }

    GLint realInternalFormat;
    glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_INTERNAL_FORMAT_EXT, &realInternalFormat );
    rbFormat = realInternalFormat;

#ifdef GLES_DEBUG
    GLint rsize, gsize, bsize, asize;
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_RED_SIZE_EXT,&rsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_GREEN_SIZE_EXT,&gsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_BLUE_SIZE_EXT,&bsize);
    glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_ALPHA_SIZE_EXT,&asize);

    log_info("Renderbuffer internal format requested: %s actual: %s sizes: r=%d g=%d b=%d a=%d\n",
              GetGLFormatName( internalFormat ), GetGLFormatName( realInternalFormat ),
              rsize, gsize, bsize, asize );
#endif

    // Create and bind a framebuffer to render with
    glGenFramebuffersEXT( 1, outFramebuffer );
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, *outFramebuffer );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Unable to bind framebuffer : Error %s\n",
                  gluErrorString( err ));

        *outError = -1;
        return NULL;
    }

    // Attach to the framebuffer
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, *outRenderbuffer );
    CHECK_ERROR();
    GLint status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
    if( status != GL_FRAMEBUFFER_COMPLETE_EXT )
    {
        *outError = -1;
        log_error( "ERROR: Unable to attach renderbuffer to framebuffer (%s, status %x)\n", gluErrorString( err ), (int)status );
        return NULL;
    }

    void* buffer = CreateRandomData(type, width * height * 4, d);

#ifdef GLES_DEBUG
    log_info( "- Fillling renderbuffer: %d : %d : %s : %s \n",
             (int)width, (int)height,
             GetGLFormatName(glFormat),
             GetGLTypeName(glType));

    DumpGLBuffer(glType, (int)width, (int)height, (void*)buffer);
#endif

    CHECK_ERROR();

    // Fill a texture with our input data
    glTextureWrapper texture;
    glGenTextures( 1, &texture );
    glBindTexture( GL_TEXTURE_2D, texture );
    CHECK_ERROR();
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    CHECK_ERROR();
    glTexImage2D( GL_TEXTURE_2D, 0, texFormat, width, height, 0, texFormat, texType, buffer );
    CHECK_ERROR();

    // Render fullscreen textured quad
    glViewport(0, 0, width, height);
    DrawQuad();
    CHECK_ERROR();

    // Read back the data in the renderbuffer
    memset(buffer, 0, width * height * 4 * get_explicit_type_size( type ));
    glReadPixels( 0, 0, (GLsizei)width, (GLsizei)height, texFormat, texType, buffer );

    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Unable to read data via glReadPixels : %d : %d : %s : %s : Error %s\n",
                  (int)width, (int)height,
                  GetGLFormatName(texFormat),
                  GetGLTypeName(texType),
                  gluErrorString( err ));
        *outError = -1;
    }

#ifdef GLES_DEBUG
    log_info( "- glReadPixels: %d : %d : %s : %s \n",
             (int)width, (int)height,
             GetGLFormatName(glFormat),
             GetGLTypeName(glType));

    DumpGLBuffer(glType, (int)width, (int)height, (void*)buffer);
#endif

    if( !allocateMem )
    {
        free( buffer );
        return NULL;
    }

#ifndef GL_ES_VERSION_2_0
    if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
    {
        // Reverse and reorder to validate since in the
        // kernel the read_imagef() call always returns RGBA
        cl_uchar *p = (cl_uchar *)buffer;
        for( size_t i = 0; i < (size_t)width * height; i++ )
        {
            cl_uchar uc0 = p[i * 4 + 0];
            cl_uchar uc1 = p[i * 4 + 1];
            cl_uchar uc2 = p[i * 4 + 2];
            cl_uchar uc3 = p[i * 4 + 3];

            p[ i * 4 + 0 ] = uc2;
            p[ i * 4 + 1 ] = uc1;
            p[ i * 4 + 2 ] = uc0;
            p[ i * 4 + 3 ] = uc3;
        }
    }
#endif

    return buffer;
}

void * ReadGLRenderbuffer( GLuint glFramebuffer, GLuint glRenderbuffer,
                           GLenum attachment,
                           GLenum rbFormat, GLenum rbType,
                           GLenum texFormat, GLenum texType,
                           ExplicitType typeToReadAs,
                           size_t outWidth, size_t outHeight )
{
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, glFramebuffer );
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, glRenderbuffer );

    // Attach to the framebuffer
    GLint err = glGetError();
    if( glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT ) != GL_FRAMEBUFFER_COMPLETE_EXT )
    {
        log_error( "ERROR: Unable to attach renderbuffer to framebuffer (%s)\n", gluErrorString( err ) );
        return NULL;
    }

    // Read results from the GL renderbuffer
#ifdef GLES_DEBUG
    log_info( "- Reading back from GL: %d x %d : %s : %s : %s\n",
             (int)outWidth, (int)outHeight,
             GetGLFormatName( glInternalFormat ),
             GetGLFormatName( glFormat ),
             GetGLTypeName( glType ));
#endif

    GLenum readBackFormat = GL_RGBA;
    GLenum readBackType = texType;

    size_t outBytes = outWidth * outHeight * 4 * GetGLTypeSize(readBackType);
    void *outBuffer = malloc( outBytes );
    memset(outBuffer, 0, outBytes);

    glReadPixels( 0, 0, (GLsizei)outWidth, (GLsizei)outHeight, readBackFormat, readBackType, outBuffer );

#ifdef GLES_DEBUG
    log_info( "- glReadPixels: %d : %d : %s : %s \n",
             (int)outWidth, (int)outHeight,
             GetGLFormatName(readBackFormat),
             GetGLTypeName(readBackType));

    DumpGLBuffer(readBackType, outWidth, outHeight, outBuffer);
#endif

    return (void *)outBuffer;
}

GLenum
GetGLFormat(GLenum internalFormat)
{
    GLenum glFormat;
    switch (internalFormat)
    {
        case GL_BGRA:
#ifndef GL_ES_VERSION_2_0
        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA32F_ARB:
#endif
            glFormat = GL_RGBA;
            break;
#ifndef GL_ES_VERSION_2_0
        case GL_RGBA8I_EXT:
        case GL_RGBA16I_EXT:
        case GL_RGBA32I_EXT:
        case GL_RGBA8UI_EXT:
        case GL_RGBA16UI_EXT:
        case GL_RGBA32UI_EXT:
            glFormat = GL_RGBA_INTEGER_EXT;
            break;
#endif
        default:
            glFormat = GL_RGBA;
            break;
    }

    return glFormat;
}

GLenum GetGLTypeForExplicitType(ExplicitType type)
{
    switch( type )
    {
        case kFloat:
            return GL_FLOAT;
        case kInt:
            return GL_INT;
        case kUInt:
            return GL_UNSIGNED_INT;
        case kShort:
            return GL_SHORT;
        case kUShort:
            return GL_UNSIGNED_SHORT;
        case kChar:
            return GL_BYTE;
        case kUChar:
            return GL_UNSIGNED_BYTE;
        case kHalf:
#if defined( __APPLE__ )
            return GL_HALF_FLOAT;
#else
            return GL_HALF_FLOAT_ARB;
#endif
        default:
            return GL_INT;
    };
}

size_t GetGLTypeSize(GLenum type)
{
    switch( type )
    {
        case GL_FLOAT:
            return sizeof(GLfloat);
        case GL_INT:
            return sizeof(GLint);
        case GL_UNSIGNED_INT:
            return sizeof(GLuint);
        case GL_SHORT:
            return sizeof(GLshort);
        case GL_UNSIGNED_SHORT:
            return sizeof(GLushort);
        case GL_BYTE:
            return sizeof(GLbyte);
        case GL_UNSIGNED_BYTE:
            return sizeof(GLubyte);
#if defined( __APPLE__ )
        case GL_HALF_FLOAT:
#else
        case GL_HALF_FLOAT_ARB:
#endif
            return sizeof(GLhalf);
        default:
            return kFloat;
    };
}

ExplicitType GetExplicitTypeForGLType(GLenum type)
{
    switch( type )
    {
        case GL_FLOAT:
            return kFloat;
        case GL_INT:
            return kInt;
        case GL_UNSIGNED_INT:
            return kUInt;
        case GL_SHORT:
            return kShort;
        case GL_UNSIGNED_SHORT:
            return kUShort;
        case GL_BYTE:
            return kChar;
        case GL_UNSIGNED_BYTE:
            return kUChar;
#if defined( __APPLE__ )
        case GL_HALF_FLOAT:
#else
        case GL_HALF_FLOAT_ARB:
#endif
            return kHalf;
        default:
            return kFloat;
    };
}

GLenum get_base_gl_target( GLenum target )
{
    switch( target )
    {
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            return GL_TEXTURE_CUBE_MAP;
        default:
            return target;
    }
}

const char *GetGLTypeName( GLenum type )
{
    switch( type )
    {
        case GL_BYTE:            return "GL_BYTE";
        case GL_UNSIGNED_BYTE:   return "GL_UNSIGNED_BYTE";
        case GL_INT:             return "GL_INT";
        case GL_UNSIGNED_INT:    return "GL_UNSIGNED_INT";
        case GL_SHORT:           return "GL_SHORT";
        case GL_UNSIGNED_SHORT:  return "GL_UNSIGNED_SHORT";
#if defined( __APPLE__ )
        case GL_HALF_FLOAT:      return "GL_HALF_FLOAT";
#else
        case GL_HALF_FLOAT_ARB:  return "GL_HALF_FLOAT_ARB";
#endif
        case GL_FLOAT:           return "GL_FLOAT";
#ifndef GL_ES_VERSION_2_0
        case GL_UNSIGNED_INT_8_8_8_8: return "GL_UNSIGNED_INT_8_8_8_8";
        case GL_UNSIGNED_INT_8_8_8_8_REV: return "GL_UNSIGNED_INT_8_8_8_8_REV";
#endif
        default:
        {
        static char foo[ 128 ];
        sprintf( foo, "(Unknown:0x%08x)", (int)type );
        return foo;
        }
    }
}

const char *GetGLTargetName( GLenum tgt )
{
    if( tgt == GL_TEXTURE_2D )          return "GL_TEXTURE_2D";
    if( tgt == GL_TEXTURE_3D )          return "GL_TEXTURE_3D";
#ifndef GL_ES_VERSION_2_0
    if( tgt == GL_TEXTURE_RECTANGLE_EXT ) return "GL_TEXTURE_RECTANGLE_EXT";
#endif
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_X ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_X";
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_Y ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_Y";
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_Z ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_Z";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_X ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_X";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_Y ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Y";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_Z ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Z";
    return "";
}

const char *GetGLAttachmentName( GLenum att )
{
    if( att == GL_COLOR_ATTACHMENT0_EXT ) return "GL_COLOR_ATTACHMENT0_EXT";
#ifndef GL_ES_VERSION_2_0
    if( att == GL_COLOR_ATTACHMENT1_EXT ) return "GL_COLOR_ATTACHMENT1_EXT";
    if( att == GL_COLOR_ATTACHMENT2_EXT ) return "GL_COLOR_ATTACHMENT2_EXT";
    if( att == GL_COLOR_ATTACHMENT3_EXT ) return "GL_COLOR_ATTACHMENT3_EXT";
    if( att == GL_COLOR_ATTACHMENT4_EXT ) return "GL_COLOR_ATTACHMENT4_EXT";
    if( att == GL_COLOR_ATTACHMENT5_EXT ) return "GL_COLOR_ATTACHMENT5_EXT";
    if( att == GL_COLOR_ATTACHMENT6_EXT ) return "GL_COLOR_ATTACHMENT6_EXT";
    if( att == GL_COLOR_ATTACHMENT7_EXT ) return "GL_COLOR_ATTACHMENT7_EXT";
    if( att == GL_COLOR_ATTACHMENT8_EXT ) return "GL_COLOR_ATTACHMENT8_EXT";
#endif
    if( att == GL_DEPTH_ATTACHMENT_EXT ) return "GL_DEPTH_ATTACHMENT_EXT";
    return "";
}
const char *GetGLBaseFormatName( GLenum baseformat )
{
    switch( baseformat )
    {
        case GL_RGBA:            return "GL_RGBA";
#ifdef GL_ES_VERSION_2_0
        case GL_BGRA_EXT:            return "GL_BGRA_EXT";
#else
        case GL_RGBA8:            return "GL_RGBA";
        case GL_RGBA16:            return "GL_RGBA";
        case GL_BGRA:            return "GL_BGRA";
        case GL_RGBA8I_EXT:        return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA16I_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32I_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA8UI_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA16UI_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32UI_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32F_ARB:    return "GL_RGBA";

        case GL_RGBA_INTEGER_EXT:    return "GL_RGBA_INTEGER_EXT";

        case GL_ALPHA4: return "GL_ALPHA";
        case GL_ALPHA8: return "GL_ALPHA";
        case GL_ALPHA12: return "GL_ALPHA";
        case GL_ALPHA16: return "GL_ALPHA";
        case GL_LUMINANCE4: return "GL_LUMINANCE";
        case GL_LUMINANCE8: return "GL_LUMINANCE";
        case GL_LUMINANCE12: return "GL_LUMINANCE";
        case GL_LUMINANCE16: return "GL_LUMINANCE";
        case GL_LUMINANCE4_ALPHA4: return "GL_LUMINANCE_ALPHA";
        case GL_LUMINANCE6_ALPHA2: return "GL_LUMINANCE_ALPHA";
        case GL_LUMINANCE8_ALPHA8: return "GL_LUMINANCE_ALPHA";
        case GL_LUMINANCE12_ALPHA4: return "GL_LUMINANCE_ALPHA";
        case GL_LUMINANCE12_ALPHA12: return "GL_LUMINANCE_ALPHA";
        case GL_LUMINANCE16_ALPHA16: return "GL_LUMINANCE_ALPHA";
        case GL_INTENSITY: return "GL_INTENSITY";
        case GL_INTENSITY4: return "GL_INTENSITY";
        case GL_INTENSITY8: return "GL_INTENSITY";
        case GL_INTENSITY12: return "GL_INTENSITY";
        case GL_INTENSITY16: return "GL_INTENSITY";
        case GL_R3_G3_B2: return "GL_RGB";
        case GL_RGB4: return "GL_RGB";
        case GL_RGB5: return "GL_RGB";
        case GL_RGB8: return "GL_RGB";
        case GL_RGB10: return "GL_RGB";
        case GL_RGB12: return "GL_RGB";
        case GL_RGB16: return "GL_RGB";
        case GL_RGBA2: return "GL_RGBA";
        case GL_RGBA4: return "GL_RGBA";
        case GL_RGB5_A1: return "GL_RGBA";
        case GL_RGB10_A2: return "GL_RGBA";
        case GL_RGBA12: return "GL_RGBA";
#endif

        default:
        {
            static char foo[ 128 ];
            sprintf( foo, "(Unknown:0x%08x)", (int)baseformat );
            return foo;
        }
    }
}

const char *GetGLFormatName( GLenum format )
{
    switch( format )
    {
        case GL_RGBA:            return "GL_RGBA";
#ifdef GL_ES_VERSION_2_0
        case GL_BGRA_EXT:            return "GL_BGRA_EXT";
#else
        case GL_RGBA8:            return "GL_RGBA8";
        case GL_RGBA16:            return "GL_RGBA16";
        case GL_BGRA:            return "GL_BGRA";
        case GL_RGBA8I_EXT:        return "GL_RGBA8I_EXT";
        case GL_RGBA16I_EXT:    return "GL_RGBA16I_EXT";
        case GL_RGBA32I_EXT:    return "GL_RGBA32I_EXT";
        case GL_RGBA8UI_EXT:    return "GL_RGBA8UI_EXT";
        case GL_RGBA16UI_EXT:    return "GL_RGBA16UI_EXT";
        case GL_RGBA32UI_EXT:    return "GL_RGBA32UI_EXT";
        case GL_RGBA32F_ARB:    return "GL_RGBA32F_ARB";

        case GL_RGBA_INTEGER_EXT:    return "GL_RGBA_INTEGER_EXT";

        case GL_ALPHA4: return "GL_ALPHA4";
        case GL_ALPHA8: return "GL_ALPHA8";
        case GL_ALPHA12: return "GL_ALPHA12";
        case GL_ALPHA16: return "GL_ALPHA16";
        case GL_LUMINANCE4: return "GL_LUMINANCE4";
        case GL_LUMINANCE8: return "GL_LUMINANCE8";
        case GL_LUMINANCE12: return "GL_LUMINANCE12";
        case GL_LUMINANCE16: return "GL_LUMINANCE16";
        case GL_LUMINANCE4_ALPHA4: return "GL_LUMINANCE4_ALPHA4";
        case GL_LUMINANCE6_ALPHA2: return "GL_LUMINANCE6_ALPHA2";
        case GL_LUMINANCE8_ALPHA8: return "GL_LUMINANCE8_ALPHA8";
        case GL_LUMINANCE12_ALPHA4: return "GL_LUMINANCE12_ALPHA4";
        case GL_LUMINANCE12_ALPHA12: return "GL_LUMINANCE12_ALPHA12";
        case GL_LUMINANCE16_ALPHA16: return "GL_LUMINANCE16_ALPHA16";
        case GL_INTENSITY: return "GL_INTENSITY";
        case GL_INTENSITY4: return "GL_INTENSITY4";
        case GL_INTENSITY8: return "GL_INTENSITY8";
        case GL_INTENSITY12: return "GL_INTENSITY12";
        case GL_INTENSITY16: return "GL_INTENSITY16";
        case GL_R3_G3_B2: return "GL_R3_G3_B2";
        case GL_RGB4: return "GL_RGB4";
        case GL_RGB5: return "GL_RGB5";
        case GL_RGB8: return "GL_RGB8";
        case GL_RGB10: return "GL_RGB10";
        case GL_RGB12: return "GL_RGB12";
        case GL_RGB16: return "GL_RGB16";
        case GL_RGBA2: return "GL_RGBA2";
        case GL_RGBA4: return "GL_RGBA4";
        case GL_RGB5_A1: return "GL_RGB5_A1";
        case GL_RGB10_A2: return "GL_RGB10_A2";
        case GL_RGBA12: return "GL_RGBA12";
#endif
        case GL_INT:            return "GL_INT";
        case GL_UNSIGNED_INT:    return "GL_UNSIGNED_INT";
        case GL_SHORT:            return "GL_SHORT";
        case GL_UNSIGNED_SHORT:    return "GL_UNSIGNED_SHORT";
        case GL_BYTE:            return "GL_BYTE";
        case GL_UNSIGNED_BYTE:    return "GL_UNSIGNED_BYTE";
        case GL_FLOAT:            return "GL_FLOAT";
#ifdef GL_ES_VERSION_2_0
        case GL_HALF_FLOAT_OES: return "GL_HALF_FLOAT_OES";
#else
#if defined( __APPLE__ )
        case GL_HALF_FLOAT:        return "GL_HALF_FLOAT";
#else
        case GL_HALF_FLOAT_ARB: return "GL_HALF_FLOAT_ARB";
#endif
#endif

        default:
        {
            static char foo[ 128 ];
            sprintf( foo, "(Unknown:0x%08x)", (int)format );
            return foo;
        }
    }
}

cl_ushort float2half_rte( float f )
{
    union{ float f; cl_uint u; } u = {f};
    cl_uint sign = (u.u >> 16) & 0x8000;
    float x = fabsf(f);

    //Nan
    if( x != x )
    {
        u.u >>= (24-11);
        u.u &= 0x7fff;
        u.u |= 0x0200;      //silence the NaN
        return u.u | sign;
    }

    // overflow
    if( x >= MAKE_HEX_FLOAT(0x1.ffep15f, 0x1ffeL, 3) )
        return 0x7c00 | sign;

    // underflow
    if( x <= MAKE_HEX_FLOAT(0x1.0p-25f, 0x1L, -25) )
        return sign;    // The halfway case can return 0x0001 or 0. 0 is even.

    // very small
    if( x < MAKE_HEX_FLOAT(0x1.8p-24f, 0x18L, -28) )
        return sign | 1;

    // half denormal
    if( x < MAKE_HEX_FLOAT(0x1.0p-14f, 0x1L, -14) )
    {
        u.f = x * MAKE_HEX_FLOAT(0x1.0p-125f, 0x1L, -125);
        return sign | u.u;
    }

    u.f *= MAKE_HEX_FLOAT(0x1.0p13f, 0x1L, 13);
    u.u &= 0x7f800000;
    x += u.f;
    u.f = x - u.f;
    u.f *= MAKE_HEX_FLOAT(0x1.0p-112f, 0x1L, -112);

    return (u.u >> (24-11)) | sign;
}

void* CreateRandomData( ExplicitType type, size_t count, MTdata d )
{
    switch(type)
    {
        case (kChar):
        {
            cl_char *p = (cl_char *)malloc(count * sizeof(cl_char));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = (cl_char)genrand_int32(d);
            }
            return (void*)p;
        }
        case (kUChar):
        {
            cl_uchar *p = (cl_uchar *)malloc(count * sizeof(cl_uchar));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] =  (cl_uchar)genrand_int32(d);
            }

            return (void*)p;
        }
        case (kShort):
        {
            cl_short *p = (cl_short *)malloc(count * sizeof(cl_short));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = (cl_short)genrand_int32(d);
            }

            return (void*)p;
        }
        case (kUShort):
        {
            cl_ushort *p = (cl_ushort *)malloc(count * sizeof(cl_ushort));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = (cl_ushort)genrand_int32(d);
            }

            return (void*)p;
        }
        case (kInt):
        {
            cl_int *p = (cl_int *)malloc(count * sizeof(cl_int));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = (cl_int)genrand_int32(d);
            }

            return (void*)p;
        }
        case (kUInt):
        {
            cl_uint *p = (cl_uint *)malloc(count * sizeof(cl_uint));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] =  (cl_uint)genrand_int32(d);
            }

            return (void*)p;
        }

        case (kFloat):
        {
            cl_float *p = (cl_float *)malloc(count * sizeof(cl_float));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = get_random_float( 0.f, 1.f, d );
            }

            return (void*)p;
        }
        /* added support for half floats */
        case (kHalf):
        {
            cl_half *p = (cl_half *)malloc(count * sizeof(cl_half));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = float2half_rte(get_random_float( 0.f, 1.f, d ));
            }

            return (void*)p;
        }
        default:
        {
            log_error("Invalid explicit type specified for create random data!\n");
            return 0;
        }
    }
    return 0;
}

void DumpGLBuffer(GLenum type, size_t width, size_t height, void* buffer)
{
    size_t i;
    size_t count = width * height;
    if(type == GL_BYTE)
    {
        cl_char* p = (cl_char*)buffer;
        for(i = 0; i < count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_UNSIGNED_BYTE)
    {
        cl_uchar* p = (cl_uchar*)buffer;
        for(i = 0; i < count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_INT)
    {
        cl_int* p = (cl_int*)buffer;
        for(i = 0; i < count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_UNSIGNED_INT)
    {
        cl_uint* p = (cl_uint*)buffer;
        for(i = 0; i < count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_SHORT)
    {
        cl_short* p = (cl_short*)buffer;
        for(i = 0; i < count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_UNSIGNED_SHORT)
    {
        cl_ushort* p = (cl_ushort*)buffer;
        for(i = 0; i <  count; i++)
            log_info("[%4d] %3d %3d %3d %3d\n", (unsigned int)(i),
                p[i* 4 + 0],
                p[i* 4 + 1],
                p[i* 4 + 2],
                p[i* 4 + 3]);
    }
    else if(type == GL_FLOAT)
    {
        cl_float* p = (cl_float*)buffer;
        for(i = 0; i < count; i++)
        log_info("[%4d] %#f %#f %#f %#f\n", (unsigned int)(i),
            p[i* 4 + 0],
            p[i* 4 + 1],
            p[i* 4 + 2],
            p[i* 4 + 3]);
    }
}

#if defined(_WIN32)
#include <string.h>

GLboolean gluCheckExtension(const GLubyte *extName, const GLubyte *extString)
{
  const size_t len = strlen((const char*)extName);
  const char* str = (const char*)extString;

  while (str != NULL) {
    str = strstr(str, (const char*)extName);
    if (str == NULL) {
      break;
    }
    if ((str > (const char*)extString || str[-1] == ' ')
        && (str[len] == ' ' || str[len] == '\0')) {
      return GL_TRUE;
    }
    str = strchr(str + len, ' ');
  }

  return GL_FALSE;
}

#endif

// Function pointers for the GL/CL calls
clCreateFromGLBuffer_fn clCreateFromGLBuffer_ptr;
clCreateFromGLTexture_fn clCreateFromGLTexture_ptr;
clCreateFromGLRenderbuffer_fn clCreateFromGLRenderbuffer_ptr;
clGetGLObjectInfo_fn clGetGLObjectInfo_ptr;
clGetGLTextureInfo_fn clGetGLTextureInfo_ptr;
clEnqueueAcquireGLObjects_fn clEnqueueAcquireGLObjects_ptr;
clEnqueueReleaseGLObjects_fn clEnqueueReleaseGLObjects_ptr;

int init_clgl_ext(cl_platform_id platform_id)
{
    // Create the function pointer table
    clCreateFromGLBuffer_ptr = (clCreateFromGLBuffer_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clCreateFromGLBuffer");
    if (clCreateFromGLBuffer_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clCreateFromGLBuffer) returned NULL.\n");
        return -1;
    }

    clCreateFromGLTexture_ptr = (clCreateFromGLTexture_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clCreateFromGLTexture");
    if (clCreateFromGLTexture_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clCreateFromGLTexture) returned NULL.\n");
        return -1;
    }

    clCreateFromGLRenderbuffer_ptr = (clCreateFromGLRenderbuffer_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clCreateFromGLRenderbuffer");
    if (clCreateFromGLRenderbuffer_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clCreateFromGLRenderbuffer) returned NULL.\n");
        return -1;
    }

    clGetGLObjectInfo_ptr = (clGetGLObjectInfo_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clGetGLObjectInfo");
    if (clGetGLObjectInfo_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clGetGLObjectInfo) returned NULL.\n");
        return -1;
    }

    clGetGLTextureInfo_ptr = (clGetGLTextureInfo_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clGetGLTextureInfo");
    if (clGetGLTextureInfo_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clGetGLTextureInfo) returned NULL.\n");
        return -1;
    }

    clEnqueueAcquireGLObjects_ptr = (clEnqueueAcquireGLObjects_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueAcquireGLObjects");
    if (clEnqueueAcquireGLObjects_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clEnqueueAcquireGLObjects) returned NULL.\n");
        return -1;
    }

    clEnqueueReleaseGLObjects_ptr = (clEnqueueReleaseGLObjects_fn)clGetExtensionFunctionAddressForPlatform(platform_id, "clEnqueueReleaseGLObjects");
    if (clEnqueueReleaseGLObjects_ptr == NULL)
    {
        log_error("clGetExtensionFunctionAddressForPlatform(clEnqueueReleaseGLObjects) returned NULL.\n");
        return -1;
    }

    return 0;
}


