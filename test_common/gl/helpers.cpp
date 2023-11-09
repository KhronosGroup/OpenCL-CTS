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
#include "harness/imageHelpers.h"

#if defined( __APPLE__ )
    #include <OpenGL/glu.h>
#else
    #include <GL/glu.h>
#endif

#if defined(__linux__)
// On linux we dont link to GLU library to avoid comaptibility issues with
// libstdc++
// FIXME: Implement this
const GLubyte* gluErrorString (GLenum error)
{
    const char* gl_Error = "OpenGL Error";
    return (const GLubyte*)gl_Error;
}
#endif

void * CreateGLTexture1DArray(size_t width, size_t length,
  GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
  ExplicitType type, GLuint *outTextureID, int *outError,
  bool allocateMem, MTdata d)
{
  *outError = 0;
  GLenum err = 0;

  char * buffer;
  unsigned int size = 0;

  // width_in_pixels * pixel_width * number_of_images:
  if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
  {
    size = width * length;
  }
  else
  {
    size = width * length * 4;
  }

  buffer = (char *)CreateRandomData(type, size, d);

  glGenTextures( 1, outTextureID );
  glBindTexture( get_base_gl_target( target ), *outTextureID );
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  err = glGetError();
  if( err != GL_NO_ERROR ) {
    log_error( "ERROR: Failed to create GL texture object: %s!\n", gluErrorString( err ));
    *outError = -1;
    free( buffer );
    return NULL;
  }

  // use TexImage2D to pump the 1D array fill of bits:
  glTexImage2D( get_base_gl_target(target), 0, internalFormat, (GLsizei)width,
    (GLsizei)length, 0, glFormat, glType, buffer );

  err = glGetError();
  if( err != GL_NO_ERROR ) {
    if (err != GL_OUT_OF_MEMORY) {
        log_error( "ERROR: Unable to load data using glTexImage2D for "
          "TEXTURE_1D_ARRAY : %s : %s : %d : %d : %s : %s : Error %s\n",
        GetGLTargetName(target),
        GetGLFormatName(internalFormat),
        (int)(width), (int)(length),
        GetGLFormatName(glFormat),
        GetGLTypeName(glType),
        gluErrorString( err ));

        *outError = -1;
    } else {
        log_info( "WARNING: Unable to load data using glTexImage2D for "
          "TEXTURE_1D_ARRAY : %s : %s : %d : %d : %s : %s : Error %s\n",
        GetGLTargetName(target),
        GetGLFormatName(internalFormat),
        (int)(width), (int)(length),
        GetGLFormatName(glFormat),
        GetGLTypeName(glType),
        gluErrorString( err ));

        *outError = -2;
    }
    free( buffer );
    return NULL;
  }

  if( !allocateMem ) {
    free( buffer );
    return NULL;
  }

  if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width * length; i++ ) {
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
  else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width * length; i++ )
    {
      cl_uchar uc0 = p[i * 4 + 0];
      cl_uchar uc1 = p[i * 4 + 1];
      cl_uchar uc2 = p[i * 4 + 2];
      cl_uchar uc3 = p[i * 4 + 3];

      p[ i * 4 + 0 ] = uc1;
      p[ i * 4 + 1 ] = uc2;
      p[ i * 4 + 2 ] = uc3;
      p[ i * 4 + 3 ] = uc0;
    }
  }

  return buffer;
}

void * CreateGLTexture2DArray(size_t width, size_t height, size_t length,
  GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
  ExplicitType type, GLuint *outTextureID, int *outError,
  bool allocateMem, MTdata d)
{
  *outError = 0;

  char * buffer;
  unsigned int size = 0;

  if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
  {
    size = width * height * length;
  }
  else
  {
    size = width * height * length * 4;
  }

  buffer = (char *)CreateRandomData(type, size, d);

  if( type == kFloat && allocateMem )
  {
    // Re-fill the created buffer to just have [0-1] floats, since that's what it'd expect
    cl_float *p = (cl_float *)buffer;
    for( size_t i = 0; i < size; i++ )
    {
      p[ i ] = (float) genrand_real1( d );
    }
  }
  else if( !allocateMem )
    memset( buffer, 0, size * get_explicit_type_size( type ) );

  glGenTextures( 1, outTextureID );

  glBindTexture( target, *outTextureID );
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glTexParameteri( target, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( target, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

  glGetError();
  //the default alignment in OpenGL is 4 bytes and need to be changed for GL_DEPTH_COMPONENT16 which is aligned to 2 bytes
  if (internalFormat == GL_DEPTH_COMPONENT16)
    glPixelStorei(GL_UNPACK_ALIGNMENT, get_explicit_type_size( type ));

  glTexImage3D( target, 0, internalFormat, (GLsizei)width, (GLsizei)height,
    (GLsizei)length, 0, glFormat, glType, buffer );

  if (internalFormat == GL_DEPTH_COMPONENT16)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  GLenum err = glGetError();
  if( err != GL_NO_ERROR )
  {
    if (err != GL_OUT_OF_MEMORY) {
        log_error( "ERROR: Unable to load data into GL texture (%s) format %s "
          "type %s internal format %s\n", gluErrorString( err ),
          GetGLFormatName( glFormat ), get_explicit_type_name( type ),
          GetGLFormatName( internalFormat ) );
        *outError = -1;
    } else {
        log_info( "WARNING: Unable to load data into GL texture (%s) format %s "
          "type %s internal format %s\n", gluErrorString( err ),
          GetGLFormatName( glFormat ), get_explicit_type_name( type ),
          GetGLFormatName( internalFormat ) );
        *outError = -2;
    }
    delete [] buffer;
    return NULL;
  }

  if( !allocateMem )
  {
    delete [] buffer;
    return NULL;
  }

  if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width * height * length; i++ )
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
  else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width * length; i++ )
    {
      cl_uchar uc0 = p[i * 4 + 0];
      cl_uchar uc1 = p[i * 4 + 1];
      cl_uchar uc2 = p[i * 4 + 2];
      cl_uchar uc3 = p[i * 4 + 3];

      p[ i * 4 + 0 ] = uc1;
      p[ i * 4 + 1 ] = uc2;
      p[ i * 4 + 2 ] = uc3;
      p[ i * 4 + 3 ] = uc0;
    }
  }

  return buffer;
}

void * CreateGLTextureBuffer(size_t width, GLenum target,
  GLenum glFormat, GLenum internalFormat, GLenum glType, ExplicitType type,
  GLuint *outTex, GLuint *outBuf, int *outError, bool allocateMem, MTdata d)
{
  // First, generate a regular GL Buffer from random data.
  *outError = 0;
  GLenum err = 0;

  char * buffer;
  unsigned int size = 0;

  // The buffer should be the array width * number of elements * element pitch
  if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
  {
    size = width;
  }
  else
  {
    size = width * 4;
  }

  buffer = (char*)CreateRandomData(type, size, d);

  err = glGetError();

  glGenBuffers(1, outBuf);
  glBindBuffer(GL_TEXTURE_BUFFER, *outBuf);

  // Need to multiply by the type size:
  size *= ( GetGLTypeSize( GetGLTypeForExplicitType(type) ) );

  glBufferData(GL_TEXTURE_BUFFER, size, buffer, GL_DYNAMIC_DRAW);

  // Now make a Texture out of this Buffer:

  glGenTextures(1, outTex);
  glBindTexture(GL_TEXTURE_BUFFER, *outTex);
  glTexBuffer(GL_TEXTURE_BUFFER, internalFormat, *outBuf);

  if ((err = glGetError())) {
    log_error( "ERROR: Unable to load data into glTexBuffer : %s : %s : %d : %s : %s : Error %s\n",
              GetGLTargetName(target),
              GetGLFormatName(internalFormat),
              (int)(size),
              GetGLFormatName(glFormat),
              GetGLTypeName(glType),
              gluErrorString( err ));
    *outError = -1;
    delete [] buffer;
    return NULL;
  }

  if( !allocateMem ) {
    free( buffer );
    return NULL;
  }

  if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width; i++ ) {
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
  else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width; i++ )
    {
      cl_uchar uc0 = p[i * 4 + 0];
      cl_uchar uc1 = p[i * 4 + 1];
      cl_uchar uc2 = p[i * 4 + 2];
      cl_uchar uc3 = p[i * 4 + 3];

      p[ i * 4 + 0 ] = uc1;
      p[ i * 4 + 1 ] = uc2;
      p[ i * 4 + 2 ] = uc3;
      p[ i * 4 + 3 ] = uc0;
    }
  }

  return buffer;
}

void* CreateGLTexture1D( size_t width, GLenum target, GLenum glFormat,
    GLenum internalFormat, GLenum glType, ExplicitType type,
    GLuint *outTextureID, int *outError, bool allocateMem, MTdata d )
{
  *outError = 0;
  GLenum err = 0;

  char * buffer;
  unsigned int size = 0;

  // The buffer should be the array width * number of elements * element pitch
  if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
  {
    size = width;
  }
  else
  {
    size = width * 4;
  }

  buffer = (char*)CreateRandomData(type, size, d);

  glGenTextures( 1, outTextureID );
  glBindTexture( get_base_gl_target( target ), *outTextureID );
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  err = glGetError();
  if( err != GL_NO_ERROR )
  {
    log_error( "ERROR: Failed to create GL texture object: %s!\n", gluErrorString( err ));
    *outError = -1;
    free( buffer );
    return NULL;
  }

  glTexImage1D( get_base_gl_target(target), 0, internalFormat, (GLsizei)width,
    0, glFormat, glType, buffer );

  err = glGetError();
  if( err != GL_NO_ERROR )
  {
        if (err != GL_OUT_OF_MEMORY) {
            log_error( "ERROR: Unable to load data into glTexImage1D : %s : %s : %d : %s : %s : Error %s\n",
              GetGLTargetName(target),
              GetGLFormatName(internalFormat),
              (int)(width),
              GetGLFormatName(glFormat),
              GetGLTypeName(glType),
              gluErrorString( err ));
            *outError = -1;
        } else {
            log_info( "WARNING: Unable to load data into glTexImage1D : %s : %s : %d : %s : %s : Error %s\n",
              GetGLTargetName(target),
              GetGLFormatName(internalFormat),
              (int)(width),
              GetGLFormatName(glFormat),
              GetGLTypeName(glType),
              gluErrorString( err ));
            *outError = -2;
        }
      free( buffer );
      return NULL;
  }

  if( !allocateMem ) {
    free( buffer );
    return NULL;
  }

  if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width; i++ ) {
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
  else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < width; i++ )
    {
      cl_uchar uc0 = p[i * 4 + 0];
      cl_uchar uc1 = p[i * 4 + 1];
      cl_uchar uc2 = p[i * 4 + 2];
      cl_uchar uc3 = p[i * 4 + 3];

      p[ i * 4 + 0 ] = uc1;
      p[ i * 4 + 1 ] = uc2;
      p[ i * 4 + 2 ] = uc3;
      p[ i * 4 + 3 ] = uc0;
    }
  }

  return buffer;
}

void * CreateGLTexture2D( size_t width, size_t height,
                        GLenum target, GLenum glFormat,
                        GLenum internalFormat, GLenum glType,
                        ExplicitType type, GLuint *outTextureID,
                        int *outError, bool allocateMem, MTdata d )
{
    *outError = 0;
    GLenum err = 0;

    char * buffer;
    unsigned int size = 0;

    if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
    {
      size = width * height;
    }
    else
    {
      size = width * height * 4;
    }

    buffer = (char *)CreateRandomData(type, size, d);

    glGenTextures( 1, outTextureID );
    glBindTexture( get_base_gl_target( target ), *outTextureID );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( get_base_gl_target( target ), GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Failed to create GL texture object: %s!\n", gluErrorString( err ));
        *outError = -1;
        free( buffer );
        return NULL;
    }

    if( get_base_gl_target( target ) == GL_TEXTURE_CUBE_MAP )
    {
        char * temp = (char *)malloc(size * get_explicit_type_size( type ) * sizeof(cl_char));
        if(allocateMem)
            memcpy( temp, buffer, size * get_explicit_type_size( type ) );
        else
            memset( temp, 0, size * get_explicit_type_size( type ) );

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
#ifdef DEBUG
        log_info("- glTexImage2D : %s : %s : %d : %d : %s : %s\n",
            GetGLTargetName(target),
            GetGLFormatName(internalFormat),
            width, height,
            GetGLFormatName(glFormat),
            GetGLTypeName(glType));

        DumpGLBuffer(glType, width, height, buffer);
#endif

        //the default alignment in OpenGL is 4 bytes and need to be changed for GL_DEPTH_COMPONENT16 which is aligned to 2 bytes
        if (internalFormat == GL_DEPTH_COMPONENT16)
          glPixelStorei(GL_UNPACK_ALIGNMENT, get_explicit_type_size( type ));

        glTexImage2D( get_base_gl_target(target), 0, internalFormat, (GLsizei)width, (GLsizei)height, 0, glFormat, glType, buffer );

        if (internalFormat == GL_DEPTH_COMPONENT16)
          glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        if (err != GL_OUT_OF_MEMORY) {
            log_error( "ERROR: Unable to load data into glTexImage2D : %s : %s : %d : %d : %s : %s : Error %s\n",
                GetGLTargetName(target),
                GetGLFormatName(internalFormat),
                (int)(width), (int)(height),
                GetGLFormatName(glFormat),
                GetGLTypeName(glType),
                gluErrorString( err ));
            *outError = -1;
        } else {
            log_info( "WARNING: Unable to load data into glTexImage2D : %s : %s : %d : %d : %s : %s : Error %s\n",
                GetGLTargetName(target),
                GetGLFormatName(internalFormat),
                (int)(width), (int)(height),
                GetGLFormatName(glFormat),
                GetGLTypeName(glType),
                gluErrorString( err ));
            *outError = -2;
        }
        free( buffer );
        return NULL;
    }

#ifdef DEBUG
    char * test = (char *)malloc(width * height * 4 * get_explicit_type_size( type ));
    memset(test, 0, width * height * 4 * get_explicit_type_size( type ));

    if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
    {
      glFormat = GL_RGBA;
      glType = GL_FLOAT;
    }

    log_info("- glGetTexImage : %s : %s : %s\n",
        GetGLTargetName(target),
        GetGLFormatName(glFormat),
        GetGLTypeName(glType));

    glGetTexImage(target, 0, glFormat, glType, test);

    DumpGLBuffer(glType, width, height, test);

    free(test);

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
    else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
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

        p[ i * 4 + 0 ] = uc1;
        p[ i * 4 + 1 ] = uc2;
        p[ i * 4 + 2 ] = uc3;
        p[ i * 4 + 3 ] = uc0;
      }
    }

    return buffer;
}

void * CreateGLTexture3D( size_t width, size_t height, size_t depth,
                          GLenum target, GLenum glFormat,
                          GLenum internalFormat, GLenum glType,
                          ExplicitType type, GLuint *outTextureID,
                          int *outError, MTdata d, bool allocateMem)
{
    *outError = 0;

    char * buffer;
    unsigned int size = 0;

    if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) )
    {
        size = width * height * depth;
    }
    else
    {
        size = width * height * depth * 4;
    }

    buffer = (char *)create_random_data( type, d, size );

    if( type == kFloat && allocateMem )
    {
        // Re-fill the created buffer to just have [0-1] floats, since that's what it'd expect
        cl_float *p = (cl_float *)buffer;
        for( size_t i = 0; i < size; i++ )
        {
            p[ i ] = (float) genrand_real1( d );
        }
    }
    else if( !allocateMem )
        memset( buffer, 0, size * get_explicit_type_size( type ) );

    glGenTextures( 1, outTextureID );

    glBindTexture( target, *outTextureID );
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( target, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( target, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

    glGetError();
    glTexImage3D( target, 0, internalFormat, (GLsizei)width, (GLsizei)height, (GLsizei)depth, 0, glFormat, glType, buffer );
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        if (err != GL_OUT_OF_MEMORY) {
            log_error( "ERROR: Unable to load data into GL texture (%s) format %s type %s internal format %s\n", gluErrorString( err ), GetGLFormatName( glFormat ), get_explicit_type_name( type ), GetGLFormatName( internalFormat ) );
            *outError = -1;
        } else {
            log_info( "WARNING: Unable to load data into GL texture (%s) format %s type %s internal format %s\n", gluErrorString( err ), GetGLFormatName( glFormat ), get_explicit_type_name( type ), GetGLFormatName( internalFormat ) );
            *outError = -2;
        }
        delete [] buffer;
        return NULL;
    }

    if( !allocateMem )
    {
        delete [] buffer;
        return NULL;
    }

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
    else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
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

        p[ i * 4 + 0 ] = uc1;
        p[ i * 4 + 1 ] = uc2;
        p[ i * 4 + 2 ] = uc3;
        p[ i * 4 + 3 ] = uc0;
      }
    }

    return buffer;
}

void * ReadGLTexture( GLenum glTarget, GLuint glTexture, GLuint glBuf, GLint width,
                        GLenum glFormat, GLenum glInternalFormat,
                        GLenum glType, ExplicitType typeToReadAs,
                        size_t outWidth, size_t outHeight )
{
    // Read results from the GL texture
    glBindTexture(get_base_gl_target(glTarget), glTexture);

    GLint realWidth, realHeight, realDepth;
    glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_WIDTH, &realWidth );
    glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_HEIGHT, &realHeight );
    glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_DEPTH, &realDepth );

    realDepth = (realDepth) ? realDepth : 1;

    GLint realInternalFormat;
    glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &realInternalFormat );

#ifdef DEBUG
    log_info( "- Reading back from GL: %d x %d : %s : %s : %s : %s (stored as %s)\n",
        realWidth, realHeight,
        GetGLTargetName( glTarget),
        GetGLFormatName( glInternalFormat ),
        GetGLFormatName( glFormat ),
        GetGLTypeName( glType ),
        GetGLFormatName( realInternalFormat ));
#endif

    GLenum readBackFormat;
    switch(glFormat)
    {
    case GL_RGBA_INTEGER_EXT:
      readBackFormat = GL_RGBA_INTEGER_EXT;
      break;
    case GL_DEPTH_COMPONENT:
      readBackFormat = GL_DEPTH_COMPONENT;
      break;
    case GL_DEPTH_STENCIL:
      readBackFormat = GL_DEPTH_STENCIL;
      break;
    default:
      readBackFormat = GL_RGBA;
      break;
    }

    GLenum readBackType;
    switch (glType) {
#ifdef __APPLE__
      case GL_UNSIGNED_INT_8_8_8_8:
      case GL_UNSIGNED_INT_8_8_8_8_REV:
        readBackType = GL_UNSIGNED_BYTE;
        break;
#endif
      case GL_HALF_FLOAT:
      case GL_UNSIGNED_BYTE:
      case GL_UNSIGNED_SHORT:
      case GL_UNSIGNED_INT:
      case GL_BYTE:
      case GL_SHORT:
      case GL_INT:
      case GL_FLOAT:
      default:
        readBackType = glType;
    }

    size_t outBytes;
    if (get_base_gl_target(glTarget) != GL_TEXTURE_BUFFER) {
        outBytes = realWidth * realHeight * realDepth * 4
          * GetGLTypeSize(readBackType);
    }
    else {
        outBytes = width * 4;

        outBytes *= ( GetGLTypeSize( GetGLTypeForExplicitType(typeToReadAs) ) );
    }

    cl_char *outBuffer = (cl_char *)malloc( outBytes );
    memset(outBuffer, 0, outBytes);

    if (get_base_gl_target(glTarget) != GL_TEXTURE_BUFFER) {
        //the default alignment in OpenGL is 4 bytes and need to be changed for GL_DEPTH_COMPONENT16 which is aligned to 2 bytes
        if (realInternalFormat == GL_DEPTH_COMPONENT16)
          glPixelStorei(GL_PACK_ALIGNMENT, 2);

        glGetTexImage( glTarget, 0, readBackFormat, readBackType, outBuffer );

        if (realInternalFormat == GL_DEPTH_COMPONENT16)
          glPixelStorei(GL_PACK_ALIGNMENT, 4);
    }
    else {
        glBindBuffer(GL_ARRAY_BUFFER, glBuf);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, outBytes, outBuffer);
    }

#ifdef DEBUG

    log_info( "- glGetTexImage: %s : %s : %s \n",
        GetGLTargetName( glTarget),
        GetGLFormatName(readBackFormat),
        GetGLTypeName(readBackType));

    DumpGLBuffer(readBackType, realWidth, realHeight, (void*)outBuffer);

#endif

    return (void *)outBuffer;
}

int CreateGLRenderbufferRaw( GLsizei width, GLsizei height,
                            GLenum attachment, GLenum glFormat,
                            GLenum internalFormat, GLenum glType,
                            GLuint *outFramebuffer,
                            GLuint *outRenderbuffer )
{
    GLenum err = 0;

    // Generate a renderbuffer and bind
    glGenRenderbuffersEXT( 1, outRenderbuffer );
    glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, *outRenderbuffer );

    // Allocate storage to the renderbuffer
    glGetError();
    glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, internalFormat, (GLsizei)width,  (GLsizei)height );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error("Failed to allocate render buffer storage!\n");
        return 1701;
    }

    GLint realInternalFormat;
    glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_INTERNAL_FORMAT_EXT, &realInternalFormat );
    internalFormat = realInternalFormat;

#ifdef DEBUG
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


void reorder_verification_buffer(GLenum glFormat, GLenum glType, char* buffer, size_t num_pixels)
{
  if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA)
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < num_pixels; i++ )
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
  else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA)
  {
    // Reverse and reorder to validate since in the
    // kernel the read_imagef() call always returns RGBA
    cl_uchar *p = (cl_uchar *)buffer;
    for( size_t i = 0; i < num_pixels; i++ )
    {
      cl_uchar uc0 = p[i * 4 + 0];
      cl_uchar uc1 = p[i * 4 + 1];
      cl_uchar uc2 = p[i * 4 + 2];
      cl_uchar uc3 = p[i * 4 + 3];

      p[ i * 4 + 0 ] = uc1;
      p[ i * 4 + 1 ] = uc2;
      p[ i * 4 + 2 ] = uc3;
      p[ i * 4 + 3 ] = uc0;
    }
  }
}


#ifdef GL_VERSION_3_2

#define CHECK_GL_ERROR()                                                       \
    {                                                                          \
        GLenum errnom = GL_NO_ERROR;                                           \
        if ((errnom = glGetError()) != GL_NO_ERROR)                            \
            log_error("GL Error: 0x%04X at %s:%d\n", errnom, __FILE__,         \
                      __LINE__);                                               \
    }

const char *get_gl_vector_type( GLenum internalformat )
{
  switch (internalformat) {
  case GL_RGBA8:
  case GL_RGBA16:
  case GL_RGBA32F_ARB:
  case GL_RGBA16F_ARB:
  case GL_DEPTH_COMPONENT16:
  case GL_DEPTH_COMPONENT32F:
  case GL_DEPTH24_STENCIL8:
  case GL_DEPTH32F_STENCIL8:
    return "vec4";
    break;
  case GL_RGBA8I_EXT:
  case GL_RGBA16I_EXT:
  case GL_RGBA32I_EXT:
    return "ivec4";
    break;
  case GL_RGBA8UI_EXT:
  case GL_RGBA16UI_EXT:
  case GL_RGBA32UI_EXT:
    return "uvec4";
    break;
  default:
    log_error("Test error: unsupported data type\n");
    return "";
    break;
  }
}

const char *get_gl_data_type( GLenum internalformat )
{
  switch (internalformat) {
  case GL_RGBA8:
  case GL_RGBA16:
  case GL_RGBA32F_ARB:
  case GL_RGBA16F_ARB:
  case GL_DEPTH_COMPONENT16:
  case GL_DEPTH_COMPONENT32F:
  case GL_DEPTH24_STENCIL8:
  case GL_DEPTH32F_STENCIL8:
    return "float";
    break;
  case GL_RGBA8I_EXT:
  case GL_RGBA16I_EXT:
  case GL_RGBA32I_EXT:
    return "int";
    break;
  case GL_RGBA8UI_EXT:
  case GL_RGBA16UI_EXT:
  case GL_RGBA32UI_EXT:
    return "uint";
    break;
  default:
    log_error("Test error: unsupported data type\n");
    return "";
    break;
  }
}


void * CreateGLTexture2DMultisample( size_t width, size_t height, size_t samples,
                                    GLenum target, GLenum glFormat,
                                    GLenum internalFormat, GLenum glType,
                                    ExplicitType type, GLuint *outTextureID,
                                    int *outError, bool allocateMem, MTdata d , bool fixedSampleLocations)
{
  // This function creates a multisample texture and renders into each sample
  // using a GLSL shader

  // Check if the renderer supports enough samples
  GLint max_samples = get_gl_max_samples(target, internalFormat);
  CHECK_GL_ERROR()

  if (max_samples < (GLint)samples)
      log_error("GL error: requested samples (%zu) exceeds renderer max "
                "samples (%d)\n",
                samples, max_samples);

  // Setup the GLSL program
  const GLchar *vertex_source =
  "#version 140\n"
  "in vec4 att0;\n"
  "void main (void) {\n"
  " gl_Position = vec4(att0.xy,0.0,1.0);\n"
  "}\n";

  const GLchar *fragmentSource =
  "#version 140\n"
    "out %s out0;\n"
    "uniform %s colorVal;\n"
    "uniform float depthVal;\n"
  "void main (void) {\n"
    "    out0 = %s(colorVal);\n"
    " gl_FragDepth = depthVal;\n"
  "}\n";

  GLchar fragmentShader[256];
  sprintf(fragmentShader, fragmentSource, get_gl_vector_type(internalFormat), get_gl_data_type(internalFormat), get_gl_vector_type(internalFormat));
  const GLchar *fragment_source = &fragmentShader[0];

  glShaderWrapper vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_source, NULL);
  glCompileShader(vertex_shader);
  CHECK_GL_ERROR()

  glShaderWrapper fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_source, NULL);
  glCompileShader(fragment_shader);
  CHECK_GL_ERROR()

  GLuint prog = glCreateProgram();
  glAttachShader(prog, vertex_shader);
  glAttachShader(prog, fragment_shader);
  CHECK_GL_ERROR()

  glBindAttribLocation(prog, 0, "att0");
  glLinkProgram(prog);
  CHECK_GL_ERROR()

  // Setup the FBO and texture
  glFramebufferWrapper fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  CHECK_GL_ERROR()

  glViewport(0, 0, width, height);
  CHECK_GL_ERROR()

  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
  glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, internalFormat, width, height, fixedSampleLocations);
  CHECK_GL_ERROR()

  GLint attachment;
  switch (internalFormat) {
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT32F:
    attachment = GL_DEPTH_ATTACHMENT;
    break;
    case GL_DEPTH24_STENCIL8:
    case GL_DEPTH32F_STENCIL8:
    attachment = GL_DEPTH_STENCIL_ATTACHMENT;
      break;
    default:
    attachment = GL_COLOR_ATTACHMENT0;
      break;
  }

  glFramebufferTexture(GL_FRAMEBUFFER, attachment, tex, 0);
  CHECK_GL_ERROR()

  GLint status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status == GL_FRAMEBUFFER_UNSUPPORTED) {
    log_info("GL status: GL_FRAMEBUFFER_UNSUPPORTED format %s multisample is not supported\n", GetGLFormatName(internalFormat));
    *outTextureID = 0;
    *outError = GL_FRAMEBUFFER_UNSUPPORTED;
    return NULL;
  }

  if (status != GL_FRAMEBUFFER_COMPLETE) {
    log_error("GL error: framebuffer incomplete status 0x%04X\n",status);
    *outTextureID = 0;
    *outError = status;
    return NULL;
  }

  // Check if the framebuffer supports enough samples
  GLint fbo_samples = 0;
  glGetIntegerv(GL_SAMPLES, &fbo_samples);
  CHECK_GL_ERROR();

  if (fbo_samples < (GLint)samples)
      log_error(
          "GL Error: requested samples (%zu) exceeds FBO capability (%d)\n",
          samples, fbo_samples);

  glUseProgram(prog);
  CHECK_GL_ERROR()

  if (attachment != GL_DEPTH_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT) {
    glDisable(GL_DEPTH_TEST);
    CHECK_GL_ERROR()
  }
  else {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    CHECK_GL_ERROR()
  }

  // Setup the VBO for rendering a quad
  GLfloat quad[] = {
    -1.0f, -1.0f,
    1.0f, -1.0f,
    1.0f,  1.0f,
    -1.0f,  1.0f
  };

  glBufferWrapper vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
  CHECK_GL_ERROR()

  glVertexArraysWrapper vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*2, 0);
  CHECK_GL_ERROR()

  //clearing color and depth buffer
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  glClearDepth(1.0);
  glClear(GL_DEPTH_BUFFER_BIT);

  //calculating colors
  double color_delta = 1.0 / samples;
  double color = color_delta;

  glEnable(GL_SAMPLE_MASK);
  for (size_t i=0;i!=samples;++i) {
    glSampleMaski(0, 1<<i);
    GLint colorUniformLocation = glGetUniformLocation(prog, "colorVal");
    switch (internalFormat) {
    case GL_RGBA8I_EXT:
      glUniform1i(colorUniformLocation, color * 0x7f);
      break;
    case GL_RGBA16I_EXT:
      glUniform1i(colorUniformLocation, color * 0x7fff);
      break;
    case GL_RGBA32I_EXT:
      glUniform1i(colorUniformLocation, color * 0x7fffffff);
      break;
    case GL_RGBA8UI_EXT:
      glUniform1ui(colorUniformLocation, color * 0xff);
      break;
    case GL_RGBA16UI_EXT:
      glUniform1ui(colorUniformLocation, color * 0xffff);
      break;
    case GL_RGBA32UI_EXT:
      glUniform1ui(colorUniformLocation, color * 0xffffffff);
      break;
    default:
      glUniform1f(colorUniformLocation, color);
      break;
    }

    glUniform1f(glGetUniformLocation(prog, "depthVal"), color);
    color += color_delta;

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    CHECK_GL_ERROR();

    glFlush();
  }

  glDisable(GL_SAMPLE_MASK);
  CHECK_GL_ERROR();

  *outTextureID = tex;

  // Create an output buffer containing the expected results.
  unsigned int size = 0;
  if ( glType == GL_FLOAT_32_UNSIGNED_INT_24_8_REV )
  {
    size = width * height * 2;
  }
  else if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) || (attachment == GL_DEPTH_ATTACHMENT) || (attachment == GL_DEPTH_STENCIL_ATTACHMENT))
  {
    size = width * height;
  }
  else
  {
    size = width * height * 4;
  }

  char *data = (char *)malloc(get_explicit_type_size(type) * size * samples);
  char *p = data;
  size_t stride = get_explicit_type_size(type);

  for (size_t s=0;s!=samples;++s) {
    double val = color_delta + (color_delta * s);
    for (size_t i=0;i!=size;++i) {
      switch (type) {
      case kChar:
        *((char*)p) = val * 0x7f;
        break;
        case kUChar:
        *((unsigned char*)p) = val * 0xff;
          break;
        case kFloat:
          *((float*)p) = val;
          break;
      case kShort:
        *((short*)p) = val*0x7fff;
        break;
        case kUShort:
        *((unsigned short*)p) = val*0xffff;
          break;
      case kInt:
        *((int*)p) = val*0x7fffffff;
          break;
        case kUInt:
        *((unsigned int*)p) = val*0xffffffff;
          break;
        case kHalf: *((cl_half *)p) = convert_float_to_half(val); break;
        default:
          log_error("Test error: unexpected type enum 0x%x\n",type);
      }

      p += stride;
    }
  }


  if (allocateMem)
    reorder_verification_buffer(glFormat,glType,data,width*height*samples);

  return data;
}

void * CreateGLTexture2DArrayMultisample(size_t width, size_t height,
                              size_t total_layers, size_t samples,
                              GLenum target, GLenum glFormat, GLenum internalFormat, GLenum glType,
                              ExplicitType type, GLuint *outTextureID, int *outError,
                              bool allocateMem, MTdata d, bool fixedSampleLocations)
{
  // This function creates a multisample texture and renders into each sample
  // using a GLSL shader

  // Check if the renderer supports enough samples
  GLint max_samples = get_gl_max_samples(target, internalFormat);

  if (max_samples < (GLint)samples)
      log_error("GL error: requested samples (%zu) exceeds renderer max "
                "samples (%d)\n",
                samples, max_samples);

  // Setup the GLSL program
  const GLchar *vertex_source =
  "#version 140\n"
  "in vec4 att0;\n"
  "void main (void) {\n"
  " gl_Position = vec4(att0.xy,0.0,1.0);\n"
  "}\n";

  const GLchar *fragmentSource =
  "#version 140\n"
    "out %s out0;\n"
    "uniform %s colorVal;\n"
    "uniform float depthVal;\n"
  "void main (void) {\n"
    "    out0 = %s(colorVal);\n"
    " gl_FragDepth = depthVal;\n"
  "}\n";

  GLchar fragmentShader[256];
  sprintf(fragmentShader, fragmentSource, get_gl_vector_type(internalFormat), get_gl_data_type(internalFormat), get_gl_vector_type(internalFormat));
  const GLchar *fragment_source = &fragmentShader[0];

  glShaderWrapper vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_source, NULL);
  glCompileShader(vertex_shader);
  CHECK_GL_ERROR()

  glShaderWrapper fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_source, NULL);
  glCompileShader(fragment_shader);
  CHECK_GL_ERROR()

  glProgramWrapper prog = glCreateProgram();
  glAttachShader(prog, vertex_shader);
  glAttachShader(prog, fragment_shader);
  CHECK_GL_ERROR()

  glBindAttribLocation(prog, 0, "att0");
  glLinkProgram(prog);
  CHECK_GL_ERROR()

  // Setup the FBO and texture
  glFramebufferWrapper fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  CHECK_GL_ERROR()

  glViewport(0, 0, width, height);
  CHECK_GL_ERROR()

  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, tex);
  glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, samples, internalFormat, width, height, total_layers, fixedSampleLocations);
  CHECK_GL_ERROR()

  GLint attachment;
  switch (internalFormat) {
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT32F:
    attachment = GL_DEPTH_ATTACHMENT;
    break;
    case GL_DEPTH24_STENCIL8:
    case GL_DEPTH32F_STENCIL8:
    attachment = GL_DEPTH_STENCIL_ATTACHMENT;
      break;
    default:
    attachment = GL_COLOR_ATTACHMENT0;
    break;
  }

  //calculating colors
  double color_delta = 1.0 / (total_layers * samples);

  if (attachment != GL_DEPTH_ATTACHMENT && attachment != GL_DEPTH_STENCIL_ATTACHMENT) {
    glDisable(GL_DEPTH_TEST);
    CHECK_GL_ERROR()
  }
  else {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    CHECK_GL_ERROR()
  }

  // Setup the VBO for rendering a quad
  GLfloat quad[] = {
    -1.0f, -1.0f,
    1.0f, -1.0f,
    1.0f,  1.0f,
    -1.0f,  1.0f
  };

  glBufferWrapper vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
  CHECK_GL_ERROR()

  glVertexArraysWrapper vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*2, 0);
  CHECK_GL_ERROR()

  for (size_t l=0; l!=total_layers; ++l) {
    glFramebufferTextureLayer(GL_FRAMEBUFFER, attachment, tex, 0, l);
    CHECK_GL_ERROR()

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status == GL_FRAMEBUFFER_UNSUPPORTED) {
      log_info("GL status: GL_FRAMEBUFFER_UNSUPPORTED format %s multisample array is not supported\n", GetGLFormatName(internalFormat));
      *outTextureID = 0;
      *outError = GL_FRAMEBUFFER_UNSUPPORTED;
      return NULL;
    }

    if (status != GL_FRAMEBUFFER_COMPLETE) {
      log_error("GL error: framebuffer incomplete status 0x%04X\n",status);
      *outTextureID = 0;
      *outError = status;
      return NULL;
    }

    // Check if the framebuffer supports enough samples
    GLint fbo_samples = 0;
    glGetIntegerv(GL_SAMPLES, &fbo_samples);
    CHECK_GL_ERROR();

    if (fbo_samples < (GLint)samples)
        log_error(
            "GL Error: requested samples (%zu) exceeds FBO capability (%d)\n",
            samples, fbo_samples);

    glUseProgram(prog);
    CHECK_GL_ERROR()

    //clearing color and depth buffer
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);

    glEnable(GL_SAMPLE_MASK);
    for (size_t s=0;s!=samples;++s) {
      double val = color_delta + color_delta * (l * samples + s);

      glSampleMaski(0, 1<<s);
      GLint colorUniformLocation = glGetUniformLocation(prog, "colorVal");
      switch (internalFormat) {
      case GL_RGBA8I_EXT:
        glUniform1i(colorUniformLocation, val * 0x7f);
        break;
      case GL_RGBA16I_EXT:
        glUniform1i(colorUniformLocation, val * 0x7fff);
        break;
      case GL_RGBA32I_EXT:
        glUniform1i(colorUniformLocation, val * 0x7fffffff);
        break;
      case GL_RGBA8UI_EXT:
        glUniform1ui(colorUniformLocation, val * 0xff);
        break;
      case GL_RGBA16UI_EXT:
        glUniform1ui(colorUniformLocation, val * 0xffff);
        break;
      case GL_RGBA32UI_EXT:
        glUniform1ui(colorUniformLocation, val * 0xffffffff);
        break;
      default:
        glUniform1f(colorUniformLocation, val);
        break;
      }

      glUniform1f(glGetUniformLocation(prog, "depthVal"), val);

      glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
      CHECK_GL_ERROR();

      glFlush();
    }

    glDisable(GL_SAMPLE_MASK);
    CHECK_GL_ERROR();
  }

  *outTextureID = tex;

  // Create an output buffer containing the expected results.
  unsigned int size = 0;
  if ( glType == GL_FLOAT_32_UNSIGNED_INT_24_8_REV )
  {
    size = width * height * 2;
  }
  else if ( (glType == GL_UNSIGNED_INT_2_10_10_10_REV) || (glType == GL_UNSIGNED_INT_10_10_10_2) || (attachment == GL_DEPTH_ATTACHMENT) || (attachment == GL_DEPTH_STENCIL_ATTACHMENT))
  {
    size = width * height;
  }
  else
  {
    size = width * height * 4;
  }

  char *data = (char *)malloc(get_explicit_type_size(type) * size * total_layers * samples);
  char *p = data;
  size_t stride = get_explicit_type_size(type);

  for (size_t s=0;s!=samples;++s) {
    for (size_t l=0;l!=total_layers;++l) {
      double val = color_delta + color_delta * (l * samples + s);
    for (size_t i=0;i!=size;++i) {
      switch (type) {
        case kChar:
          *((char*)p) = val * 0x7f;
          break;
        case kUChar:
          *((unsigned char*)p) = val*0xff;
          break;
        case kFloat:
          *((float*)p) = val;
          break;
        case kShort:
          *((short*)p) = val * 0x7fff;
          break;
        case kUShort:
          *((unsigned short*)p) = val * 0xffff;
          break;
        case kInt:
          *((int*)p) = val * 0x7fffffff;
          break;
        case kUInt:
          *((unsigned int*)p) = val*0xffffffff;
          break;
        case kHalf: *((cl_half *)p) = convert_float_to_half(val); break;
        default:
          log_error("Test error: unexpected type enum 0x%x\n",type);
      }

      p += stride;
    }
  }
  }

  if (allocateMem)
    reorder_verification_buffer(glFormat,glType,data,width*height*samples*total_layers);

  return data;
}

#endif // GL_VERSION_3_2

void * CreateGLRenderbuffer( GLsizei width, GLsizei height,
                             GLenum attachment, GLenum glFormat,
                             GLenum internalFormat, GLenum glType,
                             ExplicitType type,
                             GLuint *outFramebuffer,
                             GLuint *outRenderbuffer,
                             int *outError, MTdata d, bool allocateMem )
{
    *outError = CreateGLRenderbufferRaw( width, height, attachment, glFormat, internalFormat,
                            glType, outFramebuffer, outRenderbuffer );

    if( *outError != 0 )
        return NULL;

    GLenum err = 0;

    // Generate a renderbuffer and bind
    glGenRenderbuffersEXT( 1, outRenderbuffer );
    glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, *outRenderbuffer );

    // Allocate storage to the renderbuffer
    glGetError();
    glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, internalFormat, (GLsizei)width,  (GLsizei)height );
    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        *outError = 1701;
        log_error("Failed to allocate render buffer storage!\n");
        return NULL;
    }

    GLint realInternalFormat;
    glGetRenderbufferParameterivEXT( GL_RENDERBUFFER_EXT, GL_RENDERBUFFER_INTERNAL_FORMAT_EXT, &realInternalFormat );
    internalFormat = realInternalFormat;

#ifdef DEBUG
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

        *outError = -1;
        return NULL;
    }

    // Attach to the framebuffer
    glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, attachment, GL_RENDERBUFFER_EXT, *outRenderbuffer );
    err = glGetError();
    GLint status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
    if( status != GL_FRAMEBUFFER_COMPLETE_EXT )
    {
        *outError = -1;
        log_error( "ERROR: Unable to attach renderbuffer to framebuffer (%s, status %x)\n", gluErrorString( err ), (int)status );
        return NULL;
    }

    void* buffer = CreateRandomData(type, width * height * 4, d);

#ifdef DEBUG
    log_info( "- Fillling renderbuffer: %d : %d : %s : %s \n",
             (int)width, (int)height,
             GetGLFormatName(glFormat),
             GetGLTypeName(glType));

    DumpGLBuffer(glType, (int)width, (int)height, (void*)buffer);
#endif

    // Fill a texture with our input data
    glTextureWrapper texture;
    glGenTextures( 1, &texture );
    glBindTexture( GL_TEXTURE_RECTANGLE_ARB, texture );
    glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_RECTANGLE_ARB, 0, internalFormat, width, height, 0, glFormat, glType, buffer );
    glEnable( GL_TEXTURE_RECTANGLE_ARB );

    // Render fullscreen textured quad
    glDisable( GL_LIGHTING );
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode( GL_TEXTURE );
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT);
    gluOrtho2D( -1.0, 1.0, -1.0, 1.0 );
    glMatrixMode( GL_MODELVIEW );
    glBegin( GL_QUADS );
    {
        glColor3f(1.0f, 1.0f, 1.0f);
        glTexCoord2f( 0.0f, 0.0f );
        glVertex3f( -1.0f, -1.0f, 0.0f );
        glTexCoord2f( 0.0f, height );
        glVertex3f( -1.0f, 1.0f, 0.0f );
        glTexCoord2f( width, height );
        glVertex3f( 1.0f, 1.0f, 0.0f );
        glTexCoord2f( width, 0.0f );
        glVertex3f( 1.0f, -1.0f, 0.0f );
    }
    glEnd();
    glBindTexture( GL_TEXTURE_RECTANGLE_ARB, 0 );
    glDisable( GL_TEXTURE_RECTANGLE_ARB );

    glFlush();

    // Read back the data in the renderbuffer
    memset(buffer, 0, width * height * 4 * get_explicit_type_size( type ));
    glReadBuffer( attachment );

    glReadPixels( 0, 0, (GLsizei)width, (GLsizei)height, glFormat, glType, buffer );

    err = glGetError();
    if( err != GL_NO_ERROR )
    {
        log_error( "ERROR: Unable to read data via glReadPixels : %d : %d : %s : %s : Error %s\n",
                  (int)width, (int)height,
                  GetGLFormatName(glFormat),
                  GetGLTypeName(glType),
                  gluErrorString( err ));
        *outError = -1;
    }

#ifdef DEBUG
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

    if( glType == GL_UNSIGNED_INT_8_8_8_8_REV && glFormat == GL_BGRA && allocateMem )
    {
        // Reverse and reorder to validate since in the
        // kernel the read_imagef() call always returns RGBA
        cl_uchar *p = (cl_uchar *)buffer;
        for (GLsizei i = 0; i < width * height; i++)
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
    else if( glType == GL_UNSIGNED_INT_8_8_8_8 && glFormat == GL_BGRA && allocateMem )
    {
      // Reverse and reorder to validate since in the
      // kernel the read_imagef() call always returns RGBA
      cl_uchar *p = (cl_uchar *)buffer;
      for (GLsizei i = 0; i < width * height; i++)
      {
        cl_uchar uc0 = p[i * 4 + 0];
        cl_uchar uc1 = p[i * 4 + 1];
        cl_uchar uc2 = p[i * 4 + 2];
        cl_uchar uc3 = p[i * 4 + 3];

        p[ i * 4 + 0 ] = uc1;
        p[ i * 4 + 1 ] = uc2;
        p[ i * 4 + 2 ] = uc3;
        p[ i * 4 + 3 ] = uc0;
      }
    }

    return buffer;
}

void * ReadGLRenderbuffer( GLuint glFramebuffer, GLuint glRenderbuffer,
                           GLenum attachment, GLenum glFormat,
                           GLenum glInternalFormat, GLenum glType,
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
#ifdef DEBUG
    log_info( "- Reading back from GL: %d x %d : %s : %s : %s\n",
             (int)outWidth, (int)outHeight,
             GetGLFormatName( glInternalFormat ),
             GetGLFormatName( glFormat ),
             GetGLTypeName( glType ));
#endif

    GLenum readBackFormat = glFormat == GL_RGBA_INTEGER_EXT ? GL_RGBA_INTEGER_EXT : GL_RGBA;
    GLenum readBackType = glType;

    size_t outBytes = outWidth * outHeight * 4 * GetGLTypeSize(readBackType);
    void *outBuffer = malloc( outBytes );
    memset(outBuffer, 0, outBytes);

    glReadPixels( 0, 0, (GLsizei)outWidth, (GLsizei)outHeight, readBackFormat, readBackType, outBuffer );

#ifdef DEBUG
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
        case GL_RGBA8:
        case GL_RGBA16:
        case GL_RGBA32F_ARB:
            glFormat = GL_RGBA;
            break;
        case GL_RGBA8I_EXT:
        case GL_RGBA16I_EXT:
        case GL_RGBA32I_EXT:
        case GL_RGBA8UI_EXT:
        case GL_RGBA16UI_EXT:
        case GL_RGBA32UI_EXT:
            glFormat = GL_RGBA_INTEGER_EXT;
            break;
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
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8:
            return sizeof(GLuint);
        case GL_SHORT:
            return sizeof(GLshort);
        case GL_UNSIGNED_SHORT:
            return sizeof(GLushort);
        case GL_UNSIGNED_INT_8_8_8_8_REV:
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
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 2 * sizeof(GLfloat);
        default:
            log_error("Unknown type 0x%x\n",type);
            return 0;
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
        case GL_UNSIGNED_INT_8_8_8_8:        return "GL_UNSIGNED_INT_8_8_8_8";
        case GL_UNSIGNED_INT_8_8_8_8_REV:    return "GL_UNSIGNED_INT_8_8_8_8_REV";
        case GL_UNSIGNED_INT_10_10_10_2:     return "GL_UNSIGNED_INT_10_10_10_2";
        case GL_UNSIGNED_INT_2_10_10_10_REV: return "GL_UNSIGNED_INT_2_10_10_10_REV";
#ifdef GL_VERSION_3_2
        case GL_UNSIGNED_INT_24_8: return "GL_UNSIGNED_INT_24_8";
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: return "GL_FLOAT_32_UNSIGNED_INT_24_8_REV";
#endif
        default:
        {
            static char foo[ 128 ];
            sprintf( foo, "0x%04x", (int)type);
            return foo;
        }
    }
}

const char *GetGLTargetName( GLenum tgt )
{
    if( tgt == GL_TEXTURE_BUFFER ) return "GL_TEXTURE_BUFFER";
    if( tgt == GL_TEXTURE_1D ) return "GL_TEXTURE_1D";
    if( tgt == GL_TEXTURE_2D ) return "GL_TEXTURE_2D";
    if( tgt == GL_TEXTURE_3D ) return "GL_TEXTURE_3D";
    if( tgt == GL_TEXTURE_RECTANGLE_EXT ) return "GL_TEXTURE_RECTANGLE_EXT";
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_X ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_X";
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_Y ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_Y";
    if( tgt == GL_TEXTURE_CUBE_MAP_POSITIVE_Z ) return "GL_TEXTURE_CUBE_MAP_POSITIVE_Z";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_X ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_X";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_Y ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Y";
    if( tgt == GL_TEXTURE_CUBE_MAP_NEGATIVE_Z ) return "GL_TEXTURE_CUBE_MAP_NEGATIVE_Z";
    if( tgt == GL_TEXTURE_2D_MULTISAMPLE ) return "GL_TEXTURE_2D_MULTISAMPLE";
    if( tgt == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ) return "GL_TEXTURE_2D_MULTISAMPLE_ARRAY";

    static char foo[ 128 ];
    sprintf( foo, "0x%04x", (int)tgt);
    return foo;
}

const char *GetGLAttachmentName( GLenum att )
{
    if( att == GL_COLOR_ATTACHMENT0_EXT ) return "GL_COLOR_ATTACHMENT0_EXT";
    if( att == GL_COLOR_ATTACHMENT1_EXT ) return "GL_COLOR_ATTACHMENT1_EXT";
    if( att == GL_COLOR_ATTACHMENT2_EXT ) return "GL_COLOR_ATTACHMENT2_EXT";
    if( att == GL_COLOR_ATTACHMENT3_EXT ) return "GL_COLOR_ATTACHMENT3_EXT";
    if( att == GL_COLOR_ATTACHMENT4_EXT ) return "GL_COLOR_ATTACHMENT4_EXT";
    if( att == GL_COLOR_ATTACHMENT5_EXT ) return "GL_COLOR_ATTACHMENT5_EXT";
    if( att == GL_COLOR_ATTACHMENT6_EXT ) return "GL_COLOR_ATTACHMENT6_EXT";
    if( att == GL_COLOR_ATTACHMENT7_EXT ) return "GL_COLOR_ATTACHMENT7_EXT";
    if( att == GL_COLOR_ATTACHMENT8_EXT ) return "GL_COLOR_ATTACHMENT8_EXT";
    if( att == GL_DEPTH_ATTACHMENT_EXT ) return "GL_DEPTH_ATTACHMENT_EXT";
    return "";
}
const char *GetGLBaseFormatName( GLenum baseformat )
{
    switch( baseformat )
    {
        case GL_RGBA8:          return "GL_RGBA";
        case GL_RGBA16:         return "GL_RGBA";
        case GL_RGBA:           return "GL_RGBA";
        case GL_BGRA:           return "GL_BGRA";
        case GL_RGBA8I_EXT:     return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA16I_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32I_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA8UI_EXT:    return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA16UI_EXT:   return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32UI_EXT:   return "GL_RGBA_INTEGER_EXT";
        case GL_RGBA32F_ARB:    return "GL_RGBA";

        case GL_RGBA_INTEGER_EXT:   return "GL_RGBA_INTEGER_EXT";

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
#ifdef GL_VERSION_3_2
        case GL_DEPTH_COMPONENT: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_COMPONENT16: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_COMPONENT24: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_COMPONENT32: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_COMPONENT32F: return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_STENCIL:      return "GL_DEPTH_STENCIL";
        case GL_DEPTH24_STENCIL8:   return "GL_DEPTH_STENCIL";
        case GL_DEPTH32F_STENCIL8:  return "GL_DEPTH_STENCIL";
#endif
        default:
        {
            static char foo[ 128 ];
            sprintf( foo, "0x%04x", (int)baseformat );
            return foo;
        }
    }
}

const char *GetGLFormatName( GLenum format )
{
    switch( format )
    {
        case GL_RGBA8:          return "GL_RGBA8";
        case GL_RGBA16:         return "GL_RGBA16";
        case GL_RGBA:           return "GL_RGBA";
        case GL_BGRA:           return "GL_BGRA";
        case GL_RGBA8I_EXT:     return "GL_RGBA8I_EXT";
        case GL_RGBA16I_EXT:    return "GL_RGBA16I_EXT";
        case GL_RGBA32I_EXT:    return "GL_RGBA32I_EXT";
        case GL_RGBA8UI_EXT:    return "GL_RGBA8UI_EXT";
        case GL_RGBA16UI_EXT:   return "GL_RGBA16UI_EXT";
        case GL_RGBA32UI_EXT:   return "GL_RGBA32UI_EXT";
        case GL_RGBA16F:        return "GL_RGBA16F";
        case GL_RGBA32F:        return "GL_RGBA32F";

        case GL_RGBA_INTEGER_EXT:   return "GL_RGBA_INTEGER_EXT";

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

        case GL_INT:            return "GL_INT";
        case GL_UNSIGNED_INT:   return "GL_UNSIGNED_INT";
        case GL_SHORT:          return "GL_SHORT";
        case GL_UNSIGNED_SHORT: return "GL_UNSIGNED_SHORT";
        case GL_BYTE:           return "GL_BYTE";
        case GL_UNSIGNED_BYTE:  return "GL_UNSIGNED_BYTE";
        case GL_FLOAT:          return "GL_FLOAT";
#if defined( __APPLE__ )
        case GL_HALF_FLOAT:     return "GL_HALF_FLOAT";
#else
        case GL_HALF_FLOAT_ARB: return "GL_HALF_FLOAT_ARB";
#endif
#ifdef GL_VERSION_3_2
        case GL_DEPTH_STENCIL:      return "GL_DEPTH_STENCIL";
        case GL_DEPTH_COMPONENT:    return "GL_DEPTH_COMPONENT";
        case GL_DEPTH_COMPONENT16:  return "GL_DEPTH_COMPONENT16";
        case GL_DEPTH_COMPONENT24:  return "GL_DEPTH_COMPONENT24";
        case GL_DEPTH_COMPONENT32:  return "GL_DEPTH_COMPONENT32";
        case GL_DEPTH_COMPONENT32F: return "GL_DEPTH_COMPONENT32F";
        case GL_DEPTH24_STENCIL8:   return "GL_DEPTH24_STENCIL8";
        case GL_DEPTH32F_STENCIL8:  return "GL_DEPTH32F_STENCIL8";
#endif
        default:
        {
            static char foo[ 128 ];
            sprintf( foo, "0x%04x", (int)format);
            return foo;
        }
    }
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
        case (kUnsignedChar):
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
        case (kUnsignedShort):
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
        case (kUnsignedInt):
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
        case (kHalf):
        {
            cl_half *p = (cl_half *)malloc(count * sizeof(cl_half));
            if(!p) return 0;

            for( size_t i = 0; i < count; i++ )
            {
                p[ i ] = convert_float_to_half( get_random_float( 0.f, 1.f, d ) );
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
clCreateFromGLTexture2D_fn clCreateFromGLTexture2D_ptr;
clCreateFromGLTexture3D_fn clCreateFromGLTexture3D_ptr;
clCreateFromGLRenderbuffer_fn clCreateFromGLRenderbuffer_ptr;
clGetGLObjectInfo_fn clGetGLObjectInfo_ptr;
clGetGLTextureInfo_fn clGetGLTextureInfo_ptr;
clEnqueueAcquireGLObjects_fn clEnqueueAcquireGLObjects_ptr;
clEnqueueReleaseGLObjects_fn clEnqueueReleaseGLObjects_ptr;

int init_clgl_ext() {

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

    // Create the function pointer table
    clCreateFromGLBuffer_ptr = (clCreateFromGLBuffer_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromGLBuffer");
    if (clCreateFromGLBuffer_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clCreateFromGLBuffer) returned NULL.\n");
        return -1;
    }
    clCreateFromGLTexture2D_ptr = (clCreateFromGLTexture2D_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromGLTexture2D");
    if (clCreateFromGLTexture2D_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clCreateFromGLTexture2D) returned NULL.\n");
        return -1;
    }
    clCreateFromGLTexture3D_ptr = (clCreateFromGLTexture3D_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromGLTexture3D");
    if (clCreateFromGLTexture3D_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clCreateFromGLTexture3D\") returned NULL.\n");
        return -1;
    }
    clCreateFromGLTexture_ptr = (clCreateFromGLTexture_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromGLTexture");
    if (clCreateFromGLTexture_ptr == NULL) {
         log_error("clGetExtensionFunctionAddressForPlatform(platform,\"clCreateFromGLTexture\") returned NULL.\n");
         return -1;
    }
    clCreateFromGLRenderbuffer_ptr = (clCreateFromGLRenderbuffer_fn)clGetExtensionFunctionAddressForPlatform(platform,"clCreateFromGLRenderbuffer");
    if (clCreateFromGLRenderbuffer_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clCreateFromGLRenderbuffer) returned NULL.\n");
        return -1;
    }
    clGetGLObjectInfo_ptr = (clGetGLObjectInfo_fn)clGetExtensionFunctionAddressForPlatform(platform,"clGetGLObjectInfo");
    if (clGetGLObjectInfo_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clGetGLObjectInfo) returned NULL.\n");
        return -1;
    }
    clGetGLTextureInfo_ptr = (clGetGLTextureInfo_fn)clGetExtensionFunctionAddressForPlatform(platform,"clGetGLTextureInfo");
    if (clGetGLTextureInfo_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clGetGLTextureInfo) returned NULL.\n");
        return -1;
    }
    clEnqueueAcquireGLObjects_ptr = (clEnqueueAcquireGLObjects_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueAcquireGLObjects");
    if (clEnqueueAcquireGLObjects_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clEnqueueAcquireGLObjects) returned NULL.\n");
        return -1;
    }
    clEnqueueReleaseGLObjects_ptr = (clEnqueueReleaseGLObjects_fn)clGetExtensionFunctionAddressForPlatform(platform,"clEnqueueReleaseGLObjects");
    if (clEnqueueReleaseGLObjects_ptr == NULL) {
        log_error("clGetExtensionFunctionAddressForPlatform(platform,clEnqueueReleaseGLObjects) returned NULL.\n");
        return -1;
    }

    return 0;
}

GLint get_gl_max_samples( GLenum target, GLenum internalformat )
{
    GLint max_samples = 0;
#ifdef GL_VERSION_4_2
    glGetInternalformativ(target, internalformat, GL_SAMPLES, 1, &max_samples);
#else
    switch (internalformat)
    {
        case GL_RGBA8I:
        case GL_RGBA16I:
        case GL_RGBA32I:
        case GL_RGBA8UI:
        case GL_RGBA16UI:
        case GL_RGBA32UI:
            glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &max_samples);
            break;
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            glGetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES, &max_samples);
            break;
        default:
            glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &max_samples);
            break;
    }
#endif
    return max_samples;
}
