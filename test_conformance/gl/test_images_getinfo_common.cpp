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
#include "testBase.h"
#include "common.h"

#if defined( __APPLE__ )
    #include <OpenGL/glu.h>
#else
    #include <GL/glu.h>
    #include <CL/cl_gl.h>
#endif

extern int supportsHalf(cl_context context, bool* supports_half);

static int test_image_info( cl_context context, cl_command_queue queue,
  GLenum glTarget, GLuint glTexture, size_t imageWidth, size_t imageHeight,
  size_t imageDepth, cl_image_format *outFormat, ExplicitType *outType,
  void **outResultBuffer )
{
  clMemWrapper streams[ 2 ];

  int error;

  // Create a CL image from the supplied GL texture
  streams[ 0 ] = (*clCreateFromGLTexture_ptr)( context, CL_MEM_READ_ONLY,
    glTarget, 0, glTexture, &error );
  if( error != CL_SUCCESS )
  {
    print_error( error, "Unable to create CL image from GL texture" );
    GLint fmt;
    glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt );
    log_error( "    Supplied GL texture was format %s\n", GetGLFormatName( fmt ) );
    return error;
  }

  // Determine data type and format that CL came up with
  error = clGetImageInfo( streams[ 0 ], CL_IMAGE_FORMAT,
    sizeof( cl_image_format ), outFormat, NULL );
  test_error( error, "Unable to get CL image format" );

  cl_gl_object_type object_type;
  switch (glTarget) {
    case GL_TEXTURE_1D:
      object_type = CL_GL_OBJECT_TEXTURE1D;
      break;
    case GL_TEXTURE_BUFFER:
      object_type = CL_GL_OBJECT_TEXTURE_BUFFER;
      break;
    case GL_TEXTURE_1D_ARRAY:
      object_type = CL_GL_OBJECT_TEXTURE1D_ARRAY;
      break;
    case GL_TEXTURE_2D:
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
      object_type = CL_GL_OBJECT_TEXTURE2D;
      break;
    case GL_TEXTURE_2D_ARRAY:
      object_type = CL_GL_OBJECT_TEXTURE2D_ARRAY;
      break;
    case GL_TEXTURE_3D:
      object_type = CL_GL_OBJECT_TEXTURE3D;
      break;
    default:
      log_error("Unsupported texture target.");
      return 1;
  }

  return CheckGLObjectInfo(streams[0], object_type, glTexture, glTarget, 0);
}

static int test_image_format_get_info(
    cl_context context, cl_command_queue queue,
    size_t width, size_t height, size_t depth,
    GLenum target, struct format* fmt, MTdata data)
{
  int error = 0;

  // If we're testing a half float format, then we need to determine the
  // rounding mode of this machine.  Punt if we fail to do so.

  if( fmt->type == kHalf )
  {
    if( DetectFloatToHalfRoundingMode(queue) )
      return 0;
    bool supports_half = false;
    error = supportsHalf(context, &supports_half);
    if( error != 0 )
      return error;
    if (!supports_half) return 0;
  }

  size_t w = width, h = height, d = depth;

  // Unpack the format and use it, along with the target, to create an
  // appropriate GL texture.

  GLenum gl_fmt          = fmt->formattype;
  GLenum gl_internal_fmt = fmt->internal;
  GLenum gl_type         = fmt->datatype;
  ExplicitType type      = fmt->type;

  glTextureWrapper texture;
  glBufferWrapper glbuf;

  // If we're testing a half float format, then we need to determine the
  // rounding mode of this machine.  Punt if we fail to do so.

  if( fmt->type == kHalf )
    if( DetectFloatToHalfRoundingMode(queue) )
      return 1;

  // Use the correct texture creation function depending on the target, and
  // adjust width, height, depth as appropriate so subsequent size calculations
  // succeed.

  switch (target) {
    case GL_TEXTURE_1D:
      h = 1; d = 1;
      CreateGLTexture1D( width, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, false, data );
      break;
    case GL_TEXTURE_BUFFER:
      h = 1; d = 1;
      CreateGLTextureBuffer( width, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &glbuf, &error, false, data );
      break;
    case GL_TEXTURE_1D_ARRAY:
      d = 1;
      CreateGLTexture1DArray( width, height, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, false, data );
      break;
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_2D:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
      d = 1;
      CreateGLTexture2D( width, height, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, false, data );
      break;
    case GL_TEXTURE_2D_ARRAY:
      CreateGLTexture2DArray( width, height, depth, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, false, data );
      break;
    case GL_TEXTURE_3D:
      d = 1;
      CreateGLTexture3D( width, height, depth, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, data, false );
      break;
    default:
      log_error("Unsupported texture target.\n");
      return 1;
  }

  if ( error == -2 ) {
    log_info("OpenGL texture couldn't be created, because a texture is too big. Skipping test.\n");
    return 0;
  }

  if ( error != 0 ) {
    if ((gl_fmt == GL_RGBA_INTEGER_EXT) && (!CheckGLIntegerExtensionSupport())) {
      log_info("OpenGL version does not support GL_RGBA_INTEGER_EXT. "
        "Skipping test.\n");
      return 0;
    } else {
      return error;
    }
  }

  cl_image_format clFormat;
  ExplicitType actualType;
  char *outBuffer;

  // Perform the info check:
  return test_image_info( context, queue, target, texture, w, h, d, &clFormat,
    &actualType, (void **)&outBuffer );
}

int test_images_get_info_common( cl_device_id device, cl_context context,
  cl_command_queue queue, struct format* formats, size_t nformats,
  GLenum *targets, size_t ntargets, sizevec_t *sizes, size_t nsizes )
{
  int error = 0;
  RandomSeed seed(gRandomSeed);

  // First, ensure this device supports images.

  if (checkForImageSupport(device)) {
    log_info("Device does not support images.  Skipping test.\n");
    return 0;
  }

  size_t fidx, tidx, sidx;

  // Test each format on every target, every size.

  for ( fidx = 0; fidx < nformats; fidx++ ) {
    for ( tidx = 0; tidx < ntargets; tidx++ ) {

      if ( formats[ fidx ].datatype == GL_UNSIGNED_INT_2_10_10_10_REV )
      {
        // Check if the RGB 101010 format is supported
        if ( is_rgb_101010_supported( context, targets[ tidx ] ) == 0 )
          break; // skip
      }

      log_info( "Testing image info for GL format %s : %s : %s : %s\n",
        GetGLTargetName( targets[ tidx ] ),
        GetGLFormatName( formats[ fidx ].internal ),
        GetGLBaseFormatName( formats[ fidx ].formattype ),
        GetGLTypeName( formats[ fidx ].datatype ) );

      for ( sidx = 0; sidx < nsizes; sidx++ ) {

        // Test this format + size:

        if ( test_image_format_get_info(context, queue,
                                        sizes[sidx].width, sizes[sidx].height, sizes[sidx].depth,
                                        targets[tidx], &formats[fidx], seed) )
        {
          // We land here in the event of test failure.

          log_error( "ERROR: Image info test failed for %s : %s : %s : %s\n\n",
            GetGLTargetName( targets[ tidx ] ),
            GetGLFormatName( formats[ fidx ].internal ),
            GetGLBaseFormatName( formats[ fidx ].formattype ),
            GetGLTypeName( formats[ fidx ].datatype ) );
          error++;

          // Skip the other sizes for this format.

          break;
        }
      }
    }
  }

  return error;
}
