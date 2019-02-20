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
#include "common.h"
#include "testBase.h"

#if defined( __APPLE__ )
    #include <OpenGL/glu.h>
#else
    #include <GL/glu.h>
    #include <CL/cl_gl.h>
#endif

extern "C" { extern cl_uint gRandomSeed; };

static const char *kernelpattern_image_read_1d =
"__kernel void sample_test( read_only image1d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"  int offset = get_global_id(0);\n"
"  results[ offset ] = read_image%s( source, sampler, offset );\n"
"}\n";

static const char *kernelpattern_image_read_1darray =
"__kernel void sample_test( read_only image1d_array_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    results[ tidY * get_image_width( source ) + tidX ] = read_image%s( source, sampler, (int2)( tidX, tidY ) );\n"
"}\n";

static const char *kernelpattern_image_read_2d =
"__kernel void sample_test( read_only image2d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    results[ tidY * get_image_width( source ) + tidX ] = read_image%s( source, sampler, (int2)( tidX, tidY ) );\n"
"}\n";

static const char *kernelpattern_image_read_2darray =
"__kernel void sample_test( read_only image2d_array_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  tidZ = get_global_id(2);\n"
"    int  width = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
"    int offset = tidZ * width * height + tidY * width + tidX;\n"
"\n"
"     results[ offset ] = read_image%s( source, sampler, (int4)( tidX, tidY, tidZ, 0 ) );\n"
"}\n";

static const char *kernelpattern_image_read_3d =
"__kernel void sample_test( read_only image3d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  tidZ = get_global_id(2);\n"
"    int  width = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
"    int offset = tidZ * width * height + tidY * width + tidX;\n"
"\n"
"     results[ offset ] = read_image%s( source, sampler, (int4)( tidX, tidY, tidZ, 0 ) );\n"
"}\n";

static const char* get_appropriate_kernel_for_target(GLenum target) {

    switch (get_base_gl_target(target)) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_BUFFER:
      return kernelpattern_image_read_1d;
    case GL_TEXTURE_1D_ARRAY:
      return kernelpattern_image_read_1darray;
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_2D:
    case GL_COLOR_ATTACHMENT0:
    case GL_RENDERBUFFER:
      return kernelpattern_image_read_2d;
    case GL_TEXTURE_2D_ARRAY:
      return kernelpattern_image_read_2darray;
    case GL_TEXTURE_3D:
      return kernelpattern_image_read_3d;

    default:
      log_error("Unsupported texture target (%s); cannot determine "
        "appropriate kernel.", GetGLTargetName(target));
      return NULL;
  }
}

int test_cl_image_read( cl_context context, cl_command_queue queue,
  GLenum gl_target, cl_mem image, size_t width, size_t height, size_t depth,
  cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
  clProgramWrapper program;
  clKernelWrapper kernel;
  clMemWrapper streams[ 2 ];

  int error;
  char kernelSource[1024];
  char *programPtr;

  // Use the image created from the GL texture.
  streams[ 0 ] = image;

  // Determine data type and format that CL came up with
  error = clGetImageInfo( streams[ 0 ], CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
  test_error( error, "Unable to get CL image format" );

  // Create the source
  *outType = get_read_kernel_type( outFormat );
  size_t channelSize = get_explicit_type_size( *outType );

  const char* source = get_appropriate_kernel_for_target(gl_target);

  sprintf( kernelSource, source, get_explicit_type_name( *outType ),
    get_kernel_suffix( outFormat ) );

  programPtr = kernelSource;
  if( create_single_kernel_helper( context, &program, &kernel, 1,
    (const char **)&programPtr, "sample_test" ) )
  {
    return -1;
  }

  // Create a vanilla output buffer
  streams[ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE,
    channelSize * 4 * width * height * depth, NULL, &error );
  test_error( error, "Unable to create output buffer" );

  /* Assign streams and execute */
  clSamplerWrapper sampler = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
  test_error( error, "Unable to create sampler" );

  error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
  test_error( error, "Unable to set kernel arguments" );
  error = clSetKernelArg( kernel, 1, sizeof( sampler ), &sampler );
  test_error( error, "Unable to set kernel arguments" );
  error = clSetKernelArg( kernel, 2, sizeof( streams[ 1 ] ), &streams[ 1 ] );
  test_error( error, "Unable to set kernel arguments" );

  glFlush();

  error = (*clEnqueueAcquireGLObjects_ptr)( queue, 1, &streams[ 0 ], 0, NULL, NULL);
  test_error( error, "Unable to acquire GL obejcts");

  // The ND range we use is a function of the dimensionality of the image.
  size_t global_range[3] = { width, height, depth };
  size_t *local_range = NULL;
  int ndim = 1;

  switch (get_base_gl_target(gl_target)) {
    case GL_TEXTURE_1D:
    case GL_TEXTURE_BUFFER:
      ndim = 1;
      break;
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_2D:
    case GL_TEXTURE_1D_ARRAY:
    case GL_COLOR_ATTACHMENT0:
    case GL_RENDERBUFFER:
      ndim = 2;
      break;
    case GL_TEXTURE_3D:
    case GL_TEXTURE_2D_ARRAY:
      ndim = 3;
      break;
    default:
      log_error("Unsupported texture target.");
      return 1;
  }

  // 2D and 3D images have a special way to set the local size (legacy).
  // Otherwise, we let CL select by leaving local_range as NULL.

  if (gl_target == GL_TEXTURE_2D) {
    local_range = (size_t*)malloc(sizeof(size_t) * ndim);
    get_max_common_2D_work_group_size( context, kernel, global_range, local_range );

  } else if (gl_target == GL_TEXTURE_3D) {
    local_range = (size_t*)malloc(sizeof(size_t) * ndim);
    get_max_common_3D_work_group_size( context, kernel, global_range, local_range );

  }

  error = clEnqueueNDRangeKernel( queue, kernel, ndim, NULL, global_range,
    local_range, 0, NULL, NULL );
  test_error( error, "Unable to execute test kernel" );

  error = (*clEnqueueReleaseGLObjects_ptr)( queue, 1, &streams[ 0 ],
    0, NULL, NULL );
  test_error(error, "clEnqueueReleaseGLObjects failed");

  // Read results from the CL buffer
  *outResultBuffer = (void *)( new char[ channelSize * 4 * width * height * depth ] );
  error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0,
    channelSize * 4 * width * height * depth, *outResultBuffer, 0, NULL, NULL );
  test_error( error, "Unable to read output CL buffer!" );

  // free the ranges
  if (local_range) free(local_range);

  return 0;
}

static int test_image_read( cl_context context, cl_command_queue queue,
  GLenum target, GLuint globj, size_t width, size_t height, size_t depth,
  cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
  int error;

  // Create a CL image from the supplied GL texture or renderbuffer.

  cl_mem image;

  if (target == GL_RENDERBUFFER || target == GL_COLOR_ATTACHMENT0) {
    image = (*clCreateFromGLRenderbuffer_ptr)( context, CL_MEM_READ_ONLY, globj, &error );
  } else {
    image = (*clCreateFromGLTexture_ptr)( context, CL_MEM_READ_ONLY,
      target, 0, globj, &error );
  }

  if( error != CL_SUCCESS ) {
    if (target == GL_RENDERBUFFER || target == GL_COLOR_ATTACHMENT0) {
      print_error( error, "Unable to create CL image from GL renderbuffer" );
    } else {
      print_error( error, "Unable to create CL image from GL texture" );
      GLint fmt;
      glGetTexLevelParameteriv( target, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt );
      log_error( "    Supplied GL texture was base format %s and internal "
        "format %s\n", GetGLBaseFormatName( fmt ), GetGLFormatName( fmt ) );
    }
    return error;
  }

  return test_cl_image_read( context, queue, target, image,
    width, height, depth, outFormat, outType, outResultBuffer );
}

static int test_image_format_read(
    cl_context context, cl_command_queue queue,
    size_t width, size_t height, size_t depth,
    GLenum target, struct format* fmt, MTdata data)
{
  int error = 0;

  // If we're testing a half float format, then we need to determine the
  // rounding mode of this machine.  Punt if we fail to do so.

  if( fmt->type == kHalf )
    if( DetectFloatToHalfRoundingMode(queue) )
      return 1;

  size_t w = width, h = height, d = depth;

  // Unpack the format and use it, along with the target, to create an
  // appropriate GL texture.

  GLenum gl_fmt          = fmt->formattype;
  GLenum gl_internal_fmt = fmt->internal;
  GLenum gl_type         = fmt->datatype;
  ExplicitType type      = fmt->type;

  // Required for most of the texture-backed cases:
  glTextureWrapper texture;

  // Required for the special case of TextureBuffer textures:
  glBufferWrapper glbuf;

  // And these are required for the case of Renderbuffer images:
  glFramebufferWrapper glFramebuffer;
  glRenderbufferWrapper glRenderbuffer;

  void* buffer = NULL;

  // Use the correct texture creation function depending on the target, and
  // adjust width, height, depth as appropriate so subsequent size calculations
  // succeed.

  switch (get_base_gl_target(target)) {
    case GL_TEXTURE_1D:
      h = 1; d = 1;
      buffer = CreateGLTexture1D( width, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, true, data );
      break;
    case GL_TEXTURE_BUFFER:
      h = 1; d = 1;
      buffer = CreateGLTextureBuffer(width, target, gl_fmt, gl_internal_fmt,
        gl_type, type, &texture, &glbuf, &error, true, data);
      break;
    case GL_RENDERBUFFER:
    case GL_COLOR_ATTACHMENT0:
      buffer = CreateGLRenderbuffer(width, height, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &glFramebuffer, &glRenderbuffer, &error,
        data, true);
      break;
    case GL_TEXTURE_2D:
    case GL_TEXTURE_RECTANGLE_EXT:
      d = 1;
      buffer = CreateGLTexture2D(width, height, target, gl_fmt, gl_internal_fmt,
        gl_type, type, &texture, &error, true, data);
      break;
    case GL_TEXTURE_1D_ARRAY:
      d = 1;
      buffer = CreateGLTexture1DArray( width, height, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, true, data );
      break;
    case GL_TEXTURE_2D_ARRAY:
      buffer = CreateGLTexture2DArray( width, height, depth, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, true, data );
      break;
    case GL_TEXTURE_3D:
      buffer = CreateGLTexture3D( width, height, depth, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, data, true );
      break;
    default:
      log_error("Unsupported texture target.");
      return 1;
  }

  if ( error != 0 ) {
    if ((gl_fmt == GL_RGBA_INTEGER_EXT) && (!CheckGLIntegerExtensionSupport())){
      log_info("OpenGL version does not support GL_RGBA_INTEGER_EXT. "
        "Skipping test.\n");
      return 0;
    } else {
      return error;
    }
  }

  BufferOwningPtr<char> inputBuffer(buffer);

  cl_image_format clFormat;
  ExplicitType actualType;
  char *outBuffer;

  // Perform the read:

  GLuint globj = texture;
  if (target == GL_RENDERBUFFER || target == GL_COLOR_ATTACHMENT0) {
    globj = glRenderbuffer;
  }

  error = test_image_read( context, queue, target, globj, w, h, d, &clFormat,
    &actualType, (void **)&outBuffer );

  if( error != 0 )
    return error;
  BufferOwningPtr<char> actualResults(outBuffer);

  log_info( "- Read [%4d x %4d x %4d] : GL Texture : %s : %s : %s => CL Image : %s : %s \n",
    (int)w, (int)h, (int)d, GetGLFormatName( gl_fmt ), GetGLFormatName( gl_internal_fmt ),
    GetGLTypeName( gl_type ), GetChannelOrderName( clFormat.image_channel_order ),
    GetChannelTypeName( clFormat.image_channel_data_type ));

  // We have to convert our input buffer to the returned type, so we can validate.
  // This is necessary because OpenCL might not actually pick an internal format
  // that actually matches our input format (for example, if it picks a normalized
  // format, the results will come out as floats instead of going in as ints).

  BufferOwningPtr<char> convertedInputs(convert_to_expected( inputBuffer,
    w * h * d, type, actualType ));
  if( convertedInputs == NULL )
    return -1;

  // Now we validate
  if( actualType == kFloat ) {
    return validate_float_results( convertedInputs, actualResults, w, h, d );
  } else {
    return validate_integer_results( convertedInputs, actualResults, w, h, d,
      get_explicit_type_size( actualType ) );
  }
}

int test_images_read_common( cl_device_id device, cl_context context,
  cl_command_queue queue, struct format* formats, size_t nformats,
  GLenum *targets, size_t ntargets, size_t *sizes, size_t nsizes )
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

      log_info( "Testing image read for GL format %s : %s : %s : %s\n",
        GetGLTargetName( targets[ tidx ] ),
        GetGLFormatName( formats[ fidx ].internal ),
        GetGLBaseFormatName( formats[ fidx ].formattype ),
        GetGLTypeName( formats[ fidx ].datatype ) );

      for ( sidx = 0; sidx < nsizes; sidx++ ) {

        // Test this format + size:

        if ( test_image_format_read(context, queue,
                                    sizes[sidx], sizes[sidx], sizes[sidx],
                                    targets[tidx], &formats[fidx], seed) )
        {
          // We land here in the event of test failure.

          log_error( "ERROR: Image read test failed for %s : %s : %s : %s\n\n",
            GetGLTargetName( targets[ tidx ] ),
            GetGLFormatName( formats[ fidx ].internal ),
            GetGLBaseFormatName( formats[ fidx ].formattype ),
            GetGLTypeName( formats[ fidx ].datatype ) );
          error++;

          // Skip the other sizes for this format.

          break;
        }
      }

      // Note a successful format test, if we passed every size.

      if( sidx == sizeof (sizes) / sizeof( sizes[0] ) ) {
        log_info( "passed: Image read test for GL format  %s : %s : %s : %s\n\n",
        GetGLTargetName( targets[ tidx ] ),
        GetGLFormatName( formats[ fidx ].internal ),
        GetGLBaseFormatName( formats[ fidx ].formattype ),
        GetGLTypeName( formats[ fidx ].datatype ) );
      }
    }
  }

  return error;
}



