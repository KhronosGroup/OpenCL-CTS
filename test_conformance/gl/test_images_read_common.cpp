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

extern int supportsHalf(cl_context context, bool* supports_half);
extern int supportsMsaa(cl_context context, bool* supports_msaa);
extern int supportsDepth(cl_context context, bool* supports_depth);

static const char *kernelpattern_image_read_1d =
"__kernel void sample_test( read_only image1d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"  int offset = get_global_id(0);\n"
"  results[ offset ] = read_image%s( source, sampler, offset );\n"
"}\n";

static const char *kernelpattern_image_read_1d_buffer =
"__kernel void sample_test( read_only image1d_buffer_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"  int offset = get_global_id(0);\n"
"  results[ offset ] = read_image%s( source, offset );\n"
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

static const char *kernelpattern_image_read_2d_depth =
"__kernel void sample_test( read_only image2d_depth_t source, sampler_t sampler, __global %s *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    results[ tidY * get_image_width( source ) + tidX ] = read_image%s( source, sampler, (int2)( tidX, tidY ) );\n"
"}\n";

static const char *kernelpattern_image_read_2darray_depth =
"__kernel void sample_test( read_only image2d_array_depth_t source, sampler_t sampler, __global %s *results )\n"
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

static const char *kernelpattern_image_multisample_read_2d =
"#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable\n"
"__kernel void sample_test( read_only image2d_msaa_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  width = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
"    int  num_samples = get_image_num_samples( source );\n"
"    for(size_t sample = 0; sample < num_samples; sample++ ) {\n"
"    int  offset = sample * width * height + tidY * width + tidX;\n"
"     results[ offset ] = read_image%s( source, (int2)( tidX, tidY ), sample );\n"
"    }\n"
"}\n";

static const char *kernelpattern_image_multisample_read_2d_depth =
  "#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable\n"
  "__kernel void sample_test( read_only image2d_msaa_depth_t source, sampler_t sampler, __global %s *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  width = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
  "    int  num_samples = get_image_num_samples( source );\n"
  "    for(size_t sample = 0; sample < num_samples; sample++ ) {\n"
"    int  offset = sample * width * height + tidY * width + tidX;\n"
"     results[ offset ] = read_image%s( source, (int2)( tidX, tidY ), sample );\n"
  "    }\n"
"}\n";

static const char *kernelpattern_image_multisample_read_2darray =
"#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable\n"
"__kernel void sample_test( read_only image2d_array_msaa_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  tidZ = get_global_id(2);\n"
"    int  num_samples = get_image_num_samples( source );\n"
"    int  width  = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
"    int  array_size = get_image_array_size( source );\n"
"    for(size_t sample = 0; sample< num_samples; ++sample) {\n"
"      int offset = (array_size * width * height) * sample + (width * height) * tidZ + tidY * width + tidX;\n"
"         results[ offset ] = read_image%s( source, (int4)( tidX, tidY, tidZ, 1 ), sample );\n"
"    }\n"
"}\n";

static const char *kernelpattern_image_multisample_read_2darray_depth =
  "#pragma OPENCL EXTENSION cl_khr_gl_msaa_sharing : enable\n"
  "__kernel void sample_test( read_only image2d_array_msaa_depth_t source, sampler_t sampler, __global %s *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  tidZ = get_global_id(2);\n"
"    int  num_samples = get_image_num_samples( source );\n"
"    int  width  = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
  "    int  array_size = get_image_array_size( source );\n"
  "    for(size_t sample = 0; sample < num_samples; ++sample) {\n"
  "      int offset = (array_size * width * height) * sample + (width * height) * tidZ + tidY * width + tidX;\n"
  "         results[ offset ] = read_image%s( source, (int4)( tidX, tidY, tidZ, 1 ), sample );\n"
  "    }\n"
"}\n";

static const char* get_appropriate_kernel_for_target(GLenum target, cl_channel_order channel_order) {

    switch (get_base_gl_target(target)) {
    case GL_TEXTURE_1D:
      return kernelpattern_image_read_1d;
    case GL_TEXTURE_BUFFER:
      return kernelpattern_image_read_1d_buffer;
    case GL_TEXTURE_1D_ARRAY:
      return kernelpattern_image_read_1darray;
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_2D:
    case GL_COLOR_ATTACHMENT0:
    case GL_RENDERBUFFER:
    case GL_TEXTURE_CUBE_MAP:
#ifdef GL_VERSION_3_2
    if(channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
      return kernelpattern_image_read_2d_depth;
#endif
      return kernelpattern_image_read_2d;
    case GL_TEXTURE_2D_ARRAY:
#ifdef GL_VERSION_3_2
      if(channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
        return kernelpattern_image_read_2darray_depth;
#endif
      return kernelpattern_image_read_2darray;
    case GL_TEXTURE_3D:
      return kernelpattern_image_read_3d;
    case GL_TEXTURE_2D_MULTISAMPLE:
#ifdef GL_VERSION_3_2
        if(channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
          return kernelpattern_image_multisample_read_2d_depth;
#endif
      return kernelpattern_image_multisample_read_2d;
      break;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
#ifdef GL_VERSION_3_2
        if(channel_order == CL_DEPTH || channel_order == CL_DEPTH_STENCIL)
          return kernelpattern_image_multisample_read_2darray_depth;
#endif
      return kernelpattern_image_multisample_read_2darray;
      break;
    default:
      log_error("Unsupported texture target (%s); cannot determine "
        "appropriate kernel.", GetGLTargetName(target));
      return NULL;
  }
}

int test_cl_image_read( cl_context context, cl_command_queue queue,
  GLenum gl_target, cl_mem image, size_t width, size_t height, size_t depth, size_t sampleNum,
  cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
  clProgramWrapper program;
  clKernelWrapper kernel;
  clMemWrapper streams[ 2 ];

  int error;
  char kernelSource[2048];
  char *programPtr;

  // Use the image created from the GL texture.
  streams[ 0 ] = image;

  // Determine data type and format that CL came up with
  error = clGetImageInfo( streams[ 0 ], CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
  test_error( error, "Unable to get CL image format" );

  // Determine the number of samples
  cl_uint samples = 0;
  error = clGetImageInfo( streams[ 0 ], CL_IMAGE_NUM_SAMPLES, sizeof( samples ), &samples, NULL );
  test_error( error, "Unable to get CL_IMAGE_NUM_SAMPLES" );

  // Create the source
  *outType = get_read_kernel_type( outFormat );
  size_t channelSize = get_explicit_type_size( *outType );

  const char* source = get_appropriate_kernel_for_target(gl_target, outFormat->image_channel_order);

  sprintf( kernelSource, source, get_explicit_type_name( *outType ),
    get_kernel_suffix( outFormat ) );

  programPtr = kernelSource;
  if( create_single_kernel_helper( context, &program, &kernel, 1,
    (const char **)&programPtr, "sample_test", "" ) )
  {
    return -1;
  }

  // Create a vanilla output buffer
  cl_device_id device;
  error = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, NULL);
  test_error( error, "Unable to get queue device" );

  cl_ulong maxAllocSize = 0;
  error = clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( maxAllocSize ), &maxAllocSize, NULL );
  test_error( error, "Unable to get CL_DEVICE_MAX_MEM_ALLOC_SIZE" );

  size_t buffer_bytes = channelSize * get_channel_order_channel_count(outFormat->image_channel_order) * width * height * depth * sampleNum;
  if (buffer_bytes > maxAllocSize) {
    log_info("Output buffer size %d is too large for device (max alloc size %d) Skipping...\n",
             (int)buffer_bytes, (int)maxAllocSize);
    return 1;
  }

  streams[ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE, buffer_bytes, NULL, &error );
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

  glFinish();

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
    case GL_TEXTURE_CUBE_MAP:
      ndim = 2;
      break;
    case GL_TEXTURE_3D:
    case GL_TEXTURE_2D_ARRAY:
#ifdef GL_VERSION_3_2
    case GL_TEXTURE_2D_MULTISAMPLE:
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      ndim = 3;
      break;
#endif
    default:
      log_error("Test error: Unsupported texture target.\n");
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
  *outResultBuffer = (void *)( new char[ channelSize * get_channel_order_channel_count(outFormat->image_channel_order) * width * height * depth * sampleNum] );
  error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0,
    channelSize * get_channel_order_channel_count(outFormat->image_channel_order) * width * height * depth * sampleNum, *outResultBuffer, 0, NULL, NULL );
  test_error( error, "Unable to read output CL buffer!" );

  // free the ranges
  if (local_range) free(local_range);

  return 0;
}

static int test_image_read( cl_context context, cl_command_queue queue,
  GLenum target, GLuint globj, size_t width, size_t height, size_t depth, size_t sampleNum,
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
    width, height, depth, sampleNum, outFormat, outType, outResultBuffer );
}

static int test_image_format_read(
    cl_context context, cl_command_queue queue,
    size_t width, size_t height, size_t depth,
    GLenum target, struct format* fmt, MTdata data)
{
  int error = 0;

  // Determine the maximum number of supported samples
  GLint samples = 1;
  if (target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    samples = get_gl_max_samples(target, fmt->internal);

  // If we're testing a half float format, then we need to determine the
  // rounding mode of this machine.  Punt if we fail to do so.

  if( fmt->type == kHalf )
  {
    if( DetectFloatToHalfRoundingMode(queue) )
      return 1;
    bool supports_half = false;
    error = supportsHalf(context, &supports_half);
    if( error != 0 )
      return error;
    if (!supports_half) return 0;
  }
#ifdef GL_VERSION_3_2
    if (get_base_gl_target(target) == GL_TEXTURE_2D_MULTISAMPLE ||
        get_base_gl_target(target) == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    {
        bool supports_msaa;
        error = supportsMsaa(context, &supports_msaa);
        if( error != 0 ) return error;
        if (!supports_msaa) return 0;
    }
    if (fmt->formattype == GL_DEPTH_COMPONENT ||
        fmt->formattype == GL_DEPTH_STENCIL)
    {
        bool supports_depth;
        error = supportsDepth(context, &supports_depth);
        if( error != 0 ) return error;
        if (!supports_depth) return 0;
    }
#endif
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
      d = 1;
      buffer = CreateGLRenderbuffer(width, height, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &glFramebuffer, &glRenderbuffer, &error,
        data, true);
      break;
    case GL_TEXTURE_2D:
    case GL_TEXTURE_RECTANGLE_EXT:
    case GL_TEXTURE_CUBE_MAP:
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
#ifdef GL_VERSION_3_2
    case GL_TEXTURE_2D_MULTISAMPLE:
      d = 1;
      buffer = CreateGLTexture2DMultisample( width, height, samples, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, true, data, true );
      break;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
      buffer = CreateGLTexture2DArrayMultisample( width, height, depth, samples, target, gl_fmt,
        gl_internal_fmt, gl_type, type, &texture, &error, true, data, true );
      break;
#endif
    default:
      log_error("Unsupported texture target.");
      return 1;
  }

  if ( error == -2 ) {
    log_info("OpenGL texture couldn't be created, because a texture is too big. Skipping test.\n");
    return 0;
  }

  // Check to see if the texture could not be created for some other reason like
  // GL_FRAMEBUFFER_UNSUPPORTED
  if (error == GL_FRAMEBUFFER_UNSUPPORTED) {
    log_info("Skipping...\n");
    return 0;
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
  if( inputBuffer == NULL )
    return -1;

  cl_image_format clFormat;
  ExplicitType actualType;
  char *outBuffer;

  // Perform the read:

  GLuint globj = texture;
  if (target == GL_RENDERBUFFER || target == GL_COLOR_ATTACHMENT0) {
    globj = glRenderbuffer;
  }

  error = test_image_read( context, queue, target, globj, w, h, d, samples, &clFormat,
                          &actualType, (void **)&outBuffer );

  if( error != 0 )
    return error;

  BufferOwningPtr<char> actualResults(outBuffer);
  if( actualResults == NULL )
    return -1;

  log_info( "- Read [%4d x %4d x %4d x %4d] : GL Texture : %s : %s : %s => CL Image : %s : %s \n",
    (int)w, (int)h, (int)d, (int)samples, GetGLFormatName( gl_fmt ), GetGLFormatName( gl_internal_fmt ),
    GetGLTypeName( gl_type ), GetChannelOrderName( clFormat.image_channel_order ),
    GetChannelTypeName( clFormat.image_channel_data_type ));

  BufferOwningPtr<char> convertedInputs;

  // We have to convert our input buffer to the returned type, so we can validate.
  // This is necessary because OpenCL might not actually pick an internal format
  // that actually matches our input format (for example, if it picks a normalized
  // format, the results will come out as floats instead of going in as ints).

  if ( gl_type == GL_UNSIGNED_INT_2_10_10_10_REV )
  {
    cl_uint *p = (cl_uint *)buffer;
    float *inData = (float *)malloc( w * h * d * samples * sizeof(float) );

    for( size_t i = 0; i < 4 * w * h * d * samples; i += 4 )
    {
      inData[ i + 0 ] = (float)( ( p[ 0 ] >> 20 ) & 0x3ff ) / (float)1023;
      inData[ i + 1 ] = (float)( ( p[ 0 ] >> 10 ) & 0x3ff ) / (float)1023;
      inData[ i + 2 ] = (float)( p[ 0 ] & 0x3ff ) / (float)1023;
      p++;
    }

    convertedInputs.reset( inData );
    if( convertedInputs == NULL )
      return -1;
  }
  else if ( gl_type == GL_DEPTH24_STENCIL8 )
  {
    // GL_DEPTH24_STENCIL8 is treated as CL_UNORM_INT24 + CL_DEPTH_STENCIL where
    // the stencil is ignored.
    cl_uint *p = (cl_uint *)buffer;
    float *inData = (float *)malloc( w * h * d * samples * sizeof(float) );

    for( size_t i = 0; i < w * h * d * samples; i++ )
    {
      inData[ i ] = (float)((p[i] >> 8) & 0xffffff) / (float)0xfffffe;
    }

    convertedInputs.reset( inData );
    if( convertedInputs == NULL )
      return -1;
  }
  else if ( gl_type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
  {
    // GL_FLOAT_32_UNSIGNED_INT_24_8_REV is treated as a CL_FLOAT +
    // unused 24 + CL_DEPTH_STENCIL; we check the float value and ignore the
    // second word

    float *p = (float *)buffer;
    float *inData = (float *)malloc( w * h * d * samples * sizeof(float) );

    for( size_t i = 0; i < w * h * d * samples; i++ )
    {
      inData[ i ] = p[i*2];
    }

    convertedInputs.reset( inData );
    if( convertedInputs == NULL )
      return -1;
  }
  else
  {
    convertedInputs.reset(convert_to_expected( inputBuffer,
      w * h * d * samples, type, actualType, get_channel_order_channel_count(clFormat.image_channel_order) ));
    if( convertedInputs == NULL )
      return -1;
  }

  // Now we validate
  if( actualType == kFloat )
  {
    if ( clFormat.image_channel_data_type == CL_UNORM_INT_101010 )
    {
      return validate_float_results_rgb_101010( convertedInputs, actualResults, w, h, d, samples );
    }
    else
    {
      return validate_float_results( convertedInputs, actualResults, w, h, d, samples, get_channel_order_channel_count(clFormat.image_channel_order) );
    }
  }
  else
  {
    return validate_integer_results( convertedInputs, actualResults, w, h, d, samples, get_explicit_type_size( actualType ) );
  }
}

int test_images_read_common( cl_device_id device, cl_context context,
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

      // Texture buffer only takes an internal format, so the level data passed
      // by the test and used for verification must match the internal format
      if ((targets[tidx] == GL_TEXTURE_BUFFER) && (GetGLFormat(formats[ fidx ].internal) != formats[fidx].formattype))
        continue;

      if ( formats[ fidx ].datatype == GL_UNSIGNED_INT_2_10_10_10_REV )
      {
        // Check if the RGB 101010 format is supported
        if ( is_rgb_101010_supported( context, targets[ tidx ] ) == 0 )
          break; // skip
      }

      if (targets[tidx] != GL_TEXTURE_BUFFER)
        log_info( "Testing image read for GL format %s : %s : %s : %s\n",
          GetGLTargetName( targets[ tidx ] ),
          GetGLFormatName( formats[ fidx ].internal ),
          GetGLBaseFormatName( formats[ fidx ].formattype ),
          GetGLTypeName( formats[ fidx ].datatype ) );
      else
        log_info( "Testing image read for GL format %s : %s\n",
                 GetGLTargetName( targets[ tidx ] ),
                 GetGLFormatName( formats[ fidx ].internal ));

      for ( sidx = 0; sidx < nsizes; sidx++ ) {

        // Test this format + size:
        int err;
        if ((err = test_image_format_read(context, queue,
                                    sizes[sidx].width, sizes[sidx].height, sizes[sidx].depth,
                                    targets[tidx], &formats[fidx], seed) ))
        {
          // Negative return values are errors, positive mean the test was skipped
          if (err < 0) {

            // We land here in the event of test failure.

            log_error( "ERROR: Image read test failed for %s : %s : %s : %s\n\n",
              GetGLTargetName( targets[ tidx ] ),
              GetGLFormatName( formats[ fidx ].internal ),
              GetGLBaseFormatName( formats[ fidx ].formattype ),
              GetGLTypeName( formats[ fidx ].datatype ) );
            error++;
          }

          // Skip the other sizes for this format.
          printf("Skipping remaining sizes for this format\n");

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
