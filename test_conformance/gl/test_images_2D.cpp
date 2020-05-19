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
#include <algorithm>

using namespace std;

#pragma mark -
#pragma mark _2D read tests

void calc_2D_test_size_descriptors(sizevec_t* sizes, size_t nsizes)
{
  // Need to limit array size according to GL device properties
  // Need to limit texture size according to GL device properties
  GLint maxTextureSize = 4096, maxTextureRectangleSize = 4096, size;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
  glGetIntegerv(GL_MAX_RECTANGLE_TEXTURE_SIZE_EXT, &maxTextureRectangleSize);

  size = min(maxTextureSize, maxTextureRectangleSize);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (size_t i = 0; i < nsizes; i++) {
    sizes[i].width  = random_in_range( 2, min(size, 1<<(i+4)), seed );
    sizes[i].height = random_in_range( 2, min(size, 1<<(i+4)), seed );
    sizes[i].depth  = 1;
  }
}

void calc_cube_test_size_descriptors(sizevec_t* sizes, size_t nsizes)
{
  // Need to limit array size according to GL device properties
  // Need to limit texture size according to GL device properties
  GLint maxQubeMapSize = 4096;
  glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &maxQubeMapSize);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (size_t i = 0; i < nsizes; i++) {
    sizes[i].width  = sizes[i].height = random_in_range( 2, min(maxQubeMapSize, 1<<(i+4)), seed );
    sizes[i].depth  = 1;
  }
}

int test_images_read_2D( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  GLenum targets[] = { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_2D_test_size_descriptors(sizes, nsizes);

  return test_images_read_common(device, context, queue, common_formats,
    nformats, targets, ntargets, sizes, nsizes);
}

int test_images_read_cube( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  GLenum targets[] = {
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };

  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_cube_test_size_descriptors(sizes, nsizes);

  return test_images_read_common(device, context, queue, common_formats,
    nformats, targets, ntargets, sizes, nsizes);
}

#pragma mark -
#pragma mark _2D write tests

#include "common.h"

int test_images_write( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  GLenum targets[] = { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_2D_test_size_descriptors(sizes, nsizes);

  return test_images_write_common( device, context, queue, common_formats,
    nformats, targets, ntargets, sizes, nsizes );
}

int test_images_write_cube( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  GLenum targets[] = {
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
  };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_cube_test_size_descriptors(sizes, nsizes);

  return test_images_write_common( device, context, queue, common_formats,
    nformats, targets, ntargets, sizes, nsizes );
}

#pragma mark -
#pragma mark _2D get info tests

int test_images_2D_getinfo( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  GLenum targets[] = { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_2D_test_size_descriptors(sizes, nsizes);

  return test_images_get_info_common(device, context, queue, common_formats,
      nformats, targets, ntargets, sizes, nsizes);
}

int test_images_cube_getinfo( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
    GLenum targets[] = {
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
  };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_cube_test_size_descriptors(sizes, nsizes);

  return test_images_get_info_common(device, context, queue, common_formats,
      nformats, targets, ntargets, sizes, nsizes);
}
