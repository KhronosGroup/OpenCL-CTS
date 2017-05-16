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

#include <algorithm>

using namespace std;

void calc_2D_multisample_size_descriptors(sizevec_t* sizes, size_t nsizes)
{
  // Need to limit texture size according to GL device properties
  GLint maxTextureSize = 4096;
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (size_t i = 0; i < nsizes; i++) {
    sizes[i].width  = random_in_range( 2, min(maxTextureSize, 1<<(i+4)), seed );
    sizes[i].height = random_in_range( 2, min(maxTextureSize, 1<<(i+4)), seed );
    sizes[i].depth  = 1;
  }
}

void calc_2D_array_multisample_size_descriptors(sizevec_t* sizes, size_t nsizes)
{
  // Need to limit array size according to GL device properties
  GLint maxTextureLayers = 16, maxTextureSize = 4096;
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxTextureLayers);
  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (size_t i = 0; i < nsizes; i++) {
    sizes[i].width  = random_in_range( 2, min(maxTextureSize, 1<<(i+4)), seed );
    sizes[i].height = random_in_range( 2, min(maxTextureSize, 1<<(i+4)), seed );
    sizes[i].depth  = random_in_range( 2, min(maxTextureLayers, 1<<(i+4)), seed );
  }
}

int test_images_read_2D_multisample( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  if (!is_extension_available(device, "cl_khr_gl_msaa_sharing")) {
    log_info("Test not run because 'cl_khr_gl_msaa_sharing' extension is not supported by the tested device\n");
    return 0;
  }

  glEnable(GL_MULTISAMPLE);

  const size_t nsizes = 8;
  sizevec_t sizes[nsizes];
  calc_2D_multisample_size_descriptors(sizes, nsizes);

  size_t nformats;

  GLenum targets[] = { GL_TEXTURE_2D_MULTISAMPLE };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  nformats = sizeof(common_formats) / sizeof(common_formats[0]);
  int ret_common = test_images_read_common(device, context, queue, common_formats, nformats, targets, ntargets, sizes, nsizes);

  nformats = sizeof(depth_formats) / sizeof(depth_formats[0]);
  int ret_depth = test_images_read_common(device, context, queue, depth_formats, nformats, targets, ntargets, sizes, nsizes);

  return (ret_common) ? ret_common : ret_depth;
}

int test_images_read_2Darray_multisample( cl_device_id device, cl_context context,
                             cl_command_queue queue, int )
{
  if (!is_extension_available(device, "cl_khr_gl_msaa_sharing")) {
    log_info("Test not run because 'cl_khr_gl_msaa_sharing' extension is not supported by the tested device\n");
    return 0;
  }

  glEnable(GL_MULTISAMPLE);

  const size_t nsizes = 4;
  sizevec_t sizes[nsizes];
  calc_2D_array_multisample_size_descriptors(sizes, nsizes);

  size_t nformats;

  GLenum targets[] = { GL_TEXTURE_2D_MULTISAMPLE_ARRAY };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  nformats = sizeof(common_formats) / sizeof(common_formats[0]);
  int ret_common = test_images_read_common(device, context, queue, common_formats, nformats, targets, ntargets, sizes, nsizes);

  nformats = sizeof(depth_formats) / sizeof(depth_formats[0]);
  int ret_depth = test_images_read_common(device, context, queue, depth_formats, nformats, targets, ntargets, sizes, nsizes);

  return (ret_common) ? ret_common : ret_depth;
}

