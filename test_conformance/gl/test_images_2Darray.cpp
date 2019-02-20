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

int test_images_read_2Darray( cl_device_id device, cl_context context,
  cl_command_queue queue, int )
{
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  size_t sizes[] = { 2, 4, 8, 16, 32, 64, 128 };
  size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);

  GLenum targets[] = { GL_TEXTURE_2D_ARRAY };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  return test_images_read_common(device, context, queue, common_formats,
      nformats, targets, ntargets, sizes, nsizes);
}

int test_images_write_2Darray( cl_device_id device, cl_context context,
  cl_command_queue queue, int numElements )
{
  int error = 0;
  size_t i;
  const size_t nsizes = 6;
  sizevec_t sizes[nsizes];

  // FIXME: Query for 2D image array write support.

  GLenum targets[] = { GL_TEXTURE_2D_ARRAY };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (i = 0; i < nsizes; i++) {
    sizes[i].width  = random_in_range( 16, 512, seed );
    sizes[i].height = random_in_range( 16, 512, seed );
    sizes[i].depth  = random_in_range( 4, 24, seed );
  }

  return test_images_write_common( device, context, queue, common_formats,
    nformats, targets, ntargets, sizes, nsizes );
}

int test_images_2Darray_getinfo( cl_device_id device, cl_context context,
  cl_command_queue queue, int )
{
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  size_t sizes[] = { 2, 4, 8, 16, 32, 64, 128 };
  size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);

  GLenum targets[] = { GL_TEXTURE_2D_ARRAY };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);

  return test_images_get_info_common(device, context, queue, common_formats,
      nformats, targets, ntargets, sizes, nsizes);
}