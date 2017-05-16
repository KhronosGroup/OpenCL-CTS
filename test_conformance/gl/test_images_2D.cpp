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

extern "C" { extern cl_uint gRandomSeed; }

#pragma mark -
#pragma mark _2D read tests

int test_images_read_2D( cl_device_id device, cl_context context, 
  cl_command_queue queue, int numElements )
{
  RandomSeed seed( gRandomSeed );
  
  GLenum targets[] = { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);
  
  const size_t nsizes = 6;
  size_t sizes[nsizes];
  for (int i = 0; i < nsizes; i++) {
    sizes[i] = random_in_range(16, 512, seed);
  }
  
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
  
  size_t sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
  size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);
    
  return test_images_read_common(device, context, queue, common_formats, 
    nformats, targets, ntargets, sizes, nsizes);
}

#pragma mark -
#pragma mark _2D write tests

#include "common.h"

int test_images_write( cl_device_id device, cl_context context, 
  cl_command_queue queue, int numElements )
{
  int error = 0;
  size_t i;
  const size_t nsizes = 6;
  sizevec_t sizes[nsizes];
  
  GLenum targets[] = { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
  size_t ntargets = sizeof(targets) / sizeof(targets[0]);
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);

  RandomSeed seed( gRandomSeed );

  // Generate some random sizes (within reasonable ranges)
  for (i = 0; i < nsizes; i++) {
    sizes[i].width  = random_in_range( 16, 512, seed );
    sizes[i].height = random_in_range( 16, 512, seed );
    sizes[i].depth  = 1;
  }

  return test_images_write_common( device, context, queue, common_formats, 
    nformats, targets, ntargets, sizes, nsizes );
}

int test_images_write_cube( cl_device_id device, cl_context context, 
  cl_command_queue queue, int numElements )
{
  size_t i;
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

  const size_t nsizes = 9;
  size_t base_size = 16;
  sizevec_t sizes[nsizes];
  
  // Generate power-of-two 2D sizes, 16-4096:
  
  for (i = 0; i < nsizes; i++) {
    sizes[i].width = sizes[i].height = base_size;
    sizes[i].depth = 1;
    base_size *= 2;
  }
        
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
  
  size_t sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
  size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);
  
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);
  
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
  
  size_t sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
  size_t nsizes = sizeof(sizes) / sizeof(sizes[0]);
  
  size_t nformats = sizeof(common_formats) / sizeof(common_formats[0]);
  
  return test_images_get_info_common(device, context, queue, common_formats, 
      nformats, targets, ntargets, sizes, nsizes);
}
