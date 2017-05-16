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
#ifndef __COMMON_H__
#define __COMMON_H__

#include "testBase.h"

typedef struct {
  size_t width;
  size_t height;
  size_t depth;
} sizevec_t;

struct format {
  GLenum internal;
  GLenum formattype;
  GLenum datatype;
  ExplicitType type;
};

// These are the typically tested formats.

static struct format common_formats[] = {
#ifdef __APPLE__
  { GL_RGBA8,        GL_BGRA,             GL_UNSIGNED_INT_8_8_8_8,         kUChar },
  { GL_RGBA8,        GL_BGRA,             GL_UNSIGNED_INT_8_8_8_8_REV,     kUChar },
  { GL_RGBA8,        GL_RGBA,             GL_UNSIGNED_INT_8_8_8_8_REV,     kUChar },
  // { GL_RGB10,        GL_BGRA,             GL_UNSIGNED_INT_2_10_10_10_REV,  kFloat },
#endif
  { GL_RGBA8,        GL_RGBA,             GL_UNSIGNED_BYTE,                kUChar },
  { GL_RGBA16,       GL_RGBA,             GL_UNSIGNED_SHORT,               kUShort },
  { GL_RGBA8I_EXT,   GL_RGBA_INTEGER_EXT, GL_BYTE,                         kChar },
  { GL_RGBA16I_EXT,  GL_RGBA_INTEGER_EXT, GL_SHORT,                        kShort },
  { GL_RGBA32I_EXT,  GL_RGBA_INTEGER_EXT, GL_INT,                          kInt },
  { GL_RGBA8UI_EXT,  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE,                kUChar },
  { GL_RGBA16UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_SHORT,               kUShort },
  { GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT,                 kUInt },
  { GL_RGBA32F_ARB,  GL_RGBA,             GL_FLOAT,                        kFloat },
  { GL_RGBA16F_ARB,  GL_RGBA,             GL_HALF_FLOAT,                   kHalf }
};

#ifdef GL_VERSION_3_2
static struct format depth_formats[] = {
  { GL_DEPTH_COMPONENT16,  GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT,                 kUShort },
  { GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT,                          kFloat },
  { GL_DEPTH24_STENCIL8,   GL_DEPTH_STENCIL,   GL_UNSIGNED_INT_24_8,              kUInt },
  { GL_DEPTH32F_STENCIL8,  GL_DEPTH_STENCIL,   GL_FLOAT_32_UNSIGNED_INT_24_8_REV, kFloat },
};
#endif

int test_images_write_common(cl_device_id device, cl_context context,
  cl_command_queue queue, struct format* formats, size_t nformats,
  GLenum *targets, size_t ntargets, sizevec_t* sizes, size_t nsizes );

int test_images_read_common( cl_device_id device, cl_context context,
  cl_command_queue queue, struct format* formats, size_t nformats,
  GLenum *targets, size_t ntargets, sizevec_t *sizes, size_t nsizes );

int test_images_get_info_common( cl_device_id device, cl_context context,
  cl_command_queue queue, struct format* formats, size_t nformats,
  GLenum *targets, size_t ntargets, sizevec_t *sizes, size_t nsizes );

int is_rgb_101010_supported( cl_context context, GLenum gl_target );

#endif // __COMMON_H__
