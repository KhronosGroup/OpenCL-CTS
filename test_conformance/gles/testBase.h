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
#ifndef _testBase_h
#define _testBase_h

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#if !defined(_WIN32)
#include <stdbool.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#if !defined (__APPLE__)
#include <CL/cl.h>
#include "gles/gl_headers.h"
#include <CL/cl_gl.h>
#else
#include "gl/gl_headers.h"
#endif

#include "harness/errorHelpers.h"
#include "harness/kernelHelpers.h"
#include "harness/threadTesting.h"
#include "harness/typeWrappers.h"
#include "harness/conversions.h"
#include "harness/mt19937.h"

#ifdef GL_ES_VERSION_2_0
#include "gles/helpers.h"
#else
#include "gl/helpers.h"
#endif

extern const char *get_kernel_suffix( cl_image_format *format );
extern const char *get_write_conversion( cl_image_format *format, ExplicitType type);
extern ExplicitType get_read_kernel_type( cl_image_format *format );
extern ExplicitType get_write_kernel_type( cl_image_format *format );

extern char * convert_to_expected( void * inputBuffer, size_t numPixels, ExplicitType inType, ExplicitType outType );
extern int validate_integer_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t typeSize );
extern int validate_integer_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t depth, size_t typeSize );
extern int validate_float_results( void *expectedResults, void *actualResults, size_t width, size_t height );
extern int validate_float_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t depth );

extern int CheckGLObjectInfo(cl_mem mem, cl_gl_object_type expected_cl_gl_type, GLuint expected_gl_name,
                             GLenum expected_cl_gl_texture_target, GLint expected_cl_gl_mipmap_level);

extern bool CheckGLIntegerExtensionSupport();

#endif // _testBase_h



