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

#include "gl_headers.h"

const char *get_kernel_suffix( cl_image_format *format )
{
    switch( format->image_channel_data_type )
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_FLOAT:
            return "f";
        case CL_HALF_FLOAT:
            return "h";
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
            return "i";
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
            return "ui";
        default:
            return "";
    }
}

ExplicitType get_read_kernel_type( cl_image_format *format )
{
    switch( format->image_channel_data_type )
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_FLOAT:
            return kFloat;
        case CL_HALF_FLOAT:
            return kHalf;
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
            return kInt;
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
            return kUInt;
        default:
            return kInt;
    }
}

ExplicitType get_write_kernel_type( cl_image_format *format )
{
    switch( format->image_channel_data_type )
    {
        case CL_UNORM_INT8:
            return kFloat;
        case CL_UNORM_INT16:
            return kFloat;
        case CL_SNORM_INT8:
            return kFloat;
        case CL_SNORM_INT16:
            return kFloat;
        case CL_HALF_FLOAT:
            return kHalf;
        case CL_FLOAT:
            return kFloat;
        case CL_SIGNED_INT8:
            return kChar;
        case CL_SIGNED_INT16:
            return kShort;
        case CL_SIGNED_INT32:
            return kInt;
        case CL_UNSIGNED_INT8:
            return kUChar;
        case CL_UNSIGNED_INT16:
            return kUShort;
        case CL_UNSIGNED_INT32:
            return kUInt;
        default:
            return kInt;
    }
}

const char* get_write_conversion( cl_image_format *format, ExplicitType type )
{
    switch( format->image_channel_data_type )
    {
        case CL_UNORM_INT8:
        case CL_UNORM_INT16:
        case CL_SNORM_INT8:
        case CL_SNORM_INT16:
        case CL_FLOAT:
            if(type != kFloat) return "convert_float4";
            break;
        case CL_HALF_FLOAT:
            break;
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
            if(type != kInt) return "convert_int4";
            break;
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
            if(type != kUInt) return "convert_uint4";
            break;
        default:
            return "";
    }
    return "";
}

// The only three input types to this function are kInt, kUInt and kFloat, due to the way we set up our tests
// The output types, though, are pretty much anything valid for GL to receive

#define DOWNSCALE_INTEGER_CASE( enum, type, bitShift )    \
    case enum:    \
    {        \
        cl_##type *dst = new cl_##type[ numPixels * 4 ]; \
        for( size_t i = 0; i < numPixels * 4; i++ ) \
            dst[ i ] = src[ i ];    \
        return (char *)dst;        \
    }

#define UPSCALE_FLOAT_CASE( enum, type, typeMax )    \
    case enum:    \
    {        \
        cl_##type *dst = new cl_##type[ numPixels * 4 ]; \
        for( size_t i = 0; i < numPixels * 4; i++ ) \
            dst[ i ] = (cl_##type)( src[ i ] * typeMax );    \
        return (char *)dst;        \
    }

char * convert_to_expected( void * inputBuffer, size_t numPixels, ExplicitType inType, ExplicitType outType )
{
#ifdef GLES_DEBUG
    log_info( "- Converting from input type '%s' to output type '%s'\n",
             get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
#endif

    if( inType == outType )
    {
        char *outData = new char[ numPixels * 4 * get_explicit_type_size(outType) ] ; // sizeof( cl_int ) ];
        memcpy( outData, inputBuffer, numPixels * 4 * get_explicit_type_size(inType)  );
        return outData;
    }
    else if( inType == kChar )
    {
        cl_char *src = (cl_char *)inputBuffer;

        switch( outType )
        {
            case kInt:
            {
                cl_int *outData = new cl_int[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_int)((src[ i ]));
                }
                return (char *)outData;
            }
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)src[ i ] / 127.0f;
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else if( inType == kUChar )
    {
        cl_uchar *src = (cl_uchar *)inputBuffer;

        switch( outType )
        {
            case kUInt:
            {
                cl_uint *outData = new cl_uint[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_uint)((src[ i ]));
                }
                return (char *)outData;
            }
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)(src[ i ]) / 256.0f;
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else if( inType == kShort )
    {
        cl_short *src = (cl_short *)inputBuffer;

        switch( outType )
        {
            case kInt:
            {
                cl_int *outData = new cl_int[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_int)((src[ i ]));
                }
                return (char *)outData;
            }
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)src[ i ] / 32768.0f;
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else if( inType == kUShort )
    {
        cl_ushort *src = (cl_ushort *)inputBuffer;

        switch( outType )
        {
            case kUInt:
            {
                cl_uint *outData = new cl_uint[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_uint)((src[ i ]));
                }
                return (char *)outData;
            }
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)(src[ i ]) / 65535.0f;
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else if( inType == kInt )
    {
        cl_int *src = (cl_int *)inputBuffer;

        switch( outType )
        {
                DOWNSCALE_INTEGER_CASE( kShort, short, 16 )
                DOWNSCALE_INTEGER_CASE( kChar, char, 24 )
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)fmaxf( (float)src[ i ] / 2147483647.f, -1.f );
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else if( inType == kUInt )
    {
        cl_uint *src = (cl_uint *)inputBuffer;

        switch( outType )
        {
                DOWNSCALE_INTEGER_CASE( kUShort, ushort, 16 )
                DOWNSCALE_INTEGER_CASE( kUChar, uchar, 24 )
            case kFloat:
            {
                // If we're converting to float, then CL decided that we should be normalized
                cl_float *outData = new cl_float[ numPixels * 4 ];
                for( size_t i = 0; i < numPixels * 4; i++ )
                {
                    outData[ i ] = (cl_float)src[ i ] / 4294967295.f;
                }
                return (char *)outData;
            }
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }
    else
    {
        cl_float *src = (cl_float *)inputBuffer;

        switch( outType )
        {
                UPSCALE_FLOAT_CASE( kChar, char, 127.f )
                UPSCALE_FLOAT_CASE( kUChar, uchar, 255.f )
                UPSCALE_FLOAT_CASE( kShort, short, 32767.f )
                UPSCALE_FLOAT_CASE( kUShort, ushort, 65535.f )
                UPSCALE_FLOAT_CASE( kInt, int, 2147483647.f )
                UPSCALE_FLOAT_CASE( kUInt, uint, 4294967295.f )
            default:
                log_error( "ERROR: Unsupported conversion from %s to %s!\n", get_explicit_type_name( inType ), get_explicit_type_name( outType ) );
                return NULL;
        }
    }

    return NULL;
}

int validate_integer_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t typeSize )
{
    return validate_integer_results( expectedResults, actualResults, width, height, 0, typeSize );
}

int validate_integer_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t depth, size_t typeSize )
{
    char *expected = (char *)expectedResults;
    char *actual = (char *)actualResults;
    for( size_t z = 0; z < ( ( depth == 0 ) ? 1 : depth ); z++ )
    {
        for( size_t y = 0; y < height; y++ )
        {
            for( size_t x = 0; x < width; x++ )
            {
                if( memcmp( expected, actual, typeSize * 4 ) != 0 )
                {
                    char scratch[ 1024 ];

                    if( depth == 0 )
                        log_error( "ERROR: Data sample %d,%d did not validate!\n", (int)x, (int)y );
                    else
                        log_error( "ERROR: Data sample %d,%d,%d did not validate!\n", (int)x, (int)y, (int)z );
                    log_error( "\tExpected: %s\n", GetDataVectorString( expected, typeSize, 4, scratch ) );
                    log_error( "\t  Actual: %s\n", GetDataVectorString( actual, typeSize, 4, scratch ) );
                    return -1;
                }
                expected += typeSize * 4;
                actual += typeSize * 4;
            }
        }
    }

    return 0;
}

int validate_float_results( void *expectedResults, void *actualResults, size_t width, size_t height )
{
    return validate_float_results( expectedResults, actualResults, width, height, 0 );
}

int validate_float_results( void *expectedResults, void *actualResults, size_t width, size_t height, size_t depth )
{
    cl_float *expected = (cl_float *)expectedResults;
    cl_float *actual = (cl_float *)actualResults;
    for( size_t z = 0; z < ( ( depth == 0 ) ? 1 : depth ); z++ )
    {
        for( size_t y = 0; y < height; y++ )
        {
            for( size_t x = 0; x < width; x++ )
            {
                float err = 0.f;
                for( size_t i = 0; i < 4; i++ )
                {
                    float error = fabsf( expected[ i ] - actual[ i ] );
                    if( error > err )
                        err = error;
                }

                if( err > 1.f / 127.f ) // Max expected range of error if we converted from an 8-bit integer to a normalized float
                {
                    if( depth == 0 )
                        log_error( "ERROR: Data sample %d,%d did not validate!\n", (int)x, (int)y );
                    else
                        log_error( "ERROR: Data sample %d,%d,%d did not validate!\n", (int)x, (int)y, (int)z );
                    log_error( "\tExpected: %f %f %f %f\n", expected[ 0 ], expected[ 1 ], expected[ 2 ], expected[ 3 ] );
                    log_error( "\t        : %a %a %a %a\n", expected[ 0 ], expected[ 1 ], expected[ 2 ], expected[ 3 ] );
                    log_error( "\t  Actual: %f %f %f %f\n", actual[ 0 ], actual[ 1 ], actual[ 2 ], actual[ 3 ] );
                    log_error( "\t        : %a %a %a %a\n", actual[ 0 ], actual[ 1 ], actual[ 2 ], actual[ 3 ] );
                    return -1;
                }
                expected += 4;
                actual += 4;
            }
        }
    }

    return 0;
}

int CheckGLObjectInfo(cl_mem mem, cl_gl_object_type expected_cl_gl_type, GLuint expected_gl_name,
                  GLenum expected_cl_gl_texture_target, GLint expected_cl_gl_mipmap_level)
{
  cl_gl_object_type object_type;
  GLuint object_name;
  GLenum texture_target;
  GLint mipmap_level;
    int error;

  error = (*clGetGLObjectInfo_ptr)(mem, &object_type, &object_name);
  test_error( error, "clGetGLObjectInfo failed");
  if (object_type != expected_cl_gl_type) {
    log_error("clGetGLObjectInfo did not return expected object type: expected %d, got %d.\n", expected_cl_gl_type, object_type);
    return -1;
  }
  if (object_name != expected_gl_name) {
    log_error("clGetGLObjectInfo did not return expected object name: expected %d, got %d.\n", expected_gl_name, object_name);
    return -1;
  }

  if (object_type == CL_GL_OBJECT_TEXTURE2D || object_type == CL_GL_OBJECT_TEXTURE3D) {
       error = (*clGetGLTextureInfo_ptr)(mem, CL_GL_TEXTURE_TARGET, sizeof(texture_target), &texture_target, NULL);
    test_error( error, "clGetGLTextureInfo for CL_GL_TEXTURE_TARGET failed");

    if (texture_target != expected_cl_gl_texture_target) {
      log_error("clGetGLTextureInfo did not return expected texture target: expected %d, got %d.\n", expected_cl_gl_texture_target, texture_target);
      return -1;
    }

       error = (*clGetGLTextureInfo_ptr)(mem, CL_GL_MIPMAP_LEVEL, sizeof(mipmap_level), &mipmap_level, NULL);
    test_error( error, "clGetGLTextureInfo for CL_GL_MIPMAP_LEVEL failed");

    if (mipmap_level != expected_cl_gl_mipmap_level) {
      log_error("clGetGLTextureInfo did not return expected mipmap level: expected %d, got %d.\n", expected_cl_gl_mipmap_level, mipmap_level);
      return -1;
    }
  }
  return 0;
}

bool CheckGLIntegerExtensionSupport()
{
    // Get the OpenGL version and supported extensions
    const GLubyte *glVersion = glGetString(GL_VERSION);
    const GLubyte *glExtensionList = glGetString(GL_EXTENSIONS);

    // Check if the OpenGL vrsion is 3.0 or grater or GL_EXT_texture_integer is supported
    return (((glVersion[0] - '0') >= 3) || (strstr((const char*)glExtensionList, "GL_EXT_texture_integer")));
}
