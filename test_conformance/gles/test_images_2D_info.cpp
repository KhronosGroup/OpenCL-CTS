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

static int test_image_read( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glTexture,
                            size_t imageWidth, size_t imageHeight, cl_image_format *outFormat,
                            ExplicitType *outType, void **outResultBuffer )
{
    // Create a CL image from the supplied GL texture
    int error;
    clMemWrapper image = (*clCreateFromGLTexture_ptr)( context, CL_MEM_READ_ONLY, glTarget, 0, glTexture, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create CL image from GL texture" );
#ifndef GL_ES_VERSION_2_0
        GLint fmt;
        glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt );
        log_error( "    Supplied GL texture was baseformat %s and internalformat %s\n", GetGLBaseFormatName( fmt ), GetGLFormatName( fmt ) );
#endif
        return error;
    }

    // Determine data type and format that CL came up with
    error = clGetImageInfo( image, CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
    test_error( error, "Unable to get CL image format" );

    return CheckGLObjectInfo(image, CL_GL_OBJECT_TEXTURE2D, glTexture, glTarget, 0);
}

static int test_image_object_info( cl_context context, cl_command_queue queue,
                                   size_t width, size_t height, GLenum target,
                                   GLenum format, GLenum internalFormat,
                                   GLenum glType, ExplicitType type, MTdata d )
{
    int error;

    // Create the GL texture
    glTextureWrapper glTexture;
    void *tmp = CreateGLTexture2D( width, height, target, format, internalFormat, glType, type, &glTexture, &error, true, d );
    BufferOwningPtr<char> inputBuffer(tmp);
    if( error != 0 )
    {
        // GL_RGBA_INTEGER_EXT doesn't exist in GLES2. No need to check for it.
        return error;
    }

    /* skip formats not supported by OpenGL */
    if(!tmp)
    {
        return 0;
    }

    // Run and get the results
    cl_image_format clFormat;
    ExplicitType actualType;
    char *outBuffer;
    error = test_image_read( context, queue, target, glTexture, width, height, &clFormat, &actualType, (void **)&outBuffer );

    return error;
}

int test_images_2D_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum targets[] =
#ifdef GL_ES_VERSION_2_0
        { GL_TEXTURE_2D };
#else // GL_ES_VERSION_2_0
        { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
#endif // GL_ES_VERSION_2_0

    struct {
        GLenum internal;
        GLenum format;
        GLenum datatype;
        ExplicitType type;

    } formats[] = {
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_SHORT,           kUShort },
        { GL_RGBA,         GL_RGBA,             GL_HALF_FLOAT_OES,           kHalf },
        { GL_RGBA,         GL_RGBA,             GL_FLOAT,                    kFloat },
    };

    size_t sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

    size_t fmtIdx, tgtIdx;
    int error = 0;
    size_t iter = 6;
    RandomSeed seed( gRandomSeed );

    // Check if images are supported
  if (checkForImageSupport(device)) {
    log_info("Device does not support images. Skipping test.\n");
    return 0;
  }

    // Loop through a set of GL formats, testing a set of sizes against each one
    for( fmtIdx = 0; fmtIdx < sizeof( formats ) / sizeof( formats[ 0 ] ); fmtIdx++ )
    {
        for( tgtIdx = 0; tgtIdx < sizeof( targets ) / sizeof( targets[ 0 ] ); tgtIdx++ )
        {
            size_t i;
            log_info( "Testing image texture object info test for %s : %s : %s : %s\n",
                GetGLTargetName( targets[ tgtIdx ] ),
                GetGLFormatName( formats[ fmtIdx ].internal ),
                GetGLBaseFormatName( formats[ fmtIdx ].format ),
                GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                if( test_image_object_info( context, queue, sizes[i], sizes[i],
                                            targets[ tgtIdx ],
                                            formats[ fmtIdx ].format,
                                            formats[ fmtIdx ].internal,
                                            formats[ fmtIdx ].datatype,
                                            formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Image texture object info test failed for %s : %s : %s : %s\n\n",
                        GetGLTargetName( targets[ tgtIdx ] ),
                        GetGLFormatName( formats[ fmtIdx ].internal ),
                        GetGLBaseFormatName( formats[ fmtIdx ].format ),
                        GetGLTypeName( formats[ fmtIdx ].datatype ) );


                    error++;
                    break;    // Skip other sizes for this combination
                }
            }
            if( i == iter )
            {
                log_info( "passed: Image texture object info test passed for %s : %s : %s : %s\n\n",
                    GetGLTargetName( targets[ tgtIdx ] ),
                    GetGLFormatName( formats[ fmtIdx ].internal ),
                    GetGLBaseFormatName( formats[ fmtIdx ].format ),
                    GetGLTypeName( formats[ fmtIdx ].datatype ) );
            }
            else
                break;    // Skip other cube map targets; they're unlikely to pass either
        }
    }

    return error;
}
int test_images_cube_getinfo( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum targets[] = {
        GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };

    struct {
        GLenum internal;
        GLenum format;
        GLenum datatype;
        ExplicitType type;

    } formats[] = {
#ifdef GL_ES_VERSION_2_0
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_SHORT,           kUShort },
        // XXX add others
#else // GL_ES_VERSION_2_0
        { GL_RGBA,         GL_BGRA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA8,        GL_RGBA,             GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16,       GL_RGBA,             GL_UNSIGNED_SHORT,           kUShort },
        { GL_RGBA8I_EXT,   GL_RGBA_INTEGER_EXT, GL_BYTE,                     kChar },
        { GL_RGBA16I_EXT,  GL_RGBA_INTEGER_EXT, GL_SHORT,                    kShort },
        { GL_RGBA32I_EXT,  GL_RGBA_INTEGER_EXT, GL_INT,                      kInt },
        { GL_RGBA8UI_EXT,  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_SHORT,           kUShort },
        { GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT,             kUInt },
        { GL_RGBA32F_ARB,  GL_RGBA,             GL_FLOAT,                    kFloat }
#endif
    };

    size_t sizes[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };

    size_t fmtIdx, tgtIdx;
    int error = 0;
    size_t iter = 6;
    RandomSeed seed( gRandomSeed );

    // Check if images are supported
  if (checkForImageSupport(device)) {
    log_info("Device does not support images. Skipping test.\n");
    return 0;
  }

    // Loop through a set of GL formats, testing a set of sizes against each one
    for( fmtIdx = 0; fmtIdx < sizeof( formats ) / sizeof( formats[ 0 ] ); fmtIdx++ )
    {
        for( tgtIdx = 0; tgtIdx < sizeof( targets ) / sizeof( targets[ 0 ] ); tgtIdx++ )
        {
            size_t i;
            log_info( "Testing cube map object info test for %s : %s : %s : %s\n",
                GetGLTargetName( targets[ tgtIdx ] ),
                GetGLFormatName( formats[ fmtIdx ].internal ),
                GetGLBaseFormatName( formats[ fmtIdx ].format ),
                GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                if( test_image_object_info( context, queue, sizes[i], sizes[i],
                                            targets[ tgtIdx ],
                                            formats[ fmtIdx ].format,
                                            formats[ fmtIdx ].internal,
                                            formats[ fmtIdx ].datatype,
                                            formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Cube map object info test failed for %s : %s : %s : %s\n\n",
                        GetGLTargetName( targets[ tgtIdx ] ),
                        GetGLFormatName( formats[ fmtIdx ].internal ),
                        GetGLBaseFormatName( formats[ fmtIdx ].format ),
                        GetGLTypeName( formats[ fmtIdx ].datatype ) );


                    error++;
                    break;    // Skip other sizes for this combination
                }
            }
            if( i == iter )
            {
                log_info( "passed: Cube map object info test passed for %s : %s : %s : %s\n\n",
                    GetGLTargetName( targets[ tgtIdx ] ),
                    GetGLFormatName( formats[ fmtIdx ].internal ),
                    GetGLBaseFormatName( formats[ fmtIdx ].format ),
                    GetGLTypeName( formats[ fmtIdx ].datatype ) );
            }
            else
                break;    // Skip other cube map targets; they're unlikely to pass either
        }
    }

    return error;
}
