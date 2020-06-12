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

#if defined( __APPLE__ )
    #include <OpenGL/glu.h>
#else
    #include <GL/glu.h>
    #include <CL/cl_gl.h>
#endif

#if defined (__linux__)
GLboolean
gluCheckExtension(const GLubyte *extension, const GLubyte *extensions)
{
  const GLubyte *start;
  GLubyte *where, *terminator;

  /* Extension names should not have spaces. */
  where = (GLubyte *) strchr((const char*)extension, ' ');
  if (where || *extension == '\0')
    return 0;
  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  start = extensions;
  for (;;) {
    where = (GLubyte *) strstr((const char *) start, (const char*) extension);
    if (!where)
      break;
    terminator = where + strlen((const char*) extension);
    if (where == start || *(where - 1) == ' ')
      if (*terminator == ' ' || *terminator == '\0')
        return 1;
    start = terminator;
  }
  return 0;
}
#endif


// This is defined in the write common code:
extern int test_cl_image_write( cl_context context, cl_command_queue queue,
  GLenum target, cl_mem clImage, size_t width, size_t height, size_t depth,
  cl_image_format *outFormat, ExplicitType *outType, void **outSourceBuffer,
  MTdata d, bool supports_half );

extern int test_cl_image_read( cl_context context, cl_command_queue queue,
  GLenum gl_target, cl_mem image, size_t width, size_t height, size_t depth, size_t sampleNum,
  cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer );

extern int supportsHalf(cl_context context, bool* supports_half);

static int test_attach_renderbuffer_read_image( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glRenderbuffer,
                    size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
    int error;

    // Create a CL image from the supplied GL renderbuffer
    cl_mem image = (*clCreateFromGLRenderbuffer_ptr)( context, CL_MEM_READ_ONLY, glRenderbuffer, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create CL image from GL renderbuffer" );
        return error;
    }

    return test_cl_image_read( context, queue, glTarget, image, imageWidth,
    imageHeight, 1, 1, outFormat, outType, outResultBuffer );
}

int test_renderbuffer_read_image( cl_context context, cl_command_queue queue,
                            GLsizei width, GLsizei height, GLenum attachment,
                            GLenum format, GLenum internalFormat,
                            GLenum glType, ExplicitType type, MTdata d )
{
    int error;

    if( type == kHalf )
        if( DetectFloatToHalfRoundingMode(queue) )
            return 1;

    // Create the GL renderbuffer
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    void *tmp = CreateGLRenderbuffer( width, height, attachment, format, internalFormat, glType, type, &glFramebuffer, &glRenderbuffer, &error, d, true );
    BufferOwningPtr<char> inputBuffer(tmp);
    if( error != 0 )
    {
        if ((format == GL_RGBA_INTEGER_EXT) && (!CheckGLIntegerExtensionSupport()))
        {
            log_info("OpenGL version does not support GL_RGBA_INTEGER_EXT. Skipping test.\n");
            return 0;
        }
        else
        {
            return error;
        }
    }

    // Run and get the results
    cl_image_format clFormat;
    ExplicitType actualType;
    char *outBuffer;
    error = test_attach_renderbuffer_read_image( context, queue, attachment, glRenderbuffer, width, height, &clFormat, &actualType, (void **)&outBuffer );
    if( error != 0 )
        return error;
    BufferOwningPtr<char> actualResults(outBuffer);

    log_info( "- Read [%4d x %4d] : GL renderbuffer : %s : %s : %s => CL Image : %s : %s \n", width, height,
                    GetGLFormatName( format ), GetGLFormatName( internalFormat ), GetGLTypeName( glType),
                    GetChannelOrderName( clFormat.image_channel_order ), GetChannelTypeName( clFormat.image_channel_data_type ));

#ifdef DEBUG
    log_info("- start read GL data -- \n");
    DumpGLBuffer(glType, width, height, actualResults);
    log_info("- end read GL data -- \n");
#endif

    // We have to convert our input buffer to the returned type, so we can validate.
    BufferOwningPtr<char> convertedInput(convert_to_expected( inputBuffer, width * height, type, actualType, get_channel_order_channel_count(clFormat.image_channel_order) ));

#ifdef DEBUG
    log_info("- start input data -- \n");
    DumpGLBuffer(GetGLTypeForExplicitType(actualType), width, height, convertedInput);
    log_info("- end input data -- \n");
#endif

#ifdef DEBUG
    log_info("- start converted data -- \n");
    DumpGLBuffer(GetGLTypeForExplicitType(actualType), width, height, actualResults);
    log_info("- end converted data -- \n");
#endif

    // Now we validate
    int valid = 0;
    if(convertedInput) {
        if( actualType == kFloat )
            valid = validate_float_results( convertedInput, actualResults, width, height, 1, get_channel_order_channel_count(clFormat.image_channel_order) );
        else
            valid = validate_integer_results( convertedInput, actualResults, width, height, 1, get_explicit_type_size( actualType ) );
    }

    return valid;
}

int test_renderbuffer_read( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum attachments[] = { GL_COLOR_ATTACHMENT0_EXT };

    struct {
        GLenum internal;
        GLenum format;
        GLenum datatype;
        ExplicitType type;

    } formats[] = {
        { GL_RGBA,         GL_BGRA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA8,        GL_RGBA,             GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16,       GL_RGBA,             GL_UNSIGNED_SHORT,           kUShort },

// Renderbuffers with integer formats do not seem to work reliably across
// platforms/implementations. Disabling this in version 1.0 of CL conformance tests.

#ifdef TEST_INTEGER_FORMATS

        { GL_RGBA8I_EXT,   GL_RGBA_INTEGER_EXT, GL_BYTE,                     kChar },
        { GL_RGBA16I_EXT,  GL_RGBA_INTEGER_EXT, GL_SHORT,                    kShort },
        { GL_RGBA32I_EXT,  GL_RGBA_INTEGER_EXT, GL_INT,                      kInt },
        { GL_RGBA8UI_EXT,  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_SHORT,           kUShort },
        { GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT,             kUInt },
#endif
        { GL_RGBA32F_ARB,  GL_RGBA,             GL_FLOAT,                    kFloat },
        { GL_RGBA16F_ARB,  GL_RGBA,             GL_HALF_FLOAT,               kHalf }
    };

    size_t fmtIdx, attIdx;
    int error = 0;
#ifdef DEBUG
    size_t iter = 1;
#else
    size_t iter = 6;
#endif
    RandomSeed seed( gRandomSeed );

  // Check if images are supported
  if (checkForImageSupport(device)) {
    log_info("Device does not support images. Skipping test.\n");
    return 0;
  }

    if( !gluCheckExtension( (const GLubyte *)"GL_EXT_framebuffer_object", glGetString( GL_EXTENSIONS ) ) )
    {
        log_info( "Renderbuffers are not supported by this OpenGL implementation; skipping test\n" );
        return 0;
    }

    // Loop through a set of GL formats, testing a set of sizes against each one
    for( fmtIdx = 0; fmtIdx < sizeof( formats ) / sizeof( formats[ 0 ] ); fmtIdx++ )
    {
        for( attIdx = 0; attIdx < sizeof( attachments ) / sizeof( attachments[ 0 ] ); attIdx++ )
        {
            size_t i;

            log_info( "Testing renderbuffer read for %s : %s : %s : %s\n",
                GetGLAttachmentName( attachments[ attIdx ] ),
                GetGLFormatName( formats[ fmtIdx ].internal ),
                GetGLBaseFormatName( formats[ fmtIdx ].format ),
                GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                GLsizei width = random_in_range( 16, 512, seed );
                GLsizei height = random_in_range( 16, 512, seed );
#ifdef DEBUG
                width = height = 4;
#endif

                if( test_renderbuffer_read_image( context, queue, width, height,
                                                  attachments[ attIdx ],
                                                  formats[ fmtIdx ].format,
                                                  formats[ fmtIdx ].internal,
                                                  formats[ fmtIdx ].datatype,
                                                  formats[ fmtIdx ].type, seed ) )

                {
                    log_error( "ERROR: Renderbuffer read test failed for %s : %s : %s : %s\n\n",
                                GetGLAttachmentName( attachments[ attIdx ] ),
                                GetGLFormatName( formats[ fmtIdx ].internal ),
                                GetGLBaseFormatName( formats[ fmtIdx ].format ),
                                GetGLTypeName( formats[ fmtIdx ].datatype ) );

                    error++;
                    break;    // Skip other sizes for this combination
                }
            }
            if( i == iter )
            {
                log_info( "passed: Renderbuffer read test passed for %s : %s : %s : %s\n\n",
                          GetGLAttachmentName( attachments[ attIdx ] ),
                          GetGLFormatName( formats[ fmtIdx ].internal ),
                          GetGLBaseFormatName( formats[ fmtIdx ].format ),
                          GetGLTypeName( formats[ fmtIdx ].datatype ) );
            }
        }
    }

    return error;
}


#pragma mark -------------------- Write tests -------------------------

int test_attach_renderbuffer_write_to_image( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glRenderbuffer,
                     size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, MTdata d, void **outSourceBuffer, bool supports_half )
{
    int error;

    // Create a CL image from the supplied GL renderbuffer
    clMemWrapper image = (*clCreateFromGLRenderbuffer_ptr)( context, CL_MEM_WRITE_ONLY, glRenderbuffer, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create CL image from GL renderbuffer" );
        return error;
    }

    return test_cl_image_write( context, queue, glTarget, image, imageWidth,
    imageHeight, 1, outFormat, outType, outSourceBuffer, d, supports_half );
}

int test_renderbuffer_image_write( cl_context context, cl_command_queue queue,
                                   GLsizei width, GLsizei height, GLenum attachment,
                                   GLenum format, GLenum internalFormat,
                                     GLenum glType, ExplicitType type, MTdata d )
{
    int error;

    if( type == kHalf )
        if( DetectFloatToHalfRoundingMode(queue) )
            return 1;

    // Create the GL renderbuffer
    glFramebufferWrapper glFramebuffer;
    glRenderbufferWrapper glRenderbuffer;
    CreateGLRenderbuffer( width, height, attachment, format, internalFormat, glType, type, &glFramebuffer, &glRenderbuffer, &error, d, false );
    if( error != 0 )
    {
        if ((format == GL_RGBA_INTEGER_EXT) && (!CheckGLIntegerExtensionSupport()))
        {
            log_info("OpenGL version does not support GL_RGBA_INTEGER_EXT. Skipping test.\n");
            return 0;
        }
        else
        {
            return error;
        }
    }

    // Run and get the results
    cl_image_format clFormat;
    ExplicitType sourceType;
    ExplicitType validationType;
    void *outSourceBuffer;

    bool supports_half = false;
    error = supportsHalf(context, &supports_half);
    if( error != 0 )
        return error;

    error = test_attach_renderbuffer_write_to_image( context, queue, attachment, glRenderbuffer, width, height, &clFormat, &sourceType, d, (void **)&outSourceBuffer, supports_half );
    if( error != 0 || ((sourceType == kHalf ) && !supports_half))
        return error;

    // If actual source type was half, convert to float for validation.
    if( sourceType == kHalf )
        validationType = kFloat;
    else
        validationType = sourceType;

    BufferOwningPtr<char> validationSource( convert_to_expected( outSourceBuffer, width * height, sourceType, validationType, get_channel_order_channel_count(clFormat.image_channel_order) ) );

    log_info( "- Write [%4d x %4d] : GL Renderbuffer : %s : %s : %s => CL Image : %s : %s \n", width, height,
                    GetGLFormatName( format ), GetGLFormatName( internalFormat ), GetGLTypeName( glType),
                    GetChannelOrderName( clFormat.image_channel_order ), GetChannelTypeName( clFormat.image_channel_data_type ));

    // Now read the results from the GL renderbuffer
    BufferOwningPtr<char> resultData( ReadGLRenderbuffer( glFramebuffer, glRenderbuffer, attachment, format, internalFormat, glType, type, width, height ) );

#ifdef DEBUG
    log_info("- start result data -- \n");
    DumpGLBuffer(glType, width, height, resultData);
    log_info("- end result data -- \n");
#endif

    // We have to convert our input buffer to the returned type, so we can validate.
    BufferOwningPtr<char> convertedData( convert_to_expected( resultData, width * height, type, validationType, get_channel_order_channel_count(clFormat.image_channel_order) ) );

#ifdef DEBUG
    log_info("- start input data -- \n");
    DumpGLBuffer(GetGLTypeForExplicitType(validationType), width, height, validationSource);
    log_info("- end input data -- \n");
#endif

#ifdef DEBUG
    log_info("- start converted data -- \n");
    DumpGLBuffer(GetGLTypeForExplicitType(validationType), width, height, convertedData);
    log_info("- end converted data -- \n");
#endif

    // Now we validate
    int valid = 0;
    if(convertedData) {
        if( sourceType == kFloat || sourceType == kHalf )
            valid = validate_float_results( validationSource, convertedData, width, height, 1, get_channel_order_channel_count(clFormat.image_channel_order) );
        else
            valid = validate_integer_results( validationSource, convertedData, width, height, 1, get_explicit_type_size( type ) );
    }

    return valid;
}

int test_renderbuffer_write( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum attachments[] = { GL_COLOR_ATTACHMENT0_EXT };

    struct {
        GLenum internal;
        GLenum format;
        GLenum datatype;
        ExplicitType type;

    } formats[] = {
        { GL_RGBA,         GL_BGRA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA,         GL_RGBA,             GL_UNSIGNED_INT_8_8_8_8_REV, kUChar },
        { GL_RGBA8,        GL_RGBA,             GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16,       GL_RGBA,             GL_UNSIGNED_SHORT,           kUShort },

// Renderbuffers with integer formats do not seem to work reliably across
// platforms/implementations. Disabling this in version 1.0 of CL conformance tests.

#ifdef TEST_INTEGER_FORMATS

        { GL_RGBA8I_EXT,   GL_RGBA_INTEGER_EXT, GL_BYTE,                     kChar },
        { GL_RGBA16I_EXT,  GL_RGBA_INTEGER_EXT, GL_SHORT,                    kShort },
        { GL_RGBA32I_EXT,  GL_RGBA_INTEGER_EXT, GL_INT,                      kInt },
        { GL_RGBA8UI_EXT,  GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE,            kUChar },
        { GL_RGBA16UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_SHORT,           kUShort },
        { GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT,             kUInt },
#endif
        { GL_RGBA32F_ARB,  GL_RGBA,             GL_FLOAT,                    kFloat },
        { GL_RGBA16F_ARB,  GL_RGBA,             GL_HALF_FLOAT,               kHalf }
    };

    size_t fmtIdx, attIdx;
    int error = 0;
    size_t iter = 6;
#ifdef DEBUG
    iter = 1;
#endif
    RandomSeed seed( gRandomSeed );

  // Check if images are supported
  if (checkForImageSupport(device)) {
    log_info("Device does not support images. Skipping test.\n");
    return 0;
  }

    if( !gluCheckExtension( (const GLubyte *)"GL_EXT_framebuffer_object", glGetString( GL_EXTENSIONS ) ) )
    {
        log_info( "Renderbuffers are not supported by this OpenGL implementation; skipping test\n" );
        return 0;
    }

    // Loop through a set of GL formats, testing a set of sizes against each one
    for( fmtIdx = 0; fmtIdx < sizeof( formats ) / sizeof( formats[ 0 ] ); fmtIdx++ )
    {
        for( attIdx = 0; attIdx < sizeof( attachments ) / sizeof( attachments[ 0 ] ); attIdx++ )
        {
            log_info( "Testing Renderbuffer write test for %s : %s : %s : %s\n",
                GetGLAttachmentName( attachments[ attIdx ] ),
                GetGLFormatName( formats[ fmtIdx ].internal ),
                GetGLBaseFormatName( formats[ fmtIdx ].format ),
                GetGLTypeName( formats[ fmtIdx ].datatype ) );

            size_t i;
            for( i = 0; i < iter; i++ )
            {
                GLsizei width = random_in_range( 16, 512, seed );
                GLsizei height = random_in_range( 16, 512, seed );
#ifdef DEBUG
                width = height = 4;
#endif

                if( test_renderbuffer_image_write( context, queue, width, height,
                                                   attachments[ attIdx ],
                                                   formats[ fmtIdx ].format,
                                                   formats[ fmtIdx ].internal,
                                                   formats[ fmtIdx ].datatype,
                                                   formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Renderbuffer write test failed for %s : %s : %s : %s\n\n",
                          GetGLAttachmentName( attachments[ attIdx ] ),
                          GetGLFormatName( formats[ fmtIdx ].internal ),
                          GetGLBaseFormatName( formats[ fmtIdx ].format ),
                          GetGLTypeName( formats[ fmtIdx ].datatype ) );

                    error++;
                    break;    // Skip other sizes for this combination
                }
            }
            if( i == iter )
            {
                log_info( "passed: Renderbuffer write test passed for %s : %s : %s : %s\n\n",
                          GetGLAttachmentName( attachments[ attIdx ] ),
                          GetGLFormatName( formats[ fmtIdx ].internal ),
                          GetGLBaseFormatName( formats[ fmtIdx ].format ),
                          GetGLTypeName( formats[ fmtIdx ].datatype ) );
            }
        }
    }

    return error;
}
