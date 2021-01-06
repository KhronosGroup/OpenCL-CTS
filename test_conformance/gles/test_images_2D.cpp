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

static const char *imageReadKernelPattern =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"  /* added support for half floats */
"__kernel void sample_test( read_only image2d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    results[ tidY * get_image_width( source ) + tidX ] = read_image%s( source, sampler, (int2)( tidX, tidY ) );\n"
"}\n";

static const char *imageWriteKernelPattern =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"  /* added support for half floats */
"__kernel void sample_test( __global %s4 *source, write_only image2d_t dest )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    uint index = tidY * get_image_width( dest ) + tidX;\n"
"    %s4 value = source[index];\n"
"    write_image%s( dest, (int2)( tidX, tidY ), %s(value));\n"
"}\n";

int test_cl_image_read( cl_context context, cl_command_queue queue, cl_mem clImage,
                       size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper outStream;

    int error;
    size_t threads[ 2 ], localThreads[ 2 ];
    char kernelSource[10240];
    char *programPtr;


    // Determine data type and format that CL came up with
    error = clGetImageInfo( clImage, CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
    test_error( error, "Unable to get CL image format" );

    /* Create the source */
    *outType = get_read_kernel_type( outFormat );
    size_t channelSize = get_explicit_type_size( *outType );

    sprintf( kernelSource, imageReadKernelPattern, get_explicit_type_name( *outType ), get_kernel_suffix( outFormat ) );

#ifdef GLES_DEBUG
    log_info("-- start cl image read kernel --\n");
    log_info("%s", kernelSource);
    log_info("-- end cl image read kernel --\n");
#endif

    /* Create kernel */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }


    // Create a vanilla output buffer
    outStream = clCreateBuffer( context, CL_MEM_READ_WRITE, channelSize * 4 * imageWidth * imageHeight, NULL, &error );
    test_error( error, "Unable to create output buffer" );


    /* Assign streams and execute */
    clSamplerWrapper sampler = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
    test_error( error, "Unable to create sampler" );

    error = clSetKernelArg( kernel, 0, sizeof( clImage ), &clImage );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( sampler ), &sampler );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 2, sizeof( outStream ), &outStream );
    test_error( error, "Unable to set kernel arguments" );

    glFlush();

    error = (*clEnqueueAcquireGLObjects_ptr)( queue, 1, &clImage, 0, NULL, NULL);
    test_error( error, "Unable to acquire GL obejcts");

    /* Run the kernel */
    threads[ 0 ] = imageWidth;
    threads[ 1 ] = imageHeight;

    error = get_max_common_2D_work_group_size( context, kernel, threads, localThreads );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );


    error = (*clEnqueueReleaseGLObjects_ptr)( queue, 1, &clImage, 0, NULL, NULL );
    test_error(error, "clEnqueueReleaseGLObjects failed");

    // Read results from the CL buffer
    *outResultBuffer = malloc(channelSize * 4 * imageWidth * imageHeight);
    error = clEnqueueReadBuffer( queue, outStream, CL_TRUE, 0, channelSize * 4 * imageWidth * imageHeight,
                                *outResultBuffer, 0, NULL, NULL );
    test_error( error, "Unable to read output CL buffer!" );

    return 0;
}

static int test_image_read( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glTexture,
                           size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
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

    return test_cl_image_read( context, queue, image, imageWidth, imageHeight, outFormat, outType, outResultBuffer );
}

int test_image_format_read( cl_context context, cl_command_queue queue,
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
    if( error != 0 )
        return error;
    BufferOwningPtr<char> actualResults(outBuffer);

    log_info( "- Read [%4d x %4d] : GL Texture : %s : %s : %s => CL Image : %s : %s \n", (int)width, (int)height,
             GetGLFormatName( format ), GetGLFormatName( internalFormat ), GetGLTypeName( glType),
             GetChannelOrderName( clFormat.image_channel_order ), GetChannelTypeName( clFormat.image_channel_data_type ));

    // We have to convert our input buffer to the returned type, so we can validate.
    BufferOwningPtr<char> convertedInputs(convert_to_expected( inputBuffer, width * height, type, actualType ));

    // Now we validate
    int valid = 0;
    if(convertedInputs) {
        if( actualType == kFloat )
            valid = validate_float_results( convertedInputs, actualResults, width, height );
        else
            valid = validate_integer_results( convertedInputs, actualResults, width, height, get_explicit_type_size( actualType ) );
    }

    return valid;
}

int test_images_read( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
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
        { GL_RGBA,         GL_RGBA,             GL_FLOAT,                    kFloat },
    };

    size_t fmtIdx, tgtIdx;
    int error = 0;
    size_t iter = 6;
    RandomSeed seed(gRandomSeed );

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

            log_info( "Testing image read for GL format %s : %s : %s : %s\n",
                     GetGLTargetName( targets[ tgtIdx ] ),
                     GetGLFormatName( formats[ fmtIdx ].internal ),
                     GetGLBaseFormatName( formats[ fmtIdx ].format ),
                     GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                size_t width = random_in_range( 16, 512, seed );
                size_t height = random_in_range( 16, 512, seed );

                if( test_image_format_read( context, queue, width, height,
                                           targets[ tgtIdx ],
                                           formats[ fmtIdx ].format,
                                           formats[ fmtIdx ].internal,
                                           formats[ fmtIdx ].datatype,
                                           formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Image read test failed for %s : %s : %s : %s\n\n",
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
                log_info( "passed: Image read for GL format %s : %s : %s : %s\n\n",
                         GetGLTargetName( targets[ tgtIdx ] ),
                         GetGLFormatName( formats[ fmtIdx ].internal ),
                         GetGLBaseFormatName( formats[ fmtIdx ].format ),
                         GetGLTypeName( formats[ fmtIdx ].datatype ) );
            }
        }
    }

    return error;
}

int test_images_read_cube( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
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
    RandomSeed seed(gRandomSeed);

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

            log_info( "Testing image read cubemap for GL format  %s : %s : %s : %s\n\n",
                     GetGLTargetName( targets[ tgtIdx ] ),
                     GetGLFormatName( formats[ fmtIdx ].internal ),
                     GetGLBaseFormatName( formats[ fmtIdx ].format ),
                     GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                if( test_image_format_read( context, queue, sizes[i], sizes[i],
                                           targets[ tgtIdx ],
                                           formats[ fmtIdx ].format,
                                           formats[ fmtIdx ].internal,
                                           formats[ fmtIdx ].datatype,
                                           formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Image read cubemap test failed for %s : %s : %s : %s\n\n",
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
                log_info( "passed: Image read cubemap for GL format  %s : %s : %s : %s\n\n",
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


#pragma mark -------------------- Write tests -------------------------


int test_cl_image_write( cl_context context, cl_command_queue queue, cl_mem clImage,
                        size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, void **outSourceBuffer, MTdata d )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper inStream;

    int error;
    size_t threads[ 2 ], localThreads[ 2 ];
    char kernelSource[10240];
    char *programPtr;

    // Determine data type and format that CL came up with
    error = clGetImageInfo( clImage, CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
    test_error( error, "Unable to get CL image format" );

    /* Create the source */
    *outType = get_write_kernel_type( outFormat );
    size_t channelSize = get_explicit_type_size( *outType );

    const char* suffix = get_kernel_suffix( outFormat );
    const char* convert = get_write_conversion( outFormat, *outType );

    sprintf( kernelSource, imageWriteKernelPattern, get_explicit_type_name( *outType ), get_explicit_type_name( *outType ), suffix, convert);

#ifdef GLES_DEBUG
    log_info("-- start cl image write kernel --\n");
    log_info("%s", kernelSource);
    log_info("-- end cl image write kernel --\n");
#endif

    /* Create kernel */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }

    // Generate some source data based on the input type we need
    *outSourceBuffer = CreateRandomData(*outType, imageWidth * imageHeight * 4, d);

    // Create a vanilla input buffer
    inStream = clCreateBuffer( context, CL_MEM_COPY_HOST_PTR, channelSize * 4 * imageWidth * imageHeight, *outSourceBuffer, &error );
    test_error( error, "Unable to create output buffer" );

    /* Assign streams and execute */
    clSamplerWrapper sampler = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
    test_error( error, "Unable to create sampler" );

    error = clSetKernelArg( kernel, 0, sizeof( inStream ), &inStream );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( clImage ), &clImage );
    test_error( error, "Unable to set kernel arguments" );

    glFlush();

    error = (*clEnqueueAcquireGLObjects_ptr)( queue, 1, &clImage, 0, NULL, NULL);
    test_error( error, "Unable to acquire GL obejcts");

    /* Run the kernel */
    threads[ 0 ] = imageWidth;
    threads[ 1 ] = imageHeight;

    error = get_max_common_2D_work_group_size( context, kernel, threads, localThreads );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );

    clEventWrapper event;
    error = (*clEnqueueReleaseGLObjects_ptr)( queue, 1, &clImage, 0, NULL, &event );
    test_error(error, "clEnqueueReleaseGLObjects failed");

    error = clWaitForEvents( 1, &event );
    test_error(error, "clWaitForEvents failed");

#ifdef GLES_DEBUG
    int i;
    size_t origin[] = {0, 0, 0,};
    size_t region[] = {imageWidth, imageHeight, 1 };
    void* cldata = malloc( channelSize * 4 * imageWidth * imageHeight );
    clEnqueueReadImage( queue, clImage, 1, origin, region, 0, 0, cldata, 0, 0, 0);
    log_info("- start CL Image Data -- \n");
    DumpGLBuffer(GetGLTypeForExplicitType(*outType), imageWidth, imageHeight, cldata);
    log_info("- end CL Image Data -- \n");
    free(cldata);
#endif

    // All done!
    return 0;
}

int test_image_write( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glTexture,
                     size_t imageWidth, size_t imageHeight, cl_image_format *outFormat, ExplicitType *outType, void **outSourceBuffer, MTdata d )
{
    int error;

    // Create a CL image from the supplied GL texture
    clMemWrapper image = (*clCreateFromGLTexture_ptr)( context, CL_MEM_WRITE_ONLY, glTarget, 0, glTexture, &error );
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

    return test_cl_image_write( context, queue, image, imageWidth, imageHeight, outFormat, outType, outSourceBuffer, d );
}


int test_image_format_write( cl_context context, cl_command_queue queue,
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
        return error;
    }

    /* skip formats not supported by OpenGL */
    if(!tmp)
    {
        return 0;
    }

    // Run and get the results
    cl_image_format clFormat;
    ExplicitType sourceType;
    void *outSourceBuffer;
    error = test_image_write( context, queue, target, glTexture, width, height, &clFormat, &sourceType, (void **)&outSourceBuffer, d );
    if( error != 0 )
        return error;

    BufferOwningPtr<char> actualSource(outSourceBuffer);

    log_info( "- Write [%4d x %4d] : GL Texture : %s : %s : %s => CL Image : %s : %s \n", (int)width, (int)height,
             GetGLFormatName( format ), GetGLFormatName( internalFormat ), GetGLTypeName( glType),
             GetChannelOrderName( clFormat.image_channel_order ), GetChannelTypeName( clFormat.image_channel_data_type ));

    // Now read the results from the GL texture
    ExplicitType readType = type;
    BufferOwningPtr<char> glResults( ReadGLTexture( target, glTexture, format, internalFormat, glType, readType, width, height ) );

    // We have to convert our input buffer to the returned type, so we can validate.
    BufferOwningPtr<char> convertedGLResults( convert_to_expected( glResults, width * height, readType, sourceType ) );

#ifdef GLES_DEBUG
    log_info("- start read GL data -- \n");
    DumpGLBuffer(glType, width, height, glResults);
    log_info("- end read GL data -- \n");

    log_info("- start converted data -- \n");
    DumpGLBuffer(glType, width, height, convertedGLResults);
    log_info("- end converted data -- \n");
#endif

    // Now we validate
    int valid = 0;
    if(convertedGLResults) {
        if( sourceType == kFloat )
            valid = validate_float_results( actualSource, convertedGLResults, width, height );
        else
            valid = validate_integer_results( actualSource, convertedGLResults, width, height, get_explicit_type_size( readType ) );
    }

    return valid;
}

int test_images_write( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum targets[] =
#ifdef GL_ES_VERSION_2_0
            { GL_TEXTURE_2D };
#else // GL_ES_VERSION_2_0
            { GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE_EXT };
#endif

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

    size_t fmtIdx, tgtIdx;
    int error = 0;
    size_t iter = 6;
    RandomSeed seed(gRandomSeed);

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
            log_info( "Testing image write test for %s : %s : %s : %s\n",
                     GetGLTargetName( targets[ tgtIdx ] ),
                     GetGLFormatName( formats[ fmtIdx ].internal ),
                     GetGLBaseFormatName( formats[ fmtIdx ].format ),
                     GetGLTypeName( formats[ fmtIdx ].datatype ) );

            size_t i;
            for( i = 0; i < iter; i++ )
            {
                size_t width = random_in_range( 16, 512, seed );
                size_t height = random_in_range( 16, 512, seed );

                if( targets[ tgtIdx ] == GL_TEXTURE_2D )
                    width = height;

                if( test_image_format_write( context, queue, width, height,
                                            targets[ tgtIdx ],
                                            formats[ fmtIdx ].format,
                                            formats[ fmtIdx ].internal,
                                            formats[ fmtIdx ].datatype,
                                            formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Image write test failed for %s : %s : %s : %s\n\n",
                              GetGLTargetName( targets[ tgtIdx ] ),
                              GetGLFormatName( formats[ fmtIdx ].internal ),
                              GetGLBaseFormatName( formats[ fmtIdx ].format ),
                              GetGLTypeName( formats[ fmtIdx ].datatype ) );

                    error++;
                    break;    // Skip other sizes for this combination
                }
            }
            if( i == 6 )
            {
                log_info( "passed: Image write for GL format  %s : %s : %s : %s\n\n",
                         GetGLTargetName( targets[ tgtIdx ] ),
                         GetGLFormatName( formats[ fmtIdx ].internal ),
                         GetGLBaseFormatName( formats[ fmtIdx ].format ),
                         GetGLTypeName( formats[ fmtIdx ].datatype ) );

            }
        }
    }

    return error;
}

int test_images_write_cube( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
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
            log_info( "Testing image write cubemap test for %s : %s : %s : %s\n",
                     GetGLTargetName( targets[ tgtIdx ] ),
                     GetGLFormatName( formats[ fmtIdx ].internal ),
                     GetGLBaseFormatName( formats[ fmtIdx ].format ),
                     GetGLTypeName( formats[ fmtIdx ].datatype ) );

            for( i = 0; i < iter; i++ )
            {
                if( test_image_format_write( context, queue, sizes[i], sizes[i],
                                            targets[ tgtIdx ],
                                            formats[ fmtIdx ].format,
                                            formats[ fmtIdx ].internal,
                                            formats[ fmtIdx ].datatype,
                                            formats[ fmtIdx ].type, seed ) )
                {
                    log_error( "ERROR: Image write cubemap test failed for %s : %s : %s : %s\n\n",
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
                log_info( "passed: Image write cubemap for GL format  %s : %s : %s : %s\n\n",
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
