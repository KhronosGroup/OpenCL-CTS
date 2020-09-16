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
"__kernel void sample_test( read_only image3d_t source, sampler_t sampler, __global %s4 *results )\n"
"{\n"
"    int  tidX = get_global_id(0);\n"
"    int  tidY = get_global_id(1);\n"
"    int  tidZ = get_global_id(2);\n"
"    int  width = get_image_width( source );\n"
"    int  height = get_image_height( source );\n"
"    int offset = tidZ * width * height + tidY * width + tidX;\n"
"\n"
"     results[ offset ] = read_image%s( source, sampler, (int4)( tidX, tidY, tidZ, 0 ) );\n"
"}\n";

static int test_image_read( cl_context context, cl_command_queue queue, GLenum glTarget, GLuint glTexture,
                            size_t imageWidth, size_t imageHeight, size_t imageDepth,
                            cl_image_format *outFormat, ExplicitType *outType, void **outResultBuffer )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ 2 ];

    int error;
    size_t threads[ 3 ], localThreads[ 3 ];
    char kernelSource[1024];
    char *programPtr;


    // Create a CL image from the supplied GL texture
    streams[ 0 ] = (*clCreateFromGLTexture_ptr)( context, CL_MEM_READ_ONLY, glTarget, 0, glTexture, &error );
    if( error != CL_SUCCESS )
    {
        print_error( error, "Unable to create CL image from GL texture" );
#ifndef GL_ES_VERSION_2_0
        GLint fmt;
        glGetTexLevelParameteriv( glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt );
        log_error( "    Supplied GL texture was format %s\n", GetGLFormatName( fmt ) );
#endif
        return error;
    }

    // Determine data type and format that CL came up with
    error = clGetImageInfo( streams[ 0 ], CL_IMAGE_FORMAT, sizeof( cl_image_format ), outFormat, NULL );
    test_error( error, "Unable to get CL image format" );

    /* Create the source */
    *outType = get_read_kernel_type( outFormat );
    size_t channelSize = get_explicit_type_size( *outType );

    sprintf( kernelSource, imageReadKernelPattern, get_explicit_type_name( *outType ), get_kernel_suffix( outFormat ) );

    /* Create kernel */
    programPtr = kernelSource;
    if( create_single_kernel_helper( context, &program, &kernel, 1, (const char **)&programPtr, "sample_test" ) )
    {
        return -1;
    }


    // Create a vanilla output buffer
    streams[ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE, channelSize * 4 * imageWidth * imageHeight * imageDepth, NULL, &error );
    test_error( error, "Unable to create output buffer" );


    /* Assign streams and execute */
    clSamplerWrapper sampler = clCreateSampler( context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error );
    test_error( error, "Unable to create sampler" );

    error = clSetKernelArg( kernel, 0, sizeof( streams[ 0 ] ), &streams[ 0 ] );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 1, sizeof( sampler ), &sampler );
    test_error( error, "Unable to set kernel arguments" );
    error = clSetKernelArg( kernel, 2, sizeof( streams[ 1 ] ), &streams[ 1 ] );
    test_error( error, "Unable to set kernel arguments" );

    glFlush();

    error = (*clEnqueueAcquireGLObjects_ptr)( queue, 1, &streams[ 0 ], 0, NULL, NULL);
    test_error( error, "Unable to acquire GL obejcts");

    /* Run the kernel */
    threads[ 0 ] = imageWidth;
    threads[ 1 ] = imageHeight;
    threads[ 2 ] = imageDepth;

    error = get_max_common_3D_work_group_size( context, kernel, threads, localThreads );
    test_error( error, "Unable to get work group size to use" );

    error = clEnqueueNDRangeKernel( queue, kernel, 3, NULL, threads, localThreads, 0, NULL, NULL );
    test_error( error, "Unable to execute test kernel" );


    error = (*clEnqueueReleaseGLObjects_ptr)( queue, 1, &streams[ 0 ], 0, NULL, NULL );
    test_error(error, "clEnqueueReleaseGLObjects failed");

    // Read results from the CL buffer
    *outResultBuffer = (void *)( new char[ channelSize * 4 * imageWidth * imageHeight * imageDepth ] );
    error = clEnqueueReadBuffer( queue, streams[ 1 ], CL_TRUE, 0, channelSize * 4 * imageWidth * imageHeight * imageDepth,
                                *outResultBuffer, 0, NULL, NULL );
    test_error( error, "Unable to read output CL buffer!" );

    return 0;
}

int test_image_format_read( cl_context context, cl_command_queue queue,
                            size_t width, size_t height, size_t depth,
                            GLenum target, GLenum format, GLenum internalFormat,
                            GLenum glType, ExplicitType type, MTdata d )
{
    int error;


    // Create the GL texture
    glTextureWrapper glTexture;
    void* tmp = CreateGLTexture3D( width, height, depth, target, format, internalFormat, glType, type, &glTexture, &error, d );
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
    error = test_image_read( context, queue, target, glTexture, width, height, depth, &clFormat, &actualType, (void **)&outBuffer );
    if( error != 0 )
        return error;
    BufferOwningPtr<char> actualResults(outBuffer);

    log_info( "- Read [%4d x %4d x %4d] : GL Texture : %s : %s : %s => CL Image : %s : %s \n",
                    (int)width, (int)height, (int)depth,
                    GetGLFormatName( format ), GetGLFormatName( internalFormat ), GetGLTypeName( glType),
                    GetChannelOrderName( clFormat.image_channel_order ), GetChannelTypeName( clFormat.image_channel_data_type ));

    // We have to convert our input buffer to the returned type, so we can validate.
    // This is necessary because OpenCL might not actually pick an internal format that actually matches our
    // input format (for example, if it picks a normalized format, the results will come out as floats instead of
    // going in as ints).

    BufferOwningPtr<char> convertedInputs(convert_to_expected( inputBuffer, width * height * depth, type, actualType ));
    if( convertedInputs == NULL )
        return -1;

    // Now we validate
    if( actualType == kFloat )
        return validate_float_results( convertedInputs, actualResults, width, height, depth );
    else
        return validate_integer_results( convertedInputs, actualResults, width, height, depth, get_explicit_type_size( actualType ) );
}


int test_images_read_3D( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    GLenum targets[] = { GL_TEXTURE_3D };

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

    size_t sizes[] = { 2, 4, 8, 16, 32, 64, 128 };
    size_t fmtIdx, tgtIdx;
    int error = 0;
    RandomSeed seed(gRandomSeed);

    size_t iter = sizeof(sizes)/sizeof(sizes[0]);

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
                if( test_image_format_read( context, queue, sizes[i], sizes[i], sizes[i],
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
            if( i == sizeof (sizes) / sizeof( sizes[0] ) )
            {
                log_info( "passed: Image read test for GL format  %s : %s : %s : %s\n\n",
                    GetGLTargetName( targets[ tgtIdx ] ),
                    GetGLFormatName( formats[ fmtIdx ].internal ),
                    GetGLBaseFormatName( formats[ fmtIdx ].format ),
                    GetGLTypeName( formats[ fmtIdx ].datatype ) );

            }
        }
    }

    return error;
}

