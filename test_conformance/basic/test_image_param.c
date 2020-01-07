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
#include "harness/compat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"
#include "harness/typeWrappers.h"
#include "harness/imageHelpers.h"
#include "harness/conversions.h"


static const char *param_kernel[] = {
"__kernel void test_fn(read_only image2d_t srcimg, sampler_t sampler, __global float4 *results )\n"
"{\n"
"    int            tid_x = get_global_id(0);\n"
"    int            tid_y = get_global_id(1);\n"
"    results[ tid_y * get_image_width( srcimg ) + tid_x ] = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"\n"
"}\n" };

int validate_results( size_t width, size_t height, cl_image_format &format, char *inputData, cl_float *actualResults )
{
    for( size_t i = 0; i < width * height; i++ )
    {
        cl_float expected[ 4 ], tolerance;

        switch( format.image_channel_data_type )
        {
            case CL_UNORM_INT8:
            {
                cl_uchar *p = (cl_uchar *)inputData;
                expected[ 0 ] = p[ 0 ] / 255.f;
                expected[ 1 ] = p[ 1 ] / 255.f;
                expected[ 2 ] = p[ 2 ] / 255.f;
                expected[ 3 ] = p[ 3 ] / 255.f;
                tolerance = 1.f / 255.f;
                break;
            }
            case CL_SNORM_INT8:
            {
                cl_char *p = (cl_char *)inputData;
                expected[ 0 ] = fmaxf( p[ 0 ] / 127.f, -1.f );
                expected[ 1 ] = fmaxf( p[ 1 ] / 127.f, -1.f );
                expected[ 2 ] = fmaxf( p[ 2 ] / 127.f, -1.f );
                expected[ 3 ] = fmaxf( p[ 3 ] / 127.f, -1.f );
                tolerance = 1.f / 127.f;
                break;
            }
            case CL_UNSIGNED_INT8:
            {
                cl_uchar *p = (cl_uchar *)inputData;
                expected[ 0 ] = p[ 0 ];
                expected[ 1 ] = p[ 1 ];
                expected[ 2 ] = p[ 2 ];
                expected[ 3 ] = p[ 3 ];
                tolerance = 1.f / 127.f;
                break;
            }
            case CL_SIGNED_INT8:
            {
                cl_short *p = (cl_short *)inputData;
                expected[ 0 ] = p[ 0 ];
                expected[ 1 ] = p[ 1 ];
                expected[ 2 ] = p[ 2 ];
                expected[ 3 ] = p[ 3 ];
                tolerance = 1.f / 127.f;
                break;
            }
            case CL_UNORM_INT16:
            {
                cl_ushort *p = (cl_ushort *)inputData;
                expected[ 0 ] = p[ 0 ] / 65535.f;
                expected[ 1 ] = p[ 1 ] / 65535.f;
                expected[ 2 ] = p[ 2 ] / 65535.f;
                expected[ 3 ] = p[ 3 ] / 65535.f;
                tolerance = 1.f / 65535.f;
                break;
            }
            case CL_UNSIGNED_INT32:
            {
                cl_uint *p = (cl_uint *)inputData;
                expected[ 0 ] = p[ 0 ];
                expected[ 1 ] = p[ 1 ];
                expected[ 2 ] = p[ 2 ];
                expected[ 3 ] = p[ 3 ];
                tolerance = 0.0001f;
                break;
            }
            case CL_FLOAT:
            {
                cl_float *p = (cl_float *)inputData;
                expected[ 0 ] = p[ 0 ];
                expected[ 1 ] = p[ 1 ];
                expected[ 2 ] = p[ 2 ];
                expected[ 3 ] = p[ 3 ];
                tolerance = 0.0001f;
                break;
            }
            default:
                // Should never get here
                break;
        }

        if( format.image_channel_order == CL_BGRA )
        {
            cl_float tmp = expected[ 0 ];
            expected[ 0 ] = expected[ 2 ];
            expected[ 2 ] = tmp;
        }

        // Within an error tolerance, make sure the results match
        cl_float error1 = fabsf( expected[ 0 ] - actualResults[ 0 ] );
        cl_float error2 = fabsf( expected[ 1 ] - actualResults[ 1 ] );
        cl_float error3 = fabsf( expected[ 2 ] - actualResults[ 2 ] );
        cl_float error4 = fabsf( expected[ 3 ] - actualResults[ 3 ] );

        if( error1 > tolerance || error2 > tolerance || error3 > tolerance || error4 > tolerance )
        {
            log_error( "ERROR: Sample %d did not validate against expected results for %d x %d %s:%s image\n", (int)i, (int)width, (int)height,
                            GetChannelOrderName( format.image_channel_order ), GetChannelTypeName( format.image_channel_data_type ) );
            log_error( "    Expected: %f %f %f %f\n", (float)expected[ 0 ], (float)expected[ 1 ], (float)expected[ 2 ], (float)expected[ 3 ] );
            log_error( "      Actual: %f %f %f %f\n", (float)actualResults[ 0 ], (float)actualResults[ 1 ], (float)actualResults[ 2 ], (float)actualResults[ 3 ] );

            // Check real quick a special case error here
            cl_float error1 = fabsf( expected[ 3 ] - actualResults[ 0 ] );
            cl_float error2 = fabsf( expected[ 2 ] - actualResults[ 1 ] );
            cl_float error3 = fabsf( expected[ 1 ] - actualResults[ 2 ] );
            cl_float error4 = fabsf( expected[ 0 ] - actualResults[ 3 ] );
            if( error1 <= tolerance && error2 <= tolerance && error3 <= tolerance && error4 <= tolerance )
            {
                log_error( "\t(Kernel did not respect change in channel order)\n" );
            }
            return -1;
        }

        // Increment and go
        actualResults += 4;
        inputData += get_format_type_size( &format ) * 4;
    }

    return 0;
}

int test_image_param(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t              sizes[] = { 64, 100, 128, 250, 512 };
    cl_image_format      formats[] = { { CL_RGBA, CL_UNORM_INT8 }, { CL_RGBA, CL_UNORM_INT16 }, { CL_RGBA, CL_FLOAT }, { CL_BGRA, CL_UNORM_INT8 } };
    cl_image_format  *supported_formats;
    ExplicitType      types[] =  { kUChar, kUShort, kFloat, kUChar };
    int               error;
    size_t            i, j, idx;
    size_t            threads[ 2 ];
    MTdata            d;
    int supportsBGRA = 0;
    cl_uint numSupportedFormats = 0;

    const size_t numSizes = sizeof( sizes ) / sizeof( sizes[ 0 ] );
    const size_t numFormats = sizeof( formats ) / sizeof( formats[ 0 ] );
    const size_t numAttempts = numSizes * numFormats;


    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper streams[ numAttempts ][ 2 ];
    BufferOwningPtr<char> inputs[ numAttempts ];

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

       if(gIsEmbedded)
    {
        /* Get the supported image formats to see if BGRA is supported */
        clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, NULL, &numSupportedFormats);
        supported_formats = (cl_image_format *) malloc(sizeof(cl_image_format) * numSupportedFormats);
        clGetSupportedImageFormats (context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, numFormats, supported_formats, NULL);

        for(i = 0; i < numSupportedFormats; i++)
        {
            if(supported_formats[i].image_channel_order == CL_BGRA)
            {
                supportsBGRA = 1;
                break;
            }
        }
    }
    else
    {
        supportsBGRA = 1;
    }

    d = init_genrand( gRandomSeed );
    for( i = 0, idx = 0; i < numSizes; i++ )
    {
        for( j = 0; j < numFormats; j++, idx++ )
        {
            if(formats[j].image_channel_order == CL_BGRA && !supportsBGRA)
                continue;

            // For each attempt, we create a pair: an input image, whose parameters keep changing, and an output buffer
            // that we can read values from. The output buffer will remain consistent to ensure that any changes we
            // witness are due to the image changes
            inputs[ idx ].reset(create_random_data( types[ j ], d, sizes[ i ] * sizes[ i ] * 4 ));

            streams[ idx ][ 0 ] = create_image_2d( context, CL_MEM_COPY_HOST_PTR, &formats[ j ], sizes[ i ], sizes[ i ], 0, inputs[ idx ], &error );
            {
                char err_str[256];
                sprintf(err_str, "Unable to create input image for format %s order %s" ,
                                  GetChannelOrderName( formats[j].image_channel_order ),
                                  GetChannelTypeName( formats[j].image_channel_data_type ));
                test_error( error, err_str);
            }

            streams[ idx ][ 1 ] = clCreateBuffer( context, CL_MEM_READ_WRITE, sizes[ i ] * sizes[ i ] * 4 * sizeof( cl_float ), NULL, &error );
            test_error( error, "Unable to create output buffer" );
        }
    }
    free_mtdata(d); d = NULL;

    // Create a single kernel to use for all the tests
    error = create_single_kernel_helper( context, &program, &kernel, 1, param_kernel, "test_fn" );
    test_error( error, "Unable to create testing kernel" );

    // Also create a sampler to use for all the runs
    clSamplerWrapper sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &error );
    test_error( error, "clCreateSampler failed" );

    // Set up the arguments for each and queue
    for( i = 0, idx = 0; i < numSizes; i++ )
    {
        for( j = 0; j < numFormats; j++, idx++ )
        {
            if(formats[j].image_channel_order == CL_BGRA && !supportsBGRA)
                continue;

            error = clSetKernelArg( kernel, 0, sizeof( streams[ idx ][ 0 ] ), &streams[ idx ][ 0 ] );
            error |= clSetKernelArg( kernel, 1, sizeof( sampler ), &sampler );
            error |= clSetKernelArg( kernel, 2, sizeof( streams[ idx ][ 1 ] ), &streams[ idx ][ 1 ]);
            test_error( error, "Unable to set kernel arguments" );

            threads[ 0 ] = threads[ 1 ] = (size_t)sizes[ i ];

            error = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, NULL );
            test_error( error, "clEnqueueNDRangeKernel failed" );
        }
    }

    // Now go through each combo and validate the results
    for( i = 0, idx = 0; i < numSizes; i++ )
    {
        for( j = 0; j < numFormats; j++, idx++ )
        {
            if(formats[j].image_channel_order == CL_BGRA && !supportsBGRA)
                continue;

            BufferOwningPtr<cl_float> output(malloc(sizeof(cl_float) * sizes[ i ] * sizes[ i ] * 4 ));

            error = clEnqueueReadBuffer( queue, streams[ idx ][ 1 ], CL_TRUE, 0, sizes[ i ] * sizes[ i ] * 4 * sizeof( cl_float ), output, 0, NULL, NULL );
            test_error( error, "Unable to read results" );

            error = validate_results( sizes[ i ], sizes[ i ], formats[ j ], inputs[ idx ], output );
            if( error )
                return -1;
        }
    }

    return 0;
}
