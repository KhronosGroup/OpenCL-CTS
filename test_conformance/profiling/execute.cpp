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
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <algorithm>

#include "procs.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"

#ifndef uchar
typedef unsigned char uchar;
#endif

//#define CREATE_OUTPUT    1

extern int writePPM( const char *filename, uchar *buf, int xsize, int ysize );



//--- the code for kernel executables
static const char *image_filter_src =
"constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
"\n"
"__kernel void image_filter( int n, int m, __global float *filter_weights,\n"
"              read_only image2d_t src_image, write_only image2d_t dst_image )\n"
"{\n"
"   int     i, j;\n"
"   int     indx = 0;\n"
"    int        tid_x = get_global_id(0);\n"
"    int        tid_y = get_global_id(1);\n"
"    float4  filter_result = (float4)( 0.f, 0.f, 0.f, 0.f );\n"
"\n"
"    for (i=-m/2; i<(m+1)/2; i++){\n"
"        for (j=-n/2; j<(n+1)/2; j++){\n"
"            float   w = filter_weights[indx++];\n"
"\n"
"            if (w != 0.0f){\n"
"                filter_result += w * read_imagef(src_image, sampler,\n"
"                                                 (int2)(tid_x + j, tid_y + i));\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    write_imagef(dst_image, (int2)(tid_x, tid_y), filter_result);\n"
"}\n";


//--- equivalent non-kernel code
static void read_imagef( int x, int y, int w, int h, int nChannels, uchar *src, float *srcRgb )
{
    // clamp the coords
    int x0 = std::min(std::max(x, 0), w - 1);
    int y0 = std::min(std::max(y, 0), h - 1);

    // get tine index
    int    indx = ( y0 * w + x0 ) * nChannels;

    // seed the return array
    int    i;
    for( i = 0; i < nChannels; i++ ){
        srcRgb[i] = (float)src[indx+i];
    }
}    // end read_imagef()


static void write_imagef( uchar *dst, int x, int y, int w, int h, int nChannels, float *dstRgb )
{
    // get tine index
    int    indx = ( y * w + x ) * nChannels;

    // seed the return array
    int    i;
    for( i = 0; i < nChannels; i++ ){
        dst[indx+i] = (uchar)dstRgb[i];
    }
}    // end write_imagef()


static void basicFilterPixel( int x, int y, int n, int m, int xsize, int ysize, int nChannels, const float *filter_weights, uchar *src, uchar *dst )
{
    int        i, j, k;
    int        indx = 0;
    float    filter_result[] = { 0.f, 0.f, 0.f, 0.f };
    float    srcRgb[4];

    for( i = -m/2; i < (m+1)/2; i++ ){
        for( j = -n/2; j < (n+1)/2; j++ ){
            float    w = filter_weights[indx++];

            if( w != 0 ){
                read_imagef( x + j, y + i, xsize, ysize, nChannels, src, srcRgb );
                for( k = 0; k < nChannels; k++ ){
                    filter_result[k] += w * srcRgb[k];
                }
            }
        }
    }

    write_imagef( dst, x, y, xsize, ysize, nChannels, filter_result );

}    // end basicFilterPixel()


//--- helper functions
static uchar *createImage( int elements, MTdata d)
{
    int        i;
    uchar    *ptr = (uchar *)malloc( elements * sizeof( cl_uchar ) );
    if( ! ptr )
        return NULL;

    for( i = 0; i < elements; i++ ){
        ptr[i] = (uchar)genrand_int32(d);
    }

    return ptr;

}    // end createImage()


static int verifyImages( uchar *ptr0, uchar *ptr1, uchar tolerance, int xsize, int ysize, int nChannels )
{
    int        x, y, z;
    uchar    *p0 = ptr0;
    uchar    *p1 = ptr1;

    for( y = 0; y < ysize; y++ ){
        for( x = 0; x < xsize; x++ ){
            for( z = 0; z < nChannels; z++ ){
                if( (uchar)abs( (int)( *p0++ - *p1++ ) ) > tolerance ){
                    log_error( "  images differ at x,y = %d,%d, channel = %d, %d to %d\n", x, y, z,
                              (int)p0[-1], (int)p1[-1] );
                    return -1;
                }
            }
        }
    }

    return 0;

}    // end verifyImages()


static int kernelFilter( cl_device_id device, cl_context context, cl_command_queue queue, int w, int h, int nChannels,
                         uchar *inptr, uchar *outptr )
{
    cl_program            program[1];
    cl_kernel            kernel[1];
    cl_mem                memobjs[3];
    cl_image_format        image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    cl_event            executeEvent;
    cl_ulong    queueStart, submitStart, writeStart, writeEnd;
    size_t                threads[2];
    float                filter_weights[] = { .1f, .1f, .1f, .1f, .2f, .1f, .1f, .1f, .1f };
    int                    filter_w = 3, filter_h = 3;
    int                    err = 0;

    // set thread dimensions
    threads[0] = w;
    threads[1] = h;

    // allocate the input and output image memory objects
    memobjs[0] =
        create_image_2d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format_desc, w, h, 0, inptr, &err);
    if( memobjs[0] == (cl_mem)0 ){
        log_error( " unable to create 2D image using create_image_2d\n" );
        return -1;
    }

    memobjs[1] = create_image_2d( context, CL_MEM_WRITE_ONLY, &image_format_desc, w, h, 0, NULL, &err );
    if( memobjs[1] == (cl_mem)0 ){
        log_error( " unable to create 2D image using create_image_2d\n" );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // allocate an array memory object to load the filter weights
    memobjs[2] = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * filter_w * filter_h, &filter_weights, &err);
    if( memobjs[2] == (cl_mem)0 ){
        log_error( " unable to create array using clCreateBuffer\n" );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // create the compute program
    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &image_filter_src, "image_filter" );
    if( err ){
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }


    // create kernel args object and set arg values.
    // set the args values
    err = clSetKernelArg( kernel[0], 0, sizeof( cl_int ), (void *)&filter_w );
    err |= clSetKernelArg( kernel[0], 1, sizeof( cl_int ), (void *)&filter_h );
    err |= clSetKernelArg( kernel[0], 2, sizeof( cl_mem ), (void *)&memobjs[2] );
    err |= clSetKernelArg( kernel[0], 3, sizeof( cl_mem ), (void *)&memobjs[0] );
    err |= clSetKernelArg( kernel[0], 4, sizeof( cl_mem ), (void *)&memobjs[1] );

    if( err != CL_SUCCESS ){
        print_error( err, "clSetKernelArg failed\n" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, &executeEvent );

    if( err != CL_SUCCESS ){
        print_error( err, "clEnqueueNDRangeKernel failed\n" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &executeEvent );
    if( err != CL_SUCCESS )
    {
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // test profiling
    while( ( err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // read output image
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { w, h, 1 };
    err = clEnqueueReadImage( queue, memobjs[1], true, origin, region, 0, 0, outptr, 0, NULL, NULL);
    if( err != CL_SUCCESS ){
        print_error( err, "clReadImage failed\n" );
    clReleaseEvent( executeEvent );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject( memobjs[2] );
        clReleaseMemObject( memobjs[1] );
        clReleaseMemObject( memobjs[0] );
        return -1;
    }

    // release event, kernel, program, and memory objects
  clReleaseEvent( executeEvent );
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    clReleaseMemObject( memobjs[2] );
    clReleaseMemObject( memobjs[1] );
    clReleaseMemObject( memobjs[0] );

    if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
        err = -1;

    return err;

}    // end kernelFilter()


static int basicFilter( int w, int h, int nChannels, uchar *inptr, uchar *outptr )
{
    const float    filter_weights[] = { .1f, .1f, .1f, .1f, .2f, .1f, .1f, .1f, .1f };
    int            filter_w = 3, filter_h = 3;
    int            x, y;

    for( y = 0; y < h; y++ ){
        for( x = 0; x < w; x++ ){
            basicFilterPixel( x, y, filter_w, filter_h, w, h, nChannels, filter_weights, inptr, outptr );
        }
    }

    return 0;

}    // end of basicFilter()


int test_execute( cl_device_id device, cl_context context, cl_command_queue queue, int num_elements )
{
    uchar    *inptr;
    uchar    *outptr[2];
    int        w = 256, h = 256;
    int        nChannels = 4;
    int        nElements = w * h * nChannels;
    int        err = 0;
    MTdata  d;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    d = init_genrand( gRandomSeed );
    inptr = createImage( nElements, d );
    free_mtdata( d);    d = NULL;

    if( ! inptr ){
        log_error( " unable to allocate %d bytes of memory for image\n", nElements );
        return -1;
    }

    outptr[0] = (uchar *)malloc( nElements * sizeof( cl_uchar ) );
    if( ! outptr[0] ){
        log_error( " unable to allocate %d bytes of memory for output image #1\n", nElements );
        free( (void *)inptr );
        return -1;
    }

    outptr[1] = (uchar *)malloc( nElements * sizeof( cl_uchar ) );
    if( ! outptr[1] ){
        log_error( " unable to allocate %d bytes of memory for output image #2\n", nElements );
        free( (void *)outptr[0] );
        free( (void *)inptr );
        return -1;
    }

    err = kernelFilter( device, context, queue, w, h, nChannels, inptr, outptr[0] );

    if( ! err ){
        basicFilter( w, h, nChannels, inptr, outptr[1] );

        // verify that the images are the same
        err = verifyImages( outptr[0], outptr[1], (uchar)0x1, w, h, nChannels );
        if( err )
            log_error( " images do not match\n" );
    }

    // clean up
    free( (void *)outptr[1] );
    free( (void *)outptr[0] );
    free( (void *)inptr );

    return err;

}    // end execute()



