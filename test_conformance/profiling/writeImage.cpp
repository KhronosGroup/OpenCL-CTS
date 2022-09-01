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
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"
#include "harness/errorHelpers.h"

//--- the code for the kernel executables
static const char *readKernelCode[] = {
"__kernel void testReadf(read_only image2d_t srcimg, __global float4 *dst)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"     dst[indx].x = color.x;\n"
"     dst[indx].y = color.y;\n"
"     dst[indx].z = color.z;\n"
"     dst[indx].w = color.w;\n"
"\n"
"}\n",

"__kernel void testReadi(read_only image2d_t srcimg, __global uchar4 *dst)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    int4    color;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    color = read_imagei(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"  uchar4 dst_write;\n"
"     dst_write.x = (uchar)color.x;\n"
"     dst_write.y = (uchar)color.y;\n"
"     dst_write.z = (uchar)color.z;\n"
"     dst_write.w = (uchar)color.w;\n"
"  dst[indx] = dst_write;\n"
"\n"
"}\n",

"__kernel void testReadui(read_only image2d_t srcimg, __global uchar4 *dst)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(srcimg) + tid_x;\n"
"    uint4    color;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    color = read_imageui(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"  uchar4 dst_write;\n"
"     dst_write.x = (uchar)color.x;\n"
"     dst_write.y = (uchar)color.y;\n"
"     dst_write.z = (uchar)color.z;\n"
"     dst_write.w = (uchar)color.w;\n"
"  dst[indx] = dst_write;\n"
"\n"
"}\n",

"__kernel void testWritef(__global uchar *src, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    indx *= 4;\n"
"    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);\n"
"     color /= (float4)(255.f, 255.f, 255.f, 255.f);\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testWritei(__global char *src, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    int4    color;\n"
"\n"
"    indx *= 4;\n"
"     color.x = (int)src[indx+0];\n"
"     color.y = (int)src[indx+1];\n"
"     color.z = (int)src[indx+2];\n"
"     color.w = (int)src[indx+3];\n"
"    write_imagei(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testWriteui(__global uchar *src, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    indx = tid_y * get_image_width(dstimg) + tid_x;\n"
"    uint4    color;\n"
"\n"
"    indx *= 4;\n"
"     color.x = (uint)src[indx+0];\n"
"     color.y = (uint)src[indx+1];\n"
"     color.z = (uint)src[indx+2];\n"
"     color.w = (uint)src[indx+3];\n"
"    write_imageui(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testReadWriteff(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, CLK_DEFAULT_SAMPLER, (int2)(tid_x, tid_y));\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testReadWriteii(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int4    color;\n"
"\n"
"    color = read_imagei(srcimg, CLK_DEFAULT_SAMPLER, (int2)(tid_x, tid_y));\n"
"    write_imagei(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testReadWriteuiui(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    uint4    color;\n"
"\n"
"    color = read_imageui(srcimg, CLK_DEFAULT_SAMPLER, (int2)(tid_x, tid_y));\n"
"    write_imageui(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n",

"__kernel void testReadWritefi(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 colorf;\n"
"     int4    colori;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colorf = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
// since we are going from unsigned to signed, be sure to convert
// values greater 0.5 to negative values
"     if( colorf.x >= 0.5f )\n"
"         colori.x = (int)( ( colorf.x - 1.f ) * 255.f );\n"
"     else\n"
"         colori.x = (int)( colorf.x * 255.f );\n"
"     if( colorf.y >= 0.5f )\n"
"         colori.y = (int)( ( colorf.y - 1.f ) * 255.f );\n"
"     else\n"
"         colori.y = (int)( colorf.y * 255.f );\n"
"     if( colorf.z >= 0.5f )\n"
"         colori.z = (int)( ( colorf.z - 1.f ) * 255.f );\n"
"     else\n"
"         colori.z = (int)( colorf.z * 255.f );\n"
"     if( colorf.w >= 0.5f )\n"
"         colori.w = (int)( ( colorf.w - 1.f ) * 255.f );\n"
"     else\n"
"         colori.w = (int)( colorf.w * 255.f );\n"
"    write_imagei(dstimg, (int2)(tid_x, tid_y), colori);\n"
"\n"
"}\n",

"__kernel void testReadWritefui(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4    colorf;\n"
"     uint4    colorui;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colorf = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"     colorui.x = (uint)( colorf.x * 255.f );\n"
"     colorui.y = (uint)( colorf.y * 255.f );\n"
"     colorui.z = (uint)( colorf.z * 255.f );\n"
"     colorui.w = (uint)( colorf.w * 255.f );\n"
"    write_imageui(dstimg, (int2)(tid_x, tid_y), colorui);\n"
"\n"
"}\n",

"__kernel void testReadWriteif(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int4    colori;\n"
"    float4    colorf;\n"
"\n"
// since we are going from signed to unsigned, we need to adjust the rgba values from
// from the signed image to add 256 to the signed image values less than 0.
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colori = read_imagei(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"     if( colori.x < 0 )\n"
"        colorf.x = ( (float)colori.x + 256.f ) / 255.f;\n"
"     else\n"
"        colorf.x = (float)colori.x / 255.f;\n"
"     if( colori.y < 0 )\n"
"        colorf.y = ( (float)colori.y + 256.f ) / 255.f;\n"
"     else\n"
"        colorf.y = (float)colori.y / 255.f;\n"
"     if( colori.z < 0 )\n"
"        colorf.z = ( (float)colori.z + 256.f ) / 255.f;\n"
"     else\n"
"        colorf.z = (float)colori.z / 255.f;\n"
"     if( colori.w < 0 )\n"
"        colorf.w = ( (float)colori.w + 256.f ) / 255.f;\n"
"     else\n"
"        colorf.w = (float)colori.w / 255.f;\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), colorf);\n"
"\n"
"}\n",

"__kernel void testReadWriteiui(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int4    colori;\n"
"    uint4    colorui;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colori = read_imagei(srcimg, sampler, (int2)(tid_x, tid_y));\n"
// since we are going from signed to unsigned, we need to adjust the rgba values from
// from the signed image to add 256 to the signed image values less than 0.
"     if( colori.x < 0 )\n"
"        colorui.x = (uint)( colori.x + 256 );\n"
"     else\n"
"        colorui.x = (uint)colori.x;\n"
"     if( colori.y < 0 )\n"
"        colorui.y = (uint)( colori.y + 256 );\n"
"     else\n"
"        colorui.y = (uint)colori.y;\n"
"     if( colori.z < 0 )\n"
"        colorui.z = (uint)( colori.z + 256 );\n"
"     else\n"
"        colorui.z = (uint)colori.z;\n"
"     if( colori.w < 0 )\n"
"        colorui.w = (uint)( colori.w + 256 );\n"
"     else\n"
"        colorui.w = (uint)colori.w;\n"
"    write_imageui(dstimg, (int2)(tid_x, tid_y), colorui);\n"
"\n"
"}\n",

"__kernel void testReadWriteuif(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    uint4    colorui;\n"
"    float4    colorf;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colorui = read_imageui(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"     colorf.x = (float)colorui.x / 255.f;\n"
"     colorf.y = (float)colorui.y / 255.f;\n"
"     colorf.z = (float)colorui.z / 255.f;\n"
"     colorf.w = (float)colorui.w / 255.f;\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), colorf);\n"
"\n"
"}\n",

"__kernel void testReadWriteuii(read_only image2d_t srcimg, write_only image2d_t dstimg)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    uint4    colorui;\n"
"    int4    colori;\n"
"\n"
"    const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE;\n"
"    colorui = read_imageui(srcimg, sampler, (int2)(tid_x, tid_y));\n"
// since we are going from unsigned to signed, be sure to convert
// values greater 0.5 to negative values
"     if( colorui.x >= 128U )\n"
"         colori.x = (int)colorui.x - 256;\n"
"     else\n"
"         colori.x = (int)colorui.x;\n"
"     if( colorui.y >= 128U )\n"
"         colori.y = (int)colorui.y - 256;\n"
"     else\n"
"         colori.y = (int)colorui.y;\n"
"     if( colorui.z >= 128U )\n"
"         colori.z = (int)colorui.z - 256;\n"
"     else\n"
"         colori.z = (int)colorui.z;\n"
"     if( colorui.w >= 128U )\n"
"         colori.w = (int)colorui.w - 256;\n"
"     else\n"
"         colori.w = (int)colorui.w;\n"
"    write_imagei(dstimg, (int2)(tid_x, tid_y), colori);\n"
"\n"
"}\n" };

static const char *readKernelName[] = { "testReadf", "testReadi", "testReadui", "testWritef", "testWritei", "testWriteui",
"testReadWriteff", "testReadWriteii", "testReadWriteuiui", "testReadWritefi",
"testReadWritefui", "testReadWriteif", "testReadWriteiui", "testReadWriteuif",
"testReadWriteuii" };


static cl_uchar *generateImage( int n, MTdata d )
{
    cl_uchar    *ptr = (cl_uchar *)malloc( n * sizeof( cl_uchar ) );
    int        i;

    for( i = 0; i < n; i++ ){
        ptr[i] = (cl_uchar)genrand_int32(d);
    }

    return ptr;

}


static char *generateSignedImage( int n, MTdata d )
{
    char    *ptr = (char *)malloc( n * sizeof( char ) );
    int        i;

    for( i = 0; i < n; i++ ){
        ptr[i] = (char)genrand_int32(d);
    }

    return ptr;

}


static int verifyImage( cl_uchar *image, cl_uchar *outptr, int w, int h )
{
    int     i;

    for( i = 0; i < w * h * 4; i++ ){
        if( outptr[i] != image[i] ){
            log_error("Image verification failed at offset %d. Actual value=%d, expected value=%d\n", i, outptr[i], image[i]);
            return -1;
        }
    }

    return 0;
}

static int verifyImageFloat ( cl_double *refptr, cl_float *outptr, int w, int h )
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != (float)refptr[i])
        {
            float ulps = Ulp_Error( outptr[i], refptr[i]);

            if(! (fabsf(ulps) < 1.5f) )
            {
                log_error( "ERROR: Data sample %d does not validate! Expected (%a), got (%a), ulp %f\n",
                    (int)i, refptr[i], outptr[ i ],  ulps );
                return -1;
            }
        }
    }

    return 0;
}

static double *prepareReference( cl_uchar *inptr, int w, int h)
{
    int        i;
    double    *refptr = (double *)malloc( w * h * 4*sizeof( double ) );
    if ( !refptr )
    {
        log_error( "Unable to allocate refptr at %d x %d\n", (int)w, (int)h );
        return 0;
    }
    for( i = 0; i < w * h * 4; i++ ) {
        refptr[i] = ((double)inptr[i])/255;
    }
    return refptr;
}

//----- the test functions
int write_image( cl_device_id device, cl_context context, cl_command_queue queue, int numElements, const char *code,
                 const char *name, cl_image_format image_format_desc, int readFloat )
{
    cl_mem            memobjs[2];
    cl_program        program[1];
    void            *inptr;
    double            *refptr = NULL;
    void            *dst = NULL;
    cl_kernel        kernel[1];
    cl_event        writeEvent;
    cl_ulong    queueStart, submitStart, writeStart, writeEnd;
    size_t    threads[2];
    int                err;
    int                w = 64, h = 64;
    cl_mem_flags    flags;
    size_t            element_nbytes;
    size_t            num_bytes;
    size_t            channel_nbytes = sizeof( cl_uchar );
    MTdata          d;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    if (readFloat)
        channel_nbytes = sizeof( cl_float );

    element_nbytes = channel_nbytes * get_format_channel_count( &image_format_desc );
    num_bytes = w * h * element_nbytes;

    threads[0] = (size_t)w;
    threads[1] = (size_t)h;

    d = init_genrand( gRandomSeed );
    if( image_format_desc.image_channel_data_type == CL_SIGNED_INT8 )
        inptr = (void *)generateSignedImage( w * h * 4, d );
    else
        inptr = (void *)generateImage( w * h * 4, d );
    free_mtdata(d); d = NULL;
    if( ! inptr ){
        log_error("unable to allocate inptr at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    dst = malloc( num_bytes );
    if( ! dst ){
        free( (void *)inptr );
        log_error("unable to allocate dst at %d x %d\n", (int)w, (int)h );
        return -1;
    }

    // allocate the input and output image memory objects
    flags = CL_MEM_READ_WRITE;
    memobjs[0] = create_image_2d( context, flags, &image_format_desc, w, h, 0, NULL, &err );
    if( memobjs[0] == (cl_mem)0 ){
        free( dst );
        free( (void *)inptr );
        log_error("unable to create Image2D\n");
        return -1;
    }

    memobjs[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                channel_nbytes * 4 * w * h, NULL, &err);
    if( memobjs[1] == (cl_mem)0 ){
        free( dst );
        free( (void *)inptr );
        clReleaseMemObject(memobjs[0]);
        log_error("unable to create array\n");
        return -1;
    }

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { w, h, 1 };
    err = clEnqueueWriteImage( queue, memobjs[0], false, origin, region, 0, 0, inptr, 0, NULL, &writeEvent );
    if( err != CL_SUCCESS ){
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        print_error(err, "clWriteImage failed");
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &writeEvent );
    if( err != CL_SUCCESS )
    {
        print_error( err, "clWaitForEvents failed" );
        clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    // test profiling
    while( ( err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( writeEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &code, name );
    if( err ){
        log_error( "Unable to create program and kernel\n" );
    clReleaseEvent(writeEvent);
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&memobjs[0] );
    err |= clSetKernelArg( kernel[0], 1, sizeof( cl_mem ), (void *)&memobjs[1] );
    if( err != CL_SUCCESS ){
        log_error( "clSetKernelArg failed\n" );
    clReleaseEvent(writeEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clEnqueueNDRangeKernel(queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );

    if( err != CL_SUCCESS ){
        print_error( err, "clEnqueueNDRangeKernel failed" );
    clReleaseEvent(writeEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clEnqueueReadBuffer( queue, memobjs[1], true, 0, num_bytes, dst, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clEnqueueReadBuffer failed" );
    clReleaseEvent(writeEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    if ( readFloat )
    {
        refptr = prepareReference( (cl_uchar *)inptr, w, h );
        if ( refptr )
        {
            err = verifyImageFloat( refptr, (cl_float *)dst, w, h );
            free ( refptr );
        }
        else
            err = -1;
    }
    else
        err = verifyImage( (cl_uchar *)inptr, (cl_uchar *)dst, w, h );

    if( err )
    {
        log_error( "Image failed to verify.\n" );
    }
    else
    {
        log_info( "Image verified.\n" );
    }

    // cleanup
  clReleaseEvent(writeEvent);
    clReleaseKernel( kernel[0] );
    clReleaseProgram( program[0] );
    clReleaseMemObject(memobjs[0]);
    clReleaseMemObject(memobjs[1]);
    free( dst );
    free( inptr );

    if (check_times(queueStart, submitStart, writeStart, writeEnd, device))
        err = -1;

    return err;

}    // end write_image()


int test_write_image_float( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // 0 to 255 for unsigned image data
    return write_image( device, context, queue, numElements, readKernelCode[0], readKernelName[0], image_format_desc, 1 );

}


int test_write_image_char( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_SIGNED_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // -128 to 127 for signed iamge data
    return write_image( device, context, queue, numElements, readKernelCode[1], readKernelName[1], image_format_desc, 0 );

}


int test_write_image_uchar( cl_device_id device, cl_context context, cl_command_queue queue, int numElements )
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_UNSIGNED_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // 0 to 255 for unsigned image data
    return write_image( device, context, queue, numElements, readKernelCode[2], readKernelName[2], image_format_desc, 0 );

}


