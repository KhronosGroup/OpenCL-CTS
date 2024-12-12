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
"}\n" };

static const char *readKernelName[] = { "testWritef", "testWritei", "testWriteui" };


//--- helper functions
static cl_uchar *generateImage( int n, MTdata d )
{
    cl_uchar    *ptr = (cl_uchar *)malloc( n * sizeof( cl_uchar ) );
    int        i;

    for( i = 0; i < n; i++ ){
        ptr[i] = (cl_uchar)genrand_int32( d );
    }

    return ptr;

}


static char *generateSignedImage( int n, MTdata d )
{
    char    *ptr = (char *)malloc( n * sizeof( char ) );
    int        i;

    for( i = 0; i < n; i++ ){
        ptr[i] = (char)genrand_int32( d );
    }

    return ptr;

}


static int verifyImage( cl_uchar *image, cl_uchar *outptr, int w, int h )
{
    int     i;

    for( i = 0; i < w * h * 4; i++ ){
        if( outptr[i] != image[i] ){
            return -1;
        }
    }

    return 0;
}


//----- the test functions
int read_image( cl_device_id device, cl_context context, cl_command_queue queue, int numElements, const char *code, const char *name,
                   cl_image_format image_format_desc )
{
    cl_mem            memobjs[2];
    cl_program        program[1];
    void            *inptr;
    void            *dst = NULL;
    cl_kernel        kernel[1];
    cl_event        readEvent;
    cl_ulong        queueStart, submitStart, readStart, readEnd;
    size_t          threads[2];
    int                err;
    int                w = 64, h = 64;
    cl_mem_flags    flags;
    size_t            element_nbytes;
    size_t            num_bytes;
    size_t            channel_nbytes = sizeof( cl_uchar );
    MTdata          d;


    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    element_nbytes = channel_nbytes * get_format_channel_count( &image_format_desc );
    num_bytes = w * h * element_nbytes;

    threads[0] = (size_t)w;
    threads[1] = (size_t)h;

    d = init_genrand( gRandomSeed );
    if( image_format_desc.image_channel_data_type == CL_SIGNED_INT8 )
        inptr = (void *)generateSignedImage( w * h * 4, d );
    else
        inptr = (void *)generateImage( w * h * 4, d );
    free_mtdata(d);         d = NULL;

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

    err = clEnqueueWriteBuffer( queue, memobjs[1], true, 0, num_bytes, inptr, 0, NULL, NULL );
    if( err != CL_SUCCESS ){
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        log_error("clWriteArray failed\n");
        return -1;
    }

    err = create_single_kernel_helper( context, &program[0], &kernel[0], 1, &code, name );
    if( err ){
        log_error( "Unable to create program and kernel\n" );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clSetKernelArg( kernel[0], 0, sizeof( cl_mem ), (void *)&memobjs[1] );
    err |= clSetKernelArg( kernel[0], 1, sizeof( cl_mem ), (void *)&memobjs[0] );
    if( err != CL_SUCCESS ){
        log_error( "clSetKernelArg failed\n" );
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
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t)w, (size_t)h, 1 };
    err = clEnqueueReadImage( queue, memobjs[0], false, origin, region, 0, 0, dst, 0, NULL, &readEvent );
    if( err != CL_SUCCESS ){
        print_error( err, "clReadImage2D failed" );
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    // This synchronization point is needed in order to assume the data is valid.
    // Getting profiling information is not a synchronization point.
    err = clWaitForEvents( 1, &readEvent );
    if( err != CL_SUCCESS )
    {
    clReleaseEvent(readEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_QUEUED, sizeof( cl_ulong ), &queueStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(readEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    while( ( err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof( cl_ulong ), &submitStart, NULL ) ) ==
          CL_PROFILING_INFO_NOT_AVAILABLE );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(readEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &readStart, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(readEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = clGetEventProfilingInfo( readEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &readEnd, NULL );
    if( err != CL_SUCCESS ){
        print_error( err, "clGetEventProfilingInfo failed" );
    clReleaseEvent(readEvent);
        clReleaseKernel( kernel[0] );
        clReleaseProgram( program[0] );
        clReleaseMemObject(memobjs[0]);
        clReleaseMemObject(memobjs[1]);
        free( dst );
        free( inptr );
        return -1;
    }

    err = verifyImage( (cl_uchar *)inptr, (cl_uchar *)dst, w, h );
    if( err ){
        log_error( "Image failed to verify.\n" );
    }
    else{
        log_info( "Image verified.\n" );
    }

  clReleaseEvent(readEvent);
    clReleaseKernel(kernel[0]);
    clReleaseProgram(program[0]);
    clReleaseMemObject(memobjs[0]);
    clReleaseMemObject(memobjs[1]);
    free(dst);
    free(inptr);

  if (check_times(queueStart, submitStart, readStart, readEnd, device))
      err = -1;

  return err;

}    // end read_image()


REGISTER_TEST(read_image_float)
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // 0 to 255 for unsigned image data
    return read_image(device, context, queue, num_elements, readKernelCode[0],
                      readKernelName[0], image_format_desc);
}


REGISTER_TEST(read_image_char)
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_SIGNED_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // -128 to 127 for signed iamge data
    return read_image(device, context, queue, num_elements, readKernelCode[1],
                      readKernelName[1], image_format_desc);
}


REGISTER_TEST(read_image_uchar)
{
    cl_image_format    image_format_desc = { CL_RGBA, CL_UNSIGNED_INT8 };
    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )
    // 0 to 255 for unsigned image data
    return read_image(device, context, queue, num_elements, readKernelCode[2],
                      readKernelName[2], image_format_desc);
}


