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

#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "procs.h"
#include "harness/testHarness.h"
#include "harness/typeWrappers.h"
#include "harness/errorHelpers.h"

static const char *read3d_kernel_code =
"\n"
"__kernel void read3d(read_only image3d_t srcimg, __global unsigned char *dst, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    int    tid_z = get_global_id(2);\n"
"    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));\n"
"    indx *= 4;\n"
"    dst[indx+0] = (unsigned char)(color.x * 255.0f);\n"
"    dst[indx+1] = (unsigned char)(color.y * 255.0f);\n"
"    dst[indx+2] = (unsigned char)(color.z * 255.0f);\n"
"    dst[indx+3] = (unsigned char)(color.w * 255.0f);\n"
"\n"
"}\n";


static cl_uchar *createImage( int elements, MTdata d )
{
    int i;
    cl_uchar *ptr = (cl_uchar *)malloc( elements * sizeof( cl_uchar ) );
    if( ! ptr )
        return NULL;

    for( i = 0; i < elements; i++ ){
        ptr[i] = (cl_uchar)genrand_int32(d);
    }

    return ptr;

}    // end createImage()


static int verifyImages( cl_uchar *ptr0, cl_uchar *ptr1, cl_uchar tolerance, int xsize, int ysize, int zsize, int nChannels )
{
    int x, y, z, c;
    cl_uchar *p0 = ptr0;
    cl_uchar *p1 = ptr1;

    for( z = 0; z < zsize; z++ ){
        for( y = 0; y < ysize; y++ ){
            for( x = 0; x < xsize; x++ ){
                for( c = 0; c < nChannels; c++ ){
                    if( (cl_uchar)abs( (int)( *p0++ - *p1++ ) ) > tolerance ){
                        log_error( "  images differ at x,y,z = %d,%d,%d channel = %d, %d to %d\n",
                                  x, y, z, c, (int)p0[-1], (int)p1[-1] );
                        return -1;
                    }
                }
            }
        }
    }

    return 0;

}    // end verifyImages()


static int run_kernel( cl_device_id device, cl_context context, cl_command_queue queue,
                      int w, int h, int d, int nChannels, cl_uchar *inptr, cl_uchar *outptr )
{
    clProgramWrapper program;
    clKernelWrapper kernel;
    clMemWrapper memobjs[2];
    cl_image_format image_format_desc = { CL_RGBA, CL_UNORM_INT8 };
    clEventWrapper executeEvent = NULL;
    cl_ulong queueStart, submitStart, writeStart, writeEnd;
    size_t threads[3];
    size_t localThreads[3];
    size_t maxWorkgroupSize;
    int err = 0;

    // set thread dimensions
    threads[0] = w;
    threads[1] = h;
    threads[2] = d;

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          3 * sizeof(size_t), (size_t *)localThreads, NULL);
    test_error(err, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES) failed");
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                          &maxWorkgroupSize, NULL);
    test_error(err, "clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE) failed");

    localThreads[0] =
        std::min({ localThreads[0], threads[0], maxWorkgroupSize });
    localThreads[1] = std::min(
        { localThreads[1], threads[1], maxWorkgroupSize / localThreads[0] });
    localThreads[2] =
        std::min({ localThreads[2], threads[2],
                   maxWorkgroupSize / (localThreads[0] * localThreads[1]) });

    clSamplerWrapper sampler = clCreateSampler(
        context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    // allocate the input and output image memory objects
    memobjs[0] =
        create_image_3d(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format_desc, w, h, d, 0, 0, inptr, &err);
    test_error(err, "unable to create 3D image using create_image_3d");

    // allocate an array memory object to load the filter weights
    size_t outptr_size = sizeof(cl_uchar) * w * h * d * nChannels;
    memobjs[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, outptr_size, NULL, &err);
    test_error(err, "unable to create array using clCreateBuffer");

    // create the compute program
    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &read3d_kernel_code, "read3d");
    test_error(err, "create_single_kernel_helper failed");

    // create kernel args object and set arg values.
    // set the args values
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjs[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjs[1]);
    err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
    test_error(err, "clSetKernelArg failed");

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, threads, localThreads,
                                 0, NULL, &executeEvent);
    test_error(err, "clEnqueueNDRangeKernel failed");

    if (executeEvent) {

        // This synchronization point is needed in order to assume the data is valid.
        // Getting profiling information is not a synchronization point.
        err = clWaitForEvents( 1, &executeEvent );
        test_error(err, "clWaitForEvents failed");

        // test profiling
        while ((err = clGetEventProfilingInfo(
                    executeEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong),
                    &queueStart, NULL))
               == CL_PROFILING_INFO_NOT_AVAILABLE)
            ;
        test_error(err, "clGetEventProfilingInfo failed");

        while ((err = clGetEventProfilingInfo(
                    executeEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong),
                    &submitStart, NULL))
               == CL_PROFILING_INFO_NOT_AVAILABLE)
            ;
        test_error(err, "clGetEventProfilingInfo failed");

        err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &writeStart, NULL );
        test_error(err, "clGetEventProfilingInfo failed");

        err = clGetEventProfilingInfo( executeEvent, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &writeEnd, NULL );
        test_error(err, "clGetEventProfilingInfo failed");

        log_info( "Profiling info:\n" );
        log_info( "Time from queue to start of clEnqueueNDRangeKernel: %f seconds\n", (double)(writeStart - queueStart) / 1000000000000.f );
        log_info( "Time from start of clEnqueueNDRangeKernel to end: %f seconds\n", (double)(writeEnd - writeStart) / 1000000000000.f );
    }

    // read output image
    err = clEnqueueReadBuffer(queue, memobjs[1], CL_TRUE, 0, outptr_size,
                              outptr, 0, NULL, NULL);
    test_error(err, "clReadImage failed");

    return err;
}    // end run_kernel()


// The main point of this test is to exercise code that causes a multipass cld launch for a single
// kernel exec at the cl level. This is done on the gpu for 3d launches, and it's also done
// to handle gdims that excede the maximums allowed by the hardware. In this case we
// use 3d to exercise the multipass events. In the future 3d may not be multpass, in which
// case we will need to ensure that we use gdims large enough to force multipass.

REGISTER_TEST(execute_multipass)
{
    cl_uchar *inptr;
    cl_uchar *outptr;
    int w = 256, h = 128, d = 32;
    int nChannels = 4;
    int nElements = w * h * d * nChannels;
    int err = 0;
    MTdata mtData;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    mtData = init_genrand( gRandomSeed );
    inptr = createImage( nElements, mtData );
    free_mtdata( mtData); mtData = NULL;
    if( ! inptr ){
        log_error( " unable to allocate %d bytes of memory for image\n", nElements );
        return -1;
    }

    outptr = (cl_uchar *)malloc( nElements * sizeof( cl_uchar ) );
    if( ! outptr ){
        log_error( " unable to allocate %d bytes of memory for output image #1\n", nElements );
        free( (void *)inptr );
        return -1;
    }


    err = run_kernel( device, context, queue, w, h, d, nChannels, inptr, outptr );

    if( ! err ){
        // verify that the images are the same
        err = verifyImages( outptr, inptr, (cl_uchar)0x1, w, h, d, nChannels );
        if( err )
            log_error( " images do not match\n" );
    }

    // clean up
    free( (void *)outptr );
    free( (void *)inptr );

    return err;

}    // end execute()



