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

const char *hostptr_kernel_code =
"__kernel void test_hostptr(__global float *srcA, __global float *srcB, __global float *dst)\n"
"{\n"
"    int  tid = get_global_id(0);\n"
"\n"
"    dst[tid] = srcA[tid] + srcB[tid];\n"
"}\n";

static int verify_hostptr(cl_float *inptrA, cl_float *inptrB, cl_float *outptr, int n)
{
    cl_float       r;
    int         i;

    for (i=0; i<n; i++)
    {
        r = inptrA[i] + inptrB[i];
        if (r != outptr[i])
        {
            return -1;
        }
    }
    return 0;
}

static void make_random_data(unsigned count, float *ptr, MTdata d)
{
    cl_uint     i;
    for (i=0; i<count; i++)
        ptr[i] = get_random_float(-MAKE_HEX_FLOAT( 0x1.0p32f, 0x1, 32), MAKE_HEX_FLOAT( 0x1.0p32f, 0x1, 32), d);
}

static unsigned char *
generate_rgba8_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static unsigned char *
randomize_rgba8_image(unsigned char *ptr, int w, int h, MTdata d)
{
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_rgba8_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h*4; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}

int
test_hostptr(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    cl_float            *input_ptr[2], *output_ptr;
    cl_program            program;
    cl_kernel           kernel;
    size_t              threads[3]={0,0,0};
    cl_image_format     img_format;
    cl_uchar            *rgba8_inptr, *rgba8_outptr;
    void                *lock_buffer;
    int                 img_width = 512;
    int                 img_height = 512;
    cl_int              err;
    MTdata              d;
    RoundingMode        oldRoundMode;
    int                    isRTZ = 0;

    // Block to mark deletion of streams before deletion of host_ptr
    {
        clMemWrapper        streams[7];

        PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

        // Alloc buffers
        input_ptr[0] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
        input_ptr[1] = (cl_float*)malloc(sizeof(cl_float) * num_elements);
        output_ptr = (cl_float*)malloc(sizeof(cl_float) * num_elements);

        d = init_genrand( gRandomSeed );
        rgba8_inptr = (cl_uchar *)generate_rgba8_image(img_width, img_height, d);
        rgba8_outptr = (cl_uchar *)malloc(sizeof(cl_uchar) * 4 * img_width * img_height);

        // Random data
        make_random_data(num_elements, input_ptr[0], d);
        make_random_data(num_elements, input_ptr[1], d);

        // Create host-side input
        streams[0] =
            clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                           sizeof(cl_float) * num_elements, input_ptr[0], &err);
        test_error(err, "clCreateBuffer 0 failed");

        // Create a copied input
        streams[1] =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_float) * num_elements, input_ptr[1], &err);
        test_error(err, "clCreateBuffer 1 failed");

        // Create a host-side output
        streams[2] =
            clCreateBuffer(context, CL_MEM_USE_HOST_PTR,
                           sizeof(cl_float) * num_elements, output_ptr, &err);
        test_error(err, "clCreateBuffer 2 failed");

        // Create a host-side input
        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[3] =
            create_image_2d(context, CL_MEM_USE_HOST_PTR, &img_format,
                            img_width, img_height, 0, rgba8_inptr, &err);
        test_error(err, "create_image_2d 3 failed");

        // Create a copied input
        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[4] =
            create_image_2d(context, CL_MEM_COPY_HOST_PTR, &img_format,
                            img_width, img_height, 0, rgba8_inptr, &err);
        test_error(err, "create_image_2d 4 failed");

        // Create a host-side output
        img_format.image_channel_order = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[5] =
            create_image_2d(context, CL_MEM_USE_HOST_PTR, &img_format,
                            img_width, img_height, 0, rgba8_outptr, &err);
        test_error(err, "create_image_2d 5 failed");

        // Create a copied output
        img_format.image_channel_data_type = CL_RGBA;
        img_format.image_channel_data_type = CL_UNORM_INT8;
        streams[6] =
            create_image_2d(context, CL_MEM_COPY_HOST_PTR, &img_format,
                            img_width, img_height, 0, rgba8_outptr, &err);
        test_error(err, "create_image_2d 6 failed");

        err = create_single_kernel_helper(context, &program, &kernel,1, &hostptr_kernel_code, "test_hostptr" );
        test_error(err, "create_single_kernel_helper failed");

        // Execute kernel
        err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
        err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
        err |= clSetKernelArg(kernel, 2, sizeof streams[2], &streams[2]);
        test_error(err, "clSetKernelArg failed");

        threads[0] = (size_t)num_elements;
        err = clEnqueueNDRangeKernel( queue, kernel, 1, NULL, threads, NULL, 0, NULL, NULL );
        test_error(err, "clEnqueueNDRangeKernel failed");

        cl_float *data = (cl_float*) clEnqueueMapBuffer( queue, streams[2], CL_TRUE, CL_MAP_READ, 0, sizeof(cl_float) * num_elements, 0, NULL, NULL, &err );
        test_error( err, "clEnqueueMapBuffer failed" );

        //If we only support rtz mode
        if( CL_FP_ROUND_TO_ZERO == get_default_rounding_mode(device) && gIsEmbedded)
        {
            oldRoundMode = set_round(kRoundTowardZero, kfloat);
            isRTZ = 1;
        }

        if (isRTZ)
            oldRoundMode = set_round(kRoundTowardZero, kfloat);

        // Verify that we got the expected results back on the host side
        err = verify_hostptr(input_ptr[0], input_ptr[1], data, num_elements);
        if (err)
        {
            log_error("Checking mapped data for kernel executed with CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR inputs "
                      "and a CL_MEM_USE_HOST_PTR output did not return the expected results.\n");
        } else {
            log_info("Checking mapped data for kernel executed with CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR inputs "
                     "and a CL_MEM_USE_HOST_PTR output returned the expected results.\n");
        }

        if (isRTZ)
            set_round(oldRoundMode, kfloat);

        err = clEnqueueUnmapMemObject( queue, streams[2], data, 0, NULL, NULL );
        test_error( err, "clEnqueueUnmapMemObject failed" );

        size_t origin[3]={0,0,0}, region[3]={img_width, img_height, 1};
        randomize_rgba8_image(rgba8_outptr, img_width, img_height, d);
        free_mtdata(d); d = NULL;

        // Copy from host-side to host-side
        log_info("clEnqueueCopyImage from CL_MEM_USE_HOST_PTR to CL_MEM_USE_HOST_PTR...\n");
        err = clEnqueueCopyImage(queue, streams[3], streams[5],
                                 origin, origin, region,  0, NULL, NULL);
        test_error(err, "clEnqueueCopyImage failed");
        log_info("clEnqueueCopyImage from CL_MEM_USE_HOST_PTR to CL_MEM_USE_HOST_PTR image passed.\n");

        // test the lock buffer interface
        log_info("Mapping the CL_MEM_USE_HOST_PTR image with clEnqueueMapImage...\n");
        size_t row_pitch;
        lock_buffer = clEnqueueMapImage(queue, streams[5], CL_TRUE,
                                        CL_MAP_READ, origin, region,
                                        &row_pitch, NULL,
                                        0, NULL, NULL, &err);
        test_error(err, "clEnqueueMapImage failed");

        err = verify_rgba8_image(rgba8_inptr, (unsigned char*)lock_buffer, img_width, img_height);
        if (err != CL_SUCCESS)
        {
            log_error("verify_rgba8_image FAILED after clEnqueueMapImage\n");
            return -1;
        }
        log_info("verify_rgba8_image passed after clEnqueueMapImage\n");

        err = clEnqueueUnmapMemObject(queue, streams[5], lock_buffer, 0, NULL, NULL);
        test_error(err, "clEnqueueUnmapMemObject failed");

        // Copy host-side to device-side and read back
        log_info("clEnqueueCopyImage CL_MEM_USE_HOST_PTR to CL_MEM_COPY_HOST_PTR...\n");
        err = clEnqueueCopyImage(queue, streams[3], streams[5],
                                 origin, origin, region,
                                 0, NULL, NULL);
        test_error(err, "clEnqueueCopyImage failed");

        err = clEnqueueReadImage(queue, streams[5], CL_TRUE, origin, region, 4*img_width, 0, rgba8_outptr, 0, NULL, NULL);
        test_error(err, "clEnqueueReadImage failed");

        err = verify_rgba8_image(rgba8_inptr, rgba8_outptr, img_width, img_height);
        if (err != CL_SUCCESS)
        {
            log_error("verify_rgba8_image FAILED after clEnqueueCopyImage, clEnqueueReadImage\n");
            return -1;
        }
        log_info("verify_rgba8_image passed after clEnqueueCopyImage, clEnqueueReadImage\n");
    }
    // cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    free(input_ptr[0]);
    free(input_ptr[1]);
    free(output_ptr);

    free(rgba8_inptr);
    free(rgba8_outptr);

    return err;
}





