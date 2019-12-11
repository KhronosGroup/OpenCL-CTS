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

static const char *image_to_image_kernel_integer_coord_code =
"\n"
"__kernel void image_to_image_copy(read_only image2d_t srcimg, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";

static const char *image_to_image_kernel_float_coord_code =
"\n"
"__kernel void image_to_image_copy(read_only image2d_t srcimg, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (float2)((float)tid_x, (float)tid_y));\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static const char *image_sum_kernel_integer_coord_code =
"\n"
"__kernel void image_sum(read_only image2d_t srcimg0, read_only image2d_t srcimg1, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color0;\n"
"    float4 color1;\n"
"\n"
"    color0 = read_imagef(srcimg0, sampler, (int2)(tid_x, tid_y));\n"
"    color1 = read_imagef(srcimg1, sampler, (int2)(tid_x, tid_y));\n"
"    write_imagef(dstimg, (int2)(tid_x, tid_y), color0 + color1);\n"
"\n"
"}\n";


static const char *image_sum_kernel_float_coord_code =
"\n"
"__kernel void image_sum(read_only image2d_t srcimg0, read_only image2d_t srcimg1, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color0;\n"
"    float4 color1;\n"
"\n"
"    color0 = read_imagef(srcimg0, sampler, (float2)((float)tid_x, (float)tid_y));\n"
"    color1 = read_imagef(srcimg1, sampler, (float2)((float)tid_x, (float)tid_y));\n"
"    write_imagef(dstimg,(int2)(tid_x, tid_y), color0 + color1);\n"
"\n"
"}\n";


static unsigned char *
generate_initial_byte_image(int w, int h, int num_elements, unsigned char value)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * num_elements);
    int             i;

    for (i = 0; i < w*h*num_elements; i++)
        ptr[i] = value;

    return ptr;
}

static unsigned char *
generate_expected_byte_image(unsigned char **input_data, int num_inputs, int w, int h, int num_elements)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * num_elements);
    int             i;

    for (i = 0; i < w*h*num_elements; i++)
    {
        int j;
        ptr[i] = 0;
        for (j = 0; j < num_inputs; j++)
        {
            unsigned char *input = *(input_data + j);
            ptr[i] += input[i];
        }
    }

    return ptr;
}


static unsigned char *
generate_byte_image(int w, int h, int num_elements, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * num_elements);
    int             i;

    for (i = 0; i < w*h*num_elements; i++)
        ptr[i] = (unsigned char)genrand_int32(d) & 31;

    return ptr;
}

static int
verify_byte_image(unsigned char *image, unsigned char *outptr, int w, int h, int num_elements)
{
    int     i;

    for (i = 0; i < w*h*num_elements; i++)
    {
        if (outptr[i] != image[i])
        {
            return -1;
        }
    }
    return 0;
}

int
test_image_multipass_integer_coord(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int                 img_width = 512;
    int                 img_height = 512;
    cl_image_format     img_format;

    int                 num_input_streams = 8;
    cl_mem              *input_streams;
    cl_mem                accum_streams[2];
    unsigned char       *expected_output;
    unsigned char       *output_ptr;
    cl_kernel           kernel[2];
    int                 err;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;

    expected_output = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height);
    output_ptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height);

    // Create the accum images with initial data.
    {
        unsigned char          *initial_data;
        cl_mem_flags        flags;

        initial_data = generate_initial_byte_image(img_width, img_height, 4, 0xF0);
        flags = (cl_mem_flags)(CL_MEM_READ_WRITE);

        accum_streams[0] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!accum_streams[0])
        {
            log_error("create_image_2d failed\n");
            free(expected_output);
            free(output_ptr);
            return -1;
        }

        size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
        err = clEnqueueWriteImage(queue, accum_streams[0], CL_TRUE,
                                  origin, region, 0, 0,
                                  initial_data, 0, NULL, NULL);
        if (err)
        {
            log_error("clWriteImage failed: %d\n", err);
            free(expected_output);
            free(output_ptr);
            return -1;
        }

        accum_streams[1] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!accum_streams[1])
        {
            log_error("create_image_2d failed\n");
            free(expected_output);
            free(output_ptr);
            return -1;
        }
        err = clEnqueueWriteImage(queue, accum_streams[1], CL_TRUE,
                                  origin, region, 0, 0,
                                  initial_data, 0, NULL, NULL);
        if (err)
        {
            log_error("clWriteImage failed: %d\n", err);
            free(expected_output);
            free(output_ptr);
            return -1;
        }

        free(initial_data);
    }

    // Set up the input data.
    {
        cl_mem_flags        flags;
        unsigned char       **input_data = (unsigned char **)malloc(sizeof(unsigned char*) * num_input_streams);
        MTdata              d;

        input_streams = (cl_mem*)malloc(sizeof(cl_mem) * num_input_streams);
        flags = (cl_mem_flags)(CL_MEM_READ_WRITE);

        int i;
        d = init_genrand( gRandomSeed );
        for ( i = 0; i < num_input_streams; i++)
        {
            input_data[i] = generate_byte_image(img_width, img_height, 4, d);
            input_streams[i] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
            if (!input_streams[i])
            {
                log_error("create_image_2d failed\n");
                free_mtdata(d);
                free(expected_output);
                free(output_ptr);
                return -1;
            }

            size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
            err = clEnqueueWriteImage(queue, input_streams[i], CL_TRUE,
                                      origin, region, 0, 0,
                                      input_data[i], 0, NULL, NULL);
            if (err)
            {
                log_error("clWriteImage failed: %d\n", err);
                free_mtdata(d);
                free(expected_output);
                free(output_ptr);
                free(input_streams);
                return -1;
            }


        }
        free_mtdata(d); d = NULL;
        expected_output = generate_expected_byte_image(input_data, num_input_streams, img_width, img_height, 4);
        for ( i = 0; i < num_input_streams; i++)
        {
            free(input_data[i]);
        }
        free( input_data );
    }

    // Set up the kernels.
    {
        cl_program          program[4];

        err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &image_to_image_kernel_integer_coord_code, "image_to_image_copy");
        if (err)
        {
            log_error("Failed to create kernel 0: %d\n", err);
            return -1;
        }
        err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &image_sum_kernel_integer_coord_code, "image_sum");
        if (err)
        {
            log_error("Failed to create kernel 1: %d\n", err);
            return -1;
        }
        clReleaseProgram(program[0]);
        clReleaseProgram(program[1]);
    }

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    {
        size_t        threads[3] = {0, 0, 0};
        threads[0] = (size_t)img_width;
        threads[1] = (size_t)img_height;
        int i;

        {
            cl_mem accum_input;
            cl_mem accum_output;

            err = clSetKernelArg(kernel[0], 0, sizeof input_streams[0], &input_streams[0]);
            err |= clSetKernelArg(kernel[0], 1, sizeof accum_streams[0], &accum_streams[0]);
            err |= clSetKernelArg(kernel[0], 2, sizeof sampler, &sampler);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }
            err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            for (i = 1; i < num_input_streams; i++)
            {
                accum_input = accum_streams[(i-1)%2];
                accum_output = accum_streams[i%2];

                err = clSetKernelArg(kernel[1], 0, sizeof accum_input, &accum_input);
                err |= clSetKernelArg(kernel[1], 1, sizeof input_streams[i], &input_streams[i]);
                err |= clSetKernelArg(kernel[1], 2, sizeof accum_output, &accum_output);
                err |= clSetKernelArg(kernel[1], 3, sizeof sampler, &sampler);

                if (err != CL_SUCCESS)
                {
                    log_error("clSetKernelArgs failed\n");
                    return -1;
                }
                err = clEnqueueNDRangeKernel( queue, kernel[1], 2, NULL, threads, NULL, 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    log_error("clEnqueueNDRangeKernel failed\n");
                    return -1;
                }
            }

            // Copy the last accum into the other one.
            accum_input = accum_streams[(i-1)%2];
            accum_output = accum_streams[i%2];
            err = clSetKernelArg(kernel[0], 0, sizeof accum_input, &accum_input);
            err |= clSetKernelArg(kernel[0], 1, sizeof accum_output, &accum_output);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }
            err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
            err = clEnqueueReadImage(queue, accum_output, CL_TRUE,
                                     origin, region, 0, 0,
                                     (void *)output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clReadImage failed\n");
                return -1;
            }
            err = verify_byte_image(expected_output, output_ptr, img_width, img_height, 4);
            if (err)
            {
                log_error("IMAGE_MULTIPASS test failed.\n");
            }
            else
            {
                log_info("IMAGE_MULTIPASS test passed\n");
            }
        }

        clReleaseSampler(sampler);
    }


    // cleanup
    clReleaseMemObject(accum_streams[0]);
    clReleaseMemObject(accum_streams[1]);
    {
        int i;
        for (i = 0; i < num_input_streams; i++)
        {
            clReleaseMemObject(input_streams[i]);
        }
    }
    free(input_streams);
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    free(expected_output);
    free(output_ptr);

    return err;
}

int
test_image_multipass_float_coord(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    int                 img_width = 512;
    int                 img_height = 512;
    cl_image_format     img_format;

    int                 num_input_streams = 8;
    cl_mem              *input_streams;
    cl_mem                accum_streams[2];
    unsigned char       *expected_output;
    unsigned char       *output_ptr;
    cl_kernel           kernel[2];
    int                 err;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    img_format.image_channel_order = CL_RGBA;
    img_format.image_channel_data_type = CL_UNORM_INT8;

    output_ptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * img_width * img_height);

    // Create the accum images with initial data.
    {
        unsigned char          *initial_data;
        cl_mem_flags        flags;

        initial_data = generate_initial_byte_image(img_width, img_height, 4, 0xF0);
        flags = (cl_mem_flags)(CL_MEM_READ_WRITE);

        accum_streams[0] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!accum_streams[0])
        {
            log_error("create_image_2d failed\n");
            return -1;
        }

        size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
        err = clEnqueueWriteImage(queue, accum_streams[0], CL_TRUE,
                                  origin, region, 0, 0,
                                  initial_data, 0, NULL, NULL);
        if (err)
        {
            log_error("clWriteImage failed: %d\n", err);
            return -1;
        }

        accum_streams[1] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
        if (!accum_streams[1])
        {
            log_error("create_image_2d failed\n");
            return -1;
        }
        err = clEnqueueWriteImage(queue, accum_streams[1], CL_TRUE,
                                  origin, region, 0, 0,
                                  initial_data, 0, NULL, NULL);
        if (err)
        {
            log_error("clWriteImage failed: %d\n", err);
            return -1;
        }

        free(initial_data);
    }

    // Set up the input data.
    {
        cl_mem_flags        flags;
        unsigned char       **input_data = (unsigned char **)malloc(sizeof(unsigned char*) * num_input_streams);
        MTdata              d;

        input_streams = (cl_mem*)malloc(sizeof(cl_mem) * num_input_streams);
        flags = (cl_mem_flags)(CL_MEM_READ_WRITE);

        int i;
        d = init_genrand( gRandomSeed );
        for ( i = 0; i < num_input_streams; i++)
        {
            input_data[i] = generate_byte_image(img_width, img_height, 4, d);
            input_streams[i] = create_image_2d(context, flags, &img_format, img_width, img_height, 0, NULL, NULL);
            if (!input_streams[i])
            {
                log_error("create_image_2d failed\n");
                free(input_data);
                free(input_streams);
                return -1;
            }

            size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
            err = clEnqueueWriteImage(queue, input_streams[i], CL_TRUE,
                                      origin, region, 0, 0,
                                      input_data[i], 0, NULL, NULL);
            if (err)
            {
                log_error("clWriteImage failed: %d\n", err);
                free(input_data);
                free(input_streams);
                return -1;
            }
        }
        free_mtdata(d); d = NULL;
        expected_output = generate_expected_byte_image(input_data, num_input_streams, img_width, img_height, 4);
        for ( i = 0; i < num_input_streams; i++)
        {
            free(input_data[i]);
        }
        free(input_data);
    }

    // Set up the kernels.
    {
        cl_program          program[2];

        err = create_single_kernel_helper(context, &program[0], &kernel[0], 1, &image_to_image_kernel_float_coord_code, "image_to_image_copy");
        if (err)
        {
            log_error("Failed to create kernel 2: %d\n", err);
            return -1;
        }
        err = create_single_kernel_helper(context, &program[1], &kernel[1], 1, &image_sum_kernel_float_coord_code, "image_sum");
        if (err)
        {
            log_error("Failed to create kernel 3: %d\n", err);
            return -1;
        }

        clReleaseProgram(program[0]);
        clReleaseProgram(program[1]);
    }

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    {
        size_t        threads[3] = {0, 0, 0};
        threads[0] = (size_t)img_width;
        threads[1] = (size_t)img_height;
        int i;

        {
            cl_mem accum_input;
            cl_mem accum_output;

            err = clSetKernelArg(kernel[0], 0, sizeof input_streams[0], &input_streams[0]);
            err |= clSetKernelArg(kernel[0], 1, sizeof accum_streams[0], &accum_streams[0]);
            err |= clSetKernelArg(kernel[0], 2, sizeof sampler, &sampler);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }
            err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            for (i = 1; i < num_input_streams; i++)
            {
                accum_input = accum_streams[(i-1)%2];
                accum_output = accum_streams[i%2];

                err = clSetKernelArg(kernel[1], 0, sizeof accum_input, &accum_input);
                err |= clSetKernelArg(kernel[1], 1, sizeof input_streams[i], &input_streams[i]);
                err |= clSetKernelArg(kernel[1], 2, sizeof accum_output, &accum_output);
                err |= clSetKernelArg(kernel[1], 3, sizeof sampler, &sampler);

                if (err != CL_SUCCESS)
                {
                    log_error("clSetKernelArgs failed\n");
                    return -1;
                }
                err = clEnqueueNDRangeKernel( queue, kernel[1], 2, NULL, threads, NULL, 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    log_error("clEnqueueNDRangeKernel failed\n");
                    return -1;
                }
            }

            // Copy the last accum into the other one.
            accum_input = accum_streams[(i-1)%2];
            accum_output = accum_streams[i%2];
            err = clSetKernelArg(kernel[0], 0, sizeof accum_input, &accum_input);
            err |= clSetKernelArg(kernel[0], 1, sizeof accum_output, &accum_output);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                return -1;
            }
            err = clEnqueueNDRangeKernel( queue, kernel[0], 2, NULL, threads, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                return -1;
            }

            size_t origin[3] = {0, 0, 0}, region[3] = {img_width, img_height, 1};
            err = clEnqueueReadImage(queue, accum_output, CL_TRUE,
                                     origin, region, 0, 0,
                                     (void *)output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clReadImage failed\n");
                return -1;
            }
            err = verify_byte_image(expected_output, output_ptr, img_width, img_height, 4);
            if (err)
            {
                log_error("IMAGE_MULTIPASS test failed.\n");
            }
            else
            {
                log_info("IMAGE_MULTIPASS test passed\n");
            }
        }

    }


    // cleanup
    clReleaseSampler(sampler);
    clReleaseMemObject(accum_streams[0]);
    clReleaseMemObject(accum_streams[1]);
    {
        int i;
        for (i = 0; i < num_input_streams; i++)
        {
            clReleaseMemObject(input_streams[i]);
        }
    }
    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    free(expected_output);
    free(output_ptr);
    free(input_streams);

    return err;
}





