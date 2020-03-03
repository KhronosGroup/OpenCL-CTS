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

static const char *image_dim_kernel_code =
"\n"
"__kernel void test_image_dim(read_only image2d_t srcimg, write_only image2d_t dstimg, sampler_t sampler)\n"
"{\n"
"    int    tid_x = get_global_id(0);\n"
"    int    tid_y = get_global_id(1);\n"
"    float4 color;\n"
"\n"
"    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));\n"
"     write_imagef(dstimg, (int2)(tid_x, tid_y), color);\n"
"\n"
"}\n";


static unsigned char *
generate_8888_image(int w, int h, MTdata d)
{
    unsigned char   *ptr = (unsigned char*)malloc(w * h * 4);
    int             i;

    for (i=0; i<w*h*4; i++)
        ptr[i] = (unsigned char)genrand_int32(d);

    return ptr;
}

static int
verify_8888_image(unsigned char *image, unsigned char *outptr, int w, int h)
{
    int     i;

    for (i=0; i<w*h; i++)
    {
        if (outptr[i] != image[i])
            return -1;
    }

    return 0;
}


int
test_imagedim_pow2(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem            streams[2];
    cl_image_format    img_format;
    unsigned char    *input_ptr, *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    threads[2];
     cl_ulong    max_mem_size;
    int                img_width, max_img_width;
    int                img_height, max_img_height;
    int                max_img_dim;
    int                i, j, i2, j2, err=0;
    size_t            max_image2d_width, max_image2d_height;
    int total_errors = 0;
    MTdata  d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    err = create_single_kernel_helper( context, &program, &kernel, 1, &image_dim_kernel_code, "test_image_dim" );
    if (err)
    {
        log_error("create_program_and_kernel_with_sources failed\n");
        return -1;
    }

    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(max_mem_size), &max_mem_size, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_GLOBAL_MEM_SIZE failed (%d)\n", err);
        return -1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(max_image2d_width), &max_image2d_width, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_WIDTH failed (%d)\n", err);
        return -1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_image2d_width), &max_image2d_height, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_HEIGHT failed (%d)\n", err);
        return -1;
    }
    log_info("Device reported max image sizes of %lu x %lu, and max mem size of %gMB.\n",
           max_image2d_width, max_image2d_height, max_mem_size/(1024.0*1024.0));

    if (max_mem_size > (cl_ulong)SIZE_MAX) {
        max_mem_size = (cl_ulong)SIZE_MAX;
    }

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    max_img_width = (int)max_image2d_width;
    max_img_height = (int)max_image2d_height;

    // determine max image dim we can allocate - assume RGBA image, 4 bytes per pixel,
  //  and we want to consume 1/4 of global memory (this is the minimum required to be
  //  supported by the spec)
    max_mem_size /= 4; // use 1/4
    max_mem_size /= 4; // 4 bytes per pixel
    max_img_dim = (int)sqrt((double)max_mem_size);
    // convert to a power of 2
    {
        unsigned int    n = (unsigned int)max_img_dim;
        unsigned int    m = 0x80000000;

        // round-down to the nearest power of 2
        while (m > n)
            m >>= 1;

        max_img_dim = (int)m;
    }

    if (max_img_width > max_img_dim)
        max_img_width = max_img_dim;
    if (max_img_height > max_img_dim)
        max_img_height = max_img_dim;

    log_info("Adjusted maximum image size to test is %d x %d, which is a max mem size of %gMB.\n",
                max_img_width, max_img_height, (max_img_width*max_img_height*4)/(1024.0*1024.0));

    d = init_genrand( gRandomSeed );
    input_ptr = generate_8888_image(max_img_width, max_img_height, d);
    output_ptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * max_img_width * max_img_height);

    // test power of 2 width, height starting at 1 to 4K
    for (i=1,i2=0; i<=max_img_height; i<<=1,i2++)
    {
        img_height = (1 << i2);
        for (j=1,j2=0; j<=max_img_width; j<<=1,j2++)
        {
            img_width = (1 << j2);

            img_format.image_channel_order = CL_RGBA;
            img_format.image_channel_data_type = CL_UNORM_INT8;
            streams[0] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, NULL);
            if (!streams[0])
            {
                log_error("create_image_2d failed.  width = %d, height = %d\n", img_width, img_height);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }
            img_format.image_channel_order = CL_RGBA;
            img_format.image_channel_data_type = CL_UNORM_INT8;
            streams[1] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, img_width, img_height, 0, NULL, NULL);
            if (!streams[1])
            {
                log_error("create_image_2d failed.  width = %d, height = %d\n", img_width, img_height);
                clReleaseMemObject(streams[0]);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }

            size_t origin[3] = {0,0,0};
            size_t region[3] = {img_width, img_height, 1};
            err = clEnqueueWriteImage(queue, streams[0], CL_FALSE, origin, region, 0, 0, input_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clWriteImage failed\n");
                clReleaseMemObject(streams[0]);
                clReleaseMemObject(streams[1]);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }

            err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
            err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
            err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
            if (err != CL_SUCCESS)
            {
                log_error("clSetKernelArgs failed\n");
                clReleaseMemObject(streams[0]);
                clReleaseMemObject(streams[1]);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }

            threads[0] = (size_t)img_width;
            threads[1] = (size_t)img_height;
            log_info("Testing image dimensions %d x %d with local threads NULL.\n", img_width, img_height);
            err = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, NULL, 0, NULL, NULL );
            if (err != CL_SUCCESS)
            {
                log_error("clEnqueueNDRangeKernel failed\n");
                log_error("Image Dimension test failed.  image width = %d, image height = %d, local NULL\n",
                            img_width, img_height);
                clReleaseMemObject(streams[0]);
                clReleaseMemObject(streams[1]);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }
            err = clEnqueueReadImage(queue, streams[1], CL_TRUE, origin, region, 0, 0, output_ptr, 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                log_error("clReadImage failed\n");
                log_error("Image Dimension test failed.  image width = %d, image height = %d, local NULL\n",
                            img_width, img_height);
                clReleaseMemObject(streams[0]);
                clReleaseMemObject(streams[1]);
                free(input_ptr);
                free(output_ptr);
                free_mtdata(d);
                return -1;
            }
            err = verify_8888_image(input_ptr, output_ptr, img_width, img_height);
            if (err)
            {
                total_errors++;
                log_error("Image Dimension test failed.  image width = %d, image height = %d\n", img_width, img_height);
            }

            clReleaseMemObject(streams[0]);
            clReleaseMemObject(streams[1]);
        }
    }

    // cleanup
    free(input_ptr);
    free(output_ptr);
    free_mtdata(d);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return total_errors;
}



int
test_imagedim_non_pow2(cl_device_id device, cl_context context, cl_command_queue queue, int n_elems)
{
    cl_mem            streams[2];
    cl_image_format    img_format;
    unsigned char    *input_ptr, *output_ptr;
    cl_program        program;
    cl_kernel        kernel;
    size_t    threads[2], local_threads[2];
    cl_ulong    max_mem_size;
    int                img_width, max_img_width;
    int                img_height, max_img_height;
    int                max_img_dim;
    int                i, j, i2, j2, err=0;
    size_t            max_image2d_width, max_image2d_height;
    int total_errors = 0;
    size_t max_local_workgroup_size[3];
    MTdata d;

    PASSIVE_REQUIRE_IMAGE_SUPPORT( device )

    err = create_single_kernel_helper( context, &program, &kernel, 1, &image_dim_kernel_code, "test_image_dim" );
    if (err)
    {
        log_error("create_program_and_kernel_with_sources failed\n");
        return -1;
    }

    size_t work_group_size = 0;
    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
    test_error(err, "clGetKerenlWorkgroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_local_workgroup_size), max_local_workgroup_size, NULL);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(max_mem_size), &max_mem_size, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_GLOBAL_MEM_SIZE failed (%d)\n", err);
        return -1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(max_image2d_width), &max_image2d_width, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_WIDTH failed (%d)\n", err);
        return -1;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(max_image2d_width), &max_image2d_height, NULL);
    if (err)
    {
        log_error("clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_HEIGHT failed (%d)\n", err);
        return -1;
    }
    log_info("Device reported max image sizes of %lu x %lu, and max mem size of %gMB.\n",
           max_image2d_width, max_image2d_height, max_mem_size/(1024.0*1024.0));

    cl_sampler sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    max_img_width = (int)max_image2d_width;
    max_img_height = (int)max_image2d_height;

  if (max_mem_size > (cl_ulong)SIZE_MAX) {
    max_mem_size = (cl_ulong)SIZE_MAX;
  }

    // determine max image dim we can allocate - assume RGBA image, 4 bytes per pixel,
    //  and we want to consume 1/4 of global memory (this is the minimum required to be
    //  supported by the spec)
    max_mem_size /= 4; // use 1/4
    max_mem_size /= 4; // 4 bytes per pixel
    max_img_dim = (int)sqrt((double)max_mem_size);
    // convert to a power of 2
    {
        unsigned int    n = (unsigned int)max_img_dim;
        unsigned int    m = 0x80000000;

        // round-down to the nearest power of 2
        while (m > n)
            m >>= 1;

        max_img_dim = (int)m;
    }

    if (max_img_width > max_img_dim)
        max_img_width = max_img_dim;
    if (max_img_height > max_img_dim)
        max_img_height = max_img_dim;

    log_info("Adjusted maximum image size to test is %d x %d, which is a max mem size of %gMB.\n",
            max_img_width, max_img_height, (max_img_width*max_img_height*4)/(1024.0*1024.0));

    d = init_genrand( gRandomSeed );
    input_ptr = generate_8888_image(max_img_width, max_img_height, d);
    output_ptr = (unsigned char*)malloc(sizeof(unsigned char) * 4 * max_img_width * max_img_height);

    int plus_minus;
    for (plus_minus=0; plus_minus < 3; plus_minus++)
    {

    // test power of 2 width, height starting at 1 to 4K
        for (i=2,i2=1; i<=max_img_height; i<<=1,i2++)
        {
            img_height = (1 << i2);
            for (j=2,j2=1; j<=max_img_width; j<<=1,j2++)
            {
                img_width = (1 << j2);

                int effective_img_height = img_height;
                int effective_img_width = img_width;

                local_threads[0] = 1;
                local_threads[1] = 1;

                switch (plus_minus) {
                    case 0:
                      effective_img_height--;
                      local_threads[0] = work_group_size > max_local_workgroup_size[0] ? max_local_workgroup_size[0] : work_group_size;
                      while (img_width%local_threads[0] != 0)
                        local_threads[0]--;
                      break;
                    case 1:
                      effective_img_width--;
                      local_threads[1] = work_group_size > max_local_workgroup_size[1] ? max_local_workgroup_size[1] : work_group_size;
                      while (img_height%local_threads[1] != 0)
                        local_threads[1]--;
                      break;
                    case 2:
                      effective_img_width--;
                      effective_img_height--;
                      break;
                    default:
                      break;
                }

                img_format.image_channel_order = CL_RGBA;
                img_format.image_channel_data_type = CL_UNORM_INT8;
                streams[0] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, effective_img_width, effective_img_height, 0, NULL, NULL);
                if (!streams[0])
                {
                    log_error("create_image_2d failed.  width = %d, height = %d\n", effective_img_width, effective_img_height);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }
                img_format.image_channel_order = CL_RGBA;
                img_format.image_channel_data_type = CL_UNORM_INT8;
                streams[1] = create_image_2d(context, (cl_mem_flags)(CL_MEM_READ_WRITE),  &img_format, effective_img_width, effective_img_height, 0, NULL, NULL);
                if (!streams[1])
                {
                    log_error("create_image_2d failed.  width = %d, height = %d\n", effective_img_width, effective_img_height);
                    clReleaseMemObject(streams[0]);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }

                  size_t origin[3] = {0,0,0};
                  size_t region[3] = {effective_img_width, effective_img_height, 1};
                  err = clEnqueueWriteImage(queue, streams[0], CL_FALSE, origin, region, 0, 0, input_ptr, 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    log_error("clWriteImage failed\n");
                    clReleaseMemObject(streams[0]);
                    clReleaseMemObject(streams[1]);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }

                err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
                err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
                err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
                if (err != CL_SUCCESS)
                {
                    log_error("clSetKernelArgs failed\n");
                    clReleaseMemObject(streams[0]);
                    clReleaseMemObject(streams[1]);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }

                threads[0] = (size_t)effective_img_width;
                threads[1] = (size_t)effective_img_height;
                log_info("Testing image dimensions %d x %d with local threads %d x %d.\n",
                            effective_img_width, effective_img_height, (int)local_threads[0], (int)local_threads[1]);
                err = clEnqueueNDRangeKernel( queue, kernel, 2, NULL, threads, local_threads, 0, NULL, NULL );
                if (err != CL_SUCCESS)
                {
                    log_error("clEnqueueNDRangeKernel failed\n");
                    log_error("Image Dimension test failed.  image width = %d, image height = %d, local %d x %d\n",
                                effective_img_width, effective_img_height, (int)local_threads[0], (int)local_threads[1]);
                    clReleaseMemObject(streams[0]);
                    clReleaseMemObject(streams[1]);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }
                err = clEnqueueReadImage(queue, streams[1], CL_TRUE, origin, region, 0, 0, output_ptr, 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    log_error("clReadImage failed\n");
                    log_error("Image Dimension test failed.  image width = %d, image height = %d, local %d x %d\n",
                                effective_img_width, effective_img_height, (int)local_threads[0], (int)local_threads[1]);
                    clReleaseMemObject(streams[0]);
                    clReleaseMemObject(streams[1]);
                    free(input_ptr);
                    free(output_ptr);
                    free_mtdata(d);
                    return -1;
                }
                err = verify_8888_image(input_ptr, output_ptr, effective_img_width, effective_img_height);
                if (err)
                {
                    total_errors++;
                    log_error("Image Dimension test failed.  image width = %d, image height = %d\n", effective_img_width, effective_img_height);
                }

                clReleaseMemObject(streams[0]);
                clReleaseMemObject(streams[1]);
            }
        }

  }

    // cleanup
    free(input_ptr);
    free(output_ptr);
    free_mtdata(d);
    clReleaseSampler(sampler);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return total_errors;
}




