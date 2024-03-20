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

#include <algorithm>
#include <vector>

#include "procs.h"

namespace {
const char *image_dim_kernel_code = R"(
__kernel void test_image_dim(read_only image2d_t srcimg, write_only image2d_t dstimg, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    float4 color;

    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y));
    write_imagef(dstimg, (int2)(tid_x, tid_y), color);
}
)";

void generate_random_inputs(std::vector<cl_uchar> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() { return genrand_int32(seed); };

    std::generate(v.begin(), v.end(), random_generator);
}

int get_max_image_dimensions(cl_device_id device, size_t &max_img_width,
                             size_t &max_img_height)
{
    int err = 0;

    cl_ulong max_mem_size;
    size_t max_image2d_width, max_image2d_height;

    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(max_mem_size), &max_mem_size, nullptr);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_GLOBAL_MEM_SIZE failed");
    err =
        clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                        sizeof(max_image2d_width), &max_image2d_width, nullptr);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_WIDTH failed");
    err = clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                          sizeof(max_image2d_width), &max_image2d_height,
                          nullptr);
    test_error(err, "clGetDeviceInfo for CL_DEVICE_IMAGE2D_MAX_HEIGHT failed");

    log_info("Device reported max image sizes of %lu x %lu, and max mem size "
             "of %gMB.\n",
             max_image2d_width, max_image2d_height,
             max_mem_size / (1024.0 * 1024.0));


    max_mem_size = std::min(max_mem_size, (cl_ulong)SIZE_MAX);

    // determine max image dim we can allocate - assume RGBA image, 4 bytes per
    // pixel, and we want to consume 1/4 of global memory (this is the minimum
    // required to be supported by the spec)
    max_mem_size /= 4; // use 1/4
    max_mem_size /= 4; // 4 bytes per pixel

    size_t max_img_dim =
        static_cast<size_t>(sqrt(static_cast<double>(max_mem_size)));
    // convert to a power of 2
    {
        unsigned int n = static_cast<unsigned int>(max_img_dim);
        unsigned int m = 0x80000000;

        // round-down to the nearest power of 2
        while (m > n) m >>= 1;

        max_img_dim = m;
    }

    max_img_width = std::min(max_image2d_width, max_img_dim);
    max_img_height = std::min(max_image2d_height, max_img_dim);

    log_info("Adjusted maximum image size to test is %d x %d, which is a max "
             "mem size of %gMB.\n",
             max_img_width, max_img_height,
             (max_img_width * max_img_height * 4) / (1024.0 * 1024.0));
    return err;
}

int test_imagedim_common(cl_context context, cl_command_queue queue,
                         cl_kernel kernel, size_t *local_threads,
                         size_t img_width, size_t img_height)
{

    int err;
    int total_errors = 0;

    clMemWrapper streams[2];

    std::vector<cl_uchar> input(4 * img_width * img_height);
    std::vector<cl_uchar> output(4 * img_width * img_height);

    generate_random_inputs(input);

    const cl_image_format img_format = { CL_RGBA, CL_UNORM_INT8 };

    streams[0] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format,
                                 img_width, img_height, 0, nullptr, &err);
    test_error(err, "create_image_2d failed");

    streams[1] = create_image_2d(context, CL_MEM_READ_WRITE, &img_format,
                                 img_width, img_height, 0, nullptr, &err);
    test_error(err, "create_image_2d failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { img_width, img_height, 1 };
    err = clEnqueueWriteImage(queue, streams[0], CL_FALSE, origin, region, 0, 0,
                              input.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteImage failed");

    clSamplerWrapper sampler = clCreateSampler(
        context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
    test_error(err, "clSetKernelArg failed");

    size_t threads[] = { img_width, img_height };
    if (local_threads)
        log_info(
            "Testing image dimensions %d x %d with local threads %d x %d.\n",
            img_width, img_height, local_threads[0], local_threads[1]);
    else
        log_info(
            "Testing image dimensions %d x %d with local threads nullptr.\n",
            img_width, img_height);
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, threads,
                                 local_threads, 0, nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed");

    err = clEnqueueReadImage(queue, streams[1], CL_TRUE, origin, region, 0, 0,
                             output.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadImage failed");

    if (0 != memcmp(input.data(), output.data(), 4 * img_width * img_height))
    {
        total_errors++;
        log_error("Image Dimension test failed.  image width = %d, "
                  "image height = %d\n",
                  img_width, img_height);
    }
    return total_errors;
}
}

int test_imagedim_pow2(cl_device_id device, cl_context context,
                       cl_command_queue queue, int n_elems)
{
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t max_img_width;
    size_t max_img_height;

    int err = 0;
    int total_errors = 0;

    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &image_dim_kernel_code, "test_image_dim");
    test_error(err, "create_single_kernel_helper failed");

    err = get_max_image_dimensions(device, max_img_width, max_img_height);
    test_error(err, "get_max_image_dimensions failed");

    // test power of 2 width, height starting at 1 to 4K
    for (size_t i = 1, i2 = 0; i <= max_img_height; i <<= 1, i2++)
    {
        size_t img_height = (1 << i2);
        for (size_t j = 1, j2 = 0; j <= max_img_width; j <<= 1, j2++)
        {
            size_t img_width = (1 << j2);

            total_errors += test_imagedim_common(
                context, queue, kernel, nullptr, img_width, img_height);
        }
    }

    return total_errors;
}


int test_imagedim_non_pow2(cl_device_id device, cl_context context,
                           cl_command_queue queue, int n_elems)
{
    clProgramWrapper program;
    clKernelWrapper kernel;

    size_t max_img_width;
    size_t max_img_height;
    size_t max_local_workgroup_size[3] = {};
    size_t work_group_size = 0;
    int err = 0;
    int total_errors = 0;


    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &image_dim_kernel_code, "test_image_dim");
    test_error(err, "create_single_kernel_helper failed");

    err = get_max_image_dimensions(device, max_img_width, max_img_height);
    test_error(err, "get_max_image_dimensions failed");

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(work_group_size), &work_group_size,
                                   nullptr);
    test_error(err,
               "clGetKernelWorkgroupInfo failed for CL_KERNEL_WORK_GROUP_SIZE");

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                          sizeof(max_local_workgroup_size),
                          max_local_workgroup_size, nullptr);
    test_error(err, "clGetDeviceInfo failed for CL_DEVICE_MAX_WORK_ITEM_SIZES");

    // clamp max_local_workgroup_size to CL_KERNEL_WORK_GROUP_SIZE
    for (auto &max_lws : max_local_workgroup_size)
        max_lws = std::min(max_lws, work_group_size);

    for (int plus_minus = 0; plus_minus < 3; plus_minus++)
    {

        // test power of 2 width, height starting at 1 to 4K
        for (size_t i = 2, i2 = 1; i <= max_img_height; i <<= 1, i2++)
        {
            size_t img_height = (1 << i2);
            for (size_t j = 2, j2 = 1; j <= max_img_width; j <<= 1, j2++)
            {
                size_t img_width = (1 << j2);

                size_t effective_img_height = img_height;
                size_t effective_img_width = img_width;

                size_t local_threads[] = { 1, 1 };

                switch (plus_minus)
                {
                    case 0:
                        effective_img_height--;
                        local_threads[0] = max_local_workgroup_size[0];
                        while (img_width % local_threads[0] != 0)
                            local_threads[0]--;
                        break;
                    case 1:
                        effective_img_width--;
                        local_threads[1] = max_local_workgroup_size[1];
                        while (img_height % local_threads[1] != 0)
                            local_threads[1]--;
                        break;
                    case 2:
                        effective_img_width--;
                        effective_img_height--;
                        break;
                    default: break;
                }

                total_errors += test_imagedim_common(
                    context, queue, kernel, local_threads, effective_img_width,
                    effective_img_height);
            }
        }
    }

    return total_errors;
}
