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
#include "harness/imageHelpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>


#include "procs.h"

#include <algorithm>
#include <string>
#include <vector>

#include "procs.h"

namespace {
const char *kernel_source = R"(
__kernel void test_CL_BGRACL_UNORM_INT8(__global unsigned char *src, write_only image2d_t dstimg)
{
    int            tid_x = get_global_id(0);
    int            tid_y = get_global_id(1);
    int            indx = tid_y * get_image_width(dstimg) + tid_x;
    float4         color;

    indx *= 4;
    color = (float4)((float)src[indx+2], (float)src[indx+1], (float)src[indx+0], (float)src[indx+3]);
    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);
    write_imagef(dstimg, (int2)(tid_x, tid_y), color);
}

__kernel void test_CL_RGBACL_UNORM_INT8(__global unsigned char *src, write_only image2d_t dstimg)
{
    int            tid_x = get_global_id(0);
    int            tid_y = get_global_id(1);
    int            indx = tid_y * get_image_width(dstimg) + tid_x;
    float4         color;

    indx *= 4;
    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);
    color /= (float4)(255.0f, 255.0f, 255.0f, 255.0f);
    write_imagef(dstimg, (int2)(tid_x, tid_y), color);
}

__kernel void test_CL_RGBACL_UNORM_INT16(__global unsigned short *src, write_only image2d_t dstimg)
{
    int            tid_x = get_global_id(0);
    int            tid_y = get_global_id(1);
    int            indx = tid_y * get_image_width(dstimg) + tid_x;
    float4         color;

    indx *= 4;
    color = (float4)((float)src[indx+0], (float)src[indx+1], (float)src[indx+2], (float)src[indx+3]);
    color /= 65535.0f;
    write_imagef(dstimg, (int2)(tid_x, tid_y), color);
}

__kernel void test_CL_RGBACL_FLOAT(__global float *src, write_only image2d_t dstimg)
{
    int            tid_x = get_global_id(0);
    int            tid_y = get_global_id(1);
    int            indx = tid_y * get_image_width(dstimg) + tid_x;
    float4         color;

    indx *= 4;
    color = (float4)(src[indx+0], src[indx+1], src[indx+2], src[indx+3]);
    write_imagef(dstimg, (int2)(tid_x, tid_y), color);
}
)";


template <typename T> void generate_random_inputs(std::vector<T> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return static_cast<T>(genrand_int32(seed));
    };

    std::generate(v.begin(), v.end(), random_generator);
}

template <> void generate_random_inputs<float>(std::vector<float> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return get_random_float(-0x40000000, 0x40000000, seed);
    };

    std::generate(v.begin(), v.end(), random_generator);
}


const char *get_mem_flag_name(cl_mem_flags flags)
{
    switch (flags)
    {
        case CL_MEM_READ_WRITE: return "CL_MEM_READ_WRITE";
        case CL_MEM_WRITE_ONLY: return "CL_MEM_WRITE_ONLY";
        default: return "Unsupported cl_mem_flags value";
    }
}

template <typename T>
int test_writeimage(cl_device_id device, cl_context context,
                    cl_command_queue queue, const cl_image_format *img_format,
                    cl_mem_flags img_flags)
{
    clMemWrapper streams[2];
    clProgramWrapper program;
    clKernelWrapper kernel;

    std::string kernel_name("test_");

    size_t img_width = 512;
    size_t img_height = 512;

    int err;

    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { img_width, img_height, 1 };

    const size_t num_elements = img_width * img_height * 4;
    const size_t length = num_elements * sizeof(T);

    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    std::vector<T> input(num_elements);
    std::vector<T> output(num_elements);

    generate_random_inputs(input);

    streams[0] = create_image_2d(context, img_flags, img_format, img_width,
                                 img_height, 0, nullptr, &err);
    test_error(err, "create_image failed.");

    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, length, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");

    err = clEnqueueWriteBuffer(queue, streams[1], CL_TRUE, 0, length,
                               input.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteImage failed.");

    kernel_name += GetChannelOrderName(img_format->image_channel_order);
    kernel_name += GetChannelTypeName(img_format->image_channel_data_type);

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &kernel_source, kernel_name.c_str());
    test_error(err, "create_single_kernel_helper failed.");

    err |= clSetKernelArg(kernel, 0, sizeof(streams[1]), &streams[1]);
    err |= clSetKernelArg(kernel, 1, sizeof(streams[0]), &streams[0]);
    test_error(err, "clSetKernelArgs failed\n");

    size_t threads[] = { img_width, img_height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, threads, nullptr, 0,
                                 nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed\n");

    err = clEnqueueReadImage(queue, streams[0], CL_TRUE, origin, region, 0, 0,
                             output.data(), 0, nullptr, nullptr);

    if (0 != memcmp(input.data(), output.data(), length))
    {
        log_error("WRITE_IMAGE_%s_%s with %s test failed\n",
                  GetChannelOrderName(img_format->image_channel_order),
                  GetChannelTypeName(img_format->image_channel_data_type),
                  get_mem_flag_name(img_flags));
        err = -1;
    }
    else
    {
        log_info("WRITE_IMAGE_%s_%s with %s test passed\n",
                 GetChannelOrderName(img_format->image_channel_order),
                 GetChannelTypeName(img_format->image_channel_data_type),
                 get_mem_flag_name(img_flags));
    }

    return err;
}

bool check_format(cl_device_id device, cl_context context,
                  cl_mem_object_type image_type,
                  const cl_image_format img_format, cl_mem_flags test_flags)
{
    return is_image_format_required(img_format, test_flags, image_type, device)
        || is_image_format_supported(context, test_flags, image_type,
                                     &img_format);
}
}
int test_writeimage(cl_device_id device, cl_context context,
                    cl_command_queue queue, int num_elements)
{
    int err = 0;
    const cl_image_format format[] = { { CL_RGBA, CL_UNORM_INT8 },
                                       { CL_BGRA, CL_UNORM_INT8 } };
    const cl_mem_flags test_flags[] = { CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE };

    for (size_t i = 0; i < ARRAY_SIZE(test_flags) && !err; i++)
    {
        err = test_writeimage<cl_uchar>(device, context, queue, &format[0],
                                        test_flags[i]);

        if (check_format(device, context, CL_MEM_OBJECT_IMAGE2D, format[1],
                         test_flags[i]))
        {
            err |= test_writeimage<cl_uchar>(device, context, queue, &format[1],
                                             test_flags[i]);
        }
    }
    return err;
}

int test_writeimage_int16(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    int err = 0;
    const cl_image_format format = { CL_RGBA, CL_UNORM_INT16 };
    const cl_mem_flags test_flags[] = { CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE };

    for (size_t i = 0; i < ARRAY_SIZE(test_flags) && !err; i++)
    {
        err = test_writeimage<cl_ushort>(device, context, queue, &format,
                                         test_flags[i]);
    }
    return err;
}

int test_writeimage_fp32(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    int err = 0;
    const cl_image_format format = { CL_RGBA, CL_FLOAT };
    const cl_mem_flags test_flags[] = { CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE };

    for (size_t i = 0; i < ARRAY_SIZE(test_flags) && !err; i++)
    {
        err = test_writeimage<cl_float>(device, context, queue, &format,
                                        test_flags[i]);
    }
    return err;
}
