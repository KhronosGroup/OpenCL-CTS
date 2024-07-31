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

#include <algorithm>
#include <string>
#include <vector>

#include "procs.h"

#define TEST_IMAGE_WIDTH_2D (512)
#define TEST_IMAGE_HEIGHT_2D (512)

#define TEST_IMAGE_WIDTH_3D (64)
#define TEST_IMAGE_HEIGHT_3D (64)
#define TEST_IMAGE_DEPTH_3D (64)

#define TEST_IMAGE_WIDTH(TYPE)                                                 \
    ((CL_MEM_OBJECT_IMAGE2D == TYPE) ? TEST_IMAGE_WIDTH_2D                     \
                                     : TEST_IMAGE_WIDTH_3D)
#define TEST_IMAGE_HEIGHT(TYPE)                                                \
    ((CL_MEM_OBJECT_IMAGE2D == TYPE) ? TEST_IMAGE_HEIGHT_2D                    \
                                     : TEST_IMAGE_HEIGHT_3D)
#define TEST_IMAGE_DEPTH(TYPE)                                                 \
    ((CL_MEM_OBJECT_IMAGE2D == TYPE) ? 1 : TEST_IMAGE_DEPTH_3D)

namespace {
const char *kernel_source_2d = R"(
__kernel void test_CL_BGRACL_UNORM_INT8(read_only image2d_t srcimg, __global uchar4 *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    indx = tid_y * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y)) * 255.0f;
    dst[indx] = convert_uchar4_rte(color.zyxw);
}

__kernel void test_CL_RGBACL_UNORM_INT8(read_only image2d_t srcimg, __global uchar4 *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    indx = tid_y * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int2)(tid_x, tid_y)) * 255.0f;
    dst[indx] = convert_uchar4_rte(color);
}

__kernel void test_CL_RGBACL_UNORM_INT16(read_only image2d_t srcimg, __global ushort4 *dst, sampler_t smp)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    indx = tid_y * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, smp, (int2)(tid_x, tid_y));
    ushort4 dst_write;
    dst_write.x = convert_ushort_rte(color.x * 65535.0f);
    dst_write.y = convert_ushort_rte(color.y * 65535.0f);
    dst_write.z = convert_ushort_rte(color.z * 65535.0f);
    dst_write.w = convert_ushort_rte(color.w * 65535.0f);
    dst[indx] = dst_write;
}

__kernel void test_CL_RGBACL_FLOAT(read_only image2d_t srcimg, __global float4 *dst, sampler_t smp)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    indx = tid_y * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, smp, (int2)(tid_x, tid_y));
    
    dst[indx].x = color.x;
    dst[indx].y = color.y;
    dst[indx].z = color.z;
    dst[indx].w = color.w;

}
)";

static const char *kernel_source_3d = R"(
__kernel void test_CL_BGRACL_UNORM_INT8(read_only image3d_t srcimg, __global uchar4 *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    tid_z = get_global_id(2);
    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0))* 255.0f;
    dst[indx].x = color.z;
    dst[indx].y = color.y;
    dst[indx].z = color.x;
    dst[indx].w = color.w;

}

__kernel void test_CL_RGBACL_UNORM_INT8(read_only image3d_t srcimg, __global uchar4 *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    tid_z = get_global_id(2);
    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0))* 255.0f;

    dst[indx].x = color.x;
    dst[indx].y = color.y;
    dst[indx].z = color.z;
    dst[indx].w = color.w;

}

__kernel void test_CL_RGBACL_UNORM_INT16(read_only image3d_t srcimg, __global ushort4 *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    tid_z = get_global_id(2);
    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));
    ushort4 dst_write;
    dst_write.x = convert_ushort_rte(color.x * 65535.0f);
    dst_write.y = convert_ushort_rte(color.y * 65535.0f);
    dst_write.z = convert_ushort_rte(color.z * 65535.0f);
    dst_write.w = convert_ushort_rte(color.w * 65535.0f);
    dst[indx] = dst_write;

}

__kernel void test_CL_RGBACL_FLOAT(read_only image3d_t srcimg, __global float *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    tid_z = get_global_id(2);
    int    indx = (tid_z * get_image_height(srcimg) + tid_y) * get_image_width(srcimg) + tid_x;
    float4 color;

    color = read_imagef(srcimg, sampler, (int4)(tid_x, tid_y, tid_z, 0));
    indx *= 4;
    dst[indx+0] = color.x;
    dst[indx+1] = color.y;
    dst[indx+2] = color.z;
    dst[indx+3] = color.w;

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

cl_mem create_image_xd(cl_context context, cl_mem_flags flags,
                       cl_mem_object_type type, const cl_image_format *fmt,
                       size_t x, size_t y, size_t z, cl_int *err)
{

    return (CL_MEM_OBJECT_IMAGE2D == type)
        ? create_image_2d(context, flags, fmt, x, y, 0, nullptr, err)
        : create_image_3d(context, flags, fmt, x, y, z, 0, 0, nullptr, err);
}

template <cl_mem_object_type IMG_TYPE, typename T>
int test_readimage(cl_device_id device, cl_context context,
                   cl_command_queue queue, const cl_image_format *img_format)
{
    clMemWrapper streams[2];
    clProgramWrapper program;
    clKernelWrapper kernel;
    clSamplerWrapper sampler;

    std::string kernel_name("test_");

    size_t img_width = TEST_IMAGE_WIDTH(IMG_TYPE);
    size_t img_height = TEST_IMAGE_HEIGHT(IMG_TYPE);
    size_t img_depth = TEST_IMAGE_DEPTH(IMG_TYPE);

    int err;

    const size_t origin[3] = { 0, 0, 0 };
    const size_t region[3] = { img_width, img_height, img_depth };

    const size_t num_elements = img_width * img_height * img_depth * 4;
    const size_t length = num_elements * sizeof(T);

    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    std::vector<T> input(num_elements);
    std::vector<T> output(num_elements);

    generate_random_inputs(input);

    streams[0] =
        create_image_xd(context, CL_MEM_READ_ONLY, IMG_TYPE, img_format,
                        img_width, img_height, img_depth, &err);
    test_error(err, "create_image failed.");

    streams[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, length, NULL, &err);
    test_error(err, "clCreateBuffer failed.");

    sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE,
                              CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0,
                              input.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueWriteImage failed.");

    kernel_name += GetChannelOrderName(img_format->image_channel_order);
    kernel_name += GetChannelTypeName(img_format->image_channel_data_type);

    const char **kernel_source = (CL_MEM_OBJECT_IMAGE2D == IMG_TYPE)
        ? &kernel_source_2d
        : &kernel_source_3d;

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      kernel_source, kernel_name.c_str());
    test_error(err, "create_single_kernel_helper failed.");

    err = clSetKernelArg(kernel, 0, sizeof(streams[0]), &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(streams[1]), &streams[1]);
    err |= clSetKernelArg(kernel, 2, sizeof(sampler), &sampler);
    test_error(err, "clSetKernelArgs failed\n");

    err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, region, NULL, 0, NULL,
                                 NULL);
    test_error(err, "clEnqueueNDRangeKernel failed\n");

    err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length,
                              output.data(), 0, NULL, NULL);
    test_error(err, "clEnqueueReadBuffer failed\n");

    if (0 != memcmp(input.data(), output.data(), length))
    {
        log_error("READ_IMAGE_%s_%s test failed\n",
                  GetChannelOrderName(img_format->image_channel_order),
                  GetChannelTypeName(img_format->image_channel_data_type));
        err = -1;
    }
    else
    {
        log_info("READ_IMAGE_%s_%s test passed\n",
                 GetChannelOrderName(img_format->image_channel_order),
                 GetChannelTypeName(img_format->image_channel_data_type));
    }

    return err;
}

bool check_format(cl_device_id device, cl_context context,
                  cl_mem_object_type image_type,
                  const cl_image_format img_format)
{
    return is_image_format_required(img_format, CL_MEM_READ_ONLY, image_type,
                                    device)
        || is_image_format_supported(context, CL_MEM_READ_ONLY, image_type,
                                     &img_format);
}

}
int test_readimage(cl_device_id device, cl_context context,
                   cl_command_queue queue, int num_elements)
{
    const cl_image_format format[] = { { CL_RGBA, CL_UNORM_INT8 },
                                       { CL_BGRA, CL_UNORM_INT8 } };

    int err = test_readimage<CL_MEM_OBJECT_IMAGE2D, cl_uchar>(
        device, context, queue, &format[0]);

    if (check_format(device, context, CL_MEM_OBJECT_IMAGE2D, format[1]))
    {
        err |= test_readimage<CL_MEM_OBJECT_IMAGE2D, cl_uchar>(
            device, context, queue, &format[1]);
    }

    return err;
}

int test_readimage_int16(cl_device_id device, cl_context context,
                         cl_command_queue queue, int num_elements)
{
    const cl_image_format format = { CL_RGBA, CL_UNORM_INT16 };
    return test_readimage<CL_MEM_OBJECT_IMAGE2D, cl_ushort>(device, context,
                                                            queue, &format);
}

int test_readimage_fp32(cl_device_id device, cl_context context,
                        cl_command_queue queue, int num_elements)
{
    const cl_image_format format = { CL_RGBA, CL_FLOAT };
    return test_readimage<CL_MEM_OBJECT_IMAGE2D, cl_float>(device, context,
                                                           queue, &format);
}

int test_readimage3d(cl_device_id device, cl_context context,
                     cl_command_queue queue, int num_elements)
{
    const cl_image_format format[] = { { CL_RGBA, CL_UNORM_INT8 },
                                       { CL_BGRA, CL_UNORM_INT8 } };

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)

    int err = test_readimage<CL_MEM_OBJECT_IMAGE3D, cl_uchar>(
        device, context, queue, &format[0]);

    if (check_format(device, context, CL_MEM_OBJECT_IMAGE3D, format[1]))
    {
        err |= test_readimage<CL_MEM_OBJECT_IMAGE3D, cl_uchar>(
            device, context, queue, &format[1]);
    }

    return err;
}

int test_readimage3d_int16(cl_device_id device, cl_context context,
                           cl_command_queue queue, int num_elements)
{
    const cl_image_format format = { CL_RGBA, CL_UNORM_INT16 };

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)

    return test_readimage<CL_MEM_OBJECT_IMAGE3D, cl_ushort>(device, context,
                                                            queue, &format);
}
int test_readimage3d_fp32(cl_device_id device, cl_context context,
                          cl_command_queue queue, int num_elements)
{
    const cl_image_format format = { CL_RGBA, CL_FLOAT };

    PASSIVE_REQUIRE_3D_IMAGE_SUPPORT(device)

    return test_readimage<CL_MEM_OBJECT_IMAGE3D, cl_float>(device, context,
                                                           queue, &format);
}
