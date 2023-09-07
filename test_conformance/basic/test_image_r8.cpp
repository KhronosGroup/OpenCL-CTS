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
const char *r_uint8_kernel_code = R"(
__kernel void test_r_uint8(read_only image2d_t srcimg, __global unsigned char *dst, sampler_t sampler)
{
    int    tid_x = get_global_id(0);
    int    tid_y = get_global_id(1);
    int    indx = tid_y * get_image_width(srcimg) + tid_x;
    uint4  color;

    color = read_imageui(srcimg, sampler, (int2)(tid_x, tid_y));
    dst[indx] = (unsigned char)(color.x);
})";


void generate_random_inputs(std::vector<cl_uchar> &v)
{
    RandomSeed seed(gRandomSeed);

    auto random_generator = [&seed]() {
        return static_cast<cl_uchar>(genrand_int32(seed));
    };

    std::generate(v.begin(), v.end(), random_generator);
}

}
int test_image_r8(cl_device_id device, cl_context context,
                  cl_command_queue queue, int num_elements)
{
    clMemWrapper streams[2];
    clProgramWrapper program;
    clKernelWrapper kernel;
    const size_t img_width = 512;
    const size_t img_height = 512;
    const size_t length = img_width * img_height;
    int err;

    PASSIVE_REQUIRE_IMAGE_SUPPORT(device)

    const cl_image_format img_format = { CL_R, CL_UNSIGNED_INT8 };

    // early out if this image type is not supported
    if (!is_image_format_supported(context, CL_MEM_READ_ONLY,
                                   CL_MEM_OBJECT_IMAGE2D, &img_format))
    {
        log_info("WARNING: Image type not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    std::vector<cl_uchar> input(length);
    std::vector<cl_uchar> output(length);

    generate_random_inputs(input);

    streams[0] = create_image_2d(context, CL_MEM_READ_ONLY, &img_format,
                                 img_width, img_height, 0, nullptr, &err);
    test_error(err, "create_image_2d failed.");

    streams[1] =
        clCreateBuffer(context, CL_MEM_READ_WRITE, length, nullptr, &err);
    test_error(err, "clCreateBuffer failed.");

    const size_t origin[3] = { 0, 0, 0 },
                 region[3] = { img_width, img_height, 1 };
    err = clEnqueueWriteImage(queue, streams[0], CL_TRUE, origin, region, 0, 0,
                              input.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueWriteImage failed.");

    err = create_single_kernel_helper(context, &program, &kernel, 1,
                                      &r_uint8_kernel_code, "test_r_uint8");
    test_error(err, "create_single_kernel_helper failed.");

    clSamplerWrapper sampler = clCreateSampler(
        context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    test_error(err, "clCreateSampler failed");

    err = clSetKernelArg(kernel, 0, sizeof streams[0], &streams[0]);
    err |= clSetKernelArg(kernel, 1, sizeof streams[1], &streams[1]);
    err |= clSetKernelArg(kernel, 2, sizeof sampler, &sampler);
    test_error(err, "clSetKernelArgs failed\n");

    size_t threads[] = { img_width, img_height };
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, threads, nullptr, 0,
                                 nullptr, nullptr);
    test_error(err, "clEnqueueNDRangeKernel failed\n");


    err = clEnqueueReadBuffer(queue, streams[1], CL_TRUE, 0, length,
                              output.data(), 0, nullptr, nullptr);
    test_error(err, "clEnqueueReadBuffer failed\n");

    if (0 != memcmp(input.data(), output.data(), length))
    {
        log_error("READ_IMAGE_R_UNSIGNED_INT8 test failed\n");
        err = -1;
    }
    else
    {
        log_info("READ_IMAGE_R_UNSIGNED_INT8 test passed\n");
    }

    return err;
}
